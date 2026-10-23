"""
The evaluation orchestrator.

This is the only module that knows the shape of the whole pipeline. It lives
outside the Streamlit layer deliberately: `app.py` collects inputs and renders
output, while the sequence below is plain Python that can be unit tested,
scripted or driven from a CLI with no browser involved.

Per supplier (parallelised across suppliers)::

    pdf.extract      deterministic  bytes            -> pages + diagnostics
    pdf.prepare      deterministic  pages + criteria -> budgeted prompt text
    llm.evaluate     model          text + criteria  -> schema-validated JSON
    schema.normalize deterministic  JSON             -> one row per criterion
    evidence.verify  deterministic  rows + pages     -> grounding reports

Then once, across the whole peer group::

    score.metrics    deterministic  all suppliers    -> weighted scores, benchmarks, PPI
    score.rank       deterministic  all suppliers    -> total deterministic order
    db.persist       deterministic  run              -> stored, reproducible run

Two properties are load-bearing. First, the only model step is the judgement
step; every number is computed by `scoring`, so no arithmetic can be
hallucinated. Second, the peer-group stage runs after all suppliers complete,
because a benchmark is by definition not knowable from one proposal alone.

Supplier evaluation runs concurrently since each proposal is independent and the
work is dominated by network latency. Ranking stays deterministic regardless of
completion order: results are re-sorted into submission order before scoring and
the ranking key is a strict total order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import database
import evidence
import pdf_processor
import scoring
import tools
from evaluator import default_max_chars, diagnostics_warnings, evaluate_supplier

#: Upper bound on concurrent supplier evaluations. Enough to hide network
#: latency across a six-proposal run without tripping provider rate limits.
MAX_WORKERS = 4

APP_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def build_registry() -> tools.ToolRegistry:
    """
    Declare every capability the pipeline uses, and whether it is model or maths.

    The manifest this produces is stored with each run, so the separation of
    concerns is an auditable property of the record rather than a claim in a
    README.
    """
    registry = tools.ToolRegistry()

    registry.register(
        "pdf.extract", pdf_processor.extract_document, tools.DETERMINISTIC,
        "Extract per-page text from PDF bytes and report empty or scanned pages.",
    )
    registry.register(
        "pdf.prepare", pdf_processor.prepare_document_for_llm, tools.DETERMINISTIC,
        "Fit a proposal inside the model input budget, condensing by criterion "
        "relevance instead of truncating silently.",
    )
    registry.register(
        "llm.evaluate", evaluate_supplier, tools.MODEL,
        "Ask the configured model to score one proposal against the active criteria "
        "and quote its evidence. The only non-deterministic step.",
    )
    registry.register(
        "schema.normalize", scoring.validate_and_normalize, tools.DETERMINISTIC,
        "Validate and normalize the model reply into exactly one trustworthy row "
        "per active criterion.",
    )
    registry.register(
        "evidence.verify", evidence.verify_criteria_evidence, tools.DETERMINISTIC,
        "Check every quoted passage back against the extracted document text.",
    )
    registry.register(
        "score.metrics", scoring.calculate_supplier_metrics, tools.DETERMINISTIC,
        "Compute weighted absolute scores, peer benchmarks, gaps and the PPI.",
    )
    registry.register(
        "score.rank", scoring.rank_suppliers, tools.DETERMINISTIC,
        "Apply the four-level deterministic tie-break order and explain each placement.",
    )
    registry.register(
        "db.persist", database.save_run, tools.DETERMINISTIC,
        "Store the run, its criteria snapshot and every result in SQLite.",
    )
    return registry


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


def validate_run_inputs(
    proposals: Sequence[Dict[str, Any]],
    criteria: Sequence[Dict[str, Any]],
    max_suppliers: int = 6,
) -> List[str]:
    """
    Check everything that would make a run meaningless, and report all of it at once.

    Returning the full list rather than the first failure means a user fixes one
    form, not one error at a time.
    """
    problems: List[str] = []

    if not criteria:
        problems.append("No active criteria. Seed or enable criteria before running an evaluation.")
    else:
        total = sum(float(c["weight"]) for c in criteria)
        if abs(total - 100.0) > 1e-3:
            problems.append(f"Active criterion weights total {total:g}%, not 100%.")
        if any(float(c["max_score"]) <= 0 for c in criteria):
            problems.append("Every criterion needs a max score greater than zero.")
        ids = [int(c["criterion_id"]) for c in criteria]
        if len(set(ids)) != len(ids):
            problems.append("Active criteria contain duplicate criterion ids.")

    if not proposals:
        problems.append("Upload at least one proposal PDF.")
    if len(proposals) > max_suppliers:
        problems.append(f"Upload no more than {max_suppliers} proposals per run.")

    names = [str(p.get("supplier_name", "")).strip() for p in proposals]
    if any(not name for name in names):
        problems.append("Every supplier needs a name.")
    folded = [name.casefold() for name in names if name]
    if len(set(folded)) != len(folded):
        problems.append("Supplier names must be unique within a run.")
    if any(not p.get("bytes") for p in proposals):
        problems.append("Every supplier needs an uploaded PDF.")

    return problems


# ---------------------------------------------------------------------------
# Per-supplier pipeline
# ---------------------------------------------------------------------------


def evaluate_one(
    proposal: Dict[str, Any],
    criteria: Sequence[Dict[str, Any]],
    llm_config: Dict[str, Any],
    registry: tools.ToolRegistry,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Run the per-supplier stage end to end.

    A failure here is contained: the supplier comes back with zeroed criteria,
    ``evaluation_failed`` set and the error recorded in ``warnings``, so one
    unreadable PDF cannot abort an otherwise valid six-supplier run. It still
    appears in the leaderboard, visibly unscored, which is safer than quietly
    dropping a bidder.
    """
    trace = tools.ToolTrace(label=proposal.get("supplier_name", "supplier"))
    warnings: List[str] = []
    base = {
        "supplier_name": proposal["supplier_name"],
        "submission_date": proposal["submission_date"],
        "experience_rating": float(proposal["experience_rating"]),
        "source_filename": proposal.get("source_filename", ""),
    }

    try:
        extraction = registry.invoke(
            "pdf.extract", proposal["bytes"], trace=trace,
            summarize=lambda out: f"{out['page_count']} page(s), {out['char_count']:,} characters",
        )
        warnings.extend(extraction["warnings"])

        if not extraction["has_text"]:
            raise ValueError(
                "No text could be extracted from this PDF. It appears to be a scanned or "
                "image-only document; OCR is not part of this pipeline. Supply a text-based PDF."
            )
        if extraction["is_likely_scanned"]:
            warnings.append(
                f"{len(extraction['empty_pages'])} of {extraction['page_count']} pages held no "
                "extractable text, so this proposal may be partly scanned. Scores reflect only "
                "the machine-readable portion."
            )

        prepared = registry.invoke(
            "pdf.prepare", extraction["pages"], criteria,
            default_max_chars(llm_config), proposal["supplier_name"],
            trace=trace,
            summarize=lambda out: (
                f"{out['report']['strategy']}, {out['report']['chars_sent']:,} chars "
                f"({out['report']['coverage']:.0%} of document)"
            ),
        )
        preparation_report = prepared["report"]
        warnings.extend(preparation_report["warnings"])

        note = ""
        if preparation_report["truncated"]:
            note = (
                "[NOTE] Only part of this proposal is shown. Judge strictly on the passages "
                'present and mark anything unsupported as "not specified".'
            )

        ai_result = registry.invoke(
            "llm.evaluate",
            supplier_name=proposal["supplier_name"],
            document_text=prepared["text"],
            criteria=list(criteria),
            llm_config=llm_config,
            preparation_note=note,
            max_attempts=max_attempts,
            trace=trace,
            summarize=lambda out: (
                f"{len(out.get('criteria') or [])} criterion result(s) in "
                f"{out.get('_diagnostics', {}).get('attempt_count', 1)} attempt(s)"
            ),
        )
        warnings.extend(diagnostics_warnings(ai_result.get("_diagnostics", {})))

        normalized, normalize_warnings = registry.invoke(
            "schema.normalize", ai_result, list(criteria), trace=trace,
            summarize=lambda out: f"{len(out[0])} row(s), {len(out[1])} correction(s)",
        )
        warnings.extend(normalize_warnings)

        normalized, evidence_warnings, evidence_summary = registry.invoke(
            "evidence.verify", normalized, extraction["pages"], trace=trace,
            summarize=lambda out: (
                f"{out[2]['verified']}/{out[2]['checked']} quote(s) grounded"
                if out[2]["checked"] else "no quotes to check"
            ),
        )
        warnings.extend(evidence_warnings)

        risks = ai_result.get("risks")
        return {
            **base,
            "criteria": normalized,
            "risks": [str(r) for r in risks] if isinstance(risks, list) else [],
            "overall_summary": str(ai_result.get("overall_summary", "") or ""),
            "warnings": warnings,
            "pages_count": extraction["page_count"],
            "document_stats": {
                "page_count": extraction["page_count"],
                "char_count": extraction["char_count"],
                "empty_pages": extraction["empty_pages"],
                "is_likely_scanned": extraction["is_likely_scanned"],
            },
            "preparation": preparation_report,
            "evidence_summary": evidence_summary,
            "llm_diagnostics": ai_result.get("_diagnostics", {}),
            "tool_trace": trace.as_dict(),
            "evaluation_failed": False,
        }

    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        normalized, _ = scoring.validate_and_normalize({}, list(criteria))
        for row in normalized:
            row["evidence_check"] = {
                "verified": None, "status": "not applicable", "method": "none",
                "coverage": 0.0, "located_page": None, "cited_page": None,
                "page_matches": None, "note": "Evaluation failed before evidence could be checked.",
            }
            row["evidence_verified"] = None
            row["evidence_status"] = "not applicable"
        return {
            **base,
            "criteria": normalized,
            "risks": [f"Evaluation failed: {message}"],
            "overall_summary": "Evaluation failed; this supplier's scores are not reliable.",
            "warnings": warnings + [message],
            "pages_count": 0,
            "document_stats": {},
            "preparation": {},
            "evidence_summary": {
                "checked": 0, "verified": 0, "partial": 0, "unverified": 0,
                "page_mismatches": 0, "grounding_rate": None, "document_word_count": 0,
            },
            "llm_diagnostics": {},
            "tool_trace": trace.as_dict(),
            "evaluation_failed": True,
            "error": message,
        }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run_evaluation(
    proposals: Sequence[Dict[str, Any]],
    criteria: Sequence[Dict[str, Any]],
    llm_config: Dict[str, Any],
    registry: Optional[tools.ToolRegistry] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    persist: bool = True,
    max_workers: int = MAX_WORKERS,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Execute a complete evaluation run and return everything needed to render and audit it.

    ``progress(completed, total, label)`` is called as suppliers finish, which
    lets a UI show progress without the orchestrator importing Streamlit.

    Returns ``{"run_id", "results", "criteria", "meta", "warnings"}``.
    """
    registry = registry or build_registry()
    criteria = list(criteria)
    proposals = list(proposals)

    problems = validate_run_inputs(proposals, criteria)
    if problems:
        raise ValueError("; ".join(problems))

    started = datetime.now(timezone.utc)
    results: List[Dict[str, Any]] = []
    total = len(proposals)

    def work(index_and_proposal):
        index, proposal = index_and_proposal
        return index, evaluate_one(proposal, criteria, llm_config, registry, max_attempts)

    if total == 1 or max_workers <= 1:
        for index, proposal in enumerate(proposals):
            results.append((index, evaluate_one(proposal, criteria, llm_config, registry, max_attempts)))
            if progress:
                progress(len(results), total, proposal["supplier_name"])
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as pool:
            futures = {pool.submit(work, item): item for item in enumerate(proposals)}
            for future in as_completed(futures):
                index, result = future.result()
                results.append((index, result))
                if progress:
                    progress(len(results), total, result["supplier_name"])

    # Restore submission order so the peer-group stage never depends on which
    # network call happened to return first.
    ordered = [result for _, result in sorted(results, key=lambda pair: pair[0])]

    scored = registry.invoke("score.metrics", ordered, criteria)
    ranked = registry.invoke("score.rank", scored)

    failed = [r["supplier_name"] for r in ranked if r.get("evaluation_failed")]
    run_warnings: List[str] = []
    if failed:
        run_warnings.append(
            f"{len(failed)} supplier(s) could not be evaluated and are ranked with zero scores: "
            + ", ".join(failed)
        )

    summary = scoring.ranking_summary(ranked)
    if summary["neutral_criteria_excluded_from_ppi"]:
        names = {int(c["criterion_id"]): c["name"] for c in criteria}
        labels = ", ".join(names.get(cid, str(cid)) for cid in summary["neutral_criteria_excluded_from_ppi"])
        run_warnings.append(
            f"No supplier scored on: {labels}. These criteria cannot separate the field and were "
            "excluded from the PPI; they still affect the absolute score."
        )

    finished = datetime.now(timezone.utc)
    grounding_checked = sum((r.get("evidence_summary") or {}).get("checked", 0) for r in ranked)
    grounding_verified = sum((r.get("evidence_summary") or {}).get("verified", 0) for r in ranked)

    meta: Dict[str, Any] = {
        "created_at": started.isoformat(),
        "completed_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "status": "completed_with_failures" if failed else "completed",
        "app_version": APP_VERSION,
        "provider": llm_config.get("provider"),
        "model": llm_config.get("model"),
        "temperature": llm_config.get("temperature"),
        "max_document_chars": default_max_chars(llm_config),
        "supplier_count": len(ranked),
        "criteria_count": len(criteria),
        "criteria_weight_total": sum(float(c["weight"]) for c in criteria),
        "failed_suppliers": failed,
        "ranking": summary,
        "tie_break_order": summary["tie_break_order"],
        "tool_manifest": registry.describe(),
        "evidence_grounding": {
            "quotes_checked": grounding_checked,
            "quotes_verified": grounding_verified,
            "grounding_rate": round(grounding_verified / grounding_checked, 4) if grounding_checked else None,
        },
        "warnings": run_warnings,
        "notes": f"{len(ranked)} supplier(s) scored against {len(criteria)} criteria.",
    }

    run_id = None
    if persist:
        run_id = registry.invoke("db.persist", ranked, criteria, meta)
    meta["rfp_run_id"] = run_id

    return {
        "run_id": run_id,
        "results": ranked,
        "criteria": criteria,
        "meta": meta,
        "warnings": run_warnings,
    }


def export_payload(run_id: Optional[str], results: Sequence[Dict[str, Any]],
                   criteria: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble the downloadable JSON audit record.

    Self-contained on purpose: the criteria snapshot, the run metadata, the
    tie-break order and the tool manifest all travel with the scores, so the file
    can be re-read years later without the database or this codebase.
    """
    return {
        "rfp_run_id": run_id,
        "schema_version": APP_VERSION,
        "run": meta,
        "criteria": list(criteria),
        "tie_break_order": meta.get("tie_break_order", []),
        "results": list(results),
    }
