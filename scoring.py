"""
Deterministic scoring, peer benchmarking and ranking.

Everything in this module is a pure function of its inputs. No network calls, no
model access, no randomness, no clock reads. Given the same normalized criterion
rows the output is bit-for-bit identical on every run, which is what makes an
evaluation defensible in a procurement review.

Pipeline position::

    LLM reply -> schemas.validate_evaluation_payload (strict, may trigger repair)
              -> validate_and_normalize             (total, always usable)
              -> calculate_supplier_metrics         (weighted score, benchmarks, PPI)
              -> rank_suppliers                     (total deterministic order)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

#: Documented tie-break order, surfaced in the UI and stored with every run so a
#: historical ranking can always be explained.
TIE_BREAK_ORDER: Tuple[Dict[str, str], ...] = (
    {"key": "ppi", "direction": "descending", "label": "Peer Performance Index"},
    {"key": "submission_date", "direction": "ascending", "label": "Submission date (earlier wins)"},
    {"key": "experience_rating", "direction": "descending", "label": "Historical experience rating"},
    {"key": "supplier_name", "direction": "ascending", "label": "Supplier name (A–Z)"},
)

#: Tolerance used for every float comparison in this module.
EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def validate_and_normalize(ai_result: Any, criteria: List[Dict[str, Any]]):
    """
    Produce exactly one trustworthy row per active criterion.

    This function never raises and never returns a partial result. Anything the
    model got wrong is corrected here and recorded in ``warnings`` so the UI and
    the stored run both show what had to be repaired:

    * missing criterion            -> score 0, status "not specified"
    * unparseable / infinite score -> score 0
    * score outside 0..max_score   -> clipped to the nearest bound
    * unrecognised status           -> "specified"
    * status "not specified" with a non-zero score -> score forced to 0
    * duplicate criterion_id        -> first occurrence kept, duplicate reported
    * unknown criterion_id          -> dropped and reported

    Forcing not-specified rows to zero is what keeps the absolute score and the
    peer benchmark consistent: a row excluded from benchmarking can no longer
    quietly contribute points to its own weighted total.
    """
    warnings: List[str] = []
    raw_rows = ai_result.get("criteria", []) if isinstance(ai_result, dict) else []
    if not isinstance(raw_rows, list):
        warnings.append("The 'criteria' field was not a list; every criterion was defaulted.")
        raw_rows = []

    active_ids = {int(c["criterion_id"]) for c in criteria}
    name_by_id = {int(c["criterion_id"]): c["name"] for c in criteria}

    by_id: Dict[int, Dict[str, Any]] = {}
    for position, item in enumerate(raw_rows):
        if not isinstance(item, dict):
            warnings.append(f"Ignored criteria[{position}]: expected an object.")
            continue
        raw_id = item.get("criterion_id")
        if raw_id is None:
            warnings.append(f"Ignored criteria[{position}]: no criterion_id was supplied.")
            continue
        try:
            cid = int(float(raw_id))
        except (ValueError, TypeError):
            warnings.append(f"Ignored criteria[{position}]: criterion_id {raw_id!r} is not a number.")
            continue
        if cid not in active_ids:
            warnings.append(
                f"Ignored criteria[{position}]: criterion_id {cid} is not an active criterion."
            )
            continue
        if cid in by_id:
            warnings.append(
                f"Duplicate result for '{name_by_id[cid]}' (criterion_id {cid}); "
                "the first occurrence was kept."
            )
            continue
        by_id[cid] = item

    normalized: List[Dict[str, Any]] = []
    for criterion in criteria:
        cid = int(criterion["criterion_id"])
        maximum = float(criterion["max_score"])
        weight = float(criterion["weight"])
        name = criterion["name"]
        item = by_id.get(cid)

        if item is None:
            warnings.append(
                f"Missing criterion result: {name}; assigned score 0 and marked not specified."
            )
            normalized.append({
                "criterion_id": cid,
                "name": name,
                "description": criterion.get("description", ""),
                "weight": weight,
                "score": 0.0,
                "max_score": maximum,
                "justification": "No valid result was returned for this criterion.",
                "evidence": "",
                "status": "not specified",
            })
            continue

        try:
            score = float(item.get("score", 0))
            if not math.isfinite(score):
                raise ValueError
        except (ValueError, TypeError):
            score = 0.0
            warnings.append(f"Invalid score for {name}; assigned 0.")

        clipped = min(max(score, 0.0), maximum)
        if abs(clipped - score) > EPSILON:
            warnings.append(f"Score for {name} was outside 0–{maximum:g} and was clipped to {clipped:g}.")

        status = str(item.get("status", "specified")).strip().lower()
        if status not in {"specified", "not specified"}:
            warnings.append(f"Unrecognised status {item.get('status')!r} for {name}; treated as specified.")
            status = "specified"

        if status == "not specified" and clipped > 0:
            warnings.append(
                f"{name} was marked not specified but carried a score of {clipped:g}; "
                "the score was reset to 0 for consistency with peer benchmarking."
            )
            clipped = 0.0

        evidence = str(item.get("evidence", "") or "").strip()
        if status == "specified" and not evidence:
            warnings.append(f"{name} was scored as specified but no evidence quote was supplied.")

        normalized.append({
            "criterion_id": cid,
            "name": name,
            "description": criterion.get("description", ""),
            "weight": weight,
            "score": clipped,
            "max_score": maximum,
            "justification": str(item.get("justification", "")).strip() or "No justification supplied.",
            "evidence": evidence,
            "status": status,
        })

    return normalized, warnings


#: Readable alias. `normalize_evaluation` says what the function does; the older
#: name is kept because callers and tests already use it.
normalize_evaluation = validate_and_normalize


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def assert_weights_total_100(criteria: List[Dict[str, Any]]) -> float:
    """Return the total weight, raising when it is not 100%."""
    total = sum(float(c["weight"]) for c in criteria)
    if abs(total - 100.0) > 1e-3:
        raise ValueError(f"Active criterion weights must total 100; got {total:g}.")
    return total


def calculate_supplier_metrics(results: List[Dict[str, Any]], criteria: List[Dict[str, Any]]):
    """
    Compute the absolute weighted score, peer benchmarks and the PPI.

    Absolute score (0–10), independent of the peer group::

        absolute = ( Σ_i (score_i / max_i) × weight_i ) / 10

    Peer benchmark for criterion *i* is the highest score any supplier achieved
    on that criterion while marked "specified".

    Relative performance is ``score / benchmark × 100``.

    A criterion whose benchmark is zero carries no information: no supplier
    earned a point on it, so it cannot separate the field. Such a criterion is
    marked ``discriminating: False`` and is excluded from both the numerator and
    the denominator of the PPI, rather than being awarded an arbitrary 0% or
    100%. The PPI therefore stays a weighted mean over the criteria that
    genuinely differentiate the suppliers::

        PPI = Σ_{i ∈ D} (relative_i × weight_i) / Σ_{i ∈ D} weight_i

    where *D* is the set of discriminating criteria.
    """
    assert_weights_total_100(criteria)

    benchmarks: Dict[int, float] = {}
    for criterion in criteria:
        cid = int(criterion["criterion_id"])
        specified = [
            row["score"]
            for supplier in results
            for row in supplier["criteria"]
            if row["criterion_id"] == cid and row["status"] == "specified"
        ]
        benchmarks[cid] = max(specified) if specified else 0.0

    discriminating = {cid: value > EPSILON for cid, value in benchmarks.items()}
    neutral = sorted(cid for cid, is_live in discriminating.items() if not is_live)

    for supplier in results:
        weighted_total = 0.0
        ppi_numerator = 0.0
        ppi_denominator = 0.0

        for row in supplier["criteria"]:
            maximum = float(row["max_score"])
            weight = float(row["weight"])
            cid = row["criterion_id"]
            benchmark = benchmarks[cid]

            achieved_fraction = (row["score"] / maximum) if maximum > EPSILON else 0.0
            weighted_points = achieved_fraction * weight
            weighted_total += weighted_points

            row["benchmark"] = round(benchmark, 4)
            row["gap"] = round(row["score"] - benchmark, 4)
            row["weighted_points"] = round(weighted_points / 10.0, 4)
            row["max_weighted_points"] = round(weight / 10.0, 4)
            row["is_benchmark_setter"] = bool(
                discriminating[cid]
                and row["status"] == "specified"
                and abs(row["score"] - benchmark) <= EPSILON
            )

            if discriminating[cid]:
                relative = row["score"] / benchmark * 100.0
                row["relative_performance_pct"] = round(relative, 2)
                row["discriminating"] = True
                ppi_numerator += relative * weight
                ppi_denominator += weight
            else:
                # No supplier scored on this criterion; it is excluded from the PPI.
                row["relative_performance_pct"] = None
                row["discriminating"] = False

        supplier["absolute_score"] = round(weighted_total / 10.0, 4)
        supplier["absolute_score_pct"] = round(weighted_total, 4)
        supplier["ppi"] = round(ppi_numerator / ppi_denominator, 4) if ppi_denominator > EPSILON else 0.0
        supplier["ppi_weight_basis"] = round(ppi_denominator, 4)
        supplier["benchmarks"] = {str(cid): round(value, 4) for cid, value in benchmarks.items()}
        supplier["neutral_criteria"] = neutral
        supplier["benchmark_criteria_won"] = sorted(
            row["criterion_id"] for row in supplier["criteria"] if row.get("is_benchmark_setter")
        )

    return results


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _sort_key(result: Dict[str, Any]):
    """The documented four-level total order, as a sortable tuple."""
    return (
        -float(result.get("ppi", 0.0)),
        str(result.get("submission_date", "9999-12-31")),
        -float(result.get("experience_rating", 0.0)),
        str(result.get("supplier_name", "")).casefold(),
    )


def _separation_reason(higher: Dict[str, Any], lower: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify which tie-break level actually separated two adjacent suppliers.

    Returns the level label plus the two competing values, so the UI can state
    "ranked above X on submission date" instead of leaving the user to guess.
    """
    higher_key = _sort_key(higher)
    lower_key = _sort_key(lower)
    for index, rule in enumerate(TIE_BREAK_ORDER):
        left, right = higher_key[index], lower_key[index]
        if isinstance(left, float) and isinstance(right, float):
            differs = abs(left - right) > EPSILON
        else:
            differs = left != right
        if differs:
            return {
                "level": index + 1,
                "key": rule["key"],
                "label": rule["label"],
                "tie_break_used": index > 0,
                "value": higher.get(rule["key"]),
                "compared_with": lower.get(rule["key"]),
            }
    return {
        "level": None,
        "key": None,
        "label": "Fully identical on every tie-break level",
        "tie_break_used": True,
        "value": None,
        "compared_with": None,
    }


def rank_suppliers(results: List[Dict[str, Any]]):
    """
    Sort suppliers into the documented total order and explain every placement.

    Order: PPI descending -> submission date ascending -> experience rating
    descending -> supplier name ascending (case-folded).

    Because supplier names are unique within a run, this key is a strict total
    order: the result does not depend on the input order, on dict iteration, or
    on the stability of the sort. Each supplier additionally receives
    ``rank_basis`` naming the level that put it ahead of the supplier below it,
    and ``rank_explanation``, a one-line human-readable version of the same.
    """
    results.sort(key=_sort_key)

    for index, result in enumerate(results):
        result["final_rank"] = index + 1
        result["tie_break_order"] = [rule["label"] for rule in TIE_BREAK_ORDER]

    for index, result in enumerate(results):
        following = results[index + 1] if index + 1 < len(results) else None
        if following is None:
            result["rank_basis"] = {
                "level": None,
                "key": None,
                "label": "Lowest ranked supplier in this run",
                "tie_break_used": False,
                "value": None,
                "compared_with": None,
            }
            result["rank_explanation"] = (
                f"Ranked #{result['final_rank']} of {len(results)} — lowest PPI in this run."
            )
            continue

        basis = _separation_reason(result, following)
        result["rank_basis"] = basis
        if basis["key"] == "ppi":
            result["rank_explanation"] = (
                f"Ranked #{result['final_rank']} ahead of {following['supplier_name']} on PPI "
                f"({float(result['ppi']):.2f}% vs {float(following['ppi']):.2f}%)."
            )
        elif basis["key"] is None:
            result["rank_explanation"] = (
                f"Ranked #{result['final_rank']}; identical to {following['supplier_name']} on every "
                "tie-break level."
            )
        else:
            result["rank_explanation"] = (
                f"Ranked #{result['final_rank']} ahead of {following['supplier_name']} on a tie-break: "
                f"{basis['label']} ({basis['value']} vs {basis['compared_with']}) after matching on PPI."
            )

    return results


def ranking_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run-level ranking facts worth storing alongside the per-supplier rows."""
    tie_broken = [r["supplier_name"] for r in results if r.get("rank_basis", {}).get("tie_break_used")]
    neutral = results[0].get("neutral_criteria", []) if results else []
    return {
        "tie_break_order": [rule["label"] for rule in TIE_BREAK_ORDER],
        "suppliers_separated_by_tie_break": tie_broken,
        "neutral_criteria_excluded_from_ppi": neutral,
        "ppi_weight_basis": results[0].get("ppi_weight_basis") if results else 0.0,
    }
