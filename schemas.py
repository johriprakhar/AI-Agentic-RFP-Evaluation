"""
Declarative schema definitions and a dependency-free validator for every
structured payload the application exchanges with an LLM.

The validator is deliberately separate from `scoring.normalize_evaluation`:

* `schemas.validate_evaluation_payload` is **strict**. It reports exactly what is
  wrong with a model reply so the caller can ask the model to repair it.
* `scoring.normalize_evaluation` is **total**. It always returns one usable row
  per active criterion, whatever the model sent.

Running strict validation first and normalization second means a malformed reply
gets one or more repair attempts before the pipeline falls back to safe defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Field specification primitives
# ---------------------------------------------------------------------------

#: Allowed values for a criterion result's ``status`` field.
STATUS_VALUES = ("specified", "not specified")


@dataclass(frozen=True)
class Field:
    """One declarative field rule inside a schema."""

    name: str
    kind: str                      # "number" | "string" | "enum" | "list" | "object"
    required: bool = True
    enum: Optional[tuple] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    item_kind: Optional[str] = None
    allow_empty: bool = True
    description: str = ""


#: Schema for a single element of the ``criteria`` array.
CRITERION_RESULT_SCHEMA: tuple = (
    Field("criterion_id", "number", required=True,
          description="Must match one of the active criterion ids supplied in the prompt."),
    Field("score", "number", required=True, min_value=0.0,
          description="Numeric score between 0 and the criterion's max_score."),
    Field("status", "enum", required=True, enum=STATUS_VALUES,
          description='Either "specified" or "not specified".'),
    Field("justification", "string", required=True, allow_empty=False,
          description="Explanation of why this score was given."),
    Field("evidence", "string", required=True, allow_empty=True,
          description="Verbatim quote from the proposal, with a page number when identifiable."),
)

#: Schema for the whole evaluation reply.
EVALUATION_SCHEMA: tuple = (
    Field("criteria", "list", required=True, item_kind="object", allow_empty=False,
          description="One result object per active criterion."),
    Field("risks", "list", required=False, item_kind="string",
          description="Zero or more plain-string risk statements."),
    Field("overall_summary", "string", required=False,
          description="Short narrative summary of the proposal."),
)

#: Schema for a single element of the search engine's ``chunk_evaluations`` array.
CHUNK_EVALUATION_SCHEMA: tuple = (
    Field("chunk_id", "string", required=True, allow_empty=False,
          description="Must be one of the candidate chunk_id values supplied in the prompt."),
    Field("relevance_score", "number", required=True, min_value=0.0, max_value=100.0,
          description="Semantic relevance from 0 to 100."),
    Field("reasoning", "string", required=False,
          description="One sentence explaining the match."),
    Field("matched_quote", "string", required=False,
          description="Exact quote taken from the chunk text."),
)

#: Schema for the whole LLM search reply.
SEARCH_SCHEMA: tuple = (
    Field("chunk_evaluations", "list", required=True, item_kind="object", allow_empty=False,
          description="One evaluation object per candidate chunk judged relevant."),
    Field("most_relevant_document", "string", required=False,
          description="Name of the document that best answers the query."),
    Field("overall_answer", "string", required=False,
          description="One or two sentence synthesis of the answer."),
)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Outcome of a strict schema check."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found."""
        return not self.errors

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def as_dict(self) -> Dict[str, List[str]]:
        return {"errors": list(self.errors), "warnings": list(self.warnings)}


# ---------------------------------------------------------------------------
# Primitive checks
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        try:
            return math.isfinite(float(value.strip()))
        except (ValueError, AttributeError):
            return False
    return False


def _check_field(container: Dict[str, Any], spec: Field, where: str) -> ValidationReport:
    report = ValidationReport()
    present = spec.name in container and container[spec.name] is not None

    if not present:
        if spec.required:
            report.errors.append(f"{where}: missing required field '{spec.name}' ({spec.description})")
        return report

    value = container[spec.name]

    if spec.kind == "number":
        if not _is_number(value):
            report.errors.append(f"{where}: field '{spec.name}' must be a finite number, got {value!r}")
            return report
        numeric = float(value)
        if spec.min_value is not None and numeric < spec.min_value:
            report.errors.append(
                f"{where}: field '{spec.name}' is {numeric:g}, below the minimum of {spec.min_value:g}"
            )
        if spec.max_value is not None and numeric > spec.max_value:
            report.errors.append(
                f"{where}: field '{spec.name}' is {numeric:g}, above the maximum of {spec.max_value:g}"
            )

    elif spec.kind == "string":
        if not isinstance(value, str):
            report.errors.append(f"{where}: field '{spec.name}' must be a string, got {type(value).__name__}")
        elif not spec.allow_empty and not value.strip():
            report.errors.append(f"{where}: field '{spec.name}' must not be empty ({spec.description})")

    elif spec.kind == "enum":
        text = str(value).strip().lower()
        if text not in (spec.enum or ()):
            report.errors.append(
                f"{where}: field '{spec.name}' must be one of {list(spec.enum or ())}, got {value!r}"
            )

    elif spec.kind == "list":
        if not isinstance(value, list):
            report.errors.append(f"{where}: field '{spec.name}' must be a list, got {type(value).__name__}")
        elif not spec.allow_empty and not value:
            report.errors.append(f"{where}: field '{spec.name}' must not be an empty list")
        elif spec.item_kind == "string":
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    report.warnings.append(
                        f"{where}: '{spec.name}'[{index}] is not a string and was ignored."
                    )
        elif spec.item_kind == "object":
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    report.errors.append(
                        f"{where}: '{spec.name}'[{index}] must be an object, got {type(item).__name__}"
                    )

    elif spec.kind == "object" and not isinstance(value, dict):
        report.errors.append(f"{where}: field '{spec.name}' must be an object, got {type(value).__name__}")

    return report


# ---------------------------------------------------------------------------
# Evaluation payload validation
# ---------------------------------------------------------------------------


def validate_evaluation_payload(payload: Any, criteria: List[Dict[str, Any]]) -> ValidationReport:
    """
    Strictly validate an `evaluate_supplier` reply against the active criteria.

    Blocking errors cover anything that would make the result untrustworthy:
    wrong container types, missing or non-numeric scores, scores above the
    criterion maximum, unknown or duplicated criterion ids, and missing
    justifications. Non-blocking observations are returned as warnings.
    """
    report = ValidationReport()

    if not isinstance(payload, dict):
        report.errors.append(
            f"Top level must be a JSON object with a 'criteria' array, got {type(payload).__name__}."
        )
        return report

    for spec in EVALUATION_SCHEMA:
        report.merge(_check_field(payload, spec, "response"))

    rows = payload.get("criteria")
    if not isinstance(rows, list):
        return report

    expected = {int(c["criterion_id"]): c for c in criteria}
    seen: Dict[int, int] = {}

    for index, row in enumerate(rows):
        where = f"criteria[{index}]"
        if not isinstance(row, dict):
            continue

        for spec in CRITERION_RESULT_SCHEMA:
            report.merge(_check_field(row, spec, where))

        raw_id = row.get("criterion_id")
        if not _is_number(raw_id):
            continue
        cid = int(float(raw_id))

        if cid not in expected:
            report.errors.append(
                f"{where}: criterion_id {cid} is not an active criterion. "
                f"Valid ids are {sorted(expected)}."
            )
            continue

        seen[cid] = seen.get(cid, 0) + 1
        if seen[cid] == 2:
            report.errors.append(
                f"{where}: criterion_id {cid} ('{expected[cid]['name']}') appears more than once. "
                "Return exactly one result per criterion."
            )

        if _is_number(row.get("score")):
            score = float(row["score"])
            maximum = float(expected[cid]["max_score"])
            if score > maximum:
                report.errors.append(
                    f"{where}: score {score:g} exceeds max_score {maximum:g} for "
                    f"'{expected[cid]['name']}'."
                )

        status = str(row.get("status", "")).strip().lower()
        evidence = str(row.get("evidence", "") or "").strip()
        if status == "specified" and not evidence:
            report.warnings.append(
                f"{where}: '{expected[cid]['name']}' is marked specified but carries no evidence quote."
            )

    missing = sorted(set(expected) - set(seen))
    if missing:
        names = ", ".join(f"{cid} ({expected[cid]['name']})" for cid in missing)
        report.errors.append(f"response: no result returned for criterion id(s) {names}.")

    return report


def validate_search_payload(payload: Any, valid_chunk_ids: Any = None) -> ValidationReport:
    """Strictly validate a `search_with_llm` reply."""
    report = ValidationReport()

    if not isinstance(payload, dict):
        report.errors.append(
            f"Top level must be a JSON object with a 'chunk_evaluations' array, "
            f"got {type(payload).__name__}."
        )
        return report

    for spec in SEARCH_SCHEMA:
        report.merge(_check_field(payload, spec, "response"))

    rows = payload.get("chunk_evaluations")
    if not isinstance(rows, list):
        return report

    allowed = set(valid_chunk_ids) if valid_chunk_ids is not None else None
    for index, row in enumerate(rows):
        where = f"chunk_evaluations[{index}]"
        if not isinstance(row, dict):
            continue
        for spec in CHUNK_EVALUATION_SCHEMA:
            report.merge(_check_field(row, spec, where))
        if allowed is not None:
            chunk_id = row.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id not in allowed:
                report.warnings.append(
                    f"{where}: chunk_id '{chunk_id}' was not among the candidates and was ignored."
                )

    return report


# ---------------------------------------------------------------------------
# Repair prompting
# ---------------------------------------------------------------------------


def repair_instruction(report: ValidationReport, limit: int = 12) -> str:
    """
    Turn a failed validation report into a corrective follow-up prompt.

    Feeding the precise errors back to the model is far more effective than
    simply re-sending the original request, because the model is told exactly
    which fields to fix rather than having to guess.
    """
    problems = report.errors[:limit]
    bullet_list = "\n".join(f"- {problem}" for problem in problems)
    overflow = ""
    if len(report.errors) > limit:
        overflow = f"\n- …and {len(report.errors) - limit} further problem(s) of the same kind."
    return (
        "Your previous reply did not satisfy the required JSON schema.\n"
        "Fix exactly these problems and return the complete corrected JSON object:\n"
        f"{bullet_list}{overflow}\n\n"
        "Rules for the corrected reply:\n"
        "- Return the whole object again, not a patch or a diff.\n"
        "- Return JSON only, with no prose, commentary or code fences.\n"
        "- Include exactly one entry per active criterion id, with no duplicates.\n"
        "- Keep every score within 0 and that criterion's max_score.\n"
        '- Use only "specified" or "not specified" for status.\n'
        "- Do not invent evidence. If the proposal is silent on a criterion, score it low "
        'and set status to "not specified".'
    )


def schema_reference(schema: tuple) -> str:
    """Render a schema as human-readable lines, used in prompts and docs."""
    lines = []
    for spec in schema:
        flag = "required" if spec.required else "optional"
        kind = spec.kind if not spec.item_kind else f"list[{spec.item_kind}]"
        detail = f" — {spec.description}" if spec.description else ""
        lines.append(f"- {spec.name} ({kind}, {flag}){detail}")
    return "\n".join(lines)
