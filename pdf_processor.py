"""
PDF text extraction and model-input preparation.

Two responsibilities, kept separate:

* **Extraction** turns PDF bytes into per-page text plus a diagnostic report
  (page count, empty pages, whether the file looks scanned).
* **Preparation** turns those pages into the exact string handed to a model,
  under a character budget, and reports precisely what it did.

The preparation step exists because silently slicing a document at N characters
is the quietest way to produce a confidently wrong evaluation: everything after
the cut is invisible to the model, yet the result looks complete. Instead, when
a proposal exceeds the budget, the most relevant passages for the active
criteria are selected with BM25, assembled in reading order with explicit
omission markers, and the resulting coverage figure is attached to the run.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import pymupdf

from retrieval import chunks_from_pages, select_relevant_chunks

#: A page with fewer than this many characters of text is treated as having no
#: extractable text, which is the signature of a scanned or image-only page.
MIN_PAGE_CHARS = 20

#: Share of pages that must be text-free before the file is reported as scanned.
SCANNED_PAGE_RATIO = 0.8


def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text page by page so evidence can be tied back to a source page.

    Each page carries both ``text`` (stripped, for prompting) and ``raw_text``
    (untouched, so line numbering stays faithful to the PDF).
    """
    pages: List[Dict[str, Any]] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for index, page in enumerate(pdf, 1):
            raw_text = page.get_text("text")
            stripped = raw_text.strip()
            pages.append({
                "page": index,
                "text": stripped,
                "raw_text": raw_text,
                "char_count": len(stripped),
                "line_count": len([line for line in raw_text.splitlines() if line.strip()]),
                "has_text": len(stripped) >= MIN_PAGE_CHARS,
            })
    return pages


def extract_document(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Extract a PDF and describe what came out.

    Returns the pages plus a report containing the page count, which pages had
    no usable text, the total character count and ``is_likely_scanned``. The
    caller can then fail loudly with an actionable message instead of sending an
    empty string to a model and scoring a supplier zero for it.
    """
    pages = extract_pdf_pages(pdf_bytes)
    empty_pages = [page["page"] for page in pages if not page["has_text"]]
    total_chars = sum(page["char_count"] for page in pages)
    page_count = len(pages)
    # Compare the ratio directly. Flooring the threshold at one page would brand
    # any two-page proposal with a single image page as a scanned document.
    likely_scanned = bool(
        page_count and (len(empty_pages) / page_count) >= SCANNED_PAGE_RATIO
    )

    warnings: List[str] = []
    if empty_pages and not likely_scanned:
        shown = ", ".join(str(number) for number in empty_pages[:10])
        suffix = "…" if len(empty_pages) > 10 else ""
        warnings.append(
            f"{len(empty_pages)} of {page_count} page(s) contained no extractable text "
            f"(page {shown}{suffix}); they may be images or scans."
        )

    return {
        "pages": pages,
        "page_count": page_count,
        "empty_pages": empty_pages,
        "char_count": total_chars,
        "has_text": total_chars > 0,
        "is_likely_scanned": likely_scanned,
        "warnings": warnings,
    }


def build_document_text(pages: Iterable[Dict[str, Any]]) -> str:
    """Join pages into one string with ``[Page N]`` markers for evidence citation."""
    return "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}"
        for page in pages
        if str(page.get("text", "")).strip()
    )


def _criteria_queries(criteria: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
    """One retrieval query per active criterion, built from its name and description."""
    queries: List[str] = []
    for criterion in criteria or []:
        name = str(criterion.get("name", "")).strip()
        description = str(criterion.get("description", "")).strip()
        combined = f"{name} {description}".strip()
        if combined:
            queries.append(combined)
    return queries


#: Characters each selected passage costs on top of its own text once it is
#: wrapped in a "(lines N–M) " prefix and separated by newlines and page markers.
#: Budgeting for this up front is what keeps the even sample reaching the final
#: pages instead of being trimmed off the tail during assembly.
CHUNK_MARKUP_OVERHEAD = 30


def _digest_header(strategy: str) -> str:
    """The explanatory preamble placed at the top of a condensed document."""
    how = (
        "selected by relevance to the active evaluation criteria"
        if strategy == "criteria_targeted"
        else "sampled evenly across the document, because relevance ranking could not "
             "separate the passages"
    )
    return (
        f"[NOTE] This proposal exceeded the model input budget. The passages below were {how} "
        "and are presented in document order. Omitted regions are marked. Judge only what is "
        'present; if a criterion has no supporting passage here, mark it "not specified".\n'
    )


def _uniform_sample(
    chunks: Sequence[Dict[str, Any]],
    budget_chars: int,
    overhead_per_chunk: int = 0,
) -> List[Dict[str, Any]]:
    """
    Spread a budget evenly across a document, in reading order.

    Used when relevance ranking cannot separate the passages — a document whose
    every passage uses the same vocabulary gives BM25 almost no signal, because
    a term appearing everywhere has near-zero inverse document frequency. An
    even stride is strictly better than keeping the first N characters: it at
    least reaches every part of the document instead of stopping at the head.
    """
    if not chunks or budget_chars <= 0:
        return []

    average = max(
        1,
        sum(len(c["text"]) + 1 + overhead_per_chunk for c in chunks) // len(chunks),
    )
    capacity = max(1, budget_chars // average)
    if capacity >= len(chunks):
        candidates = list(chunks)
    else:
        stride = len(chunks) / capacity
        candidates = [chunks[min(len(chunks) - 1, int(i * stride))] for i in range(capacity)]

    selected: List[Dict[str, Any]] = []
    used = 0
    for chunk in candidates:
        size = len(chunk["text"]) + 1 + overhead_per_chunk
        if used + size > budget_chars:
            break
        selected.append(chunk)
        used += size
    return selected or [candidates[0]]


def _condense(
    pages: Sequence[Dict[str, Any]],
    criteria: Optional[Sequence[Dict[str, Any]]],
    max_chars: int,
    supplier_name: str,
) -> Dict[str, Any]:
    """
    Build a digest of an over-long document.

    Passages relevant to each active criterion are preferred. If relevance
    ranking returns nothing, an even sample across the document is used instead,
    so the digest still spans the whole proposal.

    Reserves roughly 92% of the budget for passage text, leaving room for the
    page markers and omission notes that keep the digest honest about its gaps.
    """
    queries = _criteria_queries(criteria)
    chunks = chunks_from_pages(pages, filename=f"{supplier_name}.pdf", supplier_name=supplier_name)

    if not queries or not chunks:
        return {"text": "", "selected": [], "pages_covered": [], "strategy": None}

    # Reserve room for whichever header wins, plus slack for page markers.
    reserve = max(len(_digest_header("criteria_targeted")),
                  len(_digest_header("uniform_sample"))) + 64
    budget = max(200, int(max_chars) - reserve)

    strategy = "criteria_targeted"
    selected = select_relevant_chunks(
        chunks, queries, budget_chars=budget, overhead_per_chunk=CHUNK_MARKUP_OVERHEAD,
    )
    if not selected:
        strategy = "uniform_sample"
        selected = _uniform_sample(chunks, budget, CHUNK_MARKUP_OVERHEAD)
    if not selected:
        return {"text": "", "selected": [], "pages_covered": [], "strategy": None}

    header = _digest_header(strategy)

    # Assemble with a running length check so the finished digest honours
    # max_chars. Selection only budgets the passage text; the page markers,
    # line-range prefixes and this header are real tokens too, and billing them
    # afterwards is how a "budgeted" prompt quietly ends up over budget.
    blocks: List[str] = []
    kept: List[Dict[str, Any]] = []
    length = len(header)
    current_page: Optional[int] = None
    previous_end: Optional[int] = None

    for chunk in selected:
        pieces: List[str] = []
        if chunk["page"] != current_page:
            pieces.append(f"\n[Page {chunk['page']}]")
        elif previous_end is not None and chunk["start_line"] > previous_end + 1:
            pieces.append("[… passage omitted as not relevant to the active criteria …]")
        pieces.append(f"(lines {chunk['start_line']}–{chunk['end_line']}) {chunk['text']}")

        addition = sum(len(piece) + 1 for piece in pieces)
        if kept and length + addition > max_chars:
            break

        if chunk["page"] != current_page:
            current_page = chunk["page"]
            previous_end = None
        blocks.extend(pieces)
        kept.append(chunk)
        length += addition
        previous_end = max(previous_end or 0, chunk["end_line"])

    return {
        "text": header + "\n".join(blocks).strip(),
        "selected": kept,
        "pages_covered": sorted({chunk["page"] for chunk in kept}),
        "strategy": strategy,
    }


def prepare_document_for_llm(
    pages: Sequence[Dict[str, Any]],
    criteria: Optional[Sequence[Dict[str, Any]]] = None,
    max_chars: int = 70000,
    supplier_name: str = "supplier",
) -> Dict[str, Any]:
    """
    Produce the exact text to send to a model, plus a report of how it was built.

    Strategies:

    * ``full`` — the whole document fitted inside the budget.
    * ``criteria_targeted`` — the document was too long, so BM25 selected the
      passages most relevant to each active criterion.
    * ``head_truncated`` — fallback when no criteria were supplied to retrieve
      against; the document is cut and the cut is reported.

    The report always states ``truncated``, the character counts, the fraction
    of the document that reached the model (``coverage``) and a human-readable
    ``warnings`` list. Nothing is dropped silently.
    """
    full_text = build_document_text(pages)
    total_chars = len(full_text)
    report: Dict[str, Any] = {
        "strategy": "full",
        "truncated": False,
        "chars_total": total_chars,
        "chars_sent": total_chars,
        "coverage": 1.0 if total_chars else 0.0,
        "max_chars": int(max_chars),
        "pages_total": len(list(pages)),
        "pages_covered": [page["page"] for page in pages if str(page.get("text", "")).strip()],
        "warnings": [],
    }

    if total_chars <= max_chars:
        return {"text": full_text, "report": report}

    condensed = _condense(pages, criteria, max_chars, supplier_name)
    if condensed["text"]:
        text = condensed["text"]
        report.update({
            "strategy": condensed["strategy"],
            "truncated": True,
            "chars_sent": len(text),
            "coverage": round(len(text) / total_chars, 4) if total_chars else 0.0,
            "pages_covered": condensed["pages_covered"],
            "chunks_selected": len(condensed["selected"]),
        })
        how = (
            "passages most relevant to the active criteria"
            if condensed["strategy"] == "criteria_targeted"
            else "passages sampled evenly across the document (relevance ranking found no "
                 "distinguishing passages)"
        )
        report["warnings"].append(
            f"Proposal is {total_chars:,} characters, above the {int(max_chars):,}-character budget. "
            f"{len(condensed['selected'])} {how} were sent "
            f"({report['coverage']:.0%} of the document, pages "
            f"{_format_pages(condensed['pages_covered'])}). Criteria with no selected passage may be "
            'reported as "not specified".'
        )
        return {"text": text, "report": report}

    text = full_text[:max_chars]
    report.update({
        "strategy": "head_truncated",
        "truncated": True,
        "chars_sent": len(text),
        "coverage": round(len(text) / total_chars, 4) if total_chars else 0.0,
    })
    report["warnings"].append(
        f"Proposal is {total_chars:,} characters and could not be condensed by relevance, so only "
        f"the first {len(text):,} characters ({report['coverage']:.0%}) were evaluated."
    )
    return {"text": text, "report": report}


def _format_pages(numbers: Sequence[int]) -> str:
    """Render a page list compactly: [1,2,3,7] -> '1–3, 7'."""
    if not numbers:
        return "none"
    ordered = sorted(set(numbers))
    spans: List[str] = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        spans.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = number
    spans.append(str(start) if start == previous else f"{start}–{previous}")
    return ", ".join(spans)
