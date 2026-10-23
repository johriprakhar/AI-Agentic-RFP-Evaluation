"""
Evidence grounding checks.

An LLM asked for a verbatim quote will usually supply one, but "usually" is not
a basis for a procurement decision. Every quote the model returns is checked
back against the text that was actually extracted from the PDF, and the outcome
travels with the result into the UI, the JSON export and the database.

The check is deliberately not a plain substring test. Models routinely re-wrap
lines, normalise smart quotes, collapse table whitespace or trim a trailing
clause, all of which are faithful quotations that an exact match would reject.
Matching is therefore done on word shingles:

1. Normalise both strings (case, unicode punctuation, whitespace, hyphenation).
2. Try a direct normalised substring match.                      -> "exact"
3. Otherwise measure what fraction of the quote's word 4-grams
   appear anywhere in the document.                              -> "shingle"
4. Short quotes fall back to token containment.                  -> "tokens"

Anything at or above VERIFIED_THRESHOLD counts as grounded, anything above
PARTIAL_THRESHOLD is reported as partial, and the rest is flagged as
unverifiable so a reviewer knows not to lean on it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

#: Shingle size used for fuzzy containment.
SHINGLE_SIZE = 4

#: Coverage at or above this fraction is treated as a grounded quote.
VERIFIED_THRESHOLD = 0.80

#: Coverage at or above this fraction is reported as a partial match.
PARTIAL_THRESHOLD = 0.50

#: Quotes shorter than this many words are matched by token containment.
MIN_SHINGLE_WORDS = 6

_PAGE_CITATION = re.compile(r"(?:\[?\s*page\s*|\bpg\.?\s*|\bp\.\s*)(\d{1,4})", re.IGNORECASE)

#: The prompt asks the model to append a page citation to each quote, so the
#: citation is part of the evidence string by design. It must be removed before
#: matching: left in place it contributes word shingles that appear nowhere in
#: the document and silently drags a perfectly faithful quote below the
#: verification threshold.
_PAGE_CITATION_STRIP = re.compile(
    r"[(\[{]?\s*(?:page|pg\.?|p\.)\s*\d{1,4}\s*[)\]}]?",
    re.IGNORECASE,
)

_PUNCTUATION_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2026": "...", "\u2022": " ",
}


def normalize_text(text: str) -> str:
    """Lowercase, fold unicode punctuation, de-hyphenate line breaks, collapse whitespace."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", str(text))
    for source, target in _PUNCTUATION_MAP.items():
        folded = folded.replace(source, target)
    # Re-join words split across a line break by hyphenation ("compli-\nance").
    folded = re.sub(r"-\s*\n\s*", "", folded)
    folded = folded.lower()
    # Drop the page markers the pipeline injects so they never count as evidence.
    folded = re.sub(r"\[page\s*\d+\]", " ", folded)
    folded = re.sub(r"[^a-z0-9$%./-]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _words(text: str) -> List[str]:
    return [word for word in normalize_text(text).split(" ") if word]


def _shingles(words: List[str], size: int = SHINGLE_SIZE) -> List[tuple]:
    if len(words) < size:
        return []
    return [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]


def parse_cited_page(text: str) -> Optional[int]:
    """Pull the page number a quote claims to come from, if it states one."""
    if not text:
        return None
    match = _PAGE_CITATION.search(str(text))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (ValueError, TypeError):
        return None


def strip_citation(text: str) -> str:
    """Remove page citations so only the quoted wording is matched."""
    if not text:
        return ""
    return _PAGE_CITATION_STRIP.sub(" ", str(text))


class EvidenceIndex:
    """
    A reusable, normalised index of one proposal's extracted text.

    Built once per supplier and queried once per criterion, so a six-supplier
    run with five criteria normalises each document a single time rather than
    thirty times.
    """

    def __init__(self, pages: Iterable[Dict[str, Any]]):
        self.pages: List[Dict[str, Any]] = []
        document_words: List[str] = []
        for page in pages or []:
            number = int(page.get("page", 0) or 0)
            words = _words(page.get("text", ""))
            self.pages.append({
                "page": number,
                "normalized": normalize_text(page.get("text", "")),
                "words": words,
                "shingles": set(_shingles(words)),
                "word_set": set(words),
            })
            document_words.extend(words)

        self.document_normalized = " ".join(
            page["normalized"] for page in self.pages if page["normalized"]
        )
        self.document_words = document_words
        self.document_shingles = set(_shingles(document_words))
        self.document_word_set = set(document_words)

    @property
    def word_count(self) -> int:
        return len(self.document_words)

    def _locate_page(self, quote_words: List[str], quote_shingles: List[tuple]) -> Optional[int]:
        """Return the page number with the strongest overlap for this quote."""
        best_page, best_overlap = None, 0.0
        for page in self.pages:
            if quote_shingles and page["shingles"]:
                hits = sum(1 for shingle in quote_shingles if shingle in page["shingles"])
                overlap = hits / len(quote_shingles)
            elif quote_words and page["word_set"]:
                hits = sum(1 for word in set(quote_words) if word in page["word_set"])
                overlap = hits / len(set(quote_words))
            else:
                overlap = 0.0
            if overlap > best_overlap:
                best_page, best_overlap = page["page"], overlap
        return best_page if best_overlap >= PARTIAL_THRESHOLD else None

    def verify(self, quote: str) -> Dict[str, Any]:
        """
        Check one quote against the indexed document.

        Returns a report with ``verified``, ``status`` (one of ``verified``,
        ``partial``, ``unverified``, ``empty``), the match ``method``, a
        ``coverage`` fraction, the page the quote was actually found on and the
        page it claimed to come from.
        """
        cited_page = parse_cited_page(quote)
        report: Dict[str, Any] = {
            "verified": False,
            "status": "empty",
            "method": "none",
            "coverage": 0.0,
            "located_page": None,
            "cited_page": cited_page,
            "page_matches": None,
            "note": "",
        }

        quoted_only = strip_citation(quote)
        quote_normalized = normalize_text(quoted_only)
        if not quote_normalized:
            report["note"] = "No evidence quote was supplied."
            return report

        if not self.document_normalized:
            report["status"] = "unverified"
            report["note"] = "No extractable document text was available to verify against."
            return report

        quote_words = _words(quoted_only)
        quote_shingles = _shingles(quote_words)

        # 1. Direct normalised containment.
        if quote_normalized and quote_normalized in self.document_normalized:
            report.update({
                "verified": True,
                "status": "verified",
                "method": "exact",
                "coverage": 1.0,
            })
        # 2. Shingle containment for quotes long enough to shingle.
        elif quote_shingles:
            hits = sum(1 for shingle in quote_shingles if shingle in self.document_shingles)
            coverage = hits / len(quote_shingles)
            report["coverage"] = round(coverage, 4)
            report["method"] = "shingle"
            if coverage >= VERIFIED_THRESHOLD:
                report.update({"verified": True, "status": "verified"})
            elif coverage >= PARTIAL_THRESHOLD:
                report["status"] = "partial"
                report["note"] = (
                    f"Only {coverage:.0%} of the quoted wording was found in the extracted text."
                )
            else:
                report["status"] = "unverified"
                report["note"] = (
                    f"The quote does not appear in the extracted text "
                    f"({coverage:.0%} of its wording matched)."
                )
        # 3. Token containment for very short quotes.
        else:
            unique = set(quote_words)
            hits = sum(1 for word in unique if word in self.document_word_set)
            coverage = hits / len(unique) if unique else 0.0
            report["coverage"] = round(coverage, 4)
            report["method"] = "tokens"
            if len(quote_words) < MIN_SHINGLE_WORDS and coverage >= VERIFIED_THRESHOLD:
                report.update({"verified": True, "status": "verified"})
                report["note"] = "Quote was too short for phrase matching; verified on token containment."
            elif coverage >= PARTIAL_THRESHOLD:
                report["status"] = "partial"
                report["note"] = "Quote was too short to verify as a phrase."
            else:
                report["status"] = "unverified"
                report["note"] = "The quoted terms do not appear in the extracted text."

        report["located_page"] = self._locate_page(quote_words, quote_shingles)
        if cited_page is not None and report["located_page"] is not None:
            report["page_matches"] = cited_page == report["located_page"]
            if not report["page_matches"]:
                suffix = (
                    f"Quote cites page {cited_page} but the wording was found on "
                    f"page {report['located_page']}."
                )
                report["note"] = f"{report['note']} {suffix}".strip()
        return report


def verify_criteria_evidence(rows: List[Dict[str, Any]], pages: Iterable[Dict[str, Any]]):
    """
    Annotate normalized criterion rows with evidence checks.

    Each row gains ``evidence_check`` (the full report) plus the flat
    ``evidence_verified`` / ``evidence_status`` fields the UI renders. Scores are
    never altered: an unverifiable quote is a transparency signal for the
    reviewer, not grounds for silently rewriting the model's judgement.

    Returns ``(rows, warnings, summary)``.
    """
    index = EvidenceIndex(pages)
    warnings: List[str] = []
    verified = partial = unverified = 0
    page_mismatches = 0

    for row in rows:
        if row.get("status") == "not specified":
            row["evidence_check"] = {
                "verified": None,
                "status": "not applicable",
                "method": "none",
                "coverage": 0.0,
                "located_page": None,
                "cited_page": None,
                "page_matches": None,
                "note": "Criterion was not addressed in the proposal.",
            }
            row["evidence_verified"] = None
            row["evidence_status"] = "not applicable"
            continue

        report = index.verify(row.get("evidence", ""))
        row["evidence_check"] = report
        row["evidence_verified"] = report["verified"]
        row["evidence_status"] = report["status"]

        if report["status"] == "verified":
            verified += 1
        elif report["status"] == "partial":
            partial += 1
            warnings.append(f"Evidence for {row['name']} only partially matches the document text.")
        else:
            unverified += 1
            warnings.append(
                f"Evidence for {row['name']} could not be found in the document text; "
                "treat this justification with caution."
            )
        if report.get("page_matches") is False:
            page_mismatches += 1
            warnings.append(
                f"Evidence for {row['name']} cites page {report['cited_page']} but matches text on "
                f"page {report['located_page']}."
            )

    checked = verified + partial + unverified
    summary = {
        "checked": checked,
        "verified": verified,
        "partial": partial,
        "unverified": unverified,
        "page_mismatches": page_mismatches,
        "grounding_rate": round(verified / checked, 4) if checked else None,
        "document_word_count": index.word_count,
    }
    return rows, warnings, summary
