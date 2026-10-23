"""
Tests for evidence grounding: quote verification, page citation checks and the
per-supplier grounding summary.

Runnable either way::

    python test_evidence.py
    pytest test_evidence.py
"""

from __future__ import annotations

import unittest

import conftest
from evidence import (
    EvidenceIndex,
    normalize_text,
    parse_cited_page,
    verify_criteria_evidence,
)
from pdf_processor import extract_pdf_pages

PAGES = [
    {"page": 1, "text":
        "Apex proposes a cloud-native architecture on Kubernetes across three availability zones.\n"
        "All integrations are delivered as versioned REST APIs with an event bus for batch loads.\n"
        "Fixed implementation fee: $240,000 covering design, build, migration and go-live support."},
    {"page": 2, "text":
        "Apex holds ISO 27001 certification, renewed March 2026, and SOC 2 Type II attestation.\n"
        "Immutable audit logs capture every privileged action and are retained for seven years."},
]


def rows(*specs):
    """Build minimal normalized criterion rows for the summary tests."""
    return [{
        "criterion_id": index + 1,
        "name": name,
        "status": status,
        "evidence": quote,
        "score": 5.0,
        "max_score": 10.0,
        "weight": 100.0 / len(specs),
    } for index, (name, status, quote) in enumerate(specs)]


class TestNormalization(unittest.TestCase):
    def test_case_and_whitespace_are_folded(self):
        self.assertEqual(normalize_text("  Fixed   FEE\n\n of the plan "), "fixed fee of the plan")

    def test_thousands_separators_are_folded_consistently_on_both_sides(self):
        """
        Separators are dropped rather than preserved. What matters is that the
        quote and the document are folded identically, so "$240,000" still
        matches "$240,000" in the source.
        """
        self.assertEqual(normalize_text("$240,000"), normalize_text("$240,000"))
        self.assertIn(normalize_text("$240,000"), normalize_text("fee of $240,000 total"))

    def test_smart_punctuation_is_folded(self):
        self.assertEqual(
            normalize_text("\u201cISO 27001\u201d \u2014 renewed"),
            normalize_text('"ISO 27001" - renewed'),
        )

    def test_hyphenation_across_a_line_break_is_rejoined(self):
        self.assertIn("compliance", normalize_text("compli-\nance is maintained"))

    def test_injected_page_markers_are_stripped(self):
        self.assertEqual(normalize_text("[Page 3] the fee applies"), "the fee applies")

    def test_empty_inputs_are_safe(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")


class TestPageCitationParsing(unittest.TestCase):
    def test_common_citation_forms_are_recognised(self):
        for text, expected in [
            ("quoted text (page 2)", 2), ("quoted text, Page 11", 11),
            ("[Page 4] quoted", 4), ("see p. 7", 7), ("see pg 3", 3),
            ("PAGE 5 reference", 5),
        ]:
            self.assertEqual(parse_cited_page(text), expected, text)

    def test_absent_or_unparseable_citations_return_none(self):
        for text in ["no citation here", "", None, "page number unknown"]:
            self.assertIsNone(parse_cited_page(text))


class TestQuoteVerification(unittest.TestCase):
    def setUp(self):
        self.index = EvidenceIndex(PAGES)

    def test_an_exact_quote_is_verified(self):
        report = self.index.verify("Fixed implementation fee: $240,000")
        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["method"], "exact")
        self.assertEqual(report["coverage"], 1.0)

    def test_a_rewrapped_quote_is_verified(self):
        """Models re-wrap lines; that is still a faithful quotation."""
        report = self.index.verify(
            "All integrations are delivered\nas versioned REST APIs\nwith an event bus for batch loads."
        )
        self.assertTrue(report["verified"])

    def test_a_quote_with_smart_punctuation_is_verified(self):
        report = self.index.verify("Apex holds ISO 27001 certification, renewed March 2026")
        self.assertTrue(report["verified"])

    def test_a_quote_with_a_trimmed_clause_is_verified(self):
        report = self.index.verify(
            "cloud-native architecture on Kubernetes across three availability zones"
        )
        self.assertTrue(report["verified"])

    def test_a_fabricated_quote_is_not_verified(self):
        report = self.index.verify(
            "Apex guarantees a dedicated Tier 4 datacentre in Reykjavik with free hardware."
        )
        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "unverified")
        self.assertIn("does not appear", report["note"])

    def test_a_partly_invented_quote_is_reported_as_partial_or_unverified(self):
        report = self.index.verify(
            "Immutable audit logs capture every privileged action and are retained "
            "for seven years under our FedRAMP High authorisation and quantum key vault."
        )
        self.assertIn(report["status"], {"partial", "unverified"})
        self.assertFalse(report["verified"])

    def test_the_page_a_quote_came_from_is_located(self):
        report = self.index.verify("ISO 27001 certification, renewed March 2026")
        self.assertEqual(report["located_page"], 2)

    def test_a_correct_page_citation_is_confirmed(self):
        report = self.index.verify("Immutable audit logs capture every privileged action (page 2)")
        self.assertTrue(report["verified"])
        self.assertEqual(report["cited_page"], 2)
        self.assertTrue(report["page_matches"])

    def test_a_wrong_page_citation_is_flagged(self):
        report = self.index.verify("Immutable audit logs capture every privileged action (page 1)")
        self.assertTrue(report["verified"])
        self.assertFalse(report["page_matches"])
        self.assertIn("cites page 1", report["note"])

    def test_a_short_quote_is_still_verified(self):
        """Too short to shingle, so it falls through to substring or token matching."""
        report = self.index.verify("$240,000")
        self.assertTrue(report["verified"])
        self.assertIn(report["method"], {"exact", "tokens"})

    def test_a_citation_appended_to_a_quote_does_not_break_verification(self):
        """
        The prompt asks for "quote (page N)", so the citation is part of the
        string. Leaving it in the match text would invent word shingles that
        appear nowhere in the document.
        """
        bare = self.index.verify("All integrations are delivered as versioned REST APIs")
        cited = self.index.verify("All integrations are delivered as versioned REST APIs (page 1)")
        self.assertTrue(bare["verified"])
        self.assertTrue(cited["verified"])
        self.assertEqual(cited["coverage"], bare["coverage"])
        self.assertEqual(cited["cited_page"], 1)

    def test_a_short_invented_quote_is_not_verified(self):
        report = self.index.verify("zzqq xylophone")
        self.assertFalse(report["verified"])

    def test_an_empty_quote_is_reported_as_empty(self):
        for value in ["", "   ", None]:
            report = self.index.verify(value)
            self.assertEqual(report["status"], "empty")
            self.assertFalse(report["verified"])

    def test_an_empty_document_cannot_verify_anything(self):
        empty = EvidenceIndex([])
        report = empty.verify("any quote at all")
        self.assertFalse(report["verified"])
        self.assertEqual(report["status"], "unverified")
        self.assertIn("No extractable document text", report["note"])

    def test_word_count_reflects_the_indexed_document(self):
        self.assertGreater(EvidenceIndex(PAGES).word_count, 20)
        self.assertEqual(EvidenceIndex([]).word_count, 0)


class TestCriteriaEvidenceSummary(unittest.TestCase):
    def test_grounded_quotes_produce_a_clean_summary(self):
        annotated, warnings, summary = verify_criteria_evidence(
            rows(
                ("Technical", "specified", "cloud-native architecture on Kubernetes"),
                ("Security", "specified", "ISO 27001 certification, renewed March 2026"),
            ),
            PAGES,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(summary["checked"], 2)
        self.assertEqual(summary["verified"], 2)
        self.assertEqual(summary["grounding_rate"], 1.0)
        self.assertTrue(all(row["evidence_verified"] for row in annotated))

    def test_an_ungrounded_quote_is_warned_about_without_changing_the_score(self):
        original = rows(("Technical", "specified", "We promise a free quantum datacentre."))
        annotated, warnings, summary = verify_criteria_evidence(original, PAGES)
        self.assertEqual(summary["unverified"], 1)
        self.assertEqual(summary["grounding_rate"], 0.0)
        self.assertTrue(any("could not be found" in w for w in warnings))
        self.assertEqual(annotated[0]["score"], 5.0)  # scores are never rewritten

    def test_not_specified_rows_are_skipped_as_not_applicable(self):
        annotated, warnings, summary = verify_criteria_evidence(
            rows(("Commercial", "not specified", "")), PAGES,
        )
        self.assertEqual(summary["checked"], 0)
        self.assertIsNone(summary["grounding_rate"])
        self.assertEqual(annotated[0]["evidence_status"], "not applicable")
        self.assertIsNone(annotated[0]["evidence_verified"])
        self.assertEqual(warnings, [])

    def test_a_page_mismatch_is_counted_and_warned(self):
        _, warnings, summary = verify_criteria_evidence(
            rows(("Security", "specified",
                  "Immutable audit logs capture every privileged action (page 1)")),
            PAGES,
        )
        self.assertEqual(summary["page_mismatches"], 1)
        self.assertTrue(any("cites page 1" in w for w in warnings))

    def test_a_specified_row_with_no_quote_is_reported(self):
        _, warnings, summary = verify_criteria_evidence(
            rows(("Technical", "specified", "")), PAGES,
        )
        self.assertEqual(summary["unverified"] + summary["partial"], 1)
        self.assertTrue(warnings)

    def test_mixed_outcomes_are_counted_correctly(self):
        _, _, summary = verify_criteria_evidence(
            rows(
                ("A", "specified", "Fixed implementation fee: $240,000"),
                ("B", "specified", "invented guarantee of free hardware forever"),
                ("C", "not specified", ""),
            ),
            PAGES,
        )
        self.assertEqual(summary["checked"], 2)
        self.assertEqual(summary["verified"], 1)
        self.assertEqual(summary["grounding_rate"], 0.5)


class TestAgainstRealPdf(unittest.TestCase):
    """Verification must work on text as PyMuPDF actually extracts it."""

    @classmethod
    def setUpClass(cls):
        paths = conftest.ensure_sample_pdfs()
        apex = next(path for path in paths if path.name.startswith("Apex"))
        cls.pages = extract_pdf_pages(apex.read_bytes())
        cls.index = EvidenceIndex(cls.pages)

    def test_a_real_quote_from_the_pdf_verifies(self):
        report = self.index.verify("Annual support cost: $36,000 per year for three years")
        self.assertTrue(report["verified"], report)
        self.assertEqual(report["located_page"], 1)

    def test_a_quote_from_page_two_locates_page_two(self):
        report = self.index.verify(
            "Annual penetration testing is performed by an external CREST-accredited assessor"
        )
        self.assertTrue(report["verified"], report)
        self.assertEqual(report["located_page"], 2)

    def test_a_plausible_but_absent_claim_is_rejected(self):
        report = self.index.verify(
            "Apex provides unlimited free consultancy days for the lifetime of the contract"
        )
        self.assertFalse(report["verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
