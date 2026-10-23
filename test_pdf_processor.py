"""
Tests for PDF extraction, scanned-page diagnostics and model-input preparation.

The preparation tests are the important ones: they pin the guarantee that a
proposal is never silently truncated, because invisible truncation is what turns
a confident evaluation into a wrong one.

Runnable either way::

    python test_pdf_processor.py
    pytest test_pdf_processor.py
"""

from __future__ import annotations

import unittest

import conftest
from pdf_processor import (
    _format_pages,
    build_document_text,
    extract_document,
    extract_pdf_pages,
    prepare_document_for_llm,
)

CRITERIA = [
    {"criterion_id": 1, "name": "Technical Capability",
     "description": "Architecture, integrations, scalability and technical fit.",
     "weight": 40.0, "max_score": 10.0},
    {"criterion_id": 2, "name": "Commercial Value",
     "description": "Pricing clarity, total cost and value for money.",
     "weight": 30.0, "max_score": 10.0},
    {"criterion_id": 3, "name": "Security & Compliance",
     "description": "Security controls, certifications and auditability.",
     "weight": 30.0, "max_score": 10.0},
]


def synthetic_pages(count=3, lines_per_page=40, filler="Clause text about architecture"):
    """Build page dicts without touching a PDF."""
    pages = []
    for number in range(1, count + 1):
        raw = "\n".join(f"{filler} {number}.{line}" for line in range(1, lines_per_page + 1))
        pages.append({
            "page": number, "raw_text": raw + "\n", "text": raw,
            "char_count": len(raw), "line_count": lines_per_page, "has_text": True,
        })
    return pages


class TestExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = conftest.ensure_sample_pdfs()
        cls.apex_bytes = next(p for p in paths if p.name.startswith("Apex")).read_bytes()

    def test_pages_are_numbered_from_one(self):
        pages = extract_pdf_pages(self.apex_bytes)
        self.assertEqual([page["page"] for page in pages], [1, 2])

    def test_each_page_carries_both_raw_and_stripped_text(self):
        for page in extract_pdf_pages(self.apex_bytes):
            self.assertTrue(page["text"])
            self.assertIn("raw_text", page)
            self.assertEqual(page["text"], page["raw_text"].strip())
            self.assertTrue(page["has_text"])
            self.assertGreater(page["line_count"], 0)

    def test_extract_document_reports_a_healthy_text_pdf(self):
        report = extract_document(self.apex_bytes)
        self.assertEqual(report["page_count"], 2)
        self.assertTrue(report["has_text"])
        self.assertFalse(report["is_likely_scanned"])
        self.assertEqual(report["empty_pages"], [])
        self.assertEqual(report["warnings"], [])
        self.assertGreater(report["char_count"], 500)

    def test_document_text_carries_page_markers_for_citation(self):
        text = build_document_text(extract_pdf_pages(self.apex_bytes))
        self.assertIn("[Page 1]", text)
        self.assertIn("[Page 2]", text)
        self.assertIn("$240,000", text)

    def test_pages_without_text_are_excluded_from_the_prompt_text(self):
        pages = [
            {"page": 1, "text": "real content"},
            {"page": 2, "text": "   "},
            {"page": 3, "text": "more content"},
        ]
        text = build_document_text(pages)
        self.assertIn("[Page 1]", text)
        self.assertNotIn("[Page 2]", text)
        self.assertIn("[Page 3]", text)

    def test_invalid_pdf_bytes_raise(self):
        with self.assertRaises(Exception):
            extract_pdf_pages(b"this is not a pdf")


class TestScannedDetection(unittest.TestCase):
    """A scanned PDF must be reported, never scored as an empty proposal."""

    def _blank_pdf(self, page_count):
        import pymupdf
        document = pymupdf.open()
        for _ in range(page_count):
            document.new_page()
        data = document.tobytes()
        document.close()
        return data

    def test_an_image_only_pdf_is_flagged_as_scanned(self):
        report = extract_document(self._blank_pdf(3))
        self.assertFalse(report["has_text"])
        self.assertTrue(report["is_likely_scanned"])
        self.assertEqual(report["empty_pages"], [1, 2, 3])

    def test_mixed_text_and_blank_pages_are_reported(self):
        import pymupdf
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Substantive proposal content for the first page of the bid.")
        document.new_page()
        data = document.tobytes()
        document.close()

        report = extract_document(data)
        self.assertTrue(report["has_text"])
        self.assertEqual(report["empty_pages"], [2])
        self.assertFalse(report["is_likely_scanned"])
        self.assertTrue(any("no extractable text" in w for w in report["warnings"]))


class TestPreparation(unittest.TestCase):
    def test_a_short_document_is_sent_whole(self):
        pages = synthetic_pages(count=1, lines_per_page=5)
        prepared = prepare_document_for_llm(pages, CRITERIA, max_chars=100000)
        report = prepared["report"]
        self.assertEqual(report["strategy"], "full")
        self.assertFalse(report["truncated"])
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["chars_sent"], report["chars_total"])

    def test_an_over_long_document_is_condensed_not_cut(self):
        pages = synthetic_pages(count=6, lines_per_page=60)
        prepared = prepare_document_for_llm(pages, CRITERIA, max_chars=2000, supplier_name="Big Co")
        report = prepared["report"]
        self.assertIn(report["strategy"], {"criteria_targeted", "uniform_sample"})
        self.assertTrue(report["truncated"])
        self.assertLessEqual(report["chars_sent"], 2000)
        self.assertLess(report["coverage"], 1.0)
        self.assertGreater(report["chunks_selected"], 0)

    def test_a_uniform_document_is_sampled_across_all_pages_not_just_the_head(self):
        """
        When every passage shares the same vocabulary, BM25 has no signal to
        rank on. Falling back to an even sample still reaches the later pages,
        which blind head truncation never would.
        """
        pages = synthetic_pages(count=6, lines_per_page=60)
        prepared = prepare_document_for_llm(pages, CRITERIA, max_chars=2000)
        report = prepared["report"]
        self.assertEqual(report["strategy"], "uniform_sample")
        self.assertGreater(len(report["pages_covered"]), 1)
        self.assertIn(6, report["pages_covered"])
        self.assertTrue(any("sampled evenly" in w for w in report["warnings"]))

    def test_truncation_is_always_warned_about(self):
        """The core guarantee: nothing is dropped silently."""
        pages = synthetic_pages(count=6, lines_per_page=60)
        prepared = prepare_document_for_llm(pages, CRITERIA, max_chars=2000)
        warnings = prepared["report"]["warnings"]
        self.assertTrue(warnings)
        self.assertIn("above the", warnings[0])
        self.assertIn("% of the document", warnings[0])
        self.assertTrue(
            "most relevant" in warnings[0] or "sampled evenly" in warnings[0], warnings[0]
        )

    def test_the_assembled_digest_respects_the_character_budget(self):
        """Markers and the header are billed against the budget, not added on top."""
        for max_chars in (1200, 2000, 5000):
            pages = synthetic_pages(count=6, lines_per_page=60)
            report = prepare_document_for_llm(pages, CRITERIA, max_chars=max_chars)["report"]
            self.assertLessEqual(report["chars_sent"], max_chars, f"budget {max_chars}")

    def test_a_condensed_document_tells_the_model_it_is_partial(self):
        pages = synthetic_pages(count=6, lines_per_page=60)
        text = prepare_document_for_llm(pages, CRITERIA, max_chars=2000)["text"]
        self.assertIn("exceeded the model input budget", text)
        self.assertIn("not specified", text)

    def test_a_condensed_document_keeps_page_markers_and_line_ranges(self):
        pages = synthetic_pages(count=6, lines_per_page=60)
        text = prepare_document_for_llm(pages, CRITERIA, max_chars=3000)["text"]
        self.assertIn("[Page ", text)
        self.assertIn("(lines ", text)

    def test_omitted_regions_are_marked(self):
        pages = synthetic_pages(count=4, lines_per_page=80)
        text = prepare_document_for_llm(pages, CRITERIA, max_chars=1500)["text"]
        self.assertIn("passage omitted", text)

    def test_condensation_covers_content_for_several_criteria(self):
        pages = [{
            "page": 1,
            "raw_text": "\n".join([
                "The architecture is cloud-native with Kubernetes and REST integrations.",
                "Scalability targets ten thousand concurrent users across regions.",
                "Pricing is a fixed fee of $240,000 with annual support of $36,000.",
                "Total cost of ownership is itemised with all assumptions stated.",
                "Security controls include ISO 27001 certification and AES-256 encryption.",
                "Audit logs are immutable and retained for seven years for compliance.",
            ] + [f"Filler sentence number {i} with no bearing on any criterion." for i in range(200)]),
        }]
        pages[0]["text"] = pages[0]["raw_text"]
        prepared = prepare_document_for_llm(pages, CRITERIA, max_chars=900)
        text = prepared["text"].lower()
        self.assertEqual(prepared["report"]["strategy"], "criteria_targeted")
        self.assertIn("architecture", text)
        self.assertIn("240,000", text)
        self.assertIn("iso 27001", text)

    def test_without_criteria_an_over_long_document_falls_back_to_head_truncation(self):
        pages = synthetic_pages(count=6, lines_per_page=60)
        prepared = prepare_document_for_llm(pages, criteria=None, max_chars=1500)
        report = prepared["report"]
        self.assertEqual(report["strategy"], "head_truncated")
        self.assertTrue(report["truncated"])
        self.assertTrue(any("could not be condensed" in w for w in report["warnings"]))

    def test_an_empty_document_reports_zero_coverage_without_raising(self):
        prepared = prepare_document_for_llm([], CRITERIA, max_chars=1000)
        self.assertEqual(prepared["text"], "")
        self.assertEqual(prepared["report"]["coverage"], 0.0)
        self.assertFalse(prepared["report"]["truncated"])

    def test_coverage_is_reported_as_a_fraction_between_zero_and_one(self):
        for max_chars in (500, 2000, 10000, 100000):
            pages = synthetic_pages(count=4, lines_per_page=40)
            report = prepare_document_for_llm(pages, CRITERIA, max_chars=max_chars)["report"]
            self.assertGreaterEqual(report["coverage"], 0.0)
            self.assertLessEqual(report["coverage"], 1.0)


class TestPageFormatting(unittest.TestCase):
    def test_contiguous_pages_collapse_into_ranges(self):
        self.assertEqual(_format_pages([1, 2, 3, 7]), "1–3, 7")
        self.assertEqual(_format_pages([2]), "2")
        self.assertEqual(_format_pages([5, 3, 4]), "3–5")
        self.assertEqual(_format_pages([1, 3, 5]), "1, 3, 5")

    def test_an_empty_list_reads_as_none(self):
        self.assertEqual(_format_pages([]), "none")


if __name__ == "__main__":
    unittest.main(verbosity=2)
