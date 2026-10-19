"""Unit tests for the semantic search engine (chunking, BM25, line & page tracking, rankings)."""

import glob
import os
import unittest

from search_engine import (
    extract_document_chunks,
    search_without_llm,
    stem,
    tokenize,
)


class TestSearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_files = glob.glob("sample_pdfs/*.pdf")
        cls.chunks = []
        for pdf_path in cls.sample_files:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            name = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title()
            cls.chunks.extend(extract_document_chunks(pdf_bytes, pdf_path, supplier_name=name))

    def test_chunking_metadata(self):
        self.assertGreater(len(self.chunks), 0)
        for c in self.chunks:
            self.assertIn("document_name", c)
            self.assertIn("page", c)
            self.assertIn("start_line", c)
            self.assertIn("end_line", c)
            self.assertIn("text", c)
            self.assertIn("lines", c)
            self.assertGreaterEqual(c["page"], 1)
            self.assertGreaterEqual(c["start_line"], 1)
            self.assertGreaterEqual(c["end_line"], c["start_line"])
            self.assertGreater(len(c["lines"]), 0)

    def test_search_without_llm_technical_query(self):
        query = "cloud-native architecture with REST integrations and disaster recovery"
        res = search_without_llm(query, self.chunks, top_k=5)
        self.assertEqual(res["mode"], "without_llm")
        self.assertIsNotNone(res["top_document"])
        # Apex Systems describes cloud-native architecture with REST integrations and disaster recovery
        self.assertEqual(res["top_document"], "Apex Systems")
        self.assertGreater(res["top_document_score"], 80.0)

        # Top chunk verification
        top_chunk = res["results"][0]
        self.assertEqual(top_chunk["document_name"], "Apex Systems")
        self.assertEqual(top_chunk["page"], 1)
        self.assertTrue(top_chunk["start_line"] <= 8)
        self.assertIn("disaster recovery", top_chunk["text"].lower())

    def test_search_without_llm_pricing_query(self):
        query = "$240,000 fixed implementation fee annual support $36,000"
        res = search_without_llm(query, self.chunks, top_k=5)
        self.assertEqual(res["top_document"], "Apex Systems")
        top_chunk = res["results"][0]
        self.assertEqual(top_chunk["document_name"], "Apex Systems")
        self.assertIn("240,000", top_chunk["text"])
        self.assertIn("lines", top_chunk)

    def test_search_empty_query(self):
        res = search_without_llm("", self.chunks)
        self.assertEqual(len(res["results"]), 0)
        self.assertIsNone(res["top_document"])

    def test_document_rankings_included(self):
        query = "security and compliance ISO 27001 encryption"
        res = search_without_llm(query, self.chunks)
        self.assertIn("document_rankings", res)
        self.assertGreater(len(res["document_rankings"]), 0)
        self.assertIn("document_name", res["document_rankings"][0])
        self.assertIn("score", res["document_rankings"][0])


if __name__ == "__main__":
    unittest.main()

