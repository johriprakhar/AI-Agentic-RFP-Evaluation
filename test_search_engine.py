"""
Tests for chunking, page/line citation accuracy, BM25 ranking and passage selection.

These run entirely offline against the synthetic proposals, which `conftest`
generates on demand, so the suite needs no API key and no network.

Runnable either way::

    python test_search_engine.py
    pytest test_search_engine.py
"""

from __future__ import annotations

import os
import unittest

import conftest
from retrieval import (
    BM25Index,
    aggregate_document_scores,
    chunks_from_pages,
    extract_document_chunks,
    search_without_llm,
    select_relevant_chunks,
    stem,
    tokenize,
)

CRITERIA = [
    {"criterion_id": 1, "name": "Technical Capability",
     "description": "Architecture, integrations, scalability, and technical fit.",
     "weight": 30.0, "max_score": 10.0},
    {"criterion_id": 2, "name": "Commercial Value",
     "description": "Pricing clarity, total cost, assumptions, and value for money.",
     "weight": 20.0, "max_score": 10.0},
    {"criterion_id": 3, "name": "Security & Compliance",
     "description": "Security controls, certifications, privacy, and auditability.",
     "weight": 50.0, "max_score": 10.0},
]


class SampleCorpus(unittest.TestCase):
    """Shared fixture: every sample proposal chunked once."""

    @classmethod
    def setUpClass(cls):
        cls.paths = conftest.ensure_sample_pdfs()
        cls.chunks = []
        for path in cls.paths:
            name = os.path.splitext(path.name)[0].replace("_", " ").title()
            cls.chunks.extend(
                extract_document_chunks(path.read_bytes(), str(path), supplier_name=name)
            )
        cls.documents = sorted({chunk["document_name"] for chunk in cls.chunks})


class TestTokenizer(unittest.TestCase):
    def test_stop_words_and_single_characters_are_dropped(self):
        self.assertEqual(tokenize("The cost of a plan is high"), ["cost", "plan", "high"])

    def test_currency_and_identifiers_survive_tokenization(self):
        tokens = tokenize("Fixed fee: $240,000 under ISO-27001 and TLS 1.3")
        self.assertIn("$240", tokens)
        self.assertIn("000", tokens)
        self.assertIn("iso", tokens)
        self.assertIn("27001", tokens)

    def test_stemming_merges_morphological_variants(self):
        self.assertEqual(stem("integrations"), stem("integration"))
        self.assertEqual(stem("requirements"), stem("requirement"))

    def test_stemming_leaves_words_too_short_to_stem_alone(self):
        """The length guard stops the suffix stripper from mangling short words."""
        self.assertEqual(stem("is"), "is")
        self.assertEqual(stem("gas"), "gas")
        self.assertEqual(stem("fees"), "fee")

    def test_empty_and_none_inputs_are_safe(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])


class TestChunking(SampleCorpus):
    def test_every_chunk_carries_complete_citation_metadata(self):
        self.assertGreater(len(self.chunks), 0)
        for chunk in self.chunks:
            for key in ("chunk_id", "document_name", "filename", "page",
                        "start_line", "end_line", "text", "lines"):
                self.assertIn(key, chunk)
            self.assertGreaterEqual(chunk["page"], 1)
            self.assertGreaterEqual(chunk["start_line"], 1)
            self.assertGreaterEqual(chunk["end_line"], chunk["start_line"])
            self.assertTrue(chunk["text"].strip())
            self.assertTrue(chunk["lines"])
            for line in chunk["lines"]:
                self.assertGreaterEqual(line["line_num"], 1)
                self.assertTrue(line["text"].strip())

    def test_chunk_ids_are_unique(self):
        ids = [chunk["chunk_id"] for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_line_numbers_match_the_generator_layout_exactly(self):
        """
        The citation contract: page 1 line 4 of the Apex fixture is the
        cloud-native architecture sentence. If chunking ever shifts line
        numbering, this fails rather than silently mis-citing evidence.
        """
        apex = [c for c in self.chunks if c["document_name"] == "Apex Systems" and c["page"] == 1]
        lines = {line["line_num"]: line["text"] for chunk in apex for line in chunk["lines"]}
        self.assertIn("cloud-native architecture", lines[4])
        self.assertIn("REST APIs", lines[5])
        self.assertIn("disaster recovery", lines[7].lower())
        self.assertIn("$240,000", lines[14])
        self.assertIn("$36,000", lines[15])

    def test_windows_overlap_so_context_is_not_split(self):
        apex = [c for c in self.chunks if c["document_name"] == "Apex Systems" and c["page"] == 1]
        starts = sorted(chunk["start_line"] for chunk in apex)
        self.assertEqual(starts[0], 1)
        self.assertLess(starts[1], apex[0]["end_line"] + 1)

    def test_pages_without_text_are_skipped(self):
        pages = [
            {"page": 1, "raw_text": "   \n\n  \n"},
            {"page": 2, "raw_text": "Real content on the second page\nSecond line here\n"},
        ]
        chunks = chunks_from_pages(pages, "doc.pdf", supplier_name="Doc")
        self.assertEqual({chunk["page"] for chunk in chunks}, {2})

    def test_line_numbers_come_from_raw_text_not_stripped_text(self):
        """Stripping a page before indexing would shift every line number."""
        pages = [{"page": 1, "raw_text": "\n\n\nFirst visible line\n", "text": "First visible line"}]
        chunks = chunks_from_pages(pages, "doc.pdf", supplier_name="Doc")
        self.assertEqual(chunks[0]["start_line"], 4)

    def test_empty_document_yields_no_chunks(self):
        self.assertEqual(chunks_from_pages([], "doc.pdf"), [])


class TestBM25Search(SampleCorpus):
    def test_technical_query_ranks_apex_first_with_exact_citation(self):
        found = search_without_llm(
            "cloud-native architecture with REST integrations and disaster recovery",
            self.chunks, top_k=5,
        )
        self.assertEqual(found["mode"], "without_llm")
        self.assertEqual(found["top_document"], "Apex Systems")
        self.assertGreater(found["top_document_score"], 80.0)

        top = found["results"][0]
        self.assertEqual(top["document_name"], "Apex Systems")
        self.assertEqual(top["page"], 1)
        self.assertLessEqual(top["start_line"], 8)
        self.assertIn("disaster recovery", top["text"].lower())
        self.assertTrue(top["matched_terms"])

    def test_pricing_query_retrieves_the_exact_figures(self):
        found = search_without_llm(
            "$240,000 fixed implementation fee annual support $36,000", self.chunks, top_k=5,
        )
        self.assertEqual(found["top_document"], "Apex Systems")
        top = found["results"][0]
        self.assertIn("240,000", top["text"])
        self.assertEqual(top["page"], 1)
        self.assertTrue(any(line["line_num"] == 14 for line in top["lines"]))

    def test_scores_are_descending_and_bounded(self):
        found = search_without_llm("implementation timeline milestones rollback", self.chunks, top_k=10)
        scores = [row["score"] for row in found["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 < score <= 100 for score in scores))

    def test_top_k_is_respected(self):
        found = search_without_llm("support", self.chunks, top_k=2)
        self.assertLessEqual(len(found["results"]), 2)

    def test_document_rankings_list_every_document(self):
        found = search_without_llm("ISO 27001 certification", self.chunks)
        ranked = {row["document_name"] for row in found["document_rankings"]}
        self.assertEqual(ranked, set(self.documents))
        scores = [row["score"] for row in found["document_rankings"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_document_filtering_restricts_results(self):
        subset = [c for c in self.chunks if c["document_name"] == "Nexaworks"]
        found = search_without_llm("ISO 27001 encryption audit logs", subset, top_k=5)
        self.assertEqual(found["top_document"], "Nexaworks")
        self.assertTrue(all(r["document_name"] == "Nexaworks" for r in found["results"]))

    def test_empty_blank_and_stopword_queries_return_nothing(self):
        for query in ["", "   ", "\n", "the and of a"]:
            found = search_without_llm(query, self.chunks)
            self.assertEqual(found["results"], [], msg=f"query {query!r}")
            self.assertIsNone(found["top_document"])

    def test_query_with_no_corpus_match_returns_nothing(self):
        found = search_without_llm("zzzqqq nonexistent terminology xylophone", self.chunks)
        self.assertEqual(found["results"], [])
        self.assertIsNone(found["top_document"])

    def test_search_over_an_empty_corpus_is_safe(self):
        found = search_without_llm("pricing", [])
        self.assertEqual(found["results"], [])
        self.assertEqual(found["document_rankings"], [])

    def test_lexical_search_cannot_detect_negation(self):
        """
        A documented limitation, pinned by a test so it stays a known property
        rather than a surprise. BrightPath states it is *not* ISO 27001
        certified, yet shares the query's vocabulary, so purely lexical
        retrieval scores it highly. Resolving that is exactly what LLM mode is
        for.
        """
        found = search_without_llm("ISO 27001 certified", self.chunks, top_k=6)
        matched = {row["document_name"] for row in found["results"]}
        self.assertIn("Brightpath Tech", matched)
        brightpath = next(r for r in found["results"] if r["document_name"] == "Brightpath Tech")
        self.assertIn("not currently iso 27001", brightpath["text"].lower())


class TestBM25Index(SampleCorpus):
    def test_index_statistics_are_computed_once_and_reused(self):
        index = BM25Index(self.chunks)
        self.assertEqual(len(index), len(self.chunks))
        self.assertGreater(index.avg_len, 0)
        self.assertTrue(index.idf)
        first = index.rank("disaster recovery", top_k=3)
        second = index.rank("disaster recovery", top_k=3)
        self.assertEqual([r["chunk_id"] for r in first], [r["chunk_id"] for r in second])

    def test_rarer_terms_carry_more_weight(self):
        index = BM25Index(self.chunks)
        rare = index.idf.get(stem("kubernetes"))
        common = index.idf.get(stem("support"))
        self.assertIsNotNone(rare)
        self.assertIsNotNone(common)
        self.assertGreater(rare, common)

    def test_empty_index_scores_nothing(self):
        index = BM25Index([])
        self.assertEqual(index.score("anything"), [])
        self.assertEqual(index.rank("anything"), [])


class TestAggregation(SampleCorpus):
    def test_documents_with_no_match_score_zero_but_still_appear(self):
        scored = [row for row in search_without_llm(
            "Kubernetes availability zones", self.chunks, top_k=50)["results"]]
        rankings, top = aggregate_document_scores(scored, self.chunks)
        self.assertEqual(len(rankings), len(self.documents))
        self.assertIsNotNone(top)
        zeros = [row for row in rankings if row["match_count"] == 0]
        self.assertTrue(all(row["score"] == 0.0 for row in zeros))

    def test_no_matches_means_no_top_document(self):
        rankings, top = aggregate_document_scores([], self.chunks)
        self.assertIsNone(top)
        self.assertTrue(all(row["score"] == 0.0 for row in rankings))

    def test_breadth_breaks_a_tie_between_equal_peaks(self):
        """
        Composite = 0.7 x peak + 0.3 x mean of the top three. The peak dominates,
        so breadth is a tie-breaker between comparable documents rather than an
        override: with identical best passages, the document with more
        supporting passages wins.
        """
        all_chunks = [
            {"document_name": "Broad", "text": "a"}, {"document_name": "Narrow", "text": "b"},
        ]
        scored = [
            {"document_name": "Broad", "score": 90.0},
            {"document_name": "Broad", "score": 88.0},
            {"document_name": "Broad", "score": 86.0},
            {"document_name": "Narrow", "score": 90.0},
        ]
        rankings, top = aggregate_document_scores(scored, all_chunks)
        self.assertEqual(top["document_name"], "Broad")
        self.assertGreater(rankings[0]["score"], rankings[1]["score"])

    def test_a_higher_peak_still_outranks_more_weak_passages(self):
        all_chunks = [
            {"document_name": "Deep", "text": "a"}, {"document_name": "Wide", "text": "b"},
        ]
        scored = [
            {"document_name": "Deep", "score": 98.0},
            {"document_name": "Wide", "score": 40.0},
            {"document_name": "Wide", "score": 38.0},
            {"document_name": "Wide", "score": 36.0},
        ]
        _, top = aggregate_document_scores(scored, all_chunks)
        self.assertEqual(top["document_name"], "Deep")


class TestPassageSelection(SampleCorpus):
    def test_selection_stays_within_the_character_budget(self):
        apex = [c for c in self.chunks if c["document_name"] == "Apex Systems"]
        selected = select_relevant_chunks(
            apex, [f"{c['name']} {c['description']}" for c in CRITERIA], budget_chars=600,
        )
        self.assertTrue(selected)
        self.assertLessEqual(sum(len(c["text"]) + 1 for c in selected), 600)

    def test_selection_is_returned_in_reading_order(self):
        apex = [c for c in self.chunks if c["document_name"] == "Apex Systems"]
        selected = select_relevant_chunks(
            apex, [f"{c['name']} {c['description']}" for c in CRITERIA], budget_chars=4000,
        )
        order = [(c["page"], c["start_line"]) for c in selected]
        self.assertEqual(order, sorted(order))

    def test_interleaving_represents_every_criterion(self):
        """
        A tight budget must not be consumed by one verbose criterion. With one
        passage's worth of room per criterion, each criterion's best passage
        should be represented.
        """
        apex = [c for c in self.chunks if c["document_name"] == "Apex Systems"]
        queries = [f"{c['name']} {c['description']}" for c in CRITERIA]
        selected = select_relevant_chunks(apex, queries, budget_chars=1200)
        joined = " ".join(c["text"] for c in selected).lower()
        self.assertIn("architecture", joined)
        self.assertTrue("$240,000" in joined or "cost" in joined)
        self.assertTrue("iso 27001" in joined or "encrypted" in joined)

    def test_no_chunks_or_no_budget_selects_nothing(self):
        self.assertEqual(select_relevant_chunks([], ["anything"], 1000), [])
        self.assertEqual(select_relevant_chunks(self.chunks, ["pricing"], 0), [])

    def test_a_chunk_larger_than_the_budget_still_returns_the_best_one(self):
        selected = select_relevant_chunks(self.chunks, ["fixed implementation fee"], budget_chars=10)
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
