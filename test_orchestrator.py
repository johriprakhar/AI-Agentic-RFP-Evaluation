"""
End-to-end orchestration tests.

The model is replaced by a scripted stub, so the whole pipeline runs offline:
real PDFs in, real extraction, real evidence verification, real arithmetic, real
SQLite persistence, with only the judgement step faked. That is exactly the
separation the architecture claims, and these tests are what prove it: if any
number depended on the model, stubbing the model would change the numbers.

Runnable either way::

    python test_orchestrator.py
    pytest test_orchestrator.py
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import conftest
import database
import evaluator
import orchestrator
import tools

CRITERIA = [
    {"criterion_id": 1, "name": "Technical Capability",
     "description": "Architecture, integrations, scalability and technical fit.",
     "weight": 50.0, "max_score": 10.0},
    {"criterion_id": 2, "name": "Commercial Value",
     "description": "Pricing clarity, total cost and value for money.",
     "weight": 30.0, "max_score": 10.0},
    {"criterion_id": 3, "name": "Security & Compliance",
     "description": "Security controls, certifications and auditability.",
     "weight": 20.0, "max_score": 10.0},
]

LLM_CONFIG = {
    "provider": "StubProvider", "model": "stub-model", "api_key": "test-key",
    "temperature": 0.0, "max_document_chars": 70000, "ready": True,
}

#: Quotes that genuinely appear in the generated fixtures.
REAL_QUOTES = {
    "Apex Systems": {
        1: "Apex proposes a cloud-native architecture on Kubernetes across three availability zones.",
        2: "Fixed implementation fee: $240,000 covering design, build, migration and go-live support.",
        3: "Apex holds ISO 27001 certification, renewed March 2026, and SOC 2 Type II attestation.",
    },
    "BrightPath Tech": {
        1: "BrightPath delivers a containerised microservices platform hosted on managed infrastructure.",
        2: "Fixed implementation fee: $155,000, the lowest total cost offered in this submission.",
        3: "BrightPath is not currently ISO 27001 certified; certification is targeted for late 2027.",
    },
}

SCORES = {
    "Apex Systems": {1: 9.0, 2: 7.0, 3: 9.0},
    "BrightPath Tech": {1: 6.0, 2: 9.0, 3: 3.0},
}


class StubModel:
    """Stands in for `evaluator.chat_json_validated`."""

    def __init__(self, scores=None, quotes=None, overrides=None, diagnostics=None):
        self.scores = scores or SCORES
        self.quotes = quotes or REAL_QUOTES
        self.overrides = overrides or {}
        self.diagnostics = diagnostics or {}
        self.prompts = []

    def __call__(self, llm_config, system_prompt, user_prompt, validator=None, max_attempts=3):
        self.prompts.append(user_prompt)
        match = re.search(r"SUPPLIER:\s*(.+)", user_prompt)
        supplier = match.group(1).strip() if match else "Unknown"

        if supplier in self.overrides:
            payload = self.overrides[supplier]
        else:
            scores = self.scores.get(supplier, {1: 5.0, 2: 5.0, 3: 5.0})
            quotes = self.quotes.get(supplier, {})
            payload = {
                "criteria": [{
                    "criterion_id": cid,
                    "score": score,
                    "status": "specified",
                    "justification": f"Stubbed judgement for criterion {cid}.",
                    "evidence": quotes.get(cid, ""),
                } for cid, score in sorted(scores.items())],
                "risks": [f"Stub risk for {supplier}."],
                "overall_summary": f"Stub summary for {supplier}.",
            }

        diagnostics = {
            "attempts": [], "attempt_count": 1, "repaired": False,
            "schema_failed": False, "schema_warnings": [],
        }
        diagnostics.update(self.diagnostics)
        return {"payload": payload, "diagnostics": diagnostics}


class OrchestratorCase(unittest.TestCase):
    """Fresh database per test; proposals loaded from the generated fixtures."""

    @classmethod
    def setUpClass(cls):
        paths = conftest.ensure_sample_pdfs()
        # Keyed on the stem with underscores replaced, preserving the original
        # capitalisation ("BrightPath_Tech" -> "BrightPath Tech").
        cls.pdf_bytes = {path.stem.replace("_", " "): path.read_bytes() for path in paths}

    def setUp(self):
        self._previous = database.get_db_path()
        self._dir = tempfile.TemporaryDirectory()
        database.set_db_path(Path(self._dir.name) / "orchestrator.db")
        database.init_db()

    def tearDown(self):
        database.set_db_path(self._previous)
        self._dir.cleanup()

    def proposals(self, names=("Apex Systems", "BrightPath Tech")):
        return [{
            "supplier_name": name,
            "submission_date": f"2026-01-0{index + 1}",
            "experience_rating": 5.0 + index,
            "source_filename": f"{name.replace(' ', '_')}.pdf",
            "bytes": self.pdf_bytes[name],
        } for index, name in enumerate(names)]

    def run_pipeline(self, proposals=None, stub=None, **kwargs):
        stub = stub or StubModel()
        with patch.object(evaluator, "chat_json_validated", stub):
            outcome = orchestrator.run_evaluation(
                proposals if proposals is not None else self.proposals(),
                CRITERIA, LLM_CONFIG, **kwargs,
            )
        return outcome, stub


class TestInputValidation(unittest.TestCase):
    def good(self):
        return [{"supplier_name": "A", "submission_date": "2026-01-01",
                 "experience_rating": 5.0, "bytes": b"%PDF-"}]

    def test_valid_inputs_produce_no_problems(self):
        self.assertEqual(orchestrator.validate_run_inputs(self.good(), CRITERIA), [])

    def test_no_proposals_is_reported(self):
        problems = orchestrator.validate_run_inputs([], CRITERIA)
        self.assertTrue(any("at least one proposal" in p for p in problems))

    def test_no_criteria_is_reported(self):
        problems = orchestrator.validate_run_inputs(self.good(), [])
        self.assertTrue(any("No active criteria" in p for p in problems))

    def test_weights_not_totalling_100_are_reported(self):
        broken = [dict(CRITERIA[0], weight=10.0)]
        problems = orchestrator.validate_run_inputs(self.good(), broken)
        self.assertTrue(any("not 100%" in p for p in problems))

    def test_duplicate_supplier_names_are_reported_case_insensitively(self):
        proposals = self.good() * 2
        proposals[1] = dict(proposals[1], supplier_name="a")
        problems = orchestrator.validate_run_inputs(proposals, CRITERIA)
        self.assertTrue(any("unique within a run" in p for p in problems))

    def test_a_blank_supplier_name_is_reported(self):
        proposals = [dict(self.good()[0], supplier_name="   ")]
        problems = orchestrator.validate_run_inputs(proposals, CRITERIA)
        self.assertTrue(any("needs a name" in p for p in problems))

    def test_a_missing_pdf_is_reported(self):
        proposals = [dict(self.good()[0], bytes=b"")]
        problems = orchestrator.validate_run_inputs(proposals, CRITERIA)
        self.assertTrue(any("needs an uploaded PDF" in p for p in problems))

    def test_too_many_proposals_are_reported(self):
        proposals = [dict(self.good()[0], supplier_name=f"S{i}") for i in range(7)]
        problems = orchestrator.validate_run_inputs(proposals, CRITERIA)
        self.assertTrue(any("no more than 6" in p for p in problems))

    def test_duplicate_criterion_ids_are_reported(self):
        duplicated = [CRITERIA[0], dict(CRITERIA[1], criterion_id=1, weight=50.0)]
        problems = orchestrator.validate_run_inputs(self.good(), duplicated)
        self.assertTrue(any("duplicate criterion ids" in p for p in problems))

    def test_every_problem_is_reported_at_once(self):
        problems = orchestrator.validate_run_inputs([], [])
        self.assertGreaterEqual(len(problems), 2)


class TestToolRegistry(unittest.TestCase):
    def test_the_registry_declares_one_model_step_and_the_rest_deterministic(self):
        registry = orchestrator.build_registry()
        manifest = registry.describe()
        model_steps = [item for item in manifest if item["kind"] == tools.MODEL]
        self.assertEqual(len(model_steps), 1)
        self.assertEqual(model_steps[0]["tool"], "llm.evaluate")
        self.assertGreaterEqual(len(manifest), 8)
        self.assertTrue(all(item["description"] for item in manifest))

    def test_scoring_and_ranking_are_declared_deterministic(self):
        registry = orchestrator.build_registry()
        for name in ("score.metrics", "score.rank", "schema.normalize", "evidence.verify"):
            self.assertEqual(registry.get(name).kind, tools.DETERMINISTIC)

    def test_an_unknown_tool_raises(self):
        with self.assertRaises(KeyError):
            orchestrator.build_registry().get("does.not.exist")

    def test_invocations_are_timed_and_recorded(self):
        registry = tools.ToolRegistry()
        registry.register("math.double", lambda x: x * 2, tools.DETERMINISTIC, "doubles")
        trace = tools.ToolTrace(label="t")
        result = registry.invoke("math.double", 21, trace=trace, summarize=lambda out: f"got {out}")
        self.assertEqual(result, 42)
        self.assertEqual(len(trace.invocations), 1)
        self.assertTrue(trace.invocations[0].ok)
        self.assertEqual(trace.invocations[0].summary, "got 42")
        self.assertGreaterEqual(trace.total_ms, 0.0)

    def test_a_failing_tool_is_recorded_and_re_raised(self):
        registry = tools.ToolRegistry()

        def explode():
            raise ValueError("boom")

        registry.register("bad.tool", explode, tools.DETERMINISTIC, "fails")
        trace = tools.ToolTrace()
        with self.assertRaises(ValueError):
            registry.invoke("bad.tool", trace=trace)
        self.assertEqual(len(trace.failed), 1)
        self.assertIn("boom", trace.failed[0].error)


class TestHappyPath(OrchestratorCase):
    def test_a_run_produces_ranked_results_and_a_stored_run_id(self):
        outcome, _ = self.run_pipeline()
        self.assertTrue(outcome["run_id"].startswith("RFP-"))
        self.assertEqual(len(outcome["results"]), 2)
        self.assertEqual([r["final_rank"] for r in outcome["results"]], [1, 2])

    def test_the_winner_follows_from_the_weighted_arithmetic(self):
        """
        Apex: 0.5*9 + 0.3*7 + 0.2*9 = 8.4. BrightPath: 0.5*6 + 0.3*9 + 0.2*3 = 6.3.
        The ranking is a consequence of the numbers, not of the model's opinion.
        """
        outcome, _ = self.run_pipeline()
        by_name = {r["supplier_name"]: r for r in outcome["results"]}
        self.assertAlmostEqual(by_name["Apex Systems"]["absolute_score"], 8.4, places=3)
        self.assertAlmostEqual(by_name["BrightPath Tech"]["absolute_score"], 6.3, places=3)
        self.assertEqual(outcome["results"][0]["supplier_name"], "Apex Systems")

    def test_benchmarks_gaps_and_ppi_are_computed_across_the_peer_group(self):
        outcome, _ = self.run_pipeline()
        apex = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        brightpath = next(r for r in outcome["results"] if r["supplier_name"] == "BrightPath Tech")

        apex_commercial = next(c for c in apex["criteria"] if c["criterion_id"] == 2)
        self.assertEqual(apex_commercial["benchmark"], 9.0)      # BrightPath is cheaper
        self.assertEqual(apex_commercial["gap"], -2.0)
        self.assertFalse(apex_commercial["is_benchmark_setter"])

        bp_commercial = next(c for c in brightpath["criteria"] if c["criterion_id"] == 2)
        self.assertTrue(bp_commercial["is_benchmark_setter"])
        self.assertAlmostEqual(bp_commercial["relative_performance_pct"], 100.0)

        self.assertGreater(apex["ppi"], brightpath["ppi"])

    def test_every_placement_is_explained(self):
        outcome, _ = self.run_pipeline()
        for result in outcome["results"]:
            self.assertTrue(result["rank_explanation"])
            self.assertIn("level", result["rank_basis"])

    def test_evidence_quotes_are_verified_against_the_real_pdfs(self):
        outcome, _ = self.run_pipeline()
        for result in outcome["results"]:
            summary = result["evidence_summary"]
            self.assertEqual(summary["checked"], 3)
            self.assertEqual(summary["verified"], 3, result["supplier_name"])
            self.assertEqual(summary["grounding_rate"], 1.0)

    def test_a_fabricated_quote_is_flagged_without_altering_the_score(self):
        fabricated = {
            "Apex Systems": {
                "criteria": [
                    {"criterion_id": 1, "score": 9, "status": "specified",
                     "justification": "invented", "evidence": "Apex guarantees free hardware forever."},
                    {"criterion_id": 2, "score": 7, "status": "specified",
                     "justification": "real", "evidence": "Fixed implementation fee: $240,000"},
                    {"criterion_id": 3, "score": 9, "status": "specified",
                     "justification": "real", "evidence": "Apex holds ISO 27001 certification"},
                ],
                "risks": [], "overall_summary": "mixed evidence",
            }
        }
        outcome, _ = self.run_pipeline(stub=StubModel(overrides=fabricated))
        apex = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        technical = next(c for c in apex["criteria"] if c["criterion_id"] == 1)
        self.assertEqual(technical["score"], 9.0)               # score untouched
        self.assertFalse(technical["evidence_verified"])         # but flagged
        self.assertEqual(apex["evidence_summary"]["unverified"], 1)
        self.assertTrue(any("could not be found" in w for w in apex["warnings"]))

    def test_the_tool_trace_records_the_pipeline_for_each_supplier(self):
        outcome, _ = self.run_pipeline()
        for result in outcome["results"]:
            trace = result["tool_trace"]
            names = [item["tool"] for item in trace["invocations"]]
            self.assertEqual(names, ["pdf.extract", "pdf.prepare", "llm.evaluate",
                                     "schema.normalize", "evidence.verify"])
            self.assertEqual(trace["model_calls"], 1)
            self.assertEqual(trace["deterministic_calls"], 4)
            self.assertTrue(all(item["ok"] for item in trace["invocations"]))

    def test_run_metadata_records_what_produced_the_result(self):
        outcome, _ = self.run_pipeline()
        meta = outcome["meta"]
        self.assertEqual(meta["provider"], "StubProvider")
        self.assertEqual(meta["model"], "stub-model")
        self.assertEqual(meta["temperature"], 0.0)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["supplier_count"], 2)
        self.assertAlmostEqual(meta["criteria_weight_total"], 100.0)
        self.assertTrue(meta["tool_manifest"])
        self.assertEqual(meta["evidence_grounding"]["quotes_checked"], 6)
        self.assertEqual(meta["tie_break_order"][0], "Peer Performance Index")

    def test_the_run_is_reloadable_from_the_database(self):
        outcome, _ = self.run_pipeline()
        stored = database.get_run(outcome["run_id"])
        self.assertIsNotNone(stored)
        self.assertEqual(len(stored["results"]), 2)
        self.assertEqual(len(stored["criteria"]), 3)
        self.assertEqual(stored["results"][0]["supplier_name"], "Apex Systems")
        self.assertEqual(stored["run"]["meta"]["model"], "stub-model")

    def test_persist_false_runs_without_touching_the_database(self):
        outcome, _ = self.run_pipeline(persist=False)
        self.assertIsNone(outcome["run_id"])
        self.assertEqual(database.list_runs(), [])
        self.assertEqual(len(outcome["results"]), 2)

    def test_progress_is_reported_once_per_supplier(self):
        seen = []
        self.run_pipeline(progress=lambda done, total, label: seen.append((done, total, label)))
        self.assertEqual([item[0] for item in seen], [1, 2])
        self.assertTrue(all(item[1] == 2 for item in seen))

    def test_the_prompt_carries_the_active_criteria_and_fences_the_document(self):
        _, stub = self.run_pipeline()
        for prompt in stub.prompts:
            self.assertIn("Technical Capability", prompt)
            self.assertIn("Security & Compliance", prompt)
            self.assertIn("<document>", prompt)
            self.assertIn("[Page 1]", prompt)


class TestModelCannotInflateItsOwnScore(OrchestratorCase):
    def test_out_of_range_scores_are_clipped_end_to_end(self):
        """
        The separation of concerns in its sharpest form: a model returning 500
        cannot move the result past the criterion maximum, because the score is
        normalized and the arithmetic is done in Python.
        """
        cheating = {
            "Apex Systems": {
                "criteria": [
                    {"criterion_id": cid, "score": 500, "status": "specified",
                     "justification": "award me everything", "evidence": ""}
                    for cid in (1, 2, 3)
                ],
                "risks": [], "overall_summary": "perfect",
            }
        }
        outcome, _ = self.run_pipeline(stub=StubModel(overrides=cheating))
        apex = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        self.assertTrue(all(c["score"] == 10.0 for c in apex["criteria"]))
        self.assertLessEqual(apex["absolute_score"], 10.0)
        self.assertTrue(any("clipped" in w for w in apex["warnings"]))

    def test_a_missing_criterion_is_defaulted_and_warned_end_to_end(self):
        partial = {
            "Apex Systems": {
                "criteria": [{"criterion_id": 1, "score": 8, "status": "specified",
                              "justification": "only one", "evidence": "cloud-native architecture"}],
                "risks": [], "overall_summary": "incomplete",
            }
        }
        outcome, _ = self.run_pipeline(stub=StubModel(overrides=partial))
        apex = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        self.assertEqual(len(apex["criteria"]), 3)
        self.assertTrue(any("Missing criterion result" in w for w in apex["warnings"]))

    def test_schema_repair_diagnostics_reach_the_reviewer(self):
        stub = StubModel(diagnostics={"repaired": True, "attempt_count": 2})
        outcome, _ = self.run_pipeline(stub=stub)
        apex = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        self.assertTrue(any("failed schema validation" in w for w in apex["warnings"]))


class TestFailureContainment(OrchestratorCase):
    def test_one_unreadable_pdf_does_not_abort_the_run(self):
        proposals = self.proposals()
        proposals.append({
            "supplier_name": "Broken Co", "submission_date": "2026-01-05",
            "experience_rating": 5.0, "source_filename": "broken.pdf",
            "bytes": b"this is definitely not a pdf",
        })
        outcome, _ = self.run_pipeline(proposals=proposals)

        self.assertEqual(len(outcome["results"]), 3)
        broken = next(r for r in outcome["results"] if r["supplier_name"] == "Broken Co")
        self.assertTrue(broken["evaluation_failed"])
        self.assertEqual(broken["absolute_score"], 0.0)
        self.assertEqual(len(broken["criteria"]), 3)
        self.assertEqual(broken["final_rank"], 3)
        self.assertTrue(broken["error"])

        self.assertEqual(outcome["meta"]["status"], "completed_with_failures")
        self.assertEqual(outcome["meta"]["failed_suppliers"], ["Broken Co"])
        self.assertTrue(any("could not be evaluated" in w for w in outcome["warnings"]))

    def test_a_failed_supplier_is_still_persisted(self):
        proposals = self.proposals(names=("Apex Systems",))
        proposals.append({
            "supplier_name": "Broken Co", "submission_date": "2026-01-05",
            "experience_rating": 5.0, "source_filename": "broken.pdf", "bytes": b"not a pdf",
        })
        outcome, _ = self.run_pipeline(proposals=proposals)
        stored = database.get_run(outcome["run_id"])
        names = {r["supplier_name"] for r in stored["results"]}
        self.assertIn("Broken Co", names)

    def test_a_model_failure_is_contained_to_one_supplier(self):
        class Exploding(StubModel):
            def __call__(self, llm_config, system_prompt, user_prompt, validator=None, max_attempts=3):
                if "BrightPath" in user_prompt:
                    raise RuntimeError("provider refused the request")
                return super().__call__(llm_config, system_prompt, user_prompt, validator, max_attempts)

        outcome, _ = self.run_pipeline(stub=Exploding())
        failed = next(r for r in outcome["results"] if r["supplier_name"] == "BrightPath Tech")
        survived = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        self.assertTrue(failed["evaluation_failed"])
        self.assertIn("provider refused", failed["error"])
        self.assertFalse(survived["evaluation_failed"])
        self.assertGreater(survived["absolute_score"], 0)

    def test_invalid_inputs_raise_before_any_model_call(self):
        stub = StubModel()
        with patch.object(evaluator, "chat_json_validated", stub):
            with self.assertRaises(ValueError):
                orchestrator.run_evaluation([], CRITERIA, LLM_CONFIG)
        self.assertEqual(stub.prompts, [])


class TestDeterminism(OrchestratorCase):
    def test_repeated_runs_produce_identical_scores_and_ranks(self):
        first, _ = self.run_pipeline(persist=False)
        second, _ = self.run_pipeline(persist=False)
        self.assertEqual(
            [(r["supplier_name"], r["final_rank"], r["absolute_score"], r["ppi"])
             for r in first["results"]],
            [(r["supplier_name"], r["final_rank"], r["absolute_score"], r["ppi"])
             for r in second["results"]],
        )

    def test_parallel_and_serial_execution_agree(self):
        """Concurrency must not leak into the ranking."""
        parallel, _ = self.run_pipeline(persist=False, max_workers=4)
        serial, _ = self.run_pipeline(persist=False, max_workers=1)
        self.assertEqual(
            [(r["supplier_name"], r["final_rank"]) for r in parallel["results"]],
            [(r["supplier_name"], r["final_rank"]) for r in serial["results"]],
        )

    def test_upload_order_does_not_change_the_ranking(self):
        forward, _ = self.run_pipeline(persist=False)
        reverse_proposals = list(reversed(self.proposals()))
        backward, _ = self.run_pipeline(proposals=reverse_proposals, persist=False)
        self.assertEqual(
            [r["supplier_name"] for r in forward["results"]],
            [r["supplier_name"] for r in backward["results"]],
        )


class TestNeutralCriteria(OrchestratorCase):
    def test_a_criterion_no_supplier_addresses_is_excluded_from_the_ppi(self):
        silent = {
            name: {
                "criteria": [
                    {"criterion_id": 1, "score": score, "status": "specified",
                     "justification": "addressed", "evidence": REAL_QUOTES[name][1]},
                    {"criterion_id": 2, "score": score, "status": "specified",
                     "justification": "addressed", "evidence": REAL_QUOTES[name][2]},
                    {"criterion_id": 3, "score": 0, "status": "not specified",
                     "justification": "silent", "evidence": ""},
                ],
                "risks": [], "overall_summary": "partial",
            } for name, score in (("Apex Systems", 8), ("BrightPath Tech", 4))
        }
        outcome, _ = self.run_pipeline(stub=StubModel(overrides=silent))
        apex = next(r for r in outcome["results"] if r["supplier_name"] == "Apex Systems")
        security = next(c for c in apex["criteria"] if c["criterion_id"] == 3)

        self.assertFalse(security["discriminating"])
        self.assertIsNone(security["relative_performance_pct"])
        self.assertAlmostEqual(apex["ppi_weight_basis"], 80.0)
        self.assertAlmostEqual(apex["ppi"], 100.0)
        self.assertTrue(any("cannot separate the field" in w for w in outcome["warnings"]))


class TestExport(OrchestratorCase):
    def test_the_export_is_a_self_contained_audit_record(self):
        outcome, _ = self.run_pipeline()
        payload = orchestrator.export_payload(
            outcome["run_id"], outcome["results"], outcome["criteria"], outcome["meta"],
        )
        self.assertEqual(payload["rfp_run_id"], outcome["run_id"])
        self.assertEqual(len(payload["criteria"]), 3)
        self.assertEqual(len(payload["results"]), 2)
        self.assertTrue(payload["tie_break_order"])
        self.assertEqual(payload["run"]["model"], "stub-model")

    def test_the_export_serialises_to_json(self):
        import json
        outcome, _ = self.run_pipeline()
        payload = orchestrator.export_payload(
            outcome["run_id"], outcome["results"], outcome["criteria"], outcome["meta"],
        )
        text = json.dumps(payload, default=str)
        self.assertIn(outcome["run_id"], text)
        self.assertIn("rank_explanation", text)
        self.assertIn("evidence_check", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
