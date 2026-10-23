"""
Tests for SQLite persistence: criteria CRUD, run storage, the per-run criteria
snapshot and history reload.

Every test runs against a fresh temporary database, so the suite never touches
the developer's `rfp_evaluation.db`.

Runnable either way::

    python test_database.py
    pytest test_database.py
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401
import database


def sample_criteria():
    return [
        {"criterion_id": 1, "name": "Technical", "description": "Fit", "weight": 60.0, "max_score": 10.0},
        {"criterion_id": 2, "name": "Commercial", "description": "Value", "weight": 40.0, "max_score": 10.0},
    ]


def sample_results():
    return [
        {
            "supplier_name": "Apex", "submission_date": "2026-01-02", "experience_rating": 8.0,
            "source_filename": "apex.pdf", "absolute_score": 8.2, "ppi": 94.0, "final_rank": 1,
            "criteria": [{"criterion_id": 1, "name": "Technical", "score": 9.0, "max_score": 10.0,
                          "weight": 60.0, "status": "specified", "evidence": "quote",
                          "justification": "strong"}],
            "warnings": ["one minor warning"],
            "evidence_summary": {"checked": 2, "verified": 2, "partial": 0, "unverified": 0,
                                 "page_mismatches": 0, "grounding_rate": 1.0,
                                 "document_word_count": 400},
            "risks": ["timeline"], "overall_summary": "good",
        },
        {
            "supplier_name": "BrightPath", "submission_date": "2026-01-01", "experience_rating": 5.0,
            "source_filename": "bp.pdf", "absolute_score": 6.4, "ppi": 72.0, "final_rank": 2,
            "criteria": [{"criterion_id": 1, "name": "Technical", "score": 6.0, "max_score": 10.0,
                          "weight": 60.0, "status": "specified", "evidence": "quote",
                          "justification": "ok"}],
            "warnings": [],
            "evidence_summary": {"checked": 2, "verified": 1, "partial": 1, "unverified": 0,
                                 "page_mismatches": 0, "grounding_rate": 0.5,
                                 "document_word_count": 350},
            "risks": [], "overall_summary": "adequate",
        },
    ]


def sample_meta():
    return {
        "provider": "OpenAI", "model": "gpt-4.1-mini", "temperature": 0.0,
        "status": "completed", "app_version": "2.0",
        "tool_manifest": [{"tool": "score.rank", "kind": "deterministic", "description": "ranks"}],
        "warnings": [], "notes": "two suppliers",
    }


class DatabaseCase(unittest.TestCase):
    """Each test gets its own database file."""

    def setUp(self):
        self._previous = database.get_db_path()
        self._dir = tempfile.TemporaryDirectory()
        database.set_db_path(Path(self._dir.name) / "test.db")
        database.init_db()

    def tearDown(self):
        database.set_db_path(self._previous)
        self._dir.cleanup()


class TestSchema(DatabaseCase):
    def test_init_creates_every_table(self):
        with database.session() as conn:
            names = {row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertTrue({"evaluation_criteria", "rfp_runs", "run_criteria",
                         "supplier_results"} <= names)

    def test_init_is_idempotent(self):
        database.init_db()
        database.init_db()
        self.assertEqual(len(database.get_all_criteria()), len(database.DEFAULT_CRITERIA))

    def test_default_criteria_are_seeded_and_total_100_percent(self):
        active = database.get_active_criteria()
        self.assertEqual(len(active), 5)
        self.assertAlmostEqual(sum(c["weight"] for c in active), 100.0)
        self.assertAlmostEqual(database.total_active_weight(), 100.0)

    def test_foreign_keys_are_enforced(self):
        with database.session() as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_indexes_exist_for_run_lookups(self):
        with database.session() as conn:
            names = {row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        self.assertIn("idx_results_run", names)
        self.assertIn("idx_run_criteria_run", names)

    def test_an_orphan_result_is_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with database.session() as conn:
                conn.execute("""
                    INSERT INTO supplier_results
                        (rfp_run_id, supplier_name, submission_date, experience_rating,
                         absolute_score, ppi, final_rank, result_json)
                    VALUES ('NO-SUCH-RUN', 'Ghost', '2026-01-01', 5, 5, 50, 1, '{}')
                """)


class TestCriteriaCrud(DatabaseCase):
    def test_a_new_criterion_gets_the_next_id(self):
        new_id = database.upsert_criterion(None, "Sustainability", "Carbon", 10.0, 10.0, True)
        self.assertEqual(new_id, 6)
        names = {c["name"] for c in database.get_all_criteria()}
        self.assertIn("Sustainability", names)

    def test_an_existing_criterion_is_updated_in_place(self):
        database.upsert_criterion(1, "Technical Capability", "Updated text", 35.0, 10.0, True)
        row = next(c for c in database.get_all_criteria() if c["criterion_id"] == 1)
        self.assertEqual(row["description"], "Updated text")
        self.assertAlmostEqual(row["weight"], 35.0)
        self.assertEqual(len(database.get_all_criteria()), 5)

    def test_deactivating_removes_a_criterion_from_the_active_set(self):
        database.set_criterion_active(5, False)
        active_ids = {c["criterion_id"] for c in database.get_active_criteria()}
        self.assertNotIn(5, active_ids)
        self.assertEqual(len(database.get_all_criteria()), 5)
        self.assertAlmostEqual(database.total_active_weight(), 90.0)

    def test_deleting_removes_a_criterion_entirely(self):
        database.delete_criterion(5)
        self.assertEqual(len(database.get_all_criteria()), 4)

    def test_replace_criteria_swaps_the_whole_set_atomically(self):
        count = database.replace_criteria([
            {"criterion_id": 1, "name": "Only", "description": "d", "weight": 100.0,
             "max_score": 10.0, "is_active": True},
        ])
        self.assertEqual(count, 1)
        self.assertEqual(len(database.get_all_criteria()), 1)
        self.assertAlmostEqual(database.total_active_weight(), 100.0)

    def test_reset_restores_the_shipped_defaults(self):
        database.replace_criteria([{"criterion_id": 1, "name": "X", "description": "",
                                    "weight": 100.0, "max_score": 10.0, "is_active": True}])
        database.reset_criteria_to_defaults()
        self.assertEqual(len(database.get_all_criteria()), 5)
        self.assertAlmostEqual(database.total_active_weight(), 100.0)

    def test_invalid_criteria_are_rejected(self):
        with self.assertRaises(ValueError):
            database.upsert_criterion(None, "   ", "d", 10.0)
        with self.assertRaises(ValueError):
            database.upsert_criterion(None, "Name", "d", -5.0)
        with self.assertRaises(ValueError):
            database.upsert_criterion(None, "Name", "d", 10.0, max_score=0.0)

    def test_active_criteria_come_back_in_id_order(self):
        ids = [c["criterion_id"] for c in database.get_active_criteria()]
        self.assertEqual(ids, sorted(ids))


class TestRunPersistence(DatabaseCase):
    def test_saving_a_run_stores_metadata_results_and_the_criteria_snapshot(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        self.assertTrue(run_id.startswith("RFP-"))

        runs = database.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["rfp_run_id"], run_id)
        self.assertEqual(runs[0]["winner"], "Apex")
        self.assertEqual(runs[0]["supplier_count"], 2)
        self.assertEqual(runs[0]["criteria_count"], 2)
        self.assertEqual(runs[0]["provider"], "OpenAI")
        self.assertEqual(runs[0]["model"], "gpt-4.1-mini")
        self.assertAlmostEqual(runs[0]["criteria_weight_total"], 100.0)

    def test_results_reload_in_rank_order(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        results = database.get_run_results(run_id)
        self.assertEqual([r["final_rank"] for r in results], [1, 2])
        self.assertEqual(results[0]["supplier_name"], "Apex")

    def test_the_full_result_payload_survives_the_round_trip(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        restored = database.get_run_results(run_id)[0]
        self.assertEqual(restored["risks"], ["timeline"])
        self.assertEqual(restored["overall_summary"], "good")
        self.assertEqual(restored["criteria"][0]["evidence"], "quote")
        self.assertEqual(restored["evidence_summary"]["grounding_rate"], 1.0)
        self.assertEqual(restored["warnings"], ["one minor warning"])

    def test_the_criteria_snapshot_is_stored_per_run(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        snapshot = database.get_run_criteria(run_id)
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot[0]["name"], "Technical")
        self.assertAlmostEqual(snapshot[0]["weight"], 60.0)

    def test_a_historical_run_keeps_its_own_criteria_after_the_active_set_changes(self):
        """
        The reproducibility guarantee. Rewriting the active criteria must not
        retroactively change what a stored ranking was measured against.
        """
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        database.replace_criteria([
            {"criterion_id": 1, "name": "Renamed Entirely", "description": "changed",
             "weight": 100.0, "max_score": 20.0, "is_active": True},
        ])
        snapshot = database.get_run_criteria(run_id)
        self.assertEqual([c["name"] for c in snapshot], ["Technical", "Commercial"])
        self.assertAlmostEqual(snapshot[0]["weight"], 60.0)
        self.assertAlmostEqual(snapshot[0]["max_score"], 10.0)

    def test_get_run_returns_metadata_criteria_and_results_together(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        loaded = database.get_run(run_id)
        self.assertEqual(loaded["run_id"], run_id)
        self.assertEqual(len(loaded["results"]), 2)
        self.assertEqual(len(loaded["criteria"]), 2)
        self.assertEqual(loaded["run"]["meta"]["provider"], "OpenAI")
        self.assertEqual(loaded["run"]["meta"]["tool_manifest"][0]["tool"], "score.rank")

    def test_queryable_grounding_columns_are_populated(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        with database.session() as conn:
            rows = conn.execute("""
                SELECT supplier_name, evidence_checked, evidence_verified, grounding_rate,
                       warning_count, evaluation_failed
                FROM supplier_results WHERE rfp_run_id = ? ORDER BY final_rank
            """, (run_id,)).fetchall()
        self.assertEqual(rows[0]["evidence_checked"], 2)
        self.assertEqual(rows[0]["evidence_verified"], 2)
        self.assertAlmostEqual(rows[0]["grounding_rate"], 1.0)
        self.assertEqual(rows[0]["warning_count"], 1)
        self.assertEqual(rows[0]["evaluation_failed"], 0)

    def test_a_failed_supplier_is_recorded_rather_than_dropped(self):
        results = sample_results()
        results[1]["evaluation_failed"] = True
        run_id = database.save_run(results, sample_criteria(), sample_meta())
        with database.session() as conn:
            failed = conn.execute(
                "SELECT COUNT(*) FROM supplier_results WHERE rfp_run_id=? AND evaluation_failed=1",
                (run_id,),
            ).fetchone()[0]
        self.assertEqual(failed, 1)
        self.assertEqual(len(database.get_run_results(run_id)), 2)

    def test_unknown_run_ids_return_nothing_without_raising(self):
        self.assertIsNone(database.get_run("RFP-DOESNOTEXIST"))
        self.assertEqual(database.get_run_results("RFP-DOESNOTEXIST"), [])
        self.assertEqual(database.get_run_criteria("RFP-DOESNOTEXIST"), [])

    def test_deleting_a_run_removes_its_children(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        database.delete_run(run_id)
        self.assertIsNone(database.get_run(run_id))
        self.assertEqual(database.get_run_results(run_id), [])
        self.assertEqual(database.get_run_criteria(run_id), [])
        self.assertEqual(database.list_runs(), [])

    def test_runs_are_listed_newest_first_and_limited(self):
        ids = []
        for index in range(5):
            meta = sample_meta()
            meta["created_at"] = f"2026-02-0{index + 1}T10:00:00+00:00"
            ids.append(database.save_run(sample_results(), sample_criteria(), meta))
        listed = database.list_runs(limit=3)
        self.assertEqual(len(listed), 3)
        self.assertEqual(listed[0]["rfp_run_id"], ids[-1])

    def test_run_ids_are_unique(self):
        generated = {database.new_run_id() for _ in range(200)}
        self.assertEqual(len(generated), 200)

    def test_non_ascii_content_survives_storage(self):
        results = sample_results()
        results[0]["overall_summary"] = "Qualité élevée — 99,9 % disponibilité ✓"
        run_id = database.save_run(results, sample_criteria(), sample_meta())
        restored = database.get_run_results(run_id)[0]
        self.assertEqual(restored["overall_summary"], "Qualité élevée — 99,9 % disponibilité ✓")

    def test_stored_run_json_is_valid_json(self):
        run_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        with database.session() as conn:
            raw = conn.execute(
                "SELECT run_json FROM rfp_runs WHERE rfp_run_id = ?", (run_id,)
            ).fetchone()["run_json"]
        self.assertIsInstance(json.loads(raw), dict)


class TestMigration(DatabaseCase):
    def test_an_older_database_is_upgraded_in_place(self):
        """A database written by the previous schema must survive an upgrade."""
        database.set_db_path(Path(self._dir.name) / "legacy.db")
        with database.session() as conn:
            conn.execute("""
                CREATE TABLE evaluation_criteria (
                    criterion_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                    description TEXT NOT NULL, weight REAL NOT NULL,
                    max_score REAL NOT NULL DEFAULT 10, is_active INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE rfp_runs (
                    rfp_run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL
                )
            """)
            conn.execute("""
                INSERT INTO evaluation_criteria VALUES (1, 'Legacy', 'old row', 100.0, 10.0, 1)
            """)
            conn.execute("INSERT INTO rfp_runs VALUES ('RFP-OLD', '2025-01-01T00:00:00', 'completed')")

        database.init_db()

        with database.session() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(rfp_runs)").fetchall()}
        self.assertTrue({"provider", "model", "temperature", "run_json", "winner"} <= columns)

        # Pre-existing data is preserved, and the new tables are available.
        self.assertEqual(len(database.get_all_criteria()), 1)
        self.assertEqual(database.get_all_criteria()[0]["name"], "Legacy")
        self.assertEqual(len(database.list_runs()), 1)
        new_id = database.save_run(sample_results(), sample_criteria(), sample_meta())
        self.assertEqual(len(database.get_run_criteria(new_id)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
