"""
Tests for normalization, weighted scoring, peer benchmarking, the PPI and ranking.

Runnable either way::

    python test_scoring.py      # unittest runner, no extra dependency
    pytest test_scoring.py      # same tests under pytest
"""

from __future__ import annotations

import unittest

import conftest  # noqa: F401  (anchors sys.path and the working directory)
from scoring import (
    TIE_BREAK_ORDER,
    assert_weights_total_100,
    calculate_supplier_metrics,
    normalize_evaluation,
    rank_suppliers,
    ranking_summary,
    validate_and_normalize,
)

CRITERIA = [
    {"criterion_id": 1, "name": "Technical", "description": "Technical fit", "weight": 60.0, "max_score": 10.0},
    {"criterion_id": 2, "name": "Commercial", "description": "Value", "weight": 40.0, "max_score": 10.0},
]


def make_supplier(name, scores, submitted="2026-01-01", experience=5.0, statuses=None):
    """Build a supplier record with normalized criterion rows."""
    statuses = statuses or {}
    payload = {"criteria": [{
        "criterion_id": cid,
        "score": score,
        "status": statuses.get(cid, "specified"),
        "justification": "test",
        "evidence": f"quote for {cid}",
    } for cid, score in scores.items()]}
    rows, _ = validate_and_normalize(payload, CRITERIA)
    return {
        "supplier_name": name,
        "submission_date": submitted,
        "experience_rating": experience,
        "criteria": rows,
    }


class TestNormalization(unittest.TestCase):
    def test_missing_criterion_is_defaulted_and_reported(self):
        rows, warnings = validate_and_normalize(
            {"criteria": [{"criterion_id": 1, "score": 8, "justification": "Good", "evidence": "Page 1"}]},
            CRITERIA,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["score"], 0.0)
        self.assertEqual(rows[1]["status"], "not specified")
        self.assertTrue(any("Missing criterion result" in w for w in warnings))

    def test_out_of_range_score_is_clipped(self):
        rows, warnings = validate_and_normalize(
            {"criteria": [
                {"criterion_id": 1, "score": 12, "justification": "Good", "evidence": "Page 1"},
                {"criterion_id": 2, "score": -4, "justification": "Bad", "evidence": "Page 2"},
            ]},
            CRITERIA,
        )
        self.assertEqual(rows[0]["score"], 10.0)
        self.assertEqual(rows[1]["score"], 0.0)
        self.assertEqual(sum("clipped" in w for w in warnings), 2)

    def test_non_numeric_and_infinite_scores_default_to_zero(self):
        rows, warnings = validate_and_normalize(
            {"criteria": [
                {"criterion_id": 1, "score": "not a number"},
                {"criterion_id": 2, "score": float("inf")},
            ]},
            CRITERIA,
        )
        self.assertEqual([row["score"] for row in rows], [0.0, 0.0])
        self.assertEqual(sum("Invalid score" in w for w in warnings), 2)

    def test_duplicate_criterion_id_keeps_first_and_warns(self):
        rows, warnings = validate_and_normalize(
            {"criteria": [
                {"criterion_id": 1, "score": 9, "evidence": "first"},
                {"criterion_id": 1, "score": 2, "evidence": "second"},
                {"criterion_id": 2, "score": 5, "evidence": "ok"},
            ]},
            CRITERIA,
        )
        self.assertEqual(rows[0]["score"], 9.0)
        self.assertEqual(rows[0]["evidence"], "first")
        self.assertTrue(any("Duplicate result" in w for w in warnings))

    def test_unknown_criterion_id_is_dropped_and_warned(self):
        rows, warnings = validate_and_normalize(
            {"criteria": [
                {"criterion_id": 99, "score": 10, "evidence": "not a real criterion"},
                {"criterion_id": 1, "score": 6, "evidence": "ok"},
                {"criterion_id": 2, "score": 6, "evidence": "ok"},
            ]},
            CRITERIA,
        )
        self.assertEqual([row["criterion_id"] for row in rows], [1, 2])
        self.assertTrue(any("not an active criterion" in w for w in warnings))

    def test_not_specified_with_a_score_is_zeroed(self):
        """A row excluded from benchmarking must not still earn its own points."""
        rows, warnings = validate_and_normalize(
            {"criteria": [
                {"criterion_id": 1, "score": 7, "status": "not specified", "evidence": ""},
                {"criterion_id": 2, "score": 5, "status": "specified", "evidence": "ok"},
            ]},
            CRITERIA,
        )
        self.assertEqual(rows[0]["score"], 0.0)
        self.assertTrue(any("reset to 0" in w for w in warnings))

    def test_unrecognised_status_falls_back_to_specified(self):
        rows, warnings = validate_and_normalize(
            {"criteria": [
                {"criterion_id": 1, "score": 5, "status": "maybe", "evidence": "ok"},
                {"criterion_id": 2, "score": 5, "status": "SPECIFIED", "evidence": "ok"},
            ]},
            CRITERIA,
        )
        self.assertEqual(rows[0]["status"], "specified")
        self.assertEqual(rows[1]["status"], "specified")
        self.assertTrue(any("Unrecognised status" in w for w in warnings))

    def test_garbage_payloads_never_raise(self):
        for payload in [None, [], "text", 42, {"criteria": "nope"}, {"criteria": [None, 7]}, {}]:
            rows, warnings = validate_and_normalize(payload, CRITERIA)
            self.assertEqual(len(rows), len(CRITERIA))
            self.assertTrue(all(row["score"] == 0.0 for row in rows))
            self.assertTrue(warnings)

    def test_alias_points_at_the_same_function(self):
        self.assertIs(normalize_evaluation, validate_and_normalize)


class TestWeightGuard(unittest.TestCase):
    def test_weights_totalling_100_are_accepted(self):
        self.assertAlmostEqual(assert_weights_total_100(CRITERIA), 100.0)

    def test_weights_not_totalling_100_raise(self):
        broken = [dict(CRITERIA[0], weight=50.0), dict(CRITERIA[1], weight=40.0)]
        with self.assertRaises(ValueError):
            assert_weights_total_100(broken)
        with self.assertRaises(ValueError):
            calculate_supplier_metrics([make_supplier("A", {1: 5, 2: 5})], broken)


class TestMetrics(unittest.TestCase):
    def test_absolute_score_is_the_weighted_fraction_of_the_maximum(self):
        suppliers = [make_supplier("A", {1: 10, 2: 5})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        # 0.6 * 10 + 0.4 * 5 = 8.0
        self.assertAlmostEqual(suppliers[0]["absolute_score"], 8.0)
        self.assertAlmostEqual(suppliers[0]["absolute_score_pct"], 80.0)

    def test_absolute_score_handles_a_non_ten_point_scale(self):
        criteria = [
            {"criterion_id": 1, "name": "A", "description": "", "weight": 50.0, "max_score": 5.0},
            {"criterion_id": 2, "name": "B", "description": "", "weight": 50.0, "max_score": 20.0},
        ]
        payload = {"criteria": [
            {"criterion_id": 1, "score": 5, "evidence": "x"},
            {"criterion_id": 2, "score": 10, "evidence": "y"},
        ]}
        rows, _ = validate_and_normalize(payload, criteria)
        supplier = {"supplier_name": "A", "submission_date": "2026-01-01",
                    "experience_rating": 5.0, "criteria": rows}
        calculate_supplier_metrics([supplier], criteria)
        # (5/5)*50 + (10/20)*50 = 75 -> 7.5 / 10
        self.assertAlmostEqual(supplier["absolute_score"], 7.5)

    def test_perfect_scores_reach_the_top_of_the_scale(self):
        suppliers = [make_supplier("A", {1: 10, 2: 10})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        self.assertAlmostEqual(suppliers[0]["absolute_score"], 10.0)
        self.assertAlmostEqual(suppliers[0]["ppi"], 100.0)

    def test_benchmark_is_the_best_specified_peer_score(self):
        suppliers = [make_supplier("A", {1: 6, 2: 4}), make_supplier("B", {1: 9, 2: 2})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        self.assertEqual(suppliers[0]["criteria"][0]["benchmark"], 9.0)
        self.assertEqual(suppliers[0]["criteria"][0]["gap"], -3.0)
        self.assertTrue(suppliers[1]["criteria"][0]["is_benchmark_setter"])
        self.assertFalse(suppliers[0]["criteria"][0]["is_benchmark_setter"])

    def test_not_specified_scores_are_excluded_from_the_benchmark(self):
        suppliers = [
            make_supplier("A", {1: 8, 2: 5}),
            make_supplier("B", {1: 10, 2: 5}, statuses={1: "not specified"}),
        ]
        calculate_supplier_metrics(suppliers, CRITERIA)
        # B's 10 was marked not specified, so it is zeroed and A sets the benchmark.
        self.assertEqual(suppliers[0]["criteria"][0]["benchmark"], 8.0)
        self.assertEqual(suppliers[1]["criteria"][0]["score"], 0.0)

    def test_ppi_is_the_weighted_mean_of_relative_performance(self):
        suppliers = [make_supplier("A", {1: 10, 2: 5}), make_supplier("B", {1: 5, 2: 10})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        # A: 100% technical, 50% commercial -> (100*60 + 50*40) / 100 = 80
        self.assertAlmostEqual(suppliers[0]["ppi"], 80.0)
        # B: 50% technical, 100% commercial -> (50*60 + 100*40) / 100 = 70
        self.assertAlmostEqual(suppliers[1]["ppi"], 70.0)

    def test_ppi_never_exceeds_100(self):
        suppliers = [make_supplier("A", {1: 10, 2: 10}), make_supplier("B", {1: 1, 2: 1})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        for supplier in suppliers:
            self.assertLessEqual(supplier["ppi"], 100.0)
            self.assertGreaterEqual(supplier["ppi"], 0.0)

    def test_criterion_nobody_scored_is_excluded_from_the_ppi(self):
        """
        A criterion with a zero benchmark cannot separate the field, so it must
        not silently hand every supplier 0% or 100% relative performance.
        """
        suppliers = [
            make_supplier("A", {1: 8, 2: 0}, statuses={2: "not specified"}),
            make_supplier("B", {1: 4, 2: 0}, statuses={2: "not specified"}),
        ]
        calculate_supplier_metrics(suppliers, CRITERIA)
        commercial = suppliers[0]["criteria"][1]
        self.assertFalse(commercial["discriminating"])
        self.assertIsNone(commercial["relative_performance_pct"])
        # Only the technical criterion (weight 60) forms the PPI basis.
        self.assertAlmostEqual(suppliers[0]["ppi_weight_basis"], 60.0)
        self.assertAlmostEqual(suppliers[0]["ppi"], 100.0)
        self.assertAlmostEqual(suppliers[1]["ppi"], 50.0)
        self.assertEqual(suppliers[0]["neutral_criteria"], [2])

    def test_all_zero_field_is_handled_without_dividing_by_zero(self):
        suppliers = [make_supplier("A", {1: 0, 2: 0}), make_supplier("B", {1: 0, 2: 0})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        for supplier in suppliers:
            self.assertEqual(supplier["ppi"], 0.0)
            self.assertEqual(supplier["absolute_score"], 0.0)

    def test_weighted_points_sum_to_the_absolute_score(self):
        suppliers = [make_supplier("A", {1: 7, 2: 3})]
        calculate_supplier_metrics(suppliers, CRITERIA)
        total = sum(row["weighted_points"] for row in suppliers[0]["criteria"])
        self.assertAlmostEqual(total, suppliers[0]["absolute_score"], places=3)


class TestRanking(unittest.TestCase):
    def test_higher_ppi_ranks_first(self):
        suppliers = [make_supplier("Low", {1: 4, 2: 4}), make_supplier("High", {1: 9, 2: 9})]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        self.assertEqual(ranked[0]["supplier_name"], "High")
        self.assertEqual(ranked[0]["final_rank"], 1)
        self.assertEqual(ranked[0]["rank_basis"]["key"], "ppi")
        self.assertFalse(ranked[0]["rank_basis"]["tie_break_used"])

    def test_tie_break_1_earlier_submission_wins(self):
        suppliers = [
            make_supplier("B Supplier", {1: 8, 2: 7}, submitted="2026-01-02", experience=8),
            make_supplier("A Supplier", {1: 8, 2: 7}, submitted="2026-01-01", experience=5),
        ]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        self.assertEqual(ranked[0]["supplier_name"], "A Supplier")
        self.assertEqual(ranked[0]["rank_basis"]["key"], "submission_date")
        self.assertTrue(ranked[0]["rank_basis"]["tie_break_used"])

    def test_tie_break_2_higher_experience_wins(self):
        suppliers = [
            make_supplier("Alpha", {1: 8, 2: 7}, submitted="2026-01-01", experience=4),
            make_supplier("Beta", {1: 8, 2: 7}, submitted="2026-01-01", experience=9),
        ]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        self.assertEqual(ranked[0]["supplier_name"], "Beta")
        self.assertEqual(ranked[0]["rank_basis"]["key"], "experience_rating")

    def test_tie_break_3_alphabetical_name_wins(self):
        suppliers = [
            make_supplier("zeta corp", {1: 8, 2: 7}, submitted="2026-01-01", experience=5),
            make_supplier("Alpha Corp", {1: 8, 2: 7}, submitted="2026-01-01", experience=5),
        ]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        self.assertEqual(ranked[0]["supplier_name"], "Alpha Corp")
        self.assertEqual(ranked[0]["rank_basis"]["key"], "supplier_name")

    def test_ranking_is_independent_of_input_order(self):
        """The sort key must be a strict total order, not merely a stable sort."""
        specs = [
            ("Alpha", {1: 8, 2: 7}, "2026-01-01", 5.0),
            ("Bravo", {1: 8, 2: 7}, "2026-01-01", 5.0),
            ("Charlie", {1: 9, 2: 6}, "2026-01-03", 7.0),
            ("Delta", {1: 6, 2: 9}, "2026-01-02", 9.0),
        ]
        orderings = [specs, list(reversed(specs)), [specs[2], specs[0], specs[3], specs[1]]]
        results = []
        for ordering in orderings:
            suppliers = [make_supplier(n, s, d, e) for n, s, d, e in ordering]
            ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
            results.append([r["supplier_name"] for r in ranked])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_ranks_are_contiguous_and_unique(self):
        suppliers = [make_supplier(f"S{i}", {1: i, 2: 10 - i}) for i in range(1, 6)]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        self.assertEqual(sorted(r["final_rank"] for r in ranked), [1, 2, 3, 4, 5])

    def test_single_supplier_run_is_rank_one(self):
        suppliers = [make_supplier("Only", {1: 5, 2: 5})]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        self.assertEqual(ranked[0]["final_rank"], 1)
        self.assertAlmostEqual(ranked[0]["ppi"], 100.0)
        self.assertIn("lowest PPI", ranked[0]["rank_explanation"])

    def test_every_supplier_gets_an_explanation(self):
        suppliers = [make_supplier("A", {1: 9, 2: 9}), make_supplier("B", {1: 3, 2: 3})]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        for result in ranked:
            self.assertTrue(result["rank_explanation"])
            self.assertEqual(result["tie_break_order"], [rule["label"] for rule in TIE_BREAK_ORDER])

    def test_ranking_summary_reports_tie_breaks_and_neutral_criteria(self):
        suppliers = [
            make_supplier("A Co", {1: 8, 2: 7}, submitted="2026-01-01"),
            make_supplier("B Co", {1: 8, 2: 7}, submitted="2026-01-02"),
        ]
        ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
        summary = ranking_summary(ranked)
        self.assertEqual(summary["suppliers_separated_by_tie_break"], ["A Co"])
        self.assertEqual(summary["neutral_criteria_excluded_from_ppi"], [])
        self.assertEqual(len(summary["tie_break_order"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
