from scoring import validate_and_normalize, calculate_supplier_metrics, rank_suppliers

CRITERIA = [
    {"criterion_id": 1, "name": "Technical", "description": "Technical fit", "weight": 60.0, "max_score": 10.0},
    {"criterion_id": 2, "name": "Commercial", "description": "Value", "weight": 40.0, "max_score": 10.0},
]

def test_missing_criterion_is_normalized():
    normalized, warnings = validate_and_normalize(
        {"criteria": [{"criterion_id": 1, "score": 8, "justification": "Good", "evidence": "Page 1"}]},
        CRITERIA
    )
    assert len(normalized) == 2
    assert normalized[1]["score"] == 0
    assert warnings

def test_out_of_range_score_is_clipped():
    normalized, warnings = validate_and_normalize(
        {"criteria": [
            {"criterion_id": 1, "score": 12, "justification": "Good", "evidence": "Page 1"},
            {"criterion_id": 2, "score": 5, "justification": "Fair", "evidence": "Page 2"}
        ]}, CRITERIA
    )
    assert normalized[0]["score"] == 10
    assert warnings

def test_metrics_and_ranking():
    suppliers = []
    for name, tech, commercial, submitted, experience in [
        ("B Supplier", 8, 7, "2026-01-02", 8),
        ("A Supplier", 8, 7, "2026-01-01", 5)
    ]:
        normalized, _ = validate_and_normalize({"criteria": [
            {"criterion_id": 1, "score": tech, "justification": "test", "evidence": "Page 1"},
            {"criterion_id": 2, "score": commercial, "justification": "test", "evidence": "Page 2"}
        ]}, CRITERIA)
        suppliers.append({
            "supplier_name": name, "submission_date": submitted,
            "experience_rating": experience, "criteria": normalized
        })
    ranked = rank_suppliers(calculate_supplier_metrics(suppliers, CRITERIA))
    assert ranked[0]["supplier_name"] == "A Supplier"  # Earlier submission wins tie
    assert ranked[0]["final_rank"] == 1
