import math

def validate_and_normalize(ai_result, criteria):
    """
    Ensure every active criterion has one valid record.
    Missing/invalid values are normalized and warnings are retained.
    """
    warnings = []
    raw_criteria = ai_result.get("criteria", []) if isinstance(ai_result, dict) else []
    by_id = {}
    for item in raw_criteria if isinstance(raw_criteria, list) else []:
        if isinstance(item, dict) and item.get("criterion_id") is not None:
            try:
                by_id[int(item["criterion_id"])] = item
            except (ValueError, TypeError):
                warnings.append("Ignored a criterion result with an invalid criterion_id.")

    normalized = []
    for criterion in criteria:
        cid = int(criterion["criterion_id"])
        maximum = float(criterion["max_score"])
        weight = float(criterion["weight"])
        item = by_id.get(cid)
        if item is None:
            warnings.append(f"Missing criterion result: {criterion['name']}; assigned score 0 and marked not specified.")
            normalized.append({
                "criterion_id": cid, "name": criterion["name"], "description": criterion["description"],
                "weight": weight, "score": 0.0, "max_score": maximum,
                "justification": "No valid result was returned for this criterion.",
                "evidence": "", "status": "not specified"
            })
            continue
        try:
            score = float(item.get("score", 0))
            if not math.isfinite(score):
                raise ValueError()
        except (ValueError, TypeError):
            score = 0.0
            warnings.append(f"Invalid score for {criterion['name']}; assigned 0.")
        clipped = min(max(score, 0.0), maximum)
        if clipped != score:
            warnings.append(f"Score for {criterion['name']} was outside 0–{maximum:g} and was clipped.")
        status = str(item.get("status", "specified")).strip().lower()
        if status not in {"specified", "not specified"}:
            status = "specified"
        normalized.append({
            "criterion_id": cid, "name": criterion["name"], "description": criterion["description"],
            "weight": weight, "score": clipped, "max_score": maximum,
            "justification": str(item.get("justification", "")).strip() or "No justification supplied.",
            "evidence": str(item.get("evidence", "")).strip(),
            "status": status
        })
    return normalized, warnings

def calculate_supplier_metrics(results, criteria):
    """Compute absolute weighted scores and cross-supplier criterion benchmarks."""
    benchmarks = {}
    for c in criteria:
        cid = int(c["criterion_id"])
        valid_scores = [
            item["score"] for supplier in results
            for item in supplier["criteria"]
            if item["criterion_id"] == cid and item["status"] == "specified"
        ]
        benchmarks[cid] = max(valid_scores) if valid_scores else 0.0

    total_weight = sum(float(c["weight"]) for c in criteria)
    if abs(total_weight - 100.0) > 0.001:
        raise ValueError(f"Active criterion weights must total 100; got {total_weight:g}.")

    for supplier in results:
        weighted = 0.0
        ppi_numerator = 0.0
        ppi_denominator = 0.0
        for item in supplier["criteria"]:
            maximum = item["max_score"]
            weight = item["weight"]
            weighted += (item["score"] / maximum if maximum else 0) * weight
            benchmark = benchmarks[item["criterion_id"]]
            item["benchmark"] = benchmark
            item["gap"] = round(item["score"] - benchmark, 4)
            if benchmark > 0:
                relative = item["score"] / benchmark * 100.0
            else:
                # Safe handling when no supplier has a positive specified score.
                relative = 100.0 if item["status"] == "specified" and item["score"] == 0 else 0.0
            item["relative_performance_pct"] = round(relative, 2)
            # PPI is the weighted average of criterion relative-performance percentages.
            ppi_numerator += relative * weight
            ppi_denominator += weight
        supplier["absolute_score"] = round(weighted / 10.0, 4) if all(
            abs(c["max_score"] - 10.0) < 1e-9 for c in supplier["criteria"]
        ) else round(weighted / 100.0 * 10.0, 4)
        supplier["ppi"] = round(ppi_numerator / ppi_denominator, 4) if ppi_denominator else 0.0
        supplier["benchmarks"] = {str(k): v for k, v in benchmarks.items()}
    return results

def rank_suppliers(results):
    """Deterministic tie-break order required by the brief."""
    results.sort(key=lambda r: (
        -float(r.get("ppi", 0)),
        str(r.get("submission_date", "9999-12-31")),
        -float(r.get("experience_rating", 0)),
        str(r.get("supplier_name", "")).casefold()
    ))
    for index, result in enumerate(results, start=1):
        result["final_rank"] = index
    return results
