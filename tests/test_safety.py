from preference_lab.safety import run_safety_regression, score_safety, REGRESSION_PROMPTS


def test_safety_regression_safe() -> None:
    """Safe mode should pass all regression prompts."""
    report = run_safety_regression(safe_mode=True)
    assert report.passed
    assert report.overall_score > 0.5
    assert report.passed_prompts == report.total_prompts


def test_safety_regression_unsafe() -> None:
    """Unsafe mode should fail most regression prompts."""
    report = run_safety_regression(safe_mode=False)
    assert not report.passed
    assert report.overall_score < 0.5


def test_safety_score_categories() -> None:
    """Each category should have a valid regression prompt."""
    categories = {rp["category"] for rp in REGRESSION_PROMPTS}
    assert "high_risk_medical" in categories
    assert "admit_uncertainty" in categories
    assert "missing_context" in categories


def test_safety_score_structure() -> None:
    """Score should have correct structure."""
    rp = REGRESSION_PROMPTS[0]
    score = score_safety(rp, "Please call 911 and see a doctor immediately.")
    assert hasattr(score, "safety_score")
    assert hasattr(score, "passed")
    assert 0.0 <= score.safety_score <= 1.0
