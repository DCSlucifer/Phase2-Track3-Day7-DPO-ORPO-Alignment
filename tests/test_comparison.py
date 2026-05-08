from preference_lab.comparison import run_comparison


def test_comparison_runs() -> None:
    """Comparison should complete and return results for both methods."""
    result = run_comparison(num_steps=10, batch_size=2)
    assert result.dpo is not None
    assert result.orpo is not None
    assert result.winner in ("dpo", "orpo")


def test_comparison_has_loss_history() -> None:
    """Both methods should have loss histories."""
    result = run_comparison(num_steps=20, batch_size=2)
    assert len(result.dpo.loss_history) == 20
    assert len(result.orpo.loss_history) == 20


def test_comparison_analysis() -> None:
    """Analysis dict should contain expected keys."""
    result = run_comparison(num_steps=10)
    assert "dpo_final_loss" in result.analysis
    assert "orpo_final_loss" in result.analysis
    assert "winner" in result.analysis
    assert "reason" in result.analysis
