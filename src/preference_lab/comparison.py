"""DPO vs ORPO comparison module.

Runs both methods on the same data and generates a structured comparison.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from .trainers import TrainingConfig, PreferenceTrainer, TrainingResult


@dataclass
class ComparisonResult:
    """Side-by-side comparison of DPO and ORPO."""
    dpo: TrainingResult | None = None
    orpo: TrainingResult | None = None
    winner: str = ""
    analysis: dict[str, str] = field(default_factory=dict)


def run_comparison(
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    num_steps: int = 50,
    batch_size: int = 4,
    seed: int = 42,
) -> ComparisonResult:
    """Run DPO and ORPO training with identical settings and compare.

    Returns:
        ComparisonResult with both results and analysis.
    """
    # DPO training
    dpo_config = TrainingConfig(
        method="dpo",
        beta=beta,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        output_dir="outputs",
    )
    dpo_trainer = PreferenceTrainer(dpo_config)
    dpo_result = dpo_trainer.train_dpo()

    # ORPO training (same seed for fair comparison)
    orpo_config = TrainingConfig(
        method="orpo",
        lambda_orpo=lambda_orpo,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        output_dir="outputs",
    )
    orpo_trainer = PreferenceTrainer(orpo_config)
    orpo_result = orpo_trainer.train_orpo()

    # Analysis
    dpo_reduction = dpo_result.loss_history[0] - dpo_result.final_loss
    orpo_reduction = orpo_result.loss_history[0] - orpo_result.final_loss

    dpo_convergence = _convergence_step(dpo_result.loss_history)
    orpo_convergence = _convergence_step(orpo_result.loss_history)

    winner = "dpo" if dpo_result.final_loss < orpo_result.final_loss else "orpo"

    comparison = ComparisonResult(
        dpo=dpo_result,
        orpo=orpo_result,
        winner=winner,
        analysis={
            "dpo_initial_loss": f"{dpo_result.loss_history[0]:.6f}",
            "dpo_final_loss": f"{dpo_result.final_loss:.6f}",
            "dpo_reduction": f"{dpo_reduction:.6f}",
            "dpo_convergence_step": str(dpo_convergence),
            "orpo_initial_loss": f"{orpo_result.loss_history[0]:.6f}",
            "orpo_final_loss": f"{orpo_result.final_loss:.6f}",
            "orpo_reduction": f"{orpo_reduction:.6f}",
            "orpo_convergence_step": str(orpo_convergence),
            "winner": winner,
            "reason": (
                f"{winner.upper()} achieved lower final loss. "
                f"DPO: {dpo_result.final_loss:.4f} vs ORPO: {orpo_result.final_loss:.4f}"
            ),
        },
    )

    return comparison


def _convergence_step(loss_history: list[float], threshold: float = 0.01) -> int:
    """Find the step where loss change falls below threshold (convergence)."""
    for i in range(1, len(loss_history)):
        if abs(loss_history[i] - loss_history[i - 1]) < threshold:
            return i
    return len(loss_history)


def save_comparison(comparison: ComparisonResult, output_dir: str | Path) -> Path:
    """Save comparison results as JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "comparison.json"

    data = {
        "winner": comparison.winner,
        "analysis": comparison.analysis,
        "dpo": {
            "loss_history": [round(l, 6) for l in (comparison.dpo.loss_history if comparison.dpo else [])],
            "final_loss": comparison.dpo.final_loss if comparison.dpo else 0,
            "steps": comparison.dpo.steps if comparison.dpo else 0,
        },
        "orpo": {
            "loss_history": [round(l, 6) for l in (comparison.orpo.loss_history if comparison.orpo else [])],
            "final_loss": comparison.orpo.final_loss if comparison.orpo else 0,
            "steps": comparison.orpo.steps if comparison.orpo else 0,
        },
    }

    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
