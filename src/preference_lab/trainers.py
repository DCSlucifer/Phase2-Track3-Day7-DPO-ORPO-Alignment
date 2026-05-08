from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .losses import dpo_loss, orpo_loss


@dataclass(frozen=True)
class TrainingConfig:
    method: str  # "dpo" | "orpo" | "both"
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    num_steps: int = 50
    learning_rate: float = 1e-3
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class TrainingResult:
    """Captures training run results."""
    method: str
    loss_history: list[float] = field(default_factory=list)
    final_loss: float = 0.0
    steps: int = 0
    elapsed_seconds: float = 0.0
    config: dict = field(default_factory=dict)


class PreferenceTrainer:
    """Mock CPU trainer for DPO/ORPO that demonstrates correct loss computation.

    Uses simulated log-probabilities to show loss convergence without needing
    a real model or GPU. The math is identical to production DPO/ORPO.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _simulate_logprobs(
        self, n: int, step: int, total_steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate simulated log-probs that improve over training.

        As training progresses, the policy learns to assign higher logprobs
        to chosen responses and lower to rejected ones.
        """
        progress = step / max(total_steps, 1)
        noise_scale = 0.3 * (1 - 0.5 * progress)  # noise decreases over training

        # Reference model: fixed preferences (slight preference for chosen)
        ref_chosen = self.rng.normal(-0.5, 0.2, n)
        ref_rejected = self.rng.normal(-1.0, 0.2, n)

        # Policy model: improving preferences over training
        margin = 0.3 + 1.5 * progress  # margin grows as training progresses
        policy_chosen = ref_chosen + margin * 0.5 + self.rng.normal(0, noise_scale, n)
        policy_rejected = ref_rejected - margin * 0.5 + self.rng.normal(0, noise_scale, n)

        return policy_chosen, policy_rejected, ref_chosen, ref_rejected

    def train_dpo(self) -> TrainingResult:
        """Run mock DPO training loop."""
        result = TrainingResult(
            method="dpo",
            config={
                "beta": self.config.beta,
                "num_steps": self.config.num_steps,
                "batch_size": self.config.batch_size,
            },
        )

        start = time.time()
        for step in range(self.config.num_steps):
            pc, pr, rc, rr = self._simulate_logprobs(
                self.config.batch_size, step, self.config.num_steps
            )
            loss = dpo_loss(pc, pr, rc, rr, beta=self.config.beta)
            result.loss_history.append(loss)

        result.steps = self.config.num_steps
        result.final_loss = result.loss_history[-1]
        result.elapsed_seconds = time.time() - start
        return result

    def train_orpo(self) -> TrainingResult:
        """Run mock ORPO training loop."""
        result = TrainingResult(
            method="orpo",
            config={
                "lambda_orpo": self.config.lambda_orpo,
                "num_steps": self.config.num_steps,
                "batch_size": self.config.batch_size,
            },
        )

        start = time.time()
        for step in range(self.config.num_steps):
            progress = step / max(self.config.num_steps, 1)
            n = self.config.batch_size
            noise_scale = 0.3 * (1 - 0.5 * progress)

            # SFT loss decreases over training
            sft_nll = self.rng.normal(2.0 - 1.0 * progress, 0.2, n)
            sft_nll = np.clip(sft_nll, 0.1, None)

            # Log probs for chosen/rejected
            chosen_logps = self.rng.normal(-0.3 - 0.2 * (1 - progress), noise_scale, n)
            chosen_logps = np.clip(chosen_logps, None, -1e-10)
            rejected_logps = self.rng.normal(-1.5 + 0.3 * (1 - progress), noise_scale, n)
            rejected_logps = np.clip(rejected_logps, None, -1e-10)

            loss = orpo_loss(sft_nll, chosen_logps, rejected_logps, self.config.lambda_orpo)
            result.loss_history.append(loss)

        result.steps = self.config.num_steps
        result.final_loss = result.loss_history[-1]
        result.elapsed_seconds = time.time() - start
        return result

    def train(self) -> list[TrainingResult]:
        """Train the policy using the configured method.

        Returns a list of TrainingResult (one per method).
        Side effects: saves checkpoints and metrics to output_dir.
        """
        results: list[TrainingResult] = []

        if self.config.method in ("dpo", "both"):
            results.append(self.train_dpo())

        if self.config.method in ("orpo", "both"):
            results.append(self.train_orpo())

        if not results:
            raise ValueError(f"Unknown training method: {self.config.method}")

        # Save results to output_dir
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for r in results:
            metrics = {
                "method": r.method,
                "final_loss": r.final_loss,
                "steps": r.steps,
                "elapsed_seconds": round(r.elapsed_seconds, 3),
                "loss_history": [round(l, 6) for l in r.loss_history],
                "config": r.config,
            }
            out_file = output_path / f"training_{r.method}.json"
            out_file.write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

        return results
