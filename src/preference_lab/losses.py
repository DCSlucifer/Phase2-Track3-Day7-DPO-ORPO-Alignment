from __future__ import annotations
import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-sigmoid: log(σ(x)) = -log(1 + exp(-x)).

    Uses np.logaddexp(0, -x) to avoid overflow for large negative x.
    """
    return -np.logaddexp(0.0, -x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    DPO loss = -E[log σ(β · (log π(y_w|x)/π(y_l|x) - log π_ref(y_w|x)/π_ref(y_l|x)))]

    The loss encourages the policy to increase the gap between chosen and
    rejected log-probs relative to the reference model.

    Args:
        policy_chosen_logps: Log probs of chosen responses under policy.
        policy_rejected_logps: Log probs of rejected responses under policy.
        ref_chosen_logps: Log probs of chosen responses under reference model.
        ref_rejected_logps: Log probs of rejected responses under reference model.
        beta: Temperature parameter controlling deviation from reference.

    Returns:
        Scalar DPO loss value.
    """
    # Policy log-ratio: how much policy prefers chosen over rejected
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps

    # Reference log-ratio: how much reference prefers chosen over rejected
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps

    # DPO logits: scaled difference
    logits = beta * (policy_log_ratio - ref_log_ratio)

    # Loss = -E[log σ(logits)]
    loss = -_log_sigmoid(logits).mean()

    return float(loss)


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    ORPO = SFT_loss + λ · E[-log σ(log(odds_chosen/odds_rejected))]

    ORPO combines standard SFT loss with an odds-ratio preference penalty,
    eliminating the need for a separate reference model.

    Args:
        sft_nll: Per-example negative log-likelihood (SFT loss).
        chosen_logps: Log probs of chosen responses (must be <= 0).
        rejected_logps: Log probs of rejected responses (must be <= 0).
        lambda_orpo: Weight for the preference penalty term.

    Returns:
        Scalar ORPO loss value.
    """
    # Clamp logprobs to avoid log(0) — ensure they stay in valid range
    eps = 1e-10
    chosen_logps_clamped = np.clip(chosen_logps, a_min=None, a_max=-eps)
    rejected_logps_clamped = np.clip(rejected_logps, a_min=None, a_max=-eps)

    # Log-odds: log(p / (1-p)) = logp - log(1 - exp(logp))
    # Use log1p(-exp(logp)) for numerical stability since logp < 0
    log_odds_chosen = chosen_logps_clamped - np.log1p(-np.exp(chosen_logps_clamped))
    log_odds_rejected = rejected_logps_clamped - np.log1p(-np.exp(rejected_logps_clamped))

    # Log odds ratio
    log_odds_ratio = log_odds_chosen - log_odds_rejected

    # Preference loss: -E[log σ(log_odds_ratio)]
    preference_loss = -_log_sigmoid(log_odds_ratio).mean()

    # Combined ORPO loss
    loss = sft_nll.mean() + lambda_orpo * preference_loss

    return float(loss)
