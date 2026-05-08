import numpy as np
import pytest
from preference_lab.losses import dpo_loss, orpo_loss


class TestDPOLoss:
    """Tests for DPO loss function."""

    def test_dpo_loss_basic(self) -> None:
        """When policy matches reference, loss should be ~log(2) ≈ 0.693."""
        loss = dpo_loss(
            np.array([-0.5, -0.5]),
            np.array([-1.5, -1.5]),
            np.array([-0.5, -0.5]),
            np.array([-1.5, -1.5]),
            beta=0.1,
        )
        # When policy = reference, logits = 0, loss = -log(sigmoid(0)) = log(2)
        assert abs(loss - np.log(2)) < 0.01

    def test_dpo_loss_positive(self) -> None:
        """DPO loss should always be positive."""
        loss = dpo_loss(
            np.array([-0.3, -0.4]),
            np.array([-1.5, -2.0]),
            np.array([-0.5, -0.6]),
            np.array([-1.0, -1.2]),
            beta=0.1,
        )
        assert loss > 0

    def test_dpo_loss_decreases_with_margin(self) -> None:
        """Higher policy preference margin should give lower loss."""
        # Small margin
        loss_small = dpo_loss(
            np.array([-0.5]), np.array([-1.0]),
            np.array([-0.5]), np.array([-1.0]),
            beta=0.1,
        )
        # Large margin
        loss_large = dpo_loss(
            np.array([-0.1]), np.array([-3.0]),
            np.array([-0.5]), np.array([-1.0]),
            beta=0.1,
        )
        assert loss_large < loss_small

    def test_dpo_loss_numerical_stability(self) -> None:
        """Should handle extreme values without NaN or Inf."""
        loss = dpo_loss(
            np.array([-0.001]), np.array([-100.0]),
            np.array([-0.001]), np.array([-100.0]),
            beta=10.0,
        )
        assert np.isfinite(loss)

    def test_dpo_loss_batch(self) -> None:
        """Should handle batches correctly."""
        loss = dpo_loss(
            np.array([-0.5, -0.3, -0.7]),
            np.array([-1.5, -2.0, -1.0]),
            np.array([-0.6, -0.4, -0.8]),
            np.array([-1.0, -1.5, -0.9]),
            beta=0.1,
        )
        assert isinstance(loss, float)
        assert np.isfinite(loss)


class TestORPOLoss:
    """Tests for ORPO loss function."""

    def test_orpo_loss_basic(self) -> None:
        """ORPO loss should return a finite value."""
        loss = orpo_loss(
            np.array([1.0, 1.5]),
            np.array([-0.5, -0.3]),
            np.array([-1.5, -2.0]),
            lambda_orpo=0.1,
        )
        assert np.isfinite(loss)
        assert loss > 0

    def test_orpo_loss_sft_component(self) -> None:
        """Higher SFT NLL should increase total loss."""
        loss_low = orpo_loss(
            np.array([0.5]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1
        )
        loss_high = orpo_loss(
            np.array([5.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1
        )
        assert loss_high > loss_low

    def test_orpo_loss_preference_component(self) -> None:
        """When chosen is much better, preference loss should be lower."""
        # Chosen much better than rejected
        loss_good = orpo_loss(
            np.array([1.0]), np.array([-0.1]), np.array([-3.0]), lambda_orpo=1.0
        )
        # Chosen slightly better
        loss_close = orpo_loss(
            np.array([1.0]), np.array([-0.9]), np.array([-1.0]), lambda_orpo=1.0
        )
        assert loss_good < loss_close

    def test_orpo_loss_numerical_stability(self) -> None:
        """Should handle extreme logprob values."""
        loss = orpo_loss(
            np.array([1.0]),
            np.array([-0.001]),
            np.array([-50.0]),
            lambda_orpo=0.1,
        )
        assert np.isfinite(loss)
