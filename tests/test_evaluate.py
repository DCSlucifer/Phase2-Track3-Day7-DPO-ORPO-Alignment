import pytest
from preference_lab.evaluate import (
    pairwise_accuracy,
    reward_margin,
    reward_std,
    score_by_length,
    score_by_keyword_overlap,
    combined_scorer,
    score_examples,
)
from preference_lab.schemas import PreferenceExample


def _make_example(prompt: str = "p", chosen: str = "good answer", rejected: str = "bad") -> PreferenceExample:
    return PreferenceExample(prompt=prompt, chosen=chosen, rejected=rejected)


class TestPairwiseAccuracy:
    def test_perfect_accuracy(self) -> None:
        examples = [_make_example()]
        result = pairwise_accuracy(examples, [2.0], [1.0])
        assert result["accuracy"] == 1.0
        assert result["win_count"] == 1

    def test_zero_accuracy(self) -> None:
        examples = [_make_example()]
        result = pairwise_accuracy(examples, [1.0], [2.0])
        assert result["accuracy"] == 0.0
        assert result["loss_count"] == 1

    def test_ties(self) -> None:
        examples = [_make_example(), _make_example()]
        result = pairwise_accuracy(examples, [1.0, 2.0], [1.0, 1.0])
        assert result["tie_count"] == 1
        assert result["win_count"] == 1
        assert result["tie_rate"] == 0.5

    def test_empty(self) -> None:
        result = pairwise_accuracy([], [], [])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_length_mismatch(self) -> None:
        examples = [_make_example()]
        with pytest.raises(ValueError, match="Length mismatch"):
            pairwise_accuracy(examples, [1.0, 2.0], [1.0])


class TestScorers:
    def test_length_scorer(self) -> None:
        short = score_by_length("hello")
        long = score_by_length("hello world this is a longer response with more words")
        assert long > short

    def test_keyword_overlap(self) -> None:
        score = score_by_keyword_overlap(
            "Explain machine learning",
            "Machine learning is a technique that uses data to learn patterns"
        )
        assert score > 0

    def test_combined_scorer(self) -> None:
        score = combined_scorer(
            "What is deep learning?",
            "Deep learning is a subset of machine learning using neural networks"
        )
        assert 0 <= score <= 1

    def test_score_examples(self) -> None:
        examples = [_make_example("What is AI?", "Artificial intelligence is...", "It's magic")]
        chosen_scores, rejected_scores = score_examples(examples, scorer="combined")
        assert len(chosen_scores) == 1
        assert len(rejected_scores) == 1


class TestRewardMetrics:
    def test_reward_margin(self) -> None:
        margin = reward_margin([2.0, 3.0], [1.0, 1.0])
        assert margin == 1.5

    def test_reward_std(self) -> None:
        std = reward_std([1.0, 1.0, 1.0])
        assert std == 0.0

    def test_reward_std_nonzero(self) -> None:
        std = reward_std([1.0, 3.0])
        assert std > 0
