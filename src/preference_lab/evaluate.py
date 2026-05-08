from __future__ import annotations
import json
import math
from pathlib import Path
from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> dict[str, float | int]:
    """Return pairwise accuracy with detailed breakdown.

    Validates input lengths, handles ties explicitly, and returns
    enriched metrics including win/loss/tie counts and rates.
    """
    if not examples:
        return {
            "accuracy": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "tie_count": 0,
            "tie_rate": 0.0,
            "total": 0,
        }

    # Validate lengths
    if len(examples) != len(chosen_scores):
        raise ValueError(
            f"Length mismatch: {len(examples)} examples vs {len(chosen_scores)} chosen_scores"
        )
    if len(examples) != len(rejected_scores):
        raise ValueError(
            f"Length mismatch: {len(examples)} examples vs {len(rejected_scores)} rejected_scores"
        )

    wins = 0
    losses = 0
    ties = 0
    for c, r in zip(chosen_scores, rejected_scores, strict=True):
        if c > r:
            wins += 1
        elif c < r:
            losses += 1
        else:
            ties += 1

    total = len(examples)
    return {
        "accuracy": wins / total,
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
        "tie_rate": ties / total,
        "total": total,
    }


def reward_margin(
    chosen_scores: list[float], rejected_scores: list[float]
) -> float:
    """Average margin between chosen and rejected scores.

    Higher margin means the scorer more confidently prefers chosen.
    """
    if not chosen_scores:
        return 0.0
    margins = [c - r for c, r in zip(chosen_scores, rejected_scores, strict=True)]
    return sum(margins) / len(margins)


def reward_std(scores: list[float]) -> float:
    """Standard deviation of reward scores."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
    return math.sqrt(variance)


def score_by_length(response: str) -> float:
    """Score based on response length (log-scaled).

    Longer, more detailed responses tend to be higher quality.
    """
    return math.log1p(len(response.split()))


def score_by_keyword_overlap(prompt: str, response: str) -> float:
    """Score based on keyword overlap between prompt and response.

    Relevant responses should contain key terms from the prompt.
    """
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    # Remove stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "not", "with", "what",
        "how", "why", "when", "where", "which", "do", "does", "did",
        "this", "that", "it", "its", "be", "been", "being", "have",
        "has", "had", "will", "would", "could", "should", "can",
    }
    prompt_keywords = prompt_words - stop_words
    if not prompt_keywords:
        return 0.0
    overlap = prompt_keywords & response_words
    return len(overlap) / len(prompt_keywords)


def combined_scorer(prompt: str, response: str) -> float:
    """Combined scoring: 0.5 * length_score + 0.5 * keyword_score."""
    length_score = score_by_length(response)
    keyword_score = score_by_keyword_overlap(prompt, response)
    # Normalize length score to roughly 0-1 range (log of typical response ~2-4)
    normalized_length = min(length_score / 4.0, 1.0)
    return 0.5 * normalized_length + 0.5 * keyword_score


def score_examples(
    examples: list[PreferenceExample],
    scorer: str = "combined",
) -> tuple[list[float], list[float]]:
    """Score all examples using the specified scorer.

    Args:
        examples: List of preference examples.
        scorer: One of "mock", "length", "keyword", "combined".

    Returns:
        Tuple of (chosen_scores, rejected_scores).
    """
    score_fn = {
        "mock": lambda p, r: 1.0,
        "length": lambda p, r: score_by_length(r),
        "keyword": lambda p, r: score_by_keyword_overlap(p, r),
        "combined": lambda p, r: combined_scorer(p, r),
    }.get(scorer, lambda p, r: combined_scorer(p, r))

    chosen_scores = [score_fn(ex.prompt, ex.chosen) for ex in examples]
    rejected_scores = [score_fn(ex.prompt, ex.rejected) for ex in examples]

    return chosen_scores, rejected_scores


def write_metrics(metrics: dict, output_dir: str | Path) -> Path:
    """Write metrics dict to JSON file."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
