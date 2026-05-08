import pytest
from preference_lab.data import load_jsonl, load_jsonl_with_diagnostics, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_no_leakage() -> None:
    """Ensure no prompt appears in both train and val."""
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.3)
    train_prompts = {ex.prompt.strip().lower() for ex in train}
    val_prompts = {ex.prompt.strip().lower() for ex in val}
    assert train_prompts.isdisjoint(val_prompts), "Data leakage: prompts overlap"


def test_split_deterministic() -> None:
    """Same seed should produce same split."""
    examples = load_jsonl("data/sample_preferences.jsonl")
    train1, val1 = split_by_prompt(examples, seed=42)
    train2, val2 = split_by_prompt(examples, seed=42)
    assert [e.prompt for e in train1] == [e.prompt for e in train2]
    assert [e.prompt for e in val1] == [e.prompt for e in val2]


def test_split_different_seeds() -> None:
    """Different seeds should (very likely) produce different splits."""
    examples = load_jsonl("data/sample_preferences.jsonl")
    train1, _ = split_by_prompt(examples, seed=42)
    train2, _ = split_by_prompt(examples, seed=99)
    # With 24 examples, different seeds should give different orderings
    prompts1 = [e.prompt for e in train1]
    prompts2 = [e.prompt for e in train2]
    assert prompts1 != prompts2


def test_load_diagnostics() -> None:
    """Test that diagnostics reporting works."""
    result = load_jsonl_with_diagnostics("data/sample_preferences.jsonl")
    assert len(result.examples) == 24
    assert result.total_lines >= 24
    assert len(result.errors) == 0


def test_split_empty() -> None:
    """Split of empty list returns two empty lists."""
    train, val = split_by_prompt([])
    assert train == []
    assert val == []


def test_schema_whitespace_robust() -> None:
    """Chosen and rejected that differ only by whitespace should be rejected."""
    with pytest.raises(Exception):
        PreferenceExample(
            prompt="test",
            chosen="  hello world  ",
            rejected="hello    world",
        )


def test_schema_case_robust() -> None:
    """Chosen and rejected that differ only by case should be rejected."""
    with pytest.raises(Exception):
        PreferenceExample(
            prompt="test",
            chosen="Hello World",
            rejected="hello world",
        )
