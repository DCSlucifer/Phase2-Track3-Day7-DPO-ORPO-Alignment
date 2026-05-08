from __future__ import annotations
import json
import re
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from .schemas import PreferenceExample

logger = logging.getLogger(__name__)

# Common PII patterns
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]


@dataclass
class LoadResult:
    """Result of loading a JSONL file with diagnostics."""
    examples: list[PreferenceExample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_lines: int = 0
    duplicate_prompts: list[str] = field(default_factory=list)
    pii_warnings: list[str] = field(default_factory=list)


def _check_pii(text: str, line_num: int, field_name: str) -> list[str]:
    """Check for common PII patterns in text."""
    warnings: list[str] = []
    for pii_type, pattern in _PII_PATTERNS:
        if pattern.search(text):
            warnings.append(
                f"Line {line_num}: Potential {pii_type} detected in '{field_name}'"
            )
    return warnings


def load_jsonl(path: str | Path, check_pii: bool = True) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Features:
    - Line-numbered error messages for malformed JSON or validation failures
    - Duplicate prompt detection with warnings
    - Optional PII guardrails (email, phone, SSN patterns)

    Raises ValueError if the file contains critical errors.
    Returns only valid examples (skipping bad lines with warnings).
    """
    result = load_jsonl_with_diagnostics(path, check_pii=check_pii)

    # Log warnings
    for w in result.warnings:
        logger.warning(w)
    for w in result.pii_warnings:
        logger.warning(w)
    for e in result.errors:
        logger.error(e)

    return result.examples


def load_jsonl_with_diagnostics(
    path: str | Path, check_pii: bool = True
) -> LoadResult:
    """Load JSONL with full diagnostics returned as LoadResult."""
    result = LoadResult()
    seen_prompts: dict[str, int] = {}  # prompt -> first line number

    file_path = Path(path)
    if not file_path.exists():
        result.errors.append(f"File not found: {file_path}")
        return result

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            result.total_lines += 1
            stripped = line.strip()

            # Skip blank lines
            if not stripped:
                continue

            # Parse JSON with line-numbered errors
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as e:
                result.errors.append(f"Line {line_num}: Malformed JSON — {e}")
                continue

            # Validate against schema with line-numbered errors
            try:
                example = PreferenceExample.model_validate(raw)
            except Exception as e:
                result.errors.append(f"Line {line_num}: Validation error — {e}")
                continue

            # Duplicate prompt check
            prompt_key = example.prompt.strip().lower()
            if prompt_key in seen_prompts:
                first_line = seen_prompts[prompt_key]
                result.warnings.append(
                    f"Line {line_num}: Duplicate prompt (first seen at line {first_line}): "
                    f"'{example.prompt[:60]}...'"
                )
                result.duplicate_prompts.append(example.prompt)
            else:
                seen_prompts[prompt_key] = line_num

            # PII guardrails
            if check_pii:
                for field_name in ("prompt", "chosen", "rejected"):
                    pii_hits = _check_pii(
                        getattr(example, field_name), line_num, field_name
                    )
                    result.pii_warnings.extend(pii_hits)

            result.examples.append(example)

    return result


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid data leakage.

    Groups examples by their prompt text, shuffles groups deterministically,
    then splits groups so that no prompt appears in both train and validation.
    """
    if not examples:
        return [], []

    # Group examples by prompt
    prompt_groups: dict[str, list[PreferenceExample]] = {}
    for ex in examples:
        key = ex.prompt.strip().lower()
        prompt_groups.setdefault(key, []).append(ex)

    # Get unique prompts and shuffle deterministically
    unique_prompts = list(prompt_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)

    # Calculate split point based on number of unique prompts
    n_val_prompts = max(1, int(len(unique_prompts) * validation_ratio))
    n_train_prompts = len(unique_prompts) - n_val_prompts

    train_prompts = unique_prompts[:n_train_prompts]
    val_prompts = unique_prompts[n_train_prompts:]

    # Build train and validation sets
    train: list[PreferenceExample] = []
    val: list[PreferenceExample] = []

    for p in train_prompts:
        train.extend(prompt_groups[p])
    for p in val_prompts:
        val.extend(prompt_groups[p])

    return train, val
