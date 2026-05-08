from __future__ import annotations
import re
import difflib
from typing import Any
from pydantic import BaseModel, Field, field_validator


def _normalize(text: str) -> str:
    """Normalize text for comparison: strip, lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is None:
            return rejected

        # Normalize both for robust comparison (whitespace, case)
        norm_chosen = _normalize(chosen)
        norm_rejected = _normalize(rejected)

        # Exact match after normalization
        if norm_chosen == norm_rejected:
            raise ValueError("chosen and rejected must differ (identical after normalization)")

        # Near-duplicate check using SequenceMatcher (threshold: 95% similarity)
        similarity = difflib.SequenceMatcher(None, norm_chosen, norm_rejected).ratio()
        if similarity > 0.95:
            raise ValueError(
                f"chosen and rejected are near-duplicates "
                f"(similarity={similarity:.2%}, threshold=95%)"
            )

        return rejected
