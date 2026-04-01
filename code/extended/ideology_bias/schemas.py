"""Minimal shared helpers for released classification and ICL files."""

from __future__ import annotations

VALID_SIGNS = ("+", "-", "None", "mixed")


def normalize_sign(value: object) -> str | None:
    """Normalize raw sign strings to the released sign set."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "none":
        return "None"
    if lowered == "mixed":
        return "mixed"
    if text in VALID_SIGNS:
        return text
    mapping = {
        "positive": "+",
        "negative": "-",
        "no effect": "None",
        "null": "None",
    }
    return mapping.get(lowered)
