"""Helpers for parsing and grouping JEL codes."""

from __future__ import annotations

import json
from collections import Counter

JEL_PREFIX_NAMES = {
    "A": "General Economics and Teaching",
    "B": "History of Economic Thought",
    "C": "Mathematical and Quantitative Methods",
    "D": "Microeconomics",
    "E": "Macroeconomics and Monetary Economics",
    "F": "International Economics",
    "G": "Financial Economics",
    "H": "Public Economics",
    "I": "Health, Education, and Welfare",
    "J": "Labor and Demographic Economics",
    "K": "Law and Economics",
    "L": "Industrial Organization",
    "M": "Business Administration and Business Economics",
    "N": "Economic History",
    "O": "Economic Development and Technological Change",
    "P": "Economic Systems",
    "Q": "Agricultural and Natural Resource Economics",
    "R": "Urban, Rural, Regional, Real Estate Economics",
    "Z": "Other Special Topics",
}

COLLAPSED_JEL_GROUPS = {
    "labor": {"J"},
    "public": {"H"},
    "finance": {"G"},
    "trade": {"F"},
    "development": {"O"},
    "environment": {"Q"},
    "io": {"L"},
    "macro": {"E"},
}

_HEALTH_PREFIXES = ("I1",)
_EDUCATION_PREFIXES = ("I2",)
_WELFARE_PREFIXES = ("H5", "I3")
_TAX_PREFIXES = ("H2",)
_LABOR_PREFIXES = ("J",)
_FINANCIAL_REGULATION_PREFIXES = ("G",)
_TRADE_PREFIXES = ("F1",)

IDEOLOGY_THEME_ORDER = [
    "taxation",
    "healthcare",
    "education",
    "welfare_redistribution",
    "labor",
    "financial_regulation",
    "trade",
    "other",
]

IDEOLOGY_VOTE_THEME_PREFIXES = {
    "taxation": ("H2",),
    "healthcare": ("I1",),
    "education": ("I2",),
    "welfare_redistribution": (
        "H5",
        "I3",
    ),
    "labor": ("J",),
    "financial_regulation": ("G",),
    "trade": ("F1",),
}

IDEOLOGY_VOTE_THEME_ORDER = list(IDEOLOGY_VOTE_THEME_PREFIXES) + ["other"]


def _vote_theme_sort_key(theme: str) -> int:
    try:
        return IDEOLOGY_VOTE_THEME_ORDER.index(theme)
    except ValueError:
        return len(IDEOLOGY_VOTE_THEME_ORDER)


def _normalize_vote_theme_counts(value: object) -> dict[str, int]:
    """Normalize vote-theme count payloads from JSONL/CSV-backed sources."""
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        items = parsed.items()
    else:
        return {}

    counts: dict[str, int] = {}
    for theme, count in items:
        try:
            numeric_count = int(count)
        except (TypeError, ValueError):
            continue
        if numeric_count > 0:
            counts[str(theme)] = numeric_count
    return dict(sorted(counts.items(), key=lambda item: _vote_theme_sort_key(item[0])))


def ideology_theme_group(raw_value: object) -> str:
    """Map JEL codes into the ideology-paper themes requested for subgroup analysis."""
    codes = split_jel_codes(raw_value)
    if any(code.startswith(_WELFARE_PREFIXES) for code in codes):
        return "welfare_redistribution"
    if any(code.startswith(_HEALTH_PREFIXES) for code in codes):
        return "healthcare"
    if any(code.startswith(_EDUCATION_PREFIXES) for code in codes):
        return "education"
    if any(code.startswith(_TAX_PREFIXES) for code in codes):
        return "taxation"
    if any(code.startswith(_FINANCIAL_REGULATION_PREFIXES) for code in codes):
        return "financial_regulation"
    if any(code.startswith(_LABOR_PREFIXES) for code in codes):
        return "labor"
    if any(code.startswith(_TRADE_PREFIXES) for code in codes):
        return "trade"
    return "other"


def ideology_theme_vote_details(
    raw_value: object,
    *,
    tie_break: str = "priority",
    priority_order: list[str] | None = None,
) -> dict[str, object]:
    """Return multi-label vote-based ideology-theme assignment details for JEL codes."""
    codes = split_jel_codes(raw_value)
    counts = {theme: 0 for theme in IDEOLOGY_VOTE_THEME_PREFIXES}
    matched_themes: list[str] = []

    for code in codes:
        for theme, prefixes in IDEOLOGY_VOTE_THEME_PREFIXES.items():
            if any(code.startswith(prefix) for prefix in prefixes):
                counts[theme] += 1
                if theme not in matched_themes:
                    matched_themes.append(theme)

    nonzero_counts = {theme: count for theme, count in counts.items() if count > 0}
    if not nonzero_counts:
        return {
            "primary_theme": "other",
            "theme_counts": {},
            "theme_weights": {"other": 1.0},
            "matched_themes": [],
            "tied_themes": [],
            "max_count": 0,
            "tie_break_rule": tie_break,
            "jel_codes": codes,
        }

    max_count = max(nonzero_counts.values())
    tied_themes = [theme for theme, count in nonzero_counts.items() if count == max_count]

    if len(tied_themes) == 1:
        primary_theme = tied_themes[0]
    elif tie_break == "priority":
        ranking = priority_order or list(IDEOLOGY_VOTE_THEME_PREFIXES)
        primary_theme = min(tied_themes, key=lambda theme: ranking.index(theme) if theme in ranking else len(ranking))
    else:
        primary_theme = "other"

    if len(tied_themes) == 1:
        theme_weights = {tied_themes[0]: 1.0}
    else:
        equal_weight = 1.0 / len(tied_themes)
        theme_weights = {
            theme: equal_weight
            for theme in sorted(tied_themes, key=_vote_theme_sort_key)
        }

    return {
        "primary_theme": primary_theme,
        "theme_counts": nonzero_counts,
        "theme_weights": theme_weights,
        "matched_themes": matched_themes,
        "tied_themes": tied_themes,
        "max_count": max_count,
        "tie_break_rule": tie_break,
        "jel_codes": codes,
    }


def ideology_theme_vote_weights(
    raw_value: object | None = None,
    *,
    theme_counts: object | None = None,
) -> dict[str, float]:
    """Return analysis weights: exact top ties stay multi, otherwise use the unique max theme."""
    counts = _normalize_vote_theme_counts(theme_counts)
    if not counts:
        if raw_value is None:
            return {"other": 1.0}
        return ideology_theme_vote_details(raw_value)["theme_weights"]
    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return {"other": 1.0}
    tied_themes = [theme for theme, count in counts.items() if count == max_count]
    if len(tied_themes) == 1:
        return {tied_themes[0]: 1.0}
    equal_weight = 1.0 / len(tied_themes)
    return {
        theme: equal_weight
        for theme in sorted(tied_themes, key=_vote_theme_sort_key)
    }


def split_jel_codes(raw_value: object) -> list[str]:
    """Split raw JEL code strings into cleaned tokens."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raw_value = parsed
    raw_items = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
    tokens = []
    for raw_item in raw_items:
        text = str(raw_item).strip()
        if not text or text.lower() == "nan":
            continue
        for piece in text.replace(";", ",").split(","):
            cleaned = piece.strip().upper()
            if not cleaned or cleaned == "NAN":
                continue
            tokens.append(cleaned)
    return list(dict.fromkeys(tokens))


def jel_prefixes(raw_value: object) -> list[str]:
    """Return unique JEL letter prefixes in order of appearance."""
    prefixes: list[str] = []
    for code in split_jel_codes(raw_value):
        prefix = code[0]
        if prefix.isalpha() and prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def primary_jel_prefix(raw_value: object) -> str:
    """Return the first JEL prefix, if any."""
    prefixes = jel_prefixes(raw_value)
    return prefixes[0] if prefixes else ""


def primary_jel_name(raw_value: object) -> str:
    """Return the long-form name for the primary JEL prefix."""
    prefix = primary_jel_prefix(raw_value)
    return JEL_PREFIX_NAMES.get(prefix, "Unknown")


def collapsed_jel_group(raw_value: object) -> str:
    """Map raw JEL codes to a coarse paper subgroup used in the paper outline."""
    codes = split_jel_codes(raw_value)
    for code in codes:
        if code.startswith(_HEALTH_PREFIXES):
            return "health"
        if code.startswith(_EDUCATION_PREFIXES):
            return "education"
    prefixes = jel_prefixes(raw_value)
    for group, allowed_prefixes in COLLAPSED_JEL_GROUPS.items():
        if any(prefix in allowed_prefixes for prefix in prefixes):
            return group
    if not prefixes:
        return "other"
    return prefixes[0].lower()


def most_common_collapsed_group(values: list[object]) -> str:
    """Return the most frequent collapsed JEL group across many raw code strings."""
    counts = Counter(collapsed_jel_group(value) for value in values if value is not None)
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]
