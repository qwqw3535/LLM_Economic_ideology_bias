"""Heuristic, keyword-based reasoning-frame annotation."""

from __future__ import annotations

import re
from collections import defaultdict

from .schemas import REASONING_FRAME_VALUES


FRAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "efficiency": (
        "efficient",
        "efficiency",
        "allocative",
        "deadweight",
        "surplus",
        "welfare gain",
        "welfare loss",
        "cost-benefit",
        "optimal allocation",
    ),
    "incentives": (
        "incentive",
        "incentives",
        "behavioral response",
        "respond",
        "discourage",
        "encourage",
        "moral hazard",
        "work effort",
        "labor supply",
        "take-up",
    ),
    "market_distortion": (
        "distortion",
        "distortions",
        "misallocation",
        "market power",
        "crowding out",
        "crowd out",
        "rent-seeking",
        "price control",
        "binding constraint",
        "regulation",
        "deregulation",
    ),
    "productivity": (
        "productivity",
        "tfp",
        "output per worker",
        "innovation",
        "technology adoption",
        "human capital",
        "capital accumulation",
    ),
    "redistribution": (
        "redistribution",
        "redistributive",
        "inequality",
        "poverty",
        "transfer",
        "welfare",
        "income support",
        "safety net",
        "distributional",
        "equity",
        "progressive",
        "regressive",
    ),
    "insurance": (
        "insurance",
        "risk-sharing",
        "risk sharing",
        "consumption smoothing",
        "smoothing",
        "uncertainty",
        "precautionary",
        "buffer stock",
        "hedging",
        "social insurance",
    ),
    "externalities": (
        "externality",
        "externalities",
        "spillover",
        "spillovers",
        "pollution",
        "congestion",
        "social cost",
        "peer effect",
        "network effect",
        "third-party",
    ),
    "fiscal_burden": (
        "fiscal",
        "budget",
        "deficit",
        "debt",
        "revenue",
        "taxpayer",
        "public spending",
        "government spending",
        "expenditure",
        "budgetary",
    ),
    "state_capacity": (
        "state capacity",
        "administrative capacity",
        "implementation capacity",
        "institutional capacity",
        "enforcement",
        "compliance",
        "monitoring",
        "bureaucracy",
        "governance",
        "tax capacity",
    ),
}


def _count_keyword(text: str, keyword: str) -> int:
    if " " in keyword or "-" in keyword:
        return text.count(keyword)
    pattern = re.compile(rf"\b{re.escape(keyword)}\b")
    return len(pattern.findall(text))


def annotate_reasoning_heuristic(reasoning: str) -> dict:
    """Return a controlled-vocabulary frame annotation without using an LLM."""
    text = (reasoning or "").strip().lower()
    if not text:
        return {
            "reasoning_frames": {
                "primary_frame": "other",
                "secondary_frames": [],
                "justification": "No reasoning text was available.",
            },
            "matched_keywords": {},
            "frame_scores": {},
        }

    frame_matches: dict[str, list[str]] = defaultdict(list)
    frame_scores: dict[str, int] = {}

    for frame in REASONING_FRAME_VALUES:
        if frame == "other":
            continue
        score = 0
        for keyword in FRAME_KEYWORDS.get(frame, ()):
            hits = _count_keyword(text, keyword)
            if hits > 0:
                frame_matches[frame].append(keyword)
                score += hits
        frame_scores[frame] = score

    ranked = sorted(frame_scores.items(), key=lambda item: (-item[1], item[0]))
    positive = [(frame, score) for frame, score in ranked if score > 0]
    if not positive:
        primary = "other"
        secondary: list[str] = []
        justification = "No controlled-vocabulary keyword matches were found in the reasoning text."
    else:
        primary = positive[0][0]
        secondary = [frame for frame, _ in positive[1:4]]
        matched = frame_matches.get(primary, [])
        explanation = ", ".join(matched[:4]) if matched else "scored highest by keyword overlap"
        justification = f"Heuristic keyword match favored `{primary}` based on: {explanation}."

    return {
        "reasoning_frames": {
            "primary_frame": primary,
            "secondary_frames": secondary,
            "justification": justification,
        },
        "matched_keywords": {frame: keywords for frame, keywords in sorted(frame_matches.items())},
        "frame_scores": frame_scores,
    }
