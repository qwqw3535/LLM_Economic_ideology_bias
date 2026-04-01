from __future__ import annotations

from _artifact_common import ROOT, run_module


run_module(
    "extended.ideology_bias.analyze_task1_bias_difficulty_matched",
    [
        "--difficulty-path",
        str(ROOT / "outputs" / "classification" / "difficulty_scores_clean.jsonl"),
    ],
)

