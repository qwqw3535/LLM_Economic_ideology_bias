from __future__ import annotations

from _artifact_common import ROOT, run_module


run_module(
    "extended.ideology_bias.analyze_main_results_difficulty_matched",
    [
        "--difficulty-path",
        str(ROOT / "difficulty_matching" / "outputs" / "difficulty_scores_clean.jsonl"),
    ],
)
