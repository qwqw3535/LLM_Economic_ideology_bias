from __future__ import annotations

from _artifact_common import ROOT, run_script


run_script(
    "code/extended/evaluate_difficulty.py",
    [
        "--input",
        str(ROOT / "metadata_curated" / "classification" / "causal_triplets_gpt-5-mini_classified_merged.jsonl"),
        "--output",
        str(ROOT / "outputs" / "classification" / "difficulty_scores.jsonl"),
    ],
)

