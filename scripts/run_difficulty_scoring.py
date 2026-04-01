from __future__ import annotations

from _artifact_common import ROOT, run_script


run_script(
    "code/extended/evaluate_difficulty.py",
    [
        "--input",
        str(ROOT / "classification" / "outputs" / "classified_triplets_merged.jsonl.gz"),
        "--output",
        str(ROOT / "outputs" / "difficulty_matching" / "difficulty_scores.jsonl"),
    ],
)
