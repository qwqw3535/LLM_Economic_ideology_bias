from __future__ import annotations

from _artifact_common import ROOT, run_module


run_module(
    "extended.ideology_bias.build_ideology_triplet_subsets",
    [
        "--results",
        str(ROOT / "metadata_curated" / "classification" / "causal_triplets_multillm_ideology_qwen32b.jsonl"),
        "--task1-analysis",
        str(ROOT / "outputs" / "ideology_bias" / "analysis_datasets" / "task1_analysis_rows.csv"),
        "--subset-output",
        str(ROOT / "outputs" / "classification" / "ideology_triplet_subset_current.jsonl"),
        "--review-output",
        str(ROOT / "outputs" / "classification" / "ideology_triplet_review_sample_balanced126.jsonl"),
    ],
)

