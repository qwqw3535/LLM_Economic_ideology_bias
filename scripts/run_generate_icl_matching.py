from __future__ import annotations

import sys

from _artifact_common import ROOT, run_module


GENERATOR = "ideology_subset_shared2"
if len(sys.argv) >= 3 and sys.argv[1] == "--generator":
    GENERATOR = sys.argv[2]
    del sys.argv[1:3]

if GENERATOR == "full_exact50":
    run_module("econ_eval.evaluation.generate_task23_exact50")
elif GENERATOR == "ideology_subset_shared2":
    run_module(
        "extended.ideology_bias.generate_task23_ideology_subset_shared2",
        [
            "--subset-path",
            str(ROOT / "metadata_curated" / "classification" / "ideology_triplet_subset_current.jsonl"),
            "--task2-path",
            str(ROOT / "outputs" / "matching" / "task2_ideology_subset_jel_similarity_side_capped_jaccard05_shared2.jsonl"),
            "--task3-path",
            str(ROOT / "outputs" / "matching" / "task3_ideology_subset_jel_similarity_side_capped_jaccard05_shared2.jsonl"),
        ],
    )
elif GENERATOR == "ideology_subset_variants":
    run_module(
        "extended.ideology_bias.generate_task23_ideology_subset_variants",
        [
            "--subset-path",
            str(ROOT / "metadata_curated" / "classification" / "ideology_triplet_subset_current.jsonl"),
            "--output-dir",
            str(ROOT / "outputs" / "matching"),
            "--summary-path",
            str(ROOT / "outputs" / "matching" / "task23_ideology_subset_variant_summaries.json"),
        ],
    )
else:
    raise SystemExit(f"Unknown --generator value: {GENERATOR}")

