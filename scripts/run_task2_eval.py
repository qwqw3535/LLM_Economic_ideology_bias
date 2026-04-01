from __future__ import annotations

from _artifact_common import ROOT, run_script


run_script(
    "code/econ_eval/evaluation/run_evaluation.py",
    [
        "--tasks",
        "task2",
        "--data-path",
        str(ROOT / "data_derived" / "task1_ideology_subset_1056.jsonl"),
        "--task2-source-path",
        str(ROOT / "data_derived" / "task2_jel_similarity_side_capped_jaccard05_shared2.jsonl.gz"),
        "--task3-source-path",
        str(ROOT / "data_derived" / "task2_jel_similarity_side_capped_jaccard05_shared2.jsonl.gz"),
        "--output-dir",
        str(ROOT / "outputs" / "evaluation" / "task2"),
    ],
)
