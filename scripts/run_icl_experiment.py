from __future__ import annotations

from _artifact_common import ROOT, run_script


run_script(
    "code/econ_eval/evaluation/run_evaluation.py",
    [
        "--tasks",
        "icl_experiment",
        "--data-path",
        str(ROOT / "main_results" / "input" / "ideology_sensitive_subset_1056.jsonl"),
        "--icl-source-path",
        str(ROOT / "icl_experiment" / "input" / "jel_similarity_shared2.jsonl.gz"),
        "--output-dir",
        str(ROOT / "outputs" / "icl_experiment"),
    ],
)
