from __future__ import annotations

from _artifact_common import ROOT, run_script


run_script(
    "code/econ_eval/evaluation/run_evaluation.py",
    [
        "--tasks",
        "main_results",
        "--data-path",
        str(ROOT / "main_results" / "input" / "ideology_sensitive_subset_1056.jsonl"),
        "--output-dir",
        str(ROOT / "outputs" / "main_results"),
    ],
)
