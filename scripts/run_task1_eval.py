from __future__ import annotations

from _artifact_common import ROOT, run_script


run_script(
    "code/econ_eval/evaluation/run_evaluation.py",
    [
        "--tasks",
        "task1",
        "--data-path",
        str(ROOT / "data_derived" / "task1_ideology_subset_1056.jsonl"),
        "--output-dir",
        str(ROOT / "outputs" / "evaluation" / "task1"),
    ],
)

