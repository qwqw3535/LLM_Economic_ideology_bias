from __future__ import annotations

from _artifact_common import ROOT, run_module


run_module(
    "extended.ideology_bias.generate_icl_experiment_shared2",
    [
        "--subset-path",
        str(ROOT / "classification" / "outputs" / "ideology_sensitive_subset_current.jsonl"),
        "--icl-path",
        str(ROOT / "outputs" / "icl_experiment" / "jel_similarity_shared2.jsonl"),
    ],
)
