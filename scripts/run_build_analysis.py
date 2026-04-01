from __future__ import annotations

from _artifact_common import ROOT, run_module


run_module(
    "extended.ideology_bias.build_analysis_datasets",
    [
        "--metadata-jsonl",
        str(ROOT / "metadata_curated" / "metadata" / "full_corpus_metadata_canonical.jsonl"),
        "--task1-results-dir",
        str(ROOT / "outputs" / "evaluation" / "task1" / "results"),
        "--task23-results-dir",
        str(ROOT / "outputs" / "evaluation" / "task2" / "results"),
        "--task2-source",
        str(ROOT / "data_derived" / "task2_jel_similarity_side_capped_jaccard05_shared2.jsonl"),
    ],
)

