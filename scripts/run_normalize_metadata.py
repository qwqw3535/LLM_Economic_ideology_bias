from __future__ import annotations

from pathlib import Path

from _artifact_common import ROOT, run_module


run_module(
    "extended.ideology_bias.normalize_metadata",
    [
        "--input",
        str(ROOT / "metadata_curated" / "classification" / "causal_triplets_gpt-5-mini_classified_merged.jsonl"),
        "--source",
        "legacy",
        "--output-jsonl",
        str(ROOT / "outputs" / "ideology_bias" / "metadata" / "full_corpus_metadata_canonical.jsonl"),
        "--output-csv",
        str(ROOT / "outputs" / "ideology_bias" / "metadata" / "full_corpus_metadata_canonical.csv"),
    ],
)

