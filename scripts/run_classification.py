from __future__ import annotations

from pathlib import Path

from _artifact_common import ROOT, run_script


run_script(
    "code/extended/classify_triplets.py",
    [
        "--output",
        str(ROOT / "outputs" / "classification" / "classified_triplets_multillm.jsonl"),
    ],
)
