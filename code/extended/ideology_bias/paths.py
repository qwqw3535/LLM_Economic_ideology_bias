"""Path helpers for the released artifact only."""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = CODE_ROOT.parent
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"
OUTPUT_DIR = Path(os.environ.get("IDEOLOGY_BIAS_OUTPUT_DIR", str(OUTPUT_ROOT / "difficulty_matching"))).resolve()
CLASSIFICATION_RESULTS_DIR = ARTIFACT_ROOT / "classification" / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MAIN_RESULTS_SUBSET_PATH = ARTIFACT_ROOT / "classification" / "outputs" / "ideology_sensitive_subset_current.jsonl"
MAIN_RESULTS_RESULTS_DIR = ARTIFACT_ROOT / "main_results" / "results"


def ensure_output_dirs() -> None:
    """Create the released output tree."""
    for path in (OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
