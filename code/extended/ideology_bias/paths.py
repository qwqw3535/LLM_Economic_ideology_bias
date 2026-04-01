"""Path helpers and venue/domain normalization for ideology bias analyses."""

from __future__ import annotations

import os
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = CODE_ROOT.parent
PACKAGE_DIR = Path(__file__).resolve().parent
EXTENDED_DIR = CODE_ROOT / "extended"
DATA_DIR = ARTIFACT_ROOT / "data_derived"
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"
OUTPUT_DIR = Path(os.environ.get("IDEOLOGY_BIAS_OUTPUT_DIR", str(OUTPUT_ROOT / "ideology_bias"))).resolve()
CLASSIFICATION_RESULTS_DIR = Path(os.environ.get("CLASSIFICATION_RESULTS_DIR", str(OUTPUT_ROOT / "classification"))).resolve()
METADATA_DIR = OUTPUT_DIR / "metadata"
ANALYSIS_DATASETS_DIR = OUTPUT_DIR / "analysis_datasets"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
AUDITS_DIR = OUTPUT_DIR / "audits"

FULL_CORPUS_PATH = DATA_DIR / "causal_triplets.jsonl"
DEFAULT_MERGED_CLASSIFICATION_PATH = CLASSIFICATION_RESULTS_DIR / "causal_triplets_gpt-5-mini_classified_merged.jsonl"
TASK1_ECON_PATH = DATA_DIR / "task1_econ.jsonl"
TASK1_FINANCE_PATH = DATA_DIR / "task1_finance.jsonl"
TASK2_PATH = DATA_DIR / "task2.jsonl"
TASK3_PATH = DATA_DIR / "task3.jsonl"
TASK2_EXACT50_SIDE_CAPPED_PATH = DATA_DIR / "task2_exact50_side_capped.jsonl"
TASK3_EXACT50_SIDE_CAPPED_PATH = DATA_DIR / "task3_exact50_side_capped.jsonl"
TASK2_EXACT50_ALL_PAIRS_PATH = DATA_DIR / "task2_exact50_all_pairs.jsonl"
TASK3_EXACT50_ALL_PAIRS_PATH = DATA_DIR / "task3_exact50_all_pairs.jsonl"
TASK2_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH = (
    DATA_DIR / "task2_jel_similarity_side_capped_jaccard05_shared2.jsonl"
)
TASK3_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH = (
    DATA_DIR / "task3_jel_similarity_side_capped_jaccard05_shared2.jsonl"
)

LEGACY_TASK1_CLASSIFICATION_PATHS = {
    "econ": EXTENDED_DIR / "classification_results" / "task1_econ_gpt-5-mini_classified.jsonl",
    "finance": EXTENDED_DIR / "classification_results" / "task1_finance_gpt-5-mini_classified.jsonl",
}

TASK1_RESULT_DIRS = {
    "econ": OUTPUT_ROOT / "evaluation" / "task1" / "results",
    "finance": OUTPUT_ROOT / "evaluation" / "task1" / "results",
}
TASK23_RESULT_DIR = OUTPUT_ROOT / "evaluation" / "task23" / "results"
TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR = OUTPUT_ROOT / "evaluation" / "task23_exact50_side_capped" / "results"
TASK23_EXACT50_ALL_PAIRS_RESULT_DIR = OUTPUT_ROOT / "evaluation" / "task23_exact50_all_pairs" / "results"
TASK23_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_RESULT_DIR = (
    OUTPUT_ROOT / "evaluation" / "task2" / "results"
)

DEFAULT_FULL_METADATA_RAW_PATH = METADATA_DIR / "full_corpus_metadata_raw.jsonl"
DEFAULT_FULL_METADATA_CANONICAL_JSONL = METADATA_DIR / "full_corpus_metadata_canonical.jsonl"
DEFAULT_FULL_METADATA_CANONICAL_CSV = METADATA_DIR / "full_corpus_metadata_canonical.csv"
DEFAULT_BOOTSTRAP_METADATA_JSONL = METADATA_DIR / "bootstrap_task1_metadata_canonical.jsonl"
DEFAULT_BOOTSTRAP_METADATA_CSV = METADATA_DIR / "bootstrap_task1_metadata_canonical.csv"
DEFAULT_REASONING_FRAMES_RAW_PATH = METADATA_DIR / "reasoning_frames_raw.jsonl"
DEFAULT_REASONING_FRAMES_CANONICAL_JSONL = METADATA_DIR / "reasoning_frames_canonical.jsonl"
DEFAULT_REASONING_FRAMES_CANONICAL_CSV = METADATA_DIR / "reasoning_frames_canonical.csv"
DEFAULT_REASONING_FRAMES_HEURISTIC_RAW_PATH = METADATA_DIR / "reasoning_frames_heuristic_raw.jsonl"
DEFAULT_REASONING_FRAMES_HEURISTIC_CANONICAL_JSONL = METADATA_DIR / "reasoning_frames_heuristic_canonical.jsonl"
DEFAULT_REASONING_FRAMES_HEURISTIC_CANONICAL_CSV = METADATA_DIR / "reasoning_frames_heuristic_canonical.csv"

SOURCE_CATALOG_CSV = ANALYSIS_DATASETS_DIR / "source_catalog.csv"
SOURCE_CATALOG_JSONL = ANALYSIS_DATASETS_DIR / "source_catalog.jsonl"
TASK1_ANALYSIS_CSV = ANALYSIS_DATASETS_DIR / "task1_analysis_rows.csv"
TASK1_ANALYSIS_JSONL = ANALYSIS_DATASETS_DIR / "task1_analysis_rows.jsonl"
TASK2_ANALYSIS_CSV = ANALYSIS_DATASETS_DIR / "task2_analysis_rows.csv"
TASK2_ANALYSIS_JSONL = ANALYSIS_DATASETS_DIR / "task2_analysis_rows.jsonl"
TASK3_ANALYSIS_CSV = ANALYSIS_DATASETS_DIR / "task3_analysis_rows.csv"
TASK3_ANALYSIS_JSONL = ANALYSIS_DATASETS_DIR / "task3_analysis_rows.jsonl"

NOTEBOOK_PATH = EXTENDED_DIR / "notebooks" / "ideology_bias_paper.ipynb"

_VENUE_TO_DOMAIN = {
    "american_economic_review": "econ",
    "quarterly_journal_of_economics": "econ",
    "journal_of_political_economy": "econ",
    "econometrica": "econ",
    "review_of_economic_studies": "econ",
    "journal_of_financial_economics": "finance",
    "journal_of_finance": "finance",
    "review_of_financial_studies": "finance",
}

_VENUE_ALIASES = {
    "american economic review": "american_economic_review",
    "journal of political economy": "journal_of_political_economy",
    "quarterly journal of economics": "quarterly_journal_of_economics",
    "journal of finance": "journal_of_finance",
    "journal of financial economics": "journal_of_financial_economics",
    "review of financial studies": "review_of_financial_studies",
    "review of economic studies": "review_of_economic_studies",
}


def normalize_venue(value: object) -> str:
    """Normalize raw venue strings into a stable underscore form."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    lowered = text.lower()
    if lowered in _VENUE_ALIASES:
        return _VENUE_ALIASES[lowered]
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return _VENUE_ALIASES.get(normalized.replace("_", " "), normalized)


def infer_domain_from_venue(venue: object, default: str = "unknown") -> str:
    """Infer econ vs finance domain from a published venue string."""
    normalized = normalize_venue(venue)
    return _VENUE_TO_DOMAIN.get(normalized, default)


def ensure_output_dirs() -> None:
    """Create the default output tree."""
    for path in (
        OUTPUT_DIR,
        METADATA_DIR,
        ANALYSIS_DATASETS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        AUDITS_DIR,
        NOTEBOOK_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def preferred_metadata_jsonl() -> Path:
    """Return the best available canonical metadata file."""
    if DEFAULT_FULL_METADATA_CANONICAL_JSONL.exists():
        return DEFAULT_FULL_METADATA_CANONICAL_JSONL
    return DEFAULT_BOOTSTRAP_METADATA_JSONL


def analysis_dataset_paths(task_name: str, suffix: str = "") -> tuple[Path, Path]:
    """Return CSV/JSONL output paths for a task analysis dataset."""
    stem = f"{task_name}_analysis_rows"
    if suffix:
        stem = f"{stem}_{suffix}"
    return ANALYSIS_DATASETS_DIR / f"{stem}.csv", ANALYSIS_DATASETS_DIR / f"{stem}.jsonl"


def preferred_metadata_csv() -> Path:
    """Return the best available canonical metadata CSV file."""
    if DEFAULT_FULL_METADATA_CANONICAL_CSV.exists():
        return DEFAULT_FULL_METADATA_CANONICAL_CSV
    return DEFAULT_BOOTSTRAP_METADATA_CSV


def reasoning_frame_output_paths(method: str) -> tuple[Path, Path, Path]:
    """Return default raw/jsonl/csv paths for a reasoning-frame annotation method."""
    if method == "heuristic":
        return (
            DEFAULT_REASONING_FRAMES_HEURISTIC_RAW_PATH,
            DEFAULT_REASONING_FRAMES_HEURISTIC_CANONICAL_JSONL,
            DEFAULT_REASONING_FRAMES_HEURISTIC_CANONICAL_CSV,
        )
    return (
        DEFAULT_REASONING_FRAMES_RAW_PATH,
        DEFAULT_REASONING_FRAMES_CANONICAL_JSONL,
        DEFAULT_REASONING_FRAMES_CANONICAL_CSV,
    )


def legacy_classification_output_path(input_path: str | Path, model: str) -> Path:
    """Return the legacy-style classification output path for an input file and model."""
    input_path = Path(input_path)
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(model))
    return CLASSIFICATION_RESULTS_DIR / f"{input_path.stem}_{safe_model}_classified.jsonl"
