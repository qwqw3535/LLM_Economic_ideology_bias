"""Task 1 ideology-side analysis with difficulty-matched liberal/conservative triplets.

Loads LLM-generated difficulty scores, matches liberal and conservative triplets
so that the difficulty distribution is balanced, then re-runs accuracy comparisons
per model (overall and by vote theme).

This controls for the possibility that accuracy gaps between liberal- and
conservative-ground-truth triplets are driven by inherent question difficulty
rather than model ideological priors.

Usage:
    python -m extended.ideology_bias.analyze_task1_bias_difficulty_matched \
        [--difficulty-path extended/classification_results/difficulty_scores_clean.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by, load_dataset, save_frame
from .jel import ideology_theme_vote_details
from .paths import (
    CLASSIFICATION_RESULTS_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    SOURCE_CATALOG_JSONL,
    TABLES_DIR,
    TASK1_ANALYSIS_JSONL,
    ensure_output_dirs,
)


IDEOLOGY_GROUND_TRUTH = {"liberal", "conservative"}
THEME_COL = "jel_policy_theme_vote_primary"
DEFAULT_DIFFICULTY_PATH = CLASSIFICATION_RESULTS_DIR / "difficulty_scores_clean.jsonl"

# Dedicated output directories for difficulty-matched results
DIFF_TABLES_DIR = OUTPUT_DIR / "difficulty_matched" / "tables"
DIFF_FIGURES_DIR = OUTPUT_DIR / "difficulty_matched" / "figures"
DIFF_REPORTS_DIR = OUTPUT_DIR / "difficulty_matched" / "reports"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_frame(path: str) -> pd.DataFrame:
    frame = load_dataset(path)
    if frame.empty:
        raise RuntimeError(f"No rows found in {path}")

    for column, default in (
        (THEME_COL, "other"),
        ("model", "unknown"),
        ("family", "unknown"),
        ("parameter_bucket", "unknown"),
        ("ground_truth_side", "unlabeled"),
    ):
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)

    if THEME_COL not in frame.columns or frame[THEME_COL].eq("other").all():
        raw_jel = None
        if "jel_codes" in frame.columns:
            raw_jel = frame["jel_codes"]
        elif "jel_codes_raw" in frame.columns:
            raw_jel = frame["jel_codes_raw"]
        if raw_jel is not None:
            frame[THEME_COL] = raw_jel.apply(lambda v: ideology_theme_vote_details(v)["primary_theme"])

    frame["jel_policy_theme"] = frame[THEME_COL].fillna("other")

    for column in (
        "ground_truth_liberal",
        "predicted_liberal",
        "predicted_conservative",
        "ideology_triplet_labeled",
        "liberal_leaning_error",
        "conservative_leaning_error",
        "correct",
    ):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype(int)

    return frame


def _load_difficulty(path: str | Path) -> dict[str, int]:
    """Load triplet_key -> overall_difficulty mapping."""
    lookup: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("overall_difficulty") is not None:
                lookup[row["triplet_key"]] = int(row["overall_difficulty"])
    return lookup


def _difficulty_level_label(value: object) -> str:
    """Render a stable human-readable difficulty label."""
    return f"difficulty level {int(value)}"


def _add_difficulty_label_column(
    frame: pd.DataFrame,
    source_col: str = "overall_difficulty",
    target_col: str = "difficulty_level",
) -> pd.DataFrame:
    """Attach a string difficulty label so figure rows can show the level."""
    enriched = frame.copy()
    enriched[target_col] = enriched[source_col].apply(_difficulty_level_label)
    return enriched


# ---------------------------------------------------------------------------
# Difficulty matching
# ---------------------------------------------------------------------------


def difficulty_match_triplets(
    frame: pd.DataFrame,
    difficulty_lookup: dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """Match liberal and conservative triplets by difficulty level.

    For each difficulty level (1-5), take the minimum count between liberal
    and conservative triplets, then randomly sample that many from each side.
    Returns a filtered frame containing only the matched triplet predictions.
    """
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled["overall_difficulty"] = labeled["triplet_key"].map(difficulty_lookup)
    labeled = labeled.dropna(subset=["overall_difficulty"])
    labeled["overall_difficulty"] = labeled["overall_difficulty"].astype(int)

    # Get unique triplets per side per difficulty
    triplet_side = (
        labeled[["triplet_key", "ground_truth_side", "overall_difficulty"]]
        .drop_duplicates(subset=["triplet_key"])
    )

    rng = np.random.RandomState(seed)
    matched_keys: set[str] = set()

    for diff_level in sorted(triplet_side["overall_difficulty"].unique()):
        lib_keys = triplet_side[
            (triplet_side["overall_difficulty"] == diff_level)
            & (triplet_side["ground_truth_side"] == "liberal")
        ]["triplet_key"].tolist()
        cons_keys = triplet_side[
            (triplet_side["overall_difficulty"] == diff_level)
            & (triplet_side["ground_truth_side"] == "conservative")
        ]["triplet_key"].tolist()

        n_match = min(len(lib_keys), len(cons_keys))
        if n_match == 0:
            continue

        rng.shuffle(lib_keys)
        rng.shuffle(cons_keys)
        matched_keys.update(lib_keys[:n_match])
        matched_keys.update(cons_keys[:n_match])

    return labeled[labeled["triplet_key"].isin(matched_keys)].copy()


def difficulty_match_by_theme(
    frame: pd.DataFrame,
    difficulty_lookup: dict[str, int],
    theme_col: str = "jel_policy_theme",
    seed: int = 42,
) -> pd.DataFrame:
    """Match liberal/conservative triplets by difficulty within each theme."""
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled["overall_difficulty"] = labeled["triplet_key"].map(difficulty_lookup)
    labeled = labeled.dropna(subset=["overall_difficulty"])
    labeled["overall_difficulty"] = labeled["overall_difficulty"].astype(int)

    triplet_info = (
        labeled[["triplet_key", "ground_truth_side", "overall_difficulty", theme_col]]
        .drop_duplicates(subset=["triplet_key"])
    )

    rng = np.random.RandomState(seed)
    matched_keys: set[str] = set()

    for theme in triplet_info[theme_col].unique():
        theme_triplets = triplet_info[triplet_info[theme_col] == theme]
        for diff_level in theme_triplets["overall_difficulty"].unique():
            lib_keys = theme_triplets[
                (theme_triplets["overall_difficulty"] == diff_level)
                & (theme_triplets["ground_truth_side"] == "liberal")
            ]["triplet_key"].tolist()
            cons_keys = theme_triplets[
                (theme_triplets["overall_difficulty"] == diff_level)
                & (theme_triplets["ground_truth_side"] == "conservative")
            ]["triplet_key"].tolist()

            n_match = min(len(lib_keys), len(cons_keys))
            if n_match == 0:
                continue

            rng.shuffle(lib_keys)
            rng.shuffle(cons_keys)
            matched_keys.update(lib_keys[:n_match])
            matched_keys.update(cons_keys[:n_match])

    return labeled[labeled["triplet_key"].isin(matched_keys)].copy()


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _difficulty_distribution(
    frame: pd.DataFrame,
    difficulty_lookup: dict[str, int],
) -> pd.DataFrame:
    """Show difficulty distribution by ground_truth_side."""
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled["overall_difficulty"] = labeled["triplet_key"].map(difficulty_lookup)
    labeled = labeled.dropna(subset=["overall_difficulty"])

    triplet_info = (
        labeled[["triplet_key", "ground_truth_side", "overall_difficulty"]]
        .drop_duplicates(subset=["triplet_key"])
    )

    dist = (
        triplet_info.groupby(["ground_truth_side", "overall_difficulty"])
        .size()
        .reset_index(name="n_triplets")
        .sort_values(["ground_truth_side", "overall_difficulty"])
    )
    return _add_difficulty_label_column(dist)


def _triplet_theme_difficulty_frame(
    frame: pd.DataFrame,
    difficulty_lookup: dict[str, int],
    theme_col: str = "jel_policy_theme",
) -> pd.DataFrame:
    """Return one row per unique triplet with side/theme/difficulty metadata."""
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled["overall_difficulty"] = labeled["triplet_key"].map(difficulty_lookup)
    labeled = labeled.dropna(subset=["overall_difficulty"])
    labeled["overall_difficulty"] = labeled["overall_difficulty"].astype(int)
    triplets = (
        labeled[["triplet_key", "ground_truth_side", "overall_difficulty", theme_col]]
        .drop_duplicates(subset=["triplet_key"])
        .rename(columns={theme_col: "jel_policy_theme"})
    )
    return _add_difficulty_label_column(triplets)


def _theme_matching_summary(
    frame: pd.DataFrame,
    matched_theme: pd.DataFrame,
    difficulty_lookup: dict[str, int],
    theme_col: str = "jel_policy_theme",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize theme-level matching before/after by difficulty level."""
    before = _triplet_theme_difficulty_frame(frame, difficulty_lookup, theme_col=theme_col)
    after = _triplet_theme_difficulty_frame(matched_theme, difficulty_lookup, theme_col=theme_col)

    before_counts = (
        before.groupby(
            ["jel_policy_theme", "overall_difficulty", "difficulty_level", "ground_truth_side"],
            dropna=False,
            observed=False,
        )["triplet_key"]
        .nunique()
        .reset_index(name="n_triplets_before")
    )
    after_counts = (
        after.groupby(
            ["jel_policy_theme", "overall_difficulty", "difficulty_level", "ground_truth_side"],
            dropna=False,
            observed=False,
        )["triplet_key"]
        .nunique()
        .reset_index(name="n_triplets_after")
    )

    long = before_counts.merge(
        after_counts,
        on=["jel_policy_theme", "overall_difficulty", "difficulty_level", "ground_truth_side"],
        how="outer",
    )
    long["n_triplets_before"] = long["n_triplets_before"].fillna(0).astype(int)
    long["n_triplets_after"] = long["n_triplets_after"].fillna(0).astype(int)
    denominator = long["n_triplets_before"].where(long["n_triplets_before"] != 0)
    long["retained_rate"] = (long["n_triplets_after"] / denominator).astype(float).fillna(0.0)
    long = long.sort_values(
        ["jel_policy_theme", "overall_difficulty", "ground_truth_side"]
    ).reset_index(drop=True)

    side_pivot = (
        long.pivot_table(
            index=["jel_policy_theme", "overall_difficulty", "difficulty_level"],
            columns="ground_truth_side",
            values=["n_triplets_before", "n_triplets_after", "retained_rate"],
            aggfunc="first",
        )
        .fillna(0)
    )
    side_pivot.columns = [
        f"{metric}_{side}"
        for metric, side in side_pivot.columns.to_flat_index()
    ]
    summary = side_pivot.reset_index()
    for column in [
        "n_triplets_before_liberal",
        "n_triplets_before_conservative",
        "n_triplets_after_liberal",
        "n_triplets_after_conservative",
    ]:
        if column in summary.columns:
            summary[column] = summary[column].astype(int)
    summary["matched_triplets_per_side"] = summary[
        ["n_triplets_after_liberal", "n_triplets_after_conservative"]
    ].min(axis=1)
    summary["liberal_minus_conservative_before"] = (
        summary["n_triplets_before_liberal"] - summary["n_triplets_before_conservative"]
    )
    summary["matching_status"] = np.where(
        summary["matched_triplets_per_side"] > 0,
        "matched",
        "unmatched",
    )
    summary = summary.sort_values(["jel_policy_theme", "overall_difficulty"]).reset_index(drop=True)
    return long, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save(frame: pd.DataFrame, name: str, **kwargs) -> None:
    """Save CSV to DIFF_TABLES_DIR and figure to DIFF_FIGURES_DIR."""
    csv_path = DIFF_TABLES_DIR / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # save_frame derives figure path by swapping "tables" -> "figures" in the path.
    # We place our tables dir under .../difficulty_matched/tables so it will
    # look for .../difficulty_matched/figures automatically.
    DIFF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_frame(frame, csv_path, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task1 bias analysis with difficulty-matched triplets."
    )
    parser.add_argument(
        "--difficulty-path",
        default=str(DEFAULT_DIFFICULTY_PATH),
        help="Path to difficulty_scores_clean.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_output_dirs()
    DIFF_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    difficulty_lookup = _load_difficulty(args.difficulty_path)
    print(f"Loaded difficulty scores for {len(difficulty_lookup)} triplets.")

    frame = _prepare_frame(str(TASK1_ANALYSIS_JSONL))
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()

    # --- 1. Difficulty distribution (before matching) ---
    dist_before = _difficulty_distribution(frame, difficulty_lookup)
    _save(dist_before, "task1_difficulty_distribution_before_matching")

    # --- 2. Difficulty-matched analysis ---
    matched = difficulty_match_triplets(frame, difficulty_lookup, seed=args.seed)
    overall_n_lib_matched = matched[matched["ground_truth_side"] == "liberal"]["triplet_key"].nunique()
    overall_n_cons_matched = matched[matched["ground_truth_side"] == "conservative"]["triplet_key"].nunique()

    matched_theme = difficulty_match_by_theme(
        frame, difficulty_lookup, theme_col="jel_policy_theme", seed=args.seed
    )
    dist_after = _difficulty_distribution(matched_theme, difficulty_lookup)
    _save(dist_after, "task1_difficulty_distribution_after_matching")

    n_lib_matched = matched_theme[matched_theme["ground_truth_side"] == "liberal"]["triplet_key"].nunique()
    n_cons_matched = matched_theme[matched_theme["ground_truth_side"] == "conservative"]["triplet_key"].nunique()

    # Save the headline "difficulty matched" outputs using the stricter
    # theme-level matched sample so all aggregate views are based on the same rows.
    accuracy_matched_overall = accuracy_by(matched_theme, ["ground_truth_side"])
    accuracy_matched_by_model = accuracy_by(matched_theme, ["family", "model", "ground_truth_side"])
    bias_matched_by_model = bias_summary_by(matched_theme, ["family", "model"])

    _save(accuracy_matched_overall, "task1_difficulty_matched_accuracy_by_ground_truth_side")
    _save(accuracy_matched_by_model, "task1_difficulty_matched_accuracy_by_model_and_ground_truth_side")
    _save(bias_matched_by_model, "task1_difficulty_matched_bias_by_model")

    # --- 3. Per-theme difficulty-matched analysis ---

    if not matched_theme.empty:
        accuracy_theme_gt = accuracy_by(matched_theme, ["jel_policy_theme", "ground_truth_side"])
        accuracy_model_theme_gt = accuracy_by(
            matched_theme, ["model", "jel_policy_theme", "ground_truth_side"]
        )
        accuracy_family_theme_gt = accuracy_by(
            matched_theme, ["family", "jel_policy_theme", "ground_truth_side"]
        )
        bias_theme = bias_summary_by(matched_theme, ["jel_policy_theme"])
        bias_model_theme = bias_summary_by(matched_theme, ["model", "jel_policy_theme"])

        _save(accuracy_theme_gt, "task1_difficulty_matched_accuracy_by_theme_and_ground_truth_side")
        _save(accuracy_model_theme_gt, "task1_difficulty_matched_accuracy_by_model_theme_and_ground_truth_side")
        _save(accuracy_family_theme_gt, "task1_difficulty_matched_accuracy_by_family_theme_and_ground_truth_side")
        _save(bias_theme, "task1_difficulty_matched_bias_by_theme")
        _save(bias_model_theme, "task1_difficulty_matched_bias_by_model_and_theme")
        theme_matching_long, theme_matching_summary = _theme_matching_summary(
            frame,
            matched_theme,
            difficulty_lookup,
            theme_col="jel_policy_theme",
        )
        _save(theme_matching_long, "task1_difficulty_theme_matching_retention_by_theme_level_and_side")
        _save(theme_matching_summary, "task1_difficulty_theme_matching_summary")

    # --- 4. Unmatched (original) vs matched comparison ---
    accuracy_original = accuracy_by(labeled, ["family", "model", "ground_truth_side"])
    _save(accuracy_original, "task1_original_accuracy_by_model_and_ground_truth_side")

    # --- 5. Per-difficulty-level accuracy (not matched, diagnostic) ---
    labeled_with_diff = labeled.copy()
    labeled_with_diff["overall_difficulty"] = labeled_with_diff["triplet_key"].map(difficulty_lookup)
    labeled_with_diff = labeled_with_diff.dropna(subset=["overall_difficulty"])
    labeled_with_diff["overall_difficulty"] = labeled_with_diff["overall_difficulty"].astype(int)

    labeled_with_diff = _add_difficulty_label_column(labeled_with_diff)

    accuracy_by_difficulty = accuracy_by(labeled_with_diff, ["difficulty_level", "ground_truth_side"])
    accuracy_by_difficulty_model = accuracy_by(
        labeled_with_diff, ["model", "difficulty_level", "ground_truth_side"]
    )
    _save(accuracy_by_difficulty, "task1_accuracy_by_difficulty_and_ground_truth_side")
    _save(accuracy_by_difficulty_model, "task1_accuracy_by_model_difficulty_and_ground_truth_side")

    # --- Report ---
    report_path = DIFF_REPORTS_DIR / "task1_bias_difficulty_matched_report.md"
    with open(report_path, "w", encoding="utf-8") as h:
        h.write("# Task1 Difficulty-Matched Bias Analysis Report\n\n")

        h.write("## 1. Motivation\n\n")
        h.write("Liberal/conservative accuracy gaps may reflect inherent question difficulty\n")
        h.write("rather than model ideological priors. This analysis controls for difficulty\n")
        h.write("by matching liberal and conservative triplets on LLM-evaluated difficulty levels.\n\n")

        h.write("## 2. Methodology\n\n")
        h.write("- Each ideology-sensitive triplet receives an overall difficulty score (1-5)\n")
        h.write("  from GPT-5-mini based on domain knowledge, context dependence, ambiguity,\n")
        h.write("  causal complexity, and evidence sufficiency.\n")
        h.write("- For each difficulty level, we take min(n_liberal, n_conservative) triplets\n")
        h.write("  from each side (random sample, seed=42).\n")
        h.write("- This ensures identical difficulty distributions for both sides.\n")
        h.write("- Theme-level matching does the same within each JEL policy theme.\n\n")

        h.write("## 3. Data Summary\n\n")
        h.write(f"- Difficulty scores available: {len(difficulty_lookup)} triplets\n")
        h.write(f"- Labeled triplets in Task1: {labeled['triplet_key'].nunique()}\n")
        h.write(
            "- After difficulty matching without theme constraint "
            f"(diagnostic only): lib={overall_n_lib_matched}, cons={overall_n_cons_matched}\n"
        )
        if not matched_theme.empty:
            h.write(
                "- After theme-level matching "
                f"(used for the saved difficulty-matched tables/figures): "
                f"lib={n_lib_matched}, cons={n_cons_matched}\n"
            )
        h.write("\n")

        if not matched_theme.empty:
            h.write("## 4. How Theme-Level Matching Worked\n\n")
            h.write("Within each `jel_policy_theme`, triplets were first split by difficulty level.\n")
            h.write("At each theme-difficulty cell, the matched count per side is\n")
            h.write("`min(n_liberal_before, n_conservative_before)`.\n")
            h.write("Rows with zero on either side are dropped for that cell.\n\n")

            matched_theme_summary = theme_matching_summary[
                [
                    "jel_policy_theme",
                    "difficulty_level",
                    "n_triplets_before_liberal",
                    "n_triplets_before_conservative",
                    "matched_triplets_per_side",
                    "matching_status",
                ]
            ].copy()
            for _, row in matched_theme_summary.iterrows():
                h.write(
                    "- "
                    f"{row['jel_policy_theme']} / {row['difficulty_level']}: "
                    f"before lib={int(row['n_triplets_before_liberal'])}, "
                    f"cons={int(row['n_triplets_before_conservative'])}, "
                    f"matched per side={int(row['matched_triplets_per_side'])} "
                    f"({row['matching_status']})\n"
                )
            h.write("\n")

        h.write("## 5. Output Files\n\n")
        h.write("### Difficulty Distribution\n")
        h.write("- `task1_difficulty_distribution_before_matching.csv`\n")
        h.write("- `task1_difficulty_distribution_after_matching.csv`\n\n")

        h.write("### Theme-Level-Matched Overall Aggregates\n")
        h.write("- `task1_difficulty_matched_accuracy_by_ground_truth_side.csv`\n")
        h.write("- `task1_difficulty_matched_accuracy_by_model_and_ground_truth_side.csv`\n")
        h.write("- `task1_difficulty_matched_bias_by_model.csv`\n\n")

        h.write("### Theme-Level Difficulty-Matched\n")
        h.write("- `task1_difficulty_matched_accuracy_by_theme_and_ground_truth_side.csv`\n")
        h.write("- `task1_difficulty_matched_accuracy_by_model_theme_and_ground_truth_side.csv`\n")
        h.write("- `task1_difficulty_matched_accuracy_by_family_theme_and_ground_truth_side.csv`\n")
        h.write("- `task1_difficulty_matched_bias_by_theme.csv`\n")
        h.write("- `task1_difficulty_matched_bias_by_model_and_theme.csv`\n\n")
        h.write("- `task1_difficulty_theme_matching_retention_by_theme_level_and_side.csv`\n")
        h.write("- `task1_difficulty_theme_matching_summary.csv`\n\n")

        h.write("### Diagnostic (per difficulty level, unmatched)\n")
        h.write("- `task1_accuracy_by_difficulty_and_ground_truth_side.csv`\n")
        h.write("- `task1_accuracy_by_model_difficulty_and_ground_truth_side.csv`\n\n")

        h.write("### Comparison Baseline\n")
        h.write("- `task1_original_accuracy_by_model_and_ground_truth_side.csv`\n")

    print(
        json.dumps(
            {
                "difficulty_scores_loaded": len(difficulty_lookup),
                "labeled_triplets": int(labeled["triplet_key"].nunique()),
                "matched_liberal_triplets": n_lib_matched,
                "matched_conservative_triplets": n_cons_matched,
                "matched_theme_rows": len(matched_theme),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
