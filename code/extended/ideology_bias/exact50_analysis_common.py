"""Shared helpers for exact50 task2/task3 analyses."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by, load_dataset, save_frame
from .paths import REPORTS_DIR, TABLES_DIR, ensure_output_dirs


VALID_CASE_TYPES = {"lib-lib", "lib-cons", "cons-lib", "cons-cons"}


def _prepare_common(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("No analysis rows found.")
    defaults = {
        "model": "unknown",
        "pair_case_type": "unlabeled",
        "variant": "unknown",
        "ground_truth_side": "unlabeled",
    }
    for column, default in defaults.items():
        if column in frame.columns:
            frame[column] = frame[column].fillna(default)
    for column in (
        "ideology_triplet_labeled",
        "predicted_liberal",
        "predicted_conservative",
        "liberal_leaning_error",
        "conservative_leaning_error",
        "different_paper",
    ):
        if column in frame.columns:
            frame[column] = frame[column].fillna(0).astype(int)
    return frame


def _inventory(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(group_cols, dropna=False, observed=False)
        .agg(
            n_predictions=("case_id", "size"),
            n_unique_targets=("triplet_key", "nunique"),
            n_unique_pairs=("case_id", "nunique"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    return summary


def run_task2_analysis(dataset_path: str | Path, stem: str) -> dict[str, object]:
    ensure_output_dirs()
    frame = _prepare_common(load_dataset(dataset_path))
    focused = frame[frame["pair_case_type"].isin(VALID_CASE_TYPES)].copy()

    accuracy_overall = accuracy_by(focused, ["pair_case_type"])
    accuracy_by_model = accuracy_by(focused, ["model", "pair_case_type"])
    bias_overall = bias_summary_by(focused, ["pair_case_type"])
    bias_by_model = bias_summary_by(focused, ["model", "pair_case_type"])
    inventory = _inventory(focused, ["pair_case_type"])

    save_frame(accuracy_overall, TABLES_DIR / f"{stem}_accuracy_by_case_type.csv")
    save_frame(accuracy_by_model, TABLES_DIR / f"{stem}_accuracy_by_model_and_case_type.csv")
    save_frame(bias_overall, TABLES_DIR / f"{stem}_bias_by_case_type.csv")
    save_frame(bias_by_model, TABLES_DIR / f"{stem}_bias_by_model_and_case_type.csv")
    save_frame(inventory, TABLES_DIR / f"{stem}_case_inventory.csv")

    report_path = REPORTS_DIR / f"{stem}_report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {stem} report\n\n")
        handle.write("- Grouping key: `pair_case_type`\n")
        handle.write(f"- Prediction rows: {len(frame)}\n")
        handle.write(f"- Focused rows: {len(focused)}\n")
        handle.write(f"- Unique targets: {focused['triplet_key'].nunique()}\n")
        handle.write(f"- Unique example-target pairs: {focused['case_id'].nunique()}\n")

    return {
        "dataset_path": str(dataset_path),
        "prediction_rows": int(len(frame)),
        "focused_rows": int(len(focused)),
        "unique_targets": int(focused["triplet_key"].nunique()),
        "unique_pairs": int(focused["case_id"].nunique()),
        "report": str(report_path),
    }


def run_task3_analysis(dataset_path: str | Path, stem: str) -> dict[str, object]:
    ensure_output_dirs()
    frame = _prepare_common(load_dataset(dataset_path))
    for column in ("follows_displayed_example_sign", "follows_original_example_sign"):
        if column in frame.columns:
            frame[column] = frame[column].fillna(0).astype(int)
    focused = frame[frame["pair_case_type"].isin(VALID_CASE_TYPES)].copy()

    accuracy_overall = accuracy_by(focused, ["pair_case_type"])
    accuracy_by_model = accuracy_by(focused, ["model", "pair_case_type"])
    bias_overall = bias_summary_by(focused, ["pair_case_type"])
    bias_by_model = bias_summary_by(focused, ["model", "pair_case_type"])
    follow_rates = (
        focused.groupby(["model", "pair_case_type"], dropna=False, observed=False)
        .agg(
            n_predictions=("case_id", "size"),
            n_unique_targets=("triplet_key", "nunique"),
            n_unique_pairs=("case_id", "nunique"),
            follows_displayed_example_rate=("follows_displayed_example_sign", "mean"),
            recovers_original_rate=("follows_original_example_sign", "mean"),
        )
        .reset_index()
        .sort_values(["pair_case_type", "model"])
    )
    inventory = _inventory(focused, ["pair_case_type"])

    save_frame(accuracy_overall, TABLES_DIR / f"{stem}_accuracy_by_case_type.csv")
    save_frame(accuracy_by_model, TABLES_DIR / f"{stem}_accuracy_by_model_and_case_type.csv")
    save_frame(bias_overall, TABLES_DIR / f"{stem}_bias_by_case_type.csv")
    save_frame(bias_by_model, TABLES_DIR / f"{stem}_bias_by_model_and_case_type.csv")
    save_frame(follow_rates, TABLES_DIR / f"{stem}_follow_rates_by_case_type_and_model.csv")
    save_frame(inventory, TABLES_DIR / f"{stem}_case_inventory.csv")

    report_path = REPORTS_DIR / f"{stem}_report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {stem} report\n\n")
        handle.write("- Grouping key: `pair_case_type`\n")
        handle.write(f"- Prediction rows: {len(frame)}\n")
        handle.write(f"- Focused rows: {len(focused)}\n")
        handle.write(f"- Unique targets: {focused['triplet_key'].nunique()}\n")
        handle.write(f"- Unique example-target pairs: {focused['case_id'].nunique()}\n")

    return {
        "dataset_path": str(dataset_path),
        "prediction_rows": int(len(frame)),
        "focused_rows": int(len(focused)),
        "unique_targets": int(focused["triplet_key"].nunique()),
        "unique_pairs": int(focused["case_id"].nunique()),
        "report": str(report_path),
    }


def print_summary(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
