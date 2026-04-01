"""Build target-side-fixed Task1/2/3 comparison figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .viz import _append_alpha, _family_color_map, write_horizontal_bar_chart
from .utils import family_sort_key, model_sort_key

BASE_OUTPUT = Path(__file__).resolve().parents[3] / "outputs" / "ideology_bias"
TASK1_ROWS = BASE_OUTPUT / "analysis_datasets" / "task1_analysis_rows.csv"
TASK23_ANALYSIS_ROOT = BASE_OUTPUT / "task23_jel_similarity_side_capped_jaccard05_shared2_analysis"
COMPARISON_ROOT = TASK23_ANALYSIS_ROOT / "target_side_fixed_comparison"
TABLES_DIR = COMPARISON_ROOT / "tables"
FIGURES_DIR = COMPARISON_ROOT / "figures"

TASK23_ROWS = {
    "task2": TASK23_ANALYSIS_ROOT / "analysis_datasets" / "task2_analysis_rows_jel_similarity_side_capped_jaccard05_shared2.csv",
    "task3": TASK23_ANALYSIS_ROOT / "analysis_datasets" / "task3_analysis_rows_jel_similarity_side_capped_jaccard05_shared2.csv",
}

SIDE_SPECS = {
    "conservative": {
        "task1_side": "conservative",
        "task1_label": "conservative-only",
        "task23_case_types": ["lib-cons", "cons-cons"],
    },
    "liberal": {
        "task1_side": "liberal",
        "task1_label": "liberal-only",
        "task23_case_types": ["lib-lib", "cons-lib"],
    },
}

COMBINED_CASE_ORDER = [
    "liberal-only",
    "lib-lib",
    "cons-lib",
    "conservative-only",
    "lib-cons",
    "cons-cons",
]
CASE_COLOR_MAP = {
    "liberal-only": "#2563eb",
    "lib-lib": "#0f766e",
    "cons-lib": "#65a30d",
    "conservative-only": "#dc2626",
    "lib-cons": "#ea580c",
    "cons-cons": "#7c3aed",
}
MODEL_BASE_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
    "#db2777",
]
CASE_SHADE_SUFFIX = {
    "liberal-only": "FF",
    "lib-lib": "CC",
    "cons-lib": "99",
    "conservative-only": "FF",
    "lib-cons": "CC",
    "cons-cons": "99",
}


def _short_model_label(model: str) -> str:
    text = str(model)
    return text.split("/")[-1] if "/" in text else text


def _family_sort_key(family: str) -> tuple[int, str]:
    return family_sort_key(family)


def _case_sort_key(case_type: str, ordered_case_types: list[str]) -> int:
    try:
        return ordered_case_types.index(str(case_type))
    except ValueError:
        return len(ordered_case_types)


def _aggregate_accuracy(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False, observed=False)
    summary = grouped.agg(
        n_predictions=("correct", "size"),
        correct_predictions=("correct", "sum"),
        n_triplets=("triplet_key", "nunique"),
    ).reset_index()
    summary["accuracy"] = summary["correct_predictions"] / summary["n_predictions"]
    return summary


def _load_task1_rows() -> pd.DataFrame:
    return pd.read_csv(
        TASK1_ROWS,
        usecols=["triplet_key", "ground_truth_side", "family", "model", "correct"],
    )


def _load_task23_rows(task_name: str, family_map: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(
        TASK23_ROWS[task_name],
        usecols=["triplet_key", "pair_case_type", "model", "correct"],
    )
    frame["family"] = frame["model"].map(family_map).fillna("unknown")
    return frame


def _shared_triplets(task1: pd.DataFrame, task23: pd.DataFrame, target_side: str) -> set[str]:
    spec = SIDE_SPECS[target_side]
    task1_triplets = set(task1.loc[task1["ground_truth_side"] == spec["task1_side"], "triplet_key"])
    shared = task1_triplets.copy()
    for case_type in spec["task23_case_types"]:
        shared &= set(task23.loc[task23["pair_case_type"] == case_type, "triplet_key"])
    return shared


def _build_plot_frame(task_name: str, target_side: str) -> pd.DataFrame:
    task1 = _load_task1_rows()
    family_map = (
        task1[["model", "family"]]
        .drop_duplicates()
        .set_index("model")["family"]
        .to_dict()
    )
    task23 = _load_task23_rows(task_name, family_map)
    shared_triplets = _shared_triplets(task1, task23, target_side)
    spec = SIDE_SPECS[target_side]

    task1_filtered = task1[
        (task1["ground_truth_side"] == spec["task1_side"]) & task1["triplet_key"].isin(shared_triplets)
    ].copy()
    task1_filtered["case_type"] = spec["task1_label"]
    task1_summary = _aggregate_accuracy(task1_filtered, ["family", "model", "case_type"])

    task23_filtered = task23[
        task23["pair_case_type"].isin(spec["task23_case_types"]) & task23["triplet_key"].isin(shared_triplets)
    ].copy()
    task23_filtered = task23_filtered.rename(columns={"pair_case_type": "case_type"})
    task23_summary = _aggregate_accuracy(task23_filtered, ["family", "model", "case_type"])

    combined = pd.concat([task1_summary, task23_summary], ignore_index=True)
    ordered_case_types = [spec["task1_label"], *spec["task23_case_types"]]
    combined["_model_label"] = combined["model"].map(_short_model_label)
    combined["_model_key"] = combined.apply(lambda row: model_sort_key(row["model"], row.get("family")), axis=1)
    combined["_family_key"] = combined["family"].map(_family_sort_key)
    combined["_case_key"] = combined["case_type"].map(lambda value: _case_sort_key(value, ordered_case_types))
    combined["_shared_triplets"] = len(shared_triplets)
    combined = combined.sort_values(["_family_key", "_model_key", "_case_key"]).reset_index(drop=True)
    return combined


def _write_chart(frame: pd.DataFrame, title: str, output_path: Path) -> None:
    family_color_map = _family_color_map(frame["family"].tolist())
    labels = [f"{row['_model_label']} - {row['case_type']}" for _, row in frame.iterrows()]
    colors = [
        _append_alpha(family_color_map.get(str(row["family"]), "#2563eb"), "E6")
        for _, row in frame.iterrows()
    ]
    write_horizontal_bar_chart(
        output_path,
        title,
        labels=labels,
        values=frame["accuracy"].astype(float).tolist(),
        counts=frame["n_triplets"].fillna(0).astype(int).tolist(),
        colors=colors,
        metric_label="Accuracy on shared target triplets",
    )


def _write_case_colored_chart(frame: pd.DataFrame, title: str, output_path: Path) -> None:
    labels = [f"{row['_model_label']} - {row['case_type']}" for _, row in frame.iterrows()]
    model_order = []
    for model in frame.sort_values("_model_key")["model"].tolist():
        if model not in model_order:
            model_order.append(model)
    model_color_map = {
        model: MODEL_BASE_PALETTE[idx % len(MODEL_BASE_PALETTE)]
        for idx, model in enumerate(model_order)
    }
    colors = [
        _append_alpha(
            model_color_map.get(str(row["model"]), "#2563eb"),
            CASE_SHADE_SUFFIX.get(str(row["case_type"]), "FF"),
        )
        for _, row in frame.iterrows()
    ]
    write_horizontal_bar_chart(
        output_path,
        title,
        labels=labels,
        values=frame["accuracy"].astype(float).tolist(),
        counts=frame["n_triplets"].fillna(0).astype(int).tolist(),
        colors=colors,
        metric_label="Accuracy on shared target triplets",
    )


def build_outputs(task_name: str, target_side: str) -> None:
    frame = _build_plot_frame(task_name, target_side)
    shared_n = int(frame["_shared_triplets"].iloc[0]) if not frame.empty else 0
    stem = f"{task_name}_{target_side}_target_fixed_accuracy"
    title = f"{task_name.title()} {target_side.title()} Target Fixed Accuracy (N={shared_n})"

    export_frame = frame.drop(columns=["_model_label", "_family_key", "_case_key", "_shared_triplets"])
    if "_model_key" in export_frame.columns:
        export_frame = export_frame.drop(columns=["_model_key"])
    export_frame.to_csv(TABLES_DIR / f"{stem}.csv", index=False)

    overall_path = FIGURES_DIR / f"{stem}.png"
    _write_chart(frame, title, overall_path)

    family_dir = FIGURES_DIR / stem
    family_dir.mkdir(parents=True, exist_ok=True)
    for family, family_frame in frame.groupby("family", sort=False):
        family_frame = family_frame.sort_values(["model", "_case_key"]).reset_index(drop=True)
        family_path = family_dir / f"{family}_accuracy.png"
        _write_chart(family_frame, f"{title} - {family}", family_path)


def build_family_combined_outputs(task_name: str) -> None:
    liberal_frame = _build_plot_frame(task_name, "liberal")
    conservative_frame = _build_plot_frame(task_name, "conservative")
    combined = pd.concat([liberal_frame, conservative_frame], ignore_index=True)
    if combined.empty:
        return

    combined["_combined_case_key"] = combined["case_type"].map(
        lambda value: _case_sort_key(value, COMBINED_CASE_ORDER)
    )
    combined = combined.sort_values(["_family_key", "model", "_combined_case_key"]).reset_index(drop=True)

    export_frame = combined.drop(
        columns=["_model_label", "_family_key", "_case_key", "_shared_triplets", "_combined_case_key"]
    )
    export_frame.to_csv(TABLES_DIR / f"{task_name}_family_combined_target_fixed_accuracy.csv", index=False)

    family_dir = FIGURES_DIR / f"{task_name}_family_combined_target_fixed_accuracy"
    family_dir.mkdir(parents=True, exist_ok=True)
    for family, family_frame in combined.groupby("family", sort=False):
        family_frame = family_frame.sort_values(["model", "_combined_case_key"]).reset_index(drop=True)
        shared_lib = int(family_frame.loc[family_frame["case_type"] == "liberal-only", "_shared_triplets"].iloc[0])
        shared_con = int(
            family_frame.loc[family_frame["case_type"] == "conservative-only", "_shared_triplets"].iloc[0]
        )
        title = (
            f"{task_name.title()} Target-Fixed Accuracy - {family} "
            f"(lib N={shared_lib}, con N={shared_con})"
        )
        _write_case_colored_chart(family_frame, title, family_dir / f"{family}_accuracy.png")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for task_name in TASK23_ROWS:
        for target_side in SIDE_SPECS:
            build_outputs(task_name, target_side)
        build_family_combined_outputs(task_name)


if __name__ == "__main__":
    main()
