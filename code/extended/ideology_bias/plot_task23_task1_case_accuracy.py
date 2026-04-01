"""Build Task1-vs-Task2/3 case-type accuracy comparison figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .viz import _append_alpha, _family_color_map, write_horizontal_bar_chart
from .utils import family_sort_key, model_sort_key

ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
TASK1_TABLE = (
    ARTIFACT_ROOT / "outputs" / "ideology_bias" / "tables" / "task1_accuracy_by_model_and_ground_truth_side.csv"
)
ANALYSIS_ROOT = (
    ARTIFACT_ROOT
    / "outputs"
    / "ideology_bias"
    / "task23_jel_similarity_side_capped_jaccard05_shared2_analysis"
)
TABLES_DIR = ANALYSIS_ROOT / "tables"
FIGURES_DIR = ANALYSIS_ROOT / "figures"

TASK_SPECS = {
    "task2": {
        "input_table": TABLES_DIR / "task2_jel_similarity_side_capped_jaccard05_shared2_accuracy_by_model_and_case_type.csv",
        "output_stem": "task2_jel_similarity_side_capped_jaccard05_shared2_accuracy_by_model_and_case_type_with_task1",
        "title": "Task2 Accuracy By Model And Case Type With Task1",
    },
    "task3": {
        "input_table": TABLES_DIR / "task3_jel_similarity_side_capped_jaccard05_shared2_accuracy_by_model_and_case_type.csv",
        "output_stem": "task3_jel_similarity_side_capped_jaccard05_shared2_accuracy_by_model_and_case_type_with_task1",
        "title": "Task3 Accuracy By Model And Case Type With Task1",
    },
}

CASE_ORDER = ["liberal-only", "conservative-only", "lib-lib", "lib-cons", "cons-lib", "cons-cons"]
def _short_model_label(model: str) -> str:
    text = str(model)
    if "/" in text:
        return text.split("/")[-1]
    return text


def _family_sort_key(family: str) -> tuple[int, str]:
    return family_sort_key(family)


def _case_sort_key(case_type: str) -> int:
    try:
        return CASE_ORDER.index(str(case_type))
    except ValueError:
        return len(CASE_ORDER)


def _load_task1_summary() -> pd.DataFrame:
    frame = pd.read_csv(TASK1_TABLE)
    frame = frame[frame["ground_truth_side"].isin(["liberal", "conservative"])].copy()
    frame["pair_case_type"] = frame["ground_truth_side"].map(
        {
            "liberal": "liberal-only",
            "conservative": "conservative-only",
        }
    )
    return frame[["family", "model", "pair_case_type", "n_predictions", "correct_predictions", "n_triplets", "accuracy"]]


def _load_task23_summary(path: Path, task1_family_map: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["family"] = frame["model"].map(task1_family_map).fillna("unknown")
    return frame[["family", "model", "pair_case_type", "n_predictions", "correct_predictions", "n_triplets", "accuracy"]]


def _build_combined_frame(task_name: str) -> pd.DataFrame:
    task1 = _load_task1_summary()
    task1_family_map = (
        task1[["model", "family"]]
        .drop_duplicates()
        .set_index("model")["family"]
        .to_dict()
    )
    task23 = _load_task23_summary(TASK_SPECS[task_name]["input_table"], task1_family_map)
    combined = pd.concat([task1, task23], ignore_index=True)
    combined["family"] = combined["family"].fillna("unknown")
    combined["_family_key"] = combined["family"].map(lambda value: _family_sort_key(str(value)))
    combined["_model_label"] = combined["model"].map(_short_model_label)
    combined["_model_key"] = combined.apply(lambda row: model_sort_key(row["model"], row.get("family")), axis=1)
    combined["_case_key"] = combined["pair_case_type"].map(_case_sort_key)
    combined = combined.sort_values(["_family_key", "_model_key", "_case_key"]).reset_index(drop=True)
    return combined.drop(columns=["_family_key"])


def _write_chart(frame: pd.DataFrame, title: str, output_path: Path, include_family_in_label: bool) -> None:
    if frame.empty:
        return

    family_color_map = _family_color_map(frame["family"].tolist())
    labels = []
    for _, row in frame.iterrows():
        model_label = row["_model_label"]
        if include_family_in_label:
            labels.append(f"{row['family']} / {model_label} - {row['pair_case_type']}")
        else:
            labels.append(f"{model_label} - {row['pair_case_type']}")
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
        metric_label="Accuracy (proportion of correct predictions)",
    )


def build_outputs(task_name: str) -> None:
    spec = TASK_SPECS[task_name]
    combined = _build_combined_frame(task_name)

    output_csv = TABLES_DIR / f"{spec['output_stem']}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export = combined.drop(columns=["_model_label", "_case_key"])
    if "_model_key" in export.columns:
        export = export.drop(columns=["_model_key"])
    export.to_csv(output_csv, index=False)

    overall_path = FIGURES_DIR / f"{spec['output_stem']}.png"
    _write_chart(combined, spec["title"], overall_path, include_family_in_label=False)

    family_dir = FIGURES_DIR / spec["output_stem"]
    family_dir.mkdir(parents=True, exist_ok=True)
    for family, family_frame in combined.groupby("family", sort=False):
        family_frame = family_frame.sort_values(["_model_key", "_case_key"]).reset_index(drop=True)
        family_path = family_dir / f"{family}_accuracy.png"
        _write_chart(
            family_frame,
            f"{spec['title']} - {family}",
            family_path,
            include_family_in_label=False,
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for task_name in TASK_SPECS:
        build_outputs(task_name)


if __name__ == "__main__":
    main()
