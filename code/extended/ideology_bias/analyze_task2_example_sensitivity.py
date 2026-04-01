"""Task 2 analysis using true example ideology-side labels."""

from __future__ import annotations

import json

import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by, load_dataset, save_frame
from .paths import REPORTS_DIR, TABLES_DIR, TASK2_ANALYSIS_JSONL, ensure_output_dirs


VALID_SIDES = {"liberal", "conservative"}


def _prepare_frame() -> pd.DataFrame:
    frame = load_dataset(TASK2_ANALYSIS_JSONL)
    if frame.empty:
        raise RuntimeError(f"No Task2 analysis rows found in {TASK2_ANALYSIS_JSONL}")
    for column, default in (
        ("model", "unknown"),
        ("ground_truth_side", "unlabeled"),
        ("example_true_side_label", "unlabeled"),
    ):
        frame[column] = frame[column].fillna(default)
    frame["ideology_triplet_labeled"] = frame["ideology_triplet_labeled"].fillna(0).astype(int)
    frame["predicted_liberal"] = frame["predicted_liberal"].fillna(0).astype(int)
    frame["predicted_conservative"] = frame["predicted_conservative"].fillna(0).astype(int)
    frame["liberal_leaning_error"] = frame["liberal_leaning_error"].fillna(0).astype(int)
    frame["conservative_leaning_error"] = frame["conservative_leaning_error"].fillna(0).astype(int)
    return frame


def main() -> None:
    ensure_output_dirs()
    frame = _prepare_frame()
    focused = frame[
        frame["ground_truth_side"].isin(VALID_SIDES)
        & frame["example_true_side_label"].isin(VALID_SIDES)
    ].copy()

    accuracy_overall = accuracy_by(focused, ["example_true_side_label", "ground_truth_side"])
    accuracy_by_model = accuracy_by(focused, ["model", "example_true_side_label", "ground_truth_side"])
    bias_overall = bias_summary_by(focused, ["example_true_side_label"])
    bias_by_model = bias_summary_by(focused, ["model", "example_true_side_label"])

    save_frame(accuracy_overall, TABLES_DIR / "task2_accuracy_by_example_true_side_and_ground_truth_side.csv")
    save_frame(accuracy_by_model, TABLES_DIR / "task2_accuracy_by_model_example_true_side_and_ground_truth_side.csv")
    save_frame(bias_overall, TABLES_DIR / "task2_bias_by_example_true_side.csv")
    save_frame(bias_by_model, TABLES_DIR / "task2_bias_by_model_and_example_true_side.csv")

    report_path = REPORTS_DIR / "task2_example_sensitivity_report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Task2 Example Sensitivity Report / Task2 예시 민감도 보고서\n\n")
        handle.write("## 1. 분석 정의\n\n")
        handle.write("- Task2에서는 example sign이 원래 참값이다.\n")
        handle.write("- `example_true_side_label=liberal`은 example의 original sign이 그 example triplet의 liberal preferred sign과 일치한다는 뜻이다.\n")
        handle.write("- `example_true_side_label=conservative`는 example의 original sign이 conservative preferred sign과 일치한다는 뜻이다.\n")
        handle.write("- target 문제는 `ground_truth_side`를 기준으로 liberal 문제 / conservative 문제로 나눈다.\n")
        handle.write("- bias score 정의는 Task1과 동일하다.\n\n")

        handle.write("## 2. 주요 산출물\n\n")
        handle.write("- `task2_accuracy_by_example_true_side_and_ground_truth_side.csv`\n")
        handle.write("- `task2_accuracy_by_model_example_true_side_and_ground_truth_side.csv`\n")
        handle.write("- `task2_bias_by_example_true_side.csv`\n")
        handle.write("- `task2_bias_by_model_and_example_true_side.csv`\n\n")

        handle.write("## 3. 데이터 범위\n\n")
        handle.write(f"- 전체 prediction rows: {len(frame)}\n")
        handle.write(f"- example truth side와 target ground truth side가 모두 라벨링된 prediction rows: {len(focused)}\n")
        handle.write(f"- 해당 고유 target triplet 수: {focused['triplet_key'].nunique()}\n")

    print(
        json.dumps(
            {
                "prediction_rows": int(len(frame)),
                "focused_rows": int(len(focused)),
                "focused_triplets": int(focused["triplet_key"].nunique()),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
