"""Task 3 analysis using false example ideology-side labels."""

from __future__ import annotations

import json

import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by, load_dataset, save_frame
from .paths import REPORTS_DIR, TABLES_DIR, TASK3_ANALYSIS_JSONL, ensure_output_dirs


VALID_SIDES = {"liberal", "conservative"}


def _prepare_frame() -> pd.DataFrame:
    frame = load_dataset(TASK3_ANALYSIS_JSONL)
    if frame.empty:
        raise RuntimeError(f"No Task3 analysis rows found in {TASK3_ANALYSIS_JSONL}")
    for column, default in (
        ("model", "unknown"),
        ("ground_truth_side", "unlabeled"),
        ("example_false_side_label", "unlabeled"),
    ):
        frame[column] = frame[column].fillna(default)
    frame["ideology_triplet_labeled"] = frame["ideology_triplet_labeled"].fillna(0).astype(int)
    frame["predicted_liberal"] = frame["predicted_liberal"].fillna(0).astype(int)
    frame["predicted_conservative"] = frame["predicted_conservative"].fillna(0).astype(int)
    frame["liberal_leaning_error"] = frame["liberal_leaning_error"].fillna(0).astype(int)
    frame["conservative_leaning_error"] = frame["conservative_leaning_error"].fillna(0).astype(int)
    frame["follows_displayed_example_sign"] = frame["follows_displayed_example_sign"].fillna(0).astype(int)
    frame["follows_original_example_sign"] = frame["follows_original_example_sign"].fillna(0).astype(int)
    return frame


def main() -> None:
    ensure_output_dirs()
    frame = _prepare_frame()
    focused = frame[
        frame["ground_truth_side"].isin(VALID_SIDES)
        & frame["example_false_side_label"].isin(VALID_SIDES)
    ].copy()

    accuracy_overall = accuracy_by(focused, ["example_false_side_label", "ground_truth_side"])
    accuracy_by_model = accuracy_by(focused, ["model", "example_false_side_label", "ground_truth_side"])
    bias_overall = bias_summary_by(focused, ["example_false_side_label"])
    bias_by_model = bias_summary_by(focused, ["model", "example_false_side_label"])
    follow_rates = (
        focused.groupby(["model", "example_false_side_label"], dropna=False, observed=False)
        .agg(
            n_predictions=("follows_displayed_example_sign", "size"),
            n_triplets=("triplet_key", "nunique"),
            follows_false_example_rate=("follows_displayed_example_sign", "mean"),
            recovers_original_rate=("follows_original_example_sign", "mean"),
        )
        .reset_index()
        .sort_values(["example_false_side_label", "model"])
    )

    save_frame(accuracy_overall, TABLES_DIR / "task3_accuracy_by_example_false_side_and_ground_truth_side.csv")
    save_frame(accuracy_by_model, TABLES_DIR / "task3_accuracy_by_model_example_false_side_and_ground_truth_side.csv")
    save_frame(bias_overall, TABLES_DIR / "task3_bias_by_example_false_side.csv")
    save_frame(bias_by_model, TABLES_DIR / "task3_bias_by_model_and_example_false_side.csv")
    save_frame(follow_rates, TABLES_DIR / "task3_false_example_follow_rates_by_model.csv")

    report_path = REPORTS_DIR / "task3_noise_robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Task3 Noise Robustness Report / Task3 노이즈 예시 강건성 보고서\n\n")
        handle.write("## 1. 분석 정의\n\n")
        handle.write("- Task3에서는 example sign이 거짓으로 뒤집힌 noisy evidence다.\n")
        handle.write("- `example_false_side_label=liberal`은 displayed noisy sign이 example triplet의 liberal preferred sign과 일치한다는 뜻이다.\n")
        handle.write("- `example_false_side_label=conservative`는 displayed noisy sign이 conservative preferred sign과 일치한다는 뜻이다.\n")
        handle.write("- target 문제는 `ground_truth_side`를 기준으로 liberal 문제 / conservative 문제로 나눈다.\n")
        handle.write("- bias score 정의는 Task1과 동일하다.\n\n")

        handle.write("## 2. 주요 산출물\n\n")
        handle.write("- `task3_accuracy_by_example_false_side_and_ground_truth_side.csv`\n")
        handle.write("- `task3_accuracy_by_model_example_false_side_and_ground_truth_side.csv`\n")
        handle.write("- `task3_bias_by_example_false_side.csv`\n")
        handle.write("- `task3_bias_by_model_and_example_false_side.csv`\n")
        handle.write("- `task3_false_example_follow_rates_by_model.csv`\n\n")

        handle.write("## 3. 데이터 범위\n\n")
        handle.write(f"- 전체 prediction rows: {len(frame)}\n")
        handle.write(f"- example false side와 target ground truth side가 모두 라벨링된 prediction rows: {len(focused)}\n")
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
