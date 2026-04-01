"""Qualitative reasoning-frame analysis over annotated model explanations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from .analysis_helpers import accuracy_by, load_dataset, save_frame
from .paths import (
    DEFAULT_REASONING_FRAMES_CANONICAL_JSONL,
    FIGURES_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    TASK1_ANALYSIS_JSONL,
    TASK2_ANALYSIS_JSONL,
    TASK3_ANALYSIS_JSONL,
    ensure_output_dirs,
)
from .viz import write_horizontal_bar_chart, write_markdown_table


def _task_frame(path: str, task_name: str) -> pd.DataFrame:
    frame = load_dataset(path)
    if frame.empty:
        return frame
    frame["reasoning_key"] = frame.apply(
        lambda row: "|".join([row["paper_task"], row["family"], row["model"], row["case_id"]]),
        axis=1,
    )
    frame["task_name"] = task_name
    return frame


def _suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze annotated reasoning frames.")
    parser.add_argument("--input", default=str(DEFAULT_REASONING_FRAMES_CANONICAL_JSONL))
    parser.add_argument("--tag", default=None, help="Optional output tag. If omitted, heuristic inputs auto-tag as 'heuristic'.")
    args = parser.parse_args()

    ensure_output_dirs()
    input_path = Path(args.input)
    frames = load_dataset(input_path)
    if frames.empty:
        raise RuntimeError(f"No reasoning-frame annotations found in {input_path}")

    tag = args.tag
    if tag is None:
        tag = "heuristic" if "heuristic" in input_path.stem else ""
    suffix = _suffix(tag)

    task_frame = pd.concat(
        [
            _task_frame(TASK1_ANALYSIS_JSONL, "task1"),
            _task_frame(TASK2_ANALYSIS_JSONL, "task2"),
            _task_frame(TASK3_ANALYSIS_JSONL, "task3"),
        ],
        ignore_index=True,
    )
    merged = frames.merge(
        task_frame[
            [
                "reasoning_key",
                "paper_task",
                "family",
                "model",
                "triplet_key",
                "correct",
                "directional_prediction",
            ]
        ],
        on=["reasoning_key", "paper_task", "family", "model", "triplet_key"],
        how="left",
    )
    merged["primary_frame"] = merged["primary_frame"].fillna("other")
    merged["annotation_method"] = merged["annotation_method"].fillna("unknown")

    primary_overall = (
        merged.groupby(["primary_frame"], dropna=False, observed=False)
        .agg(n_predictions=("primary_frame", "size"), n_triplets=("triplet_key", "nunique"))
        .reset_index()
        .sort_values("n_predictions", ascending=False)
    )
    primary_by_task = (
        merged.groupby(["paper_task", "primary_frame"], dropna=False, observed=False)
        .agg(n_predictions=("primary_frame", "size"), n_triplets=("triplet_key", "nunique"))
        .reset_index()
        .sort_values(["paper_task", "n_predictions"], ascending=[True, False])
    )
    primary_by_family = (
        merged.groupby(["family", "primary_frame"], dropna=False, observed=False)
        .agg(n_predictions=("primary_frame", "size"), n_triplets=("triplet_key", "nunique"))
        .reset_index()
        .sort_values(["family", "n_predictions"], ascending=[True, False])
    )
    annotation_method_summary = (
        merged.groupby(["annotation_method"], dropna=False, observed=False)
        .agg(n_predictions=("annotation_method", "size"), n_triplets=("triplet_key", "nunique"))
        .reset_index()
        .sort_values("n_predictions", ascending=False)
    )
    task1_frame = merged[merged["paper_task"] == "task1"].copy()
    task1_accuracy_by_frame = accuracy_by(task1_frame.dropna(subset=["correct"]), ["primary_frame"])

    cooccurrence_counter: Counter[tuple[str, str]] = Counter()
    for row in merged.to_dict(orient="records"):
        labels = [row.get("primary_frame")] + list(row.get("secondary_frames") or [])
        unique_labels = sorted({label for label in labels if label})
        for left, right in combinations(unique_labels, 2):
            cooccurrence_counter[(left, right)] += 1
    cooccurrence_rows = [
        {"frame_left": left, "frame_right": right, "n_predictions": count}
        for (left, right), count in cooccurrence_counter.most_common()
    ]
    cooccurrence = pd.DataFrame(cooccurrence_rows)

    save_frame(primary_overall, TABLES_DIR / f"reasoning_primary_frame_overall{suffix}.csv")
    save_frame(primary_by_task, TABLES_DIR / f"reasoning_primary_frame_by_task{suffix}.csv")
    save_frame(primary_by_family, TABLES_DIR / f"reasoning_primary_frame_by_family{suffix}.csv")
    save_frame(annotation_method_summary, TABLES_DIR / f"reasoning_annotation_method_summary{suffix}.csv")
    save_frame(task1_accuracy_by_frame, TABLES_DIR / f"reasoning_task1_accuracy_by_primary_frame{suffix}.csv")
    save_frame(cooccurrence, TABLES_DIR / f"reasoning_frame_cooccurrence{suffix}.csv")

    write_markdown_table(
        TABLES_DIR / f"reasoning_primary_frame_overall{suffix}.md",
        primary_overall.to_dict(orient="records"),
        ["primary_frame", "n_triplets", "n_predictions"],
        title="Primary Reasoning Frames Overall",
    )
    if not primary_overall.empty:
        write_horizontal_bar_chart(
            FIGURES_DIR / f"reasoning_primary_frame_overall{suffix}.png",
            "Primary Reasoning Frames Overall",
            primary_overall["primary_frame"].astype(str).tolist(),
            (primary_overall["n_predictions"] / primary_overall["n_predictions"].sum()).astype(float).tolist(),
            counts=primary_overall["n_triplets"].astype(int).tolist(),
            color="#7c3aed",
        )

    report_path = REPORTS_DIR / f"reasoning_frames_report{suffix}.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Reasoning Frames Report / 추론 프레임 분석 보고서\n\n")
        handle.write("## 1. 이 분석이 하는 일\n\n")
        handle.write("- 목적: 모델이 sign을 예측할 때 어떤 경제적 reasoning frame을 사용하는지 정리한다.\n")
        handle.write("- `primary_frame`: reasoning에서 가장 중심이 되는 프레임 1개\n")
        handle.write("- `secondary_frames`: 같이 보조적으로 등장하는 프레임 최대 3개\n\n")

        handle.write("## 2. 프레임 사전\n\n")
        handle.write("- efficiency, incentives, market_distortion, productivity, redistribution, insurance, externalities, fiscal_burden, state_capacity, other\n\n")

        handle.write("## 3. 프레임은 어떻게 붙였는가\n\n")
        methods = sorted(set(merged["annotation_method"].astype(str)))
        handle.write(f"- 사용된 annotation method: {methods}\n")
        if "llm" in methods:
            handle.write("- LLM 방식: `extended/ideology_bias/reasoning_frames_prompt.py`의 프롬프트로 reasoning 텍스트를 읽고 controlled vocabulary 안에서 `primary_frame` 1개와 `secondary_frames` 최대 3개를 고르게 했다.\n")
            handle.write("- LLM prompt의 핵심 규칙: reasoning만 보고, correctness는 보지 말고, vocabulary 밖의 label은 쓰지 말라는 것이다.\n")
        if "heuristic" in methods:
            handle.write("- heuristic 방식: `extended/ideology_bias/reasoning_frames_heuristic.py`에서 frame별 키워드 사전을 두고, reasoning 텍스트 안의 단어/구문 일치 수를 세어 가장 높은 frame을 `primary_frame`으로 붙인다.\n")
            handle.write("- heuristic 결과에는 `matched_keywords`와 `frame_scores`가 함께 저장되므로, 왜 특정 frame이 붙었는지 역추적할 수 있다.\n")
        handle.write("\n")

        handle.write("## 4. 데이터 범위\n\n")
        handle.write(f"- 입력 파일: {input_path}\n")
        handle.write(f"- output tag: {tag or 'default'}\n")
        handle.write(f"- annotated reasoning row 수: {len(merged)}\n")
        handle.write(f"- 고유 triplet 수: {merged['triplet_key'].nunique()}\n")
        handle.write(f"- 고유 primary frame 수: {merged['primary_frame'].nunique()}\n")
        if not primary_overall.empty:
            handle.write(
                f"- 가장 많이 나온 primary frame: {primary_overall.iloc[0]['primary_frame']} "
                f"(triplets={int(primary_overall.iloc[0]['n_triplets'])}, predictions={int(primary_overall.iloc[0]['n_predictions'])})\n"
            )
        handle.write("\n")

        handle.write("## 5. 파일을 어떻게 읽으면 되는가\n\n")
        handle.write(f"- `reasoning_primary_frame_overall{suffix}.csv`: 전체 reasoning에서 어떤 frame이 가장 많이 나오는지 본다.\n")
        handle.write(f"- `reasoning_primary_frame_by_family{suffix}.csv`: model family별 주력 reasoning frame을 비교한다.\n")
        handle.write(f"- `reasoning_task1_accuracy_by_primary_frame{suffix}.csv`: Task1에서 어떤 frame을 주로 쓸 때 accuracy가 높거나 낮은지 본다.\n")
        handle.write(f"- `reasoning_frame_cooccurrence{suffix}.csv`: 두 frame이 함께 등장하는 조합을 본다.\n\n")

        handle.write("## 6. 해석 팁\n\n")
        handle.write("- primary frame 빈도는 모델이 어떤 경제 직관을 자주 호출하는지 보여준다.\n")
        handle.write("- co-occurrence는 예를 들어 `incentives`와 `efficiency`가 같이 많이 붙는지 같은 패턴을 보여준다.\n")
        handle.write("- Task1 accuracy by frame은 특정 frame이 편향이라기보다, 어떤 frame을 사용할 때 모델이 더 안정적/불안정한지를 보는 보조 지표다.\n")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "tag": tag or "default",
                "annotated_reasonings": len(merged),
                "n_primary_frames": int(merged["primary_frame"].nunique()),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
