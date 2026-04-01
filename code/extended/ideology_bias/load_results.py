"""Load existing evaluation results into canonical row dictionaries."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import TASK1_RESULT_DIRS, TASK23_RESULT_DIR
from .schemas import VALID_SIGNS
from .utils import make_triplet_key, normalize_sign, parse_model_meta

EXCLUDED_MODELS = {"gemini-2.5-pro"}


def _result_file_iter(results_dir: Path, prefix: str):
    for path in sorted(results_dir.glob(f"{prefix}*_results.json")):
        yield path


def _task_spec(
    task_name: str,
    results_dir: str | Path | None = None,
    task1_results_dir: str | Path | None = None,
) -> tuple[str, list[tuple[str, Path]]]:
    if task_name == "task1":
        if task1_results_dir is not None:
            return "task1", [("mixed", Path(task1_results_dir))]
        return "task1", list(TASK1_RESULT_DIRS.items())
    if task_name == "task2":
        return "task2", [("mixed", Path(results_dir) if results_dir is not None else TASK23_RESULT_DIR)]
    if task_name == "task3":
        return "task3", [("mixed", Path(results_dir) if results_dir is not None else TASK23_RESULT_DIR)]
    raise ValueError(f"Unknown task_name: {task_name}")


def _example_triplets_from_input(input_data: dict) -> list[dict]:
    examples = input_data.get("examples") or []
    paper_ids = input_data.get("paper_ids") or []
    example_paper_ids = []
    if len(paper_ids) >= len(examples) + 1:
        example_paper_ids = paper_ids[:-1]
    else:
        example_paper_ids = [None] * len(examples)
    rows: list[dict] = []
    for idx, example in enumerate(examples):
        paper_id = example_paper_ids[idx] if idx < len(example_paper_ids) else None
        rows.append(
            {
                "paper_id": paper_id,
                "treatment": example.get("treatment"),
                "outcome": example.get("outcome"),
                "context": example.get("context"),
                "displayed_sign": normalize_sign(example.get("sign")),
                "original_sign": normalize_sign(example.get("original_sign") or example.get("sign")),
                "triplet_key": make_triplet_key(paper_id, example.get("treatment"), example.get("outcome")),
                "example_true_side": example.get("example_true_side"),
                "example_false_side": example.get("example_false_side"),
                "example_exact_jel_codes": example.get("example_exact_jel_codes") or [],
                "shared_exact_jel_codes": example.get("shared_exact_jel_codes") or [],
                "shared_exact_jel_count": example.get("shared_exact_jel_count"),
                "jel_overlap_ratio": example.get("jel_overlap_ratio"),
            }
        )
    return rows


def load_result_rows(
    task_name: str,
    results_dir: str | Path | None = None,
    task1_results_dir: str | Path | None = None,
) -> list[dict]:
    """Load canonical result rows for a paper task."""
    eval_task_prefix, dirs = _task_spec(task_name, results_dir=results_dir, task1_results_dir=task1_results_dir)
    rows: list[dict] = []
    for domain_hint, results_dir in dirs:
        if not results_dir.exists():
            continue
        for path in _result_file_iter(results_dir, eval_task_prefix):
            with open(path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            family = document.get("family", "unknown")
            model = document.get("model", "unknown")
            if model in EXCLUDED_MODELS:
                continue
            model_meta = parse_model_meta(family, model)
            for result in document.get("results", []):
                input_data = result.get("input_data", {})
                output_data = result.get("output_data", {}) or {}
                if task_name == "task1":
                    paper_id = input_data.get("paper_id")
                else:
                    paper_ids = input_data.get("paper_ids") or []
                    paper_id = paper_ids[-1] if paper_ids else input_data.get("paper_id")
                row = {
                    "paper_task": task_name,
                    "eval_task": eval_task_prefix,
                    "domain_hint": domain_hint,
                    "case_id": result.get("case_id"),
                    "paper_id": str(paper_id).strip() if paper_id is not None else "",
                    "treatment": input_data.get("treatment"),
                    "outcome": input_data.get("outcome"),
                    "triplet_key": make_triplet_key(paper_id, input_data.get("treatment"), input_data.get("outcome")),
                    "context": input_data.get("context") or input_data.get("test_context"),
                    "expected_sign": normalize_sign(result.get("expected") or input_data.get("expected_sign")),
                    "predicted_sign": normalize_sign(result.get("predicted") or output_data.get("predicted_sign")),
                    "correct": int(bool(result.get("correct"))),
                    "error": result.get("error"),
                    "latency_ms": result.get("latency_ms"),
                    "avg_logprob": result.get("avg_logprob"),
                    "reasoning": result.get("reasoning") or output_data.get("reasoning"),
                    "variant": input_data.get("variant"),
                    "pair_case_type": input_data.get("pair_case_type"),
                    "matching_rule": input_data.get("matching_rule"),
                    "target_side": input_data.get("target_side"),
                    "example_true_side": input_data.get("example_true_side"),
                    "example_false_side": input_data.get("example_false_side"),
                    "target_exact_jel_codes": input_data.get("target_exact_jel_codes") or [],
                    "example_exact_jel_codes": input_data.get("example_exact_jel_codes") or [],
                    "shared_exact_jel_codes": input_data.get("shared_exact_jel_codes") or [],
                    "shared_exact_jel_count": input_data.get("shared_exact_jel_count"),
                    "target_exact_jel_count": input_data.get("target_exact_jel_count"),
                    "jel_overlap_ratio": input_data.get("jel_overlap_ratio"),
                    "different_paper": int(bool(input_data.get("different_paper"))),
                    **model_meta,
                }
                if task_name in {"task2", "task3"}:
                    example_rows = _example_triplets_from_input(input_data)
                    row.update(
                        {
                            "test_context": input_data.get("test_context"),
                            "example_count": len(example_rows),
                            "avg_similarity": input_data.get("avg_similarity"),
                            "sign_differs": bool(input_data.get("sign_differs")),
                            "example_triplets": example_rows,
                            "example_triplet_keys": [example["triplet_key"] for example in example_rows],
                            "example_displayed_signs": [example["displayed_sign"] for example in example_rows],
                            "example_original_signs": [example["original_sign"] for example in example_rows],
                            "example_paper_ids": [example["paper_id"] for example in example_rows],
                        }
                    )
                rows.append(row)
    return rows


def load_task1_rows(results_dir: str | Path | None = None) -> list[dict]:
    """Load paper-task-1 rows from existing sign-prediction results."""
    return load_result_rows("task1", task1_results_dir=results_dir)


def load_task2_rows(results_dir: str | Path | None = None) -> list[dict]:
    """Load paper-task-2 rows from existing example-sensitivity results."""
    return load_result_rows("task2", results_dir=results_dir)


def load_task3_rows(results_dir: str | Path | None = None) -> list[dict]:
    """Load paper-task-3 rows from existing noisy-example results."""
    return load_result_rows("task3", results_dir=results_dir)


def result_sign_coverage(rows: list[dict]) -> dict[str, int]:
    """Count predicted sign coverage for a collection of result rows."""
    counts = {sign: 0 for sign in VALID_SIGNS}
    counts["missing"] = 0
    for row in rows:
        sign = row.get("predicted_sign")
        if sign in counts:
            counts[sign] += 1
        else:
            counts["missing"] += 1
    return counts
