"""Prepare seeded task2/task3 eval bundles for the current ideology subset shared2 files."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extended.ideology_bias.utils import iter_jsonl


DEFAULT_TASK1_DATA_PATH = Path("data/task1_ideology_subset_1056.jsonl")
DEFAULT_TASK2_SOURCE_PATH = Path("data/task2_ideology_subset_jel_similarity_side_capped_jaccard05_shared2.jsonl")
DEFAULT_TASK3_SOURCE_PATH = Path("data/task3_ideology_subset_jel_similarity_side_capped_jaccard05_shared2.jsonl")
DEFAULT_OLD_RESULTS_DIR = Path("econ_eval/evaluation_results_final/task23_jel_similarity_side_capped_jaccard05_shared2")
DEFAULT_OUTPUT_DIR = Path("econ_eval/evaluation_results_final/task23_ideology_subset_jel_similarity_side_capped_jaccard05_shared2")
DEFAULT_SUMMARY_PATH = Path("extended/classification_results/task23_ideology_subset_eval_seed_summary.json")


def _norm(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)


def _case_signature_from_generated_row(row: dict) -> tuple[str, ...]:
    examples = row.get("examples") or row.get("example_details") or []
    if not examples:
        raise KeyError(f"Missing examples for generated row {row.get('case_id')}")
    example = examples[0]
    return (
        _norm(row.get("treatment")),
        _norm(row.get("outcome")),
        _norm(row.get("sign")),
        _norm(row.get("test_context") or row.get("context")),
        _norm(row.get("title")),
        _norm(row.get("author")),
        _norm(row.get("publication_year")),
        _norm(row.get("published_venue")),
        _norm(example.get("treatment")),
        _norm(example.get("outcome")),
        _norm(example.get("sign")),
        _norm(example.get("context")),
        _norm(example.get("title")),
        _norm(example.get("author")),
        _norm(example.get("publication_year")),
        _norm(example.get("published_venue")),
        _norm(row.get("pair_case_type")),
    )


def _case_signature_from_old_result(result_row: dict) -> tuple[str, ...]:
    input_data = result_row.get("input_data", {}) or {}
    examples = input_data.get("examples") or []
    if not examples:
        raise KeyError(f"Missing examples for old result {result_row.get('case_id')}")
    example = examples[0]
    return (
        _norm(input_data.get("treatment")),
        _norm(input_data.get("outcome")),
        _norm(input_data.get("sign")),
        _norm(input_data.get("test_context") or input_data.get("context")),
        _norm(input_data.get("title")),
        _norm(input_data.get("author")),
        _norm(input_data.get("publication_year")),
        _norm(input_data.get("published_venue")),
        _norm(example.get("treatment")),
        _norm(example.get("outcome")),
        _norm(example.get("sign")),
        _norm(example.get("context")),
        _norm(example.get("title")),
        _norm(example.get("author")),
        _norm(example.get("publication_year")),
        _norm(example.get("published_venue")),
        _norm(input_data.get("pair_case_type")),
    )


def _load_generated_rows(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def _load_old_result_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_new_row_map(rows: list[dict]) -> dict[tuple[str, ...], dict]:
    mapping: dict[tuple[str, ...], dict] = {}
    for row in rows:
        sig = _case_signature_from_generated_row(row)
        if sig in mapping:
            raise ValueError(f"Duplicate generated case signature for {row.get('case_id')}")
        mapping[sig] = row
    return mapping


def _remap_result_row(old_row: dict, new_input_row: dict) -> dict:
    seeded = copy.deepcopy(old_row)
    seeded["case_id"] = new_input_row["case_id"]
    seeded["input_data"] = copy.deepcopy(new_input_row)
    seeded["input_data"]["case_id"] = new_input_row["case_id"]
    return seeded


def _seed_task_results(
    old_results_dir: Path,
    output_dir: Path,
    task_name: str,
    new_rows: list[dict],
) -> dict:
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    new_row_map = _build_new_row_map(new_rows)
    reused_case_ids: set[str] = set()
    seeded_files = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for old_result_path in sorted((old_results_dir / "results").glob(f"{task_name}_*_results.json")):
        payload = _load_old_result_payload(old_result_path)
        family = payload.get("family")
        model_name = payload.get("model")
        if not family or not model_name:
            continue

        remapped_results: list[dict] = []
        seen_new_case_ids: set[str] = set()

        for old_row in payload.get("results", []):
            sig = _case_signature_from_old_result(old_row)
            new_input_row = new_row_map.get(sig)
            if not new_input_row:
                continue
            new_case_id = new_input_row["case_id"]
            if new_case_id in seen_new_case_ids:
                continue
            remapped_results.append(_remap_result_row(old_row, new_input_row))
            seen_new_case_ids.add(new_case_id)

        safe_model_name = str(model_name).replace("/", "_").replace(":", "_")
        new_result_path = results_dir / f"{task_name}_{family}_{safe_model_name}_results.json"
        new_checkpoint_path = checkpoints_dir / f"{task_name}_{family}_{safe_model_name}_checkpoint.json"

        result_payload = {
            "task": task_name,
            "family": family,
            "model": model_name,
            "timestamp": timestamp,
            "n_test_cases": len(new_rows),
            "n_completed": len(remapped_results),
            "results": remapped_results,
        }
        checkpoint_payload = {
            "model": model_name,
            "completed_ids": [row["case_id"] for row in remapped_results],
            "results": remapped_results,
            "last_updated": timestamp,
        }

        with open(new_result_path, "w", encoding="utf-8") as handle:
            json.dump(result_payload, handle, ensure_ascii=False, indent=2)
        with open(new_checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint_payload, handle, ensure_ascii=False, indent=2)

        reused_case_ids.update(seen_new_case_ids)
        seeded_files += 1

    return {
        "task": task_name,
        "total_cases": len(new_rows),
        "reused_cases": len(reused_case_ids),
        "new_cases_to_eval": len(new_rows) - len(reused_case_ids),
        "seeded_model_files": seeded_files,
    }


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare seeded task2/task3 eval bundle for ideology subset shared2.")
    parser.add_argument("--task1-data", type=Path, default=DEFAULT_TASK1_DATA_PATH)
    parser.add_argument("--task2-source", type=Path, default=DEFAULT_TASK2_SOURCE_PATH)
    parser.add_argument("--task3-source", type=Path, default=DEFAULT_TASK3_SOURCE_PATH)
    parser.add_argument("--old-results-dir", type=Path, default=DEFAULT_OLD_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    task2_rows = _load_generated_rows(args.task2_source)
    task3_rows = _load_generated_rows(args.task3_source)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    task2_summary = _seed_task_results(args.old_results_dir, args.output_dir, "task2", task2_rows)
    task3_summary = _seed_task_results(args.old_results_dir, args.output_dir, "task3", task3_rows)

    summary = {
        "task1_data_path": str(args.task1_data),
        "task2_source_path": str(args.task2_source),
        "task3_source_path": str(args.task3_source),
        "old_results_dir": str(args.old_results_dir),
        "seeded_output_dir": str(args.output_dir),
        "task2": task2_summary,
        "task3": task3_summary,
    }
    _write_summary(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
