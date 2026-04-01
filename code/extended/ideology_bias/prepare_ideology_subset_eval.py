"""Prepare a seeded task1 eval bundle for the current 1056 ideology subset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from extended.ideology_bias.utils import iter_jsonl, make_triplet_key


DEFAULT_SUBSET_PATH = Path("extended/classification_results/ideology_triplet_subset_current.jsonl")
DEFAULT_OLD_CLASSIFICATION_PATH = Path("extended/classification_results/causal_triplets_gpt-5-mini_classified.jsonl")
DEFAULT_OLD_RESULTS_DIR = Path("econ_eval/evaluation_results_final/task1_causal_triplets")
DEFAULT_OUTPUT_DATA_PATH = Path("data/task1_ideology_subset_1056.jsonl")
DEFAULT_MISSING_DATA_PATH = Path("data/task1_ideology_subset_missing158.jsonl")
DEFAULT_OUTPUT_DIR = Path("econ_eval/evaluation_results_final/task1_ideology_subset_1056")


def _load_current_subset(path: Path) -> dict[str, dict]:
    return {row["triplet_key"]: row for row in iter_jsonl(path)}


def _load_old_sensitive_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in iter_jsonl(path):
        pref = (row.get("classification", {}).get("ideology_preference", {}) or {})
        if pref.get("is_ideologically_sensitive") is True:
            keys.add(make_triplet_key(row.get("paper_id"), row.get("treatment"), row.get("outcome")))
    return keys


def _extract_triplet_key_from_result(result_row: dict) -> str:
    input_data = result_row.get("input_data", {}) or {}
    return make_triplet_key(
        input_data.get("paper_id"),
        input_data.get("treatment"),
        input_data.get("outcome"),
    )


def _load_case_id_map(results_path: Path) -> dict[str, str]:
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    case_map: dict[str, str] = {}
    for row in payload.get("results", []):
        case_map[_extract_triplet_key_from_result(row)] = row.get("case_id")
    return case_map


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sorted_rows_with_case_ids(subset_rows: dict[str, dict], case_id_map: dict[str, str]) -> list[dict]:
    rows = []
    for triplet_key, row in subset_rows.items():
        case_id = case_id_map.get(triplet_key)
        if not case_id:
            raise KeyError(f"Missing old task1 case_id for {triplet_key}")
        rows.append(
            {
                "case_id": case_id,
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "author": row.get("author"),
                "publication_year": row.get("publication_year"),
                "published_venue": row.get("published_venue"),
                "paper_url": row.get("paper_url"),
                "treatment": row.get("treatment"),
                "outcome": row.get("outcome"),
                "sign": row.get("sign"),
                "context": row.get("context"),
                "identification_methods": row.get("identification_methods"),
                "triplet_key": triplet_key,
            }
        )
    return sorted(rows, key=lambda row: int(str(row["case_id"]).split("_", 1)[1]))


def _seed_output_dir(old_results_dir: Path, output_dir: Path, reused_case_ids: set[str], n_test_cases: int) -> int:
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seeded_models = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for old_result_path in sorted((old_results_dir / "results").glob("task1_*_results.json")):
        with open(old_result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        filtered_results = [row for row in payload.get("results", []) if row.get("case_id") in reused_case_ids]
        family = payload.get("family")
        model_name = payload.get("model")
        if not family or not model_name:
            continue

        safe_model_name = str(model_name).replace("/", "_").replace(":", "_")
        new_result_path = results_dir / f"task1_{family}_{safe_model_name}_results.json"
        new_checkpoint_path = checkpoints_dir / f"task1_{family}_{safe_model_name}_checkpoint.json"

        result_payload = {
            "task": "task1",
            "family": family,
            "model": model_name,
            "timestamp": timestamp,
            "n_test_cases": n_test_cases,
            "n_completed": len(filtered_results),
            "results": filtered_results,
        }
        checkpoint_payload = {
            "model": model_name,
            "completed_ids": [row["case_id"] for row in filtered_results],
            "results": filtered_results,
            "last_updated": timestamp,
        }

        with open(new_result_path, "w", encoding="utf-8") as handle:
            json.dump(result_payload, handle, ensure_ascii=False, indent=2)
        with open(new_checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint_payload, handle, ensure_ascii=False, indent=2)
        seeded_models += 1

    return seeded_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare seeded eval inputs for the 1056 ideology subset.")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--old-classification", type=Path, default=DEFAULT_OLD_CLASSIFICATION_PATH)
    parser.add_argument("--old-results-dir", type=Path, default=DEFAULT_OLD_RESULTS_DIR)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_OUTPUT_DATA_PATH)
    parser.add_argument("--missing-data", type=Path, default=DEFAULT_MISSING_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    subset_rows = _load_current_subset(args.subset)
    old_sensitive_keys = _load_old_sensitive_keys(args.old_classification)
    reference_results_path = args.old_results_dir / "results" / "task1_openai_gpt-5-mini_results.json"
    case_id_map = _load_case_id_map(reference_results_path)

    all_rows = _sorted_rows_with_case_ids(subset_rows, case_id_map)
    reused_rows = [row for row in all_rows if row["triplet_key"] in old_sensitive_keys]
    missing_rows = [row for row in all_rows if row["triplet_key"] not in old_sensitive_keys]

    _write_jsonl(args.output_data, all_rows)
    _write_jsonl(args.missing_data, missing_rows)
    seeded_models = _seed_output_dir(
        args.old_results_dir,
        args.output_dir,
        {row["case_id"] for row in reused_rows},
        len(all_rows),
    )

    summary = {
        "subset_total": len(all_rows),
        "reused_from_old": len(reused_rows),
        "missing_to_eval": len(missing_rows),
        "output_data": str(args.output_data),
        "missing_data": str(args.missing_data),
        "seeded_output_dir": str(args.output_dir),
        "seeded_model_files": seeded_models,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
