"""Seed a single task eval directory from prior result bundles."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from extended.ideology_bias.utils import iter_jsonl
else:
    from .utils import iter_jsonl


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed one task's eval results from prior bundles.")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--old-results-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()

    new_rows = _load_generated_rows(args.source_path)
    new_row_map = _build_new_row_map(new_rows)
    results_dir = args.output_dir / "results"
    checkpoints_dir = args.output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    model_rows: dict[tuple[str, str], list[dict]] = {}
    reused_case_ids: set[str] = set()

    for old_root in args.old_results_dir:
        old_results = old_root / "results"
        if not old_results.exists():
            continue
        for old_result_path in sorted(old_results.glob(f"{args.task_name}_*_results.json")):
            payload = json.loads(old_result_path.read_text(encoding="utf-8"))
            family = payload.get("family")
            model_name = payload.get("model")
            if not family or not model_name:
                continue
            key = (family, model_name)
            model_rows.setdefault(key, [])
            seen_existing = {row["case_id"] for row in model_rows[key]}
            for old_row in payload.get("results", []):
                sig = _case_signature_from_old_result(old_row)
                new_input_row = new_row_map.get(sig)
                if not new_input_row:
                    continue
                new_case_id = new_input_row["case_id"]
                if new_case_id in seen_existing:
                    continue
                model_rows[key].append(_remap_result_row(old_row, new_input_row))
                reused_case_ids.add(new_case_id)
                seen_existing.add(new_case_id)

    for (family, model_name), remapped_results in model_rows.items():
        safe_model_name = str(model_name).replace("/", "_").replace(":", "_")
        result_payload = {
            "task": args.task_name,
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
        (results_dir / f"{args.task_name}_{family}_{safe_model_name}_results.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (checkpoints_dir / f"{args.task_name}_{family}_{safe_model_name}_checkpoint.json").write_text(
            json.dumps(checkpoint_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "task_name": args.task_name,
        "source_path": str(args.source_path),
        "old_results_dirs": [str(path) for path in args.old_results_dir],
        "seeded_output_dir": str(args.output_dir),
        "total_cases": len(new_rows),
        "reused_cases": len(reused_case_ids),
        "new_cases_to_eval": len(new_rows) - len(reused_case_ids),
        "seeded_model_files": len(model_rows),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
