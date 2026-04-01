"""Annotate model reasoning strings with qualitative reasoning frames."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .llm import OpenAIJsonClient
from .load_results import load_task1_rows, load_task2_rows, load_task3_rows
from .paths import (
    ensure_output_dirs,
    reasoning_frame_output_paths,
)
from .reasoning_frames_prompt import REASONING_FRAMES_PROMPT_TEMPLATE
from .reasoning_frames_heuristic import annotate_reasoning_heuristic
from .schemas import REASONING_FRAME_RESPONSE_SCHEMA, validate_reasoning_frames
from .utils import append_jsonl, load_done_keys, stringify_for_csv, write_jsonl


logger = logging.getLogger(__name__)


def _task_rows(task_name: str) -> list[dict]:
    if task_name == "task1":
        return load_task1_rows()
    if task_name == "task2":
        return load_task2_rows()
    if task_name == "task3":
        return load_task3_rows()
    if task_name == "all":
        return load_task1_rows() + load_task2_rows() + load_task3_rows()
    raise ValueError(f"Unknown task: {task_name}")


def _reasoning_rows(task_name: str) -> list[dict]:
    rows: list[dict] = []
    for row in _task_rows(task_name):
        reasoning = (row.get("reasoning") or "").strip()
        if not reasoning:
            continue
        rows.append(
            {
                "reasoning_key": "|".join([row["paper_task"], row["family"], row["model"], row["case_id"]]),
                "paper_task": row["paper_task"],
                "family": row["family"],
                "model": row["model"],
                "case_id": row["case_id"],
                "triplet_key": row["triplet_key"],
                "paper_id": row["paper_id"],
                "reasoning": reasoning,
            }
        )
    return rows


def _annotate_one(client: OpenAIJsonClient, row: dict) -> dict:
    prompt = REASONING_FRAMES_PROMPT_TEMPLATE.format(
        task_name=row["paper_task"],
        family=row["family"],
        model=row["model"],
        reasoning=row["reasoning"],
    )
    call = client.call_json_schema(prompt, REASONING_FRAME_RESPONSE_SCHEMA)
    out = {
        **row,
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_model": client.model,
        "annotation_method": "llm",
    }
    if not call.success:
        out["annotation_error"] = call.error
        return out
    out["frames_annotation"] = call.payload
    out["validation_errors"] = validate_reasoning_frames(call.payload)
    return out


def _annotate_one_heuristic(row: dict) -> dict:
    payload = annotate_reasoning_heuristic(row["reasoning"])
    out = {
        **row,
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_model": "keyword_heuristic",
        "annotation_method": "heuristic",
        "frames_annotation": {"reasoning_frames": payload["reasoning_frames"]},
        "matched_keywords": payload.get("matched_keywords", {}),
        "frame_scores": payload.get("frame_scores", {}),
    }
    out["validation_errors"] = validate_reasoning_frames(out["frames_annotation"])
    return out


def _canonical_rows(raw_rows: list[dict]) -> list[dict]:
    canonical: list[dict] = []
    for row in raw_rows:
        payload = row.get("frames_annotation") or {}
        frames = payload.get("reasoning_frames") or {}
        canonical.append(
            {
                "reasoning_key": row["reasoning_key"],
                "paper_task": row["paper_task"],
                "family": row["family"],
                "model": row["model"],
                "case_id": row["case_id"],
                "triplet_key": row["triplet_key"],
                "paper_id": row["paper_id"],
                "annotation_method": row.get("annotation_method", "llm"),
                "annotation_model": row.get("annotation_model"),
                "primary_frame": frames.get("primary_frame"),
                "secondary_frames": frames.get("secondary_frames") or [],
                "justification": frames.get("justification"),
                "matched_keywords": row.get("matched_keywords") or {},
                "frame_scores": row.get("frame_scores") or {},
                "validation_errors": row.get("validation_errors") or [],
                "annotation_error": row.get("annotation_error"),
            }
        )
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate reasoning explanations with reasoning frames.")
    parser.add_argument("--task", choices=["task1", "task2", "task3", "all"], default="task1")
    parser.add_argument("--method", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--output-raw", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ensure_output_dirs()
    default_raw, default_jsonl, default_csv = reasoning_frame_output_paths(args.method)
    raw_output = Path(args.output_raw or default_raw)
    output_jsonl = Path(args.output_jsonl or default_jsonl)
    output_csv = Path(args.output_csv or default_csv)
    rows = _reasoning_rows(args.task)
    done_keys = load_done_keys(raw_output, key_field="reasoning_key")
    pending = [row for row in rows if row["reasoning_key"] not in done_keys]
    if args.limit is not None:
        pending = pending[: args.limit]

    logger.info("reasoning_rows=%s pending=%s method=%s", len(rows), len(pending), args.method)
    if pending:
        if args.method == "heuristic":
            for row in tqdm(pending, total=len(pending), desc="annotate_reasoning_heuristic"):
                append_jsonl(raw_output, [_annotate_one_heuristic(row)])
        else:
            client = OpenAIJsonClient(
                model=args.model,
                api_key_env=args.api_key_env,
                max_retries=args.max_retries,
            )
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(_annotate_one, client, row): row["reasoning_key"] for row in pending}
                for future in tqdm(as_completed(futures), total=len(futures), desc="annotate_reasoning"):
                    append_jsonl(raw_output, [future.result()])

    raw_rows = []
    if raw_output.exists():
        with open(raw_output, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw_rows.append(json.loads(line))
    canonical = _canonical_rows(raw_rows)
    write_jsonl(output_jsonl, canonical)
    pd.DataFrame([{key: stringify_for_csv(value) for key, value in row.items()} for row in canonical]).to_csv(
        output_csv,
        index=False,
    )
    print(
        json.dumps(
            {
                "task": args.task,
                "method": args.method,
                "n_reasonings": len(rows),
                "n_annotated_total": len(canonical),
                "raw_output": str(raw_output),
                "canonical_output": str(output_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
