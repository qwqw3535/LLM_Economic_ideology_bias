"""Multi-LLM ideology classification for econcausal triplets."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from ideology_classification_common import (
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_KEYS,
    MODEL_SPECS,
    RESULTS_DIR,
    build_clients,
    call_model,
    load_jsonl,
    load_prompt,
    make_triplet_key,
    render_prompt,
    resolve_consensus,
    save_jsonl,
)

DEFAULT_OUTPUT_PATH = RESULTS_DIR / "causal_triplets_multillm_ideology_classified.jsonl"
DEFAULT_MAX_PARALLEL = 128
DEFAULT_QWEN_MAX_PARALLEL = 64
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for logger_name in ("httpx", "httpcore", "openai"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)
for logger_name in (
    "google",
    "google.genai",
    "google_genai",
    "urllib3",
    "econ_eval.common.utils",
):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def model_key_to_filename(model_key: str) -> str:
    """Convert a model key to a filesystem-safe stem."""
    return model_key.replace("/", "_")


def load_partial_results(partial_path: Path) -> dict[str, dict]:
    """Load partial per-model responses keyed by triplet key."""
    partials: dict[str, dict] = {}
    if not partial_path.exists():
        return partials
    with partial_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            triplet_key = row.get("triplet_key")
            output = row.get("output")
            if triplet_key and isinstance(output, dict):
                partials[triplet_key] = output
    return partials


def parse_model_keys(raw: str | None) -> list[str]:
    """Parse comma-separated model keys."""
    if not raw:
        return list(DEFAULT_MODEL_KEYS)
    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(parsed) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Supported: {sorted(MODEL_SPECS)}")
    return parsed


def prepare_output_for_resume(output_path: Path, model_keys: list[str]) -> set[str]:
    """Keep only fully successful merged rows for the current model set and return their keys."""
    if not output_path.exists():
        return set()

    required_models = set(model_keys)
    done_keys: set[str] = set()
    kept_lines: list[str] = []
    pruned_rows = 0

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            aggregate = row.get("classification", {}).get("aggregate", {})
            row_models = set(aggregate.get("models_requested", []))
            if aggregate.get("failed_model_count", 0) or row_models != required_models:
                pruned_rows += 1
                continue

            kept_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
            done_keys.add(make_triplet_key(row))

    if pruned_rows:
        tmp_path = output_path.with_name(f"{output_path.stem}.resume_tmp{output_path.suffix}")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.writelines(kept_lines)
        tmp_path.replace(output_path)
        logger.info(
            "Pruned %d merged rows incompatible with current model set from %s",
            pruned_rows,
            output_path,
        )

    return done_keys


def write_partition_outputs(output_path: Path, write_agreement_splits: bool) -> None:
    """Create a compact summary, and optional split files, after the main run."""
    rows = load_jsonl(output_path)
    sensitive_rows = [
        row
        for row in rows
        if row.get("classification", {}).get("aggregate", {}).get("meets_sensitive_consensus")
    ]

    agreement_buckets: dict[int, list[dict]] = {count: [] for count in range(1, 5)}
    for row in sensitive_rows:
        agreement = row.get("classification", {}).get("aggregate", {}).get("consensus_signature_agreement", 0)
        if agreement in agreement_buckets:
            agreement_buckets[agreement].append(row)

    base_name = output_path.stem
    if write_agreement_splits:
        save_jsonl(
            sensitive_rows,
            output_path.with_name(f"{base_name}_sensitive_consensus.jsonl"),
        )
        for agreement, bucket_rows in agreement_buckets.items():
            save_jsonl(
                bucket_rows,
                output_path.with_name(f"{base_name}_agreement_{agreement}.jsonl"),
            )

    summary = {
        "total_rows": len(rows),
        "sensitive_consensus_rows": len(sensitive_rows),
        "agreement_counts": {
            str(agreement): len(bucket_rows)
            for agreement, bucket_rows in agreement_buckets.items()
        },
        "ground_truth_side_counts": {},
        "division_label_counts": {},
    }
    for row in sensitive_rows:
        aggregate = row.get("classification", {}).get("aggregate", {})
        side = aggregate.get("consensus_ground_truth_side", "unknown")
        division = aggregate.get("division_label", "unknown")
        summary["ground_truth_side_counts"][side] = summary["ground_truth_side_counts"].get(side, 0) + 1
        summary["division_label_counts"][division] = summary["division_label_counts"].get(division, 0) + 1

    summary_path = output_path.with_name(f"{base_name}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def process_items(
    items: list[dict],
    output_path: Path,
    prompt_template: str,
    clients: dict[str, object],
    max_parallel: int,
    qwen_max_parallel: int,
) -> list[dict]:
    """Process items with provider-specific executors and merge after all model calls finish."""
    total_items = len(items)
    all_item_keys = {make_triplet_key(item) for item in items}
    done_keys = prepare_output_for_resume(output_path, list(clients))
    remaining = [item for item in items if make_triplet_key(item) not in done_keys]
    logger.info("Loaded %d items, %d remaining after resume", len(items), len(remaining))

    if not remaining:
        return []

    total_triplets = len(remaining)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    item_by_key = {make_triplet_key(item): item for item in remaining}
    rendered_prompts = {
        triplet_key: render_prompt(prompt_template, item)
        for triplet_key, item in item_by_key.items()
    }
    partial_dir = output_path.with_name(f"{output_path.stem}_partials")
    partial_dir.mkdir(parents=True, exist_ok=True)
    provider_max_workers = {
        model_key: (
            qwen_max_parallel
            if model_key.startswith("qwen") or model_key.startswith("llama")
            else max_parallel
        )
        for model_key in clients
    }

    progress_lock = Lock()
    overall_bar = tqdm(total=total_triplets, desc="Merged triplets", position=0)
    partial_results: dict[str, dict[str, dict]] = defaultdict(dict)
    completed_triplets: set[str] = set()
    merged_results: list[dict] = []
    future_map = {}
    executors: dict[str, ThreadPoolExecutor] = {}
    partial_handles: dict[str, object] = {}
    partial_paths = {
        model_key: partial_dir / f"{model_key_to_filename(model_key)}.jsonl"
        for model_key in clients
    }
    loaded_partials = {
        model_key: load_partial_results(partial_paths[model_key])
        for model_key in clients
    }
    cached_success_counts = {
        model_key: sum(
            1
            for triplet_key, output in loaded_partials[model_key].items()
            if triplet_key in all_item_keys and isinstance(output, dict) and not output.get("error")
        )
        for model_key in clients
    }
    model_bars = {
        model_key: tqdm(total=total_items, desc=model_key, position=index + 1)
        for index, model_key in enumerate(clients)
    }
    model_stats = {
        model_key: {
            "cached_success": cached_success_counts[model_key],
            "submitted": 0,
            "attempted": 0,
            "retried_success": 0,
            "errors": 0,
            "inflight": 0,
        }
        for model_key in clients
    }

    for triplet_key in item_by_key:
        for model_key in clients:
            output = loaded_partials[model_key].get(triplet_key)
            if output and not output.get("error"):
                partial_results[triplet_key][model_key] = output

    def update_bar(model_key: str) -> None:
        stats = model_stats[model_key]
        model_bars[model_key].set_postfix(
            cached=stats["cached_success"],
            submitted=stats["submitted"],
            api_done=stats["attempted"],
            ok_new=stats["retried_success"],
            inflight=stats["inflight"],
            errors=stats["errors"],
        )

    def run_model_for_triplet(model_key: str, triplet_key: str) -> tuple[str, str, dict]:
        output = call_model(
            client=clients[model_key],
            model_key=model_key,
            prompt=rendered_prompts[triplet_key],
            paper_id=str(item_by_key[triplet_key].get("paper_id", "")),
        )
        return model_key, triplet_key, output

    def emit_merged_if_ready(triplet_key: str, handle) -> None:
        if len(partial_results[triplet_key]) != len(clients):
            return
        if triplet_key in completed_triplets:
            return
        completed_triplets.add(triplet_key)
        item = item_by_key[triplet_key]
        aggregate = resolve_consensus(item, partial_results[triplet_key])
        merged = {
            **item,
            "classification": {
                "per_model": partial_results[triplet_key],
                "aggregate": aggregate,
            },
        }
        merged_results.append(merged)
        handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
        handle.flush()
        with progress_lock:
            overall_bar.update(1)

    try:
        for model_key, bar in model_bars.items():
            cached_success = model_stats[model_key]["cached_success"]
            if cached_success:
                bar.update(cached_success)
            update_bar(model_key)

        for model_key in clients:
            partial_handles[model_key] = partial_paths[model_key].open("a", encoding="utf-8")

        with output_path.open("a", encoding="utf-8") as handle:
            for triplet_key in item_by_key:
                emit_merged_if_ready(triplet_key, handle)

            for model_key, client in clients.items():
                del client  # unused beyond presence check
                executors[model_key] = ThreadPoolExecutor(max_workers=provider_max_workers[model_key])
                for triplet_key in item_by_key:
                    if triplet_key in partial_results and model_key in partial_results[triplet_key]:
                        continue
                    with progress_lock:
                        model_stats[model_key]["submitted"] += 1
                        model_stats[model_key]["inflight"] += 1
                        update_bar(model_key)
                    future = executors[model_key].submit(run_model_for_triplet, model_key, triplet_key)
                    future_map[future] = model_key

            pending = set(future_map)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    model_key = future_map[future]
                    try:
                        result_model_key, triplet_key, output = future.result()
                    except Exception as exc:  # pragma: no cover
                        result_model_key = model_key
                        triplet_key = ""
                        output = {"model": model_key, "error": str(exc)}

                    with progress_lock:
                        model_stats[result_model_key]["attempted"] += 1
                        model_stats[result_model_key]["inflight"] -= 1
                        if output.get("error"):
                            model_stats[result_model_key]["errors"] += 1
                        else:
                            model_stats[result_model_key]["retried_success"] += 1
                            model_bars[result_model_key].update(1)
                        update_bar(result_model_key)

                    if not triplet_key:
                        continue

                    partial_results[triplet_key][result_model_key] = output
                    partial_handles[result_model_key].write(
                        json.dumps(
                            {"triplet_key": triplet_key, "output": output},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    partial_handles[result_model_key].flush()
                    emit_merged_if_ready(triplet_key, handle)
    finally:
        overall_bar.close()
        for bar in model_bars.values():
            bar.close()
        for handle in partial_handles.values():
            handle.close()
        for executor in executors.values():
            executor.shutdown(wait=False, cancel_futures=False)

    return merged_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-LLM ideology classification for causal triplets.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input JSONL file of causal triplets.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Main output JSONL path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt path. Defaults to classification_prompt_ideology.py.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_KEYS),
        help="Comma-separated model keys.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Maximum concurrent calls for non-Qwen providers.",
    )
    parser.add_argument(
        "--qwen_max_parallel",
        type=int,
        default=DEFAULT_QWEN_MAX_PARALLEL,
        help="Maximum concurrent Qwen/OpenRouter calls.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per provider call.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Provider timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N remaining triplets.",
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=12,
        default=None,
        help="Alias for --limit, defaulting to 12.",
    )
    parser.add_argument(
        "--write_agreement_splits",
        action="store_true",
        help="Also write separate JSONL files for sensitive consensus and agreement 1/2/3/4.",
    )
    args = parser.parse_args()

    model_keys = parse_model_keys(args.models)
    prompt_template = load_prompt(Path(args.prompt) if args.prompt else None)
    items = load_jsonl(args.input)
    effective_limit = args.test if args.test is not None else args.limit
    if effective_limit is not None:
        items = items[:effective_limit]

    logger.info("Using models: %s", ", ".join(model_keys))
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info(
        "Provider concurrency: default=%d, qwen=%d",
        args.max_parallel,
        args.qwen_max_parallel,
    )

    clients = build_clients(
        model_keys=model_keys,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    output_path = Path(args.output)
    process_items(
        items=items,
        output_path=output_path,
        prompt_template=prompt_template,
        clients=clients,
        max_parallel=args.max_parallel,
        qwen_max_parallel=args.qwen_max_parallel,
    )
    write_partition_outputs(output_path, write_agreement_splits=args.write_agreement_splits)
    logger.info("Wrote summary for %s", output_path)


if __name__ == "__main__":
    main()
