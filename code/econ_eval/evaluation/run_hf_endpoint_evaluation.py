#!/usr/bin/env python3
"""CLI for running label-only evaluation against a dedicated HF endpoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import (
    DEFAULT_TASK2_SOURCE_PATH,
    DEFAULT_TASK3_SOURCE_PATH,
    EvaluationConfig,
    TASK_TYPES,
    VERIFIED_HF_ENDPOINT_MODELS,
)
from evaluation.evaluator import EvaluationOrchestrator
from evaluation.metrics import MetricsComputer


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = str(ARTIFACT_ROOT / "data_derived" / "task1_ideology_subset_1056.jsonl")
DEFAULT_OUTPUT_DIR = str(ARTIFACT_ROOT / "outputs" / "evaluation_hf_endpoint")
DEFAULT_MODEL_MAX_WORKERS = {
    "meta-llama/Llama-3.1-8B": 512,
}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run label-only EconCausality evaluation on a Hugging Face Inference Endpoint.",
    )
    parser.add_argument(
        "--model-id",
        choices=VERIFIED_HF_ENDPOINT_MODELS,
        help="Exact HF model repo ID deployed on the endpoint.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("HF_ENDPOINT_URL"),
        help="HF endpoint URL. Can also be provided via HF_ENDPOINT_URL.",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="HF API token. Optional if the endpoint is publicly reachable.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_TYPES + ["all"],
        default=["all"],
        help="Tasks to run (default: all).",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Task1 source data path (default: causal_triplets.jsonl).",
    )
    parser.add_argument(
        "--task2-source-path",
        default=DEFAULT_TASK2_SOURCE_PATH,
        help="Task2 source JSONL path.",
    )
    parser.add_argument(
        "--task3-source-path",
        default=DEFAULT_TASK3_SOURCE_PATH,
        help="Task3 source JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results and checkpoints.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--task2-examples",
        type=int,
        default=1,
        help="Number of in-context examples for task2.",
    )
    parser.add_argument(
        "--task3-examples",
        type=int,
        default=1,
        help="Number of in-context examples for task3.",
    )
    parser.add_argument(
        "--task1-prompt-variant",
        choices=["default", "choice", "raw"],
        default="default",
        help="Task1 label-only prompt variant.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum client-side parallel requests. Defaults to a model-specific value when available.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Save checkpoint every N completed cases.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retries for failed endpoint calls.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum generated tokens for the label-only decode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--stop",
        nargs="+",
        default=["\n"],
        help="Stop sequences passed to the endpoint.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics from an existing output directory.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the verified model IDs and exit.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    if args.list_models:
        for model_id in VERIFIED_HF_ENDPOINT_MODELS:
            print(model_id)
        return 0

    if args.metrics_only:
        MetricsComputer(args.output_dir).print_summary()
        return 0

    if not args.model_id:
        parser.error("--model-id is required unless --list-models or --metrics-only is used.")
    if not args.endpoint_url:
        parser.error("--endpoint-url is required unless --metrics-only is used.")

    task_types = TASK_TYPES if "all" in args.tasks else args.tasks
    resolved_max_workers = args.max_workers or DEFAULT_MODEL_MAX_WORKERS.get(args.model_id, 32)

    config = EvaluationConfig(
        task_types=task_types,
        models=["hf_endpoint"],
        hf_model_ids=[args.model_id],
        hf_endpoint_url=args.endpoint_url,
        hf_api_token=args.api_token,
        label_only=True,
        hf_max_new_tokens=args.max_new_tokens,
        hf_temperature=args.temperature,
        hf_top_p=args.top_p,
        hf_stop_sequences=args.stop,
        data_path=args.data_path,
        output_dir=args.output_dir,
        task2_source_path=args.task2_source_path,
        task3_source_path=args.task3_source_path,
        max_samples_per_task=args.max_samples,
        task2_num_examples=args.task2_examples,
        task3_num_examples=args.task3_examples,
        task1_prompt_variant=args.task1_prompt_variant,
        max_workers=resolved_max_workers,
        checkpoint_interval=args.checkpoint_interval,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    logger.info("Running HF endpoint evaluation")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Endpoint: {args.endpoint_url}")
    logger.info(f"Tasks: {task_types}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Max workers: {resolved_max_workers}")
    logger.info(f"Task1 prompt variant: {args.task1_prompt_variant}")

    orchestrator = EvaluationOrchestrator(config)
    orchestrator.run_all_tasks()

    MetricsComputer(args.output_dir).print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
