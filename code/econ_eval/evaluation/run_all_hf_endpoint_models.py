#!/usr/bin/env python3
"""Run all verified HF endpoint models sequentially on a single endpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi

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
from evaluation.data_generator import TestCaseGenerator


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = str(ARTIFACT_ROOT / "data_derived" / "task1_ideology_subset_1056.jsonl")
DEFAULT_OUTPUT_ROOT = str(ARTIFACT_ROOT / "outputs" / "evaluation_hf_batch")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ProgressTracker:
    """Write a periodic progress snapshot and append event logs."""

    def __init__(self, progress_path: Path, events_path: Path, interval_sec: int = 30):
        self.progress_path = progress_path
        self.events_path = events_path
        self.interval_sec = interval_sec
        self.state: dict[str, Any] = {"updated_at": utc_now()}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(self.interval_sec + 5, 5))
        self.flush()

    def update(self, event: Optional[str] = None, **fields: Any) -> None:
        with self._lock:
            self.state.update(fields)
            self.state["updated_at"] = utc_now()
            snapshot = deepcopy(self.state)
        if event:
            self._append_event(event, snapshot)
        self._write(snapshot)

    def flush(self) -> None:
        with self._lock:
            snapshot = deepcopy(self.state)
        self._refresh_checkpoint_progress(snapshot)
        self._write(snapshot)

    def _append_event(self, event: str, snapshot: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": utc_now(), "event": event, "state": snapshot}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self.flush()

    def _refresh_checkpoint_progress(self, snapshot: dict[str, Any]) -> None:
        checkpoint_path = snapshot.get("checkpoint_path")
        total_cases = snapshot.get("task_total_cases")
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            return
        try:
            payload = load_json(path)
            completed = len(payload.get("completed_ids", []))
            snapshot["task_completed_cases"] = completed
            snapshot["task_total_cases"] = total_cases
            snapshot["task_progress_pct"] = round((completed / total_cases) * 100, 2) if total_cases else None
            with self._lock:
                self.state.update(
                    {
                        "task_completed_cases": completed,
                        "task_total_cases": total_cases,
                        "task_progress_pct": snapshot["task_progress_pct"],
                        "updated_at": utc_now(),
                    }
                )
        except Exception:
            return

    def _write(self, snapshot: dict[str, Any]) -> None:
        self._refresh_checkpoint_progress(snapshot)
        save_json(self.progress_path, snapshot)


@dataclass
class BatchRunArgs:
    endpoint_name: str
    namespace: Optional[str]
    token: str
    output_root: str
    run_name: Optional[str]
    task_types: list[str]
    models: list[str]
    data_path: str
    task2_source_path: str
    task3_source_path: str
    max_samples: Optional[int]
    task2_examples: int
    task3_examples: int
    max_workers: int
    checkpoint_interval: int
    timeout: int
    max_retries: int
    max_new_tokens: int
    temperature: float
    top_p: float
    stop: list[str]
    endpoint_timeout: int
    endpoint_refresh_every: int
    progress_interval: int
    final_action: str
    create_if_missing: bool
    vendor: str
    region: str
    accelerator: str
    instance_size: str
    instance_type: str
    framework: str
    task_name: str
    endpoint_type: str
    verbose: bool


class HFEndpointBatchRunner:
    def __init__(self, args: BatchRunArgs, api: Optional[HfApi] = None):
        self.args = args
        self.api = api or HfApi(token=args.token)

        run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(args.output_root) / run_name
        self.log_path = self.run_dir / "runner.log"
        self.progress_path = self.run_dir / "progress.json"
        self.events_path = self.run_dir / "progress.events.jsonl"
        self.summary_path = self.run_dir / "batch_summary.json"
        self.logger = self._setup_logging(self.log_path, args.verbose)
        self.progress = ProgressTracker(self.progress_path, self.events_path, args.progress_interval)
        self.summary: dict[str, Any] = {
            "started_at": utc_now(),
            "endpoint_name": args.endpoint_name,
            "namespace": args.namespace,
            "models": [],
        }

    def _setup_logging(self, log_path: Path, verbose: bool) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("hf_endpoint_batch")
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def run(self) -> int:
        self.progress.start()
        try:
            endpoint = self._get_or_create_endpoint()
            self.progress.update(
                event="batch_started",
                status="starting",
                run_dir=str(self.run_dir),
                endpoint_name=self.args.endpoint_name,
                namespace=self.args.namespace,
                models_total=len(self.args.models),
                models_completed=0,
            )

            completed_models = 0
            for model_index, model_id in enumerate(self.args.models, start=1):
                model_dir = self.run_dir / sanitize_model_id(model_id)
                if self._all_tasks_complete(model_dir, model_id):
                    self.logger.info(f"Skipping {model_id}: all requested tasks already complete")
                    completed_models += 1
                    self.progress.update(
                        event="model_skipped",
                        current_model=model_id,
                        current_model_index=model_index,
                        models_completed=completed_models,
                        status="skipped_completed_model",
                    )
                    continue

                self.progress.update(
                    event="deploy_started",
                    status="deploying",
                    current_model=model_id,
                    current_model_index=model_index,
                    checkpoint_path=None,
                    task_completed_cases=None,
                    task_total_cases=None,
                    task_progress_pct=None,
                )
                endpoint = self._deploy_model(endpoint, model_id)
                self.progress.update(
                    event="deploy_ready",
                    status="endpoint_ready",
                    endpoint_url=endpoint.url,
                    current_model=model_id,
                )

                self._run_model(endpoint.url, model_id, model_dir)
                completed_models += 1
                self.progress.update(
                    event="model_completed",
                    status="model_completed",
                    current_model=model_id,
                    models_completed=completed_models,
                    checkpoint_path=None,
                    task_completed_cases=None,
                    task_total_cases=None,
                    task_progress_pct=None,
                )

            self._finalize_endpoint(endpoint)
            self.summary["finished_at"] = utc_now()
            self.summary["status"] = "completed"
            save_json(self.summary_path, self.summary)
            self.progress.update(event="batch_completed", status="completed", summary_path=str(self.summary_path))
            return 0
        except Exception as e:
            self.logger.exception(f"Batch run failed: {e}")
            self.summary["finished_at"] = utc_now()
            self.summary["status"] = "failed"
            self.summary["error"] = str(e)
            save_json(self.summary_path, self.summary)
            self.progress.update(event="batch_failed", status="failed", error=str(e))
            return 1
        finally:
            self.progress.stop()

    def _get_or_create_endpoint(self):
        try:
            endpoint = self.api.get_inference_endpoint(
                name=self.args.endpoint_name,
                namespace=self.args.namespace,
                token=self.args.token,
            )
            self.logger.info(f"Loaded existing endpoint {self.args.endpoint_name}")
            return endpoint
        except Exception:
            if not self.args.create_if_missing:
                raise

        self.logger.info(f"Creating endpoint {self.args.endpoint_name}")
        return self.api.create_inference_endpoint(
            name=self.args.endpoint_name,
            namespace=self.args.namespace,
            repository=self.args.models[0],
            framework=self.args.framework,
            accelerator=self.args.accelerator,
            instance_size=self.args.instance_size,
            instance_type=self.args.instance_type,
            region=self.args.region,
            vendor=self.args.vendor,
            task=self.args.task_name,
            type=self.args.endpoint_type,
            min_replica=1,
            max_replica=1,
            token=self.args.token,
        ).wait(timeout=self.args.endpoint_timeout, refresh_every=self.args.endpoint_refresh_every)

    def _deploy_model(self, endpoint, model_id: str):
        self.logger.info(f"Updating endpoint to repository={model_id}")
        endpoint = endpoint.update(
            repository=model_id,
            framework=self.args.framework,
            task=self.args.task_name,
        )
        endpoint.wait(timeout=self.args.endpoint_timeout, refresh_every=self.args.endpoint_refresh_every)
        self.logger.info(f"Endpoint ready for {model_id}: {endpoint.url}")
        return endpoint

    def _run_model(self, endpoint_url: str, model_id: str, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        config = EvaluationConfig(
            task_types=self.args.task_types,
            models=["hf_endpoint"],
            hf_model_ids=[model_id],
            hf_endpoint_url=endpoint_url,
            hf_api_token=self.args.token,
            label_only=True,
            hf_max_new_tokens=self.args.max_new_tokens,
            hf_temperature=self.args.temperature,
            hf_top_p=self.args.top_p,
            hf_stop_sequences=self.args.stop,
            data_path=self.args.data_path,
            output_dir=str(model_dir),
            task2_source_path=self.args.task2_source_path,
            task3_source_path=self.args.task3_source_path,
            max_samples_per_task=self.args.max_samples,
            task2_num_examples=self.args.task2_examples,
            task3_num_examples=self.args.task3_examples,
            max_workers=self.args.max_workers,
            checkpoint_interval=self.args.checkpoint_interval,
            timeout=self.args.timeout,
            max_retries=self.args.max_retries,
        )
        orchestrator = EvaluationOrchestrator(config)
        model_entry = {
            "model_id": model_id,
            "output_dir": str(model_dir),
            "started_at": utc_now(),
            "tasks": [],
        }
        self.summary["models"].append(model_entry)
        save_json(self.summary_path, self.summary)

        for task_type in self.args.task_types:
            total_cases = self._count_cases(orchestrator.data_generator, task_type, config)
            checkpoint_path = orchestrator._get_checkpoint_path(task_type, "hf_endpoint", model_id)
            self.progress.update(
                event="task_started",
                status="evaluating",
                current_model=model_id,
                current_task=task_type,
                current_output_dir=str(model_dir),
                checkpoint_path=str(checkpoint_path),
                task_total_cases=total_cases,
            )
            self.logger.info(f"Running {task_type} for {model_id} ({total_cases} cases)")
            orchestrator.run_task(task_type)
            model_entry["tasks"].append(
                {
                    "task": task_type,
                    "finished_at": utc_now(),
                    "checkpoint_path": str(checkpoint_path),
                    "total_cases": total_cases,
                }
            )
            save_json(self.summary_path, self.summary)
            self.progress.update(
                event="task_completed",
                current_model=model_id,
                current_task=task_type,
                checkpoint_path=None,
                task_completed_cases=total_cases,
                task_total_cases=total_cases,
                task_progress_pct=100.0,
            )

        MetricsComputer(str(model_dir)).compute_all_metrics()
        model_entry["finished_at"] = utc_now()
        save_json(self.summary_path, self.summary)

    def _count_cases(self, generator: TestCaseGenerator, task_type: str, config: EvaluationConfig) -> int:
        if task_type == "task1":
            return len(generator.generate_task2_cases(config.max_samples_per_task))
        if task_type == "task2":
            return len(generator.generate_task3_cases(config.max_samples_per_task, num_examples=config.task2_num_examples))
        if task_type == "task3":
            return len(generator.generate_task4_cases(config.max_samples_per_task, num_examples=config.task3_num_examples))
        raise ValueError(f"Unknown task type: {task_type}")

    def _all_tasks_complete(self, model_dir: Path, model_id: str) -> bool:
        result_dir = model_dir / "results"
        if not result_dir.exists():
            return False
        safe_model = sanitize_model_id(model_id)
        for task_type in self.args.task_types:
            path = result_dir / f"{task_type}_hf_endpoint_{safe_model}_results.json"
            if not path.exists():
                return False
            try:
                payload = load_json(path)
            except Exception:
                return False
            if payload.get("n_completed") != payload.get("n_test_cases"):
                return False
        return True

    def _finalize_endpoint(self, endpoint) -> None:
        action = self.args.final_action
        if action == "leave-running":
            self.logger.info("Leaving endpoint running")
            return
        if action == "pause":
            self.logger.info("Pausing endpoint")
            endpoint.pause()
            return
        if action == "scale-to-zero":
            self.logger.info("Scaling endpoint to zero")
            endpoint.scale_to_zero()
            return
        raise ValueError(f"Unknown final action: {action}")


def parse_args() -> BatchRunArgs:
    parser = argparse.ArgumentParser(description="Run all verified HF endpoint models on one endpoint.")
    parser.add_argument("--endpoint-name", required=True, help="Inference Endpoint name.")
    parser.add_argument("--namespace", default=None, help="HF namespace or organization.")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        required=False,
        help="HF access token. Defaults to HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for this batch run.")
    parser.add_argument("--run-name", default=None, help="Fixed run directory name. Reuse this to resume into the same directory.")
    parser.add_argument("--tasks", nargs="+", choices=TASK_TYPES + ["all"], default=["all"])
    parser.add_argument("--models", nargs="+", choices=VERIFIED_HF_ENDPOINT_MODELS + ["all"], default=["all"])
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--task2-source-path", default=DEFAULT_TASK2_SOURCE_PATH)
    parser.add_argument("--task3-source-path", default=DEFAULT_TASK3_SOURCE_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--task2-examples", type=int, default=1)
    parser.add_argument("--task3-examples", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", nargs="+", default=["\n"])
    parser.add_argument("--endpoint-timeout", type=int, default=1800)
    parser.add_argument("--endpoint-refresh-every", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument(
        "--final-action",
        choices=["scale-to-zero", "pause", "leave-running"],
        default="scale-to-zero",
    )
    parser.add_argument("--create-if-missing", action="store_true")
    parser.add_argument("--vendor", default="aws")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--instance-size", default="x1")
    parser.add_argument("--instance-type", default="nvidia-h200")
    parser.add_argument("--framework", default="vllm")
    parser.add_argument("--task-name", default="text-generation")
    parser.add_argument("--endpoint-type", default="protected")
    parser.add_argument("--verbose", "-v", action="store_true")
    parsed = parser.parse_args()

    if not parsed.token:
        parser.error("--token is required, or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN.")

    task_types = TASK_TYPES if "all" in parsed.tasks else parsed.tasks
    models = VERIFIED_HF_ENDPOINT_MODELS if "all" in parsed.models else parsed.models
    return BatchRunArgs(
        endpoint_name=parsed.endpoint_name,
        namespace=parsed.namespace,
        token=parsed.token,
        output_root=parsed.output_root,
        run_name=parsed.run_name,
        task_types=task_types,
        models=models,
        data_path=parsed.data_path,
        task2_source_path=parsed.task2_source_path,
        task3_source_path=parsed.task3_source_path,
        max_samples=parsed.max_samples,
        task2_examples=parsed.task2_examples,
        task3_examples=parsed.task3_examples,
        max_workers=parsed.max_workers,
        checkpoint_interval=parsed.checkpoint_interval,
        timeout=parsed.timeout,
        max_retries=parsed.max_retries,
        max_new_tokens=parsed.max_new_tokens,
        temperature=parsed.temperature,
        top_p=parsed.top_p,
        stop=parsed.stop,
        endpoint_timeout=parsed.endpoint_timeout,
        endpoint_refresh_every=parsed.endpoint_refresh_every,
        progress_interval=parsed.progress_interval,
        final_action=parsed.final_action,
        create_if_missing=parsed.create_if_missing,
        vendor=parsed.vendor,
        region=parsed.region,
        accelerator=parsed.accelerator,
        instance_size=parsed.instance_size,
        instance_type=parsed.instance_type,
        framework=parsed.framework,
        task_name=parsed.task_name,
        endpoint_type=parsed.endpoint_type,
        verbose=parsed.verbose,
    )


def main() -> int:
    args = parse_args()
    runner = HFEndpointBatchRunner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
