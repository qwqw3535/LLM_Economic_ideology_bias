#!/usr/bin/env python3
"""Run dedicated HF endpoints and HF router models in parallel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import (
    DEDICATED_ENDPOINT_MODELS,
    HF_ROUTER_MODELS,
    DEFAULT_TASK2_SOURCE_PATH,
    DEFAULT_TASK3_SOURCE_PATH,
    EvaluationConfig,
    TASK_TYPES,
)
from evaluation.data_generator import TestCaseGenerator
from evaluation.evaluator import EvaluationOrchestrator
from evaluation.metrics import MetricsComputer


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = str(ARTIFACT_ROOT / "data_derived" / "task1_ideology_subset_1056.jsonl")
DEFAULT_OUTPUT_ROOT = str(ARTIFACT_ROOT / "outputs" / "evaluation_hf_hybrid")

DEDICATED_ENDPOINT_ENV_MAP = {
    "meta-llama/Llama-3.2-1B": "LLAMA32_1B_ENDPOINT_URL",
    "meta-llama/Llama-3.2-3B": "LLAMA32_3B_ENDPOINT_URL",
    "meta-llama/Llama-3.1-70B": "LLAMA31_70B_ENDPOINT_URL",
}

DEDICATED_ENDPOINT_HARDWARE = {
    "meta-llama/Llama-3.2-1B": "A100 80GB x1",
    "meta-llama/Llama-3.2-3B": "H200 141GB x1",
    "meta-llama/Llama-3.1-70B": "H100 80GB x2",
}

DEDICATED_ENDPOINT_MAX_WORKERS = {
    "meta-llama/Llama-3.2-1B": 48,
    "meta-llama/Llama-3.2-3B": 64,
    "meta-llama/Llama-3.1-70B": 24,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ModelJob:
    model_id: str
    family: str
    endpoint_url: Optional[str] = None
    hardware: Optional[str] = None
    max_workers: Optional[int] = None


class HybridProgressTracker:
    """Track per-model progress in a single progress.json file."""

    def __init__(self, progress_path: Path, events_path: Path, interval_sec: int = 30):
        self.progress_path = progress_path
        self.events_path = events_path
        self.interval_sec = interval_sec
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.state: dict[str, Any] = {
            "updated_at": utc_now(),
            "models": {},
        }

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(self.interval_sec + 5, 5))
        self.flush()

    def register_model(self, model_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.state["models"][model_id] = payload
            self.state["updated_at"] = utc_now()
            snapshot = json.loads(json.dumps(self.state))
        self._write(snapshot)

    def update_model(self, model_id: str, event: Optional[str] = None, **fields: Any) -> None:
        with self._lock:
            model_state = self.state["models"].setdefault(model_id, {})
            model_state.update(fields)
            model_state["updated_at"] = utc_now()
            self.state["updated_at"] = utc_now()
            snapshot = json.loads(json.dumps(self.state))
        if event:
            self._append_event(model_id, event, snapshot["models"].get(model_id, {}))
        self._write(snapshot)

    def flush(self) -> None:
        with self._lock:
            snapshot = json.loads(json.dumps(self.state))
        for model_id, model_state in snapshot["models"].items():
            self._refresh_model_progress(model_id, model_state)
        snapshot["updated_at"] = utc_now()
        self._write(snapshot)

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self.flush()

    def _append_event(self, model_id: str, event: str, model_state: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": utc_now(),
                "model_id": model_id,
                "event": event,
                "state": model_state,
            }, ensure_ascii=False) + "\n")

    def _refresh_model_progress(self, model_id: str, model_state: dict[str, Any]) -> None:
        checkpoint_path = model_state.get("checkpoint_path")
        total_cases = model_state.get("task_total_cases")
        if not checkpoint_path or not total_cases:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            return
        try:
            payload = load_json(path)
            completed = len(payload.get("completed_ids", []))
            model_state["task_completed_cases"] = completed
            model_state["task_progress_pct"] = round((completed / total_cases) * 100, 2)
        except Exception:
            return

    def _write(self, snapshot: dict[str, Any]) -> None:
        save_json(self.progress_path, snapshot)


class HybridRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(args.output_root) / run_name
        self.log_path = self.run_dir / "runner.log"
        self.progress_path = self.run_dir / "progress.json"
        self.events_path = self.run_dir / "progress.events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.logger = self._setup_logging(self.log_path, args.verbose)
        self.progress = HybridProgressTracker(self.progress_path, self.events_path, args.progress_interval)
        self._summary_lock = threading.Lock()
        self.summary: dict[str, Any] = {
            "started_at": utc_now(),
            "run_dir": str(self.run_dir),
            "models": {},
        }

    def _setup_logging(self, log_path: Path, verbose: bool) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("hf_hybrid_runner")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def run(self) -> int:
        jobs = self._build_jobs()
        self.progress.start()
        try:
            for job in jobs:
                self.progress.register_model(job.model_id, {
                    "family": job.family,
                    "status": "pending",
                    "endpoint_url": job.endpoint_url,
                    "hardware": job.hardware,
                    "max_workers": job.max_workers,
                    "output_dir": str(self.run_dir / sanitize_model_id(job.model_id)),
                })

            with ThreadPoolExecutor(max_workers=min(self.args.max_model_parallelism, len(jobs))) as executor:
                futures = {executor.submit(self._run_job, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.exception(f"Model run failed for {job.model_id}: {e}")
                        self.progress.update_model(job.model_id, event="model_failed", status="failed", error=str(e))
                        with self._summary_lock:
                            self.summary["models"][job.model_id] = {"status": "failed", "error": str(e)}
                            save_json(self.summary_path, self.summary)

            with self._summary_lock:
                self.summary["finished_at"] = utc_now()
                self.summary["status"] = "completed_with_failures" if any(
                    m.get("status") == "failed" for m in self.summary["models"].values()
                ) else "completed"
                save_json(self.summary_path, self.summary)
            return 0 if self.summary["status"] == "completed" else 1
        finally:
            self.progress.stop()

    def _build_jobs(self) -> list[ModelJob]:
        jobs: list[ModelJob] = []
        dedicated_urls = {
            "meta-llama/Llama-3.2-1B": self.args.llama32_1b_endpoint_url or os.environ.get(DEDICATED_ENDPOINT_ENV_MAP["meta-llama/Llama-3.2-1B"]),
            "meta-llama/Llama-3.2-3B": self.args.llama32_3b_endpoint_url or os.environ.get(DEDICATED_ENDPOINT_ENV_MAP["meta-llama/Llama-3.2-3B"]),
            "meta-llama/Llama-3.1-70B": self.args.llama31_70b_endpoint_url or os.environ.get(DEDICATED_ENDPOINT_ENV_MAP["meta-llama/Llama-3.1-70B"]),
        }
        for model_id in DEDICATED_ENDPOINT_MODELS:
            endpoint_url = dedicated_urls.get(model_id)
            if not endpoint_url:
                raise ValueError(f"Missing dedicated endpoint URL for {model_id}")
            jobs.append(
                ModelJob(
                    model_id=model_id,
                    family="hf_endpoint",
                    endpoint_url=endpoint_url,
                    hardware=DEDICATED_ENDPOINT_HARDWARE.get(model_id),
                    max_workers=DEDICATED_ENDPOINT_MAX_WORKERS.get(model_id),
                )
            )
        for model_id in HF_ROUTER_MODELS:
            jobs.append(
                ModelJob(
                    model_id=model_id,
                    family="hf_router",
                    hardware="HF Router Provider",
                    max_workers=self.args.max_workers_per_model,
                )
            )
        return jobs

    def _run_job(self, job: ModelJob) -> None:
        model_dir = self.run_dir / sanitize_model_id(job.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.progress.update_model(job.model_id, event="model_started", status="running")
        self.logger.info(
            f"Starting {job.model_id} via {job.family}"
            + (f" on {job.hardware}" if job.hardware else "")
            + (f" with max_workers={job.max_workers}" if job.max_workers else "")
        )

        max_workers = job.max_workers or self.args.max_workers_per_model

        config = EvaluationConfig(
            task_types=self.args.task_types,
            models=[job.family],
            hf_model_ids=[job.model_id],
            hf_endpoint_url=job.endpoint_url,
            hf_api_token=self.args.token,
            hf_router_base_url=self.args.router_base_url,
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
            max_workers=max_workers,
            checkpoint_interval=self.args.checkpoint_interval,
            timeout=self.args.timeout,
            max_retries=self.args.max_retries,
        )
        orchestrator = EvaluationOrchestrator(config)
        generator = orchestrator.data_generator

        for task_type in self.args.task_types:
            total_cases = self._count_cases(generator, task_type, config)
            checkpoint_path = orchestrator._get_checkpoint_path(task_type, job.family, job.model_id)
            self.progress.update_model(
                job.model_id,
                event="task_started",
                current_task=task_type,
                checkpoint_path=str(checkpoint_path),
                task_total_cases=total_cases,
                task_completed_cases=0,
                task_progress_pct=0.0,
            )
            orchestrator.run_task(task_type)
            self.progress.update_model(
                job.model_id,
                event="task_completed",
                current_task=task_type,
                checkpoint_path=None,
                task_completed_cases=total_cases,
                task_total_cases=total_cases,
                task_progress_pct=100.0,
            )

        metrics = MetricsComputer(str(model_dir)).compute_all_metrics()
        with self._summary_lock:
            self.summary["models"][job.model_id] = {
                "status": "completed",
                "family": job.family,
                "hardware": job.hardware,
                "max_workers": max_workers,
                "output_dir": str(model_dir),
                "metrics_tasks": list(metrics.keys()),
                "finished_at": utc_now(),
            }
            save_json(self.summary_path, self.summary)
        self.progress.update_model(job.model_id, event="model_completed", status="completed", checkpoint_path=None)
        self.logger.info(f"Completed {job.model_id}")

    def _count_cases(self, generator: TestCaseGenerator, task_type: str, config: EvaluationConfig) -> int:
        if task_type == "task1":
            return len(generator.generate_task2_cases(config.max_samples_per_task))
        if task_type == "task2":
            return len(generator.generate_task3_cases(config.max_samples_per_task, num_examples=config.task2_num_examples))
        if task_type == "task3":
            return len(generator.generate_task4_cases(config.max_samples_per_task, num_examples=config.task3_num_examples))
        raise ValueError(f"Unknown task type: {task_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF dedicated endpoints and HF router models in parallel.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    parser.add_argument("--llama32-1b-endpoint-url", default=None)
    parser.add_argument("--llama32-3b-endpoint-url", default=None)
    parser.add_argument("--llama31-70b-endpoint-url", default=None)
    parser.add_argument("--router-base-url", default="https://router.huggingface.co/v1")
    parser.add_argument("--tasks", nargs="+", choices=TASK_TYPES + ["all"], default=["all"])
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--task2-source-path", default=DEFAULT_TASK2_SOURCE_PATH)
    parser.add_argument("--task3-source-path", default=DEFAULT_TASK3_SOURCE_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--task2-examples", type=int, default=1)
    parser.add_argument("--task3-examples", type=int, default=1)
    parser.add_argument("--max-workers-per-model", type=int, default=32)
    parser.add_argument("--max-model-parallelism", type=int, default=7)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", nargs="+", default=["\n"])
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token is required, or set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN.")
    args.task_types = TASK_TYPES if "all" in args.tasks else args.tasks
    return args


def main() -> int:
    args = parse_args()
    runner = HybridRunner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
