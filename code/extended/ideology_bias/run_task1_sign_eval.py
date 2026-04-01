"""Future-ready Task1 sign-prediction rerun over the full corpus."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .load_sources import load_source_file
from .paths import OUTPUT_DIR, ensure_output_dirs
from .schemas import VALID_SIGNS, normalize_sign
from .utils import safe_slug

SIGN_PREDICTION_PROMPT = """# Role
You are an expert economist.

# Task
Given a context and a treatment–outcome pair, predict the most likely sign of the causal effect.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Context
{context}

# Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", or "mixed"
- reasoning: A concise explanation of your reasoning
"""

SIGN_PREDICTION_SCHEMA = {
    "name": "sign_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": list(VALID_SIGNS),
            },
            "reasoning": {"type": "string"},
        },
        "required": ["predicted_sign", "reasoning"],
        "additionalProperties": False,
    },
}

MODEL_SPECS = {
    "openai": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.2"],
    },
    "gemini": {
        "models": ["gemini-2.5-flash", "gemini-3-flash-preview"],
    },
    "grok": {
        "models": ["grok-3-mini", "grok-3", "grok-4-1-fast-reasoning"],
    },
    "qwen": {
        "models": ["qwen/qwen3-8b", "qwen/qwen3-14b", "qwen/qwen3-32b"],
    },
    "llama": {
        "models": [
            "meta-llama/llama-3.2-1b-instruct",
            "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
        ],
    },
}

logger = logging.getLogger(__name__)


def _create_client(family: str, model: str, timeout: int, max_retries: int, max_workers: int):
    from econ_eval.common.utils import GeminiClient, GrokClient, OpenAIClient, OpenRouterClient

    if family == "openai":
        return OpenAIClient(model=model, timeout=timeout, max_retries=max_retries, max_workers=max_workers)
    if family == "gemini":
        return GeminiClient(model=model, timeout=timeout, max_retries=max_retries)
    if family == "grok":
        return GrokClient(model=model, timeout=timeout, max_retries=max_retries)
    if family == "qwen":
        return OpenRouterClient(model=model, timeout=timeout, max_retries=max_retries, return_logprobs=False)
    if family == "llama":
        return OpenRouterClient(model=model, timeout=timeout, max_retries=max_retries, return_logprobs=True)
    raise ValueError(f"Unknown family: {family}")


def _checkpoint_path(output_dir: Path, family: str, model: str) -> Path:
    return output_dir / "checkpoints" / f"task2_{family}_{safe_slug(model)}_checkpoint.json"


def _results_path(output_dir: Path, family: str, model: str) -> Path:
    return output_dir / "results" / f"task2_{family}_{safe_slug(model)}_results.json"


def _load_checkpoint(path: Path) -> tuple[set[str], list[dict]]:
    if not path.exists():
        return set(), []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    completed_ids = {row["case_id"] for row in results if row.get("case_id")}
    return completed_ids, results


def _save_checkpoint(path: Path, model: str, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model,
                "results": results,
                "completed_ids": [row["case_id"] for row in results if row.get("case_id")],
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _save_results(path: Path, family: str, model: str, results: list[dict], n_test_cases: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "task2",
                "family": family,
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_test_cases": n_test_cases,
                "n_completed": len(results),
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _make_cases(input_path: str, limit: int | None = None) -> list[dict]:
    rows = load_source_file(input_path, source_name=Path(input_path).stem)
    cases: list[dict] = []
    for idx, row in enumerate(rows):
        cases.append(
            {
                "case_id": f"t2_{idx}",
                "context": row.get("context"),
                "treatment": row.get("treatment"),
                "outcome": row.get("outcome"),
                "expected_sign": row.get("sign"),
                "paper_id": row.get("paper_id"),
            }
        )
    if limit is not None:
        return cases[:limit]
    return cases


def _call_case(client, case: dict) -> dict:
    prompt = SIGN_PREDICTION_PROMPT.format(
        context=case["context"] or "",
        treatment=case["treatment"] or "",
        outcome=case["outcome"] or "",
    )
    started = time.time()
    response = client.call_api(
        user_prompt=prompt,
        response_schema=SIGN_PREDICTION_SCHEMA,
        paper_id=case["case_id"],
    )
    latency_ms = (time.time() - started) * 1000
    payload = response.data or {}
    predicted = normalize_sign(payload.get("predicted_sign"))
    result = {
        "case_id": case["case_id"],
        "input_data": case,
        "output_data": payload,
        "expected": case["expected_sign"],
        "predicted": predicted,
        "correct": bool(predicted == case["expected_sign"]),
        "reasoning": payload.get("reasoning"),
        "error": response.error,
        "latency_ms": latency_ms,
        "avg_logprob": getattr(response, "avg_logprob", None),
        "logprobs": getattr(response, "logprobs", None),
    }
    return result


def _run_model(
    family: str,
    model: str,
    cases: list[dict],
    output_dir: Path,
    max_workers: int,
    timeout: int,
    max_retries: int,
    checkpoint_interval: int,
) -> None:
    client = _create_client(family, model, timeout=timeout, max_retries=max_retries, max_workers=max_workers)
    checkpoint_path = _checkpoint_path(output_dir, family, model)
    results_path = _results_path(output_dir, family, model)
    completed_ids, results = _load_checkpoint(checkpoint_path)
    pending = [case for case in cases if case["case_id"] not in completed_ids]

    if not pending:
        logger.info("skip %s/%s (already completed)", family, model)
        _save_results(results_path, family, model, results, n_test_cases=len(cases))
        return

    logger.info("run %s/%s pending=%s", family, model, len(pending))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_case, client, case): case["case_id"] for case in pending}
        for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"{family}:{model}"), start=1):
            results.append(future.result())
            if idx % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_path, model, results)

    results = sorted(results, key=lambda row: row["case_id"])
    _save_checkpoint(checkpoint_path, model, results)
    _save_results(results_path, family, model, results, n_test_cases=len(cases))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run future-ready Task1 sign prediction reruns over the full corpus.")
    parser.add_argument("--input", required=True, help="Input JSONL path, typically data/causal_triplets.jsonl")
    parser.add_argument("--families", nargs="+", choices=sorted(MODEL_SPECS), default=sorted(MODEL_SPECS))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "task1_rerun"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
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
    output_dir = Path(args.output_dir)
    cases = _make_cases(args.input, limit=args.limit)
    logger.info("loaded %s cases from %s", len(cases), args.input)

    for family in args.families:
        for model in MODEL_SPECS[family]["models"]:
            _run_model(
                family=family,
                model=model,
                cases=cases,
                output_dir=output_dir,
                max_workers=args.max_workers,
                timeout=args.timeout,
                max_retries=args.max_retries,
                checkpoint_interval=args.checkpoint_interval,
            )

    print(
        json.dumps(
            {
                "input": args.input,
                "n_cases": len(cases),
                "families": args.families,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
