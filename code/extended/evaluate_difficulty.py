"""Evaluate difficulty of ideology-sensitive causal triplets using GPT-5-mini.

Reads the merged classification file, filters to ideology-sensitive triplets,
calls the OpenAI API with TRIPLET_DIFFICULTY_PROMPT, and saves results.

Usage:
    python extended/evaluate_difficulty.py \
        --input extended/classification_results/causal_triplets_gpt-5-mini_classified_merged.jsonl \
        --output extended/classification_results/difficulty_scores.jsonl \
        --model gpt-5-mini \
        --max-parallel 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt (extracted from econ_eval/common/prompts.py for self-containedness)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert economist. Given an economic causal triplet "
    "(treatment → outcome with a stated sign), rate how difficult it is to "
    "correctly identify the causal sign. Use these 5 dimensions (each 1–5) "
    "plus an overall score.\n\n"
    "1. Domain Knowledge (1–5): 1 = common sense sufficient, 3 = intermediate "
    "policy/market knowledge needed, 5 = requires specialized theory or "
    "identification strategies.\n\n"
    "2. Context Dependence (1–5): 1 = sign is stable regardless of setting, "
    "3 = context matters moderately, 5 = sign is entirely pinned down by "
    "specific institutional/temporal/population conditions.\n\n"
    "3. Ambiguity / Confoundability (1–5): 1 = sign is unambiguous, "
    "3 = moderately plausible alternative, 5 = opposite sign nearly as "
    "plausible without empirical analysis.\n\n"
    "4. Causal Reasoning Complexity (1–5): 1 = direct one-step link, "
    "3 = multiple mediating channels, 5 = requires reasoning through GE "
    "feedbacks, selection, and composition effects simultaneously.\n\n"
    "5. Evidence Sufficiency (1–5): 1 = context clearly determines the sign, "
    "3 = reasonable inference possible despite gaps, 5 = information so "
    "incomplete that none/mixed is a serious contender.\n\n"
    "6. Overall Difficulty (1–5): Your holistic judgment of how hard this "
    "triplet is, considering all dimensions above.\n\n"
    "Evaluate PURELY based on economic reasoning. Do NOT let political "
    "implications influence your ratings.\n\n"
    'Respond in JSON only, no other text:\n'
    '{"domain_knowledge":<int>,"context_dependence":<int>,"ambiguity":<int>,'
    '"causal_complexity":<int>,"evidence_sufficiency":<int>,'
    '"overall_difficulty":<int>,"justification":"<1-2 sentences>"}'
)

USER_PROMPT_TEMPLATE = (
    "Rate the difficulty of this economic causal triplet.\n\n"
    "Treatment: {treatment}\n"
    "Outcome: {outcome}\n"
    "Sign: {sign}\n"
    "Context: {context}"
)

# ---------------------------------------------------------------------------
# JSON schema for structured output
# ---------------------------------------------------------------------------

DIFFICULTY_SCHEMA = {
    "name": "difficulty_evaluation",
    "schema": {
        "type": "object",
        "properties": {
            "domain_knowledge": {"type": "integer"},
            "context_dependence": {"type": "integer"},
            "ambiguity": {"type": "integer"},
            "causal_complexity": {"type": "integer"},
            "evidence_sufficiency": {"type": "integer"},
            "overall_difficulty": {"type": "integer"},
            "justification": {"type": "string"},
        },
        "required": [
            "domain_knowledge",
            "context_dependence",
            "ambiguity",
            "causal_complexity",
            "evidence_sufficiency",
            "overall_difficulty",
            "justification",
        ],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------


async def evaluate_one(
    client,
    model: str,
    triplet: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    """Evaluate difficulty for a single triplet."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        treatment=triplet["treatment"],
        outcome=triplet["outcome"],
        sign=triplet["sign"],
        context=triplet.get("context", ""),
    )

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": DIFFICULTY_SCHEMA["name"],
                            "strict": True,
                            "schema": DIFFICULTY_SCHEMA["schema"],
                        },
                    },
                    timeout=120,
                )
                content = response.choices[0].message.content
                payload = json.loads(content)
                return {
                    "triplet_key": triplet["triplet_key"],
                    "paper_id": triplet.get("paper_id"),
                    "treatment": triplet["treatment"],
                    "outcome": triplet["outcome"],
                    "sign": triplet["sign"],
                    "context": triplet.get("context", ""),
                    "ground_truth_side": triplet.get("ground_truth_side"),
                    "difficulty": payload,
                    "overall_difficulty": payload["overall_difficulty"],
                    "error": None,
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        return {
            "triplet_key": triplet["triplet_key"],
            "paper_id": triplet.get("paper_id"),
            "treatment": triplet["treatment"],
            "outcome": triplet["outcome"],
            "sign": triplet["sign"],
            "context": triplet.get("context", ""),
            "ground_truth_side": triplet.get("ground_truth_side"),
            "difficulty": None,
            "overall_difficulty": None,
            "error": last_error,
        }


async def evaluate_all(
    triplets: list[dict],
    model: str,
    max_parallel: int,
    max_retries: int,
    output_path: Path,
) -> list[dict]:
    """Evaluate all triplets with progress tracking and incremental saves."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    # Load already-completed results for resume support
    done_keys: set[str] = set()
    results: list[dict] = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") is None:
                    done_keys.add(row["triplet_key"])
                    results.append(row)
        print(f"Resuming: {len(done_keys)} triplets already completed.")

    remaining = [t for t in triplets if t["triplet_key"] not in done_keys]
    if not remaining:
        print("All triplets already evaluated.")
        return results

    print(f"Evaluating {len(remaining)} triplets (max_parallel={max_parallel})...")

    from tqdm import tqdm

    semaphore = asyncio.Semaphore(max_parallel)
    tasks = [
        evaluate_one(client, model, t, semaphore, max_retries)
        for t in remaining
    ]

    total = len(tasks)
    new_results: list[dict] = []
    n_errors = 0

    # Open file in append mode for incremental saves
    with open(output_path, "a", encoding="utf-8") as f:
        pbar = tqdm(total=total, desc="Difficulty eval", unit="triplet")
        for coro in asyncio.as_completed(tasks):
            result = await coro
            new_results.append(result)
            if result["error"] is not None:
                n_errors += 1
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            pbar.set_postfix(errors=n_errors)
            pbar.update(1)
        pbar.close()

    all_results = results + new_results
    n_errors = sum(1 for r in all_results if r["error"] is not None)
    print(f"Completed: {len(all_results)} total, {n_errors} errors.")
    return all_results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_text(value: object) -> str:
    """Normalize text for deterministic matching (matches ideology_bias/utils.py)."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _make_triplet_key(paper_id: object, treatment: object, outcome: object) -> str:
    """Create a stable triplet key (matches ideology_bias/utils.py)."""
    return "|".join([str(paper_id).strip(), _normalize_text(treatment), _normalize_text(outcome)])


def load_ideology_sensitive_triplets(input_path: str) -> list[dict]:
    """Load ideology-sensitive triplets from the merged classification JSONL."""
    triplets = []
    seen_keys = set()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cls = row.get("classification", {}).get("labels", {})
            ideology_pref = cls.get("ideology_preference", {})
            if not ideology_pref.get("is_ideologically_sensitive"):
                continue

            triplet_key = _make_triplet_key(row["paper_id"], row["treatment"], row["outcome"])
            if triplet_key in seen_keys:
                continue
            seen_keys.add(triplet_key)

            # Determine ground_truth_side
            sign = row.get("sign")
            lib_sign = ideology_pref.get("economic_liberal_expected_sign")
            cons_sign = ideology_pref.get("economic_conservative_expected_sign")
            if sign and sign == lib_sign and sign != cons_sign:
                gt_side = "liberal"
            elif sign and sign == cons_sign and sign != lib_sign:
                gt_side = "conservative"
            elif sign and sign == lib_sign and sign == cons_sign:
                gt_side = "both"
            else:
                gt_side = "unlabeled"

            triplets.append({
                "triplet_key": triplet_key,
                "paper_id": row.get("paper_id"),
                "treatment": row["treatment"],
                "outcome": row["outcome"],
                "sign": row["sign"],
                "context": row.get("context", ""),
                "ground_truth_side": gt_side,
                "economic_liberal_expected_sign": lib_sign,
                "economic_conservative_expected_sign": cons_sign,
            })
    return triplets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate difficulty of ideology-sensitive causal triplets."
    )
    parser.add_argument(
        "--input",
        default="extended/classification_results/causal_triplets_gpt-5-mini_classified_merged.jsonl",
        help="Input merged classification JSONL.",
    )
    parser.add_argument(
        "--output",
        default="extended/classification_results/difficulty_scores.jsonl",
        help="Output difficulty scores JSONL.",
    )
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model name.")
    parser.add_argument("--max-parallel", type=int, default=512, help="Max concurrent API calls.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per triplet.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of triplets (for testing).")
    args = parser.parse_args()

    triplets = load_ideology_sensitive_triplets(args.input)
    print(f"Loaded {len(triplets)} ideology-sensitive triplets.")

    if args.limit:
        triplets = triplets[: args.limit]
        print(f"Limited to {len(triplets)} triplets.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        evaluate_all(
            triplets,
            model=args.model,
            max_parallel=args.max_parallel,
            max_retries=args.max_retries,
            output_path=output_path,
        )
    )

    # Write a clean consolidated version (no duplicates, no errors)
    clean_path = output_path.with_name(output_path.stem + "_clean.jsonl")
    seen = set()
    clean_rows = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["error"] is None and row["triplet_key"] not in seen:
                seen.add(row["triplet_key"])
                clean_rows.append(row)
    with open(clean_path, "w", encoding="utf-8") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Clean output: {clean_path} ({len(clean_rows)} rows)")


if __name__ == "__main__":
    main()
