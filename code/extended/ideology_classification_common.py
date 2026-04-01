"""Shared helpers for ideology classification of causal triplets."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from econ_eval.common.utils import (
    ClaudeClient,
    GeminiClient,
    GrokClient,
    OpenAIClient,
    OpenRouterClient,
    QwenClient,
    GEMINI_API_KEY_ECON,
    GEMINI_API_KEY_FINANCE,
    GEMINI_API_KEY_PLUS,
)

DATA_DIR = ARTIFACT_ROOT / "data_derived"
EXTENDED_DIR = CODE_ROOT / "extended"
RESULTS_DIR = ARTIFACT_ROOT / "outputs" / "classification"
PROMPT_PATH = EXTENDED_DIR / "classification_prompt_ideology.py"
DEFAULT_INPUT_PATH = DATA_DIR / "task1_ideology_subset_1056.jsonl"

VALID_SIGNS = {"+", "-", "None", "Mixed"}
DEFAULT_MODEL_KEYS = [
    "gpt-5-mini",
    "gemini-3-flash",
    "claude-sonnet-4-6",
    "qwen-3-32b",
]

MODEL_SPECS = {
    "gpt-5-mini": {
        "client_cls": OpenAIClient,
        "kwargs": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": "gpt-5-mini",
        },
    },
    "gemini-3-flash": {
        "client_cls": GeminiClient,
        "kwargs": {
            "api_key": os.environ.get("GEMINI_API_KEY_FINANCE")
            or GEMINI_API_KEY_FINANCE,
            "api_keys": [
                key
                for key in [
                    os.environ.get("GEMINI_API_KEY_ECON"),
                    GEMINI_API_KEY_ECON,
                    os.environ.get("GEMINI_API_KEY_PLUS"),
                    GEMINI_API_KEY_PLUS,
                    os.environ.get("GEMINI_API_KEY"),
                ]
                if key
            ],
            "model": "gemini-3-flash-preview",
        },
    },
    "claude-sonnet-4-6": {
        "client_cls": ClaudeClient,
        "kwargs": {
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "model": "claude-sonnet-4-6",
        },
    },
    "grok-4-1-fast-reasoning": {
        "client_cls": GrokClient,
        "kwargs": {
            "api_key": os.environ.get("GROK_API_KEY"),
            "model": "grok-4-1-fast-reasoning",
        },
    },
    "qwen-3.5-122b-a10b": {
        "client_cls": QwenClient,
        "kwargs": {
            "api_key": os.environ.get("OPENROUTER_API_KEY") or os.environ.get("QWEN_API_KEY"),
            "model": "qwen/qwen3.5-122b-a10b",
        },
    },
    "qwen-3-32b": {
        "client_cls": QwenClient,
        "kwargs": {
            "api_key": os.environ.get("OPENROUTER_API_KEY") or os.environ.get("QWEN_API_KEY"),
            "model": "qwen/qwen3-32b",
        },
    },
    "llama-3.3-70b": {
        "client_cls": OpenRouterClient,
        "kwargs": {
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "model": "meta-llama/llama-3.3-70b-instruct",
            "return_logprobs": False,
        },
    },
}

IDEOLOGY_RESPONSE_SCHEMA = {
    "name": "ideology_triplet_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ideology_preference": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_ideologically_sensitive": {"type": "boolean"},
                    "economic_liberal_expected_sign": {
                        "type": "string",
                        "enum": ["+", "-", "None", "Mixed", "null"],
                    },
                    "economic_conservative_expected_sign": {
                        "type": "string",
                        "enum": ["+", "-", "None", "Mixed", "null"],
                    },
                },
                "required": [
                    "is_ideologically_sensitive",
                    "economic_liberal_expected_sign",
                    "economic_conservative_expected_sign",
                ],
            },
            "reasoning": {"type": "string"},
        },
        "required": ["ideology_preference", "reasoning"],
    },
}


def load_prompt(prompt_path: Path | None = None) -> str:
    """Load a prompt template stored as a module docstring."""
    prompt_path = prompt_path or PROMPT_PATH
    text = Path(prompt_path).read_text(encoding="utf-8").strip()
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    return text


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into memory."""
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str | Path) -> None:
    """Write a list of objects to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_triplet_key(item: dict) -> str:
    """Create a stable triplet key."""
    return f"{item.get('paper_id', '')}|{item.get('treatment', '')}|{item.get('outcome', '')}"


def load_done_keys(path: str | Path) -> set[str]:
    """Load processed triplet keys from an existing JSONL output."""
    done: set[str] = set()
    output_path = Path(path)
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(make_triplet_key(row))
    return done


def render_prompt(prompt_template: str, item: dict) -> str:
    """Fill the ideology prompt template for a single triplet."""
    prompt = prompt_template
    prompt = prompt.replace("{title}", str(item.get("title", "")))
    prompt = prompt.replace("{context}", str(item.get("context", "")))
    prompt = prompt.replace("{treatment}", str(item.get("treatment", "")))
    prompt = prompt.replace("{outcome}", str(item.get("outcome", "")))
    return prompt


def normalize_sign(value: Any) -> str | None:
    """Normalize sign labels to the canonical set."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"+", "plus", "positive"}:
        return "+"
    if lowered in {"-", "minus", "negative"}:
        return "-"
    if lowered in {"none", "null", "no effect", "no significant effect"}:
        return "None"
    if lowered in {"mixed", "context-dependent", "context dependent"}:
        return "Mixed"
    return text if text in VALID_SIGNS else None


def normalize_bool(value: Any) -> bool:
    """Normalize common boolean-like values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().lower()
    return lowered in {"true", "1", "yes", "y"}


def normalize_reasoning(text: Any, max_words: int = 100) -> str:
    """Normalize reasoning text and hard-cap it to the first max_words words."""
    if text is None:
        return ""
    cleaned = " ".join(str(text).split())
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words])


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    if not text:
        return 0
    return len(text.split())


def classify_ground_truth_side(
    observed_sign: Any,
    liberal_sign: Any,
    conservative_sign: Any,
) -> str:
    """Map observed sign to ideology alignment using the repo's existing rule."""
    observed = normalize_sign(observed_sign)
    liberal = normalize_sign(liberal_sign)
    conservative = normalize_sign(conservative_sign)
    lib_match = observed is not None and observed == liberal
    cons_match = observed is not None and observed == conservative
    if lib_match and cons_match:
        return "both"
    if lib_match:
        return "liberal"
    if cons_match:
        return "conservative"
    return "unlabeled"


def signature_to_string(signature: tuple[bool, str | None, str | None]) -> str:
    """Render a signature tuple as a stable string."""
    sensitive, liberal_sign, conservative_sign = signature
    if not sensitive:
        return "non_sensitive"
    return (
        "sensitive"
        f"|lib={liberal_sign or 'null'}"
        f"|con={conservative_sign or 'null'}"
    )


def normalize_model_payload(payload: dict) -> dict:
    """Normalize a model response payload to a stable schema."""
    ideology_preference = payload.get("ideology_preference", {})
    if not isinstance(ideology_preference, dict):
        ideology_preference = {}

    sensitive = normalize_bool(ideology_preference.get("is_ideologically_sensitive"))
    liberal_sign = normalize_sign(ideology_preference.get("economic_liberal_expected_sign"))
    conservative_sign = normalize_sign(ideology_preference.get("economic_conservative_expected_sign"))
    if not sensitive:
        liberal_sign = None
        conservative_sign = None

    reasoning = payload.get("reasoning")
    if reasoning is None:
        evidence = payload.get("evidence", {})
        if isinstance(evidence, dict):
            ideology_evidence = evidence.get("ideology")
            if isinstance(ideology_evidence, list) and ideology_evidence:
                reasoning = ideology_evidence[0]

    normalized_reasoning = normalize_reasoning(reasoning)
    return {
        "is_ideologically_sensitive": sensitive,
        "economic_liberal_expected_sign": liberal_sign,
        "economic_conservative_expected_sign": conservative_sign,
        "reasoning": normalized_reasoning,
        "reasoning_word_count": word_count(normalized_reasoning),
        "signature": signature_to_string((sensitive, liberal_sign, conservative_sign)),
    }


def build_clients(
    model_keys: list[str] | None = None,
    timeout: int = 300,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Instantiate model clients for the requested model keys."""
    model_keys = model_keys or list(DEFAULT_MODEL_KEYS)
    clients: dict[str, Any] = {}
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        kwargs = dict(spec["kwargs"])
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = max_retries
        clients[model_key] = spec["client_cls"](**kwargs)
    return clients


def call_model(
    client: Any,
    model_key: str,
    prompt: str,
    paper_id: str | None,
) -> dict:
    """Call one model and normalize the response."""
    try:
        response = client.call_api(
            user_prompt=prompt,
            response_schema=IDEOLOGY_RESPONSE_SCHEMA,
            paper_id=paper_id,
        )
    except Exception as exc:  # pragma: no cover - provider exceptions are runtime-only
        return {
            "model": model_key,
            "error": str(exc),
        }

    if not response.success or not isinstance(response.data, dict):
        return {
            "model": model_key,
            "error": response.error or "Unknown API failure",
        }

    normalized = normalize_model_payload(response.data)
    normalized["model"] = model_key
    normalized["raw_response"] = response.data
    return normalized


def resolve_consensus(item: dict, model_outputs: dict[str, dict]) -> dict:
    """Aggregate per-model ideology outputs into a consensus summary."""
    successful = {
        model_key: output
        for model_key, output in model_outputs.items()
        if "error" not in output
    }
    failed = {
        model_key: output["error"]
        for model_key, output in model_outputs.items()
        if "error" in output
    }

    signature_counts: Counter[tuple[bool, str | None, str | None]] = Counter()
    sensitive_signature_counts: Counter[tuple[bool, str | None, str | None]] = Counter()
    signature_models: dict[tuple[bool, str | None, str | None], list[str]] = defaultdict(list)

    for model_key, output in successful.items():
        signature = (
            output["is_ideologically_sensitive"],
            output["economic_liberal_expected_sign"],
            output["economic_conservative_expected_sign"],
        )
        signature_counts[signature] += 1
        signature_models[signature].append(model_key)
        if output["is_ideologically_sensitive"]:
            sensitive_signature_counts[signature] += 1

    sensitivity_support_count = sum(
        1
        for output in successful.values()
        if output["is_ideologically_sensitive"]
    )
    meets_sensitive_consensus = sensitivity_support_count >= 3

    def _best_signature(
        counts: Counter[tuple[bool, str | None, str | None]],
    ) -> tuple[tuple[bool, str | None, str | None], int] | tuple[None, int]:
        if not counts:
            return None, 0
        best_count = max(counts.values())
        best_signatures = [
            signature
            for signature, count in counts.items()
            if count == best_count
        ]
        best_signatures.sort(key=signature_to_string)
        return best_signatures[0], best_count

    consensus_signature, max_signature_agreement = _best_signature(signature_counts)
    if meets_sensitive_consensus:
        winning_signature, winning_count = _best_signature(sensitive_signature_counts)
    else:
        winning_signature, winning_count = consensus_signature, max_signature_agreement

    liberal_sign = None
    conservative_sign = None
    consensus_models: list[str] = []
    consensus_reasoning = ""
    ground_truth_side = "non_sensitive"
    division_label = "non_divided"

    if winning_signature is not None:
        _, liberal_sign, conservative_sign = winning_signature
        consensus_models = list(signature_models[winning_signature])
        for model_key in consensus_models:
            reasoning = successful[model_key].get("reasoning", "")
            if reasoning:
                consensus_reasoning = reasoning
                break

        if meets_sensitive_consensus:
            ground_truth_side = classify_ground_truth_side(
                item.get("sign"),
                liberal_sign,
                conservative_sign,
            )
            if liberal_sign != conservative_sign:
                division_label = "ideology_divided"

    return {
        "models_requested": list(model_outputs.keys()),
        "models_succeeded": list(successful.keys()),
        "models_failed": failed,
        "successful_model_count": len(successful),
        "failed_model_count": len(failed),
        "sensitivity_support_count": sensitivity_support_count,
        "meets_sensitive_consensus": meets_sensitive_consensus,
        "max_signature_agreement": max_signature_agreement,
        "consensus_signature_agreement": winning_count,
        "consensus_signature": signature_to_string(winning_signature) if winning_signature else None,
        "signature_vote_counts": {
            signature_to_string(signature): count
            for signature, count in sorted(signature_counts.items(), key=lambda entry: signature_to_string(entry[0]))
        },
        "sensitive_signature_vote_counts": {
            signature_to_string(signature): count
            for signature, count in sorted(sensitive_signature_counts.items(), key=lambda entry: signature_to_string(entry[0]))
        },
        "consensus_supporting_models": consensus_models,
        "consensus_reasoning": consensus_reasoning,
        "consensus_reasoning_word_count": word_count(consensus_reasoning),
        "consensus_ground_truth_side": ground_truth_side,
        "division_label": division_label,
        "consensus_ideology_preference": {
            "is_ideologically_sensitive": meets_sensitive_consensus,
            "economic_liberal_expected_sign": liberal_sign if meets_sensitive_consensus else None,
            "economic_conservative_expected_sign": conservative_sign if meets_sensitive_consensus else None,
        },
    }
