from __future__ import annotations

"""Configuration for the evaluation pipeline."""

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen


# Default paths for the standalone anonymous artifact.
ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"

DEFAULT_DATA_PATH = str(ARTIFACT_ROOT / "main_results" / "input" / "ideology_sensitive_subset_1056.jsonl")
DEFAULT_OUTPUT_DIR = str(OUTPUT_ROOT)
DEFAULT_ICL_SOURCE_PATH = str(ARTIFACT_ROOT / "icl_experiment" / "input" / "jel_similarity_shared2.jsonl.gz")

VERIFIED_HF_ENDPOINT_MODELS = [
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-35B-A3B-Base",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
]

DEDICATED_ENDPOINT_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-70B",
]

HF_ROUTER_MODELS = [
    "Qwen/Qwen3-14B-Base:fastest",
    "Qwen/Qwen3-8B-Base:fastest",
    "meta-llama/Llama-3.1-8B:fastest",
    "meta-llama/Llama-3.1-70B-Instruct:fastest",
]

# Supported models and their configurations
SUPPORTED_MODELS = {
    # "openai": {
    #     "client": "OpenAIClient",
    #     "default_model": "gpt-4o",
    #     "models": [ "gpt-4o"],
    #     "supports_logprobs": True,
    # },
    "openai": {
        "client": "OpenAIClient",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.2"],
        "supports_logprobs": True,
    },
    "gemini": {
        "client": "GeminiClient",
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.5-flash", "gemini-3-flash-preview"  ],
        # "models": ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-pro",  ],
        "supports_logprobs": False,
    },
    "grok": {
        "client": "GrokClient",
        "default_model": "grok-3-mini",
        "models": ["grok-3-mini", "grok-3","grok-4-1-fast-reasoning",  ],
        "supports_logprobs": False,
    },
    "claude": {
        "client": "ClaudeClient",
        "default_model": "claude-haiku-4-5",
        "models": [
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ],
        "supports_logprobs": False,
    },
    "qwen": {
        "client": "OpenRouterClient",
        "default_model": "qwen/qwen3-8b",
        "models": ["qwen/qwen3-8b", "qwen/qwen3-14b", "qwen/qwen3-32b"],
        "supports_logprobs": True,
    },
    "llama": {
        "client": "OpenRouterClient",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "models": ["meta-llama/llama-3.2-1b-instruct", "meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.3-70b-instruct"],
        "supports_logprobs": True,
    },
    "hf_endpoint": {
        "client": "HFEndpointClient",
        "default_model": VERIFIED_HF_ENDPOINT_MODELS[0],
        "models": VERIFIED_HF_ENDPOINT_MODELS.copy(),
        "supports_logprobs": False,
    },
    "hf_router": {
        "client": "HFRouterClient",
        "default_model": HF_ROUTER_MODELS[0],
        "models": HF_ROUTER_MODELS.copy(),
        "supports_logprobs": False,
    },
    # "deepseek": {
    #     "client": "OpenRouterClient",
    #     "default_model": "deepseek/deepseek-r1",
    #     "models": ["deepseek/deepseek-r1", "deepseek/deepseek-chat"],
    #     "supports_logprobs": True,
    # },
}

# Base/instruct sibling pairs. We only auto-add a sibling when the provider
# catalog confirms that exact model ID is actually callable.
MODEL_VARIANT_SIBLINGS = {
    "qwen/qwen3-8b": ["qwen/qwen3-8b-base"],
    "qwen/qwen3-8b-base": ["qwen/qwen3-8b"],
    "qwen/qwen3-14b": ["qwen/qwen3-14b-base"],
    "qwen/qwen3-14b-base": ["qwen/qwen3-14b"],
    "qwen/qwen3-32b": ["qwen/qwen3-32b-base"],
    "qwen/qwen3-32b-base": ["qwen/qwen3-32b"],
    "meta-llama/llama-3.2-1b-instruct": ["meta-llama/llama-3.2-1b"],
    "meta-llama/llama-3.2-1b": ["meta-llama/llama-3.2-1b-instruct"],
    "meta-llama/llama-3.2-3b-instruct": ["meta-llama/llama-3.2-3b"],
    "meta-llama/llama-3.2-3b": ["meta-llama/llama-3.2-3b-instruct"],
    "meta-llama/llama-3.1-8b-instruct": ["meta-llama/llama-3.1-8b"],
    "meta-llama/llama-3.1-8b": ["meta-llama/llama-3.1-8b-instruct"],
    "meta-llama/llama-3.3-70b-instruct": ["meta-llama/llama-3.3-70b"],
    "meta-llama/llama-3.3-70b": ["meta-llama/llama-3.3-70b-instruct"],
    "meta-llama/llama-3.1-405b-instruct": ["meta-llama/llama-3.1-405b"],
    "meta-llama/llama-3.1-405b": ["meta-llama/llama-3.1-405b-instruct"],
}

TASK_TYPES = ["main_results", "icl_experiment"]

# Models with rate limit restrictions (max_workers capped at 20)
RATE_LIMITED_MODELS = [
    "gemini-3-pro-preview"
]
RATE_LIMITED_MAX_WORKERS = 10

# OpenRouter-backed families are more fragile under high concurrency.
OPENROUTER_FAMILIES = ["qwen", "llama", "deepseek"]
OPENROUTER_MAX_WORKERS = 128
HF_ENDPOINT_MAX_WORKERS = 512


@lru_cache(maxsize=1)
def get_openrouter_model_catalog() -> set[str]:
    """Return the currently callable OpenRouter model IDs."""
    try:
        with urlopen("https://openrouter.ai/api/v1/models", timeout=10) as response:
            payload = json.load(response)
        return {model["id"] for model in payload.get("data", []) if model.get("id")}
    except (URLError, TimeoutError, ValueError, KeyError):
        return set()


def expand_family_models(family: str, base_models: list[str], include_variant_pairs: bool = True) -> list[str]:
    """Add provider-available base/instruct siblings for the configured models."""
    resolved_models = list(base_models)
    if not include_variant_pairs or family not in OPENROUTER_FAMILIES:
        return resolved_models

    available_ids = get_openrouter_model_catalog()
    if not available_ids:
        return resolved_models

    seen = set(resolved_models)
    for model_name in list(resolved_models):
        for sibling in MODEL_VARIANT_SIBLINGS.get(model_name, []):
            if sibling in available_ids and sibling not in seen:
                resolved_models.append(sibling)
                seen.add(sibling)

    return resolved_models


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    # Task configuration
    task_types: list[str] = field(default_factory=lambda: TASK_TYPES.copy())

    # Model configuration
    models: list[str] = field(default_factory=lambda: ["openai", "gemini", "grok", "claude", "qwen", "llama"])
    model_names: Optional[dict[str, str]] = None  # Override default model names
    include_variant_pairs: bool = True
    hf_model_ids: Optional[list[str]] = None
    hf_endpoint_url: Optional[str] = None
    hf_api_token: Optional[str] = None
    hf_router_base_url: str = "https://router.huggingface.co/v1"

    # Data configuration
    data_path: str = DEFAULT_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    journal_type: str = "econ"  # "econ" or "finance" - used for API key selection
    icl_source_path: str = DEFAULT_ICL_SOURCE_PATH

    # Sampling configuration
    max_samples_per_task: Optional[int] = None
    random_seed: int = 42

    # Task-specific configuration
    icl_num_examples: int = 1
    main_results_no_context: bool = False
    label_only: bool = False

    # Processing configuration
    max_workers: int = 128
    checkpoint_interval: int = 10
    timeout: int = 300
    max_retries: int = 3
    hf_max_new_tokens: int = 8
    hf_temperature: float = 0.0
    hf_top_p: float = 1.0
    hf_stop_sequences: list[str] = field(default_factory=lambda: ["\n"])

    def __post_init__(self):
        """Validate configuration."""
        # Validate task types
        for task_type in self.task_types:
            if task_type not in TASK_TYPES:
                raise ValueError(f"Unknown task type: {task_type}. Must be one of {TASK_TYPES}")

        # Validate models
        for model in self.models:
            if model not in SUPPORTED_MODELS:
                raise ValueError(f"Unknown model: {model}. Must be one of {list(SUPPORTED_MODELS.keys())}")

        if self.hf_model_ids:
            allowed_hf_models = set(VERIFIED_HF_ENDPOINT_MODELS) | set(HF_ROUTER_MODELS) | set(DEDICATED_ENDPOINT_MODELS)
            unknown_hf_models = [model_id for model_id in self.hf_model_ids if model_id not in allowed_hf_models]
            if unknown_hf_models:
                raise ValueError(
                    f"Unsupported HF model(s): {unknown_hf_models}. "
                    f"Must be one of {sorted(allowed_hf_models)}"
                )

        # Validate num_examples
        if self.icl_num_examples < 1:
            raise ValueError(f"icl_num_examples must be >= 1, got {self.icl_num_examples}")
        if self.hf_max_new_tokens < 1:
            raise ValueError(f"hf_max_new_tokens must be >= 1, got {self.hf_max_new_tokens}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_model_name(self, provider: str) -> str:
        """Get the model name for a provider."""
        if self.model_names and provider in self.model_names:
            return self.model_names[provider]
        return SUPPORTED_MODELS[provider]["default_model"]

    def get_family_models(self, family: str) -> list[str]:
        """Get the resolved model list for a family."""
        if self.model_names and family in self.model_names:
            return [self.model_names[family]]
        if family in {"hf_endpoint", "hf_router"} and self.hf_model_ids:
            return list(self.hf_model_ids)
        return expand_family_models(
            family,
            SUPPORTED_MODELS[family]["models"],
            include_variant_pairs=self.include_variant_pairs,
        )

    def get_checkpoint_path(self, task_type: str, model: str) -> Path:
        """Get checkpoint path for a task/model combination."""
        return Path(self.output_dir) / "checkpoints" / f"{task_type}_{model}_checkpoint.json"

    def get_results_path(self, task_type: str) -> Path:
        """Get results path for a task."""
        return Path(self.output_dir) / f"{task_type}_results.json"
