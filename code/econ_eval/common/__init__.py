"""Common utilities for the causal relation extraction pipeline.

Uses lazy imports to avoid loading heavy dependencies when not needed.
"""

# Lazy imports - only load when accessed
def __getattr__(name):
    """Lazy import mechanism."""

    # Utils imports (heavy - requires pandas, openai)
    utils_exports = [
        "OpenAIClient", "GeminiClient", "GrokClient", "OpenRouterClient", "ClaudeClient",
        "HFEndpointClient", "HFRouterClient",
        "APIResponse", "load_json", "save_json", "load_csv",
        "get_pdf_files", "setup_logging", "HuggingFaceClient",
    ]

    # Prompts imports
    prompt_exports = [
        "STEP1_PROMPT", "STEP2_PROMPT", "STEP3_PROMPT", "STEP4_PROMPT",
        "EVAL_TASK1_PROMPT", "EVAL_TASK2_PROMPT", "EVAL_TASK2_PROMPT_NO_CONTEXT",
        "EVAL_TASK2_EXAMPLE_PROMPT", "EVAL_TASK2_EXAMPLE_LABEL_PROMPT",
        "EVAL_TASK3_NOISY_EXAMPLE_PROMPT", "EVAL_TASK3_NOISY_EXAMPLE_LABEL_PROMPT",
        "EVAL_TASK3_PROMPT", "EVAL_TASK4_PROMPT",
    ]

    # Schema imports
    schema_exports = [
        "STEP1_SCHEMA", "STEP2_SCHEMA", "STEP3_SCHEMA", "STEP4_SCHEMA",
        "EVAL_TASK1_SCHEMA", "EVAL_TASK2_SCHEMA",
        "EVAL_TASK2_EXAMPLE_SCHEMA", "EVAL_TASK3_NOISY_EXAMPLE_SCHEMA",
        "EVAL_TASK3_SCHEMA", "EVAL_TASK4_SCHEMA",
    ]

    if name in utils_exports:
        from . import utils
        return getattr(utils, name)

    if name in prompt_exports:
        from . import prompts
        return getattr(prompts, name)

    if name in schema_exports:
        from . import schemas
        return getattr(schemas, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Utils
    "OpenAIClient",
    "GeminiClient",
    "GrokClient",
    "OpenRouterClient",
    "ClaudeClient",
    "HFEndpointClient",
    "HFRouterClient",
    "APIResponse",
    "load_json",
    "save_json",
    "load_csv",
    "get_pdf_files",
    "setup_logging",
    # Prompts
    "STEP1_PROMPT",
    "STEP2_PROMPT",
    "STEP3_PROMPT",
    "STEP4_PROMPT",
    "EVAL_TASK1_PROMPT",
    "EVAL_TASK2_PROMPT",
    "EVAL_TASK2_PROMPT_NO_CONTEXT",
    "EVAL_TASK2_EXAMPLE_PROMPT",
    "EVAL_TASK2_EXAMPLE_LABEL_PROMPT",
    "EVAL_TASK3_NOISY_EXAMPLE_PROMPT",
    "EVAL_TASK3_NOISY_EXAMPLE_LABEL_PROMPT",
    "EVAL_TASK3_PROMPT",
    "EVAL_TASK4_PROMPT",
    # Schemas
    "STEP1_SCHEMA",
    "STEP2_SCHEMA",
    "STEP3_SCHEMA",
    "STEP4_SCHEMA",
    "EVAL_TASK1_SCHEMA",
    "EVAL_TASK2_SCHEMA",
    "EVAL_TASK2_EXAMPLE_SCHEMA",
    "EVAL_TASK3_NOISY_EXAMPLE_SCHEMA",
    "EVAL_TASK3_SCHEMA",
    "EVAL_TASK4_SCHEMA",
]
