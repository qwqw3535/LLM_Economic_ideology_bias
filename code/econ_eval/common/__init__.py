"""Common utilities for the released evaluation pipeline.

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

    # Schema imports
    schema_exports = ["EVAL_TASK2_SCHEMA", "EVAL_TASK2_EXAMPLE_SCHEMA"]

    if name in utils_exports:
        from . import utils
        return getattr(utils, name)

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
    # Schemas
    "EVAL_TASK2_SCHEMA",
    "EVAL_TASK2_EXAMPLE_SCHEMA",
]
