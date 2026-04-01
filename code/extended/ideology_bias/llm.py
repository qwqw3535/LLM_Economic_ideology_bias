"""Lightweight OpenAI JSON-schema helpers used by annotation scripts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMCallResult:
    """Structured response from a JSON-schema LLM call."""

    success: bool
    payload: dict | None
    error: str | None = None
    raw_content: str | None = None


class OpenAIJsonClient:
    """Minimal JSON-schema wrapper over the OpenAI chat completions API."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"{api_key_env} is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def call_json_schema(
        self,
        prompt: str,
        schema: dict,
        system_prompt: str = "You are a helpful assistant that strictly returns JSON.",
    ) -> LLMCallResult:
        """Call the model and parse a strict JSON-schema response."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema["name"],
                            "strict": True,
                            "schema": schema["schema"],
                        },
                    },
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content
                return LLMCallResult(success=True, payload=json.loads(content), raw_content=content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        return LLMCallResult(success=False, payload=None, error=str(last_error), raw_content=None)

