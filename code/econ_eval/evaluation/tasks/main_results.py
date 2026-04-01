"""Main Results sign-prediction task."""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.schemas import EVAL_TASK2_SCHEMA
from prompts.main_results import MAIN_RESULTS_PROMPT
from .base import BaseTask
from ..data_generator import MainResultsCase


class MainResultsTask(BaseTask):
    """Main results evaluation used in the paper."""

    task_name = "main_results"
    task_description = "Main Results: sign prediction on ideology-sensitive items"

    VALID_SIGNS = {"+", "-", "None", "mixed"}

    def __init__(self, no_context: bool = False):
        self.no_context = no_context
        super().__init__(
            schema=EVAL_TASK2_SCHEMA,
            prompt_template=MAIN_RESULTS_PROMPT,
        )

    def format_prompt(self, test_case: MainResultsCase) -> str:
        return self.prompt_template.format(
            context=test_case.context,
            treatment=test_case.treatment,
            outcome=test_case.outcome,
        )

    def extract_prediction(self, response_data: dict) -> str:
        sign = response_data.get("predicted_sign", "").strip()
        sign_mapping = {
            "positive": "+",
            "negative": "-",
            "none": "None",
            "null": "None",
            "no effect": "None",
            "mixed": "mixed",
            "+": "+",
            "-": "-",
        }
        normalized = sign_mapping.get(sign.lower(), sign)
        return normalized if normalized in self.VALID_SIGNS else sign

    def get_expected(self, test_case: MainResultsCase) -> str:
        return test_case.expected_sign

    def is_correct(self, predicted, expected) -> bool:
        return predicted is not None and predicted == expected
