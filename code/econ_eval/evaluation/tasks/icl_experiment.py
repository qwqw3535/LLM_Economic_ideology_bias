"""ICL experiment task."""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.schemas import EVAL_TASK2_EXAMPLE_SCHEMA
from prompts.icl_experiment import ICL_EXPERIMENT_PROMPT
from .base import BaseTask
from ..data_generator import ICLExperimentCase


class ICLExperimentTask(BaseTask):
    """ICL experiment used in the paper."""

    task_name = "icl_experiment"
    task_description = "ICL Experiment: predict sign with matched in-context examples"

    VALID_SIGNS = {"+", "-", "None", "mixed"}

    def __init__(self):
        super().__init__(
            schema=EVAL_TASK2_EXAMPLE_SCHEMA,
            prompt_template=ICL_EXPERIMENT_PROMPT,
        )

    def _format_examples(self, examples: list[dict]) -> str:
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"## Example {i}")
            formatted.append(f"Treatment: {ex['treatment']}")
            formatted.append(f"Outcome: {ex['outcome']}")
            formatted.append(f"Context: {ex['context']}")
            formatted.append(f"Sign: {ex['sign']}")
            formatted.append("")
        return "\n".join(formatted)

    def format_prompt(self, test_case: ICLExperimentCase) -> str:
        return self.prompt_template.format(
            treatment=test_case.treatment,
            outcome=test_case.outcome,
            examples=self._format_examples(test_case.examples),
            context_new=test_case.test_context,
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

    def get_expected(self, test_case: ICLExperimentCase) -> str:
        return test_case.expected_sign

    def is_correct(self, predicted, expected) -> bool:
        return predicted is not None and predicted == expected
