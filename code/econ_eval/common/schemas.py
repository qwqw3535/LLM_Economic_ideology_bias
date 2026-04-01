"""Structured-output schemas used by the released artifact."""

EVAL_TASK2_SCHEMA = {
    "name": "sign_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": ["+", "-", "None", "mixed"],
                "description": "Predicted direction of causal effect",
            },
            "reasoning": {
                "type": "string",
                "description": "Economic reasoning for the predicted sign",
            },
        },
        "required": ["predicted_sign", "reasoning"],
        "additionalProperties": False,
    },
}


EVAL_TASK2_EXAMPLE_SCHEMA = {
    "name": "icl_experiment",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": ["+", "-", "None", "mixed"],
                "description": "Predicted sign for the target context",
            },
            "context_analysis": {
                "type": "string",
                "description": "How the target context differs from the examples",
            },
            "reasoning": {
                "type": "string",
                "description": "Why the predicted sign follows from the matched examples and target context",
            },
        },
        "required": ["predicted_sign", "context_analysis", "reasoning"],
        "additionalProperties": False,
    },
}
