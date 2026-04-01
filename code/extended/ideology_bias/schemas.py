"""Shared schemas, enums, and validators for ideology bias analyses."""

from __future__ import annotations

from typing import Iterable

VALID_SIGNS = ("+", "-", "None", "mixed")
IDEOLOGY_SENSITIVITY_VALUES = ("ideology_sensitive", "non_sensitive")
POLICY_DIRECTION_VALUES = ("more_state", "less_state", "unclear")
REGION_BUCKET_VALUES = ("Developed", "China", "Developing", "Underdeveloped")
UNIT_OF_ANALYSIS_VALUES = ("personal_household", "firm", "government_country")
RACE_VALUES = ("black", "white", "asian", "hispanic")
GENDER_VALUES = ("male", "female", "non_binary")
AGE_VALUES = ("0_15", "15_64", "65_plus")
REASONING_FRAME_VALUES = (
    "efficiency",
    "incentives",
    "market_distortion",
    "productivity",
    "redistribution",
    "insurance",
    "externalities",
    "fiscal_burden",
    "state_capacity",
    "other",
)

CANONICAL_METADATA_FIELDS = (
    "ideology_sensitivity",
    "policy_direction",
    "economic_liberal_preferred_sign",
    "economic_conservative_preferred_sign",
    "region_bucket",
    "time_decade",
    "unit_of_analysis",
    "race",
    "gender",
    "age",
    "text_surface_cues",
)


def _nullable_enum_schema(values: Iterable[str]) -> dict:
    return {
        "anyOf": [
            {"type": "string", "enum": list(values)},
            {"type": "null"},
        ]
    }


def _enum_array_schema(values: Iterable[str]) -> dict:
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(values)},
    }


METADATA_RESPONSE_SCHEMA = {
    "name": "ideology_bias_metadata",
    "schema": {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {
                    "ideology_sensitivity": {
                        "type": "string",
                        "enum": list(IDEOLOGY_SENSITIVITY_VALUES),
                    },
                    "policy_direction": {
                        "type": "string",
                        "enum": list(POLICY_DIRECTION_VALUES),
                    },
                    "economic_liberal_preferred_sign": _nullable_enum_schema(VALID_SIGNS),
                    "economic_conservative_preferred_sign": _nullable_enum_schema(VALID_SIGNS),
                    "region_bucket": _enum_array_schema(REGION_BUCKET_VALUES),
                    "time_decade": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "unit_of_analysis": _enum_array_schema(UNIT_OF_ANALYSIS_VALUES),
                    "race": _enum_array_schema(RACE_VALUES),
                    "gender": _enum_array_schema(GENDER_VALUES),
                    "age": _enum_array_schema(AGE_VALUES),
                    "text_surface_cues": {
                        "type": "object",
                        "properties": {
                            "up_cues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "down_cues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["up_cues", "down_cues"],
                        "additionalProperties": False,
                    },
                },
                "required": list(CANONICAL_METADATA_FIELDS),
                "additionalProperties": False,
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "ideology_sensitivity": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "policy_direction": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "preferred_signs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "region_bucket": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "time_decade": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "unit_of_analysis": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "demographics": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "text_surface_cues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "ideology_sensitivity",
                    "policy_direction",
                    "preferred_signs",
                    "region_bucket",
                    "time_decade",
                    "unit_of_analysis",
                    "demographics",
                    "text_surface_cues",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["metadata", "evidence"],
        "additionalProperties": False,
    },
}

REASONING_FRAME_RESPONSE_SCHEMA = {
    "name": "reasoning_frames",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning_frames": {
                "type": "object",
                "properties": {
                    "primary_frame": {
                        "type": "string",
                        "enum": list(REASONING_FRAME_VALUES),
                    },
                    "secondary_frames": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(REASONING_FRAME_VALUES)},
                        "maxItems": 3,
                    },
                    "justification": {"type": "string"},
                },
                "required": ["primary_frame", "secondary_frames", "justification"],
                "additionalProperties": False,
            }
        },
        "required": ["reasoning_frames"],
        "additionalProperties": False,
    },
}


def normalize_sign(value: object) -> str | None:
    """Normalize raw sign strings to the canonical sign set."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "none":
        return "None"
    if lowered == "mixed":
        return "mixed"
    if text in VALID_SIGNS:
        return text
    mapping = {
        "positive": "+",
        "negative": "-",
        "no effect": "None",
        "null": "None",
    }
    return mapping.get(lowered)


def normalize_string_list(values: object) -> list[str]:
    """Normalize a scalar or list into a sorted, unique list of strings."""
    if values is None:
        return []
    if isinstance(values, list):
        raw_items = values
    else:
        raw_items = [values]
    items: list[str] = []
    for value in raw_items:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.lower() == "nan":
            continue
        items.append(text)
    return sorted(dict.fromkeys(items))


def validate_metadata_payload(payload: dict) -> list[str]:
    """Return a list of schema-like validation errors for a metadata payload."""
    errors: list[str] = []
    metadata = payload.get("metadata", {})
    evidence = payload.get("evidence", {})

    if metadata.get("ideology_sensitivity") not in IDEOLOGY_SENSITIVITY_VALUES:
        errors.append("invalid ideology_sensitivity")
    if metadata.get("policy_direction") not in POLICY_DIRECTION_VALUES:
        errors.append("invalid policy_direction")

    for field in ("economic_liberal_preferred_sign", "economic_conservative_preferred_sign"):
        value = metadata.get(field)
        if value is not None and normalize_sign(value) is None:
            errors.append(f"invalid {field}")

    for field, allowed in (
        ("region_bucket", REGION_BUCKET_VALUES),
        ("unit_of_analysis", UNIT_OF_ANALYSIS_VALUES),
        ("race", RACE_VALUES),
        ("gender", GENDER_VALUES),
        ("age", AGE_VALUES),
    ):
        values = normalize_string_list(metadata.get(field))
        bad = [value for value in values if value not in allowed]
        if bad:
            errors.append(f"invalid {field}: {bad}")

    cues = metadata.get("text_surface_cues", {})
    if not isinstance(cues, dict):
        errors.append("invalid text_surface_cues")
    else:
        for key in ("up_cues", "down_cues"):
            if not isinstance(cues.get(key, []), list):
                errors.append(f"invalid text_surface_cues.{key}")

    expected_evidence_fields = (
        "ideology_sensitivity",
        "policy_direction",
        "preferred_signs",
        "region_bucket",
        "time_decade",
        "unit_of_analysis",
        "demographics",
        "text_surface_cues",
    )
    for field in expected_evidence_fields:
        if field not in evidence:
            errors.append(f"missing evidence.{field}")
    return errors


def validate_reasoning_frames(payload: dict) -> list[str]:
    """Return validation errors for a reasoning frame payload."""
    errors: list[str] = []
    frames = payload.get("reasoning_frames", {})
    if frames.get("primary_frame") not in REASONING_FRAME_VALUES:
        errors.append("invalid primary_frame")
    secondary = normalize_string_list(frames.get("secondary_frames"))
    bad = [value for value in secondary if value not in REASONING_FRAME_VALUES]
    if bad:
        errors.append(f"invalid secondary_frames: {bad}")
    if len(secondary) > 3:
        errors.append("too many secondary_frames")
    return errors
