"""Normalize raw ideology metadata annotations into a canonical JSONL/CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .paths import (
    DEFAULT_BOOTSTRAP_METADATA_CSV,
    DEFAULT_BOOTSTRAP_METADATA_JSONL,
    DEFAULT_MERGED_CLASSIFICATION_PATH,
    DEFAULT_FULL_METADATA_CANONICAL_CSV,
    DEFAULT_FULL_METADATA_CANONICAL_JSONL,
    DEFAULT_FULL_METADATA_RAW_PATH,
    LEGACY_TASK1_CLASSIFICATION_PATHS,
    ensure_output_dirs,
)
from .schemas import (
    IDEOLOGY_SENSITIVITY_VALUES,
    POLICY_DIRECTION_VALUES,
    REGION_BUCKET_VALUES,
    UNIT_OF_ANALYSIS_VALUES,
    normalize_sign,
    validate_metadata_payload,
)
from .utils import iter_jsonl, make_triplet_key, normalize_list, normalize_text, stringify_for_csv, write_jsonl


def _extract_ideology_signs(ideology_preference: dict) -> tuple[str | None, str | None]:
    liberal_sign = normalize_sign(
        ideology_preference.get("economic_liberal_preferred_sign")
        or ideology_preference.get("economic_liberal_expected_sign")
    )
    conservative_sign = normalize_sign(
        ideology_preference.get("economic_conservative_preferred_sign")
        or ideology_preference.get("economic_conservative_expected_sign")
    )
    return liberal_sign, conservative_sign


def _normalize_legacy_payload(record: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    classification = record.get("classification") or {}

    # Check if this is the new ideology-only format (no "labels" key, just "ideology_preference" at top level)
    if "ideology_preference" in classification and "labels" not in classification:
        # New ideology-only format
        ideology_preference = classification.get("ideology_preference") or {}
        evidence = classification.get("evidence") or {}

        ideology_sensitivity = "non_sensitive"
        liberal_sign = None
        conservative_sign = None

        flag = ideology_preference.get("is_ideologically_sensitive")
        if flag is True:
            ideology_sensitivity = "ideology_sensitive"
        elif flag is False:
            ideology_sensitivity = "non_sensitive"
        else:
            ideology_sensitivity = None
            warnings.append("ideology_sensitive flag missing; preserved as missing")

        liberal_sign, conservative_sign = _extract_ideology_signs(ideology_preference)

        payload = {
            "metadata": {
                "ideology_sensitivity": ideology_sensitivity,
                "policy_direction": "unclear",
                "economic_liberal_preferred_sign": liberal_sign,
                "economic_conservative_preferred_sign": conservative_sign,
                "region_bucket": [],
                "time_decade": [],
                "unit_of_analysis": [],
                "race": [],
                "gender": [],
                "age": [],
                "text_surface_cues": {
                    "up_cues": [],
                    "down_cues": [],
                },
            },
            "evidence": {
                "ideology_sensitivity": normalize_list(evidence.get("ideology")),
                "policy_direction": [],
                "preferred_signs": normalize_list(evidence.get("ideology")),
                "region_bucket": [],
                "time_decade": [],
                "unit_of_analysis": [],
                "demographics": [],
                "text_surface_cues": [],
            },
        }
        return payload, warnings

    # Original legacy format with full "labels" structure
    labels = classification.get("labels") or {}
    evidence = classification.get("evidence") or {}

    ideology_preference = labels.get("ideology_preference")
    ideology_sensitivity = "non_sensitive"
    liberal_sign = None
    conservative_sign = None
    policy_direction = "unclear"

    if isinstance(ideology_preference, dict):
        flag = ideology_preference.get("is_ideologically_sensitive")
        if flag is True:
            ideology_sensitivity = "ideology_sensitive"
        elif flag is False:
            ideology_sensitivity = "non_sensitive"
        else:
            ideology_sensitivity = None
            warnings.append("legacy ideologically_sensitive flag missing; preserved as missing")
        liberal_sign, conservative_sign = _extract_ideology_signs(ideology_preference)
    elif isinstance(ideology_preference, str) and ideology_preference.strip():
        ideology_sensitivity = None
        warnings.append("legacy ideology_preference string has no explicit sensitivity flag; preserved as missing")
    else:
        ideology_sensitivity = None
        warnings.append("legacy ideology_preference missing; preserved as missing")

    policy_direction = "unclear"

    payload = {
        "metadata": {
            "ideology_sensitivity": ideology_sensitivity,
            "policy_direction": policy_direction,
            "economic_liberal_preferred_sign": liberal_sign,
            "economic_conservative_preferred_sign": conservative_sign,
            "region_bucket": normalize_list(labels.get("region_bucket")),
            "time_decade": normalize_list(labels.get("time_decade")),
            "unit_of_analysis": normalize_list(labels.get("unit_of_analysis")),
            "race": normalize_list(labels.get("race")),
            "gender": normalize_list(labels.get("gender")),
            "age": normalize_list(labels.get("age")),
            "text_surface_cues": {
                "up_cues": normalize_list((labels.get("text_surface_cues") or {}).get("up_cues")),
                "down_cues": normalize_list((labels.get("text_surface_cues") or {}).get("down_cues")),
            },
        },
        "evidence": {
            "ideology_sensitivity": normalize_list(evidence.get("ideology")),
            "policy_direction": [],
            "preferred_signs": normalize_list(evidence.get("ideology")),
            "region_bucket": normalize_list(evidence.get("region")),
            "time_decade": normalize_list(evidence.get("time")),
            "unit_of_analysis": normalize_list(evidence.get("unit")),
            "demographics": normalize_list(evidence.get("demographics")),
            "text_surface_cues": normalize_list(evidence.get("surface_cues")),
        },
    }
    return payload, warnings


def _normalize_new_payload(record: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    payload = record.get("metadata_annotation") or record.get("raw_metadata") or record
    if "metadata" not in payload:
        raise ValueError("input record is missing a metadata payload")
    return payload, warnings


def _canonical_row(record: dict, payload: dict, source: str, warnings: list[str]) -> dict:
    validation_errors = validate_metadata_payload(payload)
    metadata = payload["metadata"]
    if source == "legacy" and metadata.get("ideology_sensitivity") is None:
        validation_errors = [error for error in validation_errors if error != "invalid ideology_sensitivity"]
    return {
        "metadata_source": source,
        "paper_id": str(record.get("paper_id", "")).strip(),
        "title": record.get("title"),
        "published_venue": record.get("published_venue"),
        "publication_year": record.get("publication_year"),
        "treatment": record.get("treatment"),
        "outcome": record.get("outcome"),
        "sign": record.get("sign"),
        "context": record.get("context"),
        "triplet_key": record.get("triplet_key") or make_triplet_key(record.get("paper_id"), record.get("treatment"), record.get("outcome")),
        "triplet_uid": record.get("triplet_uid"),
        "ideology_sensitivity": metadata.get("ideology_sensitivity"),
        "policy_direction": metadata.get("policy_direction"),
        "economic_liberal_preferred_sign": metadata.get("economic_liberal_preferred_sign"),
        "economic_conservative_preferred_sign": metadata.get("economic_conservative_preferred_sign"),
        "region_bucket": normalize_list(metadata.get("region_bucket")),
        "time_decade": normalize_list(metadata.get("time_decade")),
        "unit_of_analysis": normalize_list(metadata.get("unit_of_analysis")),
        "race": normalize_list(metadata.get("race")),
        "gender": normalize_list(metadata.get("gender")),
        "age": normalize_list(metadata.get("age")),
        "text_surface_cues_up": normalize_list((metadata.get("text_surface_cues") or {}).get("up_cues")),
        "text_surface_cues_down": normalize_list((metadata.get("text_surface_cues") or {}).get("down_cues")),
        "evidence": payload.get("evidence", {}),
        "normalization_warnings": warnings,
        "validation_errors": validation_errors,
    }


def normalize_metadata_records(records: list[dict], source: str) -> list[dict]:
    """Normalize raw or legacy annotation rows into canonical metadata rows."""
    normalized: list[dict] = []
    for record in records:
        if source == "legacy":
            payload, warnings = _normalize_legacy_payload(record)
        else:
            payload, warnings = _normalize_new_payload(record)
        normalized.append(_canonical_row(record, payload, source, warnings))
    return normalized


def _load_default_records() -> tuple[list[dict], str]:
    if DEFAULT_FULL_METADATA_RAW_PATH.exists():
        return list(iter_jsonl(DEFAULT_FULL_METADATA_RAW_PATH)), "annotated"
    if DEFAULT_MERGED_CLASSIFICATION_PATH.exists():
        return list(iter_jsonl(DEFAULT_MERGED_CLASSIFICATION_PATH)), "legacy"
    legacy_rows: list[dict] = []
    for path in LEGACY_TASK1_CLASSIFICATION_PATHS.values():
        if path.exists():
            legacy_rows.extend(iter_jsonl(path))
    return legacy_rows, "legacy"


def _default_output_paths(source: str, input_path: str | None) -> tuple[Path, Path]:
    if source == "annotated":
        return DEFAULT_FULL_METADATA_CANONICAL_JSONL, DEFAULT_FULL_METADATA_CANONICAL_CSV
    if input_path:
        resolved = Path(input_path).resolve()
        if resolved == DEFAULT_MERGED_CLASSIFICATION_PATH.resolve():
            return DEFAULT_FULL_METADATA_CANONICAL_JSONL, DEFAULT_FULL_METADATA_CANONICAL_CSV
    elif DEFAULT_FULL_METADATA_RAW_PATH.exists() or DEFAULT_MERGED_CLASSIFICATION_PATH.exists():
        return DEFAULT_FULL_METADATA_CANONICAL_JSONL, DEFAULT_FULL_METADATA_CANONICAL_CSV
    return DEFAULT_BOOTSTRAP_METADATA_JSONL, DEFAULT_BOOTSTRAP_METADATA_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize ideology metadata into canonical JSONL/CSV.")
    parser.add_argument("--input", default=None, help="Input JSONL path. Defaults to full raw metadata if present, else legacy task1 classification files.")
    parser.add_argument("--source", choices=["annotated", "legacy"], default=None, help="Force input format. Defaults based on the selected input.")
    parser.add_argument("--output-jsonl", default=None, help="Canonical JSONL output path.")
    parser.add_argument("--output-csv", default=None, help="Canonical CSV output path.")
    args = parser.parse_args()

    ensure_output_dirs()

    if args.input:
        records = list(iter_jsonl(args.input))
        source = args.source or "annotated"
    else:
        records, source = _load_default_records()

    if not records:
        raise FileNotFoundError("No metadata records found to normalize.")

    normalized = normalize_metadata_records(records, source=source)
    default_jsonl, default_csv = _default_output_paths(source, args.input)
    output_jsonl = Path(args.output_jsonl or default_jsonl)
    output_csv = Path(args.output_csv or default_csv)

    write_jsonl(output_jsonl, normalized)
    frame = pd.DataFrame(
        [{key: stringify_for_csv(value) for key, value in row.items()} for row in normalized]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    invalid = frame["validation_errors"].astype(str).str.len().gt(2).sum() if "validation_errors" in frame else 0
    print(
        json.dumps(
            {
                "source": source,
                "n_rows": len(normalized),
                "output_jsonl": str(output_jsonl),
                "output_csv": str(output_csv),
                "n_invalid_rows": int(invalid),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
