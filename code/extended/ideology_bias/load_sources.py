"""Load source JSONL files into a canonical triplet catalog."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .jel import (
    collapsed_jel_group,
    ideology_theme_vote_details,
    jel_prefixes,
    primary_jel_name,
    primary_jel_prefix,
    split_jel_codes,
)
from .paths import (
    FULL_CORPUS_PATH,
    TASK1_ECON_PATH,
    TASK1_FINANCE_PATH,
    TASK2_PATH,
    TASK3_PATH,
    infer_domain_from_venue,
    normalize_venue,
)
from .utils import iter_jsonl, make_triplet_key, make_triplet_uid, normalize_text, parse_example_details


def _jel_derived_fields(raw_jel_codes: object) -> dict:
    vote_theme = ideology_theme_vote_details(raw_jel_codes)
    return {
        "jel_codes_raw": raw_jel_codes,
        "jel_codes": split_jel_codes(raw_jel_codes),
        "jel_prefixes": jel_prefixes(raw_jel_codes),
        "primary_jel_prefix": primary_jel_prefix(raw_jel_codes),
        "primary_jel_name": primary_jel_name(raw_jel_codes),
        "jel_group": collapsed_jel_group(raw_jel_codes),
        "jel_policy_theme_vote_primary": vote_theme["primary_theme"],
        "jel_policy_theme_vote_counts": vote_theme["theme_counts"],
        "jel_policy_theme_vote_weights": vote_theme["theme_weights"],
        "jel_policy_theme_vote_matched_themes": vote_theme["matched_themes"],
        "jel_policy_theme_vote_tied_themes": vote_theme["tied_themes"],
        "jel_policy_theme_vote_max_count": vote_theme["max_count"],
        "jel_policy_theme_vote_tie_break": vote_theme["tie_break_rule"],
    }


def _base_source_row(record: dict, source_name: str) -> dict:
    venue_normalized = normalize_venue(record.get("published_venue"))
    triplet_key = make_triplet_key(record.get("paper_id"), record.get("treatment"), record.get("outcome"))
    return {
        "source_name": source_name,
        "paper_id": str(record.get("paper_id", "")).strip(),
        "title": record.get("title"),
        "author": record.get("author"),
        "publication_year": record.get("publication_year"),
        "published_venue": record.get("published_venue"),
        "published_venue_normalized": venue_normalized,
        "domain": infer_domain_from_venue(venue_normalized),
        **_jel_derived_fields(record.get("jel_codes")),
        "paper_url": record.get("paper_url"),
        "treatment": record.get("treatment"),
        "outcome": record.get("outcome"),
        "sign": record.get("sign"),
        "context": record.get("context"),
        "identification_methods": record.get("identification_methods"),
        "triplet_key": triplet_key,
        "triplet_uid": make_triplet_uid(
            record.get("paper_id"),
            record.get("treatment"),
            record.get("outcome"),
            record.get("context"),
        ),
        "example_details": parse_example_details(record.get("example_details")),
        "case_id": record.get("case_id"),
        "variant": record.get("variant"),
        "pair_case_type": record.get("pair_case_type"),
        "matching_rule": record.get("matching_rule"),
        "target_side": record.get("target_side"),
        "example_true_side": record.get("example_true_side"),
        "example_false_side": record.get("example_false_side"),
        "target_exact_jel_codes": record.get("target_exact_jel_codes") or [],
        "example_exact_jel_codes": record.get("example_exact_jel_codes") or [],
        "shared_exact_jel_codes": record.get("shared_exact_jel_codes") or [],
        "shared_exact_jel_count": record.get("shared_exact_jel_count"),
        "target_exact_jel_count": record.get("target_exact_jel_count"),
        "jel_overlap_ratio": record.get("jel_overlap_ratio"),
        "different_paper": record.get("different_paper"),
    }


def load_source_file(path: str | Path, source_name: str) -> list[dict]:
    """Load a single source JSONL file into canonical rows."""
    return [_base_source_row(record, source_name) for record in iter_jsonl(path)]


def _merge_preferred(existing: dict, new_row: dict) -> dict:
    """Merge duplicate triplets while keeping the richest source row."""
    if not existing:
        out = dict(new_row)
        out["source_tags"] = [new_row["source_name"]]
        return out

    out = dict(existing)
    out["source_tags"] = sorted(set(existing.get("source_tags", [])) | {new_row["source_name"]})
    for key, value in new_row.items():
        if key == "source_name":
            continue
        if key == "example_details":
            if len(value) > len(out.get("example_details", [])):
                out[key] = value
            continue
        if key == "context":
            current = out.get("context") or ""
            candidate = value or ""
            if len(candidate) > len(current):
                out[key] = candidate
            continue
        if key in {"jel_codes", "jel_prefixes"}:
            merged = list(dict.fromkeys((out.get(key) or []) + (value or [])))
            out[key] = merged
            continue
        if key == "domain" and out.get("domain") == "unknown" and value:
            out[key] = value
            continue
        if not out.get(key) and value not in (None, "", []):
            out[key] = value
    out.update(_jel_derived_fields(out.get("jel_codes") or out.get("jel_codes_raw")))
    return out


def build_source_catalog(include_task_inputs: bool = True) -> list[dict]:
    """Build a deduplicated source catalog spanning full corpus and task inputs."""
    merged: dict[str, dict] = {}
    for source_name, path in [
        ("causal_triplets", FULL_CORPUS_PATH),
        ("task1_econ", TASK1_ECON_PATH),
        ("task1_finance", TASK1_FINANCE_PATH),
        ("task2", TASK2_PATH),
        ("task3", TASK3_PATH),
    ]:
        if not include_task_inputs and source_name != "causal_triplets":
            continue
        if not Path(path).exists():
            continue
        for row in load_source_file(path, source_name):
            key = row["triplet_key"]
            merged[key] = _merge_preferred(merged.get(key, {}), row)
    return list(merged.values())


def build_source_lookup(rows: list[dict]) -> dict[str, dict]:
    """Index source rows by triplet key."""
    return {row["triplet_key"]: row for row in rows}


def load_task_source_rows(task_name: str, path_override: str | Path | None = None) -> list[dict]:
    """Load the target rows for a paper task."""
    if task_name == "task1":
        return load_source_file(TASK1_ECON_PATH, "task1_econ") + load_source_file(TASK1_FINANCE_PATH, "task1_finance")
    if task_name == "task2":
        return load_source_file(path_override or TASK2_PATH, "task2")
    if task_name == "task3":
        return load_source_file(path_override or TASK3_PATH, "task3")
    raise ValueError(f"Unknown task_name: {task_name}")


def task_source_lookup(task_name: str, path_override: str | Path | None = None, key_field: str = "triplet_key") -> dict[str, dict]:
    """Build a lookup for the target rows of a paper task."""
    return {row.get(key_field): row for row in load_task_source_rows(task_name, path_override=path_override) if row.get(key_field)}


def source_catalog_dataframe(rows: list[dict] | None = None) -> pd.DataFrame:
    """Convert source catalog rows into a DataFrame."""
    rows = rows if rows is not None else build_source_catalog()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["source_tags"] = frame["source_tags"].apply(lambda value: value or [])
    return frame


def summarize_source_counts(rows: list[dict]) -> dict[str, int]:
    """Return source counts by source tag."""
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        for tag in row.get("source_tags", []):
            counts[tag] += 1
    return dict(sorted(counts.items()))


def find_source_by_triplet(rows: list[dict], paper_id: object, treatment: object, outcome: object) -> dict | None:
    """Find a source row by a raw triplet tuple."""
    triplet_key = make_triplet_key(paper_id, treatment, outcome)
    for row in rows:
        if row["triplet_key"] == triplet_key:
            return row
    return None
