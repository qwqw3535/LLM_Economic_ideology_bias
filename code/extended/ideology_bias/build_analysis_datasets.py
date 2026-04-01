"""Build merged analysis datasets for paper tasks 1-3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .jel import ideology_theme_group
from .load_results import load_task1_rows, load_task2_rows, load_task3_rows
from .load_sources import build_source_catalog, build_source_lookup, task_source_lookup
from .paths import (
    AUDITS_DIR,
    SOURCE_CATALOG_CSV,
    SOURCE_CATALOG_JSONL,
    TASK1_ANALYSIS_CSV,
    TASK1_ANALYSIS_JSONL,
    TASK2_ANALYSIS_CSV,
    TASK2_ANALYSIS_JSONL,
    TASK2_EXACT50_SIDE_CAPPED_PATH,
    TASK3_ANALYSIS_CSV,
    TASK3_ANALYSIS_JSONL,
    TASK3_EXACT50_SIDE_CAPPED_PATH,
    TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR,
    analysis_dataset_paths,
    ensure_output_dirs,
    preferred_metadata_jsonl,
)
from .schemas import VALID_SIGNS, normalize_sign
from .utils import read_jsonl, stringify_for_csv, write_csv_rows, write_jsonl


def _label_from_list(values: object) -> str:
    if not values:
        return "(missing)"
    if isinstance(values, list):
        return " / ".join(str(value) for value in values) if values else "(missing)"
    return str(values)


def _extract_decade_anchor(value: object) -> int | None:
    if value is None:
        return None
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    if len(digits) == 3:
        return int(digits) * 10
    return None


def _time_period_grouped(decades: list[str]) -> list[str]:
    groups: list[str] = []
    for decade in decades:
        year = _extract_decade_anchor(decade)
        if year is None:
            label = "(missing)"
        elif year < 1800:
            label = "pre-1800"
        elif year < 1850:
            label = "1800-1840s"
        elif year < 1900:
            label = "1850-1890s"
        elif year < 1950:
            label = "1900-1940s"
        elif year < 2000:
            label = "1950-1990s"
        elif year < 2020:
            label = "2000-2010s"
        else:
            label = "2020s+"
        if label not in groups:
            groups.append(label)
    return groups or ["(missing)"]


def _time_period_30y_grouped(decades: list[str]) -> list[str]:
    groups: list[str] = []
    for decade in decades:
        year = _extract_decade_anchor(decade)
        if year is None:
            label = "(missing)"
        elif year < 1900:
            label = "pre-1900"
        elif year < 1930:
            label = "1900-1929"
        elif year < 1960:
            label = "1930-1959"
        elif year < 1990:
            label = "1960-1989"
        elif year < 2020:
            label = "1990-2019"
        else:
            label = "2020+"
        if label not in groups:
            groups.append(label)
    return groups or ["(missing)"]


def _state_preference_label(
    sign: str | None,
    pro_state_preferred_sign: str | None,
    less_state_preferred_sign: str | None,
) -> str:
    sign = normalize_sign(sign)
    if sign not in {"+", "-"}:
        return "neutral_or_unmapped"
    pro_state_preferred_sign = normalize_sign(pro_state_preferred_sign)
    less_state_preferred_sign = normalize_sign(less_state_preferred_sign)
    pro_match = pro_state_preferred_sign in VALID_SIGNS and sign == pro_state_preferred_sign
    less_match = less_state_preferred_sign in VALID_SIGNS and sign == less_state_preferred_sign
    if pro_match and less_match:
        return "both_state_preferences"
    if pro_match:
        return "pro_state"
    if less_match:
        return "less_state"
    return "neutral_or_unmapped"


def _preferred_side_label(
    sign: str | None,
    liberal_sign: str | None,
    conservative_sign: str | None,
) -> str:
    sign = normalize_sign(sign)
    liberal_sign = normalize_sign(liberal_sign)
    conservative_sign = normalize_sign(conservative_sign)
    if sign not in VALID_SIGNS:
        return "unlabeled"
    lib_match = liberal_sign in VALID_SIGNS and sign == liberal_sign
    cons_match = conservative_sign in VALID_SIGNS and sign == conservative_sign
    if lib_match and cons_match:
        return "both"
    if lib_match:
        return "liberal"
    if cons_match:
        return "conservative"
    return "neither"


def _ideology_alignment(sign: str | None, sensitivity: str, liberal_sign: str | None, conservative_sign: str | None) -> str:
    sign = normalize_sign(sign)
    liberal_sign = normalize_sign(liberal_sign)
    conservative_sign = normalize_sign(conservative_sign)
    if sensitivity == "non_sensitive":
        return "non_sensitive"
    if sensitivity != "ideology_sensitive":
        return "unlabeled_or_other"
    lib_match = liberal_sign in VALID_SIGNS and sign == liberal_sign
    cons_match = conservative_sign in VALID_SIGNS and sign == conservative_sign
    if lib_match and cons_match:
        return "both_aligned"
    if lib_match:
        return "liberal_aligned"
    if cons_match:
        return "conservative_aligned"
    return "unlabeled_or_other"


def _metadata_defaults(triplet_key: str) -> dict:
    return {
        "triplet_key": triplet_key,
        "metadata_source": None,
        "ideology_sensitivity": None,
        "policy_direction": "unclear",
        "economic_liberal_preferred_sign": None,
        "economic_conservative_preferred_sign": None,
        "region_bucket": [],
        "time_decade": [],
        "unit_of_analysis": [],
        "race": [],
        "gender": [],
        "age": [],
        "text_surface_cues_up": [],
        "text_surface_cues_down": [],
        "pro_state_preferred_sign": None,
        "less_state_preferred_sign": None,
        "preferred_sign_profile_label": "(missing)",
        "metadata_matched": 0,
    }


def _enrich_with_metadata(row: dict, metadata_row: dict | None) -> dict:
    metadata_row = metadata_row or _metadata_defaults(row["triplet_key"])
    merged = dict(row)
    for key, value in metadata_row.items():
        if key == "triplet_key":
            continue
        merged[key] = value
    merged["metadata_matched"] = int(metadata_row is not None and metadata_row.get("metadata_source") is not None)
    merged["region_bucket_label"] = _label_from_list(merged.get("region_bucket"))
    merged["time_decade_label"] = _label_from_list(merged.get("time_decade"))
    merged["time_period_grouped"] = _time_period_grouped(merged.get("time_decade") or [])
    merged["time_period_grouped_label"] = _label_from_list(merged["time_period_grouped"])
    merged["time_period_30y_grouped"] = _time_period_30y_grouped(merged.get("time_decade") or [])
    merged["time_period_30y_grouped_label"] = _label_from_list(merged["time_period_30y_grouped"])
    merged["unit_of_analysis_label"] = _label_from_list(merged.get("unit_of_analysis"))
    merged["race_label"] = _label_from_list(merged.get("race"))
    merged["gender_label"] = _label_from_list(merged.get("gender"))
    merged["age_label"] = _label_from_list(merged.get("age"))
    merged["up_cue_present"] = "yes" if merged.get("text_surface_cues_up") else "no"
    merged["down_cue_present"] = "yes" if merged.get("text_surface_cues_down") else "no"
    merged["pro_state_preferred_sign"] = normalize_sign(merged.get("economic_liberal_preferred_sign"))
    merged["less_state_preferred_sign"] = normalize_sign(merged.get("economic_conservative_preferred_sign"))
    merged["preferred_sign_profile_label"] = (
        f"pro:{merged['pro_state_preferred_sign'] or 'null'} | less:{merged['less_state_preferred_sign'] or 'null'}"
    )
    merged["directional_ground_truth"] = _state_preference_label(
        merged.get("expected_sign"),
        merged.get("pro_state_preferred_sign"),
        merged.get("less_state_preferred_sign"),
    )
    merged["directional_prediction"] = _state_preference_label(
        merged.get("predicted_sign"),
        merged.get("pro_state_preferred_sign"),
        merged.get("less_state_preferred_sign"),
    )
    merged["ground_truth_side"] = _preferred_side_label(
        merged.get("expected_sign"),
        merged.get("economic_liberal_preferred_sign"),
        merged.get("economic_conservative_preferred_sign"),
    )
    merged["prediction_side"] = _preferred_side_label(
        merged.get("predicted_sign"),
        merged.get("economic_liberal_preferred_sign"),
        merged.get("economic_conservative_preferred_sign"),
    )
    merged["ground_truth_liberal"] = int(merged["ground_truth_side"] == "liberal")
    merged["ground_truth_conservative"] = int(merged["ground_truth_side"] == "conservative")
    merged["predicted_liberal"] = int(merged["prediction_side"] == "liberal")
    merged["predicted_conservative"] = int(merged["prediction_side"] == "conservative")
    merged["ideology_triplet_labeled"] = int(merged["ground_truth_side"] in {"liberal", "conservative", "both"})
    merged["liberal_leaning_error"] = int(merged.get("correct", 0) == 0 and merged["prediction_side"] == "liberal")
    merged["conservative_leaning_error"] = int(merged.get("correct", 0) == 0 and merged["prediction_side"] == "conservative")
    merged["ideology_alignment_ground_truth"] = _ideology_alignment(
        merged.get("expected_sign"),
        merged.get("ideology_sensitivity"),
        merged.get("economic_liberal_preferred_sign"),
        merged.get("economic_conservative_preferred_sign"),
    )
    merged["ideology_alignment_prediction"] = _ideology_alignment(
        merged.get("predicted_sign"),
        merged.get("ideology_sensitivity"),
        merged.get("economic_liberal_preferred_sign"),
        merged.get("economic_conservative_preferred_sign"),
    )
    return merged


def _enrich_with_source(row: dict, source_row: dict | None) -> dict:
    merged = dict(row)
    if source_row:
        for key in (
            "title",
            "author",
            "publication_year",
            "published_venue",
            "published_venue_normalized",
            "domain",
            "jel_codes_raw",
            "jel_codes",
            "jel_prefixes",
            "primary_jel_prefix",
            "primary_jel_name",
            "jel_group",
            "jel_policy_theme_vote_primary",
            "jel_policy_theme_vote_counts",
            "jel_policy_theme_vote_weights",
            "jel_policy_theme_vote_matched_themes",
            "jel_policy_theme_vote_tied_themes",
            "jel_policy_theme_vote_max_count",
            "jel_policy_theme_vote_tie_break",
            "paper_url",
            "identification_methods",
        ):
            merged[key] = source_row.get(key)
        merged["jel_policy_theme"] = ideology_theme_group(
            source_row.get("jel_codes") or source_row.get("jel_codes_raw")
        )
        merged["source_matched"] = 1
    else:
        merged["source_matched"] = 0
        merged["domain"] = row.get("domain_hint") if row.get("domain_hint") != "mixed" else "unknown"
        merged["jel_policy_theme"] = "other"
        merged["jel_policy_theme_vote_primary"] = "other"
        merged["jel_policy_theme_vote_counts"] = {}
        merged["jel_policy_theme_vote_weights"] = {"other": 1.0}
        merged["jel_policy_theme_vote_matched_themes"] = []
        merged["jel_policy_theme_vote_tied_themes"] = []
        merged["jel_policy_theme_vote_max_count"] = 0
        merged["jel_policy_theme_vote_tie_break"] = "priority"
    return merged


def _state_profile_relation(example_row: dict, target_pro_sign: str | None, target_less_sign: str | None) -> str:
    example_pro_sign = normalize_sign(example_row.get("economic_liberal_preferred_sign"))
    example_less_sign = normalize_sign(example_row.get("economic_conservative_preferred_sign"))
    target_pro_sign = normalize_sign(target_pro_sign)
    target_less_sign = normalize_sign(target_less_sign)
    if not any([example_pro_sign, example_less_sign, target_pro_sign, target_less_sign]):
        return "unclear"
    if example_pro_sign == target_pro_sign and example_less_sign == target_less_sign:
        return "aligned"
    if (
        example_pro_sign == target_less_sign
        and example_less_sign == target_pro_sign
        and (example_pro_sign is not None or example_less_sign is not None)
    ):
        return "cross_direction"
    return "mixed_examples"


def _label_example_side(example_rows: list[dict], sign_field: str, require_false: bool = False) -> str:
    labels: list[str] = []
    for example_row in example_rows:
        if require_false and normalize_sign(example_row.get("displayed_sign")) == normalize_sign(example_row.get("original_sign")):
            continue
        side = _preferred_side_label(
            example_row.get(sign_field),
            example_row.get("economic_liberal_preferred_sign"),
            example_row.get("economic_conservative_preferred_sign"),
        )
        if side in {"liberal", "conservative"}:
            labels.append(side)
    if not labels:
        return "unlabeled"
    if len(set(labels)) == 1:
        return labels[0]
    return "mixed"


def _example_metadata_summary(
    example_triplets: list[dict],
    metadata_lookup: dict[str, dict],
    target_pro_sign: str | None,
    target_less_sign: str | None,
) -> dict:
    metadata_rows = [metadata_lookup.get(example["triplet_key"]) for example in example_triplets]
    matched = [row for row in metadata_rows if row]
    example_rows_with_signs: list[dict] = []
    for example_triplet, metadata_row in zip(example_triplets, metadata_rows):
        if not metadata_row:
            continue
        example_rows_with_signs.append(
            {
                "economic_liberal_preferred_sign": metadata_row.get("economic_liberal_preferred_sign"),
                "economic_conservative_preferred_sign": metadata_row.get("economic_conservative_preferred_sign"),
                "displayed_sign": example_triplet.get("displayed_sign"),
                "original_sign": example_triplet.get("original_sign"),
            }
        )
    sensitivities = sorted({row.get("ideology_sensitivity") for row in matched if row.get("ideology_sensitivity")})
    profile_labels = sorted(
        {
            f"pro:{normalize_sign(row.get('economic_liberal_preferred_sign')) or 'null'} | less:{normalize_sign(row.get('economic_conservative_preferred_sign')) or 'null'}"
            for row in matched
        }
    )
    relations = [
        _state_profile_relation(row, target_pro_sign=target_pro_sign, target_less_sign=target_less_sign)
        for row in matched
    ]
    known_relations = [relation for relation in relations if relation != "unclear"]
    if not known_relations:
        relation = "unclear"
    elif set(known_relations) == {"aligned"}:
        relation = "aligned"
    elif set(known_relations) == {"cross_direction"}:
        relation = "cross_direction"
    else:
        relation = "mixed_examples"

    first = example_triplets[0] if example_triplets else {}
    return {
        "example_preferred_sign_profiles": profile_labels,
        "example_preferred_sign_profile_label": _label_from_list(profile_labels),
        "example_ideology_sensitivities": sensitivities,
        "example_ideology_sensitivity_label": _label_from_list(sensitivities),
        "example_direction_relation": relation,
        "example_true_side_label": _label_example_side(example_rows_with_signs, "original_sign", require_false=False),
        "example_false_side_label": _label_example_side(example_rows_with_signs, "displayed_sign", require_false=True),
        "example_has_sensitive": int("ideology_sensitive" in sensitivities),
        "example_metadata_coverage": len(matched) / len(example_triplets) if example_triplets else 0.0,
        "first_example_displayed_sign": first.get("displayed_sign"),
        "first_example_original_sign": first.get("original_sign"),
        "first_example_triplet_key": first.get("triplet_key"),
    }


def _merge_task_rows(
    rows: list[dict],
    metadata_lookup: dict[str, dict],
    source_lookup: dict[str, dict],
    task_lookup: dict[str, dict],
    baseline_lookup: dict[tuple[str, str, str], dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    merged_rows: list[dict] = []
    unmatched_source: list[dict] = []
    unmatched_metadata: list[dict] = []
    for row in rows:
        source_row = task_lookup.get(row["triplet_key"]) or source_lookup.get(row["triplet_key"])
        enriched = _enrich_with_source(row, source_row)
        if source_row is None:
            unmatched_source.append(
                {
                    "paper_task": row["paper_task"],
                    "case_id": row["case_id"],
                    "triplet_key": row["triplet_key"],
                    "family": row["family"],
                    "model": row["model"],
                }
            )
        metadata_row = metadata_lookup.get(row["triplet_key"])
        enriched = _enrich_with_metadata(enriched, metadata_row)
        if metadata_row is None:
            unmatched_metadata.append(
                {
                    "paper_task": row["paper_task"],
                    "case_id": row["case_id"],
                    "triplet_key": row["triplet_key"],
                    "family": row["family"],
                    "model": row["model"],
                }
            )

        if baseline_lookup is not None:
            baseline = baseline_lookup.get((row["family"], row["model"], row["triplet_key"]))
            if baseline:
                enriched["baseline_predicted_sign"] = baseline.get("predicted_sign")
                enriched["baseline_correct"] = baseline.get("correct")
                enriched["prediction_shift_vs_task1"] = int(
                    baseline.get("predicted_sign") != enriched.get("predicted_sign")
                )
                enriched["correct_shift_vs_task1"] = enriched.get("correct", 0) - baseline.get("correct", 0)
            else:
                enriched["baseline_predicted_sign"] = None
                enriched["baseline_correct"] = None
                enriched["prediction_shift_vs_task1"] = None
                enriched["correct_shift_vs_task1"] = None

            example_triplets = row.get("example_triplets") or []
            enriched.update(
                _example_metadata_summary(
                    example_triplets,
                    metadata_lookup,
                    enriched.get("pro_state_preferred_sign"),
                    enriched.get("less_state_preferred_sign"),
                )
            )
            for key in (
                "variant",
                "pair_case_type",
                "matching_rule",
                "target_side",
                "example_true_side",
                "example_false_side",
                "target_exact_jel_codes",
                "example_exact_jel_codes",
                "shared_exact_jel_codes",
                "shared_exact_jel_count",
                "target_exact_jel_count",
                "jel_overlap_ratio",
                "different_paper",
            ):
                if row.get(key) not in (None, "", []):
                    enriched[key] = row.get(key)
            if row.get("example_true_side") not in (None, ""):
                enriched["example_true_side_label"] = row.get("example_true_side")
            if row.get("example_false_side") not in (None, ""):
                enriched["example_false_side_label"] = row.get("example_false_side")
            if row["paper_task"] == "task3":
                first_displayed = enriched.get("first_example_displayed_sign")
                first_original = enriched.get("first_example_original_sign")
                enriched["noise_displayed_differs_from_original"] = int(first_displayed != first_original)
                enriched["follows_displayed_example_sign"] = int(
                    first_displayed is not None and enriched.get("predicted_sign") == first_displayed
                )
                enriched["follows_original_example_sign"] = int(
                    first_original is not None and enriched.get("predicted_sign") == first_original
                )
        merged_rows.append(enriched)
    return merged_rows, unmatched_source, unmatched_metadata


def _save_dataset(rows: list[dict], csv_path: Path, jsonl_path: Path) -> None:
    write_jsonl(jsonl_path, rows)
    frame = pd.DataFrame([{key: stringify_for_csv(value) for key, value in row.items()} for row in rows])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged analysis datasets for ideology bias analyses.")
    parser.add_argument("--metadata-jsonl", default=str(preferred_metadata_jsonl()), help="Canonical metadata JSONL path")
    parser.add_argument("--task1-results-dir", default=None, help="Task1 evaluation results directory")
    parser.add_argument("--task2-source", default=str(TASK2_EXACT50_SIDE_CAPPED_PATH), help="Task2 source JSONL path")
    parser.add_argument("--task3-source", default=str(TASK3_EXACT50_SIDE_CAPPED_PATH), help="Task3 source JSONL path")
    parser.add_argument("--task23-results-dir", default=str(TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR), help="Task2/3 evaluation results directory")
    parser.add_argument("--dataset-suffix", default="", help="Suffix for output analysis dataset filenames")
    parser.add_argument(
        "--skip-task1-output",
        action="store_true",
        help="Skip rebuilding and saving task1 analysis outputs; only load minimal task1 baseline signals for task2/task3 comparisons.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    metadata_rows = read_jsonl(args.metadata_jsonl)
    metadata_lookup = {row["triplet_key"]: row for row in metadata_rows}

    source_catalog = build_source_catalog()
    source_lookup = build_source_lookup(source_catalog)
    task1_lookup = task_source_lookup("task1")
    task2_lookup = task_source_lookup("task2", path_override=args.task2_source)
    task3_lookup = task_source_lookup("task3", path_override=args.task3_source)

    task1_csv_path, task1_jsonl_path = analysis_dataset_paths("task1", args.dataset_suffix)
    task2_csv_path, task2_jsonl_path = analysis_dataset_paths("task2", args.dataset_suffix)
    task3_csv_path, task3_jsonl_path = analysis_dataset_paths("task3", args.dataset_suffix)

    _save_dataset(source_catalog, SOURCE_CATALOG_CSV, SOURCE_CATALOG_JSONL)

    task1_rows_raw = load_task1_rows(results_dir=args.task1_results_dir)
    task1_unmatched_source: list[dict] = []
    task1_unmatched_metadata: list[dict] = []
    if args.skip_task1_output:
        baseline_lookup = {(row["family"], row["model"], row["triplet_key"]): row for row in task1_rows_raw}
    else:
        task1_rows, task1_unmatched_source, task1_unmatched_metadata = _merge_task_rows(
            task1_rows_raw,
            metadata_lookup,
            source_lookup,
            task1_lookup,
        )
        _save_dataset(
            task1_rows,
            task1_csv_path if args.dataset_suffix else TASK1_ANALYSIS_CSV,
            task1_jsonl_path if args.dataset_suffix else TASK1_ANALYSIS_JSONL,
        )
        baseline_lookup = {(row["family"], row["model"], row["triplet_key"]): row for row in task1_rows}

    task2_rows_raw = load_task2_rows(results_dir=args.task23_results_dir)
    task2_rows, task2_unmatched_source, task2_unmatched_metadata = _merge_task_rows(
        task2_rows_raw,
        metadata_lookup,
        source_lookup,
        task2_lookup,
        baseline_lookup=baseline_lookup,
    )
    _save_dataset(task2_rows, task2_csv_path if args.dataset_suffix else TASK2_ANALYSIS_CSV, task2_jsonl_path if args.dataset_suffix else TASK2_ANALYSIS_JSONL)

    task3_rows_raw = load_task3_rows(results_dir=args.task23_results_dir)
    task3_rows, task3_unmatched_source, task3_unmatched_metadata = _merge_task_rows(
        task3_rows_raw,
        metadata_lookup,
        source_lookup,
        task3_lookup,
        baseline_lookup=baseline_lookup,
    )
    _save_dataset(task3_rows, task3_csv_path if args.dataset_suffix else TASK3_ANALYSIS_CSV, task3_jsonl_path if args.dataset_suffix else TASK3_ANALYSIS_JSONL)

    suffix = f"_{args.dataset_suffix}" if args.dataset_suffix else ""
    write_csv_rows(AUDITS_DIR / f"task1_unmatched_source{suffix}.csv", task1_unmatched_source)
    write_csv_rows(AUDITS_DIR / f"task1_unmatched_metadata{suffix}.csv", task1_unmatched_metadata)
    write_csv_rows(AUDITS_DIR / f"task2_unmatched_source{suffix}.csv", task2_unmatched_source)
    write_csv_rows(AUDITS_DIR / f"task2_unmatched_metadata{suffix}.csv", task2_unmatched_metadata)
    write_csv_rows(AUDITS_DIR / f"task3_unmatched_source{suffix}.csv", task3_unmatched_source)
    write_csv_rows(AUDITS_DIR / f"task3_unmatched_metadata{suffix}.csv", task3_unmatched_metadata)

    print(
        json.dumps(
            {
                "metadata_input": args.metadata_jsonl,
                "task1_results_dir": args.task1_results_dir,
                "task2_source": args.task2_source,
                "task3_source": args.task3_source,
                "task23_results_dir": args.task23_results_dir,
                "dataset_suffix": args.dataset_suffix,
                "source_catalog_rows": len(source_catalog),
                "task1_rows": len(task1_rows_raw),
                "task2_rows": len(task2_rows),
                "task3_rows": len(task3_rows),
                "task1_unmatched_metadata": len(task1_unmatched_metadata),
                "task2_unmatched_metadata": len(task2_unmatched_metadata),
                "task3_unmatched_metadata": len(task3_unmatched_metadata),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
