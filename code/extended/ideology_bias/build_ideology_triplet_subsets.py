"""Build ideology triplet subsets and balanced review samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .jel import ideology_theme_vote_details
from .paths import CLASSIFICATION_RESULTS_DIR
from .utils import iter_jsonl, make_triplet_key, write_jsonl


DEFAULT_RESULTS_PATH = CLASSIFICATION_RESULTS_DIR / "causal_triplets_multillm_ideology_qwen32b.jsonl"
DEFAULT_TASK1_PATH = Path("extended/ideology_bias_outputs/analysis_datasets/task1_analysis_rows_jel_similarity_side_capped_jaccard05_shared2.csv")
DEFAULT_SUBSET_PATH = CLASSIFICATION_RESULTS_DIR / "ideology_triplet_subset_current.jsonl"
DEFAULT_REVIEW_PATH = CLASSIFICATION_RESULTS_DIR / "ideology_triplet_review_sample_balanced126.jsonl"

SELECTED_MODELS = [
    "gpt-5-mini",
    "claude-sonnet-4-6",
    "qwen-3-32b",
    "grok-4-1-fast-reasoning",
]
VALID_SIGNS = {"+", "-", "None", "Mixed"}
REVIEW_THEMES = [
    "taxation",
    "healthcare",
    "education",
    "welfare_redistribution",
    "labor",
    "financial_regulation",
    "trade",
]
REVIEW_SIDES = ["liberal", "conservative", "neither"]


def _normalize_sign(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _majority(values: list[str]) -> str | None:
    counts = Counter(values)
    if not counts:
        return None
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def _parse_raw_response(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _selected_subset_rows(results_path: Path) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for row in iter_jsonl(results_path):
        per_model = row.get("classification", {}).get("per_model", {})
        lib_values: list[str] = []
        con_values: list[str] = []
        model_signs: dict[str, dict[str, str | None]] = {}
        for model in SELECTED_MODELS:
            model_row = per_model.get(model, {})
            if not isinstance(model_row, dict):
                continue
            raw = _parse_raw_response(model_row.get("raw_response"))
            pref = raw.get("ideology_preference") or {}
            if not isinstance(pref, dict):
                pref = {}
            lib = _normalize_sign(pref.get("economic_liberal_expected_sign"))
            con = _normalize_sign(pref.get("economic_conservative_expected_sign"))
            model_signs[model] = {"lib": lib, "con": con}
            if lib in VALID_SIGNS and con in VALID_SIGNS:
                lib_values.append(lib)
                con_values.append(con)

        if len(lib_values) < 3:
            continue
        lib_vote = _majority(lib_values)
        con_vote = _majority(con_values)
        if lib_vote is None or con_vote is None or lib_vote == con_vote:
            continue

        triplet_key = make_triplet_key(row.get("paper_id"), row.get("treatment"), row.get("outcome"))
        selected[triplet_key] = {
            "triplet_key": triplet_key,
            "paper_id": row.get("paper_id"),
            "title": row.get("title"),
            "author": row.get("author"),
            "publication_year": row.get("publication_year"),
            "published_venue": row.get("published_venue"),
            "paper_url": row.get("paper_url"),
            "treatment": row.get("treatment"),
            "outcome": row.get("outcome"),
            "sign": row.get("sign"),
            "context": row.get("context"),
            "identification_methods": row.get("identification_methods"),
            "lib_vote": lib_vote,
            "con_vote": con_vote,
            "selected_models": list(SELECTED_MODELS),
            "model_signs": model_signs,
        }
    return selected


def _low_defined_rows(results_path: Path, max_defined: int = 2) -> dict[str, dict]:
    neutral: dict[str, dict] = {}
    for row in iter_jsonl(results_path):
        per_model = row.get("classification", {}).get("per_model", {})
        model_signs: dict[str, dict[str, str | None]] = {}
        defined_count = 0
        for model in SELECTED_MODELS:
            model_row = per_model.get(model, {})
            if not isinstance(model_row, dict):
                continue
            raw = _parse_raw_response(model_row.get("raw_response"))
            pref = raw.get("ideology_preference") or {}
            if not isinstance(pref, dict):
                pref = {}
            lib = _normalize_sign(pref.get("economic_liberal_expected_sign"))
            con = _normalize_sign(pref.get("economic_conservative_expected_sign"))
            model_signs[model] = {"lib": lib, "con": con}
            if lib in VALID_SIGNS and con in VALID_SIGNS:
                defined_count += 1

        if defined_count > max_defined:
            continue

        triplet_key = make_triplet_key(row.get("paper_id"), row.get("treatment"), row.get("outcome"))
        neutral[triplet_key] = {
            "triplet_key": triplet_key,
            "paper_id": row.get("paper_id"),
            "title": row.get("title"),
            "author": row.get("author"),
            "publication_year": row.get("publication_year"),
            "published_venue": row.get("published_venue"),
            "paper_url": row.get("paper_url"),
            "treatment": row.get("treatment"),
            "outcome": row.get("outcome"),
            "sign": row.get("sign"),
            "context": row.get("context"),
            "identification_methods": row.get("identification_methods"),
            "lib_vote": None,
            "con_vote": None,
            "defined_model_count": defined_count,
            "selected_models": list(SELECTED_MODELS),
            "model_signs": model_signs,
            "review_bucket": "neither",
        }
    return neutral


def _load_task1_lookup(task1_path: Path) -> dict[str, dict]:
    frame = pd.read_csv(task1_path, low_memory=False)
    frame["triplet_key"] = frame["triplet_key"].astype(str)
    frame = frame.drop_duplicates(subset=["triplet_key"]).copy()
    return {row["triplet_key"]: row.to_dict() for _, row in frame.iterrows()}


def _enrich_rows(selected: dict[str, dict], task1_lookup: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for triplet_key, base in selected.items():
        meta = task1_lookup.get(triplet_key, {})
        jel_codes = meta.get("jel_codes")
        theme = ideology_theme_vote_details(jel_codes)["primary_theme"]
        rows.append(
            {
                **base,
                "ground_truth_side": meta.get("ground_truth_side"),
                "review_bucket": base.get("review_bucket", meta.get("ground_truth_side")),
                "jel_codes": jel_codes,
                "jel_policy_theme": theme,
            }
        )
    return sorted(rows, key=lambda row: row["triplet_key"])


def _year_bucket(value: object) -> str:
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return "(missing)"
    start = year // 10 * 10
    return f"{start}s"


def _sample_review_rows(rows: list[dict], per_cell: int, seed: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        theme = row.get("jel_policy_theme")
        side = row.get("review_bucket")
        if theme in REVIEW_THEMES and side in REVIEW_SIDES:
            grouped[(theme, side)].append(row)

    effective_per_cell = min(len(grouped[(theme, side)]) for theme in REVIEW_THEMES for side in REVIEW_SIDES)
    effective_per_cell = min(per_cell, effective_per_cell)
    rng = random.Random(seed)
    paper_counts: Counter[str] = Counter()
    year_counts: Counter[tuple[str, str]] = Counter()
    venue_counts: Counter[str] = Counter()
    sampled: list[dict] = []
    for theme in REVIEW_THEMES:
        for side in REVIEW_SIDES:
            bucket = list(grouped[(theme, side)])
            chosen: list[dict] = []
            while len(chosen) < effective_per_cell:
                remaining = [row for row in bucket if row not in chosen]
                rng.shuffle(remaining)
                remaining.sort(
                    key=lambda row: (
                        paper_counts[str(row.get("paper_id"))],
                        year_counts[(theme, _year_bucket(row.get("publication_year")))],
                        venue_counts[str(row.get("published_venue") or "")],
                        str(row.get("triplet_key")),
                    )
                )
                picked = remaining[0]
                chosen.append(picked)
                paper_counts[str(picked.get("paper_id"))] += 1
                year_counts[(theme, _year_bucket(picked.get("publication_year")))] += 1
                venue_counts[str(picked.get("published_venue") or "")] += 1
            sampled.extend(chosen)

    return sorted(sampled, key=lambda row: (row["jel_policy_theme"], row["review_bucket"], row["triplet_key"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ideology triplet subset and balanced review sample.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--task1-analysis", type=Path, default=DEFAULT_TASK1_PATH)
    parser.add_argument("--subset-output", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--per-cell", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = _selected_subset_rows(args.results)
    neutral = _low_defined_rows(args.results, max_defined=2)
    task1_lookup = _load_task1_lookup(args.task1_analysis)
    enriched = _enrich_rows(selected, task1_lookup)
    neutral_enriched = _enrich_rows(neutral, task1_lookup)
    review_candidates = [row for row in enriched if row.get("review_bucket") in {"liberal", "conservative"}]
    sampled = _sample_review_rows(review_candidates + neutral_enriched, args.per_cell, args.seed)

    write_jsonl(args.subset_output, enriched)
    write_jsonl(args.review_output, sampled)

    subset_counts = Counter(row.get("jel_policy_theme") for row in enriched)
    review_counts = Counter((row.get("jel_policy_theme"), row.get("review_bucket")) for row in sampled)

    print(f"subset_total={len(enriched)}")
    print(f"review_total={len(sampled)}")
    print("subset_theme_counts=" + json.dumps(dict(sorted(subset_counts.items())), ensure_ascii=False))
    print(
        "review_cell_counts="
        + json.dumps(
            {
                f"{theme}|{side}": review_counts[(theme, side)]
                for theme in REVIEW_THEMES
                for side in REVIEW_SIDES
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
