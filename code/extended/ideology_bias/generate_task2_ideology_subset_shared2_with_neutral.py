"""Generate task2 source with shared2 target-fixed pairs plus neutral examples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from econ_eval.evaluation.data_generator import Task3Case
    from econ_eval.evaluation.tasks.task3_context_to_fixed import Task3ContextTOFixed

    from extended.ideology_bias.jel import split_jel_codes
    from extended.ideology_bias.utils import make_triplet_key, read_jsonl, write_jsonl
else:
    from econ_eval.evaluation.data_generator import Task3Case
    from econ_eval.evaluation.tasks.task3_context_to_fixed import Task3ContextTOFixed

    from .jel import split_jel_codes
    from .utils import make_triplet_key, read_jsonl, write_jsonl


DEFAULT_BASE_TASK2_PATH = Path("data/task2_ideology_subset_shared2_all_pairs_target_fixed.jsonl")
DEFAULT_FULL_CLASSIFICATION_PATH = Path("extended/classification_results/causal_triplets_multillm_ideology_qwen32b.jsonl")
DEFAULT_SUBSET_PATH = Path("extended/classification_results/ideology_triplet_subset_current.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/task2_ideology_subset_shared2_all_pairs_target_fixed_with_neutral_all_pairs.jsonl")
DEFAULT_SUMMARY_PATH = Path("extended/classification_results/task2_ideology_subset_shared2_all_pairs_target_fixed_with_neutral_all_pairs_summary.json")

NEUTRAL_MATCHING_RULE_TOP1 = "neutral_complement_jaccard_top1"
NEUTRAL_MATCHING_RULE_ALL = "neutral_complement_jaccard_all_pairs"
VARIANT_NAME_TOP1 = "ideology_subset_shared2_all_pairs_target_fixed_with_neutral"
VARIANT_NAME_ALL = "ideology_subset_shared2_all_pairs_target_fixed_with_neutral_all_pairs"


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _numeric_sort(value: object) -> tuple[int, str]:
    text = str(value or "").strip()
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def _informative_jel_codes(raw_value: object) -> list[str]:
    return [code for code in split_jel_codes(raw_value) if code and code[0].isalpha() and len(code) >= 2]


def _neutral_pool_rows(path: Path, excluded_triplet_keys: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            triplet_key = make_triplet_key(row.get("paper_id"), row.get("treatment"), row.get("outcome"))
            if triplet_key in excluded_triplet_keys:
                continue
            informative_codes = _informative_jel_codes(row.get("jel_codes"))
            if not informative_codes:
                continue
            rows.append(
                {
                    "triplet_key": triplet_key,
                    "paper_id": _safe_text(row.get("paper_id")),
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "publication_year": row.get("publication_year"),
                    "published_venue": row.get("published_venue"),
                    "jel_codes": row.get("jel_codes"),
                    "paper_url": row.get("paper_url"),
                    "treatment": row.get("treatment"),
                    "outcome": row.get("outcome"),
                    "sign": row.get("sign"),
                    "context": row.get("context"),
                    "identification_methods": row.get("identification_methods"),
                    "informative_jel_codes": informative_codes,
                    "informative_jel_set": set(informative_codes),
                }
            )
    return rows


def _neutral_match(target: dict, candidate: dict, *, require_shared2: bool = False) -> dict | None:
    if _safe_text(target.get("paper_id")) == candidate["paper_id"]:
        return None
    target_codes = set(target.get("target_exact_jel_codes") or [])
    example_codes = candidate["informative_jel_set"]
    if not target_codes or not example_codes:
        return None
    shared_codes = sorted(target_codes & example_codes)
    shared_count = len(shared_codes)
    if shared_count == 0:
        return None
    union_count = len(target_codes | example_codes)
    jel_similarity = shared_count / union_count if union_count else 0.0
    if require_shared2 and (shared_count < 2 or jel_similarity < 0.5):
        return None
    target_overlap = shared_count / len(target_codes)
    example_overlap = shared_count / len(example_codes)
    return {
        "target": target,
        "example": candidate,
        "shared_exact_jel_codes": shared_codes,
        "shared_exact_jel_count": shared_count,
        "target_exact_jel_count": len(target_codes),
        "example_exact_jel_count": len(example_codes),
        "jel_similarity": jel_similarity,
        "jel_overlap_ratio": target_overlap,
        "example_overlap_ratio": example_overlap,
        "selection_score": jel_similarity,
        "union_exact_jel_count": union_count,
    }


def _rank_match(pair: dict) -> tuple:
    example = pair["example"]
    target = pair["target"]
    return (
        -pair["selection_score"],
        -pair["shared_exact_jel_count"],
        abs(pair["example_exact_jel_count"] - pair["target_exact_jel_count"]),
        _numeric_sort(example["paper_id"]),
        _safe_text(example["treatment"]).lower(),
        _safe_text(example["outcome"]).lower(),
        _safe_text(target.get("triplet_key")).lower(),
    )


def _build_neutral_case(case_id: str, target: dict, match: dict, *, variant_name: str, matching_rule: str) -> Task3Case:
    example = match["example"]
    example_payload = {
        "paper_id": example["paper_id"],
        "title": example.get("title"),
        "author": example.get("author"),
        "publication_year": example.get("publication_year"),
        "published_venue": example.get("published_venue"),
        "jel_codes": example.get("jel_codes"),
        "paper_url": example.get("paper_url"),
        "treatment": example.get("treatment"),
        "outcome": example.get("outcome"),
        "context": example.get("context"),
        "sign": example.get("sign"),
        "original_sign": example.get("sign"),
        "economic_liberal_preferred_sign": None,
        "economic_conservative_preferred_sign": None,
        "example_true_side": "neutral",
        "example_false_side": "neutral",
        "example_exact_jel_codes": list(example.get("informative_jel_codes") or []),
        "shared_exact_jel_codes": list(match["shared_exact_jel_codes"]),
        "shared_exact_jel_count": match["shared_exact_jel_count"],
        "jel_overlap_ratio": round(match["jel_overlap_ratio"], 4),
        "example_overlap_ratio": round(match["example_overlap_ratio"], 4),
        "jel_similarity": round(match["jel_similarity"], 4),
        "union_exact_jel_count": match["union_exact_jel_count"],
        "selection_score": round(match["selection_score"], 4),
        "triplet_key": example["triplet_key"],
    }
    return Task3Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=[example_payload],
        test_context=target["test_context"],
        expected_sign=target["expected_sign"],
        paper_ids=[example["paper_id"], _safe_text(target.get("paper_id"))],
        avg_similarity=round(match["jel_similarity"], 4),
        sign_differs=example.get("sign") != target["expected_sign"],
        context=target["context"],
        sign=target["sign"],
        title=target.get("title"),
        author=target.get("author"),
        publication_year=target.get("publication_year"),
        published_venue=target.get("published_venue"),
        jel_codes=target.get("jel_codes"),
        paper_url=target.get("paper_url"),
        variant=variant_name,
        pair_case_type=f"neutral-{'lib' if target.get('target_side') == 'liberal' else 'cons'}",
        matching_rule=matching_rule,
        target_side=target.get("target_side"),
        example_true_side="neutral",
        example_false_side="neutral",
        target_exact_jel_codes=list(target.get("target_exact_jel_codes") or []),
        example_exact_jel_codes=list(example.get("informative_jel_codes") or []),
        shared_exact_jel_codes=list(match["shared_exact_jel_codes"]),
        shared_exact_jel_count=match["shared_exact_jel_count"],
        target_exact_jel_count=match["target_exact_jel_count"],
        jel_overlap_ratio=round(match["jel_overlap_ratio"], 4),
        example_overlap_ratio=round(match["example_overlap_ratio"], 4),
        jel_similarity=round(match["jel_similarity"], 4),
        union_exact_jel_count=match["union_exact_jel_count"],
        selection_score=round(match["selection_score"], 4),
        different_paper=True,
        triplet_key=target["triplet_key"],
    )


def _source_row(case: Task3Case, prompt: str) -> dict:
    row = asdict(case)
    row["paper_id"] = case.paper_ids[-1] if case.paper_ids else None
    row["question"] = prompt
    row["answer"] = case.expected_sign
    row["example_details"] = row["examples"]
    row["task_name"] = "task2"
    row["paper_task"] = "task2"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Add neutral top-1 matches to task2 shared2 all-pairs target-fixed source.")
    parser.add_argument("--base-task2-path", type=Path, default=DEFAULT_BASE_TASK2_PATH)
    parser.add_argument("--full-classification-path", type=Path, default=DEFAULT_FULL_CLASSIFICATION_PATH)
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--selection-mode", choices=["top1", "all_pairs"], default="all_pairs")
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_task2_path)
    subset_triplet_keys = {row["triplet_key"] for row in read_jsonl(args.subset_path)}
    target_rows: dict[str, dict] = {}
    for row in base_rows:
        target_rows.setdefault(row["triplet_key"], row)

    neutral_pool = _neutral_pool_rows(args.full_classification_path, subset_triplet_keys)
    formatter = Task3ContextTOFixed()

    neutral_rows: list[dict] = []
    match_scores: list[float] = []
    shared_counts: Counter[int] = Counter()
    unmatched_targets: list[str] = []

    for idx, target in enumerate(sorted(target_rows.values(), key=lambda row: row["triplet_key"]), 1):
        candidates = []
        for neutral in neutral_pool:
            pair = _neutral_match(target, neutral, require_shared2=(args.selection_mode == "all_pairs"))
            if pair is not None:
                candidates.append(pair)
        if not candidates:
            unmatched_targets.append(target["triplet_key"])
            continue
        selected = sorted(candidates, key=_rank_match)
        if args.selection_mode == "top1":
            selected = selected[:1]
            variant_name = VARIANT_NAME_TOP1
            matching_rule = NEUTRAL_MATCHING_RULE_TOP1
        else:
            variant_name = VARIANT_NAME_ALL
            matching_rule = NEUTRAL_MATCHING_RULE_ALL
        for match_idx, match in enumerate(selected, 1):
            case = _build_neutral_case(
                f"task2_neutral_{idx:06d}_{match_idx:03d}",
                target,
                match,
                variant_name=variant_name,
                matching_rule=matching_rule,
            )
            neutral_rows.append(_source_row(case, formatter.format_prompt(case)))
            match_scores.append(float(match["jel_similarity"]))
            shared_counts[int(match["shared_exact_jel_count"])] += 1

    combined_rows = list(base_rows) + neutral_rows
    write_jsonl(args.output_path, combined_rows)

    summary = {
        "base_task2_path": str(args.base_task2_path),
        "full_classification_path": str(args.full_classification_path),
        "subset_path": str(args.subset_path),
        "output_path": str(args.output_path),
        "selection_mode": args.selection_mode,
        "neutral_pool_size": len(neutral_pool),
        "base_rows": len(base_rows),
        "neutral_rows": len(neutral_rows),
        "combined_rows": len(combined_rows),
        "unique_targets": len(target_rows),
        "neutral_case_counts": dict(sorted(Counter(row["pair_case_type"] for row in neutral_rows).items())),
        "neutral_match_jaccard": {
            "mean": round(mean(match_scores), 4) if match_scores else 0.0,
            "median": round(median(match_scores), 4) if match_scores else 0.0,
            "min": round(min(match_scores), 4) if match_scores else 0.0,
            "max": round(max(match_scores), 4) if match_scores else 0.0,
        },
        "neutral_shared_exact_jel_count_distribution": dict(sorted(shared_counts.items())),
        "unmatched_targets": unmatched_targets,
        "unmatched_target_count": len(unmatched_targets),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
