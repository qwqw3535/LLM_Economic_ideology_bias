"""Generate task3 source with shared2 target-fixed pairs plus neutral examples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from econ_eval.evaluation.data_generator import Task4Case
    from econ_eval.evaluation.tasks.task4_context_fixed import Task4ContextFixed

    from extended.ideology_bias.generate_task2_ideology_subset_shared2_with_neutral import (
        DEFAULT_FULL_CLASSIFICATION_PATH,
        DEFAULT_SUBSET_PATH,
        NEUTRAL_MATCHING_RULE,
        VARIANT_NAME,
        _neutral_match,
        _neutral_pool_rows,
        _rank_match,
        _safe_text,
    )
    from extended.ideology_bias.utils import read_jsonl, write_jsonl
else:
    from econ_eval.evaluation.data_generator import Task4Case
    from econ_eval.evaluation.tasks.task4_context_fixed import Task4ContextFixed

    from .generate_task2_ideology_subset_shared2_with_neutral import (
        DEFAULT_FULL_CLASSIFICATION_PATH,
        DEFAULT_SUBSET_PATH,
        NEUTRAL_MATCHING_RULE,
        VARIANT_NAME,
        _neutral_match,
        _neutral_pool_rows,
        _rank_match,
        _safe_text,
    )
    from .utils import read_jsonl, write_jsonl


DEFAULT_BASE_TASK3_PATH = Path("data/task3_ideology_subset_shared2_all_pairs_target_fixed.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/task3_ideology_subset_shared2_all_pairs_target_fixed_with_neutral.jsonl")
DEFAULT_SUMMARY_PATH = Path("extended/classification_results/task3_ideology_subset_shared2_all_pairs_target_fixed_with_neutral_summary.json")


def _build_neutral_case(case_id: str, target: dict, match: dict) -> Task4Case:
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
    return Task4Case(
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
        variant=VARIANT_NAME,
        pair_case_type=f"neutral-{'lib' if target.get('target_side') == 'liberal' else 'cons'}",
        matching_rule=NEUTRAL_MATCHING_RULE,
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


def _source_row(case: Task4Case, prompt: str) -> dict:
    row = asdict(case)
    row["paper_id"] = case.paper_ids[-1] if case.paper_ids else None
    row["question"] = prompt
    row["answer"] = case.expected_sign
    row["example_details"] = row["examples"]
    row["task_name"] = "task3"
    row["paper_task"] = "task3"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Add neutral top-1 matches to task3 shared2 all-pairs target-fixed source.")
    parser.add_argument("--base-task3-path", type=Path, default=DEFAULT_BASE_TASK3_PATH)
    parser.add_argument("--full-classification-path", type=Path, default=DEFAULT_FULL_CLASSIFICATION_PATH)
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_task3_path)
    subset_triplet_keys = {row["triplet_key"] for row in read_jsonl(args.subset_path)}
    target_rows: dict[str, dict] = {}
    for row in base_rows:
        target_rows.setdefault(row["triplet_key"], row)

    neutral_pool = _neutral_pool_rows(args.full_classification_path, subset_triplet_keys)
    formatter = Task4ContextFixed()

    neutral_rows: list[dict] = []
    match_scores: list[float] = []
    unmatched_targets: list[str] = []
    shared_count_dist: dict[int, int] = {}

    for idx, target in enumerate(sorted(target_rows.values(), key=lambda row: row["triplet_key"]), 1):
        candidates = []
        for neutral in neutral_pool:
            pair = _neutral_match(target, neutral)
            if pair is not None:
                candidates.append(pair)
        if not candidates:
            unmatched_targets.append(target["triplet_key"])
            continue
        best = sorted(candidates, key=_rank_match)[0]
        case = _build_neutral_case(f"task3_neutral_{idx:06d}", target, best)
        neutral_rows.append(_source_row(case, formatter.format_prompt(case)))
        match_scores.append(float(best["jel_similarity"]))
        count = int(best["shared_exact_jel_count"])
        shared_count_dist[count] = shared_count_dist.get(count, 0) + 1

    combined_rows = list(base_rows) + neutral_rows
    write_jsonl(args.output_path, combined_rows)

    summary = {
        "base_task3_path": str(args.base_task3_path),
        "full_classification_path": str(args.full_classification_path),
        "subset_path": str(args.subset_path),
        "output_path": str(args.output_path),
        "neutral_pool_size": len(neutral_pool),
        "base_rows": len(base_rows),
        "neutral_rows": len(neutral_rows),
        "combined_rows": len(combined_rows),
        "unique_targets": len(target_rows),
        "neutral_case_counts": {
            "neutral-lib": sum(1 for row in neutral_rows if row["pair_case_type"] == "neutral-lib"),
            "neutral-cons": sum(1 for row in neutral_rows if row["pair_case_type"] == "neutral-cons"),
        },
        "neutral_match_jaccard": {
            "mean": round(sum(match_scores) / len(match_scores), 4) if match_scores else 0.0,
            "median": round(sorted(match_scores)[len(match_scores) // 2], 4) if match_scores else 0.0,
            "min": round(min(match_scores), 4) if match_scores else 0.0,
            "max": round(max(match_scores), 4) if match_scores else 0.0,
        },
        "neutral_shared_exact_jel_count_distribution": dict(sorted(shared_count_dist.items())),
        "unmatched_targets": unmatched_targets,
        "unmatched_target_count": len(unmatched_targets),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
