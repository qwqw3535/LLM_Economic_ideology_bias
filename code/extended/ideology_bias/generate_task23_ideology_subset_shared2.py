"""Generate task2/task3 shared2 datasets for the current ideology subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from econ_eval.evaluation.data_generator import Task3Case, Task4Case
from econ_eval.evaluation.tasks.task3_context_to_fixed import Task3ContextTOFixed
from econ_eval.evaluation.tasks.task4_context_fixed import Task4ContextFixed

from .jel import split_jel_codes
from .schemas import VALID_SIGNS, normalize_sign
from .utils import iter_jsonl, write_jsonl


DEFAULT_SUBSET_PATH = Path("extended/classification_results/ideology_triplet_subset_current.jsonl")
DEFAULT_TASK2_PATH = Path("data/task2_ideology_subset_jel_similarity_side_capped_jaccard05_shared2.jsonl")
DEFAULT_TASK3_PATH = Path("data/task3_ideology_subset_jel_similarity_side_capped_jaccard05_shared2.jsonl")
DEFAULT_SUMMARY_PATH = Path("extended/classification_results/task23_ideology_subset_shared2_summary.json")

MATCHING_RULE = "jaccard_jel_similarity_shared2"


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _numeric_sort(value: object) -> tuple[int, str]:
    text = str(value or "").strip()
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def _informative_jel_codes(codes: list[str]) -> list[str]:
    return [code for code in codes if code and code[0].isalpha() and len(code) >= 2]


def _side_from_sign(sign: str, liberal_sign: str, conservative_sign: str) -> str | None:
    if sign == liberal_sign and sign != conservative_sign:
        return "liberal"
    if sign == conservative_sign and sign != liberal_sign:
        return "conservative"
    return None


def _opposite_side(side: str) -> str:
    return "conservative" if side == "liberal" else "liberal"


def _short_side(side: str) -> str:
    return "lib" if side == "liberal" else "cons"


def _usable_triplets(subset_path: Path) -> list[dict]:
    usable: list[dict] = []
    for row_index, row in enumerate(iter_jsonl(subset_path)):
        sign = normalize_sign(row.get("sign"))
        liberal_sign = normalize_sign(row.get("lib_vote"))
        conservative_sign = normalize_sign(row.get("con_vote"))
        if sign not in VALID_SIGNS:
            continue
        if liberal_sign not in VALID_SIGNS or conservative_sign not in VALID_SIGNS:
            continue
        if liberal_sign == conservative_sign:
            continue
        true_side = _side_from_sign(sign, liberal_sign, conservative_sign)
        if true_side is None:
            continue

        exact_jel_codes = split_jel_codes(row.get("jel_codes"))
        informative_jel_codes = _informative_jel_codes(exact_jel_codes)
        if len(informative_jel_codes) < 2:
            continue

        usable.append(
            {
                "triplet_key": row.get("triplet_key"),
                "record_id": f"{row.get('triplet_key')}|{row_index}",
                "paper_id": _safe_text(row.get("paper_id")),
                "title": row.get("title"),
                "author": row.get("author"),
                "publication_year": row.get("publication_year"),
                "published_venue": row.get("published_venue"),
                "jel_codes": row.get("jel_codes"),
                "paper_url": row.get("paper_url"),
                "treatment": row.get("treatment"),
                "outcome": row.get("outcome"),
                "sign": sign,
                "context": row.get("context"),
                "identification_methods": row.get("identification_methods"),
                "exact_jel_codes": exact_jel_codes,
                "exact_jel_set": set(exact_jel_codes),
                "informative_jel_codes": informative_jel_codes,
                "informative_jel_set": set(informative_jel_codes),
                "informative_jel_count": len(informative_jel_codes),
                "liberal_preferred_sign": liberal_sign,
                "conservative_preferred_sign": conservative_sign,
                "true_side": true_side,
                "false_side": _opposite_side(true_side),
                "false_sign": conservative_sign if true_side == "liberal" else liberal_sign,
            }
        )
    return usable


def _eligible_pair(target: dict, example: dict) -> dict | None:
    if target["paper_id"] == example["paper_id"]:
        return None
    target_code_set = target["informative_jel_set"]
    example_code_set = example["informative_jel_set"]
    if not target_code_set or not example_code_set:
        return None
    shared_codes = sorted(target_code_set & example_code_set)
    shared_count = len(shared_codes)
    if shared_count < 2:
        return None
    union_count = len(target_code_set | example_code_set)
    jel_similarity = shared_count / union_count if union_count else 0.0
    if jel_similarity < 0.5:
        return None
    target_overlap_ratio = shared_count / len(target_code_set)
    example_overlap_ratio = shared_count / len(example_code_set)
    return {
        "target": target,
        "example": example,
        "target_match_jel_codes": list(target["informative_jel_codes"]),
        "example_match_jel_codes": list(example["informative_jel_codes"]),
        "shared_exact_jel_codes": shared_codes,
        "shared_exact_jel_count": shared_count,
        "target_exact_jel_count": len(target_code_set),
        "example_exact_jel_count": len(example_code_set),
        "jel_overlap_ratio": target_overlap_ratio,
        "example_overlap_ratio": example_overlap_ratio,
        "union_exact_jel_count": union_count,
        "jel_similarity": jel_similarity,
        "selection_score": jel_similarity,
    }


def _rank_candidate(pair: dict) -> tuple:
    example = pair["example"]
    target = pair["target"]
    return (
        -pair["selection_score"],
        -pair["shared_exact_jel_count"],
        abs(example["informative_jel_count"] - target["informative_jel_count"]),
        _numeric_sort(example["paper_id"]),
        _safe_text(example["treatment"]).lower(),
        _safe_text(example["outcome"]).lower(),
    )


def _build_pair_set(triplets: list[dict]) -> list[dict]:
    by_target_and_side: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for target in triplets:
        for example in triplets:
            pair = _eligible_pair(target, example)
            if pair is not None:
                by_target_and_side[(target["record_id"], example["true_side"])].append(pair)

    side_capped: list[dict] = []
    for candidates in by_target_and_side.values():
        side_capped.append(sorted(candidates, key=_rank_candidate)[0])

    return sorted(
        side_capped,
        key=lambda pair: (
            _numeric_sort(pair["target"]["paper_id"]),
            _safe_text(pair["target"]["treatment"]).lower(),
            _safe_text(pair["target"]["outcome"]).lower(),
            _numeric_sort(pair["example"]["paper_id"]),
            _safe_text(pair["example"]["treatment"]).lower(),
            _safe_text(pair["example"]["outcome"]).lower(),
        ),
    )


def _build_examples(pair: dict, displayed_sign: str) -> list[dict]:
    example = pair["example"]
    return [
        {
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
            "sign": displayed_sign,
            "original_sign": example.get("sign"),
            "economic_liberal_preferred_sign": example.get("liberal_preferred_sign"),
            "economic_conservative_preferred_sign": example.get("conservative_preferred_sign"),
            "example_true_side": example.get("true_side"),
            "example_false_side": example.get("false_side"),
            "example_exact_jel_codes": list(pair.get("example_match_jel_codes") or []),
            "shared_exact_jel_codes": list(pair.get("shared_exact_jel_codes") or []),
            "shared_exact_jel_count": pair.get("shared_exact_jel_count"),
            "jel_overlap_ratio": round(pair.get("jel_overlap_ratio", 0.0), 4),
            "example_overlap_ratio": round(pair.get("example_overlap_ratio", 0.0), 4),
            "jel_similarity": round(pair.get("jel_similarity", 0.0), 4),
            "union_exact_jel_count": pair.get("union_exact_jel_count"),
            "selection_score": round(pair.get("selection_score", 0.0), 4),
            "triplet_key": example.get("triplet_key"),
        }
    ]


def _task2_case(case_id: str, pair: dict) -> Task3Case:
    target = pair["target"]
    example = pair["example"]
    examples = _build_examples(pair, example["sign"])
    return Task3Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=examples,
        test_context=target["context"],
        expected_sign=target["sign"],
        paper_ids=[example["paper_id"], target["paper_id"]],
        avg_similarity=round(pair["jel_similarity"], 4),
        sign_differs=example["sign"] != target["sign"],
        context=target["context"],
        sign=target["sign"],
        title=target.get("title"),
        author=target.get("author"),
        publication_year=target.get("publication_year"),
        published_venue=target.get("published_venue"),
        jel_codes=target.get("jel_codes"),
        paper_url=target.get("paper_url"),
        variant="ideology_subset_shared2",
        pair_case_type=f"{_short_side(example['true_side'])}-{_short_side(target['true_side'])}",
        matching_rule=MATCHING_RULE,
        target_side=target["true_side"],
        example_true_side=example["true_side"],
        example_false_side=example["false_side"],
        target_exact_jel_codes=list(pair["target_match_jel_codes"]),
        example_exact_jel_codes=list(pair["example_match_jel_codes"]),
        shared_exact_jel_codes=list(pair["shared_exact_jel_codes"]),
        shared_exact_jel_count=pair["shared_exact_jel_count"],
        target_exact_jel_count=pair["target_exact_jel_count"],
        jel_overlap_ratio=round(pair["jel_overlap_ratio"], 4),
        example_overlap_ratio=round(pair["example_overlap_ratio"], 4),
        jel_similarity=round(pair["jel_similarity"], 4),
        union_exact_jel_count=pair["union_exact_jel_count"],
        selection_score=round(pair["selection_score"], 4),
        different_paper=True,
        triplet_key=target["triplet_key"],
    )


def _task3_case(case_id: str, pair: dict) -> Task4Case:
    target = pair["target"]
    example = pair["example"]
    examples = _build_examples(pair, example["false_sign"])
    displayed_side = example["false_side"]
    return Task4Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=examples,
        test_context=target["context"],
        expected_sign=target["sign"],
        paper_ids=[example["paper_id"], target["paper_id"]],
        avg_similarity=round(pair["jel_similarity"], 4),
        sign_differs=example["false_sign"] != target["sign"],
        context=target["context"],
        sign=target["sign"],
        title=target.get("title"),
        author=target.get("author"),
        publication_year=target.get("publication_year"),
        published_venue=target.get("published_venue"),
        jel_codes=target.get("jel_codes"),
        paper_url=target.get("paper_url"),
        variant="ideology_subset_shared2",
        pair_case_type=f"{_short_side(displayed_side)}-{_short_side(target['true_side'])}",
        matching_rule=MATCHING_RULE,
        target_side=target["true_side"],
        example_true_side=example["true_side"],
        example_false_side=displayed_side,
        target_exact_jel_codes=list(pair["target_match_jel_codes"]),
        example_exact_jel_codes=list(pair["example_match_jel_codes"]),
        shared_exact_jel_codes=list(pair["shared_exact_jel_codes"]),
        shared_exact_jel_count=pair["shared_exact_jel_count"],
        target_exact_jel_count=pair["target_exact_jel_count"],
        jel_overlap_ratio=round(pair["jel_overlap_ratio"], 4),
        example_overlap_ratio=round(pair["example_overlap_ratio"], 4),
        jel_similarity=round(pair["jel_similarity"], 4),
        union_exact_jel_count=pair["union_exact_jel_count"],
        selection_score=round(pair["selection_score"], 4),
        different_paper=True,
        triplet_key=target["triplet_key"],
    )


def _source_row(case, task_name: str, prompt: str) -> dict:
    row = asdict(case)
    row["paper_id"] = case.paper_ids[-1] if case.paper_ids else None
    row["question"] = prompt
    row["answer"] = case.expected_sign
    row["example_details"] = row["examples"]
    row["task_name"] = task_name
    row["paper_task"] = task_name
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared2 task2/task3 datasets for ideology subset.")
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--task2-path", type=Path, default=DEFAULT_TASK2_PATH)
    parser.add_argument("--task3-path", type=Path, default=DEFAULT_TASK3_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    triplets = _usable_triplets(args.subset_path)
    pairs = _build_pair_set(triplets)

    task2_formatter = Task3ContextTOFixed()
    task3_formatter = Task4ContextFixed()
    task2_rows: list[dict] = []
    task3_rows: list[dict] = []

    for index, pair in enumerate(pairs, 1):
        case_id_prefix = f"ideology_subset_shared2_{index:06d}"
        task2_case = _task2_case(f"task2_{case_id_prefix}", pair)
        task3_case = _task3_case(f"task3_{case_id_prefix}", pair)
        task2_rows.append(_source_row(task2_case, "task2", task2_formatter.format_prompt(task2_case)))
        task3_rows.append(_source_row(task3_case, "task3", task3_formatter.format_prompt(task3_case)))

    write_jsonl(args.task2_path, task2_rows)
    write_jsonl(args.task3_path, task3_rows)

    task2_case_counts = Counter(row["pair_case_type"] for row in task2_rows)
    task3_case_counts = Counter(row["pair_case_type"] for row in task3_rows)
    shared_counts = Counter(row["shared_exact_jel_count"] for row in task2_rows)
    target_counter = Counter(row["triplet_key"] for row in task2_rows)
    summary = {
        "subset_path": str(args.subset_path),
        "usable_triplets": len(triplets),
        "usable_triplets_by_side": dict(sorted(Counter(row["true_side"] for row in triplets).items())),
        "task2_rows": len(task2_rows),
        "task3_rows": len(task3_rows),
        "unique_targets": len(target_counter),
        "targets_with_both_example_sides": sum(1 for n in target_counter.values() if n == 2),
        "targets_with_one_example_side": sum(1 for n in target_counter.values() if n == 1),
        "task2_case_counts": dict(sorted(task2_case_counts.items())),
        "task3_case_counts": dict(sorted(task3_case_counts.items())),
        "shared_exact_jel_count_distribution": dict(sorted(shared_counts.items())),
        "mean_jel_similarity": round(sum(float(row["jel_similarity"]) for row in task2_rows) / len(task2_rows), 4) if task2_rows else 0.0,
        "mean_target_overlap_ratio": round(sum(float(row["jel_overlap_ratio"]) for row in task2_rows) / len(task2_rows), 4) if task2_rows else 0.0,
        "task2_path": str(args.task2_path),
        "task3_path": str(args.task3_path),
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
