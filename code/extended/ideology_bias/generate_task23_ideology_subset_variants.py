"""Generate target-fixed task2/task3 variants for the ideology subset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from econ_eval.evaluation.data_generator import Task3Case, Task4Case
    from econ_eval.evaluation.tasks.task3_context_to_fixed import Task3ContextTOFixed
    from econ_eval.evaluation.tasks.task4_context_fixed import Task4ContextFixed

    from extended.ideology_bias.jel import split_jel_codes
    from extended.ideology_bias.schemas import VALID_SIGNS, normalize_sign
    from extended.ideology_bias.utils import iter_jsonl, write_jsonl
else:
    from econ_eval.evaluation.data_generator import Task3Case, Task4Case
    from econ_eval.evaluation.tasks.task3_context_to_fixed import Task3ContextTOFixed
    from econ_eval.evaluation.tasks.task4_context_fixed import Task4ContextFixed

    from .jel import split_jel_codes
    from .schemas import VALID_SIGNS, normalize_sign
    from .utils import iter_jsonl, write_jsonl


DEFAULT_SUBSET_PATH = Path("extended/classification_results/ideology_triplet_subset_current.jsonl")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_SUMMARY_PATH = Path("extended/classification_results/task23_ideology_subset_variant_summaries.json")

VARIANT_SPECS = {
    "shared2_all_pairs_target_fixed": {
        "matching_rule": "jaccard_jel_similarity_shared2_all_pairs_target_fixed",
        "match_mode": "shared2",
        "selection_mode": "all_pairs",
        "task2_path": "task2_ideology_subset_shared2_all_pairs_target_fixed.jsonl",
        "task3_path": "task3_ideology_subset_shared2_all_pairs_target_fixed.jsonl",
    },
    "exact50_side_capped_target_fixed": {
        "matching_rule": "exact50_side_capped_target_fixed",
        "match_mode": "exact50",
        "selection_mode": "side_capped",
        "task2_path": "task2_ideology_subset_exact50_side_capped_target_fixed.jsonl",
        "task3_path": "task3_ideology_subset_exact50_side_capped_target_fixed.jsonl",
    },
}


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
                "exact_jel_count": len(exact_jel_codes),
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


def _eligible_pair(target: dict, example: dict, match_mode: str) -> dict | None:
    if target["paper_id"] == example["paper_id"]:
        return None

    if match_mode == "shared2":
        target_code_set = target["informative_jel_set"]
        example_code_set = example["informative_jel_set"]
        target_code_list = target["informative_jel_codes"]
        example_code_list = example["informative_jel_codes"]
    elif match_mode == "exact50":
        target_code_set = target["exact_jel_set"]
        example_code_set = example["exact_jel_set"]
        target_code_list = target["exact_jel_codes"]
        example_code_list = example["exact_jel_codes"]
    else:
        raise ValueError(f"Unknown match_mode: {match_mode}")

    if not target_code_set or not example_code_set:
        return None

    shared_codes = sorted(target_code_set & example_code_set)
    shared_count = len(shared_codes)
    if shared_count == 0:
        return None

    target_overlap_ratio = shared_count / len(target_code_set)
    example_overlap_ratio = shared_count / len(example_code_set)
    union_exact_jel_count = len(target_code_set | example_code_set)
    jel_similarity = shared_count / union_exact_jel_count if union_exact_jel_count else 0.0

    if match_mode == "shared2":
        if shared_count < 2 or jel_similarity < 0.5:
            return None
        selection_score = jel_similarity
    else:
        if target_overlap_ratio < 0.5:
            return None
        selection_score = target_overlap_ratio

    return {
        "target": target,
        "example": example,
        "target_match_jel_codes": list(target_code_list),
        "example_match_jel_codes": list(example_code_list),
        "shared_exact_jel_codes": shared_codes,
        "shared_exact_jel_count": shared_count,
        "target_exact_jel_count": len(target_code_set),
        "example_exact_jel_count": len(example_code_set),
        "jel_overlap_ratio": target_overlap_ratio,
        "example_overlap_ratio": example_overlap_ratio,
        "union_exact_jel_count": union_exact_jel_count,
        "jel_similarity": jel_similarity,
        "selection_score": selection_score,
    }


def _rank_pair(pair: dict) -> tuple:
    example = pair["example"]
    target = pair["target"]
    return (
        -pair["selection_score"],
        -pair["shared_exact_jel_count"],
        abs(example["exact_jel_count"] - target["exact_jel_count"]),
        _numeric_sort(example["paper_id"]),
        _safe_text(example["treatment"]).lower(),
        _safe_text(example["outcome"]).lower(),
    )


def _pair_sort_key(pair: dict) -> tuple:
    target = pair["target"]
    example = pair["example"]
    return (
        _numeric_sort(target["paper_id"]),
        _safe_text(target["treatment"]).lower(),
        _safe_text(target["outcome"]).lower(),
        _numeric_sort(example["paper_id"]),
        _safe_text(example["treatment"]).lower(),
        _safe_text(example["outcome"]).lower(),
        example["true_side"],
    )


def _select_pairs(triplets: list[dict], match_mode: str, selection_mode: str) -> list[dict]:
    all_pairs: list[dict] = []
    by_target_and_side: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for target in triplets:
        for example in triplets:
            pair = _eligible_pair(target, example, match_mode)
            if pair is None:
                continue
            all_pairs.append(pair)
            by_target_and_side[(target["record_id"], example["true_side"])].append(pair)

    if selection_mode == "all_pairs":
        return sorted(all_pairs, key=_pair_sort_key)
    if selection_mode == "side_capped":
        capped = [sorted(candidates, key=_rank_pair)[0] for candidates in by_target_and_side.values()]
        return sorted(capped, key=_pair_sort_key)
    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def _target_fixed_pairs(pairs: list[dict]) -> list[dict]:
    target_side_hits: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        target_side_hits[pair["target"]["triplet_key"]].add(pair["example"]["true_side"])

    fixed_targets = {
        triplet_key
        for triplet_key, sides in target_side_hits.items()
        if {"liberal", "conservative"}.issubset(sides)
    }
    return [pair for pair in pairs if pair["target"]["triplet_key"] in fixed_targets]


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


def _task2_case(case_id: str, variant_name: str, matching_rule: str, pair: dict) -> Task3Case:
    target = pair["target"]
    example = pair["example"]
    return Task3Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=_build_examples(pair, example["sign"]),
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
        variant=variant_name,
        pair_case_type=f"{_short_side(example['true_side'])}-{_short_side(target['true_side'])}",
        matching_rule=matching_rule,
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


def _task3_case(case_id: str, variant_name: str, matching_rule: str, pair: dict) -> Task4Case:
    target = pair["target"]
    example = pair["example"]
    displayed_side = example["false_side"]
    return Task4Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=_build_examples(pair, example["false_sign"]),
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
        variant=variant_name,
        pair_case_type=f"{_short_side(displayed_side)}-{_short_side(target['true_side'])}",
        matching_rule=matching_rule,
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


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "median": median(values),
        "mean": round(mean(values), 4),
        "max": max(values),
    }


def _summarize_variant(variant_name: str, matching_rule: str, pairs: list[dict], task2_rows: list[dict], task3_rows: list[dict]) -> dict:
    task2_case_counts = Counter(row["pair_case_type"] for row in task2_rows)
    task3_case_counts = Counter(row["pair_case_type"] for row in task3_rows)
    target_counter = Counter(pair["target"]["triplet_key"] for pair in pairs)
    example_counter = Counter(pair["example"]["triplet_key"] for pair in pairs)
    target_by_side: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        target_by_side[pair["target"]["true_side"]].add(pair["target"]["triplet_key"])

    lib_targets = target_by_side.get("liberal", set())
    con_targets = target_by_side.get("conservative", set())
    selection_scores = [float(pair["selection_score"]) for pair in pairs]
    jaccards = [float(pair["jel_similarity"]) for pair in pairs]
    target_overlaps = [float(pair["jel_overlap_ratio"]) for pair in pairs]
    shared_counts = Counter(int(pair["shared_exact_jel_count"]) for pair in pairs)
    target_jel_count_dist = Counter(int(pair["target_exact_jel_count"]) for pair in pairs)
    example_jel_count_dist = Counter(int(pair["example_exact_jel_count"]) for pair in pairs)

    return {
        "variant": variant_name,
        "matching_rule": matching_rule,
        "task2_rows": len(task2_rows),
        "task3_rows": len(task3_rows),
        "task2_case_counts": dict(sorted(task2_case_counts.items())),
        "task3_case_counts": dict(sorted(task3_case_counts.items())),
        "unique_targets_total": len(target_counter),
        "unique_targets_liberal": len(lib_targets),
        "unique_targets_conservative": len(con_targets),
        "target_fixed_total": len(target_counter),
        "target_counts_per_target": _distribution(list(target_counter.values())),
        "example_reuse_counts": _distribution(list(example_counter.values())),
        "selection_score": {
            "mean": round(mean(selection_scores), 4) if selection_scores else 0.0,
            "median": round(median(selection_scores), 4) if selection_scores else 0.0,
        },
        "jel_similarity": {
            "mean": round(mean(jaccards), 4) if jaccards else 0.0,
            "median": round(median(jaccards), 4) if jaccards else 0.0,
        },
        "target_overlap_ratio": {
            "mean": round(mean(target_overlaps), 4) if target_overlaps else 0.0,
            "median": round(median(target_overlaps), 4) if target_overlaps else 0.0,
        },
        "shared_exact_jel_count_distribution": dict(sorted(shared_counts.items())),
        "target_exact_jel_count_distribution": dict(sorted(target_jel_count_dist.items())),
        "example_exact_jel_count_distribution": dict(sorted(example_jel_count_dist.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate target-fixed task23 variants for the ideology subset.")
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    triplets = _usable_triplets(args.subset_path)
    task2_formatter = Task3ContextTOFixed()
    task3_formatter = Task4ContextFixed()

    summaries: dict[str, dict] = {}
    for variant_name, spec in VARIANT_SPECS.items():
        selected_pairs = _select_pairs(
            triplets,
            match_mode=spec["match_mode"],
            selection_mode=spec["selection_mode"],
        )
        target_fixed_pairs = _target_fixed_pairs(selected_pairs)

        task2_rows: list[dict] = []
        task3_rows: list[dict] = []
        for index, pair in enumerate(target_fixed_pairs, 1):
            case_id_prefix = f"{variant_name}_{index:06d}"
            task2_case = _task2_case(f"task2_{case_id_prefix}", variant_name, spec["matching_rule"], pair)
            task3_case = _task3_case(f"task3_{case_id_prefix}", variant_name, spec["matching_rule"], pair)
            task2_rows.append(_source_row(task2_case, "task2", task2_formatter.format_prompt(task2_case)))
            task3_rows.append(_source_row(task3_case, "task3", task3_formatter.format_prompt(task3_case)))

        task2_path = args.output_dir / spec["task2_path"]
        task3_path = args.output_dir / spec["task3_path"]
        write_jsonl(task2_path, task2_rows)
        write_jsonl(task3_path, task3_rows)

        summary = _summarize_variant(variant_name, spec["matching_rule"], target_fixed_pairs, task2_rows, task3_rows)
        summary["task2_path"] = str(task2_path)
        summary["task3_path"] = str(task3_path)
        summaries[variant_name] = summary

    payload = {
        "subset_path": str(args.subset_path),
        "usable_triplets": len(triplets),
        "usable_triplets_by_side": dict(sorted(Counter(row["true_side"] for row in triplets).items())),
        "variants": summaries,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
