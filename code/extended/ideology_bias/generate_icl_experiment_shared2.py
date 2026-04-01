"""Generate the released ICL-experiment shared2 dataset for the ideology subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from econ_eval.evaluation.data_generator import ICLExperimentCase
from econ_eval.evaluation.tasks.icl_experiment import ICLExperimentTask

from .jel import split_jel_codes
from .schemas import VALID_SIGNS, normalize_sign
from .utils import iter_jsonl, write_jsonl


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBSET_PATH = ARTIFACT_ROOT / "main_results" / "input" / "ideology_sensitive_subset_1056.jsonl"
DEFAULT_ICL_PATH = ARTIFACT_ROOT / "outputs" / "icl_experiment" / "jel_similarity_shared2.jsonl"
DEFAULT_SUMMARY_PATH = ARTIFACT_ROOT / "outputs" / "icl_experiment" / "jel_similarity_shared2_summary.json"

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


def _icl_case(case_id: str, pair: dict) -> ICLExperimentCase:
    target = pair["target"]
    example = pair["example"]
    examples = _build_examples(pair, example["sign"])
    return ICLExperimentCase(
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


def _source_row(case, experiment_name: str, prompt: str) -> dict:
    row = asdict(case)
    row["paper_id"] = case.paper_ids[-1] if case.paper_ids else None
    row["question"] = prompt
    row["answer"] = case.expected_sign
    row["example_details"] = row["examples"]
    row["experiment_name"] = experiment_name
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the released ICL experiment dataset for the ideology subset.")
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--icl-path", type=Path, default=DEFAULT_ICL_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    triplets = _usable_triplets(args.subset_path)
    pairs = _build_pair_set(triplets)

    formatter = ICLExperimentTask()
    icl_rows: list[dict] = []

    for index, pair in enumerate(pairs, 1):
        case_id_prefix = f"ideology_subset_shared2_{index:06d}"
        icl_case = _icl_case(f"icl_experiment_{case_id_prefix}", pair)
        icl_rows.append(_source_row(icl_case, "icl_experiment", formatter.format_prompt(icl_case)))

    write_jsonl(args.icl_path, icl_rows)

    case_counts = Counter(row["pair_case_type"] for row in icl_rows)
    shared_counts = Counter(row["shared_exact_jel_count"] for row in icl_rows)
    target_counter = Counter(row["triplet_key"] for row in icl_rows)
    summary = {
        "subset_path": str(args.subset_path),
        "usable_triplets": len(triplets),
        "usable_triplets_by_side": dict(sorted(Counter(row["true_side"] for row in triplets).items())),
        "icl_rows": len(icl_rows),
        "unique_targets": len(target_counter),
        "targets_with_both_example_sides": sum(1 for n in target_counter.values() if n == 2),
        "targets_with_one_example_side": sum(1 for n in target_counter.values() if n == 1),
        "icl_case_counts": dict(sorted(case_counts.items())),
        "shared_exact_jel_count_distribution": dict(sorted(shared_counts.items())),
        "mean_jel_similarity": round(sum(float(row["jel_similarity"]) for row in icl_rows) / len(icl_rows), 4) if icl_rows else 0.0,
        "mean_target_overlap_ratio": round(sum(float(row["jel_overlap_ratio"]) for row in icl_rows) / len(icl_rows), 4) if icl_rows else 0.0,
        "icl_path": str(args.icl_path),
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
