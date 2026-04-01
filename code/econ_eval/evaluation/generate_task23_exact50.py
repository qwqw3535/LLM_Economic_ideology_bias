"""Generate dual exact50 task2/task3 datasets and audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from extended.ideology_bias.jel import split_jel_codes
from extended.ideology_bias.schemas import VALID_SIGNS, normalize_sign
from extended.ideology_bias.utils import make_triplet_key, read_jsonl, write_csv_rows, write_jsonl

from .data_generator import Task3Case, Task4Case
from .tasks.task3_context_to_fixed import Task3ContextTOFixed
from .tasks.task4_context_fixed import Task4ContextFixed


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "extended" / "ideology_bias_outputs"
AUDITS_DIR = OUTPUT_DIR / "audits"
REPORTS_DIR = OUTPUT_DIR / "reports"

DEFAULT_CORPUS_PATH = DATA_DIR / "causal_triplets.jsonl"
DEFAULT_CLASSIFICATION_PATH = (
    REPO_ROOT
    / "extended"
    / "classification_results"
    / "causal_triplets_gpt-5-mini_classified_merged.jsonl"
)

VARIANT_OUTPUTS = {
    "side_capped": {
        "task2_path": DATA_DIR / "task2_exact50_side_capped.jsonl",
        "task3_path": DATA_DIR / "task3_exact50_side_capped.jsonl",
        "catalog_path": AUDITS_DIR / "task23_exact50_side_capped_match_catalog.csv",
        "verification_path": REPORTS_DIR / "task23_exact50_side_capped_verification.md",
    },
    "side_capped_jel_similarity": {
        "task2_path": DATA_DIR / "task2_jel_similarity_side_capped.jsonl",
        "task3_path": DATA_DIR / "task3_jel_similarity_side_capped.jsonl",
        "catalog_path": AUDITS_DIR / "task23_jel_similarity_side_capped_match_catalog.csv",
        "verification_path": REPORTS_DIR / "task23_jel_similarity_side_capped_verification.md",
    },
    "all_pairs": {
        "task2_path": DATA_DIR / "task2_exact50_all_pairs.jsonl",
        "task3_path": DATA_DIR / "task3_exact50_all_pairs.jsonl",
        "catalog_path": AUDITS_DIR / "task23_exact50_all_pairs_match_catalog.csv",
        "verification_path": REPORTS_DIR / "task23_exact50_all_pairs_verification.md",
    },
}

SUMMARY_PATH = AUDITS_DIR / "task23_exact50_generation_summary.json"
MATCHING_RULE = "exact_jel_overlap_50"
JEL_SIMILARITY_RULE = "jaccard_jel_similarity"
VALID_SIDE_SIGNS = set(VALID_SIGNS)


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


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _numeric_sort(value: object) -> tuple[int, str]:
    text = str(value or "").strip()
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def _informative_jel_codes(codes: list[str]) -> list[str]:
    return [code for code in codes if code and code[0].isalpha() and len(code) >= 2]


def _usable_triplets(corpus_path: Path, classification_path: Path) -> list[dict]:
    usable: list[dict] = []
    corpus_lookup = {
        make_triplet_key(row.get("paper_id"), row.get("treatment"), row.get("outcome")): row
        for row in read_jsonl(corpus_path)
    }
    seen_rows: set[tuple] = set()
    for row_index, classified in enumerate(read_jsonl(classification_path)):
        triplet_key = make_triplet_key(classified.get("paper_id"), classified.get("treatment"), classified.get("outcome"))
        row = corpus_lookup.get(triplet_key, classified)
        ideology = (
            classified.get("classification", {})
            .get("labels", {})
            .get("ideology_preference", {})
        )
        if not ideology.get("is_ideologically_sensitive"):
            continue
        sign = normalize_sign(row.get("sign"))
        liberal_sign = normalize_sign(ideology.get("economic_liberal_expected_sign"))
        conservative_sign = normalize_sign(ideology.get("economic_conservative_expected_sign"))
        if sign not in VALID_SIDE_SIGNS:
            continue
        if liberal_sign not in VALID_SIDE_SIGNS or conservative_sign not in VALID_SIDE_SIGNS:
            continue
        if liberal_sign == conservative_sign:
            continue
        true_side = _side_from_sign(sign, liberal_sign, conservative_sign)
        if true_side is None:
            continue
        false_side = _opposite_side(true_side)
        false_sign = conservative_sign if true_side == "liberal" else liberal_sign
        exact_jel_codes = split_jel_codes(row.get("jel_codes"))
        if not exact_jel_codes:
            continue
        informative_jel_codes = _informative_jel_codes(exact_jel_codes)
        dedupe_key = (
            _safe_text(row.get("paper_id")),
            _safe_text(row.get("treatment")),
            _safe_text(row.get("outcome")),
            _safe_text(row.get("context")),
            sign,
            _safe_text(row.get("jel_codes")),
            liberal_sign,
            conservative_sign,
        )
        if dedupe_key in seen_rows:
            continue
        seen_rows.add(dedupe_key)
        usable.append(
            {
                "triplet_key": triplet_key,
                "record_id": f"{triplet_key}|{row_index}",
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
                "exact_jel_count": len(exact_jel_codes),
                "exact_jel_set": set(exact_jel_codes),
                "informative_jel_codes": informative_jel_codes,
                "informative_jel_count": len(informative_jel_codes),
                "informative_jel_set": set(informative_jel_codes),
                "liberal_preferred_sign": liberal_sign,
                "conservative_preferred_sign": conservative_sign,
                "true_side": true_side,
                "false_side": false_side,
                "false_sign": false_sign,
            }
        )
    return usable


def _eligible_pair(target: dict, example: dict, mode: str) -> dict | None:
    if target["paper_id"] == example["paper_id"]:
        return None
    if mode == "jaccard":
        target_code_set = target["informative_jel_set"]
        example_code_set = example["informative_jel_set"]
        target_code_list = target["informative_jel_codes"]
        example_code_list = example["informative_jel_codes"]
    else:
        target_code_set = target["exact_jel_set"]
        example_code_set = example["exact_jel_set"]
        target_code_list = target["exact_jel_codes"]
        example_code_list = example["exact_jel_codes"]
    if not target_code_set or not example_code_set:
        return None
    shared_codes = sorted(target_code_set & example_code_set)
    if not shared_codes:
        return None
    shared_count = len(shared_codes)
    target_overlap_ratio = shared_count / len(target_code_set)
    example_overlap_ratio = shared_count / len(example_code_set)
    union_exact_jel_count = len(target_code_set | example_code_set)
    jel_similarity = shared_count / union_exact_jel_count if union_exact_jel_count else 0.0
    if mode == "exact50" and target_overlap_ratio < 0.5:
        return None
    selection_score = target_overlap_ratio if mode == "exact50" else jel_similarity
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


def _rank_side_capped_candidate(pair: dict) -> tuple:
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
    )


def _build_pair_sets(triplets: list[dict]) -> dict[str, list[dict]]:
    all_pairs: list[dict] = []
    by_target_and_side: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_target_and_side_similarity: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for target in triplets:
        for example in triplets:
            exact_pair = _eligible_pair(target, example, mode="exact50")
            if exact_pair is not None:
                all_pairs.append(exact_pair)
                by_target_and_side[(target["record_id"], example["true_side"])].append(exact_pair)
            similarity_pair = _eligible_pair(target, example, mode="jaccard")
            if similarity_pair is not None:
                by_target_and_side_similarity[(target["record_id"], example["true_side"])].append(similarity_pair)

    side_capped: list[dict] = []
    for candidates in by_target_and_side.values():
        side_capped.append(sorted(candidates, key=_rank_side_capped_candidate)[0])

    side_capped_similarity: list[dict] = []
    for candidates in by_target_and_side_similarity.values():
        side_capped_similarity.append(sorted(candidates, key=_rank_side_capped_candidate)[0])

    return {
        "all_pairs": sorted(all_pairs, key=_pair_sort_key),
        "side_capped": sorted(side_capped, key=_pair_sort_key),
        "side_capped_jel_similarity": sorted(side_capped_similarity, key=_pair_sort_key),
    }


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


def _task2_case(case_id: str, variant: str, pair: dict) -> Task3Case:
    target = pair["target"]
    example = pair["example"]
    examples = _build_examples(pair, example["sign"])
    matching_rule = JEL_SIMILARITY_RULE if variant == "side_capped_jel_similarity" else MATCHING_RULE
    return Task3Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=examples,
        test_context=target["context"],
        expected_sign=target["sign"],
        paper_ids=[example["paper_id"], target["paper_id"]],
        avg_similarity=round(pair["jel_overlap_ratio"], 4),
        sign_differs=example["sign"] != target["sign"],
        context=target["context"],
        sign=target["sign"],
        title=target.get("title"),
        author=target.get("author"),
        publication_year=target.get("publication_year"),
        published_venue=target.get("published_venue"),
        jel_codes=target.get("jel_codes"),
        paper_url=target.get("paper_url"),
        variant=variant,
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


def _task3_case(case_id: str, variant: str, pair: dict) -> Task4Case:
    target = pair["target"]
    example = pair["example"]
    examples = _build_examples(pair, example["false_sign"])
    displayed_side = example["false_side"]
    matching_rule = JEL_SIMILARITY_RULE if variant == "side_capped_jel_similarity" else MATCHING_RULE
    return Task4Case(
        case_id=case_id,
        treatment=target["treatment"],
        outcome=target["outcome"],
        examples=examples,
        test_context=target["context"],
        expected_sign=target["sign"],
        paper_ids=[example["paper_id"], target["paper_id"]],
        avg_similarity=round(pair["jel_overlap_ratio"], 4),
        sign_differs=example["false_sign"] != target["sign"],
        context=target["context"],
        sign=target["sign"],
        title=target.get("title"),
        author=target.get("author"),
        publication_year=target.get("publication_year"),
        published_venue=target.get("published_venue"),
        jel_codes=target.get("jel_codes"),
        paper_url=target.get("paper_url"),
        variant=variant,
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


def _catalog_rows(task_name: str, rows: list[dict]) -> list[dict]:
    catalog: list[dict] = []
    for row in rows:
        example = (row.get("examples") or [{}])[0]
        catalog.append(
            {
                "variant": row.get("variant"),
                "task_name": task_name,
                "case_id": row.get("case_id"),
                "pair_case_type": row.get("pair_case_type"),
                "target_paper_id": row.get("paper_id"),
                "target_treatment": row.get("treatment"),
                "target_outcome": row.get("outcome"),
                "target_sign": row.get("sign"),
                "target_side": row.get("target_side"),
                "example_paper_id": example.get("paper_id"),
                "example_treatment": example.get("treatment"),
                "example_outcome": example.get("outcome"),
                "example_original_sign": example.get("original_sign"),
                "example_displayed_sign": example.get("sign"),
                "example_true_side": row.get("example_true_side"),
                "example_false_side": row.get("example_false_side"),
                "target_exact_jel_codes": json.dumps(row.get("target_exact_jel_codes"), ensure_ascii=False),
                "example_exact_jel_codes": json.dumps(row.get("example_exact_jel_codes"), ensure_ascii=False),
                "shared_exact_jel_codes": json.dumps(row.get("shared_exact_jel_codes"), ensure_ascii=False),
                "shared_exact_jel_count": row.get("shared_exact_jel_count"),
                "target_exact_jel_count": row.get("target_exact_jel_count"),
                "jel_overlap_ratio": row.get("jel_overlap_ratio"),
                "example_overlap_ratio": row.get("example_overlap_ratio"),
                "jel_similarity": row.get("jel_similarity"),
                "union_exact_jel_count": row.get("union_exact_jel_count"),
                "selection_score": row.get("selection_score"),
                "matching_rule": row.get("matching_rule"),
            }
        )
    return catalog


def _select_showcases(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("task_name"),
            row.get("pair_case_type"),
            _numeric_sort(row.get("paper_id")),
            _numeric_sort((row.get("examples") or [{}])[0].get("paper_id")),
            _safe_text(row.get("treatment")).lower(),
            _safe_text((row.get("examples") or [{}])[0].get("treatment")).lower(),
        ),
    )
    for row in sorted_rows:
        key = (row.get("task_name"), row.get("pair_case_type"))
        if len(grouped[key]) < 2:
            grouped[key].append(row)
    return grouped


def _write_verification_markdown(
    variant: str,
    task2_rows: list[dict],
    task3_rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    showcase_map = _select_showcases(task2_rows + task3_rows)

    task2_counts = Counter(row["pair_case_type"] for row in task2_rows)
    task3_counts = Counter(row["pair_case_type"] for row in task3_rows)
    matching_rule = task2_rows[0].get("matching_rule") if task2_rows else MATCHING_RULE
    if matching_rule == JEL_SIMILARITY_RULE:
        rule_lines = [
            "- 매칭 규칙: target/example의 informative exact JEL 집합(한 글자 broad code와 숫자-only code 제외)에 대해 Jaccard similarity를 계산하고, side별로 가장 높은 pair를 선택",
            "- 추가 조건: example과 target은 반드시 서로 다른 paper이고, informative shared exact JEL이 최소 1개는 있어야 함",
        ]
    else:
        rule_lines = [
            "- 매칭 규칙: 문제(target)의 exact JEL code 중 50% 이상을 example이 공유해야 함",
            "- 추가 조건: example과 target은 반드시 서로 다른 paper에서 와야 함",
        ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Task23 검수 노트 ({variant})\n\n")
        handle.write("## 요약\n\n")
        for line in rule_lines:
            handle.write(f"{line}\n")
        handle.write(f"- Task2 총 케이스 수: {len(task2_rows)}\n")
        handle.write(f"- Task3 총 케이스 수: {len(task3_rows)}\n")
        handle.write(
            "- Task2 case 수: "
            + ", ".join(f"`{case}`={task2_counts.get(case, 0)}" for case in ("lib-lib", "lib-cons", "cons-lib", "cons-cons"))
            + "\n"
        )
        handle.write(
            "- Task3 case 수: "
            + ", ".join(f"`{case}`={task3_counts.get(case, 0)}" for case in ("lib-lib", "lib-cons", "cons-lib", "cons-cons"))
            + "\n\n"
        )

        for task_name in ("task2", "task3"):
            handle.write(f"## {task_name.upper()} 샘플\n\n")
            for case_type in ("lib-lib", "lib-cons", "cons-lib", "cons-cons"):
                handle.write(f"### {case_type}\n\n")
                samples = showcase_map.get((task_name, case_type), [])
                for index, row in enumerate(samples, 1):
                    example = (row.get("examples") or [{}])[0]
                    handle.write(f"#### 예시 {index}\n\n")
                    handle.write(f"- variant: `{row.get('variant')}`\n")
                    handle.write(f"- case_id: `{row.get('case_id')}`\n")
                    handle.write(f"- target paper: `{row.get('paper_id')}` / {row.get('title') or '(제목 없음)'}\n")
                    handle.write(f"- example paper: `{example.get('paper_id')}` / {example.get('title') or '(제목 없음)'}\n")
                    handle.write(f"- target: `{row.get('treatment')}` -> `{row.get('outcome')}` / sign `{row.get('sign')}` / side `{row.get('target_side')}`\n")
                    handle.write(
                        f"- example: `{example.get('treatment')}` -> `{example.get('outcome')}` / displayed sign `{example.get('sign')}` / original sign `{example.get('original_sign')}` / true side `{row.get('example_true_side')}`\n"
                    )
                    handle.write(f"- target exact JEL: `{', '.join(row.get('target_exact_jel_codes') or [])}`\n")
                    handle.write(f"- example exact JEL: `{', '.join(row.get('example_exact_jel_codes') or [])}`\n")
                    handle.write(f"- shared exact JEL: `{', '.join(row.get('shared_exact_jel_codes') or [])}`\n")
                    handle.write(f"- target overlap ratio: `{row.get('jel_overlap_ratio')}`\n")
                    handle.write(f"- example overlap ratio: `{row.get('example_overlap_ratio')}`\n")
                    handle.write(f"- JEL similarity: `{row.get('jel_similarity')}`\n")
                    handle.write("- 검수 포인트: 다른 paper인지, shared exact JEL이 존재하는지, 그리고 현재 variant의 selection score가 기대한 방식으로 큰 pair인지 확인\n\n")
                    handle.write("```text\n")
                    handle.write(row.get("question", "").strip())
                    handle.write("\n```\n\n")


def _generate_variant_rows(variant: str, pairs: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    task2_formatter = Task3ContextTOFixed()
    task3_formatter = Task4ContextFixed()
    task2_rows: list[dict] = []
    task3_rows: list[dict] = []

    for index, pair in enumerate(pairs, 1):
        prefix_tag = "jel_similarity" if variant == "side_capped_jel_similarity" else "exact50"
        case_id_prefix = f"{prefix_tag}_{variant}_{index:06d}"
        task2_case = _task2_case(f"task2_{case_id_prefix}", variant, pair)
        task3_case = _task3_case(f"task3_{case_id_prefix}", variant, pair)
        task2_rows.append(_source_row(task2_case, "task2", task2_formatter.format_prompt(task2_case)))
        task3_rows.append(_source_row(task3_case, "task3", task3_formatter.format_prompt(task3_case)))

    return task2_rows, task3_rows


def _write_variant_outputs(variant: str, task2_rows: list[dict], task3_rows: list[dict]) -> dict:
    paths = VARIANT_OUTPUTS[variant]
    write_jsonl(paths["task2_path"], task2_rows)
    write_jsonl(paths["task3_path"], task3_rows)
    catalog_rows = _catalog_rows("task2", task2_rows) + _catalog_rows("task3", task3_rows)
    write_csv_rows(paths["catalog_path"], catalog_rows)
    _write_verification_markdown(variant, task2_rows, task3_rows, paths["verification_path"])

    task2_counts = Counter(row["pair_case_type"] for row in task2_rows)
    task3_counts = Counter(row["pair_case_type"] for row in task3_rows)
    return {
        "matching_rule": task2_rows[0].get("matching_rule") if task2_rows else None,
        "task2_rows": len(task2_rows),
        "task3_rows": len(task3_rows),
        "task2_case_counts": dict(sorted(task2_counts.items())),
        "task3_case_counts": dict(sorted(task3_counts.items())),
        "task2_path": str(paths["task2_path"]),
        "task3_path": str(paths["task3_path"]),
        "catalog_path": str(paths["catalog_path"]),
        "verification_path": str(paths["verification_path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dual exact50 task2/task3 source datasets.")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--classification-path", default=str(DEFAULT_CLASSIFICATION_PATH))
    parser.add_argument(
        "--variant",
        choices=["side_capped", "side_capped_jel_similarity", "all_pairs", "both"],
        default="both",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    classification_path = Path(args.classification_path)

    triplets = _usable_triplets(corpus_path, classification_path)
    pair_sets = _build_pair_sets(triplets)

    variants = ["side_capped", "side_capped_jel_similarity", "all_pairs"] if args.variant == "both" else [args.variant]
    summary = {
        "corpus_path": str(corpus_path),
        "classification_path": str(classification_path),
        "usable_triplets": len(triplets),
        "variants": {},
    }

    for variant in variants:
        task2_rows, task3_rows = _generate_variant_rows(variant, pair_sets[variant])
        summary["variants"][variant] = _write_variant_outputs(variant, task2_rows, task3_rows)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
