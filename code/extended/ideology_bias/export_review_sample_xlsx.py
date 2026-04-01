"""Export the balanced ideology review sample to XLSX for human reviewers."""

from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path

import pandas as pd

from .utils import iter_jsonl, make_triplet_key


DEFAULT_SAMPLE_PATH = Path("extended/classification_results/ideology_triplet_review_sample_balanced126.jsonl")
DEFAULT_RESULTS_PATH = Path("extended/classification_results/causal_triplets_multillm_ideology_qwen32b.jsonl")
DEFAULT_OUTPUT_PATH = Path("extended/classification_results/ideology_triplet_review_sample_balanced126.xlsx")
SELECTED_MODELS = [
    "gpt-5-mini",
    "claude-sonnet-4-6",
    "qwen-3-32b",
    "grok-4-1-fast-reasoning",
]


def _normalize_sign(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "null":
        return ""
    return text


def _load_reasoning_lookup(results_path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in iter_jsonl(results_path):
        triplet_key = make_triplet_key(row.get("paper_id"), row.get("treatment"), row.get("outcome"))
        lookup[triplet_key] = row.get("classification", {}).get("per_model", {})
    return lookup


def _row_for_export(sample_row: dict, per_model_lookup: dict[str, dict]) -> dict[str, object]:
    row = {
        "title": sample_row.get("title"),
        "author": sample_row.get("author"),
        "venue": sample_row.get("published_venue"),
        "pub_year": sample_row.get("publication_year"),
        "category": sample_row.get("jel_policy_theme"),
        "treatment": sample_row.get("treatment"),
        "outcome": sample_row.get("outcome"),
        "context": sample_row.get("context"),
        "ground_truth_sign": sample_row.get("sign"),
        "lib_expected_sign": sample_row.get("lib_vote"),
        "con_expected_sign": sample_row.get("con_vote"),
    }

    model_signs = sample_row.get("model_signs", {}) or {}
    for model in SELECTED_MODELS:
        sample_model_signs = model_signs.get(model, {}) or {}
        pred_lib = _normalize_sign(sample_model_signs.get("lib"))
        pred_con = _normalize_sign(sample_model_signs.get("con"))
        has_expected = bool(row["lib_expected_sign"]) and bool(row["con_expected_sign"])
        matches_expected = has_expected and pred_lib == row["lib_expected_sign"] and pred_con == row["con_expected_sign"]
        reasoning = ""
        result_model = per_model_lookup.get(model, {}) or {}
        if matches_expected:
            reasoning = result_model.get("reasoning") or ""
        row[f"{model}_reasoning"] = reasoning

    row["human_label"] = ""
    return row


def _write_xlsx(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No rows to export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="review_sample")
        sheet = writer.book["review_sample"]
        sheet.freeze_panes = "A2"
        width_map = {
            "A": 32,
            "B": 28,
            "C": 24,
            "D": 10,
            "E": 28,
            "F": 34,
            "G": 100,
            "H": 8,
            "I": 16,
            "J": 16,
        }
        for col_letter, width in width_map.items():
            sheet.column_dimensions[col_letter].width = width
        for col_idx in range(11, sheet.max_column + 1):
            sheet.column_dimensions[chr(64 + col_idx)].width = 70
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                alignment.vertical = "top"
                cell.alignment = alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ideology review sample to XLSX.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    reasoning_lookup = _load_reasoning_lookup(args.results)
    export_rows = []
    for sample_row in iter_jsonl(args.sample):
        triplet_key = sample_row.get("triplet_key")
        per_model_lookup = reasoning_lookup.get(triplet_key, {})
        export_rows.append(_row_for_export(sample_row, per_model_lookup))

    _write_xlsx(args.output, export_rows)
    print(json.dumps({"rows": len(export_rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
