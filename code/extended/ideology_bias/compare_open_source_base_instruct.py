"""Compare exact-size open-source base vs instruct Task 1 results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by
from .build_analysis_datasets import _enrich_with_metadata
from .paths import REPO_ROOT, ensure_output_dirs, preferred_metadata_jsonl
from .schemas import normalize_sign
from .utils import infer_family_from_model, make_triplet_key, parse_model_meta, read_jsonl


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "extended" / "ideology_bias_outputs_task1_ideology_subset_1056"
DEFAULT_SUBSET_PATH = REPO_ROOT / "data" / "task1_ideology_subset_1056.jsonl"
DEFAULT_INSTRUCT_RESULTS_DIR = REPO_ROOT / "econ_eval" / "evaluation_results_final" / "task1_causal_triplets" / "results"
DEFAULT_HF_RESULTS_ROOT = REPO_ROOT / "econ_eval" / "evaluation_results_hf_endpoint"


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    family: str
    family_label: str
    size_label: str
    instruct_model: str
    base_model: str
    instruct_path: Path
    base_path: Path


@dataclass(frozen=True)
class ModelSpec:
    family_label: str
    model_label: str
    variant_label: str
    analysis_group: str
    path: Path


PAIR_SPECS = [
    PairSpec(
        pair_id="llama_3_1_8b",
        family="llama",
        family_label="Llama",
        size_label="3.1-8B",
        instruct_model="meta-llama/llama-3.1-8b-instruct",
        base_model="meta-llama/Llama-3.1-8B",
        instruct_path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.1-8b-instruct_results.json",
        base_path=DEFAULT_HF_RESULTS_ROOT / "llama31_8b_full_task1" / "results" / "task1_hf_endpoint_meta-llama_Llama-3.1-8B_results.json",
    ),
    PairSpec(
        pair_id="llama_3_2_1b",
        family="llama",
        family_label="Llama",
        size_label="3.2-1B",
        instruct_model="meta-llama/llama-3.2-1b-instruct",
        base_model="meta-llama/Llama-3.2-1B",
        instruct_path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.2-1b-instruct_results.json",
        base_path=DEFAULT_HF_RESULTS_ROOT / "llama32_1b_full_task1" / "results" / "task1_hf_endpoint_meta-llama_Llama-3.2-1B_results.json",
    ),
    PairSpec(
        pair_id="llama_3_2_3b",
        family="llama",
        family_label="Llama",
        size_label="3.2-3B",
        instruct_model="meta-llama/llama-3.2-3b-instruct",
        base_model="meta-llama/Llama-3.2-3B",
        instruct_path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.2-3b-instruct_results.json",
        base_path=DEFAULT_HF_RESULTS_ROOT / "llama32_3b_full_task1" / "results" / "task1_hf_endpoint_meta-llama_Llama-3.2-3B_results.json",
    ),
    PairSpec(
        pair_id="llama_3_1_70b",
        family="llama",
        family_label="Llama",
        size_label="3.1-70B",
        instruct_model="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        base_model="unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        instruct_path=DEFAULT_HF_RESULTS_ROOT
        / "llama31_70b_instruct_bnb4"
        / "results"
        / "task1_hf_endpoint_unsloth_Meta-Llama-3.1-70B-Instruct-bnb-4bit_results.json",
        base_path=DEFAULT_HF_RESULTS_ROOT
        / "llama31_70b_full_task1"
        / "results"
        / "task1_hf_endpoint_unsloth_Meta-Llama-3.1-70B-bnb-4bit_results.json",
    ),
    PairSpec(
        pair_id="qwen3_8b",
        family="qwen",
        family_label="Qwen",
        size_label="8B",
        instruct_model="qwen/qwen3-8b",
        base_model="Qwen/Qwen3-8B-Base",
        instruct_path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_qwen_qwen_qwen3-8b_results.json",
        base_path=DEFAULT_HF_RESULTS_ROOT / "qwen3_8b_base_full_task1" / "results" / "task1_hf_endpoint_Qwen_Qwen3-8B-Base_results.json",
    ),
    PairSpec(
        pair_id="qwen3_14b",
        family="qwen",
        family_label="Qwen",
        size_label="14B",
        instruct_model="qwen/qwen3-14b",
        base_model="Qwen/Qwen3-14B-Base",
        instruct_path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_qwen_qwen_qwen3-14b_results.json",
        base_path=DEFAULT_HF_RESULTS_ROOT / "qwen3_14b_base_full_task1" / "results" / "task1_hf_endpoint_Qwen_Qwen3-14B-Base_results.json",
    ),
]

PAIR_ORDER = {spec.pair_id: idx for idx, spec in enumerate(PAIR_SPECS)}


EXCLUDED_MODEL_NOTES = [
    "- `llama31_70b_full_task1/checkpoints/task1_hf_endpoint_meta-llama_Llama-3.1-70B_checkpoint.json`에는 non-quantized 70B base checkpoint가 남아 있지만 completed case가 1,025개뿐이라, 주 테이블은 10,490개 전체가 완료된 `unsloth ... bnb-4bit` base/instruct pair를 사용했다.",
    "- `meta-llama/llama-3.3-70b-instruct`는 instruct-only 결과라 exact-size base pair가 없어 주 비교표에서는 제외하고 skew 점검만 했다.",
    "- `qwen/qwen3-32b` instruct는 있지만, `evaluation_results_hf_endpoint` 아래에 exact-size base 결과가 없어 주 테이블에서 제외했다.",
]


SKEW_MODEL_SPECS = [
    ModelSpec(
        family_label="Llama",
        model_label="3.1-8B",
        variant_label="Base",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT / "llama31_8b_full_task1" / "results" / "task1_hf_endpoint_meta-llama_Llama-3.1-8B_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.2-1B",
        variant_label="Base",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT / "llama32_1b_full_task1" / "results" / "task1_hf_endpoint_meta-llama_Llama-3.2-1B_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.2-3B",
        variant_label="Base",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT / "llama32_3b_full_task1" / "results" / "task1_hf_endpoint_meta-llama_Llama-3.2-3B_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.1-70B",
        variant_label="Base",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT
        / "llama31_70b_full_task1"
        / "results"
        / "task1_hf_endpoint_unsloth_Meta-Llama-3.1-70B-bnb-4bit_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.1-70B",
        variant_label="Instruct",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT
        / "llama31_70b_instruct_bnb4"
        / "results"
        / "task1_hf_endpoint_unsloth_Meta-Llama-3.1-70B-Instruct-bnb-4bit_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.1-8B",
        variant_label="Instruct",
        analysis_group="paired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.1-8b-instruct_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.2-1B",
        variant_label="Instruct",
        analysis_group="paired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.2-1b-instruct_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.2-3B",
        variant_label="Instruct",
        analysis_group="paired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.2-3b-instruct_results.json",
    ),
    ModelSpec(
        family_label="Llama",
        model_label="3.3-70B",
        variant_label="Instruct",
        analysis_group="unpaired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_llama_meta-llama_llama-3.3-70b-instruct_results.json",
    ),
    ModelSpec(
        family_label="Qwen",
        model_label="8B",
        variant_label="Base",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT / "qwen3_8b_base_full_task1" / "results" / "task1_hf_endpoint_Qwen_Qwen3-8B-Base_results.json",
    ),
    ModelSpec(
        family_label="Qwen",
        model_label="14B",
        variant_label="Base",
        analysis_group="paired",
        path=DEFAULT_HF_RESULTS_ROOT / "qwen3_14b_base_full_task1" / "results" / "task1_hf_endpoint_Qwen_Qwen3-14B-Base_results.json",
    ),
    ModelSpec(
        family_label="Qwen",
        model_label="8B",
        variant_label="Instruct",
        analysis_group="paired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_qwen_qwen_qwen3-8b_results.json",
    ),
    ModelSpec(
        family_label="Qwen",
        model_label="14B",
        variant_label="Instruct",
        analysis_group="paired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_qwen_qwen_qwen3-14b_results.json",
    ),
    ModelSpec(
        family_label="Qwen",
        model_label="32B",
        variant_label="Instruct",
        analysis_group="unpaired",
        path=DEFAULT_INSTRUCT_RESULTS_DIR / "task1_qwen_qwen_qwen3-32b_results.json",
    ),
]


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_subset_triplet_keys(path: Path) -> set[str]:
    subset_keys: set[str] = set()
    for row in read_jsonl(path):
        triplet_key = row.get("triplet_key")
        if triplet_key:
            subset_keys.add(str(triplet_key))
    return subset_keys


def _load_pair_rows(spec: PairSpec, variant: str, path: Path, metadata_lookup: dict[str, dict], subset_keys: set[str]) -> list[dict]:
    document = _read_json(path)
    raw_family = str(document.get("family") or "unknown")
    raw_model = str(document.get("model") or "unknown")
    family = infer_family_from_model(raw_model) if raw_family == "hf_endpoint" else raw_family
    model_meta = parse_model_meta(family, raw_model)

    rows: list[dict] = []
    for result in document.get("results", []):
        input_data = result.get("input_data", {}) or {}
        paper_id = input_data.get("paper_id")
        triplet_key = make_triplet_key(paper_id, input_data.get("treatment"), input_data.get("outcome"))
        row = {
            "pair_id": spec.pair_id,
            "family": spec.family,
            "family_label": spec.family_label,
            "size_label": spec.size_label,
            "variant": variant,
            "variant_label": "Instruct" if variant == "instruct" else "Base",
            "display_model": f"{spec.family_label} {spec.size_label}",
            "triplet_key": triplet_key,
            "paper_task": "task1",
            "expected_sign": normalize_sign(result.get("expected") or input_data.get("expected_sign")),
            "predicted_sign": normalize_sign(result.get("predicted")),
            "predicted_raw": result.get("predicted"),
            "correct": int(bool(result.get("correct"))),
            **model_meta,
        }
        enriched = _enrich_with_metadata(row, metadata_lookup.get(triplet_key))
        enriched["is_sensitive_subset"] = int(triplet_key in subset_keys)
        rows.append(enriched)
    return rows


def _build_metrics_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitive = frame[frame["is_sensitive_subset"] == 1].copy()
    non_sensitive = frame[frame["is_sensitive_subset"] == 0].copy()
    labeled_sensitive = sensitive[sensitive["ground_truth_side"].isin(["liberal", "conservative"])].copy()

    acc_sensitive = accuracy_by(sensitive, ["pair_id", "family", "family_label", "size_label", "variant", "variant_label"]).rename(
        columns={"accuracy": "acc_sensitive"}
    )
    acc_non_sensitive = accuracy_by(
        non_sensitive,
        ["pair_id", "family", "family_label", "size_label", "variant", "variant_label"],
    ).rename(columns={"accuracy": "acc_non_sensitive"})
    acc_ground_truth = accuracy_by(
        labeled_sensitive,
        ["pair_id", "family", "family_label", "size_label", "variant", "variant_label", "ground_truth_side"],
    )
    acc_lib = (
        acc_ground_truth.loc[acc_ground_truth["ground_truth_side"] == "liberal"]
        .rename(columns={"accuracy": "acc_lib"})[
            ["pair_id", "family", "family_label", "size_label", "variant", "variant_label", "acc_lib"]
        ]
    )
    acc_con = (
        acc_ground_truth.loc[acc_ground_truth["ground_truth_side"] == "conservative"]
        .rename(columns={"accuracy": "acc_con"})[
            ["pair_id", "family", "family_label", "size_label", "variant", "variant_label", "acc_con"]
        ]
    )
    bias = bias_summary_by(
        labeled_sensitive,
        ["pair_id", "family", "family_label", "size_label", "variant", "variant_label"],
    ).rename(columns={"bias_score": "bias_dir"})
    bias = bias[
        [
            "pair_id",
            "family",
            "family_label",
            "size_label",
            "variant",
            "variant_label",
            "bias_dir",
            "n_predictions",
            "n_triplets",
            "ideology_errors",
            "liberal_answer_rate",
            "conservative_answer_rate",
        ]
    ]

    metrics = acc_sensitive[
        [
            "pair_id",
            "family",
            "family_label",
            "size_label",
            "variant",
            "variant_label",
            "acc_sensitive",
            "n_predictions",
            "n_triplets",
        ]
    ].merge(
        acc_non_sensitive[
            ["pair_id", "family", "family_label", "size_label", "variant", "variant_label", "acc_non_sensitive"]
        ],
        on=["pair_id", "family", "family_label", "size_label", "variant", "variant_label"],
        how="left",
    )
    metrics = metrics.merge(
        acc_lib,
        on=["pair_id", "family", "family_label", "size_label", "variant", "variant_label"],
        how="left",
    ).merge(
        acc_con,
        on=["pair_id", "family", "family_label", "size_label", "variant", "variant_label"],
        how="left",
    ).merge(
        bias,
        on=["pair_id", "family", "family_label", "size_label", "variant", "variant_label"],
        how="left",
        suffixes=("", "_labeled"),
    )

    # For Table 1, the user-defined "neutral" slice is the complement of the 1,056-triplet
    # ideology-sensitive subset inside the full 10,490-case Task 1 corpus.
    metrics["acc_neutral"] = metrics["acc_non_sensitive"]

    metrics = metrics.rename(
        columns={
            "n_predictions": "n_sensitive_predictions",
            "n_triplets": "n_sensitive_triplets",
            "n_predictions_labeled": "n_labeled_predictions",
            "n_triplets_labeled": "n_labeled_triplets",
        }
    )
    if "n_labeled_predictions" not in metrics.columns and "n_predictions" in bias.columns:
        labeled_pred_lookup = bias.set_index(["pair_id", "variant"])["n_predictions"].to_dict()
        metrics["n_labeled_predictions"] = metrics.apply(
            lambda row: labeled_pred_lookup.get((row["pair_id"], row["variant"])),
            axis=1,
        )
    if "n_labeled_triplets" not in metrics.columns and "n_triplets" in bias.columns:
        labeled_triplet_lookup = bias.set_index(["pair_id", "variant"])["n_triplets"].to_dict()
        metrics["n_labeled_triplets"] = metrics.apply(
            lambda row: labeled_triplet_lookup.get((row["pair_id"], row["variant"])),
            axis=1,
        )

    metrics["delta_acc"] = metrics["acc_lib"] - metrics["acc_con"]
    metrics["abs_delta_acc"] = metrics["delta_acc"].abs()
    metrics["abs_bias_dir"] = metrics["bias_dir"].abs()
    metrics["pair_order"] = metrics["pair_id"].map(PAIR_ORDER).fillna(999).astype(int)
    metrics["row_order"] = metrics["variant"].map({"base": 0, "instruct": 1}).fillna(9).astype(int)
    metrics = metrics.sort_values(["pair_order", "row_order"]).reset_index(drop=True)

    wide = (
        metrics[
            [
                "pair_id",
                "pair_order",
                "family",
                "family_label",
                "size_label",
                "variant",
                "acc_sensitive",
                "acc_non_sensitive",
                "acc_neutral",
                "acc_lib",
                "acc_con",
                "delta_acc",
                "bias_dir",
                "abs_delta_acc",
                "abs_bias_dir",
            ]
        ]
        .pivot(index=["pair_id", "pair_order", "family", "family_label", "size_label"], columns="variant")
        .reset_index()
    )
    wide.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in wide.columns.to_flat_index()
    ]

    for metric in ["acc_sensitive", "acc_non_sensitive", "acc_neutral", "acc_lib", "acc_con", "delta_acc", "bias_dir"]:
        wide[f"{metric}_gain"] = wide[f"{metric}_instruct"] - wide[f"{metric}_base"]
    wide["abs_delta_acc_change"] = wide["abs_delta_acc_instruct"] - wide["abs_delta_acc_base"]
    wide["abs_bias_dir_change"] = wide["abs_bias_dir_instruct"] - wide["abs_bias_dir_base"]
    wide = wide.sort_values(["pair_order"]).reset_index(drop=True)
    return metrics, wide


def _pair_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    sensitive = frame[frame["is_sensitive_subset"] == 1].copy()
    diagnostics: list[dict] = []

    for spec in PAIR_SPECS:
        pair_frame = sensitive[sensitive["pair_id"] == spec.pair_id].copy()
        if pair_frame.empty:
            continue

        counts_by_variant: dict[str, Counter] = {}
        for variant in ("base", "instruct"):
            variant_counter = Counter(pair_frame.loc[pair_frame["variant"] == variant, "predicted_raw"].tolist())
            counts_by_variant[variant] = variant_counter

        base_rows = pair_frame.loc[pair_frame["variant"] == "base", ["triplet_key", "predicted_raw", "correct"]].rename(
            columns={"predicted_raw": "predicted_raw_base", "correct": "correct_base"}
        )
        instruct_rows = pair_frame.loc[
            pair_frame["variant"] == "instruct", ["triplet_key", "predicted_raw", "correct"]
        ].rename(columns={"predicted_raw": "predicted_raw_instruct", "correct": "correct_instruct"})
        joined = base_rows.merge(instruct_rows, on="triplet_key", how="inner")

        dominant_variant = max(
            ("base", "instruct"),
            key=lambda variant: counts_by_variant[variant].most_common(1)[0][1] if counts_by_variant[variant] else -1,
        )
        dominant_prediction, dominant_count = counts_by_variant[dominant_variant].most_common(1)[0]
        diagnostics.append(
            {
                "pair_id": spec.pair_id,
                "pair_order": PAIR_ORDER.get(spec.pair_id, 999),
                "family_label": spec.family_label,
                "size_label": spec.size_label,
                "n_sensitive_rows": len(pair_frame) // 2,
                "same_predicted_count": int((joined["predicted_raw_base"] == joined["predicted_raw_instruct"]).sum()),
                "same_correct_count": int((joined["correct_base"] == joined["correct_instruct"]).sum()),
                "dominant_variant": dominant_variant,
                "dominant_prediction": str(dominant_prediction),
                "dominant_prediction_count": int(dominant_count),
                "dominant_prediction_share": float(dominant_count / max(len(pair_frame) // 2, 1)),
                "base_prediction_counts": json.dumps(counts_by_variant["base"], ensure_ascii=False),
                "instruct_prediction_counts": json.dumps(counts_by_variant["instruct"], ensure_ascii=False),
            }
        )

    return pd.DataFrame(diagnostics).sort_values(["pair_order"]).reset_index(drop=True)


def _prediction_skew_table(subset_keys: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    model_order = {f"{spec.family_label} {spec.model_label} {spec.variant_label}": idx for idx, spec in enumerate(SKEW_MODEL_SPECS)}

    for spec in SKEW_MODEL_SPECS:
        document = _read_json(spec.path)
        normalized_counts = Counter()
        raw_counts = Counter()

        for result in document.get("results", []):
            input_data = result.get("input_data", {}) or {}
            triplet_key = make_triplet_key(input_data.get("paper_id"), input_data.get("treatment"), input_data.get("outcome"))
            if triplet_key not in subset_keys:
                continue
            raw_prediction = result.get("predicted")
            normalized = normalize_sign(raw_prediction)
            bucket = normalized if normalized in {"+", "-", "None", "mixed"} else "missing"
            normalized_counts[bucket] += 1
            raw_counts[str(raw_prediction)] += 1

        dominant_bucket, dominant_count = normalized_counts.most_common(1)[0]
        rows.append(
            {
                "family_label": spec.family_label,
                "model_label": spec.model_label,
                "variant_label": spec.variant_label,
                "analysis_group": spec.analysis_group,
                "n_sensitive_rows": sum(normalized_counts.values()),
                "count_plus": normalized_counts.get("+", 0),
                "count_minus": normalized_counts.get("-", 0),
                "count_none": normalized_counts.get("None", 0),
                "count_mixed": normalized_counts.get("mixed", 0),
                "count_missing": normalized_counts.get("missing", 0),
                "dominant_bucket": dominant_bucket,
                "dominant_count": dominant_count,
                "dominant_share": dominant_count / max(sum(normalized_counts.values()), 1),
                "severe_collapse_95": int(dominant_count / max(sum(normalized_counts.values()), 1) >= 0.95),
                "high_skew_80": int(dominant_count / max(sum(normalized_counts.values()), 1) >= 0.80),
                "raw_prediction_counts": json.dumps(raw_counts, ensure_ascii=False),
                "_order": model_order[f"{spec.family_label} {spec.model_label} {spec.variant_label}"],
            }
        )

    return pd.DataFrame(rows).sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def _pp(value: float) -> str:
    return f"{value * 100:+.1f}"


def _metric_with_bold(pair_rows: pd.DataFrame, variant: str, column: str, minimize_abs: bool = False) -> str:
    value = float(pair_rows.loc[pair_rows["variant"] == variant, column].iloc[0])
    if minimize_abs:
        base_val = abs(float(pair_rows.loc[pair_rows["variant"] == "base", column].iloc[0]))
        instruct_val = abs(float(pair_rows.loc[pair_rows["variant"] == "instruct", column].iloc[0]))
        best = min(base_val, instruct_val)
        is_best = abs(abs(value) - best) < 1e-12
        text = _pp(value)
    else:
        base_val = float(pair_rows.loc[pair_rows["variant"] == "base", column].iloc[0])
        instruct_val = float(pair_rows.loc[pair_rows["variant"] == "instruct", column].iloc[0])
        best = max(base_val, instruct_val)
        is_best = abs(value - best) < 1e-12
        text = _pct(value)
    return f"\\textbf{{{text}}}" if is_best else text


def _build_tex_table(metrics: pd.DataFrame, wide: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Base vs. instruct comparison for exact-size open-source model pairs on the 1,056-triplet ideology-sensitive subset. "
        "\\textit{Acc\\textsuperscript{sen}} is accuracy on the 1,056 sensitive triplets. "
        "\\textit{Acc\\textsuperscript{neu}} is accuracy on the remaining 9,434 triplets in the full 10,490-case Task 1 corpus, treated here as the neutral complement of the ideology-sensitive subset. "
        "\\textit{Acc\\textsuperscript{lib}} and \\textit{Acc\\textsuperscript{con}} denote accuracy on the 751 ideology-labeled sensitive triplets whose ground-truth aligns with liberal vs. conservative priors. "
        "$\\Delta_{\\text{acc}} = \\text{Acc}^{\\text{lib}} - \\text{Acc}^{\\text{con}}$. "
        "\\textit{Bias\\textsubscript{dir}} measures directional error bias (${>}0$: liberal-leaning, ${<}0$: conservative-leaning). "
        "$\\Delta$ rows report instruct minus base in percentage points. "
        "\\textbf{Bold} indicates the better value within each base/instruct pair (highest for accuracy; closest to zero for $\\Delta_{\\text{acc}}$ and Bias\\textsubscript{dir}).}",
        "\\label{tab:open_source_base_vs_instruct}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{ll cccc cc}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{\\textit{Accuracy}} & \\multicolumn{2}{c}{\\textit{By GT Direction}} & \\multicolumn{2}{c}{\\textit{Bias Metrics}} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        "\\textbf{Family} & \\textbf{Model / Variant} & Acc\\textsuperscript{sen} & Acc\\textsuperscript{neu} & Acc\\textsuperscript{lib} & Acc\\textsuperscript{con} & $\\Delta_{\\text{acc}}$ & Bias\\textsubscript{dir} \\\\",
        "\\midrule",
    ]

    for family_label in ["Llama", "Qwen"]:
        family_rows = metrics[metrics["family_label"] == family_label]
        if family_rows.empty:
            continue
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{family_label}}}}} \\\\")
        for pair_id in family_rows["pair_id"].drop_duplicates():
            pair_rows = family_rows[family_rows["pair_id"] == pair_id]
            size_label = str(pair_rows["size_label"].iloc[0])
            delta_row = wide.loc[wide["pair_id"] == pair_id].iloc[0]

            lines.append(
                f"{family_label} & {size_label} Base & "
                f"{_metric_with_bold(pair_rows, 'base', 'acc_sensitive')} & "
                f"{_metric_with_bold(pair_rows, 'base', 'acc_neutral')} & "
                f"{_metric_with_bold(pair_rows, 'base', 'acc_lib')} & "
                f"{_metric_with_bold(pair_rows, 'base', 'acc_con')} & "
                f"{_metric_with_bold(pair_rows, 'base', 'delta_acc', minimize_abs=True)} & "
                f"{_metric_with_bold(pair_rows, 'base', 'bias_dir', minimize_abs=True)} \\\\"
            )
            lines.append(
                f" & {size_label} Instruct & "
                f"{_metric_with_bold(pair_rows, 'instruct', 'acc_sensitive')} & "
                f"{_metric_with_bold(pair_rows, 'instruct', 'acc_neutral')} & "
                f"{_metric_with_bold(pair_rows, 'instruct', 'acc_lib')} & "
                f"{_metric_with_bold(pair_rows, 'instruct', 'acc_con')} & "
                f"{_metric_with_bold(pair_rows, 'instruct', 'delta_acc', minimize_abs=True)} & "
                f"{_metric_with_bold(pair_rows, 'instruct', 'bias_dir', minimize_abs=True)} \\\\"
            )
            lines.append(
                f" & $\\Delta$ (Inst-Base) & "
                f"{_pp(float(delta_row['acc_sensitive_gain']))} & "
                f"{_pp(float(delta_row['acc_neutral_gain']))} & "
                f"{_pp(float(delta_row['acc_lib_gain']))} & "
                f"{_pp(float(delta_row['acc_con_gain']))} & "
                f"{_pp(float(delta_row['delta_acc_gain']))} & "
                f"{_pp(float(delta_row['bias_dir_gain']))} \\\\"
            )
            lines.append("\\addlinespace[2pt]")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_report(metrics: pd.DataFrame, wide: pd.DataFrame, diagnostics: pd.DataFrame, skew: pd.DataFrame) -> str:
    mean_acc_sensitive_gain = float(wide["acc_sensitive_gain"].mean() * 100)
    mean_acc_non_sensitive_gain = float(wide["acc_non_sensitive_gain"].mean() * 100)
    mean_abs_gap_change = float(wide["abs_delta_acc_change"].mean() * 100)
    mean_abs_bias_change = float(wide["abs_bias_dir_change"].mean() * 100)

    strongest_gain = wide.sort_values("acc_sensitive_gain", ascending=False).iloc[0]
    weakest_gain = wide.sort_values("acc_sensitive_gain", ascending=True).iloc[0]

    lines = [
        "# Open-Source Base vs Instruct Comparison Report",
        "",
        "## 1. Scope",
        "",
        f"- Exact-size paired open-source models: {len(PAIR_SPECS)}",
        f"- Sensitive subset size: {int(metrics['n_sensitive_triplets'].max())} triplets",
        f"- Neutral complement size: {10490 - int(metrics['n_sensitive_triplets'].max())} triplets",
        f"- Ideology-labeled sensitive subset used for `Acc^lib`, `Acc^con`, `Δ_acc`, `Bias_dir`: {int(metrics['n_labeled_triplets'].max())} triplets",
        "- Here `Acc^neu` is defined as accuracy on the remaining 9,434 triplets outside the 1,056-triplet ideology-sensitive subset.",
        "- Metrics otherwise follow the same definitions as `table1_main_results.tex`.",
        "",
        "## 2. Main Findings",
        "",
        f"- Across the {len(PAIR_SPECS)} exact-size pairs, instruction tuning increased `Acc^sen` by {mean_acc_sensitive_gain:.1f}pp on average and `Acc^neu` by {mean_acc_non_sensitive_gain:.1f}pp on average.",
        f"- The ideological performance gap did not shrink overall: mean `|Δ_acc|` increased by {mean_abs_gap_change:.1f}pp.",
        f"- Directional bias magnitude also did not shrink overall: mean `|Bias_dir|` increased by {mean_abs_bias_change:.1f}pp.",
        f"- The largest `Acc^sen` gain was {strongest_gain['family_label']} {strongest_gain['size_label']} ({strongest_gain['acc_sensitive_gain'] * 100:+.1f}pp).",
        f"- The smallest `Acc^sen` gain was {weakest_gain['family_label']} {weakest_gain['size_label']} ({weakest_gain['acc_sensitive_gain'] * 100:+.1f}pp).",
        "",
        "## 3. Pair-by-Pair Summary",
        "",
    ]

    for row in wide.itertuples(index=False):
        lines.append(
            f"- {row.family_label} {row.size_label}: `Acc^sen` {row.acc_sensitive_base * 100:.1f} -> {row.acc_sensitive_instruct * 100:.1f} "
            f"({row.acc_sensitive_gain * 100:+.1f}pp), `Acc^neu` {row.acc_neutral_base * 100:.1f} -> {row.acc_neutral_instruct * 100:.1f} "
            f"({row.acc_neutral_gain * 100:+.1f}pp), `Δ_acc` {row.delta_acc_base * 100:+.1f} -> {row.delta_acc_instruct * 100:+.1f} "
            f"({row.delta_acc_gain * 100:+.1f}pp), `Bias_dir` {row.bias_dir_base * 100:+.1f} -> {row.bias_dir_instruct * 100:+.1f} "
            f"({row.bias_dir_gain * 100:+.1f}pp)."
        )

    lines.extend(
        [
            "",
            "## 4. Response Skew Check",
            "",
        ]
    )

    severe_skew = skew[skew["severe_collapse_95"] == 1]
    high_skew = skew[(skew["high_skew_80"] == 1) & (skew["severe_collapse_95"] == 0)]

    if severe_skew.empty:
        lines.append("- No model crossed the `95%` dominant-sign threshold on the sensitive subset.")
    else:
        lines.append("- Severe collapse (`dominant_share >= 95%`) on the sensitive subset:")
        for row in severe_skew.itertuples(index=False):
            lines.append(
                f"  - {row.family_label} {row.model_label} {row.variant_label}: `{row.dominant_bucket}` "
                f"{row.dominant_count}/{row.n_sensitive_rows} ({row.dominant_share * 100:.1f}%)."
            )

    if not high_skew.empty:
        lines.append("- High but not extreme skew (`80% <= dominant_share < 95%`):")
        for row in high_skew.itertuples(index=False):
            lines.append(
                f"  - {row.family_label} {row.model_label} {row.variant_label}: `{row.dominant_bucket}` "
                f"{row.dominant_count}/{row.n_sensitive_rows} ({row.dominant_share * 100:.1f}%)."
            )

    balanced_70b = skew[
        (skew["family_label"] == "Llama") & (skew["model_label"] == "3.1-70B") & (skew["variant_label"] == "Instruct")
    ]
    if not balanced_70b.empty:
        row = balanced_70b.iloc[0]
        lines.append(
            f"- Llama 3.1-70B Instruct is not collapsed: its dominant sign is `{row['dominant_bucket']}` "
            f"at only {row['dominant_share'] * 100:.1f}%."
        )

    lines.extend(
        [
            "",
            "## 5. Pair Diagnostics Worth Noting",
            "",
        ]
    )

    for row in diagnostics.itertuples(index=False):
        if row.dominant_prediction_share >= 0.95:
            variant_label = "base" if row.dominant_variant == "base" else "instruct"
            lines.append(
                f"- {row.family_label} {row.size_label}: `{variant_label}` output was dominated by `{row.dominant_prediction}` "
                f"on {row.dominant_prediction_count}/{row.n_sensitive_rows} sensitive cases ({row.dominant_prediction_share * 100:.1f}%)."
            )
        if row.same_correct_count >= row.n_sensitive_rows and row.same_predicted_count >= row.n_sensitive_rows - 1:
            lines.append(
                f"- {row.family_label} {row.size_label}: base and instruct were nearly identical on the sensitive subset "
                f"({row.same_predicted_count}/{row.n_sensitive_rows} same raw predictions; {row.same_correct_count}/{row.n_sensitive_rows} same correctness labels)."
            )

    lines.extend(
        [
            "",
            "## 6. Interpretation",
            "",
            "- In these matched open-source pairs, instruction tuning usually improved task accuracy substantially.",
            "- But higher accuracy did not translate into consistently smaller liberal-conservative gap or directional bias.",
            "- So these results support a narrower claim: instruction tuning changes performance a lot, but it does not by itself reliably eliminate the lib-con gap/bias pattern.",
            "",
            "## 7. Excluded from the Main Paired Table",
            "",
        ]
    )
    lines.extend(EXCLUDED_MODEL_NOTES)
    lines.extend(
        [
            "",
            "## 8. Output Files",
            "",
            "- `tables/table1_open_source_base_vs_instruct_long.csv`",
            "- `tables/table1_open_source_base_vs_instruct_wide.csv`",
            "- `tables/task1_open_source_prediction_skew_sensitive_subset.csv`",
            "- `tables/table1_open_source_base_vs_instruct.tex`",
            "- `reports/task1_open_source_base_vs_instruct_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare open-source base vs instruct Task1 ideology-bias metrics.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH, help="Path to the 1056-triplet subset JSONL")
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=preferred_metadata_jsonl(),
        help="Canonical metadata JSONL used for ideology labels",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    output_root = args.output_root.resolve()
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata_lookup = {row["triplet_key"]: row for row in read_jsonl(args.metadata_jsonl)}
    subset_keys = _load_subset_triplet_keys(args.subset_path)

    rows: list[dict] = []
    for spec in PAIR_SPECS:
        rows.extend(_load_pair_rows(spec, "base", spec.base_path, metadata_lookup, subset_keys))
        rows.extend(_load_pair_rows(spec, "instruct", spec.instruct_path, metadata_lookup, subset_keys))

    frame = pd.DataFrame(rows)
    metrics, wide = _build_metrics_frame(frame)
    diagnostics = _pair_diagnostics(frame)
    skew = _prediction_skew_table(subset_keys)

    long_csv_path = tables_dir / "table1_open_source_base_vs_instruct_long.csv"
    wide_csv_path = tables_dir / "table1_open_source_base_vs_instruct_wide.csv"
    diagnostics_csv_path = tables_dir / "table1_open_source_base_vs_instruct_diagnostics.csv"
    skew_csv_path = tables_dir / "task1_open_source_prediction_skew_sensitive_subset.csv"
    tex_path = tables_dir / "table1_open_source_base_vs_instruct.tex"
    report_path = reports_dir / "task1_open_source_base_vs_instruct_report.md"

    metrics.to_csv(long_csv_path, index=False)
    wide.to_csv(wide_csv_path, index=False)
    diagnostics.to_csv(diagnostics_csv_path, index=False)
    skew.to_csv(skew_csv_path, index=False)
    tex_path.write_text(_build_tex_table(metrics, wide), encoding="utf-8")
    report_path.write_text(_build_report(metrics, wide, diagnostics, skew), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "paired_models": len(PAIR_SPECS),
                "subset_triplets": len(subset_keys),
                "long_table": str(long_csv_path),
                "wide_table": str(wide_csv_path),
                "diagnostics_table": str(diagnostics_csv_path),
                "skew_table": str(skew_csv_path),
                "latex_table": str(tex_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
