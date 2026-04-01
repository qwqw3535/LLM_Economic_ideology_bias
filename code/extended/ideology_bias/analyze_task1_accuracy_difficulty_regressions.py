from __future__ import annotations

"""Task 1 accuracy regressions with difficulty controls and model fixed effects.

This script writes reusable regression tables plus Markdown/HTML reports for:
1. The full ideology-labeled Task 1 sample with overall-difficulty controls.
2. The theme-level matched Task 1 sample with overall-difficulty controls.
3. The full ideology-labeled Task 1 sample with five difficulty-dimension controls.
4. The theme-level matched Task 1 sample with five difficulty-dimension controls.

Outputs are written under:
    OUTPUT_DIR / "difficulty_matched" / {"tables", "reports"}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .analysis_helpers import fit_clustered_binomial
from .analyze_task1_bias_difficulty_matched import (
    _load_difficulty,
    _prepare_frame,
    difficulty_match_by_theme,
)
from .paths import CLASSIFICATION_RESULTS_DIR, OUTPUT_DIR, TASK1_ANALYSIS_JSONL, ensure_output_dirs
from .viz import write_regression_html_report, write_regression_report


DEFAULT_DIFFICULTY_PATH = CLASSIFICATION_RESULTS_DIR / "difficulty_scores_clean.jsonl"
REG_TABLES_DIR = OUTPUT_DIR / "difficulty_matched" / "tables"
REG_REPORTS_DIR = OUTPUT_DIR / "difficulty_matched" / "reports"
IDEOLOGY_GROUND_TRUTH = {"liberal", "conservative"}


def _load_difficulty_dimensions(path: str | Path) -> dict[str, dict[str, int | None]]:
    """Load overall difficulty plus sub-dimension ratings keyed by triplet_key."""
    lookup: dict[str, dict[str, int | None]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            difficulty = row.get("difficulty") or {}
            lookup[row["triplet_key"]] = {
                "overall_difficulty": row.get("overall_difficulty"),
                "domain_knowledge": difficulty.get("domain_knowledge"),
                "context_dependence": difficulty.get("context_dependence"),
                "ambiguity": difficulty.get("ambiguity"),
                "causal_complexity": difficulty.get("causal_complexity"),
                "evidence_sufficiency": difficulty.get("evidence_sufficiency"),
            }
    return lookup


def _attach_difficulty_columns(
    frame: pd.DataFrame,
    difficulty_lookup: dict[str, dict[str, int | None]],
) -> pd.DataFrame:
    """Attach difficulty metadata to a prediction-level frame."""
    enriched = frame.copy().reset_index(drop=True)
    overlap_cols = [
        "overall_difficulty",
        "domain_knowledge",
        "context_dependence",
        "ambiguity",
        "causal_complexity",
        "evidence_sufficiency",
    ]
    existing_overlap = [column for column in overlap_cols if column in enriched.columns]
    if existing_overlap:
        enriched = enriched.drop(columns=existing_overlap)
    metadata = enriched["triplet_key"].map(difficulty_lookup)
    meta_frame = pd.DataFrame(list(metadata))
    enriched = pd.concat([enriched, meta_frame], axis=1)
    required = [
        "overall_difficulty",
        "domain_knowledge",
        "context_dependence",
        "ambiguity",
        "causal_complexity",
        "evidence_sufficiency",
    ]
    enriched = enriched.dropna(subset=required).copy()
    for column in required:
        enriched[column] = enriched[column].astype(int)
    return enriched


def _build_samples(
    task1_path: str | Path,
    difficulty_path: str | Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return full labeled and theme-level matched prediction frames with difficulty columns."""
    frame = _prepare_frame(str(task1_path))
    difficulty_dimensions = _load_difficulty_dimensions(difficulty_path)
    overall_lookup = {
        key: value["overall_difficulty"]
        for key, value in difficulty_dimensions.items()
        if value["overall_difficulty"] is not None
    }

    full_labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    full_labeled = _attach_difficulty_columns(full_labeled, difficulty_dimensions)

    matched_theme = difficulty_match_by_theme(
        frame,
        overall_lookup,
        theme_col="jel_policy_theme",
        seed=seed,
    )
    matched_theme = matched_theme[matched_theme["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    matched_theme = _attach_difficulty_columns(matched_theme, difficulty_dimensions)
    return full_labeled, matched_theme


def _sample_summary(data: pd.DataFrame) -> dict[str, float | int]:
    """Return compact sample-level summary stats for reporting."""
    liberal = data[data["ground_truth_side"] == "liberal"]
    conservative = data[data["ground_truth_side"] == "conservative"]
    return {
        "n_obs": int(len(data)),
        "n_triplets": int(data["triplet_key"].nunique()),
        "n_models": int(data["model"].nunique()),
        "mean_accuracy_liberal": float(liberal["correct"].mean()) if not liberal.empty else float("nan"),
        "mean_accuracy_conservative": float(conservative["correct"].mean()) if not conservative.empty else float("nan"),
    }


def _save_regression(table: pd.DataFrame, stem: str, formula: str) -> None:
    """Write regression CSV plus Markdown/HTML reports under difficulty_matched."""
    csv_path = REG_TABLES_DIR / f"{stem}.csv"
    md_path = REG_REPORTS_DIR / f"{stem}.md"
    html_path = REG_REPORTS_DIR / f"{stem}.html"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    title = stem.replace("_", " ").title()
    write_regression_report(md_path, title, table, formula=formula)
    write_regression_html_report(html_path, title, table, formula=formula)


def _key_term_summary(
    result_table: pd.DataFrame,
    spec_name: str,
    sample_name: str,
    formula: str,
    sample_stats: dict[str, float | int],
) -> dict[str, object]:
    """Extract the headline coefficient for ground_truth_liberal."""
    subset = result_table[result_table["term"] == "ground_truth_liberal"].copy()
    if subset.empty:
        return {
            "sample": sample_name,
            "spec": spec_name,
            "formula": formula,
            **sample_stats,
            "coef_ground_truth_liberal": None,
            "std_err": None,
            "p_value": None,
            "conf_low": None,
            "conf_high": None,
            "odds_ratio": None,
        }

    row = subset.iloc[0]
    coef = float(row["coef"])
    return {
        "sample": sample_name,
        "spec": spec_name,
        "formula": formula,
        **sample_stats,
        "coef_ground_truth_liberal": coef,
        "std_err": float(row["std_err"]) if pd.notna(row.get("std_err")) else None,
        "p_value": float(row["p_value"]) if pd.notna(row.get("p_value")) else None,
        "conf_low": float(row["conf_low"]) if pd.notna(row.get("conf_low")) else None,
        "conf_high": float(row["conf_high"]) if pd.notna(row.get("conf_high")) else None,
        "odds_ratio": math.exp(coef),
    }


def _write_summary_reports(summary_rows: list[dict[str, object]], output_stem: str) -> None:
    """Write a combined Markdown/HTML overview across all fitted specifications."""
    frame = pd.DataFrame(summary_rows)
    md_path = REG_REPORTS_DIR / f"{output_stem}.md"
    html_path = REG_REPORTS_DIR / f"{output_stem}.html"

    def _fmt(value: object, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "NA"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines: list[str] = []
    lines.append("# Task1 Accuracy Regressions With Difficulty Controls\n")
    lines.append("## Scope\n")
    lines.append(
        "- Outcome: `correct` (prediction-level binary accuracy).\n"
    )
    lines.append(
        "- Key regressor: `ground_truth_liberal` (`1` for liberal-ground-truth triplets, `0` for conservative-ground-truth triplets).\n"
    )
    lines.append("- Model fixed effects: `C(model)`.\n")
    lines.append("- Standard errors: cluster-robust by `triplet_key`.\n")
    lines.append("- Samples: full ideology-labeled Task 1 sample and theme-level difficulty-matched sample.\n")
    lines.append("")
    lines.append("## Headline Results\n")
    lines.append(
        "| Sample | Specification | N Obs | N Triplets | Liberal Acc. | Conservative Acc. | Coef. on ground_truth_liberal | Std. Error | p-value | Odds Ratio | 95% CI |\n"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n"
    )
    for row in summary_rows:
        ci_text = f"[{_fmt(row['conf_low'])}, {_fmt(row['conf_high'])}]"
        lines.append(
            "| "
            f"{row['sample']} | "
            f"{row['spec']} | "
            f"{_fmt(row['n_obs'], 0)} | "
            f"{_fmt(row['n_triplets'], 0)} | "
            f"{_fmt(row['mean_accuracy_liberal'])} | "
            f"{_fmt(row['mean_accuracy_conservative'])} | "
            f"{_fmt(row['coef_ground_truth_liberal'])} | "
            f"{_fmt(row['std_err'])} | "
            f"{_fmt(row['p_value'])} | "
            f"{_fmt(row['odds_ratio'])} | "
            f"{ci_text} |\n"
        )
    lines.append("")
    lines.append("## Formulas\n")
    for row in summary_rows:
        lines.append(f"- **{row['sample']} / {row['spec']}**: `{row['formula']}`\n")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "- A positive coefficient on `ground_truth_liberal` means higher accuracy on liberal-ground-truth triplets relative to conservative-ground-truth triplets, conditional on the included difficulty controls and model fixed effects.\n"
    )
    lines.append(
        "- The theme-matched sample is the stricter comparison because liberal and conservative triplets are balanced within each theme-by-difficulty cell.\n"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    html_frame = frame.copy()
    html_frame["95% CI"] = html_frame.apply(
        lambda row: f"[{_fmt(row['conf_low'])}, {_fmt(row['conf_high'])}]",
        axis=1,
    )
    html_frame = html_frame.rename(
        columns={
            "sample": "Sample",
            "spec": "Specification",
            "n_obs": "N Obs",
            "n_triplets": "N Triplets",
            "n_models": "N Models",
            "mean_accuracy_liberal": "Mean Accuracy (Liberal)",
            "mean_accuracy_conservative": "Mean Accuracy (Conservative)",
            "coef_ground_truth_liberal": "Coef. on ground_truth_liberal",
            "std_err": "Std. Error",
            "p_value": "p-value",
            "odds_ratio": "Odds Ratio",
            "formula": "Formula",
        }
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Task1 Accuracy Regressions With Difficulty Controls</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 1400px;
  margin: 20px auto;
  line-height: 1.6;
  color: #111827;
}}
h1 {{
  border-bottom: 3px solid #2563eb;
  padding-bottom: 8px;
}}
h2 {{
  margin-top: 28px;
  border-bottom: 1px solid #d1d5db;
  padding-bottom: 6px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 16px;
  font-size: 14px;
}}
th, td {{
  border: 1px solid #d1d5db;
  padding: 8px 10px;
  text-align: left;
}}
th {{
  background: #eff6ff;
}}
tr:nth-child(even) {{
  background: #f9fafb;
}}
code {{
  background: #f3f4f6;
  padding: 2px 4px;
}}
ul {{
  margin-top: 8px;
}}
</style>
</head>
<body>
<h1>Task1 Accuracy Regressions With Difficulty Controls</h1>
<h2>Scope</h2>
<ul>
  <li>Outcome: <code>correct</code> (prediction-level binary accuracy).</li>
  <li>Key regressor: <code>ground_truth_liberal</code> (1 for liberal-ground-truth triplets, 0 for conservative-ground-truth triplets).</li>
  <li>Model fixed effects: <code>C(model)</code>.</li>
  <li>Standard errors: cluster-robust by <code>triplet_key</code>.</li>
  <li>Samples: full ideology-labeled Task 1 sample and theme-level difficulty-matched sample.</li>
</ul>
<h2>Headline Results</h2>
{html_frame.to_html(index=False, escape=False)}
<h2>Interpretation</h2>
<ul>
  <li>A positive coefficient on <code>ground_truth_liberal</code> means higher accuracy on liberal-ground-truth triplets relative to conservative-ground-truth triplets, conditional on the included difficulty controls and model fixed effects.</li>
  <li>The theme-matched sample is the stricter comparison because liberal and conservative triplets are balanced within each theme-by-difficulty cell.</li>
</ul>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 1 accuracy regressions with difficulty controls and model fixed effects."
    )
    parser.add_argument(
        "--difficulty-path",
        default=str(DEFAULT_DIFFICULTY_PATH),
        help="Path to difficulty_scores_clean.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for theme-level matching.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    REG_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    full_labeled, matched_theme = _build_samples(
        task1_path=TASK1_ANALYSIS_JSONL,
        difficulty_path=args.difficulty_path,
        seed=args.seed,
    )

    specs = [
        (
            "full_sample",
            full_labeled,
            "overall_difficulty_plus_model_fe",
            "correct ~ ground_truth_liberal + C(overall_difficulty) + C(model)",
        ),
        (
            "theme_matched_sample",
            matched_theme,
            "overall_difficulty_plus_model_fe",
            "correct ~ ground_truth_liberal + C(overall_difficulty) + C(model)",
        ),
        (
            "theme_matched_sample",
            matched_theme,
            "overall_difficulty_plus_theme_plus_model_fe",
            "correct ~ ground_truth_liberal + C(overall_difficulty) + C(jel_policy_theme) + C(model)",
        ),
        (
            "full_sample",
            full_labeled,
            "difficulty_dimensions_plus_model_fe",
            "correct ~ ground_truth_liberal + C(domain_knowledge) + "
            "C(context_dependence) + C(ambiguity) + C(causal_complexity) + "
            "C(evidence_sufficiency) + C(model)",
        ),
        (
            "theme_matched_sample",
            matched_theme,
            "difficulty_dimensions_plus_model_fe",
            "correct ~ ground_truth_liberal + C(domain_knowledge) + "
            "C(context_dependence) + C(ambiguity) + C(causal_complexity) + "
            "C(evidence_sufficiency) + C(model)",
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    for sample_name, data, spec_name, formula in specs:
        print(f"[task1_diff_reg] fitting {sample_name} / {spec_name}")
        table = fit_clustered_binomial(formula, data, cluster_col="triplet_key")
        stem = f"task1_regression_accuracy_{spec_name}_{sample_name}"
        _save_regression(table, stem=stem, formula=formula)
        summary_rows.append(
            _key_term_summary(
                table,
                spec_name=spec_name,
                sample_name=sample_name,
                formula=formula,
                sample_stats=_sample_summary(data),
            )
        )

    _write_summary_reports(
        summary_rows,
        output_stem="task1_regression_accuracy_difficulty_controls_summary",
    )

    print(
        json.dumps(
            {
                "task1_analysis_path": str(TASK1_ANALYSIS_JSONL),
                "difficulty_path": str(args.difficulty_path),
                "full_sample_triplets": int(full_labeled["triplet_key"].nunique()),
                "theme_matched_triplets": int(matched_theme["triplet_key"].nunique()),
                "reports_dir": str(REG_REPORTS_DIR),
                "summary_report": str(
                    REG_REPORTS_DIR / "task1_regression_accuracy_difficulty_controls_summary.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
