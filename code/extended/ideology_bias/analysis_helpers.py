"""Common dataset, aggregation, and regression helpers for analyses."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import SpecificationWarning

from .utils import read_jsonl
from .viz import is_regression_table, write_frame_figure, write_regression_report, write_regression_html_report


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a JSONL dataset into a DataFrame."""
    rows = read_jsonl(path)
    return pd.DataFrame(rows)


def _iter_values(value: object) -> list[str]:
    if value is None:
        return ["(missing)"]
    if isinstance(value, list):
        return [str(item) for item in value] or ["(missing)"]
    text = str(value)
    return [text if text else "(missing)"]


def explode_for_criterion(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Explode a multi-valued column into one row per value."""
    data = frame.copy()
    data[column] = data[column].apply(_iter_values)
    return data.explode(column)


def _with_analysis_weight(frame: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    """Return a copy of the frame with a normalized analysis-weight column."""
    data = frame.copy()
    if weight_col and weight_col in data.columns:
        data["_analysis_weight"] = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0).astype(float)
    else:
        data["_analysis_weight"] = 1.0
    return data


def accuracy_by(frame: pd.DataFrame, group_cols: list[str], weight_col: str | None = None) -> pd.DataFrame:
    """Aggregate accuracy by one or more grouping columns."""
    data = _with_analysis_weight(frame, weight_col=weight_col)
    data["_weighted_correct"] = pd.to_numeric(data["correct"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    grouped = data.groupby(group_cols, dropna=False, observed=False)
    summary = grouped.agg(
        n_predictions=("_analysis_weight", "sum"),
        correct_predictions=("_weighted_correct", "sum"),
    ).reset_index()
    if "triplet_key" in data.columns:
        triplets = grouped["triplet_key"].nunique().reset_index(name="n_triplets")
        summary = summary.merge(triplets, on=group_cols, how="left")
    else:
        summary["n_triplets"] = summary["n_predictions"]
    denominator = summary["n_predictions"].where(summary["n_predictions"] != 0)
    summary["accuracy"] = (summary["correct_predictions"] / denominator).astype(float).fillna(0.0)
    return summary.sort_values(["accuracy", "n_triplets", "n_predictions"], ascending=[False, False, False])


def distribution_by(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Return counts and within-group shares for a categorical value column."""
    grouped = (
        frame.groupby(group_cols + [value_col], dropna=False, observed=False)
        .agg(n_predictions=(value_col, "size"))
        .reset_index()
    )
    if "triplet_key" in frame.columns:
        triplets = (
            frame.groupby(group_cols + [value_col], dropna=False, observed=False)["triplet_key"]
            .nunique()
            .reset_index(name="n_triplets")
        )
        grouped = grouped.merge(triplets, on=group_cols + [value_col], how="left")
    else:
        grouped["n_triplets"] = grouped["n_predictions"]
    totals = (
        grouped.groupby(group_cols, dropna=False, observed=False)["n_predictions"]
        .sum()
        .reset_index(name="total_predictions")
    )
    merged = grouped.merge(totals, on=group_cols, how="left")
    merged["share_predictions"] = merged["n_predictions"] / merged["total_predictions"]
    return merged.sort_values(group_cols + ["share_predictions"], ascending=[True] * len(group_cols) + [False])


def bias_summary_by(frame: pd.DataFrame, group_cols: list[str], weight_col: str | None = None) -> pd.DataFrame:
    """Summarize liberal/conservative answer rates and leaning-error bias by group."""
    data = _with_analysis_weight(frame, weight_col=weight_col)
    data["ideology_error"] = ((data["ideology_triplet_labeled"] == 1) & (data["correct"] == 0)).astype(int)
    data["_weighted_ideology_prediction"] = (
        pd.to_numeric(data["ideology_triplet_labeled"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    )
    data["_weighted_ideology_error"] = (
        pd.to_numeric(data["ideology_error"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    )
    data["_weighted_predicted_liberal"] = (
        pd.to_numeric(data["predicted_liberal"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    )
    data["_weighted_predicted_conservative"] = (
        pd.to_numeric(data["predicted_conservative"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    )
    data["_weighted_liberal_error"] = (
        pd.to_numeric(data["liberal_leaning_error"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    )
    data["_weighted_conservative_error"] = (
        pd.to_numeric(data["conservative_leaning_error"], errors="coerce").fillna(0.0) * data["_analysis_weight"]
    )
    grouped = data.groupby(group_cols, dropna=False, observed=False)
    summary = grouped.agg(
        n_predictions=("_analysis_weight", "sum"),
        n_triplets=("triplet_key", "nunique"),
        ideology_predictions=("_weighted_ideology_prediction", "sum"),
        ideology_errors=("_weighted_ideology_error", "sum"),
        weighted_liberal_predictions=("_weighted_predicted_liberal", "sum"),
        weighted_conservative_predictions=("_weighted_predicted_conservative", "sum"),
        liberal_leaning_errors=("_weighted_liberal_error", "sum"),
        conservative_leaning_errors=("_weighted_conservative_error", "sum"),
    ).reset_index()
    n_predictions = summary["n_predictions"].where(summary["n_predictions"] != 0)
    summary["liberal_answer_rate"] = (
        summary["weighted_liberal_predictions"] / n_predictions
    ).astype(float).fillna(0.0)
    summary["conservative_answer_rate"] = (
        summary["weighted_conservative_predictions"] / n_predictions
    ).astype(float).fillna(0.0)
    summary = summary.drop(columns=["weighted_liberal_predictions", "weighted_conservative_predictions"])
    denominator = summary["ideology_errors"].where(summary["ideology_errors"] != 0)
    summary["bias_score"] = (
        (summary["liberal_leaning_errors"] - summary["conservative_leaning_errors"]) / denominator
    ).astype(float).fillna(0.0)
    # Calculate error rate: ideology_errors / n_predictions
    summary["error_rate"] = (
        summary["ideology_errors"] / summary["n_predictions"].where(summary["n_predictions"] != 0)
    ).astype(float).fillna(0.0)
    return summary.sort_values(group_cols)


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into JSON-safe dictionaries."""
    records: list[dict] = []
    for record in frame.to_dict(orient="records"):
        records.append({key: _json_safe(value) for key, value in record.items()})
    return records


def _json_safe(value: object) -> object:
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def save_frame(
    frame: pd.DataFrame,
    csv_path: str | Path,
    json_path: str | Path | None = None,
    formula: str | None = None,
) -> None:
    """Write a DataFrame to CSV and optionally to JSON.

    For regression results, a Markdown report and HTML report are written to the reports
    directory instead of a PNG figure.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    title = csv_path.stem.replace("_", " ").title()

    if is_regression_table(frame):
        # Write both markdown and HTML reports instead of a figure image.
        from .paths import REPORTS_DIR
        report_path = REPORTS_DIR / csv_path.with_suffix(".md").name
        write_regression_report(report_path, title, frame, formula=formula)

        # Also write HTML report
        html_report_path = REPORTS_DIR / csv_path.with_suffix(".html").name
        write_regression_html_report(html_report_path, title, frame, formula=formula)
    else:
        figure_path = csv_path
        if "tables" in figure_path.parts:
            parts = list(figure_path.parts)
            parts[parts.index("tables")] = "figures"
            figure_path = Path(*parts)
        figure_path = figure_path.with_suffix(".png")
        write_frame_figure(figure_path, title, frame)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(frame_to_records(frame), ensure_ascii=False, indent=2), encoding="utf-8")


def fit_clustered_binomial(
    formula: str,
    frame: pd.DataFrame,
    cluster_col: str,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Fit a clustered binomial GLM and return a tidy coefficient table."""
    data = frame.dropna(subset=[cluster_col]).copy()
    if data.empty:
        return pd.DataFrame([{"term": "intercept", "error": "empty dataset"}])
    weights = None
    if weight_col and weight_col in data.columns:
        weights = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0).astype(float)
        keep_mask = weights > 0
        data = data.loc[keep_mask].copy()
        weights = weights.loc[keep_mask]
        if data.empty:
            return pd.DataFrame([{"term": "intercept", "error": "empty weighted dataset"}])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SpecificationWarning)
            model = smf.glm(
                formula,
                data=data,
                family=sm.families.Binomial(),
                freq_weights=weights,
            )
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": data[cluster_col]},
            )
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"term": "intercept", "error": str(exc)}])

    # Calculate per-term triplet counts
    n_triplets_per_term = _calculate_triplet_counts(formula, data, cluster_col)

    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z_or_t": result.tvalues.values,
            "p_value": result.pvalues.values,
            "conf_low": result.conf_int()[0].values,
            "conf_high": result.conf_int()[1].values,
            "n_obs": int(result.nobs),
        }
    )

    # Add triplet counts
    table["n_triplets"] = table["term"].map(n_triplets_per_term)
    table["n_triplets"] = table["n_triplets"].fillna(data[cluster_col].nunique())

    return table


def _calculate_triplet_counts(formula: str, data: pd.DataFrame, cluster_col: str) -> dict[str, int]:
    """Calculate the number of unique triplets associated with each term in the regression formula."""
    triplet_counts: dict[str, int] = {}

    # Parse the formula to extract variable names
    lhs, rhs = formula.split("~", 1)
    terms = [term.strip() for term in rhs.split("+")]

    for term in terms:
        # Handle categorical variables C(var)
        import re
        cat_match = re.match(r"C\(([^)]+)\)", term)
        if cat_match:
            var_name = cat_match.group(1)
            # Get unique values of the categorical variable
            unique_values = data[var_name].dropna().unique()
            for value in unique_values:
                subset = data[data[var_name] == value]
                n_triplets = subset[cluster_col].nunique()
                # Create term name matching statsmodels output
                clean_term = f"C({var_name})[T.{value}]"
                triplet_counts[clean_term] = n_triplets
        else:
            # Handle continuous/binary variables
            var_name = term.strip()
            if var_name in data.columns:
                subset = data[data[var_name].notna()]
                n_triplets = subset[cluster_col].nunique()
                triplet_counts[var_name] = n_triplets

    return triplet_counts


def fit_binomial(formula: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Fit a standard binomial GLM and return a tidy coefficient table."""
    try:
        result = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit()
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"term": "intercept", "error": str(exc)}])
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z_or_t": result.tvalues.values,
            "p_value": result.pvalues.values,
            "conf_low": result.conf_int()[0].values,
            "conf_high": result.conf_int()[1].values,
            "n_obs": int(result.nobs),
        }
    )
    return table


def fit_multinomial(value_col: str, rhs_formula: str, frame: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Fit a multinomial logit model and return a tidy coefficient table."""
    data = frame.dropna(subset=[value_col]).copy()
    if data.empty:
        return pd.DataFrame([{"outcome": None, "term": "intercept", "error": "empty dataset"}])

    data = data[data[value_col].isin(categories)].copy()
    if data.empty:
        return pd.DataFrame([{"outcome": None, "term": "intercept", "error": "no valid categories"}])

    y = pd.Categorical(data[value_col], categories=categories)
    X = patsy.dmatrix(rhs_formula, data=data, return_type="dataframe")
    try:
        result = sm.MNLogit(y.codes, X).fit(method="newton", disp=False, maxiter=100)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"outcome": None, "term": "intercept", "error": str(exc)}])

    tidy_rows: list[dict] = []
    for column_index, column_name in enumerate(result.params.columns):
        outcome = categories[column_name + 1] if isinstance(column_name, int) else categories[column_index + 1]
        for term, coef in result.params.iloc[:, column_index].items():
            tidy_rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": coef,
                    "std_err": result.bse.iloc[:, column_index][term],
                    "z_or_t": result.tvalues.iloc[:, column_index][term],
                    "p_value": result.pvalues.iloc[:, column_index][term],
                    "n_obs": int(result.nobs),
                }
            )
    return pd.DataFrame(tidy_rows)
