"""Task 1 ideology-side analysis using vote-based JEL theme assignment."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by, explode_for_criterion, fit_clustered_binomial, load_dataset, save_frame
from .jel import IDEOLOGY_VOTE_THEME_ORDER, ideology_theme_vote_details, ideology_theme_vote_weights
from .load_results import load_task1_rows
from .paths import REPORTS_DIR, SOURCE_CATALOG_JSONL, TABLES_DIR, TASK1_ANALYSIS_JSONL, TASK1_RESULT_DIRS, ensure_output_dirs
from .utils import family_sort_key, model_sort_key


IDEOLOGY_GROUND_TRUTH = {"liberal", "conservative"}
THEME_COL = "jel_policy_theme_vote_primary"
THEME_COUNTS_COL = "jel_policy_theme_vote_counts"
THEME_WEIGHTS_COL = "jel_policy_theme_vote_weights"
THEME_WEIGHT_COL = "jel_policy_theme_weight"


def _publication_year_5y_bucket(value: object) -> str:
    year = pd.to_numeric(value, errors="coerce")
    if pd.isna(year):
        return "(missing)"
    start = int(year) // 5 * 5
    return f"{start}-{start + 4}"


def _publication_year_5y_bucket_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    if text == "(missing)":
        return (10**9, text)
    try:
        start = int(text.split("-", 1)[0])
    except (TypeError, ValueError):
        return (10**9, text)
    return (start, text)


def _sort_by_publication_year_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "publication_year_5y_bucket" not in frame.columns:
        return frame
    data = frame.copy()
    data["_year_key"] = data["publication_year_5y_bucket"].map(_publication_year_5y_bucket_sort_key)
    side_order = {"liberal": 0, "conservative": 1}
    if "ground_truth_side" in data.columns:
        data["_side_key"] = data["ground_truth_side"].map(lambda x: side_order.get(str(x), 10))
    sort_cols = ["_year_key"]
    if "family" in data.columns:
        sort_cols.append("family")
    if "_side_key" in data.columns:
        sort_cols.append("_side_key")
    data = data.sort_values(sort_cols).drop(columns=[col for col in ["_year_key", "_side_key"] if col in data.columns])
    return data.reset_index(drop=True)


def _theme_sort_key(value: object) -> tuple[int, str]:
    theme = str(value)
    try:
        return (IDEOLOGY_VOTE_THEME_ORDER.index(theme), theme)
    except ValueError:
        return (len(IDEOLOGY_VOTE_THEME_ORDER), theme)


def _normalize_weight_map(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, Mapping):
            return {}
        items = parsed.items()
    else:
        return {}

    weights: dict[str, float] = {}
    for theme, weight in items:
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if numeric_weight > 0:
            weights[str(theme)] = numeric_weight
    return dict(sorted(weights.items(), key=lambda item: _theme_sort_key(item[0])))


def _theme_weight_map_from_row(row: pd.Series) -> dict[str, float]:
    weights = _normalize_weight_map(row.get(THEME_WEIGHTS_COL))
    if weights:
        return weights

    raw_jel = row.get("jel_codes") if "jel_codes" in row else None
    if raw_jel in (None, "", []):
        raw_jel = row.get("jel_codes_raw") if "jel_codes_raw" in row else None
    if raw_jel not in (None, "", []):
        return ideology_theme_vote_details(raw_jel)["theme_weights"]

    counts = row.get(THEME_COUNTS_COL)
    if counts not in (None, "", {}, []):
        return ideology_theme_vote_weights(theme_counts=counts)

    primary_theme = str(row.get(THEME_COL) or "other").strip() or "other"
    return {primary_theme: 1.0}


def _expand_weighted_vote_themes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        data = frame.copy()
        data["jel_policy_theme"] = []
        data[THEME_WEIGHT_COL] = []
        return data

    data = frame.copy()
    data["_vote_theme_weight_map"] = data.apply(_theme_weight_map_from_row, axis=1)
    data["jel_policy_theme"] = data["_vote_theme_weight_map"].apply(lambda value: list(value.keys()))
    data[THEME_WEIGHT_COL] = data["_vote_theme_weight_map"].apply(lambda value: list(value.values()))
    data = data.explode(["jel_policy_theme", THEME_WEIGHT_COL], ignore_index=True)
    data[THEME_WEIGHT_COL] = pd.to_numeric(data[THEME_WEIGHT_COL], errors="coerce").fillna(0.0).astype(float)
    data = data[data[THEME_WEIGHT_COL] > 0].copy()
    data = data.sort_values(["triplet_key", "jel_policy_theme"], key=lambda col: col.map(_theme_sort_key) if col.name == "jel_policy_theme" else col)
    return data.drop(columns=["_vote_theme_weight_map"]).reset_index(drop=True)


def _prepare_frame(path: str, *, require_triplet_key: bool = True) -> pd.DataFrame:
    frame = load_dataset(path)
    if frame.empty:
        raise RuntimeError(f"No rows found in {path}")
    if require_triplet_key and "triplet_key" not in frame.columns:
        raise RuntimeError(f"Expected triplet_key column in {path}")

    for column, default in (
        (THEME_COL, "other"),
        ("model", "unknown"),
        ("family", "unknown"),
        ("parameter_bucket", "unknown"),
        ("region_bucket_label", "(missing)"),
        ("age_label", "(missing)"),
        ("gender_label", "(missing)"),
        ("ground_truth_side", "unlabeled"),
    ):
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)

    raw_jel = None
    if "jel_codes" in frame.columns:
        raw_jel = frame["jel_codes"]
    elif "jel_codes_raw" in frame.columns:
        raw_jel = frame["jel_codes_raw"]
    if raw_jel is not None:
        frame[THEME_COL] = raw_jel.apply(lambda value: ideology_theme_vote_details(value)["primary_theme"])
    elif THEME_COL not in frame.columns or frame[THEME_COL].eq("other").all():
        raw_jel = None
        if "jel_codes" in frame.columns:
            raw_jel = frame["jel_codes"]
        elif "jel_codes_raw" in frame.columns:
            raw_jel = frame["jel_codes_raw"]
        if raw_jel is not None:
            frame[THEME_COL] = raw_jel.apply(lambda value: ideology_theme_vote_details(value)["primary_theme"])

    if THEME_WEIGHTS_COL not in frame.columns:
        frame[THEME_WEIGHTS_COL] = frame.apply(_theme_weight_map_from_row, axis=1)

    frame["jel_policy_theme"] = frame[THEME_COL].fillna("other")
    if "publication_year" in frame.columns:
        frame["publication_year_5y_bucket"] = frame["publication_year"].apply(_publication_year_5y_bucket)
    else:
        frame["publication_year_5y_bucket"] = "(missing)"

    for column in (
        "ground_truth_liberal",
        "predicted_liberal",
        "predicted_conservative",
        "ideology_triplet_labeled",
        "liberal_leaning_error",
        "conservative_leaning_error",
        "correct",
    ):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype(int)

    return frame


def _triplet_theme_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["jel_policy_theme", "n_triplets", "weighted_n_triplets", "share_triplets", "share_weighted_triplets"]
        )
    deduped = frame.drop_duplicates(subset=["triplet_key", "jel_policy_theme"]).copy()
    if THEME_WEIGHT_COL not in deduped.columns:
        deduped[THEME_WEIGHT_COL] = 1.0
    counts = (
        deduped.groupby("jel_policy_theme", dropna=False, observed=False)
        .agg(
            n_triplets=("triplet_key", "nunique"),
            weighted_n_triplets=(THEME_WEIGHT_COL, "sum"),
        )
        .reset_index()
    )
    counts["share_triplets"] = counts["n_triplets"] / counts["n_triplets"].sum()
    counts["share_weighted_triplets"] = counts["weighted_n_triplets"] / counts["weighted_n_triplets"].sum()
    counts["_theme_key"] = counts["jel_policy_theme"].map(_theme_sort_key)
    return counts.sort_values(["weighted_n_triplets", "_theme_key"], ascending=[False, True]).drop(columns=["_theme_key"]).reset_index(drop=True)


def _series_value(summary: pd.DataFrame, key_col: str, key: str, value_col: str) -> float | None:
    if summary.empty:
        return None
    matches = summary.loc[summary[key_col].astype(str) == key, value_col]
    if matches.empty:
        return None
    value = matches.iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _task1_sensitive_and_complement_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treat the analysis subset as fully sensitive and the full Task1 complement as non-sensitive."""
    subset_triplet_keys = set(frame["triplet_key"].astype(str).unique())
    sensitive = frame.copy()
    sensitive["triplet_key"] = sensitive["triplet_key"].astype(str)
    sensitive["ideology_sensitivity"] = "ideology_sensitive"

    full_results_dir = TASK1_RESULT_DIRS["econ"]
    full_frame = pd.DataFrame(load_task1_rows(results_dir=full_results_dir))
    if full_frame.empty:
        raise RuntimeError(f"No full Task1 rows found in {full_results_dir}")
    full_frame["triplet_key"] = full_frame["triplet_key"].astype(str)

    complement = full_frame[~full_frame["triplet_key"].isin(subset_triplet_keys)].copy()
    complement["ideology_sensitivity"] = "non_sensitive"
    return sensitive, complement


def _task1_hero_table(
    frame: pd.DataFrame,
    non_sensitive_frame: pd.DataFrame,
    accuracy_by_model_ground_truth: pd.DataFrame,
    accuracy_by_model_sensitivity: pd.DataFrame,
    bias_by_model: pd.DataFrame,
) -> pd.DataFrame:
    non_sensitive_full_complement = accuracy_by(non_sensitive_frame, ["family", "model"]).rename(
        columns={"accuracy": "Acc (Non-sensitive)"}
    )

    lib = (
        accuracy_by_model_ground_truth.loc[accuracy_by_model_ground_truth["ground_truth_side"] == "liberal", ["family", "model", "accuracy"]]
        .rename(columns={"accuracy": "Acc^lib"})
    )
    con = (
        accuracy_by_model_ground_truth.loc[accuracy_by_model_ground_truth["ground_truth_side"] == "conservative", ["family", "model", "accuracy"]]
        .rename(columns={"accuracy": "Acc^con"})
    )
    sensitive = (
        accuracy_by_model_sensitivity.loc[
            accuracy_by_model_sensitivity["ideology_sensitivity"] == "ideology_sensitive",
            ["family", "model", "accuracy"],
        ]
        .rename(columns={"accuracy": "Acc (Sensitive)"})
    )
    bias = bias_by_model[["family", "model", "bias_score"]].rename(columns={"bias_score": "Bias_dir"})

    hero = sensitive[["family", "model", "Acc (Sensitive)"]].copy()
    for piece in (non_sensitive_full_complement, lib, con, bias):
        hero = hero.merge(piece, on=["family", "model"], how="left")

    hero["Δ_acc"] = hero["Acc^lib"] - hero["Acc^con"]
    hero["_family_key"] = hero["family"].map(family_sort_key)
    hero["_model_key"] = hero.apply(lambda row: model_sort_key(row["model"], row["family"]), axis=1)
    hero = hero.sort_values(["_family_key", "_model_key"]).drop(columns=["_family_key", "_model_key"]).reset_index(drop=True)

    ordered_cols = [
        "family",
        "model",
        "Acc (Sensitive)",
        "Acc (Non-sensitive)",
        "Acc^lib",
        "Acc^con",
        "Δ_acc",
        "Bias_dir",
    ]
    hero = hero[ordered_cols]
    return hero.where(pd.notna(hero), None)


def main() -> None:
    ensure_output_dirs()
    source_frame = _prepare_frame(str(SOURCE_CATALOG_JSONL))
    frame = _prepare_frame(str(TASK1_ANALYSIS_JSONL))
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    sensitivity_sensitive_frame, sensitivity_non_sensitive_frame = _task1_sensitive_and_complement_frames(frame)
    sensitivity_frame = pd.concat(
        [sensitivity_sensitive_frame, sensitivity_non_sensitive_frame],
        ignore_index=True,
        sort=False,
    )
    print(
        "[task1_vote_theme] loaded rows:",
        f"subset_predictions={len(frame)}",
        f"subset_triplets={frame['triplet_key'].nunique()}",
        f"complement_predictions={len(sensitivity_non_sensitive_frame)}",
        f"complement_triplets={sensitivity_non_sensitive_frame['triplet_key'].nunique()}",
    )
    weighted_source_frame = _expand_weighted_vote_themes(source_frame)
    weighted_frame = _expand_weighted_vote_themes(frame)
    weighted_labeled = _expand_weighted_vote_themes(labeled)

    source_theme_counts = _triplet_theme_counts(weighted_source_frame)
    task1_theme_counts = _triplet_theme_counts(weighted_frame)
    labeled_theme_counts = _triplet_theme_counts(weighted_labeled)

    accuracy_by_ground_truth = accuracy_by(labeled, ["ground_truth_side"])
    accuracy_by_model_ground_truth = accuracy_by(labeled, ["family", "model", "ground_truth_side"])
    bias_by_model = bias_summary_by(labeled, ["family", "model"])
    accuracy_by_sensitivity = accuracy_by(sensitivity_frame, ["ideology_sensitivity"])
    accuracy_by_model_sensitivity = accuracy_by(sensitivity_frame, ["family", "model", "ideology_sensitivity"])
    accuracy_by_theme_ground_truth = accuracy_by(
        weighted_labeled, ["jel_policy_theme", "ground_truth_side"], weight_col=THEME_WEIGHT_COL
    )
    accuracy_by_model_theme_ground_truth = accuracy_by(
        weighted_labeled, ["model", "jel_policy_theme", "ground_truth_side"], weight_col=THEME_WEIGHT_COL
    )
    accuracy_by_family_theme_ground_truth = accuracy_by(
        weighted_labeled, ["family", "jel_policy_theme", "ground_truth_side"], weight_col=THEME_WEIGHT_COL
    )
    bias_by_theme = bias_summary_by(weighted_labeled, ["jel_policy_theme"], weight_col=THEME_WEIGHT_COL)
    bias_by_model_theme = bias_summary_by(weighted_labeled, ["model", "jel_policy_theme"], weight_col=THEME_WEIGHT_COL)

    accuracy_by_publication_year_5y = accuracy_by(labeled, ["publication_year_5y_bucket"])
    accuracy_by_publication_year_5y_ground_truth = accuracy_by(
        labeled, ["publication_year_5y_bucket", "ground_truth_side"]
    )
    accuracy_by_family_publication_year_5y_ground_truth = accuracy_by(
        labeled, ["publication_year_5y_bucket", "family", "ground_truth_side"]
    )
    accuracy_by_publication_year_5y = _sort_by_publication_year_bucket(accuracy_by_publication_year_5y)
    accuracy_by_publication_year_5y_ground_truth = _sort_by_publication_year_bucket(
        accuracy_by_publication_year_5y_ground_truth
    )
    accuracy_by_family_publication_year_5y_ground_truth = _sort_by_publication_year_bucket(
        accuracy_by_family_publication_year_5y_ground_truth
    )

    iv_terms = [
        "ground_truth_liberal",
        "C(jel_policy_theme)",
        "C(publication_year_5y_bucket)",
        "C(region_bucket_label)",
        "C(age_label)",
        "C(gender_label)",
        "C(family)",
    ]

    def fit_univariate_suite(dv: str, ivs: list[str], data: pd.DataFrame) -> pd.DataFrame:
        tables = []
        for iv in ivs:
            formula = f"{dv} ~ {iv}"
            print(f"[task1_vote_theme] fitting {formula}")
            table = fit_clustered_binomial(formula, data, cluster_col="triplet_key", weight_col=THEME_WEIGHT_COL)
            if tables:
                table = table[table["term"] != "Intercept"]
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    regression_accuracy = fit_univariate_suite("correct", iv_terms, weighted_labeled)
    regression_liberal_prediction = fit_univariate_suite("predicted_liberal", iv_terms, weighted_labeled)

    labeled_exploded = weighted_labeled.copy()
    for demo_col in ["region_bucket_label", "age_label", "gender_label"]:
        labeled_exploded = explode_for_criterion(labeled_exploded, demo_col)

    multivariate_formula_rhs = " + ".join(iv_terms)
    multivariate_accuracy_formula = f"correct ~ {multivariate_formula_rhs}"
    multivariate_liberal_formula = f"predicted_liberal ~ {multivariate_formula_rhs}"

    print(f"[task1_vote_theme] fitting {multivariate_accuracy_formula}")
    regression_accuracy_multivariate = fit_clustered_binomial(
        multivariate_accuracy_formula, labeled_exploded, cluster_col="triplet_key", weight_col=THEME_WEIGHT_COL
    )
    print(f"[task1_vote_theme] fitting {multivariate_liberal_formula}")
    regression_liberal_prediction_multivariate = fit_clustered_binomial(
        multivariate_liberal_formula, labeled_exploded, cluster_col="triplet_key", weight_col=THEME_WEIGHT_COL
    )
    hero_table = _task1_hero_table(
        frame,
        sensitivity_non_sensitive_frame,
        accuracy_by_model_ground_truth=accuracy_by_model_ground_truth,
        accuracy_by_model_sensitivity=accuracy_by_model_sensitivity,
        bias_by_model=bias_by_model,
    )

    print("[task1_vote_theme] saving outputs")
    save_frame(source_theme_counts, TABLES_DIR / "task1_vote_theme_source_catalog_triplet_counts.csv")
    save_frame(task1_theme_counts, TABLES_DIR / "task1_vote_theme_task1_triplet_counts.csv")
    save_frame(labeled_theme_counts, TABLES_DIR / "task1_vote_theme_task1_labeled_triplet_counts.csv")
    save_frame(accuracy_by_ground_truth, TABLES_DIR / "task1_accuracy_by_ground_truth_side.csv")
    save_frame(accuracy_by_model_ground_truth, TABLES_DIR / "task1_accuracy_by_model_and_ground_truth_side.csv")
    save_frame(accuracy_by_sensitivity, TABLES_DIR / "task1_accuracy_by_ideology_sensitivity.csv")
    save_frame(accuracy_by_model_sensitivity, TABLES_DIR / "task1_accuracy_by_model_and_ideology_sensitivity.csv")
    save_frame(bias_by_model, TABLES_DIR / "task1_bias_by_model.csv")
    save_frame(hero_table, TABLES_DIR / "table1_hero_model_performance_and_bias.csv")
    save_frame(accuracy_by_theme_ground_truth, TABLES_DIR / "task1_accuracy_by_vote_theme_and_ground_truth_side.csv")
    save_frame(
        accuracy_by_model_theme_ground_truth,
        TABLES_DIR / "task1_accuracy_by_model_vote_theme_and_ground_truth_side.csv",
    )
    save_frame(
        accuracy_by_family_theme_ground_truth,
        TABLES_DIR / "task1_accuracy_by_family_vote_theme_and_ground_truth_side.csv",
    )
    save_frame(bias_by_theme, TABLES_DIR / "task1_bias_by_vote_theme.csv")
    save_frame(bias_by_model_theme, TABLES_DIR / "task1_bias_by_model_and_vote_theme.csv")
    save_frame(accuracy_by_publication_year_5y, TABLES_DIR / "task1_accuracy_by_publication_year_5y.csv")
    save_frame(
        accuracy_by_publication_year_5y_ground_truth,
        TABLES_DIR / "task1_accuracy_by_publication_year_5y_and_ground_truth_side.csv",
    )
    save_frame(
        accuracy_by_family_publication_year_5y_ground_truth,
        TABLES_DIR / "task1_accuracy_by_family_and_publication_year_5y_and_ground_truth_side.csv",
    )
    save_frame(
        regression_accuracy,
        TABLES_DIR / "task1_regression_accuracy_vote_theme.csv",
        formula="correct ~ [univariate models for each IV; uses weighted multi-label vote-theme rows via C(jel_policy_theme)]",
    )
    save_frame(
        regression_liberal_prediction,
        TABLES_DIR / "task1_regression_predicted_liberal_vote_theme.csv",
        formula="predicted_liberal ~ [univariate models for each IV; uses weighted multi-label vote-theme rows via C(jel_policy_theme)]",
    )
    save_frame(
        regression_accuracy_multivariate,
        TABLES_DIR / "task1_regression_accuracy_multivariate_vote_theme.csv",
        formula=multivariate_accuracy_formula,
    )
    save_frame(
        regression_liberal_prediction_multivariate,
        TABLES_DIR / "task1_regression_predicted_liberal_multivariate_vote_theme.csv",
        formula=multivariate_liberal_formula,
    )

    unique_triplets = int(frame["triplet_key"].nunique())
    labeled_triplets = int(labeled["triplet_key"].nunique())
    liberal_acc = _series_value(accuracy_by_ground_truth, "ground_truth_side", "liberal", "accuracy")
    conservative_acc = _series_value(accuracy_by_ground_truth, "ground_truth_side", "conservative", "accuracy")
    weighted_theme_rows = float(weighted_labeled[THEME_WEIGHT_COL].sum()) if not weighted_labeled.empty else 0.0

    report_path = REPORTS_DIR / "task1_bias_report_vote_theme.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Task1 Bias Report (Vote Theme) / Task1 편향 분석 보고서 (Vote Theme)\n\n")
        handle.write("## 1. JEL vote 매핑 규칙\n\n")
        handle.write("- raw JEL code 문자열을 파싱해 코드 리스트로 만든다.\n")
        handle.write("- 각 코드에 대해 theme prefix와 `startswith`만 비교한다.\n")
        handle.write("- 하나의 코드가 여러 theme와 매칭되면 모든 theme count를 올린다.\n")
        handle.write("- 분석용 subgroup row는 matched theme 전체로 확장한다.\n")
        handle.write("- 최다 득표 theme가 하나면 그 theme에 100% 배정한다.\n")
        handle.write("- 최다 득표가 정확히 동률일 때만 tied theme들에 균등 분할한다. 예: 0.5/0.5, 0.33/0.33/0.33.\n")
        handle.write("- `jel_policy_theme_vote_primary`는 진단용으로 유지하지만, subgroup 통계/회귀는 weighted multi-label theme를 사용한다.\n\n")

        handle.write("## 2. 데이터 범위\n\n")
        handle.write(f"- Task1 prediction rows: {len(frame)}\n")
        handle.write(f"- Task1 unique triplets: {unique_triplets}\n")
        handle.write(f"- ideology-labeled triplets: {labeled_triplets}\n")
        handle.write(f"- weighted labeled theme mass: {weighted_theme_rows:.1f}\n")
        if liberal_acc is not None:
            handle.write(f"- liberal ground truth accuracy: {liberal_acc * 100:.2f}%\n")
        if conservative_acc is not None:
            handle.write(f"- conservative ground truth accuracy: {conservative_acc * 100:.2f}%\n")
        handle.write("\n")

        handle.write("## 3. Theme 분포 파일\n\n")
        handle.write("- 각 파일은 `n_triplets`(theme에 걸친 unique triplet 수)와 `weighted_n_triplets`(theme weight 합)를 함께 제공한다.\n")
        handle.write("- `task1_vote_theme_source_catalog_triplet_counts.csv`: 전체 source catalog 기준 theme 분포\n")
        handle.write("- `task1_vote_theme_task1_triplet_counts.csv`: Task1 기준 theme 분포\n")
        handle.write("- `task1_vote_theme_task1_labeled_triplet_counts.csv`: ideology-labeled Task1 triplet 기준 theme 분포\n\n")

        handle.write("## 4. 분석 산출물\n\n")
        handle.write("- `table1_hero_model_performance_and_bias.csv`\n")
        handle.write("- `task1_accuracy_by_ground_truth_side.csv`\n")
        handle.write("- `task1_accuracy_by_model_and_ground_truth_side.csv`\n")
        handle.write("- `task1_accuracy_by_ideology_sensitivity.csv`\n")
        handle.write("- `task1_accuracy_by_model_and_ideology_sensitivity.csv`\n")
        handle.write("- `task1_bias_by_model.csv`\n")
        handle.write("- `task1_accuracy_by_vote_theme_and_ground_truth_side.csv`\n")
        handle.write("- `task1_accuracy_by_model_vote_theme_and_ground_truth_side.csv`\n")
        handle.write("- `task1_accuracy_by_family_vote_theme_and_ground_truth_side.csv`\n")
        handle.write("- `task1_bias_by_vote_theme.csv`\n")
        handle.write("- `task1_bias_by_model_and_vote_theme.csv`\n")
        handle.write("- `task1_regression_accuracy_vote_theme.md`\n")
        handle.write("- `task1_regression_predicted_liberal_vote_theme.md`\n")
        handle.write("- `task1_regression_accuracy_multivariate_vote_theme.md`\n")
        handle.write("- `task1_regression_predicted_liberal_multivariate_vote_theme.md`\n")

    print(
        json.dumps(
            {
                "task1_prediction_rows": int(len(frame)),
                "task1_unique_triplets": unique_triplets,
                "task1_labeled_triplets": labeled_triplets,
                "weighted_theme_mass": weighted_theme_rows,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
