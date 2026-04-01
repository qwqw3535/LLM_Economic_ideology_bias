"""Task 1 ideology-side analysis on the bootstrap subset."""

from __future__ import annotations

import json

import pandas as pd

from .analysis_helpers import accuracy_by, bias_summary_by, explode_for_criterion, fit_clustered_binomial, load_dataset, save_frame
from .load_results import load_task1_rows
from .paths import REPORTS_DIR, TABLES_DIR, TASK1_ANALYSIS_JSONL, TASK1_RESULT_DIRS, ensure_output_dirs
from .utils import family_sort_key, model_sort_key


IDEOLOGY_GROUND_TRUTH = {"liberal", "conservative"}


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


def _prepare_frame() -> pd.DataFrame:
    frame = load_dataset(TASK1_ANALYSIS_JSONL)
    if frame.empty:
        raise RuntimeError(f"No Task1 analysis rows found in {TASK1_ANALYSIS_JSONL}")
    for column, default in (
        ("model", "unknown"),
        ("family", "unknown"),
        ("parameter_bucket", "unknown"),
        ("jel_policy_theme", "other"),
        ("region_bucket_label", "(missing)"),
        ("age_label", "(missing)"),
        ("gender_label", "(missing)"),
        ("ground_truth_side", "unlabeled"),
    ):
        frame[column] = frame[column].fillna(default)
    frame["publication_year_5y_bucket"] = frame["publication_year"].apply(_publication_year_5y_bucket)
    frame["ground_truth_liberal"] = frame["ground_truth_liberal"].fillna(0).astype(int)
    frame["predicted_liberal"] = frame["predicted_liberal"].fillna(0).astype(int)
    frame["predicted_conservative"] = frame["predicted_conservative"].fillna(0).astype(int)
    frame["ideology_triplet_labeled"] = frame["ideology_triplet_labeled"].fillna(0).astype(int)
    frame["liberal_leaning_error"] = frame["liberal_leaning_error"].fillna(0).astype(int)
    frame["conservative_leaning_error"] = frame["conservative_leaning_error"].fillna(0).astype(int)
    return frame


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
    accuracy_by_model_ground_truth: pd.DataFrame,
    accuracy_by_model_sensitivity: pd.DataFrame,
    bias_by_model: pd.DataFrame,
) -> pd.DataFrame:
    _, non_sensitive_frame = _task1_sensitive_and_complement_frames(frame)
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
    return hero[ordered_cols].where(pd.notna(hero), None)


def main() -> None:
    ensure_output_dirs()
    frame = _prepare_frame()
    labeled = frame[frame["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    sensitivity_sensitive_frame, sensitivity_non_sensitive_frame = _task1_sensitive_and_complement_frames(frame)
    sensitivity_frame = pd.concat(
        [sensitivity_sensitive_frame, sensitivity_non_sensitive_frame],
        ignore_index=True,
        sort=False,
    )

    unique_triplets = int(frame["triplet_key"].nunique())
    labeled_triplets = int(labeled["triplet_key"].nunique())
    labeled_share = labeled_triplets / unique_triplets if unique_triplets else 0.0

    accuracy_by_ground_truth = accuracy_by(labeled, ["ground_truth_side"])
    accuracy_by_model_ground_truth = accuracy_by(labeled, ["family", "model", "ground_truth_side"])
    bias_by_model = bias_summary_by(labeled, ["family", "model"])
    accuracy_by_theme_ground_truth = accuracy_by(labeled, ["jel_policy_theme", "ground_truth_side"])
    accuracy_by_model_theme_ground_truth = accuracy_by(labeled, ["model", "jel_policy_theme", "ground_truth_side"])
    accuracy_by_family_theme_ground_truth = accuracy_by(labeled, ["family", "jel_policy_theme", "ground_truth_side"])
    bias_by_theme = bias_summary_by(labeled, ["jel_policy_theme"])
    bias_by_model_theme = bias_summary_by(labeled, ["model", "jel_policy_theme"])

    accuracy_by_sensitivity = accuracy_by(sensitivity_frame, ["ideology_sensitivity"])
    accuracy_by_model_sensitivity = accuracy_by(sensitivity_frame, ["family", "model", "ideology_sensitivity"])

    # Demographic analyses
    # Explode demographic fields for analysis
    frame_gender = explode_for_criterion(frame, "gender_label")
    frame_race = explode_for_criterion(frame, "race_label")
    frame_region = explode_for_criterion(frame, "region_bucket_label")
    frame_age = explode_for_criterion(frame, "age_label")

    labeled_gender = frame_gender[frame_gender["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled_race = frame_race[frame_race["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled_region = frame_region[frame_region["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()
    labeled_age = frame_age[frame_age["ground_truth_side"].isin(IDEOLOGY_GROUND_TRUTH)].copy()

    # Gender-based analyses
    accuracy_by_gender = accuracy_by(labeled_gender, ["gender_label"])
    accuracy_by_gender_ground_truth = accuracy_by(labeled_gender, ["gender_label", "ground_truth_side"])
    bias_by_gender = bias_summary_by(labeled_gender, ["gender_label"])
    accuracy_by_model_gender = accuracy_by(labeled_gender, ["family", "model", "gender_label"])

    # Race-based analyses
    accuracy_by_race = accuracy_by(labeled_race, ["race_label"])
    accuracy_by_race_ground_truth = accuracy_by(labeled_race, ["race_label", "ground_truth_side"])
    bias_by_race = bias_summary_by(labeled_race, ["race_label"])
    accuracy_by_model_race = accuracy_by(labeled_race, ["family", "model", "race_label"])

    # Region-based analyses
    accuracy_by_region = accuracy_by(labeled_region, ["region_bucket_label"])
    accuracy_by_region_ground_truth = accuracy_by(labeled_region, ["region_bucket_label", "ground_truth_side"])
    bias_by_region = bias_summary_by(labeled_region, ["region_bucket_label"])
    accuracy_by_model_region = accuracy_by(labeled_region, ["family", "model", "region_bucket_label"])

    # Age-based analyses
    accuracy_by_age = accuracy_by(labeled_age, ["age_label"])
    accuracy_by_age_ground_truth = accuracy_by(labeled_age, ["age_label", "ground_truth_side"])
    bias_by_age = bias_summary_by(labeled_age, ["age_label"])
    accuracy_by_model_age = accuracy_by(labeled_age, ["family", "model", "age_label"])

    # Publication year (5-year bucket) analyses
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
            table = fit_clustered_binomial(formula, data, cluster_col="triplet_key")
            # Only keep the Intercept for the first IV to keep the table clean
            if tables:
                table = table[table["term"] != "Intercept"]
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    regression_accuracy = fit_univariate_suite("correct", iv_terms, labeled)
    regression_liberal_prediction = fit_univariate_suite("predicted_liberal", iv_terms, labeled)

    # Multivariate regressions with all IVs together
    # Need to explode all demographic fields for multivariate regression
    labeled_exploded = labeled.copy()
    for demo_col in ["region_bucket_label", "age_label", "gender_label"]:
        labeled_exploded = explode_for_criterion(labeled_exploded, demo_col)

    multivariate_formula_rhs = " + ".join(iv_terms)
    multivariate_accuracy_formula = f"correct ~ {multivariate_formula_rhs}"
    multivariate_liberal_formula = f"predicted_liberal ~ {multivariate_formula_rhs}"

    regression_accuracy_multivariate = fit_clustered_binomial(
        multivariate_accuracy_formula, labeled_exploded, cluster_col="triplet_key"
    )
    regression_liberal_prediction_multivariate = fit_clustered_binomial(
        multivariate_liberal_formula, labeled_exploded, cluster_col="triplet_key"
    )
    hero_table = _task1_hero_table(
        frame,
        accuracy_by_model_ground_truth=accuracy_by_model_ground_truth,
        accuracy_by_model_sensitivity=accuracy_by_model_sensitivity,
        bias_by_model=bias_by_model,
    )

    accuracy_formula = "correct ~ [univariate models for each IV]"
    liberal_formula = "predicted_liberal ~ [univariate models for each IV]"


    save_frame(accuracy_by_ground_truth, TABLES_DIR / "task1_accuracy_by_ground_truth_side.csv")
    save_frame(accuracy_by_model_ground_truth, TABLES_DIR / "task1_accuracy_by_model_and_ground_truth_side.csv")
    save_frame(bias_by_model, TABLES_DIR / "task1_bias_by_model.csv")
    save_frame(hero_table, TABLES_DIR / "table1_hero_model_performance_and_bias.csv")
    save_frame(accuracy_by_theme_ground_truth, TABLES_DIR / "task1_accuracy_by_jel_theme_and_ground_truth_side.csv")
    save_frame(accuracy_by_model_theme_ground_truth, TABLES_DIR / "task1_accuracy_by_model_jel_theme_and_ground_truth_side.csv")
    save_frame(accuracy_by_family_theme_ground_truth, TABLES_DIR / "task1_accuracy_by_family_jel_theme_and_ground_truth_side.csv")
    save_frame(bias_by_theme, TABLES_DIR / "task1_bias_by_jel_theme.csv")
    save_frame(bias_by_model_theme, TABLES_DIR / "task1_bias_by_model_and_jel_theme.csv")
    save_frame(accuracy_by_sensitivity, TABLES_DIR / "task1_accuracy_by_ideology_sensitivity.csv")
    save_frame(accuracy_by_model_sensitivity, TABLES_DIR / "task1_accuracy_by_model_and_ideology_sensitivity.csv")

    # Save demographic analyses
    save_frame(accuracy_by_gender, TABLES_DIR / "task1_accuracy_by_gender.csv")
    save_frame(accuracy_by_gender_ground_truth, TABLES_DIR / "task1_accuracy_by_gender_and_ground_truth_side.csv")
    save_frame(bias_by_gender, TABLES_DIR / "task1_bias_by_gender.csv")
    save_frame(accuracy_by_model_gender, TABLES_DIR / "task1_accuracy_by_model_and_gender.csv")

    save_frame(accuracy_by_race, TABLES_DIR / "task1_accuracy_by_race.csv")
    save_frame(accuracy_by_race_ground_truth, TABLES_DIR / "task1_accuracy_by_race_and_ground_truth_side.csv")
    save_frame(bias_by_race, TABLES_DIR / "task1_bias_by_race.csv")
    save_frame(accuracy_by_model_race, TABLES_DIR / "task1_accuracy_by_model_and_race.csv")

    save_frame(accuracy_by_region, TABLES_DIR / "task1_accuracy_by_region.csv")
    save_frame(accuracy_by_region_ground_truth, TABLES_DIR / "task1_accuracy_by_region_and_ground_truth_side.csv")
    save_frame(bias_by_region, TABLES_DIR / "task1_bias_by_region.csv")
    save_frame(accuracy_by_model_region, TABLES_DIR / "task1_accuracy_by_model_and_region.csv")

    save_frame(accuracy_by_age, TABLES_DIR / "task1_accuracy_by_age.csv")
    save_frame(accuracy_by_age_ground_truth, TABLES_DIR / "task1_accuracy_by_age_and_ground_truth_side.csv")
    save_frame(bias_by_age, TABLES_DIR / "task1_bias_by_age.csv")
    save_frame(accuracy_by_model_age, TABLES_DIR / "task1_accuracy_by_model_and_age.csv")

    save_frame(accuracy_by_publication_year_5y, TABLES_DIR / "task1_accuracy_by_publication_year_5y.csv")
    save_frame(
        accuracy_by_publication_year_5y_ground_truth,
        TABLES_DIR / "task1_accuracy_by_publication_year_5y_and_ground_truth_side.csv",
    )
    save_frame(
        accuracy_by_family_publication_year_5y_ground_truth,
        TABLES_DIR / "task1_accuracy_by_family_and_publication_year_5y_and_ground_truth_side.csv",
    )
    save_frame(regression_accuracy, TABLES_DIR / "task1_regression_accuracy.csv", formula=accuracy_formula)
    save_frame(regression_liberal_prediction, TABLES_DIR / "task1_regression_predicted_liberal.csv", formula=liberal_formula)
    save_frame(
        regression_accuracy_multivariate,
        TABLES_DIR / "task1_regression_accuracy_multivariate.csv",
        formula=multivariate_accuracy_formula,
    )
    save_frame(
        regression_liberal_prediction_multivariate,
        TABLES_DIR / "task1_regression_predicted_liberal_multivariate.csv",
        formula=multivariate_liberal_formula,
    )

    liberal_acc = _series_value(accuracy_by_ground_truth, "ground_truth_side", "liberal", "accuracy")
    conservative_acc = _series_value(accuracy_by_ground_truth, "ground_truth_side", "conservative", "accuracy")

    report_path = REPORTS_DIR / "task1_bias_report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Task1 Bias Report / Task1 편향 분석 보고서\n\n")
        handle.write("## 1. 분석 정의\n\n")
        handle.write("- 이번 Task1 분석은 triplet 자체를 `more_state/less_state`로 나누지 않는다.\n")
        handle.write("- 각 triplet에 이미 있는 `economic_liberal_preferred_sign`과 `economic_conservative_preferred_sign`를 그대로 쓴다.\n")
        handle.write("- `ground_truth_side=liberal`은 정답 sign이 liberal preferred sign과 일치한다는 뜻이다.\n")
        handle.write("- `ground_truth_side=conservative`는 정답 sign이 conservative preferred sign과 일치한다는 뜻이다.\n")
        handle.write("- `predicted_liberal`/`predicted_conservative`도 같은 방식으로 모델 예측 sign이 어느 preferred sign과 맞았는지 본다.\n")
        handle.write("- bias score 정의: `(liberal_leaning_errors - conservative_leaning_errors) / all ideology triplet errors`\n\n")

        handle.write("## 2. 데이터 범위\n\n")
        handle.write(f"- 전체 고유 triplet 수: {unique_triplets}\n")
        handle.write(f"- liberal/conservative 정답 축으로 라벨링 가능한 고유 triplet 수: {labeled_triplets} ({labeled_share * 100:.2f}%)\n")
        if liberal_acc is not None:
            handle.write(f"- liberal ground truth 정확도: {liberal_acc * 100:.2f}%\n")
        if conservative_acc is not None:
            handle.write(f"- conservative ground truth 정확도: {conservative_acc * 100:.2f}%\n")
        handle.write("\n")

        handle.write("## 3. 주요 산출물\n\n")
        handle.write("### 3.1 Ideology 기반 분석\n\n")
        handle.write("- `task1_accuracy_by_model_and_ground_truth_side.csv`: 모델별 liberal/conservative 정답 문제 정확도 비교\n")
        handle.write("- `task1_bias_by_model.csv`: 모델별 liberal answer rate, conservative answer rate, bias score\n")
        handle.write("- `task1_accuracy_by_model_jel_theme_and_ground_truth_side.csv`: JEL theme별, 모델별 liberal/conservative 정답 문제 정확도\n")
        handle.write("- `task1_bias_by_model_and_jel_theme.csv`: JEL theme별, 모델별 bias summary\n")
        handle.write("- `task1_accuracy_by_ideology_sensitivity.csv`: 이념 민감도(sensitive vs non-sensitive)별 정확도 비교\n")
        handle.write("- `task1_accuracy_by_model_and_ideology_sensitivity.csv`: 모델별 이념 민감도 정확도 비교\n")
        handle.write("- `task1_regression_accuracy.md`: 정확도를 종속변수로 한 회귀 (Markdown 보고서)\n")
        handle.write("- `task1_regression_predicted_liberal.md`: liberal preferred sign을 대답할 확률을 종속변수로 한 회귀 (Markdown 보고서)\n\n")
        handle.write("### 3.2 Demographic 기반 분석\n\n")
        handle.write("#### Gender 분석\n")
        handle.write("- `task1_accuracy_by_gender.csv`: Gender별 정확도\n")
        handle.write("- `task1_accuracy_by_gender_and_ground_truth_side.csv`: Gender별 liberal/conservative 정답 문제 정확도\n")
        handle.write("- `task1_bias_by_gender.csv`: Gender별 bias summary\n")
        handle.write("- `task1_accuracy_by_model_and_gender.csv`: 모델별, Gender별 정확도\n\n")
        handle.write("#### Race 분석\n")
        handle.write("- `task1_accuracy_by_race.csv`: Race별 정확도\n")
        handle.write("- `task1_accuracy_by_race_and_ground_truth_side.csv`: Race별 liberal/conservative 정답 문제 정확도\n")
        handle.write("- `task1_bias_by_race.csv`: Race별 bias summary\n")
        handle.write("- `task1_accuracy_by_model_and_race.csv`: 모델별, Race별 정확도\n\n")
        handle.write("#### Region 분석\n")
        handle.write("- `task1_accuracy_by_region.csv`: Region별 정확도\n")
        handle.write("- `task1_accuracy_by_region_and_ground_truth_side.csv`: Region별 liberal/conservative 정답 문제 정확도\n")
        handle.write("- `task1_bias_by_region.csv`: Region별 bias summary\n")
        handle.write("- `task1_accuracy_by_model_and_region.csv`: 모델별, Region별 정확도\n\n")
        handle.write("#### Age 분석\n")
        handle.write("- `task1_accuracy_by_age.csv`: Age별 정확도\n")
        handle.write("- `task1_accuracy_by_age_and_ground_truth_side.csv`: Age별 liberal/conservative 정답 문제 정확도\n")
        handle.write("- `task1_bias_by_age.csv`: Age별 bias summary\n")
        handle.write("- `task1_accuracy_by_model_and_age.csv`: 모델별, Age별 정확도\n\n")
        handle.write("#### Publication Year (5-year bucket) 분석\n")
        handle.write("- `task1_accuracy_by_publication_year_5y.csv`: publication year 5년 구간별 정확도\n")
        handle.write("- `task1_accuracy_by_publication_year_5y_and_ground_truth_side.csv`: publication year 5년 구간별 liberal/conservative 정답 문제 정확도\n")
        handle.write("- `task1_accuracy_by_family_and_publication_year_5y_and_ground_truth_side.csv`: publication year 5년 구간별, 모델 family별 liberal/conservative 정답 문제 정확도\n\n")

        handle.write("## 4. JEL Theme 매핑\n\n")
        handle.write("- `labor_market`\n")
        handle.write("- `welfare_redistribution`\n")
        handle.write("- `taxation`\n")
        handle.write("- `financial_regulation`\n")
        handle.write("- `trade`\n")
        handle.write("- `immigration`\n")
        handle.write("- `environment_climate`\n")
        handle.write("- `education_health`\n")
        handle.write("- `development_economics`\n")
        handle.write("- 매핑 구현 파일: `extended/ideology_bias/jel.py`\n")

    print(
        json.dumps(
            {
                "prediction_rows": int(len(frame)),
                "unique_triplets": unique_triplets,
                "labeled_triplets": labeled_triplets,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
