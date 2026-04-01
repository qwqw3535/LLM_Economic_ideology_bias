import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re

from .jel import IDEOLOGY_THEME_ORDER, IDEOLOGY_VOTE_THEME_ORDER
from .paths import FIGURES_DIR, TABLES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THEME_ORDER = IDEOLOGY_VOTE_THEME_ORDER
THEME_SOURCE = "vote"

SIDE_ORDER = ["liberal", "conservative"]

SIDE_PALETTE = {"liberal": "#2563eb", "conservative": "#dc2626"}


def _theme_accuracy_table() -> Path:
    if THEME_SOURCE == "legacy":
        return TABLES_DIR / "task1_accuracy_by_jel_theme_and_ground_truth_side.csv"
    return TABLES_DIR / "task1_accuracy_by_vote_theme_and_ground_truth_side.csv"


def _theme_family_accuracy_table() -> Path:
    if THEME_SOURCE == "legacy":
        return TABLES_DIR / "task1_accuracy_by_family_jel_theme_and_ground_truth_side.csv"
    return TABLES_DIR / "task1_accuracy_by_family_vote_theme_and_ground_truth_side.csv"


def _theme_gap_table() -> Path:
    if THEME_SOURCE == "legacy":
        return TABLES_DIR / "task1_accuracy_by_family_jel_theme_and_ground_truth_side.csv"
    return TABLES_DIR / "task1_accuracy_by_family_vote_theme_and_ground_truth_side.csv"


def _year_bucket_sort_key(label: str) -> tuple[int, str]:
    if pd.isna(label):
        return (10**9, "(missing)")
    text = str(label)
    match = re.match(r"^(\d{4})-\d{4}$", text)
    if match:
        return (int(match.group(1)), text)
    return (10**9, text)


def _save_empty_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_jel_and_ideology():
    """Figure 3: bar chart with counts + environment_climate conservative shown as 0."""
    df = pd.read_csv(_theme_accuracy_table())
    df = df[df["ground_truth_side"].isin(["liberal", "conservative"])]

    # Ensure all theme × side combinations exist (fill missing with 0)
    themes_in_data = sorted(df["jel_policy_theme"].unique())
    theme_order_used = [t for t in THEME_ORDER if t in themes_in_data]
    # Add any themes not in the canonical list
    for t in themes_in_data:
        if t not in theme_order_used:
            theme_order_used.append(t)

    if not theme_order_used:
        _save_empty_figure(
            FIGURES_DIR / "task1_bias_accuracy_by_jel_and_ideology.png",
            "Task 1 Accuracy by Policy Theme and Ideology",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    full_index = pd.MultiIndex.from_product(
        [theme_order_used, SIDE_ORDER],
        names=["jel_policy_theme", "ground_truth_side"],
    )
    df = df.set_index(["jel_policy_theme", "ground_truth_side"]).reindex(full_index).reset_index()
    df["accuracy"] = df["accuracy"].fillna(0)
    df["n_triplets"] = df["n_triplets"].fillna(0).astype(int)

    # Custom order
    df["jel_policy_theme"] = pd.Categorical(df["jel_policy_theme"], categories=theme_order_used, ordered=True)
    df["ground_truth_side"] = pd.Categorical(df["ground_truth_side"], categories=SIDE_ORDER, ordered=True)
    df = df.sort_values(["jel_policy_theme", "ground_truth_side"])

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.35
    x = np.arange(len(theme_order_used))

    for i, side in enumerate(SIDE_ORDER):
        subset = df[df["ground_truth_side"] == side]
        bars = ax.bar(
            x + i * bar_width,
            subset["accuracy"].values,
            bar_width,
            label=side,
            color=SIDE_PALETTE[side],
            alpha=0.85,
        )
        # Add triplet count annotation on each bar
        for bar, (_, row) in zip(bars, subset.iterrows()):
            count = int(row["n_triplets"])
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_title("Task 1 Accuracy by Policy Theme and Ideology", fontsize=14, fontweight="bold")
    ax.set_xlabel("Policy Theme", fontsize=11)
    ax.set_ylabel("Accuracy (proportion of correct predictions)", fontsize=11)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(theme_order_used, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(title="ground_truth_side")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_bias_accuracy_by_jel_and_ideology.png", dpi=300)
    plt.close()


def plot_accuracy_by_jel_ideology_model_family():
    """Figure 4: split by jel_policy_theme, within each: model family × lib/cons order."""
    df = pd.read_csv(_theme_family_accuracy_table())
    df = df[df["ground_truth_side"].isin(["liberal", "conservative"])]

    themes_in_data = sorted(df["jel_policy_theme"].unique())
    theme_order_used = [t for t in THEME_ORDER if t in themes_in_data]
    for t in themes_in_data:
        if t not in theme_order_used:
            theme_order_used.append(t)

    families = sorted(df["family"].unique())
    if not theme_order_used or not families:
        _save_empty_figure(
            FIGURES_DIR / "task1_bias_accuracy_by_jel_ideology_and_model.png",
            "Task 1 Accuracy by Policy Theme, Ideology, and Model Family",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    # Ensure all theme × family × side combinations exist
    full_index = pd.MultiIndex.from_product(
        [theme_order_used, families, SIDE_ORDER],
        names=["jel_policy_theme", "family", "ground_truth_side"],
    )
    df = df.set_index(["jel_policy_theme", "family", "ground_truth_side"]).reindex(full_index).reset_index()
    df["accuracy"] = df["accuracy"].fillna(0)
    df["n_triplets"] = df["n_triplets"].fillna(0).astype(int)

    # Create the faceted figure by jel_policy_theme
    per_theme_dir = FIGURES_DIR / "task1_by_jel_theme"
    per_theme_dir.mkdir(parents=True, exist_ok=True)

    for theme in theme_order_used:
        theme_df = df[df["jel_policy_theme"] == theme].copy()
        # Sort: family (alphabetical), then liberal before conservative
        theme_df["family"] = pd.Categorical(theme_df["family"], categories=families, ordered=True)
        theme_df["ground_truth_side"] = pd.Categorical(theme_df["ground_truth_side"], categories=SIDE_ORDER, ordered=True)
        theme_df = theme_df.sort_values(["family", "ground_truth_side"])

        # Create combined label for x-axis
        theme_df["x_label"] = theme_df["family"] .astype(str)+ " (" + theme_df["ground_truth_side"].astype(str) + ")"

        n_bars = len(theme_df)
        fig_height = max(4, 0.5 * n_bars + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        colors = [SIDE_PALETTE[s] for s in theme_df["ground_truth_side"]]
        bars = ax.barh(
            range(n_bars),
            theme_df["accuracy"].values,
            color=colors,
            alpha=0.85,
        )

        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(theme_df["x_label"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Accuracy (proportion of correct predictions)", fontsize=11)
        ax.set_title(f"Task 1 Accuracy: {theme}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add triplet count and accuracy value
        for bar_idx, (_, row) in enumerate(theme_df.iterrows()):
            count = int(row["n_triplets"])
            acc = row["accuracy"]
            ax.text(
                acc + 0.01,
                bar_idx,
                f"{acc*100:.1f}% (n={count})",
                ha="left",
                va="center",
                fontsize=8,
            )

        # Legend
        patches = [
            mpatches.Patch(color=SIDE_PALETTE["liberal"], label="liberal"),
            mpatches.Patch(color=SIDE_PALETTE["conservative"], label="conservative"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(per_theme_dir / f"task1_accuracy_{theme}.png", dpi=300)
        plt.close()

    # Also create an overview combined figure
    n_themes = len(theme_order_used)
    n_cols = 2
    n_rows = (n_themes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes_flat = axes.flatten() if n_themes > 1 else [axes]

    for idx, theme in enumerate(theme_order_used):
        ax = axes_flat[idx]
        theme_df = df[df["jel_policy_theme"] == theme].copy()
        theme_df["family"] = pd.Categorical(theme_df["family"], categories=families, ordered=True)
        theme_df["ground_truth_side"] = pd.Categorical(theme_df["ground_truth_side"], categories=SIDE_ORDER, ordered=True)
        theme_df = theme_df.sort_values(["family", "ground_truth_side"])

        bar_width = 0.35
        x = np.arange(len(families))

        for i, side in enumerate(SIDE_ORDER):
            subset = theme_df[theme_df["ground_truth_side"] == side]
            bars = ax.bar(
                x + i * bar_width,
                subset["accuracy"].values,
                bar_width,
                label=side if idx == 0 else "",
                color=SIDE_PALETTE[side],
                alpha=0.85,
            )
            for bar, (_, row) in zip(bars, subset.iterrows()):
                count = int(row["n_triplets"])
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )

        ax.set_title(theme, fontsize=11, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy (proportion correct)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(n_themes, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add shared legend
    patches = [
        mpatches.Patch(color=SIDE_PALETTE["liberal"], label="liberal"),
        mpatches.Patch(color=SIDE_PALETTE["conservative"], label="conservative"),
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.suptitle("Task 1 Accuracy by Policy Theme, Ideology, and Model Family", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURES_DIR / "task1_bias_accuracy_by_jel_ideology_and_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_publication_year_5y():
    df = pd.read_csv(TABLES_DIR / "task1_accuracy_by_publication_year_5y.csv")
    if df.empty:
        _save_empty_figure(
            FIGURES_DIR / "task1_accuracy_by_publication_year_5y.png",
            "Task 1 Accuracy by Publication Year (5-Year Buckets)",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    bucket_order = sorted(df["publication_year_5y_bucket"].astype(str).unique(), key=_year_bucket_sort_key)
    df["publication_year_5y_bucket"] = pd.Categorical(
        df["publication_year_5y_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    df = df.sort_values("publication_year_5y_bucket")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(
        df["publication_year_5y_bucket"].astype(str),
        df["accuracy"].values,
        color="#0f766e",
        alpha=0.85,
    )

    for bar, (_, row) in zip(bars, df.iterrows()):
        count = int(row["n_triplets"])
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("Task 1 Accuracy by Publication Year (5-Year Buckets)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Publication year (5-year bucket)", fontsize=11)
    ax.set_ylabel("Accuracy (proportion of correct predictions)", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_accuracy_by_publication_year_5y.png", dpi=300)
    plt.close()


def plot_accuracy_by_publication_year_5y_and_ideology():
    df = pd.read_csv(TABLES_DIR / "task1_accuracy_by_publication_year_5y_and_ground_truth_side.csv")
    df = df[df["ground_truth_side"].isin(SIDE_ORDER)]

    bucket_order = sorted(df["publication_year_5y_bucket"].astype(str).unique(), key=_year_bucket_sort_key)
    if not bucket_order:
        _save_empty_figure(
            FIGURES_DIR / "task1_accuracy_by_publication_year_5y_and_ideology.png",
            "Task 1 Accuracy by Publication Year (5-Year Buckets) and Ideology",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    full_index = pd.MultiIndex.from_product(
        [bucket_order, SIDE_ORDER],
        names=["publication_year_5y_bucket", "ground_truth_side"],
    )
    df = df.set_index(["publication_year_5y_bucket", "ground_truth_side"]).reindex(full_index).reset_index()
    df["accuracy"] = df["accuracy"].fillna(0)
    df["n_triplets"] = df["n_triplets"].fillna(0).astype(int)
    df["publication_year_5y_bucket"] = pd.Categorical(
        df["publication_year_5y_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    df["ground_truth_side"] = pd.Categorical(df["ground_truth_side"], categories=SIDE_ORDER, ordered=True)
    df = df.sort_values(["publication_year_5y_bucket", "ground_truth_side"])

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.35
    x = np.arange(len(bucket_order))

    for i, side in enumerate(SIDE_ORDER):
        subset = df[df["ground_truth_side"] == side]
        bars = ax.bar(
            x + i * bar_width,
            subset["accuracy"].values,
            bar_width,
            label=side,
            color=SIDE_PALETTE[side],
            alpha=0.85,
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            count = int(row["n_triplets"])
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_title("Task 1 Accuracy by Publication Year (5-Year Buckets) and Ideology", fontsize=14, fontweight="bold")
    ax.set_xlabel("Publication year (5-year bucket)", fontsize=11)
    ax.set_ylabel("Accuracy (proportion of correct predictions)", fontsize=11)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(bucket_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(title="ground_truth_side")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_accuracy_by_publication_year_5y_and_ideology.png", dpi=300)
    plt.close()


def plot_accuracy_by_publication_year_5y_ideology_model_family():
    df = pd.read_csv(TABLES_DIR / "task1_accuracy_by_family_and_publication_year_5y_and_ground_truth_side.csv")
    df = df[df["ground_truth_side"].isin(SIDE_ORDER)]

    bucket_order = sorted(df["publication_year_5y_bucket"].astype(str).unique(), key=_year_bucket_sort_key)
    families = sorted(df["family"].astype(str).unique())
    if not bucket_order or not families:
        _save_empty_figure(
            FIGURES_DIR / "task1_accuracy_by_publication_year_5y_ideology_and_model.png",
            "Task 1 Accuracy by Publication Year (5-Year Buckets), Ideology, and Model Family",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    full_index = pd.MultiIndex.from_product(
        [bucket_order, families, SIDE_ORDER],
        names=["publication_year_5y_bucket", "family", "ground_truth_side"],
    )
    df = df.set_index(["publication_year_5y_bucket", "family", "ground_truth_side"]).reindex(full_index).reset_index()
    df["accuracy"] = df["accuracy"].fillna(0)
    df["n_triplets"] = df["n_triplets"].fillna(0).astype(int)

    per_bucket_dir = FIGURES_DIR / "task1_by_publication_year_5y"
    per_bucket_dir.mkdir(parents=True, exist_ok=True)

    for bucket in bucket_order:
        bucket_df = df[df["publication_year_5y_bucket"] == bucket].copy()
        bucket_df["family"] = pd.Categorical(bucket_df["family"], categories=families, ordered=True)
        bucket_df["ground_truth_side"] = pd.Categorical(
            bucket_df["ground_truth_side"],
            categories=SIDE_ORDER,
            ordered=True,
        )
        bucket_df = bucket_df.sort_values(["family", "ground_truth_side"])
        bucket_df["x_label"] = bucket_df["family"].astype(str) + " (" + bucket_df["ground_truth_side"].astype(str) + ")"

        n_bars = len(bucket_df)
        fig_height = max(4, 0.5 * n_bars + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        colors = [SIDE_PALETTE[s] for s in bucket_df["ground_truth_side"]]
        ax.barh(range(n_bars), bucket_df["accuracy"].values, color=colors, alpha=0.85)

        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(bucket_df["x_label"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Accuracy (proportion of correct predictions)", fontsize=11)
        ax.set_title(f"Task 1 Accuracy: {bucket}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for bar_idx, (_, row) in enumerate(bucket_df.iterrows()):
            count = int(row["n_triplets"])
            acc = row["accuracy"]
            ax.text(
                acc + 0.01,
                bar_idx,
                f"{acc*100:.1f}% (n={count})",
                ha="left",
                va="center",
                fontsize=8,
            )

        patches = [
            mpatches.Patch(color=SIDE_PALETTE["liberal"], label="liberal"),
            mpatches.Patch(color=SIDE_PALETTE["conservative"], label="conservative"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(per_bucket_dir / f"task1_accuracy_{bucket}.png", dpi=300)
        plt.close()

    n_buckets = len(bucket_order)
    n_cols = 2
    n_rows = (n_buckets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes_flat = axes.flatten() if n_buckets > 1 else [axes]

    for idx, bucket in enumerate(bucket_order):
        ax = axes_flat[idx]
        bucket_df = df[df["publication_year_5y_bucket"] == bucket].copy()
        bucket_df["family"] = pd.Categorical(bucket_df["family"], categories=families, ordered=True)
        bucket_df["ground_truth_side"] = pd.Categorical(
            bucket_df["ground_truth_side"],
            categories=SIDE_ORDER,
            ordered=True,
        )
        bucket_df = bucket_df.sort_values(["family", "ground_truth_side"])

        bar_width = 0.35
        x = np.arange(len(families))

        for i, side in enumerate(SIDE_ORDER):
            subset = bucket_df[bucket_df["ground_truth_side"] == side]
            bars = ax.bar(
                x + i * bar_width,
                subset["accuracy"].values,
                bar_width,
                label=side if idx == 0 else "",
                color=SIDE_PALETTE[side],
                alpha=0.85,
            )
            for bar, (_, row) in zip(bars, subset.iterrows()):
                count = int(row["n_triplets"])
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )

        ax.set_title(bucket, fontsize=11, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy (proportion correct)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(n_buckets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    patches = [
        mpatches.Patch(color=SIDE_PALETTE["liberal"], label="liberal"),
        mpatches.Patch(color=SIDE_PALETTE["conservative"], label="conservative"),
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.suptitle(
        "Task 1 Accuracy by Publication Year (5-Year Buckets), Ideology, and Model Family",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        FIGURES_DIR / "task1_accuracy_by_publication_year_5y_ideology_and_model.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_bias_score_by_model():
    df = pd.read_csv(TABLES_DIR / "task1_bias_by_model.csv")
    if df.empty:
        _save_empty_figure(
            FIGURES_DIR / "task1_true_bias_score_by_model.png",
            "Task 1 Bias Score by Model",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return
    df = df.sort_values("bias_score", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["model"].astype(str), df["bias_score"].values, color="#8b5cf6", alpha=0.85)
    ax.invert_yaxis()
    ax.set_title("Task 1 Bias Score by Model")
    ax.set_xlabel("Bias Score ((Liberal Errors - Conservative Errors) / Total Errors)")
    ax.set_ylabel("")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_true_bias_score_by_model.png", dpi=300)
    plt.close()


def plot_accuracy_gap_by_family_and_theme():
    """
    Figure: Liberal-Conservative accuracy gap by policy theme.

    Uses the current theme-level accuracy table, excludes other(s), and renders
    a single diverging horizontal bar chart where positive gaps extend right in
    blue and negative gaps extend left in red.
    """
    df = pd.read_csv(_theme_accuracy_table())
    df = df[df["ground_truth_side"].isin(["liberal", "conservative"])].copy()
    if df.empty:
        _save_empty_figure(
            FIGURES_DIR / "task1_accuracy_gap_by_family_and_theme.png",
            "Accuracy Gap by Economic Subfield",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    pivot = (
        df.pivot_table(
            index="jel_policy_theme",
            columns="ground_truth_side",
            values="accuracy",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if pivot.empty:
        _save_empty_figure(
            FIGURES_DIR / "task1_accuracy_gap_by_family_and_theme.png",
            "Accuracy Gap by Economic Subfield",
            "No ideology-labeled Task 1 rows were available for plotting.",
        )
        return

    pivot = pivot[~pivot["jel_policy_theme"].isin(["other", "others"])].copy()
    if pivot.empty:
        _save_empty_figure(
            FIGURES_DIR / "task1_accuracy_gap_by_family_and_theme.png",
            "Accuracy Gap by Economic Subfield",
            "No non-other policy themes were available for plotting.",
        )
        return

    pivot["accuracy_gap_pp"] = (pivot["liberal"] - pivot["conservative"]) * 100.0
    pivot = pivot.sort_values("accuracy_gap_pp", ascending=False).reset_index(drop=True)

    theme_label_map = {
        "healthcare": "Healthcare",
        "health_policy": "Health Policy",
        "welfare_redistribution": "Welfare &\nRedistribution",
        "education": "Education",
        "education_policy": "Education Policy",
        "labor": "Labor",
        "labor_wages_unions": "Labor, Wages,\n& Unions",
        "financial_regulation": "Financial\nRegulation",
        "trade": "Trade",
        "taxation": "Taxation",
        "macro_fiscal_state": "Macro, Fiscal,\n& State",
        "market_regulation_antitrust": "Market Regulation\n& Antitrust",
        "industrial_policy_development": "Industrial Policy\n& Development",
        "environment_climate_energy": "Environment,\nClimate & Energy",
        "ambiguous": "Ambiguous",
    }
    pivot["theme_label"] = pivot["jel_policy_theme"].map(theme_label_map).fillna(
        pivot["jel_policy_theme"].str.replace("_", " ").str.title()
    )

    positive_color = "#4F7DBE"
    negative_color = "#D96459"
    bar_colors = np.where(pivot["accuracy_gap_pp"] >= 0, positive_color, negative_color)

    serif = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(12, 6.8))
    y_positions = np.arange(len(pivot))
    bars = ax.barh(
        y_positions,
        pivot["accuracy_gap_pp"].values,
        color=bar_colors,
        edgecolor="white",
        height=0.6,
    )

    x_max = float(np.abs(pivot["accuracy_gap_pp"]).max())
    x_pad = max(1.2, x_max * 0.12)
    ax.set_xlim(-(x_max + x_pad + 4.5), x_max + x_pad + 4.5)
    ax.axvline(0, color="black", linewidth=1.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot["theme_label"].tolist(), fontsize=16, fontfamily=serif)
    ax.invert_yaxis()

    ax.set_title("Accuracy Gap by Economic Subfield", fontsize=21, pad=18, fontfamily=serif)
    ax.set_xlabel("Liberal-Conservative Accuracy Gap (pp)", fontsize=17, labelpad=10, fontfamily=serif)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="x", labelsize=14, width=1.0, length=6)
    ax.tick_params(axis="y", width=1.0, length=7, pad=8)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(serif)

    for bar, gap in zip(bars, pivot["accuracy_gap_pp"].tolist()):
        y = bar.get_y() + bar.get_height() / 2
        if gap >= 0:
            x = gap + 0.45
            ha = "left"
            label = f"+{gap:.1f}"
        else:
            x = gap - 0.45
            ha = "right"
            label = f"{gap:.1f}"
        ax.text(x, y, label, va="center", ha=ha, fontsize=15, fontfamily=serif)

    ax.text(
        0.02,
        0.02,
        "Conservative-truth\nadvantage",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        color=negative_color,
        fontfamily=serif,
        fontstyle="italic",
    )
    ax.text(
        0.98,
        0.02,
        "Liberal-truth\nadvantage",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color=positive_color,
        fontfamily=serif,
        fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_accuracy_gap_by_family_and_theme.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {FIGURES_DIR / 'task1_accuracy_gap_by_family_and_theme.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Task 1 ideology-bias figures.")
    parser.add_argument("--theme-source", choices=["vote", "legacy"], default="vote")
    args = parser.parse_args()
    THEME_SOURCE = args.theme_source
    THEME_ORDER = IDEOLOGY_VOTE_THEME_ORDER if THEME_SOURCE == "vote" else IDEOLOGY_THEME_ORDER
    plot_accuracy_by_jel_and_ideology()
    plot_accuracy_by_jel_ideology_model_family()
    plot_accuracy_by_publication_year_5y()
    plot_accuracy_by_publication_year_5y_and_ideology()
    plot_accuracy_by_publication_year_5y_ideology_model_family()
    plot_bias_score_by_model()
    plot_accuracy_gap_by_family_and_theme()
