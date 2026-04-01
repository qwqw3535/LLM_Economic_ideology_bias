"""Build polished Table 1 figures from the hero model-performance table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE_PATH = (
    ARTIFACT_ROOT / "outputs" / "task1_ideology_subset_1056" / "tables" / "table1_hero_model_performance_and_bias.csv"
)
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "outputs" / "task1_ideology_subset_1056" / "figures"

FAMILY_ORDER = ["openai", "claude", "gemini", "grok", "llama", "qwen"]
MODEL_ORDER_WITHIN_FAMILY = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.2"],
    "claude": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
    "gemini": ["gemini-2.5-flash", "gemini-3-flash-preview"],
    "grok": ["grok-3-mini", "grok-3", "grok-4-1-fast-reasoning"],
    "llama": [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "qwen": ["qwen/qwen3-8b", "qwen/qwen3-14b", "qwen/qwen3-32b"],
}
FAMILY_STYLES = {
    "openai": {"label": "GPT", "color": "#1b9e77", "marker": "o"},
    "llama": {"label": "LLAMA", "color": "#6a5acd", "marker": "D"},
    "grok": {"label": "GROK", "color": "#4a90d9", "marker": "s"},
    "gemini": {"label": "GEMINI", "color": "#4f6fe5", "marker": "^"},
    "qwen": {"label": "QWEN", "color": "#ef6a5b", "marker": "P"},
    "claude": {"label": "CLAUDE", "color": "#c08a00", "marker": "X"},
}
PRETTY_MODEL_NAMES = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o": "GPT-4o",
    "gpt-5-nano": "GPT-5-nano",
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5.2": "GPT-5.2",
    "claude-haiku-4-5": "Claude-haiku-4.5",
    "claude-sonnet-4-6": "Claude-sonnet-4.6",
    "claude-opus-4-6": "Claude-opus-4.6",
    "gemini-2.5-flash": "Gemini-2.5-flash",
    "gemini-3-flash-preview": "Gemini-3-flash",
    "grok-3-mini": "Grok-3-mini",
    "grok-3": "Grok-3",
    "grok-4-1-fast-reasoning": "Grok-4.1-fast",
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8b",
    "meta-llama/llama-3.2-1b-instruct": "Llama-3.2-1b",
    "meta-llama/llama-3.2-3b-instruct": "Llama-3.2-3b",
    "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70b",
    "qwen/qwen3-8b": "Qwen3-8b",
    "qwen/qwen3-14b": "Qwen3-14b",
    "qwen/qwen3-32b": "Qwen3-32b",
}
LABEL_OFFSETS = {
    "openai": (6, 6),
    "claude": (6, 4),
    "gemini": (8, 8),
    "grok": (8, 2),
    "llama": (6, -6),
    "qwen": (6, 6),
}


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(str(family)), str(family))
    except ValueError:
        return (len(FAMILY_ORDER), str(family))


def _model_sort_key(model: str, family: str) -> tuple[int, str]:
    ordered = MODEL_ORDER_WITHIN_FAMILY.get(str(family), [])
    try:
        return (ordered.index(str(model)), str(model))
    except ValueError:
        return (len(ordered), str(model))


def _pretty_model_name(model: str) -> str:
    return PRETTY_MODEL_NAMES.get(str(model), str(model).split("/")[-1])


def _load_table(table_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(table_path)
    frame["_family_key"] = frame["family"].map(_family_sort_key)
    frame["_model_key"] = frame.apply(lambda row: _model_sort_key(row["model"], row["family"]), axis=1)
    frame = frame.sort_values(["_family_key", "_model_key"]).reset_index(drop=True)
    frame["pretty_model"] = frame["model"].map(_pretty_model_name)
    return frame.drop(columns=["_family_key", "_model_key"])


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.edgecolor": "#b8bcc2",
            "grid.color": "#c6c6c6",
            "grid.alpha": 0.7,
            "grid.linewidth": 1.1,
        }
    )


def _family_legend_handles(include_reference: bool = False, reference_label: str = "Reference") -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for family in ["openai", "llama", "grok", "gemini", "qwen", "claude"]:
        style = FAMILY_STYLES[family]
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                linestyle="None",
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=style["label"],
            )
        )
    if include_reference:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="#9e9e9e",
                linestyle="--",
                linewidth=1.5,
                label=reference_label,
            )
        )
    return handles


def _annotate_points(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    for _, row in frame.iterrows():
        dx, dy = LABEL_OFFSETS.get(str(row["family"]), (6, 6))
        ax.annotate(
            row["pretty_model"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            va="center",
            fontsize=11,
            color=FAMILY_STYLES[str(row["family"])]["color"],
        )


def _scatter_by_family(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    for family in ["openai", "llama", "grok", "gemini", "qwen", "claude"]:
        subset = frame[frame["family"] == family]
        if subset.empty:
            continue
        style = FAMILY_STYLES[family]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=120,
            c=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.9,
            alpha=0.96,
            zorder=3,
        )


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_figure(frame: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 10))
    plot_frame = frame.copy()
    plot_frame["acc_con_pct"] = plot_frame["Acc^con"] * 100.0
    plot_frame["acc_lib_pct"] = plot_frame["Acc^lib"] * 100.0

    _scatter_by_family(ax, plot_frame, "acc_con_pct", "acc_lib_pct")
    _annotate_points(ax, plot_frame, "acc_con_pct", "acc_lib_pct")

    low = float(np.floor(min(plot_frame["acc_con_pct"].min(), plot_frame["acc_lib_pct"].min()) / 5.0) * 5.0) - 2.0
    high = float(np.ceil(max(plot_frame["acc_con_pct"].max(), plot_frame["acc_lib_pct"].max()) / 5.0) * 5.0) + 2.0
    diagonal = np.linspace(low, high, 100)
    ax.plot(diagonal, diagonal, linestyle="--", color="#9e9e9e", linewidth=1.5, zorder=1)

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Conservative-case Accuracy (%)")
    ax.set_ylabel("Liberal-case Accuracy (%)")
    ax.set_title(
        "Liberal vs Conservative Accuracy per Model\n"
        "Above diagonal = better on liberal cases | Below diagonal = better on conservative cases"
    )
    ax.legend(
        handles=_family_legend_handles(include_reference=True, reference_label="Equal accuracy"),
        loc="upper left",
        frameon=True,
        facecolor="white",
    )

    output_path = output_dir / "table1_accuracy_by_ground_truth_direction.png"
    _save_figure(fig, output_path)
    return output_path


def plot_bias_figure(frame: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 10))
    plot_frame = frame.copy()
    plot_frame["liberal_error_pct"] = (1.0 - plot_frame["Acc^con"]) * 100.0
    plot_frame["conservative_error_pct"] = (1.0 - plot_frame["Acc^lib"]) * 100.0

    _scatter_by_family(ax, plot_frame, "liberal_error_pct", "conservative_error_pct")
    _annotate_points(ax, plot_frame, "liberal_error_pct", "conservative_error_pct")

    max_error = float(
        np.ceil(
            max(
                plot_frame["liberal_error_pct"].max(),
                plot_frame["conservative_error_pct"].max(),
            )
            / 5.0
        )
        * 5.0
    ) + 5.0
    diagonal = np.linspace(0.0, max_error, 100)
    ax.plot(diagonal, diagonal, linestyle="--", color="#9e9e9e", linewidth=1.5, zorder=1)

    ax.set_xlim(max_error, 0.0)
    ax.set_ylim(max_error, 0.0)
    ax.set_xlabel("Liberal Error Rate (%) — wrong on conservative cases")
    ax.set_ylabel("Conservative Error Rate (%) — wrong on liberal cases")
    ax.set_title(
        "Liberal vs Conservative Error Rate per Model\n"
        "Below diagonal = liberal bias | Above diagonal = conservative bias"
    )
    ax.legend(
        handles=_family_legend_handles(include_reference=True, reference_label="No bias"),
        loc="upper left",
        frameon=True,
        facecolor="white",
    )

    output_path = output_dir / "table1_bias_error_rate_scatter.png"
    _save_figure(fig, output_path)
    return output_path


def plot_sensitive_vs_non_sensitive_figure(frame: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 10))
    plot_frame = frame.copy()
    plot_frame["acc_non_pct"] = plot_frame["Acc (Non-sensitive)"] * 100.0
    plot_frame["acc_sen_pct"] = plot_frame["Acc (Sensitive)"] * 100.0

    _scatter_by_family(ax, plot_frame, "acc_non_pct", "acc_sen_pct")
    _annotate_points(ax, plot_frame, "acc_non_pct", "acc_sen_pct")

    low = float(np.floor(min(plot_frame["acc_non_pct"].min(), plot_frame["acc_sen_pct"].min()) / 5.0) * 5.0) - 2.0
    high = float(np.ceil(max(plot_frame["acc_non_pct"].max(), plot_frame["acc_sen_pct"].max()) / 5.0) * 5.0) + 2.0
    diagonal = np.linspace(low, high, 100)
    ax.plot(diagonal, diagonal, linestyle="--", color="#9e9e9e", linewidth=1.5, zorder=1)

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Non-sensitive Accuracy (%)")
    ax.set_ylabel("Sensitive Accuracy (%)")
    ax.set_title(
        "Sensitive vs Non-sensitive Accuracy per Model\n"
        "Below diagonal = larger drop on ideology-sensitive triplets"
    )
    ax.legend(
        handles=_family_legend_handles(include_reference=True, reference_label="Equal accuracy"),
        loc="upper left",
        frameon=True,
        facecolor="white",
    )

    output_path = output_dir / "table1_sensitive_vs_non_sensitive_accuracy.png"
    _save_figure(fig, output_path)
    return output_path


def build_figures(table_path: Path, output_dir: Path) -> list[Path]:
    _apply_plot_style()
    frame = _load_table(table_path)
    outputs = [
        plot_accuracy_figure(frame, output_dir),
        plot_bias_figure(frame, output_dir),
        plot_sensitive_vs_non_sensitive_figure(frame, output_dir),
    ]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Table 1 main-results figures.")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_PATH, help="Path to table1 hero CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figures(args.table, args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
