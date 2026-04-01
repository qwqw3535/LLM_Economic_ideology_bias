"""Minimal visualization helpers for released tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def is_regression_table(frame: pd.DataFrame) -> bool:
    return "term" in frame.columns and "coef" in frame.columns


def write_frame_figure(path: str | Path, title: str, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = frame.head(20).copy()
    fig_height = max(3.0, 0.35 * (len(preview) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=preview.astype(str).values,
        colLabels=list(preview.columns),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_regression_report(path: str | Path, title: str, frame: pd.DataFrame, formula: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if formula:
        lines.extend([f"`{formula}`", ""])
    lines.append(frame.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regression_html_report(path: str | Path, title: str, frame: pd.DataFrame, formula: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formula_html = f"<p><code>{formula}</code></p>" if formula else ""
    html = f"<html><body><h1>{title}</h1>{formula_html}{frame.to_html(index=False)}</body></html>"
    path.write_text(html, encoding="utf-8")
