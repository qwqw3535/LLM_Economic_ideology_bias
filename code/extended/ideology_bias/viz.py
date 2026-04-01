"""Raster chart and Markdown output helpers."""

from __future__ import annotations

from pathlib import Path
import re

from PIL import Image, ImageColor, ImageDraw, ImageFont
import pandas as pd

from .jel import IDEOLOGY_THEME_ORDER
from .utils import family_sort_key, infer_family_from_model, model_sort_key


def write_horizontal_bar_chart(
    path: str | Path,
    title: str,
    labels: list[str],
    values: list[float],
    counts: list[int] | None = None,
    colors: list[str] | None = None,
    metric_label: str | None = None,
    error_counts: list[int] | None = None,
    error_rates: list[float] | None = None,
    value_range: tuple[float, float] | None = None,
) -> None:
    """Write a minimal horizontal bar chart to PNG or JPEG.

    Args:
        value_range: Optional (min, max) range for values. If None, uses (0, 1).
                    For bias_score, should be (-0.5, 0.5).
        error_rates: List of error rates (0-1 scale) to display as percentages.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = counts or [None] * len(labels)
    colors = colors or ["#2563eb"] * len(labels)
    error_counts = error_counts or [None] * len(labels)
    error_rates = error_rates or [None] * len(labels)
    value_range = value_range or (0.0, 1.0)

    width = 1400
    row_h = 34
    left = 520
    right = 120
    top = 90
    bottom = 40
    plot_w = width - left - right
    height = top + bottom + row_h * max(1, len(labels))

    image = Image.new("RGB", (width, height), "white")
    if len(labels) > 25: 
        image = image.resize((width, min(2000, height))) # Safety limit visually? not needed here just keeping it
        
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    bg_color = ImageColor.getrgb("#f3f4f6")
    text_color = ImageColor.getrgb("#111827")
    axis_color = ImageColor.getrgb("#9ca3af")

    draw.text((left, 24), title, fill=text_color, font=title_font)
    if metric_label:
        draw.text((left, 46), f"Metric: {metric_label}", fill=ImageColor.getrgb("#6b7280"), font=font)
    draw.line((left, top - 12, left + plot_w, top - 12), fill=axis_color, width=1)

    # Calculate zero position for centered ranges (e.g., -0.5 to 0.5)
    min_val, max_val = value_range
    range_span = max_val - min_val
    zero_pos = left + int(plot_w * (-min_val / range_span)) if min_val < 0 else left

    # Draw vertical line at zero for centered ranges
    if min_val < 0 < max_val:
        draw.line((zero_pos, top - 12, zero_pos, height - bottom), fill=ImageColor.getrgb("#374151"), width=2)

    for idx, (label, value, count, color, err_count, err_rate) in enumerate(zip(labels, values, counts, colors, error_counts, error_rates)):
        y = top + idx * row_h
        draw.rectangle((left, y + 6, left + plot_w, y + 24), fill=bg_color)

        # Calculate bar position and width based on value_range
        normalized_value = (float(value) - min_val) / range_span
        bar_w = max(0, min(plot_w, int(plot_w * normalized_value)))

        # For centered ranges, draw from zero position
        if min_val < 0 < max_val:
            if value >= 0:
                bar_left = zero_pos
                bar_right = left + bar_w
            else:
                bar_left = left + bar_w
                bar_right = zero_pos
        else:
            bar_left = left
            bar_right = left + bar_w

        c = ImageColor.getrgb(color)
        if len(c) == 3:
            c = (*c, 255)
        draw.rectangle((bar_left, y + 6, bar_right, y + 24), fill=c)

        label_text = str(label)
        if count is not None and err_rate is not None:
            label_text = f"{label_text} (triplets={count}, err={err_rate*100:.1f}%)"
        elif count is not None:
            label_text = f"{label_text} (triplets={count})"
        draw.text((12, y + 8), label_text, fill=text_color, font=font)
        draw.text((left + plot_w + 12, y + 8), f"{float(value) * 100:.2f}%", fill=text_color, font=font)

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, format="JPEG", quality=95)
    else:
        image.save(path, format="PNG")


def write_text_card(path: str | Path, title: str, lines: list[str]) -> None:
    """Write a simple text card as PNG/JPEG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1600
    line_h = 24
    top = 80
    height = max(240, top + 50 + line_h * max(1, len(lines)))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    draw.text((40, 24), title, fill="#111827", font=title_font)
    y = top
    for line in lines:
        draw.text((40, y), str(line), fill="#111827", font=font)
        y += line_h
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=95)
    else:
        image.save(path, format="PNG")


def write_table_card(path: str | Path, title: str, frame: pd.DataFrame, max_rows: int = 25, max_cols: int = 8) -> None:
    """Render a dataframe preview as an image."""
    path = Path(path)
    subset = frame.copy().head(max_rows)
    columns = list(subset.columns[:max_cols])
    lines = [" | ".join(columns)]
    for _, row in subset.iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append(" | ".join(rendered))
    if len(frame) > max_rows:
        lines.append(f"... ({len(frame) - max_rows} more rows)")
    write_text_card(path, title, lines)


_THEME_ORDER = IDEOLOGY_THEME_ORDER

_SIDE_ORDER = ["liberal", "conservative"]
_SIDE_COLORS = {"liberal": "#2563eb", "conservative": "#dc2626"}
_CASE_TYPE_ORDER = ["lib-lib", "lib-cons", "cons-lib", "cons-cons"]
_FAMILY_PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#6366f1", "#84cc16",
]
def _theme_sort_key(theme: str) -> int:
    """Return sort key for a jel_policy_theme string."""
    try:
        return _THEME_ORDER.index(theme)
    except ValueError:
        return len(_THEME_ORDER)


def _side_sort_key(side: str) -> int:
    """Return sort key for a ground_truth_side string."""
    try:
        return _SIDE_ORDER.index(side)
    except ValueError:
        return len(_SIDE_ORDER)


def _case_type_sort_key(case_type: str) -> int:
    """Return canonical sort key for pair_case_type."""
    try:
        return _CASE_TYPE_ORDER.index(case_type)
    except ValueError:
        return len(_CASE_TYPE_ORDER)


def _family_color_map(families: list[str]) -> dict[str, str]:
    """Create a stable family-to-color mapping."""
    ordered_unique = []
    known_family_order = ["openai", "claude", "gemini", "grok", "llama", "qwen", "deepseek", "hf_endpoint"]
    for family in known_family_order + sorted(set(families) - set(known_family_order)):
        if family in families and family not in ordered_unique:
            ordered_unique.append(family)
    return {family: _FAMILY_PALETTE[i % len(_FAMILY_PALETTE)] for i, family in enumerate(ordered_unique)}


def _append_alpha(color: str, alpha_hex: str) -> str:
    """Append an alpha channel to a hex color."""
    return color if len(color) == 9 else f"{color}{alpha_hex}"


def _publication_year_bucket_sort_key(label: str) -> tuple[int, str]:
    """Return chronological sort key for 5-year publication-year buckets."""
    match = re.match(r"^(\d{4})-\d{4}$", str(label))
    if match:
        return (int(match.group(1)), str(label))
    return (10**9, str(label))


def _difficulty_level_sort_key(label: str) -> tuple[int, str]:
    """Return numeric sort key for labels like 'difficulty level 3'."""
    match = re.search(r"(\d+)", str(label))
    if match:
        return (int(match.group(1)), str(label))
    return (10**9, str(label))


def write_frame_figure(path: str | Path, title: str, frame: pd.DataFrame) -> None:
    """Write a figure for any saved table: chart when possible, table card otherwise."""
    path = Path(path)
    if frame.empty:
        write_text_card(path, title, ["No rows to display."])
        return

    metric_candidates = [
        "accuracy",
        "bias_score",
        "prediction_shift_rate",
        "pro_state_rate",
        "follows_displayed_rate",
        "follows_original_rate",
        "share_predictions",
    ]
    metric_col = next((column for column in metric_candidates if column in frame.columns), None)
    if metric_col is None:
        metric_col = next((column for column in frame.columns if column.endswith("_rate") or column.endswith("_score")), None)

    if metric_col is None:
        write_table_card(path, title, frame)
        return

    label_cols = [
        column
        for column in frame.columns
        if column not in {metric_col, "n_triplets", "n_predictions", "correct_predictions", "total_predictions"}
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not label_cols:
        write_table_card(path, title, frame)
        return

    # Determine metric description
    metric_descriptions = {
        "accuracy": "Accuracy (proportion of correct predictions)",
        "bias_score": "Bias Score = (Liberal Errors - Conservative Errors) / Total Ideology Errors",
        "prediction_shift_rate": "Prediction Shift Rate",
        "pro_state_rate": "Pro-State Rate",
        "follows_displayed_rate": "Follows Displayed Rate",
        "follows_original_rate": "Follows Original Rate",
        "follows_displayed_example_rate": "Follows Displayed Example Rate",
        "recovers_original_rate": "Recovers Original Rate",
        "share_predictions": "Share of Predictions",
    }
    metric_label = metric_descriptions.get(metric_col, metric_col.replace("_", " ").title())

    has_theme = "jel_policy_theme" in label_cols
    has_side = "ground_truth_side" in label_cols
    has_model = "model" in label_cols
    has_family = "family" in label_cols
    has_publication_year_bucket = "publication_year_5y_bucket" in label_cols
    has_difficulty_level = "difficulty_level" in label_cols
    has_case_type = "pair_case_type" in label_cols

    plot = frame.copy()

    bar_colors: list[str] | None = None

    # ── Case A: model-level breakdown with jel_policy_theme ──
    # Split into per-theme figures stored in a subdirectory.
    if has_model and has_theme and has_side and not has_family:
        per_theme_dir = path.parent / path.stem
        per_theme_dir.mkdir(parents=True, exist_ok=True)

        themes = sorted(plot["jel_policy_theme"].unique(), key=_theme_sort_key)
        for theme in themes:
            tdf = plot[plot["jel_policy_theme"] == theme].copy()
            # Sort by model, then liberal before conservative
            tdf["_side_key"] = tdf["ground_truth_side"].map(_side_sort_key)
            tdf["_model_key"] = tdf["model"].map(lambda value: model_sort_key(value))
            tdf = tdf.sort_values(["_model_key", "_side_key"])
            tdf.drop(columns=["_model_key"], inplace=True)
            tdf.drop(columns=["_side_key"], inplace=True)

            sub_label_cols = [c for c in label_cols if c != "jel_policy_theme"]
            labels = [" | ".join(str(val) for val in row) for _, row in tdf[sub_label_cols].iterrows()]
            values = tdf[metric_col].astype(float).tolist()
            counts = None
            if "n_triplets" in tdf.columns:
                counts = tdf["n_triplets"].fillna(0).astype(int).tolist()
            elif "n_predictions" in tdf.columns:
                counts = tdf["n_predictions"].fillna(0).astype(int).tolist()

            colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in tdf["ground_truth_side"]]

            # For bias_score metric, also include error rates and set proper range
            err_counts = None
            err_rates = None
            val_range = None
            if metric_col == "bias_score":
                if "error_rate" in tdf.columns:
                    err_rates = tdf["error_rate"].fillna(0).astype(float).tolist()
                val_range = (-0.5, 0.5)

            sub_path = per_theme_dir / f"{path.stem}_{theme}.png"
            write_horizontal_bar_chart(
                sub_path,
                f"{title} — {theme}",
                labels,
                values,
                counts=counts,
                colors=colors,
                metric_label=metric_label,
                error_counts=err_counts,
                error_rates=err_rates,
                value_range=val_range,
            )
        # Also write the original combined figure (first 50 rows, same ordering)
        plot["_theme_key"] = plot["jel_policy_theme"].map(_theme_sort_key)
        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        plot["_model_key"] = plot["model"].map(lambda value: model_sort_key(value))
        plot = plot.sort_values(["_theme_key", "jel_policy_theme", "_model_key", "_side_key"]).head(50)
        plot.drop(columns=["_theme_key", "_side_key", "_model_key"], inplace=True)

        bar_colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in plot["ground_truth_side"]]

    # ── Case B: generic model + ... + ground-truth side ──
    # Keep lib/con rows adjacent within the same environment.
    elif has_model and has_side and not has_theme and not has_publication_year_bucket:
        extra_label_cols = [c for c in label_cols if c not in {"family", "model", "ground_truth_side"}]
        plot["_model_key"] = plot.apply(lambda row: model_sort_key(row["model"], row.get("family")), axis=1)
        sort_cols = ["_model_key"]

        for column in extra_label_cols:
            key_col = f"_{column}_key"
            if column == "difficulty_level":
                plot[key_col] = plot[column].map(_difficulty_level_sort_key)
                sort_cols.append(key_col)
            elif column == "pair_case_type":
                plot[key_col] = plot[column].map(_case_type_sort_key)
                sort_cols.append(key_col)
            else:
                sort_cols.append(column)

        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        sort_cols.append("_side_key")
        plot = plot.sort_values(sort_cols)

        def _side_compact_label(side: str) -> str:
            if side == "liberal":
                return "lib"
            if side == "conservative":
                return "con"
            return str(side)

        labels = []
        for _, row in plot.iterrows():
            parts = [str(row["model"])]
            parts.extend(str(row[column]) for column in extra_label_cols)
            parts.append(_side_compact_label(str(row["ground_truth_side"])))
            labels.append(" - ".join(parts))

        values = plot[metric_col].astype(float).tolist()
        counts = None
        if "n_triplets" in plot.columns:
            counts = plot["n_triplets"].fillna(0).astype(int).tolist()
        elif "n_predictions" in plot.columns:
            counts = plot["n_predictions"].fillna(0).astype(int).tolist()

        bar_colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in plot["ground_truth_side"]]

        error_counts = None
        error_rates = None
        value_range = None
        if metric_col == "bias_score":
            if "error_rate" in plot.columns:
                error_rates = plot["error_rate"].fillna(0).astype(float).tolist()
            value_range = (-0.5, 0.5)

        write_horizontal_bar_chart(
            path,
            title,
            labels,
            values,
            counts=counts,
            colors=bar_colors,
            metric_label=metric_label,
            error_counts=error_counts,
            error_rates=error_rates,
            value_range=value_range,
        )
        return

    # ── Case C: model + pair_case_type ──
    # Keep each model's case-type rows adjacent in canonical case order.
    elif has_model and has_case_type and not has_theme and not has_side and not has_family:
        plot["_case_type_key"] = plot["pair_case_type"].map(_case_type_sort_key)
        plot["_family"] = plot["model"].map(infer_family_from_model)
        plot["_model_key"] = plot["model"].map(lambda value: model_sort_key(value))
        plot = plot.sort_values(["_model_key", "_case_type_key"])

        labels = [
            f"{row['model']} - {row['pair_case_type']}"
            for _, row in plot.iterrows()
        ]
        values = plot[metric_col].astype(float).tolist()
        counts = None
        if "n_triplets" in plot.columns:
            counts = plot["n_triplets"].fillna(0).astype(int).tolist()
        elif "n_predictions" in plot.columns:
            counts = plot["n_predictions"].fillna(0).astype(int).tolist()

        family_color_map = _family_color_map(plot["_family"].tolist())
        bar_colors = [
            _append_alpha(family_color_map.get(str(row["_family"]), "#2563eb"), "E6")
            for _, row in plot.iterrows()
        ]

        error_counts = None
        error_rates = None
        value_range = None
        if metric_col == "bias_score":
            if "error_rate" in plot.columns:
                error_rates = plot["error_rate"].fillna(0).astype(float).tolist()
            value_range = (-0.5, 0.5)

        write_horizontal_bar_chart(
            path,
            title,
            labels,
            values,
            counts=counts,
            colors=bar_colors,
            metric_label=metric_label,
            error_counts=error_counts,
            error_rates=error_rates,
            value_range=value_range,
        )
        return

    # ── Case D: family + theme + ground-truth side ──
    # Keep lib/con adjacent within each theme-family environment.
    elif has_family and has_theme and has_side and not has_model:
        plot["_theme_key"] = plot["jel_policy_theme"].map(_theme_sort_key)
        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        plot = plot.sort_values(["_theme_key", "jel_policy_theme", "family", "_side_key"])

        labels = []
        for _, row in plot.iterrows():
            side = "lib" if str(row["ground_truth_side"]) == "liberal" else "con"
            labels.append(f"{row['jel_policy_theme']} - {row['family']} - {side}")

        values = plot[metric_col].astype(float).tolist()
        counts = None
        if "n_triplets" in plot.columns:
            counts = plot["n_triplets"].fillna(0).astype(int).tolist()
        elif "n_predictions" in plot.columns:
            counts = plot["n_predictions"].fillna(0).astype(int).tolist()

        unique_families = plot["family"].unique().tolist()
        family_color_map = _family_color_map(unique_families)
        bar_colors = []
        for _, row in plot.iterrows():
            base_col = family_color_map.get(row["family"], "#2563eb")
            base_col = _append_alpha(base_col, "FF" if str(row["ground_truth_side"]) == "liberal" else "66")
            bar_colors.append(base_col)

        error_counts = None
        error_rates = None
        value_range = None
        if metric_col == "bias_score":
            if "error_rate" in plot.columns:
                error_rates = plot["error_rate"].fillna(0).astype(float).tolist()
            value_range = (-0.5, 0.5)

        write_horizontal_bar_chart(
            path,
            title,
            labels,
            values,
            counts=counts,
            colors=bar_colors,
            metric_label=metric_label,
            error_counts=error_counts,
            error_rates=error_rates,
            value_range=value_range,
        )
        return

    # ── Case E: family + model present ──
    # New logic: Sort by jel_policy_theme (subdomain) first, then family, then ideology (liberal/conservative)
    elif has_family and has_model:
        if has_theme:
            # Sort by theme first (development, finance, etc.), then family, then ideology
            plot["_theme_key"] = plot["jel_policy_theme"].map(_theme_sort_key)
            if has_side:
                # Then by ideology: liberal before conservative
                plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
                plot["_family_key"] = plot["family"].map(family_sort_key)
                plot["_model_key"] = plot.apply(lambda row: model_sort_key(row["model"], row.get("family")), axis=1)
                sort_cols = ["_theme_key", "jel_policy_theme", "_family_key", "_model_key", "_side_key"]
            else:
                plot["_family_key"] = plot["family"].map(family_sort_key)
                plot["_model_key"] = plot.apply(lambda row: model_sort_key(row["model"], row.get("family")), axis=1)
                sort_cols = ["_theme_key", "jel_policy_theme", "_family_key", "_model_key"]
        else:
            # If no theme, just sort by family and model
            plot["_family_key"] = plot["family"].map(family_sort_key)
            plot["_model_key"] = plot.apply(lambda row: model_sort_key(row["model"], row.get("family")), axis=1)
            sort_cols = ["_family_key", "_model_key"]
            for c in label_cols:
                if c not in sort_cols:
                    sort_cols.append(c)

        plot = plot.sort_values(sort_cols).head(50)

        # Clean up temporary sort columns if they were added
        if "_theme_key" in plot.columns:
            plot.drop(columns=["_theme_key"], inplace=True)
        if "_side_key" in plot.columns:
            plot.drop(columns=["_side_key"], inplace=True)
        if "_family_key" in plot.columns:
            plot.drop(columns=["_family_key"], inplace=True)
        if "_model_key" in plot.columns:
            plot.drop(columns=["_model_key"], inplace=True)

        unique_families = plot["family"].unique().tolist()
        family_color_map = _family_color_map(unique_families)

        bar_colors = []
        for _, row in plot.iterrows():
            base_col = family_color_map.get(row["family"], "#2563eb")
            label_text = " ".join([str(row[c]).lower() for c in label_cols])

            if "conservative" in label_text:
                base_col = _append_alpha(base_col, "66")
            elif "liberal" in label_text:
                base_col = _append_alpha(base_col, "FF")
            else:
                base_col = _append_alpha(base_col, "E6")
            bar_colors.append(base_col)

    # ── Case F: publication_year_5y_bucket based tables ──
    elif has_publication_year_bucket and has_family and has_side and not has_model:
        plot["_year_key"] = plot["publication_year_5y_bucket"].map(_publication_year_bucket_sort_key)
        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        plot = plot.sort_values(["_year_key", "family", "_side_key"])
        plot.drop(columns=["_year_key", "_side_key"], inplace=True)

        bar_colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in plot["ground_truth_side"]]

    elif has_publication_year_bucket and has_side and not has_model and not has_family:
        plot["_year_key"] = plot["publication_year_5y_bucket"].map(_publication_year_bucket_sort_key)
        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        plot = plot.sort_values(["_year_key", "_side_key"])
        plot.drop(columns=["_year_key", "_side_key"], inplace=True)

        bar_colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in plot["ground_truth_side"]]

    elif has_publication_year_bucket and has_family and not has_model:
        plot["_year_key"] = plot["publication_year_5y_bucket"].map(_publication_year_bucket_sort_key)
        plot = plot.sort_values(["_year_key", "family"])
        plot.drop(columns=["_year_key"], inplace=True)

    elif has_publication_year_bucket:
        plot["_year_key"] = plot["publication_year_5y_bucket"].map(_publication_year_bucket_sort_key)
        plot = plot.sort_values(["_year_key"])
        plot.drop(columns=["_year_key"], inplace=True)

    # ── Case G: pair_case_type only ──
    elif has_case_type and not has_model:
        plot["_case_type_key"] = plot["pair_case_type"].map(_case_type_sort_key)
        plot = plot.sort_values(["_case_type_key"])
        plot.drop(columns=["_case_type_key"], inplace=True)

    # ── Case H: jel_policy_theme + ground_truth_side (no model) ──
    # Keep difficulty levels explicit when present.
    elif has_theme and has_side and has_difficulty_level and not has_model and not has_family:
        plot["_theme_key"] = plot["jel_policy_theme"].map(_theme_sort_key)
        plot["_difficulty_key"] = plot["difficulty_level"].map(_difficulty_level_sort_key)
        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        plot = plot.sort_values(["_theme_key", "_difficulty_key", "_side_key"])

        labels = []
        for _, row in plot.iterrows():
            side = "lib" if str(row["ground_truth_side"]) == "liberal" else "con"
            labels.append(f"{row['jel_policy_theme']} - {row['difficulty_level']} - {side}")

        values = plot[metric_col].astype(float).tolist()
        counts = None
        if "n_triplets_after" in plot.columns:
            counts = plot["n_triplets_after"].fillna(0).astype(int).tolist()
        elif "n_triplets" in plot.columns:
            counts = plot["n_triplets"].fillna(0).astype(int).tolist()
        elif "n_predictions" in plot.columns:
            counts = plot["n_predictions"].fillna(0).astype(int).tolist()

        bar_colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in plot["ground_truth_side"]]

        write_horizontal_bar_chart(
            path,
            title,
            labels,
            values,
            counts=counts,
            colors=bar_colors,
            metric_label=metric_label,
        )
        return

    # ── Case H: jel_policy_theme + ground_truth_side (no model) ──
    # Keep each theme's lib/con rows adjacent with explicit labels.
    elif has_theme and has_side:
        plot["_theme_key"] = plot["jel_policy_theme"].map(_theme_sort_key)
        plot["_side_key"] = plot["ground_truth_side"].map(_side_sort_key)
        plot = plot.sort_values(["_theme_key", "jel_policy_theme", "_side_key"])
        labels = []
        for _, row in plot.iterrows():
            side = "lib" if str(row["ground_truth_side"]) == "liberal" else "con"
            labels.append(f"{row['jel_policy_theme']} - {side}")

        values = plot[metric_col].astype(float).tolist()
        counts = None
        if "n_triplets" in plot.columns:
            counts = plot["n_triplets"].fillna(0).astype(int).tolist()
        elif "n_predictions" in plot.columns:
            counts = plot["n_predictions"].fillna(0).astype(int).tolist()

        bar_colors = [_SIDE_COLORS.get(str(s), "#2563eb") for s in plot["ground_truth_side"]]

        error_counts = None
        error_rates = None
        value_range = None
        if metric_col == "bias_score":
            if "error_rate" in plot.columns:
                error_rates = plot["error_rate"].fillna(0).astype(float).tolist()
            value_range = (-0.5, 0.5)

        write_horizontal_bar_chart(
            path,
            title,
            labels,
            values,
            counts=counts,
            colors=bar_colors,
            metric_label=metric_label,
            error_counts=error_counts,
            error_rates=error_rates,
            value_range=value_range,
        )
        return

    # ── Case I: default – sort by metric descending ──
    else:
        plot = plot.sort_values(metric_col, ascending=False).head(25)

    labels = [" | ".join([str(val) for val in row]) for _, row in plot[label_cols].iterrows()]
    values = plot[metric_col].astype(float).tolist()
    counts = None
    if "n_triplets" in plot.columns:
        counts = plot["n_triplets"].fillna(0).astype(int).tolist()
    elif "n_predictions" in plot.columns:
        counts = plot["n_predictions"].fillna(0).astype(int).tolist()

    # For bias_score metric, also include error rates and set proper range
    error_counts = None
    error_rates = None
    value_range = None
    if metric_col == "bias_score":
        if "error_rate" in plot.columns:
            error_rates = plot["error_rate"].fillna(0).astype(float).tolist()
        value_range = (-0.5, 0.5)

    write_horizontal_bar_chart(path, title, labels, values, counts=counts, colors=bar_colors, metric_label=metric_label, error_counts=error_counts, error_rates=error_rates, value_range=value_range)


def write_markdown_table(
    path: str | Path,
    rows: list[dict],
    columns: list[str],
    percent_columns: set[str] | None = None,
    title: str | None = None,
) -> None:
    """Write a list of dictionaries as a Markdown table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    percent_columns = percent_columns or set()
    with open(path, "w", encoding="utf-8") as handle:
        if title:
            handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in rows:
            rendered = []
            for column in columns:
                value = row.get(column, "")
                if column in percent_columns and isinstance(value, (int, float)):
                    rendered.append(f"{value*100:.2f}%")
                elif isinstance(value, float):
                    rendered.append(f"{value:.4f}")
                else:
                    rendered.append(str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def is_regression_table(frame: pd.DataFrame) -> bool:
    """Check whether a DataFrame looks like regression output."""
    regression_cols = {"term", "coef"}
    return regression_cols.issubset(set(frame.columns))


def _format_coef(value: float) -> str:
    """Format a coefficient for display."""
    if pd.isna(value):
        return ""
    abs_val = abs(value)
    if abs_val == 0:
        return "0.0000"
    if abs_val >= 1e6:
        return f"{value:.4e}"
    if abs_val < 0.0001:
        return f"{value:.4e}"
    return f"{value:.4f}"


def _significance_star(p_value: float) -> str:
    """Return significance stars for a p-value."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "†"
    return ""


def _clean_term_name(term: str) -> str:
    """Make regression term names more readable."""
    import re
    # C(var)[T.value] -> var: value
    match = re.match(r"C\(([^)]+)\)\[T\.(.+)\]", term)
    if match:
        return f"{match.group(1)}: {match.group(2)}"
    return term


def write_regression_report(
    path: str | Path,
    title: str,
    frame: pd.DataFrame,
    formula: str | None = None,
) -> None:
    """Write a regression result DataFrame as a complete Markdown report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_error = "error" in frame.columns and frame["error"].notna().any()
    has_outcome = "outcome" in frame.columns

    n_obs = None
    if "n_obs" in frame.columns:
        valid_nobs = frame["n_obs"].dropna()
        if not valid_nobs.empty:
            n_obs = int(valid_nobs.iloc[0])

    lines: list[str] = []
    lines.append(f"# {title}\n")

    # Model info section
    lines.append("## Model Information\n")
    if formula:
        lines.append(f"- **Formula**: `{formula}`\n")
    if n_obs is not None:
        lines.append(f"- **N (observations)**: {n_obs:,}\n")
    lines.append("")

    if has_error:
        error_msg = frame["error"].dropna().iloc[0] if frame["error"].notna().any() else "unknown"
        lines.append("> **⚠ Warning**: Model fitting encountered an issue.\n")
        lines.append(f"> `{error_msg}`\n")
        lines.append("")

    # Check whether std_err / p_value are all NaN (convergence issue)
    has_se = "std_err" in frame.columns and frame["std_err"].notna().any()
    has_pval = "p_value" in frame.columns and frame["p_value"].notna().any()
    has_ci = "conf_low" in frame.columns and frame["conf_low"].notna().any()

    if not has_se and not has_pval:
        lines.append("> **⚠ Note**: Standard errors, p-values, and confidence intervals are unavailable.")
        lines.append("> This typically indicates a model convergence issue (e.g., perfect separation or multicollinearity).")
        lines.append("> Coefficient estimates below should be interpreted with extreme caution.\n")

    # Group by outcome if multinomial
    if has_outcome:
        outcomes = frame["outcome"].unique()
        for outcome in outcomes:
            subset = frame[frame["outcome"] == outcome].copy()
            lines.append(f"## Outcome: `{outcome}`\n")
            lines.extend(_render_coef_table(subset, has_se, has_pval, has_ci))
            lines.append("")
    else:
        lines.append("## Coefficient Estimates\n")
        lines.extend(_render_coef_table(frame, has_se, has_pval, has_ci))
        lines.append("")

    # Significance legend
    if has_pval:
        lines.append("---\n")
        lines.append("**Significance**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `†` p<0.1\n")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _render_coef_table(
    frame: pd.DataFrame,
    has_se: bool,
    has_pval: bool,
    has_ci: bool,
) -> list[str]:
    """Render a coefficient table as markdown lines."""
    lines: list[str] = []

    # Build header
    header_parts = ["Term", "Coefficient"]
    if has_se:
        header_parts.append("Std. Error")
    if has_pval:
        header_parts.extend(["z / t", "p-value", "Sig."])
    if has_ci:
        header_parts.extend(["CI Low", "CI High"])

    lines.append("| " + " | ".join(header_parts) + " |")
    align = [":---"] + ["---:"] * (len(header_parts) - 1)
    lines.append("| " + " | ".join(align) + " |")

    for _, row in frame.iterrows():
        term = _clean_term_name(str(row.get("term", "")))
        coef = _format_coef(row.get("coef", float("nan")))

        cells = [term, coef]

        if has_se:
            se = row.get("std_err", float("nan"))
            cells.append(_format_coef(se) if not pd.isna(se) else "—")

        if has_pval:
            z = row.get("z_or_t", float("nan"))
            p = row.get("p_value", float("nan"))
            cells.append(_format_coef(z) if not pd.isna(z) else "—")
            if not pd.isna(p):
                cells.append(f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}")
                cells.append(_significance_star(p))
            else:
                cells.extend(["—", ""])

        if has_ci:
            cl = row.get("conf_low", float("nan"))
            ch = row.get("conf_high", float("nan"))
            cells.append(_format_coef(cl) if not pd.isna(cl) else "—")
            cells.append(_format_coef(ch) if not pd.isna(ch) else "—")

        lines.append("| " + " | ".join(cells) + " |")

    return lines


def write_regression_html_report(
    path: str | Path,
    title: str,
    frame: pd.DataFrame,
    formula: str | None = None,
) -> None:
    """Write a regression result DataFrame as a complete HTML report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_error = "error" in frame.columns and frame["error"].notna().any()
    has_outcome = "outcome" in frame.columns

    n_obs = None
    n_triplets = None
    if "n_obs" in frame.columns:
        valid_nobs = frame["n_obs"].dropna()
        if not valid_nobs.empty:
            n_obs = int(valid_nobs.iloc[0])
    if "n_triplets" in frame.columns:
        valid_ntriplets = frame["n_triplets"].dropna()
        if not valid_ntriplets.empty:
            n_triplets = int(valid_ntriplets.iloc[0])

    # Check whether std_err / p_value are all NaN (convergence issue)
    has_se = "std_err" in frame.columns and frame["std_err"].notna().any()
    has_pval = "p_value" in frame.columns and frame["p_value"].notna().any()
    has_ci = "conf_low" in frame.columns and frame["conf_low"].notna().any()

    html_parts: list[str] = []

    # HTML header and CSS
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {
  font-family: Arial, sans-serif;
  max-width: 1400px;
  margin: 20px;
  line-height: 1.6;
}
h1 {
  color: #2c3e50;
  border-bottom: 3px solid #3498db;
  padding-bottom: 10px;
}
h2 {
  color: #34495e;
  margin-top: 30px;
  border-bottom: 2px solid #bdc3c7;
  padding-bottom: 5px;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 20px 0;
  font-size: 13px;
}
th {
  background-color: #3498db;
  color: white;
  padding: 12px 8px;
  text-align: left;
  font-weight: bold;
}
td {
  padding: 8px;
  border-bottom: 1px solid #ddd;
}
tr:nth-child(even) {
  background-color: #f8f9fa;
}
tr:hover {
  background-color: #e8f4f8;
}
.numeric {
  text-align: right;
}
.sig-star {
  color: #e74c3c;
  font-weight: bold;
}
ul {
  list-style-type: disc;
  padding-left: 30px;
}
li {
  margin: 8px 0;
}
strong {
  color: #2c3e50;
}
hr {
  border: none;
  border-top: 2px solid #bdc3c7;
  margin: 30px 0;
}
.note {
  background-color: #fff3cd;
  border-left: 4px solid #ffc107;
  padding: 10px 15px;
  margin: 20px 0;
}
</style>
</head>
<body>
""")

    html_parts.append(f"<h1>{title}</h1>\n\n")

    # Model info section
    html_parts.append("<h2>Model Information</h2>\n\n<ul>\n")
    if formula:
        html_parts.append(f"<li><strong>Formula</strong>: <code>{formula}</code></li>\n")
    if n_obs is not None:
        html_parts.append(f"<li><strong>N (observations)</strong>: {n_obs:,}</li>\n")
    if n_triplets is not None:
        html_parts.append(f"<li><strong>N (unique triplets)</strong>: {n_triplets:,}</li>\n")
    html_parts.append("</ul>\n\n")

    if has_error:
        error_msg = frame["error"].dropna().iloc[0] if frame["error"].notna().any() else "unknown"
        html_parts.append(f'<div class="note">\n<strong>⚠ Warning</strong>: Model fitting encountered an issue.<br>\n<code>{error_msg}</code>\n</div>\n\n')

    if not has_se and not has_pval:
        html_parts.append('<div class="note">\n<strong>⚠ Note</strong>: Standard errors, p-values, and confidence intervals are unavailable.<br>\n')
        html_parts.append('This typically indicates a model convergence issue (e.g., perfect separation or multicollinearity).<br>\n')
        html_parts.append('Coefficient estimates below should be interpreted with extreme caution.\n</div>\n\n')

    # Group by outcome if multinomial
    if has_outcome:
        outcomes = frame["outcome"].unique()
        for outcome in outcomes:
            subset = frame[frame["outcome"] == outcome].copy()
            html_parts.append(f"<h2>Outcome: {outcome}</h2>\n\n")
            html_parts.extend(_render_html_coef_table(subset, has_se, has_pval, has_ci))
    else:
        html_parts.append("<h2>Coefficient Estimates</h2>\n\n")
        html_parts.extend(_render_html_coef_table(frame, has_se, has_pval, has_ci))

    # Significance legend
    if has_pval:
        html_parts.append('\n<hr>\n\n<div class="note">\n')
        html_parts.append('<strong>Significance</strong>: <span class="sig-star">***</span> p&lt;0.001, ')
        html_parts.append('<span class="sig-star">**</span> p&lt;0.01, ')
        html_parts.append('<span class="sig-star">*</span> p&lt;0.05, † p&lt;0.1\n')
        html_parts.append('</div>\n\n')

    html_parts.append("</body>\n</html>\n")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(html_parts))


def _render_html_coef_table(
    frame: pd.DataFrame,
    has_se: bool,
    has_pval: bool,
    has_ci: bool,
) -> list[str]:
    """Render a coefficient table as HTML."""
    lines: list[str] = []

    lines.append("<table>\n<thead>\n<tr>\n")

    # Build header
    lines.append("<th>Term</th>\n")
    lines.append('<th class="numeric">Coefficient</th>\n')
    if has_se:
        lines.append('<th class="numeric">Std. Error</th>\n')
    if has_pval:
        lines.append('<th class="numeric">z / t</th>\n')
        lines.append('<th class="numeric">p-value</th>\n')
        lines.append("<th>Sig.</th>\n")
    if has_ci:
        lines.append('<th class="numeric">CI Low</th>\n')
        lines.append('<th class="numeric">CI High</th>\n')

    # Add N Obs and N Triplets columns if available
    if "n_obs" in frame.columns:
        lines.append('<th class="numeric">N Obs</th>\n')
    if "n_triplets" in frame.columns:
        lines.append('<th class="numeric">N Triplets</th>\n')

    lines.append("</tr>\n</thead>\n<tbody>\n")

    for _, row in frame.iterrows():
        lines.append("<tr>\n")

        term = _clean_term_name(str(row.get("term", "")))
        coef = row.get("coef", float("nan"))

        lines.append(f"<td>{term}</td>\n")
        lines.append(f'<td class="numeric">{_format_coef(coef)}</td>\n')

        if has_se:
            se = row.get("std_err", float("nan"))
            se_str = _format_coef(se) if not pd.isna(se) else "—"
            lines.append(f'<td class="numeric">{se_str}</td>\n')

        if has_pval:
            z = row.get("z_or_t", float("nan"))
            p = row.get("p_value", float("nan"))
            z_str = _format_coef(z) if not pd.isna(z) else "—"
            lines.append(f'<td class="numeric">{z_str}</td>\n')

            if not pd.isna(p):
                p_str = f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
                sig_str = _significance_star(p)
                lines.append(f'<td class="numeric">{p_str}</td>\n')
                lines.append(f'<td class="sig-star">{sig_str}</td>\n')
            else:
                lines.append('<td class="numeric">—</td>\n')
                lines.append('<td></td>\n')

        if has_ci:
            cl = row.get("conf_low", float("nan"))
            ch = row.get("conf_high", float("nan"))
            cl_str = _format_coef(cl) if not pd.isna(cl) else "—"
            ch_str = _format_coef(ch) if not pd.isna(ch) else "—"
            lines.append(f'<td class="numeric">{cl_str}</td>\n')
            lines.append(f'<td class="numeric">{ch_str}</td>\n')

        # Add N Obs and N Triplets if available
        if "n_obs" in frame.columns:
            n_obs_val = row.get("n_obs", "")
            if pd.notna(n_obs_val):
                lines.append(f'<td class="numeric">{int(n_obs_val):,}</td>\n')
            else:
                lines.append('<td class="numeric">—</td>\n')
        if "n_triplets" in frame.columns:
            n_triplets_val = row.get("n_triplets", "")
            if pd.notna(n_triplets_val):
                lines.append(f'<td class="numeric">{int(n_triplets_val):,}</td>\n')
            else:
                lines.append('<td class="numeric">—</td>\n')

        lines.append("</tr>\n")

    lines.append("</tbody>\n</table>\n")
    return lines
