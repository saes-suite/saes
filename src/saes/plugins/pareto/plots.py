"""Publication-grade Matplotlib and interactive Plotly visualizations for Pareto fronts & EAFs.

Supported plot modes:
- ``runs``: Overlay all runs of 1 or multiple algorithms + optional reference front (or reference-only).
- ``representative``: Faint background runs + bold median/best/worst run + accumulated front.
- ``attainment``: 2D EAF median surface (50%), IQR attainment band (25%-75%), optional 0%/100% bounds.
- ``eaf-diff``: Two-panel mirrored ΔEAF(A - B) showing spatial dominance + KS-EAF permutation p-value.
- ``small-multiples``: Clean facet grid with one panel per algorithm against the reference front.
- ``3d``: 3-objective Pareto front projection (interactive 3D Plotly / static Matplotlib Axes3D).
- ``pcp``: Parallel Coordinates Plot normalized by ideal/nadir of the reference front (m >= 2).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from saes.plugins.pareto.dominance import (
    compute_accumulated_front,
    compute_coverage_and_representatives,
    compute_ideal_nadir,
    filter_nondominated_front,
    to_minimization_space,
)
from saes.plugins.pareto.eaf import compute_attainment, compute_eaf_difference
from saes.plugins.pareto.models import FrontCollection, ParetoFront
from saes.visualization.utils import get_color_map, resolve_figure_dimensions

matplotlib.use("Agg")

VALID_PLOT_MODES = frozenset({"runs", "representative", "attainment", "eaf-diff", "small-multiples", "3d", "pcp"})
VALID_TITLE_MODES = frozenset({"full", "compact", "none"})
VALID_LEGEND_PLACEMENTS = frozenset({"top", "bottom", "right", "inside", "none"})
VALID_LABEL_MODES = frozenset({"all", "x-only", "y-only", "ticks-only", "none"})
VALID_BOLD_MODES = frozenset({"none", "all", "title-axes", "title", "axes"})


@dataclass(frozen=True)
class _PlotStyle:
    title: str | None
    title_mode: str
    legend_placement: str
    font_size: int
    font_family: str
    label_mode: str
    bold_mode: str
    show_grid: bool
    dark_theme: bool
    width_px: int
    height_px: int

    @property
    def bold_title(self) -> bool:
        return self.bold_mode in {"all", "title-axes", "title"}

    @property
    def bold_axes(self) -> bool:
        return self.bold_mode in {"all", "title-axes", "axes"}

    @property
    def bold_ticks(self) -> bool:
        return self.bold_mode == "all"

    @property
    def show_x_title(self) -> bool:
        return self.label_mode in {"all", "x-only"}

    @property
    def show_y_title(self) -> bool:
        return self.label_mode in {"all", "y-only"}

    @property
    def show_x_ticks(self) -> bool:
        return self.label_mode not in {"none", "y-only"}

    @property
    def show_y_ticks(self) -> bool:
        return self.label_mode not in {"none", "x-only"}

    @property
    def fg_primary(self) -> str:
        return "#f8fafc" if self.dark_theme else "#0f172a"

    @property
    def fg_secondary(self) -> str:
        return "#cbd5e1" if self.dark_theme else "#334155"

    @property
    def ref_color(self) -> str:
        return "#f8fafc" if self.dark_theme else "#0f172a"

    @property
    def axis_line_color(self) -> str:
        return "#475569" if self.dark_theme else "#cbd5e1"

    @property
    def grid_color(self) -> str:
        return "rgba(148,163,184,0.20)" if self.dark_theme else "rgba(148,163,184,0.28)"

    @property
    def paper_bg(self) -> str:
        return "#0f172a" if self.dark_theme else "#ffffff"

    @property
    def inside_legend_bg(self) -> str:
        return "rgba(15,23,42,0.88)" if self.dark_theme else "rgba(255,255,255,0.92)"

    def fmt_html(self, text: str, *, bold: bool) -> str:
        return f"<b>{text}</b>" if bold else text

    def resolve_title(self, full_title: str, compact_title: str) -> str | None:
        if self.title is not None and self.title.strip():
            return self.title.strip()
        if self.title_mode == "none":
            return None
        if self.title_mode == "compact":
            return compact_title
        return full_title


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(31, 119, 180, {alpha:.2f})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def _resolve_mpl_rc(font_family: str, font_size: int, dark_theme: bool) -> dict[str, Any]:
    low = font_family.lower()
    if "georgia" in low:
        rc_family = "serif"
        serif_list = ["Georgia", "DejaVu Serif", "Times New Roman"]
        sans_list = ["DejaVu Sans"]
    elif "times" in low or ("serif" in low and "sans" not in low):
        rc_family = "serif"
        serif_list = ["Times New Roman", "DejaVu Serif", "Georgia"]
        sans_list = ["DejaVu Sans"]
    elif "arial" in low or "helvetica" in low:
        rc_family = "sans-serif"
        serif_list = ["DejaVu Serif"]
        sans_list = ["Arial", "Helvetica", "DejaVu Sans"]
    else:
        rc_family = "sans-serif"
        serif_list = ["DejaVu Serif"]
        sans_list = ["Inter", "DejaVu Sans", "Arial"]

    fg = "#f8fafc" if dark_theme else "#0f172a"
    bg = "#0f172a" if dark_theme else "#ffffff"
    edge = "#475569" if dark_theme else "#cbd5e1"
    return {
        "font.family": rc_family,
        "font.serif": serif_list,
        "font.sans-serif": sans_list,
        "font.size": max(8, font_size - 2),
        "axes.facecolor": bg,
        "figure.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": fg,
        "axes.labelcolor": fg,
        "axes.edgecolor": edge,
        "xtick.color": "#cbd5e1" if dark_theme else "#334155",
        "ytick.color": "#cbd5e1" if dark_theme else "#334155",
    }


def _sort_2d_for_line(arr: np.ndarray, minimize: tuple[bool, bool]) -> np.ndarray:
    if arr.shape[0] <= 1:
        return arr
    order = np.argsort(arr[:, 0]) if minimize[0] else np.argsort(-arr[:, 0])
    return arr[order]


def _step_curve_2d(
    points: tuple[tuple[float, float], ...],
    minimize: tuple[bool, bool],
) -> tuple[list[float], list[float]]:
    """Build exact horizontal-vertical staircase coordinates for a 2D attainment surface."""
    if not points:
        return [], []
    arr = np.asarray(points, dtype=np.float64)
    arr = _sort_2d_for_line(arr, minimize)
    xs: list[float] = [float(arr[0, 0])]
    ys: list[float] = [float(arr[0, 1])]
    for i in range(1, arr.shape[0]):
        xs.append(float(arr[i, 0]))
        ys.append(float(arr[i - 1, 1]))
        xs.append(float(arr[i, 0]))
        ys.append(float(arr[i, 1]))
    return xs, ys


def _build_band_polygon_2d(
    lower_pts: tuple[tuple[float, float], ...],
    upper_pts: tuple[tuple[float, float], ...],
    minimize: tuple[bool, bool],
) -> tuple[list[float], list[float]]:
    """Build closed polygon coordinates between two 2D attainment surfaces (e.g. 25% and 75%)."""
    lx, ly = _step_curve_2d(lower_pts, minimize)
    ux, uy = _step_curve_2d(upper_pts, minimize)
    if not lx or not ux:
        return [], []
    return lx + ux[::-1], ly + uy[::-1]


def _resolve_fronts_and_meta(
    collection: FrontCollection,
    instance: str,
    algorithms: list[str] | None,
    run_ids: list[int] | None,
    filter_dominated: bool,
    show_reference: bool,
) -> tuple[list[ParetoFront], ParetoFront | None, list[str], tuple[str, ...], tuple[bool, ...]]:
    ref = collection.get_reference(instance)
    fronts = collection.get_fronts(instance, algorithms=algorithms, run_ids=run_ids)
    if filter_dominated and fronts:
        fronts = [filter_nondominated_front(f) for f in fronts]

    has_visible_ref = show_reference and ref is not None and ref.n_points > 0
    if not fronts and not has_visible_ref:
        raise ValueError(
            f"Nothing to plot for instance='{instance}': select at least one algorithm or enable the reference front."
        )

    if fronts:
        algs = sorted({f.algorithm for f in fronts})
        obj_names = fronts[0].objective_names
        minimize = fronts[0].minimize
    else:
        all_inst_fronts = collection.get_fronts(instance)
        sample = all_inst_fronts[0] if all_inst_fronts else ref
        assert sample is not None
        algs = []
        obj_names = sample.objective_names
        minimize = sample.minimize

    return fronts, ref, algs, obj_names, minimize


def _collect_visible_legend_labels(fig: go.Figure) -> list[str]:
    labels: list[str] = []
    for tr in fig.data:
        show = getattr(tr, "showlegend", None)
        if show is False:
            continue
        name = getattr(tr, "name", None)
        if name:
            labels.append(str(name))
    return labels


def _apply_plotly_chrome(
    fig: go.Figure,
    style: _PlotStyle,
    *,
    resolved_title: str | None,
    has_subplot_titles: bool = False,
    has_right_colorbar: bool = False,
    is_3d: bool = False,
) -> None:
    """Apply consistent non-overlapping title, legend, font, and margin chrome matching SAES Core."""
    fs = style.font_size
    legend_fs = max(9, fs - 1)
    legend_labels = _collect_visible_legend_labels(fig)
    show_legend = style.legend_placement != "none" and len(legend_labels) > 0

    # Estimate wrapped rows for horizontal legend so margins never collide with axes or titles
    avg_len = (sum(len(s) for s in legend_labels) / len(legend_labels)) if legend_labels else 10.0
    max_len = max((len(s) for s in legend_labels), default=10)
    est_item_w = max(92, int(avg_len * legend_fs * 0.64) + 46)
    usable_w = max(280, style.width_px - 110)
    items_per_row = max(1, usable_w // est_item_w)
    legend_rows = max(1, math.ceil(len(legend_labels) / items_per_row)) if show_legend else 0
    legend_row_h = int(legend_fs * 1.85) + 6

    title_block_h = 0
    title_dict: dict[str, Any] | None = None
    if resolved_title:
        n_lines = resolved_title.count("<br>") + 1
        title_fs = fs + 2
        title_block_h = int(title_fs * 1.55 * n_lines) + 22
        title_dict = {
            "text": style.fmt_html(resolved_title, bold=style.bold_title),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
            "yref": "container",
            "yanchor": "top",
            "font": {"family": style.font_family, "size": title_fs, "color": style.fg_primary},
        }
    else:
        title_dict = {"text": ""}

    subplot_header_h = (int(fs * 1.6) + 14) if has_subplot_titles else 0

    left_margin = 24 if is_3d else (max(64, int(40 + fs * 2.2)) if (style.show_y_title or style.show_y_ticks) else 32)
    base_right = 24 if is_3d else (72 if has_right_colorbar else 28)
    right_margin = base_right

    axis_bottom_h = 24 if is_3d else (max(52, int(30 + fs * 2.0)) if (style.show_x_title or style.show_x_ticks) else 28)
    top_margin = max(26, title_block_h + subplot_header_h + (10 if resolved_title else 6))
    bottom_margin = axis_bottom_h

    legend_dict: dict[str, Any] = {
        "font": {"family": style.font_family, "size": legend_fs, "color": style.fg_primary},
        "bgcolor": "rgba(0,0,0,0)",
        "itemsizing": "constant",
        "tracegroupgap": 8,
    }

    if show_legend:
        if style.legend_placement == "top":
            legend_h = legend_rows * legend_row_h + 12
            top_margin = max(26, title_block_h + subplot_header_h + legend_h + 12)
            plot_h = max(180, style.height_px - top_margin - bottom_margin)
            y_pos = 1.0 + (subplot_header_h + 8) / plot_h
            legend_dict.update(
                {
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": round(y_pos, 4),
                    "yanchor": "bottom",
                }
            )
        elif style.legend_placement == "bottom":
            legend_h = legend_rows * legend_row_h + 10
            bottom_margin = axis_bottom_h + legend_h + 14
            plot_h = max(180, style.height_px - top_margin - bottom_margin)
            clearance = max(26, axis_bottom_h - 4)
            y_pos = -(clearance / plot_h)
            legend_dict.update(
                {
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": round(y_pos, 4),
                    "yanchor": "top",
                }
            )
        elif style.legend_placement == "right":
            legend_box_w = min(240, max(125, int(max_len * legend_fs * 0.64) + 46))
            right_margin = base_right + legend_box_w
            legend_dict.update(
                {
                    "orientation": "v",
                    "x": 1.11 if has_right_colorbar else 1.03,
                    "xanchor": "left",
                    "y": 1.0,
                    "yanchor": "top",
                }
            )
        elif style.legend_placement == "inside":
            legend_dict.update(
                {
                    "orientation": "v",
                    "x": 0.98,
                    "xanchor": "right",
                    "y": 0.98,
                    "yanchor": "top",
                    "bgcolor": style.inside_legend_bg,
                    "bordercolor": style.axis_line_color,
                    "borderwidth": 1,
                }
            )

    if has_subplot_titles and fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann_text = str(getattr(ann, "text", "") or "")
            if ann_text:
                ann.text = style.fmt_html(ann_text, bold=style.bold_axes)
                ann.font = {"family": style.font_family, "size": max(10, fs), "color": style.fg_primary}

    fig.update_layout(
        title=title_dict,
        font={"family": style.font_family, "size": fs, "color": style.fg_secondary},
        width=style.width_px,
        height=style.height_px,
        autosize=False,
        showlegend=show_legend,
        legend=legend_dict,
        margin={"l": left_margin, "r": right_margin, "t": top_margin, "b": bottom_margin},
        paper_bgcolor=style.paper_bg,
        plot_bgcolor=style.paper_bg,
    )


def _plotly_axis_kwargs(style: _PlotStyle, title_text: str | None, *, is_x: bool) -> dict[str, Any]:
    show_title = style.show_x_title if is_x else style.show_y_title
    show_ticks = style.show_x_ticks if is_x else style.show_y_ticks
    tick_size = max(9, style.font_size - 1)
    out: dict[str, Any] = {
        "showline": True,
        "linecolor": style.axis_line_color,
        "linewidth": 1,
        "mirror": True,
        "showgrid": style.show_grid,
        "gridcolor": style.grid_color,
        "zeroline": False,
        "showticklabels": show_ticks,
        "ticks": "outside" if show_ticks else "",
        "ticklen": 4,
        "tickcolor": style.axis_line_color,
        "tickfont": {"family": style.font_family, "size": tick_size, "color": style.fg_secondary},
        "automargin": True,
    }
    if show_title and title_text:
        out["title"] = {
            "text": style.fmt_html(title_text, bold=style.bold_axes),
            "font": {"family": style.font_family, "size": style.font_size, "color": style.fg_primary},
            "standoff": 10,
        }
    else:
        out["title"] = {"text": ""}
    return out


def _apply_mpl_axes_style(
    ax: Axes,
    style: _PlotStyle,
    *,
    x_label: str | None,
    y_label: str | None,
) -> None:
    label_weight = "bold" if style.bold_axes else "normal"
    tick_size = max(7, style.font_size - 3)
    axis_size = max(8, style.font_size - 2)

    if style.show_x_title and x_label:
        ax.set_xlabel(x_label, fontsize=axis_size, fontweight=label_weight, color=style.fg_primary)
    else:
        ax.set_xlabel("")

    if style.show_y_title and y_label:
        ax.set_ylabel(y_label, fontsize=axis_size, fontweight=label_weight, color=style.fg_primary)
    else:
        ax.set_ylabel("")

    ax.tick_params(
        axis="x",
        labelsize=tick_size,
        labelbottom=style.show_x_ticks,
        bottom=style.show_x_ticks,
    )
    ax.tick_params(
        axis="y",
        labelsize=tick_size,
        labelleft=style.show_y_ticks,
        left=style.show_y_ticks,
    )
    if style.bold_ticks:
        for tick_lbl in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick_lbl.set_fontweight("bold")

    ax.grid(style.show_grid, alpha=0.25, linestyle="--", color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color(style.axis_line_color)


def _apply_mpl_legend(fig: Figure, ax: Axes, style: _PlotStyle, *, multi_panel: bool = False) -> None:
    if style.legend_placement == "none":
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # Deduplicate labels while preserving order
    seen: set[str] = set()
    uniq_h: list[Any] = []
    uniq_l: list[str] = []
    for h, lbl in zip(handles, labels, strict=True):
        if lbl not in seen:
            seen.add(lbl)
            uniq_h.append(h)
            uniq_l.append(lbl)

    leg_fs = max(7, style.font_size - 3)
    ncol = min(len(uniq_l), max(1, style.width_px // 155))
    frame_fc = "#1e293b" if style.dark_theme else "#ffffff"
    frame_ec = style.axis_line_color

    if style.legend_placement == "inside" and not multi_panel:
        leg = ax.legend(
            uniq_h,
            uniq_l,
            loc="upper right",
            fontsize=leg_fs,
            frameon=True,
            facecolor=frame_fc,
            edgecolor=frame_ec,
            framealpha=0.9,
        )
    elif style.legend_placement == "right":
        if multi_panel:
            leg = fig.legend(
                uniq_h,
                uniq_l,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=leg_fs,
                frameon=False,
            )
        else:
            leg = ax.legend(
                uniq_h,
                uniq_l,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=leg_fs,
                frameon=False,
            )
    elif style.legend_placement == "bottom":
        if multi_panel:
            leg = fig.legend(
                uniq_h,
                uniq_l,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=ncol,
                fontsize=leg_fs,
                frameon=False,
            )
        else:
            leg = ax.legend(
                uniq_h,
                uniq_l,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=ncol,
                fontsize=leg_fs,
                frameon=False,
            )
    else:  # "top" (or "inside" fallback on multi-panel)
        if multi_panel:
            leg = fig.legend(
                uniq_h,
                uniq_l,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94 if style.title_mode != "none" or style.title else 0.99),
                ncol=ncol,
                fontsize=leg_fs,
                frameon=False,
            )
        else:
            leg = ax.legend(
                uniq_h,
                uniq_l,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                fontsize=leg_fs,
                frameon=False,
            )

    for text in leg.get_texts():
        text.set_color(style.fg_primary)


def plot_fronts(
    collection: FrontCollection,
    *,
    instance: str,
    mode: str = "attainment",
    algorithms: list[str] | None = None,
    run_ids: list[int] | None = None,
    show_reference: bool = True,
    filter_dominated: bool = True,
    run_opacity: float = 0.45,
    show_extreme_bounds: bool = False,
    show_accumulated: bool = True,
    algo_a: str | None = None,
    algo_b: str | None = None,
    n_permutations: int = 300,
    seed: int = 42,
    palette: str = "academic",
    paper_format: str = "golden",
    title: str | None = None,
    title_mode: str = "full",
    legend_placement: str = "top",
    font_size: int = 13,
    font_family: str = "Inter, -apple-system, sans-serif",
    label_mode: str = "all",
    bold_mode: str = "title-axes",
    show_grid: bool = True,
    dark_theme: bool = False,
    width_px: int | None = None,
    height_px: int | None = None,
    interactive: bool = False,
    **kwargs: Any,
) -> Figure | go.Figure:
    """Generate a static Matplotlib or interactive Plotly visualization for multi-objective Pareto fronts.

    Supports reference-only rendering when ``algorithms=[]`` and ``show_reference=True`` (for all modes
    except ``eaf-diff``, which compares two distinct algorithms).
    """
    mode_norm = mode.strip().lower()
    if mode_norm not in VALID_PLOT_MODES:
        raise ValueError(f"Unsupported Pareto plot mode '{mode}'. Supported: {sorted(VALID_PLOT_MODES)}.")

    title_mode_norm = title_mode.strip().lower()
    if title_mode_norm not in VALID_TITLE_MODES:
        raise ValueError(f"Unsupported title_mode '{title_mode}'. Supported: {sorted(VALID_TITLE_MODES)}.")

    legend_norm = legend_placement.strip().lower()
    if legend_norm not in VALID_LEGEND_PLACEMENTS:
        raise ValueError(
            f"Unsupported legend_placement '{legend_placement}'. Supported: {sorted(VALID_LEGEND_PLACEMENTS)}."
        )

    label_norm = label_mode.strip().lower()
    if label_norm not in VALID_LABEL_MODES:
        raise ValueError(f"Unsupported label_mode '{label_mode}'. Supported: {sorted(VALID_LABEL_MODES)}.")

    bold_norm = bold_mode.strip().lower()
    if bold_norm not in VALID_BOLD_MODES:
        raise ValueError(f"Unsupported bold_mode '{bold_mode}'. Supported: {sorted(VALID_BOLD_MODES)}.")

    fronts, ref, algs, obj_names, minimize = _resolve_fronts_and_meta(
        collection,
        instance,
        algorithms,
        run_ids,
        filter_dominated,
        show_reference,
    )
    n_obj = len(obj_names)

    # Smart dimension guards
    if mode_norm in {"attainment", "eaf-diff", "small-multiples"} and n_obj != 2:
        raise ValueError(
            f"Mode '{mode_norm}' requires 2 objectives, but instance '{instance}' has {n_obj} objectives. "
            f"Use mode='3d' or mode='pcp' instead."
        )
    if mode_norm == "3d" and n_obj < 3:
        raise ValueError(f"Mode '3d' requires at least 3 objectives, got {n_obj} for '{instance}'.")

    # Preserve deterministic algorithm color mapping across the full instance so toggling an algorithm
    # does not shift colors of the remaining algorithms!
    all_instance_algs = sorted({f.algorithm for f in collection.get_fronts(instance)})
    color_map = get_color_map(all_instance_algs or algs, palette_name=palette)

    dim_kwargs: dict[str, Any] = {"paper_format": paper_format, **kwargs}
    if width_px is not None:
        dim_kwargs["width"] = int(width_px)
    if height_px is not None:
        dim_kwargs["height"] = int(height_px)
    resolved_w, resolved_h = resolve_figure_dimensions(dim_kwargs)

    style = _PlotStyle(
        title=title,
        title_mode=title_mode_norm,
        legend_placement=legend_norm,
        font_size=max(8, min(36, int(font_size))),
        font_family=font_family,
        label_mode=label_norm,
        bold_mode=bold_norm,
        show_grid=bool(show_grid),
        dark_theme=bool(dark_theme),
        width_px=max(360, resolved_w),
        height_px=max(280, resolved_h),
    )

    if interactive:
        return _plot_interactive(
            collection=collection,
            fronts=fronts,
            ref=ref,
            algs=algs,
            obj_names=obj_names,
            minimize=minimize,
            instance=instance,
            mode=mode_norm,
            show_reference=show_reference,
            run_opacity=run_opacity,
            show_extreme_bounds=show_extreme_bounds,
            show_accumulated=show_accumulated,
            algo_a=algo_a,
            algo_b=algo_b,
            n_permutations=n_permutations,
            seed=seed,
            color_map=color_map,
            style=style,
        )

    return _plot_matplotlib(
        collection=collection,
        fronts=fronts,
        ref=ref,
        algs=algs,
        obj_names=obj_names,
        minimize=minimize,
        instance=instance,
        mode=mode_norm,
        show_reference=show_reference,
        run_opacity=run_opacity,
        show_extreme_bounds=show_extreme_bounds,
        show_accumulated=show_accumulated,
        algo_a=algo_a,
        algo_b=algo_b,
        n_permutations=n_permutations,
        seed=seed,
        color_map=color_map,
        style=style,
    )


def _plot_interactive(
    *,
    collection: FrontCollection,
    fronts: list[ParetoFront],
    ref: ParetoFront | None,
    algs: list[str],
    obj_names: tuple[str, ...],
    minimize: tuple[bool, ...],
    instance: str,
    mode: str,
    show_reference: bool,
    run_opacity: float,
    show_extreme_bounds: bool,
    show_accumulated: bool,
    algo_a: str | None,
    algo_b: str | None,
    n_permutations: int,
    seed: int,
    color_map: dict[str, str],
    style: _PlotStyle,
) -> go.Figure:
    n_obj = len(obj_names)
    ref_visible = show_reference and ref is not None and ref.n_points > 0

    # ── 1. EAF DIFFERENCE (Mirrored 2-panel) ────────────────────────────
    if mode == "eaf-diff":
        if len(algs) < 2 and not (algo_a and algo_b):
            raise ValueError("Mode 'eaf-diff' requires at least two algorithms to compare.")
        a_name = algo_a if algo_a else algs[0]
        b_name = algo_b if algo_b else next((a for a in algs if a != a_name), algs[0])
        if a_name == b_name:
            raise ValueError("Mode 'eaf-diff' requires two distinct algorithms.")
        diff = compute_eaf_difference(
            collection,
            instance,
            a_name,
            b_name,
            n_permutations=n_permutations,
            seed=seed,
        )
        att = compute_attainment(collection, instance, algorithms=[a_name, b_name], levels=(0.5,))
        med_map = {item.algorithm: item.surfaces[0].points for item in att.algorithms if item.surfaces}

        diff_arr = np.asarray(diff.diff_matrix, dtype=np.float64)
        pos_z = np.where(diff_arr > 1e-6, diff_arr, np.nan)
        neg_z = np.where(diff_arr < -1e-6, -diff_arr, np.nan)

        cb_font = {"family": style.font_family, "size": max(9, style.font_size - 2), "color": style.fg_secondary}
        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.17,
            subplot_titles=(
                f"{a_name} > {b_name} (+{diff.max_advantage_a * 100:.0f}%)",
                f"{b_name} > {a_name} (+{diff.max_advantage_b * 100:.0f}%)",
            ),
        )

        fig.add_trace(
            go.Heatmap(
                x=diff.x_grid,
                y=diff.y_grid,
                z=pos_z,
                zmin=0.0,
                zmax=1.0,
                colorscale=[[0.0, "#e0f2fe"], [0.4, "#38bdf8"], [0.75, "#0284c7"], [1.0, "#0c4a6e"]],
                colorbar={
                    "title": {"text": "ΔEAF", "font": cb_font},
                    "tickfont": cb_font,
                    "x": 0.432,
                    "len": 0.78,
                    "thickness": 10,
                    "xpad": 4,
                },
                hovertemplate=(
                    f"{obj_names[0]}=%{{x:.4g}}<br>{obj_names[1]}=%{{y:.4g}}"
                    f"<br>ΔEAF=+%{{z:.2f}}<extra>{a_name}</extra>"
                ),
                name=f"{a_name} > {b_name}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=diff.x_grid,
                y=diff.y_grid,
                z=neg_z,
                zmin=0.0,
                zmax=1.0,
                colorscale=[[0.0, "#ffedd5"], [0.4, "#fb923c"], [0.75, "#ea580c"], [1.0, "#7c2d12"]],
                colorbar={
                    "title": {"text": "−ΔEAF", "font": cb_font},
                    "tickfont": cb_font,
                    "x": 1.015,
                    "len": 0.78,
                    "thickness": 10,
                    "xpad": 4,
                },
                hovertemplate=(
                    f"{obj_names[0]}=%{{x:.4g}}<br>{obj_names[1]}=%{{y:.4g}}"
                    f"<br>ΔEAF=+%{{z:.2f}}<extra>{b_name}</extra>"
                ),
                name=f"{b_name} > {a_name}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        min2 = (minimize[0], minimize[1])
        for col_idx in (1, 2):
            if ref_visible and ref is not None:
                rarr = _sort_2d_for_line(ref.to_array(), min2)
                fig.add_trace(
                    go.Scatter(
                        x=rarr[:, 0],
                        y=rarr[:, 1],
                        mode="lines",
                        line={"color": style.ref_color, "width": 2, "dash": "dash"},
                        name="Reference",
                        legendgroup="Reference",
                        showlegend=(col_idx == 1),
                    ),
                    row=1,
                    col=col_idx,
                )
            for alg_key, dash_style in ((a_name, "solid"), (b_name, "dot")):
                pts = med_map.get(alg_key, ())
                sx, sy = _step_curve_2d(pts, min2)
                if sx:
                    fig.add_trace(
                        go.Scatter(
                            x=sx,
                            y=sy,
                            mode="lines",
                            line={"color": color_map.get(alg_key, "#0284c7"), "width": 2.2, "dash": dash_style},
                            name=f"{alg_key} (50%)",
                            legendgroup=alg_key,
                            showlegend=(col_idx == 1),
                        ),
                        row=1,
                        col=col_idx,
                    )

        fig.update_xaxes(**_plotly_axis_kwargs(style, obj_names[0], is_x=True))
        fig.update_yaxes(**_plotly_axis_kwargs(style, obj_names[1], is_x=False), row=1, col=1)
        fig.update_yaxes(**_plotly_axis_kwargs(style, None, is_x=False), row=1, col=2)

        resolved_title = style.resolve_title(
            f"EAF Difference: {a_name} vs {b_name} — {instance} (KS = {diff.ks_statistic:.3f}, p = {diff.p_value:.4g})",
            f"{a_name} vs {b_name} ({instance})",
        )
        _apply_plotly_chrome(
            fig,
            style,
            resolved_title=resolved_title,
            has_subplot_titles=True,
            has_right_colorbar=True,
        )
        return fig

    # ── 2. SMALL MULTIPLES (2D facet per algorithm, or single Reference panel) ──
    if mode == "small-multiples":
        min2 = (minimize[0], minimize[1])
        panels = algs if algs else ["Reference Front"]
        n_panels = len(panels)
        n_cols = min(2, max(1, n_panels))
        n_rows = math.ceil(n_panels / n_cols)
        att_map: dict[str, dict[float, tuple[tuple[float, float], ...]]] = {}
        if algs:
            att = compute_attainment(collection, instance, algorithms=algs, levels=(0.5,))
            att_map = {a.algorithm: {s.level: s.points for s in a.surfaces} for a in att.algorithms}

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=panels,
            horizontal_spacing=0.08,
            vertical_spacing=0.14 if n_rows > 1 else 0.08,
        )

        for idx, panel_name in enumerate(panels):
            r_idx = idx // n_cols + 1
            c_idx = idx % n_cols + 1

            if ref_visible and ref is not None:
                rarr = _sort_2d_for_line(ref.to_array(), min2)
                fig.add_trace(
                    go.Scatter(
                        x=rarr[:, 0],
                        y=rarr[:, 1],
                        mode="lines",
                        line={"color": style.ref_color, "width": 2.2, "dash": "dash"},
                        name="Reference",
                        legendgroup="Reference",
                        showlegend=(idx == 0),
                    ),
                    row=r_idx,
                    col=c_idx,
                )

            if not algs:
                continue

            alg = panel_name
            col_hex = color_map[alg]
            alg_runs = [f for f in fronts if f.algorithm == alg]
            for f in alg_runs:
                arr = f.to_array()
                if arr.shape[0] == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=arr[:, 0],
                        y=arr[:, 1],
                        mode="markers",
                        marker={"color": col_hex, "size": 4, "opacity": max(0.12, min(0.45, run_opacity))},
                        name=alg,
                        legendgroup=alg,
                        showlegend=False,
                        hovertemplate=(
                            f"{alg} (run {f.run_id})<br>{obj_names[0]}=%{{x:.4g}}"
                            f"<br>{obj_names[1]}=%{{y:.4g}}<extra></extra>"
                        ),
                    ),
                    row=r_idx,
                    col=c_idx,
                )

            med_pts = att_map.get(alg, {}).get(0.5, ())
            sx, sy = _step_curve_2d(med_pts, min2)
            if sx:
                fig.add_trace(
                    go.Scatter(
                        x=sx,
                        y=sy,
                        mode="lines",
                        line={"color": col_hex, "width": 2.4},
                        name=alg,
                        legendgroup=alg,
                        showlegend=True,
                    ),
                    row=r_idx,
                    col=c_idx,
                )

        # Only show X title on bottom row and Y title on leftmost column so interior subplot titles never collide
        for r_idx in range(1, n_rows + 1):
            for c_idx in range(1, n_cols + 1):
                fig.update_xaxes(
                    **_plotly_axis_kwargs(style, obj_names[0] if r_idx == n_rows else None, is_x=True),
                    row=r_idx,
                    col=c_idx,
                )
                fig.update_yaxes(
                    **_plotly_axis_kwargs(style, obj_names[1] if c_idx == 1 else None, is_x=False),
                    row=r_idx,
                    col=c_idx,
                )

        default_full = (
            f"Reference Pareto Front — {instance}"
            if not algs
            else f"Per-Algorithm Fronts & 50% EAF — {instance}"
        )
        resolved_title = style.resolve_title(default_full, f"{instance} — Small Multiples")
        _apply_plotly_chrome(fig, style, resolved_title=resolved_title, has_subplot_titles=True)
        return fig

    # ── 3. 3D INTERACTIVE FRONTS ────────────────────────────────────────
    if mode == "3d":
        fig = go.Figure()
        if ref_visible and ref is not None:
            rarr = ref.to_array()
            fig.add_trace(
                go.Scatter3d(
                    x=rarr[:, 0],
                    y=rarr[:, 1],
                    z=rarr[:, 2],
                    mode="markers",
                    marker={
                        "size": 3.5 if not algs else 3,
                        "color": style.ref_color,
                        "opacity": 0.75 if not algs else 0.45,
                        "symbol": "diamond",
                    },
                    name="Reference",
                )
            )
        for alg in algs:
            alg_runs = [f for f in fronts if f.algorithm == alg]
            valid_arrs = [f.to_array() for f in alg_runs if f.n_points > 0]
            if not valid_arrs:
                continue
            pts_3d = np.vstack(valid_arrs)
            fig.add_trace(
                go.Scatter3d(
                    x=pts_3d[:, 0],
                    y=pts_3d[:, 1],
                    z=pts_3d[:, 2],
                    mode="markers",
                    marker={"size": 3.5, "color": color_map[alg], "opacity": max(0.25, run_opacity)},
                    name=f"{alg} ({len(alg_runs)}r)",
                )
            )

        def _scene_axis(name: str, show_title: bool, show_ticks: bool) -> dict[str, Any]:
            return {
                "title": {
                    "text": style.fmt_html(name, bold=style.bold_axes) if show_title else "",
                    "font": {"family": style.font_family, "size": style.font_size, "color": style.fg_primary},
                },
                "showticklabels": show_ticks,
                "showgrid": style.show_grid,
                "gridcolor": style.grid_color,
                "linecolor": style.axis_line_color,
                "backgroundcolor": style.paper_bg,
                "tickfont": {
                    "family": style.font_family,
                    "size": max(9, style.font_size - 2),
                    "color": style.fg_secondary,
                },
            }

        fig.update_layout(
            scene={
                "bgcolor": style.paper_bg,
                "xaxis": _scene_axis(obj_names[0], style.show_x_title, style.show_x_ticks),
                "yaxis": _scene_axis(obj_names[1], style.show_y_title, style.show_y_ticks),
                "zaxis": _scene_axis(
                    obj_names[2],
                    style.label_mode == "all",
                    style.label_mode != "none",
                ),
            }
        )
        default_full = (
            f"3D Reference Pareto Front — {instance}" if not algs else f"3D Pareto Fronts — {instance}"
        )
        resolved_title = style.resolve_title(default_full, f"{instance} (3D)")
        _apply_plotly_chrome(fig, style, resolved_title=resolved_title, is_3d=True)
        return fig

    # ── 4. PARALLEL COORDINATES (Many-Objective normalized) ─────────────
    if mode == "pcp":
        fig = go.Figure()
        all_arrs = [f.to_array() for f in fronts if f.n_points > 0]
        if ref is not None and ref.n_points > 0:
            all_arrs.append(ref.to_array())
        ideal, nadir = compute_ideal_nadir(all_arrs, minimize=minimize)
        span = np.where((nadir - ideal) > 1e-12, nadir - ideal, 1.0)
        x_axes = list(obj_names)

        if ref_visible and ref is not None:
            r_norm = (to_minimization_space(ref.to_array(), minimize) - ideal) / span
            step = max(1, r_norm.shape[0] // (80 if not algs else 40))
            ref_rgba = _hex_to_rgba(style.ref_color, 0.55 if not algs else 0.22)
            for idx, row in enumerate(r_norm[::step]):
                fig.add_trace(
                    go.Scatter(
                        x=x_axes,
                        y=row.tolist(),
                        mode="lines",
                        line={"color": ref_rgba, "width": 1.6 if not algs else 1.2},
                        name="Reference",
                        legendgroup="Reference",
                        showlegend=(idx == 0),
                    )
                )

        if algs:
            cov_rep = compute_coverage_and_representatives(collection, instance, algorithms=algs)
            med_run_by_alg = {r.algorithm: r.median_run_id for r in cov_rep.representative_runs}
            for alg in algs:
                target_run = med_run_by_alg.get(alg)
                matching = [f for f in fronts if f.algorithm == alg and f.run_id == target_run]
                front_use = matching[0] if matching else next(f for f in fronts if f.algorithm == alg)
                norm = (to_minimization_space(front_use.to_array(), minimize) - ideal) / span
                step = max(1, norm.shape[0] // 35)
                col_hex = color_map[alg]
                for idx, row in enumerate(norm[::step]):
                    fig.add_trace(
                        go.Scatter(
                            x=x_axes,
                            y=row.tolist(),
                            mode="lines",
                            line={"color": _hex_to_rgba(col_hex, 0.55), "width": 1.6},
                            name=f"{alg} (#{front_use.run_id})",
                            legendgroup=alg,
                            showlegend=(idx == 0),
                        )
                    )

        fig.update_xaxes(**_plotly_axis_kwargs(style, "Objectives", is_x=True))
        y_cfg = _plotly_axis_kwargs(style, "Normalized Value (0 = ideal, 1 = nadir)", is_x=False)
        y_cfg["range"] = [-0.05, 1.08]
        fig.update_yaxes(**y_cfg)

        default_full = (
            f"Reference Parallel Coordinates ({n_obj}D) — {instance}"
            if not algs
            else f"Normalized Parallel Coordinates ({n_obj} Objectives) — {instance}"
        )
        resolved_title = style.resolve_title(default_full, f"{instance} ({n_obj}D PCP)")
        _apply_plotly_chrome(fig, style, resolved_title=resolved_title)
        return fig

    # ── 5. 2D STANDARD MODES: runs, representative, attainment ──────────
    fig = go.Figure()
    min2 = (minimize[0], minimize[1])

    if ref_visible and ref is not None:
        rarr = _sort_2d_for_line(ref.to_array(), min2)
        fig.add_trace(
            go.Scatter(
                x=rarr[:, 0],
                y=rarr[:, 1],
                mode="lines",
                line={"color": style.ref_color, "width": 2.6 if not algs else 2.2, "dash": "dash"},
                name="Reference",
                legendgroup="Reference",
                hovertemplate=(
                    f"Reference<br>{obj_names[0]}=%{{x:.4g}}<br>{obj_names[1]}=%{{y:.4g}}<extra></extra>"
                ),
            )
        )

    if not algs:
        default_full = f"Reference Pareto Front — {instance}"
        default_compact = f"{instance} (Reference)"
    elif mode == "runs":
        for alg in algs:
            alg_runs = sorted([f for f in fronts if f.algorithm == alg], key=lambda f: f.run_id)
            col_hex = color_map[alg]
            for idx, f in enumerate(alg_runs):
                arr = _sort_2d_for_line(f.to_array(), min2)
                if arr.shape[0] == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=arr[:, 0],
                        y=arr[:, 1],
                        mode="markers+lines" if len(alg_runs) <= 5 else "markers",
                        marker={"color": col_hex, "size": 5, "opacity": run_opacity},
                        line={"color": _hex_to_rgba(col_hex, run_opacity * 0.7), "width": 1.1},
                        name=f"{alg} ({len(alg_runs)}r)" if len(alg_runs) > 1 else alg,
                        legendgroup=alg,
                        showlegend=(idx == 0),
                        hovertemplate=(
                            f"{alg} — run #{f.run_id}<br>{obj_names[0]}=%{{x:.4g}}"
                            f"<br>{obj_names[1]}=%{{y:.4g}}<extra></extra>"
                        ),
                    )
                )
        default_full = f"Pareto Front Runs Overlay — {instance}"
        default_compact = f"{instance} — Runs"

    elif mode == "representative":
        cov_rep = compute_coverage_and_representatives(collection, instance, algorithms=algs)
        rep_map = {r.algorithm: r for r in cov_rep.representative_runs}
        single_alg = len(algs) == 1
        for alg in algs:
            alg_runs = sorted([f for f in fronts if f.algorithm == alg], key=lambda f: f.run_id)
            col_hex = color_map[alg]
            rep = rep_map[alg]

            bg_pts = [f.to_array() for f in alg_runs if f.n_points > 0]
            if bg_pts:
                stacked = np.vstack(bg_pts)
                fig.add_trace(
                    go.Scatter(
                        x=stacked[:, 0],
                        y=stacked[:, 1],
                        mode="markers",
                        marker={"color": col_hex, "size": 4, "opacity": min(0.22, run_opacity * 0.45)},
                        name=f"{alg} runs",
                        legendgroup=alg,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            run_lookup = {f.run_id: f for f in alg_runs}
            for label, rid, dash, width in (
                ("median", rep.median_run_id, "solid", 2.6),
                ("best", rep.best_run_id, "dash", 1.8),
                ("worst", rep.worst_run_id, "dot", 1.5),
            ):
                rf = run_lookup.get(rid)
                if rf is None or rf.n_points == 0:
                    continue
                arr = _sort_2d_for_line(rf.to_array(), min2)
                trace_name = f"{label.title()} (#{rid})" if single_alg else f"{alg} (#{rid})"
                fig.add_trace(
                    go.Scatter(
                        x=arr[:, 0],
                        y=arr[:, 1],
                        mode="lines+markers" if label == "median" else "lines",
                        marker={"size": 5, "color": col_hex},
                        line={"color": col_hex, "width": width, "dash": dash},
                        name=trace_name,
                        legendgroup=alg,
                        showlegend=(single_alg or label == "median"),
                        hovertemplate=(
                            f"{alg} {label} (run #{rid})<br>{obj_names[0]}=%{{x:.4g}}"
                            f"<br>{obj_names[1]}=%{{y:.4g}}<extra></extra>"
                        ),
                    )
                )

            if show_accumulated and len(alg_runs) > 1:
                acc = compute_accumulated_front(alg_runs)
                a_arr = _sort_2d_for_line(acc.to_array(), min2)
                if a_arr.shape[0] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=a_arr[:, 0],
                            y=a_arr[:, 1],
                            mode="markers",
                            marker={"symbol": "star-open", "size": 6, "color": col_hex},
                            name="Accumulated ND" if single_alg else f"{alg} ND union",
                            legendgroup=alg,
                            showlegend=single_alg,
                            visible="legendonly" if single_alg else True,
                        )
                    )
        default_full = f"Representative Fronts (Solid=Median, Dashed=Best, Dotted=Worst) — {instance}"
        default_compact = f"{instance} — Representative"

    else:  # attainment
        levels = (0.0, 0.25, 0.5, 0.75, 1.0) if show_extreme_bounds else (0.25, 0.5, 0.75)
        att = compute_attainment(collection, instance, algorithms=algs, levels=levels)
        single_alg = len(att.algorithms) == 1
        for item in att.algorithms:
            col_hex = color_map[item.algorithm]
            smap = {s.level: s.points for s in item.surfaces}

            # Shaded IQR band (25% - 75%) — grouped with algorithm so legend stays concise and never overlaps
            bx, by = _build_band_polygon_2d(smap.get(0.25, ()), smap.get(0.75, ()), min2)
            if bx:
                fig.add_trace(
                    go.Scatter(
                        x=bx,
                        y=by,
                        fill="toself",
                        fillcolor=_hex_to_rgba(col_hex, 0.20),
                        line={"width": 0},
                        mode="lines",
                        name="IQR (25%–75%)" if single_alg else f"{item.algorithm} IQR",
                        legendgroup=item.algorithm,
                        showlegend=single_alg,
                        hoverinfo="skip",
                    )
                )

            # Optional 0% (best) and 100% (worst) outer attainment envelopes
            if show_extreme_bounds:
                for lvl_idx, (lvl, lbl) in enumerate(((0.0, "0% / 100% bounds"), (1.0, "worst (100%)"))):
                    ex, ey = _step_curve_2d(smap.get(lvl, ()), min2)
                    if ex:
                        fig.add_trace(
                            go.Scatter(
                                x=ex,
                                y=ey,
                                mode="lines",
                                line={"color": _hex_to_rgba(col_hex, 0.65), "width": 1.2, "dash": "dot"},
                                name=lbl,
                                legendgroup=item.algorithm,
                                showlegend=(single_alg and lvl_idx == 0),
                                hovertemplate=(
                                    f"{item.algorithm} {int(lvl * 100)}% EAF<br>{obj_names[0]}=%{{x:.4g}}"
                                    f"<br>{obj_names[1]}=%{{y:.4g}}<extra></extra>"
                                ),
                            )
                        )

            # Bold 50% median attainment surface (primary legend entry)
            mx, my = _step_curve_2d(smap.get(0.5, ()), min2)
            if mx:
                fig.add_trace(
                    go.Scatter(
                        x=mx,
                        y=my,
                        mode="lines",
                        line={"color": col_hex, "width": 2.8},
                        name=f"{item.algorithm} (50% EAF)" if single_alg else item.algorithm,
                        legendgroup=item.algorithm,
                        showlegend=True,
                        hovertemplate=(
                            f"{item.algorithm} 50% EAF (n={item.n_runs})<br>{obj_names[0]}=%{{x:.4g}}"
                            f"<br>{obj_names[1]}=%{{y:.4g}}<extra></extra>"
                        ),
                    )
                )
        default_full = f"Empirical Attainment Surfaces (50% Median & 25%–75% IQR) — {instance}"
        default_compact = f"{instance} — EAF Attainment"

    fig.update_xaxes(**_plotly_axis_kwargs(style, obj_names[0], is_x=True))
    fig.update_yaxes(**_plotly_axis_kwargs(style, obj_names[1], is_x=False))
    resolved_title = style.resolve_title(default_full, default_compact)
    _apply_plotly_chrome(fig, style, resolved_title=resolved_title)
    return fig


def _plot_matplotlib(
    *,
    collection: FrontCollection,
    fronts: list[ParetoFront],
    ref: ParetoFront | None,
    algs: list[str],
    obj_names: tuple[str, ...],
    minimize: tuple[bool, ...],
    instance: str,
    mode: str,
    show_reference: bool,
    run_opacity: float,
    show_extreme_bounds: bool,
    show_accumulated: bool,
    algo_a: str | None,
    algo_b: str | None,
    n_permutations: int,
    seed: int,
    color_map: dict[str, str],
    style: _PlotStyle,
) -> Figure:
    dpi = 110
    fig_w = max(4.2, style.width_px / dpi)
    fig_h = max(3.2, style.height_px / dpi)
    ref_visible = show_reference and ref is not None and ref.n_points > 0
    title_fs = max(9, style.font_size)
    title_weight = "bold" if style.bold_title else "normal"
    sub_fs = max(8, style.font_size - 2)
    sub_weight = "bold" if style.bold_axes else "normal"

    with plt.rc_context(_resolve_mpl_rc(style.font_family, style.font_size, style.dark_theme)):  # type: ignore[arg-type]
        # ── 1. EAF DIFFERENCE ───────────────────────────────────────────────
        if mode == "eaf-diff":
            if len(algs) < 2 and not (algo_a and algo_b):
                raise ValueError("Mode 'eaf-diff' requires at least two algorithms to compare.")
            a_name = algo_a if algo_a else algs[0]
            b_name = algo_b if algo_b else next((a for a in algs if a != a_name), algs[0])
            if a_name == b_name:
                raise ValueError("Mode 'eaf-diff' requires two distinct algorithms.")
            diff = compute_eaf_difference(
                collection, instance, a_name, b_name, n_permutations=n_permutations, seed=seed
            )
            att = compute_attainment(collection, instance, algorithms=[a_name, b_name], levels=(0.5,))
            med_map = {item.algorithm: item.surfaces[0].points for item in att.algorithms if item.surfaces}

            diff_arr = np.asarray(diff.diff_matrix, dtype=np.float64)
            pos_z = np.ma.masked_where(diff_arr <= 1e-6, diff_arr)
            neg_z = np.ma.masked_where(diff_arr >= -1e-6, -diff_arr)
            x_g = np.asarray(diff.x_grid, dtype=np.float64)
            y_g = np.asarray(diff.y_grid, dtype=np.float64)
            min2 = (minimize[0], minimize[1])

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(max(7.6, fig_w), fig_h), sharex=True, sharey=True
            )
            m1 = ax1.pcolormesh(x_g, y_g, pos_z, shading="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
            m2 = ax2.pcolormesh(x_g, y_g, neg_z, shading="nearest", cmap="Oranges", vmin=0.0, vmax=1.0)
            fig.colorbar(m1, ax=ax1, fraction=0.046, pad=0.03, label="ΔEAF")
            fig.colorbar(m2, ax=ax2, fraction=0.046, pad=0.03, label="−ΔEAF")

            for idx_ax, (ax, sub_title) in enumerate(
                (
                    (ax1, f"{a_name} > {b_name} (+{diff.max_advantage_a * 100:.0f}%)"),
                    (ax2, f"{b_name} > {a_name} (+{diff.max_advantage_b * 100:.0f}%)"),
                )
            ):
                if ref_visible and ref is not None:
                    rarr = _sort_2d_for_line(ref.to_array(), min2)
                    ax.plot(rarr[:, 0], rarr[:, 1], "--", color=style.ref_color, lw=1.6, label="Reference")
                for alg_key, ls in ((a_name, "-"), (b_name, ":")):
                    sx, sy = _step_curve_2d(med_map.get(alg_key, ()), min2)
                    if sx:
                        ax.plot(sx, sy, ls, color=color_map.get(alg_key, "#0284c7"), lw=1.8, label=f"{alg_key} (50%)")
                ax.set_title(sub_title, fontsize=sub_fs, fontweight=sub_weight, color=style.fg_primary)
                _apply_mpl_axes_style(
                    ax,
                    style,
                    x_label=obj_names[0],
                    y_label=obj_names[1] if idx_ax == 0 else None,
                )

            resolved = style.resolve_title(
                f"EAF Difference ({instance}): {a_name} vs {b_name} | KS={diff.ks_statistic:.3f}, p={diff.p_value:.4g}",
                f"{a_name} vs {b_name} ({instance})",
            )
            if resolved:
                fig.suptitle(resolved, fontsize=title_fs, fontweight=title_weight, color=style.fg_primary)
            _apply_mpl_legend(fig, ax1, style, multi_panel=True)
            top_rect = 0.86 if (resolved and style.legend_placement == "top") else (0.93 if resolved else 0.96)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
            return fig

        # ── 2. SMALL MULTIPLES ──────────────────────────────────────────────
        if mode == "small-multiples":
            min2 = (minimize[0], minimize[1])
            panels = algs if algs else ["Reference Front"]
            n_panels = len(panels)
            n_cols = min(2, max(1, n_panels))
            n_rows = math.ceil(n_panels / n_cols)
            med_map = {}
            if algs:
                att = compute_attainment(collection, instance, algorithms=algs, levels=(0.5,))
                med_map = {a.algorithm: a.surfaces[0].points for a in att.algorithms if a.surfaces}

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(max(fig_w, 3.8 * n_cols), max(fig_h, 3.1 * n_rows)),
                sharex=True,
                sharey=True,
            )
            axes_flat = np.atleast_1d(axes).ravel()
            for idx, panel_name in enumerate(panels):
                ax = axes_flat[idx]
                r_idx = idx // n_cols
                c_idx = idx % n_cols
                if ref_visible and ref is not None:
                    rarr = _sort_2d_for_line(ref.to_array(), min2)
                    ax.plot(rarr[:, 0], rarr[:, 1], "--", color=style.ref_color, lw=1.6, label="Reference")
                if algs:
                    alg = panel_name
                    col_hex = color_map[alg]
                    for f in (fr for fr in fronts if fr.algorithm == alg):
                        arr = f.to_array()
                        if arr.shape[0] > 0:
                            ax.scatter(arr[:, 0], arr[:, 1], s=10, color=col_hex, alpha=min(0.4, run_opacity))
                    sx, sy = _step_curve_2d(med_map.get(alg, ()), min2)
                    if sx:
                        ax.plot(sx, sy, "-", color=col_hex, lw=2.0, label="50% EAF")
                ax.set_title(panel_name, fontsize=sub_fs, fontweight=sub_weight, color=style.fg_primary)
                _apply_mpl_axes_style(
                    ax,
                    style,
                    x_label=obj_names[0] if r_idx == n_rows - 1 else None,
                    y_label=obj_names[1] if c_idx == 0 else None,
                )
            for extra_ax in axes_flat[n_panels:]:
                extra_ax.set_visible(False)

            default_full = (
                f"Reference Pareto Front — {instance}"
                if not algs
                else f"Per-Algorithm Fronts & 50% EAF — {instance}"
            )
            resolved = style.resolve_title(default_full, f"{instance} — Small Multiples")
            if resolved:
                fig.suptitle(resolved, fontsize=title_fs, fontweight=title_weight, color=style.fg_primary)
            _apply_mpl_legend(fig, axes_flat[0], style, multi_panel=True)
            top_rect = 0.87 if (resolved and style.legend_placement == "top") else (0.93 if resolved else 0.96)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
            return fig

        # ── 3. 3D STATIC MATPLOTLIB ─────────────────────────────────────────
        if mode == "3d":
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax_3d: Any = fig.add_subplot(111, projection="3d")
            ax_3d.set_facecolor(style.paper_bg)
            if ref_visible and ref is not None:
                rarr = ref.to_array()
                ax_3d.scatter(
                    rarr[:, 0],
                    rarr[:, 1],
                    rarr[:, 2],
                    s=14 if not algs else 10,
                    color=style.ref_color,
                    alpha=0.75 if not algs else 0.35,
                    label="Reference",
                )
            for alg in algs:
                alg_runs = [f for f in fronts if f.algorithm == alg]
                valid_arrs = [f.to_array() for f in alg_runs if f.n_points > 0]
                if not valid_arrs:
                    continue
                pts = np.vstack(valid_arrs)
                ax_3d.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=14,
                    color=color_map[alg],
                    alpha=max(0.3, run_opacity),
                    label=alg,
                )
            axis_size = max(8, style.font_size - 2)
            label_weight = "bold" if style.bold_axes else "normal"
            if style.show_x_title:
                ax_3d.set_xlabel(obj_names[0], fontsize=axis_size, fontweight=label_weight)
            if style.show_y_title:
                ax_3d.set_ylabel(obj_names[1], fontsize=axis_size, fontweight=label_weight)
            if style.label_mode == "all":
                ax_3d.set_zlabel(obj_names[2], fontsize=axis_size, fontweight=label_weight)
            ax_3d.grid(style.show_grid)
            default_full = (
                f"3D Reference Pareto Front — {instance}" if not algs else f"3D Pareto Fronts — {instance}"
            )
            resolved = style.resolve_title(default_full, f"{instance} (3D)")
            if resolved:
                ax_3d.set_title(resolved, fontsize=title_fs, fontweight=title_weight, color=style.fg_primary)
            _apply_mpl_legend(fig, ax_3d, style)
            fig.tight_layout()
            return fig

        # ── 4. PARALLEL COORDINATES (Many-Objective) ────────────────────────
        if mode == "pcp":
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            all_arrs = [f.to_array() for f in fronts if f.n_points > 0]
            if ref is not None and ref.n_points > 0:
                all_arrs.append(ref.to_array())
            ideal, nadir = compute_ideal_nadir(all_arrs, minimize=minimize)
            span = np.where((nadir - ideal) > 1e-12, nadir - ideal, 1.0)
            x_idx = np.arange(len(obj_names))

            if ref_visible and ref is not None:
                r_norm = (to_minimization_space(ref.to_array(), minimize) - ideal) / span
                step = max(1, r_norm.shape[0] // (80 if not algs else 40))
                for idx, row in enumerate(r_norm[::step]):
                    ax.plot(
                        x_idx,
                        row,
                        color=style.ref_color,
                        alpha=0.5 if not algs else 0.2,
                        lw=1.3 if not algs else 1.0,
                        label="Reference" if idx == 0 else None,
                    )

            if algs:
                cov_rep = compute_coverage_and_representatives(collection, instance, algorithms=algs)
                med_run_by_alg = {r.algorithm: r.median_run_id for r in cov_rep.representative_runs}
                for alg in algs:
                    target_run = med_run_by_alg.get(alg)
                    matching = [f for f in fronts if f.algorithm == alg and f.run_id == target_run]
                    front_use = matching[0] if matching else next(f for f in fronts if f.algorithm == alg)
                    norm = (to_minimization_space(front_use.to_array(), minimize) - ideal) / span
                    step = max(1, norm.shape[0] // 35)
                    for idx, row in enumerate(norm[::step]):
                        ax.plot(
                            x_idx,
                            row,
                            color=color_map[alg],
                            alpha=0.5,
                            lw=1.3,
                            label=f"{alg} (#{front_use.run_id})" if idx == 0 else None,
                        )

            ax.set_xticks(x_idx)
            ax.set_xticklabels(list(obj_names))
            _apply_mpl_axes_style(
                ax,
                style,
                x_label="Objectives",
                y_label="Normalized Objective (0 = ideal, 1 = nadir)",
            )
            default_full = (
                f"Reference Parallel Coordinates — {instance}"
                if not algs
                else f"Normalized Parallel Coordinates — {instance}"
            )
            resolved = style.resolve_title(default_full, f"{instance} (PCP)")
            if resolved:
                pad = 24 if style.legend_placement == "top" else 10
                ax.set_title(resolved, fontsize=title_fs, fontweight=title_weight, color=style.fg_primary, pad=pad)
            _apply_mpl_legend(fig, ax, style)
            fig.tight_layout()
            return fig

        # ── 5. 2D STANDARD MODES: runs, representative, attainment ──────────
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        min2 = (minimize[0], minimize[1])

        if ref_visible and ref is not None:
            rarr = _sort_2d_for_line(ref.to_array(), min2)
            ax.plot(
                rarr[:, 0],
                rarr[:, 1],
                "--",
                color=style.ref_color,
                lw=2.1 if not algs else 1.8,
                label="Reference",
                zorder=10,
            )

        if not algs:
            default_full = f"Reference Pareto Front — {instance}"
            default_compact = f"{instance} (Reference)"
        elif mode == "runs":
            for alg in algs:
                alg_runs = sorted([f for f in fronts if f.algorithm == alg], key=lambda f: f.run_id)
                col_hex = color_map[alg]
                for idx, f in enumerate(alg_runs):
                    arr = _sort_2d_for_line(f.to_array(), min2)
                    if arr.shape[0] == 0:
                        continue
                    ax.scatter(
                        arr[:, 0],
                        arr[:, 1],
                        s=14,
                        color=col_hex,
                        alpha=run_opacity,
                        label=(f"{alg} ({len(alg_runs)}r)" if len(alg_runs) > 1 else alg) if idx == 0 else None,
                    )
            default_full = f"Pareto Front Runs Overlay — {instance}"
            default_compact = f"{instance} — Runs"

        elif mode == "representative":
            cov_rep = compute_coverage_and_representatives(collection, instance, algorithms=algs)
            rep_map = {r.algorithm: r for r in cov_rep.representative_runs}
            single_alg = len(algs) == 1
            for alg in algs:
                alg_runs = sorted([f for f in fronts if f.algorithm == alg], key=lambda f: f.run_id)
                col_hex = color_map[alg]
                rep = rep_map[alg]
                bg = [f.to_array() for f in alg_runs if f.n_points > 0]
                if bg:
                    stk = np.vstack(bg)
                    ax.scatter(stk[:, 0], stk[:, 1], s=8, color=col_hex, alpha=min(0.2, run_opacity * 0.4))
                run_lookup = {f.run_id: f for f in alg_runs}
                for label, rid, ls, lw in (
                    ("median", rep.median_run_id, "-", 2.2),
                    ("best", rep.best_run_id, "--", 1.4),
                    ("worst", rep.worst_run_id, ":", 1.2),
                ):
                    rf = run_lookup.get(rid)
                    if rf is None or rf.n_points == 0:
                        continue
                    arr = _sort_2d_for_line(rf.to_array(), min2)
                    lbl = (f"{label.title()} (#{rid})" if single_alg else f"{alg} (#{rid})") if (
                        single_alg or label == "median"
                    ) else None
                    ax.plot(arr[:, 0], arr[:, 1], ls, color=col_hex, lw=lw, label=lbl)
                if show_accumulated and len(alg_runs) > 1:
                    acc = compute_accumulated_front(alg_runs)
                    a_arr = _sort_2d_for_line(acc.to_array(), min2)
                    if a_arr.shape[0] > 0:
                        ax.scatter(
                            a_arr[:, 0],
                            a_arr[:, 1],
                            s=18,
                            marker="*",
                            color=col_hex,
                            alpha=0.8,
                            label="Accumulated ND" if single_alg else None,
                        )
            default_full = f"Representative Fronts (Best / Median / Worst) — {instance}"
            default_compact = f"{instance} — Representative"

        else:  # attainment
            levels = (0.0, 0.25, 0.5, 0.75, 1.0) if show_extreme_bounds else (0.25, 0.5, 0.75)
            att = compute_attainment(collection, instance, algorithms=algs, levels=levels)
            single_alg = len(att.algorithms) == 1
            for item in att.algorithms:
                col_hex = color_map[item.algorithm]
                smap = {s.level: s.points for s in item.surfaces}
                bx, by = _build_band_polygon_2d(smap.get(0.25, ()), smap.get(0.75, ()), min2)
                if bx:
                    ax.fill(
                        bx,
                        by,
                        color=col_hex,
                        alpha=0.20,
                        label="IQR (25%–75%)" if single_alg else None,
                    )
                if show_extreme_bounds:
                    for lvl_idx, lvl in enumerate((0.0, 1.0)):
                        ex, ey = _step_curve_2d(smap.get(lvl, ()), min2)
                        if ex:
                            ax.plot(
                                ex,
                                ey,
                                ":",
                                color=col_hex,
                                lw=1.0,
                                alpha=0.65,
                                label="0% / 100% bounds" if (single_alg and lvl_idx == 0) else None,
                            )
                mx, my = _step_curve_2d(smap.get(0.5, ()), min2)
                if mx:
                    ax.plot(
                        mx,
                        my,
                        "-",
                        color=col_hex,
                        lw=2.3,
                        label=f"{item.algorithm} (50% EAF)" if single_alg else item.algorithm,
                    )
            default_full = f"Empirical Attainment Surfaces (50% Median & IQR) — {instance}"
            default_compact = f"{instance} — EAF Attainment"

        _apply_mpl_axes_style(ax, style, x_label=obj_names[0], y_label=obj_names[1])
        resolved = style.resolve_title(default_full, default_compact)
        if resolved:
            pad = 26 if style.legend_placement == "top" else 10
            ax.set_title(resolved, fontsize=title_fs, fontweight=title_weight, color=style.fg_primary, pad=pad)
        _apply_mpl_legend(fig, ax, style)
        fig.tight_layout()
        return fig


def export_front_plot(
    collection: FrontCollection,
    *,
    instance: str,
    export_format: str = "pdf",
    dpi: int = 300,
    **kwargs: Any,
) -> tuple[bytes, str, str]:
    """Export any Pareto/EAF figure as publication-grade PNG, SVG, or PDF bytes."""
    fmt = export_format.strip().lower()
    format_map: dict[str, tuple[str, str]] = {
        "png": ("image/png", "png"),
        "svg": ("image/svg+xml", "svg"),
        "pdf": ("application/pdf", "pdf"),
    }
    if fmt not in format_map:
        raise ValueError(f"Unsupported export format '{export_format}'. Supported: png, svg, pdf.")

    fig = plot_fronts(collection, instance=instance, interactive=False, **kwargs)
    assert isinstance(fig, Figure)
    buf = io.BytesIO()
    try:
        save_kwargs: dict[str, Any] = {
            "format": fmt,
            "bbox_inches": "tight",
            "facecolor": fig.get_facecolor(),
        }
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(buf, **save_kwargs)
    finally:
        plt.close(fig)
    media_type, ext = format_map[fmt]
    return buf.getvalue(), media_type, ext
