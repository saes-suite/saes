"""Standardized Effect-Size Forest Plot generator with exact 95% confidence intervals."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from saes.domain.exceptions import VisualizationError
from saes.domain.models import EffectSizeForestResult
from saes.stats.effect_sizes import compute_effect_size_forest
from saes.visualization.base import Plotter
from saes.visualization.utils import resolve_figure_dimensions


class EffectSizeForestPlotGenerator(Plotter):
    """Generates publication-grade Forest Plots for Hedges' g, Cohen's d, Vargha-Delaney A12, or Cliff's delta."""

    def __init__(self, palette: str = "academic") -> None:
        self.palette = palette

    @property
    def name(self) -> str:
        return "Effect-Size Forest Plot (95% CI)"

    @property
    def description(self) -> str:
        return (
            "Displays pairwise standardized effect sizes with exact noncentral-t (Hedges' g / Cohen's d) "
            "or Brunner-Munzel logit (Vargha-Delaney A12 / Cliff's delta) 95% confidence intervals."
        )

    def _resolve_result(self, df: pd.DataFrame, **kwargs: Any) -> EffectSizeForestResult:
        existing = kwargs.get("forest_result")
        if isinstance(existing, EffectSizeForestResult):
            return existing

        work_df = df.copy()
        if not {"algorithm", "value"}.issubset(work_df.columns):
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns.")
        metric = str(kwargs.get("metric") or (work_df["metric"].iloc[0] if "metric" in work_df.columns else "value"))
        if "metric" not in work_df.columns:
            work_df["metric"] = metric
        if "instance" not in work_df.columns:
            work_df["instance"] = str(kwargs.get("instance") or "Benchmark")
        if "maximize" not in work_df.columns:
            work_df["maximize"] = bool(kwargs.get("maximize", True))

        try:
            res = compute_effect_size_forest(
                df=work_df,
                metric=metric,
                scope=str(kwargs.get("scope", "single-instance")),
                instance=str(kwargs["instance"]) if kwargs.get("instance") else None,
                reference_algorithm=str(kwargs["reference_algorithm"]) if kwargs.get("reference_algorithm") else None,
                alpha=float(kwargs.get("alpha", 0.05)),
                paired=bool(kwargs.get("paired", False)),
                p_adjust=str(kwargs.get("p_adjust", "holm")),
            )
            assert isinstance(res, EffectSizeForestResult)
            return res
        except Exception as e:
            raise VisualizationError(f"Failed to compute effect-size forest: {e!s}") from e

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate publication-grade Matplotlib Forest Plot with 95% confidence intervals."""
        res = self._resolve_result(df, **kwargs)
        effect_metric = str(kwargs.get("effect_metric", "hedges_g")).lower()
        width_px, height_px = resolve_figure_dimensions(kwargs, default_width=820, default_height=460)
        figsize = kwargs.get("figsize", (width_px / 100.0, height_px / 100.0))

        labels = [f"{it.algorithm} vs {it.reference}" for it in res.items]
        y_pos = np.arange(len(labels))

        if effect_metric == "a12":
            vals = [it.a12 for it in res.items]
            lows = [it.a12 - it.a12_ci_lower for it in res.items]
            highs = [it.a12_ci_upper - it.a12 for it in res.items]
            null_val = 0.5
            x_label = "Vargha-Delaney A12 (95% Brunner-Munzel Logit CI; >0.5 favors competitor)"
        elif effect_metric == "cliffs_delta":
            vals = [it.cliffs_delta for it in res.items]
            lows = [it.cliffs_delta - it.cliffs_delta_ci_lower for it in res.items]
            highs = [it.cliffs_delta_ci_upper - it.cliffs_delta for it in res.items]
            null_val = 0.0
            x_label = "Cliff's delta (95% Brunner-Munzel Logit CI; >0 favors competitor)"
        elif effect_metric == "cohens_d":
            vals = [it.cohens_d for it in res.items]
            lows = [it.cohens_d - it.cohens_d_ci_lower for it in res.items]
            highs = [it.cohens_d_ci_upper - it.cohens_d for it in res.items]
            null_val = 0.0
            x_label = "Cohen's d (95% Exact Noncentral-t CI; >0 favors competitor)"
        else:
            vals = [it.hedges_g for it in res.items]
            lows = [it.hedges_g - it.hedges_g_ci_lower for it in res.items]
            highs = [it.hedges_g_ci_upper - it.hedges_g for it in res.items]
            null_val = 0.0
            x_label = "Hedges' g (95% Exact Noncentral-t CI; >0 favors competitor)"

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.axvline(null_val, color="#64748b", linestyle="--", linewidth=1.3, zorder=1)

        for idx, (val, lo, hi, it) in enumerate(zip(vals, lows, highs, res.items, strict=False)):
            col = "#059669" if (it.is_significant and val > null_val) else (
                "#e11d48" if (it.is_significant and val < null_val) else "#475569"
            )
            ax.errorbar(
                val,
                idx,
                xerr=[[max(0.0, lo)], [max(0.0, hi)]],
                fmt="o",
                color=col,
                ecolor=col,
                elinewidth=2.2,
                capsize=4,
                markersize=7.5,
                zorder=4,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
        scope_tag = res.instance if res.instance else "All Instances"
        ax.set_title(
            f"Effect-Size Forest Plot — {res.metric} ({scope_tag}, ref={res.reference_algorithm}, α={res.alpha})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate interactive Plotly Forest Plot."""
        res = self._resolve_result(df, **kwargs)
        effect_metric = str(kwargs.get("effect_metric", "hedges_g")).lower()
        labels = [f"{it.algorithm} vs {it.reference}" for it in res.items]

        if effect_metric == "a12":
            vals = [it.a12 for it in res.items]
            err_hi = [it.a12_ci_upper - it.a12 for it in res.items]
            err_lo = [it.a12 - it.a12_ci_lower for it in res.items]
            null_val = 0.5
            x_title = "Vargha-Delaney A12 (95% CI)"
        elif effect_metric == "cliffs_delta":
            vals = [it.cliffs_delta for it in res.items]
            err_hi = [it.cliffs_delta_ci_upper - it.cliffs_delta for it in res.items]
            err_lo = [it.cliffs_delta - it.cliffs_delta_ci_lower for it in res.items]
            null_val = 0.0
            x_title = "Cliff's delta (95% CI)"
        else:
            vals = [it.hedges_g for it in res.items]
            err_hi = [it.hedges_g_ci_upper - it.hedges_g for it in res.items]
            err_lo = [it.hedges_g - it.hedges_g_ci_lower for it in res.items]
            null_val = 0.0
            x_title = "Hedges' g (95% Exact Noncentral-t CI)"

        colors = [
            "#059669" if (it.is_significant and v > null_val) else (
                "#e11d48" if (it.is_significant and v < null_val) else "#64748b"
            )
            for v, it in zip(vals, res.items, strict=False)
        ]

        fig = go.Figure()
        fig.add_vline(x=null_val, line_dash="dash", line_color="#64748b", line_width=1.5)
        fig.add_trace(
            go.Scatter(
                x=vals,
                y=labels,
                mode="markers",
                marker={"size": 11, "color": colors, "line": {"color": "#0f172a", "width": 1}},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": err_hi,
                    "arrayminus": err_lo,
                    "thickness": 2.2,
                    "color": "#334155",
                },
                hovertemplate="<b>%{y}</b><br>Effect: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title=f"Effect-Size Forest Plot — {res.metric} (ref={res.reference_algorithm})",
            xaxis_title=x_title,
            template="plotly_white",
            showlegend=False,
            height=max(360, 54 * len(labels) + 120),
        )
        return fig
