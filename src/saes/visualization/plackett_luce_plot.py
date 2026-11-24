"""Bayesian Plackett-Luce multi-algorithm ranking plot generator (Calvo et al. 2018)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from saes.domain.exceptions import VisualizationError
from saes.domain.models import BayesianPlackettLuceResult
from saes.stats.bayesian import BayesianPlackettLuceTest
from saes.stats.ranking import _extract_block_matrix
from saes.visualization.base import Plotter
from saes.visualization.utils import get_color_map, resolve_figure_dimensions


class PlackettLucePlotGenerator(Plotter):
    """Generates dual-panel Bayesian Plackett-Luce posterior P(best) and expected-rank CrI figures."""

    def __init__(self, palette: str = "academic") -> None:
        self.palette = palette

    @property
    def name(self) -> str:
        return "Bayesian Plackett-Luce Ranking Plot"

    @property
    def description(self) -> str:
        return (
            "Displays posterior winning probabilities P(best) and 95% credible intervals for expected ranks "
            "under the Bayesian Plackett-Luce model (Calvo et al. 2018)."
        )

    def _resolve_result(self, df: pd.DataFrame, **kwargs: Any) -> BayesianPlackettLuceResult:
        existing = kwargs.get("plackett_luce_result")
        if isinstance(existing, BayesianPlackettLuceResult):
            return existing

        work_df = df.copy()
        if not {"algorithm", "value"}.issubset(work_df.columns):
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns.")
        metric = str(kwargs.get("metric") or (work_df["metric"].iloc[0] if "metric" in work_df.columns else "value"))
        if "metric" not in work_df.columns:
            work_df["metric"] = metric
        if "instance" not in work_df.columns:
            work_df["instance"] = str(kwargs.get("instance") or "Benchmark")
        if "run_id" not in work_df.columns:
            work_df["run_id"] = work_df.groupby(["algorithm", "instance"]).cumcount()
        if "maximize" not in work_df.columns:
            work_df["maximize"] = bool(kwargs.get("maximize", True))

        scope = str(kwargs.get("scope", "all-instances"))
        instance = kwargs.get("instance")
        num_samples = int(kwargs.get("num_samples", 400))
        prior_alpha = float(kwargs.get("prior_alpha", 1.0))

        try:
            algorithms, matrix, maximize, *_ = _extract_block_matrix(
                df=work_df,
                metric=metric,
                scope=scope,
                instance=str(instance) if instance else None,
            )
            return BayesianPlackettLuceTest().run(
                matrix=matrix,
                algorithms=algorithms,
                metric=metric,
                maximize=maximize,
                num_samples=num_samples,
                prior_alpha=prior_alpha,
            )
        except Exception as e:
            raise VisualizationError(f"Failed to compute Bayesian Plackett-Luce ranking: {e!s}") from e

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate publication-grade Matplotlib dual-panel Plackett-Luce figure."""
        res = self._resolve_result(df, **kwargs)
        width_px, height_px = resolve_figure_dimensions(kwargs, default_width=900, default_height=460)
        figsize = kwargs.get("figsize", (width_px / 100.0, height_px / 100.0))
        palette_name = str(kwargs.get("palette", self.palette))
        algos = [it.algorithm for it in res.items]
        color_map = get_color_map(algos, palette_name=palette_name, custom_colors=kwargs.get("custom_colors"))
        colors = [color_map[a] for a in algos]
        probs = [it.prob_best for it in res.items]
        exp_ranks = [it.expected_rank for it in res.items]
        ci_low = [it.expected_rank - it.rank_ci_lower for it in res.items]
        ci_high = [it.rank_ci_upper - it.expected_rank for it in res.items]
        y_pos = np.arange(len(algos))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150, sharey=True)

        # Left panel: Posterior P(best)
        bars = ax1.barh(y_pos, probs, color=colors, edgecolor="#1e293b", linewidth=0.8, alpha=0.9)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(algos, fontsize=10, fontweight="bold")
        ax1.invert_yaxis()
        ax1.set_xlabel("Posterior Probability P(Best)", fontsize=10, fontweight="bold")
        ax1.set_xlim(0.0, max(1.0, max(probs) * 1.18))
        ax1.grid(axis="x", linestyle="--", alpha=0.35)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for bar, p in zip(bars, probs, strict=False):
            ax1.text(
                p + 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                f"{p * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
            )

        # Right panel: Expected Rank + 95% CrI
        ax2.errorbar(
            exp_ranks,
            y_pos,
            xerr=[ci_low, ci_high],
            fmt="o",
            color="#0f172a",
            ecolor="#475569",
            elinewidth=2.0,
            capsize=4,
            markersize=7,
            zorder=4,
        )
        for idx, (rk, col) in enumerate(zip(exp_ranks, colors, strict=False)):
            ax2.scatter([rk], [idx], color=col, s=64, zorder=5, edgecolors="#0f172a", linewidths=0.8)
        ax2.set_xlabel("Expected Rank (95% CrI, lower is better)", fontsize=10, fontweight="bold")
        ax2.set_xlim(0.7, float(res.k) + 0.4)
        ax2.grid(axis="x", linestyle="--", alpha=0.35)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig.suptitle(
            f"Bayesian Plackett-Luce Posterior Ranking — {res.metric} (k={res.k}, N={res.n_blocks} blocks)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        return fig

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate interactive Plotly dual-panel Plackett-Luce figure."""
        res = self._resolve_result(df, **kwargs)
        palette_name = str(kwargs.get("palette", self.palette))
        algos = [it.algorithm for it in res.items]
        color_map = get_color_map(algos, palette_name=palette_name, custom_colors=kwargs.get("custom_colors"))
        colors = [color_map[a] for a in algos]

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            subplot_titles=("Posterior P(Best)", "Expected Rank (95% CrI)"),
            horizontal_spacing=0.08,
        )
        fig.add_trace(
            go.Bar(
                y=algos,
                x=[it.prob_best for it in res.items],
                orientation="h",
                marker={"color": colors, "line": {"color": "#1e293b", "width": 1}},
                text=[f"{it.prob_best * 100:.1f}%" for it in res.items],
                textposition="outside",
                name="P(Best)",
                hovertemplate="<b>%{y}</b><br>P(Best): %{x:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=algos,
                x=[it.expected_rank for it in res.items],
                mode="markers",
                marker={"size": 11, "color": colors, "line": {"color": "#0f172a", "width": 1.2}},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [it.rank_ci_upper - it.expected_rank for it in res.items],
                    "arrayminus": [it.expected_rank - it.rank_ci_lower for it in res.items],
                    "color": "#475569",
                    "thickness": 2,
                },
                name="Expected Rank",
                hovertemplate="<b>%{y}</b><br>E[Rank]: %{x:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="P(Best)", range=[0, 1.12], row=1, col=1)
        fig.update_xaxes(title_text="Expected Rank (lower = better)", range=[0.7, res.k + 0.4], row=1, col=2)
        fig.update_layout(
            title=f"Bayesian Plackett-Luce Ranking — {res.metric} (N={res.n_blocks} blocks)",
            showlegend=False,
            template="plotly_white",
            height=max(360, 56 * len(algos) + 120),
        )
        return fig
