"""Public API surface for SAES library."""

from typing import Any

import numpy as np

from saes.app.pipeline import AnalysisPipeline, AnalysisReport
from saes.domain.models import (
    AssumptionResult,
    BayesianCorrelatedTResult,
    BayesianPlackettLuceResult,
    BayesianRankResult,
    BlockedOmnibusResult,
    CriticalDistanceResult,
    DataExplorationResult,
    Dataset,
    EffectSizeForestResult,
    PerformanceProfilesResult,
    PostHocResult,
    PowerAnalysisResult,
    ReproducibilityResult,
    StatisticalResult,
    StatisticalTableResult,
    TOSTResult,
    ValidationReport,
)
from saes.reports.reproducibility import (
    generate_environment_manifest,
    generate_jupyter_notebook,
    generate_python_script,
    generate_reproducibility_artifacts,
)
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import (
    AndersonDarlingChecker,
    HomogeneityChecker,
    LillieforsChecker,
    NormalityChecker,
)
from saes.stats.bayesian import BayesianCorrelatedTTest, BayesianPlackettLuceTest
from saes.stats.effect_sizes import (
    cohens_d_ci,
    compute_cliffs_delta_with_ci,
    compute_effect_size_forest,
    compute_vargha_delaney_a12_with_ci,
    eta_squared_ci,
)
from saes.stats.explorer import DataExplorerEngine, export_exploration_table
from saes.stats.nonparametric import (
    KruskalWallisTest,
    MannWhitneyUTest,
    WilcoxonSignedRankTest,
    adjust_pvalues,
    compute_cliffs_delta,
    compute_vargha_delaney_a12,
)
from saes.stats.posthoc import (
    BrunnerMunzelTest,
    adjust_pvalues_bergmann_hommel,
    adjust_pvalues_shaffer,
    conover_iman_test,
    dunn_test,
    dunnett,
    games_howell,
    tukey_hsd,
)
from saes.stats.power import (
    analyze_statistical_power,
    calculate_statistical_power,
    compute_minimum_detectable_effect,
    estimate_required_sample_size,
    generate_power_curves,
    tost_equivalence_test,
)
from saes.stats.profiles import compute_performance_profiles
from saes.stats.ranking import (
    BayesianRankingTest,
    FriedmanAlignedRanksTest,
    FriedmanNemenyiTest,
    QuadeTest,
    RepeatedMeasuresANOVA,
    _extract_block_matrix,
)
from saes.stats.tables import StatisticalTableGenerator, export_statistical_table
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def load(source: Any) -> Dataset:
    """Load a CSV dataset into a SAES Dataset object.

    Args:
        source: Filepath or file-like buffer.

    Returns:
        Dataset entity.
    """
    pipeline = AnalysisPipeline()
    return pipeline.load_csv(source)


def validate(dataset: Dataset) -> ValidationReport:
    """Validate dataset structure, schema, and types.

    Args:
        dataset: Dataset instance.

    Returns:
        ValidationReport object.
    """
    pipeline = AnalysisPipeline()
    return pipeline.validate(dataset)


def analyze(
    dataset: Dataset,
    alpha: float = 0.05,
    generate_plots: bool = True,
    paired: bool = False,
) -> AnalysisReport:
    """Run comprehensive automated statistical analysis across dataset.

    Args:
        dataset: Validated Dataset instance.
        alpha: Significance level (default 0.05).
        generate_plots: Whether to generate distribution boxplots (default True).
        paired: Whether per-instance observations are explicitly statistically paired
            (e.g., identical CV folds or Common Random Numbers). Defaults to False.

    Returns:
        AnalysisReport: Comprehensive report of all statistical tests and plots.
    """
    pipeline = AnalysisPipeline(alpha=alpha, paired=paired)
    return pipeline.analyze(dataset, generate_plots=generate_plots, paired=paired)


def plot(
    dataset: Dataset,
    plot_type: str = "boxplot",
    metric: str | None = None,
    instance: str | None = None,
    interactive: bool = False,
    **kwargs: Any,
) -> Any:
    """Generate a publication-grade figure directly from a SAES Dataset.

    Supported plot_type values:
        - 'boxplot': Quartile box plot with centered observations
        - 'violin': Kernel density + internal box plot
        - 'critical-distance': Demšar (2006) Friedman + Nemenyi CD diagram
        - 'bayesian-rank': Dirichlet Process posterior rank distribution & ROPE
        - 'barplot', 'ecdf', 'lineplot', 'scatter'

    Args:
        dataset: SAES Dataset object.
        plot_type: Key registered in PlotterRegistry.
        metric: Metric identifier (defaults to first metric in dataset).
        instance: Problem instance identifier (defaults to first instance for cell plots).
        interactive: If True, returns a Plotly Figure; otherwise returns a Matplotlib Figure.
        **kwargs: Additional styling parameters (paper_format='golden'|'single-col'|..., palette, alpha, etc.).

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure.
    """
    from saes.visualization.registry import PlotterRegistry

    target_metric = metric or (dataset.metrics[0] if dataset.metrics else "value")
    resolved_plot_type = plot_type.lower()
    if resolved_plot_type in {"bayesian-sign", "bayesian-signed-rank"}:
        kwargs = {
            **kwargs,
            "method": "sign" if resolved_plot_type == "bayesian-sign" else "signed-rank",
        }
        plotter = PlotterRegistry.get("bayesian-rank")
    else:
        plotter = PlotterRegistry.get(plot_type)

    is_ranking = resolved_plot_type in {
        "critical-distance",
        "bayesian-rank",
        "bayesian-sign",
        "bayesian-signed-rank",
        "plackett-luce",
        "effect-size-forest",
    }
    default_scope = "single-instance" if resolved_plot_type == "effect-size-forest" else "all-instances"
    scope = str(kwargs.get("scope", default_scope))

    if is_ranking and scope == "all-instances":
        df_sub = dataset.df[dataset.df["metric"] == target_metric].copy()
        target_inst = instance or ""
    else:
        target_inst = instance or (dataset.instances[0] if dataset.instances else "")
        df_sub = dataset.get_cell(target_inst, target_metric) if target_inst else dataset.df.copy()

    plot_kwargs = {
        "metric": target_metric,
        "instance": target_inst,
        **kwargs,
    }
    if interactive:
        return plotter.plot_interactive(df_sub, **plot_kwargs)
    return plotter.plot(df_sub, **plot_kwargs)


def export_plot(
    dataset: Dataset,
    plot_type: str = "boxplot",
    metric: str | None = None,
    instance: str | None = None,
    export_format: str = "png",
    dpi: int = 300,
    **kwargs: Any,
) -> tuple[bytes, str, str]:
    """Generate and export a publication-grade plot as PNG, SVG, or PDF bytes.

    Args:
        dataset: SAES Dataset object.
        plot_type: Key registered in PlotterRegistry.
        metric: Metric identifier (defaults to first metric in dataset).
        instance: Problem instance identifier (defaults to first instance for cell plots).
        export_format: Target format ('png', 'svg', or 'pdf').
        dpi: Resolution in dots per inch for raster formats (default 300).
        **kwargs: Additional plotting parameters passed to `plot()`.

    Returns:
        Tuple of (file_bytes, media_type, extension).
    """
    import io

    import matplotlib.pyplot as plt

    fmt = export_format.strip().lower()
    format_map: dict[str, tuple[str, str]] = {
        "png": ("image/png", "png"),
        "svg": ("image/svg+xml", "svg"),
        "pdf": ("application/pdf", "pdf"),
    }
    if fmt not in format_map:
        raise ValueError(f"Unsupported plot export format '{export_format}'. Supported: png, svg, pdf.")

    fig = plot(
        dataset,
        plot_type=plot_type,
        metric=metric,
        instance=instance,
        interactive=False,
        **kwargs,
    )
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)

    media_type, ext = format_map[fmt]
    return buf.getvalue(), media_type, ext


def generate_table(
    dataset: Dataset,
    metric: str | None = None,
    table_type: str = "instance_algorithm",
    test_type: str = "auto",
    posthoc_correction: str = "holm",
    summary_stat: str = "auto",
    reference_algorithm: str | None = None,
    scope: str = "all-instances",
    selected_instance: str | None = None,
    algorithms_filter: list[str] | None = None,
    instances_filter: list[str] | None = None,
    alpha: float = 0.05,
    precision: int = 4,
    latex_document_mode: str = "standalone",
    latex_notation: str = "scientific",
    paired: bool = False,
) -> StatisticalTableResult:
    """Generate a publication-grade ANOVA, Friedman, Wilcoxon, or Pairwise statistical benchmark table."""
    return StatisticalTableGenerator().generate(
        dataset=dataset,
        metric=metric,
        table_type=table_type,
        test_type=test_type,
        posthoc_correction=posthoc_correction,
        summary_stat=summary_stat,
        reference_algorithm=reference_algorithm,
        scope=scope,
        selected_instance=selected_instance,
        algorithms_filter=algorithms_filter,
        instances_filter=instances_filter,
        alpha=alpha,
        precision=precision,
        latex_document_mode=latex_document_mode,
        latex_notation=latex_notation,
        paired=paired,
    )


def export_table(
    table_result: StatisticalTableResult,
    export_format: str = "latex",
    document_mode: str | None = None,
    notation: str | None = None,
    title: str | None = None,
    caption: str | None = None,
    pdf_theme: str = "slate",
    pdf_orientation: str = "landscape",
    pdf_font_size: float = 8.5,
    pdf_highlight_best: bool = True,
    pdf_show_zebra: bool = True,
    pdf_grid_style: str = "full",
) -> bytes:
    """Export a generated StatisticalTableResult to LaTeX, CSV, Markdown, HTML, DOCX, or PDF bytes."""
    file_bytes, _, _ = export_statistical_table(
        table_result,
        export_format=export_format,
        document_mode=document_mode,
        notation=notation,
        title=title,
        caption=caption,
        pdf_theme=pdf_theme,
        pdf_orientation=pdf_orientation,
        pdf_font_size=pdf_font_size,
        pdf_highlight_best=pdf_highlight_best,
        pdf_show_zebra=pdf_show_zebra,
        pdf_grid_style=pdf_grid_style,
    )
    return file_bytes


def explore(
    dataset: Dataset,
    metrics: list[str] | None = None,
    algorithms: list[str] | None = None,
    instances: list[str] | None = None,
    alpha: float = 0.05,
) -> DataExplorationResult:
    """Explore dataset observations, reveal multi-metric tables, check normality, and get test recommendations."""
    return DataExplorerEngine().explore(
        dataset=dataset,
        metrics=metrics,
        algorithms=algorithms,
        instances=instances,
        alpha=alpha,
    )


def export_exploration(
    exploration: DataExplorationResult,
    view_mode: str = "reveal_metrics",
    export_format: str = "csv",
    stat_format: str = "mean_std",
    normality_filter: str = "all",
    output_path: str | None = None,
) -> bytes:
    """Export an Observations & Data exploration result into PDF, Word (.docx), LaTeX, CSV, Markdown, or HTML."""
    file_bytes, _, _ = export_exploration_table(
        exploration=exploration,
        view_mode=view_mode,
        export_format=export_format,
        stat_format=stat_format,
        normality_filter=normality_filter,
    )
    if output_path:
        with open(output_path, "wb") as f:
            f.write(file_bytes)
    return file_bytes


def profiles(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    max_tau: float | None = None,
    num_tau_points: int = 60,
    paired: bool = False,
    posthoc_correction: str = "holm",
) -> PerformanceProfilesResult:
    """Compute Dolan-Moré performance profiles and global Win/Tie/Loss meta-ranking."""
    return compute_performance_profiles(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        max_tau=max_tau,
        num_tau_points=num_tau_points,
        paired=paired,
        posthoc_correction=posthoc_correction,
    )


def power(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    target_effect_size: float = 0.50,
    paired: bool = False,
) -> PowerAnalysisResult:
    """Run full a priori power analysis, sensitivity (MDES) analysis, and design audit."""
    return analyze_statistical_power(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        target_power=target_power,
        target_effect_size=target_effect_size,
        paired=paired,
    )


def tost(
    x1: np.ndarray,
    x2: np.ndarray,
    low_eqbound: float,
    high_eqbound: float | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    equal_var: bool = False,
    group_1: str = "Algorithm A",
    group_2: str = "Algorithm B",
) -> TOSTResult:
    """Run Two One-Sided Tests (TOST) for statistical equivalence (Schuirmann 1987; Lakens 2017)."""
    return tost_equivalence_test(
        x1=x1,
        x2=x2,
        low_eqbound=low_eqbound,
        high_eqbound=high_eqbound,
        alpha=alpha,
        paired=paired,
        equal_var=equal_var,
        group_1=group_1,
        group_2=group_2,
    )


def reproducibility(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    filename: str = "dataset.csv",
    raw_csv: str | None = None,
) -> ReproducibilityResult:
    """Generate standalone reproducibility code artifacts (Python script, Jupyter Notebook, manifest)."""
    return generate_reproducibility_artifacts(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        filename=filename,
        raw_csv=raw_csv,
    )


# Low-level programmatic statistics sub-namespace
class StatisticsNamespace:
    """Programmatic statistical test API namespace (saes.statistics.*)."""

    @staticmethod
    def ttest_ind(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Student's two-sample independent t-test."""
        return IndependentTTest().run(groups, alpha=alpha)

    @staticmethod
    def ttest_welch(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Welch's two-sample independent t-test (unequal variances)."""
        return WelchTTest().run(groups, alpha=alpha)

    @staticmethod
    def ttest_rel(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Paired samples t-test."""
        return PairedTTest().run(groups, alpha=alpha)

    @staticmethod
    def wilcoxon(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Wilcoxon Signed-Rank paired non-parametric test."""
        return WilcoxonSignedRankTest().run(groups, alpha=alpha)

    @staticmethod
    def mann_whitney(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Mann-Whitney U (Wilcoxon Rank-Sum) independent non-parametric test."""
        return MannWhitneyUTest().run(groups, alpha=alpha)

    @staticmethod
    def brunner_munzel(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Brunner-Munzel (2000) non-parametric Behrens-Fisher test for stochastic equality."""
        return BrunnerMunzelTest().run(groups, alpha=alpha)

    @staticmethod
    def kruskal(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Kruskal-Wallis H multi-group non-parametric omnibus test."""
        return KruskalWallisTest().run(groups, alpha=alpha)

    @staticmethod
    def anova_oneway(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Fisher One-Way ANOVA."""
        return OneWayANOVA().run(groups, alpha=alpha)

    @staticmethod
    def anova_welch(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Welch's One-Way ANOVA."""
        return WelchANOVA().run(groups, alpha=alpha)

    @staticmethod
    def vargha_delaney_a12(x1: np.ndarray, x2: np.ndarray, maximize: bool = True) -> tuple[float, str]:
        """Compute Vargha-Delaney A12 non-parametric effect size and magnitude."""
        return compute_vargha_delaney_a12(x1, x2, maximize=maximize)

    @staticmethod
    def vargha_delaney_a12_ci(
        x1: np.ndarray,
        x2: np.ndarray,
        maximize: bool = True,
        alpha: float = 0.05,
    ) -> tuple[float, str, tuple[float, float]]:
        """Compute Vargha-Delaney A12 with Brunner-Munzel asymptotic confidence interval."""
        return compute_vargha_delaney_a12_with_ci(x1, x2, maximize=maximize, alpha=alpha)

    @staticmethod
    def cliffs_delta(x1: np.ndarray, x2: np.ndarray, maximize: bool = True) -> tuple[float, str]:
        """Compute Cliff's delta non-parametric effect size and magnitude."""
        return compute_cliffs_delta(x1, x2, maximize=maximize)

    @staticmethod
    def cliffs_delta_ci(
        x1: np.ndarray,
        x2: np.ndarray,
        maximize: bool = True,
        alpha: float = 0.05,
    ) -> tuple[float, str, tuple[float, float]]:
        """Compute Cliff's delta with Brunner-Munzel range-preserving confidence interval."""
        return compute_cliffs_delta_with_ci(x1, x2, maximize=maximize, alpha=alpha)

    @staticmethod
    def cohens_d_ci(
        d: float,
        n1: int,
        n2: int | None = None,
        alpha: float = 0.05,
        paired: bool = False,
        hedges_correction: bool = False,
    ) -> tuple[float, float]:
        """Compute exact noncentral-t confidence interval for Cohen's d or Hedges' g."""
        return cohens_d_ci(
            d=d,
            n1=n1,
            n2=n2,
            alpha=alpha,
            paired=paired,
            hedges_correction=hedges_correction,
        )

    @staticmethod
    def eta_squared_ci(
        f_stat: float,
        df_between: float,
        df_within: float,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute exact noncentral-F confidence interval for ANOVA Eta-squared."""
        return eta_squared_ci(f_stat=f_stat, df_between=df_between, df_within=df_within, alpha=alpha)

    @staticmethod
    def effect_size_forest(
        dataset: Dataset,
        metric: str,
        scope: str = "single-instance",
        instance: str | None = None,
        reference_algorithm: str | None = None,
        alpha: float = 0.05,
        paired: bool = False,
        p_adjust: str = "holm",
    ) -> EffectSizeForestResult:
        """Compute pairwise standardized effect sizes (d, g, A12, Cliff's delta) with 95% CIs vs reference."""
        res = compute_effect_size_forest(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            reference_algorithm=reference_algorithm,
            alpha=alpha,
            paired=paired,
            p_adjust=p_adjust,
        )
        assert isinstance(res, EffectSizeForestResult)
        return res

    @staticmethod
    def adjust_pvalues(p_values: list[float], method: str = "holm") -> list[float]:
        """Adjust p-values for multiple comparisons (Holm, Bonferroni, Hochberg, Hommel, Finner, Rom, Li, FDR)."""
        return adjust_pvalues(p_values, method=method)

    @staticmethod
    def adjust_pvalues_shaffer(p_matrix: np.ndarray) -> np.ndarray:
        """Apply Shaffer's (1986) static multiple-comparison adjustment to a symmetric k x k p-value matrix."""
        return adjust_pvalues_shaffer(p_matrix)

    @staticmethod
    def adjust_pvalues_bergmann_hommel(
        p_matrix: np.ndarray,
        max_exact_k: int = 9,
    ) -> tuple[np.ndarray, list[str]]:
        """Apply exact Bergmann-Hommel (1988) exhaustive adjustment to a symmetric k x k p-value matrix."""
        return adjust_pvalues_bergmann_hommel(p_matrix, max_exact_k=max_exact_k)

    @staticmethod
    def tukey_hsd(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
        """Tukey's Honestly Significant Difference (HSD) parametric all-pairs post-hoc test."""
        return tukey_hsd(groups, alpha=alpha)

    @staticmethod
    def games_howell(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
        """Games-Howell heteroscedastic all-pairs post-hoc test (after Welch's ANOVA)."""
        return games_howell(groups, alpha=alpha)

    @staticmethod
    def dunnett(groups: dict[str, np.ndarray], control: str, alpha: float = 0.05) -> PostHocResult:
        """Dunnett's many-to-one post-hoc comparison against a control group."""
        return dunnett(groups, control=control, alpha=alpha)

    @staticmethod
    def dunn(groups: dict[str, np.ndarray], alpha: float = 0.05, p_adjust: str = "holm") -> PostHocResult:
        """Dunn's (1964) non-parametric all-pairs post-hoc test (after Kruskal-Wallis)."""
        return dunn_test(groups, alpha=alpha, p_adjust=p_adjust)

    @staticmethod
    def conover_iman(groups: dict[str, np.ndarray], alpha: float = 0.05, p_adjust: str = "holm") -> PostHocResult:
        """Conover-Iman (1979) non-parametric all-pairs post-hoc test (after Kruskal-Wallis)."""
        return conover_iman_test(groups, alpha=alpha, p_adjust=p_adjust)

    @staticmethod
    def table(
        dataset: Dataset,
        metric: str | None = None,
        table_type: str = "instance_algorithm",
        test_type: str = "auto",
        posthoc_correction: str = "holm",
        summary_stat: str = "auto",
        reference_algorithm: str | None = None,
        scope: str = "all-instances",
        selected_instance: str | None = None,
        algorithms_filter: list[str] | None = None,
        instances_filter: list[str] | None = None,
        alpha: float = 0.05,
        precision: int = 4,
        paired: bool = False,
    ) -> StatisticalTableResult:
        """Generate a statistical benchmark table (instance_algorithm, pairwise_matrix, omnibus_posthoc)."""
        return generate_table(
            dataset=dataset,
            metric=metric,
            table_type=table_type,
            test_type=test_type,
            posthoc_correction=posthoc_correction,
            summary_stat=summary_stat,
            reference_algorithm=reference_algorithm,
            scope=scope,
            selected_instance=selected_instance,
            algorithms_filter=algorithms_filter,
            instances_filter=instances_filter,
            alpha=alpha,
            precision=precision,
            paired=paired,
        )

    @staticmethod
    def explore(
        dataset: Dataset,
        metrics: list[str] | None = None,
        algorithms: list[str] | None = None,
        instances: list[str] | None = None,
        alpha: float = 0.05,
    ) -> DataExplorationResult:
        """Explore dataset distributions, reveal multi-metric pivots, and recommend statistical tests."""
        return explore(
            dataset=dataset,
            metrics=metrics,
            algorithms=algorithms,
            instances=instances,
            alpha=alpha,
        )

    @staticmethod
    def shapiro(groups: dict[str, np.ndarray], alpha: float = 0.05) -> list[AssumptionResult]:
        """Check normality assumption via Shapiro-Wilk test."""
        return NormalityChecker(alpha=alpha).check(groups)

    @staticmethod
    def lilliefors(groups: dict[str, np.ndarray], alpha: float = 0.05) -> list[AssumptionResult]:
        """Check normality assumption via Lilliefors-corrected Kolmogorov-Smirnov test."""
        return LillieforsChecker(alpha=alpha).check(groups)

    @staticmethod
    def anderson(groups: dict[str, np.ndarray], alpha: float = 0.05) -> list[AssumptionResult]:
        """Check normality assumption via Anderson-Darling test (Stephens 1974)."""
        return AndersonDarlingChecker(alpha=alpha).check(groups)

    @staticmethod
    def levene(groups: dict[str, np.ndarray], alpha: float = 0.05) -> AssumptionResult:
        """Check homogeneity of variance assumption via Levene's test."""
        return HomogeneityChecker(alpha=alpha).check(groups)

    @staticmethod
    def friedman_nemenyi(
        dataset: Dataset,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> CriticalDistanceResult:
        """Friedman non-parametric rank test with Nemenyi post-hoc Critical Distance (CD)."""
        return FriedmanNemenyiTest().run(df=dataset.df, metric=metric, alpha=alpha, scope=scope, instance=instance)

    @staticmethod
    def friedman_aligned_ranks(
        dataset: Dataset,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> BlockedOmnibusResult:
        """Friedman Aligned-Ranks omnibus test across benchmark blocks (Hodges & Lehmann 1962; García et al. 2010)."""
        return FriedmanAlignedRanksTest().run(
            df=dataset.df,
            metric=metric,
            alpha=alpha,
            scope=scope,
            instance=instance,
        )

    @staticmethod
    def quade(
        dataset: Dataset,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> BlockedOmnibusResult:
        """Quade weighted-block non-parametric omnibus test (Quade 1979; García et al. 2010)."""
        return QuadeTest().run(df=dataset.df, metric=metric, alpha=alpha, scope=scope, instance=instance)

    @staticmethod
    def rm_anova(
        dataset: Dataset,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
        greenhouse_geisser: bool = True,
    ) -> BlockedOmnibusResult:
        """One-way Repeated-Measures ANOVA across matched blocks with Greenhouse-Geisser sphericity correction."""
        return RepeatedMeasuresANOVA().run(
            df=dataset.df,
            metric=metric,
            alpha=alpha,
            scope=scope,
            instance=instance,
            greenhouse_geisser=greenhouse_geisser,
        )

    @staticmethod
    def bayesian_ranking(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        rope: float | None = None,
        prior_strength: float = 1.0,
        num_samples: int = 1500,
        algo_a: str | None = None,
        algo_b: str | None = None,
        method: str = "signed-rank",
    ) -> BayesianRankResult:
        """Bayesian Signed-Rank / Sign-Test Simplex (ROPE) and Posterior Ranking test (Benavoli et al. 2017)."""
        return BayesianRankingTest().run(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            rope=rope,
            prior_strength=prior_strength,
            num_samples=num_samples,
            algo_a=algo_a,
            algo_b=algo_b,
            method=method,
        )

    @staticmethod
    def bayesian_sign(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        rope: float | None = None,
        prior_strength: float = 1.0,
        num_samples: int = 1500,
        algo_a: str | None = None,
        algo_b: str | None = None,
    ) -> BayesianRankResult:
        """Bayesian Sign Test Simplex (ROPE) (Benavoli et al. 2017)."""
        return BayesianRankingTest().run(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            rope=rope,
            prior_strength=prior_strength,
            num_samples=num_samples,
            algo_a=algo_a,
            algo_b=algo_b,
            method="sign",
        )

    @staticmethod
    def bayesian_signed_rank(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        rope: float | None = None,
        prior_strength: float = 1.0,
        num_samples: int = 1500,
        algo_a: str | None = None,
        algo_b: str | None = None,
    ) -> BayesianRankResult:
        """Bayesian Signed-Rank Test Simplex (ROPE) (Benavoli et al. 2017)."""
        return BayesianRankingTest().run(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            rope=rope,
            prior_strength=prior_strength,
            num_samples=num_samples,
            algo_a=algo_a,
            algo_b=algo_b,
            method="signed-rank",
        )

    @staticmethod
    def bayesian_correlated_ttest(
        x_a: np.ndarray,
        x_b: np.ndarray,
        algo_a: str = "Algorithm A",
        algo_b: str = "Algorithm B",
        rope: float = 0.01,
        rho: float | None = None,
        maximize: bool = True,
    ) -> BayesianCorrelatedTResult:
        """Bayesian Correlated t-test with ROPE (Nadeau & Bengio 2003; Benavoli et al. 2017)."""
        return BayesianCorrelatedTTest().run(
            x_a=x_a,
            x_b=x_b,
            algo_a=algo_a,
            algo_b=algo_b,
            rope=rope,
            rho=rho,
            maximize=maximize,
        )

    @staticmethod
    def bayesian_plackett_luce(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        num_samples: int = 400,
        prior_alpha: float = 1.0,
        seed: int = 42,
    ) -> BayesianPlackettLuceResult:
        """Bayesian Plackett-Luce multi-algorithm ranking model (Calvo et al. 2018)."""
        algorithms, matrix, maximize, *_ = _extract_block_matrix(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
        )
        return BayesianPlackettLuceTest().run(
            matrix=matrix,
            algorithms=algorithms,
            metric=metric,
            maximize=maximize,
            num_samples=num_samples,
            prior_alpha=prior_alpha,
            seed=seed,
        )

    @staticmethod
    def minimum_detectable_effect(
        n: int,
        target_power: float = 0.80,
        alpha: float = 0.05,
        paired: bool = False,
    ) -> float:
        """Compute Minimum Detectable Effect Size (MDES, Cohen's d) for a fixed sample size n."""
        return compute_minimum_detectable_effect(n=n, target_power=target_power, alpha=alpha, paired=paired)

    @staticmethod
    def tost(
        x1: np.ndarray,
        x2: np.ndarray,
        low_eqbound: float,
        high_eqbound: float | None = None,
        alpha: float = 0.05,
        paired: bool = False,
        equal_var: bool = False,
        group_1: str = "Algorithm A",
        group_2: str = "Algorithm B",
    ) -> TOSTResult:
        """Two One-Sided Tests (TOST) procedure for statistical equivalence (Schuirmann 1987; Lakens 2017)."""
        return tost_equivalence_test(
            x1=x1,
            x2=x2,
            low_eqbound=low_eqbound,
            high_eqbound=high_eqbound,
            alpha=alpha,
            paired=paired,
            equal_var=equal_var,
            group_1=group_1,
            group_2=group_2,
        )


statistics = StatisticsNamespace()

__all__ = [
    "load",
    "validate",
    "analyze",
    "plot",
    "export_plot",
    "generate_table",
    "export_table",
    "explore",
    "export_exploration",
    "profiles",
    "power",
    "tost",
    "tost_equivalence_test",
    "reproducibility",
    "compute_performance_profiles",
    "analyze_statistical_power",
    "calculate_statistical_power",
    "compute_minimum_detectable_effect",
    "estimate_required_sample_size",
    "generate_power_curves",
    "generate_reproducibility_artifacts",
    "generate_python_script",
    "generate_jupyter_notebook",
    "generate_environment_manifest",
    "statistics",
]

