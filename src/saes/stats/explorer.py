"""Data exploration, multi-metric reveal pivoting, normality profiling, and test recommendations."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors

from saes.domain.models import (
    AggregatedMetricRow,
    ContextDiagnostics,
    DataExplorationResult,
    Dataset,
    DistributionProfile,
    PivotedObservationRow,
    TestRecommendation,
)
from saes.stats.assumptions import AndersonDarlingChecker, HomogeneityChecker
from saes.stats.decision import is_paired_data


def _fmt_num(val: float) -> str:
    if not math.isfinite(val):
        return "—"
    abs_v = abs(val)
    if abs_v == 0.0:
        return "0.0000"
    if abs_v < 1e-4 or abs_v >= 1e5:
        return f"{val:.3e}"
    return f"{val:.4f}"


class DataExplorerEngine:
    """Evaluates dataset slices, reveals multi-metric tables, checks distributions, and recommends statistical tests."""

    def explore(
        self,
        dataset: Dataset,
        metrics: list[str] | None = None,
        algorithms: list[str] | None = None,
        instances: list[str] | None = None,
        alpha: float = 0.05,
    ) -> DataExplorationResult:
        df = dataset.df.copy()
        available_metrics = sorted(df["metric"].unique().tolist()) if not df.empty else []
        available_algorithms = sorted(df["algorithm"].unique().tolist()) if not df.empty else []
        available_instances = sorted(df["instance"].unique().tolist()) if not df.empty else []

        metric_directions: dict[str, bool] = {}
        for m in available_metrics:
            m_sub = df[df["metric"] == m]
            if not m_sub.empty:
                metric_directions[m] = bool(m_sub["maximize"].iloc[0])

        active_metrics = [m for m in metrics if m in available_metrics] if metrics else list(available_metrics)
        if not active_metrics and available_metrics:
            active_metrics = list(available_metrics)

        active_algorithms = (
            [a for a in algorithms if a in available_algorithms] if algorithms else list(available_algorithms)
        )
        if not active_algorithms and available_algorithms:
            active_algorithms = list(available_algorithms)

        active_instances = (
            [i for i in instances if i in available_instances] if instances else list(available_instances)
        )
        if not active_instances and available_instances:
            active_instances = list(available_instances)

        sub_df = df[
            df["metric"].isin(active_metrics)
            & df["algorithm"].isin(active_algorithms)
            & df["instance"].isin(active_instances)
        ].copy()

        # 1. Build Distribution Profiles per (instance, metric, algorithm)
        profiles: list[DistributionProfile] = []
        total_outliers = 0
        normality_pass_count = 0

        for (inst, met, alg), grp in sub_df.groupby(["instance", "metric", "algorithm"]):
            arr = grp["value"].dropna().to_numpy(dtype=float)
            n = len(arr)
            if n == 0:
                continue

            max_dir = bool(grp["maximize"].iloc[0])
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            med_v = float(np.median(arr))
            q05, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
            iqr_v = float(q75 - q25)
            min_v = float(np.min(arr))
            max_v = float(np.max(arr))
            cv_v = float(abs(std_v / mean_v)) if abs(mean_v) > 1e-12 else 0.0

            # IQR outlier detection (1.5 * IQR rule)
            if iqr_v > 0:
                lower_fence = q25 - 1.5 * iqr_v
                upper_fence = q75 + 1.5 * iqr_v
                outliers = int(np.sum((arr < lower_fence) | (arr > upper_fence)))
            else:
                outliers = 0
            total_outliers += outliers

            is_constant = bool(n >= 1 and np.all(arr == arr[0]))
            skew_v = float(stats.skew(arr)) if (n >= 3 and not is_constant) else 0.0
            kurt_v = float(stats.kurtosis(arr)) if (n >= 4 and not is_constant) else 0.0

            # Shapiro-Wilk test
            if n < 3 or is_constant:
                shapiro_stat = 1.0
                shapiro_p = 1.0
                shapiro_ok = True
            else:
                sw_res = stats.shapiro(arr)
                shapiro_stat = float(sw_res.statistic)
                shapiro_p = float(sw_res.pvalue)
                shapiro_ok = bool(shapiro_p >= alpha)

            # D'Agostino-Pearson K^2 test (requires n >= 8)
            dag_stat: float | None = None
            dag_p: float | None = None
            dag_ok: bool | None = None
            if n >= 8 and not is_constant:
                try:
                    nt_res = stats.normaltest(arr)
                    dag_stat = float(nt_res.statistic)
                    dag_p = float(nt_res.pvalue)
                    dag_ok = bool(dag_p >= alpha)
                except Exception:
                    pass

            # Lilliefors-corrected Kolmogorov-Smirnov test for normality with estimated mean/SD
            if n < 4 or is_constant or std_v == 0:
                ks_stat = 0.0
                ks_p = 1.0
                ks_ok = True
            else:
                lf_stat, lf_p = sm_lilliefors(arr, dist="norm", pvalmethod="table")
                ks_stat = float(lf_stat)
                ks_p = float(lf_p)
                ks_ok = bool(ks_p >= alpha)

            # Anderson-Darling normality test (Stephens 1974 / SciPy interpolated p-value)
            ad_stat: float | None = None
            ad_p: float | None = None
            ad_ok: bool | None = None
            if n >= 4 and not is_constant:
                ad_check = AndersonDarlingChecker(alpha=alpha).check({str(alg): arr})
                if ad_check and math.isfinite(ad_check[0].statistic):
                    ad_stat = float(ad_check[0].statistic)
                    ad_p = float(ad_check[0].p_value)
                    ad_ok = bool(ad_check[0].passed)

            is_normal = shapiro_ok
            if is_normal:
                normality_pass_count += 1

            if is_constant:
                shape_label = "Constant (Zero Variance)"
            elif not is_normal:
                if skew_v > 0.8:
                    shape_label = "Right-Skewed (Non-Normal)"
                elif skew_v < -0.8:
                    shape_label = "Left-Skewed (Non-Normal)"
                elif kurt_v > 1.5:
                    shape_label = "Heavy-Tailed / Outliers (Non-Normal)"
                else:
                    shape_label = "Non-Gaussian / Multimodal"
            else:
                if abs(skew_v) > 0.5:
                    shape_label = "Approximately Normal (Mild Skew)"
                else:
                    shape_label = "Gaussian (Symmetric)"

            profiles.append(
                DistributionProfile(
                    algorithm=str(alg),
                    instance=str(inst),
                    metric=str(met),
                    maximize=max_dir,
                    n=n,
                    mean=mean_v,
                    std=std_v,
                    median=med_v,
                    iqr=iqr_v,
                    min_val=min_v,
                    max_val=max_v,
                    cv=cv_v,
                    skewness=skew_v,
                    kurtosis=kurt_v,
                    outlier_count=outliers,
                    shapiro_stat=shapiro_stat,
                    shapiro_p=shapiro_p,
                    shapiro_passed=shapiro_ok,
                    dagostino_stat=dag_stat,
                    dagostino_p=dag_p,
                    dagostino_passed=dag_ok,
                    ks_stat=ks_stat,
                    ks_p=ks_p,
                    ks_passed=ks_ok,
                    anderson_stat=ad_stat,
                    anderson_p=ad_p,
                    anderson_passed=ad_ok,
                    is_normal=is_normal,
                    distribution_shape=shape_label,
                    quantiles={
                        "q05": float(q05),
                        "q25": float(q25),
                        "q50": float(q50),
                        "q75": float(q75),
                        "q95": float(q95),
                    },
                    raw_values=[round(float(x), 6) for x in arr.tolist()],
                )
            )

        # 2. Build Pivoted Observation Rows (reveal all metrics side-by-side per run)
        pivoted_rows: list[PivotedObservationRow] = []
        if not sub_df.empty:
            for (alg, inst, run_id), grp in sub_df.groupby(["algorithm", "instance", "run_id"]):
                m_vals: dict[str, float] = {}
                for _, row in grp.iterrows():
                    val = float(row["value"])
                    if math.isfinite(val):
                        m_vals[str(row["metric"])] = val
                pivoted_rows.append(
                    PivotedObservationRow(
                        algorithm=str(alg),
                        instance=str(inst),
                        run_id=int(str(run_id)),
                        metrics_values=m_vals,
                    )
                )

        # 3. Build Aggregated Multi-Metric Rows per (instance, algorithm)
        aggregated_rows: list[AggregatedMetricRow] = []
        prof_lookup: dict[tuple[str, str, str], DistributionProfile] = {
            (p.instance, p.algorithm, p.metric): p for p in profiles
        }
        for inst in active_instances:
            for alg in active_algorithms:
                mean_std_map: dict[str, str] = {}
                med_iqr_map: dict[str, str] = {}
                norm_map: dict[str, bool] = {}
                shap_map: dict[str, float] = {}
                n_runs = 0
                for met in active_metrics:
                    p = prof_lookup.get((inst, alg, met))
                    if p:
                        n_runs = max(n_runs, p.n)
                        mean_std_map[met] = f"{_fmt_num(p.mean)} ± {_fmt_num(p.std)}"
                        med_iqr_map[met] = f"{_fmt_num(p.median)} ({_fmt_num(p.iqr)})"
                        norm_map[met] = p.is_normal
                        shap_map[met] = p.shapiro_p
                if n_runs > 0:
                    aggregated_rows.append(
                        AggregatedMetricRow(
                            algorithm=alg,
                            instance=inst,
                            n_runs=n_runs,
                            metrics_mean_std=mean_std_map,
                            metrics_median_iqr=med_iqr_map,
                            metrics_normal=norm_map,
                            metrics_shapiro_p=shap_map,
                        )
                    )

        # 4. Compute Context Diagnostics (Levene, Bartlett, Pairing, Balance)
        k_alg = len(active_algorithms)
        n_inst = len(active_instances)
        n_met = len(active_metrics)
        total_obs = len(sub_df)
        total_groups = len(profiles)
        norm_rate = float(normality_pass_count / total_groups) if total_groups > 0 else 1.0
        all_normal = normality_pass_count == total_groups and total_groups > 0

        counts = [p.n for p in profiles]
        runs_per_cell = int(round(float(np.median(counts)))) if counts else 0
        is_balanced = len(set(counts)) <= 1 if counts else True

        # Check pairing & homoscedasticity on primary active metric
        primary_metric = active_metrics[0] if active_metrics else ""
        primary_df = sub_df[sub_df["metric"] == primary_metric] if primary_metric else sub_df
        paired_flags: list[bool] = []
        levene_ps: list[float] = []
        levene_stats: list[float] = []
        bartlett_ps: list[float] = []
        bartlett_stats: list[float] = []

        for inst in active_instances:
            cell_df = primary_df[primary_df["instance"] == inst]
            if cell_df.empty:
                continue
            paired_flags.append(is_paired_data(cell_df))
            grp_dict: dict[str, np.ndarray] = {}
            for alg, g in cell_df.groupby("algorithm"):
                arr = g["value"].dropna().to_numpy(dtype=float)
                if len(arr) >= 2:
                    grp_dict[str(alg)] = arr
            if len(grp_dict) >= 2:
                lev = HomogeneityChecker(alpha=alpha).check(grp_dict)
                if math.isfinite(lev.p_value):
                    levene_ps.append(lev.p_value)
                    levene_stats.append(lev.statistic if math.isfinite(lev.statistic) else 0.0)
                try:
                    stds = [np.std(v, ddof=1) for v in grp_dict.values()]
                    if all(s > 0 for s in stds):
                        b_stat, b_p = stats.bartlett(*grp_dict.values())
                        if math.isfinite(b_p):
                            bartlett_ps.append(float(b_p))
                            bartlett_stats.append(float(b_stat))
                except Exception:
                    pass

        is_paired = all(paired_flags) if paired_flags else False
        min_levene_p = float(min(levene_ps)) if levene_ps else 1.0
        mean_levene_stat = float(np.mean(levene_stats)) if levene_stats else 0.0
        homo_passed = min_levene_p >= alpha
        min_bartlett_p = float(min(bartlett_ps)) if bartlett_ps else 1.0
        mean_bartlett_stat = float(np.mean(bartlett_stats)) if bartlett_stats else 0.0

        diagnostics = ContextDiagnostics(
            k_algorithms=k_alg,
            n_instances=n_inst,
            n_metrics=n_met,
            total_observations=total_obs,
            runs_per_cell=runs_per_cell,
            is_paired=is_paired,
            is_balanced=is_balanced,
            normality_pass_count=normality_pass_count,
            normality_total_groups=total_groups,
            normality_rate=norm_rate,
            all_normal=all_normal,
            homoscedasticity_levene_stat=mean_levene_stat,
            homoscedasticity_levene_p=min_levene_p,
            homoscedasticity_passed=homo_passed,
            homoscedasticity_bartlett_stat=mean_bartlett_stat,
            homoscedasticity_bartlett_p=min_bartlett_p,
            total_outliers=total_outliers,
            can_run_cell_tests=(runs_per_cell >= 2),
            can_run_blocked_omnibus=(k_alg >= 2 and n_inst >= 2),
        )

        recommendations = self._build_recommendations(diagnostics=diagnostics, alpha=alpha)

        return DataExplorationResult(
            available_metrics=available_metrics,
            available_algorithms=available_algorithms,
            available_instances=available_instances,
            metric_directions=metric_directions,
            active_metrics=active_metrics,
            active_algorithms=active_algorithms,
            active_instances=active_instances,
            alpha=alpha,
            context_diagnostics=diagnostics,
            recommendations=recommendations,
            distribution_profiles=profiles,
            pivoted_rows=pivoted_rows,
            aggregated_rows=aggregated_rows,
        )

    def _build_recommendations(
        self,
        diagnostics: ContextDiagnostics,
        alpha: float,
    ) -> list[TestRecommendation]:
        """Generate prioritized, dataset-aware recommendations covering frequentist, post-hoc, Bayesian, and TOST."""
        k = diagnostics.k_algorithms
        n_inst = diagnostics.n_instances
        runs = diagnostics.runs_per_cell
        all_normal = diagnostics.all_normal
        norm_pct = round(diagnostics.normality_rate * 100, 1)
        homo_ok = diagnostics.homoscedasticity_passed
        is_paired = diagnostics.is_paired
        outliers = diagnostics.total_outliers
        has_replicates = runs >= 2

        recs: list[TestRecommendation] = []

        # ── 1. Multi-instance blocked omnibus & Bayesian multi-algorithm ranking ──
        if k >= 3 and n_inst >= 2:
            recs.append(
                TestRecommendation(
                    test_id="friedman",
                    test_name="Friedman Omnibus Test + Iman-Davenport F & Holm / Nemenyi Post-Hoc",
                    category="Non-Parametric",
                    suitability="recommended",
                    suitability_score=98,
                    rationale=(
                        f"Comparing k={k} algorithms across N={n_inst} matched benchmark instances "
                        f"(normality pass rate: {norm_pct}%, {outliers} outlier(s)). "
                        "Tie-corrected Friedman rank test with Iman-Davenport F is the multi-problem standard."
                    ),
                    when_to_use="Comparing >= 3 algorithms across multiple problem instances (blocked design).",
                    assumptions_satisfied=[
                        f"k = {k} algorithms (>= 3 required)",
                        f"N = {n_inst} matched problem instance blocks",
                        "Distribution-free & exact tie-corrected",
                    ],
                    assumptions_violated=(
                        [] if n_inst >= 5 else [f"Modest number of blocks (N={n_inst}); check Iman-Davenport F"]
                    ),
                    suggested_table_type="omnibus_posthoc",
                    target_tab="stats",
                    target_mode="omnibus_posthoc",
                    posthoc_suggestion="Holm / Finner / Shaffer / Bergmann-Hommel or Nemenyi CD",
                    effect_size_suggestion="Kendall's W Concordance & Vargha-Delaney A12 (95% CI)",
                    reference_citation="Demšar (2006), JMLR; García & Herrera (2008)",
                )
            )

            aligned_score = 95 if k <= 5 else 87
            recs.append(
                TestRecommendation(
                    test_id="friedman_aligned",
                    test_name="Friedman Aligned-Ranks Test (Hodges-Lehmann)",
                    category="Non-Parametric",
                    suitability="recommended" if k <= 5 else "suitable",
                    suitability_score=aligned_score,
                    rationale=(
                        f"Centers each of the N={n_inst} blocks by its location before ranking all "
                        f"N×k={n_inst * k} values jointly. Offers higher power than Friedman when k ≤ 5."
                    ),
                    when_to_use="Multi-instance blocked comparisons with a moderate number of algorithms (k ≤ 5).",
                    assumptions_satisfied=[
                        f"k = {k} algorithms across N = {n_inst} blocks",
                        "Preserves inter-block magnitude information via alignment",
                    ],
                    assumptions_violated=(
                        [] if k <= 6 else [f"With k={k} > 6, standard Friedman ranks are typically preferred"]
                    ),
                    suggested_table_type="omnibus_posthoc",
                    target_tab="stats",
                    target_mode="omnibus_posthoc",
                    posthoc_suggestion="Holm / Bergmann-Hommel post-hoc comparisons",
                    effect_size_suggestion="Aligned mean rank differences",
                    reference_citation="Hodges & Lehmann (1962); García et al. (2010), Inf. Sci.",
                )
            )

            quade_score = 93 if k <= 5 else 84
            recs.append(
                TestRecommendation(
                    test_id="quade",
                    test_name="Quade Weighted-Block Non-Parametric Test",
                    category="Non-Parametric",
                    suitability="recommended" if k <= 5 else "suitable",
                    suitability_score=quade_score,
                    rationale=(
                        f"Weights within-block ranks by the sample range of each instance across N={n_inst} blocks, "
                        "giving greater influence to benchmark instances with larger performance spread."
                    ),
                    when_to_use="Blocked multi-instance benchmarks where instances differ substantially in difficulty.",
                    assumptions_satisfied=[
                        f"N = {n_inst} blocks with range-weighted ranking",
                        "Non-parametric (no Gaussian residual requirement)",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="omnibus_posthoc",
                    target_tab="stats",
                    target_mode="omnibus_posthoc",
                    posthoc_suggestion="Holm / Shaffer post-hoc adjustment",
                    effect_size_suggestion="Weighted average ranks",
                    reference_citation="Quade (1979), JASA; García et al. (2010)",
                )
            )

            rm_score = 92 if all_normal else 64
            recs.append(
                TestRecommendation(
                    test_id="rm_anova",
                    test_name="Repeated-Measures ANOVA (Greenhouse-Geisser ε)",
                    category="Parametric",
                    suitability="recommended" if all_normal else "caution",
                    suitability_score=rm_score,
                    rationale=(
                        f"Parametric blocked omnibus F-test across N={n_inst} matched instances with automatic "
                        "Greenhouse-Geisser sphericity adjustment. "
                        + (
                            "Normality assumptions are satisfied across groups."
                            if all_normal
                            else f"Caution: only {norm_pct}% of groups pass normality; prefer Friedman/Quade."
                        )
                    ),
                    when_to_use="Comparing k >= 3 algorithms across matched instances when block values are Gaussian.",
                    assumptions_satisfied=(
                        ["Matched blocked design", "Greenhouse-Geisser sphericity protection"]
                        + (["Normality satisfied"] if all_normal else [])
                    ),
                    assumptions_violated=[] if all_normal else [f"Normality violated ({norm_pct}% pass rate)"],
                    suggested_table_type="omnibus_posthoc",
                    target_tab="stats",
                    target_mode="omnibus_posthoc",
                    posthoc_suggestion="Paired t-tests with Holm adjustment",
                    effect_size_suggestion="Greenhouse-Geisser ε & Partial η²",
                    reference_citation="Greenhouse & Geisser (1959), Psychometrika",
                )
            )

            recs.append(
                TestRecommendation(
                    test_id="bayesian_plackett_luce",
                    test_name="Bayesian Plackett-Luce Multi-Algorithm Ranking",
                    category="Bayesian",
                    suitability="recommended",
                    suitability_score=94,
                    rationale=(
                        f"Jointly models complete rankings of all k={k} algorithms across N={n_inst} blocks "
                        "for coherent posterior P(best) and 95% expected-rank credible intervals."
                    ),
                    when_to_use="Ranking >= 3 algorithms simultaneously with direct Bayesian winning probabilities.",
                    assumptions_satisfied=[
                        f"k = {k} algorithms across N = {n_inst} blocks",
                        "Probabilities sum strictly to 1.0 (coherent joint ranking)",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="omnibus_posthoc",
                    target_tab="charts",
                    target_mode="plackett-luce",
                    posthoc_suggestion="95% Credible Intervals on expected ranks",
                    effect_size_suggestion="Posterior worth w_j & P(Best)",
                    reference_citation="Calvo et al. (2018), GECCO; Hunter (2004)",
                )
            )

        # ── 2. Pairwise & cell-level comparisons ──
        if k >= 2:
            if has_replicates:
                bm_score = 95 if (not all_normal and not homo_ok) else (91 if not all_normal else 83)
                recs.append(
                    TestRecommendation(
                        test_id="brunner_munzel",
                        test_name="Brunner-Munzel Test (Non-Parametric Behrens-Fisher)",
                        category="Non-Parametric",
                        suitability="recommended" if (not all_normal and not homo_ok) else "suitable",
                        suitability_score=bm_score,
                        rationale=(
                            "Tests true stochastic equality P(X < Y) + 0.5 P(X = Y) = 0.5 without assuming equal "
                            f"shapes or variances (Levene min p = {diagnostics.homoscedasticity_levene_p:.4f}). "
                            "Reports exact Brunner-Munzel logit 95% CI for Vargha-Delaney A12."
                        ),
                        when_to_use="Comparing two independent stochastic algorithms with unequal variances/shapes.",
                        assumptions_satisfied=[
                            f"Independent runs per cell (n ≈ {runs})",
                            "Robust to both non-normality and heteroscedasticity",
                        ],
                        assumptions_violated=[],
                        suggested_table_type="instance_algorithm",
                        target_tab="stats",
                        target_mode="instance_algorithm",
                        posthoc_suggestion=f"Holm / Finner adjustment (α = {alpha})",
                        effect_size_suggestion="Vargha-Delaney A12 (95% Brunner-Munzel CI)",
                        reference_citation="Brunner & Munzel (2000); Konietschke et al. (2015, JSS)",
                    )
                )

                mw_score = 93 if not all_normal else 82
                recs.append(
                    TestRecommendation(
                        test_id="mann_whitney",
                        test_name="Mann-Whitney U Test (Wilcoxon Rank-Sum) + A12 CI",
                        category="Non-Parametric",
                        suitability="recommended" if not all_normal else "suitable",
                        suitability_score=mw_score,
                        rationale=(
                            f"Compares independent random-seed runs per instance without Gaussian assumptions "
                            f"(normality rate: {norm_pct}%). Default choice for independent stochastic runs."
                        ),
                        when_to_use="Independent random-seed executions on individual benchmark instances.",
                        assumptions_satisfied=[
                            f"Independent runs per cell (n ≈ {runs})",
                            "No normality assumption required",
                        ],
                        assumptions_violated=[],
                        suggested_table_type="instance_algorithm",
                        target_tab="stats",
                        target_mode="instance_algorithm",
                        posthoc_suggestion="Holm / Shaffer / Bergmann-Hommel adjustment",
                        effect_size_suggestion="Vargha-Delaney A12 & Cliff's δ (95% CI)",
                        reference_citation="Arcuri & Briand (2014); Vargha & Delaney (2000)",
                    )
                )

            wilcoxon_score = (
                94
                if (n_inst >= 2 and not all_normal)
                else (90 if (has_replicates and is_paired and not all_normal) else 80)
            )
            recs.append(
                TestRecommendation(
                    test_id="wilcoxon",
                    test_name="Wilcoxon Signed-Rank Test (Paired Blocks / Matched Runs)",
                    category="Non-Parametric",
                    suitability="recommended" if wilcoxon_score >= 90 else "suitable",
                    suitability_score=wilcoxon_score,
                    rationale=(
                        f"Evaluates paired differences across N={n_inst} matched instances or explicitly paired folds "
                        "(enable 'Paired Design' in the Workbench when runs share CV folds or Common Random Numbers)."
                    ),
                    when_to_use="Paired comparison across matched instances or explicitly paired folds/seeds.",
                    assumptions_satisfied=[
                        "Distribution-free paired comparison",
                        "Resistant to outliers",
                    ],
                    assumptions_violated=(
                        []
                        if (n_inst >= 2 or is_paired)
                        else ["Requires explicit pairing (CV folds or Common Random Numbers) within a single instance"]
                    ),
                    suggested_table_type="instance_algorithm",
                    target_tab="stats",
                    target_mode="instance_algorithm",
                    posthoc_suggestion=f"Holm-Bonferroni / Shaffer adjustment (α = {alpha})",
                    effect_size_suggestion="Vargha-Delaney A12 & Cliff's δ (95% CI)",
                    reference_citation="Demšar (2006); Derrac et al. (2011)",
                )
            )

        # ── 3. Multi-group cell omnibus & dedicated post-hoc procedures ──
        if k >= 3 and has_replicates:
            kw_score = 92 if not all_normal else 80
            recs.append(
                TestRecommendation(
                    test_id="kruskal",
                    test_name="Kruskal-Wallis H Omnibus + Dunn / Conover-Iman Post-Hoc",
                    category="Non-Parametric",
                    suitability="recommended" if not all_normal else "suitable",
                    suitability_score=kw_score,
                    rationale=(
                        f"Non-parametric omnibus alternative to ANOVA for k={k} algorithms within each instance "
                        f"(normality rate: {norm_pct}%). Follow with Dunn or Conover-Iman in the Post-Hoc Lab."
                    ),
                    when_to_use="Omnibus multi-group comparison within individual instances when normality fails.",
                    assumptions_satisfied=[
                        f"k = {k} groups with n ≈ {runs} runs/cell",
                        "Robust to skewness and heavy tails",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="instance_algorithm",
                    target_tab="stats",
                    target_mode="posthoc_lab",
                    posthoc_suggestion="Dunn's test or Conover-Iman with Holm / Finner correction",
                    effect_size_suggestion="Epsilon-squared (ε² with 95% bootstrap CI)",
                    reference_citation="Kruskal & Wallis (1952); Dunn (1964); Conover & Iman (1979)",
                )
            )

            welch_anova_score = 91 if (all_normal and not homo_ok) else (84 if all_normal else 60)
            recs.append(
                TestRecommendation(
                    test_id="anova_welch",
                    test_name="Welch's Heteroscedastic ANOVA + Games-Howell Post-Hoc",
                    category="Parametric",
                    suitability=(
                        "recommended"
                        if (all_normal and not homo_ok)
                        else ("suitable" if all_normal else "caution")
                    ),
                    suitability_score=welch_anova_score,
                    rationale=(
                        f"Parametric omnibus test for k={k} groups that does not assume equal variances "
                        f"(Levene min p = {diagnostics.homoscedasticity_levene_p:.4f}). Pairs naturally with "
                        "Games-Howell post-hoc comparisons."
                    ),
                    when_to_use="Comparing >= 3 normal groups with unequal variances.",
                    assumptions_satisfied=(
                        ["Robust to heteroscedasticity"] + (["Normality satisfied"] if all_normal else [])
                    ),
                    assumptions_violated=(
                        [] if all_normal else [f"Normality violated in {100.0 - norm_pct:.1f}% of groups"]
                    ),
                    suggested_table_type="instance_algorithm",
                    target_tab="stats",
                    target_mode="posthoc_lab",
                    posthoc_suggestion="Games-Howell all-pairs post-hoc (in Post-Hoc Lab)",
                    effect_size_suggestion="Eta-squared (η² with exact noncentral-F CI)",
                    reference_citation="Welch (1951); Games & Howell (1976)",
                )
            )

            anova_score = 93 if (all_normal and homo_ok) else (48 if not all_normal else 58)
            levene_viol = f"Homoscedasticity violated (Levene p={diagnostics.homoscedasticity_levene_p:.4f})"
            recs.append(
                TestRecommendation(
                    test_id="anova_oneway",
                    test_name="Fisher's One-Way ANOVA + Tukey HSD / Dunnett Post-Hoc",
                    category="Parametric",
                    suitability="recommended" if (all_normal and homo_ok) else "caution",
                    suitability_score=anova_score,
                    rationale=(
                        f"Classic parametric omnibus F-test across k={k} algorithms paired with Tukey's HSD "
                        "(all-pairs) or Dunnett's test (many-to-one vs control) in the Post-Hoc Lab."
                    ),
                    when_to_use="Comparing >= 3 Gaussian, homoscedastic algorithm distributions.",
                    assumptions_satisfied=(
                        (["Normality satisfied"] if all_normal else [])
                        + (["Levene homoscedasticity satisfied"] if homo_ok else [])
                    ),
                    assumptions_violated=(
                        ([] if all_normal else [f"Normality violated ({norm_pct}% pass rate)"])
                        + ([] if homo_ok else [levene_viol])
                    ),
                    suggested_table_type="instance_algorithm",
                    target_tab="stats",
                    target_mode="posthoc_lab",
                    posthoc_suggestion="Tukey's HSD (all-pairs) or Dunnett's test (vs control)",
                    effect_size_suggestion="Eta-squared (η² with exact noncentral-F CI)",
                    reference_citation="Fisher (1925); Tukey (1949); Dunnett (1955)",
                )
            )

        # ── 4. Two-group parametric tests ──
        if k == 2 and has_replicates:
            welch_t_score = 94 if (all_normal and not homo_ok) else (88 if all_normal else 62)
            recs.append(
                TestRecommendation(
                    test_id="ttest_welch",
                    test_name="Welch's Two-Sample t-test (Unequal Variances)",
                    category="Parametric",
                    suitability="recommended" if all_normal else "caution",
                    suitability_score=welch_t_score,
                    rationale=(
                        "Compares means of 2 independent algorithms using Satterthwaite degrees of freedom "
                        "with exact noncentral-t 95% CI for Hedges' g."
                    ),
                    when_to_use="Two-algorithm independent parametric comparison.",
                    assumptions_satisfied=(
                        ["Robust to unequal variances"] + (["Normality satisfied"] if all_normal else [])
                    ),
                    assumptions_violated=[] if all_normal else [f"Normality violated ({norm_pct}% pass rate)"],
                    suggested_table_type="instance_algorithm",
                    target_tab="stats",
                    target_mode="instance_algorithm",
                    posthoc_suggestion="Not required (k = 2)",
                    effect_size_suggestion="Hedges' g (exact noncentral-t 95% CI)",
                    reference_citation="Welch (1947); Steiger & Fouladi (1997)",
                )
            )

        # ── 5. Bayesian Simplex, Bayesian Correlated t & TOST Equivalence ──
        if k >= 2:
            recs.append(
                TestRecommendation(
                    test_id="bayesian_rope",
                    test_name="Bayesian Signed-Rank / Sign Simplex (ROPE)",
                    category="Bayesian",
                    suitability="recommended",
                    suitability_score=91,
                    rationale=(
                        "Exact Dirichlet Process Bayesian alternative to NHST p-values. Computes P(A better), "
                        "P(ROPE practical equivalence), and P(B better) with relative or absolute metric ROPE."
                    ),
                    when_to_use="Quantifying practical superiority vs equivalence (ROPE) without p-value inflation.",
                    assumptions_satisfied=[
                        "Exact Dirichlet Process prior DP(s, δ_0)",
                        "Supports absolute metric-unit ROPE & custom prior strength",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="pairwise_matrix",
                    target_tab="charts",
                    target_mode="bayesian-rank",
                    posthoc_suggestion="Posterior probability simplex",
                    effect_size_suggestion="ROPE Probability P(|Δ| ≤ ROPE)",
                    reference_citation="Benavoli et al. (2017), JMLR",
                )
            )

            if has_replicates or n_inst >= 2:
                bct_score = 90 if (is_paired and has_replicates) else 82
                recs.append(
                    TestRecommendation(
                        test_id="bayesian_correlated_ttest",
                        test_name="Bayesian Correlated t-Test (Repeated k-Fold CV)",
                        category="Bayesian",
                        suitability="recommended" if (is_paired and has_replicates) else "suitable",
                        suitability_score=bct_score,
                        rationale=(
                            "Exact analytical Student-t posterior with Nadeau & Bengio correlation correction ρ "
                            "(e.g., ρ = 1/9 for 10-fold CV) preventing overconfident inference on overlapping splits."
                        ),
                        when_to_use="Comparing two models evaluated on matched k-fold cross-validation splits.",
                        assumptions_satisfied=[
                            "Analytical Student-t posterior with exact ROPE integrals",
                            "Accounts for fold overlap via correlation parameter ρ",
                        ],
                        assumptions_violated=[],
                        suggested_table_type="pairwise_matrix",
                        target_tab="charts",
                        target_mode="bayesian-correlated-t",
                        posthoc_suggestion="Posterior density plot with ROPE region",
                        effect_size_suggestion="Posterior mean difference & P(ROPE)",
                        reference_citation="Nadeau & Bengio (2003); Benavoli et al. (2017)",
                    )
                )

                recs.append(
                    TestRecommendation(
                        test_id="tost",
                        test_name="TOST Statistical Equivalence Test & MDES Sensitivity",
                        category="Equivalence & Power",
                        suitability="suitable",
                        suitability_score=85,
                        rationale=(
                            "Formally tests whether two algorithms are practically equivalent within user-defined "
                            "bounds [Δ_L, Δ_U] (avoiding the fallacy of interpreting p ≥ α as equivalence)."
                        ),
                        when_to_use="Establishing practical equivalence or checking Minimum Detectable Effect Size.",
                        assumptions_satisfied=[
                            "Exact (1 − 2α) equivalence confidence interval",
                            "Supports Welch, Student, and Paired designs",
                        ],
                        assumptions_violated=[],
                        suggested_table_type="instance_algorithm",
                        target_tab="power",
                        target_mode="tost",
                        posthoc_suggestion="Equivalence interval plot + MDES sensitivity table",
                        effect_size_suggestion="Mean difference (90% CI) & MDES Cohen's d",
                        reference_citation="Schuirmann (1987); Lakens (2017); Hoenig & Heisey (2001)",
                    )
                )

        if not has_replicates:
            recs.append(
                TestRecommendation(
                    test_id="cell_tests_unavailable",
                    test_name="Single-Instance Parametric / Non-Parametric Cell Tests (Unavailable)",
                    category="Design Guard",
                    suitability="not_recommended",
                    suitability_score=10,
                    rationale=(
                        "This dataset has only 1 observation per (instance, algorithm) cell (e.g., Demšar-style "
                        "multi-dataset summary). Use blocked multi-instance tests (Friedman, Aligned-Ranks, Quade, "
                        "RM-ANOVA, or Bayesian Plackett-Luce / Signed-Rank) instead of within-instance tests."
                    ),
                    when_to_use="Requires at least n >= 2 independent runs per (instance, algorithm) cell.",
                    assumptions_satisfied=[],
                    assumptions_violated=["Only n = 1 observation per (instance, algorithm) cell"],
                    suggested_table_type="omnibus_posthoc",
                    target_tab="stats",
                    target_mode="omnibus_posthoc",
                    reference_citation="Demšar (2006), JMLR",
                )
            )

        recs.sort(key=lambda r: r.suitability_score, reverse=True)
        return recs


def _fmt_opt_num(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return _fmt_num(val)


def _fmt_p(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    if val < 0.0001:
        return f"{val:.2e}"
    return f"{val:.4f}"


def export_exploration_table(
    exploration: DataExplorationResult,
    view_mode: str = "reveal_metrics",
    export_format: str = "csv",
    stat_format: str = "mean_std",
    normality_filter: str = "all",
) -> tuple[bytes, str, str]:
    """Export an Observations & Data exploration view into PDF, DOCX, LaTeX, CSV, Markdown, or HTML.

    Args:
        exploration: DataExplorationResult produced by DataExplorerEngine.explore().
        view_mode: One of 'reveal_metrics' (aggregated), 'by_run' (pivoted runs),
                   'normality' (distribution diagnostics), or 'recommendations'.
        export_format: One of 'pdf', 'docx', 'latex', 'csv', 'markdown', 'html'.
        stat_format: 'mean_std' or 'median_iqr' for the 'reveal_metrics' view.
        normality_filter: 'all', 'normal', 'non_normal', or 'outliers' for the 'normality' view.

    Returns:
        Tuple of (file_bytes, media_type, extension).
    """
    from saes.stats.tables import export_flat_table

    mode = view_mode.strip().lower()
    metrics = exploration.active_metrics

    if mode == "by_run":
        headers = ["Algorithm", "Instance", "Run ID"] + metrics
        rows: list[list[str]] = []
        bolds: list[list[bool]] = []
        for r in exploration.pivoted_rows:
            row_vals = [r.algorithm, r.instance, str(r.run_id)] + [
                _fmt_opt_num(r.metrics_values.get(m)) for m in metrics
            ]
            rows.append(row_vals)
            bolds.append([False] * len(headers))
        title = "SAES Observations & Multi-Metric Run Explorer"
        caption = (
            f"Pivoted per-run observations across {len(exploration.active_algorithms)} algorithm(s), "
            f"{len(exploration.active_instances)} instance(s), and {len(metrics)} revealed metric(s)."
        )
        return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)

    if mode == "normality":
        headers = [
            "Metric",
            "Instance",
            "Algorithm",
            "n",
            "Mean ± SD",
            "Median (IQR)",
            "Skewness",
            "Kurtosis",
            "Shapiro p",
            "KS p",
            "Outliers",
            "Shape",
            "Suggested Paradigm",
        ]
        rows = []
        bolds = []
        for p in exploration.distribution_profiles:
            if normality_filter == "normal" and not p.is_normal:
                continue
            if normality_filter == "non_normal" and p.is_normal:
                continue
            if normality_filter == "outliers" and p.outlier_count == 0:
                continue
            paradigm = "Parametric / Non-Parametric" if p.is_normal else "Non-Parametric (Rank-Based)"
            rows.append(
                [
                    p.metric,
                    p.instance,
                    p.algorithm,
                    str(p.n),
                    f"{_fmt_num(p.mean)} ± {_fmt_num(p.std)}",
                    f"{_fmt_num(p.median)} ({_fmt_num(p.iqr)})",
                    f"{p.skewness:+.2f}",
                    f"{p.kurtosis:+.2f}",
                    _fmt_p(p.shapiro_p),
                    _fmt_p(p.ks_p),
                    str(p.outlier_count),
                    p.distribution_shape,
                    paradigm,
                ]
            )
            bolds.append([p.is_normal] * len(headers))
        title = "SAES Normality & Distribution Diagnostics Table"
        caption = (
            f"Shapiro-Wilk, D'Agostino-Pearson K², Lilliefors-corrected KS, skewness, kurtosis, and IQR outlier "
            f"diagnostics across {len(rows)} cell distribution(s)."
        )
        return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)

    if mode == "recommendations":
        headers = [
            "Suitability",
            "Score",
            "Statistical Test",
            "Category",
            "Post-Hoc Correction",
            "Effect Size",
            "Reference",
        ]
        rows = []
        bolds = []
        for rec in exploration.recommendations:
            is_rec = rec.suitability == "recommended"
            rows.append(
                [
                    rec.suitability.upper(),
                    f"{rec.suitability_score}/100",
                    rec.test_name,
                    rec.category,
                    rec.posthoc_suggestion,
                    rec.effect_size_suggestion,
                    rec.reference_citation,
                ]
            )
            bolds.append([is_rec] * len(headers))
        title = "SAES Context-Aware Statistical Test Recommendations"
        diag = exploration.context_diagnostics
        caption = (
            f"Context-aware recommendations for {diag.k_algorithms} algorithm(s), {diag.n_instances} instance(s), "
            f"and {diag.n_metrics} metric(s) (normality pass rate: {diag.normality_rate * 100:.1f}%)."
        )
        return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)

    # Default: "reveal_metrics" (Multi-Metric Aggregated Table)
    stat_label = "Median (IQR)" if stat_format == "median_iqr" else "Mean ± SD"
    headers = ["Algorithm", "Instance", "Runs (n)"]
    for m in metrics:
        headers.append(f"{m} ({stat_label})")
        headers.append(f"{m} Normality")

    rows = []
    bolds = []
    for agg in exploration.aggregated_rows:
        r_vals = [agg.algorithm, agg.instance, str(agg.n_runs)]
        b_vals = [True, False, False]
        for m in metrics:
            if stat_format == "median_iqr":
                val_str = agg.metrics_median_iqr.get(m, "—")
            else:
                val_str = agg.metrics_mean_std.get(m, "—")
            norm_ok = agg.metrics_normal.get(m)
            norm_str = "Normal" if norm_ok is True else ("Non-Normal" if norm_ok is False else "—")
            r_vals.extend([val_str, norm_str])
            b_vals.extend([False, bool(norm_ok)])
        rows.append(r_vals)
        bolds.append(b_vals)

    title = f"SAES Multi-Metric Reveal Table ({stat_label})"
    caption = (
        f"Side-by-side multi-metric comparison across {len(exploration.active_algorithms)} algorithm(s), "
        f"{len(exploration.active_instances)} instance(s), and {len(metrics)} revealed metric(s)."
    )
    return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)
