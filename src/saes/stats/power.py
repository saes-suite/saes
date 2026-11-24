"""Experimental design, a priori power analysis, sensitivity (MDES), and TOST equivalence testing.

Methodological note:
    Observed ("post-hoc") power computed from sample effect sizes is a 1-to-1 function of the
    observed p-value and should NOT be interpreted as independent evidence of Type II error
    (Hoenig & Heisey 2001, The American Statistician). For rigorous empirical analysis, prefer:
      1. A priori sample size planning (`estimate_required_sample_size`)
      2. Sensitivity / Minimum Detectable Effect Size (`compute_minimum_detectable_effect`)
      3. Two One-Sided Tests of equivalence (`tost_equivalence_test`, Schuirmann 1987; Lakens 2017)
    `compute_post_hoc_power_audit` is retained for backward compatibility with existing UI schemas
    with internally consistent effect-size/design calculations and explicit MDES reporting.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    Dataset,
    PostHocPowerItem,
    PowerAnalysisResult,
    PowerCurveSeries,
    SampleSizeRecommendation,
    SensitivityAuditItem,
    TOSTResult,
)
from saes.stats.nonparametric import compute_vargha_delaney_a12


def calculate_statistical_power(
    n: int,
    effect_size: float,
    alpha: float = 0.05,
    paired: bool = False,
) -> float:
    """Calculate exact two-tailed statistical power (1 - beta) for a Student t-test comparison.

    Uses the exact non-central Student t distribution (`scipy.stats.nct`).

    Args:
        n: Number of repetitions (runs) per sample group (n >= 2).
        effect_size: Standardized mean difference (Cohen's d for independent, d_z for paired; >= 0).
        alpha: Significance level (default 0.05).
        paired: Whether observations are explicitly paired across runs.

    Returns:
        Statistical power in [alpha, 1.0].
    """
    if n < 2 or effect_size <= 1e-9:
        return float(alpha)

    abs_d = abs(effect_size)

    if paired:
        df = n - 1
        delta = abs_d * math.sqrt(n)
    else:
        df = 2 * n - 2
        delta = abs_d * math.sqrt(n / 2.0)

    if df <= 0:
        return float(alpha)

    try:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        power = float(1.0 - stats.nct.cdf(t_crit, df, delta) + stats.nct.cdf(-t_crit, df, delta))
        if math.isfinite(power):
            return max(float(alpha), min(1.0, power))
    except Exception:
        pass

    se = math.sqrt(2.0 / n) if not paired else math.sqrt(1.0 / n)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = (abs_d / se) - z_crit
    return max(float(alpha), min(1.0, float(stats.norm.cdf(z_power))))


def compute_minimum_detectable_effect(
    n: int,
    target_power: float = 0.80,
    alpha: float = 0.05,
    paired: bool = False,
) -> float:
    """Compute the Minimum Detectable Effect Size (MDES, Cohen's d) for a fixed sample size n.

    Solves `calculate_statistical_power(n, d, alpha=alpha, paired=paired) = target_power`
    via Brent root-finding. Provides a valid sensitivity analysis without the observed-power fallacy.
    """
    if n < 2:
        raise StatisticalAnalysisError(f"Sample size n must be >= 2 for MDES calculation, got {n}.")
    if not (alpha < target_power < 1.0):
        raise StatisticalAnalysisError(
            f"target_power must satisfy alpha < target_power < 1.0 (got alpha={alpha}, target_power={target_power})."
        )

    def obj(d: float) -> float:
        return calculate_statistical_power(n=n, effect_size=d, alpha=alpha, paired=paired) - target_power

    hi = 1.0
    while obj(hi) < 0.0 and hi < 100.0:
        hi *= 2.0
    if obj(hi) < 0.0:
        return float(hi)
    return float(optimize.brentq(obj, 1e-6, hi, xtol=1e-6, maxiter=200))


def tost_equivalence_test(
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
    """Two One-Sided Tests (TOST) procedure for statistical equivalence (Schuirmann 1987; Lakens 2017).

    Tests H01: (mu1 - mu2) <= low_eqbound  vs  H11: (mu1 - mu2) > low_eqbound
      and H02: (mu1 - mu2) >= high_eqbound vs  H12: (mu1 - mu2) < high_eqbound.
    Practical equivalence is declared at level `alpha` iff max(p_lower, p_upper) < alpha
    (equivalently, the (1 - 2*alpha) CI lies strictly inside [low_eqbound, high_eqbound]).

    Args:
        x1: First group observations.
        x2: Second group observations.
        low_eqbound: Lower equivalence bound (if `high_eqbound` is None, uses symmetric [-|bound|, +|bound|]).
        high_eqbound: Optional upper equivalence bound (> low_eqbound).
        alpha: Equivalence significance level (default 0.05 -> 90% CI).
        paired: Whether observations are explicitly paired.
        equal_var: If False (default), uses Welch-Satterthwaite degrees of freedom when unpaired.
        group_1: Label for group 1.
        group_2: Label for group 2.
    """
    c1 = np.asarray(x1, dtype=float)
    c2 = np.asarray(x2, dtype=float)

    if high_eqbound is None:
        bound = abs(float(low_eqbound))
        lo_b, hi_b = -bound, bound
    else:
        lo_b, hi_b = float(low_eqbound), float(high_eqbound)
    if lo_b >= hi_b:
        raise StatisticalAnalysisError(f"low_eqbound ({lo_b}) must be strictly less than high_eqbound ({hi_b}).")

    if paired:
        mask = ~(np.isnan(c1) | np.isnan(c2))
        c1, c2 = c1[mask], c2[mask]
        if len(c1) != len(c2) or len(c1) < 2:
            raise StatisticalAnalysisError("Paired TOST requires equal sample sizes with at least 2 pairs.")
        diffs = c1 - c2
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        se = sd_diff / math.sqrt(n)
        df = float(n - 1)
        test_name = "Paired TOST Equivalence Test"
    else:
        c1 = c1[~np.isnan(c1)]
        c2 = c2[~np.isnan(c2)]
        n1, n2 = len(c1), len(c2)
        if n1 < 2 or n2 < 2:
            raise StatisticalAnalysisError("Independent TOST requires at least 2 observations per group.")
        mean_diff = float(np.mean(c1) - np.mean(c2))
        v1, v2 = float(np.var(c1, ddof=1)), float(np.var(c2, ddof=1))
        if equal_var:
            df = float(n1 + n2 - 2)
            pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            se = math.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
            test_name = "Student's Independent TOST Equivalence Test"
        else:
            se1_sq, se2_sq = v1 / n1, v2 / n2
            se_sq = se1_sq + se2_sq
            se = math.sqrt(se_sq)
            den = (se1_sq**2) / (n1 - 1) + (se2_sq**2) / (n2 - 1)
            df = float((se_sq**2) / den) if den > 0 else float(n1 + n2 - 2)
            test_name = "Welch's Independent TOST Equivalence Test"

    if se <= 1e-15:
        t_lower = 1e6 if mean_diff > lo_b else -1e6
        p_lower = 0.0 if mean_diff > lo_b else 1.0
        t_upper = -1e6 if mean_diff < hi_b else 1e6
        p_upper = 0.0 if mean_diff < hi_b else 1.0
        ci_low, ci_high = mean_diff, mean_diff
    else:
        t_lower = float((mean_diff - lo_b) / se)
        p_lower = float(stats.t.sf(t_lower, df))  # Upper tail: H11: diff > low_eqbound
        t_upper = float((mean_diff - hi_b) / se)
        p_upper = float(stats.t.cdf(t_upper, df))  # Lower tail: H12: diff < high_eqbound
        t_crit = float(stats.t.ppf(1.0 - alpha, df))  # (1 - 2*alpha) CI
        ci_low = float(mean_diff - t_crit * se)
        ci_high = float(mean_diff + t_crit * se)

    p_tost = max(p_lower, p_upper)
    is_equiv = bool(p_tost < alpha)
    ci_pct = (1.0 - 2.0 * alpha) * 100.0
    interp = (
        f"{test_name} establishes statistical equivalence between '{group_1}' and '{group_2}' "
        f"within [{lo_b:.4g}, {hi_b:.4g}] (diff = {mean_diff:.4g}, {ci_pct:.0f}% CI [{ci_low:.4g}, {ci_high:.4g}], "
        f"p_TOST = {p_tost:.4e} < {alpha})."
        if is_equiv
        else f"{test_name} does NOT establish statistical equivalence between '{group_1}' and '{group_2}' "
        f"within [{lo_b:.4g}, {hi_b:.4g}] (diff = {mean_diff:.4g}, {ci_pct:.0f}% CI [{ci_low:.4g}, {ci_high:.4g}], "
        f"p_TOST = {p_tost:.4f} >= {alpha})."
    )
    return TOSTResult(
        test_name=test_name,
        group_1=group_1,
        group_2=group_2,
        mean_diff=mean_diff,
        low_eqbound=lo_b,
        high_eqbound=hi_b,
        t_lower=t_lower,
        p_lower=p_lower,
        t_upper=t_upper,
        p_upper=p_upper,
        p_value=p_tost,
        degrees_of_freedom=df,
        alpha=alpha,
        is_equivalent=is_equiv,
        ci_lower=ci_low,
        ci_upper=ci_high,
        paired=paired,
        interpretation=interp,
    )


def estimate_required_sample_size(
    target_power: float = 0.80,
    effect_size: float = 0.50,
    alpha: float = 0.05,
) -> SampleSizeRecommendation:
    """Estimate required sample size (runs per instance) for independent and paired tests.

    Applies the Hodges-Lehmann asymptotic relative efficiency (A.R.E. = 0.864) lower bound
    for non-parametric rank-sum comparisons.
    """
    abs_d = max(0.05, abs(effect_size))

    if abs_d < 0.35:
        label = "Small (d ≈ 0.20)"
    elif abs_d < 0.65:
        label = "Medium (d ≈ 0.50)"
    else:
        label = "Large (d ≈ 0.80)"

    n_ind = 2
    while n_ind < 2000:
        if calculate_statistical_power(n_ind, abs_d, alpha=alpha, paired=False) >= target_power:
            break
        n_ind += 1

    n_paired = 2
    while n_paired < 2000:
        if calculate_statistical_power(n_paired, abs_d, alpha=alpha, paired=True) >= target_power:
            break
        n_paired += 1

    n_nonparam = max(n_ind, math.ceil(n_ind / 0.864))

    return SampleSizeRecommendation(
        target_power=target_power,
        alpha=alpha,
        effect_size=round(abs_d, 3),
        effect_size_label=label,
        recommended_n_independent=n_ind,
        recommended_n_paired=n_paired,
        recommended_n_nonparametric=n_nonparam,
    )


def generate_power_curves(
    alpha: float = 0.05,
    custom_effect_size: float | None = None,
    paired: bool = False,
) -> list[PowerCurveSeries]:
    """Generate sample size vs power curves for standard and custom effect sizes."""
    sample_sizes = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]

    curves_specs = [
        ("Small Effect (d = 0.20)", 0.20),
        ("Medium Effect (d = 0.50)", 0.50),
        ("Large Effect (d = 0.80)", 0.80),
    ]

    if custom_effect_size is not None and custom_effect_size not in (0.2, 0.5, 0.8):
        curves_specs.append((f"Custom Effect (d = {custom_effect_size:.2f})", custom_effect_size))

    series_list: list[PowerCurveSeries] = []
    for label, d_val in curves_specs:
        powers = [
            round(calculate_statistical_power(n, d_val, alpha=alpha, paired=paired), 4) for n in sample_sizes
        ]
        series_list.append(
            PowerCurveSeries(
                effect_size_label=label,
                effect_size_value=d_val,
                sample_sizes=sample_sizes,
                power_values=powers,
            )
        )

    return series_list


def compute_sensitivity_audit(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    paired: bool = False,
) -> list[SensitivityAuditItem]:
    """Compute sensitivity (Minimum Detectable Effect Size, MDES) for each benchmark pair."""
    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]
    metric_df = dataset.df[dataset.df["metric"] == target_metric].copy()
    maximize: bool = bool(metric_df["maximize"].iloc[0]) if not metric_df.empty else True

    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    instances = sorted(metric_df["instance"].unique().tolist())

    items: list[SensitivityAuditItem] = []
    for inst in instances:
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                a1, a2 = algorithms[i], algorithms[j]
                s1 = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == a1)]["value"].to_numpy(
                    dtype=float
                )
                s2 = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == a2)]["value"].to_numpy(
                    dtype=float
                )
                n1, n2 = len(s1), len(s2)
                if n1 < 2 or n2 < 2:
                    continue

                use_paired = paired and (n1 == n2)
                n_eff = min(n1, n2)
                mdes = compute_minimum_detectable_effect(
                    n=n_eff, target_power=target_power, alpha=alpha, paired=use_paired
                )
                if use_paired:
                    diffs = s1 - s2
                    sd_d = float(np.std(diffs, ddof=1))
                    obs_d = abs(float(np.mean(diffs)) / sd_d) if sd_d > 1e-12 else 0.0
                else:
                    v1, v2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))
                    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, n1 + n2 - 2))
                    obs_d = abs(float(np.mean(s1) - np.mean(s2))) / pooled_sd if pooled_sd > 1e-12 else 0.0

                a12_val, _ = compute_vargha_delaney_a12(s1, s2, maximize=maximize)
                rec = (
                    f"With N={n_eff} {'paired' if use_paired else 'independent'} runs at α={alpha}, "
                    f"this design achieves {target_power:.0%} power only for effects |d| ≥ {mdes:.2f} (MDES)."
                )
                items.append(
                    SensitivityAuditItem(
                        algo_1=a1,
                        algo_2=a2,
                        instance=inst,
                        metric=target_metric,
                        n_samples=n_eff,
                        target_power=target_power,
                        alpha=alpha,
                        paired=use_paired,
                        mdes_cohens_d=round(mdes, 4),
                        observed_cohens_d=round(obs_d, 4),
                        observed_a12=round(a12_val, 4),
                        recommendation=rec,
                    )
                )
    return items


def compute_post_hoc_power_audit(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    target_power: float = 0.80,
) -> tuple[list[PostHocPowerItem], str]:
    """Audit sensitivity and retrospective power across pairwise algorithm comparisons.

    Note:
        Observed power is computed strictly consistently with the declared design (`paired=False`
        by default, never inferred from `n1 == n2`), and recommendations emphasize MDES sensitivity
        rather than treating observed power as independent Type II error evidence (Hoenig & Heisey 2001).
    """
    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]
    metric_df = dataset.df[dataset.df["metric"] == target_metric].copy()
    maximize: bool = bool(metric_df["maximize"].iloc[0]) if not metric_df.empty else True

    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    instances = sorted(metric_df["instance"].unique().tolist())

    audit_items: list[PostHocPowerItem] = []
    optimal_count = 0
    total_comparisons = 0

    for inst in instances:
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                a1 = algorithms[i]
                a2 = algorithms[j]

                s1 = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == a1)]["value"].to_numpy(
                    dtype=float
                )
                s2 = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == a2)]["value"].to_numpy(
                    dtype=float
                )

                n1, n2 = len(s1), len(s2)
                if n1 < 2 or n2 < 2:
                    continue

                total_comparisons += 1
                use_paired = paired and (n1 == n2)
                n_eff = min(n1, n2)
                mdes = compute_minimum_detectable_effect(
                    n=n_eff, target_power=target_power, alpha=alpha, paired=use_paired
                )

                if use_paired:
                    diffs = s1 - s2
                    sd_d = float(np.std(diffs, ddof=1))
                    cohens_d = abs(float(np.mean(diffs)) / sd_d) if sd_d > 1e-12 else 0.0
                    try:
                        res = stats.ttest_rel(s1, s2)
                        p_val = float(res.pvalue) if math.isfinite(res.pvalue) else 1.0
                    except Exception:
                        p_val = 1.0
                else:
                    m1, m2 = float(np.mean(s1)), float(np.mean(s2))
                    var1, var2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))
                    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, n1 + n2 - 2))
                    cohens_d = (abs(m1 - m2) / pooled_sd) if pooled_sd > 1e-12 else 0.0
                    try:
                        res = stats.ttest_ind(s1, s2, equal_var=False)
                        p_val = float(res.pvalue) if math.isfinite(res.pvalue) else 1.0
                    except Exception:
                        p_val = 1.0

                a12_val, _ = compute_vargha_delaney_a12(s1, s2, maximize=maximize)
                power = calculate_statistical_power(n_eff, cohens_d, alpha=alpha, paired=use_paired)
                is_sig = p_val < alpha

                if power >= 0.80:
                    status = "Optimal"
                    optimal_count += 1
                    recommendation = (
                        f"Design sensitivity MDES = {mdes:.2f} at {target_power:.0%} power (N={n_eff}). "
                        f"Observed effect |d|={cohens_d:.2f} exceeds MDES."
                    )
                elif power >= 0.70:
                    status = "Adequate"
                    recommendation = (
                        f"Design sensitivity MDES = {mdes:.2f} (N={n_eff}). "
                        "Consider a priori planning or TOST equivalence testing rather than relying on post-hoc power."
                    )
                else:
                    status = "Underpowered"
                    rec_spec = estimate_required_sample_size(
                        target_power=target_power, effect_size=max(0.2, cohens_d), alpha=alpha
                    )
                    rec_n = rec_spec.recommended_n_paired if use_paired else rec_spec.recommended_n_independent
                    recommendation = (
                        f"Design MDES = {mdes:.2f} at {target_power:.0%} power (N={n_eff}). "
                        f"To detect an a priori effect of d={max(0.2, cohens_d):.2f}, plan N={rec_n} runs "
                        "(note: interpret via MDES/TOST, not post-hoc power; Hoenig & Heisey 2001)."
                    )

                audit_items.append(
                    PostHocPowerItem(
                        algo_1=a1,
                        algo_2=a2,
                        instance=inst,
                        metric=target_metric,
                        n_samples=n_eff,
                        cohens_d=round(cohens_d, 4),
                        a12=round(a12_val, 4),
                        achieved_power=round(power, 4),
                        power_status=status,
                        is_significant=is_sig,
                        recommendation=recommendation,
                    )
                )

    if total_comparisons == 0:
        overall_health = "Optimal"
    else:
        ratio = optimal_count / total_comparisons
        if ratio >= 0.75:
            overall_health = "Optimal"
        elif ratio >= 0.50:
            overall_health = "Adequate"
        else:
            overall_health = "Underpowered"

    return audit_items, overall_health


def analyze_statistical_power(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    target_effect_size: float = 0.50,
    paired: bool = False,
) -> PowerAnalysisResult:
    """Run full a priori power planning, sensitivity (MDES) analysis, and design audit."""
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no metrics for power analysis.")

    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]

    standard_effects = [
        (0.20, "Small (d = 0.20)"),
        (0.50, "Medium (d = 0.50)"),
        (0.80, "Large (d = 0.80)"),
    ]
    if target_effect_size not in (0.20, 0.50, 0.80):
        standard_effects.append((target_effect_size, f"Custom (d = {target_effect_size:.2f})"))

    recommendations: list[SampleSizeRecommendation] = [
        estimate_required_sample_size(target_power=target_power, effect_size=d_val, alpha=alpha)
        for d_val, _ in standard_effects
    ]

    power_curves = generate_power_curves(alpha=alpha, custom_effect_size=target_effect_size, paired=paired)

    audit_items, overall_health = compute_post_hoc_power_audit(
        dataset=dataset,
        metric=target_metric,
        alpha=alpha,
        paired=paired,
        target_power=target_power,
    )
    sensitivity_items = compute_sensitivity_audit(
        dataset=dataset,
        metric=target_metric,
        alpha=alpha,
        target_power=target_power,
        paired=paired,
    )

    return PowerAnalysisResult(
        alpha=alpha,
        target_power=target_power,
        target_effect_size=target_effect_size,
        sample_size_recommendations=recommendations,
        power_curves=power_curves,
        post_hoc_audit=audit_items,
        sensitivity_audit=sensitivity_items,
        overall_power_health=overall_health,
    )
