"""Standardized effect size estimators with exact and asymptotic confidence intervals.

Implements:
    - Cohen's d & Hedges' g with exact noncentral-t confidence intervals
      (Steiger & Fouladi 1997; Smithson 2001; Kelley 2007).
    - Eta-squared (η²) & Omega-squared (ω²) with exact noncentral-F confidence intervals
      (Smithson 2001).
    - Vargha-Delaney A12 & Cliff's δ with Brunner-Munzel (2000) / Konietschke et al. (2015)
      asymptotic logit-transformed confidence intervals.
    - Kruskal-Wallis Epsilon-squared (ε²) with deterministic percentile bootstrap CI.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats


def classify_cohen_d(d_val: float) -> str:
    """Classify magnitude of Cohen's d or Hedges' g (Cohen 1988)."""
    abs_d = abs(d_val)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def classify_eta_squared(eta_sq: float) -> str:
    """Classify magnitude of Eta-squared / Omega-squared effect size (Cohen 1988)."""
    if eta_sq < 0.01:
        return "negligible"
    if eta_sq < 0.06:
        return "small"
    if eta_sq < 0.14:
        return "medium"
    return "large"


def classify_vargha_delaney_a12(a12: float) -> str:
    """Classify Vargha-Delaney A12 effect size magnitude (Vargha & Delaney 2000; Arcuri & Briand 2014).

    Thresholds for |A12 - 0.5|:
      - < 0.06 (0.44 < A12 < 0.56): negligible
      - < 0.14 (0.36 < A12 < 0.64): small
      - < 0.21 (0.29 < A12 < 0.71): medium
      - >= 0.21 (A12 <= 0.29 or A12 >= 0.71): large
    """
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    if diff < 0.14:
        return "small"
    if diff < 0.21:
        return "medium"
    return "large"


def classify_epsilon_squared(eps_sq: float) -> str:
    """Classify Kruskal-Wallis epsilon-squared effect size."""
    if eps_sq < 0.01:
        return "negligible"
    if eps_sq < 0.08:
        return "small"
    if eps_sq < 0.26:
        return "medium"
    return "large"


def _solve_nct_ncp_bounds(t_obs: float, df: float, alpha: float) -> tuple[float, float] | None:
    """Invert noncentral-t CDF to obtain exact (1 - alpha) CI for the noncentrality parameter."""
    if not math.isfinite(t_obs) or df <= 0:
        return None

    target_lower = 1.0 - alpha / 2.0  # F(t_obs; df, ncp_low) = 1 - alpha/2
    target_upper = alpha / 2.0  # F(t_obs; df, ncp_high) = alpha/2

    def _find_root(target_cdf: float) -> float | None:
        def f(ncp: float) -> float:
            val = float(stats.nct.cdf(t_obs, df, ncp))
            return val - target_cdf

        step = max(2.0, abs(t_obs) * 0.5 + 2.0)
        lo = t_obs - step
        hi = t_obs + step
        for _ in range(12):
            f_lo = f(lo)
            f_hi = f(hi)
            if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi <= 0:
                try:
                    return float(optimize.brentq(f, lo, hi, xtol=1e-7, maxiter=200))
                except Exception:
                    return None
            step *= 2.0
            lo = t_obs - step
            hi = t_obs + step
        return None

    ncp_low = _find_root(target_lower)
    ncp_high = _find_root(target_upper)
    if ncp_low is None or ncp_high is None:
        return None
    return (min(ncp_low, ncp_high), max(ncp_low, ncp_high))


def cohens_d_ci(
    d: float,
    n1: int,
    n2: int | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    hedges_correction: bool = False,
) -> tuple[float, float]:
    """Compute exact noncentral-t (with Hedges-Olkin fallback) confidence interval for Cohen's d / Hedges' g.

    Args:
        d: Point estimate of Cohen's d (or Hedges' g if `hedges_correction=True`).
        n1: Sample size of group 1 (or number of pairs when `paired=True`).
        n2: Sample size of group 2 (required when `paired=False`).
        alpha: Two-sided significance level (default 0.05 -> 95% CI).
        paired: If True, treats `d` as paired standardized mean difference d_z.
        hedges_correction: If True, scales noncentral-t bounds by small-sample factor J.

    Returns:
        Tuple (ci_lower, ci_upper).
    """
    if not math.isfinite(d):
        return (0.0, 0.0)

    if paired:
        if n1 < 2:
            return (d, d)
        df = float(n1 - 1)
        scale = math.sqrt(float(n1))
        j_factor = 1.0 - (3.0 / (4.0 * df - 1.0)) if (hedges_correction and df > 1) else 1.0
        d_uncorr = d / j_factor if j_factor > 0 else d
        t_obs = d_uncorr * scale
        bounds = _solve_nct_ncp_bounds(t_obs, df, alpha)
        if bounds is not None:
            return (float(bounds[0] / scale * j_factor), float(bounds[1] / scale * j_factor))
        se = math.sqrt((1.0 / n1) + (d**2) / (2.0 * n1))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return (float(d - z_crit * se), float(d + z_crit * se))

    n2_eff = n2 if n2 is not None else n1
    if n1 < 2 or n2_eff < 2:
        return (d, d)

    df = float(n1 + n2_eff - 2)
    scale = math.sqrt((n1 * n2_eff) / float(n1 + n2_eff))
    j_factor = 1.0 - (3.0 / (4.0 * (n1 + n2_eff) - 9.0)) if (hedges_correction and (n1 + n2_eff) > 2) else 1.0
    d_uncorr = d / j_factor if j_factor > 0 else d
    t_obs = d_uncorr * scale
    bounds = _solve_nct_ncp_bounds(t_obs, df, alpha)
    if bounds is not None:
        return (float(bounds[0] / scale * j_factor), float(bounds[1] / scale * j_factor))

    # Hedges & Olkin (1985) asymptotic variance fallback
    se = math.sqrt((n1 + n2_eff) / float(n1 * n2_eff) + (d**2) / (2.0 * (n1 + n2_eff)))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return (float(d - z_crit * se), float(d + z_crit * se))


def eta_squared_ci(
    f_stat: float,
    df_between: float,
    df_within: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute exact noncentral-F confidence interval for ANOVA Eta-squared (Smithson 2001).

    Uses the standard two-sided (1 - alpha) noncentral-F inversion mapping noncentrality
    parameter lambda to partial/one-way eta-squared: eta^2 = lambda / (lambda + df1 + df2 + 1).
    """
    if not math.isfinite(f_stat) or f_stat <= 0.0 or df_between <= 0.0 or df_within <= 0.0:
        return (0.0, 0.0)

    n_total = df_between + df_within + 1.0

    def _ncp_to_eta(ncp: float) -> float:
        return float(max(0.0, min(1.0, ncp / (ncp + n_total))))

    def _find_ncp(target_cdf: float) -> float:
        # At ncp = 0, F follows central F. If central CDF <= target, lower bound is 0.
        cdf_zero = float(stats.f.cdf(f_stat, df_between, df_within))
        if cdf_zero <= target_cdf:
            return 0.0

        def obj(ncp: float) -> float:
            return float(stats.ncf.cdf(f_stat, df_between, df_within, ncp)) - target_cdf

        hi = max(10.0, f_stat * df_between * 2.0)
        for _ in range(15):
            val_hi = obj(hi)
            if math.isfinite(val_hi) and val_hi <= 0.0:
                try:
                    return float(optimize.brentq(obj, 0.0, hi, xtol=1e-7, maxiter=200))
                except Exception:
                    return 0.0
            hi *= 2.0
        return 0.0

    ncp_low = _find_ncp(1.0 - alpha / 2.0)
    ncp_high = _find_ncp(alpha / 2.0)
    return (_ncp_to_eta(min(ncp_low, ncp_high)), _ncp_to_eta(max(ncp_low, ncp_high)))


def compute_vargha_delaney_a12_with_ci(
    x1: np.ndarray,
    x2: np.ndarray,
    maximize: bool = True,
    alpha: float = 0.05,
) -> tuple[float, str, tuple[float, float]]:
    """Compute Vargha-Delaney A12 and Brunner-Munzel (2000) asymptotic confidence interval.

    Reference:
        Brunner, E., & Munzel, U. (2000). The nonparametric Behrens-Fisher problem:
        Asymptotic theory and a small-sample approximation. Biometrical Journal, 42(1), 17-25.
        Konietschke, F., et al. (2015). nparcomp (JSS, 64(9)).

    Returns:
        Tuple (a12, magnitude, (ci_lower, ci_upper)).
    """
    c1 = np.asarray(x1, dtype=float)
    c2 = np.asarray(x2, dtype=float)
    c1 = c1[~np.isnan(c1)]
    c2 = c2[~np.isnan(c2)]
    n1, n2 = len(c1), len(c2)
    if n1 == 0 or n2 == 0:
        return 0.5, "negligible", (0.5, 0.5)

    # Orient so higher values are better
    v1 = c1 if maximize else -c1
    v2 = c2 if maximize else -c2

    # Pairwise placement matrix: 1 if v1_i > v2_j, 0.5 if tie, 0 otherwise
    diff_matrix = v1[:, None] - v2[None, :]
    win_mat = (diff_matrix > 0).astype(float) + 0.5 * (diff_matrix == 0).astype(float)
    a12 = float(np.mean(win_mat))
    mag = classify_vargha_delaney_a12(a12)

    if n1 < 2 or n2 < 2:
        return a12, mag, (max(0.0, a12), min(1.0, a12))

    # Placement vectors (Brunner-Munzel variance estimator):
    # p_1i = average score of v1_i against sample 2; p_2j = average score of v2_j against sample 1
    p1 = np.mean(win_mat, axis=1)
    p2 = np.mean(1.0 - win_mat, axis=0)
    s1_sq = float(np.var(p1, ddof=1))
    s2_sq = float(np.var(p2, ddof=1))
    var_a12 = (s1_sq / n1) + (s2_sq / n2)

    if var_a12 <= 1e-15:
        return a12, mag, (a12, a12)

    # Welch-Satterthwaite degrees of freedom for Brunner-Munzel t-approximation
    num_df = var_a12**2
    den_df = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
    df = float(num_df / den_df) if den_df > 0 else float(n1 + n2 - 2)
    df = max(1.0, df)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    se = math.sqrt(var_a12)

    # Range-preserving logit transformation when strictly interior to (0, 1) (Konietschke et al. 2015)
    if 1e-6 < a12 < 1.0 - 1e-6:
        logit_p = math.log(a12 / (1.0 - a12))
        se_logit = se / (a12 * (1.0 - a12))
        lo = 1.0 / (1.0 + math.exp(-(logit_p - t_crit * se_logit)))
        hi = 1.0 / (1.0 + math.exp(-(logit_p + t_crit * se_logit)))
    else:
        lo = max(0.0, a12 - t_crit * se)
        hi = min(1.0, a12 + t_crit * se)

    return a12, mag, (float(max(0.0, lo)), float(min(1.0, hi)))


def compute_cliffs_delta_with_ci(
    x1: np.ndarray,
    x2: np.ndarray,
    maximize: bool = True,
    alpha: float = 0.05,
) -> tuple[float, str, tuple[float, float]]:
    """Compute Cliff's delta = 2 * A12 - 1 in [-1, 1] and its range-preserving CI."""
    a12, mag, (lo_a, hi_a) = compute_vargha_delaney_a12_with_ci(x1, x2, maximize=maximize, alpha=alpha)
    delta = float(2.0 * a12 - 1.0)
    return delta, mag, (float(2.0 * lo_a - 1.0), float(2.0 * hi_a - 1.0))


def epsilon_squared_ci(
    groups: list[np.ndarray],
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute deterministic percentile bootstrap CI for Kruskal-Wallis epsilon-squared."""
    clean = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]
    if len(clean) < 2:
        return (0.0, 0.0)

    n_total = sum(len(g) for g in clean)
    if n_total <= 1:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_vals: list[float] = []
    for _ in range(n_bootstrap):
        resampled = [g[rng.integers(0, len(g), size=len(g))] for g in clean]
        all_v = np.concatenate(resampled)
        if np.all(all_v == all_v[0]):
            boot_vals.append(0.0)
            continue
        try:
            res = stats.kruskal(*resampled)
            h_val = float(res.statistic)
            boot_vals.append(max(0.0, min(1.0, h_val / float(n_total - 1))))
        except Exception:
            boot_vals.append(0.0)

    lo, hi = np.percentile(boot_vals, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (float(lo), float(hi))


def compute_effect_size_forest(
    df: object,
    metric: str,
    scope: str = "single-instance",
    instance: str | None = None,
    reference_algorithm: str | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    p_adjust: str = "holm",
) -> object:
    """Compute standardized effect sizes (Cohen's d, Hedges' g, A12, Cliff's delta) with exact CIs vs reference.

    Positive d / g / Cliff's delta (and A12 > 0.5) indicate that `algorithm` outperforms `reference_algorithm`
    accounting for the metric's `maximize` direction.
    """
    import pandas as pd

    from saes.domain.exceptions import StatisticalAnalysisError
    from saes.domain.models import EffectSizeComparisonItem, EffectSizeForestResult
    from saes.stats.posthoc import adjust_pvalues

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise StatisticalAnalysisError("Dataset is empty; cannot compute effect-size forest.")

    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        raise StatisticalAnalysisError(f"Metric '{metric}' not found in dataset.")

    maximize = bool(sub["maximize"].iloc[0])
    algorithms = sorted(sub["algorithm"].unique().tolist())
    if len(algorithms) < 2:
        raise StatisticalAnalysisError(f"Effect-size forest requires >= 2 algorithms, got {len(algorithms)}.")

    instances = sorted(sub["instance"].unique().tolist())
    warnings_list: list[str] = []

    norm_scope = scope.strip().lower()
    use_all_inst = norm_scope == "all-instances" and len(instances) >= 2
    selected_inst = instance if (instance and instance in instances) else instances[0]

    vectors: dict[str, np.ndarray] = {}
    if use_all_inst:
        piv = sub.groupby(["instance", "algorithm"])["value"].mean().unstack()
        piv = piv.dropna(axis=0, how="any")
        for alg in algorithms:
            if alg in piv.columns:
                vectors[alg] = piv[alg].to_numpy(dtype=float)
        effective_paired = True  # Instances form matched blocks
        resolved_scope = "all-instances"
        resolved_inst: str | None = None
    else:
        cell = sub[sub["instance"] == selected_inst]
        for alg in algorithms:
            arr = cell[cell["algorithm"] == alg]["value"].dropna().to_numpy(dtype=float)
            if len(arr) >= 2:
                vectors[alg] = arr
        effective_paired = paired
        resolved_scope = "single-instance"
        resolved_inst = selected_inst

    valid_algos = [a for a in algorithms if a in vectors and len(vectors[a]) >= 2]
    if len(valid_algos) < 2:
        raise StatisticalAnalysisError(
            "At least 2 algorithms with >= 2 observations are required for effect-size CI computation."
        )

    # Default reference: best median performer
    if reference_algorithm and reference_algorithm in valid_algos:
        ref_alg = reference_algorithm
    else:
        medians = {a: float(np.median(vectors[a])) for a in valid_algos}
        ref_alg = sorted(valid_algos, key=lambda a: medians[a], reverse=maximize)[0]

    ref_vec = vectors[ref_alg]
    competitors = [a for a in valid_algos if a != ref_alg]
    raw_items: list[dict[str, float | str | int]] = []
    raw_ps: list[float] = []

    for alg in competitors:
        vec = vectors[alg]
        n1, n2 = len(vec), len(ref_vec)
        can_pair = effective_paired and n1 == n2 and n1 >= 2

        # Orient so positive effect means `alg` is better than `ref_alg`
        v1 = vec if maximize else -vec
        v2 = ref_vec if maximize else -ref_vec

        if can_pair:
            diffs = v1 - v2
            sd_diff = float(np.std(diffs, ddof=1))
            d_val = float(np.mean(diffs) / sd_diff) if sd_diff > 1e-12 else 0.0
            df_val = float(n1 - 1)
            j_fac = 1.0 - (3.0 / (4.0 * df_val - 1.0)) if df_val > 1 else 1.0
            g_val = d_val * j_fac
            d_lo, d_hi = cohens_d_ci(d_val, n1=n1, alpha=alpha, paired=True, hedges_correction=False)
            g_lo, g_hi = cohens_d_ci(g_val, n1=n1, alpha=alpha, paired=True, hedges_correction=True)
            try:
                p_val = float(stats.wilcoxon(v1, v2).pvalue) if not np.allclose(v1, v2) else 1.0
            except Exception:
                p_val = 1.0
        else:
            s1, s2 = float(np.var(v1, ddof=1)), float(np.var(v2, ddof=1))
            pooled_sd = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1.0, float(n1 + n2 - 2)))
            d_val = float((np.mean(v1) - np.mean(v2)) / pooled_sd) if pooled_sd > 1e-12 else 0.0
            j_fac = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2) > 2 else 1.0
            g_val = d_val * j_fac
            d_lo, d_hi = cohens_d_ci(d_val, n1=n1, n2=n2, alpha=alpha, paired=False, hedges_correction=False)
            g_lo, g_hi = cohens_d_ci(g_val, n1=n1, n2=n2, alpha=alpha, paired=False, hedges_correction=True)
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                bm = stats.brunnermunzel(v1, v2, alternative="two-sided", distribution="t")
            p_raw = float(bm.pvalue)
            if not math.isfinite(p_raw):
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", RuntimeWarning)
                    bm_norm = stats.brunnermunzel(v1, v2, alternative="two-sided", distribution="normal")
                p_raw = float(bm_norm.pvalue)
            p_val = p_raw if math.isfinite(p_raw) else (1.0 if np.allclose(v1, v2) else 0.0)

        a12_val, a12_mag, (a12_lo, a12_hi) = compute_vargha_delaney_a12_with_ci(
            vec, ref_vec, maximize=maximize, alpha=alpha
        )
        cd_val, _, (cd_lo, cd_hi) = compute_cliffs_delta_with_ci(vec, ref_vec, maximize=maximize, alpha=alpha)

        raw_ps.append(p_val)
        raw_items.append(
            {
                "algorithm": alg,
                "n_algorithm": n1,
                "n_reference": n2,
                "cohens_d": d_val,
                "cohens_d_ci_lower": d_lo,
                "cohens_d_ci_upper": d_hi,
                "cohens_d_magnitude": classify_cohen_d(d_val),
                "hedges_g": g_val,
                "hedges_g_ci_lower": g_lo,
                "hedges_g_ci_upper": g_hi,
                "a12": a12_val,
                "a12_ci_lower": a12_lo,
                "a12_ci_upper": a12_hi,
                "a12_magnitude": a12_mag,
                "cliffs_delta": cd_val,
                "cliffs_delta_ci_lower": cd_lo,
                "cliffs_delta_ci_upper": cd_hi,
                "p_value": p_val,
            }
        )

    adj_ps = adjust_pvalues(raw_ps, method=p_adjust)
    items: list[EffectSizeComparisonItem] = []
    for row, p_adj in zip(raw_items, adj_ps, strict=False):
        items.append(
            EffectSizeComparisonItem(
                algorithm=str(row["algorithm"]),
                reference=ref_alg,
                n_algorithm=int(row["n_algorithm"]),
                n_reference=int(row["n_reference"]),
                cohens_d=float(row["cohens_d"]),
                cohens_d_ci_lower=float(row["cohens_d_ci_lower"]),
                cohens_d_ci_upper=float(row["cohens_d_ci_upper"]),
                cohens_d_magnitude=str(row["cohens_d_magnitude"]),
                hedges_g=float(row["hedges_g"]),
                hedges_g_ci_lower=float(row["hedges_g_ci_lower"]),
                hedges_g_ci_upper=float(row["hedges_g_ci_upper"]),
                a12=float(row["a12"]),
                a12_ci_lower=float(row["a12_ci_lower"]),
                a12_ci_upper=float(row["a12_ci_upper"]),
                a12_magnitude=str(row["a12_magnitude"]),
                cliffs_delta=float(row["cliffs_delta"]),
                cliffs_delta_ci_lower=float(row["cliffs_delta_ci_lower"]),
                cliffs_delta_ci_upper=float(row["cliffs_delta_ci_upper"]),
                p_value=float(row["p_value"]),
                p_adjusted=float(p_adj),
                is_significant=bool(p_adj < alpha),
            )
        )
    items.sort(key=lambda it: it.hedges_g, reverse=True)
    return EffectSizeForestResult(
        metric=metric,
        scope=resolved_scope,
        instance=resolved_inst,
        reference_algorithm=ref_alg,
        maximize=maximize,
        alpha=alpha,
        paired=effective_paired,
        p_adjust=p_adjust,
        items=items,
        warnings=tuple(warnings_list),
    )
