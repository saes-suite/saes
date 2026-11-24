"""Non-parametric hypothesis tests, effect sizes (Vargha-Delaney A12, Cliff's delta), and p-value corrections."""

from __future__ import annotations

import numpy as np
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import EffectSize, StatisticalResult
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.base import StatisticalTest
from saes.stats.effect_sizes import (
    classify_epsilon_squared,
    classify_vargha_delaney_a12,
    compute_cliffs_delta_with_ci,
    compute_vargha_delaney_a12_with_ci,
    epsilon_squared_ci,
)
from saes.stats.posthoc import adjust_pvalues
from saes.stats.ttest import _compute_group_summary


def compute_vargha_delaney_a12(
    x1: np.ndarray,
    x2: np.ndarray,
    maximize: bool = True,
) -> tuple[float, str]:
    """Compute Vargha-Delaney A12 non-parametric stochastic superiority effect size.

    A12 measures P(X1 > X2) + 0.5 * P(X1 == X2) when maximize=True (or reversed when maximize=False).
    """
    a12, mag, _ = compute_vargha_delaney_a12_with_ci(x1, x2, maximize=maximize)
    return a12, mag


def compute_cliffs_delta(
    x1: np.ndarray,
    x2: np.ndarray,
    maximize: bool = True,
) -> tuple[float, str]:
    """Compute Cliff's delta = 2 * A12 - 1 in [-1, 1]."""
    delta, mag, _ = compute_cliffs_delta_with_ci(x1, x2, maximize=maximize)
    return delta, mag


class WilcoxonSignedRankTest(StatisticalTest):
    """Wilcoxon Signed-Rank non-parametric paired two-sample test."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"WilcoxonSignedRankTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        if len(x1) != len(x2):
            raise StatisticalAnalysisError(
                f"WilcoxonSignedRankTest requires equal paired sample sizes. Got N1={len(x1)}, N2={len(x2)}."
            )

        n = len(x1)
        if n < 2:
            raise StatisticalAnalysisError("WilcoxonSignedRankTest requires at least 2 paired observations.")

        diffs = x1 - x2
        normality_res = NormalityChecker(alpha=alpha).check({"Paired Differences": diffs})
        assumptions = tuple(normality_res)

        warnings_list: list[str] = []
        if n < 6:
            warnings_list.append(
                f"Small paired sample size (N={n}). Exact Wilcoxon Signed-Rank critical values have limited power."
            )

        if np.all(diffs == 0):
            w_stat = 0.0
            p_val = 1.0
        else:
            try:
                res = stats.wilcoxon(x1, x2, zero_method="zsplit", alternative="two-sided")
                w_stat = float(res.statistic)
                p_val = float(res.pvalue)
            except Exception:
                w_stat = 0.0
                p_val = 1.0

        # A12 = P(X1 > X2) + 0.5*P(X1 == X2) with Brunner-Munzel confidence interval
        a12, a12_mag, (a12_lo, a12_hi) = compute_vargha_delaney_a12_with_ci(x1, x2, maximize=True, alpha=alpha)
        effect_size = EffectSize(
            name="Vargha-Delaney A12",
            value=a12,
            ci_lower=a12_lo,
            ci_upper=a12_hi,
            magnitude=a12_mag,
        )

        is_sig = p_val < alpha
        med1, med2 = float(np.median(x1)), float(np.median(x2))
        interp = (
            f"Statistically significant paired non-parametric difference detected between '{g1_name}' "
            f"(median={med1:.4g}) and '{g2_name}' (median={med2:.4g}) via Wilcoxon Signed-Rank test "
            f"(W = {w_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {a12:.4f} (95% CI [{a12_lo:.4f}, {a12_hi:.4f}], {a12_mag})."
            if is_sig
            else f"No statistically significant paired non-parametric difference detected between '{g1_name}' "
            f"and '{g2_name}' via Wilcoxon Signed-Rank test (W = {w_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=w_stat,
            p_value=p_val,
            degrees_of_freedom=float(n),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )


class MannWhitneyUTest(StatisticalTest):
    """Mann-Whitney U (Wilcoxon Rank-Sum) non-parametric two-sample independent test."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"MannWhitneyUTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        n1, n2 = len(x1), len(x2)
        if n1 < 2 or n2 < 2:
            raise StatisticalAnalysisError("Each group must contain at least 2 observations for MannWhitneyUTest.")

        normality_res = NormalityChecker(alpha=alpha).check(groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        res = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)

        a12, a12_mag, (a12_lo, a12_hi) = compute_vargha_delaney_a12_with_ci(x1, x2, maximize=True, alpha=alpha)
        effect_size = EffectSize(
            name="Vargha-Delaney A12",
            value=a12,
            ci_lower=a12_lo,
            ci_upper=a12_hi,
            magnitude=a12_mag,
        )

        is_sig = p_val < alpha
        med1, med2 = float(np.median(x1)), float(np.median(x2))
        interp = (
            f"Statistically significant non-parametric difference detected between '{g1_name}' "
            f"(median={med1:.4g}) and '{g2_name}' (median={med2:.4g}) via Mann-Whitney U test "
            f"(U = {u_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {a12:.4f} (95% CI [{a12_lo:.4f}, {a12_hi:.4f}], {a12_mag})."
            if is_sig
            else f"No statistically significant non-parametric difference detected between '{g1_name}' "
            f"and '{g2_name}' via Mann-Whitney U test (U = {u_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Mann-Whitney U Test (Wilcoxon Rank-Sum)",
            statistic=u_stat,
            p_value=p_val,
            degrees_of_freedom=(float(n1), float(n2)),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=(),
            interpretation=interp,
            group_summaries=summaries,
        )


class KruskalWallisTest(StatisticalTest):
    """Kruskal-Wallis H non-parametric omnibus test across 2 or more independent groups."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) < 2:
            raise StatisticalAnalysisError(f"KruskalWallisTest requires at least 2 groups, got {len(groups)}.")

        clean_groups = {k: v[~np.isnan(v)] for k, v in groups.items()}
        for grp_name, v in clean_groups.items():
            if len(v) < 2:
                raise StatisticalAnalysisError(
                    f"Group '{grp_name}' has only {len(v)} observations (minimum 2 required)."
                )

        normality_res = NormalityChecker(alpha=alpha).check(clean_groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(clean_groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        group_list = list(clean_groups.values())
        k = len(group_list)
        n_total = sum(len(g) for g in group_list)
        df_between = float(k - 1)

        all_vals = np.concatenate(group_list)
        if np.all(all_vals == all_vals[0]):
            h_stat = 0.0
            p_val = 1.0
            eps_sq = 0.0
            eps_lo, eps_hi = 0.0, 0.0
        else:
            try:
                res = stats.kruskal(*group_list)
                h_stat = float(res.statistic)
                p_val = float(res.pvalue)
                eps_sq = float(h_stat / float(n_total - 1)) if n_total > 1 else 0.0
                eps_lo, eps_hi = epsilon_squared_ci(group_list, alpha=alpha, n_bootstrap=300, seed=42)
            except Exception:
                h_stat = 0.0
                p_val = 1.0
                eps_sq = 0.0
                eps_lo, eps_hi = 0.0, 0.0

        effect_size = EffectSize(
            name="Epsilon-squared (ε²)",
            value=eps_sq,
            ci_lower=eps_lo,
            ci_upper=eps_hi,
            magnitude=classify_epsilon_squared(eps_sq),
        )

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant non-parametric difference detected among {k} groups via Kruskal-Wallis H test "
            f"(H({df_between:.0f}) = {h_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {eps_sq:.4f} (95% CI [{eps_lo:.4f}, {eps_hi:.4f}], "
            f"{effect_size.magnitude})."
            if is_sig
            else f"No statistically significant difference detected among {k} groups via Kruskal-Wallis H test "
            f"(H({df_between:.0f}) = {h_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {name: _compute_group_summary(data) for name, data in clean_groups.items()}

        return StatisticalResult(
            test_name="Kruskal-Wallis H Test",
            statistic=h_stat,
            p_value=p_val,
            degrees_of_freedom=df_between,
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=(),
            interpretation=interp,
            group_summaries=summaries,
        )


__all__ = [
    "classify_vargha_delaney_a12",
    "compute_vargha_delaney_a12",
    "compute_vargha_delaney_a12_with_ci",
    "compute_cliffs_delta",
    "compute_cliffs_delta_with_ci",
    "adjust_pvalues",
    "WilcoxonSignedRankTest",
    "MannWhitneyUTest",
    "KruskalWallisTest",
]

