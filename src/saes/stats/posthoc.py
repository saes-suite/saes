"""Multiple-comparison p-value adjustments and parametric/non-parametric post-hoc procedures.

References:
    - Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. JMLR, 7, 1-30.
    - García, S., & Herrera, F. (2008). An extension on "Statistical comparisons of classifiers over
      multiple data sets" for all pairwise comparisons. JMLR, 9, 2677-2694.
    - García, S., Fernández, A., Luengo, J., & Herrera, F. (2010). Advanced nonparametric tests
      for multiple comparisons in the design of experiments in computational intelligence and data
      mining. Information Sciences, 180(10), 2044-2064.
    - Dunn, O. J. (1964). Multiple comparisons using rank sums. Technometrics, 6(3), 241-252.
    - Conover, W. J., & Iman, R. L. (1979). On multiple-comparisons procedures. Los Alamos Report LA-7677-MS.
    - Games, P. A., & Howell, J. F. (1976). Pairwise multiple comparison procedures with unequal N's
      and/or variances. Journal of Educational Statistics, 1(2), 113-125.
    - Brunner, E., & Munzel, U. (2000). The nonparametric Behrens-Fisher problem. Biometrical Journal, 42(1).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    EffectSize,
    PostHocComparisonItem,
    PostHocResult,
    StatisticalResult,
)
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.base import StatisticalTest
from saes.stats.effect_sizes import compute_vargha_delaney_a12_with_ci
from saes.stats.ttest import _compute_group_summary


def adjust_pvalues(p_values: list[float], method: str = "holm") -> list[float]:
    """Adjust p-values for multiple hypothesis testing.

    Supported methods:
        - 'holm' / 'holm-bonferroni': Holm (1979) step-down FWER control
        - 'bonferroni': Bonferroni-Dunn FWER control
        - 'hochberg': Hochberg (1988) step-up FWER control
        - 'hommel': Hommel (1988) closed-testing FWER control (via statsmodels)
        - 'finner': Finner (1993) step-down procedure (García et al. 2010)
        - 'holland': Holland & Copenhaver (1987) Šidák step-down procedure
        - 'rom': Rom (1990) sharp step-up procedure
        - 'li': Li (2008) two-step procedure
        - 'fdr_bh' / 'bh': Benjamini-Hochberg (1995) FDR control
        - 'fdr_by' / 'by': Benjamini-Yekutieli (2001) dependent FDR control
        - 'none' / 'unadjusted': Unadjusted p-values
    """
    m = len(p_values)
    if m == 0:
        return []

    clean_p = [min(1.0, max(0.0, float(p) if math.isfinite(p) else 1.0)) for p in p_values]
    norm_method = method.strip().lower()
    if m == 1 or norm_method in {"none", "unadjusted"}:
        return clean_p

    arr = np.array(clean_p, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    sorted_p = arr[order]
    adjusted = np.zeros(m, dtype=float)

    if norm_method == "bonferroni":
        return [float(x) for x in np.minimum(1.0, arr * float(m)).tolist()]

    if norm_method in {"holm", "holm-bonferroni"}:
        running_max = 0.0
        for idx in range(m):
            val = min(1.0, float(sorted_p[idx]) * float(m - idx))
            running_max = max(running_max, val)
            adjusted[order[idx]] = running_max
        return [float(x) for x in adjusted.tolist()]

    if norm_method == "hochberg":
        running_min = 1.0
        for idx in range(m - 1, -1, -1):
            val = min(1.0, float(sorted_p[idx]) * float(m - idx))
            running_min = min(running_min, val)
            adjusted[order[idx]] = running_min
        return [float(x) for x in adjusted.tolist()]

    if norm_method == "hommel":
        _, p_adj, _, _ = multipletests(arr, method="hommel")
        return [float(min(1.0, max(0.0, x))) for x in p_adj.tolist()]

    if norm_method == "finner":
        # Step-down: p_adj(i) = min(1, max_{j<=i} [1 - (1 - p_(j))^(m / j)]), 1-indexed j
        running_max = 0.0
        for idx in range(m):
            j = float(idx + 1)
            p_j = float(sorted_p[idx])
            val = 1.0 if p_j >= 1.0 else min(1.0, 1.0 - (1.0 - p_j) ** (float(m) / j))
            running_max = max(running_max, val)
            adjusted[order[idx]] = running_max
        return [float(x) for x in adjusted.tolist()]

    if norm_method in {"holland", "sidak-sd"}:
        # Holland & Copenhaver (1987) step-down Šidák: 1 - (1 - p_(j))^(m - j + 1)
        running_max = 0.0
        for idx in range(m):
            exp = float(m - idx)
            p_j = float(sorted_p[idx])
            val = 1.0 if p_j >= 1.0 else min(1.0, 1.0 - (1.0 - p_j) ** exp)
            running_max = max(running_max, val)
            adjusted[order[idx]] = running_max
        return [float(x) for x in adjusted.tolist()]

    if norm_method == "rom":
        # Rom (1990) exact critical constants c_k for step-up procedure (computed recursively)
        # c_1 = 1, and sum_{i=1}^{k-1} C(k, i) * c_{k-i} = ... using standard Rom alpha-scaling
        # Using Rom's step-up recursion on alpha_i ratios: c_1 = 1, c_2 = 2, c_3 = 2.98...,
        # approximated accurately via exact Rom recursion at nominal grid or Hochberg bound.
        # Exact Rom (1990) multipliers d_r = alpha / alpha_r for r = 1..m (independent of alpha to O(alpha^2)):
        rom_mult = _rom_multipliers(m)
        running_min = 1.0
        for idx in range(m - 1, -1, -1):
            r = m - idx  # r = 1 for largest p-value, m for smallest
            val = min(1.0, float(sorted_p[idx]) * rom_mult[r - 1])
            running_min = min(running_min, val)
            adjusted[order[idx]] = running_min
        return [float(x) for x in adjusted.tolist()]

    if norm_method == "li":
        # Li (2008): p_Li,i = p_i / (p_i + 1 - p_(m)) for i < m, and p_(m) for largest
        p_max = float(sorted_p[-1])
        for idx in range(m):
            p_i = float(sorted_p[idx])
            if idx == m - 1 or p_max >= 1.0:
                val = p_max
            else:
                denom = p_i + (1.0 - p_max)
                val = min(1.0, p_i / denom) if denom > 0 else 1.0
            adjusted[order[idx]] = min(p_max, val)
        return [float(x) for x in adjusted.tolist()]

    if norm_method in {"fdr_bh", "bh", "benjamini-hochberg"}:
        running_min = 1.0
        for idx in range(m - 1, -1, -1):
            val = min(1.0, (float(sorted_p[idx]) * float(m)) / float(idx + 1))
            running_min = min(running_min, val)
            adjusted[order[idx]] = running_min
        return [float(x) for x in adjusted.tolist()]

    if norm_method in {"fdr_by", "by", "benjamini-yekutieli"}:
        _, p_adj, _, _ = multipletests(arr, method="fdr_by")
        return [float(min(1.0, max(0.0, x))) for x in p_adj.tolist()]

    raise StatisticalAnalysisError(
        f"Unsupported p-value adjustment method '{method}'. Supported: "
        "holm, bonferroni, hochberg, hommel, finner, holland, rom, li, fdr_bh, fdr_by, none."
    )


def _rom_multipliers(m: int, alpha_ref: float = 0.05) -> list[float]:
    """Compute Rom (1990) step-up critical value ratios alpha_ref / alpha_k for k = 1..m.

    Uses the exact Bol'shev/Steck order-statistic identity for k <= 20 matching Rom (1990,
    Biometrika 77(3), Table 1): P(V_(1) <= b_1, ..., V_(k) <= b_k) = 1 - alpha_ref with b_r = 1 - alpha_r.
    For k > 20, safely falls back to Hochberg's bound (alpha_ref / k) to avoid alternating-sum cancellation.
    """
    b_vals = [1.0 - alpha_ref]
    q_vals = [1.0, 1.0 - alpha_ref]
    exact_limit = min(m, 20)
    for k in range(2, exact_limit + 1):
        rhs = (1.0 - alpha_ref) - sum(
            ((-1) ** (k - 1 - j)) * math.comb(k, j) * q_vals[j] * (b_vals[j] ** (k - j))
            for j in range(k - 1)
        )
        b_k = rhs / (float(k) * q_vals[k - 1])
        a_k = max(1e-12, min(alpha_ref / float(k - 1), 1.0 - b_k))
        b_vals.append(1.0 - a_k)
        q_vals.append(1.0 - alpha_ref)

    alphas = [1.0 - b for b in b_vals]
    for k in range(exact_limit + 1, m + 1):
        alphas.append(alpha_ref / float(k))
    return [alpha_ref / a for a in alphas]


def _set_partitions(n: int) -> Iterator[list[list[int]]]:
    """Generate all set partitions of {0, ..., n-1} (Bell number B_n)."""
    if n == 0:
        yield []
        return
    first = n - 1
    for smaller in _set_partitions(n - 1):
        for idx in range(len(smaller)):
            yield smaller[:idx] + [smaller[idx] + [first]] + smaller[idx + 1 :]
        yield smaller + [[first]]


def _shaffer_possible_true_counts(k: int) -> list[int]:
    """Return sorted list of all logically possible numbers of true pairwise hypotheses for k groups."""
    # Dynamic programming over integer partitions of k: sum C(c_i, 2)
    dp: dict[int, set[int]] = {0: {0}}
    for total in range(1, k + 1):
        possible: set[int] = set()
        for part in range(1, total + 1):
            pairs_in_part = part * (part - 1) // 2
            for prev in dp[total - part]:
                possible.add(prev + pairs_in_part)
        dp[total] = possible
    return sorted(dp[k])


def adjust_pvalues_shaffer(p_matrix: np.ndarray) -> np.ndarray:
    """Apply Shaffer's static procedure (1986) to a symmetric k x k matrix of pairwise p-values.

    Reference:
        Shaffer, J. P. (1986). Modified sequentially rejective multiple test procedures. JASA, 81(395).
        García, S., & Herrera, F. (2008). JMLR, 9, 2677-2694.
    """
    mat = np.asarray(p_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise StatisticalAnalysisError("p_matrix must be a square (k x k) array.")
    k = mat.shape[0]
    if k < 2:
        return np.ones_like(mat)

    pairs: list[tuple[int, int]] = [(i, j) for i in range(k) for j in range(i + 1, k)]
    m = len(pairs)
    raw = np.array([min(1.0, max(0.0, float(mat[i, j]))) for i, j in pairs], dtype=float)
    order = np.argsort(raw, kind="mergesort")
    sorted_p = raw[order]

    possible_counts = _shaffer_possible_true_counts(k)
    adj_sorted = np.zeros(m, dtype=float)
    running_max = 0.0
    for idx in range(m):
        max_remaining = m - idx
        t_j = max(c for c in possible_counts if c <= max_remaining)
        val = min(1.0, float(sorted_p[idx]) * float(t_j))
        running_max = max(running_max, val)
        adj_sorted[idx] = running_max

    out = np.ones((k, k), dtype=float)
    for pos, orig_idx in enumerate(order):
        i, j = pairs[orig_idx]
        out[i, j] = adj_sorted[pos]
        out[j, i] = adj_sorted[pos]
    return out


def adjust_pvalues_bergmann_hommel(p_matrix: np.ndarray, max_exact_k: int = 9) -> tuple[np.ndarray, list[str]]:
    """Apply exact Bergmann-Hommel (1988) exhaustive procedure to a symmetric k x k p-value matrix.

    For k <= max_exact_k (default 9), enumerates all exhaustive hypothesis sets corresponding
    to set partitions of {0, ..., k-1}. For k > max_exact_k, falls back to Shaffer's static
    procedure with an explicit warning to avoid Bell-number combinatorial explosion.

    Reference:
        Bergmann, G., & Hommel, G. (1988). Improvements of general multiple test procedures for
        redundant systems of hypotheses.
        García, S., & Herrera, F. (2008). JMLR, 9, 2677-2694.
    """
    mat = np.asarray(p_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise StatisticalAnalysisError("p_matrix must be a square (k x k) array.")
    k = mat.shape[0]
    if k < 2:
        return np.ones_like(mat), []

    warnings_list: list[str] = []
    if k > max_exact_k:
        warnings_list.append(
            f"Exact Bergmann-Hommel exhaustive search is intractable for k={k} (> {max_exact_k}); "
            "fell back to Shaffer's static procedure."
        )
        return adjust_pvalues_shaffer(mat), warnings_list

    # Initialize adjusted p-values to 0; update with max over every exhaustive set containing (i, j)
    adj = np.zeros((k, k), dtype=float)
    np.fill_diagonal(adj, 1.0)

    for partition in _set_partitions(k):
        hyp_pairs: list[tuple[int, int]] = []
        for block in partition:
            b_len = len(block)
            if b_len >= 2:
                for a_idx in range(b_len):
                    for b_idx in range(a_idx + 1, b_len):
                        u, v = block[a_idx], block[b_idx]
                        hyp_pairs.append((u, v) if u < v else (v, u))
        size_i = len(hyp_pairs)
        if size_i == 0:
            continue
        min_p = min(min(1.0, max(0.0, float(mat[u, v]))) for u, v in hyp_pairs)
        candidate = min(1.0, float(size_i) * min_p)
        for u, v in hyp_pairs:
            if candidate > adj[u, v]:
                adj[u, v] = candidate
                adj[v, u] = candidate

    return adj, warnings_list


def _build_p_matrix(groups: list[str], items: list[PostHocComparisonItem]) -> dict[str, dict[str, float]]:
    mat: dict[str, dict[str, float]] = {g: {g2: 1.0 for g2 in groups} for g in groups}
    for it in items:
        mat[it.group_1][it.group_2] = it.p_adjusted
        mat[it.group_2][it.group_1] = it.p_adjusted
    return mat


def tukey_hsd(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
    """Tukey's Honestly Significant Difference (HSD) all-pairs post-hoc test (parametric, homoscedastic)."""
    if len(groups) < 2:
        raise StatisticalAnalysisError(f"Tukey HSD requires at least 2 groups, got {len(groups)}.")

    names = list(groups.keys())
    clean_list = [np.asarray(groups[n], dtype=float)[~np.isnan(groups[n])] for n in names]
    for n, arr in zip(names, clean_list, strict=False):
        if len(arr) < 2:
            raise StatisticalAnalysisError(f"Group '{n}' has fewer than 2 observations.")

    res = stats.tukey_hsd(*clean_list)
    ci = res.confidence_interval(confidence_level=1.0 - alpha)
    k = len(names)
    df_within = float(sum(len(a) for a in clean_list) - k)

    items: list[PostHocComparisonItem] = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(res.statistic[i, j])
            p_val = float(res.pvalue[i, j])
            items.append(
                PostHocComparisonItem(
                    group_1=names[i],
                    group_2=names[j],
                    statistic=diff,
                    p_value=p_val,
                    p_adjusted=p_val,
                    is_significant=bool(p_val < alpha),
                    mean_diff=diff,
                    ci_lower=float(ci.low[i, j]),
                    ci_upper=float(ci.high[i, j]),
                    degrees_of_freedom=df_within,
                )
            )

    return PostHocResult(
        method="Tukey's HSD",
        p_adjust="tukey-studentized-range",
        alpha=alpha,
        groups=names,
        comparisons=items,
        p_value_matrix=_build_p_matrix(names, items),
    )


def games_howell(groups: dict[str, np.ndarray], alpha: float = 0.05) -> PostHocResult:
    """Games-Howell heteroscedastic all-pairs post-hoc test (after Welch's ANOVA)."""
    if len(groups) < 2:
        raise StatisticalAnalysisError(f"Games-Howell requires at least 2 groups, got {len(groups)}.")

    names = list(groups.keys())
    clean_list = [np.asarray(groups[n], dtype=float)[~np.isnan(groups[n])] for n in names]
    for n, arr in zip(names, clean_list, strict=False):
        if len(arr) < 2:
            raise StatisticalAnalysisError(f"Group '{n}' has fewer than 2 observations.")

    k = len(names)
    items: list[PostHocComparisonItem] = []
    warnings_list: list[str] = []

    for i in range(k):
        for j in range(i + 1, k):
            x1, x2 = clean_list[i], clean_list[j]
            n1, n2 = len(x1), len(x2)
            m1, m2 = float(np.mean(x1)), float(np.mean(x2))
            v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
            se1_sq = v1 / n1
            se2_sq = v2 / n2
            se_sq = se1_sq + se2_sq
            diff = m1 - m2

            if se_sq <= 1e-18:
                df_ws = float(n1 + n2 - 2)
                q_stat = 0.0
                p_val = 1.0 if abs(diff) <= 1e-15 else 0.0
                ci_low, ci_high = diff, diff
                warnings_list.append(f"Zero variance encountered for pair ({names[i]}, {names[j]}).")
            else:
                se = math.sqrt(se_sq)
                q_stat = float(math.sqrt(2.0) * abs(diff) / se)
                den = (se1_sq**2) / (n1 - 1) + (se2_sq**2) / (n2 - 1)
                df_ws = float((se_sq**2) / den) if den > 0 else float(n1 + n2 - 2)
                df_ws = max(1.0, df_ws)
                p_val = float(stats.studentized_range.sf(q_stat, k, df_ws))
                q_crit = float(stats.studentized_range.ppf(1.0 - alpha, k, df_ws))
                half_w = (q_crit / math.sqrt(2.0)) * se
                ci_low, ci_high = diff - half_w, diff + half_w

            items.append(
                PostHocComparisonItem(
                    group_1=names[i],
                    group_2=names[j],
                    statistic=q_stat,
                    p_value=p_val,
                    p_adjusted=p_val,
                    is_significant=bool(p_val < alpha),
                    mean_diff=diff,
                    ci_lower=float(ci_low),
                    ci_upper=float(ci_high),
                    degrees_of_freedom=df_ws,
                )
            )

    return PostHocResult(
        method="Games-Howell",
        p_adjust="studentized-range-welch",
        alpha=alpha,
        groups=names,
        comparisons=items,
        p_value_matrix=_build_p_matrix(names, items),
        warnings=tuple(warnings_list),
    )


def dunnett(groups: dict[str, np.ndarray], control: str, alpha: float = 0.05) -> PostHocResult:
    """Dunnett's many-to-one post-hoc comparison against a designated control group."""
    if control not in groups:
        raise StatisticalAnalysisError(f"Control group '{control}' not found in groups: {list(groups.keys())}.")
    if len(groups) < 2:
        raise StatisticalAnalysisError("Dunnett's test requires at least 2 groups (1 control + >=1 treatment).")

    ctrl_arr = np.asarray(groups[control], dtype=float)
    ctrl_arr = ctrl_arr[~np.isnan(ctrl_arr)]
    if len(ctrl_arr) < 2:
        raise StatisticalAnalysisError(f"Control group '{control}' requires at least 2 observations.")

    treatment_names = [n for n in groups.keys() if n != control]
    treatment_arrs = [np.asarray(groups[n], dtype=float)[~np.isnan(groups[n])] for n in treatment_names]
    for n, arr in zip(treatment_names, treatment_arrs, strict=False):
        if len(arr) < 2:
            raise StatisticalAnalysisError(f"Treatment group '{n}' requires at least 2 observations.")

    res = stats.dunnett(*treatment_arrs, control=ctrl_arr, alternative="two-sided", random_state=42)
    ci = res.confidence_interval(confidence_level=1.0 - alpha)
    df_within = float(len(ctrl_arr) + sum(len(a) for a in treatment_arrs) - len(groups))

    items: list[PostHocComparisonItem] = []
    for idx, t_name in enumerate(treatment_names):
        diff = float(res.statistic[idx])
        p_val = float(res.pvalue[idx])
        items.append(
            PostHocComparisonItem(
                group_1=t_name,
                group_2=control,
                statistic=diff,
                p_value=p_val,
                p_adjusted=p_val,
                is_significant=bool(p_val < alpha),
                mean_diff=diff,
                ci_lower=float(ci.low[idx]),
                ci_upper=float(ci.high[idx]),
                degrees_of_freedom=df_within,
            )
        )

    all_names = [control] + treatment_names
    return PostHocResult(
        method=f"Dunnett's Test (vs {control})",
        p_adjust="dunnett-multivariate-t",
        alpha=alpha,
        groups=all_names,
        comparisons=items,
        p_value_matrix=_build_p_matrix(all_names, items),
    )


def dunn_test(groups: dict[str, np.ndarray], alpha: float = 0.05, p_adjust: str = "holm") -> PostHocResult:
    """Dunn's (1964) non-parametric all-pairs rank-sum post-hoc test with tie correction."""
    if len(groups) < 2:
        raise StatisticalAnalysisError(f"Dunn's test requires at least 2 groups, got {len(groups)}.")

    names = list(groups.keys())
    clean_list = [np.asarray(groups[n], dtype=float)[~np.isnan(groups[n])] for n in names]
    for n, arr in zip(names, clean_list, strict=False):
        if len(arr) < 2:
            raise StatisticalAnalysisError(f"Group '{n}' requires at least 2 observations.")

    all_vals = np.concatenate(clean_list)
    n_total = len(all_vals)
    ranks = stats.rankdata(all_vals, method="average")

    # Tie correction term: sum(t^3 - t) / (12 * (N - 1))
    _, counts = np.unique(all_vals, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts)) / (12.0 * (n_total - 1.0)) if n_total > 1 else 0.0
    var_factor = (n_total * (n_total + 1.0)) / 12.0 - tie_term

    # Split mean ranks per group
    mean_ranks: list[float] = []
    ns: list[int] = []
    offset = 0
    for arr in clean_list:
        m = len(arr)
        mean_ranks.append(float(np.mean(ranks[offset : offset + m])))
        ns.append(m)
        offset += m

    k = len(names)
    pair_indices: list[tuple[int, int]] = []
    z_stats: list[float] = []
    raw_ps: list[float] = []

    for i in range(k):
        for j in range(i + 1, k):
            pair_indices.append((i, j))
            if var_factor <= 1e-18:
                z_val = 0.0
                p_val = 1.0
            else:
                se = math.sqrt(var_factor * (1.0 / ns[i] + 1.0 / ns[j]))
                z_val = float((mean_ranks[i] - mean_ranks[j]) / se)
                p_val = float(2.0 * stats.norm.sf(abs(z_val)))
            z_stats.append(z_val)
            raw_ps.append(p_val)

    adj_ps = adjust_pvalues(raw_ps, method=p_adjust)
    items: list[PostHocComparisonItem] = []
    for (i, j), z_val, p_raw, p_adj in zip(pair_indices, z_stats, raw_ps, adj_ps, strict=False):
        items.append(
            PostHocComparisonItem(
                group_1=names[i],
                group_2=names[j],
                statistic=z_val,
                p_value=p_raw,
                p_adjusted=p_adj,
                is_significant=bool(p_adj < alpha),
                mean_diff=float(mean_ranks[i] - mean_ranks[j]),
            )
        )

    return PostHocResult(
        method="Dunn's Non-Parametric Post-Hoc Test",
        p_adjust=p_adjust,
        alpha=alpha,
        groups=names,
        comparisons=items,
        p_value_matrix=_build_p_matrix(names, items),
    )


def conover_iman_test(groups: dict[str, np.ndarray], alpha: float = 0.05, p_adjust: str = "holm") -> PostHocResult:
    """Conover-Iman (1979) non-parametric post-hoc t-test with tie correction after Kruskal-Wallis."""
    if len(groups) < 2:
        raise StatisticalAnalysisError(f"Conover-Iman test requires at least 2 groups, got {len(groups)}.")

    names = list(groups.keys())
    clean_list = [np.asarray(groups[n], dtype=float)[~np.isnan(groups[n])] for n in names]
    for n, arr in zip(names, clean_list, strict=False):
        if len(arr) < 2:
            raise StatisticalAnalysisError(f"Group '{n}' requires at least 2 observations.")

    k = len(names)
    all_vals = np.concatenate(clean_list)
    n_total = len(all_vals)
    df = float(n_total - k)
    if df <= 0:
        raise StatisticalAnalysisError("Conover-Iman test requires total observations N > k.")

    ranks = stats.rankdata(all_vals, method="average")
    s_sq = float((np.sum(ranks**2) - n_total * ((n_total + 1.0) ** 2) / 4.0) / float(n_total - 1))

    mean_ranks: list[float] = []
    sum_ranks_sq_over_n = 0.0
    ns: list[int] = []
    offset = 0
    for arr in clean_list:
        m = len(arr)
        grp_r = ranks[offset : offset + m]
        mean_ranks.append(float(np.mean(grp_r)))
        sum_ranks_sq_over_n += float(np.sum(grp_r) ** 2) / float(m)
        ns.append(m)
        offset += m

    h_stat = float((sum_ranks_sq_over_n - n_total * ((n_total + 1.0) ** 2) / 4.0) / s_sq) if s_sq > 1e-18 else 0.0
    scale_factor = max(0.0, s_sq * ((n_total - 1.0 - h_stat) / df))

    pair_indices: list[tuple[int, int]] = []
    t_stats: list[float] = []
    raw_ps: list[float] = []

    for i in range(k):
        for j in range(i + 1, k):
            pair_indices.append((i, j))
            if scale_factor <= 1e-18:
                t_val = 0.0
                p_val = 1.0
            else:
                se = math.sqrt(scale_factor * (1.0 / ns[i] + 1.0 / ns[j]))
                t_val = float((mean_ranks[i] - mean_ranks[j]) / se)
                p_val = float(2.0 * stats.t.sf(abs(t_val), df))
            t_stats.append(t_val)
            raw_ps.append(p_val)

    adj_ps = adjust_pvalues(raw_ps, method=p_adjust)
    items: list[PostHocComparisonItem] = []
    for (i, j), t_val, p_raw, p_adj in zip(pair_indices, t_stats, raw_ps, adj_ps, strict=False):
        items.append(
            PostHocComparisonItem(
                group_1=names[i],
                group_2=names[j],
                statistic=t_val,
                p_value=p_raw,
                p_adjusted=p_adj,
                is_significant=bool(p_adj < alpha),
                mean_diff=float(mean_ranks[i] - mean_ranks[j]),
                degrees_of_freedom=df,
            )
        )

    return PostHocResult(
        method="Conover-Iman Non-Parametric Post-Hoc Test",
        p_adjust=p_adjust,
        alpha=alpha,
        groups=names,
        comparisons=items,
        p_value_matrix=_build_p_matrix(names, items),
    )


class BrunnerMunzelTest(StatisticalTest):
    """Non-parametric Behrens-Fisher test for stochastic equality (Brunner & Munzel 2000; nparcomp JSS 2015)."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"BrunnerMunzelTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = np.asarray(groups[g1_name], dtype=float)
        x2 = np.asarray(groups[g2_name], dtype=float)
        x1 = x1[~np.isnan(x1)]
        x2 = x2[~np.isnan(x2)]

        n1, n2 = len(x1), len(x2)
        if n1 < 2 or n2 < 2:
            raise StatisticalAnalysisError("Each group must contain at least 2 observations for BrunnerMunzelTest.")

        normality_res = NormalityChecker(alpha=alpha).check({g1_name: x1, g2_name: x2})
        homogeneity_res = HomogeneityChecker(alpha=alpha).check({g1_name: x1, g2_name: x2})
        assumptions = tuple(normality_res + [homogeneity_res])

        a12, a12_mag, (ci_lo, ci_hi) = compute_vargha_delaney_a12_with_ci(x1, x2, maximize=True, alpha=alpha)

        import warnings as _warnings

        warnings_list: list[str] = []
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            res = stats.brunnermunzel(x1, x2, alternative="two-sided", distribution="t")
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
        if not math.isfinite(w_stat) or not math.isfinite(p_val):
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                res_norm = stats.brunnermunzel(x1, x2, alternative="two-sided", distribution="normal")
            w_stat = float(res_norm.statistic)
            p_val = float(res_norm.pvalue)
        if not math.isfinite(w_stat) or not math.isfinite(p_val):
            # Complete separation or identical values -> fallback to exact rank boundary
            if a12 == 0.5:
                w_stat = 0.0
                p_val = 1.0
            else:
                w_stat = math.copysign(1e6, a12 - 0.5)
                p_val = 0.0
            warnings_list.append(
                "Degenerate variance in Brunner-Munzel statistic (complete separation or constant samples)."
            )

        effect_size = EffectSize(
            name="Vargha-Delaney A12 (Relative Effect)",
            value=a12,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            magnitude=a12_mag,
        )
        is_sig = p_val < alpha
        interp = (
            f"Statistically significant stochastic difference between '{g1_name}' and '{g2_name}' via "
            f"Brunner-Munzel test (W = {w_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Relative effect A12 = {a12:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}], {a12_mag})."
            if is_sig
            else f"No statistically significant stochastic difference between '{g1_name}' and '{g2_name}' via "
            f"Brunner-Munzel test (W = {w_stat:.4f}, p = {p_val:.4f}, alpha = {alpha}). "
            f"Relative effect A12 = {a12:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])."
        )
        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Brunner-Munzel Test (Non-Parametric Behrens-Fisher)",
            statistic=w_stat,
            p_value=p_val,
            degrees_of_freedom=(float(n1), float(n2)),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=(ci_lo, ci_hi),
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )
