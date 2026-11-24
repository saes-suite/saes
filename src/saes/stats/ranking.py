"""Multi-algorithm blocked omnibus tests & Bayesian ranking procedures.

Implements:
    - Friedman omnibus test with exact tie correction, Iman-Davenport F, and Nemenyi CD (Demšar 2006)
    - Friedman Aligned-Ranks test (Hodges & Lehmann 1962; García et al. 2010)
    - Quade weighted-block rank test (Quade 1979; García et al. 2010)
    - Repeated-Measures One-Way ANOVA with Greenhouse-Geisser sphericity correction
    - Exact Bayesian Sign & Signed-Rank Simplex ROPE tests (Benavoli et al. 2017)
    - Re-exports BayesianCorrelatedTTest & BayesianPlackettLuceTest
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    AlgorithmRankItem,
    BayesianPosteriorItem,
    BayesianRankResult,
    BlockedOmnibusResult,
    CriticalDistanceResult,
)
from saes.stats.bayesian import BayesianCorrelatedTTest, BayesianPlackettLuceTest

# Studentized range statistic q_alpha / sqrt(2) critical values for two-tailed Nemenyi test
# (Demšar 2006, Table 5a; exact studentized_range via SciPy for arbitrary k and alpha)
_NEMENYI_Q_ALPHA: dict[float, dict[int, float]] = {
    0.01: {
        2: 2.576,
        3: 2.913,
        4: 3.113,
        5: 3.255,
        6: 3.364,
        7: 3.452,
        8: 3.526,
        9: 3.590,
        10: 3.646,
    },
    0.05: {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    },
    0.10: {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    },
}


def _get_nemenyi_q_alpha(k: int, alpha: float) -> float:
    """Return critical value q_alpha for Nemenyi two-tailed test."""
    try:
        q_exact = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0))
        if math.isfinite(q_exact) and q_exact > 0:
            return q_exact
    except Exception:
        pass

    alpha_key = 0.01 if alpha <= 0.01 else (0.10 if alpha >= 0.10 else 0.05)
    table = _NEMENYI_Q_ALPHA.get(alpha_key, _NEMENYI_Q_ALPHA[0.05])
    clamped_k = min(10, max(2, k))
    return table[clamped_k]


def _extract_block_matrix(
    df: pd.DataFrame,
    metric: str,
    scope: str = "all-instances",
    instance: str | None = None,
) -> tuple[list[str], np.ndarray, bool, str, list[str], list[str], str, str, list[str]]:
    """Extract a complete (n_blocks x k_algorithms) matrix of observations for blocked ranking.

    When `metric="all-metrics"` across multiple metrics, per-metric normalized scores are
    averaged *within each instance* (or run) so that exactly one independent block per
    problem instance is preserved (avoiding artificial N inflation from correlated metrics).

    Returns:
        (algorithms, matrix [n_blocks, k], maximize, scope_label,
         metrics_list, instances_list, metrics_label, instances_label, warnings_list)
    """
    all_available_metrics = sorted(df["metric"].unique().tolist())
    if metric.lower() in {"all-metrics", "all", "*"}:
        target_metrics = all_available_metrics
    else:
        target_metrics = [metric]

    metric_df = df[df["metric"].isin(target_metrics)].copy()
    if metric_df.empty:
        raise StatisticalAnalysisError(f"No observations found for metric '{metric}'.")

    is_multi_metric = len(target_metrics) > 1
    maximize = True if is_multi_metric else bool(metric_df["maximize"].iloc[0])
    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    k = len(algorithms)
    if k < 2:
        raise StatisticalAnalysisError(f"Ranking tests require at least 2 algorithms, got {k}.")

    all_instances = sorted(metric_df["instance"].unique().tolist())
    use_single_instance = scope == "single-instance" or (len(all_instances) == 1 and instance is not None)

    if not use_single_instance and len(all_instances) > 1:
        target_instances = all_instances
    else:
        resolved_inst = instance if (instance and instance in all_instances) else all_instances[0]
        target_instances = [resolved_inst]

    warnings_list: list[str] = []
    rows: list[list[float]] = []

    if is_multi_metric:
        warnings_list.append(
            f"Multi-metric composite ranking averages oriented normalized scores across {len(target_metrics)} "
            "metrics within each instance/block to preserve block independence (N = number of instances)."
        )
        if not use_single_instance and len(all_instances) > 1:
            for inst in target_instances:
                inst_df = metric_df[metric_df["instance"] == inst]
                per_metric_normed: list[np.ndarray] = []
                for m_name in target_metrics:
                    m_sub = inst_df[inst_df["metric"] == m_name]
                    if m_sub.empty:
                        continue
                    m_max = bool(m_sub["maximize"].iloc[0])
                    block_vals: list[float] = []
                    for algo in algorithms:
                        vals = m_sub[m_sub["algorithm"] == algo]["value"].dropna().to_numpy(dtype=float)
                        if len(vals) > 0:
                            block_vals.append(float(np.mean(vals)))
                    if len(block_vals) == k:
                        arr = np.array(block_vals, dtype=float)
                        oriented = arr if m_max else -arr
                        b_min, b_max = float(np.min(oriented)), float(np.max(oriented))
                        span = b_max - b_min
                        if span > 1e-12:
                            normed = 1.0 + (oriented - b_min) / span
                        else:
                            normed = np.ones(k, dtype=float)
                        per_metric_normed.append(normed)
                if per_metric_normed:
                    composite = np.mean(np.vstack(per_metric_normed), axis=0)
                    rows.append(composite.tolist())
        else:
            target_inst = target_instances[0]
            inst_df = metric_df[metric_df["instance"] == target_inst]
            run_ids = sorted(inst_df["run_id"].unique().tolist())
            for run_id in run_ids:
                run_df = inst_df[inst_df["run_id"] == run_id]
                per_metric_normed = []
                for m_name in target_metrics:
                    m_sub = run_df[run_df["metric"] == m_name]
                    if m_sub.empty:
                        continue
                    m_max = bool(m_sub["maximize"].iloc[0])
                    block_vals = []
                    for algo in algorithms:
                        vals = m_sub[m_sub["algorithm"] == algo]["value"].dropna().to_numpy(dtype=float)
                        if len(vals) > 0:
                            block_vals.append(float(vals[0]))
                    if len(block_vals) == k:
                        arr = np.array(block_vals, dtype=float)
                        oriented = arr if m_max else -arr
                        b_min, b_max = float(np.min(oriented)), float(np.max(oriented))
                        span = b_max - b_min
                        normed = 1.0 + (oriented - b_min) / span if span > 1e-12 else np.ones(k, dtype=float)
                        per_metric_normed.append(normed)
                if per_metric_normed:
                    composite = np.mean(np.vstack(per_metric_normed), axis=0)
                    rows.append(composite.tolist())
    else:
        m_sub = metric_df[metric_df["metric"] == target_metrics[0]]
        if not use_single_instance and len(all_instances) > 1:
            for inst in target_instances:
                inst_df = m_sub[m_sub["instance"] == inst]
                block_vals = []
                for algo in algorithms:
                    vals = inst_df[inst_df["algorithm"] == algo]["value"].dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        block_vals.append(float(np.mean(vals)))
                if len(block_vals) == k:
                    rows.append(block_vals)
        else:
            target_inst = target_instances[0]
            inst_df = m_sub[m_sub["instance"] == target_inst]
            run_ids = sorted(inst_df["run_id"].unique().tolist())
            for run_id in run_ids:
                run_df = inst_df[inst_df["run_id"] == run_id]
                block_vals = []
                for algo in algorithms:
                    vals = run_df[run_df["algorithm"] == algo]["value"].dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        block_vals.append(float(vals[0]))
                if len(block_vals) == k:
                    rows.append(block_vals)

    if not rows:
        raise StatisticalAnalysisError(f"No complete blocks found across all {k} algorithms for metric '{metric}'.")

    if len(target_metrics) == 1:
        metrics_label = f"Metric: {target_metrics[0]}"
    else:
        metrics_label = f"Metrics ({len(target_metrics)}): {', '.join(target_metrics)}"

    if len(target_instances) == 1:
        instances_label = f"Instance: {target_instances[0]} ({len(rows)} Blocks)"
        scope_label = f"Instance: {target_instances[0]} ({len(rows)} Runs)"
    else:
        inst_str = (
            ", ".join(target_instances)
            if len(target_instances) <= 6
            else f"{', '.join(target_instances[:5])}, …, {target_instances[-1]}"
        )
        instances_label = f"Instances ({len(target_instances)}): {inst_str}"
        composite_tag = " Composite" if is_multi_metric else ""
        scope_label = f"Across {len(target_instances)} Instances ({len(rows)}{composite_tag} Blocks)"

    matrix = np.array(rows, dtype=float)
    return (
        algorithms,
        matrix,
        maximize,
        scope_label,
        target_metrics,
        target_instances,
        metrics_label,
        instances_label,
        warnings_list,
    )


def _compute_fractional_ranks_matrix(matrix: np.ndarray, maximize: bool) -> np.ndarray:
    """Compute fractional (average tie) ranks per row (block). Rank 1 is best."""
    n_blocks, k = matrix.shape
    ranks_matrix = np.zeros((n_blocks, k), dtype=float)
    for b in range(n_blocks):
        row = -matrix[b] if maximize else matrix[b]
        ranks_matrix[b] = stats.rankdata(row, method="average")
    return ranks_matrix


class FriedmanNemenyiTest:
    """Friedman non-parametric omnibus test (with exact tie correction), Iman-Davenport F, and Nemenyi CD."""

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> CriticalDistanceResult:
        (
            algorithms,
            matrix,
            maximize,
            scope_label,
            metrics_list,
            instances_list,
            metrics_label,
            instances_label,
            warnings_list,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        ranks_matrix = _compute_fractional_ranks_matrix(matrix, maximize=maximize)

        avg_ranks = np.mean(ranks_matrix, axis=0)
        mean_values = np.mean(matrix, axis=0)

        order = np.argsort(avg_ranks)
        rank_items = [
            AlgorithmRankItem(
                algorithm=algorithms[idx],
                avg_rank=float(avg_ranks[idx]),
                mean_value=float(mean_values[idx]),
            )
            for idx in order
        ]

        # Tie-corrected Friedman chi-square statistic (Conover 1999; matches scipy.stats.friedmanchisquare & R)
        # Q = [12 * N^2 * sum((R_bar_j - (k+1)/2)^2)] / [N * k * (k+1) - (1/(k-1)) * sum_b sum_t (t^3 - t)]
        tie_sum = 0.0
        for b in range(n_blocks):
            _, counts = np.unique(ranks_matrix[b], return_counts=True)
            tie_sum += float(np.sum(counts**3 - counts))

        q_denom = float(n_blocks * k * (k + 1.0)) - (tie_sum / float(k - 1))
        centered_sq = float(np.sum((avg_ranks - (k + 1.0) / 2.0) ** 2))
        if q_denom <= 1e-12:
            friedman_stat = 0.0
            friedman_p = 1.0
        else:
            friedman_stat = max(0.0, float((12.0 * (n_blocks**2) * centered_sq) / q_denom))
            friedman_p = float(stats.chi2.sf(friedman_stat, df=k - 1))

        # Iman-Davenport F-statistic
        if n_blocks <= 1:
            iman_f = 0.0
            iman_p = 1.0
        else:
            id_denom = n_blocks * (k - 1.0) - friedman_stat
            if id_denom <= 1e-12:
                iman_f = float("inf") if friedman_stat > 0 else 0.0
                iman_p = 0.0 if friedman_stat > 0 else 1.0
            else:
                iman_f = float(((n_blocks - 1.0) * friedman_stat) / id_denom)
                iman_p = float(stats.f.sf(iman_f, k - 1, (k - 1) * (n_blocks - 1)))
        if not math.isfinite(iman_f):
            iman_f = 1e6

        # Nemenyi Critical Distance (CD)
        q_alpha = _get_nemenyi_q_alpha(k=k, alpha=alpha)
        cd = float(q_alpha * math.sqrt((k * (k + 1.0)) / (6.0 * n_blocks)))

        # Maximal cliques of non-significantly different algorithms (|R_j - R_i| <= CD)
        raw_cliques: list[tuple[int, int]] = []
        for i in range(len(rank_items)):
            max_j = i
            for j in range(i + 1, len(rank_items)):
                if rank_items[j].avg_rank - rank_items[i].avg_rank <= cd + 1e-9:
                    max_j = j
            if max_j > i:
                is_subset = any(s <= i and e >= max_j for s, e in raw_cliques)
                if not is_subset:
                    raw_cliques.append((i, max_j))

        cliques = [(rank_items[i].avg_rank, rank_items[j].avg_rank) for i, j in raw_cliques]
        p_str = f"{friedman_p:.2e}" if friedman_p < 0.0001 else f"{friedman_p:.4f}"
        params_label = (
            f"α = {alpha}  •  CD = {cd:.3f}  •  k = {k}, N = {n_blocks}  •  χ² = {friedman_stat:.2f} (p = {p_str})"
        )

        return CriticalDistanceResult(
            metric=metric,
            scope=scope,
            scope_label=scope_label,
            metrics_list=metrics_list,
            instances_list=instances_list,
            metrics_label=metrics_label,
            instances_label=instances_label,
            params_label=params_label,
            alpha=alpha,
            k=k,
            n_blocks=n_blocks,
            cd=cd,
            friedman_stat=friedman_stat,
            friedman_p_value=friedman_p,
            iman_davenport_f=iman_f,
            iman_davenport_p_value=iman_p,
            maximize=maximize,
            ranks=rank_items,
            cliques=cliques,
            warnings=tuple(warnings_list),
        )


class FriedmanAlignedRanksTest:
    """Friedman Aligned-Ranks omnibus test (Hodges & Lehmann 1962; García et al. 2010)."""

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> BlockedOmnibusResult:
        (
            algorithms,
            matrix,
            maximize,
            _,
            _,
            _,
            _,
            _,
            warnings_list,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        oriented = -matrix if maximize else matrix

        # Align by subtracting block location (mean), then rank all N*k values jointly
        block_means = np.mean(oriented, axis=1, keepdims=True)
        aligned = oriented - block_means
        flat_ranks = stats.rankdata(aligned.ravel(), method="average")
        aligned_ranks = flat_ranks.reshape((n_blocks, k))

        col_sums = np.sum(aligned_ranks, axis=0)
        row_sums = np.sum(aligned_ranks, axis=1)
        avg_aligned_ranks = col_sums / float(n_blocks)
        mean_values = np.mean(matrix, axis=0)

        # García et al. (2010) Eq. 5 (generalized with sum(r^2) for exact tie handling)
        nk = n_blocks * k
        num = float(k - 1) * (float(np.sum(col_sums**2)) - (k * (n_blocks**2) * ((nk + 1.0) ** 2)) / 4.0)
        den = float(np.sum(aligned_ranks**2)) - float(np.sum(row_sums**2)) / float(k)

        if den <= 1e-12:
            t_ar = 0.0
            p_val = 1.0
        else:
            t_ar = max(0.0, float(num / den))
            p_val = float(stats.chi2.sf(t_ar, df=k - 1))

        order = np.argsort(avg_aligned_ranks)
        rank_items = [
            AlgorithmRankItem(
                algorithm=algorithms[idx],
                avg_rank=float(avg_aligned_ranks[idx]),
                mean_value=float(mean_values[idx]),
            )
            for idx in order
        ]
        is_sig = p_val < alpha
        interp = (
            f"Friedman Aligned-Ranks test on '{metric}' (k={k}, N={n_blocks}): "
            f"T_AR = {t_ar:.4f} (df={k - 1}, p = {p_val:.4e}, alpha = {alpha}) — "
            + ("significant differences detected." if is_sig else "no significant differences detected.")
        )
        return BlockedOmnibusResult(
            test_name="Friedman Aligned-Ranks Test",
            metric=metric,
            statistic=t_ar,
            p_value=p_val,
            degrees_of_freedom=float(k - 1),
            alpha=alpha,
            is_significant=is_sig,
            k=k,
            n_blocks=n_blocks,
            maximize=maximize,
            ranks=rank_items,
            warnings=tuple(warnings_list),
            interpretation=interp,
        )


class QuadeTest:
    """Quade weighted-block non-parametric omnibus test (Quade 1979; Conover 1999; García et al. 2010)."""

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> BlockedOmnibusResult:
        (
            algorithms,
            matrix,
            maximize,
            _,
            _,
            _,
            _,
            _,
            warnings_list,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        if n_blocks < 2:
            raise StatisticalAnalysisError("QuadeTest requires at least 2 blocks (instances).")

        ranks_matrix = _compute_fractional_ranks_matrix(matrix, maximize=maximize)
        ranges = np.max(matrix, axis=1) - np.min(matrix, axis=1)
        q_weights = stats.rankdata(ranges, method="average")

        # Weighted centered rank scores S_{b, j} = Q_b * (r_{b, j} - (k + 1) / 2)
        s_mat = q_weights[:, None] * (ranks_matrix - (k + 1.0) / 2.0)
        s_col = np.sum(s_mat, axis=0)

        a_term = float(np.sum(s_mat**2))
        b_term = float(np.sum(s_col**2)) / float(n_blocks)
        df1 = float(k - 1)
        df2 = float((k - 1) * (n_blocks - 1))

        if a_term - b_term <= 1e-12:
            f_stat = 1e6 if b_term > 1e-12 else 0.0
            p_val = 0.0 if b_term > 1e-12 else 1.0
        else:
            f_stat = max(0.0, float(((n_blocks - 1.0) * b_term) / (a_term - b_term)))
            p_val = float(stats.f.sf(f_stat, df1, df2))

        # Weighted average ranks W_j = sum_b Q_b * r_{b,j} / sum_b Q_b
        q_sum = max(1e-12, float(np.sum(q_weights)))
        weighted_ranks = np.sum(q_weights[:, None] * ranks_matrix, axis=0) / q_sum
        mean_values = np.mean(matrix, axis=0)

        order = np.argsort(weighted_ranks)
        rank_items = [
            AlgorithmRankItem(
                algorithm=algorithms[idx],
                avg_rank=float(weighted_ranks[idx]),
                mean_value=float(mean_values[idx]),
            )
            for idx in order
        ]
        is_sig = p_val < alpha
        interp = (
            f"Quade weighted-block rank test on '{metric}' (k={k}, N={n_blocks}): "
            f"F({df1:.0f}, {df2:.0f}) = {f_stat:.4f} (p = {p_val:.4e}, alpha = {alpha}) — "
            + ("significant differences detected." if is_sig else "no significant differences detected.")
        )
        return BlockedOmnibusResult(
            test_name="Quade Test",
            metric=metric,
            statistic=f_stat,
            p_value=p_val,
            degrees_of_freedom=(df1, df2),
            alpha=alpha,
            is_significant=is_sig,
            k=k,
            n_blocks=n_blocks,
            maximize=maximize,
            ranks=rank_items,
            warnings=tuple(warnings_list),
            interpretation=interp,
        )


class RepeatedMeasuresANOVA:
    """One-way Repeated-Measures ANOVA across matched blocks with Greenhouse-Geisser sphericity correction."""

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
        greenhouse_geisser: bool = True,
    ) -> BlockedOmnibusResult:
        (
            algorithms,
            matrix,
            maximize,
            _,
            _,
            _,
            _,
            _,
            warnings_list,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        if n_blocks < 2:
            raise StatisticalAnalysisError("RepeatedMeasuresANOVA requires at least 2 blocks (instances).")

        grand_mean = float(np.mean(matrix))
        alg_means = np.mean(matrix, axis=0)
        block_means = np.mean(matrix, axis=1)

        ss_treat = float(n_blocks * np.sum((alg_means - grand_mean) ** 2))
        ss_blocks = float(k * np.sum((block_means - grand_mean) ** 2))
        ss_total = float(np.sum((matrix - grand_mean) ** 2))
        ss_error = max(0.0, ss_total - ss_treat - ss_blocks)

        df1 = float(k - 1)
        df2 = float((k - 1) * (n_blocks - 1))
        ms_treat = ss_treat / df1
        ms_error = ss_error / df2 if df2 > 0 else 0.0

        # Greenhouse-Geisser epsilon for sphericity (exact when k >= 3; 1.0 when k == 2)
        epsilon_gg = 1.0
        if k >= 3 and n_blocks >= 2:
            cov_mat = np.cov(matrix, rowvar=False, ddof=1)
            row_m = np.mean(cov_mat, axis=1, keepdims=True)
            col_m = np.mean(cov_mat, axis=0, keepdims=True)
            dc_mat = cov_mat - row_m - col_m + float(np.mean(cov_mat))
            tr_dc = float(np.trace(dc_mat))
            tr_dc2 = float(np.sum(dc_mat**2))
            if tr_dc2 > 1e-18:
                epsilon_gg = (tr_dc**2) / (float(k - 1) * tr_dc2)
                epsilon_gg = min(1.0, max(1.0 / float(k - 1), epsilon_gg))

        eff_df1 = df1 * epsilon_gg if greenhouse_geisser else df1
        eff_df2 = df2 * epsilon_gg if greenhouse_geisser else df2

        if ms_error <= 1e-15:
            f_stat = 1e6 if ms_treat > 1e-15 else 0.0
            p_val = 0.0 if ms_treat > 1e-15 else 1.0
        else:
            f_stat = float(ms_treat / ms_error)
            p_val = float(stats.f.sf(f_stat, eff_df1, eff_df2))

        ranks_matrix = _compute_fractional_ranks_matrix(matrix, maximize=maximize)
        avg_ranks = np.mean(ranks_matrix, axis=0)
        order = np.argsort(-alg_means if maximize else alg_means)
        rank_items = [
            AlgorithmRankItem(
                algorithm=algorithms[idx],
                avg_rank=float(avg_ranks[idx]),
                mean_value=float(alg_means[idx]),
            )
            for idx in order
        ]
        is_sig = p_val < alpha
        gg_note = f", GG ε = {epsilon_gg:.3f}" if greenhouse_geisser and k >= 3 else ""
        interp = (
            f"Repeated-Measures ANOVA on '{metric}' (k={k}, N={n_blocks}{gg_note}): "
            f"F({eff_df1:.2f}, {eff_df2:.2f}) = {f_stat:.4f} (p = {p_val:.4e}, alpha = {alpha}) — "
            + ("significant differences detected." if is_sig else "no significant differences detected.")
        )
        return BlockedOmnibusResult(
            test_name="Repeated-Measures One-Way ANOVA (Greenhouse-Geisser)"
            if greenhouse_geisser
            else "Repeated-Measures One-Way ANOVA",
            metric=metric,
            statistic=f_stat,
            p_value=p_val,
            degrees_of_freedom=(eff_df1, eff_df2),
            alpha=alpha,
            is_significant=is_sig,
            k=k,
            n_blocks=n_blocks,
            maximize=maximize,
            ranks=rank_items,
            sphericity_epsilon=epsilon_gg,
            warnings=tuple(warnings_list),
            interpretation=interp,
        )


class BayesianRankingTest:
    """Exact Bayesian Sign-Test and Signed-Rank Simplex (ROPE) test (Benavoli et al. 2017, JMLR).

    Implements:
        - Bayesian Sign Test with Dirichlet Process prior DP(s, delta_0) (Benavoli et al. 2017, Sec. 4.1).
        - Bayesian Signed-Rank Test with DP(s, delta_0) exact quadratic form over pairwise sums
          (z_i + z_j) vs +/- 2*ROPE (Benavoli et al. 2017, Sec. 4.2, Eq. 11).
        - Supports both exact absolute ROPE (`rope` in metric units) and relative ROPE (`rope_percent`).
        - Every observed block contributes exact Dirichlet weight 1.0 (no sample-size truncation).
    """

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        rope: float | None = None,
        prior_strength: float = 1.0,
        num_samples: int = 1500,
        seed: int = 42,
        algo_a: str | None = None,
        algo_b: str | None = None,
        method: str = "signed-rank",
    ) -> BayesianRankResult:
        is_sign = method.strip().lower() in {"sign", "bayesian-sign", "sign-test"}
        normalized_method = "sign" if is_sign else "signed-rank"
        if prior_strength <= 0:
            raise StatisticalAnalysisError(f"prior_strength must be > 0, got {prior_strength}.")

        (
            algorithms,
            matrix,
            maximize,
            scope_label,
            metrics_list,
            instances_list,
            metrics_label,
            instances_label,
            warnings_list,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        ranks_matrix = _compute_fractional_ranks_matrix(matrix, maximize=maximize)

        method_seed_offset = 101 if normalized_method == "sign" else 0
        rng = np.random.default_rng(seed + n_blocks * 31 + k * 17 + method_seed_offset)

        # Bayesian bootstrap over per-block ranks for multi-algorithm expected-rank posterior
        weights = rng.dirichlet(np.ones(n_blocks, dtype=float), size=num_samples)
        posterior_ranks = weights @ ranks_matrix

        # Multi-algorithm P(best) via Bayesian Plackett-Luce (Calvo et al. 2018) when k >= 2
        pl_res = BayesianPlackettLuceTest().run(
            matrix=matrix,
            algorithms=algorithms,
            metric=metric,
            maximize=maximize,
            num_samples=min(400, max(150, num_samples // 4)),
            prior_alpha=prior_strength,
            seed=seed,
        )
        pl_prob_map = {it.algorithm: it.prob_best for it in pl_res.items}

        mean_ranks = np.mean(posterior_ranks, axis=0)
        ref_idx = int(np.argmin(mean_ranks))
        reference_algo = algorithms[ref_idx]

        use_absolute_rope = rope is not None
        rope_threshold = float(rope) if rope is not None else max(0.0, rope_percent / 100.0)
        if rope_threshold < 0:
            raise StatisticalAnalysisError(f"ROPE threshold must be >= 0, got {rope_threshold}.")

        items: list[BayesianPosteriorItem] = []
        for idx, algo in enumerate(algorithms):
            samples_arr = posterior_ranks[:, idx]
            q025, q25, q50, q75, q975 = np.percentile(samples_arr, [2.5, 25.0, 50.0, 75.0, 97.5])

            wins = 0
            rope_ties = 0
            for b in range(n_blocks):
                v_algo = float(matrix[b, idx])
                v_ref = float(matrix[b, ref_idx])
                raw_d = (v_algo - v_ref) if maximize else (v_ref - v_algo)
                if use_absolute_rope:
                    diff_val = raw_d
                else:
                    scale = max(1e-12, (abs(v_algo) + abs(v_ref)) / 2.0)
                    diff_val = raw_d / scale
                if abs(diff_val) <= rope_threshold:
                    rope_ties += 1
                elif diff_val > rope_threshold:
                    wins += 1

            # Exact posterior mean under DP(s, delta_0) prior concentrated at z_0 = 0 (ROPE)
            denom_post = float(n_blocks) + float(prior_strength)
            prob_better = 0.0 if idx == ref_idx else float(wins / denom_post)
            prob_rope = 1.0 if idx == ref_idx else float((rope_ties + prior_strength) / denom_post)

            items.append(
                BayesianPosteriorItem(
                    algorithm=algo,
                    mean_rank=float(mean_ranks[idx]),
                    median_rank=float(q50),
                    ci_lower=float(q025),
                    ci_upper=float(q975),
                    q25=float(q25),
                    q75=float(q75),
                    prob_best=float(pl_prob_map.get(algo, 0.0)),
                    prob_better_than_ref=prob_better,
                    prob_rope_with_ref=prob_rope,
                    samples=[round(float(x), 4) for x in samples_arr.tolist()],
                )
            )

        items.sort(key=lambda item: item.mean_rank)

        sorted_algos_by_rank = [item.algorithm for item in items]
        resolved_b = algo_b if (algo_b and algo_b in algorithms) else sorted_algos_by_rank[0]
        if algo_a and algo_a in algorithms and algo_a != resolved_b:
            resolved_a = algo_a
        else:
            fallback_candidates = [a for a in sorted_algos_by_rank if a != resolved_b]
            resolved_a = fallback_candidates[0] if fallback_candidates else sorted_algos_by_rank[0]

        idx_a = algorithms.index(resolved_a)
        idx_b = algorithms.index(resolved_b)
        vals_a = matrix[:, idx_a]
        vals_b = matrix[:, idx_b]

        # Paired differences where > 0 means algo_a (Left) beats algo_b (Right)
        raw_diffs = (vals_a - vals_b) if maximize else (vals_b - vals_a)
        if use_absolute_rope:
            z_obs = raw_diffs
        else:
            pair_scales = np.maximum(1e-12, (np.abs(vals_a) + np.abs(vals_b)) / 2.0)
            z_obs = raw_diffs / pair_scales

        if normalized_method == "sign":
            # Benavoli et al. (2017, Sec. 4.1, Eq. 9): exact Dirichlet posterior with DP(s, delta_0)
            # Every observation has exact weight 1.0; single pseudo-observation z_0 = 0 adds prior_strength to ROPE.
            eps = 1e-6
            n_left = float(np.sum(z_obs > rope_threshold))
            n_rope = float(np.sum(np.abs(z_obs) <= rope_threshold))
            n_right = float(np.sum(z_obs < -rope_threshold))
            dir_params = np.array(
                [max(eps, n_left), max(eps, n_rope + prior_strength), max(eps, n_right)],
                dtype=float,
            )
            w_sign = rng.dirichlet(dir_params, size=num_samples)
            theta_left = w_sign[:, 0]
            theta_rope = w_sign[:, 1]
            theta_right = w_sign[:, 2]
        else:
            # Benavoli et al. (2017, Sec. 4.2, Eq. 11): exact Bayesian Signed-Rank Test
            # Single pseudo-observation z_0 = 0 with weight prior_strength; each real z_i has exact weight 1.0.
            z_aug = np.concatenate([[0.0], z_obs])
            pairwise_sums = z_aug[:, None] + z_aug[None, :]
            two_rope = 2.0 * rope_threshold

            ind_left = (pairwise_sums > two_rope).astype(float)
            ind_rope = (np.abs(pairwise_sums) <= two_rope).astype(float)
            ind_right = (pairwise_sums < -two_rope).astype(float)

            alpha_sr = np.concatenate([[float(prior_strength)], np.ones(n_blocks, dtype=float)])
            w_sr = rng.dirichlet(alpha_sr, size=num_samples)

            theta_left = np.einsum("mi,ij,mj->m", w_sr, ind_left, w_sr)
            theta_rope = np.einsum("mi,ij,mj->m", w_sr, ind_rope, w_sr)
            theta_right = np.einsum("mi,ij,mj->m", w_sr, ind_right, w_sr)

        theta_sum = np.maximum(1e-12, theta_left + theta_rope + theta_right)
        theta_left = theta_left / theta_sum
        theta_rope = theta_rope / theta_sum
        theta_right = theta_right / theta_sum

        theta_stack = np.column_stack([theta_left, theta_rope, theta_right])
        dominant_region = np.argmax(theta_stack, axis=1)
        p_left_val = round(float(np.mean(dominant_region == 0)), 3)
        p_rope_val = round(float(np.mean(dominant_region == 1)), 3)
        p_right_val = round(max(0.0, 1.0 - p_left_val - p_rope_val), 3)

        h_tri = math.sqrt(3.0) / 2.0
        x_coords = theta_right + 0.5 * theta_rope
        y_coords = h_tri * theta_rope

        try:
            kde = stats.gaussian_kde(np.vstack([x_coords, y_coords]))
            raw_density = kde(np.vstack([x_coords, y_coords]))
        except Exception:
            dist_from_center = np.hypot(x_coords - np.mean(x_coords), y_coords - np.mean(y_coords))
            raw_density = np.exp(-dist_from_center / max(1e-6, float(np.std(dist_from_center))))

        d_min = float(np.min(raw_density))
        d_max = float(np.max(raw_density))
        norm_density = (raw_density - d_min) / max(1e-9, d_max - d_min)

        density_order = np.argsort(norm_density)
        x_sorted = x_coords[density_order]
        y_sorted = y_coords[density_order]
        d_sorted = norm_density[density_order]
        p_left_sorted = theta_left[density_order]
        p_rope_sorted = theta_rope[density_order]
        p_right_sorted = theta_right[density_order]

        cx, cy = 0.5, round(h_tri / 3.0, 6)
        half_h = round(h_tri / 2.0, 6)
        top_h = round(h_tri, 6)
        triangle_x = [0.0, 1.0, 0.5, 0.0]
        triangle_y = [0.0, 0.0, top_h, 0.0]
        boundary_lines: list[tuple[float, float, float, float]] = [
            (cx, cy, 0.5, 0.0),
            (cx, cy, 0.25, half_h),
            (cx, cy, 0.75, half_h),
        ]

        method_tag = "Bayesian Sign" if normalized_method == "sign" else "Bayesian Signed-Rank"
        rope_tag = f"±{rope_threshold:.4g}" if use_absolute_rope else f"±{rope_percent}%"
        params_label = (
            f"{method_tag}: {resolved_a} vs {resolved_b}  •  ROPE = {rope_tag}  •  N = {n_blocks}  •  "
            f"p({resolved_a}) = {p_left_val:.3f}, p(rope) = {p_rope_val:.3f}, p({resolved_b}) = {p_right_val:.3f}"
        )

        return BayesianRankResult(
            metric=metric,
            scope=scope,
            scope_label=scope_label,
            bayesian_method=normalized_method,
            metrics_list=metrics_list,
            instances_list=instances_list,
            metrics_label=metrics_label,
            instances_label=instances_label,
            params_label=params_label,
            k=k,
            n_blocks=n_blocks,
            rope_percent=rope_percent,
            rope_absolute=rope,
            prior_strength=prior_strength,
            reference_algo=reference_algo,
            maximize=maximize,
            algo_a=resolved_a,
            algo_b=resolved_b,
            algorithms=algorithms,
            p_left=p_left_val,
            p_rope=p_rope_val,
            p_right=p_right_val,
            label_left=f"p({resolved_a}) = {p_left_val:.3f}",
            label_rope=f"p(rope) = {p_rope_val:.3f}",
            label_right=f"p({resolved_b}) = {p_right_val:.3f}",
            points_x=[round(float(v), 5) for v in x_sorted.tolist()],
            points_y=[round(float(v), 5) for v in y_sorted.tolist()],
            points_density=[round(float(v), 4) for v in d_sorted.tolist()],
            points_p_left=[round(float(v), 4) for v in p_left_sorted.tolist()],
            points_p_rope=[round(float(v), 4) for v in p_rope_sorted.tolist()],
            points_p_right=[round(float(v), 4) for v in p_right_sorted.tolist()],
            triangle_x=triangle_x,
            triangle_y=triangle_y,
            boundary_lines=boundary_lines,
            items=items,
            warnings=tuple(warnings_list),
        )


__all__ = [
    "FriedmanNemenyiTest",
    "FriedmanAlignedRanksTest",
    "QuadeTest",
    "RepeatedMeasuresANOVA",
    "BayesianRankingTest",
    "BayesianCorrelatedTTest",
    "BayesianPlackettLuceTest",
]
