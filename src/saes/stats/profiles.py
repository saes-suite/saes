"""Dolan-Moré performance profiles, data profiles, and global meta-benchmarking."""

from __future__ import annotations

import math

import numpy as np

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    AlgorithmProfileCurve,
    Dataset,
    GlobalMetaRankingItem,
    PerformanceProfilesResult,
    WinTieLossCell,
)
from saes.stats.nonparametric import (
    MannWhitneyUTest,
    WilcoxonSignedRankTest,
    adjust_pvalues,
    compute_vargha_delaney_a12,
)


def compute_performance_profiles(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    max_tau: float | None = None,
    num_tau_points: int = 60,
    paired: bool = False,
    posthoc_correction: str = "holm",
) -> PerformanceProfilesResult:
    """Compute Dolan-Moré performance profiles and multiplicity-adjusted Win/Tie/Loss meta-ranking.

    Reference:
        Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software
        with performance profiles. Mathematical Programming, 91(2), 201-213.

    Args:
        dataset: Validated SAES Dataset.
        metric: Evaluation metric identifier (defaults to first metric).
        alpha: Statistical significance threshold for Win/Tie/Loss comparisons.
        max_tau: Optional upper bound for performance ratio tau.
        num_tau_points: Number of evaluation points along the tau axis.
        paired: Whether per-instance runs are explicitly paired across algorithms (default False ->
            Mann-Whitney U; True -> Wilcoxon Signed-Rank when equal lengths).
        posthoc_correction: Multiple-testing p-value adjustment applied to the Win/Tie/Loss
            comparisons within each instance (default 'holm').

    Returns:
        PerformanceProfilesResult with curves, W/T/L matrix, and meta-rankings.
    """
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no evaluation metrics.")

    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]
    metric_df = dataset.df[dataset.df["metric"] == target_metric].copy()

    if metric_df.empty:
        raise StatisticalAnalysisError(f"No records found for metric '{target_metric}'.")

    maximize: bool = bool(metric_df["maximize"].iloc[0])
    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    instances = sorted(metric_df["instance"].unique().tolist())

    if len(algorithms) < 1 or len(instances) < 1:
        raise StatisticalAnalysisError("Need at least 1 algorithm and 1 instance for performance profiling.")

    # 1. Representative performance per (instance, algorithm) cell (median across runs)
    perf_matrix: dict[str, dict[str, float]] = {}
    for inst in instances:
        perf_matrix[inst] = {}
        for algo in algorithms:
            sub = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == algo)]
            perf_matrix[inst][algo] = float(sub["value"].median()) if not sub.empty else float("nan")

    # 2. Dolan-Moré (2002) performance ratios r_{p, s}
    ratio_matrix: dict[str, dict[str, float]] = {inst: {} for inst in instances}
    all_finite_ratios: list[float] = []

    for inst in instances:
        row_vals = [perf_matrix[inst][a] for a in algorithms if not math.isnan(perf_matrix[inst][a])]
        if not row_vals:
            continue

        min_val = min(row_vals)
        max_val = max(row_vals)
        shift = (abs(min_val) + 1.0) if min_val <= 0.0 else 0.0

        for algo in algorithms:
            val = perf_matrix[inst][algo]
            if math.isnan(val):
                ratio = float("inf")
            else:
                shifted_val = val + shift
                if maximize:
                    best_shifted = max_val + shift
                    ratio = best_shifted / max(1e-12, shifted_val)
                else:
                    best_shifted = min_val + shift
                    ratio = shifted_val / max(1e-12, best_shifted)

            if abs(ratio - 1.0) < 1e-9:
                ratio = 1.0

            ratio_matrix[inst][algo] = ratio
            if math.isfinite(ratio):
                all_finite_ratios.append(ratio)

    if not all_finite_ratios:
        all_finite_ratios = [1.0]

    computed_max_tau = max(all_finite_ratios)
    upper_tau = max_tau if max_tau is not None else min(10.0, max(2.0, computed_max_tau))
    if upper_tau <= 1.0:
        upper_tau = 2.0

    tau_grid = np.unique(
        np.concatenate(
            [
                np.array([1.0]),
                np.linspace(1.0, upper_tau, num_tau_points),
                np.array(sorted(all_finite_ratios)),
            ]
        )
    )
    tau_grid = tau_grid[(tau_grid >= 1.0) & (tau_grid <= upper_tau)]
    tau_values = [round(float(t), 4) for t in sorted(set(tau_grid.tolist()))]

    # 3. Profile curves rho_s(tau) = fraction of problems where r_{p,s} <= tau
    total_instances = len(instances)
    curves: list[AlgorithmProfileCurve] = []

    for algo in algorithms:
        algo_ratios = [ratio_matrix[inst].get(algo, float("inf")) for inst in instances]
        rho_vals: list[float] = []
        for t in tau_values:
            solved = sum(1 for r in algo_ratios if r <= t)
            rho_vals.append(round(solved / total_instances, 4))

        efficiency = rho_vals[0] if rho_vals else 0.0
        robustness = rho_vals[-1] if rho_vals else 0.0

        curves.append(
            AlgorithmProfileCurve(
                algorithm=algo,
                tau_values=tau_values,
                rho_values=rho_vals,
                efficiency=efficiency,
                robustness=robustness,
            )
        )

    # 4. Multiplicity-adjusted Win / Tie / Loss (W/T/L) Score Matrix
    # Compute symmetric upper-triangle tests per instance, adjust p-values per instance family,
    # and assign win/loss direction via non-parametric stochastic dominance (A12 with median tie-break).
    wilcoxon = WilcoxonSignedRankTest()
    mannwhitney = MannWhitneyUTest()

    k = len(algorithms)
    pair_wtl: dict[tuple[int, int], list[int]] = {(i, j): [0, 0, 0] for i in range(k) for j in range(k) if i != j}

    for inst in instances:
        inst_df = metric_df[metric_df["instance"] == inst]
        pair_indices: list[tuple[int, int]] = []
        raw_ps: list[float] = []
        pair_dirs: list[int] = []  # +1 if i better than j, -1 if j better than i, 0 if tie

        for i in range(k):
            for j in range(i + 1, k):
                a_i, a_j = algorithms[i], algorithms[j]
                v_i = inst_df[inst_df["algorithm"] == a_i]["value"].dropna().to_numpy(dtype=float)
                v_j = inst_df[inst_df["algorithm"] == a_j]["value"].dropna().to_numpy(dtype=float)

                pair_indices.append((i, j))
                if len(v_i) < 2 or len(v_j) < 2:
                    raw_ps.append(1.0)
                    pair_dirs.append(0)
                    continue

                try:
                    if paired and len(v_i) == len(v_j) and len(v_i) >= 3:
                        res = wilcoxon.run({a_i: v_i, a_j: v_j}, alpha=alpha)
                    else:
                        res = mannwhitney.run({a_i: v_i, a_j: v_j}, alpha=alpha)
                    p_val = res.p_value
                except Exception:
                    p_val = 1.0

                # Stochastic superiority direction (Vargha-Delaney A12, median fallback on exact 0.5 tie)
                a12, _ = compute_vargha_delaney_a12(v_i, v_j, maximize=maximize)
                if a12 > 0.5 + 1e-12:
                    direction = 1
                elif a12 < 0.5 - 1e-12:
                    direction = -1
                else:
                    med_i, med_j = float(np.median(v_i)), float(np.median(v_j))
                    if abs(med_i - med_j) <= 1e-12:
                        direction = 0
                    else:
                        direction = 1 if ((med_i > med_j) if maximize else (med_i < med_j)) else -1

                raw_ps.append(p_val)
                pair_dirs.append(direction)

        adj_ps = adjust_pvalues(raw_ps, method=posthoc_correction) if raw_ps else []
        for (i, j), p_adj, direction in zip(pair_indices, adj_ps, pair_dirs, strict=False):
            if p_adj < alpha and direction != 0:
                if direction > 0:
                    pair_wtl[(i, j)][0] += 1  # i wins over j
                    pair_wtl[(j, i)][2] += 1  # j loses to i
                else:
                    pair_wtl[(i, j)][2] += 1  # i loses to j
                    pair_wtl[(j, i)][0] += 1  # j wins over i
            else:
                pair_wtl[(i, j)][1] += 1
                pair_wtl[(j, i)][1] += 1

    wtl_matrix: list[list[WinTieLossCell]] = []
    algo_wins: dict[str, int] = {a: 0 for a in algorithms}
    algo_ties: dict[str, int] = {a: 0 for a in algorithms}
    algo_losses: dict[str, int] = {a: 0 for a in algorithms}

    for i, algo_row in enumerate(algorithms):
        matrix_row: list[WinTieLossCell] = []
        for j, algo_col in enumerate(algorithms):
            if i == j:
                matrix_row.append(
                    WinTieLossCell(
                        algo_row=algo_row,
                        algo_col=algo_col,
                        wins=0,
                        ties=total_instances,
                        losses=0,
                        net=0,
                    )
                )
                continue
            wins, ties, losses = pair_wtl[(i, j)]
            net = wins - losses
            algo_wins[algo_row] += wins
            algo_ties[algo_row] += ties
            algo_losses[algo_row] += losses
            matrix_row.append(
                WinTieLossCell(
                    algo_row=algo_row,
                    algo_col=algo_col,
                    wins=wins,
                    ties=ties,
                    losses=losses,
                    net=net,
                )
            )
        wtl_matrix.append(matrix_row)

    # 5. Global Meta-Rankings
    curve_map = {c.algorithm: c for c in curves}
    num_comparisons = max(1, (len(algorithms) - 1) * total_instances)

    meta_items: list[GlobalMetaRankingItem] = []
    for algo in algorithms:
        c = curve_map.get(algo)
        eff = c.efficiency if c else 0.0
        rob = c.robustness if c else 0.0
        win_cnt = algo_wins[algo]
        tie_cnt = algo_ties[algo]
        loss_cnt = algo_losses[algo]
        net = win_cnt - loss_cnt
        copeland = round(net / num_comparisons, 4)

        meta_items.append(
            GlobalMetaRankingItem(
                rank=1,
                algorithm=algo,
                efficiency=eff,
                robustness=rob,
                total_wins=win_cnt,
                total_ties=tie_cnt,
                total_losses=loss_cnt,
                net_score=net,
                copeland_score=copeland,
            )
        )

    meta_items.sort(key=lambda x: (x.net_score, x.efficiency, x.robustness), reverse=True)
    for rank_idx, item in enumerate(meta_items, start=1):
        object.__setattr__(item, "rank", rank_idx)

    return PerformanceProfilesResult(
        metric=target_metric,
        maximize=maximize,
        alpha=alpha,
        tau_values=tau_values,
        curves=curves,
        wtl_matrix=wtl_matrix,
        meta_rankings=meta_items,
    )
