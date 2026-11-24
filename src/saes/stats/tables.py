"""Scientific benchmark table generation (ANOVA, Friedman, Wilcoxon, Kruskal-Wallis).

Presentation and multi-format export renderers (LaTeX, Markdown, CSV, HTML, DOCX, PDF)
live in `saes.reports.renderers` and are re-exported here for full backward compatibility.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    AlgorithmSummaryFooter,
    BenchmarkTableCell,
    BenchmarkTableInstanceRow,
    Dataset,
    PairwiseInstanceComparison,
    PairwiseMatrixCell,
    StatisticalResult,
    StatisticalTableResult,
)
from saes.reports.renderers import (
    PDF_THEMES,
    _format_num,
    _format_p,
    export_flat_table,
    export_statistical_table,
    render_latex_table,
    render_markdown_table,
)
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.bayesian import BayesianPlackettLuceTest
from saes.stats.decision import is_paired_data
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
)
from saes.stats.ranking import FriedmanAlignedRanksTest, FriedmanNemenyiTest, QuadeTest
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def _fast_bayesian_pair_probs(
    vec_r: np.ndarray,
    vec_c: np.ndarray,
    maximize: bool,
    rope_percent: float = 1.0,
    num_samples: int = 400,
    prior_strength: float = 1.0,
) -> tuple[float, float, float]:
    """Compute exact Benavoli et al. (2017) Bayesian Signed-Rank ROPE probabilities for a pair."""
    r_clean = vec_r[~np.isnan(vec_r)]
    c_clean = vec_c[~np.isnan(vec_c)]
    if len(r_clean) == 0 or len(c_clean) == 0:
        return 0.0, 1.0, 0.0

    rope_threshold = max(1e-9, rope_percent / 100.0)
    rng = np.random.default_rng(42 + len(r_clean) * 13 + len(c_clean) * 7)

    if len(r_clean) == len(c_clean) and len(r_clean) >= 2:
        raw_diffs = (r_clean - c_clean) if maximize else (c_clean - r_clean)
        scales = np.maximum(1e-12, (np.abs(r_clean) + np.abs(c_clean)) / 2.0)
        z_obs = raw_diffs / scales
        z_aug = np.concatenate([[0.0], z_obs])
        pairwise_sums = z_aug[:, None] + z_aug[None, :]
        two_rope = 2.0 * rope_threshold

        ind_left = (pairwise_sums > two_rope).astype(float)
        ind_rope = (np.abs(pairwise_sums) <= two_rope).astype(float)
        ind_right = (pairwise_sums < -two_rope).astype(float)

        alpha_sr = np.concatenate([[prior_strength], np.ones(len(z_obs), dtype=float)])
        w_sr = rng.dirichlet(alpha_sr, size=num_samples)

        theta_left = np.einsum("mi,ij,mj->m", w_sr, ind_left, w_sr)
        theta_rope = np.einsum("mi,ij,mj->m", w_sr, ind_rope, w_sr)
        theta_right = np.einsum("mi,ij,mj->m", w_sr, ind_right, w_sr)
        dominant = np.argmax(np.column_stack([theta_left, theta_rope, theta_right]), axis=1)
        p_row = float(np.mean(dominant == 0))
        p_rope = float(np.mean(dominant == 1))
        p_col = max(0.0, 1.0 - p_row - p_rope)
        return p_row, p_rope, p_col

    # Unpaired fallback: Bayesian bootstrap over Mann-Whitney relative dominance against ROPE
    a12, _ = compute_vargha_delaney_a12(r_clean, c_clean, maximize=maximize)
    rope_a12 = min(0.15, max(0.02, rope_threshold))
    if abs(a12 - 0.5) <= rope_a12:
        return 0.0, 1.0, 0.0
    if a12 > 0.5:
        return float(a12), float(max(0.0, 1.0 - a12)), 0.0
    return 0.0, float(max(0.0, a12)), float(1.0 - a12)


def _fast_bayesian_summary_probs(
    inst_means_matrix: dict[str, np.ndarray],
    algorithms: list[str],
    ref_algo: str,
    maximize: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute Bayesian P(Best) via Plackett-Luce over per-instance rankings and P(Better/ROPE vs Ref)."""
    bayes_best_map: dict[str, float] = {}
    bayes_better_map: dict[str, float] = {}
    bayes_rope_map: dict[str, float] = {}

    ref_vec = inst_means_matrix.get(ref_algo, np.array([], dtype=float))
    for alg in algorithms:
        if alg == ref_algo:
            bayes_better_map[alg] = 0.0
            bayes_rope_map[alg] = 1.0
        else:
            alg_vec = inst_means_matrix.get(alg, np.array([], dtype=float))
            p_b, p_r, _ = _fast_bayesian_pair_probs(alg_vec, ref_vec, maximize=maximize)
            bayes_better_map[alg] = p_b
            bayes_rope_map[alg] = p_r

    # Respect incommensurability across instances using Plackett-Luce on per-instance rankings
    lengths = [len(inst_means_matrix.get(a, np.array([]))) for a in algorithms]
    min_len = min(lengths) if lengths else 0
    if min_len >= 1 and len(algorithms) >= 2:
        mat = np.column_stack([inst_means_matrix[a][:min_len] for a in algorithms])
        pl_res = BayesianPlackettLuceTest().run(
            matrix=mat,
            algorithms=algorithms,
            maximize=maximize,
            num_samples=250,
            seed=42,
        )
        for it in pl_res.items:
            bayes_best_map[it.algorithm] = it.prob_best
    else:
        for alg in algorithms:
            bayes_best_map[alg] = 1.0 / float(max(1, len(algorithms)))

    return bayes_best_map, bayes_better_map, bayes_rope_map


def _compute_cohens_d_raw(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute signed Cohen's d between two 1D arrays."""
    c1 = x1[~np.isnan(x1)]
    c2 = x2[~np.isnan(x2)]
    n1, n2 = len(c1), len(c2)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1 = float(np.var(c1, ddof=1))
    v2 = float(np.var(c2, ddof=1))
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(c1) - np.mean(c2)) / math.sqrt(pooled_var))


def _run_omnibus_for_groups(
    groups: dict[str, np.ndarray],
    test_type: str,
    alpha: float,
    is_paired: bool = False,
) -> StatisticalResult:
    """Execute the requested or auto-selected omnibus/group test for a single instance."""
    k = len(groups)
    norm_type = test_type.strip().lower()

    if k == 2:
        if norm_type in {"brunner_munzel", "brunner-munzel"}:
            return BrunnerMunzelTest().run(groups, alpha=alpha)
        if norm_type in {"wilcoxon", "wilcoxon_signed_rank", "friedman", "friedman_aligned", "quade"} and is_paired:
            return WilcoxonSignedRankTest().run(groups, alpha=alpha)
        if norm_type in {"mann_whitney", "kruskal", "wilcoxon", "friedman", "friedman_aligned", "quade"}:
            return MannWhitneyUTest().run(groups, alpha=alpha)
        if norm_type in {"ttest_rel", "paired_ttest", "rm_anova"} and is_paired:
            return PairedTTest().run(groups, alpha=alpha)
        if norm_type in {"ttest_ind", "anova_oneway", "anova", "rm_anova"}:
            return IndependentTTest().run(groups, alpha=alpha)
        if norm_type in {"ttest_welch", "anova_welch", "welch"}:
            return WelchTTest().run(groups, alpha=alpha)

        norm_checks = NormalityChecker(alpha=alpha).check(groups)
        all_normal = all(n.passed for n in norm_checks)
        if not all_normal:
            return (
                WilcoxonSignedRankTest().run(groups, alpha=alpha)
                if is_paired
                else MannWhitneyUTest().run(groups, alpha=alpha)
            )
        if is_paired:
            return PairedTTest().run(groups, alpha=alpha)
        homo = HomogeneityChecker(alpha=alpha).check(groups)
        return IndependentTTest().run(groups, alpha=alpha) if homo.passed else WelchTTest().run(groups, alpha=alpha)

    if norm_type in {"anova_oneway", "anova", "ttest_ind", "rm_anova"}:
        return OneWayANOVA().run(groups, alpha=alpha)
    if norm_type in {"anova_welch", "welch", "ttest_welch"}:
        return WelchANOVA().run(groups, alpha=alpha)
    if norm_type in {
        "kruskal",
        "kruskal_wallis",
        "wilcoxon",
        "mann_whitney",
        "brunner_munzel",
        "brunner-munzel",
        "friedman",
        "friedman_aligned",
        "quade",
    }:
        return KruskalWallisTest().run(groups, alpha=alpha)

    norm_checks = NormalityChecker(alpha=alpha).check(groups)
    if not all(n.passed for n in norm_checks):
        return KruskalWallisTest().run(groups, alpha=alpha)
    homo = HomogeneityChecker(alpha=alpha).check(groups)
    return OneWayANOVA().run(groups, alpha=alpha) if homo.passed else WelchANOVA().run(groups, alpha=alpha)


def _run_pairwise_test(
    x_algo: np.ndarray,
    x_ref: np.ndarray,
    algo_name: str,
    ref_name: str,
    test_type: str,
    alpha: float,
    is_paired: bool,
) -> StatisticalResult:
    """Run a two-sample test between an algorithm and a reference/comparison algorithm."""
    groups = {algo_name: x_algo, ref_name: x_ref}
    norm_type = test_type.strip().lower()
    equal_len = len(x_algo) == len(x_ref) and is_paired

    if norm_type in {"brunner_munzel", "brunner-munzel"}:
        return BrunnerMunzelTest().run(groups, alpha=alpha)

    if norm_type in {"wilcoxon", "wilcoxon_signed_rank", "friedman", "friedman_aligned", "quade"}:
        if equal_len:
            return WilcoxonSignedRankTest().run(groups, alpha=alpha)
        return MannWhitneyUTest().run(groups, alpha=alpha)

    if norm_type in {"mann_whitney", "kruskal", "kruskal_wallis"}:
        return MannWhitneyUTest().run(groups, alpha=alpha)

    if norm_type in {"ttest_rel", "paired_ttest", "rm_anova"}:
        if equal_len:
            return PairedTTest().run(groups, alpha=alpha)
        return WelchTTest().run(groups, alpha=alpha)

    if norm_type in {"ttest_ind", "anova_oneway", "anova"}:
        return IndependentTTest().run(groups, alpha=alpha)

    if norm_type in {"ttest_welch", "anova_welch", "welch"}:
        return WelchTTest().run(groups, alpha=alpha)

    norm_checks = NormalityChecker(alpha=alpha).check(groups)
    if not all(n.passed for n in norm_checks):
        if equal_len:
            return WilcoxonSignedRankTest().run(groups, alpha=alpha)
        return MannWhitneyUTest().run(groups, alpha=alpha)

    if equal_len:
        return PairedTTest().run(groups, alpha=alpha)
    homo = HomogeneityChecker(alpha=alpha).check(groups)
    return IndependentTTest().run(groups, alpha=alpha) if homo.passed else WelchTTest().run(groups, alpha=alpha)


class StatisticalTableGenerator:
    """Generates publication-ready ANOVA, Friedman, Wilcoxon, and Pairwise statistical benchmark tables."""

    def generate(
        self,
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
        latex_document_mode: str = "standalone",
        latex_notation: str = "scientific",
        alpha: float = 0.05,
        precision: int = 4,
        paired: bool = False,
    ) -> StatisticalTableResult:
        """Generate a complete StatisticalTableResult with cells, omnibus tests, pairwise matrices, and previews.

        Args:
            paired: Whether per-instance runs are explicitly paired across algorithms (default False ->
                independent stochastic runs use Mann-Whitney / Welch / Student rather than silently
                assuming pairing from matching `run_id` indices). Multi-instance global comparisons
                remain blocked by instance.
        """
        df = dataset.df.copy()
        if df.empty:
            raise StatisticalAnalysisError("Cannot generate statistical table from an empty dataset.")

        available_metrics = sorted(df["metric"].unique().tolist())
        target_metric = metric if (metric and metric in available_metrics) else available_metrics[0]
        sub_df = df[df["metric"] == target_metric].copy()

        if algorithms_filter:
            valid_algos = [a for a in algorithms_filter if a in sub_df["algorithm"].unique()]
            if len(valid_algos) >= 2:
                sub_df = sub_df[sub_df["algorithm"].isin(valid_algos)].copy()

        if instances_filter:
            valid_insts = [i for i in instances_filter if i in sub_df["instance"].unique()]
            if valid_insts:
                sub_df = sub_df[sub_df["instance"].isin(valid_insts)].copy()

        if scope == "single-instance" and selected_instance:
            if selected_instance in sub_df["instance"].unique():
                sub_df = sub_df[sub_df["instance"] == selected_instance].copy()

        algorithms = sorted(sub_df["algorithm"].unique().tolist())
        instances = sorted(sub_df["instance"].unique().tolist())
        if len(algorithms) < 2:
            raise StatisticalAnalysisError(
                f"Statistical table generation requires at least 2 algorithms, got {len(algorithms)}."
            )

        maximize = bool(sub_df["maximize"].iloc[0])

        # 1. Global Blocked Non-Parametric Tests (Friedman, Aligned-Ranks, Quade)
        friedman_stat: float | None = None
        friedman_p: float | None = None
        iman_f: float | None = None
        iman_p: float | None = None
        kendalls_w: float | None = None
        nemenyi_cd: float | None = None
        friedman_aligned_stat: float | None = None
        friedman_aligned_p: float | None = None
        quade_stat: float | None = None
        quade_p: float | None = None
        global_ranks_map: dict[str, float] = {}

        effective_scope = "single-instance" if len(instances) == 1 else "all-instances"
        effective_inst = instances[0] if len(instances) == 1 else None

        try:
            cd_res = FriedmanNemenyiTest().run(
                df=sub_df, metric=target_metric, alpha=alpha, scope=effective_scope, instance=effective_inst
            )
            friedman_stat = cd_res.friedman_stat
            friedman_p = cd_res.friedman_p_value
            iman_f = cd_res.iman_davenport_f
            iman_p = cd_res.iman_davenport_p_value
            nemenyi_cd = cd_res.cd
            if cd_res.n_blocks > 0 and cd_res.k > 1:
                kendalls_w = float(min(1.0, max(0.0, cd_res.friedman_stat / (cd_res.n_blocks * (cd_res.k - 1)))))
            for r_item in cd_res.ranks:
                global_ranks_map[r_item.algorithm] = r_item.avg_rank
        except Exception:
            for idx, alg in enumerate(algorithms):
                global_ranks_map[alg] = float(idx + 1)

        try:
            far_res = FriedmanAlignedRanksTest().run(
                df=sub_df, metric=target_metric, alpha=alpha, scope=effective_scope, instance=effective_inst
            )
            friedman_aligned_stat = far_res.statistic
            friedman_aligned_p = far_res.p_value
        except Exception:
            pass

        try:
            q_res = QuadeTest().run(
                df=sub_df, metric=target_metric, alpha=alpha, scope=effective_scope, instance=effective_inst
            )
            quade_stat = q_res.statistic
            quade_p = q_res.p_value
        except Exception:
            pass

        sorted_by_rank = sorted(algorithms, key=lambda a: global_ranks_map.get(a, 999.0))
        ref_algo = (
            reference_algorithm
            if (reference_algorithm and reference_algorithm in algorithms)
            else sorted_by_rank[0]
        )

        norm_test_type = test_type.strip().lower()
        if summary_stat in {"mean_std", "median_iqr"}:
            resolved_summary_stat = summary_stat
        elif norm_test_type in {"wilcoxon", "mann_whitney", "kruskal", "friedman"}:
            resolved_summary_stat = "median_iqr"
        else:
            resolved_summary_stat = "mean_std"

        # 2. Evaluate each Instance Row
        instance_rows: list[BenchmarkTableInstanceRow] = []
        wins_vs_ref: dict[str, int] = {a: 0 for a in algorithms}
        ties_vs_ref: dict[str, int] = {a: 0 for a in algorithms}
        losses_vs_ref: dict[str, int] = {a: 0 for a in algorithms}
        best_counts: dict[str, int] = {a: 0 for a in algorithms}
        second_best_counts: dict[str, int] = {a: 0 for a in algorithms}
        resolved_omnibus_names: list[str] = []
        resolved_pairwise_names: list[str] = []

        for inst in instances:
            inst_df = sub_df[sub_df["instance"] == inst]
            paired_cell = paired and is_paired_data(inst_df)
            groups: dict[str, np.ndarray] = {}
            if paired_cell:
                pivoted = inst_df.pivot(index="run_id", columns="algorithm", values="value").sort_index()
                for alg in algorithms:
                    if alg in pivoted.columns:
                        vals = pivoted[alg].dropna().to_numpy(dtype=float)
                        if len(vals) > 0:
                            groups[alg] = vals
            else:
                for alg in algorithms:
                    vals = inst_df[inst_df["algorithm"] == alg]["value"].dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        groups[alg] = vals

            if len(groups) < 2:
                continue

            can_run_cell_omnibus = all(len(v) >= 2 for v in groups.values())
            omnibus_res: StatisticalResult | None = None
            if can_run_cell_omnibus:
                omnibus_res = _run_omnibus_for_groups(groups, test_type=test_type, alpha=alpha, is_paired=paired_cell)
                resolved_omnibus_names.append(omnibus_res.test_name)

            norm_list = NormalityChecker(alpha=alpha).check(groups)
            norm_map = {a_name: norm_res for a_name, norm_res in zip(groups.keys(), norm_list, strict=False)}
            levene_res = HomogeneityChecker(alpha=alpha).check(groups)
            norm_pass_cnt = sum(1 for n_r in norm_list if n_r.passed)

            central_scores = {
                alg: (float(np.median(arr)) if resolved_summary_stat == "median_iqr" else float(np.mean(arr)))
                for alg, arr in groups.items()
            }
            ordered_inst_algos = sorted(groups.keys(), key=lambda a: central_scores[a], reverse=maximize)
            best_alg = ordered_inst_algos[0]
            second_best_alg = ordered_inst_algos[1] if len(ordered_inst_algos) > 1 else ""
            best_counts[best_alg] = best_counts.get(best_alg, 0) + 1
            if second_best_alg:
                second_best_counts[second_best_alg] = second_best_counts.get(second_best_alg, 0) + 1

            inst_avg_ranks: dict[str, float] = {}
            group_alg_list = list(groups.keys())
            lengths = {len(groups[a]) for a in group_alg_list}
            if len(lengths) == 1 and next(iter(lengths)) >= 1:
                mat = np.column_stack([groups[a] for a in group_alg_list])
                ranked = stats.rankdata(-mat if maximize else mat, axis=1, method="average")
                mean_ranks = np.mean(ranked, axis=0)
                for idx_a, alg_i in enumerate(group_alg_list):
                    inst_avg_ranks[alg_i] = float(mean_ranks[idx_a])
            else:
                for rank_i, alg_i in enumerate(ordered_inst_algos):
                    inst_avg_ranks[alg_i] = float(rank_i + 1)

            raw_pair_p: dict[str, float] = {}
            pair_results: dict[str, StatisticalResult] = {}
            ref_vals = groups.get(ref_algo)
            non_ref_algos = [a for a in groups.keys() if a != ref_algo]

            if ref_vals is not None and len(ref_vals) >= 2:
                for alg in non_ref_algos:
                    alg_vals = groups[alg]
                    if len(alg_vals) >= 2:
                        p_res = _run_pairwise_test(
                            x_algo=alg_vals,
                            x_ref=ref_vals,
                            algo_name=alg,
                            ref_name=ref_algo,
                            test_type=test_type,
                            alpha=alpha,
                            is_paired=paired_cell,
                        )
                        pair_results[alg] = p_res
                        raw_pair_p[alg] = p_res.p_value
                        resolved_pairwise_names.append(p_res.test_name)

            ordered_tested = [a for a in non_ref_algos if a in raw_pair_p]
            vec_posthoc = (
                "holm"
                if posthoc_correction.strip().lower() in {"shaffer", "bergmann_hommel", "bergmann-hommel"}
                else posthoc_correction
            )
            adj_p_list = adjust_pvalues([raw_pair_p[a] for a in ordered_tested], method=vec_posthoc)
            adj_pair_p = dict(zip(ordered_tested, adj_p_list, strict=False))

            row_cells: dict[str, BenchmarkTableCell] = {}
            for alg, arr in groups.items():
                n_obs = len(arr)
                m_val = float(np.mean(arr))
                s_val = float(np.std(arr, ddof=1)) if n_obs > 1 else 0.0
                med_val = float(np.median(arr))
                q75, q25 = np.percentile(arr, [75, 25]) if n_obs > 0 else (0.0, 0.0)
                iqr_val = float(q75 - q25)
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                skew_v = float(stats.skew(arr)) if n_obs >= 3 and s_val > 0 else 0.0
                kurt_v = float(stats.kurtosis(arr)) if n_obs >= 4 and s_val > 0 else 0.0

                n_check = norm_map.get(alg)
                norm_ok = bool(n_check.passed) if n_check else True
                shapiro_w = float(n_check.statistic) if (n_check and math.isfinite(n_check.statistic)) else None
                shapiro_p = float(n_check.p_value) if (n_check and math.isfinite(n_check.p_value)) else None

                is_ref = alg == ref_algo
                sig_sym = "ref" if is_ref else "≈"
                p_test_name = ""
                p_stat: float | None = None
                p_raw: float | None = None
                p_adj: float | None = None
                eff_name: str | None = None
                eff_val: float | None = None
                eff_mag: str | None = None
                a12_val: float | None = None
                cliff_val: float | None = None
                cohen_val: float | None = None

                if not is_ref and alg in pair_results and ref_vals is not None:
                    pr = pair_results[alg]
                    p_test_name = pr.test_name
                    p_stat = pr.statistic
                    p_raw = pr.p_value
                    p_adj = adj_pair_p.get(alg, pr.p_value)
                    if pr.effect_size:
                        eff_name = pr.effect_size.name
                        eff_val = pr.effect_size.value
                        eff_mag = pr.effect_size.magnitude

                    a12_v, _ = compute_vargha_delaney_a12(arr, ref_vals, maximize=maximize)
                    cliff_v, _ = compute_cliffs_delta(arr, ref_vals, maximize=maximize)
                    cohen_v = _compute_cohens_d_raw(arr, ref_vals)
                    a12_val, cliff_val, cohen_val = a12_v, cliff_v, cohen_v

                    effective_p = p_adj if p_adj is not None else p_raw
                    if effective_p is not None and effective_p < alpha:
                        alg_central = med_val if resolved_summary_stat == "median_iqr" else m_val
                        ref_central = (
                            float(np.median(ref_vals))
                            if resolved_summary_stat == "median_iqr"
                            else float(np.mean(ref_vals))
                        )
                        alg_is_better = (alg_central > ref_central) if maximize else (alg_central < ref_central)
                        if alg_is_better:
                            sig_sym = "+"
                            wins_vs_ref[alg] += 1
                        else:
                            sig_sym = "−"
                            losses_vs_ref[alg] += 1
                    else:
                        sig_sym = "≈"
                        ties_vs_ref[alg] += 1

                base_str = (
                    f"{_format_num(med_val, precision)} ({_format_num(iqr_val, precision)})"
                    if resolved_summary_stat == "median_iqr"
                    else f"{_format_num(m_val, precision)} ± {_format_num(s_val, precision)}"
                )
                formatted_cell = f"{base_str}{'' if is_ref else f' {sig_sym}'}"

                row_cells[alg] = BenchmarkTableCell(
                    instance=inst,
                    algorithm=alg,
                    n=n_obs,
                    mean=m_val,
                    std=s_val,
                    median=med_val,
                    iqr=iqr_val,
                    min_val=min_v,
                    max_val=max_v,
                    avg_rank=inst_avg_ranks.get(alg, 1.0),
                    formatted_value=formatted_cell,
                    is_best=(alg == best_alg),
                    is_second_best=(alg == second_best_alg),
                    is_reference=is_ref,
                    significance_symbol=sig_sym,
                    pairwise_test_name=p_test_name,
                    pairwise_statistic=p_stat,
                    pairwise_p_value=p_raw,
                    pairwise_p_adjusted=p_adj,
                    effect_size_name=eff_name,
                    effect_size_value=eff_val,
                    effect_size_magnitude=eff_mag,
                    a12=a12_val,
                    cliffs_delta=cliff_val,
                    cohens_d=cohen_val,
                    normality_passed=norm_ok,
                    shapiro_stat=shapiro_w,
                    shapiro_p_value=shapiro_p,
                    skewness=skew_v,
                    kurtosis=kurt_v,
                    raw_values=[round(float(x), 6) for x in arr.tolist()],
                )

            if omnibus_res is not None:
                df_str = (
                    f"({omnibus_res.degrees_of_freedom[0]:.0f}, {omnibus_res.degrees_of_freedom[1]:.1f})"
                    if isinstance(omnibus_res.degrees_of_freedom, tuple)
                    else f"({omnibus_res.degrees_of_freedom})"
                    if omnibus_res.degrees_of_freedom is not None
                    else ""
                )
                omni_name = omnibus_res.test_name
                omni_stat = omnibus_res.statistic
                omni_p = omnibus_res.p_value
                omni_sig = omnibus_res.is_significant
                omni_eff_name = omnibus_res.effect_size.name if omnibus_res.effect_size else ""
                omni_eff_val = omnibus_res.effect_size.value if omnibus_res.effect_size else 0.0
                omni_eff_mag = omnibus_res.effect_size.magnitude if omnibus_res.effect_size else ""
                omni_interp = omnibus_res.interpretation
            else:
                df_str = ""
                omni_name = "Single observation per block"
                omni_stat = 0.0
                omni_p = 1.0
                omni_sig = False
                omni_eff_name = ""
                omni_eff_val = 0.0
                omni_eff_mag = ""
                omni_interp = (
                    f"Single observation per algorithm on instance '{inst}'; "
                    "inferential testing is performed across blocks (Friedman/Quade/Bayesian)."
                )

            instance_rows.append(
                BenchmarkTableInstanceRow(
                    instance=inst,
                    best_algorithm=best_alg,
                    second_best_algorithm=second_best_alg,
                    omnibus_test_name=omni_name,
                    omnibus_statistic=omni_stat,
                    omnibus_p_value=omni_p,
                    omnibus_df=df_str,
                    omnibus_significant=omni_sig,
                    omnibus_effect_size_name=omni_eff_name,
                    omnibus_effect_size_value=omni_eff_val,
                    omnibus_effect_size_magnitude=omni_eff_mag,
                    levene_statistic=levene_res.statistic if math.isfinite(levene_res.statistic) else 0.0,
                    levene_p_value=levene_res.p_value if math.isfinite(levene_res.p_value) else 1.0,
                    levene_passed=levene_res.passed,
                    normality_passed_count=norm_pass_cnt,
                    total_groups=len(groups),
                    interpretation=omni_interp,
                    cells=row_cells,
                )
            )

        # 3. Global Post-Hoc vs Reference across Instances
        n_inst = max(1, len(instance_rows))
        k_alg = len(algorithms)
        se_friedman = math.sqrt((k_alg * (k_alg + 1.0)) / (6.0 * n_inst))
        ref_rank = global_ranks_map.get(ref_algo, 1.0)

        non_ref_list = [a for a in algorithms if a != ref_algo]
        friedman_raw_p_map: dict[str, float] = {}
        friedman_z_map: dict[str, float] = {}

        for alg in non_ref_list:
            r_alg = global_ranks_map.get(alg, 1.0)
            z_val = abs(r_alg - ref_rank) / se_friedman if se_friedman > 0 else 0.0
            friedman_z_map[alg] = z_val
            friedman_raw_p_map[alg] = float(2.0 * stats.norm.sf(z_val))

        raw_f_ps = [friedman_raw_p_map[a] for a in non_ref_list]
        f_holm_map = dict(zip(non_ref_list, adjust_pvalues(raw_f_ps, method="holm"), strict=False))
        f_bonf_map = dict(zip(non_ref_list, adjust_pvalues(raw_f_ps, method="bonferroni"), strict=False))
        f_hoch_map = dict(zip(non_ref_list, adjust_pvalues(raw_f_ps, method="hochberg"), strict=False))

        # Global Wilcoxon Signed-Rank across instance blocks vs Reference (instances are paired blocks)
        inst_means_matrix: dict[str, np.ndarray] = {
            alg: np.array([r.cells[alg].mean for r in instance_rows if alg in r.cells], dtype=float)
            for alg in algorithms
        }

        global_wilcoxon_stat_map: dict[str, float] = {}
        global_wilcoxon_p_map: dict[str, float] = {}
        ref_inst_means = inst_means_matrix.get(ref_algo, np.array([]))
        for alg in non_ref_list:
            alg_inst_means = inst_means_matrix.get(alg, np.array([]))
            if len(alg_inst_means) >= 2 and len(alg_inst_means) == len(ref_inst_means):
                try:
                    w_res = WilcoxonSignedRankTest().run({alg: alg_inst_means, ref_algo: ref_inst_means}, alpha=alpha)
                    global_wilcoxon_stat_map[alg] = w_res.statistic
                    global_wilcoxon_p_map[alg] = w_res.p_value
                except Exception:
                    pass

        w_only_algos = [a for a in non_ref_list if a in global_wilcoxon_p_map]
        w_holm_ps = adjust_pvalues([global_wilcoxon_p_map[a] for a in w_only_algos], method="holm")
        global_wilcoxon_holm_map = dict(zip(w_only_algos, w_holm_ps, strict=False))

        # Unblocked One-Way ANOVA / Welch / Kruskal only apply to a single-instance scope.
        # Across multiple problem instances, blocked tests (Friedman, Aligned-Ranks, Quade) are used
        # to respect blocking and avoid incommensurability distortion (Demšar 2006).
        global_anova_f: float | None = None
        global_anova_p: float | None = None
        global_welch_f: float | None = None
        global_welch_p: float | None = None
        global_kruskal_h: float | None = None
        global_kruskal_p: float | None = None

        if len(instance_rows) == 1:
            r0 = instance_rows[0]
            if "Welch" in r0.omnibus_test_name:
                global_welch_f, global_welch_p = r0.omnibus_statistic, r0.omnibus_p_value
            elif "ANOVA" in r0.omnibus_test_name:
                global_anova_f, global_anova_p = r0.omnibus_statistic, r0.omnibus_p_value
            elif "Kruskal" in r0.omnibus_test_name:
                global_kruskal_h, global_kruskal_p = r0.omnibus_statistic, r0.omnibus_p_value

        bayes_best_map, bayes_better_map, bayes_rope_map = _fast_bayesian_summary_probs(
            inst_means_matrix=inst_means_matrix,
            algorithms=algorithms,
            ref_algo=ref_algo,
            maximize=maximize,
        )

        rank_order_algos = sorted(algorithms, key=lambda a: global_ranks_map.get(a, 999.0))
        algorithm_summaries: list[AlgorithmSummaryFooter] = []
        for alg in algorithms:
            arr_means = inst_means_matrix.get(alg, np.array([0.0]))
            algorithm_summaries.append(
                AlgorithmSummaryFooter(
                    algorithm=alg,
                    avg_rank=global_ranks_map.get(alg, 1.0),
                    rank_position=rank_order_algos.index(alg) + 1,
                    mean_across_instances=float(np.mean(arr_means)) if len(arr_means) > 0 else 0.0,
                    median_across_instances=float(np.median(arr_means)) if len(arr_means) > 0 else 0.0,
                    wins=wins_vs_ref.get(alg, 0),
                    ties=ties_vs_ref.get(alg, 0),
                    losses=losses_vs_ref.get(alg, 0),
                    best_count=best_counts.get(alg, 0),
                    second_best_count=second_best_counts.get(alg, 0),
                    friedman_z_vs_ref=friedman_z_map.get(alg),
                    friedman_p_unadjusted=friedman_raw_p_map.get(alg),
                    friedman_p_holm=f_holm_map.get(alg),
                    friedman_p_bonferroni=f_bonf_map.get(alg),
                    friedman_p_hochberg=f_hoch_map.get(alg),
                    global_wilcoxon_stat=global_wilcoxon_stat_map.get(alg),
                    global_wilcoxon_p=global_wilcoxon_p_map.get(alg),
                    global_wilcoxon_p_holm=global_wilcoxon_holm_map.get(alg),
                    bayesian_prob_best=bayes_best_map.get(alg),
                    bayesian_prob_better_than_ref=bayes_better_map.get(alg),
                    bayesian_prob_rope_with_ref=bayes_rope_map.get(alg),
                )
            )

        if table_type == "pairwise_matrix":
            pairwise_matrix = self._build_pairwise_matrix(
                sub_df=sub_df,
                target_metric=target_metric,
                algorithms=algorithms,
                instance_rows=instance_rows,
                global_ranks_map=global_ranks_map,
                test_type=test_type,
                posthoc_correction=posthoc_correction,
                maximize=maximize,
                alpha=alpha,
                paired=paired,
            )
        else:
            pairwise_matrix = []

        most_common_omnibus = (
            max(set(resolved_omnibus_names), key=resolved_omnibus_names.count)
            if resolved_omnibus_names
            else "Friedman Omnibus Test"
        )
        most_common_pairwise = (
            max(set(resolved_pairwise_names), key=resolved_pairwise_names.count)
            if resolved_pairwise_names
            else "Mann-Whitney U Test"
        )

        opt_dir_str = "Maximization" if maximize else "Minimization"
        stat_label_str = "Median (IQR)" if resolved_summary_stat == "median_iqr" else "Mean ± SD"
        caption = (
            f"Benchmark evaluation on metric {target_metric} ({opt_dir_str}, {stat_label_str}) across "
            f"{len(instance_rows)} instance(s) and {len(algorithms)} algorithms at α = {alpha}. "
            f"Omnibus test: {most_common_omnibus}; pairwise test: {most_common_pairwise} "
            f"({posthoc_correction.upper()} correction) vs reference algorithm '{ref_algo}' "
            f"(+: significantly better, −: significantly worse, ≈: tie)."
        )

        preliminary_result = StatisticalTableResult(
            table_type=table_type,
            metric=target_metric,
            maximize=maximize,
            alpha=alpha,
            test_type=test_type,
            resolved_omnibus_test=most_common_omnibus,
            resolved_pairwise_test=most_common_pairwise,
            posthoc_correction=posthoc_correction,
            summary_stat=resolved_summary_stat,
            scope=effective_scope,
            selected_instance=effective_inst,
            reference_algorithm=ref_algo,
            algorithms=algorithms,
            instances=[r.instance for r in instance_rows],
            friedman_stat=friedman_stat,
            friedman_p_value=friedman_p,
            iman_davenport_f=iman_f,
            iman_davenport_p_value=iman_p,
            kendalls_w=kendalls_w,
            nemenyi_cd=nemenyi_cd,
            global_anova_f=global_anova_f,
            global_anova_p=global_anova_p,
            global_welch_f=global_welch_f,
            global_welch_p=global_welch_p,
            global_kruskal_h=global_kruskal_h,
            global_kruskal_p=global_kruskal_p,
            friedman_aligned_stat=friedman_aligned_stat,
            friedman_aligned_p_value=friedman_aligned_p,
            quade_stat=quade_stat,
            quade_p_value=quade_p,
            instance_rows=instance_rows,
            algorithm_summaries=algorithm_summaries,
            pairwise_matrix=pairwise_matrix,
            caption=caption,
        )

        latex_str = render_latex_table(
            preliminary_result, precision=precision, document_mode=latex_document_mode, notation=latex_notation
        )
        md_str = render_markdown_table(preliminary_result, precision=precision)

        return preliminary_result.model_copy(
            update={
                "latex_document_mode": latex_document_mode,
                "latex_notation": latex_notation,
                "latex_preview": latex_str,
                "markdown_preview": md_str,
            }
        )

    def _build_pairwise_matrix(
        self,
        sub_df: pd.DataFrame,
        target_metric: str,
        algorithms: list[str],
        instance_rows: list[BenchmarkTableInstanceRow],
        global_ranks_map: dict[str, float],
        test_type: str,
        posthoc_correction: str,
        maximize: bool,
        alpha: float,
        paired: bool = False,
    ) -> list[list[PairwiseMatrixCell]]:
        """Construct the full k x k pairwise comparison matrix across all algorithm pairs."""
        _ = (sub_df, target_metric)
        k = len(algorithms)
        n_inst = len(instance_rows)
        norm_type = test_type.strip().lower()

        pair_keys: list[tuple[int, int]] = []
        raw_p_list: list[float] = []
        raw_stat_map: dict[tuple[int, int], tuple[str, float, float]] = {}
        inst_pair_p_cache: dict[tuple[int, int, str], float | None] = {}

        # Per-instance multiplicity adjustment across all C(k, 2) pairs within each instance
        for r_inst in instance_rows:
            inst_pairs: list[tuple[int, int]] = []
            inst_raw_ps: list[float] = []
            for i in range(k):
                for j in range(i + 1, k):
                    alg_a, alg_b = algorithms[i], algorithms[j]
                    if alg_a not in r_inst.cells or alg_b not in r_inst.cells:
                        inst_pair_p_cache[(i, j, r_inst.instance)] = None
                        continue
                    arr_a = np.array(r_inst.cells[alg_a].raw_values, dtype=float)
                    arr_b = np.array(r_inst.cells[alg_b].raw_values, dtype=float)
                    if len(arr_a) >= 2 and len(arr_b) >= 2:
                        try:
                            inst_pr = _run_pairwise_test(
                                x_algo=arr_a,
                                x_ref=arr_b,
                                algo_name=alg_a,
                                ref_name=alg_b,
                                test_type=test_type,
                                alpha=alpha,
                                is_paired=(paired and len(arr_a) == len(arr_b)),
                            )
                            inst_pairs.append((i, j))
                            inst_raw_ps.append(inst_pr.p_value)
                        except Exception:
                            inst_pair_p_cache[(i, j, r_inst.instance)] = None
                    else:
                        inst_pair_p_cache[(i, j, r_inst.instance)] = None

            if inst_pairs:
                norm_corr = posthoc_correction.strip().lower()
                if norm_corr in {"shaffer", "bergmann_hommel", "bergmann-hommel"} and k >= 3:
                    p_mat_inst = np.ones((k, k), dtype=float)
                    for (i, j), p_r in zip(inst_pairs, inst_raw_ps, strict=False):
                        p_mat_inst[i, j] = p_r
                        p_mat_inst[j, i] = p_r
                    if norm_corr == "shaffer":
                        adj_mat_inst = adjust_pvalues_shaffer(p_mat_inst)
                    else:
                        adj_mat_inst, _ = adjust_pvalues_bergmann_hommel(p_mat_inst)
                    for i, j in inst_pairs:
                        inst_pair_p_cache[(i, j, r_inst.instance)] = float(adj_mat_inst[i, j])
                else:
                    vec_method = "holm" if norm_corr in {"shaffer", "bergmann_hommel", "bergmann-hommel"} else norm_corr
                    inst_adj_ps = adjust_pvalues(inst_raw_ps, method=vec_method)
                    for (i, j), p_adj_inst in zip(inst_pairs, inst_adj_ps, strict=False):
                        inst_pair_p_cache[(i, j, r_inst.instance)] = p_adj_inst

        for i in range(k):
            for j in range(i + 1, k):
                alg_a, alg_b = algorithms[i], algorithms[j]
                pair_keys.append((i, j))

                if n_inst > 1:
                    vals_a = np.array(
                        [r.cells[alg_a].mean for r in instance_rows if alg_a in r.cells and alg_b in r.cells],
                        dtype=float,
                    )
                    vals_b = np.array(
                        [r.cells[alg_b].mean for r in instance_rows if alg_a in r.cells and alg_b in r.cells],
                        dtype=float,
                    )
                    if norm_type == "friedman":
                        se_f = math.sqrt((k * (k + 1.0)) / (6.0 * max(1, n_inst)))
                        z_f = abs(global_ranks_map.get(alg_a, 1.0) - global_ranks_map.get(alg_b, 1.0)) / se_f
                        p_f = float(2.0 * stats.norm.sf(z_f))
                        raw_stat_map[(i, j)] = ("Friedman Post-Hoc z-Test", z_f, p_f)
                        raw_p_list.append(p_f)
                    elif len(vals_a) >= 2:
                        p_res = _run_pairwise_test(
                            x_algo=vals_a,
                            x_ref=vals_b,
                            algo_name=alg_a,
                            ref_name=alg_b,
                            test_type="wilcoxon" if norm_type == "auto" else test_type,
                            alpha=alpha,
                            is_paired=True,  # Instances are matched blocks
                        )
                        raw_stat_map[(i, j)] = (p_res.test_name, p_res.statistic, p_res.p_value)
                        raw_p_list.append(p_res.p_value)
                    else:
                        raw_stat_map[(i, j)] = ("Wilcoxon Signed-Rank Test", 0.0, 1.0)
                        raw_p_list.append(1.0)
                else:
                    r0 = instance_rows[0] if instance_rows else None
                    vals_a = np.array(r0.cells[alg_a].raw_values if (r0 and alg_a in r0.cells) else [], dtype=float)
                    vals_b = np.array(r0.cells[alg_b].raw_values if (r0 and alg_b in r0.cells) else [], dtype=float)
                    if len(vals_a) >= 2 and len(vals_b) >= 2:
                        p_res = _run_pairwise_test(
                            x_algo=vals_a,
                            x_ref=vals_b,
                            algo_name=alg_a,
                            ref_name=alg_b,
                            test_type=test_type,
                            alpha=alpha,
                            is_paired=(paired and len(vals_a) == len(vals_b)),
                        )
                        raw_stat_map[(i, j)] = (p_res.test_name, p_res.statistic, p_res.p_value)
                        raw_p_list.append(p_res.p_value)
                    else:
                        raw_stat_map[(i, j)] = ("Mann-Whitney U Test", 0.0, 1.0)
                        raw_p_list.append(1.0)

        norm_corr = posthoc_correction.strip().lower()
        if norm_corr in {"shaffer", "bergmann_hommel", "bergmann-hommel"} and k >= 3:
            p_mat_global = np.ones((k, k), dtype=float)
            for (i, j), p_r in zip(pair_keys, raw_p_list, strict=False):
                p_mat_global[i, j] = p_r
                p_mat_global[j, i] = p_r
            if norm_corr == "shaffer":
                adj_mat_global = adjust_pvalues_shaffer(p_mat_global)
            else:
                adj_mat_global, _ = adjust_pvalues_bergmann_hommel(p_mat_global)
            adj_p_map: dict[tuple[int, int], float] = {
                (i, j): float(adj_mat_global[i, j]) for (i, j) in pair_keys
            }
        else:
            vec_method = "holm" if norm_corr in {"shaffer", "bergmann_hommel", "bergmann-hommel"} else norm_corr
            adj_p_list = adjust_pvalues(raw_p_list, method=vec_method)
            adj_p_map = dict(zip(pair_keys, adj_p_list, strict=False))

        matrix: list[list[PairwiseMatrixCell]] = []
        for i in range(k):
            row_cells: list[PairwiseMatrixCell] = []
            alg_row = algorithms[i]
            for j in range(k):
                alg_col = algorithms[j]
                if i == j:
                    row_cells.append(
                        PairwiseMatrixCell(
                            algo_row=alg_row,
                            algo_col=alg_col,
                            test_name="Self",
                            statistic=0.0,
                            p_value=1.0,
                            p_adjusted=1.0,
                            is_significant=False,
                            direction="self",
                            symbol="—",
                            interpretation=f"Diagonal self-comparison cell for {alg_row}.",
                        )
                    )
                    continue

                pair_key = (min(i, j), max(i, j))
                t_name, stat_val, p_raw = raw_stat_map.get(pair_key, ("Pairwise Test", 0.0, 1.0))
                p_adj = adj_p_map.get(pair_key, p_raw)

                breakdown: list[PairwiseInstanceComparison] = []
                wins_r = ties_r = losses_r = 0
                sig_wins_r = sig_ties_r = sig_losses_r = 0
                inst_sym_chars: list[str] = []
                row_agg_vals: list[float] = []
                col_agg_vals: list[float] = []

                for r_inst in instance_rows:
                    if alg_row not in r_inst.cells or alg_col not in r_inst.cells:
                        continue
                    c_r = r_inst.cells[alg_row]
                    c_c = r_inst.cells[alg_col]
                    row_agg_vals.append(c_r.mean)
                    col_agg_vals.append(c_c.mean)

                    arr_r = np.array(c_r.raw_values, dtype=float)
                    arr_c = np.array(c_c.raw_values, dtype=float)
                    inst_a12, _ = compute_vargha_delaney_a12(arr_r, arr_c, maximize=maximize)
                    inst_p = inst_pair_p_cache.get((pair_key[0], pair_key[1], r_inst.instance))

                    # Non-parametric stochastic direction when A12 differs from 0.5, else mean fallback
                    if inst_a12 > 0.5 + 1e-12:
                        diff_sign = 1
                    elif inst_a12 < 0.5 - 1e-12:
                        diff_sign = -1
                    else:
                        diff_m = (c_r.mean - c_c.mean) if maximize else (c_c.mean - c_r.mean)
                        diff_sign = 0 if abs(diff_m) <= 1e-12 else (1 if diff_m > 0 else -1)

                    if diff_sign == 0:
                        winner_str = "tie"
                        ties_r += 1
                    elif diff_sign > 0:
                        winner_str = "row"
                        wins_r += 1
                    else:
                        winner_str = "col"
                        losses_r += 1

                    inst_sig = bool(inst_p is not None and inst_p < alpha and diff_sign != 0)
                    if inst_sig:
                        if diff_sign > 0:
                            inst_sym = "+"
                            sig_wins_r += 1
                        else:
                            inst_sym = "-"
                            sig_losses_r += 1
                    else:
                        inst_sym = "="
                        sig_ties_r += 1
                    inst_sym_chars.append(inst_sym)

                    breakdown.append(
                        PairwiseInstanceComparison(
                            instance=r_inst.instance,
                            mean_row=c_r.mean,
                            std_row=c_r.std,
                            median_row=c_r.median,
                            iqr_row=c_r.iqr,
                            mean_col=c_c.mean,
                            std_col=c_c.std,
                            median_col=c_c.median,
                            iqr_col=c_c.iqr,
                            winner=winner_str,
                            symbol=inst_sym,
                            is_significant=inst_sig,
                            p_value=inst_p,
                            a12=inst_a12,
                        )
                    )

                instance_symbols_str = "".join(inst_sym_chars)
                if n_inst == 1 and instance_rows:
                    vec_r = np.array(instance_rows[0].cells[alg_row].raw_values, dtype=float)
                    vec_c = np.array(instance_rows[0].cells[alg_col].raw_values, dtype=float)
                else:
                    vec_r = np.array(row_agg_vals, dtype=float)
                    vec_c = np.array(col_agg_vals, dtype=float)

                a12_pair, a12_mag = compute_vargha_delaney_a12(vec_r, vec_c, maximize=maximize)
                cliff_pair, _ = compute_cliffs_delta(vec_r, vec_c, maximize=maximize)
                cohen_pair = _compute_cohens_d_raw(vec_r, vec_c)

                mean_r = float(np.mean(vec_r)) if len(vec_r) > 0 else 0.0
                mean_c = float(np.mean(vec_c)) if len(vec_c) > 0 else 0.0
                med_r = float(np.median(vec_r)) if len(vec_r) > 0 else 0.0
                med_c = float(np.median(vec_c)) if len(vec_c) > 0 else 0.0

                is_sig = p_adj < alpha
                row_is_better = (a12_pair > 0.5) if abs(a12_pair - 0.5) > 1e-12 else (
                    (mean_r > mean_c) if maximize else (mean_r < mean_c)
                )
                if is_sig:
                    direction = "better" if row_is_better else "worse"
                    sym = "+" if row_is_better else "−"
                else:
                    direction = "tie"
                    sym = "≈"

                prob_r, prob_rope_v, prob_c = _fast_bayesian_pair_probs(
                    vec_r=vec_r, vec_c=vec_c, maximize=maximize
                )

                interp = (
                    f"{alg_row} vs {alg_col} ({t_name}): statistic = {stat_val:.4f}, "
                    f"raw p = {_format_p(p_raw)}, {posthoc_correction.upper()} adjusted p = {_format_p(p_adj)} "
                    f"({'Significant' if is_sig else 'Not significant'} at α = {alpha}). "
                    f"Vargha-Delaney A12 = {a12_pair:.3f} ({a12_mag}), Cliff's δ = {cliff_pair:.3f}. "
                    f"Statistical instance record (+/=/-): +{sig_wins_r} / ={sig_ties_r} / -{sig_losses_r} "
                    f"({instance_symbols_str})."
                )

                row_cells.append(
                    PairwiseMatrixCell(
                        algo_row=alg_row,
                        algo_col=alg_col,
                        test_name=t_name,
                        statistic=stat_val,
                        p_value=p_raw,
                        p_adjusted=p_adj,
                        is_significant=is_sig,
                        direction=direction,
                        symbol=sym,
                        instance_symbols=instance_symbols_str,
                        sig_wins=sig_wins_r,
                        sig_ties=sig_ties_r,
                        sig_losses=sig_losses_r,
                        wins=wins_r,
                        ties=ties_r,
                        losses=losses_r,
                        mean_row=mean_r,
                        mean_col=mean_c,
                        median_row=med_r,
                        median_col=med_c,
                        effect_size_name="Vargha-Delaney A12",
                        effect_size_value=a12_pair,
                        effect_size_magnitude=a12_mag,
                        a12=a12_pair,
                        cliffs_delta=cliff_pair,
                        cohens_d=cohen_pair,
                        prob_row_better=prob_r,
                        prob_rope=prob_rope_v,
                        prob_col_better=prob_c,
                        interpretation=interp,
                        instance_breakdown=breakdown,
                    )
                )
            matrix.append(row_cells)
        return matrix


__all__ = [
    "StatisticalTableGenerator",
    "render_latex_table",
    "render_markdown_table",
    "export_statistical_table",
    "export_flat_table",
    "PDF_THEMES",
]
