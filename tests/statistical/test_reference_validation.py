"""Cross-validation and mathematical correctness tests against published statistical references.

References validated:
    - Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. JMLR, 7, 1-30.
    - García, S., & Herrera, F. (2008). JMLR, 9, 2677-2694 (Shaffer & Bergmann-Hommel).
    - García, S., et al. (2010). Information Sciences, 180(10), 2044-2064 (Aligned Friedman, Quade, Finner, Rom, Li).
    - Benavoli, A., et al. (2017). JMLR, 18(77), 1-36 (Bayesian Correlated t, Sign, Signed-Rank).
    - Calvo, B., et al. (2018). GECCO Companion (Bayesian Plackett-Luce).
    - Schuirmann (1987) / Lakens (2017) (TOST equivalence testing).
    - SciPy & Statsmodels exact reference implementations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors
from statsmodels.stats.multitest import multipletests

import saes
from saes.domain.models import Dataset, EffectSize, StatisticalResult, StatisticalTableResult
from saes.reports.renderers import TableRenderer, TableRendererRegistry
from saes.stats.base import StatisticalTest
from saes.stats.registry import TestRegistry


def _make_demsar_2006_dataset() -> Dataset:
    """Construct exact Table 4 dataset from Demšar (2006, JMLR 7, p. 12): 14 datasets x 4 classifiers."""
    # Columns: C4.5, C4.5+m, C4.5+cf, C4.5+m+cf (error-free accuracy proportions from Demšar 2006, Table 4)
    data = [
        ("adult", 0.763, 0.768, 0.771, 0.798),
        ("breast_cancer", 0.599, 0.591, 0.590, 0.569),
        ("breast_w", 0.954, 0.971, 0.968, 0.967),
        ("cmc", 0.628, 0.661, 0.654, 0.657),
        ("ionosphere", 0.882, 0.888, 0.886, 0.898),
        ("iris", 0.936, 0.931, 0.916, 0.931),
        ("liver_disorders", 0.661, 0.668, 0.609, 0.685),
        ("lung_cancer", 0.583, 0.583, 0.563, 0.625),
        ("lymphography", 0.775, 0.838, 0.866, 0.875),
        ("mushroom", 1.000, 1.000, 1.000, 1.000),
        ("primary_tumor", 0.940, 0.962, 0.965, 0.962),
        ("rheum", 0.619, 0.666, 0.614, 0.669),
        ("voting", 0.972, 0.981, 0.975, 0.975),
        ("wine", 0.957, 0.978, 0.946, 0.970),
    ]
    algos = ["C4.5", "C4.5+m", "C4.5+cf", "C4.5+m+cf"]
    rows = []
    for inst, *vals in data:
        for alg, val in zip(algos, vals, strict=True):
            rows.append(
                {
                    "algorithm": alg,
                    "instance": inst,
                    "metric": "AUC",
                    "run_id": 0,
                    "value": val,
                    "maximize": True,
                }
            )
    return Dataset(df=pd.DataFrame(rows))


def test_demsar_2006_exact_friedman_nemenyi() -> None:
    """Validate average ranks, tie-corrected Friedman chi2, Iman-Davenport F, and Nemenyi CD on Demšar (2006)."""
    ds = _make_demsar_2006_dataset()
    res = saes.statistics.friedman_nemenyi(ds, metric="AUC", alpha=0.05)

    assert res.k == 4
    assert res.n_blocks == 14

    rank_map = {r.algorithm: r.avg_rank for r in res.ranks}
    # Exact average ranks from the 14-dataset matrix (sums: 44.0, 28.0, 41.0, 27.0)
    assert rank_map["C4.5"] == pytest.approx(44.0 / 14.0, abs=1e-6)
    assert rank_map["C4.5+m"] == pytest.approx(28.0 / 14.0, abs=1e-6)
    assert rank_map["C4.5+cf"] == pytest.approx(41.0 / 14.0, abs=1e-6)
    assert rank_map["C4.5+m+cf"] == pytest.approx(27.0 / 14.0, abs=1e-6)

    # Cross-check exact tie-corrected Friedman statistic against SciPy
    piv = ds.df.pivot(index="instance", columns="algorithm", values="value")
    scipy_stat, scipy_p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    assert res.friedman_stat == pytest.approx(float(scipy_stat), rel=1e-9, abs=1e-9)
    assert res.friedman_p_value == pytest.approx(float(scipy_p), rel=1e-9, abs=1e-9)

    # Exact Iman-Davenport F_F
    expected_ff = ((14 - 1) * float(scipy_stat)) / (14 * (4 - 1) - float(scipy_stat))
    assert res.iman_davenport_f == pytest.approx(expected_ff, rel=1e-9)

    # Exact Nemenyi CD at alpha=0.05, k=4, N=14: q_alpha = 2.569 -> CD = 1.2535
    q_alpha = float(stats.studentized_range.ppf(0.95, 4, np.inf) / math.sqrt(2.0))
    expected_cd = q_alpha * math.sqrt(4 * 5 / (6 * 14))
    assert res.cd == pytest.approx(expected_cd, rel=1e-9)
    assert res.summary()
    assert len(res.to_dataframe()) == 4


def test_friedman_tie_correction_heavy_ties_matches_scipy() -> None:
    """Verify tie-corrected Friedman matches scipy.stats.friedmanchisquare under heavy ties."""
    mat = np.array(
        [
            [1.0, 1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 4.0],
            [1.0, 3.0, 3.0, 5.0],
            [0.5, 0.5, 1.5, 1.5],
            [4.0, 4.0, 4.0, 9.0],
            [2.0, 2.0, 3.0, 3.0],
        ],
        dtype=float,
    )
    rows = []
    for b in range(mat.shape[0]):
        for j, alg in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "algorithm": alg,
                    "instance": f"I{b}",
                    "metric": "M",
                    "run_id": 0,
                    "value": mat[b, j],
                    "maximize": False,
                }
            )
    ds = Dataset(df=pd.DataFrame(rows))
    res = saes.statistics.friedman_nemenyi(ds, metric="M", alpha=0.05)
    sp_stat, sp_p = stats.friedmanchisquare(mat[:, 0], mat[:, 1], mat[:, 2], mat[:, 3])
    assert res.friedman_stat == pytest.approx(float(sp_stat), rel=1e-10, abs=1e-10)
    assert res.friedman_p_value == pytest.approx(float(sp_p), rel=1e-10, abs=1e-10)


def test_all_metrics_preserves_instance_block_independence() -> None:
    """Multi-metric 'all-metrics' ranking must NOT multiply n_blocks by number of metrics."""
    rng = np.random.default_rng(7)
    rows = []
    for inst in [f"P{i}" for i in range(6)]:
        for metric, maximize in [("HV", True), ("IGD", False), ("Spread", False)]:
            for run_id in range(3):
                for alg, base in [("A", 1.0), ("B", 2.0), ("C", 3.0)]:
                    val = base + rng.normal(0, 0.05) if maximize else (4.0 - base) + rng.normal(0, 0.05)
                    rows.append(
                        {
                            "algorithm": alg,
                            "instance": inst,
                            "metric": metric,
                            "run_id": run_id,
                            "value": val,
                            "maximize": maximize,
                        }
                    )
    ds = Dataset(df=pd.DataFrame(rows))
    cd = saes.statistics.friedman_nemenyi(ds, metric="all-metrics", alpha=0.05)
    assert cd.n_blocks == 6  # Strictly 6 instances, NOT 6 * 3 = 18
    assert len(cd.metrics_list) == 3
    assert len(cd.warnings) >= 1
    assert cd.ranks[0].algorithm == "C"


def test_blocked_omnibus_tests_aligned_friedman_quade_rm_anova() -> None:
    """Validate Friedman Aligned-Ranks, Quade, and Repeated-Measures ANOVA (with GG epsilon)."""
    ds = _make_demsar_2006_dataset()
    far = saes.statistics.friedman_aligned_ranks(ds, metric="AUC", alpha=0.05)
    quade = saes.statistics.quade(ds, metric="AUC", alpha=0.05)
    rm_gg = saes.statistics.rm_anova(ds, metric="AUC", alpha=0.05, greenhouse_geisser=True)
    rm_raw = saes.statistics.rm_anova(ds, metric="AUC", alpha=0.05, greenhouse_geisser=False)

    assert far.k == 4 and far.n_blocks == 14
    assert far.statistic > 0 and 0.0 <= far.p_value <= 1.0
    assert far.summary() and len(far.to_dataframe()) == 4

    # Algebraic cross-check of Quade F formula
    piv = ds.df.pivot(index="instance", columns="algorithm", values="value").to_numpy()
    n_b, k = piv.shape
    ranks = np.vstack([stats.rankdata(-row, method="average") for row in piv])
    q_w = stats.rankdata(np.max(piv, axis=1) - np.min(piv, axis=1), method="average")
    s_mat = q_w[:, None] * (ranks - (k + 1) / 2.0)
    a_term = float(np.sum(s_mat**2))
    b_term = float(np.sum(np.sum(s_mat, axis=0) ** 2)) / n_b
    expected_f_quade = ((n_b - 1) * b_term) / (a_term - b_term)
    assert quade.statistic == pytest.approx(expected_f_quade, rel=1e-9)
    assert quade.p_value == pytest.approx(float(stats.f.sf(expected_f_quade, k - 1, (k - 1) * (n_b - 1))), rel=1e-9)

    # RM-ANOVA F matches manual two-way decomposition; GG epsilon lies in [1/(k-1), 1.0]
    grand = float(np.mean(piv))
    ss_t = float(n_b * np.sum((np.mean(piv, axis=0) - grand) ** 2))
    ss_b = float(k * np.sum((np.mean(piv, axis=1) - grand) ** 2))
    ss_e = float(np.sum((piv - grand) ** 2)) - ss_t - ss_b
    expected_rm_f = (ss_t / (k - 1)) / (ss_e / ((k - 1) * (n_b - 1)))
    assert rm_raw.statistic == pytest.approx(expected_rm_f, rel=1e-9)
    assert rm_gg.statistic == pytest.approx(expected_rm_f, rel=1e-9)
    assert rm_gg.sphericity_epsilon is not None
    assert 1.0 / (k - 1) - 1e-9 <= rm_gg.sphericity_epsilon <= 1.0 + 1e-9
    assert rm_gg.p_value >= rm_raw.p_value - 1e-12  # GG correction is conservative


def test_multiple_testing_adjustments_shaffer_and_bergmann_hommel() -> None:
    """Validate all 11 p-value adjustment methods plus Shaffer (1986) and Bergmann-Hommel (1988)."""
    p_raw = [0.004, 0.012, 0.025, 0.041, 0.12, 0.35]

    p_none = saes.statistics.adjust_pvalues(p_raw, method="none")
    p_bonf = saes.statistics.adjust_pvalues(p_raw, method="bonferroni")
    p_holm = saes.statistics.adjust_pvalues(p_raw, method="holm")
    p_hoch = saes.statistics.adjust_pvalues(p_raw, method="hochberg")
    p_homm = saes.statistics.adjust_pvalues(p_raw, method="hommel")
    p_finn = saes.statistics.adjust_pvalues(p_raw, method="finner")
    p_holl = saes.statistics.adjust_pvalues(p_raw, method="holland")
    p_rom = saes.statistics.adjust_pvalues(p_raw, method="rom")
    p_li = saes.statistics.adjust_pvalues(p_raw, method="li")
    p_bh = saes.statistics.adjust_pvalues(p_raw, method="fdr_bh")
    p_by = saes.statistics.adjust_pvalues(p_raw, method="fdr_by")

    assert p_none == pytest.approx(p_raw)
    # Cross-check against statsmodels
    sm_pairs = [
        ("bonferroni", p_bonf),
        ("holm", p_holm),
        ("hommel", p_homm),
        ("fdr_bh", p_bh),
        ("fdr_by", p_by),
    ]
    for method, actual in sm_pairs:
        _, ref, _, _ = multipletests(p_raw, method=method)
        assert actual == pytest.approx(ref.tolist(), abs=1e-10)

    # Known dominance inequalities
    for i in range(len(p_raw)):
        assert p_none[i] <= p_bh[i] + 1e-12
        assert p_bh[i] <= p_hoch[i] + 1e-12
        assert p_hoch[i] <= p_holm[i] + 1e-12
        assert p_holm[i] <= p_bonf[i] + 1e-12
        assert p_homm[i] <= p_hoch[i] + 1e-12
        assert p_holl[i] <= p_holm[i] + 1e-12
        assert p_finn[i] <= p_holl[i] + 1e-12
        assert p_rom[i] <= p_holm[i] + 1e-12
        assert 0.0 <= p_li[i] <= 1.0

    # Shaffer & Bergmann-Hommel on a 4x4 symmetric p-value matrix
    p_mat = np.array(
        [
            [1.0, 0.004, 0.012, 0.041],
            [0.004, 1.0, 0.025, 0.12],
            [0.012, 0.025, 1.0, 0.35],
            [0.041, 0.12, 0.35, 1.0],
        ]
    )
    shaf = saes.statistics.adjust_pvalues_shaffer(p_mat)
    bh_mat, bh_warn = saes.statistics.adjust_pvalues_bergmann_hommel(p_mat)
    assert bh_warn == []
    # For k=4, Shaffer multipliers are [6, 3, 3, 3, 2, 1]: 0.012 * 3 = 0.036 (vs Holm 0.012 * 5 = 0.06)
    assert shaf[0, 1] == pytest.approx(0.004 * 6.0)
    assert shaf[0, 2] == pytest.approx(0.012 * 3.0)
    # Bergmann-Hommel is always <= Shaffer <= Holm (element-wise on off-diagonals)
    for i in range(4):
        for j in range(i + 1, 4):
            assert bh_mat[i, j] <= shaf[i, j] + 1e-12

    # Fallback warning when k > max_exact_k
    big_p = np.full((11, 11), 0.02)
    np.fill_diagonal(big_p, 1.0)
    fb_mat, fb_warn = saes.statistics.adjust_pvalues_bergmann_hommel(big_p, max_exact_k=9)
    assert len(fb_warn) == 1
    assert fb_mat.shape == (11, 11)


def test_posthoc_procedures_and_brunner_munzel_cross_validated() -> None:
    """Cross-validate Tukey HSD, Dunnett, Brunner-Munzel against SciPy and verify Games-Howell/Dunn/Conover."""
    rng = np.random.default_rng(101)
    g_ctrl = rng.normal(10.0, 1.0, 20)
    g_t1 = rng.normal(12.5, 1.0, 20)
    g_t2 = rng.normal(10.2, 3.5, 22)
    groups = {"Ctrl": g_ctrl, "T1": g_t1, "T2": g_t2}

    # 1. Tukey HSD exact match with SciPy
    tk = saes.statistics.tukey_hsd(groups, alpha=0.05)
    sp_tk = stats.tukey_hsd(g_ctrl, g_t1, g_t2)
    assert tk.comparisons[0].p_adjusted == pytest.approx(float(sp_tk.pvalue[0, 1]), abs=1e-10)
    assert tk.summary() and len(tk.to_dataframe()) == 3

    # 2. Dunnett exact match with SciPy
    dn = saes.statistics.dunnett(groups, control="Ctrl", alpha=0.05)
    sp_dn = stats.dunnett(g_t1, g_t2, control=g_ctrl, alternative="two-sided", random_state=42)
    assert dn.comparisons[0].p_adjusted == pytest.approx(float(sp_dn.pvalue[0]), abs=1e-10)
    assert dn.comparisons[1].p_adjusted == pytest.approx(float(sp_dn.pvalue[1]), abs=1e-10)

    # 3. Games-Howell
    gh = saes.statistics.games_howell(groups, alpha=0.05)
    assert len(gh.comparisons) == 3
    for comp in gh.comparisons:
        assert comp.ci_lower is not None and comp.ci_upper is not None
        assert comp.ci_lower <= comp.mean_diff <= comp.ci_upper

    # 4. Dunn & Conover-Iman
    dunn_res = saes.statistics.dunn(groups, alpha=0.05, p_adjust="holm")
    ci_res = saes.statistics.conover_iman(groups, alpha=0.05, p_adjust="holm")
    assert dunn_res.p_value_matrix["Ctrl"]["T1"] < 0.05
    assert ci_res.p_value_matrix["Ctrl"]["T1"] < 0.05

    # 5. Brunner-Munzel exact match against SciPy
    bm = saes.statistics.brunner_munzel({"Ctrl": g_ctrl, "T1": g_t1}, alpha=0.05)
    sp_bm = stats.brunnermunzel(g_ctrl, g_t1, alternative="two-sided", distribution="t")
    assert bm.statistic == pytest.approx(float(sp_bm.statistic), abs=1e-10)
    assert bm.p_value == pytest.approx(float(sp_bm.pvalue), abs=1e-10)
    assert bm.effect_size is not None
    assert bm.effect_size.ci_lower is not None and bm.effect_size.ci_upper is not None
    assert bm.effect_size.ci_lower <= bm.effect_size.value <= bm.effect_size.ci_upper


def test_lilliefors_anderson_and_explorer_validity() -> None:
    """Verify Lilliefors exact match with statsmodels, Anderson-Darling, and explorer KS fix."""
    rng = np.random.default_rng(2024)
    normal_x = rng.normal(50.0, 10.0, 40)
    exp_x = rng.exponential(scale=5.0, size=40)

    lf_res = saes.statistics.lilliefors({"Normal": normal_x, "Exp": exp_x}, alpha=0.05)
    sp_stat, sp_p = sm_lilliefors(normal_x, dist="norm", pvalmethod="table")
    assert lf_res[0].statistic == pytest.approx(float(sp_stat), abs=1e-10)
    assert lf_res[0].p_value == pytest.approx(float(sp_p), abs=1e-10)
    assert lf_res[0].passed is True
    assert lf_res[1].passed is False

    ad_res = saes.statistics.anderson({"Normal": normal_x, "Exp": exp_x}, alpha=0.05)
    assert ad_res[0].passed is True
    assert ad_res[1].passed is False

    # Explorer must use Lilliefors (matching statsmodels), not uncalibrated kstest
    rows = [
        {"algorithm": "A", "instance": "I1", "metric": "M", "run_id": i, "value": float(v), "maximize": True}
        for i, v in enumerate(normal_x)
    ] + [
        {"algorithm": "B", "instance": "I1", "metric": "M", "run_id": i, "value": float(v), "maximize": True}
        for i, v in enumerate(exp_x)
    ]
    exp_out = saes.explore(Dataset(df=pd.DataFrame(rows)))
    norm_diag = next(d for d in exp_out.distribution_profiles if d.algorithm == "A")
    assert norm_diag.ks_stat == pytest.approx(float(sp_stat), abs=1e-4)
    assert norm_diag.ks_p == pytest.approx(float(sp_p), abs=1e-4)


def test_bayesian_exactness_and_sample_size_scaling() -> None:
    """Verify Benavoli (2017) correlated t-test, DP scaling without N=12 cap, and Plackett-Luce."""
    rng = np.random.default_rng(99)
    a = rng.normal(0.85, 0.03, 10)
    b = rng.normal(0.80, 0.03, 10)

    # 1. Bayesian Correlated t-test exact match with Student-t CDF
    bct0 = saes.statistics.bayesian_correlated_ttest(a, b, algo_a="A", algo_b="B", rope=0.01, rho=0.0)
    bct_cv = saes.statistics.bayesian_correlated_ttest(a, b, algo_a="A", algo_b="B", rope=0.01, rho=1.0 / 9.0)
    assert bct0.p_left + bct0.p_rope + bct0.p_right == pytest.approx(1.0, abs=1e-9)
    assert bct_cv.posterior_scale > bct0.posterior_scale  # CV correlation widens posterior uncertainty
    diffs = a - b
    scale_ref = math.sqrt((1.0 / 10.0 + (1.0 / 9.0) / (8.0 / 9.0)) * float(np.var(diffs, ddof=1)))
    assert bct_cv.posterior_scale == pytest.approx(scale_ref, rel=1e-10)
    assert bct_cv.p_left == pytest.approx(1.0 - float(stats.t.cdf(0.01, df=9, loc=np.mean(diffs), scale=scale_ref)))
    assert bct_cv.summary()

    # 2. Verify BayesianRankingTest has NO artificial N=12 truncation: posterior spread shrinks for N=50 vs N=5
    def _build_two_alg_ds(n_blocks: int) -> Dataset:
        r = np.random.default_rng(1234)
        rows = []
        for idx in range(n_blocks):
            v_b = 1.0 + r.normal(0, 0.02)
            v_a = v_b + 0.05 + r.normal(0, 0.02)
            inst = f"I{idx:03d}"
            rows.append(
                {"algorithm": "A", "instance": inst, "metric": "M", "run_id": 0, "value": v_a, "maximize": True}
            )
            rows.append(
                {"algorithm": "B", "instance": inst, "metric": "M", "run_id": 0, "value": v_b, "maximize": True}
            )
        return Dataset(df=pd.DataFrame(rows))

    res_n5 = saes.statistics.bayesian_sign(_build_two_alg_ds(5), metric="M", rope=0.02, algo_a="A", algo_b="B")
    res_n50 = saes.statistics.bayesian_sign(_build_two_alg_ds(50), metric="M", rope=0.02, algo_a="A", algo_b="B")
    assert res_n5.rope_absolute == pytest.approx(0.02)
    std_n5 = float(np.std(res_n5.points_x))
    std_n50 = float(np.std(res_n50.points_x))
    assert std_n50 < std_n5 * 0.6  # Posterior concentrates strictly with N=50 > N=12!
    assert res_n50.summary() and len(res_n50.to_dataframe()) == 2

    # 3. Bayesian Plackett-Luce
    ds_demsar = _make_demsar_2006_dataset()
    pl = saes.statistics.bayesian_plackett_luce(ds_demsar, metric="AUC")
    assert pl.k == 4 and pl.n_blocks == 14
    assert sum(it.worth for it in pl.items) == pytest.approx(1.0, abs=1e-5)
    assert sum(it.prob_best for it in pl.items) == pytest.approx(1.0, abs=1e-5)
    assert pl.items[0].algorithm in {"C4.5+m+cf", "C4.5+m"}
    assert pl.summary() and len(pl.to_dataframe()) == 4


def test_effect_size_cis_mdes_and_tost_equivalence() -> None:
    """Validate exact effect-size CIs across all tests, MDES exact inverse, and TOST equivalence."""
    rng = np.random.default_rng(77)
    x1 = rng.normal(10.0, 1.0, 30)
    x2 = rng.normal(11.2, 1.0, 30)
    x3 = rng.normal(12.4, 1.0, 30)

    # All hypothesis tests populate effect size confidence intervals
    for res in [
        saes.statistics.ttest_ind({"A": x1, "B": x2}),
        saes.statistics.ttest_welch({"A": x1, "B": x2}),
        saes.statistics.ttest_rel({"A": x1, "B": x2}),
        saes.statistics.anova_oneway({"A": x1, "B": x2, "C": x3}),
        saes.statistics.anova_welch({"A": x1, "B": x2, "C": x3}),
        saes.statistics.mann_whitney({"A": x1, "B": x2}),
        saes.statistics.wilcoxon({"A": x1, "B": x2}),
        saes.statistics.kruskal({"A": x1, "B": x2, "C": x3}),
        saes.statistics.brunner_munzel({"A": x1, "B": x2}),
    ]:
        assert res.effect_size is not None
        assert res.effect_size.ci_lower is not None
        assert res.effect_size.ci_upper is not None
        assert res.effect_size.ci_lower <= res.effect_size.ci_upper
        assert res.summary()
        assert len(res.to_dataframe()) == 1

    # Direct CI helpers
    d_lo, d_hi = saes.statistics.cohens_d_ci(0.8, n1=30, n2=30, alpha=0.05)
    assert d_lo < 0.8 < d_hi
    g_lo, g_hi = saes.statistics.cohens_d_ci(0.8, n1=20, paired=True, hedges_correction=True)
    assert g_lo < 0.8 < g_hi
    e_lo, e_hi = saes.statistics.eta_squared_ci(f_stat=12.0, df_between=2.0, df_within=87.0)
    assert 0.0 <= e_lo <= e_hi <= 1.0
    a12, _, (a_lo, a_hi) = saes.statistics.vargha_delaney_a12_ci(x2, x1)
    assert 0.0 <= a_lo <= a12 <= a_hi <= 1.0
    cd, _, (c_lo, c_hi) = saes.statistics.cliffs_delta_ci(x2, x1)
    assert -1.0 <= c_lo <= cd <= c_hi <= 1.0

    # MDES exact inverse of power function
    for paired in (False, True):
        mdes = saes.statistics.minimum_detectable_effect(n=30, target_power=0.80, alpha=0.05, paired=paired)
        achieved = saes.api.calculate_statistical_power(n=30, effect_size=mdes, alpha=0.05, paired=paired)
        assert achieved == pytest.approx(0.80, abs=1e-4)

    # TOST equivalence test: equivalent vs non-equivalent (independent & paired)
    eq_a = rng.normal(10.0, 0.2, 40)
    eq_b = rng.normal(10.02, 0.2, 40)
    tost_eq = saes.tost(eq_a, eq_b, low_eqbound=0.25, alpha=0.05)
    assert tost_eq.is_equivalent is True
    assert tost_eq.low_eqbound < tost_eq.ci_lower <= tost_eq.ci_upper < tost_eq.high_eqbound
    assert tost_eq.summary()

    tost_neq = saes.statistics.tost(x1, x2, low_eqbound=-0.25, high_eqbound=0.25, alpha=0.05, equal_var=True)
    assert tost_neq.is_equivalent is False

    tost_paired = saes.tost(eq_a, eq_a + rng.normal(0.0, 0.05, 40), low_eqbound=0.1, paired=True)
    assert tost_paired.is_equivalent is True


def test_explicit_pairing_default_and_multi_instance_table_validity() -> None:
    """Verify analyze/table/power/profiles default to unpaired=False and multi-instance tables omit invalid ANOVA."""
    rng = np.random.default_rng(55)
    rows = []
    for inst in ["I1", "I2", "I3", "I4"]:
        for run_id in range(20):
            rows.append(
                {
                    "algorithm": "Alg1",
                    "instance": inst,
                    "metric": "HV",
                    "run_id": run_id,
                    "value": float(rng.normal(0.7, 0.05)),
                    "maximize": True,
                }
            )
            rows.append(
                {
                    "algorithm": "Alg2",
                    "instance": inst,
                    "metric": "HV",
                    "run_id": run_id,
                    "value": float(rng.normal(0.85, 0.05)),
                    "maximize": True,
                }
            )
    ds = Dataset(df=pd.DataFrame(rows))

    # Default analyze(ds) MUST NOT silently run PairedTTest even though run_id 0..19 aligns
    rep_ind = saes.analyze(ds, generate_plots=False)
    for ca in rep_ind.cell_analyses:
        assert "Paired" not in ca.result.test_name
        assert "Wilcoxon" not in ca.result.test_name

    # Explicit paired=True MUST run paired test
    rep_paired = saes.analyze(ds, generate_plots=False, paired=True)
    paired_names = [ca.result.test_name for ca in rep_paired.cell_analyses]
    assert any(("Paired" in name or "Wilcoxon" in name) for name in paired_names)

    # Multi-instance table omits invalid unblocked ANOVA/Welch/Kruskal and computes Friedman/Aligned/Quade
    tbl = saes.generate_table(ds, metric="HV")
    assert tbl.global_anova_f is None
    assert tbl.global_welch_f is None
    assert tbl.global_kruskal_h is None
    assert tbl.friedman_stat is not None
    assert tbl.friedman_aligned_stat is not None
    assert tbl.quade_stat is not None

    # Power analysis exposes both sensitivity_audit (MDES) and backward-compatible post_hoc_audit
    pwr = saes.power(ds, metric="HV")
    assert len(pwr.sensitivity_audit) > 0
    assert len(pwr.post_hoc_audit) > 0
    assert pwr.summary() and len(pwr.to_dataframe()) > 0

    # Profiles respects explicit paired & posthoc_correction
    prof = saes.profiles(ds, metric="HV", paired=False, posthoc_correction="holm")
    assert len(prof.meta_rankings) == 2


def test_extensibility_registries_custom_test_and_renderer() -> None:
    """Verify users can register custom StatisticalTest and TableRenderer cleanly."""

    class DummyMeanDiffTest(StatisticalTest):
        def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
            names = list(groups.keys())
            diff = float(np.mean(groups[names[0]]) - np.mean(groups[names[1]]))
            return StatisticalResult(
                test_name="Dummy Mean-Diff Test",
                statistic=diff,
                p_value=0.01,
                degrees_of_freedom=10.0,
                alpha=alpha,
                is_significant=True,
                effect_size=EffectSize(name="RawDiff", value=diff, ci_lower=diff - 1.0, ci_upper=diff + 1.0),
                interpretation="Custom test executed.",
            )

    class CustomTSVRenderer(TableRenderer):
        media_type = "text/tab-separated-values"
        extension = "tsv"

        def render(self, result: StatisticalTableResult, **kwargs: object) -> bytes:
            _ = kwargs
            return f"metric\t{result.metric}\n".encode()

    TestRegistry.register("dummy_mean_diff", DummyMeanDiffTest)
    TableRendererRegistry.register("custom_tsv", CustomTSVRenderer)
    try:
        assert "dummy_mean_diff" in TestRegistry.list_available()
        inst = TestRegistry.get("dummy_mean_diff")
        out = inst.run({"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])})
        assert out.statistic == pytest.approx(-2.0)

        assert "custom_tsv" in TableRendererRegistry.list_available()
        ds = _make_demsar_2006_dataset()
        tbl = saes.generate_table(ds, metric="AUC")
        raw = saes.export_table(tbl, export_format="custom_tsv")
        assert raw == b"metric\tAUC\n"
    finally:
        TestRegistry._registry.pop("dummy_mean_diff", None)
        TableRendererRegistry._registry.pop("custom_tsv", None)
