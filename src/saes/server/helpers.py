"""Shared server-side orchestration helpers for SAES REST API endpoints."""

from __future__ import annotations

import numpy as np

import saes
from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    BayesianCorrelatedTResult,
    Dataset,
    PostHocResult,
    TOSTResult,
)
from saes.server.schemas import CellAnalysisSchema, RankingAnalysisResponse


def build_cell_analysis_schemas(
    dataset: Dataset,
    alpha: float = 0.05,
    paired: bool = False,
) -> list[CellAnalysisSchema]:
    """Run automated analysis and map results to typed REST response schemas."""
    report = saes.analyze(dataset, alpha=alpha, generate_plots=False, paired=paired)
    results: list[CellAnalysisSchema] = []
    for cell in report.cell_analyses:
        res = cell.result
        results.append(
            CellAnalysisSchema(
                instance=cell.instance,
                metric=cell.metric,
                test_name=res.test_name,
                statistic=res.statistic,
                p_value=res.p_value,
                degrees_of_freedom=res.degrees_of_freedom,
                is_significant=res.is_significant,
                effect_size_name=res.effect_size.name if res.effect_size else None,
                effect_size_value=res.effect_size.value if res.effect_size else None,
                effect_size_ci_lower=res.effect_size.ci_lower if res.effect_size else None,
                effect_size_ci_upper=res.effect_size.ci_upper if res.effect_size else None,
                effect_size_magnitude=res.effect_size.magnitude if res.effect_size else None,
                assumptions=list(res.assumptions),
                warnings=list(res.warnings),
                interpretation=res.interpretation,
                group_summaries=res.group_summaries,
            )
        )
    return results


def run_ranking_bundle(
    dataset: Dataset,
    metric: str,
    alpha: float = 0.05,
    scope: str = "all-instances",
    instance: str | None = None,
    rope_percent: float = 1.0,
    rope: float | None = None,
    prior_strength: float = 1.0,
    algo_a: str | None = None,
    algo_b: str | None = None,
    bayesian_method: str = "signed-rank",
) -> RankingAnalysisResponse:
    """Compute complete frequentist blocked-omnibus + Bayesian ranking bundle."""
    cd_res = saes.statistics.friedman_nemenyi(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        scope=scope,
        instance=instance,
    )
    bayes_sign_res = saes.statistics.bayesian_sign(
        dataset=dataset,
        metric=metric,
        scope=scope,
        instance=instance,
        rope_percent=rope_percent,
        rope=rope,
        prior_strength=prior_strength,
        algo_a=algo_a,
        algo_b=algo_b,
    )
    bayes_sr_res = saes.statistics.bayesian_signed_rank(
        dataset=dataset,
        metric=metric,
        scope=scope,
        instance=instance,
        rope_percent=rope_percent,
        rope=rope,
        prior_strength=prior_strength,
        algo_a=algo_a,
        algo_b=algo_b,
    )
    active_bayes = (
        bayes_sign_res
        if bayesian_method.strip().lower() in {"sign", "bayesian-sign", "sign-test"}
        else bayes_sr_res
    )

    pl_res = None
    try:
        pl_res = saes.statistics.bayesian_plackett_luce(
            dataset=dataset,
            metric=metric,
            scope=scope,
            instance=instance,
            prior_alpha=prior_strength,
        )
    except Exception:
        pass

    far_res = None
    try:
        far_res = saes.statistics.friedman_aligned_ranks(
            dataset=dataset, metric=metric, alpha=alpha, scope=scope, instance=instance
        )
    except Exception:
        pass

    quade_res = None
    try:
        quade_res = saes.statistics.quade(
            dataset=dataset, metric=metric, alpha=alpha, scope=scope, instance=instance
        )
    except Exception:
        pass

    rm_res = None
    try:
        rm_res = saes.statistics.rm_anova(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            scope=scope,
            instance=instance,
            greenhouse_geisser=True,
        )
    except Exception:
        pass

    return RankingAnalysisResponse(
        critical_distance=cd_res,
        bayesian_ranking=active_bayes,
        bayesian_sign=bayes_sign_res,
        bayesian_signed_rank=bayes_sr_res,
        bayesian_plackett_luce=pl_res,
        friedman_aligned=far_res,
        quade=quade_res,
        rm_anova=rm_res,
    )


def run_posthoc_for_dataset(
    dataset: Dataset,
    metric: str | None = None,
    instance: str | None = None,
    method: str = "auto",
    control: str | None = None,
    p_adjust: str = "holm",
    alpha: float = 0.05,
) -> PostHocResult:
    """Execute parametric or non-parametric post-hoc procedure on an (instance, metric) cell."""
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no evaluation metrics.")
    target_metric = metric if (metric and metric in dataset.metrics) else dataset.metrics[0]
    sub = dataset.df[dataset.df["metric"] == target_metric]
    if sub.empty:
        raise StatisticalAnalysisError(f"Metric '{target_metric}' not found in dataset.")

    instances = sorted(sub["instance"].unique().tolist())
    target_inst = instance if (instance and instance in instances) else instances[0]
    cell = sub[sub["instance"] == target_inst]

    groups: dict[str, np.ndarray] = {}
    for alg_key, grp in cell.groupby("algorithm"):
        arr = grp["value"].dropna().to_numpy(dtype=float)
        if len(arr) >= 2:
            groups[str(alg_key)] = arr

    if len(groups) < 2:
        raise StatisticalAnalysisError(
            f"Post-hoc comparisons on ({target_inst}, {target_metric}) require >= 2 algorithms with >= 2 runs each."
        )

    norm_method = method.strip().lower()
    if norm_method == "auto":
        norm_ok = all(r.passed for r in saes.statistics.shapiro(groups, alpha=alpha))
        if not norm_ok:
            norm_method = "dunn"
        else:
            homo_ok = saes.statistics.levene(groups, alpha=alpha).passed
            norm_method = "tukey_hsd" if homo_ok else "games_howell"

    if norm_method in {"tukey", "tukey_hsd"}:
        return saes.statistics.tukey_hsd(groups, alpha=alpha)
    if norm_method in {"games_howell", "games-howell"}:
        return saes.statistics.games_howell(groups, alpha=alpha)
    if norm_method == "dunnett":
        ctrl = control if (control and control in groups) else sorted(groups.keys())[0]
        return saes.statistics.dunnett(groups, control=ctrl, alpha=alpha)
    if norm_method in {"conover", "conover_iman", "conover-iman"}:
        return saes.statistics.conover_iman(groups, alpha=alpha, p_adjust=p_adjust)
    if norm_method == "dunn":
        return saes.statistics.dunn(groups, alpha=alpha, p_adjust=p_adjust)

    raise StatisticalAnalysisError(
        f"Unsupported post-hoc method '{method}'. "
        "Supported: auto, tukey_hsd, games_howell, dunnett, dunn, conover_iman."
    )


def _extract_pair_vectors(
    dataset: Dataset,
    metric: str | None,
    algo_a: str | None,
    algo_b: str | None,
    scope: str = "single-instance",
    instance: str | None = None,
    paired: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, str, bool]:
    """Extract aligned or independent vectors for two algorithms from a Dataset."""
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no evaluation metrics.")
    target_metric = metric if (metric and metric in dataset.metrics) else dataset.metrics[0]
    sub = dataset.df[dataset.df["metric"] == target_metric]
    if sub.empty:
        raise StatisticalAnalysisError(f"Metric '{target_metric}' not found in dataset.")

    maximize = bool(sub["maximize"].iloc[0])
    algos = sorted(sub["algorithm"].unique().tolist())
    if len(algos) < 2:
        raise StatisticalAnalysisError("At least 2 algorithms are required for pairwise inference.")

    resolved_a = algo_a if (algo_a and algo_a in algos) else algos[0]
    resolved_b = (
        algo_b
        if (algo_b and algo_b in algos and algo_b != resolved_a)
        else next(a for a in algos if a != resolved_a)
    )

    instances = sorted(sub["instance"].unique().tolist())
    use_all = scope.strip().lower() == "all-instances" and len(instances) >= 2

    if use_all:
        piv = sub.groupby(["instance", "algorithm"])["value"].mean().unstack()
        if resolved_a not in piv.columns or resolved_b not in piv.columns:
            raise StatisticalAnalysisError(f"Algorithms '{resolved_a}' and '{resolved_b}' lack paired instances.")
        pair_df = piv[[resolved_a, resolved_b]].dropna()
        return (
            pair_df[resolved_a].to_numpy(dtype=float),
            pair_df[resolved_b].to_numpy(dtype=float),
            resolved_a,
            resolved_b,
            maximize,
        )

    target_inst = instance if (instance and instance in instances) else instances[0]
    cell = sub[sub["instance"] == target_inst]
    if paired:
        piv_r = cell.pivot(index="run_id", columns="algorithm", values="value")
        if resolved_a in piv_r.columns and resolved_b in piv_r.columns:
            pair_r = piv_r[[resolved_a, resolved_b]].dropna()
            return (
                pair_r[resolved_a].to_numpy(dtype=float),
                pair_r[resolved_b].to_numpy(dtype=float),
                resolved_a,
                resolved_b,
                maximize,
            )

    va = cell[cell["algorithm"] == resolved_a]["value"].dropna().to_numpy(dtype=float)
    vb = cell[cell["algorithm"] == resolved_b]["value"].dropna().to_numpy(dtype=float)
    return va, vb, resolved_a, resolved_b, maximize


def run_tost_for_dataset(
    dataset: Dataset,
    metric: str | None = None,
    algo_a: str | None = None,
    algo_b: str | None = None,
    low_eqbound: float = 0.05,
    high_eqbound: float | None = None,
    scope: str = "single-instance",
    instance: str | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    equal_var: bool = False,
) -> TOSTResult:
    """Run TOST statistical equivalence test between two algorithms in a Dataset."""
    effective_paired = paired or (scope.strip().lower() == "all-instances")
    va, vb, name_a, name_b, _ = _extract_pair_vectors(
        dataset=dataset,
        metric=metric,
        algo_a=algo_a,
        algo_b=algo_b,
        scope=scope,
        instance=instance,
        paired=effective_paired,
    )
    return saes.tost(
        x1=va,
        x2=vb,
        low_eqbound=low_eqbound,
        high_eqbound=high_eqbound,
        alpha=alpha,
        paired=effective_paired,
        equal_var=equal_var,
        group_1=name_a,
        group_2=name_b,
    )


def run_bayesian_correlated_t_for_dataset(
    dataset: Dataset,
    metric: str | None = None,
    algo_a: str | None = None,
    algo_b: str | None = None,
    scope: str = "single-instance",
    instance: str | None = None,
    rope: float = 0.01,
    rho: float | None = None,
) -> BayesianCorrelatedTResult:
    """Run Bayesian Correlated t-test (Nadeau & Bengio 2003; Benavoli et al. 2017) on a Dataset."""
    va, vb, name_a, name_b, maximize = _extract_pair_vectors(
        dataset=dataset,
        metric=metric,
        algo_a=algo_a,
        algo_b=algo_b,
        scope=scope,
        instance=instance,
        paired=True,
    )
    return saes.statistics.bayesian_correlated_ttest(
        x_a=va,
        x_b=vb,
        algo_a=name_a,
        algo_b=name_b,
        rope=rope,
        rho=rho,
        maximize=maximize,
    )
