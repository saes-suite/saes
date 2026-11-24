"""Exact Bayesian statistical tests for algorithm comparison and multi-algorithm ranking.

References:
    - Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change:
      A tutorial for comparing multiple classifiers through Bayesian analysis. JMLR, 18(77), 1-36.
    - Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52, 239-281.
    - Calvo, B., Shir, O. M., Ceberio, J., Doerr, C., Wang, H., Bäck, T., & Lozano, J. A. (2018).
      Bayesian performance analysis for black-box optimization benchmarking. GECCO Companion, 1789-1797.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    BayesianCorrelatedTResult,
    BayesianPlackettLuceItem,
    BayesianPlackettLuceResult,
)


class BayesianCorrelatedTTest:
    """Bayesian Correlated t-test with ROPE (Nadeau & Bengio 2003; Benavoli et al. 2017, Sec. 3.1).

    Models paired performance differences z_i = x_{A,i} - x_{B,i} (or reversed when minimizing)
    from repeated k-fold cross-validation or correlated runs with correlation factor rho
    (typically rho = n_test / n_train, e.g., 1/9 for 10-fold CV; default 1/n for standard paired t).
    """

    def run(
        self,
        x_a: np.ndarray,
        x_b: np.ndarray,
        algo_a: str = "Algorithm A",
        algo_b: str = "Algorithm B",
        rope: float = 0.01,
        rho: float | None = None,
        maximize: bool = True,
    ) -> BayesianCorrelatedTResult:
        """Compute exact analytical posterior probabilities P(A > B), P(ROPE), P(B > A)."""
        a = np.asarray(x_a, dtype=float)
        b = np.asarray(x_b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]

        if len(a) != len(b):
            raise StatisticalAnalysisError(
                f"BayesianCorrelatedTTest requires paired samples of equal length; got {len(a)} and {len(b)}."
            )
        n = len(a)
        if n < 2:
            raise StatisticalAnalysisError("BayesianCorrelatedTTest requires at least 2 paired observations.")
        if rope < 0:
            raise StatisticalAnalysisError(f"ROPE half-width must be >= 0, got {rope}.")

        diffs = (a - b) if maximize else (b - a)
        mean_d = float(np.mean(diffs))
        var_d = float(np.var(diffs, ddof=1))

        # Standard Nadeau & Bengio (2003) / Benavoli et al. (2017) posterior variance:
        # sigma^2_post = (1/n + rho / (1 - rho)) * s^2
        # When rho is None, uses rho = 0.0 -> standard uncorrelated Bayesian paired t-test (1/n * s^2).
        rho_eff = 0.0 if rho is None else float(rho)
        if not (0.0 <= rho_eff < 1.0):
            raise StatisticalAnalysisError(f"Correlation parameter rho must lie in [0, 1), got {rho_eff}.")

        df = float(n - 1)
        scale_sq = (1.0 / float(n) + rho_eff / (1.0 - rho_eff)) * var_d
        scale = math.sqrt(max(0.0, scale_sq))

        if scale <= 1e-15:
            if abs(mean_d) <= rope:
                p_left, p_rope, p_right = 0.0, 1.0, 0.0
            elif mean_d > rope:
                p_left, p_rope, p_right = 1.0, 0.0, 0.0
            else:
                p_left, p_rope, p_right = 0.0, 0.0, 1.0
        else:
            cdf_low = float(stats.t.cdf(-rope, df=df, loc=mean_d, scale=scale))
            cdf_high = float(stats.t.cdf(rope, df=df, loc=mean_d, scale=scale))
            p_right = max(0.0, min(1.0, cdf_low))
            p_rope = max(0.0, min(1.0, cdf_high - cdf_low))
            p_left = max(0.0, min(1.0, 1.0 - cdf_high))

        # Precompute analytical Student-t posterior PDF curve for publication / UI rendering
        if scale > 1e-15:
            span = max(4.5 * scale, 1.8 * rope, abs(mean_d) + 3.0 * scale, 1e-4)
            x_min = min(-rope * 1.5, mean_d - span)
            x_max = max(rope * 1.5, mean_d + span)
            grid_x = np.linspace(x_min, x_max, 160)
            grid_y = stats.t.pdf(grid_x, df=df, loc=mean_d, scale=scale)
            curve_x = [round(float(x), 6) for x in grid_x.tolist()]
            curve_y = [round(float(y), 6) for y in grid_y.tolist()]
        else:
            curve_x = [round(mean_d, 6)]
            curve_y = [1.0]

        interp = (
            f"Bayesian Correlated t-test ({algo_a} vs {algo_b}, n={n}, rho={rho_eff:.3f}, "
            f"ROPE=[{-rope:.4g}, {rope:.4g}]): P({algo_a} better) = {p_left:.4f}, "
            f"P(ROPE) = {p_rope:.4f}, P({algo_b} better) = {p_right:.4f}."
        )
        return BayesianCorrelatedTResult(
            algo_a=algo_a,
            algo_b=algo_b,
            n_observations=n,
            rho=rho_eff,
            rope_low=-float(rope),
            rope_high=float(rope),
            mean_diff=mean_d,
            posterior_scale=scale,
            degrees_of_freedom=df,
            p_left=p_left,
            p_rope=p_rope,
            p_right=p_right,
            curve_x=curve_x,
            curve_y=curve_y,
            interpretation=interp,
        )


def _fit_plackett_luce_mm(
    orderings: np.ndarray,
    k: int,
    prior_alpha: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute MAP Plackett-Luce worth parameters w (sum w = 1) via Hunter (2004) MM algorithm.

    Each row of `orderings` is a permutation of {0, ..., k-1} from best (position 0) to worst (position k-1).
    Incorporates a symmetric Dirichlet(prior_alpha) prior (equivalent to pseudo-wins of (prior_alpha - 1)
    and regularized denominator) to ensure strictly positive finite estimates even under complete dominance.
    """
    n_blocks = orderings.shape[0]
    # Count wins (appearing in top k-1 positions, i.e., beating at least one remaining competitor)
    wins = np.full(k, max(0.5, prior_alpha), dtype=float)
    for b in range(n_blocks):
        for pos in range(k - 1):
            wins[orderings[b, pos]] += 1.0

    w = np.full(k, 1.0 / float(k), dtype=float)
    reg = max(0.5, prior_alpha)

    for _ in range(max_iter):
        denom = np.full(k, reg * float(k), dtype=float)
        for b in range(n_blocks):
            row = orderings[b]
            # Suffix sums of w for positions pos..k-1
            suffix_sum = float(np.sum(w))
            for pos in range(k - 1):
                inv_s = 1.0 / max(1e-15, suffix_sum)
                for rem in range(pos, k):
                    denom[row[rem]] += inv_s
                suffix_sum -= w[row[pos]]
        w_new = wins / np.maximum(1e-15, denom)
        w_new = w_new / float(np.sum(w_new))
        if float(np.max(np.abs(w_new - w))) < tol:
            w = w_new
            break
        w = w_new

    return w


class BayesianPlackettLuceTest:
    """Bayesian Plackett-Luce multi-algorithm ranking model (Calvo et al. 2018).

    Estimates posterior distributions over Plackett-Luce skill/worth parameters w = (w_1, ..., w_k)
    via Bayesian bootstrap over observed instance rankings combined with Hunter (2004) MM estimation,
    yielding coherent posterior winning probabilities P(algorithm j is best) and expected ranks.
    """

    def run(
        self,
        matrix: np.ndarray,
        algorithms: list[str],
        metric: str = "value",
        maximize: bool = True,
        num_samples: int = 400,
        prior_alpha: float = 1.0,
        seed: int = 42,
    ) -> BayesianPlackettLuceResult:
        """Fit Bayesian Plackett-Luce model to an (n_blocks x k) performance matrix."""
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim != 2:
            raise StatisticalAnalysisError("Performance matrix must be 2D (n_blocks x k_algorithms).")
        n_blocks, k = mat.shape
        if k < 2 or len(algorithms) != k:
            raise StatisticalAnalysisError(
                f"Expected matrix with k >= 2 matching algorithms list; got shape {mat.shape}, {len(algorithms)} names."
            )
        if n_blocks < 1:
            raise StatisticalAnalysisError("At least 1 benchmark block is required for Plackett-Luce ranking.")

        # Convert each block into an ordering from best (0) to worst (k-1)
        oriented = -mat if maximize else mat
        orderings = np.argsort(oriented, axis=1, kind="mergesort")

        # Point MAP worth estimate on full data
        map_w = _fit_plackett_luce_mm(orderings, k=k, prior_alpha=prior_alpha)

        # Posterior samples via Dirichlet-weighted Bayesian bootstrap over rankings
        rng = np.random.default_rng(seed + n_blocks * 17 + k * 7)
        w_samples = np.zeros((num_samples, k), dtype=float)
        rank_samples = np.zeros((num_samples, k), dtype=float)

        # Fast weighted MM updates using Dirichlet(1,...,1) weights over blocks
        dir_weights = rng.dirichlet(np.ones(n_blocks, dtype=float), size=num_samples) * float(n_blocks)
        reg = max(0.5, prior_alpha)

        for s_idx in range(num_samples):
            bw = dir_weights[s_idx]
            wins = np.full(k, reg, dtype=float)
            for b in range(n_blocks):
                for pos in range(k - 1):
                    wins[orderings[b, pos]] += bw[b]

            w = map_w.copy()
            for _ in range(40):
                denom = np.full(k, reg * float(k), dtype=float)
                for b in range(n_blocks):
                    row = orderings[b]
                    suffix_sum = float(np.sum(w))
                    weight_b = bw[b]
                    for pos in range(k - 1):
                        inc = weight_b / max(1e-15, suffix_sum)
                        for rem in range(pos, k):
                            denom[row[rem]] += inc
                        suffix_sum -= w[row[pos]]
                w_new = wins / np.maximum(1e-15, denom)
                w_new = w_new / float(np.sum(w_new))
                if float(np.max(np.abs(w_new - w))) < 1e-5:
                    w = w_new
                    break
                w = w_new

            w_samples[s_idx] = w
            # Fractional rank of worth parameters (higher worth -> rank 1)
            rank_samples[s_idx] = stats.rankdata(-w, method="average")

        # Under Plackett-Luce, P(algorithm j is ranked 1st | w) = w_j; posterior E[w_j | data] = mean(w_samples)
        posterior_prob_best = np.mean(w_samples, axis=0)
        posterior_prob_best = posterior_prob_best / float(np.sum(posterior_prob_best))
        expected_ranks = np.mean(rank_samples, axis=0)

        items: list[BayesianPlackettLuceItem] = []
        for idx, alg in enumerate(algorithms):
            lo, hi = np.percentile(rank_samples[:, idx], [2.5, 97.5])
            items.append(
                BayesianPlackettLuceItem(
                    algorithm=alg,
                    worth=float(map_w[idx]),
                    prob_best=float(posterior_prob_best[idx]),
                    expected_rank=float(expected_ranks[idx]),
                    rank_ci_lower=float(lo),
                    rank_ci_upper=float(hi),
                )
            )
        items.sort(key=lambda it: it.expected_rank)
        top = items[0]
        interp = (
            f"Bayesian Plackett-Luce ranking across {n_blocks} block(s) and {k} algorithms on '{metric}': "
            f"top algorithm is '{top.algorithm}' with posterior P(best) = {top.prob_best:.3f} "
            f"and expected rank = {top.expected_rank:.2f} (95% CrI [{top.rank_ci_lower:.2f}, {top.rank_ci_upper:.2f}])."
        )
        return BayesianPlackettLuceResult(
            metric=metric,
            k=k,
            n_blocks=n_blocks,
            num_samples=num_samples,
            maximize=maximize,
            items=items,
            interpretation=interp,
        )
