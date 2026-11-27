"""Exact 2D Empirical Attainment Functions (EAF), attainment surfaces, and KS-EAF permutation tests.

Implements the statistical methodology of:
- Grunert da Fonseca, V., Fonseca, C. M., & Hall, A. O. (2001). Inferential performance assessment
  of stochastic optimisers and the attainment function. EMO 2001.
- Fonseca, C. M., Grunert da Fonseca, V., & Paquete, L. (2005). Exploring the performance of
  stochastic multiobjective optimisers with the second-order attainment function. EMO 2005.
- López-Ibáñez, M., Paquete, L., & Stützle, T. (2010). Exploratory analysis of stochastic local
  search algorithms in biobjective optimization. Experimental Methods for the Analysis of
  Optimization Algorithms, Springer.
"""

from __future__ import annotations

import math

import numpy as np

from saes.plugins.pareto.dominance import filter_nondominated_array, to_minimization_space
from saes.plugins.pareto.models import (
    AlgorithmAttainment,
    AttainmentResult,
    AttainmentSurface,
    EAFDifferenceResult,
    FrontCollection,
)


def _prepare_2d_nondominated_runs(
    run_arrays: list[np.ndarray],
    minimize: tuple[bool, bool],
) -> list[np.ndarray]:
    """Filter each 2D run to non-dominated points in minimization space, sorted by x ascending."""
    cleaned: list[np.ndarray] = []
    for arr in run_arrays:
        if arr.size == 0:
            cleaned.append(np.empty((0, 2), dtype=np.float64))
            continue
        nd_orig, _ = filter_nondominated_array(arr, minimize=minimize)
        nd_min = to_minimization_space(nd_orig, minimize=minimize)
        order = np.lexsort((nd_min[:, 1], nd_min[:, 0]))
        cleaned.append(nd_min[order])
    return cleaned


def _evaluate_run_step_y(nd_min_sorted: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
    """For a non-dominated 2D front sorted ascending by x, return y_r(x) = min{p_y : p_x <= x}."""
    if nd_min_sorted.shape[0] == 0:
        return np.full(x_coords.shape[0], np.inf, dtype=np.float64)
    xs = nd_min_sorted[:, 0]
    ys = nd_min_sorted[:, 1]
    idx = np.searchsorted(xs, x_coords, side="right") - 1
    out = np.full(x_coords.shape[0], np.inf, dtype=np.float64)
    valid = idx >= 0
    out[valid] = ys[idx[valid]]
    return out


def compute_attainment_surfaces_2d(
    run_arrays: list[np.ndarray],
    levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    minimize: tuple[bool, bool] = (True, True),
) -> list[AttainmentSurface]:
    """Compute exact k-th empirical attainment surfaces for a list of 2D approximation runs.

    Level convention (matching López-Ibáñez et al., 2010 / R `eaf`):
    - ``0.0`` (best attainment surface): attained by at least 1 run (k = 1).
    - ``0.25`` (25th percentile / upper quartile): k = ceil(0.25 * n).
    - ``0.50`` (median attainment surface): k = ceil(0.50 * n).
    - ``0.75`` (75th percentile / lower quartile): k = ceil(0.75 * n).
    - ``1.0`` (worst attainment surface): attained by all n runs (k = n).
    """
    n_runs = len(run_arrays)
    if n_runs == 0:
        return []

    nd_runs = _prepare_2d_nondominated_runs(run_arrays, minimize=minimize)
    all_x = [r[:, 0] for r in nd_runs if r.shape[0] > 0]
    if not all_x:
        return [AttainmentSurface(level=lvl, rank_k=1, points=()) for lvl in levels]

    x_unique = np.unique(np.concatenate(all_x))
    y_matrix = np.vstack([_evaluate_run_step_y(r, x_unique) for r in nd_runs])  # (n_runs, |X|)
    y_sorted = np.sort(y_matrix, axis=0)  # column j sorted ascending: best (0) .. worst (n-1)

    sx = 1.0 if minimize[0] else -1.0
    sy = 1.0 if minimize[1] else -1.0

    surfaces: list[AttainmentSurface] = []
    for lvl in levels:
        lvl_clamped = max(0.0, min(1.0, float(lvl)))
        rank_k = 1 if lvl_clamped == 0.0 else max(1, min(n_runs, int(math.ceil(lvl_clamped * n_runs))))
        y_k = y_sorted[rank_k - 1]
        finite_mask = np.isfinite(y_k)
        xs_f = x_unique[finite_mask]
        ys_f = y_k[finite_mask]

        # Extract minimal non-dominated corner points of the step surface
        pts_min: list[tuple[float, float]] = []
        prev_y = np.inf
        for xv, yv in zip(xs_f, ys_f, strict=True):
            if float(yv) < prev_y:
                pts_min.append((float(xv) * sx, float(yv) * sy))
                prev_y = float(yv)

        surfaces.append(
            AttainmentSurface(
                level=lvl_clamped,
                rank_k=rank_k,
                points=tuple(pts_min),
            )
        )
    return surfaces


def compute_attainment(
    collection: FrontCollection,
    instance: str,
    algorithms: list[str] | None = None,
    levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> AttainmentResult:
    """Compute exact 2D empirical attainment surfaces for each selected algorithm on an instance."""
    inst_fronts = collection.get_fronts(instance, algorithms=algorithms)
    if not inst_fronts:
        raise ValueError(f"No fronts found for instance '{instance}'.")

    sample = inst_fronts[0]
    if sample.n_objectives != 2:
        raise ValueError(
            f"Exact 2D Empirical Attainment Surfaces require 2 objectives, got {sample.n_objectives} "
            f"for instance '{instance}'."
        )

    minimize_2d = (sample.minimize[0], sample.minimize[1])
    obj_names_2d = (sample.objective_names[0], sample.objective_names[1])
    algs = sorted({f.algorithm for f in inst_fronts})

    alg_results: list[AlgorithmAttainment] = []
    for alg in algs:
        runs = sorted([f for f in inst_fronts if f.algorithm == alg], key=lambda f: f.run_id)
        arrays = [f.to_array() for f in runs]
        surfaces = compute_attainment_surfaces_2d(arrays, levels=levels, minimize=minimize_2d)
        alg_results.append(AlgorithmAttainment(algorithm=alg, n_runs=len(runs), surfaces=surfaces))

    return AttainmentResult(
        instance=instance,
        objective_names=obj_names_2d,
        minimize=minimize_2d,
        levels=tuple(float(v) for v in levels),
        algorithms=alg_results,
    )


def compute_eaf_difference(
    collection: FrontCollection,
    instance: str,
    algo_a: str,
    algo_b: str,
    *,
    grid_resolution: int = 120,
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> EAFDifferenceResult:
    """Compute pairwise 2D EAF difference ΔEAF(z) = α_A(z) - α_B(z) and KS-EAF permutation test.

    Args:
        collection: Input FrontCollection.
        instance: Target 2-objective instance name.
        algo_a: First algorithm name (positive ΔEAF favors algo_a).
        algo_b: Second algorithm name (negative ΔEAF favors algo_b).
        grid_resolution: Maximum coordinate count per axis for evaluation grid (uses exact coordinates
            when unique coordinates <= grid_resolution).
        n_permutations: Number of Monte Carlo permutations for KS-EAF hypothesis test (0 skips test).
        alpha: Significance threshold (default 0.05).
        seed: Deterministic RNG seed for reproducible permutation p-values.
    """
    fronts_a = sorted(collection.get_fronts(instance, algorithms=[algo_a]), key=lambda f: f.run_id)
    fronts_b = sorted(collection.get_fronts(instance, algorithms=[algo_b]), key=lambda f: f.run_id)

    if not fronts_a:
        raise ValueError(f"No fronts found for algorithm '{algo_a}' on instance '{instance}'.")
    if not fronts_b:
        raise ValueError(f"No fronts found for algorithm '{algo_b}' on instance '{instance}'.")

    sample = fronts_a[0]
    if sample.n_objectives != 2:
        raise ValueError(
            f"2D EAF Difference requires 2 objectives, got {sample.n_objectives} for instance '{instance}'."
        )

    minimize_2d = (sample.minimize[0], sample.minimize[1])
    obj_names_2d = (sample.objective_names[0], sample.objective_names[1])

    nd_a = _prepare_2d_nondominated_runs([f.to_array() for f in fronts_a], minimize=minimize_2d)
    nd_b = _prepare_2d_nondominated_runs([f.to_array() for f in fronts_b], minimize=minimize_2d)
    pooled = nd_a + nd_b
    n_a = len(nd_a)
    n_b = len(nd_b)
    n_total = n_a + n_b

    non_empty = [r for r in pooled if r.shape[0] > 0]
    if not non_empty:
        raise ValueError(f"All fronts for '{algo_a}' and '{algo_b}' on '{instance}' are empty.")

    stacked = np.vstack(non_empty)
    x_unique = np.unique(stacked[:, 0])
    y_unique = np.unique(stacked[:, 1])

    # Include slight upper padding so boundary attainment is visible
    x_span = float(x_unique[-1] - x_unique[0]) if x_unique.size > 1 else 1.0
    y_span = float(y_unique[-1] - y_unique[0]) if y_unique.size > 1 else 1.0
    x_pad = 0.03 * (x_span if x_span > 1e-12 else 1.0)
    y_pad = 0.03 * (y_span if y_span > 1e-12 else 1.0)

    if x_unique.size <= grid_resolution:
        x_grid_min = np.concatenate([x_unique, [x_unique[-1] + x_pad]])
    else:
        x_grid_min = np.linspace(float(x_unique[0]), float(x_unique[-1]) + x_pad, grid_resolution)

    if y_unique.size <= grid_resolution:
        y_grid_min = np.concatenate([y_unique, [y_unique[-1] + y_pad]])
    else:
        y_grid_min = np.linspace(float(y_unique[0]), float(y_unique[-1]) + y_pad, grid_resolution)

    # Evaluate exact binary attainment indicator I[r, j, i] = 1{y_r(x_i) <= y_j}
    y_steps = np.vstack([_evaluate_run_step_y(r, x_grid_min) for r in pooled])  # (n_total, nx)
    # Broadcast to (n_total, ny, nx)
    attained = (y_steps[:, None, :] <= y_grid_min[None, :, None]).astype(np.float32)

    eaf_a = np.mean(attained[:n_a], axis=0)
    eaf_b = np.mean(attained[n_a:], axis=0)
    diff = (eaf_a - eaf_b).astype(np.float64)  # (ny, nx)

    max_adv_a = float(max(0.0, np.max(diff)))
    max_adv_b = float(max(0.0, np.max(-diff)))
    ks_obs = float(max(max_adv_a, max_adv_b))

    # Area-weighted dominance fractions using exact cell widths dx * dy
    if x_grid_min.size > 1 and y_grid_min.size > 1:
        dx = np.diff(x_grid_min)
        dy = np.diff(y_grid_min)
        cell_areas = np.outer(dy, dx)
        total_area = float(np.sum(cell_areas))
        diff_cells = diff[:-1, :-1]
        area_a = float(np.sum(cell_areas[diff_cells > 1e-9])) / total_area if total_area > 0 else 0.0
        area_b = float(np.sum(cell_areas[diff_cells < -1e-9])) / total_area if total_area > 0 else 0.0
    else:
        area_a = float(np.mean(diff > 1e-9))
        area_b = float(np.mean(diff < -1e-9))

    # Deterministic Monte Carlo permutation test on KS-EAF = sup_z |α_A(z) - α_B(z)|
    if n_permutations > 0 and n_a >= 1 and n_b >= 1:
        rng = np.random.default_rng(seed)
        flat = attained.reshape(n_total, -1)  # (n_total, ny * nx)
        total_sum = np.sum(flat, axis=0)
        exceed_count = 0
        inv_a = 1.0 / float(n_a)
        inv_b = 1.0 / float(n_b)
        tol = 1e-12
        for _ in range(n_permutations):
            perm_a = rng.choice(n_total, size=n_a, replace=False)
            sum_a = np.sum(flat[perm_a], axis=0)
            sum_b = total_sum - sum_a
            perm_ks = float(np.max(np.abs(sum_a * inv_a - sum_b * inv_b)))
            if perm_ks >= ks_obs - tol:
                exceed_count += 1
        p_value = float((exceed_count + 1) / (n_permutations + 1))
    else:
        p_value = 1.0

    is_sig = p_value < alpha

    # Map grids back to original objective orientation (maintaining ascending display order)
    sx = 1.0 if minimize_2d[0] else -1.0
    sy = 1.0 if minimize_2d[1] else -1.0
    x_orig = x_grid_min * sx
    y_orig = y_grid_min * sy
    diff_out = diff
    if sx < 0:
        x_orig = x_orig[::-1]
        diff_out = diff_out[:, ::-1]
    if sy < 0:
        y_orig = y_orig[::-1]
        diff_out = diff_out[::-1, :]

    if is_sig:
        leader = algo_a if max_adv_a >= max_adv_b else algo_b
        interp = (
            f"Statistically significant difference between {algo_a} and {algo_b} attainment surfaces "
            f"(KS-EAF = {ks_obs:.3f}, p = {p_value:.4g} < {alpha}). "
            f"{leader} achieves the largest local attainment advantage (+{max(max_adv_a, max_adv_b) * 100:.1f}%)."
        )
    else:
        interp = (
            f"No statistically significant difference between {algo_a} and {algo_b} empirical attainment "
            f"surfaces (KS-EAF = {ks_obs:.3f}, p = {p_value:.4g} >= {alpha})."
        )

    return EAFDifferenceResult(
        instance=instance,
        algo_a=algo_a,
        algo_b=algo_b,
        n_runs_a=n_a,
        n_runs_b=n_b,
        objective_names=obj_names_2d,
        minimize=minimize_2d,
        x_grid=[float(v) for v in x_orig],
        y_grid=[float(v) for v in y_orig],
        diff_matrix=[[float(v) for v in row] for row in diff_out],
        max_advantage_a=max_adv_a,
        max_advantage_b=max_adv_b,
        ks_statistic=ks_obs,
        p_value=p_value,
        n_permutations=n_permutations,
        alpha=alpha,
        is_significant=is_sig,
        area_fraction_a_better=area_a,
        area_fraction_b_better=area_b,
        interpretation=interp,
    )
