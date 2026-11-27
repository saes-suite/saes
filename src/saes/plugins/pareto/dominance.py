"""Vectorized Pareto dominance filtering, set coverage C(A, B), and representative run selection."""

from __future__ import annotations

import numpy as np

from saes.plugins.pareto.models import (
    AlgorithmRepresentativeRuns,
    CoverageAndRepresentativeResult,
    CoverageCell,
    FrontCollection,
    ParetoFront,
)


def to_minimization_space(points: np.ndarray, minimize: tuple[bool, ...] | None = None) -> np.ndarray:
    """Transform objective vectors into pure minimization space by negating maximized objectives."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D objective array, got shape {arr.shape}.")
    if minimize is None:
        return arr.copy()
    signs = np.array([1.0 if m else -1.0 for m in minimize], dtype=np.float64)
    return arr * signs


def filter_nondominated_array(
    points: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique Pareto non-dominated objective vectors and their original indices.

    Uses an exact O(N log N) sweep for 2 objectives and chunked exact pairwise checks for m >= 3.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return arr.copy(), np.empty(0, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with >= 2 objectives, got shape {arr.shape}.")

    work = to_minimization_space(arr, minimize)

    # Deduplicate exact identical rows while keeping first occurrence index
    _, unique_idx = np.unique(work, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    u_work = work[unique_idx]
    n_pts, m = u_work.shape

    if m == 2:
        order = np.lexsort((u_work[:, 1], u_work[:, 0]))
        sorted_pts = u_work[order]
        keep_mask = np.zeros(n_pts, dtype=bool)
        best_y = np.inf
        for i in range(n_pts):
            y_val = float(sorted_pts[i, 1])
            if y_val < best_y:
                keep_mask[i] = True
                best_y = y_val
        kept_indices = unique_idx[order[keep_mask]]
        return arr[kept_indices], kept_indices

    # Exact m >= 3 non-dominated filtering (chunked to keep memory bounded)
    is_nd = np.ones(n_pts, dtype=bool)
    chunk_size = 512
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        chunk = u_work[start:end]  # (c, m)
        # Compare all currently active points against chunk
        active_idx = np.flatnonzero(is_nd)
        if active_idx.size == 0:
            break
        active = u_work[active_idx]  # (a, m)
        # active[j] dominates chunk[i] iff all(active[j] <= chunk[i]) and any(active[j] < chunk[i])
        le = np.all(active[:, None, :] <= chunk[None, :, :], axis=2)
        lt = np.any(active[:, None, :] < chunk[None, :, :], axis=2)
        dominated_in_chunk = np.any(le & lt, axis=0)
        is_nd[start:end] = ~dominated_in_chunk

    kept_indices = unique_idx[is_nd]
    return arr[kept_indices], kept_indices


def filter_nondominated_front(front: ParetoFront) -> ParetoFront:
    """Return a new ParetoFront containing only non-dominated unique points."""
    if front.n_points == 0:
        return front
    nd_arr, _ = filter_nondominated_array(front.to_array(), minimize=front.minimize)
    pts = tuple(tuple(float(v) for v in row) for row in nd_arr)
    return front.model_copy(update={"points": pts})


def compute_set_coverage(
    front_a: np.ndarray,
    front_b: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> float:
    """Compute Zitzler & Thiele (1999) set coverage C(A, B) in [0, 1].

    Returns the fraction of points in B weakly dominated by at least one point in A.
    """
    a_arr = to_minimization_space(front_a, minimize)
    b_arr = to_minimization_space(front_b, minimize)
    if b_arr.shape[0] == 0:
        return 0.0
    if a_arr.shape[0] == 0:
        return 0.0

    # Weak dominance: exists a in A such that a_j <= b_j for all j
    covered = np.zeros(b_arr.shape[0], dtype=bool)
    chunk_size = 512
    for start in range(0, b_arr.shape[0], chunk_size):
        end = min(start + chunk_size, b_arr.shape[0])
        b_chunk = b_arr[start:end]
        le = np.all(a_arr[:, None, :] <= b_chunk[None, :, :], axis=2)
        covered[start:end] = np.any(le, axis=0)
    return float(np.mean(covered))


def compute_accumulated_front(fronts: list[ParetoFront]) -> ParetoFront:
    """Merge multiple runs and return their non-dominated union front."""
    if not fronts:
        raise ValueError("Cannot compute accumulated front from an empty list.")
    first = fronts[0]
    arrays = [f.to_array() for f in fronts if f.n_points > 0]
    if not arrays:
        return first.model_copy(update={"points": ()})
    merged = np.vstack(arrays)
    nd_arr, _ = filter_nondominated_array(merged, minimize=first.minimize)
    pts = tuple(tuple(float(v) for v in row) for row in nd_arr)
    return ParetoFront(
        algorithm=first.algorithm,
        instance=first.instance,
        run_id=0,
        points=pts,
        objective_names=first.objective_names,
        minimize=first.minimize,
        is_reference=False,
    )


def compute_ideal_nadir(
    arrays: list[np.ndarray],
    minimize: tuple[bool, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ideal (best) and nadir (worst) vectors in minimization space across arrays."""
    valid = [to_minimization_space(a, minimize) for a in arrays if a.size > 0]
    if not valid:
        raise ValueError("Cannot compute ideal/nadir from empty objective arrays.")
    stacked = np.vstack(valid)
    return np.min(stacked, axis=0), np.max(stacked, axis=0)


def compute_coverage_and_representatives(
    collection: FrontCollection,
    instance: str,
    algorithms: list[str] | None = None,
    filter_dominated: bool = True,
) -> CoverageAndRepresentativeResult:
    """Compute pairwise set coverage C(A, B) matrix and representative run IDs for an instance."""
    from saes.plugins.pareto.indicators import compute_hypervolume, compute_igd_plus

    inst_fronts = collection.get_fronts(instance, algorithms=algorithms)
    if not inst_fronts:
        raise ValueError(f"No fronts found for instance '{instance}'.")

    if filter_dominated:
        inst_fronts = [filter_nondominated_front(f) for f in inst_fronts]

    algs = sorted({f.algorithm for f in inst_fronts})
    minimize = inst_fronts[0].minimize
    n_obj = inst_fronts[0].n_objectives

    by_alg: dict[str, list[ParetoFront]] = {
        a: sorted([f for f in inst_fronts if f.algorithm == a], key=lambda f: f.run_id)
        for a in algs
    }

    # Reference front (true reference or pseudo-reference = non-dominated union of all runs)
    ref_front = collection.get_reference(instance)
    if ref_front is not None and ref_front.n_points > 0:
        ref_arr = ref_front.to_array()
    else:
        pseudo = compute_accumulated_front(inst_fronts)
        ref_arr = pseudo.to_array()

    # Common HV reference point (nadir + 10% span in minimization space, then mapped back)
    all_arrays = [f.to_array() for f in inst_fronts if f.n_points > 0] + [ref_arr]
    ideal_min, nadir_min = compute_ideal_nadir(all_arrays, minimize=minimize)
    span = np.where((nadir_min - ideal_min) > 1e-12, nadir_min - ideal_min, 1.0)
    hv_ref_min = nadir_min + 0.1 * span
    signs = np.array([1.0 if m else -1.0 for m in minimize], dtype=np.float64)
    hv_ref_orig = hv_ref_min * signs

    # 1. Pairwise C(A, B) coverage matrix across runs
    coverage_cells: list[CoverageCell] = []
    for a in algs:
        for b in algs:
            if a == b:
                coverage_cells.append(CoverageCell(algo_a=a, algo_b=b, mean=1.0, median=1.0, std=0.0))
                continue
            vals: list[float] = []
            for fa in by_alg[a]:
                arr_a = fa.to_array()
                for fb in by_alg[b]:
                    vals.append(compute_set_coverage(arr_a, fb.to_array(), minimize=minimize))
            arr_vals = np.asarray(vals, dtype=np.float64)
            coverage_cells.append(
                CoverageCell(
                    algo_a=a,
                    algo_b=b,
                    mean=float(np.mean(arr_vals)),
                    median=float(np.median(arr_vals)),
                    std=float(np.std(arr_vals, ddof=1)) if arr_vals.size > 1 else 0.0,
                )
            )

    # 2. Representative runs (best, median, worst by IGD+ lower-is-better)
    representatives: list[AlgorithmRepresentativeRuns] = []
    for a in algs:
        runs = by_alg[a]
        scored: list[tuple[int, float]] = []
        hv_scores: list[float] = []
        for f in runs:
            arr = f.to_array()
            igd_p = compute_igd_plus(arr, ref_arr, minimize=minimize)
            scored.append((f.run_id, igd_p))
            if n_obj in (2, 3):
                hv_scores.append(compute_hypervolume(arr, hv_ref_orig, minimize=minimize))

        scored.sort(key=lambda item: (item[1], item[0]))
        best_id, best_val = scored[0]
        med_id, med_val = scored[len(scored) // 2]
        worst_id, worst_val = scored[-1]
        acc = compute_accumulated_front(runs)

        representatives.append(
            AlgorithmRepresentativeRuns(
                algorithm=a,
                n_runs=len(runs),
                metric_used="IGD+",
                best_run_id=best_id,
                best_score=best_val,
                median_run_id=med_id,
                median_score=med_val,
                worst_run_id=worst_id,
                worst_score=worst_val,
                accumulated_points_count=acc.n_points,
                mean_igd_plus=float(np.mean([s for _, s in scored])),
                mean_hv=float(np.mean(hv_scores)) if hv_scores else None,
            )
        )

    return CoverageAndRepresentativeResult(
        instance=instance,
        algorithms=algs,
        coverage_matrix=coverage_cells,
        representative_runs=representatives,
    )
