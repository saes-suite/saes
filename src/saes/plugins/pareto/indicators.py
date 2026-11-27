"""Exact multi-objective quality indicators and direct conversion to canonical SAES Datasets.

Supported indicators:
- ``HV``: Hypervolume (exact O(N log N) sweep for 2D, exact z-slice sweep for 3D, deterministic MC for m >= 4).
- ``IGD``: Inverted Generational Distance.
- ``IGD+``: Pareto-compliant Inverted Generational Distance Plus (Ishibuchi et al., 2015).
- ``GD``: Generational Distance.
- ``EPS``: Unary additive epsilon indicator (Zitzler et al., 2003).
- ``Spacing``: Schott's uniformity/spacing metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from saes.domain.models import Dataset
from saes.plugins.pareto.dominance import (
    compute_accumulated_front,
    compute_ideal_nadir,
    filter_nondominated_array,
    to_minimization_space,
)
from saes.plugins.pareto.models import FrontCollection

INDICATOR_DIRECTIONS: dict[str, bool] = {
    "HV": True,
    "IGD": False,
    "IGD+": False,
    "GD": False,
    "EPS": False,
    "SPACING": False,
}

INDICATOR_CANONICAL_NAMES: dict[str, str] = {
    "HV": "HV",
    "IGD": "IGD",
    "IGD+": "IGD+",
    "IGD_PLUS": "IGD+",
    "GD": "GD",
    "EPS": "EPS",
    "EPSILON": "EPS",
    "SPACING": "Spacing",
}


def compute_igd(
    front: np.ndarray,
    reference: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> float:
    """Compute Inverted Generational Distance (lower is better)."""
    a = to_minimization_space(front, minimize)
    r = to_minimization_space(reference, minimize)
    if a.shape[0] == 0 or r.shape[0] == 0:
        return float("inf")
    dists = np.linalg.norm(r[:, None, :] - a[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def compute_igd_plus(
    front: np.ndarray,
    reference: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> float:
    """Compute Pareto-compliant IGD+ (Ishibuchi et al., 2015; lower is better)."""
    a = to_minimization_space(front, minimize)
    r = to_minimization_space(reference, minimize)
    if a.shape[0] == 0 or r.shape[0] == 0:
        return float("inf")
    diff = np.maximum(a[None, :, :] - r[:, None, :], 0.0)
    dists = np.linalg.norm(diff, axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def compute_gd(
    front: np.ndarray,
    reference: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> float:
    """Compute Generational Distance (lower is better)."""
    a = to_minimization_space(front, minimize)
    r = to_minimization_space(reference, minimize)
    if a.shape[0] == 0 or r.shape[0] == 0:
        return float("inf")
    dists = np.linalg.norm(a[:, None, :] - r[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def compute_additive_epsilon(
    front: np.ndarray,
    reference: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> float:
    """Compute unary additive epsilon indicator I_{eps+}(A, R) (lower is better)."""
    a = to_minimization_space(front, minimize)
    r = to_minimization_space(reference, minimize)
    if a.shape[0] == 0 or r.shape[0] == 0:
        return float("inf")
    # For each reference point r, min_{a in A} max_j (a_j - r_j), then max over r in R
    max_diff = np.max(a[None, :, :] - r[:, None, :], axis=2)
    return float(np.max(np.min(max_diff, axis=1)))


def compute_spacing(
    front: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
) -> float:
    """Compute Schott's Spacing uniformity metric using L1 nearest-neighbor distances (lower is better)."""
    a = to_minimization_space(front, minimize)
    if a.shape[0] < 2:
        return 0.0
    l1 = np.sum(np.abs(a[:, None, :] - a[None, :, :]), axis=2)
    np.fill_diagonal(l1, np.inf)
    d_min = np.min(l1, axis=1)
    return float(np.sqrt(np.mean((d_min - np.mean(d_min)) ** 2)))


def _hv_2d_exact_min(points_min: np.ndarray, ref_min: np.ndarray) -> float:
    """Exact O(N log N) 2D Hypervolume in minimization space."""
    valid = points_min[np.all(points_min < ref_min, axis=1)]
    if valid.shape[0] == 0:
        return 0.0
    nd, _ = filter_nondominated_array(valid, minimize=(True, True))
    order = np.lexsort((nd[:, 1], nd[:, 0]))
    pts = nd[order]
    hv = 0.0
    prev_y = float(ref_min[1])
    rx = float(ref_min[0])
    for i in range(pts.shape[0]):
        x_val = float(pts[i, 0])
        y_val = float(pts[i, 1])
        width = rx - x_val
        height = prev_y - y_val
        if width > 0.0 and height > 0.0:
            hv += width * height
            prev_y = y_val
    return float(hv)


def _hv_3d_exact_min(points_min: np.ndarray, ref_min: np.ndarray) -> float:
    """Exact 3D Hypervolume via z-slice decomposition in minimization space."""
    valid = points_min[np.all(points_min < ref_min, axis=1)]
    if valid.shape[0] == 0:
        return 0.0
    nd, _ = filter_nondominated_array(valid, minimize=(True, True, True))
    order = np.argsort(nd[:, 2])
    pts = nd[order]
    z_coords = pts[:, 2]
    hv = 0.0
    for i in range(pts.shape[0]):
        z_next = float(z_coords[i + 1]) if i + 1 < pts.shape[0] else float(ref_min[2])
        dz = z_next - float(z_coords[i])
        if dz <= 0.0:
            continue
        slice_2d = pts[: i + 1, :2]
        area_2d = _hv_2d_exact_min(slice_2d, ref_min[:2])
        hv += area_2d * dz
    return float(hv)


def compute_hypervolume(
    front: np.ndarray,
    ref_point: np.ndarray,
    minimize: tuple[bool, ...] | None = None,
    *,
    mc_samples: int = 20000,
    seed: int = 42,
) -> float:
    """Compute Hypervolume indicator (higher is better).

    Exact for m = 2 (O(N log N)) and m = 3 (exact z-slice sweep); deterministic Monte Carlo for m >= 4.
    """
    pts_min = to_minimization_space(front, minimize)
    ref_arr = np.asarray(ref_point, dtype=np.float64).reshape(1, -1)
    ref_min = to_minimization_space(ref_arr, minimize)[0]
    if pts_min.shape[0] == 0:
        return 0.0

    m = pts_min.shape[1]
    if m == 2:
        return _hv_2d_exact_min(pts_min, ref_min)
    if m == 3:
        return _hv_3d_exact_min(pts_min, ref_min)

    valid = pts_min[np.all(pts_min < ref_min, axis=1)]
    if valid.shape[0] == 0:
        return 0.0
    low = np.min(valid, axis=0)
    box_vol = float(np.prod(ref_min - low))
    if box_vol <= 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=low, high=ref_min, size=(mc_samples, m))
    dominated = np.zeros(mc_samples, dtype=bool)
    chunk_size = 2000
    for start in range(0, mc_samples, chunk_size):
        end = min(start + chunk_size, mc_samples)
        chunk = samples[start:end]
        dom = np.any(np.all(valid[:, None, :] <= chunk[None, :, :], axis=2), axis=0)
        dominated[start:end] = dom
    return float(box_vol * np.mean(dominated))


def compute_indicators_dataset(
    collection: FrontCollection,
    indicators: tuple[str, ...] = ("HV", "IGD+", "IGD", "EPS", "Spacing"),
    *,
    filter_dominated: bool = True,
    seed: int = 42,
) -> Dataset:
    """Compute multi-objective quality indicators across a FrontCollection and return a SAES Dataset."""
    if not collection.fronts:
        raise ValueError("FrontCollection contains no approximation fronts.")

    normalized_inds: list[str] = []
    for raw in indicators:
        key = raw.strip().upper()
        if key not in INDICATOR_CANONICAL_NAMES:
            valid = ", ".join(sorted(set(INDICATOR_CANONICAL_NAMES.values())))
            raise ValueError(f"Unsupported indicator '{raw}'. Available: {valid}.")
        canon = INDICATOR_CANONICAL_NAMES[key]
        if canon not in normalized_inds:
            normalized_inds.append(canon)

    rows: list[dict[str, object]] = []
    for inst in collection.instances:
        inst_fronts = collection.get_fronts(inst)
        if not inst_fronts:
            continue
        minimize = inst_fronts[0].minimize

        ref_front = collection.get_reference(inst)
        if ref_front is not None and ref_front.n_points > 0:
            ref_arr = ref_front.to_array()
        else:
            ref_arr = compute_accumulated_front(inst_fronts).to_array()

        # Compute consistent per-instance normalization and HV nadir reference point
        all_arrays = [f.to_array() for f in inst_fronts if f.n_points > 0] + [ref_arr]
        ideal_min, nadir_min = compute_ideal_nadir(all_arrays, minimize=minimize)
        span = np.where((nadir_min - ideal_min) > 1e-12, nadir_min - ideal_min, 1.0)

        # Normalized space [0, 1]^m with HV reference point at 1.1 in every objective
        ref_norm = (to_minimization_space(ref_arr, minimize) - ideal_min) / span
        hv_ref_norm = np.full(ref_arr.shape[1], 1.1, dtype=np.float64)

        for front in inst_fronts:
            arr = front.to_array()
            if filter_dominated and arr.shape[0] > 0:
                arr, _ = filter_nondominated_array(arr, minimize=minimize)
            arr_norm = (to_minimization_space(arr, minimize) - ideal_min) / span if arr.shape[0] > 0 else arr

            for ind in normalized_inds:
                ind_upper = ind.upper()
                if ind_upper == "HV":
                    val = compute_hypervolume(arr_norm, hv_ref_norm, minimize=None, seed=seed)
                elif ind_upper == "IGD+":
                    val = compute_igd_plus(arr_norm, ref_norm, minimize=None)
                elif ind_upper == "IGD":
                    val = compute_igd(arr_norm, ref_norm, minimize=None)
                elif ind_upper == "GD":
                    val = compute_gd(arr_norm, ref_norm, minimize=None)
                elif ind_upper == "EPS":
                    val = compute_additive_epsilon(arr_norm, ref_norm, minimize=None)
                elif ind_upper == "SPACING":
                    val = compute_spacing(arr_norm, minimize=None)
                else:
                    continue

                if not np.isfinite(val):
                    val = 0.0 if INDICATOR_DIRECTIONS[ind_upper] else 1e6

                rows.append(
                    {
                        "algorithm": front.algorithm,
                        "instance": front.instance,
                        "metric": ind,
                        "run_id": int(front.run_id),
                        "value": float(val),
                        "maximize": bool(INDICATOR_DIRECTIONS[ind_upper]),
                    }
                )

    df = pd.DataFrame(rows, columns=["algorithm", "instance", "metric", "run_id", "value", "maximize"])
    return Dataset(df)
