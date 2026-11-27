"""Mathematical verification tests for dominance filtering, set coverage, and quality indicators."""

from __future__ import annotations

import numpy as np
import pytest

import saes
from saes.plugins import pareto as saes_pareto
from saes.plugins.pareto.dominance import (
    compute_coverage_and_representatives,
    compute_set_coverage,
    filter_nondominated_array,
)
from saes.plugins.pareto.indicators import (
    compute_additive_epsilon,
    compute_gd,
    compute_hypervolume,
    compute_igd,
    compute_igd_plus,
    compute_spacing,
)


def test_nondominated_filtering_2d_and_3d() -> None:
    pts_2d = np.array(
        [
            [1.0, 4.0],
            [2.0, 2.0],
            [3.0, 3.0],  # dominated by (2, 2)
            [4.0, 1.0],
            [2.0, 2.0],  # duplicate
            [5.0, 5.0],  # dominated
        ]
    )
    nd, _ = filter_nondominated_array(pts_2d)
    assert nd.shape == (3, 2)
    expected = {(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)}
    assert {tuple(row) for row in nd} == expected

    # Maximization on both objectives: (5, 5) dominates everything
    nd_max, _ = filter_nondominated_array(pts_2d, minimize=(False, False))
    assert nd_max.shape == (1, 2)
    assert tuple(nd_max[0]) == (5.0, 5.0)

    # 3D exact filtering
    pts_3d = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 3.5],  # dominated by (1,2,3), (2,1,3), (2,2,2)
            [3.0, 3.0, 3.0],  # dominated
        ]
    )
    nd3, _ = filter_nondominated_array(pts_3d)
    assert nd3.shape == (3, 3)


def test_exact_2d_and_3d_hypervolume() -> None:
    # 2D analytic case: points (1, 3) and (2, 1) with ref (4, 4)
    # Region = [1, 4]x[3, 4] (= 3*1 = 3) + [2, 4]x[1, 3] (= 2*2 = 4) => total = 7.0
    pts_2d = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 3.5]])  # (3, 3.5) is dominated
    hv_2d = compute_hypervolume(pts_2d, np.array([4.0, 4.0]))
    assert hv_2d == pytest.approx(7.0, rel=1e-12)

    # 3D analytic case: single box [1, 3] x [2, 5] x [0, 4] => volume = 2 * 3 * 4 = 24.0
    pts_3d_single = np.array([[1.0, 2.0, 0.0]])
    assert compute_hypervolume(pts_3d_single, np.array([3.0, 5.0, 4.0])) == pytest.approx(24.0, rel=1e-12)

    # 3D union of two boxes against ref (2, 2, 2):
    # A=(0, 1, 0) -> vol 2*1*2 = 4; B=(1, 0, 0) -> vol 1*2*2 = 4; intersection [1,2]x[1,2]x[0,2] = 2 => union = 6.0
    pts_3d_two = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert compute_hypervolume(pts_3d_two, np.array([2.0, 2.0, 2.0])) == pytest.approx(6.0, rel=1e-12)


def test_igd_igd_plus_gd_epsilon_spacing_and_coverage() -> None:
    ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    shifted = ref + 0.2

    assert compute_igd(ref, ref) == pytest.approx(0.0)
    assert compute_igd_plus(ref, ref) == pytest.approx(0.0)
    assert compute_gd(ref, ref) == pytest.approx(0.0)
    assert compute_additive_epsilon(ref, ref) == pytest.approx(0.0)

    # Shifting every point by (+0.2, +0.2): additive epsilon == 0.2, IGD+ == sqrt(0.08)
    assert compute_additive_epsilon(shifted, ref) == pytest.approx(0.2, rel=1e-12)
    assert compute_igd_plus(shifted, ref) == pytest.approx(np.sqrt(0.08), rel=1e-12)
    assert compute_igd(shifted, ref) > 0.0
    assert compute_spacing(ref) == pytest.approx(0.0, abs=1e-12)

    # Set coverage: ref completely dominates shifted, shifted covers 0% of ref
    assert compute_set_coverage(ref, shifted) == pytest.approx(1.0)
    assert compute_set_coverage(shifted, ref) == pytest.approx(0.0)


def test_to_indicators_dataset_produces_valid_saes_dataset() -> None:
    coll = saes_pareto.create_demo_collection(n_runs=6, seed=7)
    ds = coll.to_indicators_dataset(indicators=["HV", "IGD+", "IGD", "EPS", "Spacing"])

    val_report = saes.validate(ds)
    assert val_report.is_valid, f"Generated SAES Dataset failed validation: {val_report.errors}"
    assert set(ds.algorithms) == {"NSGA-II", "MOEA/D", "SPEA2", "SMPSO"}
    assert set(ds.instances) == {"ZDT1", "ZDT2", "ZDT3", "DTLZ2"}
    assert set(ds.metrics) == {"HV", "IGD+", "IGD", "EPS", "Spacing"}

    # Confirm seamless end-to-end execution with SAES statistical tests
    cd = saes.statistics.friedman_nemenyi(ds, metric="IGD+")
    assert len(cd.ranks) == 4

    cov_rep = compute_coverage_and_representatives(coll, instance="ZDT1")
    assert len(cov_rep.representative_runs) == 4
    assert len(cov_rep.coverage_matrix) == 16
