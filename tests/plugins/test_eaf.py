"""Exact mathematical tests for 2D Empirical Attainment Functions (EAF) and KS-EAF permutation tests."""

from __future__ import annotations

import numpy as np
import pytest

from saes.plugins import pareto as saes_pareto
from saes.plugins.pareto.eaf import compute_attainment, compute_attainment_surfaces_2d, compute_eaf_difference
from saes.plugins.pareto.models import FrontCollection


def test_exact_2d_attainment_surfaces_known_runs() -> None:
    # 4 shifted copies of a 2-point front:
    # Run 0: (1, 4), (4, 1)
    # Run 1: (2, 5), (5, 2)
    # Run 2: (3, 6), (6, 3)
    # Run 3: (4, 7), (7, 4)
    runs = [
        np.array([[1.0, 4.0], [4.0, 1.0]]),
        np.array([[2.0, 5.0], [5.0, 2.0]]),
        np.array([[3.0, 6.0], [6.0, 3.0]]),
        np.array([[4.0, 7.0], [7.0, 4.0]]),
    ]
    surfaces = compute_attainment_surfaces_2d(runs, levels=(0.0, 0.25, 0.5, 0.75, 1.0))
    by_lvl = {s.level: s for s in surfaces}

    # k=1 (best / 0%): matches run 0 exactly
    assert by_lvl[0.0].rank_k == 1
    assert by_lvl[0.0].points == ((1.0, 4.0), (4.0, 1.0))

    # k=2 (50% median): matches run 1 exactly
    assert by_lvl[0.5].rank_k == 2
    assert by_lvl[0.5].points == ((2.0, 5.0), (5.0, 2.0))

    # k=4 (100% worst): attained by all 4 runs -> starts at x=4 with y=7, drops to y=4 at x=7
    assert by_lvl[1.0].rank_k == 4
    assert by_lvl[1.0].points == ((4.0, 7.0), (7.0, 4.0))


def test_eaf_difference_antisymmetry_and_permutation_significance() -> None:
    rng = np.random.default_rng(123)
    x = np.linspace(0.0, 1.0, 25)
    # Clearly separated algorithms: GoodAlg strictly dominates BadAlg across all runs
    good_runs = [np.column_stack([x, 1.0 - np.sqrt(x) + rng.uniform(0.01, 0.03, size=x.size)]) for _ in range(10)]
    bad_runs = [np.column_stack([x, 1.0 - np.sqrt(x) + rng.uniform(0.25, 0.35, size=x.size)]) for _ in range(10)]
    # Identical distribution control
    same_runs = [np.column_stack([x, 1.0 - np.sqrt(x) + rng.uniform(0.01, 0.03, size=x.size)]) for _ in range(10)]

    coll = FrontCollection.from_arrays(
        {"GoodAlg": good_runs, "BadAlg": bad_runs, "SameAlg": same_runs},
        instance="ZDT1_synth",
    )

    diff_ab = compute_eaf_difference(coll, "ZDT1_synth", "GoodAlg", "BadAlg", n_permutations=200, seed=42)
    diff_ba = compute_eaf_difference(coll, "ZDT1_synth", "BadAlg", "GoodAlg", n_permutations=200, seed=42)

    assert diff_ab.ks_statistic == pytest.approx(1.0, abs=1e-6)
    assert diff_ab.max_advantage_a == pytest.approx(1.0, abs=1e-6)
    assert diff_ab.max_advantage_b == pytest.approx(0.0, abs=1e-6)
    assert diff_ab.is_significant is True
    assert diff_ab.p_value < 0.01

    # Exact antisymmetry
    assert diff_ba.ks_statistic == pytest.approx(diff_ab.ks_statistic, abs=1e-12)
    assert diff_ba.max_advantage_b == pytest.approx(diff_ab.max_advantage_a, abs=1e-12)
    assert diff_ba.max_advantage_a == pytest.approx(diff_ab.max_advantage_b, abs=1e-12)
    assert np.allclose(np.asarray(diff_ab.diff_matrix), -np.asarray(diff_ba.diff_matrix), atol=1e-12)

    # Identical distribution control should not achieve KS == 1
    diff_same = compute_eaf_difference(coll, "ZDT1_synth", "GoodAlg", "SameAlg", n_permutations=200, seed=42)
    assert diff_same.p_value > 0.05
    assert "EAF Difference" in diff_ab.summary()
    assert not diff_ab.to_dataframe().empty


def test_attainment_summary_and_dataframe_on_demo() -> None:
    coll = saes_pareto.create_demo_collection(n_runs=8, seed=42)
    att = compute_attainment(coll, instance="ZDT1", levels=(0.25, 0.5, 0.75))
    assert len(att.algorithms) == 4
    assert "Empirical Attainment Surfaces" in att.summary()
    df = att.to_dataframe()
    assert {"instance", "algorithm", "level", "f1", "f2"}.issubset(df.columns)
