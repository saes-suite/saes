"""Statistical decision tree engine for SAES.

Methodological note:
    Conditional pre-testing (selecting a parametric vs. non-parametric test solely from a
    preliminary normality or variance test) can distort nominal Type I error rates under certain
    alternatives (Zimmerman 2004; Rochon et al. 2012). SAES therefore exposes explicit test
    selection (`force_test=...` or `saes.statistics.*`) alongside this transparent automated
    decision engine, and requires explicit declaration of statistical pairing (`paired=True`)
    rather than inferring dependence from equal `run_id` indices across stochastic runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import StatisticalResult
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.base import StatisticalTest
from saes.stats.nonparametric import KruskalWallisTest, MannWhitneyUTest, WilcoxonSignedRankTest
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def is_paired_data(cell_df: pd.DataFrame) -> bool:
    """Check whether observations share structurally aligned 1-to-1 `run_id` indices across algorithms.

    Note:
        Structural alignment of `run_id` is necessary (but NOT sufficient) for statistical pairing.
        Independent stochastic runs (e.g., separate random seeds numbered 0..29) are NOT paired
        unless Common Random Numbers (CRN) or identical cross-validation folds were explicitly used.
        Use `paired=True` explicitly in `TestDecisionEngine` / `analyze()` when observations are truly paired.
    """
    if "run_id" not in cell_df.columns or "algorithm" not in cell_df.columns:
        return False
    pivoted = cell_df.pivot(index="run_id", columns="algorithm", values="value")
    return bool(not pivoted.isnull().any().any() and len(pivoted) >= 2)


class TestDecisionEngine:
    """Automated statistical test selection decision engine with normality and variance gating."""

    __test__ = False

    def __init__(
        self,
        alpha: float = 0.05,
        check_normality: bool = True,
        paired: bool = False,
    ) -> None:
        self.alpha = alpha
        self.check_normality = check_normality
        self.paired = paired

    def select_test(
        self,
        groups: dict[str, np.ndarray],
        is_paired: bool = False,
        check_normality: bool | None = None,
    ) -> StatisticalTest:
        """Select the appropriate statistical test based on design and distributional checks.

        Args:
            groups: Dictionary mapping group names to 1D data arrays.
            is_paired: Explicit flag indicating whether observations are statistically paired
                (e.g., identical CV folds or Common Random Numbers). Defaults to False.
            check_normality: Override whether Shapiro-Wilk normality is checked prior to
                selecting parametric vs. non-parametric procedures.

        Returns:
            StatisticalTest strategy instance.
        """
        k = len(groups)
        if k < 2:
            raise StatisticalAnalysisError(f"Statistical test selection requires >= 2 groups, got {k}.")

        use_norm = self.check_normality if check_normality is None else check_normality

        if k == 2:
            keys = list(groups.keys())
            x1 = groups[keys[0]][~np.isnan(groups[keys[0]])]
            x2 = groups[keys[1]][~np.isnan(groups[keys[1]])]
            can_pair = is_paired and len(x1) == len(x2) and len(x1) >= 2

            if can_pair:
                if use_norm:
                    diff_norm = NormalityChecker(alpha=self.alpha).check({"Paired Differences": x1 - x2})
                    if not diff_norm[0].passed:
                        return WilcoxonSignedRankTest()
                return PairedTTest()

            if use_norm:
                norm_checks = NormalityChecker(alpha=self.alpha).check(groups)
                if not all(n.passed for n in norm_checks):
                    return MannWhitneyUTest()

            homogeneity = HomogeneityChecker(alpha=self.alpha).check(groups)
            return IndependentTTest() if homogeneity.passed else WelchTTest()

        # k > 2 independent groups within a cell
        if use_norm:
            norm_checks = NormalityChecker(alpha=self.alpha).check(groups)
            if not all(n.passed for n in norm_checks):
                return KruskalWallisTest()

        homogeneity = HomogeneityChecker(alpha=self.alpha).check(groups)
        return OneWayANOVA() if homogeneity.passed else WelchANOVA()

    def analyze_cell(
        self,
        cell_df: pd.DataFrame,
        force_test: StatisticalTest | None = None,
        paired: bool | None = None,
    ) -> StatisticalResult:
        """Analyze a single (instance, metric) DataFrame cell using automated decision engine or forced test.

        Args:
            cell_df: Filtered DataFrame for a single (instance, metric) pair.
            force_test: Optional explicit StatisticalTest instance to override decision engine.
            paired: Explicit pairing flag; defaults to `self.paired` (False). When True,
                observations are aligned by `run_id` before executing paired tests.

        Returns:
            StatisticalResult: Outcome of statistical evaluation.
        """
        use_paired = self.paired if paired is None else paired
        groups: dict[str, np.ndarray] = {}

        if use_paired and is_paired_data(cell_df):
            pivoted = cell_df.pivot(index="run_id", columns="algorithm", values="value").sort_index()
            for col in pivoted.columns:
                groups[str(col)] = pivoted[col].to_numpy(dtype=float)
            effective_paired = True
        else:
            for alg_key, grp in cell_df.groupby("algorithm"):
                groups[str(alg_key)] = grp["value"].to_numpy(dtype=float)
            effective_paired = False

        if force_test is not None:
            return force_test.run(groups, alpha=self.alpha)

        selected_test = self.select_test(groups, is_paired=effective_paired)
        return selected_test.run(groups, alpha=self.alpha)
