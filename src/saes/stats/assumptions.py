"""Statistical assumption testing engine for SAES (Shapiro-Wilk, Lilliefors, Anderson-Darling, Levene)."""

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors

from saes.domain.models import AssumptionResult


class NormalityChecker:
    """Evaluates normality assumption using Shapiro-Wilk test for each group."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def check(self, groups: dict[str, np.ndarray]) -> list[AssumptionResult]:
        """Check normality for all groups in dictionary.

        Args:
            groups: Dictionary mapping group names to 1D numpy arrays.

        Returns:
            List of AssumptionResult objects for each group.
        """
        results: list[AssumptionResult] = []
        for name, data in groups.items():
            clean_data = data[~np.isnan(data)]
            n = len(clean_data)

            if n < 3:
                results.append(
                    AssumptionResult(
                        name=f"Normality (Shapiro-Wilk) [{name}]",
                        statistic=np.nan,
                        p_value=np.nan,
                        passed=True,  # Default to True with warning due to small sample
                        alpha=self.alpha,
                        details=f"Sample size N={n} is too small for Shapiro-Wilk test (requires N >= 3).",
                    )
                )
                continue

            if np.all(clean_data == clean_data[0]):
                results.append(
                    AssumptionResult(
                        name=f"Normality (Shapiro-Wilk) [{name}]",
                        statistic=1.0,
                        p_value=1.0,
                        passed=True,
                        alpha=self.alpha,
                        details=f"All values in group '{name}' are constant (zero variance).",
                    )
                )
                continue

            stat, p_val = stats.shapiro(clean_data)
            passed = bool(p_val >= self.alpha)
            status_str = "satisfied" if passed else "violated"
            results.append(
                AssumptionResult(
                    name=f"Normality (Shapiro-Wilk) [{name}]",
                    statistic=float(stat),
                    p_value=float(p_val),
                    passed=passed,
                    alpha=self.alpha,
                    details=f"Shapiro-Wilk test W={stat:.4f}, p={p_val:.4f}. Normality {status_str}.",
                )
            )

        return results


class HomogeneityChecker:
    """Evaluates homogeneity of variance (homoscedasticity) using Levene's test."""

    def __init__(self, alpha: float = 0.05, center: str = "median") -> None:
        self.alpha = alpha
        self.center = center

    def check(self, groups: dict[str, np.ndarray]) -> AssumptionResult:
        """Check homogeneity of variance across 2 or more groups.

        Args:
            groups: Dictionary mapping group names to 1D numpy arrays.

        Returns:
            AssumptionResult representing Levene's test outcome.
        """
        clean_groups = [g[~np.isnan(g)] for g in groups.values() if len(g[~np.isnan(g)]) > 0]

        if len(clean_groups) < 2 or any(len(g) < 2 for g in clean_groups):
            return AssumptionResult(
                name="Homogeneity of Variance (Levene)",
                statistic=np.nan,
                p_value=np.nan,
                passed=True,
                alpha=self.alpha,
                details="Requires at least 2 groups with >= 2 observations each for Levene's test.",
            )

        # Check if all groups have zero variance
        stds = [np.std(g, ddof=1) for g in clean_groups]
        if all(s == 0 for s in stds):
            return AssumptionResult(
                name="Homogeneity of Variance (Levene)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                details="All groups have zero variance.",
            )

        stat, p_val = stats.levene(*clean_groups, center=self.center)
        passed = bool(p_val >= self.alpha)
        status_str = "satisfied" if passed else "violated"
        return AssumptionResult(
            name="Homogeneity of Variance (Levene)",
            statistic=float(stat),
            p_value=float(p_val),
            passed=passed,
            alpha=self.alpha,
            details=f"Levene's test F={stat:.4f}, p={p_val:.4f}. Equal variances {status_str}.",
        )


class LillieforsChecker:
    """Evaluates normality with estimated mean/SD using Lilliefors-corrected Kolmogorov-Smirnov test."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def check(self, groups: dict[str, np.ndarray]) -> list[AssumptionResult]:
        """Check normality for all groups using Lilliefors (1967) test."""
        results: list[AssumptionResult] = []
        for name, data in groups.items():
            clean_data = data[~np.isnan(data)]
            n = len(clean_data)
            if n < 4:
                results.append(
                    AssumptionResult(
                        name=f"Normality (Lilliefors KS) [{name}]",
                        statistic=np.nan,
                        p_value=np.nan,
                        passed=True,
                        alpha=self.alpha,
                        details=f"Sample size N={n} is too small for Lilliefors test (requires N >= 4).",
                    )
                )
                continue

            if np.all(clean_data == clean_data[0]):
                results.append(
                    AssumptionResult(
                        name=f"Normality (Lilliefors KS) [{name}]",
                        statistic=0.0,
                        p_value=1.0,
                        passed=True,
                        alpha=self.alpha,
                        details=f"All values in group '{name}' are constant (zero variance).",
                    )
                )
                continue

            stat_val, p_val = sm_lilliefors(clean_data, dist="norm", pvalmethod="table")
            stat_f = float(stat_val)
            p_f = float(p_val)
            passed = bool(p_f >= self.alpha)
            status_str = "satisfied" if passed else "violated"
            results.append(
                AssumptionResult(
                    name=f"Normality (Lilliefors KS) [{name}]",
                    statistic=stat_f,
                    p_value=p_f,
                    passed=passed,
                    alpha=self.alpha,
                    details=f"Lilliefors test D={stat_f:.4f}, p={p_f:.4f}. Normality {status_str}.",
                )
            )
        return results


class AndersonDarlingChecker:
    """Evaluates normality using Anderson-Darling test (Stephens 1974)."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def check(self, groups: dict[str, np.ndarray]) -> list[AssumptionResult]:
        """Check normality for all groups via Anderson-Darling statistic."""
        results: list[AssumptionResult] = []
        for name, data in groups.items():
            clean_data = data[~np.isnan(data)]
            n = len(clean_data)
            if n < 4:
                results.append(
                    AssumptionResult(
                        name=f"Normality (Anderson-Darling) [{name}]",
                        statistic=np.nan,
                        p_value=np.nan,
                        passed=True,
                        alpha=self.alpha,
                        details=f"Sample size N={n} is too small for Anderson-Darling test (requires N >= 4).",
                    )
                )
                continue

            if np.all(clean_data == clean_data[0]):
                results.append(
                    AssumptionResult(
                        name=f"Normality (Anderson-Darling) [{name}]",
                        statistic=0.0,
                        p_value=1.0,
                        passed=True,
                        alpha=self.alpha,
                        details=f"All values in group '{name}' are constant (zero variance).",
                    )
                )
                continue

            try:
                ad_res = stats.anderson(clean_data, dist="norm", method="interpolate")
                a2 = float(ad_res.statistic)
                p_approx = float(ad_res.pvalue)
            except TypeError:
                ad_res = stats.anderson(clean_data, dist="norm")
                a2 = float(ad_res.statistic)
                # D'Agostino & Stephens (1986) modified statistic A2* and p-value approximation
                a2_star = a2 * (1.0 + 0.75 / n + 2.25 / (n**2))
                if a2_star >= 0.6:
                    p_approx = float(np.exp(1.2937 - 5.709 * a2_star + 0.0186 * (a2_star**2)))
                elif a2_star >= 0.34:
                    p_approx = float(np.exp(0.9177 - 4.279 * a2_star - 1.38 * (a2_star**2)))
                elif a2_star >= 0.2:
                    p_approx = float(1.0 - np.exp(-8.318 + 42.796 * a2_star - 59.938 * (a2_star**2)))
                else:
                    p_approx = float(1.0 - np.exp(-13.436 + 101.14 * a2_star - 223.73 * (a2_star**2)))
            p_approx = min(1.0, max(0.0, p_approx))
            passed = bool(p_approx >= self.alpha)
            status_str = "satisfied" if passed else "violated"
            results.append(
                AssumptionResult(
                    name=f"Normality (Anderson-Darling) [{name}]",
                    statistic=a2,
                    p_value=p_approx,
                    passed=passed,
                    alpha=self.alpha,
                    details=f"Anderson-Darling A²={a2:.4f}, p≈{p_approx:.4f}. Normality {status_str}.",
                )
            )
        return results
