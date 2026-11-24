"""Extensible registry for StatisticalTest strategies in SAES."""

from __future__ import annotations

from typing import Any

from saes.domain.exceptions import StatisticalAnalysisError
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.base import StatisticalTest
from saes.stats.nonparametric import KruskalWallisTest, MannWhitneyUTest, WilcoxonSignedRankTest
from saes.stats.posthoc import BrunnerMunzelTest
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


class TestRegistry:
    """Central extensible registry mapping test keys to `StatisticalTest` implementations.

    Users can register custom hypothesis tests implementing `StatisticalTest` via:
        >>> TestRegistry.register("my_custom_test", MyCustomTest)
    """

    _registry: dict[str, type[StatisticalTest]] = {
        "ttest_ind": IndependentTTest,
        "ttest_welch": WelchTTest,
        "ttest_rel": PairedTTest,
        "mann_whitney": MannWhitneyUTest,
        "wilcoxon": WilcoxonSignedRankTest,
        "brunner_munzel": BrunnerMunzelTest,
        "anova_oneway": OneWayANOVA,
        "anova_welch": WelchANOVA,
        "kruskal": KruskalWallisTest,
    }

    @classmethod
    def register(cls, key: str, test_cls: type[StatisticalTest]) -> None:
        """Register a custom `StatisticalTest` subclass under `key`."""
        if not isinstance(test_cls, type) or not issubclass(test_cls, StatisticalTest):
            raise StatisticalAnalysisError(
                f"Registered class must inherit from StatisticalTest, got {test_cls!r}."
            )
        cls._registry[key.strip().lower()] = test_cls

    @classmethod
    def get(cls, key: str, **kwargs: Any) -> StatisticalTest:
        """Instantiate a registered `StatisticalTest` by key."""
        norm_key = key.strip().lower()
        if norm_key not in cls._registry:
            valid = ", ".join(sorted(cls._registry.keys()))
            raise StatisticalAnalysisError(f"Unknown statistical test '{key}'. Available: {valid}")
        return cls._registry[norm_key](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return sorted list of all registered statistical test keys."""
        return sorted(cls._registry.keys())
