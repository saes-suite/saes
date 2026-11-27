"""SAES: Stochastic Algorithm Evaluation Suite."""

from typing import Any

import saes.plugins as plugins
from saes._version import __version__ as __version__
from saes.api import (
    analyze,
    analyze_statistical_power,
    compute_minimum_detectable_effect,
    compute_performance_profiles,
    explore,
    export_exploration,
    export_plot,
    export_table,
    generate_reproducibility_artifacts,
    generate_table,
    load,
    plot,
    power,
    profiles,
    reproducibility,
    statistics,
    tost,
    tost_equivalence_test,
    validate,
)
from saes.domain.exceptions import (
    ConfigurationError,
    InfrastructureError,
    InputError,
    SAESError,
    StatisticalAnalysisError,
    ValidationError,
    VisualizationError,
)
from saes.domain.models import (
    AlgorithmProfileCurve,
    AlgorithmRankItem,
    BayesianCorrelatedTResult,
    BayesianPlackettLuceItem,
    BayesianPlackettLuceResult,
    BayesianPosteriorItem,
    BayesianRankResult,
    BlockedOmnibusResult,
    CriticalDistanceResult,
    DataExplorationResult,
    Dataset,
    EffectSizeComparisonItem,
    EffectSizeForestResult,
    EnvironmentManifest,
    GlobalMetaRankingItem,
    PerformanceProfilesResult,
    PostHocComparisonItem,
    PostHocPowerItem,
    PostHocResult,
    PowerAnalysisResult,
    PowerCurveSeries,
    ReproducibilityResult,
    SampleSizeRecommendation,
    SensitivityAuditItem,
    StatisticalResult,
    StatisticalTableResult,
    TOSTResult,
    ValidationReport,
    WinTieLossCell,
)
from saes.plugins.registry import PluginManifest, PluginRegistry, SAESPlugin
from saes.reports.renderers import TableRenderer, TableRendererRegistry
from saes.stats.base import StatisticalTest
from saes.stats.registry import TestRegistry


def __getattr__(name: str) -> Any:
    if name == "PlotterRegistry":
        from saes.visualization.registry import PlotterRegistry

        return PlotterRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Dataset",
    "StatisticalResult",
    "StatisticalTableResult",
    "DataExplorationResult",
    "ValidationReport",
    "AlgorithmRankItem",
    "CriticalDistanceResult",
    "BlockedOmnibusResult",
    "BayesianPosteriorItem",
    "BayesianRankResult",
    "BayesianCorrelatedTResult",
    "BayesianPlackettLuceItem",
    "BayesianPlackettLuceResult",
    "PostHocComparisonItem",
    "PostHocResult",
    "EffectSizeComparisonItem",
    "EffectSizeForestResult",
    "TOSTResult",
    "PerformanceProfilesResult",
    "AlgorithmProfileCurve",
    "WinTieLossCell",
    "GlobalMetaRankingItem",
    "PowerAnalysisResult",
    "SampleSizeRecommendation",
    "PowerCurveSeries",
    "PostHocPowerItem",
    "SensitivityAuditItem",
    "ReproducibilityResult",
    "EnvironmentManifest",
    "StatisticalTest",
    "TestRegistry",
    "TableRenderer",
    "TableRendererRegistry",
    "PlotterRegistry",
    "PluginManifest",
    "SAESPlugin",
    "PluginRegistry",
    "plugins",
    "SAESError",
    "InputError",
    "ValidationError",
    "ConfigurationError",
    "StatisticalAnalysisError",
    "VisualizationError",
    "InfrastructureError",
    "load",
    "validate",
    "analyze",
    "generate_table",
    "export_table",
    "explore",
    "export_exploration",
    "plot",
    "export_plot",
    "statistics",
    "profiles",
    "power",
    "tost",
    "tost_equivalence_test",
    "compute_minimum_detectable_effect",
    "reproducibility",
    "compute_performance_profiles",
    "analyze_statistical_power",
    "generate_reproducibility_artifacts",
]

