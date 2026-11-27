"""Built-in SAES Pareto & EAF Plugin — Multi-Objective Front Analytics for SAES."""

from __future__ import annotations

from typing import Any

from saes._version import __version__ as __version__
from saes.plugins.pareto.dominance import (
    compute_accumulated_front,
    compute_coverage_and_representatives,
    compute_ideal_nadir,
    compute_set_coverage,
    filter_nondominated_array,
    filter_nondominated_front,
)
from saes.plugins.pareto.eaf import (
    compute_attainment,
    compute_attainment_surfaces_2d,
    compute_eaf_difference,
)
from saes.plugins.pareto.indicators import (
    compute_additive_epsilon,
    compute_gd,
    compute_hypervolume,
    compute_igd,
    compute_igd_plus,
    compute_indicators_dataset,
    compute_spacing,
)
from saes.plugins.pareto.io import (
    create_demo_collection,
    load_csv,
    load_directory,
    load_reference_front,
    load_zip,
)
from saes.plugins.pareto.models import (
    AlgorithmAttainment,
    AlgorithmRepresentativeRuns,
    AttainmentResult,
    AttainmentSurface,
    CoverageAndRepresentativeResult,
    CoverageCell,
    EAFDifferenceResult,
    FrontCollection,
    FrontCollectionSummary,
    InstanceFrontMeta,
    ParetoFront,
)
from saes.plugins.pareto.plugin import ParetoPlugin


def plot_fronts(collection: FrontCollection, **kwargs: Any) -> Any:
    """Generate a static Matplotlib or interactive Plotly visualization (requires ``saes[viz]``)."""
    from saes.plugins.pareto.plots import plot_fronts as _plot_fronts

    return _plot_fronts(collection, **kwargs)


def export_front_plot(collection: FrontCollection, **kwargs: Any) -> tuple[bytes, str, str]:
    """Export any Pareto/EAF figure as PNG, SVG, or PDF bytes (requires ``saes[viz]``)."""
    from saes.plugins.pareto.plots import export_front_plot as _export_front_plot

    return _export_front_plot(collection, **kwargs)


# Ergonomic top-level aliases matching SAES conventions
load = load_csv
attainment = compute_attainment
eaf_difference = compute_eaf_difference
coverage = compute_coverage_and_representatives
plot = plot_fronts
export_plot = export_front_plot

__all__ = [
    "__version__",
    "ParetoPlugin",
    "ParetoFront",
    "FrontCollection",
    "FrontCollectionSummary",
    "InstanceFrontMeta",
    "AttainmentSurface",
    "AlgorithmAttainment",
    "AttainmentResult",
    "EAFDifferenceResult",
    "CoverageCell",
    "AlgorithmRepresentativeRuns",
    "CoverageAndRepresentativeResult",
    "load",
    "load_csv",
    "load_reference_front",
    "load_directory",
    "load_zip",
    "create_demo_collection",
    "filter_nondominated_array",
    "filter_nondominated_front",
    "compute_set_coverage",
    "compute_accumulated_front",
    "compute_ideal_nadir",
    "compute_coverage_and_representatives",
    "coverage",
    "compute_attainment_surfaces_2d",
    "compute_attainment",
    "attainment",
    "compute_eaf_difference",
    "eaf_difference",
    "compute_igd",
    "compute_igd_plus",
    "compute_gd",
    "compute_additive_epsilon",
    "compute_spacing",
    "compute_hypervolume",
    "compute_indicators_dataset",
    "plot_fronts",
    "plot",
    "export_front_plot",
    "export_plot",
]
