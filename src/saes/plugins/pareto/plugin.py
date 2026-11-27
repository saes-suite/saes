"""SAESPlugin implementation registered via the `saes.plugins` Python entry point."""

from __future__ import annotations

from typing import Any

from saes._version import __version__
from saes.plugins.registry import PluginManifest, SAESPlugin


class ParetoPlugin(SAESPlugin):
    """Built-in SAES domain plugin for Multi-Objective Pareto Fronts & Empirical Attainment Functions."""

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            id="pareto",
            name="SAES Pareto & EAF",
            version=__version__,
            description=(
                "Multi-objective Pareto front visualization, exact 2D Empirical Attainment Functions (EAF), "
                "KS-EAF permutation testing, set coverage C(A, B), and quality indicator extraction to SAES Datasets."
            ),
            capabilities=(
                "pareto_fronts",
                "empirical_attainment_function",
                "eaf_difference_test",
                "set_coverage",
                "quality_indicators_bridge",
                "3d_and_many_objective",
            ),
            api_prefix="/api/v1/plugins/pareto",
            author="Emilio R. Carreira & Antonio J. Nebro",
            homepage="https://github.com/saes-suite/saes",
        )

    def get_api_router(self) -> Any | None:
        try:
            from saes.plugins.pareto.router import build_router

            return build_router()
        except ImportError:
            return None

    def get_namespace(self) -> Any:
        from saes.plugins import pareto

        return pareto
