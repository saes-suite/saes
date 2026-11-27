"""REST endpoints exposing installed SAES Python plugins and extensible registries."""

from __future__ import annotations

from fastapi import APIRouter

from saes.plugins.registry import PluginRegistry
from saes.reports.renderers import TableRendererRegistry
from saes.server.schemas import PluginRegistryResponse
from saes.stats.registry import TestRegistry

router = APIRouter(prefix="/api/v1/plugins", tags=["Plugins"])


@router.get("", response_model=PluginRegistryResponse)
def list_plugins() -> PluginRegistryResponse:
    """List installed SAES Python plugins along with dynamically discovered tests, renderers, and plotters."""
    installed = PluginRegistry.list_installed(discover=True)
    tests = TestRegistry.list_available()
    renderers = TableRendererRegistry.list_available()

    try:
        from saes.visualization.registry import PlotterRegistry

        plotters = sorted(PlotterRegistry.list_available().keys())
    except ImportError:
        plotters = []

    return PluginRegistryResponse(
        plugins=installed,
        registered_tests=tests,
        registered_renderers=renderers,
        registered_plotters=plotters,
    )
