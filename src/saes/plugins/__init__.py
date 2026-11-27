"""SAES Python plugin system and built-in reference plugins (e.g. ``saes.plugins.pareto``).

Exposes the PEP ``importlib.metadata.entry_points`` plugin discovery API directly on the
``saes.plugins`` package namespace so both ``saes.plugins.list_installed()`` and
``from saes.plugins import pareto`` work seamlessly without module shadowing.
"""

from __future__ import annotations

from saes.plugins.registry import (
    ENTRYPOINT_PLOTTERS,
    ENTRYPOINT_PLUGINS,
    ENTRYPOINT_RENDERERS,
    ENTRYPOINT_TESTS,
    PluginManifest,
    PluginRegistry,
    SAESPlugin,
)

# Ergonomic package-level shortcuts delegating to PluginRegistry
register = PluginRegistry.register
unregister = PluginRegistry.unregister
reset = PluginRegistry.reset
discover = PluginRegistry.discover
discover_tests = PluginRegistry.discover_tests
discover_renderers = PluginRegistry.discover_renderers
discover_plotters = PluginRegistry.discover_plotters
list_installed = PluginRegistry.list_installed
is_installed = PluginRegistry.is_installed
get = PluginRegistry.get
get_namespace = PluginRegistry.get_namespace
mount_api_routers = PluginRegistry.mount_api_routers

__all__ = [
    "ENTRYPOINT_PLUGINS",
    "ENTRYPOINT_TESTS",
    "ENTRYPOINT_RENDERERS",
    "ENTRYPOINT_PLOTTERS",
    "PluginManifest",
    "SAESPlugin",
    "PluginRegistry",
    "register",
    "unregister",
    "reset",
    "discover",
    "discover_tests",
    "discover_renderers",
    "discover_plotters",
    "list_installed",
    "is_installed",
    "get",
    "get_namespace",
    "mount_api_routers",
]
