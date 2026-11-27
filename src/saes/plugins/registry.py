"""PEP-standard Python plugin discovery and domain extension architecture for SAES.

SAES exposes four standard ``importlib.metadata.entry_points`` groups so external
packages can extend the library without modifying or bloating the statistical core:

- ``saes.tests``: Custom ``StatisticalTest`` subclasses auto-registered in ``TestRegistry``.
- ``saes.renderers``: Custom ``TableRenderer`` subclasses auto-registered in ``TableRendererRegistry``.
- ``saes.plotters``: Custom ``Plotter`` subclasses auto-registered in ``PlotterRegistry``.
- ``saes.plugins``: Full domain extensions implementing ``SAESPlugin`` (e.g. ``saes-pareto``),
  which may bundle Python namespaces, custom tests/renderers/plotters, and optional FastAPI routers
  auto-mounted under ``/api/v1/plugins/<plugin_id>``.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from saes.domain.exceptions import ConfigurationError

if TYPE_CHECKING:
    from saes.reports.renderers import TableRendererRegistry
    from saes.stats.registry import TestRegistry
    from saes.visualization.registry import PlotterRegistry

ENTRYPOINT_PLUGINS = "saes.plugins"
ENTRYPOINT_TESTS = "saes.tests"
ENTRYPOINT_RENDERERS = "saes.renderers"
ENTRYPOINT_PLOTTERS = "saes.plotters"


class PluginManifest(BaseModel):
    """Immutable metadata descriptor for an installed SAES Python plugin."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Unique lowercase plugin slug (e.g. 'pareto')")
    name: str = Field(min_length=1, description="Human-readable plugin display name")
    version: str = Field(min_length=1, description="Installed semantic version string")
    description: str = Field(default="", description="Short summary of plugin capabilities")
    capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Machine-readable capability identifiers advertised to REST/UI clients",
    )
    api_prefix: str | None = Field(
        default=None,
        description="Mounted FastAPI route prefix when the plugin exposes REST routes",
    )
    author: str = Field(default="", description="Plugin author or maintainer")
    homepage: str = Field(default="", description="Optional documentation or repository URL")


class SAESPlugin(ABC):
    """Abstract base interface for external SAES domain plugins.

    External packages declare a plugin in ``pyproject.toml``::

        [project.entry-points."saes.plugins"]
        pareto = "saes_pareto.plugin:ParetoPlugin"
    """

    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return immutable metadata describing this plugin."""
        raise NotImplementedError

    def register_tests(self, registry: type[TestRegistry]) -> None:
        """Optionally register custom ``StatisticalTest`` classes into ``TestRegistry``."""
        _ = registry

    def register_renderers(self, registry: type[TableRendererRegistry]) -> None:
        """Optionally register custom ``TableRenderer`` classes into ``TableRendererRegistry``."""
        _ = registry

    def register_plotters(self, registry: type[PlotterRegistry]) -> None:
        """Optionally register custom ``Plotter`` classes into ``PlotterRegistry``."""
        _ = registry

    def get_api_router(self) -> Any | None:
        """Optionally return a FastAPI ``APIRouter`` to auto-mount in ``saes.server.app``."""
        return None

    def get_namespace(self) -> Any:
        """Return the Python module or API object exposed to ``saes.plugins.get_namespace(id)``."""
        return self


def _iter_entry_points(group: str) -> list[importlib_metadata.EntryPoint]:
    """Return discovered ``importlib.metadata.EntryPoint`` objects for ``group``."""
    return list(importlib_metadata.entry_points(group=group))


class PluginRegistry:
    """Central discovery and lifecycle registry for SAES Python plugins and entry points."""

    _plugins: dict[str, SAESPlugin] = {}
    _discovered: bool = False
    _tests_discovered: bool = False
    _renderers_discovered: bool = False
    _plotters_discovered: bool = False

    @classmethod
    def _resolve_manifest(cls, plugin: SAESPlugin) -> PluginManifest:
        raw = plugin.manifest
        plugin_id = raw.id.strip().lower()
        has_router = False
        try:
            has_router = plugin.get_api_router() is not None
        except Exception:
            has_router = False
        prefix = raw.api_prefix
        if has_router and not prefix:
            prefix = f"/api/v1/plugins/{plugin_id}"
        return raw.model_copy(update={"id": plugin_id, "api_prefix": prefix})

    @classmethod
    def register(cls, plugin: SAESPlugin | type[SAESPlugin]) -> PluginManifest:
        """Register a ``SAESPlugin`` instance or subclass and invoke its extension hooks."""
        instance: SAESPlugin
        if isinstance(plugin, type) and issubclass(plugin, SAESPlugin):
            instance = plugin()
        elif isinstance(plugin, SAESPlugin):
            instance = plugin
        else:
            raise ConfigurationError(
                f"Registered plugin must be a SAESPlugin instance or subclass, got {plugin!r}."
            )

        manifest = cls._resolve_manifest(instance)
        cls._plugins[manifest.id] = instance

        from saes.reports.renderers import TableRendererRegistry
        from saes.stats.registry import TestRegistry

        instance.register_tests(TestRegistry)
        instance.register_renderers(TableRendererRegistry)

        try:
            from saes.visualization.registry import PlotterRegistry

            instance.register_plotters(PlotterRegistry)
        except ImportError:
            # Optional [viz] extra not installed; skip plotter registration cleanly.
            pass

        return manifest

    @classmethod
    def unregister(cls, plugin_id: str) -> bool:
        """Remove a registered plugin by its lowercase id. Returns True if present."""
        return cls._plugins.pop(plugin_id.strip().lower(), None) is not None

    @classmethod
    def reset(cls) -> None:
        """Clear registered plugins and reset discovery flags (primarily for isolated tests)."""
        cls._plugins.clear()
        cls._discovered = False
        cls._tests_discovered = False
        cls._renderers_discovered = False
        cls._plotters_discovered = False

    @classmethod
    def discover_tests(cls, force: bool = False) -> list[str]:
        """Discover and register ``saes.tests`` entry points into ``TestRegistry``."""
        if cls._tests_discovered and not force:
            return []
        cls._tests_discovered = True

        from saes.stats.registry import TestRegistry

        loaded: list[str] = []
        for ep in _iter_entry_points(ENTRYPOINT_TESTS):
            try:
                test_cls = ep.load()
                TestRegistry.register(ep.name, test_cls)
                loaded.append(ep.name.strip().lower())
            except Exception as exc:
                warnings.warn(
                    f"Failed to load SAES test entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return loaded

    @classmethod
    def discover_renderers(cls, force: bool = False) -> list[str]:
        """Discover and register ``saes.renderers`` entry points into ``TableRendererRegistry``."""
        if cls._renderers_discovered and not force:
            return []
        cls._renderers_discovered = True

        from saes.reports.renderers import TableRendererRegistry

        loaded: list[str] = []
        for ep in _iter_entry_points(ENTRYPOINT_RENDERERS):
            try:
                renderer_cls = ep.load()
                TableRendererRegistry.register(ep.name, renderer_cls)
                loaded.append(ep.name.strip().lower())
            except Exception as exc:
                warnings.warn(
                    f"Failed to load SAES renderer entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return loaded

    @classmethod
    def discover_plotters(cls, force: bool = False) -> list[str]:
        """Discover and register ``saes.plotters`` entry points into ``PlotterRegistry``."""
        if cls._plotters_discovered and not force:
            return []
        cls._plotters_discovered = True

        try:
            from saes.visualization.registry import PlotterRegistry
        except ImportError:
            return []

        loaded: list[str] = []
        for ep in _iter_entry_points(ENTRYPOINT_PLOTTERS):
            try:
                plotter_cls = ep.load()
                PlotterRegistry.register(ep.name, plotter_cls)
                loaded.append(ep.name.strip().lower())
            except Exception as exc:
                warnings.warn(
                    f"Failed to load SAES plotter entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return loaded

    @classmethod
    def discover(cls, force: bool = False) -> list[PluginManifest]:
        """Discover all installed SAES entry points (tests, renderers, plotters, and domain plugins)."""
        if cls._discovered and not force:
            return cls.list_installed(discover=False)

        cls._discovered = True
        cls.discover_tests(force=force)
        cls.discover_renderers(force=force)
        cls.discover_plotters(force=force)

        for ep in _iter_entry_points(ENTRYPOINT_PLUGINS):
            try:
                loaded_obj = ep.load()
                cls.register(loaded_obj)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load SAES plugin entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cls.list_installed(discover=False)

    @classmethod
    def list_installed(cls, *, discover: bool = True) -> list[PluginManifest]:
        """Return sorted manifests of all active SAES plugins."""
        if discover and not cls._discovered:
            cls.discover(force=False)
        return [
            cls._resolve_manifest(cls._plugins[key])
            for key in sorted(cls._plugins.keys())
        ]

    @classmethod
    def is_installed(cls, plugin_id: str) -> bool:
        """Check whether a plugin with ``plugin_id`` is registered or installed."""
        if not cls._discovered:
            cls.discover(force=False)
        return plugin_id.strip().lower() in cls._plugins

    @classmethod
    def get(cls, plugin_id: str) -> SAESPlugin:
        """Retrieve a registered ``SAESPlugin`` instance by its identifier."""
        if not cls._discovered:
            cls.discover(force=False)
        key = plugin_id.strip().lower()
        if key not in cls._plugins:
            available = ", ".join(sorted(cls._plugins.keys())) or "none"
            raise ConfigurationError(
                f"SAES plugin '{plugin_id}' is not installed. Installed plugins: {available}."
            )
        return cls._plugins[key]

    @classmethod
    def get_namespace(cls, plugin_id: str) -> Any:
        """Return the Python API namespace exposed by ``plugin_id``."""
        return cls.get(plugin_id).get_namespace()

    @classmethod
    def mount_api_routers(cls, app: Any) -> list[PluginManifest]:
        """Discover installed plugins and mount any optional FastAPI routers onto ``app``."""
        cls.discover(force=False)
        mounted_ids: set[str] = getattr(app.state, "saes_mounted_plugins", set())
        mounted: list[PluginManifest] = []

        for key in sorted(cls._plugins.keys()):
            plugin = cls._plugins[key]
            manifest = cls._resolve_manifest(plugin)
            if key in mounted_ids:
                mounted.append(manifest)
                continue
            try:
                router = plugin.get_api_router()
            except Exception as exc:
                warnings.warn(
                    f"Failed to obtain FastAPI router for plugin '{key}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            if router is not None:
                prefix = manifest.api_prefix or f"/api/v1/plugins/{key}"
                app.include_router(router, prefix=prefix, tags=[f"Plugin: {manifest.name}"])
                mounted_ids.add(key)
                mounted.append(manifest.model_copy(update={"api_prefix": prefix}))

        app.state.saes_mounted_plugins = mounted_ids
        return mounted
