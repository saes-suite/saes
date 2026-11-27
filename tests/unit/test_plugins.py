"""Unit tests for SAES PEP-standard plugin discovery and FastAPI auto-mounting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

import saes
from saes.domain.exceptions import ConfigurationError
from saes.domain.models import StatisticalResult, StatisticalTableResult
from saes.plugins.registry import (
    ENTRYPOINT_PLOTTERS,
    ENTRYPOINT_PLUGINS,
    ENTRYPOINT_RENDERERS,
    ENTRYPOINT_TESTS,
    PluginManifest,
    PluginRegistry,
    SAESPlugin,
)
from saes.reports.renderers import TableRenderer, TableRendererRegistry
from saes.server.app import create_app
from saes.stats.base import StatisticalTest
from saes.stats.registry import TestRegistry
from saes.visualization.base import Plotter
from saes.visualization.registry import PlotterRegistry


@dataclass
class _FakeEntryPoint:
    name: str
    obj: Any
    should_fail: bool = False

    def load(self) -> Any:
        if self.should_fail:
            raise RuntimeError("boom on load")
        return self.obj


class _DummyTest(StatisticalTest):
    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        return StatisticalResult(
            test_name="Dummy Plugin Test",
            statistic=1.23,
            p_value=0.04,
            alpha=alpha,
            is_significant=0.04 < alpha,
            interpretation="dummy",
        )


class _DummyRenderer(TableRenderer):
    media_type = "text/plain"
    extension = "txt"

    def render(self, result: StatisticalTableResult, **kwargs: Any) -> bytes:
        _ = (result, kwargs)
        return b"dummy-table"


class _DummyPlotter(Plotter):
    @property
    def name(self) -> str:
        return "Dummy Plot"

    @property
    def description(self) -> str:
        return "Dummy plugin plotter"

    def plot(self, df: Any, **kwargs: Any) -> Any:
        _ = (df, kwargs)
        return None

    def plot_interactive(self, df: Any, **kwargs: Any) -> Any:
        _ = (df, kwargs)
        return None


class _SampleDomainPlugin(SAESPlugin):
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            id="demo_ext",
            name="Demo Extension",
            version="0.1.0",
            description="Sample domain plugin for testing",
            capabilities=("demo_ping",),
        )

    def register_tests(self, registry: type[TestRegistry]) -> None:
        registry.register("plugin_demo_test", _DummyTest)

    def register_renderers(self, registry: type[TableRendererRegistry]) -> None:
        registry.register("plugin_demo_txt", _DummyRenderer)

    def register_plotters(self, registry: type[PlotterRegistry]) -> None:
        registry.register("plugin-demo-plot", _DummyPlotter)

    def get_api_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/ping")
        def ping() -> dict[str, str]:
            return {"plugin": "demo_ext", "status": "pong"}

        return router

    def get_namespace(self) -> dict[str, str]:
        return {"hello": "world"}


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Snapshot and restore global registries so plugin tests stay isolated."""
    orig_tests = dict(TestRegistry._registry)
    orig_renderers = dict(TableRendererRegistry._registry)
    orig_plotters = dict(PlotterRegistry._registry)
    orig_plugins = dict(PluginRegistry._plugins)
    orig_flags = (
        PluginRegistry._discovered,
        PluginRegistry._tests_discovered,
        PluginRegistry._renderers_discovered,
        PluginRegistry._plotters_discovered,
    )
    PluginRegistry.reset()
    try:
        yield
    finally:
        TestRegistry._registry = orig_tests
        TableRendererRegistry._registry = orig_renderers
        PlotterRegistry._registry = orig_plotters
        PluginRegistry._plugins = orig_plugins
        (
            PluginRegistry._discovered,
            PluginRegistry._tests_discovered,
            PluginRegistry._renderers_discovered,
            PluginRegistry._plotters_discovered,
        ) = orig_flags


def test_top_level_exports_and_manual_registration() -> None:
    assert saes.plugins.PluginRegistry is PluginRegistry
    assert saes.SAESPlugin is SAESPlugin
    assert saes.PluginManifest is PluginManifest

    manifest = PluginRegistry.register(_SampleDomainPlugin)
    assert manifest.id == "demo_ext"
    assert manifest.api_prefix == "/api/v1/plugins/demo_ext"
    assert PluginRegistry.is_installed("demo_ext")
    assert PluginRegistry.get_namespace("demo_ext") == {"hello": "world"}
    assert "plugin_demo_test" in TestRegistry.list_available()
    assert "plugin_demo_txt" in TableRendererRegistry.list_available()
    assert "plugin-demo-plot" in PlotterRegistry.list_available()

    assert PluginRegistry.unregister("demo_ext") is True
    assert PluginRegistry.unregister("demo_ext") is False

    with pytest.raises(ConfigurationError):
        PluginRegistry.register("not-a-plugin")  # type: ignore[arg-type]


def test_entry_point_discovery_across_all_four_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_groups: dict[str, list[_FakeEntryPoint]] = {
        ENTRYPOINT_TESTS: [_FakeEntryPoint("ep_test", _DummyTest)],
        ENTRYPOINT_RENDERERS: [_FakeEntryPoint("ep_txt", _DummyRenderer)],
        ENTRYPOINT_PLOTTERS: [_FakeEntryPoint("ep-plot", _DummyPlotter)],
        ENTRYPOINT_PLUGINS: [_FakeEntryPoint("demo_ext", _SampleDomainPlugin)],
    }

    monkeypatch.setattr(
        "saes.plugins.registry._iter_entry_points",
        lambda group: fake_groups.get(group, []),
    )

    manifests = PluginRegistry.discover(force=True)
    assert [m.id for m in manifests] == ["demo_ext"]
    assert "ep_test" in TestRegistry.list_available()
    assert "ep_txt" in TableRendererRegistry.list_available()
    assert "ep-plot" in PlotterRegistry.list_available()
    assert "plugin_demo_test" in TestRegistry.list_available()

    with pytest.raises(ConfigurationError):
        PluginRegistry.get("nonexistent")


def test_fault_isolation_on_broken_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_groups: dict[str, list[_FakeEntryPoint]] = {
        ENTRYPOINT_TESTS: [_FakeEntryPoint("broken_test", None, should_fail=True)],
        ENTRYPOINT_RENDERERS: [_FakeEntryPoint("broken_renderer", None, should_fail=True)],
        ENTRYPOINT_PLOTTERS: [_FakeEntryPoint("broken_plotter", None, should_fail=True)],
        ENTRYPOINT_PLUGINS: [
            _FakeEntryPoint("broken_plugin", None, should_fail=True),
            _FakeEntryPoint("demo_ext", _SampleDomainPlugin),
        ],
    }

    monkeypatch.setattr(
        "saes.plugins.registry._iter_entry_points",
        lambda group: fake_groups.get(group, []),
    )

    with pytest.warns(RuntimeWarning):
        manifests = PluginRegistry.discover(force=True)

    assert [m.id for m in manifests] == ["demo_ext"]


def test_fastapi_plugin_discovery_and_auto_mounting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "saes.plugins.registry._iter_entry_points",
        lambda group: [_FakeEntryPoint("demo_ext", _SampleDomainPlugin)] if group == ENTRYPOINT_PLUGINS else [],
    )

    app = create_app()
    client = TestClient(app)

    res = client.get("/api/v1/plugins")
    assert res.status_code == 200
    payload = res.json()
    assert len(payload["plugins"]) == 1
    assert payload["plugins"][0]["id"] == "demo_ext"
    assert payload["plugins"][0]["api_prefix"] == "/api/v1/plugins/demo_ext"
    assert "plugin_demo_test" in payload["registered_tests"]
    assert "plugin_demo_txt" in payload["registered_renderers"]
    assert "plugin-demo-plot" in payload["registered_plotters"]

    ping_res = client.get("/api/v1/plugins/demo_ext/ping")
    assert ping_res.status_code == 200
    assert ping_res.json() == {"plugin": "demo_ext", "status": "pong"}
