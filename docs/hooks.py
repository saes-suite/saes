"""MkDocs native build hooks for automatic metadata, changelog, and API introspection.

Automatically extracts live information from the saes codebase at build time so
GitHub Pages documentation never drifts from version.txt, CHANGELOG.md, pyproject.toml,
registered plugins (PlotterRegistry, TestRegistry, TableRendererRegistry), or FastAPI routes.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _read_version() -> str:
    version_file = REPO_ROOT / "version.txt"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "0.0.0"


def _read_pyproject() -> dict[str, Any]:
    pyproject_file = REPO_ROOT / "pyproject.toml"
    with pyproject_file.open("rb") as f:
        return tomllib.load(f)


def _read_changelog_body() -> str:
    changelog_file = REPO_ROOT / "CHANGELOG.md"
    if not changelog_file.exists():
        return "_No changelog available._"
    lines = changelog_file.read_text(encoding="utf-8").splitlines()
    # Drop top-level '# Changelog' heading since docs/changelog.md provides it
    if lines and lines[0].strip().startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _build_extras_table(pyproject: dict[str, Any]) -> str:
    project = pyproject.get("project", {})
    core_deps = ", ".join(f"`{d.split('>=')[0].split('<')[0].strip()}`" for d in project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})

    purposes = {
        "viz": "Publication figures (`saes.plot`, `saes.export_plot`) & PDF tables",
        "api": "Optional typed FastAPI REST server (`saes-api` on `:8060`)",
        "docs": "Building and previewing this Material for MkDocs site locally",
        "dev": "Testing (`pytest`), linting (`ruff`), strict typing (`mypy`), viz + api",
        "all": "Full runtime bundle (`viz` + `api`)",
    }

    rows = [
        "| Extra | Packages Included | Purpose |",
        "| :--- | :--- | :--- |",
        f"| *(core)* | {core_deps} | All statistical tests, Bayesian models, MDES, TOST, tables & FAIR packs |",
    ]
    for name, pkgs in extras.items():
        pkg_list = ", ".join(f"`{p}`" for p in pkgs)
        purpose = purposes.get(name, f"Optional `{name}` dependencies")
        rows.append(f"| `[{name}]` | {pkg_list} | {purpose} |")
    return "\n".join(rows)


def _build_plotters_table() -> str:
    from saes.visualization.registry import PlotterRegistry

    available = PlotterRegistry.list_available()
    rows = [
        "| Plotter Key | Name | Description |",
        "| :--- | :--- | :--- |",
    ]
    for key, meta in sorted(available.items()):
        name = str(meta.get("name", key)).replace("|", "\\|")
        desc = str(meta.get("description", "")).replace("|", "\\|")
        rows.append(f"| `{key}` | {name} | {desc} |")
    return "\n".join(rows)


def _build_renderers_and_tests_summary() -> dict[str, Any]:
    from saes.reports.renderers import TableRendererRegistry
    from saes.stats.registry import TestRegistry
    from saes.visualization.registry import PlotterRegistry

    plotters = PlotterRegistry.list_available()
    tests = TestRegistry.list_available()
    renderers = TableRendererRegistry.list_available()
    return {
        "plotters_count": len(plotters),
        "tests_count": len(tests),
        "renderers_count": len(renderers),
        "renderers_list": ", ".join(f"`{r}`" for r in sorted(renderers)),
    }


def _build_rest_endpoints_table() -> str:
    try:
        from saes.server.app import app

        schema = app.openapi()
    except Exception:
        return "_Install `saes[api]` to introspect live FastAPI endpoints._"

    paths: dict[str, dict[str, Any]] = schema.get("paths", {})
    rows = [
        "| Path | Methods | Description |",
        "| :--- | :--- | :--- |",
    ]
    for path in sorted(paths.keys()):
        ops = paths[path]
        methods = ", ".join(sorted(m.upper() for m in ops if m.lower() not in {"parameters", "head", "options"}))
        desc = ""
        for op in ops.values():
            if isinstance(op, dict):
                raw = str(op.get("description") or op.get("summary") or "").strip()
                if raw:
                    desc = raw.splitlines()[0].replace("|", "\\|")
                    break
        rows.append(f"| `{path}` | `{methods}` | {desc} |")
    return "\n".join(rows)


_CACHE: dict[str, str] = {}


def on_config(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Populate cached live metadata once per MkDocs build."""
    version = _read_version()
    pyproject = _read_pyproject()
    project = pyproject.get("project", {})
    counts = _build_renderers_and_tests_summary()

    _CACHE.clear()
    _CACHE.update(
        {
            "saes_version": version,
            "saes_python_requires": str(project.get("requires-python", ">=3.11")),
            "saes_license": str(project.get("license", {}).get("text", "GPL-3.0-or-later")),
            "saes_plotters_count": str(counts["plotters_count"]),
            "saes_tests_count": str(counts["tests_count"]),
            "saes_renderers_count": str(counts["renderers_count"]),
            "saes_renderers_list": str(counts["renderers_list"]),
            "saes_changelog": _read_changelog_body(),
            "saes_auto_extras_table": _build_extras_table(pyproject),
            "saes_auto_plotters_table": _build_plotters_table(),
            "saes_auto_rest_endpoints_table": _build_rest_endpoints_table(),
        }
    )

    config.setdefault("extra", {})
    config["extra"]["version"] = version
    return config


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Replace dynamic {{ saes_* }} placeholders with live introspected content."""
    if "{{ saes_" not in markdown:
        return markdown
    for key, value in _CACHE.items():
        token = f"{{{{ {key} }}}}"
        if token in markdown:
            markdown = markdown.replace(token, value)
    return markdown
