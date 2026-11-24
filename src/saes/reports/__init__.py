"""Reproducibility tools, FAIR environment manifests, and extensible table renderers."""

from saes.reports.renderers import (
    PDF_THEMES,
    TableRenderer,
    TableRendererRegistry,
    export_flat_table,
    export_statistical_table,
    render_latex_table,
    render_markdown_table,
)
from saes.reports.reproducibility import (
    generate_environment_manifest,
    generate_jupyter_notebook,
    generate_python_script,
    generate_reproducibility_artifacts,
)

__all__ = [
    "PDF_THEMES",
    "TableRenderer",
    "TableRendererRegistry",
    "export_flat_table",
    "export_statistical_table",
    "render_latex_table",
    "render_markdown_table",
    "generate_environment_manifest",
    "generate_jupyter_notebook",
    "generate_python_script",
    "generate_reproducibility_artifacts",
]
