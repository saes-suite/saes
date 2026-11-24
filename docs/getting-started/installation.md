# Installation

SAES (`saes`) requires **Python 3.11 or newer** and keeps its core scientific dependency footprint lean (`numpy`, `pandas`, `pydantic>=2`, `scipy`, `statsmodels`). Visualization, REST API serving, and documentation tooling are cleanly separated into optional extras.

## Installing Released Packages

=== "pip"

    ```bash
    # Core statistical engine only (headless CI / HPC clusters)
    pip install saes

    # Add publication plotting (Matplotlib, Seaborn, Plotly)
    pip install "saes[viz]"

    # Add optional FastAPI REST API server
    pip install "saes[api]"

    # Install everything (Core + Visualization + REST API)
    pip install "saes[all]"
    ```

=== "uv"

    ```bash
    uv add saes
    uv add "saes[viz]"
    uv add "saes[all]"
    ```

!!! note "Upgrading from SAES v1 (`1.x`)"
    Version **2.0.0** consolidates the rewritten statistical suite under the canonical `saes` PyPI package (`pip install saes`, `import saes`). Legacy v1 workflows can continue using the previous series by pinning `pip install "SAES<2"`.

## Optional Dependency Extras

*(Automatically extracted from `pyproject.toml` for `saes v{{ saes_version }}`, requiring Python `{{ saes_python_requires }}`.)*

{{ saes_auto_extras_table }}

## Installing from Source (Development)

```bash
git clone https://github.com/saes-suite/saes.git
cd saes
uv sync --extra dev --extra docs
```

Verify your installation by running the full quality gate:

```bash
make ci
```

## Building the Documentation Locally

```bash
# Live-reload preview at http://127.0.0.1:8000
make docs-serve

# Strict static production build (matches GitHub Pages CI)
make docs
```
