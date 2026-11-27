# Extensibility & Python Plugins

SAES separates statistical computation, presentation rendering, visualization, and domain-specific extensions behind **four PEP-standard `importlib.metadata.entry_points` extension groups**, inspired by R's S3/S4 dispatch, `texreg` (Leifeld 2013), and the Python scientific plugin ecosystem (`pytest`, `mkdocs`, `napari`):

| Extension Mechanism | Runtime Registry | Python `entry_points` Group | Purpose |
| :--- | :--- | :--- | :--- |
| Hypothesis Tests | `saes.TestRegistry` | `"saes.tests"` | Register custom `saes.StatisticalTest` implementations |
| Table Renderers | `saes.TableRendererRegistry` | `"saes.renderers"` | Register custom `saes.TableRenderer` output formats (Typst, Org, journal templates) |
| Visualizations | `saes.PlotterRegistry` | `"saes.plotters"` | Register custom `Plotter` classes (requires `saes[viz]`) |
| **Domain Plugins** | **`saes.PluginRegistry` (`saes.plugins`)** | **`"saes.plugins"`** | Package domain-specific workflows (`saes.SAESPlugin`) with optional FastAPI routes auto-mounted under `/api/v1/plugins/<id>` |

> **Note:** SAES Python runtime plugins (built-in under `saes.plugins.*` or installed externally via `entry_points`) are distinct from the `saes-plugins` repository, which distributes Markdown `SKILL.md` guides for AI coding assistants.

---

## 1. Registering a Custom Statistical Test

Subclass `saes.StatisticalTest`, implement `run(groups, alpha=0.05) -> StatisticalResult`, and register it either at runtime or via `pyproject.toml`:

```python
import numpy as np
from scipy import stats
import saes
from saes.domain.models import EffectSize, StatisticalResult


class PermutationMeanTest(saes.StatisticalTest):
    """Exact/Monte-Carlo two-sample permutation test for difference in means."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        g1, g2 = list(groups.keys())[:2]
        x1, x2 = groups[g1], groups[g2]
        res = stats.permutation_test(
            (x1, x2),
            statistic=lambda a, b: np.mean(a) - np.mean(b),
            n_resamples=2000,
            random_state=42,
        )
        diff = float(res.statistic)
        p_val = float(res.pvalue)
        return StatisticalResult(
            test_name="Two-Sample Permutation Test",
            statistic=diff,
            p_value=p_val,
            degrees_of_freedom=float(len(x1) + len(x2) - 2),
            alpha=alpha,
            is_significant=p_val < alpha,
            effect_size=EffectSize(name="Mean Difference", value=diff),
            interpretation=f"Permutation test ({g1} vs {g2}): diff={diff:.4f}, p={p_val:.4e}.",
        )


saes.TestRegistry.register("permutation_mean", PermutationMeanTest)

# Instantiate from the registry or pass directly into AnalysisPipeline(force_test=...)
test = saes.TestRegistry.get("permutation_mean")
result = test.run({"A": np.array([1.0, 2.0, 3.0]), "B": np.array([4.0, 5.0, 6.0])})
print(result.summary())
```

To publish the test as an auto-discovered package, declare the entry point in `pyproject.toml`:

```toml
[project.entry-points."saes.tests"]
permutation_mean = "my_pkg.tests:PermutationMeanTest"
```

---

## 2. Registering a Custom Table Renderer (e.g., Typst)

Subclass `saes.TableRenderer` and register it in `saes.TableRendererRegistry` (or declare it under `[project.entry-points."saes.renderers"]`). Once registered, `saes.export_table(table, export_format="typst")` dispatches to it automatically:

```python
import saes
from saes.domain.models import StatisticalTableResult


class TypstTableRenderer(saes.TableRenderer):
    media_type = "text/plain"
    extension = "typ"

    def render(self, result: StatisticalTableResult, **kwargs: object) -> bytes:
        cols = len(result.algorithms) + 1
        header = ", ".join(f"[*{h}*]" for h in ["Instance", *result.algorithms])
        body = []
        for row in result.instance_rows:
            vals = [f"[{row.instance}]"] + [
                f"[{row.cells[alg].formatted}]" for alg in result.algorithms
            ]
            body.append("  " + ", ".join(vals) + ",")
        typst = f"#table(\n  columns: {cols},\n  {header},\n" + "\n".join(body) + "\n)\n"
        return typst.encode("utf-8")


saes.TableRendererRegistry.register("typst", TypstTableRenderer)

dataset = saes.load("datasets/swarmintelligence.csv")
table = saes.generate_table(dataset, metric="HV")
typst_bytes = saes.export_table(table, export_format="typst")
```

---

## 3. Registering a Custom Plotter

With `saes[viz]` installed, subclass `saes.visualization.base.Plotter` and register it in `saes.PlotterRegistry` (or expose it under `[project.entry-points."saes.plotters"]`) so it becomes immediately available through `saes.plot(dataset, plot_type="my_plot")` and `saes.export_plot(...)`.

---

## 4. Authoring a Full Domain Plugin (`saes.SAESPlugin`)

Domain-specific data types (such as multi-objective Pareto fronts, anytime convergence trajectories, or ML cross-validation runners) can extend SAES **without adding dependencies or domain coupling to `saes` core** by implementing `saes.SAESPlugin`:

```python
from saes import PluginManifest, SAESPlugin


class MyDomainPlugin(SAESPlugin):
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            id="my_domain",
            name="SAES My Domain",
            version="0.1.0",
            description="Domain-specific analytics bridging into canonical saes.Dataset workflows.",
            capabilities=("custom_analytics",),
        )

    def get_api_router(self):
        # Optional FastAPI APIRouter auto-mounted at /api/v1/plugins/my_domain
        return None
```

Register the package entry point in `pyproject.toml`:

```toml
[project.entry-points."saes.plugins"]
my_domain = "my_domain_pkg.plugin:MyDomainPlugin"
```

Once installed, `saes` discovers the plugin automatically:

```python
import saes

print(saes.plugins.list_installed())
plugin_api = saes.plugins.get_namespace("my_domain")
```

When running the optional REST server (`saes-api`), `GET /api/v1/plugins` advertises all discovered plugins and `create_app()` automatically mounts every plugin's optional `APIRouter` under `/api/v1/plugins/<plugin_id>`.

### Built-in Reference Plugin: `saes.plugins.pareto`

> See the dedicated [Pareto Fronts & EAF User Guide](user-guide/pareto-plugin.md) for complete input dataset specifications (multi-file `FUN.*.tsv`, tidy CSV, `.pf` reference fronts, jMetal ZIP), mathematical definitions, and REST endpoints.

To dogfood the `SAESPlugin` protocol in a real scientific domain without coupling `saes.stats` or `Dataset` to multi-objective optimization, SAES ships **`saes.plugins.pareto`** registered via the exact same `[project.entry-points."saes.plugins"]` mechanism available to third-party packages:

- Loads multi-run Pareto fronts and reference fronts (tidy CSV, jMetal/PlatEMO `FUN.*.tsv` directories, `.pf` files, ZIP bundles).
- Computes exact 2D **Empirical Attainment Functions (EAF)**, median/IQR attainment surfaces, spatial $\Delta\text{EAF}$ difference surfaces, KS-EAF permutation hypothesis tests, and set coverage $C(A, B)$.
- Exports publication figures (PDF/SVG/PNG) and interactive Plotly charts (`attainment`, `runs`, `representative`, `eaf-diff`, `small-multiples`, `3d`, `pcp`).
- Bridges back to `saes` core via `fronts.to_indicators_dataset(indicators=["HV", "IGD+", "EPS"]) -> saes.Dataset`.

```python
import saes
from saes.plugins import pareto

fronts = pareto.create_demo_collection(n_runs=12, seed=42)
diff = pareto.eaf_difference(fronts, instance="ZDT1", algo_a="NSGA-II", algo_b="MOEA/D")
print(diff.summary())

dataset = fronts.to_indicators_dataset(indicators=["HV", "IGD+", "EPS"])
print(saes.statistics.friedman_nemenyi(dataset, metric="IGD+").summary())
```
