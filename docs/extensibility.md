# Extensibility Guide

SAES separates statistical computation, presentation rendering, and visualization behind three extensible registries inspired by R's S3/S4 dispatch and `texreg` (Leifeld 2013):

1. `saes.TestRegistry` — Register custom `StatisticalTest` implementations.
2. `saes.TableRendererRegistry` — Register custom `TableRenderer` output formats (e.g., Typst, Org-mode, custom journal LaTeX templates).
3. `saes.PlotterRegistry` — Register custom `Plotter` visualizations (requires `saes[viz]`).

## 1. Registering a Custom Statistical Test

Subclass `saes.StatisticalTest`, implement `run(groups, alpha=0.05) -> StatisticalResult`, and register it:

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

## 2. Registering a Custom Table Renderer (e.g., Typst)

Subclass `saes.TableRenderer` and register it in `saes.TableRendererRegistry`. Once registered, `saes.export_table(table, export_format="typst")` dispatches to it automatically:

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

## 3. Registering a Custom Plotter

With `saes[viz]` installed, subclass `saes.visualization.base.Plotter` and register it in `saes.PlotterRegistry` so it becomes immediately available through `saes.plot(dataset, plot_type="my_plot")` and `saes.export_plot(...)`.
