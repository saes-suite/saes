# Quickstart Guide

This guide gets you from a raw benchmark CSV to validated statistical conclusions, publication-ready LaTeX tables, vector figures, and a FAIR reproducibility script in under five minutes.

## 1. Required Tidy Long-Format CSV Schema

SAES validates every benchmark dataset against a strict six-column tidy schema:

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `algorithm` | `str` | `NSGAII` | Algorithm, solver, or model identifier |
| `instance` | `str` | `DTLZ2` | Benchmark problem instance or dataset |
| `metric` | `str` | `HV` | Quality indicator or evaluation metric |
| `run_id` | `int` | `0..29` | Stochastic run index or CV fold index |
| `value` | `float` | `0.4281` | Observed scalar measurement |
| `maximize` | `bool` | `True` | `True` if higher values are better, `False` otherwise |

!!! tip "Bundled Benchmark Datasets"
    `saes` ships with two real benchmark datasets under `datasets/`:

    - `datasets/swarmintelligence.csv` — 7 multi-objective optimizers × 12 DTLZ/ZDT instances × 7 quality indicators.
    - `datasets/ml_data.csv` — 4 ML classifiers (`KNN`, `MLP`, `RandomForest`, `SVM`) across accuracy and loss.

## 2. End-to-End Python Workflow

```python
import saes

# 1. Load & validate
ds = saes.load("datasets/swarmintelligence.csv")
val = saes.validate(ds)
assert val.is_valid, f"Validation failed: {val.issues}"

# 2. Explore assumptions & dataset-aware recommendations
exp = saes.explore(ds, metrics=["HV"], alpha=0.05)
print(f"Design: k={exp.context_diagnostics.k_algorithms} algos, "
      f"N={exp.context_diagnostics.n_instances} instances, "
      f"n={exp.context_diagnostics.runs_per_cell} runs/cell")
for rec in exp.recommendations[:3]:
    print(f"  [{rec.suitability.upper()}] {rec.test_name} ({rec.suitability_score}/100)")

# 3. Automated cell-level analysis (independent runs by default!)
report = saes.analyze(ds, alpha=0.05, generate_plots=False, paired=False)
print(report.to_dataframe()[["instance", "metric", "test_name", "p_value", "effect_size_value"]].head())

# 4. Multi-instance blocked omnibus tests
cd = saes.statistics.friedman_nemenyi(ds, metric="HV", alpha=0.05)
aligned = saes.statistics.friedman_aligned_ranks(ds, metric="HV", alpha=0.05)
quade = saes.statistics.quade(ds, metric="HV", alpha=0.05)
print(cd.summary())
print(aligned.summary())
print(quade.summary())

# 5. Exact Bayesian ranking (Plackett-Luce & Signed-Rank ROPE)
pl = saes.statistics.bayesian_plackett_luce(ds, metric="HV", num_samples=400)
bayes = saes.statistics.bayesian_signed_rank(ds, metric="HV", rope=0.01)
print(pl.summary())
print(bayes.summary())

# 6. Export publication-ready LaTeX table & vector SVG figure
tbl = saes.generate_table(ds, metric="HV", posthoc_correction="holm", paired=False)
latex_bytes = saes.export_table(tbl, export_format="latex")
svg_bytes, _, _ = saes.export_plot(ds, plot_type="plackett-luce", metric="HV", export_format="svg")
```

## 3. Uniform Result Contract (`.summary()` & `.to_dataframe()`)

Every primary result object in SAES is an immutable **Pydantic V2** model supporting:

- `.summary() -> str` — concise, publication-oriented statistical summary for logs and notebooks.
- `.to_dataframe() -> pd.DataFrame` — tidy tabular export ready for downstream Pandas/Seaborn analysis.
- `.model_dump()` / `.model_dump_json()` — lossless typed JSON serialization.

## 4. Explicit Pairing Safeguard (`paired=False` Default)

!!! warning "Independent Stochastic Runs vs. Matched Cross-Validation Folds"
    In evolutionary computation and stochastic optimization, `run_id` indexes **independent random restarts**. Treating independent runs as paired artificially shrinks variance and inflates Type I error.

    SAES therefore defaults to **`paired=False`** across `analyze()`, `generate_table()`, `profiles()`, `power()`, `tost()`, and `effect_size_forest()`. Pass **`paired=True`** explicitly **only** when `run_id` represents genuinely matched experimental blocks (e.g., identical $k$-fold cross-validation splits or Common Random Numbers).
