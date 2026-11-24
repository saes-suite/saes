<p align="center">
  <img src="https://raw.githubusercontent.com/saes-suite/saes/main/assets/saes-logo.svg" alt="SAES Logo" width="128" />
</p>

<h1 align="center">SAES — Stochastic Algorithm Evaluation Suite (<code>saes</code>)</h1>

<p align="center">
  <a href="https://github.com/saes-suite/saes/actions/workflows/tests.yml"><img src="https://github.com/saes-suite/saes/actions/workflows/tests.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/saes/"><img src="https://img.shields.io/pypi/v/saes" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/saes/"><img src="https://img.shields.io/pypi/pyversions/saes" alt="Python versions" /></a>
  <a href="https://github.com/saes-suite/saes/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License: GPLv3" /></a>
  <a href="https://saes-suite.github.io/saes/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-teal.svg" alt="Documentation" /></a>
</p>

<p align="center">
  Open-source scientific Python library for rigorous <strong>frequentist</strong>, <strong>exact Bayesian</strong>, <strong>equivalence (TOST)</strong>, <strong>power/sensitivity (MDES)</strong>, and <strong>Dolan–Moré meta-benchmarking</strong> evaluation of stochastic algorithms.
</p>

<p align="center">
  <a href="https://saes-suite.github.io/saes/"><strong>📚 Documentation (GitHub Pages)</strong></a> ·
  <a href="https://saes-suite.github.io/saes/getting-started/quickstart/"><strong>🚀 Quickstart</strong></a> ·
  <a href="https://saes-suite.github.io/saes/methodology/"><strong>📐 Statistical Methodology</strong></a> ·
  <a href="https://saes-suite.github.io/saes/api-reference/"><strong>🐍 Python API</strong></a>
</p>

---

## ✨ Key Capabilities

- **Assumption-Driven Diagnostics & Safeguards**: Shapiro–Wilk, Lilliefors KS, Anderson–Darling, D'Agostino–Pearson $K^2$, and Brown–Forsythe Levene checks; explicit `paired=False` default for independent stochastic runs (preventing silent pairing inflation); automatic blocking guards against multi-instance pseudoreplication.
- **Two-Sample, Multi-Group & Blocked Omnibus Tests**: Student/Welch/Paired $t$-tests, Fisher & Welch ANOVA, Mann–Whitney $U$, Wilcoxon Signed-Rank, **Brunner–Munzel**, Kruskal–Wallis, tie-corrected **Friedman + Iman–Davenport + Nemenyi CD**, **Friedman Aligned-Ranks**, **Quade**, and **Repeated-Measures ANOVA** (Greenhouse–Geisser $\varepsilon$).
- **Post-Hoc Suite & 13 Multiplicity Adjustments**: **Tukey HSD**, **Games–Howell**, **Dunnett**, **Dunn**, and **Conover–Iman** alongside `holm`, `bonferroni`, `hochberg`, `hommel`, `finner`, `holland`, `rom`, `li`, `fdr_bh`, `fdr_by`, **Shaffer**, and exact **Bergmann–Hommel**.
- **Exact Effect-Size Confidence Intervals & Forest Analysis**: Noncentral-$t$ CIs for Cohen's $d$ / Hedges' $g$, noncentral-$F$ CIs for $\eta^2$, Brunner–Munzel logit CIs for Vargha–Delaney $A_{12}$ and Cliff's $\delta$, bootstrap CIs for $\varepsilon^2$, and `saes.statistics.effect_size_forest(...)`.
- **Exact Bayesian Inference, MDES & TOST**: Benavoli et al. (2017) **Bayesian Sign & Signed-Rank Simplex** (relative or exact absolute ROPE + configurable Dirichlet prior), **Bayesian Correlated $t$-test** (Nadeau & Bengio 2003), **Bayesian Plackett–Luce** multi-algorithm ranking (Calvo et al. 2018), a priori power & **MDES sensitivity audits**, and **Schuirmann TOST equivalence tests**.
- **10 Publication Plotters, Camera-Ready Tables & FAIR Packs**: Static (SVG/PDF/300-DPI PNG) & interactive Plotly figures (`boxplot`, `violin`, `ecdf`, `barplot`, `lineplot`, `scatter`, `critical-distance`, `bayesian-rank`, `plackett-luce`, `effect-size-forest`), LaTeX `booktabs`/PDF/DOCX/CSV/Markdown/HTML tables, Dolan–Moré performance profiles, and SHA-256 FAIR reproducibility packs.

---

## 📦 Installation

```bash
# Core statistical library (NumPy, Pandas, Pydantic V2, SciPy, Statsmodels)
pip install saes

# With publication visualization support (Matplotlib, Seaborn, Plotly)
pip install "saes[viz]"

# With optional FastAPI REST API server (:8060)
pip install "saes[api]"

# Full installation (viz + api)
pip install "saes[all]"
```

> **Upgrading from SAES v1 (`1.x`)?** Version `2.0.0` consolidates the new architecture under the canonical `saes` PyPI package (`import saes`). If you maintain legacy v1 scripts, pin `pip install "SAES<2"`.

---

## 🚀 Quick Usage

### Python Programmatic API

```python
import saes

# 1. Load and strictly validate a tidy 6-column benchmark CSV
dataset = saes.load("datasets/swarmintelligence.csv")
assert saes.validate(dataset).is_valid

# 2. Explore distributions, normality, and dataset-aware test recommendations
exploration = saes.explore(dataset, alpha=0.05)
print("Top recommendation:", exploration.recommendations[0].test_name)

# 3. Automated cell-level analysis (independent runs by default; set paired=True for matched folds/CRN)
report = saes.analyze(dataset, alpha=0.05, generate_plots=False, paired=False)
print(report.to_dataframe().head())

# 4. Multi-instance blocked frequentist & Bayesian ranking
cd = saes.statistics.friedman_nemenyi(dataset, metric="HV", alpha=0.05)
pl = saes.statistics.bayesian_plackett_luce(dataset, metric="HV", num_samples=400)
print(cd.summary())
print(pl.summary())

# 5. Standardized effect-size forest (95% CIs) & standalone LaTeX table
forest = saes.statistics.effect_size_forest(dataset, metric="HV", instance="DTLZ2")
table = saes.generate_table(dataset, metric="HV", posthoc_correction="holm", paired=False)
latex_bytes = saes.export_table(table, export_format="latex")
```

### Launch Optional REST API & Local Documentation

```bash
# Launch FastAPI REST server on http://127.0.0.1:8060 (Swagger at /docs)
make start-api

# Preview full Material for MkDocs documentation locally at http://127.0.0.1:8000
make start-docs      # live reload
make docs-preview    # exact strict GitHub Pages static build
```

---

## 🛠️ Repository Architecture

- `saes.domain`: Immutable Pydantic V2 models (`Dataset`, `StatisticalResult`, `PostHocResult`, `TOSTResult`, `BlockedOmnibusResult`, `BayesianPlackettLuceResult`, `BayesianCorrelatedTResult`, `EffectSizeForestResult`) and typed exception hierarchy.
- `saes.io` & `saes.validation`: Tidy CSV parser and strict schema/value validator.
- `saes.stats`: Parametric, non-parametric, blocked omnibus, post-hoc (`posthoc.py`), effect-size CI (`effect_sizes.py`), exact Bayesian (`bayesian.py`), sensitivity/TOST (`power.py`), Dolan–Moré profiles (`profiles.py`), explorer (`explorer.py`), decision engine, and `TestRegistry`.
- `saes.reports`: Modular table renderers (`TableRendererRegistry`: LaTeX, PDF, DOCX, CSV, Markdown, HTML) and FAIR reproducibility pack generator.
- `saes.visualization`: 10 registered static/interactive plotters (`PlotterRegistry`).
- `saes.server`: Optional typed FastAPI REST backend (`:8060`) powering `saes-ui` and `saes-ai`.
- `docs/` & `mkdocs.yml`: Material for MkDocs documentation site auto-published to [GitHub Pages](https://saes-suite.github.io/saes/) on every release PR merge to `main`.
