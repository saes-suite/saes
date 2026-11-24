# Changelog

All notable changes to the SAES core repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Exact confidence intervals on all parametric and non-parametric effect sizes (`saes.stats.effect_sizes`): noncentral-$t$ CIs for Cohen's $d$ / Hedges' $g$ (`cohens_d_ci`), noncentral-$F$ CIs for ANOVA $\eta^2$ (`eta_squared_ci`), Brunner–Munzel logit-transformed CIs for Vargha–Delaney $A_{12}$ and Cliff's $\delta$ (`vargha_delaney_a12_ci`, `cliffs_delta_ci`), and percentile bootstrap CIs for Kruskal–Wallis $\varepsilon^2$ (`epsilon_squared_ci`), automatically populated on every `StatisticalResult.effect_size`.
- Comprehensive multiple-comparison adjustments and post-hoc test suite (`saes.stats.posthoc`): extended `adjust_pvalues` to 11 procedures (`holm`, `bonferroni`, `hochberg`, `hommel`, `finner`, `holland`, `rom`, `li`, `fdr_bh`, `fdr_by`, `none`), added Shaffer's static (`adjust_pvalues_shaffer`) and exact Bergmann–Hommel exhaustive (`adjust_pvalues_bergmann_hommel`) matrix procedures, plus `tukey_hsd`, `games_howell`, `dunnett`, `dunn`, `conover_iman`, and `BrunnerMunzelTest` (`saes.statistics.brunner_munzel`).
- Blocked multi-instance omnibus tests (`saes.stats.ranking`): `FriedmanAlignedRanksTest` (`saes.statistics.friedman_aligned_ranks`), `QuadeTest` (`saes.statistics.quade`), and `RepeatedMeasuresANOVA` with Greenhouse–Geisser sphericity $\varepsilon$ (`saes.statistics.rm_anova`).
- Exact Bayesian testing suite (`saes.stats.bayesian`): `BayesianCorrelatedTTest` (Nadeau & Bengio 2003; Benavoli et al. 2017) via `saes.statistics.bayesian_correlated_ttest` and `BayesianPlackettLuceTest` (Calvo et al. 2018) via `saes.statistics.bayesian_plackett_luce`.
- Sensitivity analysis (`compute_minimum_detectable_effect` / `saes.statistics.minimum_detectable_effect`, `PowerAnalysisResult.sensitivity_audit`) and Two One-Sided Tests for statistical equivalence (`saes.tost` / `saes.statistics.tost` / `TOSTResult`, Schuirmann 1987; Lakens 2017).
- Normality assumption checkers `LillieforsChecker` (`saes.statistics.lilliefors`) and `AndersonDarlingChecker` (`saes.statistics.anderson`).
- Extensible plugin registries `TestRegistry` (`saes.stats.registry`) and `TableRenderer` / `TableRendererRegistry` (`saes.reports.renderers`), plus `.summary()` and `.to_dataframe()` convenience methods across primary Pydantic V2 result models.
- Standardized effect-size forest computation (`saes.statistics.effect_size_forest`, `EffectSizeComparisonItem`, `EffectSizeForestResult`) and two new registered publication plotters (`plackett-luce` and `effect-size-forest`, 10 total plotters in `PlotterRegistry`), plus exact analytical posterior density curves (`curve_x`, `curve_y`) on `BayesianCorrelatedTResult`.
- Dataset-aware design capability diagnostics (`can_run_cell_tests`, `can_run_blocked_omnibus`, Anderson–Darling $A^2$ metrics on `DistributionProfile`, and UI routing metadata `target_tab` / `target_mode` on `TestRecommendation`) in `DataExplorerEngine`.
- Typed FastAPI REST endpoints (stateless CSV and stored dataset routes) for full UI parity: `/posthoc`, `/tost`, `/bayesian/correlated-ttest`, `/effect-sizes`, plus expanded `RankingAnalysisResponse` (`bayesian_plackett_luce`, `friedman_aligned`, `quade`, `rm_anova`) and explicit `paired`, `rope`, `prior_strength`, and `posthoc_correction` query/form parameters across existing routes.
- Professional Material for MkDocs GitHub Pages documentation site (`mkdocs.yml`, custom CSS/MathJax 3/Mermaid, `mkdocstrings[python]` auto-API reference, live MkDocs build hooks in `docs/hooks.py` auto-syncing `version.txt`, `CHANGELOG.md`, `pyproject.toml` extras, `PlotterRegistry`, and FastAPI OpenAPI routes, 6 generated vector SVG benchmark figures, `[docs]` optional extra pinned to `mkdocs>=1.6.0,<2.0.0`, `make docs` / `make docs-serve` / `make docs-preview`, PR validation in `.github/workflows/docs.yml`, and automated post-release GitHub Pages deployment in `.github/workflows/release.yml`).
- Full PEP 621 PyPI metadata (`license`, `authors`, `keywords`, `classifiers`, `[project.urls]`) in `pyproject.toml`, `CITATION.cff` software citation metadata (with dynamic BibTeX section in docs), `statsmodels_version` tracking in `EnvironmentManifest`, and multi-version Python 3.11 / 3.12 / 3.13 GitHub Actions test matrix.

### Fixed

- Suppressed SciPy `brunnermunzel` complete-separation `RuntimeWarning` in `saes.stats.effect_sizes` and `saes.stats.posthoc` by gracefully falling back to the normal-approximation distribution before exact boundary clamping.
- Fixed `TestDecisionEngine.select_test()` to gate parametric vs. non-parametric tests on Shapiro–Wilk normality (checking paired differences for paired 2-group comparisons) and stopped silently inferring statistical pairing from matching `run_id` indices (explicit `paired: bool = False` default across `analyze`, `generate_table`, `profiles`, and `power`).
- Fixed `FriedmanNemenyiTest` to compute the exact tie-corrected Friedman $\chi^2_F$ statistic (matching `scipy.stats.friedmanchisquare` and R `friedman.test`) and aggregate multi-metric `all-metrics` scores within each instance so `n_blocks == n_instances` without inflating block sample sizes.
- Fixed `BayesianRankingTest` to follow exact Benavoli et al. (2017) Dirichlet Process posterior weights (removing the artificial $N=12$ effective sample cap and 3 pseudo-observations, adding exact optional absolute `rope` in metric units and `prior_strength`, and computing coherent multi-algorithm $P(\text{best})$ via Plackett–Luce).
- Replaced invalid uncalibrated `scipy.stats.kstest` normality check on standardized samples in `DataExplorerEngine` with the Lilliefors-corrected test (`statsmodels.stats.diagnostic.lilliefors`).
- Replaced invalid unblocked One-Way ANOVA / Welch ANOVA / Kruskal–Wallis computation over multi-instance means in `StatisticalTableGenerator` (returning `None` for multi-instance scope while exposing blocked Friedman, Aligned Friedman, and Quade statistics) and replaced parametric normal approximations in `_fast_bayesian_*` with exact DP Signed-Rank and Plackett–Luce calculations.
- Fixed `compute_post_hoc_power_audit` pairing inference (`paired=(n1 == n2)`) and pooled-SD vs. Welch inconsistency, and added per-instance multiplicity adjustment and non-parametric $A_{12}$/median win direction in `compute_performance_profiles`.
- Removed duplicated Section 6 cell in generated Jupyter notebooks (`saes.reports.reproducibility`) and replaced deprecated `seaborn.boxplot` call in `BoxplotGenerator` with native Matplotlib `ax.boxplot(..., orientation="vertical")` to eliminate Matplotlib 3.11 deprecation warnings.

### Changed

- **Breaking (v2.0.0 Consolidation)**: Consolidated the PEP 8 Python package and PyPI distribution from `saesV2` into canonical **`saes`** (`pip install saes`, `import saes`), succeeding SAES `1.5.0` on PyPI and removing Ruff `N999` suppression. Legacy SAES v1 users should pin `pip install "SAES<2"`.
- Modularized `saes.stats.tables` by extracting LaTeX, Markdown, CSV, HTML, DOCX, and PDF presentation renderers into `saes.reports.renderers` (fully re-exported for backward compatibility) with clean `ConfigurationError` handling when `[viz]` is not installed.
- Removed unused `tabulate` dependency from `pyproject.toml`.

## [Released]

## [0.3.1] — 2026-11-20 (@rorro6787)

### Changed

- Decoupled `saesV2.server.app` from `saes_ai` by removing optional runtime mounting of `saes_ai.server.routes` on `:8060`.
- Removed legacy `saes` module alias from `saesV2/__init__.py` and `ValidationIssueSchema` alias from `saesV2/server/schemas.py`.

### Fixed

- Use absolute `raw.githubusercontent.com` URL for the SAES logo in `README.md` so it renders properly on PyPI.

## [0.3.0] — 2026-11-19 (@rorro6787)

### Fixed

- Added `cells_count` property to `AnalysisReport` (`saesV2.app.pipeline`) so generated reproducibility scripts and notebooks can access `analysis.cells_count` without raising `AttributeError`.
- Resolved `__version__` dynamically from package metadata (`importlib.metadata`) with `version.txt` fallback and replaced hardcoded `"0.1.0"` versions in `saesV2.server` and `saesV2.reports.reproducibility`.
- Updated `README.md` (added SAES logo, fixed `saesV2` package/module names and removed obsolete `saes.chat` references) and `Makefile` help text.

## [0.2.0] — 2026-11-18 (@rorro6787)

### Added

- Initial release of SAES core statistics suite.
- One-way ANOVA and Welch's ANOVA tests.
- Two-sample independent, Welch's, and paired t-tests.
- Normality checking via Shapiro-Wilk test.
- Variance homogeneity checking via Levene's test.
- Automatic statistical test selection decision engine.
- Effect size calculations (Cohen's d, Eta-squared).
- High-level orchestration pipeline.
- `saesV2.export_plot(...)` programmatic Python API to export any registered statistical plot or diagram (`boxplot`, `violin`, `critical-distance`, `bayesian-rank`, `barplot`, `lineplot`, `ecdf`, `scatter`) as PNG (300 DPI), SVG, or PDF bytes.
- REST API plot export endpoints `POST /api/v1/plots/export` and `POST /api/v1/datasets/{filename}/plots/export` returning downloadable PNG, SVG, or PDF figures.
- Dolan-Moré (2002) performance profiles and meta-benchmarking engine (`saes.stats.profiles.compute_performance_profiles`, `saes.profiles`) computing solver efficiency $\rho(1)$, robustness $\lim \rho(\tau)$, $\tau$ curves, and global pairwise Win/Tie/Loss (W/T/L) and Copeland score ranking matrices across multi-problem benchmark suites.
- Statistical power and experimental design planning engine (`saes.stats.power.analyze_statistical_power`, `calculate_statistical_power`, `estimate_required_sample_size`, `generate_power_curves`, `saes.power`) implementing exact two-sample power via non-central Student $t$ distribution (`scipy.stats.nct`), Hodges-Lehmann non-parametric sample size bounds, and post-hoc power audits to prevent Type II errors in stochastic algorithm evaluation.
- FAIR science reproducibility pack generator (`saes.reports.reproducibility.generate_reproducibility_artifacts`, `saes.reproducibility`) creating standalone executable Python scripts (`reproduce.py` with `import saes`), Jupyter Notebooks (`.ipynb` nbformat 4), and environment SHA-256 integrity manifests.
- REST API endpoints for specialized scientific capabilities: `POST /api/v1/profiles`, `POST /api/v1/power`, `POST /api/v1/reproducibility`, and dataset-scoped counterparts `/api/v1/datasets/{filename}/profiles`, `/api/v1/datasets/{filename}/power`, `/api/v1/datasets/{filename}/reproducibility`.
- Publication-grade PDF styling & presentation customization in `_build_pdf_bytes`, `export_statistical_table`, and `saes.export_table` with support for custom title, custom caption, 5 distinct publication color themes (`slate`, `academic_bw`, `emerald`, `navy`, `crimson`), orientation (`landscape` vs `portrait`), font density scaling (`compact` 7pt, `standard` 8.5pt, `large` 10pt), best performer winner highlighting, zebra striping, and border rule styles (`full`, `horizontal` booktabs, `minimal`).
- REST API support for PDF table presentation options across `POST /api/v1/tables/export`, `GET /api/v1/datasets/{filename}/tables/export`, and `POST /api/v1/datasets/{filename}/tables/export`.
- Standalone LaTeX document generation (`latex_document_mode`: `standalone` vs `snippet`) with complete LaTeX document preambles and postambles (`\documentclass`, `\usepackage[table]{xcolor}`, `booktabs`, `amsmath`, `\begin{document}`, `\maketitle`) allowing immediate out-of-the-box compilation in Overleaf or TeXShop.
- Configurable LaTeX number notation modes (`latex_notation`: `scientific` `$2.55\text{e-}02_{...}$`, `siunitx` `\SI{2.55e-2}{...}`, and `standard` `$0.0255 \pm 0.0033$`) with cell shading (`\cellcolor{gray95}` for best, `\cellcolor{gray25}` for second best), adaptive `\tabcolsep` column spacing, and vertical grid borders matching standard scientific benchmark styles.
- Non-parametric hypothesis tests (`WilcoxonSignedRankTest`, `MannWhitneyUTest`, `KruskalWallisTest`), non-parametric effect sizes (`compute_vargha_delaney_a12`, `compute_cliffs_delta`, Epsilon-squared), and multiple-testing p-value corrections (`adjust_pvalues`: Holm-Bonferroni, Bonferroni-Dunn, Hochberg, Benjamini-Hochberg FDR) in `saes.stats.nonparametric` and `saes.statistics.*`.
- Scientific benchmark table generation (`StatisticalTableGenerator`, `saes.generate_table`, `saes.statistics.table`) and multi-format table exporter (`export_statistical_table`, `saes.export_table`) supporting Instance × Algorithm (+ / ≈ / −) benchmark tables, 1vs1 pairwise comparison matrices with per-instance statistical confidence symbol sequences (`+ / = / -`, `instance_symbols`, standalone LaTeX `1vs1 Table` with `\texttt{...}` and ordered instance caption), Friedman/ANOVA post-hoc control tables, and exports in PDF, Word (`.docx`), LaTeX, CSV, Markdown, and HTML.
- Dataset observation explorer, multi-metric side-by-side pivot reveal, distribution/normality profiler (Shapiro-Wilk, D'Agostino-Pearson $K^2$, Kolmogorov-Smirnov, Levene, Bartlett, IQR outliers), context-aware statistical test recommendation engine (`DataExplorerEngine`, `saes.explore`, `saes.statistics.explore`), and multi-format observation table exporter (`export_exploration_table`, `saes.export_exploration`).
- REST API endpoints `POST /api/v1/tables`, `POST /api/v1/tables/export`, `POST /api/v1/explore`, `POST /api/v1/explore/export`, and HTTP-cacheable dataset-scoped `GET`/`POST` counterparts `/api/v1/datasets/{filename}/tables`, `/api/v1/datasets/{filename}/tables/export`, `/api/v1/datasets/{filename}/explore`, and `/api/v1/datasets/{filename}/explore/export`.
- Non-parametric Friedman rank test with Iman-Davenport F and Nemenyi post-hoc Critical Distance (`FriedmanNemenyiTest`, `saes.statistics.friedman_nemenyi`), plus Bayesian Sign and Signed-Rank Simplex ROPE & posterior ranking tests (`BayesianRankingTest`, `saes.statistics.bayesian_ranking`, `bayesian_sign`, `bayesian_signed_rank`) with multi-metric (`all-metrics`) support.
- Publication-grade `CriticalDistancePlotGenerator` (`critical-distance`) and `BayesianRankPlotGenerator` (`bayesian-rank`, `bayesian-sign`, `bayesian-signed-rank`) supporting Matplotlib (`plot`) and Plotly (`plot_interactive`), registered in `PlotterRegistry` and exposed via `saes.plot(...)`.
- Scientific paper aspect ratio presets (`PAPER_FORMAT_PRESETS`: `single-col`, `golden`, `double-col`, `square`, `compact`) and palettes (`academic`, `nature`, `viridis`, `colorblind`, `set2`) in `saes.visualization.utils`.
- REST API endpoints `POST /api/v1/ranking` and `POST /api/v1/datasets/{filename}/ranking`, plus `interpretation` and `group_summaries` in `/api/v1/analyze` responses.
- PEP 561 `py.typed` marker, core `pydantic>=2.0.0` dependency, and lazy loading for `PlotterRegistry`.

### Fixed

- Fixed LaTeX table compilation errors in `render_latex_table` and `export_flat_table` where unescaped Unicode characters (`±`, `α`, `−`, `≈`, `—`, `²`) in benchmark table captions, cells, and headers caused fatal `pdflatex` compilation errors (`Unicode character not set up for use with LaTeX`). Added automated conversion of mathematical symbols to standard LaTeX math commands and `\footnotesize` scaling for publication compliance.

### Changed

- Decoupled and extracted `chat`, `mcp`, and `plugins` into the dedicated `saes-ai` package.
- `saes.server.app` now dynamically mounts `saes_ai.server.routes` when `saes-ai` is installed without internal legacy fallbacks.
- Cleaned core dependencies by removing `chat` extra and delegating AI tooling to `[ai]` optional extra (`saes-ai`).

