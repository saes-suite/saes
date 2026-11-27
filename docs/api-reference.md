# Python API Reference

Every capability in the SAES ecosystem is exposed first as a strongly typed Python API under `import saes` and `saes.statistics`.

## 1. Top-Level Workflow API (`import saes`)

| Function | Signature / Purpose | Return Type |
| :--- | :--- | :--- |
| `saes.load` | `load(source)` — Load CSV filepath or buffer | `Dataset` |
| `saes.validate` | `validate(dataset)` — Strict schema & type validation | `ValidationReport` |
| `saes.explore` | `explore(dataset, metrics=None, algorithms=None, instances=None, alpha=0.05)` | `DataExplorationResult` |
| `saes.export_exploration` | `export_exploration(exploration, view_mode="reveal_metrics", export_format="csv", ...)` | `bytes` |
| `saes.analyze` | `analyze(dataset, alpha=0.05, generate_plots=True, paired=False)` | `AnalysisReport` |
| `saes.generate_table` | `generate_table(dataset, metric=None, table_type="instance_algorithm", ..., paired=False)` | `StatisticalTableResult` |
| `saes.export_table` | `export_table(table_result, export_format="latex", ...)` | `bytes` |
| `saes.plot` | `plot(dataset, plot_type="boxplot", metric=None, instance=None, interactive=False, **kwargs)` | `Figure` |
| `saes.export_plot` | `export_plot(dataset, plot_type="boxplot", ..., export_format="png", dpi=300)` | `tuple[bytes, str, str]` |
| `saes.profiles` | `profiles(dataset, metric=None, alpha=0.05, max_tau=None, num_tau_points=60, paired=False, posthoc_correction="holm")` | `PerformanceProfilesResult` |
| `saes.power` | `power(dataset, metric=None, alpha=0.05, target_power=0.80, target_effect_size=0.50, paired=False)` | `PowerAnalysisResult` |
| `saes.tost` | `tost(x1, x2, low_eqbound, high_eqbound=None, alpha=0.05, paired=False, equal_var=False, ...)` | `TOSTResult` |
| `saes.compute_minimum_detectable_effect` | `compute_minimum_detectable_effect(n, target_power=0.80, alpha=0.05, paired=False)` | `float` |
| `saes.reproducibility` | `reproducibility(dataset, metric=None, alpha=0.05, filename="dataset.csv", raw_csv=None)` | `ReproducibilityResult` |

## 2. Statistical Namespace (`saes.statistics.*`)

### Assumption Diagnostics
- `shapiro(groups, alpha=0.05) -> list[AssumptionResult]`
- `lilliefors(groups, alpha=0.05) -> list[AssumptionResult]`
- `anderson(groups, alpha=0.05) -> list[AssumptionResult]`
- `levene(groups, alpha=0.05) -> AssumptionResult`

### Two-Sample & Multi-Group Tests (`groups: dict[str, np.ndarray]`)
- Parametric: `ttest_ind`, `ttest_welch`, `ttest_rel`, `anova_oneway`, `anova_welch`
- Non-parametric: `mann_whitney`, `wilcoxon`, `brunner_munzel`, `kruskal`

### Effect Sizes, Exact Confidence Intervals & Forest Summary
- `cohens_d_ci(d, n1, n2=None, alpha=0.05, paired=False, hedges_correction=False) -> tuple[float, float]`
- `eta_squared_ci(f_stat, df_between, df_within, alpha=0.05) -> tuple[float, float]`
- `vargha_delaney_a12(x1, x2, maximize=True) -> tuple[float, str]`
- `vargha_delaney_a12_ci(x1, x2, maximize=True, alpha=0.05) -> tuple[float, str, tuple[float, float]]`
- `cliffs_delta(x1, x2, maximize=True) -> tuple[float, str]`
- `cliffs_delta_ci(x1, x2, maximize=True, alpha=0.05) -> tuple[float, str, tuple[float, float]]`
- `effect_size_forest(dataset, metric, scope="single-instance", instance=None, reference_algorithm=None, alpha=0.05, paired=False, p_adjust="holm") -> EffectSizeForestResult`

### Multiple-Testing Adjustments & Post-Hoc Tests
- `adjust_pvalues(p_values, method="holm") -> list[float]` (`holm`, `bonferroni`, `hochberg`, `hommel`, `finner`, `holland`, `rom`, `li`, `fdr_bh`, `fdr_by`, `none`)
- `adjust_pvalues_shaffer(p_matrix: np.ndarray) -> np.ndarray`
- `adjust_pvalues_bergmann_hommel(p_matrix: np.ndarray, max_exact_k: int = 9) -> tuple[np.ndarray, list[str]]`
- `tukey_hsd(groups, alpha=0.05) -> PostHocResult`
- `games_howell(groups, alpha=0.05) -> PostHocResult`
- `dunnett(groups, control, alpha=0.05) -> PostHocResult`
- `dunn(groups, alpha=0.05, p_adjust="holm") -> PostHocResult`
- `conover_iman(groups, alpha=0.05, p_adjust="holm") -> PostHocResult`

### Blocked Multi-Instance Omnibus Tests
- `friedman_nemenyi(dataset, metric, alpha=0.05, scope="all-instances", instance=None) -> CriticalDistanceResult`
- `friedman_aligned_ranks(dataset, metric, alpha=0.05, scope="all-instances", instance=None) -> BlockedOmnibusResult`
- `quade(dataset, metric, alpha=0.05, scope="all-instances", instance=None) -> BlockedOmnibusResult`
- `rm_anova(dataset, metric, alpha=0.05, scope="all-instances", instance=None, greenhouse_geisser=True) -> BlockedOmnibusResult`

### Bayesian Comparisons & Ranking
- `bayesian_ranking(dataset, metric, scope="all-instances", instance=None, rope_percent=1.0, rope=None, prior_strength=1.0, num_samples=1500, algo_a=None, algo_b=None, method="signed-rank") -> BayesianRankResult`
- `bayesian_sign(...) -> BayesianRankResult`
- `bayesian_signed_rank(...) -> BayesianRankResult`
- `bayesian_correlated_ttest(x_a, x_b, algo_a="Algorithm A", algo_b="Algorithm B", rope=0.01, rho=None, maximize=True) -> BayesianCorrelatedTResult`
- `bayesian_plackett_luce(dataset, metric, scope="all-instances", instance=None, num_samples=400, prior_alpha=1.0, seed=42) -> BayesianPlackettLuceResult`

### Power, Sensitivity (MDES) & Equivalence (TOST)
- `minimum_detectable_effect(n, target_power=0.80, alpha=0.05, paired=False) -> float`
- `tost(x1, x2, low_eqbound, high_eqbound=None, alpha=0.05, paired=False, equal_var=False, ...) -> TOSTResult`

### Extensibility & Python Plugins (`saes.plugins` / `saes.PluginRegistry`)
- `saes.plugins.discover(force=False) -> list[PluginManifest]` — Discover installed `entry_points` (`saes.tests`, `saes.renderers`, `saes.plotters`, `saes.plugins`).
- `saes.plugins.list_installed() -> list[PluginManifest]` — List active SAES domain plugins.
- `saes.plugins.get(plugin_id) -> SAESPlugin` / `saes.plugins.get_namespace(plugin_id) -> Any` — Access a plugin instance or its Python API namespace.
- `saes.plugins.register(plugin)` / `saes.plugins.unregister(plugin_id)` — Programmatic runtime registration.

---

## 3. Auto-Generated Module Documentation

### Top-Level API (`saes.api`)

::: saes.api
    options:
      members:
        - load
        - validate
        - explore
        - analyze
        - generate_table
        - export_table
        - plot
        - export_plot
        - compute_performance_profiles
        - analyze_statistical_power
        - tost
        - compute_minimum_detectable_effect
        - generate_reproducibility_artifacts
      show_root_heading: false

### Statistical Namespace (`saes.api.StatisticsNamespace`)

::: saes.api.StatisticsNamespace
    options:
      show_root_heading: false

### Key Domain Result Models (`saes.domain.models`)

::: saes.domain.models
    options:
      members:
        - StatisticalResult
        - PostHocResult
        - TOSTResult
        - BlockedOmnibusResult
        - CriticalDistanceResult
        - BayesianRankResult
        - BayesianPlackettLuceResult
        - BayesianCorrelatedTResult
        - EffectSizeForestResult
        - PowerAnalysisResult
        - DataExplorationResult
      show_root_heading: false
