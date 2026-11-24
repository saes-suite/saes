# Statistical Decision Engine & Methodological Safeguards

SAES automates hypothesis test selection while remaining 100% transparent and methodologically conservative. Every assumption check, rationale, and safeguard is stored on the returned Pydantic V2 result object.

## Automated Cell-Level Decision Tree

For each `(instance, metric)` experimental cell, `saes.analyze(dataset, alpha=0.05, paired=False)` executes the following assumption-driven decision flow:

```mermaid
flowchart TD
    A["Input Cell (instance, metric)<br/>k algorithms, n runs/cell"] --> B{"k == 2 or k >= 3?"}
    B -- "k == 2" --> C{"paired == True?"}
    C -- "Yes (Matched Folds / CRN)" --> D{"Normality of differences<br/>(Shapiro-Wilk p >= alpha)"}
    D -- "Pass" --> E["Paired Student t-test<br/>+ Cohen's d_z (95% noncentral-t CI)"]
    D -- "Fail" --> F["Wilcoxon Signed-Rank Test<br/>+ Matched-pairs r & A12 (95% CI)"]

    C -- "No (Independent Runs, Default)" --> G{"All groups normal?<br/>(Shapiro-Wilk p >= alpha)"}
    G -- "Pass" --> H{"Homoscedastic?<br/>(Levene median p >= alpha)"}
    H -- "Pass" --> I["Independent Student t-test<br/>+ Hedges' g / Cohen's d (95% CI)"]
    H -- "Fail" --> J["Welch's t-test<br/>+ Hedges' g / Cohen's d (95% CI)"]
    G -- "Fail" --> K["Mann-Whitney U / Brunner-Munzel<br/>+ Vargha-Delaney A12 & Cliff's delta (95% CI)"]

    B -- "k >= 3" --> L{"All groups normal?<br/>(Shapiro-Wilk p >= alpha)"}
    L -- "Pass" --> M{"Homoscedastic?<br/>(Levene median p >= alpha)"}
    M -- "Pass" --> N["One-Way Fisher ANOVA<br/>+ Eta-squared (95% CI) -> Tukey HSD / Dunnett"]
    M -- "Fail" --> O["Welch's One-Way ANOVA<br/>+ Omega-squared -> Games-Howell"]
    L -- "Fail" --> P["Kruskal-Wallis H Test<br/>+ Epsilon-squared (Bootstrap CI) -> Dunn / Conover-Iman"]
```

## Normality & Variance Diagnostics

`saes.explore()` and `saes.statistics.*` compute a full suite of distribution checks:

| Check | Function | Exact Method | When It Shines |
| :--- | :--- | :--- | :--- |
| **Shapiro–Wilk** | `saes.statistics.shapiro` | Royston's $W$ | Highest power for small-to-moderate samples ($3 \le n \le 5000$) |
| **Lilliefors KS** | `saes.statistics.lilliefors` | `statsmodels.stats.diagnostic.lilliefors` | Valid KS test when $\mu, \sigma$ are estimated from sample data |
| **Anderson–Darling** | `saes.statistics.anderson` | $A^2$ with Stephens (1986) $p$-value | Sensitive to heavy tails and skewed extreme values |
| **D'Agostino–Pearson $K^2$** | Included in `explore()` | Omnibus skewness & kurtosis test | Recommended for larger run counts ($n \ge 20$) |
| **Levene (Brown–Forsythe)** | `saes.statistics.levene` | Median-centered variance test | Robust to non-Gaussian distributions |

!!! note "Why Lilliefors Instead of Raw `kstest(..., 'norm')`?"
    Standard one-sample Kolmogorov–Smirnov assumes the population mean and variance are known *a priori*. Plugging sample estimates $\bar{x}, s$ into `scipy.stats.kstest` yields severely anti-conservative $p$-values. SAES uses **Lilliefors' corrected distribution** everywhere.

## Built-In Methodological Safeguards

SAES actively prevents four common errors in empirical algorithm benchmarking:

1. **No Implicit Pairing (`paired=False` Default)**
   Stochastic optimization runs (`run_id`) are independent random seeds unless the researcher explicitly declares `paired=True` (e.g., identical cross-validation folds or Common Random Numbers).
2. **No Multi-Instance Pseudoreplication**
   When generating global multi-instance summary tables (`N > 1` instances), SAES suppresses unblocked one-way ANOVA and Kruskal–Wallis across pooled runs and reports **blocked** omnibus tests instead (Friedman, Iman–Davenport, Friedman Aligned-Ranks, Quade, and RM-ANOVA with Greenhouse–Geisser $\varepsilon$).
3. **Single-Run Cell Guard (`can_run_cell_tests`)**
   When a benchmark provides only $n=1$ observation per `(algorithm, instance)` cell (e.g., deterministic algorithms or pre-aggregated instance means), `explore().context_diagnostics.can_run_cell_tests` is `False`, guiding both Python callers and `saes-ui` toward blocked multi-instance tests and Bayesian ranking models.
4. **Multi-Metric Pooling Warning**
   When pooling heterogeneous metrics (`metric="all-metrics"`) in ranking analyses, SAES attaches explicit methodological warnings noting that correlated quality indicators (e.g., `HV` and `IGD+` on the same problem) violate block independence.
