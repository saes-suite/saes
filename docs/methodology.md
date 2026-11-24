# Statistical Methodology & Literature References

This document specifies the exact mathematical definitions, methodological safeguards, and primary literature references implemented in `saes`.

## 1. Assumption Diagnostics

| Method | API | Reference |
| :--- | :--- | :--- |
| Shapiro–Wilk normality test | `saes.statistics.shapiro` | Shapiro & Wilk (1965) |
| Lilliefors-corrected Kolmogorov–Smirnov test | `saes.statistics.lilliefors` | Lilliefors (1967) |
| Anderson–Darling normality test ($A^{2*}$) | `saes.statistics.anderson` | Stephens (1974); D'Agostino & Stephens (1986) |
| Brown–Forsythe / Levene variance homogeneity test | `saes.statistics.levene` | Levene (1960); Brown & Forsythe (1974) |

> **Methodological safeguard**: Standard Kolmogorov–Smirnov critical values assume fully known $(\mu, \sigma)$ and become severely conservative when mean and variance are estimated from the sample. Both `saes.statistics.lilliefors` and `saes.explore` therefore use the Lilliefors-corrected distribution via `statsmodels.stats.diagnostic.lilliefors`.

## 2. Explicit Pairing & Automated Test Selection

In stochastic optimization, independent solver runs indexed `run_id = 0, 1, ..., R-1` are **not** statistically paired unless Common Random Numbers (CRN) or identical cross-validation folds were explicitly used. Consequently:

- `saes.analyze`, `TestDecisionEngine`, `saes.generate_table`, `saes.profiles`, and `saes.power` all default to **`paired=False`**.
- Pass `paired=True` explicitly when comparing algorithms across matched folds or CRN seeds.
- Because conditional pre-testing (selecting parametric vs. non-parametric tests solely from a preliminary normality or Levene test) can distort nominal Type I error under certain alternatives (Zimmerman 2004; Rochon et al. 2012), researchers can always bypass automated selection using `force_test=...` or calling `saes.statistics.*` directly.

When automated selection (`TestDecisionEngine`) is used:

1. **Two groups, `paired=True`**: Checks Shapiro–Wilk normality on paired differences $d_i = x_{1,i} - x_{2,i}$. Routes to `PairedTTest` if normal, otherwise `WilcoxonSignedRankTest`.
2. **Two groups, `paired=False`**: Checks per-group Shapiro–Wilk normality. Routes to `MannWhitneyUTest` if violated; otherwise Levene homogeneity selects `IndependentTTest` (equal variances) or `WelchTTest` (unequal variances). For non-parametric heteroscedastic stochastic equality, `BrunnerMunzelTest` is also available directly (Brunner & Munzel 2000; Konietschke et al. 2015).
3. **$k > 2$ independent groups**: Checks per-group Shapiro–Wilk normality. Routes to `KruskalWallisTest` if violated; otherwise Levene selects `OneWayANOVA` or `WelchANOVA`.

## 3. Exact Effect Sizes with Confidence Intervals

Every `StatisticalTest` returns an immutable `EffectSize` object containing the point estimate, magnitude classification, and $(1-\alpha)$ confidence interval (`ci_lower`, `ci_upper`):

- **Cohen's $d$ and Hedges' $g$** (`saes.statistics.cohens_d_ci`): Exact inversion of the noncentral-$t$ CDF (`scipy.stats.nct`) via Brent root-finding (Steiger & Fouladi 1997; Smithson 2001; Kelley 2007), with Hedges–Olkin asymptotic fallback.
- **ANOVA $\eta^2$** (`saes.statistics.eta_squared_ci`): Exact noncentral-$F$ inversion mapping noncentrality parameter $\lambda$ to $\eta^2 = \lambda / (\lambda + \mathrm{df}_1 + \mathrm{df}_2 + 1)$ (Smithson 2001).
- **Vargha–Delaney $A_{12}$ and Cliff's $\delta = 2 A_{12} - 1$** (`saes.statistics.vargha_delaney_a12_ci`, `saes.statistics.cliffs_delta_ci`): Range-preserving logit-transformed Brunner–Munzel (2000) / Konietschke et al. (2015) asymptotic confidence intervals using Welch–Satterthwaite placement degrees of freedom.
- **Kruskal–Wallis $\varepsilon^2 = H / (N - 1)$**: Deterministic percentile bootstrap confidence interval.

## 4. Blocked Multi-Instance Omnibus Tests

For $k$ algorithms evaluated over $N$ matched benchmark instances (blocks), unblocked one-way tests (One-Way ANOVA, Welch ANOVA, Kruskal–Wallis) violate independence and are omitted (`None`) in multi-instance benchmark tables. Instead, SAES implements four blocked omnibus procedures:

1. **Tie-Corrected Friedman Test + Iman–Davenport $F_F$ + Nemenyi CD** (`saes.statistics.friedman_nemenyi`):
   Computes fractional mid-ranks within each block and the exact tie-corrected Friedman $\chi^2_F$ (matching `scipy.stats.friedmanchisquare` and R `friedman.test`), Iman–Davenport $F_F = \frac{(N-1)\chi^2_F}{N(k-1) - \chi^2_F}$, and exact Nemenyi Critical Distance $\mathrm{CD} = q_{\alpha, k} \sqrt{\frac{k(k+1)}{6N}}$ using the studentized range distribution (Demšar 2006).
2. **Friedman Aligned-Ranks Test** (`saes.statistics.friedman_aligned_ranks`):
   Centers each block by its location, ranks all $N \cdot k$ aligned observations jointly, and computes the tie-aware $T_{\mathrm{AR}} \sim \chi^2_{k-1}$ statistic (Hodges & Lehmann 1962; García et al. 2010).
3. **Quade Weighted-Block Test** (`saes.statistics.quade`):
   Weights within-block rank deviations by the rank of each block's performance range, yielding $F_{(k-1), (k-1)(N-1)}$ (Quade 1979; Conover 1999; García et al. 2010).
4. **Repeated-Measures One-Way ANOVA with Greenhouse–Geisser $\varepsilon$** (`saes.statistics.rm_anova`):
   Decomposes $\mathrm{SS}_{\text{total}} = \mathrm{SS}_{\text{treat}} + \mathrm{SS}_{\text{blocks}} + \mathrm{SS}_{\text{error}}$ and scales degrees of freedom by the exact Greenhouse–Geisser sphericity estimate $\hat{\varepsilon} \in [1/(k-1), 1]$ (Greenhouse & Geisser 1959).

> **Multi-metric aggregation (`metric="all-metrics"`)**: When ranking across multiple metrics simultaneously, SAES averages oriented min–max normalized scores *within each instance* before ranking so that $N_{\text{blocks}} = N_{\text{instances}}$ (never multiplying blocks by the number of metrics) and attaches an explicit warning.

## 5. Multiple-Testing Corrections & Post-Hoc Procedures

SAES implements 11 vector p-value adjustment methods in `saes.statistics.adjust_pvalues`:
`holm`, `bonferroni`, `hochberg`, `hommel`, `finner`, `holland`, `rom`, `li`, `fdr_bh`, `fdr_by`, `none`,
plus two all-pairs logical-implication matrix procedures:

- **Shaffer's Static Procedure** (`saes.statistics.adjust_pvalues_shaffer`, Shaffer 1986; García & Herrera 2008).
- **Exact Bergmann–Hommel Exhaustive Procedure** (`saes.statistics.adjust_pvalues_bergmann_hommel`, Bergmann & Hommel 1988; García & Herrera 2008), enumerating all set partitions for $k \le 9$ with documented fallback to Shaffer for $k > 9$.

Dedicated post-hoc test procedures returning `PostHocResult`:
`tukey_hsd` (Tukey 1949), `games_howell` (Games & Howell 1976), `dunnett` (Dunnett 1955), `dunn` (Dunn 1964), and `conover_iman` (Conover & Iman 1979).

## 6. Exact Bayesian Comparisons with ROPE

Following Benavoli et al. (2017) and Calvo et al. (2018):

- **Bayesian Correlated $t$-Test** (`saes.statistics.bayesian_correlated_ttest`): Exact analytical Student-$t$ posterior with Nadeau & Bengio (2003) variance inflation $\sigma^2_{\text{post}} = \left(\frac{1}{n} + \frac{\rho}{1 - \rho}\right) s^2$ for repeated $k$-fold cross-validation.
- **Bayesian Sign Test & Bayesian Signed-Rank Test** (`saes.statistics.bayesian_sign`, `bayesian_signed_rank`, `bayesian_ranking`): Exact Dirichlet Process prior $\mathrm{DP}(s, \delta_0)$ with single pseudo-observation $z_0 = 0$ and weight `prior_strength` $s$ (default `1.0`), exact unit weight per observed block (no sample-size truncation), and support for both exact absolute ROPE (`rope` in metric units) and relative ROPE (`rope_percent`).
- **Bayesian Plackett–Luce Ranking Model** (`saes.statistics.bayesian_plackett_luce`): Multi-algorithm posterior worth parameters $w = (w_1, \dots, w_k)$, winning probabilities $P(\text{best})$, and 95% credible rank intervals estimated via Hunter (2004) MM updates under a Dirichlet Bayesian bootstrap (Calvo et al. 2018).

## 7. Power, Sensitivity (MDES), and Equivalence (TOST)

Because observed ("post-hoc") power is a 1-to-1 transformation of the observed p-value and does not quantify independent evidence for a null result (Hoenig & Heisey 2001), SAES provides:

1. **A priori sample size planning** (`estimate_required_sample_size`, exact noncentral-$t$ + Hodges–Lehmann A.R.E. = 0.864 bound).
2. **Minimum Detectable Effect Size (MDES)** (`saes.statistics.minimum_detectable_effect` & `PowerAnalysisResult.sensitivity_audit`), solving $\mathrm{Power}(n, d, \alpha) = 1 - \beta$ via Brent root-finding.
3. **Two One-Sided Tests of Equivalence (TOST)** (`saes.tost` / `saes.statistics.tost`, Schuirmann 1987; Lakens 2017), declaring statistical equivalence at level $\alpha$ iff $\max(p_{\text{lower}}, p_{\text{upper}}) < \alpha$ (equivalently, the $(1 - 2\alpha)$ confidence interval lies strictly inside $[\Delta_L, \Delta_U]$).

## 8. Performance Profiles & Meta-Benchmarking

`saes.profiles` computes Dolan & Moré (2002) performance ratios $r_{p, s}$ (using per-instance medians across runs), cumulative profiles $\rho_s(\tau)$, efficiency $\rho_s(1)$, robustness $\rho_s(\tau_{\max})$, AUC, and multiplicity-adjusted Win/Tie/Loss + Copeland rankings oriented strictly via $A_{12}$ / medians.

## References

- Arcuri, A., & Briand, L. (2014). A hitchhiker's guide to statistical tests for assessing randomized algorithms in software engineering. *Software Testing, Verification and Reliability*, 24(3), 219–250.
- Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis. *Journal of Machine Learning Research*, 18(77), 1–36.
- Bergmann, G., & Hommel, G. (1988). Improvements of general multiple test procedures for redundant systems of hypotheses. In *Multiple Hypothesenprüfung*, 100–115.
- Brunner, E., & Munzel, U. (2000). The nonparametric Behrens–Fisher problem: Asymptotic theory and a small-sample approximation. *Biometrical Journal*, 42(1), 17–25.
- Calvo, B., Shir, O. M., Ceberio, J., Doerr, C., Wang, H., Bäck, T., & Lozano, J. A. (2018). Bayesian performance analysis for black-box optimization benchmarking. *GECCO Companion*, 1789–1797.
- Conover, W. J., & Iman, R. L. (1979). On multiple-comparisons procedures. *Los Alamos Scientific Laboratory Report LA-7677-MS*.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software with performance profiles. *Mathematical Programming*, 91(2), 201–213.
- Dunn, O. J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6(3), 241–252.
- Games, P. A., & Howell, J. F. (1976). Pairwise multiple comparison procedures with unequal N's and/or variances. *Journal of Educational Statistics*, 1(2), 113–125.
- García, S., & Herrera, F. (2008). An extension on "Statistical comparisons of classifiers over multiple data sets" for all pairwise comparisons. *Journal of Machine Learning Research*, 9, 2677–2694.
- García, S., Fernández, A., Luengo, J., & Herrera, F. (2010). Advanced nonparametric tests for multiple comparisons in the design of experiments in computational intelligence and data mining. *Information Sciences*, 180(10), 2044–2064.
- Hoenig, J. M., & Heisey, D. M. (2001). The abuse of power: The pervasive fallacy of power calculations for data analysis. *The American Statistician*, 55(1), 19–24.
- Hunter, D. R. (2004). MM algorithms for generalized Bradley–Terry models. *The Annals of Statistics*, 32(1), 384–406.
- Konietschke, F., Placzek, M., Schaarschmidt, F., & Hothorn, L. A. (2015). `nparcomp`: An R software package for nonparametric multiple comparisons and simultaneous confidence intervals. *Journal of Statistical Software*, 64(9), 1–17.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.
- Lilliefors, H. W. (1967). On the Kolmogorov–Smirnov test for normality with mean and variance unknown. *JASA*, 62(318), 399–402.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281.
- Quade, D. (1979). Using weighted rankings in the analysis of complete blocks with additive block effects. *JASA*, 74(367), 680–683.
- Rochon, J., Gondan, M., & Kieser, M. (2012). To test or not to test: Preliminary assessment of normality when comparing two independent samples. *BMC Medical Research Methodology*, 12(1), 81.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Shaffer, J. P. (1986). Modified sequentially rejective multiple test procedures. *JASA*, 81(395), 826–831.
- Smithson, M. (2001). Correct confidence intervals for various regression effect sizes and parameters: The importance of noncentral distributions in computing intervals. *Educational and Psychological Measurement*, 61(4), 605–632.
- Steiger, J. H., & Fouladi, R. T. (1997). Noncentrality interval estimation and the evaluation of statistical models. In *What If There Were No Significance Tests?*, 221–257.
- Vargha, A., & Delaney, H. D. (2000). A critique and improvement of the CL common language effect size statistics of McGraw and Wong. *Journal of Educational and Behavioral Statistics*, 25(2), 101–132.
- Zimmerman, D. W. (2004). A note on preliminary tests of equality of variances. *British Journal of Mathematical and Statistical Psychology*, 57(1), 173–181.
