# Blocked Omnibus Tests & Post-Hoc Multiple Comparisons

When benchmarking $k \ge 3$ algorithms across $N \ge 2$ benchmark instances, observations are grouped into **matched blocks** (each instance is a block). SAES implements both complete multi-instance blocked omnibus procedures and cell-level post-hoc multiple-comparison tests.

## 1. Multi-Instance Blocked Omnibus Suite

| Procedure | Python API | Strengths & Literature |
| :--- | :--- | :--- |
| **Tie-Corrected Friedman + Iman–Davenport $F$ + Nemenyi CD** | `saes.statistics.friedman_nemenyi` | Standard non-parametric rank omnibus with exact tie correction (Demšar 2006) |
| **Friedman Aligned-Ranks** | `saes.statistics.friedman_aligned_ranks` | Centers each block before joint ranking across $N \times k$ values; higher power than Friedman when $k \le 5$ (Hodges & Lehmann 1962; García et al. 2010) |
| **Quade Weighted-Block Test** | `saes.statistics.quade` | Weights within-block ranks by each instance's sample range; highlights difficult high-spread instances (Quade 1979) |
| **Repeated-Measures ANOVA (Greenhouse–Geisser $\varepsilon$)** | `saes.statistics.rm_anova` | Parametric blocked $F$-test with automatic sphericity $\varepsilon$-correction (Greenhouse & Geisser 1959) |

```python
import saes

ds = saes.load("datasets/swarmintelligence.csv")

cd = saes.statistics.friedman_nemenyi(ds, metric="HV", alpha=0.05)
aligned = saes.statistics.friedman_aligned_ranks(ds, metric="HV", alpha=0.05)
quade = saes.statistics.quade(ds, metric="HV", alpha=0.05)
rm = saes.statistics.rm_anova(ds, metric="HV", alpha=0.05, greenhouse_geisser=True)

for res in (cd, aligned, quade, rm):
    print(res.summary())
```

## 2. Dedicated Cell-Level Post-Hoc Procedures

After a significant cell-level omnibus test (One-Way ANOVA, Welch's ANOVA, or Kruskal–Wallis), run targeted pairwise or control-vs-all post-hoc comparisons:

| Procedure | Python API | Assumption Regime | Output Extras |
| :--- | :--- | :--- | :--- |
| **Tukey HSD** | `saes.statistics.tukey_hsd` | Normal, homoscedastic (all-pairs) | Mean diff + simultaneous 95% CIs |
| **Games–Howell** | `saes.statistics.games_howell` | Normal, heteroscedastic (all-pairs) | Welch–Satterthwaite df + 95% CIs |
| **Dunnett's Test** | `saes.statistics.dunnett` | Normal, many-to-one vs `control` | Higher power than all-pairs |
| **Dunn's Test** | `saes.statistics.dunn` | Non-parametric (after Kruskal–Wallis) | Tie-corrected $z$ + 11 FWER/FDR adjustments |
| **Conover–Iman** | `saes.statistics.conover_iman` | Non-parametric (higher power than Dunn) | Student-$t$ rank statistic + 11 adjustments |

```python
cell = ds.get_cell("DTLZ2", "HV")
groups = {alg: g["value"].to_numpy(dtype=float) for alg, g in cell.groupby("algorithm")}

# Non-parametric all-pairs with Holm adjustment
dunn_res = saes.statistics.dunn(groups, alpha=0.05, p_adjust="holm")
conover_res = saes.statistics.conover_iman(groups, alpha=0.05, p_adjust="finner")

# Parametric alternatives
gh_res = saes.statistics.games_howell(groups, alpha=0.05)
dunnett_res = saes.statistics.dunnett(groups, control="NSGAII", alpha=0.05)

print(dunn_res.to_dataframe().head())
```

## 3. Complete Multiplicity Adjustment Catalog (13 Procedures)

### Vector Adjustments (`saes.statistics.adjust_pvalues`)

| Method Key | Error Rate | Notes & Citation |
| :--- | :---: | :--- |
| `holm` | FWER | Step-down; uniformly more powerful than Bonferroni (Holm 1979) |
| `bonferroni` | FWER | Single-step conservative bound |
| `hochberg` | FWER | Step-up under independence / PRDS (Hochberg 1988) |
| `hommel` | FWER | Closed-testing Simes step-up (Hommel 1988) |
| `finner` | FWER | Step-down geometric adjustment recommended in García et al. (2010) |
| `holland` | FWER | Šidák step-down procedure (Holland & Copenhaver 1987) |
| `rom` | FWER | Exact recursive step-up improvement over Hochberg (Rom 1990) |
| `li` | FWER | Two-step control-vs-all procedure (Li 2008) |
| `fdr_bh` | FDR | Benjamini–Hochberg (1995) |
| `fdr_by` | FDR | Benjamini–Yekutieli (2001), arbitrary dependence |
| `none` | — | Raw unadjusted $p$-values |

### All-Pairs Matrix-Level Logical Adjustments

For $k \times k$ all-pairs matrices, SAES also provides logical transitivity-aware FWER procedures (García & Herrera 2008):

- **`saes.statistics.adjust_pvalues_shaffer(p_matrix)`** — Shaffer's static step-down rule exploiting maximal true-hypothesis subsets.
- **`saes.statistics.adjust_pvalues_bergmann_hommel(p_matrix)`** — Exhaustive Bergmann–Hommel closure (exact up to $k \le 9$ algorithms, with automatic Shaffer fallback and warning for $k > 9$).

## 4. Two-Sample Stochastic Equality (Brunner–Munzel)

Unlike Mann–Whitney $U$ (which requires equal distribution shapes to test median equality), **Brunner–Munzel** tests the nonparametric Behrens–Fisher hypothesis $H_0: P(X < Y) + \frac{1}{2}P(X = Y) = 0.5$ without assuming homoscedasticity or identical shapes:

```python
bm = saes.statistics.brunner_munzel({"NSGAII": groups["NSGAII"], "SMPSO": groups["SMPSO"]})
print(bm.summary())
```
