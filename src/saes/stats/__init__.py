"""Statistical engine package containing tests, assumption checks, post-hoc procedures, and registries."""

from saes.stats import bayesian, effect_sizes, posthoc, power, profiles
from saes.stats.assumptions import (
    AndersonDarlingChecker,
    HomogeneityChecker,
    LillieforsChecker,
    NormalityChecker,
)
from saes.stats.base import StatisticalTest
from saes.stats.bayesian import BayesianCorrelatedTTest, BayesianPlackettLuceTest
from saes.stats.effect_sizes import (
    cohens_d_ci,
    compute_cliffs_delta_with_ci,
    compute_effect_size_forest,
    compute_vargha_delaney_a12_with_ci,
    epsilon_squared_ci,
    eta_squared_ci,
)
from saes.stats.posthoc import (
    BrunnerMunzelTest,
    adjust_pvalues,
    adjust_pvalues_bergmann_hommel,
    adjust_pvalues_shaffer,
    conover_iman_test,
    dunn_test,
    dunnett,
    games_howell,
    tukey_hsd,
)
from saes.stats.power import (
    analyze_statistical_power,
    calculate_statistical_power,
    compute_minimum_detectable_effect,
    compute_post_hoc_power_audit,
    compute_sensitivity_audit,
    estimate_required_sample_size,
    generate_power_curves,
    tost_equivalence_test,
)
from saes.stats.profiles import compute_performance_profiles
from saes.stats.ranking import (
    BayesianRankingTest,
    FriedmanAlignedRanksTest,
    FriedmanNemenyiTest,
    QuadeTest,
    RepeatedMeasuresANOVA,
)
from saes.stats.registry import TestRegistry

__all__ = [
    "bayesian",
    "effect_sizes",
    "posthoc",
    "power",
    "profiles",
    "StatisticalTest",
    "TestRegistry",
    "NormalityChecker",
    "HomogeneityChecker",
    "LillieforsChecker",
    "AndersonDarlingChecker",
    "BrunnerMunzelTest",
    "BayesianCorrelatedTTest",
    "BayesianPlackettLuceTest",
    "BayesianRankingTest",
    "FriedmanNemenyiTest",
    "FriedmanAlignedRanksTest",
    "QuadeTest",
    "RepeatedMeasuresANOVA",
    "cohens_d_ci",
    "eta_squared_ci",
    "compute_vargha_delaney_a12_with_ci",
    "compute_cliffs_delta_with_ci",
    "compute_effect_size_forest",
    "epsilon_squared_ci",
    "adjust_pvalues",
    "adjust_pvalues_shaffer",
    "adjust_pvalues_bergmann_hommel",
    "tukey_hsd",
    "games_howell",
    "dunnett",
    "dunn_test",
    "conover_iman_test",
    "analyze_statistical_power",
    "calculate_statistical_power",
    "compute_minimum_detectable_effect",
    "compute_post_hoc_power_audit",
    "compute_sensitivity_audit",
    "estimate_required_sample_size",
    "generate_power_curves",
    "tost_equivalence_test",
    "compute_performance_profiles",
]

