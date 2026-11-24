"""Pydantic request and response schemas for SAES REST API."""

from typing import Any

from pydantic import BaseModel, Field

from saes._version import __version__
from saes.domain.models import (
    AssumptionResult,
    BayesianCorrelatedTResult,
    BayesianPlackettLuceResult,
    BayesianRankResult,
    BlockedOmnibusResult,
    CriticalDistanceResult,
    DataExplorationResult,
    EffectSizeForestResult,
    PerformanceProfilesResult,
    PostHocResult,
    PowerAnalysisResult,
    ReproducibilityResult,
    StatisticalTableResult,
    TOSTResult,
    ValidationIssue,
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = Field(default_factory=lambda: __version__)
    service: str = "saes-api"


class ValidationResponse(BaseModel):
    """Validation response report."""

    is_valid: bool
    error_count: int
    warning_count: int
    issues: list[ValidationIssue]


class CellAnalysisSchema(BaseModel):
    """Statistical summary for an experimental cell."""

    instance: str
    metric: str
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Any
    is_significant: bool
    effect_size_name: str | None = None
    effect_size_value: float | None = None
    effect_size_ci_lower: float | None = None
    effect_size_ci_upper: float | None = None
    effect_size_magnitude: str | None = None
    assumptions: list[AssumptionResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    interpretation: str = ""
    group_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Analysis report response."""

    cells_count: int
    results: list[CellAnalysisSchema]


class DatasetItemSchema(BaseModel):
    """Metadata schema for a saved benchmark dataset."""

    name: str
    filename: str
    path: str
    size_bytes: int
    row_count: int | None = None
    modified_at: float | None = None


class DatasetDetailSchema(BaseModel):
    """Detailed response for a dataset including its CSV contents."""

    name: str
    filename: str
    path: str
    size_bytes: int
    row_count: int
    content: str


class DatasetUpdateSchema(BaseModel):
    """Payload for updating or saving CSV dataset content."""

    content: str


class RankingAnalysisResponse(BaseModel):
    """Combined blocked-omnibus and Bayesian ranking response (Friedman, Aligned, Quade, RM-ANOVA, Bayes, PL)."""

    critical_distance: CriticalDistanceResult
    bayesian_ranking: BayesianRankResult
    bayesian_sign: BayesianRankResult | None = None
    bayesian_signed_rank: BayesianRankResult | None = None
    bayesian_plackett_luce: BayesianPlackettLuceResult | None = None
    friedman_aligned: BlockedOmnibusResult | None = None
    quade: BlockedOmnibusResult | None = None
    rm_anova: BlockedOmnibusResult | None = None


StatisticalTableResponse = StatisticalTableResult
DataExplorationResponse = DataExplorationResult
PerformanceProfilesResponse = PerformanceProfilesResult
PowerAnalysisResponse = PowerAnalysisResult
ReproducibilityResponse = ReproducibilityResult
PostHocResponse = PostHocResult
TOSTResponse = TOSTResult
BayesianCorrelatedTResponse = BayesianCorrelatedTResult
EffectSizeForestResponse = EffectSizeForestResult
