"""Analysis and validation REST endpoints."""

import io
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

import saes
from saes.server.helpers import (
    build_cell_analysis_schemas,
    run_bayesian_correlated_t_for_dataset,
    run_posthoc_for_dataset,
    run_ranking_bundle,
    run_tost_for_dataset,
)
from saes.server.schemas import (
    AnalysisResponse,
    BayesianCorrelatedTResponse,
    DataExplorationResponse,
    EffectSizeForestResponse,
    PerformanceProfilesResponse,
    PostHocResponse,
    PowerAnalysisResponse,
    RankingAnalysisResponse,
    ReproducibilityResponse,
    StatisticalTableResponse,
    TOSTResponse,
    ValidationResponse,
)
from saes.stats.explorer import export_exploration_table
from saes.stats.tables import export_statistical_table

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post("/validate", response_model=ValidationResponse)
async def validate_csv(file: Annotated[UploadFile, File(...)]) -> ValidationResponse:
    """Validate uploaded CSV benchmark dataset."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        report = saes.validate(dataset)
        return ValidationResponse(
            is_valid=report.is_valid,
            error_count=len(report.errors),
            warning_count=len(report.warnings),
            issues=report.issues,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {e!s}") from e


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(
    file: Annotated[UploadFile, File(...)],
    alpha: Annotated[float, Form()] = 0.05,
    paired: Annotated[bool, Form()] = False,
) -> AnalysisResponse:
    """Execute full automated statistical analysis on uploaded CSV dataset."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        results = build_cell_analysis_schemas(dataset, alpha=alpha, paired=paired)
        return AnalysisResponse(cells_count=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Statistical analysis error: {e!s}") from e


@router.post("/ranking", response_model=RankingAnalysisResponse)
async def ranking_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str, Form()],
    alpha: Annotated[float, Form()] = 0.05,
    scope: Annotated[str, Form()] = "all-instances",
    instance: Annotated[str | None, Form()] = None,
    rope_percent: Annotated[float, Form()] = 1.0,
    rope: Annotated[float | None, Form()] = None,
    prior_strength: Annotated[float, Form()] = 1.0,
    algo_a: Annotated[str | None, Form()] = None,
    algo_b: Annotated[str | None, Form()] = None,
    bayesian_method: Annotated[str, Form()] = "signed-rank",
) -> RankingAnalysisResponse:
    """Compute Friedman, Aligned-Ranks, Quade, RM-ANOVA, Bayesian Simplex ROPE, and Plackett-Luce rankings."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return run_ranking_bundle(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            rope=rope,
            prior_strength=prior_strength,
            algo_a=algo_a,
            algo_b=algo_b,
            bayesian_method=bayesian_method,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ranking analysis error: {e!s}") from e


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if not raw or not raw.strip():
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items if items else None


@router.post("/tables", response_model=StatisticalTableResponse)
async def generate_table_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    table_type: Annotated[str, Form()] = "instance_algorithm",
    test_type: Annotated[str, Form()] = "auto",
    posthoc_correction: Annotated[str, Form()] = "holm",
    summary_stat: Annotated[str, Form()] = "auto",
    reference_algorithm: Annotated[str | None, Form()] = None,
    scope: Annotated[str, Form()] = "all-instances",
    selected_instance: Annotated[str | None, Form()] = None,
    algorithms_filter: Annotated[str | None, Form()] = None,
    instances_filter: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    precision: Annotated[int, Form()] = 4,
    latex_document_mode: Annotated[str, Form()] = "standalone",
    latex_notation: Annotated[str, Form()] = "scientific",
    paired: Annotated[bool, Form()] = False,
) -> StatisticalTableResponse:
    """Generate a statistical benchmark table (ANOVA, Friedman, Wilcoxon, Pairwise Matrix) from uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return saes.generate_table(
            dataset=dataset,
            metric=metric,
            table_type=table_type,
            test_type=test_type,
            posthoc_correction=posthoc_correction,
            summary_stat=summary_stat,
            reference_algorithm=reference_algorithm,
            scope=scope,
            selected_instance=selected_instance,
            algorithms_filter=_parse_csv_list(algorithms_filter),
            instances_filter=_parse_csv_list(instances_filter),
            alpha=alpha,
            precision=precision,
            latex_document_mode=latex_document_mode,
            latex_notation=latex_notation,
            paired=paired,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table generation error: {e!s}") from e


@router.post("/tables/export")
async def export_table_csv(
    file: Annotated[UploadFile, File(...)],
    export_format: Annotated[str, Form()] = "latex",
    metric: Annotated[str | None, Form()] = None,
    table_type: Annotated[str, Form()] = "instance_algorithm",
    test_type: Annotated[str, Form()] = "auto",
    posthoc_correction: Annotated[str, Form()] = "holm",
    summary_stat: Annotated[str, Form()] = "auto",
    reference_algorithm: Annotated[str | None, Form()] = None,
    scope: Annotated[str, Form()] = "all-instances",
    selected_instance: Annotated[str | None, Form()] = None,
    algorithms_filter: Annotated[str | None, Form()] = None,
    instances_filter: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    precision: Annotated[int, Form()] = 4,
    latex_document_mode: Annotated[str, Form()] = "standalone",
    latex_notation: Annotated[str, Form()] = "scientific",
    paired: Annotated[bool, Form()] = False,
    pdf_title: Annotated[str | None, Form()] = None,
    pdf_caption: Annotated[str | None, Form()] = None,
    pdf_theme: Annotated[str, Form()] = "slate",
    pdf_orientation: Annotated[str, Form()] = "landscape",
    pdf_font_size: Annotated[float, Form()] = 8.5,
    pdf_highlight_best: Annotated[bool, Form()] = True,
    pdf_show_zebra: Annotated[bool, Form()] = True,
    pdf_grid_style: Annotated[str, Form()] = "full",
) -> Response:
    """Generate and export a statistical benchmark table in PDF, Word (DOCX), LaTeX, CSV, Markdown, or HTML."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        table_res = saes.generate_table(
            dataset=dataset,
            metric=metric,
            table_type=table_type,
            test_type=test_type,
            posthoc_correction=posthoc_correction,
            summary_stat=summary_stat,
            reference_algorithm=reference_algorithm,
            scope=scope,
            selected_instance=selected_instance,
            algorithms_filter=_parse_csv_list(algorithms_filter),
            instances_filter=_parse_csv_list(instances_filter),
            alpha=alpha,
            precision=precision,
            latex_document_mode=latex_document_mode,
            latex_notation=latex_notation,
            paired=paired,
        )
        file_bytes, media_type, ext = export_statistical_table(
            table_res,
            export_format=export_format,
            document_mode=latex_document_mode,
            notation=latex_notation,
            title=pdf_title,
            caption=pdf_caption,
            pdf_theme=pdf_theme,
            pdf_orientation=pdf_orientation,
            pdf_font_size=pdf_font_size,
            pdf_highlight_best=pdf_highlight_best,
            pdf_show_zebra=pdf_show_zebra,
            pdf_grid_style=pdf_grid_style,
        )
        out_name = f"saes_table_{table_res.metric}_{table_res.table_type}.{ext}"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table export error: {e!s}") from e


@router.post("/explore", response_model=DataExplorationResponse)
async def explore_csv(
    file: Annotated[UploadFile, File(...)],
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> DataExplorationResponse:
    """Explore dataset observations, reveal multi-metric pivots, check normality, and get test recommendations."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return saes.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data exploration error: {e!s}") from e


@router.post("/explore/export")
async def export_explore_csv(
    file: Annotated[UploadFile, File(...)],
    export_format: Annotated[str, Form()] = "csv",
    view_mode: Annotated[str, Form()] = "reveal_metrics",
    stat_format: Annotated[str, Form()] = "mean_std",
    normality_filter: Annotated[str, Form()] = "all",
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> Response:
    """Export an Observations & Data exploration view in PDF, Word (DOCX), LaTeX, CSV, Markdown, or HTML."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        exp_res = saes.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
        file_bytes, media_type, ext = export_exploration_table(
            exploration=exp_res,
            view_mode=view_mode,
            export_format=export_format,
            stat_format=stat_format,
            normality_filter=normality_filter,
        )
        out_name = f"saes_observations_{view_mode}.{ext}"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Observation export error: {e!s}") from e


@router.post("/profiles", response_model=PerformanceProfilesResponse)
async def compute_profiles_endpoint(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    max_tau: Annotated[float | None, Form()] = None,
    paired: Annotated[bool, Form()] = False,
    posthoc_correction: Annotated[str, Form()] = "holm",
) -> PerformanceProfilesResponse:
    """Compute Dolan-Moré performance profiles and multiplicity-adjusted Win/Tie/Loss meta-ranking."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return saes.compute_performance_profiles(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            max_tau=max_tau,
            paired=paired,
            posthoc_correction=posthoc_correction,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance profile error: {e!s}") from e


@router.post("/power", response_model=PowerAnalysisResponse)
async def compute_power_endpoint(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    target_power: Annotated[float, Form()] = 0.80,
    target_effect_size: Annotated[float, Form()] = 0.50,
    paired: Annotated[bool, Form()] = False,
) -> PowerAnalysisResponse:
    """Compute statistical power analysis, MDES sensitivity audit, and sample size recommendations."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return saes.analyze_statistical_power(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            target_power=target_power,
            target_effect_size=target_effect_size,
            paired=paired,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Power analysis error: {e!s}") from e


@router.post("/posthoc", response_model=PostHocResponse)
async def compute_posthoc_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    instance: Annotated[str | None, Form()] = None,
    method: Annotated[str, Form()] = "auto",
    control: Annotated[str | None, Form()] = None,
    p_adjust: Annotated[str, Form()] = "holm",
    alpha: Annotated[float, Form()] = 0.05,
) -> PostHocResponse:
    """Run Tukey HSD, Games-Howell, Dunnett, Dunn, or Conover-Iman post-hoc tests on uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return run_posthoc_for_dataset(
            dataset=dataset,
            metric=metric,
            instance=instance,
            method=method,
            control=control,
            p_adjust=p_adjust,
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Post-hoc analysis error: {e!s}") from e


@router.post("/tost", response_model=TOSTResponse)
async def compute_tost_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    algo_a: Annotated[str | None, Form()] = None,
    algo_b: Annotated[str | None, Form()] = None,
    low_eqbound: Annotated[float, Form()] = 0.05,
    high_eqbound: Annotated[float | None, Form()] = None,
    scope: Annotated[str, Form()] = "single-instance",
    instance: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    paired: Annotated[bool, Form()] = False,
    equal_var: Annotated[bool, Form()] = False,
) -> TOSTResponse:
    """Run Two One-Sided Tests (TOST) for statistical equivalence on uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return run_tost_for_dataset(
            dataset=dataset,
            metric=metric,
            algo_a=algo_a,
            algo_b=algo_b,
            low_eqbound=low_eqbound,
            high_eqbound=high_eqbound,
            scope=scope,
            instance=instance,
            alpha=alpha,
            paired=paired,
            equal_var=equal_var,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TOST equivalence error: {e!s}") from e


@router.post("/bayesian/correlated-ttest", response_model=BayesianCorrelatedTResponse)
async def compute_bayesian_correlated_t_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    algo_a: Annotated[str | None, Form()] = None,
    algo_b: Annotated[str | None, Form()] = None,
    scope: Annotated[str, Form()] = "single-instance",
    instance: Annotated[str | None, Form()] = None,
    rope: Annotated[float, Form()] = 0.01,
    rho: Annotated[float | None, Form()] = None,
) -> BayesianCorrelatedTResponse:
    """Run Bayesian Correlated t-test (Nadeau & Bengio 2003; Benavoli et al. 2017) on uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return run_bayesian_correlated_t_for_dataset(
            dataset=dataset,
            metric=metric,
            algo_a=algo_a,
            algo_b=algo_b,
            scope=scope,
            instance=instance,
            rope=rope,
            rho=rho,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bayesian correlated t-test error: {e!s}") from e


@router.post("/effect-sizes", response_model=EffectSizeForestResponse)
async def compute_effect_sizes_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    scope: Annotated[str, Form()] = "single-instance",
    instance: Annotated[str | None, Form()] = None,
    reference_algorithm: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    paired: Annotated[bool, Form()] = False,
    p_adjust: Annotated[str, Form()] = "holm",
) -> EffectSizeForestResponse:
    """Compute pairwise standardized effect sizes with exact 95% CIs on uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        target_metric = metric if (metric and metric in dataset.metrics) else dataset.metrics[0]
        return saes.statistics.effect_size_forest(
            dataset=dataset,
            metric=target_metric,
            scope=scope,
            instance=instance,
            reference_algorithm=reference_algorithm,
            alpha=alpha,
            paired=paired,
            p_adjust=p_adjust,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Effect-size computation error: {e!s}") from e


@router.post("/reproducibility", response_model=ReproducibilityResponse)
async def generate_reproducibility_endpoint(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    filename: Annotated[str, Form()] = "dataset.csv",
) -> ReproducibilityResponse:
    """Generate reproducible standalone Python script, Jupyter notebook, and FAIR manifest."""
    content = await file.read()
    try:
        raw_text = content.decode("utf-8", errors="replace")
        dataset = saes.load(io.BytesIO(content))
        return saes.generate_reproducibility_artifacts(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            filename=filename,
            raw_csv=raw_text,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reproducibility error: {e!s}") from e


@router.post("/plots/export")
async def export_plot_csv(
    file: Annotated[UploadFile, File(...)],
    plot_type: Annotated[str, Form()] = "boxplot",
    metric: Annotated[str | None, Form()] = None,
    instance: Annotated[str | None, Form()] = None,
    export_format: Annotated[str, Form()] = "png",
    scope: Annotated[str, Form()] = "all-instances",
    alpha: Annotated[float, Form()] = 0.05,
    rope_percent: Annotated[float, Form()] = 1.0,
    rope: Annotated[float | None, Form()] = None,
    prior_strength: Annotated[float, Form()] = 1.0,
    palette: Annotated[str | None, Form()] = None,
    dpi: Annotated[int, Form()] = 300,
) -> Response:
    """Generate and export a statistical plot or diagram in PNG, SVG, or PDF format."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        kwargs: dict[str, object] = {
            "alpha": alpha,
            "scope": scope,
            "rope_percent": rope_percent,
            "prior_strength": prior_strength,
        }
        if rope is not None:
            kwargs["rope"] = rope
        if palette:
            kwargs["palette"] = palette
        file_bytes, media_type, ext = saes.export_plot(
            dataset=dataset,
            plot_type=plot_type,
            metric=metric,
            instance=instance,
            export_format=export_format,
            dpi=dpi,
            **kwargs,
        )
        safe_metric = metric or (dataset.metrics[0] if dataset.metrics else "metric")
        out_name = f"saes_{plot_type}_{safe_metric}.{ext}"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Plot export error: {e!s}") from e
