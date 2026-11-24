import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile

import saes
from saes.io.csv_reader import CSVReader
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
    DatasetDetailSchema,
    DatasetItemSchema,
    DatasetUpdateSchema,
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
from saes.server.storage import get_datasets_dir
from saes.stats.explorer import export_exploration_table
from saes.stats.tables import export_statistical_table
from saes.validation.validator import DatasetValidator

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


def _count_csv_rows(file_path: Path) -> int:
    """Quickly count non-empty data rows in a CSV file, excluding header."""
    try:
        count = 0
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    count += 1
        return max(0, count - 1)
    except Exception:
        return 0


def _build_dataset_item(file_path: Path) -> DatasetItemSchema:
    """Build metadata item for a dataset file."""
    stat = file_path.stat()
    return DatasetItemSchema(
        name=file_path.stem,
        filename=file_path.name,
        path=str(file_path),
        size_bytes=stat.st_size,
        row_count=_count_csv_rows(file_path),
        modified_at=stat.st_mtime,
    )


@router.get("", response_model=list[DatasetItemSchema])
def list_datasets() -> list[DatasetItemSchema]:
    """List all saved benchmark datasets in the backend storage directory."""
    datasets_dir = get_datasets_dir()
    items: list[DatasetItemSchema] = []
    for p in sorted(datasets_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            items.append(_build_dataset_item(p))
    return items


@router.post("/upload", response_model=DatasetItemSchema)
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> DatasetItemSchema:
    """Upload, validate, and save a CSV dataset file into backend storage (gitignore-backed)."""
    filename = Path(file.filename or "uploaded_benchmark.csv").name
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Strict structural and schema validation before accepting upload
    try:
        df = CSVReader().read(io.BytesIO(content))
        report = DatasetValidator().validate(df)
        missing_col_errors = [e for e in report.errors if e.issue_type == "MissingColumns"]
        if missing_col_errors:
            error_msgs = "; ".join(e.message for e in missing_col_errors)
            raise HTTPException(
                status_code=422,
                detail=f"Dataset validation failed: {error_msgs}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV dataset: {e!s}") from e

    datasets_dir = get_datasets_dir()
    target_path = datasets_dir / filename
    target_path.write_bytes(content)
    return _build_dataset_item(target_path)


@router.put("/{filename}", response_model=DatasetDetailSchema)
def update_dataset(filename: str, payload: DatasetUpdateSchema) -> DatasetDetailSchema:
    """Save or update CSV dataset content directly in backend storage."""
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    file_path = get_datasets_dir() / safe_name
    try:
        file_path.write_text(payload.content, encoding="utf-8")
        stat = file_path.stat()
        return DatasetDetailSchema(
            name=file_path.stem,
            filename=file_path.name,
            path=str(file_path),
            size_bytes=stat.st_size,
            row_count=_count_csv_rows(file_path),
            content=payload.content,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving dataset: {e!s}") from e


@router.get("/{filename}", response_model=DatasetDetailSchema)
def get_dataset(filename: str) -> DatasetDetailSchema:
    """Retrieve full content and metadata for a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        stat = file_path.stat()
        return DatasetDetailSchema(
            name=file_path.stem,
            filename=file_path.name,
            path=str(file_path),
            size_bytes=stat.st_size,
            row_count=_count_csv_rows(file_path),
            content=content,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {e!s}") from e


@router.post("/{filename}/validate", response_model=ValidationResponse)
def validate_dataset_from_path(filename: str) -> ValidationResponse:
    """Validate a saved benchmark dataset by loading directly from its server path."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        report = saes.validate(dataset)
        return ValidationResponse(
            is_valid=report.is_valid,
            error_count=len(report.errors),
            warning_count=len(report.warnings),
            issues=report.issues,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset validation error: {e!s}") from e


@router.post("/{filename}/analyze", response_model=AnalysisResponse)
def analyze_dataset_from_path(
    filename: str,
    alpha: Annotated[float, Form()] = 0.05,
    paired: Annotated[bool, Form()] = False,
) -> AnalysisResponse:
    """Execute statistical analysis on a saved dataset by loading directly from its server path."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        results = build_cell_analysis_schemas(dataset, alpha=alpha, paired=paired)
        return AnalysisResponse(cells_count=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Statistical analysis error: {e!s}") from e


@router.post("/{filename}/ranking", response_model=RankingAnalysisResponse)
def ranking_dataset_from_path(
    filename: str,
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
    """Compute Friedman, Aligned-Ranks, Quade, RM-ANOVA, Bayesian Simplex ROPE, and Plackett-Luce tests."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/tables", response_model=StatisticalTableResponse)
def get_dataset_table(
    filename: str,
    response: Response,
    metric: Annotated[str | None, Query()] = None,
    table_type: Annotated[str, Query()] = "instance_algorithm",
    test_type: Annotated[str, Query()] = "auto",
    posthoc_correction: Annotated[str, Query()] = "holm",
    summary_stat: Annotated[str, Query()] = "auto",
    reference_algorithm: Annotated[str | None, Query()] = None,
    scope: Annotated[str, Query()] = "all-instances",
    selected_instance: Annotated[str | None, Query()] = None,
    algorithms_filter: Annotated[str | None, Query()] = None,
    instances_filter: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    precision: Annotated[int, Query()] = 4,
    paired: Annotated[bool, Query()] = False,
) -> StatisticalTableResponse:
    """Generate a statistical benchmark table via HTTP GET (cacheable by standard HTTP clients)."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        response.headers["Cache-Control"] = "public, max-age=3600"
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
            paired=paired,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table generation error: {e!s}") from e


@router.post("/{filename}/tables", response_model=StatisticalTableResponse)
def generate_dataset_table(
    filename: str,
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
    """Generate a statistical benchmark table (ANOVA, Friedman, Wilcoxon, Pairwise) for a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/tables/export")
def export_dataset_table_get(
    filename: str,
    export_format: Annotated[str, Query()] = "latex",
    metric: Annotated[str | None, Query()] = None,
    table_type: Annotated[str, Query()] = "instance_algorithm",
    test_type: Annotated[str, Query()] = "auto",
    posthoc_correction: Annotated[str, Query()] = "holm",
    summary_stat: Annotated[str, Query()] = "auto",
    reference_algorithm: Annotated[str | None, Query()] = None,
    scope: Annotated[str, Query()] = "all-instances",
    selected_instance: Annotated[str | None, Query()] = None,
    algorithms_filter: Annotated[str | None, Query()] = None,
    instances_filter: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    precision: Annotated[int, Query()] = 4,
    latex_document_mode: Annotated[str, Query()] = "standalone",
    latex_notation: Annotated[str, Query()] = "scientific",
    paired: Annotated[bool, Query()] = False,
    pdf_title: Annotated[str | None, Query()] = None,
    pdf_caption: Annotated[str | None, Query()] = None,
    pdf_theme: Annotated[str, Query()] = "slate",
    pdf_orientation: Annotated[str, Query()] = "landscape",
    pdf_font_size: Annotated[float, Query()] = 8.5,
    pdf_highlight_best: Annotated[bool, Query()] = True,
    pdf_show_zebra: Annotated[bool, Query()] = True,
    pdf_grid_style: Annotated[str, Query()] = "full",
) -> Response:
    """Export a statistical benchmark table via HTTP GET."""
    return export_dataset_table(
        filename=filename,
        export_format=export_format,
        metric=metric,
        table_type=table_type,
        test_type=test_type,
        posthoc_correction=posthoc_correction,
        summary_stat=summary_stat,
        reference_algorithm=reference_algorithm,
        scope=scope,
        selected_instance=selected_instance,
        algorithms_filter=algorithms_filter,
        instances_filter=instances_filter,
        alpha=alpha,
        precision=precision,
        latex_document_mode=latex_document_mode,
        latex_notation=latex_notation,
        paired=paired,
        pdf_title=pdf_title,
        pdf_caption=pdf_caption,
        pdf_theme=pdf_theme,
        pdf_orientation=pdf_orientation,
        pdf_font_size=pdf_font_size,
        pdf_highlight_best=pdf_highlight_best,
        pdf_show_zebra=pdf_show_zebra,
        pdf_grid_style=pdf_grid_style,
    )


@router.post("/{filename}/tables/export")
def export_dataset_table(
    filename: str,
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
    """Generate and export a statistical benchmark table for a saved dataset in PDF, DOCX, LaTeX, CSV, MD, or HTML."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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
            headers={
                "Content-Disposition": f'attachment; filename="{out_name}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table export error: {e!s}") from e


@router.post("/{filename}/plots/export")
def export_dataset_plot(
    filename: str,
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
    """Generate and export a statistical plot or diagram for a saved dataset in PNG, SVG, or PDF."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/explore", response_model=DataExplorationResponse)
def get_explore_dataset_from_path(
    filename: str,
    response: Response,
    metrics: Annotated[str | None, Query()] = None,
    algorithms: Annotated[str | None, Query()] = None,
    instances: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
) -> DataExplorationResponse:
    """Explore dataset observations via HTTP GET (cacheable by standard HTTP clients)."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        response.headers["Cache-Control"] = "public, max-age=3600"
        return saes.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data exploration error: {e!s}") from e


@router.post("/{filename}/explore", response_model=DataExplorationResponse)
def explore_dataset_from_path(
    filename: str,
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> DataExplorationResponse:
    """Explore observations, reveal multi-metric pivots, check normality, and recommend tests for a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        return saes.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data exploration error: {e!s}") from e


@router.get("/{filename}/explore/export")
def export_explore_dataset_get(
    filename: str,
    export_format: Annotated[str, Query()] = "csv",
    view_mode: Annotated[str, Query()] = "reveal_metrics",
    stat_format: Annotated[str, Query()] = "mean_std",
    normality_filter: Annotated[str, Query()] = "all",
    metrics: Annotated[str | None, Query()] = None,
    algorithms: Annotated[str | None, Query()] = None,
    instances: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
) -> Response:
    """Export an Observations & Data exploration view via HTTP GET."""
    return export_explore_dataset(
        filename=filename,
        export_format=export_format,
        view_mode=view_mode,
        stat_format=stat_format,
        normality_filter=normality_filter,
        metrics=metrics,
        algorithms=algorithms,
        instances=instances,
        alpha=alpha,
    )


@router.post("/{filename}/explore/export")
def export_explore_dataset(
    filename: str,
    export_format: Annotated[str, Form()] = "csv",
    view_mode: Annotated[str, Form()] = "reveal_metrics",
    stat_format: Annotated[str, Form()] = "mean_std",
    normality_filter: Annotated[str, Form()] = "all",
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> Response:
    """Export an Observations & Data exploration view for a saved dataset in PDF, DOCX, LaTeX, CSV, MD, or HTML."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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
            headers={
                "Content-Disposition": f'attachment; filename="{out_name}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Observation export error: {e!s}") from e


@router.get("/{filename}/profiles", response_model=PerformanceProfilesResponse)
@router.post("/{filename}/profiles", response_model=PerformanceProfilesResponse)
def get_dataset_profiles(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    max_tau: Annotated[float | None, Query()] = None,
    paired: Annotated[bool, Query()] = False,
    posthoc_correction: Annotated[str, Query()] = "holm",
) -> PerformanceProfilesResponse:
    """Compute Dolan-Moré performance profiles and multiplicity-adjusted Win/Tie/Loss matrix."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/power", response_model=PowerAnalysisResponse)
@router.post("/{filename}/power", response_model=PowerAnalysisResponse)
def get_dataset_power(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    target_power: Annotated[float, Query()] = 0.80,
    target_effect_size: Annotated[float, Query()] = 0.50,
    paired: Annotated[bool, Query()] = False,
) -> PowerAnalysisResponse:
    """Compute statistical power analysis, MDES sensitivity audit, and sample size planning."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/posthoc", response_model=PostHocResponse)
@router.post("/{filename}/posthoc", response_model=PostHocResponse)
def get_dataset_posthoc(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    instance: Annotated[str | None, Query()] = None,
    method: Annotated[str, Query()] = "auto",
    control: Annotated[str | None, Query()] = None,
    p_adjust: Annotated[str, Query()] = "holm",
    alpha: Annotated[float, Query()] = 0.05,
) -> PostHocResponse:
    """Run Tukey HSD, Games-Howell, Dunnett, Dunn, or Conover-Iman post-hoc tests on a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/tost", response_model=TOSTResponse)
@router.post("/{filename}/tost", response_model=TOSTResponse)
def get_dataset_tost(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    algo_a: Annotated[str | None, Query()] = None,
    algo_b: Annotated[str | None, Query()] = None,
    low_eqbound: Annotated[float, Query()] = 0.05,
    high_eqbound: Annotated[float | None, Query()] = None,
    scope: Annotated[str, Query()] = "single-instance",
    instance: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    paired: Annotated[bool, Query()] = False,
    equal_var: Annotated[bool, Query()] = False,
) -> TOSTResponse:
    """Run Two One-Sided Tests (TOST) for statistical equivalence on a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/bayesian/correlated-ttest", response_model=BayesianCorrelatedTResponse)
@router.post("/{filename}/bayesian/correlated-ttest", response_model=BayesianCorrelatedTResponse)
def get_dataset_bayesian_correlated_t(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    algo_a: Annotated[str | None, Query()] = None,
    algo_b: Annotated[str | None, Query()] = None,
    scope: Annotated[str, Query()] = "single-instance",
    instance: Annotated[str | None, Query()] = None,
    rope: Annotated[float, Query()] = 0.01,
    rho: Annotated[float | None, Query()] = None,
) -> BayesianCorrelatedTResponse:
    """Run Bayesian Correlated t-test (Nadeau & Bengio 2003; Benavoli et al. 2017) on a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/effect-sizes", response_model=EffectSizeForestResponse)
@router.post("/{filename}/effect-sizes", response_model=EffectSizeForestResponse)
def get_dataset_effect_sizes(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    scope: Annotated[str, Query()] = "single-instance",
    instance: Annotated[str | None, Query()] = None,
    reference_algorithm: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    paired: Annotated[bool, Query()] = False,
    p_adjust: Annotated[str, Query()] = "holm",
) -> EffectSizeForestResponse:
    """Compute pairwise standardized effect sizes with exact 95% CIs for a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
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


@router.get("/{filename}/reproducibility", response_model=ReproducibilityResponse)
@router.post("/{filename}/reproducibility", response_model=ReproducibilityResponse)
def get_dataset_reproducibility(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
) -> ReproducibilityResponse:
    """Generate reproducible Python script, Jupyter notebook, and FAIR manifest for a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        dataset = saes.load(file_path)
        return saes.generate_reproducibility_artifacts(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            filename=safe_name,
            raw_csv=raw_text,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reproducibility error: {e!s}") from e


@router.delete("/{filename}")
def delete_dataset(filename: str) -> dict[str, str]:
    """Delete a saved dataset from the backend storage directory."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        file_path.unlink()
        return {"status": "deleted", "filename": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {e!s}") from e
