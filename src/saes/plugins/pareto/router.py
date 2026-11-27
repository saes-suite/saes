"""Typed FastAPI APIRouter for the saes-pareto plugin, auto-mounted at /api/v1/plugins/pareto."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from saes.plugins.pareto.dominance import compute_coverage_and_representatives
from saes.plugins.pareto.eaf import compute_attainment, compute_eaf_difference
from saes.plugins.pareto.io import (
    _parse_numeric_matrix,
    create_demo_collection,
    load_csv,
    load_reference_front,
    load_zip,
)
from saes.plugins.pareto.models import (
    AttainmentResult,
    CoverageAndRepresentativeResult,
    EAFDifferenceResult,
    FrontCollection,
    FrontCollectionSummary,
    ParetoFront,
)
from saes.plugins.pareto.plots import export_front_plot, plot_fronts
from saes.server.storage import get_datasets_dir


def _get_collections_dir() -> Path:
    custom = os.getenv("SAES_PARETO_DIR")
    if custom:
        path = Path(custom).resolve()
    else:
        path = (get_datasets_dir() / "_pareto_collections").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._-")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid collection name.")
    return cleaned


def _save_collection(collection: FrontCollection) -> FrontCollectionSummary:
    safe = _safe_name(collection.name)
    coll = collection.model_copy(update={"name": safe})
    target = _get_collections_dir() / f"{safe}.json"
    target.write_text(coll.model_dump_json(), encoding="utf-8")
    return coll.summary()


def _load_stored_collection(name: str) -> FrontCollection:
    safe = _safe_name(name)
    target = _get_collections_dir() / f"{safe}.json"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Pareto collection '{name}' not found.")
    return FrontCollection.model_validate_json(target.read_text(encoding="utf-8"))


class CollectionListItem(BaseModel):
    name: str
    instances: list[str]
    algorithms: list[str]
    total_fronts: int
    total_points: int
    has_references: bool


class ParetoPlotRequest(BaseModel):
    instance: str
    mode: str = "attainment"
    algorithms: list[str] | None = None
    run_ids: list[int] | None = None
    show_reference: bool = True
    filter_dominated: bool = True
    run_opacity: float = Field(default=0.45, ge=0.05, le=1.0)
    show_extreme_bounds: bool = False
    show_accumulated: bool = True
    algo_a: str | None = None
    algo_b: str | None = None
    n_permutations: int = Field(default=300, ge=0, le=5000)
    seed: int = 42
    palette: str = "academic"
    paper_format: str = "golden"
    title: str | None = None
    title_mode: str = "full"
    legend_placement: str = "top"
    font_size: int = Field(default=13, ge=8, le=36)
    font_family: str = "Inter, -apple-system, sans-serif"
    label_mode: str = "all"
    bold_mode: str = "title-axes"
    show_grid: bool = True
    dark_theme: bool = False
    width_px: int | None = Field(default=None, ge=360, le=1800)
    height_px: int | None = Field(default=None, ge=280, le=1400)


class ParetoExportPlotRequest(ParetoPlotRequest):
    export_format: str = "pdf"
    dpi: int = Field(default=300, ge=72, le=600)


class ParetoPlotResponse(BaseModel):
    instance: str
    mode: str
    plotly_figure: dict[str, Any]
    eaf_difference: EAFDifferenceResult | None = None


class AttainmentRequest(BaseModel):
    instance: str
    algorithms: list[str] | None = None
    levels: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])


class EAFDiffRequest(BaseModel):
    instance: str
    algo_a: str
    algo_b: str
    grid_resolution: int = Field(default=120, ge=20, le=300)
    n_permutations: int = Field(default=500, ge=0, le=5000)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = 42


class CoverageRequest(BaseModel):
    instance: str
    algorithms: list[str] | None = None
    filter_dominated: bool = True


class ToDatasetRequest(BaseModel):
    dataset_filename: str = "pareto_indicators.csv"
    indicators: list[str] = Field(default_factory=lambda: ["HV", "IGD+", "IGD", "EPS", "Spacing"])
    instances: list[str] | None = None
    algorithms: list[str] | None = None
    seed: int = 42


class ToDatasetResponse(BaseModel):
    filename: str
    row_count: int
    algorithms: list[str]
    instances: list[str]
    metrics: list[str]


def build_router() -> APIRouter:
    """Create the FastAPI router mounted automatically by SAES under /api/v1/plugins/pareto."""
    router = APIRouter()

    @router.get("/collections", response_model=list[CollectionListItem])
    def list_collections() -> list[CollectionListItem]:
        cdir = _get_collections_dir()
        items: list[CollectionListItem] = []
        for path in sorted(cdir.glob("*.json")):
            try:
                coll = FrontCollection.model_validate_json(path.read_text(encoding="utf-8"))
                summ = coll.summary()
                items.append(
                    CollectionListItem(
                        name=summ.name,
                        instances=list(summ.instances),
                        algorithms=list(summ.algorithms),
                        total_fronts=summ.total_fronts,
                        total_points=summ.total_points,
                        has_references=len(summ.reference_instances) > 0,
                    )
                )
            except Exception:
                continue
        return items

    @router.post("/collections/demo", response_model=FrontCollectionSummary)
    def create_demo(
        name: str = "demo_moo_suite",
        n_runs: int = 12,
        seed: int = 42,
    ) -> FrontCollectionSummary:
        coll = create_demo_collection(n_runs=max(3, min(30, n_runs)), seed=seed, name=_safe_name(name))
        return _save_collection(coll)

    @router.post("/collections/upload", response_model=FrontCollectionSummary)
    async def upload_collection(
        file: UploadFile = File(...),  # noqa: B008
        reference_file: UploadFile | None = File(default=None),  # noqa: B008
        name: str | None = Form(default=None),
        instance: str = Form(default="Problem"),
    ) -> FrontCollectionSummary:
        raw_bytes = await file.read()
        fname = file.filename or "fronts.csv"
        coll_name = _safe_name(name or Path(fname).stem)
        try:
            if fname.lower().endswith(".zip"):
                coll = load_zip(raw_bytes, name=coll_name)
            else:
                ref_bytes = await reference_file.read() if reference_file is not None else None
                coll = load_csv(
                    raw_bytes,
                    reference_front=ref_bytes,
                    instance=instance,
                    name=coll_name,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _save_collection(coll)

    @router.post("/collections/create-multi", response_model=FrontCollectionSummary)
    async def create_multi_file_collection(
        name: str = Form(...),
        instance: str = Form(default="Problem"),
        algorithm_names: str = Form(...),
        front_files: list[UploadFile] = File(...),  # noqa: B008
        reference_file: UploadFile | None = File(default=None),  # noqa: B008
    ) -> FrontCollectionSummary:
        """Build a Pareto study from multiple individual front files grouped by algorithm plus an optional reference."""
        try:
            alg_list = json.loads(algorithm_names)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="algorithm_names must be a valid JSON list.") from exc

        if not isinstance(alg_list, list) or len(alg_list) != len(front_files) or not front_files:
            raise HTTPException(
                status_code=400,
                detail="algorithm_names must match the number of uploaded front_files.",
            )

        inst_clean = instance.strip() or "Problem"
        run_counters: dict[str, int] = {}
        fronts: list[ParetoFront] = []
        expected_obj: int | None = None
        obj_names_ref: tuple[str, ...] = ()

        for raw_alg, up_file in zip(alg_list, front_files, strict=True):
            alg_name = str(raw_alg).strip()
            if not alg_name:
                raise HTTPException(status_code=400, detail="Algorithm name cannot be empty.")
            raw_text = (await up_file.read()).decode("utf-8", errors="replace")
            try:
                arr, parsed_names = _parse_numeric_matrix(raw_text)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not parse numeric front '{up_file.filename}': {exc}",
                ) from exc
            if arr.shape[0] == 0:
                continue
            if expected_obj is None:
                expected_obj = int(arr.shape[1])
                obj_names_ref = parsed_names
            elif arr.shape[1] != expected_obj:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File '{up_file.filename}' has {arr.shape[1]} objectives, "
                        f"expected {expected_obj}."
                    ),
                )
            run_id = run_counters.get(alg_name, 0)
            run_counters[alg_name] = run_id + 1
            pts = tuple(tuple(float(v) for v in row) for row in arr)
            fronts.append(
                ParetoFront(
                    algorithm=alg_name,
                    instance=inst_clean,
                    run_id=run_id,
                    points=pts,
                    objective_names=obj_names_ref,
                )
            )

        if not fronts:
            raise HTTPException(status_code=400, detail="No valid objective points found in uploaded front files.")

        refs: dict[str, ParetoFront] = {}
        if reference_file is not None:
            ref_bytes = await reference_file.read()
            loaded_refs = load_reference_front(ref_bytes, instance=inst_clean)
            for ref_inst, ref_pf in loaded_refs.items():
                if expected_obj is not None and ref_pf.n_objectives == expected_obj:
                    ref_pf = ref_pf.model_copy(update={"objective_names": obj_names_ref})
                refs[ref_inst] = ref_pf

        coll = FrontCollection(name=_safe_name(name), fronts=tuple(fronts), reference_fronts=refs)
        return _save_collection(coll)

    @router.get("/collections/{name}", response_model=FrontCollectionSummary)
    def get_collection_summary(name: str) -> FrontCollectionSummary:
        return _load_stored_collection(name).summary()

    @router.delete("/collections/{name}")
    def delete_collection(name: str) -> dict[str, str]:
        safe = _safe_name(name)
        target = _get_collections_dir() / f"{safe}.json"
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Pareto collection '{name}' not found.")
        target.unlink()
        return {"deleted": safe}

    @router.post("/collections/{name}/plot", response_model=ParetoPlotResponse)
    def generate_plot(name: str, req: ParetoPlotRequest) -> ParetoPlotResponse:
        coll = _load_stored_collection(name)
        try:
            fig = plot_fronts(
                coll,
                instance=req.instance,
                mode=req.mode,
                algorithms=req.algorithms,
                run_ids=req.run_ids,
                show_reference=req.show_reference,
                filter_dominated=req.filter_dominated,
                run_opacity=req.run_opacity,
                show_extreme_bounds=req.show_extreme_bounds,
                show_accumulated=req.show_accumulated,
                algo_a=req.algo_a,
                algo_b=req.algo_b,
                n_permutations=req.n_permutations,
                seed=req.seed,
                palette=req.palette,
                paper_format=req.paper_format,
                title=req.title,
                title_mode=req.title_mode,
                legend_placement=req.legend_placement,
                font_size=req.font_size,
                font_family=req.font_family,
                label_mode=req.label_mode,
                bold_mode=req.bold_mode,
                show_grid=req.show_grid,
                dark_theme=req.dark_theme,
                width_px=req.width_px,
                height_px=req.height_px,
                interactive=True,
            )
            eaf_diff_res: EAFDifferenceResult | None = None
            if req.mode.strip().lower() == "eaf-diff":
                algs = req.algorithms or coll.summary().instance_details[req.instance].algorithms
                a_name = req.algo_a or algs[0]
                b_name = req.algo_b or next((a for a in algs if a != a_name), algs[0])
                eaf_diff_res = compute_eaf_difference(
                    coll,
                    req.instance,
                    a_name,
                    b_name,
                    n_permutations=req.n_permutations,
                    seed=req.seed,
                )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        fig_json: dict[str, Any] = json.loads(fig.to_json())  # type: ignore[union-attr]
        return ParetoPlotResponse(
            instance=req.instance,
            mode=req.mode,
            plotly_figure=fig_json,
            eaf_difference=eaf_diff_res,
        )

    @router.post("/collections/{name}/export-plot")
    def export_plot_endpoint(name: str, req: ParetoExportPlotRequest) -> Response:
        coll = _load_stored_collection(name)
        try:
            raw_bytes, media_type, ext = export_front_plot(
                coll,
                instance=req.instance,
                mode=req.mode,
                algorithms=req.algorithms,
                run_ids=req.run_ids,
                show_reference=req.show_reference,
                filter_dominated=req.filter_dominated,
                run_opacity=req.run_opacity,
                show_extreme_bounds=req.show_extreme_bounds,
                show_accumulated=req.show_accumulated,
                algo_a=req.algo_a,
                algo_b=req.algo_b,
                n_permutations=req.n_permutations,
                seed=req.seed,
                palette=req.palette,
                paper_format=req.paper_format,
                title=req.title,
                title_mode=req.title_mode,
                legend_placement=req.legend_placement,
                font_size=req.font_size,
                font_family=req.font_family,
                label_mode=req.label_mode,
                bold_mode=req.bold_mode,
                show_grid=req.show_grid,
                dark_theme=req.dark_theme,
                width_px=req.width_px,
                height_px=req.height_px,
                export_format=req.export_format,
                dpi=req.dpi,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dl_name = f"pareto_{_safe_name(req.instance)}_{_safe_name(req.mode)}.{ext}"
        return Response(
            content=raw_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
        )

    @router.post("/collections/{name}/attainment", response_model=AttainmentResult)
    def attainment_endpoint(name: str, req: AttainmentRequest) -> AttainmentResult:
        coll = _load_stored_collection(name)
        try:
            return compute_attainment(
                coll,
                req.instance,
                algorithms=req.algorithms,
                levels=tuple(req.levels),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/collections/{name}/eaf-diff", response_model=EAFDifferenceResult)
    def eaf_diff_endpoint(name: str, req: EAFDiffRequest) -> EAFDifferenceResult:
        coll = _load_stored_collection(name)
        try:
            return compute_eaf_difference(
                coll,
                req.instance,
                req.algo_a,
                req.algo_b,
                grid_resolution=req.grid_resolution,
                n_permutations=req.n_permutations,
                alpha=req.alpha,
                seed=req.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/collections/{name}/coverage", response_model=CoverageAndRepresentativeResult)
    def coverage_endpoint(name: str, req: CoverageRequest) -> CoverageAndRepresentativeResult:
        coll = _load_stored_collection(name)
        try:
            return compute_coverage_and_representatives(
                coll,
                req.instance,
                algorithms=req.algorithms,
                filter_dominated=req.filter_dominated,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/collections/{name}/to-dataset", response_model=ToDatasetResponse)
    def to_dataset_endpoint(name: str, req: ToDatasetRequest) -> ToDatasetResponse:
        coll = _load_stored_collection(name)
        try:
            ds = coll.to_indicators_dataset(
                indicators=req.indicators,
                instances=req.instances,
                algorithms=req.algorithms,
                seed=req.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        fname = _safe_name(req.dataset_filename)
        if not fname.lower().endswith(".csv"):
            fname = f"{fname}.csv"
        out_path = get_datasets_dir() / fname
        ds.df.to_csv(out_path, index=False)

        return ToDatasetResponse(
            filename=fname,
            row_count=len(ds.df),
            algorithms=ds.algorithms,
            instances=ds.instances,
            metrics=ds.metrics,
        )

    return router
