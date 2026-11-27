"""Flexible I/O loaders for Pareto fronts (tidy CSV, jMetal/PlatEMO folders, .pf reference files, ZIP, demo suites)."""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from saes.plugins.pareto.dominance import filter_nondominated_array
from saes.plugins.pareto.models import FrontCollection, ParetoFront

_REFERENCE_ALGO_ALIASES = {"reference", "ref", "pf", "true", "true_front", "pareto_front", "reference_front"}
_FUN_PATTERN = re.compile(r"^(?:FUN|fun|front|run)[._-]?(\d+)\.(?:csv|tsv|txt|pf)$")


def _parse_numeric_matrix(raw_text: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Parse a pure numeric front file (comma, tab, or whitespace delimited, with or without header)."""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return np.empty((0, 2), dtype=np.float64), ()

    first_tokens = re.split(r"[\s,;]+", lines[0])
    has_header = False
    for tok in first_tokens:
        try:
            float(tok)
        except ValueError:
            has_header = True
            break

    obj_names: tuple[str, ...] = ()
    data_lines = lines
    if has_header:
        obj_names = tuple(t.strip() for t in first_tokens if t.strip())
        data_lines = lines[1:]

    rows: list[list[float]] = []
    for ln in data_lines:
        parts = [p for p in re.split(r"[\s,;]+", ln) if p]
        if not parts:
            continue
        rows.append([float(p) for p in parts])

    if not rows:
        return np.empty((0, max(2, len(obj_names))), dtype=np.float64), obj_names
    arr = np.asarray(rows, dtype=np.float64)
    if not obj_names or len(obj_names) != arr.shape[1]:
        obj_names = tuple(f"f{i + 1}" for i in range(arr.shape[1]))
    return arr, obj_names


def load_reference_front(
    source: str | Path | bytes | io.IOBase,
    *,
    instance: str = "Problem",
    minimize: tuple[bool, ...] | list[bool] | None = None,
) -> dict[str, ParetoFront]:
    """Load one or more reference fronts from a .pf/.tsv/.csv file or buffer.

    Supports both:
    - Multi-instance tidy CSV with an ``instance`` column plus numeric objective columns.
    - Single-instance raw numeric ``.pf``/``.tsv``/``.csv`` matrix (jMetal / PlatEMO style).
    """
    if isinstance(source, bytes):
        raw_text = source.decode("utf-8")
    elif isinstance(source, (str, Path)):
        raw_text = Path(source).read_text(encoding="utf-8")
    else:
        content = source.read()
        raw_text = content.decode("utf-8") if isinstance(content, bytes) else str(content)

    min_tuple = tuple(minimize) if minimize is not None else ()

    # Try tidy CSV with 'instance' column first
    try:
        df = pd.read_csv(io.StringIO(raw_text))
        lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
        if "instance" in lower_map:
            inst_col = lower_map["instance"]
            skip = {lower_map.get(k) for k in ("algorithm", "instance", "run_id", "run", "iteration", "is_reference")}
            obj_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
            if len(obj_cols) >= 2:
                refs: dict[str, ParetoFront] = {}
                for inst_name, grp in df.groupby(inst_col, sort=True):
                    arr = grp[obj_cols].to_numpy(dtype=np.float64)
                    pts = tuple(tuple(float(v) for v in row) for row in arr)
                    refs[str(inst_name).strip()] = ParetoFront(
                        algorithm="Reference",
                        instance=str(inst_name).strip(),
                        run_id=0,
                        points=pts,
                        objective_names=tuple(str(c) for c in obj_cols),
                        minimize=min_tuple,
                        is_reference=True,
                    )
                if refs:
                    return refs
    except Exception:
        pass

    arr, obj_names = _parse_numeric_matrix(raw_text)
    pts = tuple(tuple(float(v) for v in row) for row in arr)
    return {
        instance: ParetoFront(
            algorithm="Reference",
            instance=instance,
            run_id=0,
            points=pts,
            objective_names=obj_names,
            minimize=min_tuple,
            is_reference=True,
        )
    }


def load_csv(
    source: str | Path | bytes | io.IOBase,
    *,
    reference_front: str | Path | bytes | io.IOBase | None = None,
    instance: str = "Problem",
    minimize: tuple[bool, ...] | list[bool] | None = None,
    name: str | None = None,
) -> FrontCollection:
    """Load multi-run Pareto fronts from a tidy CSV file or buffer.

    Expected columns:
    - ``algorithm`` (required)
    - ``instance`` (optional; defaults to ``instance`` argument)
    - ``run_id`` or ``run`` or ``iteration`` (optional; defaults to 0)
    - Two or more numeric objective columns (e.g. ``f1, f2[, f3, ...]``)

    Rows whose ``algorithm`` is ``Reference`` (or ``PF`` / ``TrueFront``) are automatically
    promoted to ``reference_fronts`` for their respective instance.
    """
    if isinstance(source, bytes):
        df = pd.read_csv(io.BytesIO(source))
        coll_name = name or "uploaded_fronts"
    elif isinstance(source, (str, Path)):
        p = Path(source)
        df = pd.read_csv(p)
        coll_name = name or p.stem
    else:
        raw_stream = source.read()
        raw_text = raw_stream.decode("utf-8") if isinstance(raw_stream, bytes) else str(raw_stream)
        df = pd.read_csv(io.StringIO(raw_text))
        coll_name = name or "stream_fronts"

    if df.empty:
        raise ValueError("Input Pareto front CSV is empty.")

    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    if "algorithm" not in lower_map:
        raise ValueError("Pareto front CSV must include an 'algorithm' column.")

    alg_col = lower_map["algorithm"]
    inst_col = lower_map.get("instance") or lower_map.get("problem")
    run_col = lower_map.get("run_id") or lower_map.get("run") or lower_map.get("iteration")
    meta_cols = {c for c in (alg_col, inst_col, run_col) if c is not None}

    obj_cols: list[str] = []
    for col in df.columns:
        if col in meta_cols:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().all():
            df[col] = coerced
            obj_cols.append(str(col))

    if len(obj_cols) < 2:
        raise ValueError(
            f"Pareto front CSV requires at least 2 numeric objective columns, found {len(obj_cols)}: {obj_cols}."
        )

    min_tuple = tuple(minimize) if minimize is not None else ()
    fronts: list[ParetoFront] = []
    refs: dict[str, ParetoFront] = {}

    if inst_col is None:
        df = df.copy()
        df["__instance__"] = instance
        inst_col = "__instance__"
    if run_col is None:
        df = df.copy()
        df["__run_id__"] = 0
        run_col = "__run_id__"

    for (alg_val, inst_val, run_val), grp in df.groupby([alg_col, inst_col, run_col], sort=True):
        alg_str = str(alg_val).strip()
        inst_str = str(inst_val).strip()
        run_int = max(0, int(str(run_val)))
        arr = grp[obj_cols].to_numpy(dtype=np.float64)
        pts = tuple(tuple(float(v) for v in row) for row in arr)

        if alg_str.lower() in _REFERENCE_ALGO_ALIASES:
            refs[inst_str] = ParetoFront(
                algorithm="Reference",
                instance=inst_str,
                run_id=0,
                points=pts,
                objective_names=tuple(obj_cols),
                minimize=min_tuple,
                is_reference=True,
            )
        else:
            fronts.append(
                ParetoFront(
                    algorithm=alg_str,
                    instance=inst_str,
                    run_id=run_int,
                    points=pts,
                    objective_names=tuple(obj_cols),
                    minimize=min_tuple,
                    is_reference=False,
                )
            )

    if reference_front is not None:
        default_inst = fronts[0].instance if fronts else instance
        loaded_refs = load_reference_front(reference_front, instance=default_inst, minimize=minimize)
        for ref_inst, ref_pf in loaded_refs.items():
            # Align objective names with fronts if raw .pf used generic f1, f2
            if ref_pf.n_objectives == len(obj_cols):
                ref_pf = ref_pf.model_copy(update={"objective_names": tuple(obj_cols)})
            refs[ref_inst] = ref_pf

    return FrontCollection(name=coll_name, fronts=tuple(fronts), reference_fronts=refs)


def load_directory(
    root_dir: str | Path,
    *,
    reference_dir: str | Path | None = None,
    minimize: tuple[bool, ...] | list[bool] | None = None,
    name: str | None = None,
) -> FrontCollection:
    """Load Pareto fronts from a jMetal / PlatEMO experiment directory tree.

    Expected layout::

        root_dir/
          <algorithm>/
            <instance>/
              FUN.0.tsv  (or FUN.0.csv, FUN0.tsv, run_0.csv)
              FUN.1.tsv
        reference_dir/   (or root_dir/reference_fronts/ or root_dir/referenceFronts/)
          <instance>.pf  (or <instance>.csv, <instance>.tsv)
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {root}")

    min_tuple = tuple(minimize) if minimize is not None else ()
    fronts: list[ParetoFront] = []
    refs: dict[str, ParetoFront] = {}

    ref_dirs: list[Path] = []
    if reference_dir is not None:
        ref_dirs.append(Path(reference_dir).resolve())
    for candidate in ("reference_fronts", "referenceFronts", "reference", "pf"):
        cpath = root / candidate
        if cpath.is_dir():
            ref_dirs.append(cpath)

    ref_dir_set = set(ref_dirs)
    for rdir in ref_dirs:
        for rf in sorted(rdir.iterdir()):
            if rf.is_file() and rf.suffix.lower() in {".pf", ".csv", ".tsv", ".txt"}:
                inst_name = rf.stem
                loaded = load_reference_front(rf, instance=inst_name, minimize=minimize)
                refs.update(loaded)

    for alg_dir in sorted(root.iterdir()):
        if not alg_dir.is_dir() or alg_dir in ref_dir_set or alg_dir.name.startswith("."):
            continue
        alg_name = alg_dir.name
        for inst_dir in sorted(alg_dir.iterdir()):
            if not inst_dir.is_dir() or inst_dir.name.startswith("."):
                continue
            inst_name = inst_dir.name
            for fpath in sorted(inst_dir.iterdir()):
                if not fpath.is_file():
                    continue
                match = _FUN_PATTERN.match(fpath.name)
                if not match:
                    continue
                run_id = int(match.group(1))
                arr, obj_names = _parse_numeric_matrix(fpath.read_text(encoding="utf-8"))
                if arr.shape[0] == 0:
                    continue
                pts = tuple(tuple(float(v) for v in row) for row in arr)
                fronts.append(
                    ParetoFront(
                        algorithm=alg_name,
                        instance=inst_name,
                        run_id=run_id,
                        points=pts,
                        objective_names=obj_names,
                        minimize=min_tuple,
                    )
                )

    if not fronts:
        raise ValueError(
            f"No Pareto front files (e.g. <algorithm>/<instance>/FUN.<run>.tsv) found under '{root}'."
        )

    return FrontCollection(name=name or root.name, fronts=tuple(fronts), reference_fronts=refs)


def load_zip(
    source: str | Path | bytes,
    *,
    minimize: tuple[bool, ...] | list[bool] | None = None,
    name: str = "uploaded_archive",
) -> FrontCollection:
    """Safely extract and load a ZIP archive containing either jMetal folders or tidy CSV + reference fronts."""
    buf: Any = io.BytesIO(source) if isinstance(source, bytes) else source

    with tempfile.TemporaryDirectory(prefix="saes_pareto_zip_") as tmpdir:
        tmp_root = Path(tmpdir).resolve()
        with zipfile.ZipFile(buf, "r") as zf:
            for member in zf.infolist():
                if member.filename.startswith("__MACOSX") or "/." in member.filename:
                    continue
                target = (tmp_root / member.filename).resolve()
                if not str(target).startswith(str(tmp_root)):
                    raise ValueError(f"Unsafe ZIP entry path rejected: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member.filename))

        # Unwrap single top-level directory if present
        top_entries = [p for p in tmp_root.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
        extract_root = top_entries[0] if len(top_entries) == 1 and top_entries[0].is_dir() else tmp_root

        # Try jMetal directory structure first
        try:
            return load_directory(extract_root, minimize=minimize, name=name)
        except ValueError:
            pass

        # Fallback: tidy CSV + optional .pf files inside archive
        csv_files = sorted(extract_root.rglob("*.csv"))
        if not csv_files:
            raise ValueError("ZIP archive contains neither jMetal FUN.* directories nor a tidy CSV front file.")
        main_csv = csv_files[0]
        coll = load_csv(main_csv, minimize=minimize, name=name)

        refs = dict(coll.reference_fronts)
        for pf_file in sorted(extract_root.rglob("*.pf")):
            refs.update(load_reference_front(pf_file, instance=pf_file.stem, minimize=minimize))

        return coll.model_copy(update={"reference_fronts": refs})


def create_demo_collection(
    *,
    n_runs: int = 12,
    seed: int = 42,
    name: str = "demo_moo_suite",
) -> FrontCollection:
    """Generate a realistic multi-objective benchmark suite (ZDT1, ZDT2, ZDT3, DTLZ2) with 4 algorithms.

    Provides exact analytical reference fronts plus realistic stochastic approximation runs across
    NSGA-II, MOEA/D, SPEA2, and SMPSO with distinct regional convergence/diversity trade-offs so
    attainment bands, EAF differences, permutation tests, and SAES indicator analysis are immediately demonstrable.
    """
    rng = np.random.default_rng(seed)
    fronts: list[ParetoFront] = []
    refs: dict[str, ParetoFront] = {}

    def _add_ref(inst: str, arr: np.ndarray, obj_names: tuple[str, ...]) -> None:
        nd, _ = filter_nondominated_array(arr)
        pts = tuple(tuple(float(v) for v in row) for row in nd)
        refs[inst] = ParetoFront(
            algorithm="Reference",
            instance=inst,
            run_id=0,
            points=pts,
            objective_names=obj_names,
            minimize=tuple(True for _ in obj_names),
            is_reference=True,
        )

    def _add_run(alg: str, inst: str, run_id: int, arr: np.ndarray, obj_names: tuple[str, ...]) -> None:
        nd, _ = filter_nondominated_array(arr)
        pts = tuple(tuple(float(v) for v in row) for row in nd)
        fronts.append(
            ParetoFront(
                algorithm=alg,
                instance=inst,
                run_id=run_id,
                points=pts,
                objective_names=obj_names,
                minimize=tuple(True for _ in obj_names),
                is_reference=False,
            )
        )

    # 1. ZDT1 (2D convex): f2 = 1 - sqrt(f1)
    x_ref = np.linspace(0.0, 1.0, 180)
    _add_ref("ZDT1", np.column_stack([x_ref, 1.0 - np.sqrt(x_ref)]), ("f1", "f2"))
    zdt1_profiles = {
        "NSGA-II": (0.022, 0.014, 0.0, 0.99),   # strong knee, slightly noisy extremes
        "MOEA/D": (0.015, 0.009, 0.0, 1.0),     # very consistent across full front
        "SPEA2": (0.032, 0.018, 0.01, 0.98),    # moderate gap
        "SMPSO": (0.045, 0.028, 0.02, 0.96),    # higher spread across runs
    }
    for alg, (base_bias, jitter, lo, hi) in zdt1_profiles.items():
        for r in range(n_runs):
            x = np.sort(rng.uniform(lo, hi, size=48))
            run_shift = abs(rng.normal(base_bias, jitter))
            # Regional specialization: NSGA-II excels in center, MOEA/D excels at extremes
            if alg == "NSGA-II":
                regional = 0.032 * np.abs(x - 0.45)
            elif alg == "MOEA/D":
                regional = 0.030 * np.exp(-((x - 0.45) ** 2) / 0.05)
            else:
                regional = 0.015 * np.sin(np.pi * x) ** 2
            noise = np.abs(rng.normal(0.0, 0.006, size=x.size))
            y = (1.0 - np.sqrt(np.clip(x, 0.0, 1.0))) + run_shift + regional + noise
            _add_run(alg, "ZDT1", r, np.column_stack([x, y]), ("f1", "f2"))

    # 2. ZDT2 (2D non-convex): f2 = 1 - f1^2
    _add_ref("ZDT2", np.column_stack([x_ref, 1.0 - x_ref**2]), ("f1", "f2"))
    zdt2_profiles = {
        "NSGA-II": (0.018, 0.011),
        "MOEA/D": (0.040, 0.022),
        "SPEA2": (0.026, 0.015),
        "SMPSO": (0.016, 0.012),
    }
    for alg, (base_bias, jitter) in zdt2_profiles.items():
        for r in range(n_runs):
            x = np.sort(rng.uniform(0.0, 1.0, size=48))
            run_shift = abs(rng.normal(base_bias, jitter))
            regional = 0.035 * (x**2) if alg == "MOEA/D" else 0.01 * (1.0 - x)
            y = (1.0 - x**2) + run_shift + regional + np.abs(rng.normal(0.0, 0.006, size=x.size))
            _add_run(alg, "ZDT2", r, np.column_stack([x, y]), ("f1", "f2"))

    # 3. ZDT3 (2D disconnected 5-segment front)
    x_z3 = np.linspace(0.0, 1.0, 600)
    y_z3 = 1.0 - np.sqrt(x_z3) - x_z3 * np.sin(10.0 * np.pi * x_z3)
    ref_z3, _ = filter_nondominated_array(np.column_stack([x_z3, y_z3]))
    _add_ref("ZDT3", ref_z3, ("f1", "f2"))
    for alg in ("NSGA-II", "MOEA/D", "SPEA2", "SMPSO"):
        for r in range(n_runs):
            idx = rng.choice(ref_z3.shape[0], size=min(55, ref_z3.shape[0]), replace=False)
            pts = ref_z3[np.sort(idx)].copy()
            if alg == "MOEA/D" and r % 2 == 0:
                # Misses part of upper-right segment in half the runs
                pts = pts[pts[:, 0] < 0.78]
            bias = {"NSGA-II": 0.018, "MOEA/D": 0.025, "SPEA2": 0.028, "SMPSO": 0.042}[alg]
            jitter = {"NSGA-II": 0.010, "MOEA/D": 0.016, "SPEA2": 0.014, "SMPSO": 0.024}[alg]
            shift = abs(rng.normal(bias, jitter))
            pts[:, 0] += np.abs(rng.normal(0.0, 0.004, size=pts.shape[0]))
            pts[:, 1] += shift + np.abs(rng.normal(0.0, 0.008, size=pts.shape[0]))
            _add_run(alg, "ZDT3", r, pts, ("f1", "f2"))

    # 4. DTLZ2 (3D spherical octant: f1^2 + f2^2 + f3^2 = 1)
    u = np.linspace(0.02, np.pi / 2 - 0.02, 16)
    v = np.linspace(0.02, np.pi / 2 - 0.02, 16)
    uu, vv = np.meshgrid(u, v)
    f1_ref = np.cos(uu.ravel()) * np.cos(vv.ravel())
    f2_ref = np.cos(uu.ravel()) * np.sin(vv.ravel())
    f3_ref = np.sin(uu.ravel())
    _add_ref("DTLZ2", np.column_stack([f1_ref, f2_ref, f3_ref]), ("f1", "f2", "f3"))
    dtlz2_bias = {"NSGA-II": 0.035, "MOEA/D": 0.018, "SPEA2": 0.028, "SMPSO": 0.048}
    for alg, b in dtlz2_bias.items():
        for r in range(n_runs):
            theta = rng.uniform(0.02, np.pi / 2 - 0.02, size=50)
            phi = rng.uniform(0.02, np.pi / 2 - 0.02, size=50)
            rad = 1.0 + abs(rng.normal(b, 0.014)) + np.abs(rng.normal(0.0, 0.01, size=50))
            pts3 = np.column_stack(
                [
                    rad * np.cos(theta) * np.cos(phi),
                    rad * np.cos(theta) * np.sin(phi),
                    rad * np.sin(theta),
                ]
            )
            _add_run(alg, "DTLZ2", r, pts3, ("f1", "f2", "f3"))

    return FrontCollection(name=name, fronts=tuple(fronts), reference_fronts=refs)
