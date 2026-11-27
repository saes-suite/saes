"""Immutable Pydantic V2 domain models for multi-objective Pareto fronts and EAF analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from saes.domain.models import Dataset


class ParetoFront(BaseModel):
    """Single approximation set or reference Pareto front in m-dimensional objective space."""

    model_config = ConfigDict(frozen=True)

    algorithm: str = Field(min_length=1, description="Algorithm name or 'Reference'")
    instance: str = Field(min_length=1, description="Benchmark problem instance identifier")
    run_id: int = Field(ge=0, default=0, description="Execution run identifier (>= 0)")
    points: tuple[tuple[float, ...], ...] = Field(
        default_factory=tuple,
        description="Objective vectors of shape (n_points, n_objectives)",
    )
    objective_names: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Display names for each objective axis (defaults to f1, f2, ...)",
    )
    minimize: tuple[bool, ...] = Field(
        default_factory=tuple,
        description="Per-objective direction (True = minimize, False = maximize)",
    )
    is_reference: bool = Field(default=False, description="Whether this front is the true/reference Pareto front")

    @model_validator(mode="after")
    def _validate_dimensions(self) -> ParetoFront:
        if not self.points:
            return self
        m = len(self.points[0])
        if m < 2:
            raise ValueError(f"Pareto fronts must have at least 2 objectives, got {m}.")
        for idx, pt in enumerate(self.points):
            if len(pt) != m:
                raise ValueError(f"Point {idx} has {len(pt)} objectives, expected {m}.")
            if not all(np.isfinite(val) for val in pt):
                raise ValueError(f"Point {idx} contains non-finite objective values.")

        if self.objective_names and len(self.objective_names) != m:
            raise ValueError(f"objective_names length ({len(self.objective_names)}) != n_objectives ({m}).")
        if self.minimize and len(self.minimize) != m:
            raise ValueError(f"minimize length ({len(self.minimize)}) != n_objectives ({m}).")

        if not self.objective_names:
            object.__setattr__(self, "objective_names", tuple(f"f{i + 1}" for i in range(m)))
        if not self.minimize:
            object.__setattr__(self, "minimize", tuple(True for _ in range(m)))
        return self

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_objectives(self) -> int:
        if self.points:
            return len(self.points[0])
        return len(self.objective_names)

    def to_array(self) -> np.ndarray:
        """Return objective points as a float64 NumPy array of shape (n_points, n_objectives)."""
        if not self.points:
            return np.empty((0, self.n_objectives), dtype=np.float64)
        return np.asarray(self.points, dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        """Return points as a tidy pandas DataFrame."""
        cols = list(self.objective_names) if self.objective_names else [f"f{i + 1}" for i in range(self.n_objectives)]
        df = pd.DataFrame(self.to_array(), columns=cols)
        df.insert(0, "run_id", self.run_id)
        df.insert(0, "instance", self.instance)
        df.insert(0, "algorithm", self.algorithm)
        return df


class InstanceFrontMeta(BaseModel):
    """Summary metadata for a single multi-objective benchmark instance."""

    model_config = ConfigDict(frozen=True)

    instance: str
    n_objectives: int
    objective_names: tuple[str, ...]
    minimize: tuple[bool, ...]
    algorithms: tuple[str, ...]
    runs_per_algorithm: dict[str, list[int]]
    has_reference_front: bool
    reference_points_count: int = 0
    total_points: int = 0


class FrontCollectionSummary(BaseModel):
    """Structured summary of a FrontCollection."""

    model_config = ConfigDict(frozen=True)

    name: str = "front_collection"
    instances: tuple[str, ...]
    algorithms: tuple[str, ...]
    total_fronts: int
    total_points: int
    reference_instances: tuple[str, ...]
    instance_details: dict[str, InstanceFrontMeta]


class FrontCollection(BaseModel):
    """Collection of multi-run Pareto fronts across algorithms and instances, with optional reference fronts."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="front_collection")
    fronts: tuple[ParetoFront, ...] = Field(default_factory=tuple)
    reference_fronts: dict[str, ParetoFront] = Field(default_factory=dict)

    @property
    def algorithms(self) -> list[str]:
        return sorted({f.algorithm for f in self.fronts if not f.is_reference})

    @property
    def instances(self) -> list[str]:
        insts = {f.instance for f in self.fronts} | set(self.reference_fronts.keys())
        return sorted(insts)

    def get_fronts(
        self,
        instance: str,
        algorithms: list[str] | None = None,
        run_ids: list[int] | None = None,
    ) -> list[ParetoFront]:
        """Return fronts matching instance and optional algorithm / run_id filters."""
        alg_set = set(algorithms) if algorithms is not None else None
        run_set = set(run_ids) if run_ids is not None else None
        return [
            f
            for f in self.fronts
            if f.instance == instance
            and not f.is_reference
            and (alg_set is None or f.algorithm in alg_set)
            and (run_set is None or f.run_id in run_set)
        ]

    def get_reference(self, instance: str) -> ParetoFront | None:
        """Return the reference Pareto front for ``instance`` if available."""
        return self.reference_fronts.get(instance)

    def filter(
        self,
        *,
        instances: list[str] | None = None,
        algorithms: list[str] | None = None,
        run_ids: list[int] | None = None,
    ) -> FrontCollection:
        """Return a filtered FrontCollection preserving matching reference fronts."""
        inst_set = set(instances) if instances is not None else None
        alg_set = set(algorithms) if algorithms is not None else None
        run_set = set(run_ids) if run_ids is not None else None

        kept_fronts = tuple(
            f
            for f in self.fronts
            if (inst_set is None or f.instance in inst_set)
            and (alg_set is None or f.algorithm in alg_set)
            and (run_set is None or f.run_id in run_set)
        )
        kept_refs = {
            k: v for k, v in self.reference_fronts.items() if inst_set is None or k in inst_set
        }
        return FrontCollection(name=self.name, fronts=kept_fronts, reference_fronts=kept_refs)

    def filter_nondominated(self) -> FrontCollection:
        """Return a new FrontCollection where every front contains only non-dominated points."""
        from saes.plugins.pareto.dominance import filter_nondominated_front

        cleaned = tuple(filter_nondominated_front(f) for f in self.fronts)
        cleaned_refs = {k: filter_nondominated_front(v) for k, v in self.reference_fronts.items()}
        return FrontCollection(name=self.name, fronts=cleaned, reference_fronts=cleaned_refs)

    def summary(self) -> FrontCollectionSummary:
        """Build structured metadata summary across all instances and algorithms."""
        details: dict[str, InstanceFrontMeta] = {}
        for inst in self.instances:
            inst_fronts = self.get_fronts(inst)
            ref = self.get_reference(inst)
            sample = inst_fronts[0] if inst_fronts else ref
            n_obj = sample.n_objectives if sample else 2
            obj_names = sample.objective_names if sample else tuple(f"f{i + 1}" for i in range(n_obj))
            minimize = sample.minimize if sample else tuple(True for _ in range(n_obj))
            algs = sorted({f.algorithm for f in inst_fronts})
            runs_map: dict[str, list[int]] = {
                a: sorted({f.run_id for f in inst_fronts if f.algorithm == a}) for a in algs
            }
            details[inst] = InstanceFrontMeta(
                instance=inst,
                n_objectives=n_obj,
                objective_names=obj_names,
                minimize=minimize,
                algorithms=tuple(algs),
                runs_per_algorithm=runs_map,
                has_reference_front=ref is not None and ref.n_points > 0,
                reference_points_count=ref.n_points if ref is not None else 0,
                total_points=sum(f.n_points for f in inst_fronts),
            )
        return FrontCollectionSummary(
            name=self.name,
            instances=tuple(self.instances),
            algorithms=tuple(self.algorithms),
            total_fronts=len(self.fronts),
            total_points=sum(f.n_points for f in self.fronts),
            reference_instances=tuple(sorted(self.reference_fronts.keys())),
            instance_details=details,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export all approximation fronts into a single tidy pandas DataFrame."""
        if not self.fronts:
            return pd.DataFrame(columns=["algorithm", "instance", "run_id", "f1", "f2"])
        return pd.concat([f.to_dataframe() for f in self.fronts], ignore_index=True)

    def to_indicators_dataset(
        self,
        indicators: tuple[str, ...] | list[str] = ("HV", "IGD+", "IGD", "EPS", "Spacing"),
        instances: list[str] | None = None,
        algorithms: list[str] | None = None,
        seed: int = 42,
    ) -> Dataset:
        """Compute multi-objective quality indicators and return a canonical ``saes.Dataset``."""
        from saes.plugins.pareto.indicators import compute_indicators_dataset

        sub = self.filter(instances=instances, algorithms=algorithms)
        return compute_indicators_dataset(sub, indicators=tuple(indicators), seed=seed)

    @classmethod
    def from_arrays(
        cls,
        runs_by_algorithm: dict[str, list[np.ndarray]],
        *,
        instance: str = "Problem",
        reference_front: np.ndarray | None = None,
        objective_names: tuple[str, ...] | list[str] | None = None,
        minimize: tuple[bool, ...] | list[bool] | None = None,
        name: str = "in_memory_collection",
    ) -> FrontCollection:
        """Construct a FrontCollection directly from in-memory NumPy arrays."""
        fronts: list[ParetoFront] = []
        obj_tuple = tuple(objective_names) if objective_names else ()
        min_tuple = tuple(minimize) if minimize else ()

        for alg, run_arrays in runs_by_algorithm.items():
            for run_idx, arr in enumerate(run_arrays):
                pts = tuple(tuple(float(v) for v in row) for row in np.asarray(arr, dtype=np.float64))
                fronts.append(
                    ParetoFront(
                        algorithm=alg,
                        instance=instance,
                        run_id=run_idx,
                        points=pts,
                        objective_names=obj_tuple,
                        minimize=min_tuple,
                    )
                )

        refs: dict[str, ParetoFront] = {}
        if reference_front is not None:
            ref_pts = tuple(tuple(float(v) for v in row) for row in np.asarray(reference_front, dtype=np.float64))
            refs[instance] = ParetoFront(
                algorithm="Reference",
                instance=instance,
                run_id=0,
                points=ref_pts,
                objective_names=obj_tuple,
                minimize=min_tuple,
                is_reference=True,
            )

        return cls(name=name, fronts=tuple(fronts), reference_fronts=refs)


class AttainmentSurface(BaseModel):
    """A single k-th empirical attainment surface (level in [0, 1]) in 2D objective space."""

    model_config = ConfigDict(frozen=True)

    level: float = Field(ge=0.0, le=1.0, description="Attainment probability threshold (0=best, 0.5=median, 1=worst)")
    rank_k: int = Field(ge=1, description="Number of runs attaining every point on this surface (1..n_runs)")
    points: tuple[tuple[float, float], ...] = Field(default_factory=tuple)


class AlgorithmAttainment(BaseModel):
    """Empirical attainment surfaces computed across multiple runs of one algorithm."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    n_runs: int
    surfaces: list[AttainmentSurface] = Field(default_factory=list)


class AttainmentResult(BaseModel):
    """Multi-algorithm 2D Empirical Attainment Function (EAF) summary result."""

    model_config = ConfigDict(frozen=True)

    instance: str
    objective_names: tuple[str, str]
    minimize: tuple[bool, bool]
    levels: tuple[float, ...]
    algorithms: list[AlgorithmAttainment] = Field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Empirical Attainment Surfaces — {self.instance} ({self.objective_names[0]} vs {self.objective_names[1]})",
            f"Levels: {', '.join(f'{lvl * 100:.0f}%' for lvl in self.levels)}",
        ]
        for item in self.algorithms:
            counts = ", ".join(f"{s.level * 100:.0f}%:{len(s.points)}pts" for s in item.surfaces)
            lines.append(f"  - {item.algorithm} (n={item.n_runs} runs): {counts}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in self.algorithms:
            for surf in item.surfaces:
                for x, y in surf.points:
                    rows.append(
                        {
                            "instance": self.instance,
                            "algorithm": item.algorithm,
                            "level": surf.level,
                            "rank_k": surf.rank_k,
                            self.objective_names[0]: x,
                            self.objective_names[1]: y,
                        }
                    )
        return pd.DataFrame(rows)


class EAFDifferenceResult(BaseModel):
    """Pairwise 2D Empirical Attainment Function difference ΔEAF(z) = α_A(z) - α_B(z) and KS-EAF permutation test."""

    model_config = ConfigDict(frozen=True)

    instance: str
    algo_a: str
    algo_b: str
    n_runs_a: int
    n_runs_b: int
    objective_names: tuple[str, str]
    minimize: tuple[bool, bool]
    x_grid: list[float]
    y_grid: list[float]
    diff_matrix: list[list[float]]  # shape (len(y_grid), len(x_grid)), values in [-1, 1]
    max_advantage_a: float
    max_advantage_b: float
    ks_statistic: float
    p_value: float
    n_permutations: int
    alpha: float = 0.05
    is_significant: bool = False
    area_fraction_a_better: float = 0.0
    area_fraction_b_better: float = 0.0
    interpretation: str = ""

    def summary(self) -> str:
        sig_str = "SIGNIFICANT" if self.is_significant else "not significant"
        return (
            f"EAF Difference ({self.instance}): {self.algo_a} (n={self.n_runs_a}) vs "
            f"{self.algo_b} (n={self.n_runs_b})\n"
            f"  KS-EAF = {self.ks_statistic:.4f}, p = {self.p_value:.4g} "
            f"({sig_str} at α={self.alpha}, B={self.n_permutations} permutations)\n"
            f"  Max advantage {self.algo_a}: +{self.max_advantage_a * 100:.1f}% "
            f"(area {self.area_fraction_a_better * 100:.1f}%)\n"
            f"  Max advantage {self.algo_b}: +{self.max_advantage_b * 100:.1f}% "
            f"(area {self.area_fraction_b_better * 100:.1f}%)"
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "instance": self.instance,
                    "algo_a": self.algo_a,
                    "algo_b": self.algo_b,
                    "n_runs_a": self.n_runs_a,
                    "n_runs_b": self.n_runs_b,
                    "ks_statistic": self.ks_statistic,
                    "p_value": self.p_value,
                    "is_significant": self.is_significant,
                    "max_advantage_a": self.max_advantage_a,
                    "max_advantage_b": self.max_advantage_b,
                    "area_fraction_a_better": self.area_fraction_a_better,
                    "area_fraction_b_better": self.area_fraction_b_better,
                }
            ]
        )


class CoverageCell(BaseModel):
    """Aggregated set coverage C(A, B) between two algorithms across runs."""

    model_config = ConfigDict(frozen=True)

    algo_a: str
    algo_b: str
    mean: float
    median: float
    std: float


class AlgorithmRepresentativeRuns(BaseModel):
    """Representative runs selected by IGD+ (or HV) for a single algorithm on an instance."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    n_runs: int
    metric_used: str
    best_run_id: int
    best_score: float
    median_run_id: int
    median_score: float
    worst_run_id: int
    worst_score: float
    accumulated_points_count: int
    mean_igd_plus: float | None = None
    mean_hv: float | None = None


class CoverageAndRepresentativeResult(BaseModel):
    """Combined set-coverage C(A, B) matrix and representative run summary for an instance."""

    model_config = ConfigDict(frozen=True)

    instance: str
    algorithms: list[str]
    coverage_matrix: list[CoverageCell]
    representative_runs: list[AlgorithmRepresentativeRuns]

    def summary(self) -> str:
        lines = [f"Coverage C(A, B) & Representative Runs — {self.instance}"]
        for rep in self.representative_runs:
            lines.append(
                f"  - {rep.algorithm} (n={rep.n_runs}): best=run#{rep.best_run_id} ({rep.best_score:.4g}), "
                f"median=run#{rep.median_run_id} ({rep.median_score:.4g}), "
                f"worst=run#{rep.worst_run_id} ({rep.worst_score:.4g})"
            )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([c.model_dump() for c in self.coverage_matrix])
