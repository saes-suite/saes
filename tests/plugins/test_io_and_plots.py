"""I/O loaders (CSV, jMetal directory, ZIP) and dual Matplotlib/Plotly visualization tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from saes.plugins import pareto as saes_pareto


def test_csv_jmetal_dir_and_zip_loaders(tmp_path: Path) -> None:
    # 1. Tidy CSV + separate .pf reference front
    csv_text = (
        "algorithm,instance,run_id,f1,f2\n"
        "NSGA-II,ZDT1,0,0.1,0.7\n"
        "NSGA-II,ZDT1,0,0.5,0.35\n"
        "NSGA-II,ZDT1,1,0.15,0.68\n"
        "MOEA/D,ZDT1,0,0.12,0.66\n"
        "MOEA/D,ZDT1,0,0.6,0.28\n"
    )
    pf_text = "0.0 1.0\n0.25 0.5\n1.0 0.0\n"
    coll_csv = saes_pareto.load_csv(csv_text.encode("utf-8"), reference_front=pf_text.encode("utf-8"))
    assert coll_csv.algorithms == ["MOEA/D", "NSGA-II"]
    assert coll_csv.instances == ["ZDT1"]
    assert coll_csv.get_reference("ZDT1") is not None
    assert coll_csv.get_reference("ZDT1").n_points == 3  # type: ignore[union-attr]

    # 2. jMetal-style directory tree (filesystem-safe directory names like NSGAII, MOEAD)
    exp_dir = tmp_path / "jmetal_exp"
    for alg, shift in (("NSGA-II", 0.05), ("MOEAD", 0.02)):
        inst_dir = exp_dir / alg / "ZDT1"
        inst_dir.mkdir(parents=True)
        for run_id in range(3):
            arr = np.array([[0.1, 0.7 + shift + 0.01 * run_id], [0.5, 0.3 + shift + 0.01 * run_id]])
            np.savetxt(inst_dir / f"FUN.{run_id}.tsv", arr, delimiter="\t")
    ref_dir = exp_dir / "reference_fronts"
    ref_dir.mkdir(parents=True)
    (ref_dir / "ZDT1.pf").write_text(pf_text, encoding="utf-8")

    coll_dir = saes_pareto.load_directory(exp_dir)
    assert coll_dir.algorithms == ["MOEAD", "NSGA-II"]
    assert len(coll_dir.get_fronts("ZDT1", algorithms=["NSGA-II"])) == 3
    assert coll_dir.get_reference("ZDT1") is not None

    # 3. ZIP archive of the jMetal tree
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for path in exp_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(tmp_path)))
    coll_zip = saes_pareto.load_zip(zip_buf.getvalue())
    assert coll_zip.algorithms == ["MOEAD", "NSGA-II"]
    assert coll_zip.get_reference("ZDT1") is not None


@pytest.mark.parametrize("mode", ["runs", "representative", "attainment", "eaf-diff", "small-multiples", "pcp"])
def test_all_2d_plot_modes_matplotlib_and_plotly(mode: str) -> None:
    coll = saes_pareto.create_demo_collection(n_runs=6, seed=42)

    fig_plotly = saes_pareto.plot(
        coll,
        instance="ZDT1",
        mode=mode,
        algorithms=["NSGA-II", "MOEA/D"],
        interactive=True,
        n_permutations=50,
    )
    assert isinstance(fig_plotly, go.Figure)
    assert len(fig_plotly.data) > 0

    fig_mpl = saes_pareto.plot(
        coll,
        instance="ZDT1",
        mode=mode,
        algorithms=["NSGA-II", "MOEA/D"],
        interactive=False,
        n_permutations=50,
    )
    assert isinstance(fig_mpl, Figure)
    plt.close(fig_mpl)


def test_3d_pcp_and_export_formats() -> None:
    coll = saes_pareto.create_demo_collection(n_runs=5, seed=42)

    fig_3d = saes_pareto.plot(coll, instance="DTLZ2", mode="3d", interactive=True)
    assert isinstance(fig_3d, go.Figure)

    fig_pcp = saes_pareto.plot(coll, instance="DTLZ2", mode="pcp", interactive=True)
    assert isinstance(fig_pcp, go.Figure)

    for fmt, magic in (("pdf", b"%PDF"), ("svg", b"<svg"), ("png", b"\x89PNG")):
        raw, media_type, ext = saes_pareto.export_plot(
            coll,
            instance="ZDT1",
            mode="attainment",
            export_format=fmt,
            show_extreme_bounds=True,
        )
        assert ext == fmt
        assert len(raw) > 500
        assert magic in raw[:2048]
        assert media_type


@pytest.mark.parametrize("mode", ["attainment", "runs", "representative", "small-multiples", "pcp"])
def test_reference_only_mode_2d(mode: str) -> None:
    coll = saes_pareto.create_demo_collection(n_runs=5, seed=42)

    fig_plotly = saes_pareto.plot(
        coll,
        instance="ZDT1",
        mode=mode,
        algorithms=[],
        show_reference=True,
        interactive=True,
    )
    assert isinstance(fig_plotly, go.Figure)
    assert len(fig_plotly.data) >= 1

    fig_mpl = saes_pareto.plot(
        coll,
        instance="ZDT1",
        mode=mode,
        algorithms=[],
        show_reference=True,
        interactive=False,
    )
    assert isinstance(fig_mpl, Figure)
    plt.close(fig_mpl)


def test_reference_only_3d_and_empty_guard() -> None:
    coll = saes_pareto.create_demo_collection(n_runs=4, seed=42)

    fig_3d = saes_pareto.plot(coll, instance="DTLZ2", mode="3d", algorithms=[], show_reference=True, interactive=True)
    assert isinstance(fig_3d, go.Figure)
    assert len(fig_3d.data) == 1

    fig_3d_mpl = saes_pareto.plot(
        coll, instance="DTLZ2", mode="3d", algorithms=[], show_reference=True, interactive=False
    )
    assert isinstance(fig_3d_mpl, Figure)
    plt.close(fig_3d_mpl)

    with pytest.raises(ValueError, match="Nothing to plot"):
        saes_pareto.plot(coll, instance="ZDT1", mode="attainment", algorithms=[], show_reference=False)


def test_plot_styling_parity_and_concise_attainment_legend() -> None:
    coll = saes_pareto.create_demo_collection(n_runs=6, seed=42)

    # Multi-algorithm attainment must keep 1 concise legend entry per algorithm (+ Reference) so labels never overlap
    fig_att = saes_pareto.plot(
        coll,
        instance="ZDT1",
        mode="attainment",
        show_extreme_bounds=True,
        legend_placement="top",
        font_size=16,
        font_family="Times New Roman, Times, serif",
        title_mode="compact",
        label_mode="ticks-only",
        bold_mode="all",
        show_grid=False,
        dark_theme=True,
        width_px=900,
        height_px=560,
        interactive=True,
    )
    assert isinstance(fig_att, go.Figure)
    visible_legend = [tr.name for tr in fig_att.data if getattr(tr, "showlegend", True) is not False]
    assert len(visible_legend) == 5  # Reference + 4 algorithms
    assert fig_att.layout.width == 900
    assert fig_att.layout.height == 560
    assert fig_att.layout.paper_bgcolor == "#0f172a"

    # Hidden title + hidden legend + bottom/right/inside legend placements
    for placement in ("top", "bottom", "right", "inside", "none"):
        f_p = saes_pareto.plot(
            coll,
            instance="ZDT1",
            mode="attainment",
            algorithms=["NSGA-II", "MOEA/D"],
            legend_placement=placement,
            title_mode="none",
            label_mode="none",
            interactive=True,
        )
        assert isinstance(f_p, go.Figure)
        assert f_p.layout.showlegend is (placement != "none")
        if placement == "bottom":
            assert f_p.layout.legend.yanchor == "top"

        raw_png, _, _ = saes_pareto.export_plot(
            coll,
            instance="ZDT1",
            mode="attainment",
            algorithms=["NSGA-II", "MOEA/D"],
            legend_placement=placement,
            title="Custom Paper Title",
            font_size=14,
            font_family="Georgia, serif",
            bold_mode="title-axes",
            show_grid=False,
            dark_theme=(placement == "inside"),
            width_px=720,
            height_px=480,
            export_format="png",
            dpi=120,
        )
        assert raw_png.startswith(b"\x89PNG")
