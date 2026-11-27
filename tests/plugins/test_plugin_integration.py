"""End-to-end integration tests: SAES entry-point discovery, FastAPI mounting, and Dataset bridge."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import saes
from saes.plugins.registry import PluginRegistry
from saes.server.app import create_app


def test_saes_discovers_pareto_plugin_and_mounts_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAES_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("SAES_PARETO_DIR", str(tmp_path / "pareto"))

    # Force fresh discovery from installed entry points
    PluginRegistry.reset()
    manifests = saes.plugins.discover(force=True)
    ids = [m.id for m in manifests]
    assert "pareto" in ids

    ns = saes.plugins.get_namespace("pareto")
    assert hasattr(ns, "create_demo_collection")
    assert hasattr(ns, "eaf_difference")

    app = create_app()
    client = TestClient(app)

    # 1. Core discovery endpoint advertises pareto plugin
    disc = client.get("/api/v1/plugins")
    assert disc.status_code == 200
    payload = disc.json()
    pareto_entries = [p for p in payload["plugins"] if p["id"] == "pareto"]
    assert len(pareto_entries) == 1
    assert pareto_entries[0]["api_prefix"] == "/api/v1/plugins/pareto"

    # 2. Create demo collection via plugin route
    demo_res = client.post("/api/v1/plugins/pareto/collections/demo?name=e2e_demo&n_runs=6")
    assert demo_res.status_code == 200
    summ = demo_res.json()
    assert summ["name"] == "e2e_demo"
    assert "ZDT1" in summ["instances"]
    assert "DTLZ2" in summ["instances"]

    # 3. Interactive plot generation (attainment, runs, eaf-diff, 3d)
    for mode in ("attainment", "runs", "representative", "eaf-diff", "small-multiples"):
        plot_res = client.post(
            "/api/v1/plugins/pareto/collections/e2e_demo/plot",
            json={
                "instance": "ZDT1",
                "mode": mode,
                "algorithms": ["NSGA-II", "MOEA/D"],
                "n_permutations": 40,
            },
        )
        assert plot_res.status_code == 200, f"Mode {mode} failed: {plot_res.text}"
        body = plot_res.json()
        assert "data" in body["plotly_figure"]
        if mode == "eaf-diff":
            assert body["eaf_difference"] is not None
            assert 0.0 < body["eaf_difference"]["p_value"] <= 1.0

    # 4. Vector PDF export + reference-only & styling options over REST
    exp_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/export-plot",
        json={
            "instance": "ZDT1",
            "mode": "attainment",
            "export_format": "pdf",
            "legend_placement": "bottom",
            "title_mode": "none",
            "font_size": 14,
            "font_family": "Times New Roman, Times, serif",
            "label_mode": "all",
            "bold_mode": "title-axes",
            "show_grid": False,
            "width_px": 760,
            "height_px": 480,
        },
    )
    assert exp_res.status_code == 200
    assert exp_res.content.startswith(b"%PDF")

    ref_only_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/plot",
        json={
            "instance": "ZDT1",
            "mode": "attainment",
            "algorithms": [],
            "show_reference": True,
            "legend_placement": "inside",
            "title": "Reference Front Only",
        },
    )
    assert ref_only_res.status_code == 200
    assert len(ref_only_res.json()["plotly_figure"]["data"]) == 1

    empty_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/plot",
        json={"instance": "ZDT1", "mode": "attainment", "algorithms": [], "show_reference": False},
    )
    assert empty_res.status_code == 400

    # 5. Coverage & representative runs
    cov_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/coverage",
        json={"instance": "ZDT1"},
    )
    assert cov_res.status_code == 200
    assert len(cov_res.json()["representative_runs"]) == 4

    # 6. Bridge: extract quality indicators directly into SAES Datasets storage and run core SAES ranking!
    bridge_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/to-dataset",
        json={
            "dataset_filename": "e2e_pareto_indicators.csv",
            "indicators": ["HV", "IGD+", "EPS"],
        },
    )
    assert bridge_res.status_code == 200
    bdata = bridge_res.json()
    assert bdata["filename"] == "e2e_pareto_indicators.csv"
    assert set(bdata["metrics"]) == {"HV", "IGD+", "EPS"}

    # Verify core SAES REST API immediately sees & analyzes the generated dataset!
    ds_list = client.get("/api/v1/datasets")
    assert ds_list.status_code == 200
    assert any(item["filename"] == "e2e_pareto_indicators.csv" for item in ds_list.json())

    rank_res = client.post(
        "/api/v1/datasets/e2e_pareto_indicators.csv/ranking",
        data={"metric": "IGD+"},
    )
    assert rank_res.status_code == 200
    assert "critical_distance" in rank_res.json()

    # 7. Attainment, EAF-diff, collection list, CSV upload, and delete routes
    att_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/attainment",
        json={"instance": "ZDT1", "levels": [0.25, 0.5, 0.75]},
    )
    assert att_res.status_code == 200
    assert len(att_res.json()["algorithms"]) == 4

    diff_res = client.post(
        "/api/v1/plugins/pareto/collections/e2e_demo/eaf-diff",
        json={"instance": "ZDT1", "algo_a": "NSGA-II", "algo_b": "MOEA/D", "n_permutations": 40},
    )
    assert diff_res.status_code == 200
    assert "ks_statistic" in diff_res.json()

    csv_payload = (
        b"algorithm,instance,run_id,f1,f2\n"
        b"A,P1,0,0.1,0.9\n"
        b"A,P1,0,0.9,0.1\n"
        b"B,P1,0,0.2,0.8\n"
    )
    up_res = client.post(
        "/api/v1/plugins/pareto/collections/upload",
        files={"file": ("uploaded.csv", csv_payload, "text/csv")},
        data={"name": "custom_upload"},
    )
    assert up_res.status_code == 200
    assert up_res.json()["name"] == "custom_upload"

    list_res = client.get("/api/v1/plugins/pareto/collections")
    assert list_res.status_code == 200
    assert {item["name"] for item in list_res.json()} >= {"e2e_demo", "custom_upload"}

    del_res = client.delete("/api/v1/plugins/pareto/collections/custom_upload")
    assert del_res.status_code == 200

    # 8. Guided multi-file creation (multiple FUN files per algorithm + optional .pf reference)
    multi_res = client.post(
        "/api/v1/plugins/pareto/collections/create-multi",
        data={
            "name": "multi_front_study",
            "instance": "ZDT1",
            "algorithm_names": '["NSGA-II", "NSGA-II", "MOEA/D"]',
        },
        files=[
            ("front_files", ("FUN.0.tsv", b"0.1 0.8\n0.5 0.4\n", "text/plain")),
            ("front_files", ("FUN.1.tsv", b"0.15 0.78\n0.55 0.36\n", "text/plain")),
            ("front_files", ("FUN.0.tsv", b"0.12 0.75\n0.6 0.32\n", "text/plain")),
            ("reference_file", ("ZDT1.pf", b"0.0 1.0\n0.5 0.29\n1.0 0.0\n", "text/plain")),
        ],
    )
    assert multi_res.status_code == 200
    msumm = multi_res.json()
    assert msumm["name"] == "multi_front_study"
    assert msumm["algorithms"] == ["MOEA/D", "NSGA-II"]
    assert msumm["instance_details"]["ZDT1"]["has_reference_front"] is True
    assert len(msumm["instance_details"]["ZDT1"]["runs_per_algorithm"]["NSGA-II"]) == 2
