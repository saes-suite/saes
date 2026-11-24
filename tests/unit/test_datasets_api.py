"""Unit tests for SAES dataset persistence and path-based analysis endpoints."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from saes.server.app import app

client = TestClient(app)

SAMPLE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""

MULTI_INSTANCE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.6621,True
NSGAII,ZDT1,HV,1,0.6618,True
NSGAII,ZDT1,HV,2,0.6625,True
NSGAII,ZDT1,HV,3,0.6629,True
MOEAD,ZDT1,HV,0,0.6854,True
MOEAD,ZDT1,HV,1,0.6851,True
MOEAD,ZDT1,HV,2,0.6859,True
MOEAD,ZDT1,HV,3,0.6862,True
SPEA2,ZDT1,HV,0,0.6401,True
SPEA2,ZDT1,HV,1,0.6412,True
SPEA2,ZDT1,HV,2,0.6398,True
SPEA2,ZDT1,HV,3,0.6405,True
NSGAII,ZDT2,HV,0,0.5621,True
NSGAII,ZDT2,HV,1,0.5618,True
NSGAII,ZDT2,HV,2,0.5625,True
NSGAII,ZDT2,HV,3,0.5629,True
MOEAD,ZDT2,HV,0,0.5854,True
MOEAD,ZDT2,HV,1,0.5851,True
MOEAD,ZDT2,HV,2,0.5859,True
MOEAD,ZDT2,HV,3,0.5862,True
SPEA2,ZDT2,HV,0,0.5401,True
SPEA2,ZDT2,HV,1,0.5412,True
SPEA2,ZDT2,HV,2,0.5398,True
SPEA2,ZDT2,HV,3,0.5405,True
NSGAII,ZDT3,HV,0,0.7121,True
NSGAII,ZDT3,HV,1,0.7118,True
NSGAII,ZDT3,HV,2,0.7125,True
NSGAII,ZDT3,HV,3,0.7129,True
MOEAD,ZDT3,HV,0,0.7454,True
MOEAD,ZDT3,HV,1,0.7451,True
MOEAD,ZDT3,HV,2,0.7459,True
MOEAD,ZDT3,HV,3,0.7462,True
SPEA2,ZDT3,HV,0,0.6901,True
SPEA2,ZDT3,HV,1,0.6912,True
SPEA2,ZDT3,HV,2,0.6898,True
SPEA2,ZDT3,HV,3,0.6905,True
"""


@pytest.fixture(autouse=True)
def setup_teardown_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate datasets storage to temporary directory for unit testing."""
    monkeypatch.setenv("SAES_DATASETS_DIR", str(tmp_path))
    yield


def test_datasets_crud_and_path_analysis():
    # 1. Initially empty
    res = client.get("/api/v1/datasets")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Upload dataset
    file_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    upload_res = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test_benchmark.csv", file_bytes, "text/csv")},
    )
    assert upload_res.status_code == 200
    item = upload_res.json()
    assert item["name"] == "test_benchmark"
    assert item["filename"] == "test_benchmark.csv"
    assert item["row_count"] == 6

    # 3. List datasets contains uploaded item
    list_res = client.get("/api/v1/datasets")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    # 4. Get dataset details
    detail_res = client.get("/api/v1/datasets/test_benchmark.csv")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "algorithm,instance,metric" in detail["content"]
    assert detail["row_count"] == 6

    # 5. Validate dataset directly from server path
    val_res = client.post("/api/v1/datasets/test_benchmark.csv/validate")
    assert val_res.status_code == 200
    assert val_res.json()["is_valid"] is True

    # 6. Analyze dataset directly from server path
    ana_res = client.post("/api/v1/datasets/test_benchmark.csv/analyze", data={"alpha": 0.05})
    assert ana_res.status_code == 200
    ana_data = ana_res.json()
    assert ana_data["cells_count"] == 1
    assert ana_data["results"][0]["instance"] == "ZDT1"

    # 7. Update dataset content via PUT
    updated_csv = SAMPLE_CSV + "MOEAD,ZDT1,HV,3,0.666000,True\n"
    put_res = client.put(
        "/api/v1/datasets/test_benchmark.csv",
        json={"content": updated_csv},
    )
    assert put_res.status_code == 200
    assert put_res.json()["row_count"] == 7

    # 8. Delete dataset
    del_res = client.delete("/api/v1/datasets/test_benchmark.csv")
    assert del_res.status_code == 200
    assert client.get("/api/v1/datasets").json() == []


def test_dataset_path_ranking_tables_explore_profiles_power_repro_plots():
    """Exercise all path-based scientific REST endpoints for stored datasets."""
    upload_res = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("multi.csv", io.BytesIO(MULTI_INSTANCE_CSV.encode("utf-8")), "text/csv")},
    )
    assert upload_res.status_code == 200

    # Ranking (signed-rank & sign)
    rank_res = client.post(
        "/api/v1/datasets/multi.csv/ranking",
        data={"metric": "HV", "alpha": 0.05, "bayesian_method": "sign"},
    )
    assert rank_res.status_code == 200
    r_json = rank_res.json()
    assert r_json["critical_distance"]["k"] == 3
    assert r_json["bayesian_ranking"]["bayesian_method"] == "sign"

    # Tables GET & POST
    tbl_get = client.get(
        "/api/v1/datasets/multi.csv/tables",
        params={"metric": "HV", "algorithms_filter": "NSGAII,MOEAD"},
    )
    assert tbl_get.status_code == 200
    assert "Cache-Control" in tbl_get.headers
    assert tbl_get.json()["global_anova_f"] is None

    tbl_post = client.post(
        "/api/v1/datasets/multi.csv/tables",
        data={"metric": "HV", "table_type": "pairwise_matrix", "posthoc_correction": "holm"},
    )
    assert tbl_post.status_code == 200
    assert tbl_post.json()["table_type"] == "pairwise_matrix"

    # Tables export GET & POST across formats
    for fmt in ("latex", "csv", "markdown", "html", "docx", "pdf"):
        exp_post = client.post(
            "/api/v1/datasets/multi.csv/tables/export",
            data={"metric": "HV", "export_format": fmt},
        )
        assert exp_post.status_code == 200
        assert len(exp_post.content) > 20

    exp_get = client.get("/api/v1/datasets/multi.csv/tables/export", params={"metric": "HV", "export_format": "csv"})
    assert exp_get.status_code == 200

    # Plots export
    plot_res = client.post(
        "/api/v1/datasets/multi.csv/plots/export",
        data={"plot_type": "critical-distance", "metric": "HV", "export_format": "svg", "palette": "viridis"},
    )
    assert plot_res.status_code == 200
    assert b"<svg" in plot_res.content

    # Explore GET & POST + Export GET & POST
    expl_get = client.get("/api/v1/datasets/multi.csv/explore", params={"metrics": "HV"})
    assert expl_get.status_code == 200
    assert len(expl_get.json()["distribution_profiles"]) > 0

    expl_post = client.post("/api/v1/datasets/multi.csv/explore", data={"metrics": "HV"})
    assert expl_post.status_code == 200

    expl_exp_get = client.get(
        "/api/v1/datasets/multi.csv/explore/export",
        params={"export_format": "csv", "view_mode": "reveal_metrics"},
    )
    assert expl_exp_get.status_code == 200

    expl_exp_post = client.post(
        "/api/v1/datasets/multi.csv/explore/export",
        data={"export_format": "markdown", "view_mode": "summary"},
    )
    assert expl_exp_post.status_code == 200

    # Profiles, Power, Reproducibility
    prof_res = client.get("/api/v1/datasets/multi.csv/profiles", params={"metric": "HV"})
    assert prof_res.status_code == 200
    assert len(prof_res.json()["meta_rankings"]) == 3

    pwr_res = client.get("/api/v1/datasets/multi.csv/power", params={"metric": "HV"})
    assert pwr_res.status_code == 200
    assert len(pwr_res.json()["sensitivity_audit"]) > 0

    repro_res = client.get("/api/v1/datasets/multi.csv/reproducibility", params={"metric": "HV"})
    assert repro_res.status_code == 200
    assert repro_res.json()["manifest"]["statsmodels_version"]

    # New Phase 2 endpoints: posthoc, tost, bayesian correlated-ttest, effect-sizes, new plotters
    ph_res = client.get(
        "/api/v1/datasets/multi.csv/posthoc",
        params={"metric": "HV", "instance": "ZDT1", "method": "auto"},
    )
    assert ph_res.status_code == 200
    assert len(ph_res.json()["comparisons"]) == 3

    for method in ("tukey_hsd", "games_howell", "dunnett", "dunn", "conover_iman"):
        m_res = client.get(
            "/api/v1/datasets/multi.csv/posthoc",
            params={"metric": "HV", "instance": "ZDT1", "method": method, "control": "NSGAII"},
        )
        assert m_res.status_code == 200

    tost_res = client.get(
        "/api/v1/datasets/multi.csv/tost",
        params={"metric": "HV", "instance": "ZDT1", "algo_a": "NSGAII", "algo_b": "MOEAD", "low_eqbound": 0.05},
    )
    assert tost_res.status_code == 200
    assert "is_equivalent" in tost_res.json()

    bct_res = client.get(
        "/api/v1/datasets/multi.csv/bayesian/correlated-ttest",
        params={"metric": "HV", "instance": "ZDT1", "algo_a": "MOEAD", "algo_b": "NSGAII", "rope": 0.01, "rho": 0.1},
    )
    assert bct_res.status_code == 200
    assert len(bct_res.json()["curve_x"]) > 10

    ef_res = client.get(
        "/api/v1/datasets/multi.csv/effect-sizes",
        params={"metric": "HV", "instance": "ZDT1", "scope": "single-instance"},
    )
    assert ef_res.status_code == 200
    assert len(ef_res.json()["items"]) == 2

    # Ranking bundle includes Plackett-Luce, Aligned Friedman, Quade, RM-ANOVA
    assert r_json["bayesian_plackett_luce"] is not None
    assert r_json["friedman_aligned"] is not None
    assert r_json["quade"] is not None
    assert r_json["rm_anova"] is not None

    # New plotters export via REST
    for pt in ("plackett-luce", "effect-size-forest"):
        p_exp = client.post(
            "/api/v1/datasets/multi.csv/plots/export",
            data={"plot_type": pt, "metric": "HV", "instance": "ZDT1", "export_format": "svg"},
        )
        assert p_exp.status_code == 200
        assert b"<svg" in p_exp.content


def test_dataset_not_found_handling():
    assert client.get("/api/v1/datasets/non_existent.csv").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/validate").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/analyze").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/ranking", data={"metric": "HV"}).status_code == 404
    assert client.get("/api/v1/datasets/non_existent.csv/tables").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/tables").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/tables/export").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/plots/export").status_code == 404
    assert client.get("/api/v1/datasets/non_existent.csv/explore").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/explore").status_code == 404
    assert client.post("/api/v1/datasets/non_existent.csv/explore/export").status_code == 404
    assert client.get("/api/v1/datasets/non_existent.csv/profiles").status_code == 404
    assert client.get("/api/v1/datasets/non_existent.csv/power").status_code == 404
    assert client.get("/api/v1/datasets/non_existent.csv/reproducibility").status_code == 404
    assert client.delete("/api/v1/datasets/non_existent.csv").status_code == 404


def test_upload_dataset_rejects_invalid_schema():
    # Missing 'value' column
    bad_csv = "algorithm,instance,metric,run_id,maximize\nNSGAII,ZDT1,HV,0,True\n"
    res = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("corrupt.csv", io.BytesIO(bad_csv.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 422
    assert "Missing required column(s)" in res.json()["detail"]
    assert client.get("/api/v1/datasets").json() == []
