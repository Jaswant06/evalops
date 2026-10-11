from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from evalops.engine import RunOptions


@pytest.fixture
def client(project, monkeypatch):
    project.write_app("v1")
    project.write_app("v2", {"default": "good", "cases": {"c01": "wrong", "c17": "wrong"}})
    ev = project.evaluator()
    runs = {name: ev.run(project.handle(name), RunOptions(quiet=True)).run_id for name in ("v1", "v2")}
    monkeypatch.chdir(project.root)
    monkeypatch.setenv("EVALOPS_CONFIG", "configs/evalops.yaml")
    monkeypatch.setenv("EVALOPS_DATABASE_URL", f"sqlite:///{project.root}/evalops.db")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from evalops.api import app
    with TestClient(app) as c:
        c.runs = runs
        yield c


def test_health_and_runs(client):
    assert client.get("/health").json()["status"] == "ok"
    listed = client.get("/runs").json()
    assert {r["run_id"] for r in listed} == set(client.runs.values())
    detail = client.get(f"/runs/{client.runs['v1']}").json()
    assert detail["run"]["status"] == "completed"
    assert detail["metrics"]["answerable"]["answer_correctness"]["value"] == 1.0
    assert client.get("/runs/nope").status_code == 404


def test_failed_cases_and_report(client):
    failed = client.get(f"/runs/{client.runs['v2']}/cases", params={"failed_only": True}).json()
    assert {r["case_id"] for r in failed} == {"c01", "c17"}
    assert "# Evaluation report" in client.get(f"/runs/{client.runs['v2']}/report").text


def test_compare_applies_the_gate_without_recording(client):
    text = client.get("/compare", params={"candidate": client.runs["v2"], "baseline": client.runs["v1"]}).text
    assert "# Release decision: BLOCKED" in text
    assert client.get("/decisions").json() == []


def test_start_run_rejects_paths_outside_the_repo(client):
    assert client.post("/runs", json={"app_config": "../../etc/passwd"}).status_code == 400
    assert client.post("/runs", json={"app_config": "/etc/hosts"}).status_code == 400


def test_start_run_in_the_background(client):
    job = client.post("/runs", json={"app_config": "configs/apps/v1.yaml", "mode": "replay"}).json()
    for _ in range(100):
        state = client.get(f"/jobs/{job['job_id']}").json()
        if state["state"] != "running":
            break
        time.sleep(0.05)
    assert state["state"] == "completed", state
    assert client.get(f"/runs/{state['run_id']}").json()["run"]["mode"] == "replay"


def test_prometheus_metrics(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert "evalops_api_requests_total" in body and 'route="/health"' in body


def test_loadtest_against_a_fake_service(project, monkeypatch):
    import httpx
    from evalops import loadtest
    from evalops.datasets import load

    calls = []

    def post(url, json=None, timeout=None):
        calls.append(json)
        status = 500 if json["question"].startswith("Question number 3?") else 200
        return httpx.Response(status, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(loadtest.httpx, "post", post)
    monkeypatch.setattr(loadtest.httpx, "get", lambda url, timeout=None: httpx.Response(
        200, json={}, request=httpx.Request("GET", url)))
    results = loadtest.run("http://svc", load("toy/v1:test", project.root / "datasets"), [1, 4], n_requests=20)
    assert [r["concurrency"] for r in results] == [1, 4]
    assert all(r["errors"] == 1 and r["requests_per_second"] > 0 for r in results)
