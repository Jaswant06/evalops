"""EvalOps HTTP API: runs, per-case results, comparisons, gate decisions, releases.

Run: evalops serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from . import __version__
from .config import load_app, load_platform, load_release
from .engine import Evaluator, RunOptions, app_from_config
from .datasets import load as load_dataset
from .decision_report import render_markdown
from .gate import evaluate_gate
from .release import run_data
from .store import Store
from .tracking import Tracker

REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter("evalops_api_requests_total", "API requests", ["route", "status"], registry=REGISTRY)
HTTP_LATENCY = Histogram("evalops_api_request_latency_seconds", "API latency", ["route"], registry=REGISTRY)
JOBS = Gauge("evalops_api_jobs", "Evaluation jobs by state", ["state"], registry=REGISTRY)
PRODUCTION = Gauge("evalops_production_info", "The app version in production", ["app", "version"],
                   registry=REGISTRY)

state: dict = {}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    platform = load_platform(Path(os.environ.get("EVALOPS_CONFIG", "configs/evalops.yaml")))
    store = Store()
    store.init()
    state["evaluator"] = Evaluator(platform, Path("."), store, Tracker.from_env())
    yield
    state.clear()


app = FastAPI(title="EvalOps", version=__version__, lifespan=lifespan,
              description="Evaluation and release gating for LLM applications.")


@app.middleware("http")
async def observe(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    HTTP_REQUESTS.labels(path, str(response.status_code)).inc()
    HTTP_LATENCY.labels(path).observe(time.perf_counter() - started)
    return response


def evaluator() -> Evaluator:
    return state["evaluator"]


def get_run_or_404(run_id: str) -> dict:
    run = evaluator().store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No run {run_id}")
    return run


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "database": evaluator().store.describe()}


@app.get("/runs")
def list_runs(limit: int = 50, app_name: str | None = None, status: str | None = None) -> list[dict]:
    return evaluator().store.list_runs(limit=limit, app=app_name, status=status)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = get_run_or_404(run_id)
    return {"run": run, "metrics": evaluator().store.get_metrics(run_id)}


@app.get("/runs/{run_id}/cases")
def get_cases(run_id: str, failed_only: bool = False) -> list[dict]:
    get_run_or_404(run_id)
    rows = [c["record"] for c in evaluator().store.get_cases(run_id).values() if c.get("record")]
    if failed_only:
        rows = [r for r in rows if r["status"] != "ok" or r.get("verdict") == "incorrect"
                or not r["abstention_correct"] or (r.get("claims_total") or 0) > (r.get("claims_supported_document") or 0)]
    return sorted(rows, key=lambda r: r["case_id"])


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def get_report(run_id: str) -> str:
    run = get_run_or_404(run_id)
    path = Path(run.get("report_path") or "")
    if not path.is_file():
        raise HTTPException(404, "Report not available (the run did not complete)")
    return path.read_text()


@app.get("/compare", response_class=PlainTextResponse)
def compare(candidate: str, baseline: str) -> str:
    """Apply the release gate to two existing runs. Read-only: nothing is promoted or recorded."""
    ev = evaluator()
    cand, base = run_data(ev, get_run_or_404(candidate)["run_id"]), run_data(ev, get_run_or_404(baseline)["run_id"])
    dataset = load_dataset(cand.run["dataset_ref"], ev.base_dir / "datasets")
    questions = {c.case_id: c.input.question for c in dataset.cases}
    return render_markdown(evaluate_gate(ev.gate, cand, base), cand, base, ev.gate, questions)


@app.get("/decisions")
def decisions(limit: int = 50) -> list[dict]:
    return [{k: v for k, v in d.items() if k != "report"} for d in evaluator().store.list_decisions(limit)]


@app.get("/decisions/{decision_id}", response_class=PlainTextResponse)
def decision(decision_id: str) -> str:
    d = evaluator().store.get_decision(decision_id)
    if d is None:
        raise HTTPException(404, f"No decision {decision_id}")
    return d["report"]


@app.get("/audit")
def audit(limit: int = 50) -> list[dict]:
    """Gate decisions, promotions, refused promotions and rollbacks, newest first."""
    return evaluator().store.list_audit(limit)


@app.get("/release")
def release() -> dict:
    ev = evaluator()
    record = load_release(ev.base_dir / ev.platform.release)
    return {"release": record.model_dump(mode="json"), "promotions": ev.store.list_releases()}


class RunRequest(BaseModel):
    app_config: str
    split: str | None = None
    mode: Literal["replay", "live", "rescore", "auto"] = "auto"
    limit: int | None = None


@app.post("/runs", status_code=202)
def start_run(req: RunRequest) -> dict:
    """Start an evaluation in the background. Poll /jobs/{job_id}."""
    ev = evaluator()
    path = Path(req.app_config)
    if path.is_absolute() or ".." in path.parts or not (ev.base_dir / path).is_file():
        raise HTTPException(400, "app_config must be a config file inside the repository")
    handle = app_from_config(load_app(ev.base_dir / path), ev.base_dir)
    dataset = f"{ev.platform.dataset.split(':')[0]}:{req.split}" if req.split else None
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "state": "running", "app_config": req.app_config, "run_id": None,
                        "error": None}

    def work():
        try:
            result = ev.run(handle, RunOptions(dataset=dataset, mode=req.mode, limit=req.limit, quiet=True))
            update = {"state": "completed", "run_id": result.run_id}
        except BaseException as exc:
            update = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        with jobs_lock:
            jobs[job_id].update(update)

    threading.Thread(target=work, daemon=True).start()
    return jobs[job_id]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, f"No job {job_id}")
    return jobs[job_id]


@app.get("/metrics")
def metrics() -> Response:
    with jobs_lock:
        for s in ("running", "completed", "failed"):
            JOBS.labels(s).set(sum(j["state"] == s for j in jobs.values()))
    ev = state.get("evaluator")
    if ev:
        record = load_release(ev.base_dir / ev.platform.release)
        PRODUCTION.clear()
        if record.production:
            PRODUCTION.labels(record.production.app, record.production.version).set(1)
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
