"""DocQA HTTP service with Prometheus metrics.

Run:  uvicorn docqa.service:app --port 8001

By default it serves the version that configs/release.yaml names as production,
so a promotion is what changes the deployed version; DOCQA_CONFIG overrides it.
The config is the same app-version YAML EvalOps evaluates; only its `settings`
block is read here, so the service has no dependency on EvalOps.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .app import DocQA, DocQASettings

RELEASE = Path(os.environ.get("DOCQA_RELEASE", "configs/release.yaml"))


def production_config() -> Path:
    """Serve what the production pointer says, unless DOCQA_CONFIG names a version explicitly."""
    if os.environ.get("DOCQA_CONFIG"):
        return Path(os.environ["DOCQA_CONFIG"])
    if RELEASE.exists():
        production = (yaml.safe_load(RELEASE.read_text()) or {}).get("production")
        if production:
            return Path(production["config"])
    return Path("configs/apps/docqa-v1.0.yaml")


CONFIG = production_config()
DOCUMENTS = Path(os.environ.get("DOCQA_DOCUMENTS", "datasets/policies/v1/documents.jsonl"))
BASE_DIR = Path(os.environ.get("DOCQA_BASE_DIR", "."))
TIMEOUT_S = float(os.environ.get("DOCQA_TIMEOUT_S", "60"))

REQUESTS = Counter("docqa_requests_total", "Questions answered, by outcome", ["version", "status"])
ABSTENTIONS = Counter("docqa_abstentions_total", "Questions declined as not answerable", ["version"])
LATENCY = Histogram("docqa_request_latency_seconds", "End-to-end latency", ["version"],
                    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60))
TOKENS = Counter("docqa_tokens_total", "Model tokens", ["version", "direction"])

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    raw = yaml.safe_load(CONFIG.read_text())
    qa = DocQA(DocQASettings.model_validate(raw["settings"]), BASE_DIR)
    docs = [json.loads(line) for line in DOCUMENTS.read_text().splitlines() if line.strip()]
    qa.load_documents({d["document_id"]: d["text"] for d in docs})
    state.update(qa=qa, version=str(raw["version"]), model=raw["settings"]["model"]["name"],
                 titles={d["document_id"]: d["title"] for d in docs})
    yield
    state.clear()


app = FastAPI(title="DocQA", description="Answers questions about policy documents, with citations.",
              lifespan=lifespan)


class AskRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str | None
    found: bool | None
    citations: list[int]
    status: str
    version: str
    model: str
    latency_s: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": state.get("version"), "model": state.get("model")}


@app.get("/v1/documents")
def documents() -> list[dict]:
    return [{"document_id": k, "title": v} for k, v in sorted(state["titles"].items())]


@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    qa: DocQA = state["qa"]
    if not qa.has_document(req.document_id):
        raise HTTPException(status_code=404, detail=f"Unknown document {req.document_id}")
    version = state["version"]
    started = time.perf_counter()
    result = await run_in_threadpool(qa.ask, req.document_id, req.question, TIMEOUT_S)
    elapsed = time.perf_counter() - started
    REQUESTS.labels(version, result.status).inc()
    if result.status == "ok":
        LATENCY.labels(version).observe(elapsed)
        if result.found is False:
            ABSTENTIONS.labels(version).inc()
    if result.input_tokens:
        TOKENS.labels(version, "input").inc(result.input_tokens)
    if result.output_tokens:
        TOKENS.labels(version, "output").inc(result.output_tokens)
    if result.status in ("timeout", "error"):
        raise HTTPException(status_code=504 if result.status == "timeout" else 502, detail=result.error)
    return AskResponse(answer=result.answer, found=result.found, citations=result.citations, status=result.status,
                       version=version, model=state["model"], latency_s=round(elapsed, 3))


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
