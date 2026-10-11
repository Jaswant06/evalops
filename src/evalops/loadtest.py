"""Throughput under concurrent load, measured against a running HTTP service.

Evaluation runs send requests one at a time so their latency numbers are clean;
this measures what happens when requests overlap: requests per second, latency
percentiles and errors at each concurrency level. Questions come from the
dataset, so the load is realistic. A local Ollama server processes a limited
number of requests in parallel, so past that point extra concurrency mostly
turns into queueing time; the numbers show where that happens.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from .datasets import Dataset


def run_level(url: str, dataset: Dataset, concurrency: int, n_requests: int, timeout_s: float) -> dict:
    cases = [dataset.cases[i % len(dataset.cases)] for i in range(n_requests)]
    latencies: list[float] = []
    errors = 0

    def one(case) -> None:
        nonlocal errors
        started = time.perf_counter()
        try:
            r = httpx.post(f"{url}/v1/ask", json={"document_id": case.input.document_id,
                                                  "question": case.input.question}, timeout=timeout_s)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        if ok:
            latencies.append(time.perf_counter() - started)
        else:
            errors += 1

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, cases))
    wall = time.perf_counter() - wall
    lat = np.array(latencies) if latencies else np.array([np.nan])
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "errors": errors,
        "requests_per_second": len(latencies) / wall,
        "latency_p50_s": float(np.nanpercentile(lat, 50)),
        "latency_p95_s": float(np.nanpercentile(lat, 95)),
        "wall_s": wall,
    }


def run(url: str, dataset: Dataset, levels: list[int], n_requests: int, timeout_s: float = 120) -> list[dict]:
    httpx.get(f"{url}/health", timeout=10).raise_for_status()
    # Warm up so the first level does not pay the model load.
    run_level(url, dataset, 1, 2, timeout_s)
    return [run_level(url, dataset, c, n_requests, timeout_s) for c in levels]
