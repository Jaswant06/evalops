"""Operational metrics for evaluation runs, pushed to a Prometheus Pushgateway.

An evaluation is a batch job, so it pushes its metrics when it finishes
instead of being scraped. Latency is labelled by source (live inference or
cache replay) so replayed numbers are never mistaken for fresh measurements.

Enabled when EVALOPS_PUSHGATEWAY_URL is set.
"""

from __future__ import annotations

import os
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120)


class RunTelemetry:
    def __init__(self, app: str, version: str, dataset: str):
        self.url = os.environ.get("EVALOPS_PUSHGATEWAY_URL")
        self.grouping = {"app": app, "version": version, "dataset": dataset}
        self.registry = CollectorRegistry()
        self.started = time.time()
        self.requests = Counter("evalops_app_requests_total", "Application requests made by the evaluator",
                                ["status", "source"], registry=self.registry)
        self.latency = Histogram("evalops_app_request_latency_seconds",
                                 "Original application latency of each answer, labelled by whether it was "
                                 "measured live in this run or replayed from a recording",
                                 ["source"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.evaluation_duration = Histogram(
            "evalops_case_evaluation_duration_seconds",
            "Wall time this run spent obtaining each answer (near zero when replayed; not the app's latency)",
            buckets=(0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60), registry=self.registry)
        self.tokens = Counter("evalops_app_tokens_total", "Tokens used by the application",
                              ["direction"], registry=self.registry)
        self.judge_calls = Counter("evalops_judge_calls_total", "Judge calls by outcome", ["status"],
                                   registry=self.registry)
        self.duration = Gauge("evalops_run_duration_seconds", "Wall-clock duration of the last run",
                              registry=self.registry)
        self.cases = Gauge("evalops_run_cases", "Cases in the last run", registry=self.registry)
        self.status = Gauge("evalops_run_status", "1 for the status of the last run", ["status"],
                            registry=self.registry)
        self.finished = Gauge("evalops_run_finished_timestamp_seconds", "When the last run ended",
                              registry=self.registry)
        self.model_memory = Gauge("evalops_model_memory_bytes", "Memory held by each loaded model during a live run",
                                  ["model"], registry=self.registry)
        self.live_fraction = Gauge("evalops_run_live_fraction", "Share of answers generated live (not replayed)",
                                   registry=self.registry)

    def request(self, status: str, cached: bool, latency_s: float | None,
                input_tokens: int | None, output_tokens: int | None) -> None:
        source = "cache" if cached else "live"
        self.requests.labels(status=status, source=source).inc()
        if latency_s is not None and status == "ok":
            self.latency.labels(source=source).observe(latency_s)
        if input_tokens:
            self.tokens.labels(direction="input").inc(input_tokens)
        if output_tokens:
            self.tokens.labels(direction="output").inc(output_tokens)

    def resources(self, models: list[dict]) -> None:
        for m in models:
            self.model_memory.labels(model=m["name"]).set(m["bytes"])

    def judge(self, status: str) -> None:
        self.judge_calls.labels(status=status).inc()

    def finish(self, status: str, n_cases: int) -> None:
        self.duration.set(time.time() - self.started)
        self.cases.set(n_cases)
        for s in ("completed", "failed", "interrupted"):
            self.status.labels(status=s).set(1 if s == status else 0)
        self.finished.set_to_current_time()
        self.push()

    def push(self) -> None:
        if not self.url:
            return
        try:
            push_to_gateway(self.url, job="evalops_run", grouping_key=self.grouping, registry=self.registry)
        except OSError as exc:
            # Monitoring must not change an evaluation's outcome; report and carry on.
            print(f"warning: could not push metrics to {self.url}: {exc}")
