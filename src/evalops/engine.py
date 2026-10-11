"""The evaluation pipeline.

 1. Load the dataset version and verify its hash and schema.
 2. Validate the app, scoring, gate and pricing configuration.
 3. Initialize the run: fingerprint every input, record the run as `running`.
 4. Execute the cases through the app's adapter (cache first, live on a miss).
 5. Capture responses; every one is persisted as soon as it exists.
 6. Run the deterministic checks.
 7. Run the LLM judge where semantics are needed.
 8. Aggregate metrics overall and per category.
 9. Persist results (database, MLflow) and push operational metrics.
10. Write the report, then, and only then, mark the run `completed`.

Any exception leaves the run `failed` (or `interrupted` on Ctrl-C), never
`completed`. `--resume RUN_ID` continues an unfinished run: cases already
generated or scored are skipped, and the upserts make re-running a case
harmless.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from . import __version__
from .adapters import AdapterNotReady, AppAdapter, load_adapter_class
from .cache import CacheMiss, CacheMode, ResponseCache, now_iso
from .checks import run_checks
from .config import (AppConfig, ConfigError, FaultInjection, PlatformConfig, load_gate, load_pricing,
                     load_scoring)
from .datasets import Dataset
from .datasets import load as load_dataset
from .hashing import sha256_json, sha256_text
from .judge import Judge, structured_judgment, summarize_claims, summarize_correctness
from .llm import LLMError, installed_digest
from .metrics import CaseRecord, compute_all
from .report import render_run_report
from .schemas import AppResponse, AppVersion, CaseExecution, TestCase
from .store import Store
from .telemetry import RunTelemetry
from .tracking import Tracker


class RunError(RuntimeError):
    """The run could not be started or completed."""


RunMode = Literal["replay", "live", "rescore", "auto"]

# How each mode uses recorded outputs: (application answers, judge verdicts).
MODE_POLICY: dict[str, tuple[CacheMode, CacheMode]] = {
    "replay": ("replay", "replay"),   # nothing calls a model; a missing recording is an error
    "live": ("refresh", "use"),       # the app answers every case now, so latency is measured now
    "rescore": ("replay", "use"),     # recorded answers, scored by the current scoring config
    "auto": ("use", "use"),           # recordings where they exist, live inference for the rest
}

MODE_DESCRIPTION = {
    "replay": "REPLAY: recorded answers and judge verdicts are re-scored by the current code; no model is called.",
    "live": "LIVE: the app's model answers every question now; latency is measured in this run.",
    "rescore": "RESCORE: recorded answers are scored with the current scoring config; the judge runs live "
               "only where that config has no recording.",
    "auto": "AUTO: recorded outputs are reused where they exist; anything missing runs live.",
}


MODE_POLICY_TEXT = {
    "replay": "Every answer and verdict must come from a recording of this exact version; a missing one stops the run.",
    "live": "Every answer is generated now. A failed request is recorded as a failure; it is never replaced by a "
            "recording.",
    "rescore": "Answers come from recordings of this exact version; only scoring runs now.",
    "auto": "A recording of this exact version is reused when it exists{age}; otherwise the answer is generated "
            "now. A failed live request is recorded as a failure; it is never replaced by a recording.",
}


def summarize_execution(mode: str, records: list["CaseRecord"], judge_live: int, judge_stats: dict,
                        max_recording_age_days: float | None = None) -> dict:
    """Where this run's numbers came from: live inference, recordings, or both."""
    replayed = [r for r in records if r.execution_mode == "replayed"]
    times = sorted(r.recorded_at for r in replayed if r.recorded_at)
    live = len(records) - len(replayed)
    return {
        "mode": mode,
        "app": {"live": live, "replayed": len(replayed),
                "recorded_from": times[0] if times else None, "recorded_to": times[-1] if times else None},
        "judge": {"live": judge_live, "replayed": judge_stats.get("hits", 0),
                  "rejected": judge_stats.get("rejected", 0)},
        # Latency always comes from when an answer was generated, never from replaying it.
        "latency_source": "measured in this run" if not replayed else ("recorded" if not live else "mixed"),
        "policy": MODE_POLICY_TEXT[mode].format(
            age=f" and is at most {max_recording_age_days:g} days old" if max_recording_age_days else ""),
        "evaluation_duration_s": round(sum(r.evaluation_duration_s or 0 for r in records), 3),
        "failures": {"application": sum(r.failure_class == "application" for r in records),
                     "scoring": sum(r.failure_class == "scoring" for r in records)},
    }


@dataclass
class RunOptions:
    dataset: str | None = None
    mode: RunMode = "auto"
    record: bool = True              # write new outputs to the recordings
    max_recording_age_days: float | None = None   # auto mode: older recordings count as missing
    resume: str | None = None
    limit: int | None = None
    concurrency: int | None = None
    judge_concurrency: int | None = None
    quiet: bool = False
    # Return an existing completed run with the same fingerprint instead of running again.
    reuse: bool = False


@dataclass
class RunResult:
    run_id: str
    status: str
    app_version: AppVersion
    dataset: Dataset
    records: list[CaseRecord]
    metrics: dict
    run_dir: Path
    execution: dict = field(default_factory=dict)


@dataclass
class AppHandle:
    """An app version to evaluate: from a config file (can run live) or a release snapshot (replay only)."""

    version: AppVersion
    adapter_cls: type[AppAdapter]
    adapter: AppAdapter | None
    faults: FaultInjection | None = None


def git_sha(base_dir: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base_dir, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def app_from_config(config: AppConfig, base_dir: Path) -> AppHandle:
    try:
        adapter_cls = load_adapter_class(config.adapter)
        adapter = adapter_cls(config.settings, base_dir)
        components = adapter.components()
    except Exception as exc:
        raise ConfigError(f"App {config.app} {config.version}: cannot load adapter {config.adapter}: {exc}") from None
    fingerprint = sha256_json({"app": config.app, "adapter": config.adapter, "components": components})
    version = AppVersion(app=config.app, version=config.version, adapter=config.adapter,
                         components=components, fingerprint=fingerprint)
    return AppHandle(version, adapter_cls, adapter, config.fault_injection)


def app_from_snapshot(snapshot: dict) -> AppHandle:
    version = AppVersion.model_validate(snapshot)
    expected = sha256_json({"app": version.app, "adapter": version.adapter, "components": version.components})
    if expected != version.fingerprint:
        raise ConfigError(f"Release snapshot for {version.app} {version.version} has an inconsistent fingerprint")
    return AppHandle(version, load_adapter_class(version.adapter), adapter=None)


def case_input_hash(case: TestCase, dataset: Dataset) -> str:
    document = dataset.documents[case.input.document_id]
    return sha256_json({"input": case.input.model_dump(), "document_sha256": sha256_text(document.text)})


def inject_fault(execution: CaseExecution, case: TestCase, faults: FaultInjection, timeout_s: float) -> str | None:
    """Deterministically replace some responses with failures. Returns the fault name."""
    u = int(sha256_text(f"{faults.seed}:{case.case_id}")[:8], 16) / 2**32
    r = execution.response
    if u < faults.timeout_rate:
        execution.response = AppResponse(status="timeout", passages=r.passages, error="injected timeout",
                                         timing=None)
        return "timeout"
    if u < faults.timeout_rate + faults.malformed_rate:
        # Truncate the raw output: the schema parser, not this function, must catch it.
        execution.response = r.model_copy(update={"raw_output": (r.raw_output or "")[: len(r.raw_output or "") // 2]})
        return "malformed"
    if u < faults.timeout_rate + faults.malformed_rate + faults.error_rate:
        execution.response = AppResponse(status="error", error="injected error")
        return "error"
    if faults.extra_latency_s and r.status == "ok" and r.timing is not None:
        execution.response = r.model_copy(update={"timing": r.timing.model_copy(
            update={"latency_s": round(r.timing.latency_s + faults.extra_latency_s, 4)})})
        return "slowdown"
    return None



def error_class(exc: BaseException) -> str:
    """Why a run failed. None of these are application failures: the run itself is invalid."""
    from .store import StoreUnavailable
    from .tracking import TrackingError

    if isinstance(exc, (StoreUnavailable, TrackingError, LLMError)) or (
            isinstance(exc, RunError) and str(exc).startswith("Judge")):
        return "infrastructure"      # database, MLflow or the judge model unavailable
    if isinstance(exc, CacheMiss):
        return "missing_recording"   # replay asked for an output that was never recorded
    if isinstance(exc, AdapterNotReady):
        return "application_not_ready"   # the app cannot serve this version (model missing, digest drift)
    return "internal"


class Evaluator:
    def __init__(self, platform: PlatformConfig, base_dir: Path = Path("."), store: Store | None = None,
                 tracker: Tracker | None = None):
        self.platform = platform
        self.base_dir = base_dir
        self.store = store or Store()
        self.tracker = tracker
        self.used_cache_keys: dict[str, set[str]] | None = None
        # Step 2: every config is validated before anything runs.
        self.scoring = load_scoring(base_dir / platform.scoring)
        self.scoring_fingerprint = self.scoring.fingerprint(base_dir)
        self.pricing = load_pricing(base_dir / platform.pricing)
        self.gate = load_gate(base_dir / platform.gate)

    def _log(self, options: RunOptions, msg: str) -> None:
        if not options.quiet:
            print(msg, file=sys.stderr, flush=True)

    def run(self, app: AppHandle, options: RunOptions = RunOptions()) -> RunResult:
        # Step 1: dataset, verified.
        dataset = load_dataset(options.dataset or self.platform.dataset, self.base_dir / "datasets")
        cases = dataset.cases[: options.limit] if options.limit else dataset.cases
        if options.mode not in MODE_POLICY:
            raise RunError(f"Unknown mode {options.mode!r}; use one of {sorted(MODE_POLICY)}")
        app_cache_mode, judge_cache_mode = MODE_POLICY[options.mode]
        if app.adapter is None and app_cache_mode != "replay":
            raise RunError("A release snapshot has no app to call; evaluate it with mode 'replay' or 'rescore'.")

        # Step 3: identity of this run.
        faults = app.faults.model_dump() if app.faults else None
        run_fingerprint = sha256_json({
            "dataset_hash": dataset.dataset_hash, "split": dataset.split, "limit": options.limit,
            "app_fingerprint": app.version.fingerprint, "scoring_fingerprint": self.scoring_fingerprint,
            "faults": faults,
        })
        self.store.init()  # raises StoreUnavailable: nothing runs without somewhere to record it

        if options.reuse and not options.resume:
            row = self.store.latest_completed(run_fingerprint)
            if row is not None:
                self._log(options, f"Reusing completed run {row['run_id']} (same inputs)")
                records = [CaseRecord.model_validate(c["record"]) for c in self.store.get_cases(row["run_id"]).values()]
                return RunResult(row["run_id"], "completed", app.version, dataset, records,
                                 self.store.get_metrics(row["run_id"]),
                                 self.base_dir / self.platform.runs_dir / row["run_id"], row.get("execution") or {})

        existing: dict[str, dict] = {}
        if options.resume:
            row = self.store.get_run(options.resume)
            if row is None:
                raise RunError(f"No run {options.resume} to resume")
            if row["status"] == "completed":
                raise RunError(f"Run {options.resume} already completed")
            if row["run_fingerprint"] != run_fingerprint:
                raise RunError(f"Run {options.resume} was started with different inputs; start a new run instead")
            run_id = options.resume
            existing = self.store.get_cases(run_id)
            self.store.update_run(run_id, status="running", error=None, finished_at=None)
            self._log(options, f"Resuming {run_id}: {len(existing)} of {len(cases)} cases already done")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            run_id = f"{app.version.app}-v{app.version.version}-{stamp}-{secrets.token_hex(2)}"
            self.store.create_run({
                "run_id": run_id, "app": app.version.app, "app_version_label": app.version.version,
                "app_fingerprint": app.version.fingerprint, "app_version": app.version.model_dump(),
                "dataset_ref": dataset.ref, "dataset_hash": dataset.dataset_hash,
                "scoring_fingerprint": self.scoring_fingerprint, "run_fingerprint": run_fingerprint,
                "mode": options.mode, "evalops_version": __version__,
                "git_sha": git_sha(self.base_dir), "n_cases": len(cases),
            })

        run_dir = self.base_dir / self.platform.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        telemetry = RunTelemetry(app.version.app, app.version.version, dataset.ref)
        self._log(options, f"Mode: {MODE_DESCRIPTION[options.mode]}")
        app_cache = ResponseCache(self.base_dir / self.platform.cache_dir, "app", app_cache_mode, write=options.record,
                                  max_age_days=options.max_recording_age_days if options.mode == "auto" else None)
        judge_cache = ResponseCache(self.base_dir / self.platform.cache_dir, "judge", judge_cache_mode,
                                    write=options.record)
        if self.used_cache_keys is not None:  # collected by `evalops cache-prune`
            app_cache.used = self.used_cache_keys.setdefault("app", set())
            judge_cache.used = self.used_cache_keys.setdefault("judge", set())
        status = "failed"
        try:
            records, judge_live = self._execute(app, dataset, cases, run_id, existing, options, app_cache,
                                                judge_cache, telemetry)

            # Step 8: metrics.
            metrics = compute_all(records)
            self.store.save_metrics(run_id, metrics)
            if self.store.count_cases(run_id, stage="scored") != len(cases):
                raise RunError("Not every case was scored; refusing to mark the run completed")

            # Steps 9-10: artifacts, tracking, report; then completed.
            execution = summarize_execution(options.mode, records, judge_live, judge_cache.stats.as_dict(),
                                            options.max_recording_age_days)
            telemetry.live_fraction.set(execution["app"]["live"] / max(1, len(records)))
            run_row = self.store.get_run(run_id)
            run_row["execution"] = execution
            self._write_artifacts(run_dir, run_row, records, metrics, dataset)
            if self.tracker:
                mlflow_run_id, model_id = self.tracker.log_run(run_row, metrics, run_dir)
                self.store.update_run(run_id, mlflow_run_id=mlflow_run_id, mlflow_model_id=model_id)
            self.store.update_run(run_id, status="completed", execution=execution,
                                  report_path=str(run_dir / "report.md"))
            status = "completed"
            self._log(options, f"Run {run_id} completed. Report: {run_dir / 'report.md'}")
            return RunResult(run_id, status, app.version, dataset, records, metrics, run_dir, execution)
        except KeyboardInterrupt:
            status = "interrupted"
            self._mark(run_id, "interrupted", "interrupted; resume with --resume " + run_id, "interrupted")
            raise
        except BaseException as exc:
            self._mark(run_id, "failed", f"{type(exc).__name__}: {exc}", error_class(exc))
            raise
        finally:
            telemetry.finish(status, len(cases))
            (run_dir / "status.json").write_text(json.dumps({"run_id": run_id, "status": status}) + "\n")


    def _mark(self, run_id: str, status: str, error: str, error_class: str) -> None:
        try:
            self.store.update_run(run_id, status=status, error=error[:2000], error_class=error_class)
        except Exception:
            pass  # the database may be the thing that failed; status.json still records it

    # --- steps 4-7 ----------------------------------------------------------------------

    def _execute(self, app: AppHandle, dataset: Dataset, cases: list[TestCase], run_id: str,
                 existing: dict[str, dict], options: RunOptions, app_cache: ResponseCache,
                 judge_cache: ResponseCache, telemetry: RunTelemetry) -> tuple[list[CaseRecord], int]:
        exec_cfg = self.platform.execution
        timeout_s = exec_cfg.timeout_s
        components = app.version.components
        fingerprint = app.version.fingerprint
        ready_lock = threading.Lock()
        ready = {"app": False}

        def ensure_app_ready() -> None:
            with ready_lock:
                if not ready["app"]:
                    app.adapter.check_ready()
                    app.adapter.warm_up()  # load the model before anything is timed
                    ready["app"] = True

        if app.adapter is not None:
            app.adapter.prepare(dataset.documents)

        def generate(case: TestCase) -> CaseExecution:
            input_hash = case_input_hash(case, dataset)
            entry = app_cache.get(fingerprint, case.case_id, input_hash, components)
            if entry is not None:
                return CaseExecution(response=AppResponse.model_validate(entry.payload["response"]),
                                     cache_hit=True, recorded_at=entry.recorded_at,
                                     attempts=entry.payload.get("attempts", 1))
            ensure_app_ready()
            response, attempts = None, 0
            for attempts in range(1, exec_cfg.max_retries + 2):
                try:
                    response = app.adapter.run_case(case, timeout_s)
                except Exception as exc:  # an adapter bug is still a failed request, not a crash
                    response = AppResponse(status="error", error=f"adapter raised {type(exc).__name__}: {exc}")
                if response.status not in ("timeout", "error"):
                    break
            # Infrastructure errors are not model behaviour and are not recorded;
            # outputs, malformed outputs and timeouts are.
            if response.status != "error":
                app_cache.put(fingerprint, case.case_id, input_hash, components,
                              {"response": response.model_dump(), "attempts": attempts})
            return CaseExecution(response=response, cache_hit=False, recorded_at=now_iso(), attempts=attempts)

        # Steps 4-5: execute and persist each response as soon as it exists.
        executions: dict[str, tuple[CaseExecution, str | None]] = {}
        todo = []
        for case in cases:
            row = existing.get(case.case_id)
            if row and row.get("execution"):
                ex = row["execution"]
                executions[case.case_id] = (CaseExecution.model_validate(ex["execution"]), ex.get("injected_fault"),
                                            ex.get("duration_s"))
            else:
                todo.append(case)

        done = len(executions)
        started = time.perf_counter()

        def generate_and_store(case: TestCase) -> None:
            nonlocal done
            case_started = time.perf_counter()
            execution = generate(case)
            duration = round(time.perf_counter() - case_started, 4)   # what this run spent, not the app's latency
            fault = inject_fault(execution, case, app.faults, timeout_s) if app.faults else None
            r = execution.response
            telemetry.request(r.status, execution.cache_hit, r.timing.latency_s if r.timing else None,
                              r.usage.input_tokens, r.usage.output_tokens)
            telemetry.evaluation_duration.observe(duration)
            self.store.upsert_case(run_id, case.case_id, "generated", case.categories,
                                   execution={"execution": execution.model_dump(), "injected_fault": fault,
                                              "duration_s": duration})
            executions[case.case_id] = (execution, fault, duration)
            done += 1
            if app.faults and app.faults.abort_after == done:
                raise KeyboardInterrupt(f"simulated interruption after {done} cases (fault injection)")
            if done % 25 == 0 or done == len(cases):
                self._log(options, f"  answers {done}/{len(cases)} ({time.perf_counter() - started:.0f}s)")

        self._parallel(generate_and_store, todo, options.concurrency or exec_cfg.concurrency)
        if ready["app"]:   # something ran live: record what the app's models held in memory
            telemetry.resources(app.adapter.resource_usage())

        # Step 6: deterministic checks (cheap, always recomputed).
        output_model = app.adapter_cls.output_model
        checks = {c.case_id: run_checks(c, executions[c.case_id][0].response, dataset.documents, output_model,
                                        self.scoring.checks) for c in cases}

        # Step 7: the judge, only where an answer was given.
        judge_ready = {"done": False}

        def ensure_judge_ready() -> None:
            with ready_lock:
                if judge_ready["done"]:
                    return
                model = self.scoring.judge.model
                try:
                    digest = installed_digest(model.name)
                except LLMError as exc:
                    raise RunError(f"Judge unavailable: {exc}") from None
                if digest != model.digest:
                    raise RunError(f"Judge model {model.name} digest {digest} does not match the pinned "
                                   f"{model.digest}; scores would not be comparable")
                judge_ready["done"] = True

        judge = self.make_judge(judge_cache, ensure_judge_ready)
        scored = {cid for cid, row in existing.items() if row.get("stage") == "scored"}
        pending = [c for c in cases if c.case_id not in scored and checks[c.case_id]["answered"]]
        pending.sort(key=lambda c: (c.input.document_id, c.case_id))
        claims_results: dict[str, object] = {}
        correctness_results: dict[str, object] = {}

        def judge_claims(case: TestCase) -> None:
            j = judge.claims(case, executions[case.case_id][0].response, dataset.documents[case.input.document_id])
            telemetry.judge(j.status)
            claims_results[case.case_id] = j

        def judge_correctness(case: TestCase) -> None:
            j = judge.correctness(case, executions[case.case_id][0].response)
            telemetry.judge(j.status)
            correctness_results[case.case_id] = j

        if pending:
            self._log(options, f"  judging {len(pending)} answers")
        jc = options.judge_concurrency or exec_cfg.judge_concurrency
        self._parallel(judge_claims, pending, jc)
        self._parallel(judge_correctness, [c for c in pending if c.expected.answerable], jc)

        # Build and persist the scored records.
        records: list[CaseRecord] = []
        for case in cases:
            if case.case_id in scored:
                records.append(CaseRecord.model_validate(existing[case.case_id]["record"]))
                continue
            execution, fault, duration = executions[case.case_id]
            r = execution.response
            ck = checks[case.case_id]
            claims_j = claims_results.get(case.case_id)
            corr_j = correctness_results.get(case.case_id)
            errors = [j.error for j in (claims_j, corr_j) if j is not None and j.status == "unevaluable"]
            judgments = [structured_judgment(name, j) for name, j in (("correctness", corr_j), ("claims", claims_j))
                         if j is not None]
            failure_class = ("application" if ck["status"] != "ok"
                             else "scoring" if any(j["status"] == "unevaluable" for j in judgments) else None)
            record = CaseRecord(
                case_id=case.case_id, categories=case.categories, answerable=case.expected.answerable,
                status=ck["status"], schema_valid=ck["schema_valid"], answered=ck["answered"],
                abstained=ck["abstained"], abstention_correct=ck["abstention_correct"],
                execution_mode="replayed" if execution.cache_hit else "live",
                recorded_at=execution.recorded_at, attempts=execution.attempts, injected_fault=fault,
                original_latency_s=r.timing.latency_s if r.timing else None,
                original_ttft_s=r.timing.ttft_s if r.timing else None,
                evaluation_duration_s=duration, failure_class=failure_class, judgments=judgments,
                input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                cost_usd=self._cost(app, r),
                n_citations=ck["n_citations"], citations_invalid=ck["citations_invalid"],
                citation_missing=ck["citation_missing"], retrieval_recall=ck["retrieval_recall"],
                retrieval_precision=ck["retrieval_precision"], evidence_coverage=ck["evidence_coverage"],
                reference_f1=ck["reference_f1"],
                judge_claims_status=claims_j.status if claims_j else "skipped",
                judge_correctness_status=corr_j.status if corr_j else "skipped",
                judge_error="; ".join(errors) or None,
                **(summarize_correctness(corr_j) if corr_j else {}),
                **({k: v for k, v in summarize_claims(claims_j).items()} if claims_j else {}),
                answer=r.answer, error=r.error,
            )
            self.store.upsert_case(run_id, case.case_id, "scored", case.categories, record=record.model_dump())
            records.append(record)
        return records, judge.live_calls

    def make_judge(self, cache: ResponseCache, before_live_call) -> Judge:
        """Separate method so tests can substitute a scripted judge client."""
        return Judge(self.scoring.judge, self.base_dir, cache, before_live_call=before_live_call)

    def _cost(self, app: AppHandle, r: AppResponse) -> float | None:
        model = app.version.components.get("model", {})
        price = self.pricing.models.get(model.get("name") if isinstance(model, dict) else None)
        if price is None or r.usage.input_tokens is None or r.usage.output_tokens is None:
            return None
        return (r.usage.input_tokens * price.input_per_mtok + r.usage.output_tokens * price.output_per_mtok) / 1e6

    @staticmethod
    def _parallel(fn, items: list, workers: int) -> None:
        if not items:
            return
        if workers <= 1:
            for item in items:
                fn(item)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(fn, item) for item in items]:
                future.result()  # re-raise the first failure

    def _write_artifacts(self, run_dir: Path, run_row: dict, records: list[CaseRecord], metrics: dict,
                         dataset: Dataset) -> None:
        def default(o):
            return o.isoformat() if isinstance(o, datetime) else str(o)

        (run_dir / "run.json").write_text(json.dumps(run_row, indent=2, default=default) + "\n")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with (run_dir / "cases.jsonl").open("w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
        (run_dir / "report.md").write_text(render_run_report(run_row, records, metrics, dataset, self.pricing))
