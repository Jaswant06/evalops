"""Gate, promote and roll back: three separate steps.

gate      Evaluates the candidate and production on the same cases (production
          is replayed from the snapshot in release.yaml and rescored by the
          current code), applies the rules and records a decision bound to the
          exact versions involved. It never changes production.

promote   Runs no evaluation. It accepts a candidate only with a recorded,
          passing gate decision for exactly the versions on disk now: the same
          app fingerprint (model, digest, prompt, settings), dataset, scoring
          and gate rules, decided against the version that is still in
          production. Then, under a lock, it rewrites release.yaml atomically
          and records the promotion. Promoting what is already in production is
          a no-op. Every refusal is recorded too.

rollback  Restores the previous production version from the release history.
          It is the emergency path, so it does not need a new gate decision,
          but it only accepts a version that was promoted before.
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig, ProductionRelease, ReleaseRecord, load_app, load_release, save_release
from .datasets import load as load_dataset
from .decision_report import render_markdown
from .engine import AppHandle, Evaluator, RunMode, RunOptions, app_from_config, app_from_snapshot
from .gate import GateDecision, RunData, evaluate_gate
from .metrics import CaseRecord


class PromotionError(RuntimeError):
    """The release state could not be changed safely (for example, another promotion holds the lock)."""


@dataclass
class GateOutcome:
    decision: GateDecision
    candidate: RunData
    baseline: RunData | None
    questions: dict[str, str]
    report: str
    report_path: Path


@dataclass
class PromotionResult:
    status: str                      # promoted | already_in_production | refused | rolled_back
    message: str
    problems: list[str] = field(default_factory=list)
    decision_id: str | None = None


def actor() -> str:
    if os.environ.get("GITHUB_ACTIONS"):
        return f"github-actions ({os.environ.get('GITHUB_ACTOR', 'unknown')})"
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or getpass.getuser()


def run_data(evaluator: Evaluator, run_id: str) -> RunData:
    run = evaluator.store.get_run(run_id)
    cases = evaluator.store.get_cases(run_id)
    records = [CaseRecord.model_validate(c["record"]) for c in cases.values() if c.get("record")]
    return RunData(run=run, records=sorted(records, key=lambda r: r.case_id))


def evaluate(evaluator: Evaluator, app: AppHandle, mode: RunMode, quiet: bool = False) -> RunData:
    result = evaluator.run(app, RunOptions(mode=mode, reuse=True, quiet=quiet))
    return run_data(evaluator, result.run_id)


def gate_candidate(evaluator: Evaluator, candidate: AppHandle, release: ReleaseRecord, *,
                   mode: RunMode = "auto", context: str = "cli", quiet: bool = False,
                   baseline_snapshot: dict | None = None, candidate_config: str | None = None) -> GateOutcome:
    snapshot = baseline_snapshot if baseline_snapshot is not None else (
        release.production.app_version if release.production else None)
    baseline = evaluate(evaluator, app_from_snapshot(snapshot), "replay", quiet) if snapshot else None
    cand = evaluate(evaluator, candidate, mode, quiet)
    decision = evaluate_gate(evaluator.gate, cand, baseline)
    decision.candidate_config = candidate_config

    dataset = load_dataset(cand.run["dataset_ref"], evaluator.base_dir / "datasets")
    questions = {c.case_id: c.input.question for c in dataset.cases}
    report = render_markdown(decision, cand, baseline, evaluator.gate, questions)

    out_dir = evaluator.base_dir / evaluator.platform.runs_dir / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{decision.decision_id}.md"
    report_path.write_text(report)
    (out_dir / f"{decision.decision_id}.json").write_text(json.dumps(decision.to_dict(), indent=2, default=str))
    store = evaluator.store
    store.save_decision({
        "decision_id": decision.decision_id, "candidate_run_id": decision.candidate_run_id,
        "baseline_run_id": decision.baseline_run_id, "decision": decision.decision,
        "mandatory_failures": len(decision.mandatory_failures),
        "advisory_failures": len(decision.advisory_failures),
        "rules": [r.__dict__ for r in decision.rules], "gate_fingerprint": decision.gate_fingerprint,
        "candidate_fingerprint": decision.candidate_fingerprint,
        "baseline_fingerprint": decision.baseline_fingerprint, "candidate_config": candidate_config,
        "dataset_ref": decision.dataset_ref, "dataset_hash": decision.dataset_hash,
        "scoring_fingerprint": decision.scoring_fingerprint,
        "context": context, "git_sha": cand.run.get("git_sha"), "report": report,
    })
    store.audit("gate_decision", app=candidate.version.app, version=candidate.version.version,
                fingerprint=decision.candidate_fingerprint, decision_id=decision.decision_id, actor=actor(),
                details={"decision": decision.decision, "context": context,
                         "blocking_rules": [r.rule_id for r in decision.mandatory_failures],
                         "review_rules": [r.rule_id for r in decision.advisory_failures]})
    return GateOutcome(decision, cand, baseline, questions, report, report_path)


@contextmanager
def release_lock(evaluator: Evaluator):
    """One promotion or rollback at a time, across processes."""
    lock_dir = evaluator.base_dir / evaluator.platform.runs_dir.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / "release.lock", "w") as handle:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PromotionError("Another promotion or rollback is in progress; try again when it finishes.") from None
        except ImportError:
            pass  # no fcntl (Windows): best effort
        yield


def check_decision(evaluator: Evaluator, decision: dict | None, candidate: AppHandle, config: AppConfig,
                   release: ReleaseRecord, accept_warnings: bool) -> list[str]:
    """Everything that must hold before a decision may change production. Empty list = valid."""
    if decision is None:
        return ["no gate decision exists for this exact version; run `evalops gate` first"]
    problems = []
    if config.fault_injection is not None:
        problems.append("versions with fault injection exist to test the gate and can never be promoted")
    if decision["decision"] == "fail":
        problems.append(f"the gate decision {decision['decision_id']} is BLOCKED; a failed evaluation never "
                        f"changes production")
    if decision["decision"] == "pass_with_warnings" and not accept_warnings:
        problems.append(f"the gate decision {decision['decision_id']} passed with warnings that need manual "
                        f"review; promote with --accept-warnings once reviewed")
    if decision["candidate_fingerprint"] != candidate.version.fingerprint:
        problems.append("the app version changed after it was gated (model, digest, prompt or settings differ)")
    if decision["gate_fingerprint"] != evaluator.gate.fingerprint():
        problems.append("the gate rules changed after the decision; gate the candidate again")
    if decision["scoring_fingerprint"] != evaluator.scoring_fingerprint:
        problems.append("the scoring configuration changed after the decision; gate the candidate again")
    dataset = load_dataset(evaluator.platform.dataset, evaluator.base_dir / "datasets")   # verifies hashes
    if (decision["dataset_ref"], decision["dataset_hash"]) != (dataset.ref, dataset.dataset_hash):
        problems.append(f"the decision was made on {decision['dataset_ref']} ({decision['dataset_hash'][:12]}), "
                        f"not the release dataset {dataset.ref} ({dataset.dataset_hash[:12]})")
    current = release.production.fingerprint if release.production else None
    if decision["baseline_fingerprint"] != current:
        problems.append("production changed after the decision was made; gate the candidate against the "
                        "current production")
    created = decision["created_at"]
    if created.tzinfo is None:          # SQLite returns naive datetimes; they are stored in UTC
        created = created.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    if age_hours > evaluator.gate.decision_valid_hours:
        problems.append(f"the decision is {age_hours:.0f} hours old and decisions expire after "
                        f"{evaluator.gate.decision_valid_hours:g} hours; gate the candidate again")
    run = evaluator.store.get_run(decision["candidate_run_id"])
    if run is None or run["status"] != "completed":
        problems.append(f"the evaluation run {decision['candidate_run_id']} is not a completed run")
    elif run["app_fingerprint"] != candidate.version.fingerprint:
        problems.append("the decision's evaluation run evaluated a different app version")
    elif evaluator.store.count_cases(run["run_id"], stage="scored") != run["n_cases"]:
        problems.append("the decision's evaluation run does not have a scored result for every case")
    return problems


def promote(evaluator: Evaluator, app_config: Path, *, decision_id: str | None = None,
            accept_warnings: bool = False) -> PromotionResult:
    store = evaluator.store
    store.init()
    release_path = evaluator.base_dir / evaluator.platform.release
    config = load_app(evaluator.base_dir / app_config)
    candidate = app_from_config(config, evaluator.base_dir)
    who = actor()
    with release_lock(evaluator):
        release = load_release(release_path)   # read under the lock
        ident = dict(app=config.app, version=config.version, fingerprint=candidate.version.fingerprint, actor=who)
        if config.fault_injection is not None:
            # Checked first: injected faults sit on top of the app, so the fingerprint alone cannot tell.
            problems = ["versions with fault injection exist to test the gate and can never be promoted"]
            store.audit("promotion_refused", **ident, details={"problems": problems})
            return PromotionResult("refused", "Not promoted: production is unchanged.", problems)
        if release.production and release.production.fingerprint == candidate.version.fingerprint:
            store.audit("promotion_noop", **ident, details={"config": str(app_config)})
            return PromotionResult("already_in_production",
                                   f"{config.app} {config.version} is already in production; nothing changed.")

        decision = (store.get_decision(decision_id) if decision_id
                    else store.latest_decision(candidate.version.fingerprint, evaluator.platform.dataset))
        problems = check_decision(evaluator, decision, candidate, config, release, accept_warnings)
        did = decision["decision_id"] if decision else None
        if problems:
            store.audit("promotion_refused", **ident, decision_id=did, details={"problems": problems})
            return PromotionResult("refused", "Not promoted: production is unchanged.", problems, did)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if release.production:
            release.history.append({**release.production.model_dump(mode="json"), "replaced_at": now,
                                    "replaced_by": config.version})
        release.production = ProductionRelease(
            app=config.app, version=config.version, config=app_config,
            fingerprint=candidate.version.fingerprint, app_version=candidate.version.model_dump(),
            approved_run_id=decision["candidate_run_id"], decision_id=did, promoted_at=now, promoted_by=who,
        )
        # The authoritative records first: the production pointer (release.yaml, in git) and the
        # promotion record and audit event (PostgreSQL). MLflow only mirrors them, below.
        save_release(release, release_path)
        store.save_release({"release_id": f"rel-{did.removeprefix('gate-')}", "app": config.app,
                            "version": config.version, "fingerprint": candidate.version.fingerprint,
                            "run_id": decision["candidate_run_id"], "decision_id": did})
        store.audit("promoted", **ident, decision_id=did,
                    details={"config": str(app_config),
                             "previous": release.history[-1]["version"] if release.history else None})
        warning = _mirror_to_registry(evaluator, release, release_path, ident)
    return PromotionResult("promoted", f"Promoted {config.app} {config.version} using decision {did}." + warning,
                           decision_id=did)


def _mirror_to_registry(evaluator: Evaluator, release: ReleaseRecord, release_path: Path, ident: dict) -> str:
    """Point MLflow's `production` alias at the production version. A failure leaves production as it is
    (the pointer and the database are the record) and is audited, to be repaired with `evalops sync-registry`."""
    from .tracking import TrackingError

    if evaluator.tracker is None:
        return ""
    prod = release.production
    try:
        if prod.registry_version is None:
            run = evaluator.store.get_run(prod.approved_run_id)
            if not run or not run.get("mlflow_model_id"):
                return " (MLflow alias not updated: the approving run was not logged to MLflow.)"
            prod.registry_version = evaluator.tracker.promote(prod.app, run["mlflow_model_id"], prod.decision_id,
                                                              prod.approved_run_id)
            save_release(release, release_path)
        else:
            evaluator.tracker.set_production_alias(prod.app, prod.registry_version)
        return ""
    except TrackingError as exc:
        evaluator.store.audit("registry_sync_failed", **ident, decision_id=prod.decision_id,
                              details={"error": str(exc)[:500]})
        return " WARNING: the MLflow alias could not be updated; run `evalops sync-registry`."


def sync_registry(evaluator: Evaluator) -> PromotionResult:
    """Repair the MLflow mirror from the authoritative production pointer."""
    evaluator.store.init()
    release_path = evaluator.base_dir / evaluator.platform.release
    with release_lock(evaluator):
        release = load_release(release_path)
        if release.production is None:
            return PromotionResult("refused", "Nothing is in production.", ["nothing to sync"])
        prod = release.production
        ident = dict(app=prod.app, version=prod.version, fingerprint=prod.fingerprint, actor=actor())
        warning = _mirror_to_registry(evaluator, release, release_path, ident)
        if warning:
            return PromotionResult("refused", warning.strip(), [warning.strip()])
        evaluator.store.audit("registry_synced", **ident, decision_id=prod.decision_id)
    return PromotionResult("synced", f"MLflow alias `production` now points to {prod.app} {prod.version}.")


def rollback(evaluator: Evaluator, to_version: str | None = None, *, reason: str) -> PromotionResult:
    if not reason or not reason.strip():
        raise ValueError("a rollback needs a reason (--reason), recorded in the audit log")
    store = evaluator.store
    store.init()
    release_path = evaluator.base_dir / evaluator.platform.release
    who = actor()
    with release_lock(evaluator):
        release = load_release(release_path)
        if release.production is None:
            return PromotionResult("refused", "Nothing is in production.", ["nothing to roll back"])
        current = release.production
        candidates = [h for h in reversed(release.history) if h["fingerprint"] != current.fingerprint
                      and (to_version is None or h["version"] == to_version)]
        if not candidates:
            want = f"version {to_version}" if to_version else "an earlier version"
            problem = f"the release history has no {want} to roll back to"
            store.audit("promotion_refused", app=current.app, version=to_version, actor=who,
                        details={"problems": [problem], "operation": "rollback"})
            return PromotionResult("refused", "Not rolled back: production is unchanged.", [problem])
        target = candidates[0]
        app_from_snapshot(target["app_version"])   # raises if the snapshot was tampered with
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        release.history.append({**current.model_dump(mode="json"), "replaced_at": now,
                                "replaced_by": f"{target['version']} (rollback)", "reason": reason})
        release.production = ProductionRelease.model_validate({
            **{k: v for k, v in target.items() if k not in ("replaced_at", "replaced_by")},
            "promoted_at": now, "promoted_by": who, "via": "rollback"})
        save_release(release, release_path)
        ident = dict(app=target["app"], version=target["version"], fingerprint=target["fingerprint"], actor=who)
        store.audit("rollback", **ident, decision_id=target.get("decision_id"),
                    details={"from_version": current.version, "reason": reason})
        warning = _mirror_to_registry(evaluator, release, release_path, ident)
    return PromotionResult("rolled_back", f"Rolled back production from {current.version} to {target['version']}."
                           + warning, decision_id=target.get("decision_id"))


@dataclass
class Criterion:
    name: str
    passed: bool
    detail: str


def release_criteria(evaluator: Evaluator, outcome: GateOutcome) -> list[Criterion]:
    """What a green release check means. A passing test suite is not one of them: the evaluation itself has
    to complete, meet every requirement, and leave its evidence behind."""
    d, store = outcome.decision, evaluator.store
    runs = [("candidate", outcome.candidate)] + ([("production", outcome.baseline)] if outcome.baseline else [])

    completed, persisted = [], []
    for label, data in runs:
        run = store.get_run(data.run_id)
        scored = store.count_cases(data.run_id, stage="scored")
        completed.append(run["status"] == "completed" and scored == run["n_cases"] == len(outcome.questions))
        persisted.append(bool(store.get_metrics(data.run_id)) and scored == run["n_cases"])
    absolute, relative = [], []
    for r in d.mandatory_failures:
        value = r.candidate
        breaks_floor = value is None or r.n < 1 or (r.thresholds.get("min") is not None and value < r.thresholds["min"]) \
            or (r.thresholds.get("max") is not None and value > r.thresholds["max"]) or any("cannot be verified" in x for x in r.reasons)
        (absolute if breaks_floor else relative).append(r.rule_id)
        if breaks_floor and any("vs production" in x for x in r.reasons):
            relative.append(r.rule_id)
    stored = store.get_decision(d.decision_id)
    return [
        Criterion("evaluation completed", all(completed),
                  ", ".join(f"{label}: {store.count_cases(data.run_id, stage='scored')}/{len(outcome.questions)} "
                            f"cases scored, {store.get_run(data.run_id)['status']}" for label, data in runs)),
        Criterion("comparison valid", all(p.passed for p in d.preconditions),
                  "; ".join(f"{p.name}: {p.detail}" for p in d.preconditions if not p.passed) or
                  "same dataset, cases and scoring"),
        Criterion("every required metric meets its threshold", not absolute,
                  ", ".join(absolute) or f"{sum(1 for r in d.rules if r.severity == 'mandatory')} mandatory rules"),
        Criterion("no prohibited regression against production", not relative,
                  ", ".join(relative) or "within every allowed change"),
        Criterion("results and decision persisted", all(persisted) and stored is not None
                  and stored["decision"] == d.decision and outcome.report_path.exists(),
                  f"runs, case results, metrics, decision {d.decision_id}, report"),
    ]
