"""Command line interface.

Exit codes: 0 success / gate passed, 1 gate failed / promotion blocked,
2 the evaluation itself could not be trusted (bad config, dataset tampering,
missing recordings, database down, run failure).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from . import __version__
from .cache import CacheMiss
from .config import ConfigError, ReleaseRecord, load_app, load_platform, load_release
from .datasets import DatasetError, parse_ref, verify
from .engine import Evaluator, RunError, RunOptions, app_from_config
from .decision_report import render_text
from .llm import LLMError
from .adapters import AdapterNotReady
from .metrics import compute, METRICS
from .release import (Criterion, PromotionError, evaluate, gate_candidate, promote, release_criteria, rollback,
                      sync_registry)
from .report import execution_lines, fmt
from .store import Store, StoreUnavailable
from .tracking import Tracker, TrackingError

EXIT_OK, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2
EXPECTED_ERRORS = (ConfigError, DatasetError, RunError, CacheMiss, StoreUnavailable, TrackingError,
                   AdapterNotReady, LLMError, PromotionError, FileNotFoundError, ValueError)


def make_evaluator(args) -> Evaluator:
    platform = load_platform(Path(args.config))
    if getattr(args, "scoring", None):
        # Comparison mode: score with another scoring config (judge model, prompts, check parameters).
        platform = platform.model_copy(update={"scoring": Path(args.scoring)})
    tracker = None if getattr(args, "no_mlflow", False) else Tracker.from_env()
    return Evaluator(platform, Path("."), Store(args.database), tracker)


def cmd_validate(args) -> int:
    platform = load_platform(Path(args.config))
    evaluator = Evaluator(platform, Path("."), Store("sqlite://"))  # loads scoring, gate, pricing
    print(f"platform    {args.config}: ok")
    print(f"scoring     {platform.scoring}: ok (fingerprint {evaluator.scoring_fingerprint[:12]})")
    print(f"gate        {platform.gate}: ok ({len(evaluator.gate.rules)} rules)")
    print(f"pricing     {platform.pricing}: ok")
    release = load_release(platform.release)
    print(f"release     {platform.release}: " + (f"production is {release.production.app} {release.production.version}"
                                                  if release.production else "nothing in production yet"))
    for path in sorted(Path("configs/apps").glob("*.yaml")):
        handle = app_from_config(load_app(path), Path("."))
        print(f"app         {path}: {handle.version.app} {handle.version.version} "
              f"(fingerprint {handle.version.fingerprint[:12]})")
    for manifest in sorted(Path("datasets").glob("*/*/manifest.json")):
        m = verify(manifest.parent)
        print(f"dataset     {m['name']}/{m['version']}: ok (hash {m['dataset_hash'][:12]})")
    return EXIT_OK


def cmd_dataset(args) -> int:
    name, version, _ = parse_ref(args.ref + ":x") if ":" not in args.ref else parse_ref(args.ref)
    m = verify(Path("datasets") / name / version)
    print(json.dumps({k: m[k] for k in ("name", "version", "dataset_hash", "splits", "source")}, indent=2))
    return EXIT_OK


def cmd_run(args) -> int:
    evaluator = make_evaluator(args)
    handle = app_from_config(load_app(Path(args.app)), Path("."))
    dataset = args.dataset or (f"{evaluator.platform.dataset.split(':')[0]}:{args.split}" if args.split else None)
    result = evaluator.run(handle, RunOptions(dataset=dataset, mode=args.mode, record=not args.no_record,
                                              max_recording_age_days=args.max_recording_age_days,
                                              resume=args.resume, limit=args.limit, concurrency=args.concurrency,
                                              judge_concurrency=args.judge_concurrency))
    m = result.metrics
    print(f"\n{result.app_version.app} {result.app_version.version} on {result.dataset.ref}: {result.status}")
    for line in execution_lines(result.execution):
        print(f"  {line}")
    print()
    latency_note = "" if result.execution.get("latency_source") == "measured in this run" else "  (recorded, not measured now)"
    for category, name in [("answerable", "answer_correctness"), ("unanswerable", "correct_abstention_rate"),
                           ("citation_required", "citation_support_rate"), ("all", "unsupported_claim_rate"),
                           ("all", "app_failure_rate"), ("all", "latency_p95_s"), ("all", "est_cost_per_success_usd")]:
        note = latency_note if name.startswith("latency") else ""
        print(f"  {name:28s} {category:18s} {fmt(name, m[category][name]['value']):>9}  {METRICS[name].source}{note}")
    print(f"\n  report: {result.run_dir / 'report.md'}")
    return EXIT_OK


def cmd_gate(args) -> int:
    """Evaluate a candidate against production and record a decision. Never changes production."""
    evaluator = make_evaluator(args)
    handle = app_from_config(load_app(Path(args.app)), Path("."))
    release = load_release(evaluator.platform.release)
    outcome = gate_candidate(evaluator, handle, release, mode=args.mode, context="cli", candidate_config=args.app)
    d = outcome.decision
    if d.promotable:
        flag = " --accept-warnings" if d.decision == "pass_with_warnings" else ""
        next_step = (f"Promotion: not attempted (gate never changes production).\n"
                     f"To promote this exact version: evalops promote {args.app}{flag}\n"
                     f"Report: {outcome.report_path}")
    else:
        next_step = (f"Promotion: BLOCKED. Production is unchanged.\n"
                     f"Next: fix the failing rules, then run `evalops gate {args.app}` again.\n"
                     f"Report: {outcome.report_path}")
    print("\n" + render_text(d, outcome.candidate, outcome.baseline, outcome.questions, next_step=next_step))
    return EXIT_OK if d.promotable else EXIT_BLOCKED


def cmd_promote(args) -> int:
    """Make a version production, only on the strength of a valid passing gate decision. Evaluates nothing."""
    evaluator = make_evaluator(args)
    result = promote(evaluator, Path(args.app), decision_id=args.decision, accept_warnings=args.accept_warnings)
    if result.status == "refused":
        print(f"Promotion: REFUSED. {result.message}")
        for problem in result.problems:
            print(f"  - {problem}")
        print(f"Next: evalops gate {args.app}")
        return EXIT_BLOCKED
    print(f"Promotion: {'DONE' if result.status == 'promoted' else 'NOTHING TO DO'}. {result.message}")
    if result.status == "promoted":
        print(f"{evaluator.platform.release} updated; commit it together with cache/.")
    return EXIT_OK


def cmd_rollback(args) -> int:
    evaluator = make_evaluator(args)
    result = rollback(evaluator, args.to, reason=args.reason)
    if result.status == "refused":
        print(f"Rollback: REFUSED. {result.message}")
        for problem in result.problems:
            print(f"  - {problem}")
        return EXIT_BLOCKED
    print(f"Rollback: DONE. {result.message} {evaluator.platform.release} updated; commit it.")
    return EXIT_OK


COMPARE = [("answerable", "answer_correctness"), ("answerable", "answer_quality"),
           ("unanswerable", "correct_abstention_rate"), ("citation_required", "citation_support_rate"),
           ("all", "unsupported_claim_rate"), ("all", "contradiction_rate"), ("all", "judge_unevaluable_rate"),
           ("all", "latency_p95_s"), ("all", "est_cost_per_success_usd")]


def cmd_compare(args) -> int:
    """Two runs side by side. Same app with different scoring = comparison of scoring configs."""
    store = Store(args.database)
    store.init()
    a_run, b_run = store.get_run(args.run_a), store.get_run(args.run_b)
    if a_run is None or b_run is None:
        raise ValueError("both runs must exist")
    a, b = {r.case_id: r for r in _records(store, args.run_a)}, {r.case_id: r for r in _records(store, args.run_b)}
    same_app = a_run["app_fingerprint"] == b_run["app_fingerprint"]
    same_scoring = a_run["scoring_fingerprint"] == b_run["scoring_fingerprint"]
    if same_app and not same_scoring:
        print("Comparing two SCORING configurations on the same recorded answers (not two app versions).")
    elif not same_app and same_scoring:
        print("Comparing two APP versions scored the same way. For a release decision use `evalops gate`.")
    else:
        print("Warning: the app version and the scoring both differ; differences mix the two effects.")
    print(f"  A: {args.run_a} (scoring {a_run['scoring_fingerprint'][:12]})")
    print(f"  B: {args.run_b} (scoring {b_run['scoring_fingerprint'][:12]})\n")
    print(f"  {'metric':28s} {'category':18s} {'A':>9} {'B':>9}")
    for category, name in COMPARE:
        va = compute(METRICS[name], [r for r in a.values() if category == "all" or category in r.categories])[0]
        vb = compute(METRICS[name], [r for r in b.values() if category == "all" or category in r.categories])[0]
        print(f"  {name:28s} {category:18s} {fmt(name, va):>9} {fmt(name, vb):>9}")
    both = [cid for cid in a if cid in b and a[cid].verdict and b[cid].verdict]
    flips = [cid for cid in both if a[cid].verdict != b[cid].verdict]
    print(f"\n  Correctness verdicts that differ: {len(flips)} of {len(both)}"
          + (f" (e.g. {', '.join(flips[:5])})" if flips else ""))
    return EXIT_OK


def cmd_sync_registry(args) -> int:
    result = sync_registry(make_evaluator(args))
    print(("Registry sync: DONE. " if result.status == "synced" else "Registry sync: FAILED. ") + result.message)
    return EXIT_OK if result.status == "synced" else EXIT_BLOCKED


def cmd_audit(args) -> int:
    store = Store(args.database)
    store.init()
    for e in reversed(store.list_audit(limit=args.limit)):
        what = f"{e['app'] or ''} {e['version'] or ''}".strip()
        detail = e["details"] or {}
        extra = ("; ".join(detail.get("problems", [])) or detail.get("decision", "")
                 or (f"from {detail['from_version']}: {detail.get('reason', '')}" if detail.get("from_version") else "")
                 or detail.get("error", ""))
        print(f"{e['created_at']:%Y-%m-%d %H:%M:%S}  {e['event']:<18} {what:<22} {e['decision_id'] or '':<32} "
              f"{e['actor'] or '':<20} {extra}")
    return EXIT_OK


def release_at(ref: str, path: Path) -> ReleaseRecord:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        return ReleaseRecord()  # the base has no release file yet
    return ReleaseRecord.model_validate(yaml.safe_load(out.stdout) or {})


LEADERBOARD = [("answerable", "answer_correctness"), ("unanswerable", "correct_abstention_rate"),
               ("citation_required", "citation_support_rate"), ("all", "unsupported_claim_rate"),
               ("all", "app_failure_rate"), ("all", "latency_p95_s"), ("all", "est_cost_per_success_usd")]


def cmd_ci(args) -> int:
    """The pull-request check: everything replayed, production gated against the base branch."""
    evaluator = make_evaluator(args)
    platform = evaluator.platform
    base = release_at(args.base_ref, platform.release)
    head = load_release(platform.release)
    summary = ["# EvalOps release check", "",
               "Mode: REPLAY. Recorded model outputs are re-checked, re-judged from recordings and re-scored by "
               "this pull request's code; no model runs in CI. Latency figures are from when outputs were recorded.",
               ""]
    failed = False
    criteria: list = []

    if head.production is None:
        summary.append("Nothing is in production yet; no release gate to apply.")
    else:
        config_path = head.production.config
        candidate = app_from_config(load_app(config_path), Path("."))
        approved = {h["fingerprint"] for h in base.history} | (
            {base.production.fingerprint} if base.production else set())
        is_rollback = head.production.via == "rollback" and head.production.fingerprint in approved
        stale = candidate.version.fingerprint != head.production.fingerprint
        outcome = gate_candidate(evaluator, candidate, head, mode="replay", context="ci", quiet=True,
                                 baseline_snapshot=base.production.app_version if base.production else None,
                                 candidate_config=str(config_path))
        criteria = release_criteria(evaluator, outcome)
        if is_rollback:   # a return to an approved version is not judged against the version it replaces
            criteria = [c for c in criteria if c.name in ("evaluation completed", "results and decision persisted")]
            criteria.append(Criterion("rollback to an approved version", True,
                                      f"{head.production.version} was approved before"))
        criteria.append(Criterion("release.yaml matches the approved version", not stale,
                                  f"`{config_path}` changed after promotion" if stale else "fingerprints match"))
        print(render_text(outcome.decision, outcome.candidate, outcome.baseline, outcome.questions,
                          next_step="See the release check below."))
        if stale:
            msg = (f"`{platform.release}` is out of date: `{config_path}` no longer matches the version that was "
                   f"approved. Gate it and run `evalops promote {config_path}`, then commit the result.")
            summary += [f"**FAIL: {msg}**", ""]
        if is_rollback:
            summary += [f"**Rollback** to {head.production.version}, which was approved before. "
                        f"The comparison below is for information only.", ""]
        summary.append(outcome.report)

    # Every committed app version must have recorded outputs; show how they compare.
    rows, missing = [], []
    for path in sorted(Path("configs/apps").glob("*.yaml")):
        config = load_app(path)
        try:
            handle = app_from_config(config, Path("."))
            data = evaluate(evaluator, handle, "replay", quiet=True)
        except CacheMiss:
            missing.append(str(path))
            continue
        cells = [fmt(name, compute(METRICS[name], [r for r in data.records if cat == "all" or cat in r.categories])[0])
                 for cat, name in LEADERBOARD]
        prod = " (production)" if head.production and handle.version.fingerprint == head.production.fingerprint else ""
        rows.append(f"| {config.version}{prod} | " + " | ".join(cells) + " |")
    summary += ["", "## All app versions (replayed from recorded outputs)", "",
                "| Version | Correctness | Declines unanswerable | Citation support | Unsupported claims | "
                "Failures | p95 latency (recorded) | Est. cost per success |", "|---|---|---|---|---|---|---|---|", *rows]
    if missing:
        summary += ["", "**FAIL: these app versions have no recorded outputs, so they were never evaluated:** "
                    + ", ".join(f"`{m}`" for m in missing) + ". Run `evalops run <config>` locally and commit `cache/`."]
    criteria.append(Criterion("every app version was evaluated", not missing,
                              ("never recorded: " + ", ".join(missing)) if missing else
                              f"{len(rows)} versions replayed"))

    failed = not all(c.passed for c in criteria)
    checklist = ["## Release check", "",
                 "Green means every line below holds; a passing test suite alone is not enough.", ""]
    checklist += [f"- {'PASS' if c.passed else '**FAIL**'}: {c.name} ({c.detail})" for c in criteria]
    print("\nRelease check")
    for c in criteria:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    summary = summary[:4] + checklist + [""] + summary[4:]
    text = "\n".join(summary) + "\n"
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    print("\nCI result:", "FAIL" if failed else "PASS")
    return EXIT_BLOCKED if failed else EXIT_OK


def _records(store: Store, run_id: str):
    from .metrics import CaseRecord
    store.init()
    if store.get_run(run_id) is None:
        raise ValueError(f"No run {run_id}")
    return [CaseRecord.model_validate(c["record"]) for c in store.get_cases(run_id).values() if c.get("record")]


def cmd_label(args) -> int:
    """Record human verdicts for a sample of answers, without showing the judge's verdict."""
    from .datasets import load as load_dataset
    from .judgecheck import VERDICTS, load_labels, sample_for_labelling

    store = Store(args.database)
    records = _records(store, args.run_id)
    dataset = load_dataset(store.get_run(args.run_id)["dataset_ref"])
    cases = {c.case_id: c for c in dataset.cases}
    out = Path(args.out)
    done = load_labels(out)
    todo = [r for r in sample_for_labelling(records, args.n, args.seed) if r.case_id not in done]
    print(f"{len(done)} already labelled, {len(todo)} to go. Keys: c = correct, p = partially correct, "
          f"i = incorrect, s = skip, q = quit.\n")
    keys = {"c": VERDICTS[0], "p": VERDICTS[1], "i": VERDICTS[2]}
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(todo, 1):
        case = cases[r.case_id]
        print(f"[{i}/{len(todo)}] {r.case_id}\n  Question:  {case.input.question}\n"
              f"  Reference: {case.expected.reference}\n  Answer:    {r.answer}")
        while (key := input("  Verdict [c/p/i/s/q]: ").strip().lower()) not in ("c", "p", "i", "s", "q"):
            pass
        if key == "q":
            break
        if key in keys:
            with out.open("a") as f:
                f.write(json.dumps({"case_id": r.case_id, "verdict": keys[key], "labeller": args.labeller}) + "\n")
        print()
    return EXIT_OK


def cmd_judge_agreement(args) -> int:
    from .judgecheck import agreement, f1_consistency, load_labels

    records = _records(Store(args.database), args.run_id)
    result = {"f1_consistency": f1_consistency(records)}
    labels = load_labels(Path(args.labels))
    if labels:
        result["human_agreement"] = agreement(records, labels)
    else:
        result["human_agreement"] = f"no labels in {args.labels}; create them with: evalops label {args.run_id}"
    print(json.dumps(result, indent=2))
    return EXIT_OK


def cmd_judge_probe(args) -> int:
    from .judgecheck import run_probes

    evaluator = make_evaluator(args)
    evaluator.store.init()
    result = run_probes(evaluator, args.run_id, n=args.n, seed=args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    out = evaluator.base_dir / evaluator.platform.runs_dir / args.run_id / "judge_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return EXIT_OK


def cmd_cache_prune(args) -> int:
    """Keep only cache entries that some current app version actually replays."""
    from .cache import prune

    import tempfile

    platform = load_platform(Path(args.config))
    scratch = tempfile.mkdtemp(prefix="evalops-prune-")  # replays are bookkeeping, not results
    evaluator = Evaluator(platform, Path("."), Store(f"sqlite:///{scratch}/prune.db"))
    evaluator.used_cache_keys = used = {"app": set(), "judge": set()}
    name = platform.dataset.split(":")[0]
    for path in sorted(Path("configs/apps").glob("*.yaml")):
        handle = app_from_config(load_app(path), Path("."))
        for split in args.splits.split(","):
            try:
                evaluator.run(handle, RunOptions(dataset=f"{name}:{split}", mode="replay", quiet=True))
                print(f"replayed {path} on {split}")
            except CacheMiss:
                print(f"skipped  {path} on {split} (not fully recorded)")
    for kind in ("app", "judge"):
        result = prune(platform.cache_dir, kind, used[kind], ("probe-",) if args.keep_probes else ())
        print(f"{kind}: kept {result['kept']}, removed {result['removed']}")
    return EXIT_OK


def cmd_loadtest(args) -> int:
    from datetime import datetime, timezone

    from . import loadtest
    from .datasets import load as load_dataset

    platform = load_platform(Path(args.config))
    dataset = load_dataset(args.dataset or platform.dataset)
    levels = [int(x) for x in args.concurrency.split(",")]
    results = loadtest.run(args.url, dataset, levels, args.requests)
    print(f"{'concurrency':>11} {'req/s':>7} {'p50':>7} {'p95':>7} {'errors':>6}")
    for r in results:
        print(f"{r['concurrency']:>11} {r['requests_per_second']:>7.2f} {r['latency_p50_s']:>6.2f}s "
              f"{r['latency_p95_s']:>6.2f}s {r['errors']:>6}")
    out = Path(platform.runs_dir).parent / "loadtests"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (out / f"loadtest-{stamp}.json").write_text(json.dumps({"url": args.url, "results": results}, indent=2) + "\n")
    return EXIT_OK


def cmd_recover(args) -> int:
    """Mark abandoned `running` runs as interrupted so they can be resumed, and say how."""
    store = Store(args.database)
    store.init()
    stale = store.stale_runs(args.stale_minutes)
    for run in stale:
        store.update_run(run["run_id"], status="interrupted", error_class="abandoned",
                         error=f"no activity for over {args.stale_minutes:g} minutes; the process stopped")
        print(f"interrupted {run['run_id']}  (resume: evalops run <config> --resume {run['run_id']})")
    if not stale:
        print("No abandoned runs.")
    return EXIT_OK


def cmd_runs(args) -> int:
    store = Store(args.database)
    store.init()
    for r in store.list_runs(limit=args.limit):
        source = (r.get("execution") or {}).get("latency_source", "")
        why = f"  [{r['error_class']}]" if r.get("error_class") else ""
        print(f"{r['run_id']:48s} {r['status']:11s} {r['dataset_ref']:22s} {r['mode']:8s} {source}{why}")
    return EXIT_OK


def cmd_serve(args) -> int:
    import uvicorn

    os.environ.setdefault("EVALOPS_CONFIG", args.config)
    uvicorn.run("evalops.api:app", host=args.host, port=args.port)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalops", description="Evaluate and gate LLM application releases.")
    parser.add_argument("--version", action="version", version=f"evalops {__version__}")
    parser.add_argument("--config", default="configs/evalops.yaml", help="platform config")
    parser.add_argument("--database", default=None, help="results database URL (default: $EVALOPS_DATABASE_URL or SQLite)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="validate every config and dataset").set_defaults(fn=cmd_validate)

    p = sub.add_parser("dataset", help="verify a dataset version and show its manifest")
    p.add_argument("ref", help="e.g. policies/v1")
    p.set_defaults(fn=cmd_dataset)

    def add_mode(p, default="auto"):
        p.add_argument("--mode", choices=["replay", "live", "rescore", "auto"], default=default,
                       help="replay: recorded outputs only, no model is called, fail if anything is missing; "
                            "live: the app answers every case now (latency measured now); "
                            "rescore: recorded answers, judged and scored with the current scoring config; "
                            "auto: recordings where they exist, live for the rest (default)")
        p.add_argument("--no-mlflow", action="store_true", help="skip MLflow even if MLFLOW_TRACKING_URI is set")

    p = sub.add_parser("run", help="evaluate one app version")
    p.add_argument("app", help="app config, e.g. configs/apps/docqa-v1.0.yaml")
    p.add_argument("--split", help="dataset split (default: the platform dataset's split)")
    p.add_argument("--dataset", help="full dataset reference, e.g. policies/v1:dev")
    p.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted or failed run")
    p.add_argument("--limit", type=int, help="only the first N cases (for quick checks)")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--judge-concurrency", type=int)
    p.add_argument("--no-record", action="store_true", help="do not add new outputs to the recordings in cache/")
    p.add_argument("--scoring", help="score with another scoring config (use with --mode rescore)")
    p.add_argument("--max-recording-age-days", type=float,
                   help="auto mode: treat recordings older than this as missing and run those cases live")
    add_mode(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("gate", help="evaluate a candidate against production and record a decision "
                                    "(never changes production)")
    p.add_argument("app")
    add_mode(p)
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("promote", help="make a version production, using its passing gate decision "
                                       "(evaluates nothing)")
    p.add_argument("app")
    p.add_argument("--decision", help="a specific gate decision id (default: the latest one for this version)")
    p.add_argument("--accept-warnings", action="store_true",
                   help="accept a pass_with_warnings decision after reviewing the advisory failures")
    p.add_argument("--no-mlflow", action="store_true")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("rollback", help="restore the previous production version from the release history")
    p.add_argument("--to", help="a specific earlier version (default: the one before the current)")
    p.add_argument("--reason", required=True, help="why production is being rolled back (recorded in the audit log)")
    p.add_argument("--no-mlflow", action="store_true")
    p.set_defaults(fn=cmd_rollback)

    p = sub.add_parser("compare", help="two runs side by side (e.g. one app under two scoring configs)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("sync-registry", help="point MLflow's production alias at the version in release.yaml")
    p.set_defaults(fn=cmd_sync_registry)

    p = sub.add_parser("audit", help="show gate decisions, promotions, refusals and rollbacks")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("ci", help="pull-request check: replay, gate production against the base branch")
    p.add_argument("--base-ref", default="origin/main")
    p.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"), help="append a Markdown summary here")
    p.add_argument("--no-mlflow", action="store_true")
    p.set_defaults(fn=cmd_ci)

    p = sub.add_parser("recover", help="mark runs abandoned by a crashed process as interrupted")
    p.add_argument("--stale-minutes", type=float, default=30)
    p.set_defaults(fn=cmd_recover)

    p = sub.add_parser("runs", help="list recent runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_runs)

    p = sub.add_parser("label", help="label a sample of answers by hand, to measure the judge")
    p.add_argument("run_id")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="labels/judge_labels.jsonl")
    p.add_argument("--labeller", default="human")
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("judge-agreement", help="how often the judge agrees with human labels and with token F1")
    p.add_argument("run_id")
    p.add_argument("--labels", default="labels/judge_labels.jsonl")
    p.set_defaults(fn=cmd_judge_agreement)

    p = sub.add_parser("judge-probe", help="measure the judge on answers that are wrong by construction")
    p.add_argument("run_id")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_judge_probe)

    p = sub.add_parser("cache-prune", help="drop recorded outputs no current app version uses")
    p.add_argument("--splits", default="test,dev")
    p.add_argument("--keep-probes", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(fn=cmd_cache_prune)

    p = sub.add_parser("loadtest", help="throughput and latency of a running DocQA service under concurrency")
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--requests", type=int, default=40, help="requests per concurrency level")
    p.add_argument("--dataset", help="dataset reference for the questions (default: the platform dataset)")
    p.set_defaults(fn=cmd_loadtest)

    p = sub.add_parser("serve", help="run the EvalOps API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except EXPECTED_ERRORS as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
