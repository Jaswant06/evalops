"""Extended reliability scenarios: real processes, real signals, a real database outage.

Slower than the unit tests, so they run on a schedule (`.github/workflows/reliability.yml`)
rather than on every pull request:

    pytest -m reliability

The PostgreSQL scenarios need EVALOPS_TEST_DATABASE_URL; the outage scenario also needs
EVALOPS_RELIABILITY_PG_CONTAINER, the id or name of the PostgreSQL container it may stop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from evalops.config import load_release
from evalops.engine import RunOptions
from evalops.store import Store, StoreUnavailable
from tests.conftest import N_ANSWERABLE, N_UNANSWERABLE

pytestmark = pytest.mark.reliability
N = N_ANSWERABLE + N_UNANSWERABLE
REPO = Path(__file__).resolve().parents[2]
PG_URL = os.environ.get("EVALOPS_TEST_DATABASE_URL")
PG_CONTAINER = os.environ.get("EVALOPS_RELIABILITY_PG_CONTAINER")


def cli(project, *args, db: str | None = None, wait: bool = True):
    """The real CLI in its own process (the scripted app is importable through PYTHONPATH)."""
    env = {**os.environ, "PYTHONPATH": str(REPO), "EVALOPS_DATABASE_URL": db or f"sqlite:///{project.root}/evalops.db"}
    env.pop("MLFLOW_TRACKING_URI", None)
    env.pop("EVALOPS_PUSHGATEWAY_URL", None)
    proc = subprocess.Popen([sys.executable, "-m", "evalops.cli", *args], cwd=project.root, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return proc
    out, _ = proc.communicate(timeout=300)
    return proc.returncode, out


def gate_in_process(project, name, db=None):
    """Gate with the scripted judge (no model needed); the decision lands in the shared database."""
    from evalops.config import load_release as _load
    from evalops.release import gate_candidate
    ev = project.evaluator(database=db)
    outcome = gate_candidate(ev, project.handle(name), _load(project.root / "configs/release.yaml"),
                             mode="auto", quiet=True, candidate_config=f"configs/apps/{name}.yaml")
    return outcome.decision


@pytest.mark.parametrize("round_", range(5))
def test_concurrent_promotions_from_separate_processes(project, round_):
    """Two promote processes start at the same moment: exactly one may change production."""
    for name in ("v1", "v2", "v3"):
        project.write_app(name, prompt_version=f"{name}-{round_}")
    assert gate_in_process(project, "v1").decision == "pass"
    assert cli(project, "promote", "configs/apps/v1.yaml")[0] == 0
    assert gate_in_process(project, "v2").decision == "pass"
    assert gate_in_process(project, "v3").decision == "pass"     # both decided against v1

    procs = [cli(project, "promote", f"configs/apps/{n}.yaml", wait=False) for n in ("v2", "v3")]
    results = [(p.wait(timeout=120), p.stdout.read()) for p in procs]
    codes = sorted(code for code, _ in results)
    assert codes[0] == 0 and codes[1] in (1, 2), results            # one winner; the other refused or locked out
    winner = "v2" if results[0][0] == 0 else "v3"
    release = load_release(project.root / "configs/release.yaml")
    assert release.production.version == winner
    assert [h["version"] for h in release.history] == ["v1"]
    events = [e["event"] for e in Store(f"sqlite:///{project.root}/evalops.db").list_audit()]
    assert events.count("promoted") == 2                             # v1, then exactly one of v2/v3


def test_real_interrupt_then_resume(project):
    """SIGINT a running evaluation: it is marked interrupted, and resuming finishes it without duplicates."""
    project.write_app("slow", sleep_s=0.15)
    project.evaluator().run(project.handle("slow"), RunOptions(quiet=True))   # records answers and verdicts
    proc = cli(project, "run", "configs/apps/slow.yaml", "--mode", "live", wait=False)
    time.sleep(1.2)
    proc.send_signal(signal.SIGINT)
    out = proc.communicate(timeout=60)[0]
    assert proc.returncode == 2, out
    store = Store(f"sqlite:///{project.root}/evalops.db")
    run = store.list_runs()[0]
    done = store.count_cases(run["run_id"])
    assert run["status"] == "interrupted" and 0 < done < N, (run["status"], done)

    code, out = cli(project, "run", "configs/apps/slow.yaml", "--mode", "live", "--resume", run["run_id"])
    assert code == 0, out
    assert store.get_run(run["run_id"])["status"] == "completed"
    assert store.count_cases(run["run_id"]) == N


def test_extended_timeouts_and_retries(project):
    """Transient timeouts recover within the retry budget; persistent ones are recorded, once, as failures."""
    behaviour = {"default": "good", "cases": {"c01": "flaky_timeout_1", "c02": "flaky_timeout_2", "c03": "timeout"}}
    project.write_app("flaky", behaviour)
    result = project.evaluator().run(project.handle("flaky"), RunOptions(mode="live", quiet=True))
    recs = {r.case_id: r for r in result.records}
    assert (recs["c01"].status, recs["c01"].attempts) == ("ok", 2)        # recovered on the retry
    assert (recs["c02"].status, recs["c02"].attempts) == ("timeout", 2)   # retry budget (1) exhausted
    assert recs["c03"].status == "timeout" and recs["c03"].failure_class == "application"
    assert project.evaluator().store.count_cases(result.run_id) == N
    assert result.metrics["all"]["app_timeout_rate"]["value"] == pytest.approx(2 / N)

    config = project.root / "configs/evalops.yaml"
    data = yaml.safe_load(config.read_text())
    data["execution"]["max_retries"] = 3
    config.write_text(yaml.safe_dump(data))
    result = project.evaluator().run(project.handle("flaky"), RunOptions(mode="live", quiet=True))
    recs = {r.case_id: r for r in result.records}
    assert (recs["c02"].status, recs["c02"].attempts) == ("ok", 3)        # a bigger budget recovers it
    assert recs["c03"].attempts == 4 and recs["c03"].status == "timeout"


@pytest.mark.skipif(not PG_URL, reason="needs EVALOPS_TEST_DATABASE_URL")
def test_stress_concurrent_evaluations_on_postgres(project):
    """Eight evaluation processes at once against one PostgreSQL: every run completes, nothing is duplicated."""
    names = [f"s{i}" for i in range(4)]
    for name in names:
        project.write_app(name, prompt_version=name)
        project.evaluator(database=PG_URL).run(project.handle(name), RunOptions(quiet=True))   # record outputs
    store = Store(PG_URL)
    before = {r["run_id"] for r in store.list_runs(limit=1000)}
    procs = [cli(project, "run", f"configs/apps/{n}.yaml", "--mode", "replay", db=PG_URL, wait=False)
             for n in names for _ in range(2)]
    outputs = [(p.wait(timeout=300), p.stdout.read()) for p in procs]
    assert all(code == 0 for code, _ in outputs), [o for c, o in outputs if c]
    new = [r for r in store.list_runs(limit=1000) if r["run_id"] not in before]
    assert len(new) == 8 and all(r["status"] == "completed" for r in new)
    assert all(store.count_cases(r["run_id"]) == N for r in new)
    by_version: dict[str, set] = {}
    for r in new:     # the same version gives the same numbers, however many run at once
        m = store.get_metrics(r["run_id"])["all"]
        by_version.setdefault(r["app_version_label"], set()).add(
            (m["answer_correctness"]["value"], m["app_failure_rate"]["value"]))
    assert all(len(v) == 1 for v in by_version.values())


@pytest.mark.skipif(not (PG_URL and PG_CONTAINER), reason="needs EVALOPS_TEST_DATABASE_URL and a container to stop")
def test_database_outage_mid_run_then_recovery(project):
    """PostgreSQL really stops mid-run: the run fails without reporting success, and is recovered afterwards."""
    project.write_app("v1")
    project.evaluator().run(project.handle("v1"), RunOptions(quiet=True))   # record outputs (SQLite)
    ev = project.evaluator(database=PG_URL)
    real = ev.store.upsert_case
    calls = {"n": 0}

    def outage_after_eight(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 9:
            subprocess.run(["docker", "stop", PG_CONTAINER], check=True, capture_output=True, timeout=60)
        return real(*args, **kwargs)

    ev.store.upsert_case = outage_after_eight
    try:
        with pytest.raises(StoreUnavailable):
            ev.run(project.handle("v1"), RunOptions(mode="replay", quiet=True))
    finally:
        subprocess.run(["docker", "start", PG_CONTAINER], check=True, capture_output=True, timeout=60)
    status = list((project.root / ".evalops/runs").glob("*/status.json"))
    assert any('"failed"' in p.read_text() for p in status)

    store = Store(PG_URL)
    for _ in range(60):                  # wait for the database to come back
        try:
            store.init()
            break
        except StoreUnavailable:
            time.sleep(1)
    run = store.list_runs()[0]
    assert run["status"] == "running"    # the failure could not be written while the database was down
    code, out = cli(project, "recover", "--stale-minutes", "0", db=PG_URL)
    assert code == 0 and "interrupted" in out
    assert store.get_run(run["run_id"])["status"] == "interrupted"
    resumed = project.evaluator(database=PG_URL).run(
        project.handle("v1"), RunOptions(mode="replay", quiet=True, resume=run["run_id"]))
    assert resumed.status == "completed" and store.count_cases(run["run_id"]) == N
