"""Failure injection: the evaluator must record failures, recover, and never
report a failed evaluation as successful. One test per row of the failure table
in the README."""

from __future__ import annotations

import json

import pytest

from evalops.cache import CacheMiss
from evalops.datasets import DatasetError
from evalops.engine import RunOptions
from evalops.gate import evaluate_gate
from evalops.release import run_data
from evalops.store import StoreUnavailable
from tests.conftest import N_ANSWERABLE, N_UNANSWERABLE
from tests.fakes import Interrupt, ScriptedAdapter

N = N_ANSWERABLE + N_UNANSWERABLE


def run(project, name, **options):
    return project.evaluator().run(project.handle(name), RunOptions(quiet=True, **options))


def test_malformed_json_is_recorded_and_gated(project):
    project.write_app("baseline")
    project.write_app("bad", {"default": "good", "cases": {"c01": "malformed", "c02": "malformed"}})
    base, bad = run(project, "baseline"), run(project, "bad")
    recs = {r.case_id: r for r in bad.records}
    assert recs["c01"].status == "malformed" and recs["c01"].schema_valid is False
    assert bad.metrics["all"]["app_malformed_rate"]["value"] == pytest.approx(2 / N)
    ev = project.evaluator()
    decision = evaluate_gate(ev.gate, run_data(ev, bad.run_id), run_data(ev, base.run_id))
    assert decision.decision == "fail"
    assert "failures" in {r.rule_id for r in decision.mandatory_failures}


def test_timeouts_are_retried_then_recorded(project):
    project.write_app("slow", {"default": "good", "cases": {"c03": "timeout"}})
    result = run(project, "slow")
    rec = next(r for r in result.records if r.case_id == "c03")
    assert rec.status == "timeout"
    assert rec.attempts == 2  # max_retries=1 in the toy platform config
    assert result.metrics["all"]["app_timeout_rate"]["value"] == pytest.approx(1 / N)
    # Timeouts count as wrong answers, not as missing data.
    assert result.metrics["answerable"]["answer_correctness"]["value"] == pytest.approx(15 / 16)


def test_adapter_crash_is_a_failed_request_not_a_crashed_run(project):
    project.write_app("buggy", {"default": "good", "cases": {"c04": "crash"}})
    result = run(project, "buggy")
    assert result.status == "completed"
    rec = next(r for r in result.records if r.case_id == "c04")
    assert rec.status == "error" and "adapter raised RuntimeError" in rec.error


def test_dataset_change_is_rejected_before_the_run_starts(project):
    project.write_app("good")
    path = project.root / "datasets/toy/v1/test.jsonl"
    path.write_text(path.read_text().replace("Question number 1?", "Question number one?"))
    ev = project.evaluator()
    with pytest.raises(DatasetError, match="does not match its manifest hash"):
        ev.run(project.handle("good"), RunOptions(quiet=True))
    ev.store.init()
    assert ev.store.list_runs() == []


def test_malformed_judge_output_is_retried_then_marked(project):
    project.write_app("judgy", {"default": "good", "cases": {"c05": "judge_flaky", "c06": "judge_garbage"}})
    result = run(project, "judgy")
    recs = {r.case_id: r for r in result.records}
    assert recs["c05"].judge_correctness_status == "ok"      # garbage once, fine on the retry
    assert recs["c06"].judge_correctness_status == "unevaluable"   # garbage every time
    assert "malformed judge output" in recs["c06"].judge_error
    # Excluded from correctness rather than counted as right, and counted as a judge error.
    assert result.metrics["answerable"]["answer_correctness"]["n"] == N_ANSWERABLE - 1
    assert result.metrics["all"]["judge_unevaluable_rate"]["value"] > 0


def test_quality_below_mandatory_threshold_blocks(project):
    project.write_app("baseline")
    project.write_app("worse", {"default": "good", **{"cases": {f"c{i:02d}": "wrong" for i in range(1, 6)}}})
    base, worse = run(project, "baseline"), run(project, "worse")
    ev = project.evaluator()
    decision = evaluate_gate(ev.gate, run_data(ev, worse.run_id), run_data(ev, base.run_id))
    assert decision.decision == "fail"
    failed = {r.rule_id: r for r in decision.mandatory_failures}
    assert "correctness" in failed
    assert failed["correctness"].delta == pytest.approx(-5 / 16)


def test_better_quality_but_over_latency_budget_blocks(project):
    project.write_app("baseline", {"default": "good", "cases": {"c01": "wrong"}})
    project.write_app("slow-but-better", latency={"default": 4.0})
    base, cand = run(project, "baseline"), run(project, "slow-but-better")
    ev = project.evaluator()
    decision = evaluate_gate(ev.gate, run_data(ev, cand.run_id), run_data(ev, base.run_id))
    correctness = next(r for r in decision.rules if r.rule_id == "correctness")
    assert correctness.passed and correctness.delta > 0
    assert decision.decision == "fail"
    assert [r.rule_id for r in decision.mandatory_failures] == ["latency"]


def test_database_unavailable_fails_before_anything_runs(project):
    project.write_app("good")
    ev = project.evaluator(database="postgresql+psycopg://nobody:x@127.0.0.1:1/none")
    with pytest.raises(StoreUnavailable):
        ev.run(project.handle("good"), RunOptions(quiet=True))
    assert ScriptedAdapter.calls == 0


def test_database_failure_mid_run_never_reports_success(project, monkeypatch):
    project.write_app("good")
    ev = project.evaluator()
    real = ev.store.upsert_case
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 5:
            raise StoreUnavailable("database went away")
        return real(*args, **kwargs)

    monkeypatch.setattr(ev.store, "upsert_case", flaky)
    with pytest.raises(StoreUnavailable):
        ev.run(project.handle("good"), RunOptions(quiet=True))
    run_row = ev.store.list_runs()[0]
    assert run_row["status"] == "failed"
    status = json.loads((project.root / ".evalops/runs" / run_row["run_id"] / "status.json").read_text())
    assert status["status"] == "failed"


def test_cache_entry_from_another_model_version_is_rejected(project):
    project.write_app("good")
    run(project, "good")                                  # records outputs for digest bbbb...
    # Same app, but the model weights changed: a different version.
    project.write_app("good", digest="cccccccccccc")
    with pytest.raises(CacheMiss):
        run(project, "good", mode="replay")

    # Defence in depth: an entry whose recorded identity was tampered with is rejected too.
    project.write_app("good")
    cache_file = next((project.root / "cache/app").glob("*.jsonl"))
    lines = [json.loads(line) for line in cache_file.read_text().splitlines()]
    lines[0]["identity"]["model"]["digest"] = "dddddddddddd"
    cache_file.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(CacheMiss):
        run(project, "good", mode="replay")


def test_interrupted_run_is_never_completed_and_resumes_without_duplicates(project):
    project.write_app("good")
    ev = project.evaluator()
    ScriptedAdapter.interrupt_after = 7
    with pytest.raises(KeyboardInterrupt):
        ev.run(project.handle("good"), RunOptions(quiet=True, mode="live", record=False))
    run_row = ev.store.list_runs()[0]
    assert run_row["status"] == "interrupted"
    assert ev.store.count_cases(run_row["run_id"]) == 7

    from evalops.gate import RunData
    data = run_data(ev, run_row["run_id"])
    assert evaluate_gate(ev.gate, data, None).decision == "fail"   # an unfinished run can never pass

    ScriptedAdapter.interrupt_after = None
    resumed = ev.run(project.handle("good"), RunOptions(quiet=True, mode="live", record=False, resume=run_row["run_id"]))
    assert resumed.status == "completed"
    assert ScriptedAdapter.calls == N                       # the 7 finished cases were not redone
    assert ev.store.count_cases(run_row["run_id"]) == N     # and nothing was duplicated
    clean = run(project, "good")
    assert resumed.metrics["answerable"] == clean.metrics["answerable"] | {
        "replayed_share": resumed.metrics["answerable"]["replayed_share"]}


def test_resume_refuses_a_run_with_different_inputs(project):
    project.write_app("good")
    ev = project.evaluator()
    ScriptedAdapter.interrupt_after = 3
    with pytest.raises(KeyboardInterrupt):
        ev.run(project.handle("good"), RunOptions(quiet=True))
    run_id = ev.store.list_runs()[0]["run_id"]
    project.write_app("good", prompt_version="p2")
    from evalops.engine import RunError
    with pytest.raises(RunError, match="different inputs"):
        ev.run(project.handle("good"), RunOptions(quiet=True, resume=run_id))


def test_fault_injection_is_deterministic_and_trips_the_reliability_gate(project):
    project.write_app("good")
    project.write_app("chaos", faults={"timeout_rate": 0.2, "malformed_rate": 0.1, "seed": 3})
    base = run(project, "good")
    a = run(project, "chaos")
    b = run(project, "chaos", mode="replay")
    assert [r.injected_fault for r in a.records] == [r.injected_fault for r in b.records]
    assert any(r.injected_fault == "timeout" for r in a.records)
    malformed = [r for r in a.records if r.injected_fault == "malformed"]
    assert all(r.status == "malformed" for r in malformed)   # caught by the parser, not by a flag
    ev = project.evaluator()
    decision = evaluate_gate(ev.gate, run_data(ev, a.run_id), run_data(ev, base.run_id))
    assert "failures" in {r.rule_id for r in decision.mandatory_failures}


def test_injected_slowdown_keeps_quality_and_fails_the_latency_rule(project):
    project.write_app("good")
    project.write_app("slow", faults={"extra_latency_s": 3.0, "seed": 1})
    base, slow = run(project, "good"), run(project, "slow")
    assert slow.metrics["all"]["latency_p95_s"]["value"] == pytest.approx(4.0)
    ev = project.evaluator()
    decision = evaluate_gate(ev.gate, run_data(ev, slow.run_id), run_data(ev, base.run_id))
    assert [r.rule_id for r in decision.mandatory_failures] == ["latency"]


def test_simulated_interruption_then_resume(project):
    project.write_app("abort", faults={"abort_after": 5, "seed": 1})
    ev = project.evaluator()
    with pytest.raises(KeyboardInterrupt):
        ev.run(project.handle("abort"), RunOptions(quiet=True))
    row = ev.store.list_runs()[0]
    assert row["status"] == "interrupted" and ev.store.count_cases(row["run_id"]) == 5
    resumed = ev.run(project.handle("abort"), RunOptions(quiet=True, resume=row["run_id"]))
    assert resumed.status == "completed" and ev.store.count_cases(row["run_id"]) == N
