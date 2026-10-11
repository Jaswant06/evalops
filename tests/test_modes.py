"""Run modes: what is replayed, what runs live, and what the run says about it."""

from __future__ import annotations

import pytest
import yaml

from evalops import cli
from evalops.cache import CacheMiss
from evalops.engine import RunOptions
from evalops.report import execution_lines
from tests.fakes import ScriptedAdapter


def run(project, name, **options):
    return project.evaluator().run(project.handle(name), RunOptions(quiet=True, **options))


def test_live_then_replay_reports_where_numbers_came_from(project):
    project.write_app("v1")
    live = run(project, "v1", mode="live")
    assert live.execution["app"] == {"live": 20, "replayed": 0, "recorded_from": None, "recorded_to": None}
    assert live.execution["latency_source"] == "measured in this run"
    assert ScriptedAdapter.calls == 20

    replay = run(project, "v1", mode="replay")
    assert ScriptedAdapter.calls == 20                       # nothing called the app
    assert replay.execution["app"]["replayed"] == 20 and replay.execution["judge"]["live"] == 0
    assert replay.execution["latency_source"] == "recorded"
    assert replay.metrics == {**live.metrics, "all": {**live.metrics["all"], "replayed_share": {"value": 1.0, "n": 20}},
                              "answerable": {**live.metrics["answerable"],
                                             "replayed_share": {"value": 1.0, "n": 16}},
                              "unanswerable": {**live.metrics["unanswerable"],
                                               "replayed_share": {"value": 1.0, "n": 4}},
                              "citation_required": {**live.metrics["citation_required"],
                                                    "replayed_share": {"value": 1.0, "n": 16}}}
    text = "\n".join(execution_lines(replay.execution))
    assert "Mode: REPLAY" in text and "not in this run" in text


def test_live_mode_measures_again_even_when_recorded(project):
    project.write_app("v1")
    run(project, "v1", mode="live")
    run(project, "v1", mode="live")
    assert ScriptedAdapter.calls == 40


def test_auto_mode_mixes_and_says_so(project):
    project.write_app("v1")
    run(project, "v1", mode="live", limit=5)       # record 5 cases only
    mixed = run(project, "v1", mode="auto")
    assert mixed.execution["app"]["replayed"] == 5 and mixed.execution["app"]["live"] == 15
    assert mixed.execution["latency_source"] == "mixed"


def test_no_record_leaves_the_recordings_alone(project):
    project.write_app("v1")
    run(project, "v1", mode="live", record=False)
    with pytest.raises(CacheMiss):
        run(project, "v1", mode="replay")


def test_rescore_scores_recorded_answers_with_another_scoring_config(project, monkeypatch, capsys):
    project.write_app("v1", {"default": "good", "cases": {"c01": "wrong"}})
    first = run(project, "v1", mode="live")
    # A second scoring config: same judge model, a different correctness prompt.
    (project.root / "prompts/judge/correctness_v2.txt").write_text("Grade the answer, version 2.")
    scoring = yaml.safe_load((project.root / "configs/scoring.yaml").read_text())
    scoring["judge"]["prompts"]["correctness"] = "prompts/judge/correctness_v2.txt"
    (project.root / "configs/scoring-v2.yaml").write_text(yaml.safe_dump(scoring))

    monkeypatch.chdir(project.root)
    from tests.conftest import FakeJudgeEvaluator
    from evalops.config import load_platform
    from evalops.store import Store

    def make(args):
        platform = load_platform(project.root / "configs/evalops.yaml")
        if getattr(args, "scoring", None):
            platform = platform.model_copy(update={"scoring": args.scoring})
        return FakeJudgeEvaluator(platform, project.root, Store(f"sqlite:///{project.root}/evalops.db"))

    monkeypatch.setattr(cli, "make_evaluator", make)
    calls = ScriptedAdapter.calls
    assert cli.main(["run", "configs/apps/v1.yaml", "--mode", "rescore", "--scoring", "configs/scoring-v2.yaml"]) == 0
    assert ScriptedAdapter.calls == calls                    # recorded answers, no app calls
    out = capsys.readouterr().out
    assert "RESCORE" in out
    store = Store(f"sqlite:///{project.root}/evalops.db")
    second = store.list_runs()[0]
    assert second["mode"] == "rescore"
    assert second["scoring_fingerprint"] != store.get_run(first.run_id)["scoring_fingerprint"]
    assert second["app_fingerprint"] == store.get_run(first.run_id)["app_fingerprint"]
    assert cli.main(["compare", first.run_id, second["run_id"]]) == 0
    assert "Comparing two SCORING configurations" in capsys.readouterr().out


def test_a_release_snapshot_can_only_be_replayed(project):
    from evalops.engine import RunError, app_from_snapshot
    project.write_app("v1")
    snapshot = project.handle("v1").version.model_dump()
    with pytest.raises(RunError, match="has no app to call"):
        project.evaluator().run(app_from_snapshot(snapshot), RunOptions(mode="live", quiet=True))


# --- failure classes, latency fields, structured judgments --------------------------------------------

def test_application_and_scoring_failures_are_classified_separately(project):
    project.write_app("v1", {"default": "good", "cases": {"c01": "timeout", "c02": "judge_garbage"}})
    result = run(project, "v1", mode="live")
    recs = {r.case_id: r for r in result.records}
    assert recs["c01"].failure_class == "application"
    assert recs["c02"].failure_class == "scoring" and recs["c02"].judge_correctness_status == "unevaluable"
    assert recs["c03"].failure_class is None
    assert result.execution["failures"] == {"application": 1, "scoring": 1}
    # an unevaluable answer is never silently scored
    assert recs["c02"].correctness is None
    assert result.metrics["all"]["app_failure_rate"]["value"] == pytest.approx(1 / 20)


def test_judgments_are_stored_in_one_validated_structure(project):
    project.write_app("v1", {"default": "good", "cases": {"c02": "judge_garbage"}})
    recs = {r.case_id: r for r in run(project, "v1", mode="live").records}
    good = {j["evaluation"]: j for j in recs["c01"].judgments}
    assert set(good) == {"correctness", "claims"}
    assert good["correctness"] | {"justification": None} == {
        "evaluation": "correctness", "status": "ok", "attempts": 1, "score": 1.0, "label": "correct",
        "justification": None, "error": None}
    bad = {j["evaluation"]: j for j in recs["c02"].judgments}["correctness"]
    assert bad["status"] == "unevaluable" and bad["score"] is None and bad["attempts"] == 2


def test_replayed_cases_keep_original_latency_and_report_their_own_duration(project):
    project.write_app("v1", latency={"default": 2.0})
    run(project, "v1", mode="live")
    replay = run(project, "v1", mode="replay")
    r = replay.records[0]
    assert r.execution_mode == "replayed" and r.original_latency_s == 2.0
    assert r.evaluation_duration_s is not None and r.evaluation_duration_s < 0.5
    assert replay.metrics["all"]["latency_p95_s"]["value"] == pytest.approx(2.0)
    assert replay.execution["evaluation_duration_s"] < 5


def test_auto_records_its_policy_and_honours_a_maximum_recording_age(project):
    project.write_app("v1")
    run(project, "v1", mode="live")
    fresh = run(project, "v1", mode="auto", max_recording_age_days=7)
    assert fresh.execution["app"]["replayed"] == 20
    assert "never replaced by a recording" in fresh.execution["policy"]
    assert "at most 7 days old" in fresh.execution["policy"]
    calls = ScriptedAdapter.calls
    stale = run(project, "v1", mode="auto", max_recording_age_days=0)   # every recording counts as too old
    assert stale.execution["app"]["live"] == 20 and ScriptedAdapter.calls == calls + 20


def test_failed_runs_record_why(project):
    from evalops.engine import RunError
    project.write_app("v1")
    ev = project.evaluator()
    with pytest.raises(CacheMiss):
        ev.run(project.handle("v1"), RunOptions(mode="replay", quiet=True))
    assert ev.store.list_runs()[0]["error_class"] == "missing_recording"

    project.monkeypatch.setattr("evalops.engine.installed_digest", lambda name: None)   # judge model gone
    with pytest.raises(RunError, match="Judge"):
        ev.run(project.handle("v1"), RunOptions(mode="live", quiet=True))
    assert ev.store.list_runs()[0]["error_class"] == "infrastructure"
