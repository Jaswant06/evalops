"""Gate, promote, rollback and the CI check, end to end through the CLI (scripted app and judge).

The safeguards under test:
  - gate records a decision and never changes production
  - promote evaluates nothing and needs a valid passing decision for the exact versions on disk
  - a failed evaluation never changes production, and every refusal is audited
  - promotion is idempotent and locked; the previous version stays available for rollback
"""

from __future__ import annotations

import subprocess

import pytest
import yaml

from evalops import cli
from evalops.config import load_release
from evalops.engine import RunOptions
from evalops.release import release_lock
from evalops.store import Store
from tests.fakes import ScriptedAdapter


@pytest.fixture
def repo(project, monkeypatch):
    monkeypatch.chdir(project.root)
    evaluator = project.evaluator()
    monkeypatch.setattr(cli, "make_evaluator", lambda args: evaluator)
    project.ev = evaluator
    return project


def store(repo) -> Store:
    return Store(f"sqlite:///{repo.root}/evalops.db")


def release_text(repo) -> str | None:
    path = repo.root / "configs/release.yaml"
    return path.read_text() if path.exists() else None


def events(repo) -> list[str]:
    return [e["event"] for e in reversed(store(repo).list_audit())]


def gate_and_promote(repo, name) -> None:
    assert cli.main(["gate", f"configs/apps/{name}.yaml"]) == cli.EXIT_OK
    assert cli.main(["promote", f"configs/apps/{name}.yaml"]) == cli.EXIT_OK


def test_gate_records_a_decision_and_never_changes_production(repo, capsys):
    repo.write_app("v1")
    assert cli.main(["gate", "configs/apps/v1.yaml"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Release decision: PASS" in out and "Promotion: not attempted" in out
    assert release_text(repo) is None
    assert events(repo) == ["gate_decision"]


def test_promote_without_a_gate_decision_is_refused(repo, capsys):
    repo.write_app("v1")
    assert cli.main(["promote", "configs/apps/v1.yaml"]) == cli.EXIT_BLOCKED
    assert "no gate decision exists" in capsys.readouterr().out
    assert release_text(repo) is None
    assert events(repo) == ["promotion_refused"]
    assert ScriptedAdapter.calls == 0   # promote never evaluates


def test_promote_after_a_passing_gate_updates_release_and_audit_trail(repo):
    repo.write_app("v1")
    gate_and_promote(repo, "v1")
    prod = load_release(repo.root / "configs/release.yaml").production
    assert prod.version == "v1" and prod.fingerprint == repo.handle("v1").version.fingerprint
    assert store(repo).get_decision(prod.decision_id)["decision"] == "pass"
    assert [r["version"] for r in store(repo).list_releases()] == ["v1"]
    assert events(repo) == ["gate_decision", "promoted"]


def test_a_failed_evaluation_never_changes_production(repo, capsys):
    repo.write_app("v1")
    repo.write_app("degraded", {"default": "wrong"})
    gate_and_promote(repo, "v1")
    before = release_text(repo)
    assert cli.main(["gate", "configs/apps/degraded.yaml"]) == cli.EXIT_BLOCKED
    assert "Release decision: BLOCKED" in capsys.readouterr().out
    assert cli.main(["promote", "configs/apps/degraded.yaml"]) == cli.EXIT_BLOCKED
    assert "is BLOCKED; a failed evaluation never changes production" in capsys.readouterr().out
    assert release_text(repo) == before
    assert events(repo)[-2:] == ["gate_decision", "promotion_refused"]


def test_promote_refuses_when_the_version_changed_after_gating(repo, capsys):
    repo.write_app("v2")
    assert cli.main(["gate", "configs/apps/v2.yaml"]) == cli.EXIT_OK
    decision = store(repo).list_decisions()[0]["decision_id"]
    repo.write_app("v2", prompt_version="edited-after-gating")
    assert cli.main(["promote", "configs/apps/v2.yaml", "--decision", decision]) == cli.EXIT_BLOCKED
    assert "the app version changed after it was gated" in capsys.readouterr().out
    assert release_text(repo) is None


def test_promote_refuses_when_the_gate_rules_changed_after_gating(repo, capsys):
    repo.write_app("v1")
    assert cli.main(["gate", "configs/apps/v1.yaml"]) == cli.EXIT_OK
    gate_file = repo.root / "configs/gate.yaml"
    rules = yaml.safe_load(gate_file.read_text())
    rules["rules"][0]["min"] = 0.5          # loosened after the decision
    gate_file.write_text(yaml.safe_dump(rules))
    repo.monkeypatch.setattr(cli, "make_evaluator", lambda args: repo.evaluator())
    assert cli.main(["promote", "configs/apps/v1.yaml"]) == cli.EXIT_BLOCKED
    assert "the gate rules changed after the decision" in capsys.readouterr().out


def test_promote_refuses_a_decision_made_against_an_older_production(repo, capsys):
    for name in ("v1", "v2", "v3"):
        repo.write_app(name, prompt_version=name)
    gate_and_promote(repo, "v1")
    assert cli.main(["gate", "configs/apps/v2.yaml"]) == cli.EXIT_OK
    assert cli.main(["gate", "configs/apps/v3.yaml"]) == cli.EXIT_OK     # also decided against v1
    assert cli.main(["promote", "configs/apps/v2.yaml"]) == cli.EXIT_OK
    assert cli.main(["promote", "configs/apps/v3.yaml"]) == cli.EXIT_BLOCKED
    assert "production changed after the decision was made" in capsys.readouterr().out
    assert load_release(repo.root / "configs/release.yaml").production.version == "v2"


def test_promoting_twice_is_a_no_op(repo, capsys):
    repo.write_app("v1")
    gate_and_promote(repo, "v1")
    before = release_text(repo)
    assert cli.main(["promote", "configs/apps/v1.yaml"]) == cli.EXIT_OK
    assert "already in production; nothing changed" in capsys.readouterr().out
    assert release_text(repo) == before
    assert events(repo)[-1] == "promotion_noop"


def test_concurrent_promotions_are_locked(repo):
    repo.write_app("v1")
    assert cli.main(["gate", "configs/apps/v1.yaml"]) == cli.EXIT_OK
    with release_lock(repo.ev):
        assert cli.main(["promote", "configs/apps/v1.yaml"]) == cli.EXIT_ERROR
    assert release_text(repo) is None
    assert cli.main(["promote", "configs/apps/v1.yaml"]) == cli.EXIT_OK


def test_warnings_need_explicit_review(repo, capsys):
    repo.write_app("v1")
    repo.write_app("pricey", usage_scale=2.0)            # twice the tokens: the advisory cost rule fails
    gate_and_promote(repo, "v1")
    assert cli.main(["gate", "configs/apps/pricey.yaml"]) == cli.EXIT_OK
    assert "PASS WITH WARNINGS" in capsys.readouterr().out
    assert cli.main(["promote", "configs/apps/pricey.yaml"]) == cli.EXIT_BLOCKED
    assert "--accept-warnings" in capsys.readouterr().out
    assert cli.main(["promote", "configs/apps/pricey.yaml", "--accept-warnings"]) == cli.EXIT_OK


def test_fault_injected_versions_can_never_be_promoted(repo, capsys):
    repo.write_app("chaos", faults={"timeout_rate": 0.0, "seed": 1})
    assert cli.main(["gate", "configs/apps/chaos.yaml"]) == cli.EXIT_OK
    assert cli.main(["promote", "configs/apps/chaos.yaml"]) == cli.EXIT_BLOCKED
    assert "can never be promoted" in capsys.readouterr().out
    assert release_text(repo) is None


def test_rollback_restores_the_previous_version(repo, capsys):
    repo.write_app("v1")
    repo.write_app("v2", prompt_version="p2")
    gate_and_promote(repo, "v1")
    gate_and_promote(repo, "v2")
    assert cli.main(["rollback", "--reason", "v2 misbehaves in production"]) == cli.EXIT_OK
    prod = load_release(repo.root / "configs/release.yaml").production
    assert prod.version == "v1" and prod.via == "rollback"
    assert events(repo)[-1] == "rollback"
    assert cli.main(["rollback", "--to", "v9", "--reason", "test"]) == cli.EXIT_BLOCKED


# --- the CI check -------------------------------------------------------------------------------

def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=root, check=True,
                   capture_output=True)


def record(repo, name):
    """Run an app version so its outputs are recorded in cache/ (what a developer does locally)."""
    repo.evaluator().run(repo.handle(name), RunOptions(quiet=True))


def ci_setup(repo):
    """main has v1 in production, committed with its recorded outputs."""
    repo.write_app("v1")
    gate_and_promote(repo, "v1")
    git(repo.root, "init", "-q")
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-qm", "v1 in production")


def test_ci_blocks_a_pull_request_that_releases_a_degraded_version(repo, tmp_path):
    ci_setup(repo)
    # The PR: a version that drops citations, recorded locally, with release.yaml pointed at it by hand.
    repo.write_app("v2", {"default": "no_citation"})
    record(repo, "v2")
    data = yaml.safe_load(release_text(repo))
    data["production"]["config"] = "configs/apps/v2.yaml"
    (repo.root / "configs/release.yaml").write_text(yaml.safe_dump(data))

    summary = tmp_path / "summary.md"
    assert cli.main(["ci", "--base-ref", "HEAD", "--summary", str(summary)]) == cli.EXIT_BLOCKED
    text = summary.read_text()
    assert "# Release decision: BLOCKED" in text and "**citations** (blocking, citation_required)" in text
    assert "Mode: REPLAY" in text
    # Green has a definition: the evaluation completed and was stored, but a requirement failed.
    assert "- PASS: evaluation completed" in text
    assert "- PASS: results and decision persisted" in text
    assert "- **FAIL**: every required metric meets its threshold (declines, citations, support)" in text
    assert "- **FAIL**: release.yaml matches the approved version" in text   # it was edited by hand


def test_ci_passes_a_properly_promoted_fix(repo, tmp_path):
    ci_setup(repo)
    repo.write_app("v3", prompt_version="p3")
    gate_and_promote(repo, "v3")
    summary = tmp_path / "summary.md"
    assert cli.main(["ci", "--base-ref", "HEAD", "--summary", str(summary)]) == cli.EXIT_OK
    checklist = [line for line in summary.read_text().splitlines() if line.startswith("- PASS") or "FAIL" in line[:12]]
    assert len(checklist) == 7 and all(line.startswith("- PASS") for line in checklist)


def test_ci_allows_a_rollback_to_an_approved_version(repo):
    repo.write_app("v1")
    repo.write_app("v2", prompt_version="p2")
    gate_and_promote(repo, "v1")
    gate_and_promote(repo, "v2")
    git(repo.root, "init", "-q")
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-qm", "v2 in production")
    assert cli.main(["rollback", "--reason", "v2 misbehaves in production"]) == cli.EXIT_OK
    assert cli.main(["ci", "--base-ref", "HEAD"]) == cli.EXIT_OK


def test_ci_fails_when_production_config_changed_without_promotion(repo):
    ci_setup(repo)
    repo.write_app("v1", prompt_version="edited-in-place")
    record(repo, "v1")
    assert cli.main(["ci", "--base-ref", "HEAD"]) == cli.EXIT_BLOCKED


def test_ci_fails_on_an_app_version_that_was_never_evaluated(repo, tmp_path):
    ci_setup(repo)
    repo.write_app("untested", prompt_version="never-run")
    summary = tmp_path / "summary.md"
    assert cli.main(["ci", "--base-ref", "HEAD", "--summary", str(summary)]) == cli.EXIT_BLOCKED
    assert "never evaluated" in summary.read_text()


def test_validate_command(repo, capsys):
    repo.write_app("v1")
    assert cli.main(["validate"]) == cli.EXIT_OK
    assert "dataset     toy/v1: ok" in capsys.readouterr().out


def test_a_fault_injected_copy_of_production_is_refused_not_a_no_op(repo, capsys):
    """Found by the failure drills: faults sit on top of the app, so the fingerprint matches production's."""
    repo.write_app("v1")
    repo.write_app("v1-chaos", faults={"timeout_rate": 0.2, "seed": 1})
    gate_and_promote(repo, "v1")
    assert cli.main(["promote", "configs/apps/v1-chaos.yaml"]) == cli.EXIT_BLOCKED
    assert "can never be promoted" in capsys.readouterr().out


def test_an_expired_decision_cannot_promote(repo, capsys):
    repo.write_app("v1")
    assert cli.main(["gate", "configs/apps/v1.yaml"]) == cli.EXIT_OK
    from sqlalchemy import text
    with store(repo).engine.begin() as conn:     # age the decision by ten days
        conn.execute(text("UPDATE gate_decisions SET created_at = datetime(created_at, '-10 days')"))
    assert cli.main(["promote", "configs/apps/v1.yaml"]) == cli.EXIT_BLOCKED
    assert "decisions expire after 168 hours" in capsys.readouterr().out


def test_a_rollback_needs_a_reason(repo):
    repo.write_app("v1")
    gate_and_promote(repo, "v1")
    with pytest.raises(SystemExit):
        cli.main(["rollback"])


def test_registry_mirror_failure_does_not_undo_a_promotion(repo, capsys):
    from evalops.tracking import TrackingError

    class BrokenTracker:
        def log_run(self, run, metrics, artifacts):
            return "mlflow-run", "model-1"

        def promote(self, *args):
            raise TrackingError("MLflow is down")

        def set_production_alias(self, *args):
            raise TrackingError("MLflow is down")

    repo.ev.tracker = BrokenTracker()
    repo.write_app("v1")
    gate_and_promote(repo, "v1")
    assert "run `evalops sync-registry`" in capsys.readouterr().out
    assert load_release(repo.root / "configs/release.yaml").production.version == "v1"   # the record stands
    assert events(repo)[-2:] == ["promoted", "registry_sync_failed"]
