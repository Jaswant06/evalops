"""Failure drills: break things on purpose and check that EvalOps responds correctly.

Each drill runs the real CLI in a disposable copy of the repository, on the
recorded DocQA outputs in replay mode (no models needed, so it also runs in
CI). For every drill it checks the exit code, what was recorded (run status,
gate decision, blocking rules, audit events) and that production was not
changed, except in the control drill, where a valid candidate must be promoted.

    python scripts/failure_drills.py
    python scripts/failure_drills.py --summary "$GITHUB_STEP_SUMMARY" --json drills.json

Exit code 0 when every drill behaved as expected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
COPY = ["configs", "prompts", "datasets", "cache"]
OK, BLOCKED, ERROR = 0, 1, 2


@dataclass
class Result:
    name: str
    scenario: str
    expected: str
    observed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    production_unchanged: bool | None = None

    @property
    def passed(self) -> bool:
        return not self.problems


class Sandbox:
    """A throwaway copy of the repository with its own results database."""

    def __init__(self, root: Path):
        self.root = root
        for name in COPY:
            if (REPO / name).exists():
                shutil.copytree(REPO / name, root / name)
        self.db = f"sqlite:///{root}/drill.db"
        self.release_before = self.release_bytes()

    def release_bytes(self) -> bytes:
        path = self.root / "configs/release.yaml"
        return path.read_bytes() if path.exists() else b""

    def cli(self, *args: str, db: str | None = None) -> tuple[int, str]:
        env = {k: v for k, v in os.environ.items()
               if k not in ("MLFLOW_TRACKING_URI", "EVALOPS_PUSHGATEWAY_URL", "GITHUB_STEP_SUMMARY")}
        env["EVALOPS_DATABASE_URL"] = db or self.db
        env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        out = subprocess.run([sys.executable, "-m", "evalops.cli", *args], cwd=self.root, env=env,
                             capture_output=True, text=True, timeout=600)
        return out.returncode, out.stdout + out.stderr

    def derive_app(self, base: str, name: str, **changes) -> str:
        """A new app config derived from an existing one (for example with fault injection)."""
        data = yaml.safe_load((self.root / "configs/apps" / base).read_text())
        for key, value in changes.items():
            if key == "settings":
                for k, v in value.items():
                    data["settings"][k] = {**data["settings"].get(k, {}), **v} if isinstance(v, dict) else v
            else:
                data[key] = value
        data["version"] = f"{data['version']}+{name}"
        path = f"configs/apps/drill-{name}.yaml"
        (self.root / path).write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    def query(self, sql: str) -> list[tuple]:
        from sqlalchemy import create_engine, text
        engine = create_engine(self.db)
        with engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(sql))]

    def latest_decision(self) -> tuple[str, list[str]] | None:
        rows = self.query("SELECT decision, rules FROM gate_decisions ORDER BY created_at DESC LIMIT 1")
        if not rows:
            return None
        decision, rules = rows[0]
        rules = json.loads(rules) if isinstance(rules, str) else rules
        return decision, [r["rule_id"] for r in rules if not r["passed"] and r["severity"] == "mandatory"]

    def audit(self) -> list[str]:
        return [r[0] for r in self.query("SELECT event FROM audit_events ORDER BY created_at")]


def expect_exit(result: Result, step: str, code: int, wanted: int, output: str) -> None:
    result.observed.append(f"{step}: exit {code}")
    if code != wanted:
        tail = " ".join(output.strip().splitlines()[-2:])[:200]
        result.problems.append(f"{step} exited {code}, expected {wanted} ({tail})")


def blocked_by(result: Result, box: Sandbox, required: set[str], exactly: bool = False) -> None:
    latest = box.latest_decision()
    if latest is None:
        result.problems.append("no gate decision was recorded")
        return
    decision, blocking = latest
    result.observed.append(f"decision {decision}, blocking: {', '.join(blocking) or 'none'}")
    if decision != "fail":
        result.problems.append(f"decision was {decision}, expected fail")
    missing = required - set(blocking)
    if missing:
        result.problems.append(f"expected blocking rules missing: {sorted(missing)}")
    if exactly and set(blocking) != required:
        result.problems.append(f"expected exactly {sorted(required)} to block")


def unchanged(result: Result, box: Sandbox) -> None:
    result.production_unchanged = box.release_bytes() == box.release_before
    if not result.production_unchanged:
        result.problems.append("configs/release.yaml changed")


def gate_then_promote(result: Result, box: Sandbox, app: str, required: set[str], exactly: bool = False) -> None:
    code, out = box.cli("gate", app, "--mode", "replay")
    expect_exit(result, "gate", code, BLOCKED, out)
    blocked_by(result, box, required, exactly)
    code, out = box.cli("promote", app)
    expect_exit(result, "promote", code, BLOCKED, out)
    events = box.audit()
    if events[-2:] != ["gate_decision", "promotion_refused"]:
        result.problems.append(f"audit trail was {events}, expected a gate decision then a refused promotion")
    else:
        result.observed.append("audit: gate_decision, promotion_refused")
    unchanged(result, box)


# --- the drills -------------------------------------------------------------------------------------

def drill_incorrect_answers(box: Sandbox) -> Result:
    r = Result("incorrect-answers", "A candidate answers many questions wrongly (DocQA 2.0 on qwen3.5:0.8b)",
               "gate BLOCKED on answer correctness; promote refused; production unchanged")
    gate_then_promote(r, box, "configs/apps/docqa-v2.0.yaml", {"answer-correctness"})
    return r


def drill_citation_support(box: Sandbox) -> Result:
    r = Result("citation-support", "A prompt change makes answers cite passages that do not support them (DocQA 1.1)",
               "gate BLOCKED on citation support; promote refused; production unchanged")
    gate_then_promote(r, box, "configs/apps/docqa-v1.1.yaml", {"citation-support"})
    return r


def drill_timeouts(box: Sandbox) -> Result:
    app = box.derive_app("docqa-v1.0.yaml", "timeouts", fault_injection={"timeout_rate": 0.05, "seed": 7})
    r = Result("timeouts", "5% of model requests time out (injected on top of the production version)",
               "timeouts recorded; gate BLOCKED on the application failure rate; promote refused; production unchanged")
    gate_then_promote(r, box, app, {"app-failures"})
    rows = box.query("SELECT m.value FROM run_metrics m JOIN runs r ON r.run_id = m.run_id "
                     "WHERE m.metric = 'app_timeout_rate' AND m.category = 'all' "
                     "AND r.app_version_label LIKE '%timeouts'")
    if rows:
        r.observed.append(f"timeout_rate {rows[0][0]:.1%}")
    return r


def drill_malformed(box: Sandbox) -> Result:
    app = box.derive_app("docqa-v1.0.yaml", "malformed", fault_injection={"malformed_rate": 0.03, "seed": 7})
    r = Result("malformed-output", "3% of responses are truncated JSON (injected)",
               "the schema parser flags them; gate BLOCKED on the application failure rate; production unchanged")
    gate_then_promote(r, box, app, {"app-failures"})
    return r


def drill_slower(box: Sandbox) -> Result:
    app = box.derive_app("docqa-v1.0.yaml", "slower", fault_injection={"extra_latency_s": 2.5, "seed": 7})
    r = Result("quality-ok-but-slow", "Same answers as production, 2.5 s slower per request (injected)",
               "gate BLOCKED only by the p95 latency rule; production unchanged")
    gate_then_promote(r, box, app, {"latency-p95"}, exactly=True)
    return r


def drill_dataset_changed(box: Sandbox) -> Result:
    r = Result("dataset-changed", "A test question is edited after the dataset version was published",
               "evaluation refused before it starts (hash mismatch, exit 2); production unchanged")
    path = box.root / "datasets/policies/v1/test.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["input"]["question"] += " Please answer briefly."
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    code, out = box.cli("gate", "configs/apps/docqa-v1.2.yaml", "--mode", "replay")
    expect_exit(r, "gate", code, ERROR, out)
    if "does not match its manifest hash" not in out:
        r.problems.append("the hash mismatch was not reported")
    else:
        r.observed.append("reported: test.jsonl does not match its manifest hash")
    if box.query("SELECT count(*) FROM runs")[0][0] if (box.root / "drill.db").exists() else 0:
        r.problems.append("a run was created from the modified dataset")
    unchanged(r, box)
    return r


def drill_interrupted(box: Sandbox) -> Result:
    app = box.derive_app("docqa-v1.0.yaml", "interrupted", fault_injection={"abort_after": 100, "seed": 7})
    r = Result("interrupted", "The evaluation is interrupted after 100 of 250 cases (injected)",
               "run marked interrupted, never completed; promote refused; resume finishes without duplicates")
    code, out = box.cli("run", app, "--mode", "replay")
    expect_exit(r, "run", code, ERROR, out)
    rows = box.query("SELECT run_id, status FROM runs ORDER BY created_at DESC LIMIT 1")
    run_id, status = rows[0]
    count = box.query(f"SELECT count(*) FROM case_results WHERE run_id = '{run_id}'")[0][0]
    r.observed.append(f"run status {status}, {count} cases stored")
    if status != "interrupted":
        r.problems.append(f"run status was {status}, expected interrupted")
    code, out = box.cli("promote", app)
    expect_exit(r, "promote", code, BLOCKED, out)
    code, out = box.cli("run", app, "--mode", "replay", "--resume", run_id)
    expect_exit(r, "resume", code, OK, out)
    status, = box.query(f"SELECT status FROM runs WHERE run_id = '{run_id}'")[0]
    count = box.query(f"SELECT count(*) FROM case_results WHERE run_id = '{run_id}'")[0][0]
    r.observed.append(f"after resume: {status}, {count} cases")
    if status != "completed" or count != 250:
        r.problems.append(f"after resume the run was {status} with {count} cases, expected completed with 250")
    unchanged(r, box)
    return r


def drill_database_down(box: Sandbox) -> Result:
    r = Result("database-down", "PostgreSQL is unreachable when an evaluation starts",
               "evaluation refused (exit 2), nothing reported as successful; production unchanged")
    code, out = box.cli("gate", "configs/apps/docqa-v1.2.yaml", "--mode", "replay",
                        db="postgresql+psycopg://nobody:x@127.0.0.1:1/none")
    expect_exit(r, "gate", code, ERROR, out)
    if "Results database unavailable" in out:
        r.observed.append("reported: results database unavailable")
    else:
        r.problems.append("the database failure was not reported")
    unchanged(r, box)
    return r


def drill_model_version_changed(box: Sandbox) -> Result:
    app = box.derive_app("docqa-v1.0.yaml", "new-weights", settings={"model": {"digest": "0123456789ab"}})
    r = Result("model-version-changed", "Same app, but the model weights (digest) changed",
               "recordings of the old weights are not reused (exit 2); production unchanged")
    code, out = box.cli("gate", app, "--mode", "replay")
    expect_exit(r, "gate", code, ERROR, out)
    if "No recorded app output" in out:
        r.observed.append("reported: no recorded output for this model version")
    else:
        r.problems.append("the version mismatch was not reported")
    unchanged(r, box)
    return r


def drill_valid_candidate(box: Sandbox) -> Result:
    r = Result("control: valid candidate", "The fixed prompt (DocQA 1.2), gated against production 1.0",
               "gate PASS; promote succeeds and is recorded (production changes, as it should)")
    release = yaml.safe_load((box.root / "configs/release.yaml").read_text())
    if release["production"]["version"] != "1.0.0":
        # Make the sandbox's production 1.0 regardless of the repository's current state.
        entry = next(h for h in release["history"] if h["version"] == "1.0.0")
        production = {k: v for k, v in entry.items() if k not in ("replaced_at", "replaced_by", "reason")}
        (box.root / "configs/release.yaml").write_text(yaml.safe_dump({"production": production, "history": []}))
        box.release_before = box.release_bytes()
    code, out = box.cli("gate", "configs/apps/docqa-v1.2.yaml", "--mode", "replay")
    expect_exit(r, "gate", code, OK, out)
    code, out = box.cli("promote", "configs/apps/docqa-v1.2.yaml")
    expect_exit(r, "promote", code, OK, out)
    after = yaml.safe_load(box.release_bytes())
    r.production_unchanged = box.release_bytes() == box.release_before
    r.observed.append(f"production is now {after['production']['version']}")
    if after["production"]["version"] != "1.2.0":
        r.problems.append("the valid candidate was not promoted")
    if box.audit()[-2:] != ["gate_decision", "promoted"]:
        r.problems.append(f"audit trail was {box.audit()}")
    return r


DRILLS = [drill_incorrect_answers, drill_citation_support, drill_timeouts, drill_malformed, drill_slower,
          drill_dataset_changed, drill_interrupted, drill_database_down, drill_model_version_changed,
          drill_valid_candidate]


def to_markdown(results: list[Result]) -> str:
    out = ["# EvalOps failure drills", "",
           "Each drill breaks something on purpose in a disposable copy of the repository, using the recorded "
           "DocQA outputs, and checks what EvalOps did.", "",
           "| Drill | What was broken | Expected | Observed | Production unchanged | Result |",
           "|---|---|---|---|---|---|"]
    for r in results:
        prod = {True: "yes", False: "no (promoted)" if r.name.startswith("control") else "**NO**", None: "-"}[
            r.production_unchanged]
        res = "PASS" if r.passed else "**FAIL**: " + "; ".join(r.problems)
        out.append(f"| {r.name} | {r.scenario} | {r.expected} | {'; '.join(r.observed)} | {prod} | {res} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", help="append the Markdown table here (for example $GITHUB_STEP_SUMMARY)")
    parser.add_argument("--json", help="write the results as JSON")
    parser.add_argument("--only", help="run only drills whose name contains this text")
    args = parser.parse_args()

    results = []
    for drill in DRILLS:
        name = drill.__name__.removeprefix("drill_").replace("_", "-")
        if args.only and args.only not in name:
            continue
        with tempfile.TemporaryDirectory(prefix="evalops-drill-") as tmp:
            result = drill(Sandbox(Path(tmp)))
        results.append(result)
        print(f"{'PASS' if result.passed else 'FAIL'}  {result.name}: {'; '.join(result.observed)}"
              + ("" if result.passed else f"\n      problems: {'; '.join(result.problems)}"), flush=True)

    text = to_markdown(results)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    if args.json:
        Path(args.json).write_text(json.dumps([r.__dict__ | {"passed": r.passed} for r in results], indent=2))
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} drills behaved as expected.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
