"""Integration tests against a real PostgreSQL server (runs in CI; locally set EVALOPS_TEST_DATABASE_URL)."""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text

from evalops.engine import RunOptions
from evalops.store import Store

URL = os.environ.get("EVALOPS_TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres,
              pytest.mark.skipif(not URL, reason="set EVALOPS_TEST_DATABASE_URL to run PostgreSQL tests")]


def test_upserts_are_idempotent():
    store = Store(URL)
    store.init()
    run_id = f"pg-test-{uuid.uuid4().hex[:8]}"
    store.create_run({"run_id": run_id, "app": "a", "app_version_label": "1", "app_fingerprint": "f" * 64,
                      "app_version": {}, "dataset_ref": "d/v1:test", "dataset_hash": "0" * 64,
                      "scoring_fingerprint": "1" * 64, "run_fingerprint": "2" * 64, "mode": "auto",
                      "evalops_version": "0", "n_cases": 1})
    for _ in range(3):
        store.upsert_case(run_id, "c1", "generated", ["answerable"], execution={"x": 1})
    store.upsert_case(run_id, "c1", "scored", ["answerable"], record={"verdict": "correct"})
    assert store.count_cases(run_id) == 1
    row = store.get_cases(run_id)["c1"]
    assert row["stage"] == "scored" and row["execution"] == {"x": 1} and row["record"] == {"verdict": "correct"}


def test_full_run_on_postgres_is_queryable_as_jsonb(project):
    project.write_app("good", {"default": "good", "cases": {"c01": "wrong"}})
    ev = project.evaluator(database=URL)
    result = ev.run(project.handle("good"), RunOptions(quiet=True))
    with ev.store.engine.connect() as conn:
        verdicts = conn.execute(text("SELECT record->>'verdict' FROM case_results WHERE run_id = :r AND "
                                     "record->>'verdict' = 'incorrect'"), {"r": result.run_id}).all()
    assert len(verdicts) == 1
    assert ev.store.get_run(result.run_id)["status"] == "completed"
