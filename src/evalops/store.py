"""Results store: runs, per-case results, metrics, gate decisions and releases.

PostgreSQL in Docker and in CI; SQLite for a quick local run without Docker.
Writes are idempotent upserts keyed by (run_id, case_id), so retries and
resumed runs never create duplicate results. A run is only ever marked
`completed` by the engine's final step; anything that stops it earlier leaves
it `running`, `interrupted` or `failed`, and the gate refuses those.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine,
                        func, inspect, select, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError

DEFAULT_URL = "sqlite:///evalops.db"
# Bump when a table changes; an older database is refused instead of half-working.
SCHEMA_VERSION = "4"

RUN_STATUSES = ("running", "completed", "failed", "interrupted")

metadata = MetaData()
JSONType = JSON().with_variant(JSONB(), "postgresql")

meta = Table(
    "evalops_meta", metadata,
    Column("key", String(40), primary_key=True),
    Column("value", String(200), nullable=False),
)

runs = Table(
    "runs", metadata,
    Column("run_id", String(120), primary_key=True),
    Column("status", String(20), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("app", String(80), nullable=False),
    Column("app_version_label", String(40), nullable=False),
    Column("app_fingerprint", String(64), nullable=False),
    Column("app_version", JSONType, nullable=False),
    Column("dataset_ref", String(120), nullable=False),
    Column("dataset_hash", String(64), nullable=False),
    Column("scoring_fingerprint", String(64), nullable=False),
    Column("run_fingerprint", String(64), nullable=False),
    Column("mode", String(20), nullable=False),            # replay | live | rescore | auto
    Column("evalops_version", String(20), nullable=False),
    Column("git_sha", String(40)),
    Column("n_cases", Integer, nullable=False),
    Column("execution", JSONType),                         # live vs replayed counts, latency source
    Column("error", Text),
    Column("error_class", String(40)),                     # infrastructure | missing_recording | interrupted | ...
    Column("mlflow_run_id", String(64)),
    Column("mlflow_model_id", String(64)),
    Column("report_path", Text),
)

case_results = Table(
    "case_results", metadata,
    Column("run_id", String(120), primary_key=True),
    Column("case_id", String(80), primary_key=True),
    Column("stage", String(20), nullable=False),          # generated | scored
    Column("categories", JSONType, nullable=False),
    Column("execution", JSONType),
    Column("record", JSONType),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

run_metrics = Table(
    "run_metrics", metadata,
    Column("run_id", String(120), primary_key=True),
    Column("category", String(40), primary_key=True),
    Column("metric", String(60), primary_key=True),
    Column("value", Float),
    Column("n", Integer, nullable=False),
)

gate_decisions = Table(
    "gate_decisions", metadata,
    Column("decision_id", String(80), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("candidate_run_id", String(120), nullable=False),
    Column("baseline_run_id", String(120)),
    Column("decision", String(30), nullable=False),        # pass | pass_with_warnings | fail
    Column("mandatory_failures", Integer, nullable=False),
    Column("advisory_failures", Integer, nullable=False),
    Column("rules", JSONType, nullable=False),
    Column("gate_fingerprint", String(64), nullable=False),
    # The exact versions the decision is about; promotion re-checks every one.
    Column("candidate_fingerprint", String(64), nullable=False),
    Column("baseline_fingerprint", String(64)),
    Column("candidate_config", Text),
    Column("dataset_ref", String(120), nullable=False),
    Column("dataset_hash", String(64), nullable=False),
    Column("scoring_fingerprint", String(64), nullable=False),
    Column("context", String(20), nullable=False),         # cli | ci | api
    Column("git_sha", String(40)),
    Column("report", Text),
)

# Append-only record of everything that touched (or tried to touch) production.
audit_events = Table(
    "audit_events", metadata,
    Column("event_id", String(80), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("event", String(30), nullable=False),     # gate_decision | promoted | promotion_refused | promotion_noop | rollback
    Column("app", String(80)),
    Column("version", String(40)),
    Column("fingerprint", String(64)),
    Column("decision_id", String(80)),
    Column("actor", String(120)),
    Column("details", JSONType),
)

releases = Table(
    "releases", metadata,
    Column("release_id", String(80), primary_key=True),
    Column("app", String(80), nullable=False),
    Column("version", String(40), nullable=False),
    Column("fingerprint", String(64), nullable=False),
    Column("run_id", String(120), nullable=False),
    Column("decision_id", String(80), nullable=False),
    Column("promoted_at", DateTime(timezone=True), nullable=False),
)


class StoreUnavailable(RuntimeError):
    """The results database cannot be reached; no run may be reported as successful."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("EVALOPS_DATABASE_URL", DEFAULT_URL)
        if self.url.startswith("sqlite:///") and self.url != "sqlite:///:memory:":
            Path(self.url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"connect_timeout": 5} if self.url.startswith("postgresql") else {}
        self.engine = create_engine(self.url, future=True, pool_pre_ping=True, connect_args=connect_args)

    def describe(self) -> str:
        return self.engine.url.render_as_string(hide_password=True)

    def init(self) -> None:
        """Check the connection and create tables. Raises StoreUnavailable."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
                has_runs = inspect(conn).has_table("runs")
            metadata.create_all(self.engine)
            with self.engine.begin() as conn:
                version = conn.execute(select(meta.c.value).where(meta.c.key == "schema_version")).scalar()
                if version is None and not has_runs:
                    conn.execute(meta.insert().values(key="schema_version", value=SCHEMA_VERSION))
                    version = SCHEMA_VERSION
        except SQLAlchemyError as exc:
            raise StoreUnavailable(f"Results database unavailable ({self.describe()}): {exc.__class__.__name__}: "
                                   f"{str(exc).splitlines()[0]}") from None
        if version != SCHEMA_VERSION:
            raise StoreUnavailable(
                f"The results database at {self.describe()} was created by an older EvalOps (schema "
                f"{version or '1'}, need {SCHEMA_VERSION}). Point EVALOPS_DATABASE_URL at a new database, or reset "
                f"the Docker one with: docker compose down -v")

    def _exec(self, stmt):
        try:
            with self.engine.begin() as conn:
                return conn.execute(stmt)
        except SQLAlchemyError as exc:
            raise StoreUnavailable(f"Results database error: {str(exc).splitlines()[0]}") from None

    def _upsert(self, table: Table, values: dict, keys: list[str]):
        dialect = self.engine.dialect.name
        ins = (pg_insert if dialect == "postgresql" else sqlite_insert)(table).values(**values)
        update = {k: ins.excluded[k] for k in values if k not in keys}
        return self._exec(ins.on_conflict_do_update(index_elements=keys, set_=update))

    # --- runs ---------------------------------------------------------------------

    def create_run(self, row: dict[str, Any]) -> None:
        self._exec(runs.insert().values(status="running", created_at=utcnow(), **row))

    def get_run(self, run_id: str) -> dict | None:
        result = self._exec(select(runs).where(runs.c.run_id == run_id)).mappings().first()
        return dict(result) if result else None

    def update_run(self, run_id: str, **values) -> None:
        if "status" in values and values["status"] not in RUN_STATUSES:
            raise ValueError(f"bad run status {values['status']}")
        if values.get("status") in ("completed", "failed", "interrupted"):
            values.setdefault("finished_at", utcnow())
        self._exec(runs.update().where(runs.c.run_id == run_id).values(**values))

    def list_runs(self, limit: int = 50, app: str | None = None, status: str | None = None) -> list[dict]:
        stmt = select(runs).order_by(runs.c.created_at.desc()).limit(limit)
        if app:
            stmt = stmt.where(runs.c.app == app)
        if status:
            stmt = stmt.where(runs.c.status == status)
        return [dict(r) for r in self._exec(stmt).mappings().all()]

    def stale_runs(self, older_than_minutes: float) -> list[dict]:
        """Runs still marked `running` with no activity for a while: their process died (for example the
        database went away before the failure could be recorded)."""
        cutoff = utcnow().timestamp() - older_than_minutes * 60
        stale = []
        for run in self.list_runs(limit=1000, status="running"):
            last = self._exec(select(func.max(case_results.c.updated_at)).where(
                case_results.c.run_id == run["run_id"])).scalar() or run["created_at"]
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if last.timestamp() < cutoff:
                stale.append(run)
        return stale

    def latest_completed(self, run_fingerprint: str) -> dict | None:
        stmt = (select(runs).where(runs.c.run_fingerprint == run_fingerprint, runs.c.status == "completed")
                .order_by(runs.c.created_at.desc()).limit(1))
        r = self._exec(stmt).mappings().first()
        return dict(r) if r else None

    # --- cases --------------------------------------------------------------------

    def upsert_case(self, run_id: str, case_id: str, stage: str, categories: list[str],
                    execution: dict | None = None, record: dict | None = None) -> None:
        values = {"run_id": run_id, "case_id": case_id, "stage": stage, "categories": categories,
                  "updated_at": utcnow()}
        if execution is not None:
            values["execution"] = execution
        if record is not None:
            values["record"] = record
        self._upsert(case_results, values, ["run_id", "case_id"])

    def get_cases(self, run_id: str) -> dict[str, dict]:
        rows = self._exec(select(case_results).where(case_results.c.run_id == run_id)).mappings().all()
        return {r["case_id"]: dict(r) for r in rows}

    def count_cases(self, run_id: str, stage: str | None = None) -> int:
        stmt = select(func.count()).select_from(case_results).where(case_results.c.run_id == run_id)
        if stage:
            stmt = stmt.where(case_results.c.stage == stage)
        return int(self._exec(stmt).scalar_one())

    # --- metrics ------------------------------------------------------------------

    def save_metrics(self, run_id: str, metrics: dict[str, dict[str, dict]]) -> None:
        for category, values in metrics.items():
            for metric, v in values.items():
                self._upsert(run_metrics, {"run_id": run_id, "category": category, "metric": metric,
                                           "value": v["value"], "n": v["n"]}, ["run_id", "category", "metric"])

    def get_metrics(self, run_id: str) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        for r in self._exec(select(run_metrics).where(run_metrics.c.run_id == run_id)).mappings():
            out.setdefault(r["category"], {})[r["metric"]] = {"value": r["value"], "n": r["n"]}
        return out

    # --- gate decisions and releases ------------------------------------------------

    def save_decision(self, row: dict[str, Any]) -> None:
        self._exec(gate_decisions.insert().values(created_at=utcnow(), **row))

    def get_decision(self, decision_id: str) -> dict | None:
        r = self._exec(select(gate_decisions).where(gate_decisions.c.decision_id == decision_id)).mappings().first()
        return dict(r) if r else None

    def list_decisions(self, limit: int = 50) -> list[dict]:
        stmt = select(gate_decisions).order_by(gate_decisions.c.created_at.desc()).limit(limit)
        return [dict(r) for r in self._exec(stmt).mappings().all()]

    def latest_decision(self, candidate_fingerprint: str, dataset_ref: str) -> dict | None:
        stmt = (select(gate_decisions).where(gate_decisions.c.candidate_fingerprint == candidate_fingerprint,
                                             gate_decisions.c.dataset_ref == dataset_ref)
                .order_by(gate_decisions.c.created_at.desc()).limit(1))
        r = self._exec(stmt).mappings().first()
        return dict(r) if r else None

    def audit(self, event: str, *, app: str | None = None, version: str | None = None,
              fingerprint: str | None = None, decision_id: str | None = None, actor: str | None = None,
              details: dict | None = None) -> str:
        event_id = f"evt-{utcnow().strftime('%Y%m%dT%H%M%S%f')}-{os.urandom(2).hex()}"
        self._exec(audit_events.insert().values(
            event_id=event_id, created_at=utcnow(), event=event, app=app, version=version, fingerprint=fingerprint,
            decision_id=decision_id, actor=actor, details=details or {}))
        return event_id

    def list_audit(self, limit: int = 50) -> list[dict]:
        stmt = select(audit_events).order_by(audit_events.c.created_at.desc()).limit(limit)
        return [dict(r) for r in self._exec(stmt).mappings().all()]

    def save_release(self, row: dict[str, Any]) -> None:
        self._exec(releases.insert().values(promoted_at=utcnow(), **row))

    def list_releases(self) -> list[dict]:
        return [dict(r) for r in self._exec(select(releases).order_by(releases.c.promoted_at.desc())).mappings()]
