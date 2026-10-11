"""MLflow integration: experiments, artifacts, app versions and the release alias.

Each evaluation run becomes an MLflow run in the experiment `evalops/<app>`,
with every version input logged as a parameter (dataset hash, app fingerprint,
model digest, prompt hash, scoring fingerprint) and every metric per category.
The app version itself is an MLflow *external model*: DocQA is not packaged as
an MLflow model, but its versions are tracked, compared and registered, and
`evalops promote` moves the registry alias `production` to the approved one.

Enabled when MLFLOW_TRACKING_URI is set. If it is set but unreachable, the run
fails rather than silently skipping tracking.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")   # EvalOps reports run links itself
# An external model has no input schema to infer; MLflow warns about it on every run.
logging.getLogger("mlflow.models.signature").setLevel(logging.ERROR)
# MLflow prints run links and registry progress to the console; EvalOps reports those itself.
logging.getLogger("mlflow").setLevel(logging.WARNING)


class TrackingError(RuntimeError):
    pass


def _flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = str(value)[:500]


class Tracker:
    def __init__(self, uri: str):
        import mlflow  # optional dependency

        # MLflow configures its own loggers on import; quiet its run links and registry progress,
        # which EvalOps reports itself.
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        self.mlflow = mlflow
        self.uri = uri
        mlflow.set_tracking_uri(uri)

    @classmethod
    def from_env(cls) -> "Tracker | None":
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            return None
        try:
            return cls(uri)
        except ImportError:
            raise TrackingError("MLFLOW_TRACKING_URI is set but mlflow is not installed: pip install -e '.[mlflow]'")

    def log_run(self, run: dict[str, Any], metrics: dict[str, dict[str, dict]], artifacts_dir: Path) -> tuple[str, str]:
        """Returns (mlflow_run_id, logged_model_id)."""
        mlflow = self.mlflow
        app_version = run["app_version"]
        try:
            mlflow.set_experiment(f"evalops/{run['app']}")
            with mlflow.start_run(run_name=run["run_id"]) as active:
                params: dict[str, str] = {}
                _flatten("app", {"name": run["app"], "version": run["app_version_label"],
                                 "fingerprint": run["app_fingerprint"]}, params)
                _flatten("app.components", app_version["components"], params)
                _flatten("dataset", {"ref": run["dataset_ref"], "hash": run["dataset_hash"]}, params)
                params["scoring.fingerprint"] = run["scoring_fingerprint"]
                params["run.fingerprint"] = run["run_fingerprint"]
                params["run.mode"] = run["mode"]
                params["run.latency_source"] = (run.get("execution") or {}).get("latency_source", "unknown")
                params["evalops.version"] = run["evalops_version"]
                mlflow.log_params(params)
                mlflow.set_tags({"evalops.run_id": run["run_id"], "git_sha": run.get("git_sha") or "unknown",
                                 "app": run["app"], "app_version": run["app_version_label"]})

                model = mlflow.create_external_model(
                    # MLflow model names cannot contain / : . % or quotes.
                    name=re.sub(r"[/:.%\"']", "-", f"{run['app']}-v{run['app_version_label']}"),
                    source_run_id=active.info.run_id,
                    params={k: v for k, v in params.items() if k.startswith("app.")},
                    tags={"fingerprint": run["app_fingerprint"]},
                    model_type="llm-app",
                )
                flat = {f"{cat}.{name}": v["value"] for cat, values in metrics.items()
                        for name, v in values.items() if v["value"] is not None}
                mlflow.log_metrics(flat, model_id=model.model_id)
                mlflow.log_artifacts(str(artifacts_dir), artifact_path="evaluation")
                return active.info.run_id, model.model_id
        except Exception as exc:  # MLflow raises many exception types for connection problems
            raise TrackingError(f"MLflow logging failed ({self.uri}): {exc}") from None

    def promote(self, app: str, model_id: str, decision_id: str, run_id: str) -> str:
        """Register the approved app version and point the `production` alias at it."""
        from mlflow import MlflowClient

        try:
            version = self.mlflow.register_model(f"models:/{model_id}", app,
                                                 tags={"decision_id": decision_id, "evalops.run_id": run_id})
            MlflowClient().set_registered_model_alias(app, "production", version.version)
            return str(version.version)
        except Exception as exc:
            raise TrackingError(f"MLflow registry update failed: {exc}") from None

    def set_production_alias(self, app: str, registry_version: str) -> None:
        """Point `production` at an already registered version (rollback)."""
        from mlflow import MlflowClient

        try:
            MlflowClient().set_registered_model_alias(app, "production", registry_version)
        except Exception as exc:
            raise TrackingError(f"MLflow registry update failed: {exc}") from None
