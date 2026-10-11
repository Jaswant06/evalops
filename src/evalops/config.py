"""Typed, validated configuration files.

Every YAML file is parsed into a strict Pydantic model before an evaluation
starts: unknown keys, missing fields, a gate rule naming a metric that does not
exist, or a prompt file that is missing all stop the run before any model is
called.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from . import SCORING_VERSION
from .hashing import sha256_file, sha256_json
from .schemas import Strict


class ConfigError(RuntimeError):
    """A configuration file is missing or invalid."""


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_model(model, path: Path):
    try:
        return model.model_validate(read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from None


# --- Platform settings ----------------------------------------------------------

class ExecutionConfig(Strict):
    timeout_s: float = Field(60, gt=0)
    max_retries: int = Field(2, ge=0, le=5)        # transport errors and timeouts only
    concurrency: int = Field(1, ge=1, le=16)       # 1 keeps latency numbers clean
    judge_concurrency: int = Field(2, ge=1, le=16)


class PlatformConfig(Strict):
    dataset: str
    scoring: Path
    gate: Path
    pricing: Path
    release: Path
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path(".evalops/runs")
    execution: ExecutionConfig = ExecutionConfig()


# --- Application versions ---------------------------------------------------------

class FaultInjection(Strict):
    """Deterministic failures layered on top of real responses, for testing the gate."""

    timeout_rate: float = Field(0.0, ge=0, le=1)
    malformed_rate: float = Field(0.0, ge=0, le=1)
    error_rate: float = Field(0.0, ge=0, le=1)
    extra_latency_s: float = Field(0.0, ge=0)        # added to every successful response: a slower version
    abort_after: int | None = Field(None, ge=1)      # simulate the evaluation being interrupted after N cases
    seed: int = 0


class AppConfig(Strict):
    app: str
    version: str
    description: str = ""
    adapter: str
    settings: dict[str, Any] = {}
    fault_injection: FaultInjection | None = None


# --- Scoring (deterministic checks + LLM judge) -------------------------------------

class ModelRef(Strict):
    name: str
    digest: str  # first 12 hex chars of the Ollama manifest digest

    @field_validator("digest")
    @classmethod
    def _hex(cls, v: str) -> str:
        v = v.removeprefix("sha256:")[:12]
        if len(v) != 12 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("digest must be 12 lowercase hex characters")
        return v


class JudgeConfig(Strict):
    model: ModelRef
    prompts: dict[Literal["correctness", "claims"], Path]
    options: dict[str, Any] = {"temperature": 0, "seed": 7}
    max_attempts: int = Field(3, ge=1, le=5)
    timeout_s: float = Field(120, gt=0)


class ChecksConfig(Strict):
    # A retrieved passage counts as relevant when it contains at least this share
    # of the evidence passage's content words.
    passage_relevance_threshold: float = Field(0.5, gt=0, le=1)
    # Retrieval "hit" when the retrieved passages together cover this share.
    evidence_coverage_threshold: float = Field(0.6, gt=0, le=1)


class ScoringConfig(Strict):
    judge: JudgeConfig
    checks: ChecksConfig = ChecksConfig()

    def fingerprint(self, base_dir: Path) -> str:
        """Identifies the scoring logic: code version, judge model, judge prompts, parameters."""
        prompts = {}
        for name, path in self.judge.prompts.items():
            full = base_dir / path
            if not full.exists():
                raise ConfigError(f"Judge prompt not found: {full}")
            prompts[name] = sha256_file(full)
        return sha256_json({
            "scoring_version": SCORING_VERSION,
            "judge_model": self.judge.model.model_dump(),
            "judge_prompts": prompts,
            "judge_options": self.judge.options,
            "checks": self.checks.model_dump(),
        })


# --- Pricing -------------------------------------------------------------------

class ModelPrice(Strict):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class PricingConfig(Strict):
    currency: str = "USD"
    source: str
    as_of: str
    illustrative: bool = True
    models: dict[str, ModelPrice] = {}


# --- Release gate ------------------------------------------------------------------

class GateRule(Strict):
    id: str
    metric: str
    category: str = "all"
    description: str = ""
    severity: Literal["mandatory", "advisory"] = "mandatory"
    # Absolute thresholds on the candidate.
    min: float | None = None
    max: float | None = None
    # Limits relative to production, measured on the same cases.
    max_drop: float | None = Field(None, ge=0)
    max_increase: float | None = Field(None, ge=0)
    max_increase_pct: float | None = Field(None, ge=0)
    # Too few cases in the category makes the rule fail rather than pass vacuously.
    min_cases: int = Field(10, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "GateRule":
        from .metrics import METRICS  # avoid an import cycle

        if self.metric not in METRICS:
            raise ValueError(f"rule '{self.id}': unknown metric '{self.metric}'. Known: {sorted(METRICS)}")
        if all(v is None for v in (self.min, self.max, self.max_drop, self.max_increase, self.max_increase_pct)):
            raise ValueError(f"rule '{self.id}' sets no threshold")
        direction = METRICS[self.metric].direction
        if self.max_drop is not None and direction != "higher":
            raise ValueError(f"rule '{self.id}': max_drop only applies to higher-is-better metrics")
        if (self.max_increase is not None or self.max_increase_pct is not None) and direction != "lower":
            raise ValueError(f"rule '{self.id}': max_increase only applies to lower-is-better metrics")
        return self


class GateConfig(Strict):
    version: int
    description: str = ""
    rules: list[GateRule] = Field(min_length=1)
    # Statistical settings for the paired comparison shown in reports.
    bootstrap_samples: int = Field(2000, ge=200)
    confidence: float = Field(0.95, gt=0.5, lt=1)
    # A passing decision can be used for promotion for this long; after that, gate again.
    decision_valid_hours: float = Field(168, gt=0)

    @model_validator(mode="after")
    def _unique_ids(self) -> "GateConfig":
        ids = [r.id for r in self.rules]
        if dupes := {i for i in ids if ids.count(i) > 1}:
            raise ValueError(f"duplicate rule ids: {sorted(dupes)}")
        return self

    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


# --- Release record ----------------------------------------------------------------

class ProductionRelease(Strict):
    app: str
    version: str
    config: Path
    fingerprint: str
    app_version: dict[str, Any]   # AppVersion snapshot, so CI can replay it by fingerprint
    approved_run_id: str
    decision_id: str
    promoted_at: str
    promoted_by: str | None = None
    via: Literal["promote", "rollback"] = "promote"
    registry_version: str | None = None   # MLflow model registry version, for rollback


class ReleaseRecord(Strict):
    production: ProductionRelease | None = None
    history: list[dict[str, Any]] = []


def load_platform(path: Path = Path("configs/evalops.yaml")) -> PlatformConfig:
    return load_model(PlatformConfig, path)


def load_app(path: Path) -> AppConfig:
    return load_model(AppConfig, path)


def load_scoring(path: Path) -> ScoringConfig:
    return load_model(ScoringConfig, path)


def load_gate(path: Path) -> GateConfig:
    return load_model(GateConfig, path)


def load_pricing(path: Path) -> PricingConfig:
    return load_model(PricingConfig, path)


def load_release(path: Path) -> ReleaseRecord:
    if not path.exists():
        return ReleaseRecord()
    return load_model(ReleaseRecord, path)


def save_release(record: ReleaseRecord, path: Path) -> None:
    """Atomic: the file is either the old version or the new one, never half-written."""
    header = ("# What is in production. Written only by `evalops promote` (after a passing gate decision)\n"
              "# and `evalops rollback`. CI re-checks the production entry on every pull request.\n")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(header + yaml.safe_dump(record.model_dump(mode="json"), sort_keys=False, width=100))
    os.replace(tmp, path)
