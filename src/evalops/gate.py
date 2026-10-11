"""The release gate: decide whether a candidate may replace production.

A decision has three parts:

1. Preconditions (always mandatory). Both runs completed; same dataset hash and
   split; same cases; same scoring fingerprint. If they fail, the comparison
   is meaningless and the gate fails.
2. Rules from gate.yaml, each on one category of cases (all, answerable,
   unanswerable, citation_required), with absolute thresholds, limits
   relative to production, or both. Relative limits use the same cases
   for both versions, and every comparison carries a paired bootstrap
   confidence interval, so a report never claims a 1-point gain is real
   when the interval says it might not be.
3. Severity. A failed mandatory rule blocks promotion. A failed advisory rule
   gives `pass_with_warnings`: promotable, but flagged for manual review.

The decision, every rule result and the gate config's fingerprint are stored,
so the decision can be audited and reproduced.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .config import GateConfig, GateRule
from .metrics import METRICS, CaseRecord, compute, paired_bootstrap, select
from .report import fmt, fmt_delta

Decision = Literal["pass", "pass_with_warnings", "fail"]


@dataclass
class RunData:
    run: dict[str, Any]
    records: list[CaseRecord]

    @property
    def run_id(self) -> str:
        return self.run["run_id"]

    @property
    def label(self) -> str:
        return f"{self.run['app']} {self.run['app_version_label']}"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class RuleResult:
    rule_id: str
    metric: str
    category: str
    severity: str
    description: str
    passed: bool
    candidate: float | None
    baseline: float | None
    delta: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    verdict: str
    thresholds: dict[str, float]
    reasons: list[str] = field(default_factory=list)


@dataclass
class GateDecision:
    decision_id: str
    decision: Decision
    candidate: str
    candidate_run_id: str
    baseline: str | None
    baseline_run_id: str | None
    gate_fingerprint: str
    preconditions: list[Check]
    rules: list[RuleResult]
    created_at: str
    # The exact versions this decision is about. `promote` accepts the decision
    # only while all of them still match what is on disk and in production.
    candidate_fingerprint: str = ""
    baseline_fingerprint: str | None = None
    dataset_ref: str = ""
    dataset_hash: str = ""
    scoring_fingerprint: str = ""
    candidate_config: str | None = None

    @property
    def promotable(self) -> bool:
        return self.decision != "fail"

    @property
    def mandatory_failures(self) -> list[RuleResult]:
        return [r for r in self.rules if not r.passed and r.severity == "mandatory"]

    @property
    def advisory_failures(self) -> list[RuleResult]:
        return [r for r in self.rules if not r.passed and r.severity == "advisory"]

    def to_dict(self) -> dict:
        return asdict(self)


def _thresholds(rule: GateRule) -> dict[str, float]:
    keys = ("min", "max", "max_drop", "max_increase", "max_increase_pct")
    return {k: getattr(rule, k) for k in keys if getattr(rule, k) is not None}


def preconditions(candidate: RunData, baseline: RunData | None) -> list[Check]:
    checks = [Check("candidate run completed", candidate.run["status"] == "completed",
                    f"status is {candidate.run['status']}")]
    if baseline is None:
        return checks
    c, b = candidate.run, baseline.run
    checks += [
        Check("baseline run completed", b["status"] == "completed", f"status is {b['status']}"),
        Check("same dataset", (c["dataset_ref"], c["dataset_hash"]) == (b["dataset_ref"], b["dataset_hash"]),
              f"{c['dataset_ref']} {c['dataset_hash'][:12]} vs {b['dataset_ref']} {b['dataset_hash'][:12]}"),
        Check("same cases", {r.case_id for r in candidate.records} == {r.case_id for r in baseline.records},
              f"{len(candidate.records)} vs {len(baseline.records)} cases"),
        Check("same scoring", c["scoring_fingerprint"] == b["scoring_fingerprint"],
              f"{c['scoring_fingerprint'][:12]} vs {b['scoring_fingerprint'][:12]}; rescore both with the same judge"),
    ]
    return checks


def evaluate_rule(rule: GateRule, candidate: RunData, baseline: RunData | None, gate: GateConfig) -> RuleResult:
    metric = METRICS[rule.metric]
    value, n = compute(metric, select(candidate.records, rule.category))
    cmp = paired_bootstrap(metric, baseline.records, candidate.records, rule.category, gate.bootstrap_samples,
                           gate.confidence) if baseline else None
    result = RuleResult(
        rule_id=rule.id, metric=rule.metric, category=rule.category, severity=rule.severity,
        description=rule.description or metric.description, passed=True, candidate=value,
        baseline=cmp.baseline if cmp else None, delta=cmp.delta if cmp else None,
        ci_low=cmp.ci_low if cmp else None, ci_high=cmp.ci_high if cmp else None, n=n,
        verdict=cmp.verdict if cmp else "no baseline", thresholds=_thresholds(rule),
    )
    reasons = result.reasons
    if n < rule.min_cases:
        reasons.append(f"only {n} cases in '{rule.category}' (needs {rule.min_cases}); the rule cannot be verified")
    if value is None:
        reasons.append("metric could not be computed for the candidate")
    else:
        if rule.min is not None and value < rule.min:
            reasons.append(f"{fmt(rule.metric, value)} is below the minimum {fmt(rule.metric, rule.min)}")
        if rule.max is not None and value > rule.max:
            reasons.append(f"{fmt(rule.metric, value)} is above the maximum {fmt(rule.metric, rule.max)}")
        b = result.baseline
        if b is not None:
            if rule.max_drop is not None and b - value > rule.max_drop + 1e-12:
                reasons.append(f"dropped {fmt_delta(rule.metric, value - b)} vs production "
                               f"(allowed: {fmt_delta(rule.metric, -rule.max_drop)})")
            if rule.max_increase is not None and value - b > rule.max_increase + 1e-12:
                reasons.append(f"rose {fmt_delta(rule.metric, value - b)} vs production "
                               f"(allowed: {fmt_delta(rule.metric, rule.max_increase)})")
            if rule.max_increase_pct is not None and b > 0 and (value - b) / b * 100 > rule.max_increase_pct:
                reasons.append(f"rose {(value - b) / b * 100:.0f}% vs production (allowed: {rule.max_increase_pct:.0f}%)")
    result.passed = not reasons
    return result


def evaluate_gate(gate: GateConfig, candidate: RunData, baseline: RunData | None) -> GateDecision:
    pre = preconditions(candidate, baseline)
    comparable = all(c.passed for c in pre)
    rules = [evaluate_rule(r, candidate, baseline if comparable else None, gate) for r in gate.rules]
    if not comparable or any(not r.passed and r.severity == "mandatory" for r in rules):
        decision: Decision = "fail"
    elif any(not r.passed for r in rules):
        decision = "pass_with_warnings"
    else:
        decision = "pass"
    now = datetime.now(timezone.utc)
    return GateDecision(
        decision_id=f"gate-{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}",
        decision=decision, candidate=candidate.label, candidate_run_id=candidate.run_id,
        baseline=baseline.label if baseline else None, baseline_run_id=baseline.run_id if baseline else None,
        gate_fingerprint=gate.fingerprint(), preconditions=pre, rules=rules,
        created_at=now.isoformat(timespec="seconds"),
        candidate_fingerprint=candidate.run["app_fingerprint"],
        baseline_fingerprint=baseline.run["app_fingerprint"] if baseline else None,
        dataset_ref=candidate.run["dataset_ref"], dataset_hash=candidate.run["dataset_hash"],
        scoring_fingerprint=candidate.run["scoring_fingerprint"],
    )
