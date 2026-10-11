"""Explaining a release decision: production versus candidate, rule by rule and case by case.

Two renderings of the same decision:
  render_markdown  the comparison report (files, CI job summary, API)
  render_text      the terminal summary printed by `evalops gate` and `evalops ci`

Both show, for every rule, the production value, the candidate value, the
change with its 95% confidence interval, the requirement and the result; then
the reasons and the individual test cases responsible for each regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GateConfig
from .gate import GateDecision, RuleResult, RunData
from .metrics import METRICS, CaseRecord, MetricDef, paired_bootstrap, select
from .report import SECTIONS, execution_lines, fmt, fmt_delta

DECISION_WORD = {"pass": "PASS", "pass_with_warnings": "PASS WITH WARNINGS", "fail": "BLOCKED"}

DIMENSION = {metric: section for section, metrics in SECTIONS.items() for metric in metrics}

# Per-category metrics for the "results by category" section.
CATEGORY_METRICS = {
    "answerable": ["answer_correctness", "answer_quality", "reference_f1", "false_abstention_rate", "retrieval_recall"],
    "unanswerable": ["correct_abstention_rate", "inappropriate_answer_rate"],
    "citation_required": ["citation_missing_rate", "citation_invalid_rate", "citation_support_rate"],
    "all": ["abstention_accuracy", "unsupported_claim_rate", "contradiction_rate", "app_failure_rate", "judge_unevaluable_rate",
            "latency_p50_s", "latency_p95_s", "est_cost_per_success_usd"],
}


@dataclass
class CaseChange:
    case_id: str
    question: str
    production_answer: str | None
    candidate_answer: str | None
    what_changed: str
    severity: float


def _case_value(metric: MetricDef, r: CaseRecord) -> float | None:
    if metric.ratio is not None:
        num, den = metric.ratio([r])
        return float(num[0] / den[0]) if den[0] > 0 else None
    v = metric.values([r])[0]
    return None if np.isnan(v) else float(v)


def _explain(metric: str, b: CaseRecord, c: CaseRecord, bv: float, cv: float) -> str:
    if b.status == "ok" and c.status != "ok":
        return f"request failed: {c.status}" + (f" ({c.error})" if c.error else "")
    if metric in ("correct_abstention_rate", "abstention_accuracy", "inappropriate_answer_rate") and c.answered:
        return "answered a question the document cannot answer"
    if metric == "false_abstention_rate" or (c.abstained and not b.abstained and c.answerable):
        return "declined a question the document answers"
    if metric == "answer_correctness":
        return f"verdict {b.verdict or 'n/a'} -> {c.verdict or 'n/a'}"
    if metric == "citation_support_rate":
        return (f"claims supported by its citations {b.claims_supported_cited}/{b.claims_total} -> "
                f"{c.claims_supported_cited}/{c.claims_total}")
    if metric == "unsupported_claim_rate":
        bu = (b.claims_total or 0) - (b.claims_supported_document or 0)
        cu = (c.claims_total or 0) - (c.claims_supported_document or 0)
        unsupported = [x["claim"] for x in (c.claims or []) if not x["supported_by_document"]]
        return f"unsupported claims {bu} -> {cu}" + (f": \"{unsupported[0]}\"" if unsupported else "")
    if metric == "citation_missing_rate":
        return f"citations {b.n_citations} -> {c.n_citations}"
    if metric.startswith("latency") or metric == "ttft_p50_s":
        return f"original latency {b.original_latency_s:.2f} s -> {c.original_latency_s:.2f} s"
    return f"{fmt(metric, bv)} -> {fmt(metric, cv)}"


def regression_cases(rule: RuleResult, candidate: RunData, baseline: RunData, questions: dict[str, str],
                     limit: int = 8) -> list[CaseChange]:
    """The cases in the rule's category that got worse on the rule's metric, worst first."""
    metric = METRICS[rule.metric]
    if metric.direction == "none":
        return []
    base = {r.case_id: r for r in select(baseline.records, rule.category)}
    changes = []
    for c in select(candidate.records, rule.category):
        b = base.get(c.case_id)
        if b is None:
            continue
        if metric.values is not None:   # percentiles: per-case latency, flag clear slowdowns
            if c.original_latency_s is None or b.original_latency_s is None or c.original_latency_s < max(b.original_latency_s * 1.5, b.original_latency_s + 0.5):
                continue
            bv, cv = b.original_latency_s, c.original_latency_s
        else:
            bv, cv = _case_value(metric, b), _case_value(metric, c)
            if bv is None or cv is None:
                continue
        worse = (cv < bv) if metric.direction == "higher" else (cv > bv)
        if worse and abs(cv - bv) > 1e-9:
            changes.append(CaseChange(c.case_id, questions.get(c.case_id, ""), b.answer, c.answer,
                                      _explain(rule.metric, b, c, bv, cv), abs(cv - bv)))
    changes.sort(key=lambda x: (-x.severity, x.case_id))
    return changes[:limit] if limit else changes


def _requirement(r: RuleResult) -> str:
    parts = []
    for k, v in r.thresholds.items():
        if k == "min":
            parts.append(f">= {fmt(r.metric, v)}")
        elif k == "max":
            parts.append(f"<= {fmt(r.metric, v)}")
        elif k == "max_drop":
            parts.append(f"drop <= {fmt_delta(r.metric, v).lstrip('+')}")
        elif k == "max_increase":
            parts.append(f"rise <= {fmt_delta(r.metric, v).lstrip('+')}")
        else:
            parts.append(f"rise <= {v:.0f}%")
    return ", ".join(parts)


def _result(r: RuleResult) -> str:
    return "PASS" if r.passed else ("FAIL" if r.severity == "mandatory" else "WARN")


def _change(r: RuleResult) -> str:
    if r.delta is None:
        return "n/a"
    ci = f" [{fmt_delta(r.metric, r.ci_low)}, {fmt_delta(r.metric, r.ci_high)}]" if r.ci_low is not None else ""
    return fmt_delta(r.metric, r.delta) + ci


def _grouped(d: GateDecision) -> list[tuple[str, list[RuleResult]]]:
    order = list(SECTIONS)
    groups: dict[str, list[RuleResult]] = {}
    for r in d.rules:
        groups.setdefault(DIMENSION.get(r.metric, "Other"), []).append(r)
    return sorted(groups.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))


def _model(run: dict) -> str:
    comp = run["app_version"]["components"]
    model = comp.get("model", {})
    prompt = comp.get("prompt", {})
    return (f"{model.get('name', '?')} @ {model.get('digest', '?')}, prompt {prompt.get('path', '?')} "
            f"({str(prompt.get('sha256', '?'))[:8]})")


def render_markdown(d: GateDecision, candidate: RunData, baseline: RunData | None, gate: GateConfig,
                    questions: dict[str, str]) -> str:
    word = DECISION_WORD[d.decision]
    out = [f"# Release decision: {word}", ""]
    out.append(f"**Candidate {d.candidate}** vs **production {d.baseline or '(none: first release)'}**, "
               f"decision `{d.decision_id}`.")
    out += ["", "| | Production | Candidate |", "|---|---|---|"]
    b = baseline.run if baseline else None
    c = candidate.run
    out.append(f"| Version | {b['app_version_label'] if b else '-'} | {c['app_version_label']} |")
    out.append(f"| Model and prompt | {_model(b) if b else '-'} | {_model(c)} |")
    out.append(f"| Fingerprint | `{b['app_fingerprint'][:12]}` | `{c['app_fingerprint'][:12]}` |" if b else
               f"| Fingerprint | - | `{c['app_fingerprint'][:12]}` |")
    out.append(f"| Evaluation run | `{b['run_id']}` | `{c['run_id']}` |" if b else
               f"| Evaluation run | - | `{c['run_id']}` |")
    lat = lambda run: (run.get("execution") or {}).get("latency_source", "unknown")  # noqa: E731
    out.append(f"| Latency figures | {lat(b) if b else '-'} | {lat(c)} |")
    out.append(f"\nDataset `{d.dataset_ref}` (hash `{d.dataset_hash[:12]}`), {len(candidate.records)} cases, "
               f"scoring `{d.scoring_fingerprint[:12]}`, gate rules `{d.gate_fingerprint[:12]}`.")

    blocking = [p for p in d.preconditions if not p.passed]
    if blocking:
        out += ["", "## Preconditions failed (the comparison is not valid)", ""]
        out += [f"- {p.name}: {p.detail}" for p in blocking]

    out += ["", "## Decision by dimension", "",
            "| Dimension | Rule | Category (n) | Production | Candidate | Change [95% CI] | Requirement | Result |",
            "|---|---|---|---|---|---|---|---|"]
    for dimension, rules in _grouped(d):
        for r in rules:
            sev = "" if r.severity == "mandatory" else " (advisory)"
            out.append(f"| {dimension} | {r.rule_id}{sev} | {r.category} ({r.n}) | {fmt(r.metric, r.baseline)} | "
                       f"{fmt(r.metric, r.candidate)} | {_change(r)} | {_requirement(r)} | **{_result(r)}** |")

    failed = d.mandatory_failures + d.advisory_failures
    if failed:
        out += ["", "## Why", ""]
        for r in failed:
            tag = "blocking" if r.severity == "mandatory" else "needs review"
            out.append(f"- **{r.rule_id}** ({tag}, {r.category}): " + "; ".join(r.reasons))

    if baseline:
        out += ["", "## Results by category", "",
                "| Category | Metric | Source | Production | Candidate | Change | 95% CI | Reading |",
                "|---|---|---|---|---|---|---|---|"]
        for category, names in CATEGORY_METRICS.items():
            n = len(select(candidate.records, category))
            for name in names:
                m = METRICS[name]
                cmp = paired_bootstrap(m, baseline.records, candidate.records, category, gate.bootstrap_samples,
                                       gate.confidence)
                ci = f"[{fmt_delta(name, cmp.ci_low)}, {fmt_delta(name, cmp.ci_high)}]" if cmp.ci_low is not None else "n/a"
                out.append(f"| {category} ({n}) | {name} | {m.source} | {fmt(name, cmp.baseline)} | "
                           f"{fmt(name, cmp.candidate)} | {fmt_delta(name, cmp.delta)} | {ci} | {cmp.verdict} |")
        out += ["", "\"No clear difference\" means the 95% interval includes zero: on these cases the data cannot "
                "tell the versions apart on that metric, whatever the point estimates say."]

        if failed:
            out += ["", "## Cases behind the regressions", ""]
            for r in failed:
                cases = regression_cases(r, candidate, baseline, questions)
                total = len(regression_cases(r, candidate, baseline, questions, limit=0))
                out += [f"### {r.rule_id} ({r.category}): {total} cases got worse"
                        + (f", showing {len(cases)}" if total > len(cases) else ""), ""]
                if not cases:
                    out += ["No single case got worse on this metric: the change is spread across cases "
                            "(for example a shift in the latency distribution).", ""]
                    continue
                out += ["| Case | Question | What changed | Production answer | Candidate answer |",
                        "|---|---|---|---|---|"]
                for x in cases:
                    out.append(f"| `{x.case_id}` | {_cell(x.question)} | {_cell(x.what_changed)} | "
                               f"{_cell(x.production_answer)} | {_cell(x.candidate_answer)} |")
                out.append("")

    out += ["", "Thresholds apply to point estimates, the conservative choice for blocking regressions. "
            "The decision is stored with every rule result and the versions above, so it can be audited and "
            "reproduced. `evalops gate` never changes production; `evalops promote` only accepts a passing "
            "decision whose versions still match.", ""]
    return "\n".join(out)


def _cell(text: str | None, limit: int = 160) -> str:
    if not text:
        return "-"
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_text(d: GateDecision, candidate: RunData, baseline: RunData | None, questions: dict[str, str],
                *, next_step: str) -> str:
    """The terminal summary: decision, every rule grouped by dimension, reasons, worst cases, next step."""
    word = DECISION_WORD[d.decision]
    lines = [f"Release decision: {word}",
             f"  Candidate:   {d.candidate:<22} run {d.candidate_run_id}",
             f"  Production:  {(d.baseline or '(none, first release)'):<22} "
             + (f"run {d.baseline_run_id}" if d.baseline_run_id else ""),
             f"  Dataset:     {d.dataset_ref} ({len(candidate.records)} cases, hash {d.dataset_hash[:12]})"]
    for line in execution_lines(candidate.run.get("execution") or {}):
        lines.append(f"  {line}")
    for p in d.preconditions:
        if not p.passed:
            lines.append(f"  PRECONDITION FAILED: {p.name} ({p.detail})")
    width = max(len(r.rule_id) for r in d.rules) + 2
    for dimension, rules in _grouped(d):
        lines += ["", dimension]
        for r in rules:
            lines.append(f"  {r.rule_id:<{width}} {r.category:<18} {fmt(r.metric, r.baseline):>9} -> "
                         f"{fmt(r.metric, r.candidate):<9} {fmt_delta(r.metric, r.delta) if r.delta is not None else '':>11}"
                         f"   {_result(r)}" + ("" if r.severity == "mandatory" else " (advisory)"))
    failed = d.mandatory_failures + d.advisory_failures
    if failed:
        lines += ["", "Reasons"]
        for r in failed:
            lines.append(f"  - {r.rule_id} [{r.category}]: " + "; ".join(r.reasons))
            if baseline:
                for x in regression_cases(r, candidate, baseline, questions, limit=2):
                    lines.append(f"      e.g. {x.case_id}: {_cell(x.what_changed, 110)}")
    lines += ["", next_step]
    return "\n".join(lines)
