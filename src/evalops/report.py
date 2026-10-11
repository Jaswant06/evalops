"""Markdown reports for evaluation runs."""

from __future__ import annotations

from typing import Any

from .config import PricingConfig
from .metrics import METRICS, REPORT_CATEGORIES, CaseRecord

SECTIONS = {
    "Answer quality": ["answer_correctness", "answer_quality", "reference_f1"],
    "Abstention": ["abstention_accuracy", "correct_abstention_rate", "inappropriate_answer_rate",
                   "false_abstention_rate"],
    "Citations and grounding": ["citation_missing_rate", "citation_invalid_rate", "citation_support_rate",
                                "unsupported_claim_rate", "contradiction_rate"],
    "Retrieval": ["retrieval_recall", "retrieval_precision"],
    "Application reliability": ["app_failure_rate", "app_timeout_rate", "app_malformed_rate", "app_error_rate"],
    "Evaluation reliability": ["judge_unevaluable_rate"],
    "Performance and cost": ["latency_p50_s", "latency_p95_s", "latency_p99_s", "ttft_p50_s",
                             "requests_per_minute", "tokens_per_second", "input_tokens_mean",
                             "output_tokens_mean", "est_cost_per_1k_requests_usd", "est_cost_per_success_usd"],
}


def fmt(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    unit = METRICS[metric].unit if metric in METRICS else "count"
    if unit == "ratio":
        return f"{value * 100:.1f}%"
    if unit == "seconds":
        return f"{value:.2f} s"
    if unit == "usd":
        return f"${value:.4f}" if abs(value) >= 0.01 else f"${value:.6f}"
    if unit in ("tokens", "count"):
        return f"{value:.0f}"
    return f"{value:.2f}"


def fmt_delta(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    unit = METRICS[metric].unit
    sign = "+" if value >= 0 else "-"
    if unit == "ratio":
        return f"{sign}{abs(value) * 100:.1f} pts"
    if unit == "seconds":
        return f"{sign}{abs(value):.2f} s"
    if unit == "usd":
        return f"{sign}${abs(value):.4f}" if abs(value) >= 0.01 else f"{sign}${abs(value):.6f}"
    return f"{sign}{abs(value):.2f}"


MODE_HEADLINE = {
    "replay": "REPLAY. Recorded answers and judge verdicts were re-scored by the current code; no model was called.",
    "live": "LIVE. The app's model answered every question in this run.",
    "rescore": "RESCORE. Recorded answers were scored with the current scoring config.",
    "auto": "AUTO. Recorded outputs were reused where they existed; the rest ran live.",
}


def _day(ts: str | None) -> str:
    return ts.replace("T", " ")[:16] + " UTC" if ts else "unknown"


def execution_lines(execution: dict) -> list[str]:
    """Plain statements of where a run's numbers came from."""
    if not execution:
        return []
    app, judge = execution["app"], execution["judge"]
    lines = [f"Mode: {MODE_HEADLINE.get(execution['mode'], execution['mode'])}"]
    answers = f"Answers: {app['live']} generated live in this run, {app['replayed']} replayed"
    if app["replayed"]:
        answers += f" (recorded {_day(app['recorded_from'])} to {_day(app['recorded_to'])})"
    lines.append(answers + ".")
    lines.append(f"Judge verdicts: {judge['live']} generated live, {judge['replayed']} replayed"
                 + (f", {judge['rejected']} recordings rejected for a version mismatch" if judge["rejected"] else "")
                 + ".")
    if execution.get("policy"):
        lines.append(f"Policy: {execution['policy']}")
    failures = execution.get("failures") or {}
    if failures:
        lines.append(f"Failures: {failures.get('application', 0)} application (system under test), "
                     f"{failures.get('scoring', 0)} scoring (judge could not score; marked unevaluable). "
                     f"Infrastructure failures stop the run instead of being counted.")
    source = execution["latency_source"]
    if source == "recorded":
        lines.append(f"Latency and throughput: measured when the answers were recorded ({_day(app['recorded_from'])}"
                     f" to {_day(app['recorded_to'])}), not in this run. Replaying is not a latency measurement.")
    elif source == "mixed":
        lines.append(f"Latency and throughput: mixed. {app['live']} answers were timed in this run and "
                     f"{app['replayed']} when they were recorded; compare with care.")
    else:
        lines.append("Latency and throughput: measured in this run.")
    if execution.get("evaluation_duration_s") is not None:
        lines.append(f"Evaluation time: {execution['evaluation_duration_s']:.1f} s spent obtaining answers in this "
                     f"run (separate from the application's latency above).")
    return lines


def versions_table(run: dict[str, Any]) -> list[str]:
    comp = run["app_version"]["components"]
    model = comp.get("model", {})
    prompt = comp.get("prompt", {})
    rows = [
        ("Dataset", f"`{run['dataset_ref']}` hash `{run['dataset_hash'][:12]}`"),
        ("App version", f"{run['app']} {run['app_version_label']}, fingerprint `{run['app_fingerprint'][:12]}`"),
        ("Model", f"`{model.get('name')}` digest `{model.get('digest')}`" if model else "n/a"),
        ("Prompt", f"`{prompt.get('path')}` sha256 `{str(prompt.get('sha256'))[:12]}`" if prompt else "n/a"),
        ("Generation options", f"`{comp.get('options')}`"),
        ("Retrieval", f"`{comp.get('retrieval')}`"),
        ("Scoring", f"fingerprint `{run['scoring_fingerprint'][:12]}`"),
        ("Run fingerprint", f"`{run['run_fingerprint'][:12]}`"),
        ("Code", f"evalops {run['evalops_version']}, {comp.get('app_code', '')}, git `{(run.get('git_sha') or 'unknown')[:10]}`"),
    ]
    return ["| Input | Version |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]]


def metrics_table(metrics: dict, names: list[str]) -> list[str]:
    cats = [c for c in REPORT_CATEGORIES if c in metrics]
    head = "| Metric | Source | " + " | ".join(f"{c} (n={int(metrics[c]['cases']['value'])})" for c in cats) + " |"
    lines = [head, "|---|---|" + "---|" * len(cats)]
    for name in names:
        cells = []
        for c in cats:
            v = metrics[c][name]
            cells.append(f"{fmt(name, v['value'])} ({v['n']})" if v["value"] is not None else "-")
        lines.append(f"| {name} | {METRICS[name].source} | " + " | ".join(cells) + " |")
    return lines


def render_run_report(run: dict, records: list[CaseRecord], metrics: dict, dataset, pricing: PricingConfig) -> str:
    cases = {c.case_id: c for c in dataset.cases}
    n = len(records)
    out = [
        f"# Evaluation report: {run['app']} {run['app_version_label']}",
        "",
        f"Run `{run['run_id']}`, dataset `{run['dataset_ref']}`, {n} cases.",
        "",
        "## Versions",
        "",
        *versions_table(run),
        "",
        "## Execution",
        "",
        *[f"- {line}" for line in execution_lines(run.get("execution") or {})],
    ]
    if any(r.injected_fault for r in records):
        injected = sum(bool(r.injected_fault) for r in records)
        out.append(f"- Fault injection was enabled: {injected} responses were replaced by injected failures.")
    if pricing.illustrative:
        out.append(f"- Cost uses illustrative rates ({pricing.source}, {pricing.as_of}), not a real bill.")
    out.append("")
    out.append("Numbers in parentheses are the number of cases each metric is based on. Source: `judged` by the "
               "LLM judge, `deterministic` (parsers, labels, set lookups), `measured` (timings and token counts), "
               "`estimated` (cost from a price sheet).")
    for title, names in SECTIONS.items():
        out += ["", f"## {title}", "", *metrics_table(metrics, names)]

    failures = [r for r in records if r.status != "ok"]
    wrong = [r for r in records if r.verdict == "incorrect"]
    inappropriate = [r for r in records if not r.answerable and r.answered]
    unsupported = [r for r in records if r.claims_total and r.claims_supported_document is not None
                   and r.claims_supported_document < r.claims_total]
    judge_errors = [r for r in records if r.judge_error]

    out += ["", "## Where it failed", ""]
    out.append(f"{len(failures)} failed requests, {len(wrong)} incorrect answers, {len(inappropriate)} answers "
               f"to unanswerable questions, {len(unsupported)} answers with unsupported claims, "
               f"{len(judge_errors)} judge errors. Every case is in `cases.jsonl` and the results database.")

    def sample(title: str, rows: list[CaseRecord], extra) -> None:
        if not rows:
            return
        out.extend(["", f"### {title} (first {min(5, len(rows))} of {len(rows)})", ""])
        for r in rows[:5]:
            q = cases[r.case_id].input.question if r.case_id in cases else ""
            out.append(f"- `{r.case_id}` {q}")
            out.append(f"  - {extra(r)}")

    sample("Failed requests", failures, lambda r: f"{r.status}: {r.error or 'output failed the schema parser'}")
    sample("Incorrect answers", wrong, lambda r: f"answer: {r.answer!r}; reference: "
                                                 f"{cases[r.case_id].expected.reference!r}")
    sample("Answered although the document has no answer", inappropriate, lambda r: f"answer: {r.answer!r}")
    sample("Unsupported claims", unsupported, lambda r: "; ".join(
        c["claim"] for c in (r.claims or []) if not c["supported_by_document"]))
    sample("Judge errors", judge_errors, lambda r: r.judge_error)
    out.append("")
    return "\n".join(out)
