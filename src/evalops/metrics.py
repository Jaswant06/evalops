"""Metric definitions, per-category aggregation and paired statistics.

Every metric is defined once, as per-case arrays:
  ratio metrics       value = sum(numerator) / sum(denominator); a case with
                      denominator 0 is outside the metric (for example
                      citation support for an abstention)
  percentile metrics  a value per case (NaN when not applicable), then a percentile

That one definition gives the point estimate, the per-category breakdown, and
a paired bootstrap: production and candidate are resampled with the *same*
case indices, so the confidence interval is for the difference on identical
questions, not for two independent averages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

Direction = Literal["higher", "lower", "none"]
# Where a number comes from. Reports show it next to every metric, so nobody
# mistakes an estimate for a measurement or a judge's opinion for a fact.
Source = Literal["judged", "deterministic", "measured", "estimated", "bookkeeping"]


class CaseRecord(BaseModel):
    """The scored result for one test case: checks + judge + operational data."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    categories: list[str]
    answerable: bool
    status: str
    schema_valid: bool | None = None
    answered: bool = False
    abstained: bool = False
    abstention_correct: bool = False
    # Where the answer came from in this run, and when it was generated.
    execution_mode: Literal["live", "replayed"] = "live"
    recorded_at: str | None = None
    attempts: int = 1
    injected_fault: str | None = None
    # Latency of the application when the answer was generated. For a replayed
    # answer this is the original live measurement, not something measured now.
    original_latency_s: float | None = None
    original_ttft_s: float | None = None
    # Wall time this run spent obtaining the answer (about zero when replayed).
    evaluation_duration_s: float | None = None
    # application: the system under test failed (timeout, error, malformed output)
    # scoring: the answer could not be scored (judge output unusable after retries)
    failure_class: Literal["application", "scoring"] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    n_citations: int = 0
    citations_invalid: int | None = None
    citation_missing: bool | None = None
    retrieval_recall: float | None = None
    retrieval_precision: float | None = None
    evidence_coverage: float | None = None
    reference_f1: float | None = None
    judge_correctness_status: Literal["ok", "unevaluable", "skipped"] = "skipped"
    judge_claims_status: Literal["ok", "unevaluable", "skipped"] = "skipped"
    judge_error: str | None = None
    # One structured, validated record per judge evaluation (see judge.structured_judgment).
    judgments: list[dict] = []
    verdict: str | None = None
    correctness: float | None = None
    quality: int | None = None
    judge_reasoning: str | None = None
    claims_total: int | None = None
    claims_supported_cited: int | None = None
    claims_supported_document: int | None = None
    claims_contradicted: int | None = None
    claims: list[dict] | None = None
    answer: str | None = None
    error: str | None = None


Arrays = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class MetricDef:
    name: str
    direction: Direction
    description: str
    unit: str = "ratio"           # ratio | seconds | tokens | usd | score | per_minute
    ratio: Callable[[list[CaseRecord]], Arrays] | None = None
    values: Callable[[list[CaseRecord]], np.ndarray] | None = None
    percentile: float | None = None
    scale: float = 1.0
    source: Source = "deterministic"


def _ratio(fn: Callable[[CaseRecord], tuple[float, float]]) -> Callable[[list[CaseRecord]], Arrays]:
    def arrays(records: list[CaseRecord]) -> Arrays:
        pairs = [fn(r) for r in records]
        return np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)
    return arrays


def _values(fn: Callable[[CaseRecord], float | None]) -> Callable[[list[CaseRecord]], np.ndarray]:
    def arrays(records: list[CaseRecord]) -> np.ndarray:
        return np.array([np.nan if (v := fn(r)) is None else v for r in records], float)
    return arrays


def _ok(r: CaseRecord) -> bool:
    return r.status == "ok"


def _claims_judged(r: CaseRecord) -> bool:
    return r.answered and r.judge_claims_status == "ok" and bool(r.claims_total)


def _correctness(r: CaseRecord) -> tuple[float, float]:
    if not r.answerable:
        return 0, 0
    if not r.answered:               # failures and wrong abstentions count as incorrect
        return 0, 1
    if r.judge_correctness_status != "ok":
        return 0, 0                  # unevaluable: excluded, and counted in judge_unevaluable_rate
    return r.correctness, 1


def _success(r: CaseRecord) -> bool:
    """A successful task: a correct answer, or a correct refusal of an unanswerable question."""
    if not _ok(r) or not r.abstention_correct:
        return False
    return (not r.answerable) or r.verdict == "correct"


def _judge_error(r: CaseRecord) -> tuple[float, float]:
    required = [r.judge_claims_status] if r.answered else []
    if r.answered and r.answerable:
        required.append(r.judge_correctness_status)
    if not required:
        return 0, 0
    return float(any(s == "unevaluable" for s in required)), 1


METRIC_LIST = [
    # --- answer quality ------------------------------------------------------------
    MetricDef("answer_correctness", "higher",
              "Answerable questions judged correct (partial = 0.5). Failures and wrong abstentions count as 0.",
              ratio=_ratio(_correctness), source="judged"),
    MetricDef("answer_quality", "higher", "Judge's 1-5 quality score, answered answerable questions.", unit="score",
              ratio=_ratio(lambda r: (r.quality, 1) if r.answerable and r.answered and r.quality else (0, 0)), source="judged"),
    MetricDef("reference_f1", "higher", "Token F1 against the reference answer (no judge involved).",
              ratio=_ratio(lambda r: (r.reference_f1, 1) if r.reference_f1 is not None else (0, 0))),
    # --- abstention ---------------------------------------------------------------------
    MetricDef("abstention_accuracy", "higher", "Declined exactly when the question is unanswerable.",
              ratio=_ratio(lambda r: (float(r.abstention_correct), 1))),
    MetricDef("correct_abstention_rate", "higher", "Unanswerable questions the app declined.",
              ratio=_ratio(lambda r: (float(r.abstained), 1) if not r.answerable else (0, 0))),
    MetricDef("inappropriate_answer_rate", "lower", "Unanswerable questions the app answered anyway.",
              ratio=_ratio(lambda r: (float(r.answered), 1) if not r.answerable else (0, 0))),
    MetricDef("false_abstention_rate", "lower", "Answerable questions the app wrongly declined.",
              ratio=_ratio(lambda r: (float(r.abstained), 1) if r.answerable else (0, 0))),
    # --- citations: validity (deterministic) and support (judge) ------------------------
    MetricDef("citation_missing_rate", "lower", "Answers to citation-required questions with no citation.",
              ratio=_ratio(lambda r: (float(r.n_citations == 0), 1)
                           if r.answered and "citation_required" in r.categories else (0, 0))),
    MetricDef("citation_invalid_rate", "lower",
              "Answers with a citation that does not resolve to a retrieved passage of the right document.",
              ratio=_ratio(lambda r: (float((r.citations_invalid or 0) > 0), 1)
                           if r.answered and r.n_citations else (0, 0))),
    MetricDef("citation_support_rate", "higher", "Answer claims supported by the passages the answer cites.",
              ratio=_ratio(lambda r: (r.claims_supported_cited, r.claims_total) if _claims_judged(r) else (0, 0)), source="judged"),
    MetricDef("unsupported_claim_rate", "lower", "Answer claims not supported anywhere in the document.",
              ratio=_ratio(lambda r: (r.claims_total - r.claims_supported_document, r.claims_total)
                           if _claims_judged(r) else (0, 0)), source="judged"),
    MetricDef("contradiction_rate", "lower", "Answers with a claim the document contradicts.",
              ratio=_ratio(lambda r: (float(r.claims_contradicted > 0), 1) if _claims_judged(r) else (0, 0)), source="judged"),
    # --- retrieval --------------------------------------------------------------------
    MetricDef("retrieval_recall", "higher", "Retrieved passages contain the gold evidence passage.",
              ratio=_ratio(lambda r: (r.retrieval_recall, 1) if r.retrieval_recall is not None else (0, 0))),
    MetricDef("retrieval_precision", "higher", "Share of retrieved passages that contain the evidence.",
              ratio=_ratio(lambda r: (r.retrieval_precision, 1) if r.retrieval_precision is not None else (0, 0))),
    # --- reliability ------------------------------------------------------------------
    MetricDef("app_failure_rate", "lower",
              "Application failures: requests to the system under test that timed out, errored or returned "
              "malformed output.",
              ratio=_ratio(lambda r: (float(not _ok(r)), 1))),
    MetricDef("app_timeout_rate", "lower", "Requests that timed out.", ratio=_ratio(lambda r: (float(r.status == "timeout"), 1))),
    MetricDef("app_malformed_rate", "lower", "Responses whose output failed the schema parser.",
              ratio=_ratio(lambda r: (float(r.status == "malformed"), 1))),
    MetricDef("app_error_rate", "lower", "Requests that failed with an error.", ratio=_ratio(lambda r: (float(r.status == "error"), 1))),
    MetricDef("judge_unevaluable_rate", "lower",
              "Scoring failures: answers the judge could not score after retries (marked unevaluable, never "
              "given a score).",
              ratio=_ratio(_judge_error)),
    # --- performance and cost ------------------------------------------------------------
    MetricDef("latency_p50_s", "lower", "Median end-to-end latency of successful requests.", unit="seconds",
              values=_values(lambda r: r.original_latency_s if _ok(r) else None), percentile=50, source="measured"),
    MetricDef("latency_p95_s", "lower", "95th percentile latency of successful requests.", unit="seconds",
              values=_values(lambda r: r.original_latency_s if _ok(r) else None), percentile=95, source="measured"),
    MetricDef("latency_p99_s", "lower", "99th percentile latency of successful requests.", unit="seconds",
              values=_values(lambda r: r.original_latency_s if _ok(r) else None), percentile=99, source="measured"),
    MetricDef("ttft_p50_s", "lower", "Median time to first token (model load + prompt processing).", unit="seconds",
              values=_values(lambda r: r.original_ttft_s if _ok(r) else None), percentile=50, source="measured"),
    MetricDef("input_tokens_mean", "none", "Prompt tokens per request.", unit="tokens",
              ratio=_ratio(lambda r: (r.input_tokens, 1) if _ok(r) and r.input_tokens is not None else (0, 0)), source="measured"),
    MetricDef("output_tokens_mean", "none", "Generated tokens per request.", unit="tokens",
              ratio=_ratio(lambda r: (r.output_tokens, 1) if _ok(r) and r.output_tokens is not None else (0, 0)), source="measured"),
    MetricDef("tokens_per_second", "higher", "Generated tokens per second of request time.", unit="tokens/s",
              ratio=_ratio(lambda r: (r.output_tokens, r.original_latency_s)
                           if _ok(r) and r.output_tokens is not None and r.original_latency_s else (0, 0)), source="measured"),
    MetricDef("requests_per_minute", "higher",
              "Sequential throughput: 60 / mean latency of successful requests.", unit="per_minute",
              ratio=_ratio(lambda r: (1, r.original_latency_s) if _ok(r) and r.original_latency_s else (0, 0)), scale=60.0, source="measured"),
    MetricDef("est_cost_per_1k_requests_usd", "lower",
              "Estimated cost per 1,000 requests: measured tokens x the price sheet.", unit="usd",
              ratio=_ratio(lambda r: (r.cost_usd, 1) if r.cost_usd is not None else (0, 0)), scale=1000.0,
              source="estimated"),
    MetricDef("est_cost_per_success_usd", "lower",
              "Estimated cost per successful task: total cost / cases answered correctly or correctly declined.",
              unit="usd", ratio=_ratio(lambda r: (r.cost_usd, float(_success(r))) if r.cost_usd is not None else (0, 0)),
              source="estimated"),
    MetricDef("replayed_share", "none", "Responses replayed from recordings instead of live inference.",
              ratio=_ratio(lambda r: (float(r.execution_mode == "replayed"), 1)), source="bookkeeping"),
]
METRICS: dict[str, MetricDef] = {m.name: m for m in METRIC_LIST}

# Categories every report breaks down; "all" is the whole split.
REPORT_CATEGORIES = ["all", "answerable", "unanswerable", "citation_required"]


def select(records: list[CaseRecord], category: str) -> list[CaseRecord]:
    return records if category == "all" else [r for r in records if category in r.categories]


def compute(metric: MetricDef, records: list[CaseRecord]) -> tuple[float | None, int]:
    """Point estimate and the number of cases it is based on."""
    if metric.ratio is not None:
        num, den = metric.ratio(records)
        n = int((den > 0).sum())
        total = den.sum()
        return (float(num.sum() / total * metric.scale) if total > 0 else None), n
    values = metric.values(records)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return None, 0
    return float(np.percentile(valid, metric.percentile)), int(valid.size)


def compute_all(records: list[CaseRecord], categories: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """{category: {metric: {"value": v, "n": n}}}"""
    out: dict[str, dict[str, dict]] = {}
    for category in categories or REPORT_CATEGORIES:
        subset = select(records, category)
        out[category] = {}
        for m in METRIC_LIST:
            value, n = compute(m, subset)
            out[category][m.name] = {"value": value, "n": n}
        out[category]["cases"] = {"value": float(len(subset)), "n": len(subset)}
    return out


@dataclass
class PairedComparison:
    metric: str
    category: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    ci_low: float | None
    ci_high: float | None
    n_pairs: int
    verdict: str   # improved | regressed | no clear difference | not comparable


def paired_bootstrap(metric: MetricDef, baseline: list[CaseRecord], candidate: list[CaseRecord],
                     category: str = "all", samples: int = 2000, confidence: float = 0.95,
                     seed: int = 0) -> PairedComparison:
    """Bootstrap CI for (candidate - baseline) on the same cases, resampled together."""
    b_by_id = {r.case_id: r for r in select(baseline, category)}
    c_by_id = {r.case_id: r for r in select(candidate, category)}
    ids = sorted(b_by_id.keys() & c_by_id.keys())
    b_recs, c_recs = [b_by_id[i] for i in ids], [c_by_id[i] for i in ids]
    b_val, _ = compute(metric, b_recs)
    c_val, _ = compute(metric, c_recs)
    if not ids or b_val is None or c_val is None:
        return PairedComparison(metric.name, category, b_val, c_val, None, None, None, len(ids), "not comparable")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(samples, len(ids)))

    def resampled(recs: list[CaseRecord]) -> np.ndarray:
        if metric.ratio is not None:
            num, den = metric.ratio(recs)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num[idx].sum(1) / den[idx].sum(1) * metric.scale
        return np.nanpercentile(metric.values(recs)[idx], metric.percentile, axis=1)

    with np.errstate(invalid="ignore"):
        deltas = resampled(c_recs) - resampled(b_recs)
    deltas = deltas[~np.isnan(deltas)]
    alpha = (1 - confidence) / 2
    low, high = (float(np.quantile(deltas, alpha)), float(np.quantile(deltas, 1 - alpha))) if deltas.size else (None, None)

    delta = c_val - b_val
    verdict = "no clear difference"
    if low is not None and metric.direction != "none" and (low > 0 or high < 0):
        better = (low > 0) == (metric.direction == "higher")
        verdict = "improved" if better else "regressed"
    return PairedComparison(metric.name, category, b_val, c_val, delta, low, high, len(ids), verdict)
