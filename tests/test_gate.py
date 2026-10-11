from __future__ import annotations

import pytest

from evalops.config import GateConfig
from evalops.engine import RunOptions
from evalops.decision_report import render_markdown, render_text
from evalops.gate import RunData, evaluate_gate
from evalops.metrics import METRICS, CaseRecord, paired_bootstrap
from evalops.release import run_data


def run(project, name, **options):
    ev = project.evaluator()
    result = ev.run(project.handle(name), RunOptions(quiet=True, **options))
    return run_data(ev, result.run_id)


def gate(project) -> GateConfig:
    return project.evaluator().gate


def test_identical_candidate_passes(project):
    project.write_app("good")
    base = run(project, "good")
    project.write_app("same-behaviour", prompt_version="p2")
    cand = run(project, "same-behaviour")
    decision = evaluate_gate(gate(project), cand, base)
    assert decision.decision == "pass"
    assert all(r.verdict in ("no clear difference", "not comparable") for r in decision.rules if r.delta == 0)


def test_category_regression_hidden_by_the_overall_average_is_caught(project):
    # Baseline gets 3 answerable questions wrong; the candidate fixes them but starts
    # answering unanswerable questions. Overall accuracy goes UP, unanswerable handling collapses.
    project.write_app("base", {"default": "good", "cases": {"c01": "wrong", "c02": "wrong", "c03": "wrong"}})
    # c17 and c18 are unanswerable; "wrong" makes the app answer them instead of declining.
    project.write_app("cand", {"default": "good", "cases": {"c17": "wrong", "c18": "wrong"}})
    base, cand = run(project, "base"), run(project, "cand")
    decision = evaluate_gate(gate(project), cand, base)
    overall_before = sum(r.abstention_correct and (r.verdict != "incorrect") for r in base.records)
    overall_after = sum(r.abstention_correct and (r.verdict != "incorrect") for r in cand.records)
    assert overall_after > overall_before
    assert decision.decision == "fail"
    assert [r.rule_id for r in decision.mandatory_failures] == ["declines"]


def test_advisory_failure_gives_pass_with_warnings(project):
    project.write_app("cheap")
    project.write_app("pricey", latency={"default": 1.0})
    base = run(project, "cheap")
    cand = run(project, "pricey")
    # Make the candidate 50% more expensive per request.
    for r in cand.records:
        r.cost_usd = (r.cost_usd or 0) * 1.5
    decision = evaluate_gate(gate(project), cand, base)
    assert decision.decision == "pass_with_warnings"
    assert [r.rule_id for r in decision.advisory_failures] == ["cost"]
    assert decision.promotable


def test_preconditions_block_incomparable_runs(project):
    project.write_app("good")
    base, cand = run(project, "good"), run(project, "good", mode="live", record=False)
    cand.run["scoring_fingerprint"] = "0" * 64
    decision = evaluate_gate(gate(project), cand, base)
    assert decision.decision == "fail"
    assert [c.name for c in decision.preconditions if not c.passed] == ["same scoring"]

    cand.run["scoring_fingerprint"] = base.run["scoring_fingerprint"]
    cand.run["dataset_hash"] = "f" * 64
    assert evaluate_gate(gate(project), cand, base).decision == "fail"

    cand.run["dataset_hash"] = base.run["dataset_hash"]
    cand.run["status"] = "failed"
    assert evaluate_gate(gate(project), cand, base).decision == "fail"


def test_too_few_cases_fails_instead_of_passing_vacuously(project):
    project.write_app("good")
    data = run(project, "good")
    config = GateConfig.model_validate({"version": 1, "rules": [
        {"id": "declines", "metric": "correct_abstention_rate", "category": "unanswerable", "min": 0.5,
         "min_cases": 30}]})
    decision = evaluate_gate(config, data, None)
    assert decision.decision == "fail"
    assert "cannot be verified" in decision.rules[0].reasons[0]


def test_first_release_uses_absolute_rules_only(project):
    project.write_app("good")
    decision = evaluate_gate(gate(project), run(project, "good"), None)
    assert decision.decision == "pass"
    assert all(r.baseline is None and r.verdict == "no baseline" for r in decision.rules)


def test_report_explains_the_decision_and_the_cases_behind_it(project):
    project.write_app("base")
    project.write_app("worse", {"default": "good", "cases": {f"c{i:02d}": "no_citation" for i in range(1, 9)}})
    base, cand = run(project, "base"), run(project, "worse")
    decision = evaluate_gate(gate(project), cand, base)
    questions = {f"c{i:02d}": f"Question number {i}?" for i in range(1, 21)}
    md = render_markdown(decision, cand, base, gate(project), questions)
    assert "# Release decision: BLOCKED" in md
    assert "**citations** (blocking, citation_required)" in md
    assert "## Cases behind the regressions" in md and "citations 1 -> 0" in md and "Question number 1?" in md
    assert "| Citations and grounding | citations |" in md
    text = render_text(decision, cand, base, questions, next_step="Promotion: BLOCKED.")
    assert text.startswith("Release decision: BLOCKED")
    assert "citations" in text and "FAIL" in text and "e.g. c0" in text


def record(i: int, correct: bool, latency: float = 1.0) -> CaseRecord:
    return CaseRecord(case_id=f"c{i}", categories=["answerable"], answerable=True, status="ok", answered=True,
                      judge_correctness_status="ok", verdict="correct" if correct else "incorrect",
                      correctness=1.0 if correct else 0.0, original_latency_s=latency, abstention_correct=True)


def test_paired_bootstrap_separates_noise_from_real_change():
    metric = METRICS["answer_correctness"]
    base = [record(i, i % 10 != 0) for i in range(200)]                    # 90%
    small = [record(i, i % 10 != 0 or i == 0) for i in range(200)]         # 90.5%: one case better
    big = [record(i, i % 10 != 0 and i % 3 != 0) for i in range(200)]      # about 60%
    noise = paired_bootstrap(metric, base, small, samples=1000)
    assert noise.delta == pytest.approx(0.005) and noise.verdict == "no clear difference"
    real = paired_bootstrap(metric, base, big, samples=1000)
    assert real.verdict == "regressed" and real.ci_high < 0
    assert paired_bootstrap(metric, base, big, samples=1000).ci_low == real.ci_low  # seeded, reproducible


def test_percentile_metrics_bootstrap_too():
    metric = METRICS["latency_p95_s"]
    base = [record(i, True, latency=1.0 + (i % 20) / 10) for i in range(200)]
    slow = [record(i, True, latency=3.0 + (i % 20) / 10) for i in range(200)]
    cmp = paired_bootstrap(metric, base, slow, samples=500)
    assert cmp.delta == pytest.approx(2.0) and cmp.verdict == "regressed"
