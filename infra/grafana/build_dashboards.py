"""Generate the Grafana dashboards as JSON.

Dashboards are written in Python so their SQL and PromQL are readable and
reviewable; Grafana only sees the generated JSON. Re-run after editing:

    python infra/grafana/build_dashboards.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
PG = {"type": "grafana-postgresql-datasource", "uid": "evalops-pg"}
PROM = {"type": "prometheus", "uid": "prometheus"}

# Key metrics per category, shown side by side for production and candidate.
KEY_METRICS = [
    ("answerable", "answer_correctness", "Correct answers"),
    ("answerable", "false_abstention_rate", "Wrongly declined"),
    ("unanswerable", "correct_abstention_rate", "Declined unanswerable"),
    ("citation_required", "citation_missing_rate", "Missing citations"),
    ("citation_required", "citation_invalid_rate", "Invalid citations"),
    ("citation_required", "citation_support_rate", "Claims supported by citations"),
    ("all", "unsupported_claim_rate", "Unsupported claims"),
    ("answerable", "retrieval_recall", "Retrieval recall"),
    ("all", "app_failure_rate", "Application failures"),
    ("all", "judge_unevaluable_rate", "Unevaluable (judge)"),
]
LATENCY = [("latency_p50_s", "p50"), ("latency_p95_s", "p95"), ("latency_p99_s", "p99")]


def values_list(pairs) -> str:
    return ", ".join(f"('{c}', '{m}', '{label}', {i})" for i, (c, m, label) in enumerate(pairs))


def sql_panel(kind: str, title: str, sql: str, grid: tuple[int, int, int, int], **extra) -> dict:
    x, y, w, h = grid
    panel = {
        "type": kind, "title": title, "datasource": PG, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "datasource": PG, "editorMode": "code", "format": "table",
                     "rawQuery": True, "rawSql": sql.strip()}],
    }
    panel.update(extra)
    return panel


def prom_panel(title: str, exprs: list[tuple[str, str]], grid, unit: str = "short") -> dict:
    x, y, w, h = grid
    return {
        "type": "timeseries", "title": title, "datasource": PROM, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [{"refId": chr(65 + i), "datasource": PROM, "expr": e, "legendFormat": legend}
                    for i, (e, legend) in enumerate(exprs)],
    }


def run_variable(name: str, label: str, order_production_first: bool) -> dict:
    order = ("(rel.run_id IS NOT NULL) DESC, rel.promoted_at DESC NULLS LAST, r.created_at DESC"
             if order_production_first else "r.created_at DESC")
    sql = (
        "SELECT r.run_id AS __value, "
        "r.app_version_label || CASE WHEN rel.run_id IS NOT NULL THEN ' (released)' ELSE '' END "
        "|| ' | ' || r.dataset_ref || ' | ' || to_char(r.created_at, 'YYYY-MM-DD HH24:MI') || ' | ' || r.mode "
        "|| ' (latency ' || COALESCE(r.execution->>'latency_source', '?') || ')' AS __text "
        "FROM runs r LEFT JOIN releases rel ON rel.run_id = r.run_id "
        f"WHERE r.status = 'completed' ORDER BY {order}"
    )
    return {"name": name, "label": label, "type": "query", "datasource": PG, "query": sql, "definition": sql,
            "refresh": 1, "includeAll": False, "multi": False, "sort": 0}


def release_comparison() -> dict:
    decision_sql = """
SELECT CASE decision WHEN 'pass' THEN 'PASS' WHEN 'pass_with_warnings' THEN 'PASS (review)'
       ELSE 'BLOCKED' END AS decision
FROM gate_decisions
WHERE candidate_run_id = '$candidate' AND baseline_run_id = '$baseline'
ORDER BY created_at DESC LIMIT 1"""
    rules_sql = """
SELECT r->>'rule_id' AS rule, r->>'category' AS category, r->>'severity' AS severity,
       (r->>'baseline')::float AS production, (r->>'candidate')::float AS candidate,
       (r->>'delta')::float AS change, (r->>'ci_low')::float AS ci_low, (r->>'ci_high')::float AS ci_high,
       CASE WHEN (r->>'passed')::bool THEN 'pass' WHEN r->>'severity' = 'mandatory' THEN 'FAIL' ELSE 'warn' END AS result,
       array_to_string(ARRAY(SELECT jsonb_array_elements_text(r->'reasons')), '; ') AS why
FROM (SELECT rules FROM gate_decisions
      WHERE candidate_run_id = '$candidate' AND baseline_run_id = '$baseline'
      ORDER BY created_at DESC LIMIT 1) d, jsonb_array_elements(d.rules) r"""
    metrics_sql = f"""
WITH wanted(category, metric, label, ord) AS (VALUES {values_list(KEY_METRICS)})
SELECT w.label AS metric, w.category, b.value AS production, c.value AS candidate,
       c.value - b.value AS change, c.n AS cases
FROM wanted w
LEFT JOIN run_metrics b ON b.run_id = '$baseline' AND b.category = w.category AND b.metric = w.metric
LEFT JOIN run_metrics c ON c.run_id = '$candidate' AND c.category = w.category AND c.metric = w.metric
ORDER BY w.ord"""
    bars_sql = f"""
WITH wanted(category, metric, label, ord) AS (VALUES {values_list(KEY_METRICS[:8])})
SELECT w.label AS metric, b.value AS production, c.value AS candidate
FROM wanted w
LEFT JOIN run_metrics b ON b.run_id = '$baseline' AND b.category = w.category AND b.metric = w.metric
LEFT JOIN run_metrics c ON c.run_id = '$candidate' AND c.category = w.category AND c.metric = w.metric
ORDER BY w.ord"""
    latency_sql = f"""
WITH wanted(metric, label, ord) AS (VALUES {", ".join(f"('{m}', '{l}', {i})" for i, (m, l) in enumerate(LATENCY))})
SELECT w.label AS percentile, b.value AS production, c.value AS candidate
FROM wanted w
LEFT JOIN run_metrics b ON b.run_id = '$baseline' AND b.category = 'all' AND b.metric = w.metric
LEFT JOIN run_metrics c ON c.run_id = '$candidate' AND c.category = 'all' AND c.metric = w.metric
ORDER BY w.ord"""
    regressions_sql = """
SELECT c.case_id,
       (SELECT string_agg(x, ', ') FROM jsonb_array_elements_text(c.categories) x) AS categories,
       CASE
         WHEN b.record->>'status' = 'ok' AND c.record->>'status' <> 'ok' THEN 'request failed: ' || (c.record->>'status')
         WHEN (b.record->>'abstention_correct')::bool AND NOT (c.record->>'abstention_correct')::bool
           THEN CASE WHEN (c.record->>'answerable')::bool THEN 'wrongly declined' ELSE 'answered an unanswerable question' END
         WHEN b.record->>'verdict' = 'correct' AND c.record->>'verdict' <> 'correct' THEN 'answer now ' || (c.record->>'verdict')
         WHEN (c.record->>'n_citations')::int = 0 AND (b.record->>'n_citations')::int > 0 THEN 'citations dropped'
         ELSE 'claims lost citation support'
       END AS regression,
       left(b.record->>'answer', 120) AS production_answer,
       left(c.record->>'answer', 120) AS candidate_answer
FROM case_results c
JOIN case_results b ON b.case_id = c.case_id AND b.run_id = '$baseline'
WHERE c.run_id = '$candidate' AND (
     (b.record->>'status' = 'ok' AND c.record->>'status' <> 'ok')
  OR ((b.record->>'abstention_correct')::bool AND NOT (c.record->>'abstention_correct')::bool)
  OR (b.record->>'verdict' = 'correct' AND c.record->>'verdict' <> 'correct')
  OR ((c.record->>'n_citations')::int = 0 AND (b.record->>'n_citations')::int > 0 AND (c.record->>'answered')::bool)
  OR (COALESCE((c.record->>'claims_supported_cited')::int, 0) < COALESCE((c.record->>'claims_total')::int, 0)
      AND COALESCE((b.record->>'claims_supported_cited')::int, 0) = COALESCE((b.record->>'claims_total')::int, 0)
      AND (b.record->>'claims_total') IS NOT NULL))
ORDER BY c.case_id"""
    versions_sql = """
WITH latest AS (
  SELECT DISTINCT ON (app_version_label) run_id, app_version_label
  FROM runs WHERE status = 'completed' AND dataset_ref = 'policies/v1:test'
  ORDER BY app_version_label, created_at DESC)
SELECT l.app_version_label AS version,
  max(m.value) FILTER (WHERE m.category = 'answerable' AND m.metric = 'answer_correctness') AS correctness,
  max(m.value) FILTER (WHERE m.category = 'unanswerable' AND m.metric = 'correct_abstention_rate') AS declines_unanswerable,
  max(m.value) FILTER (WHERE m.category = 'citation_required' AND m.metric = 'citation_support_rate') AS citation_support,
  max(m.value) FILTER (WHERE m.category = 'all' AND m.metric = 'unsupported_claim_rate') AS unsupported_claims,
  max(m.value) FILTER (WHERE m.category = 'all' AND m.metric = 'app_failure_rate') AS app_failures,
  max(m.value) FILTER (WHERE m.category = 'all' AND m.metric = 'latency_p95_s') AS p95_latency_s,
  max(m.value) FILTER (WHERE m.category = 'all' AND m.metric = 'est_cost_per_1k_requests_usd') AS est_cost_per_1k_usd
FROM latest l JOIN run_metrics m ON m.run_id = l.run_id
GROUP BY l.app_version_label ORDER BY l.app_version_label"""
    decisions_sql = """
SELECT d.created_at AS time,
       CASE d.decision WHEN 'pass' THEN 'PASS' WHEN 'pass_with_warnings' THEN 'PASS (review)' ELSE 'BLOCKED' END AS decision,
       c.app_version_label AS candidate, COALESCE(b.app_version_label, '(first release)') AS production,
       d.mandatory_failures AS blocking_rules, d.advisory_failures AS review_rules, d.context
FROM gate_decisions d
JOIN runs c ON c.run_id = d.candidate_run_id
LEFT JOIN runs b ON b.run_id = d.baseline_run_id
ORDER BY d.created_at DESC LIMIT 25"""

    audit_sql = """
SELECT created_at AS time, event, app || ' ' || COALESCE(version, '') AS version, decision_id, actor,
       COALESCE(details->>'from_version', '') ||
       COALESCE((SELECT string_agg(p, '; ') FROM jsonb_array_elements_text(details->'problems') p), '') AS details
FROM audit_events WHERE event <> 'gate_decision'
ORDER BY created_at DESC LIMIT 25"""
    pct = {"unit": "percentunit", "decimals": 1}
    decision_colors = {"type": "value", "options": {
        "PASS": {"color": "green", "index": 0}, "PASS (review)": {"color": "yellow", "index": 1},
        "BLOCKED": {"color": "red", "index": 2}}}
    result_override = {"matcher": {"id": "byName", "options": "result"}, "properties": [
        {"id": "custom.cellOptions", "value": {"type": "color-background"}},
        {"id": "mappings", "value": [{"type": "value", "options": {
            "pass": {"color": "green"}, "FAIL": {"color": "red"}, "warn": {"color": "yellow"}}}]}]}
    decision_override = {"matcher": {"id": "byName", "options": "decision"}, "properties": [
        {"id": "custom.cellOptions", "value": {"type": "color-background"}},
        {"id": "mappings", "value": [decision_colors]}]}

    def pct_columns(*names):
        return [{"matcher": {"id": "byName", "options": n}, "properties": [{"id": "unit", "value": "percentunit"},
                                                                          {"id": "decimals", "value": 1}]}
                for n in names]

    panels = [
        sql_panel("stat", "Gate decision for this pair", decision_sql, (0, 0, 6, 6),
                  fieldConfig={"defaults": {"mappings": [decision_colors], "color": {"mode": "thresholds"},
                                            "thresholds": {"mode": "absolute", "steps": [{"color": "text", "value": None}]},
                                            "noValue": "not gated"}, "overrides": []},
                  options={"colorMode": "background", "graphMode": "none", "textMode": "value",
                           "reduceOptions": {"calcs": ["lastNotNull"], "fields": "/^decision$/", "values": False}}),
        sql_panel("table", "Release rules (latest decision for this pair)", rules_sql, (6, 0, 18, 10),
                  fieldConfig={"defaults": {"decimals": 3}, "overrides": [result_override]},
                  options={"showHeader": True, "cellHeight": "sm"}),
        sql_panel("table", "Key metrics: production vs candidate", metrics_sql, (0, 10, 12, 11),
                  fieldConfig={"defaults": {}, "overrides": pct_columns("production", "candidate", "change")},
                  options={"showHeader": True, "cellHeight": "sm"}),
        sql_panel("barchart", "Quality by category", bars_sql, (12, 10, 12, 11),
                  fieldConfig={"defaults": {**pct, "min": 0, "max": 1}, "overrides": []},
                  options={"xField": "metric", "orientation": "horizontal", "legend": {"showLegend": True},
                           "barWidth": 0.8, "groupWidth": 0.75, "showValue": "auto"}),
        sql_panel("barchart", "Original app latency (s)", latency_sql, (0, 21, 8, 9),
                  description="Latency measured when each answer was generated. A replay run reuses this "
                              "measurement and does not re-measure it; the run picker shows each run's latency "
                              "source (measured in this run, recorded, or mixed).",
                  fieldConfig={"defaults": {"unit": "s", "decimals": 2}, "overrides": []},
                  options={"xField": "percentile", "orientation": "vertical", "legend": {"showLegend": True},
                           "showValue": "auto"}),
        sql_panel("table", "Cases that regressed (right in production, wrong in candidate)", regressions_sql,
                  (8, 21, 16, 9), options={"showHeader": True, "cellHeight": "sm"}),
        sql_panel("table", "All app versions (latest completed run on the test split)", versions_sql,
                  (0, 30, 24, 8),
                  fieldConfig={"defaults": {}, "overrides": pct_columns(
                      "correctness", "declines_unanswerable", "citation_support", "unsupported_claims", "app_failures")},
                  options={"showHeader": True, "cellHeight": "sm"}),
        sql_panel("table", "Promotions, refusals and rollbacks (audit log)", audit_sql, (0, 46, 24, 8),
                  options={"showHeader": True, "cellHeight": "sm"}),
        sql_panel("table", "Gate decisions", decisions_sql, (0, 38, 24, 8),
                  fieldConfig={"defaults": {}, "overrides": [decision_override]},
                  options={"showHeader": True, "cellHeight": "sm"}),
    ]
    for i, p in enumerate(panels, start=1):
        p["id"] = i
    return {
        "uid": "evalops-release", "title": "EvalOps: evaluation results (PostgreSQL)", "schemaVersion": 41, "version": 1,
        "tags": ["evalops"], "timezone": "browser", "editable": False, "refresh": "",
        "time": {"from": "now-30d", "to": "now"},
        "description": "Evaluation results from PostgreSQL: production vs candidate on identical test cases, the "
                       "gate decision, every regression, and the audit trail. Operational metrics are on the "
                       "operations dashboard (Prometheus).",
        "templating": {"list": [run_variable("baseline", "Production run", True),
                                run_variable("candidate", "Candidate run", False)]},
        "panels": panels,
    }


def operations() -> dict:
    panels = [
        prom_panel("Evaluation latency by version (live inference only)", [
            ('histogram_quantile(0.95, sum by (version, le) (evalops_app_request_latency_seconds_bucket{source="live"}))',
             "p95 {{version}}"),
            ('histogram_quantile(0.5, sum by (version, le) (evalops_app_request_latency_seconds_bucket{source="live"}))',
             "p50 {{version}}")], (0, 0, 12, 8), unit="s"),
        prom_panel("Evaluation requests by outcome (last run per version)", [
            ("sum by (version, status, source) (evalops_app_requests_total)", "{{version}} {{status}} ({{source}})")],
            (12, 0, 12, 8)),
        prom_panel("Evaluation run duration", [("evalops_run_duration_seconds", "{{version}} {{dataset}}")],
                   (0, 8, 8, 8), unit="s"),
        prom_panel("Judge calls by outcome", [("sum by (version, status) (evalops_judge_calls_total)",
                                                "{{version}} {{status}}")], (8, 8, 8, 8)),
        prom_panel("Tokens used by the app under test", [("sum by (version, direction) (evalops_app_tokens_total)",
                                                          "{{version}} {{direction}}")], (16, 8, 8, 8)),
        prom_panel("DocQA service: requests per second", [
            ("sum by (version, status) (rate(docqa_requests_total[5m]))", "{{version}} {{status}}")],
            (0, 16, 8, 8), unit="reqps"),
        prom_panel("DocQA service: latency", [
            ("histogram_quantile(0.95, sum by (version, le) (rate(docqa_request_latency_seconds_bucket[5m])))",
             "p95 {{version}}"),
            ("histogram_quantile(0.5, sum by (version, le) (rate(docqa_request_latency_seconds_bucket[5m])))",
             "p50 {{version}}")], (8, 16, 8, 8), unit="s"),
        prom_panel("DocQA service: abstentions and failures", [
            ("sum by (version) (rate(docqa_abstentions_total[5m]))", "declined {{version}}"),
            ('sum by (version) (rate(docqa_requests_total{status!="ok"}[5m]))', "failed {{version}}")],
            (16, 16, 8, 8), unit="reqps"),
        prom_panel("Model memory during live evaluation runs", [("evalops_model_memory_bytes", "{{model}} ({{version}})")],
                   (0, 24, 8, 7), unit="bytes"),
        prom_panel("Share of answers generated live (rest replayed)", [("evalops_run_live_fraction", "{{version}} {{dataset}}")],
                   (8, 24, 8, 7), unit="percentunit"),
        prom_panel("EvalOps API requests", [("sum by (route) (rate(evalops_api_requests_total[5m]))", "{{route}}")],
                   (16, 24, 8, 7), unit="reqps"),
    ]
    for i, p in enumerate(panels, start=1):
        p["id"] = i
    return {
        "uid": "evalops-ops", "title": "EvalOps: operations (Prometheus)", "schemaVersion": 41, "version": 1,
        "tags": ["evalops"], "timezone": "browser", "editable": False, "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "description": "Operational metrics from Prometheus: request latency, throughput, errors and resource "
                       "usage of evaluation runs (pushed when a run finishes) and of the live DocQA service. "
                       "Evaluation scores are on the evaluation results dashboard (PostgreSQL).",
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, board in [("release-comparison", release_comparison()), ("operations", operations())]:
        (OUT / f"{name}.json").write_text(json.dumps(board, indent=2) + "\n")
        print(f"wrote {OUT / name}.json")
