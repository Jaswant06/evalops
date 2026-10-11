"""Unit tests for the building blocks: datasets, configs, cache, checks, metrics."""

from __future__ import annotations

import json

import pytest
import yaml

from evalops.cache import CacheMiss, ResponseCache
from evalops.checks import run_checks, schema_valid, token_f1
from evalops.config import ChecksConfig, ConfigError, load_app, load_gate, load_scoring
from evalops.datasets import DatasetError, load
from evalops.hashing import sha256_file, sha256_json
from evalops.metrics import METRICS, CaseRecord, compute, compute_all
from evalops.schemas import AppResponse, Citation, Document, Passage, TestCase
from tests.fakes import FakeOutput

# --- datasets -----------------------------------------------------------------------


def test_dataset_loads_with_categories(project):
    ds = load("toy/v1:test", project.root / "datasets")
    assert len(ds.cases) == 20
    assert ds.categories() == {"answerable": 16, "citation_required": 16, "unanswerable": 4}


def test_dataset_rejects_unlisted_files(project):
    (project.root / "datasets/toy/v1/extra.jsonl").write_text("{}\n")
    with pytest.raises(DatasetError, match="not in the manifest"):
        load("toy/v1:test", project.root / "datasets")


def test_dataset_schema_is_validated_even_when_hashes_match(project):
    d = project.root / "datasets/toy/v1"
    rows = (d / "test.jsonl").read_text().splitlines()
    bad = json.loads(rows[0])
    bad["expected"]["answerable"] = "maybe"
    rows[0] = json.dumps(bad)
    (d / "test.jsonl").write_text("\n".join(rows) + "\n")
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["files"]["test.jsonl"] = sha256_file(d / "test.jsonl")
    manifest["dataset_hash"] = sha256_json(manifest["files"])
    (d / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match="does not match the schema"):
        load("toy/v1:test", project.root / "datasets")


def test_bad_dataset_reference():
    with pytest.raises(DatasetError, match="Expected name/version:split"):
        load("toy-v1")


# --- configs --------------------------------------------------------------------------


def write_gate(project, rules):
    path = project.root / "configs/gate.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "rules": rules}))
    return path


@pytest.mark.parametrize("rule, message", [
    ({"id": "x", "metric": "made_up", "min": 1}, "unknown metric"),
    ({"id": "x", "metric": "app_failure_rate"}, "sets no threshold"),
    ({"id": "x", "metric": "app_failure_rate", "max_drop": 0.1}, "only applies to higher-is-better"),
    ({"id": "x", "metric": "answer_correctness", "max_increase": 0.1}, "only applies to lower-is-better"),
    ({"id": "x", "metric": "app_failure_rate", "max": 0.1, "severity": "optional"}, "severity"),
    ({"id": "x", "metric": "app_failure_rate", "max": 0.1, "thresold": 3}, "Extra inputs"),
])
def test_invalid_gate_rules_are_rejected(project, rule, message):
    with pytest.raises(ConfigError, match=message):
        load_gate(write_gate(project, [rule]))


def test_duplicate_rule_ids_are_rejected(project):
    rule = {"id": "x", "metric": "app_failure_rate", "max": 0.1}
    with pytest.raises(ConfigError, match="duplicate rule ids"):
        load_gate(write_gate(project, [rule, rule]))


def test_unknown_app_config_key_is_rejected(project):
    path = project.write_app("good")
    data = yaml.safe_load((project.root / path).read_text())
    data["promt"] = "typo"
    (project.root / path).write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_app(project.root / path)


def test_scoring_fingerprint_tracks_judge_prompt_content(project):
    scoring = load_scoring(project.root / "configs/scoring.yaml")
    before = scoring.fingerprint(project.root)
    (project.root / "prompts/judge/claims.txt").write_text("Check the claims strictly.")
    assert scoring.fingerprint(project.root) != before


def test_app_fingerprint_tracks_every_component(project):
    project.write_app("a")
    base = project.handle("a").version.fingerprint
    project.write_app("a", prompt_version="p2")
    assert project.handle("a").version.fingerprint != base
    project.write_app("a", digest="cccccccccccc")
    assert project.handle("a").version.fingerprint != base
    project.write_app("a", version="9.9.9")  # the label alone does not change behaviour
    assert project.handle("a").version.fingerprint == base


# --- cache ----------------------------------------------------------------------------

IDENTITY = {"model": {"name": "m", "digest": "abc"}}


def test_cache_roundtrip_and_identity_check(tmp_path):
    cache = ResponseCache(tmp_path, "app")
    cache.put("fp1", "case1", "in1", IDENTITY, {"x": 1})
    fresh = ResponseCache(tmp_path, "app")  # reads from disk
    assert fresh.get("fp1", "case1", "in1", IDENTITY).payload == {"x": 1}
    assert fresh.get("fp1", "case1", "in2", IDENTITY) is None          # different input
    assert fresh.get("fp1", "case1", "in1", {"model": {"name": "m", "digest": "zzz"}}) is None
    assert fresh.stats.hits == 1 and fresh.stats.misses == 2 and fresh.stats.rejected == 1


def test_cache_modes(tmp_path):
    ResponseCache(tmp_path, "app").put("fp", "c", "i", IDENTITY, {"x": 1})
    assert ResponseCache(tmp_path, "app", mode="refresh").get("fp", "c", "i", IDENTITY) is None
    no_write = ResponseCache(tmp_path, "app", mode="refresh", write=False)
    no_write.put("fp", "c2", "i", IDENTITY, {"x": 2})
    assert ResponseCache(tmp_path, "app").get("fp", "c2", "i", IDENTITY) is None
    with pytest.raises(CacheMiss, match="No recorded app output"):
        ResponseCache(tmp_path, "app", mode="replay").get("fp", "missing", "i", IDENTITY)


def test_cache_survives_a_torn_last_line(tmp_path):
    cache = ResponseCache(tmp_path, "app")
    cache.put("fp", "c", "i", IDENTITY, {"x": 1})
    path = next((tmp_path / "app").glob("*.jsonl"))
    path.write_text(path.read_text() + '{"key": "trunc')
    assert ResponseCache(tmp_path, "app").get("fp", "c", "i", IDENTITY).payload == {"x": 1}


# --- deterministic checks ------------------------------------------------------------------

CASE = TestCase.model_validate({
    "case_id": "q1", "input": {"document_id": "d1", "question": "How many days?"},
    "expected": {"answerable": True, "reference": "Staff get 12 days.",
                 "evidence": "All full-time staff get 12 days of paid leave each year."},
    "categories": ["answerable", "citation_required"]})
DOCS = {"d1": Document(document_id="d1", title="t", text="..."), "d2": Document(document_id="d2", title="t", text="...")}
PASSAGES = [Passage(passage_id="d1#s1", document_id="d1", text="All full-time staff get 12 days of paid leave each year."),
            Passage(passage_id="d1#s2", document_id="d1", text="Parking is free on weekends."),
            Passage(passage_id="d2#s1", document_id="d2", text="Other document.")]


def response(**kw) -> AppResponse:
    base = dict(status="ok", answer="Staff get 12 days.", citations=[Citation(raw="1", passage_id="d1#s1")],
                passages=PASSAGES[:2], raw_output=json.dumps({"answer": "x", "citations": [1], "found": True}))
    return AppResponse(**{**base, **kw})


def test_token_f1():
    assert token_f1("Staff get 12 days.", "staff get 12 days") == 1.0
    assert token_f1("The answer", "a different thing") == 0.0
    assert 0 < token_f1("12 days of leave", "Staff get 12 days") < 1


def test_schema_check_uses_the_parser():
    assert schema_valid('{"answer": "a", "citations": [1], "found": true}', FakeOutput) is True
    assert schema_valid('{"answer": "a", "citations": [1]}', FakeOutput) is False
    assert schema_valid('{"answer": "a", "cit', FakeOutput) is False
    assert schema_valid(None, FakeOutput) is None


def test_checks_on_a_good_answer():
    ck = run_checks(CASE, response(), DOCS, FakeOutput, ChecksConfig())
    assert ck["status"] == "ok" and ck["answered"] and ck["abstention_correct"]
    assert ck["citations_invalid"] == 0 and ck["citation_missing"] is False
    assert ck["retrieval_recall"] == 1.0 and ck["retrieval_precision"] == 0.5
    assert ck["reference_f1"] == 1.0


def test_citation_validity_is_separate_from_support():
    unresolved = response(citations=[Citation(raw="9")])
    other_doc = response(citations=[Citation(raw="1", passage_id="d2#s1")], passages=PASSAGES)
    assert run_checks(CASE, unresolved, DOCS, FakeOutput, ChecksConfig())["citations_invalid"] == 1
    assert run_checks(CASE, other_doc, DOCS, FakeOutput, ChecksConfig())["citations_invalid"] == 1
    missing = run_checks(CASE, response(citations=[]), DOCS, FakeOutput, ChecksConfig())
    assert missing["citation_missing"] is True


def test_parser_overrides_an_adapter_that_says_ok():
    ck = run_checks(CASE, response(raw_output='{"answer": '), DOCS, FakeOutput, ChecksConfig())
    assert ck["status"] == "malformed" and not ck["answered"] and ck["reference_f1"] == 0.0


def test_retrieval_miss():
    ck = run_checks(CASE, response(passages=[PASSAGES[1]]), DOCS, FakeOutput, ChecksConfig())
    assert ck["retrieval_recall"] == 0.0 and ck["retrieval_precision"] == 0.0


def test_wrong_abstention():
    ck = run_checks(CASE, response(abstained=True, answer="not found", citations=[]), DOCS, FakeOutput, ChecksConfig())
    assert ck["abstained"] and not ck["abstention_correct"] and ck["reference_f1"] == 0.0


# --- metrics -----------------------------------------------------------------------------


def rec(**kw) -> CaseRecord:
    base = dict(case_id="c", categories=["answerable", "citation_required"], answerable=True, status="ok",
                answered=True, abstention_correct=True, original_latency_s=1.0)
    return CaseRecord(**{**base, **kw})


def test_citation_support_is_micro_averaged_over_claims():
    records = [rec(case_id="a", judge_claims_status="ok", claims_total=3, claims_supported_cited=3,
                   claims_supported_document=3, claims_contradicted=0),
               rec(case_id="b", judge_claims_status="ok", claims_total=1, claims_supported_cited=0,
                   claims_supported_document=1, claims_contradicted=0)]
    value, n = compute(METRICS["citation_support_rate"], records)
    assert value == pytest.approx(3 / 4) and n == 2
    assert compute(METRICS["unsupported_claim_rate"], records)[0] == 0.0


def test_failures_count_against_correctness_and_are_excluded_from_latency():
    records = [rec(case_id="a", judge_correctness_status="ok", correctness=1.0, original_latency_s=1.0),
               rec(case_id="b", status="timeout", answered=False, original_latency_s=None)]
    assert compute(METRICS["answer_correctness"], records) == (0.5, 2)
    assert compute(METRICS["latency_p50_s"], records) == (1.0, 1)
    assert compute(METRICS["app_failure_rate"], records) == (0.5, 2)


def test_compute_all_has_every_category():
    out = compute_all([rec(case_id="a"), rec(case_id="b", categories=["unanswerable"], answerable=False,
                                             answered=False, abstained=True)])
    assert set(out) == {"all", "answerable", "unanswerable", "citation_required"}
    assert out["unanswerable"]["correct_abstention_rate"] == {"value": 1.0, "n": 1}
    assert out["answerable"]["cases"]["value"] == 1


# --- judge reliability ---------------------------------------------------------------------

def test_cohen_kappa_and_agreement():
    from evalops.judgecheck import agreement, cohen_kappa
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohen_kappa([("a", "b"), ("b", "a")]) < 0
    records = [rec(case_id=f"c{i}", verdict=v) for i, v in enumerate(["correct", "correct", "incorrect", "correct"])]
    labels = {"c0": "correct", "c1": "incorrect", "c2": "incorrect", "c3": "correct"}
    out = agreement(records, labels)
    assert out["n"] == 4 and out["agreement"] == 0.75 and out["judge_more_lenient"] == 1


def test_labelling_sample_is_stratified_and_reproducible():
    from evalops.judgecheck import sample_for_labelling
    records = [rec(case_id=f"c{i}", verdict="incorrect" if i < 3 else "correct") for i in range(50)]
    sample = sample_for_labelling(records, 10, seed=1)
    assert len(sample) == 10 and sum(r.verdict == "incorrect" for r in sample) == 3
    assert [r.case_id for r in sample] == [r.case_id for r in sample_for_labelling(records, 10, seed=1)]


def test_judge_probes_catch_constructed_errors(project):
    """The scripted judge flags 'UNSUPPORTED'/'WRONG' markers only, so swapped answers from
    other cases (which carry no marker) are NOT caught: the probe must report that honestly."""
    from evalops.engine import RunOptions
    from evalops.judgecheck import run_probes
    import evalops.llm as llm
    project.write_app("good")
    ev = project.evaluator()
    result = ev.run(project.handle("good"), RunOptions(quiet=True))
    project.monkeypatch.setattr(llm, "installed_digest", lambda name: "aaaaaaaaaaaa")
    # The probe builds its own Judge; give it the scripted client.
    from tests.fakes import FakeJudgeClient
    import evalops.judge as judge_module
    project.monkeypatch.setattr(judge_module, "OllamaChat", lambda **kw: FakeJudgeClient())
    out = run_probes(ev, result.run_id, n=5)
    assert out["probes_per_kind"] == 5
    assert out["uncited_detection_rate"] == 1.0          # no cited passages: always caught
    assert out["cross_document_detection_rate"] == 0.0   # the scripted judge cannot tell; reported as such


def test_cache_prune_keeps_only_used_entries(tmp_path):
    from evalops.cache import prune
    cache = ResponseCache(tmp_path, "app")
    keep = cache.put("fp", "c1", "i", IDENTITY, {"x": 1})
    cache.put("fp", "c2", "i", IDENTITY, {"x": 2})
    cache.put("old", "c1", "i", IDENTITY, {"x": 3})
    cache.put("fp", "probe-x-c1", "i", IDENTITY, {"x": 4})
    assert prune(tmp_path, "app", {keep.key}, ("probe-",)) == {"kept": 2, "removed": 2}
    fresh = ResponseCache(tmp_path, "app")
    assert fresh.get("fp", "c1", "i", IDENTITY) is not None and fresh.get("fp", "c2", "i", IDENTITY) is None
    assert not list((tmp_path / "app").glob("old*"))
