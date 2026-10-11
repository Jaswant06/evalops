"""DocQA, the system under test: sectioning, retrieval, output handling, adapter, service."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from docqa import app as docqa_app
from docqa.adapter import DocQAAdapter
from docqa.app import NOT_FOUND_MESSAGE, DocQA, DocQASettings
from docqa.retrieval import bm25_top_k, split_sections
from evalops.schemas import Document, TestCase

TEXT = """Leave Policy at Acme

Introduction

Acme gives every full-time employee 25 days of paid annual leave.

Parental Leave

New parents receive 16 weeks of paid parental leave, starting from the birth or adoption.

Remote Work

Employees may work remotely two days a week with manager approval."""

SETTINGS = {"model": {"name": "qwen3.5:4b", "digest": "3c3adf637187"}, "prompt": "prompt.txt",
            "options": {"temperature": 0}, "retrieval": {"top_k": 2}}


def test_headings_merge_into_the_following_section():
    sections = split_sections(TEXT)
    assert len(sections) == 3
    assert sections[0].text.startswith("Leave Policy at Acme Introduction\nAcme gives")
    assert [s.number for s in sections] == [1, 2, 3]


def test_bm25_returns_top_k_in_document_order():
    sections = split_sections(TEXT)
    top = bm25_top_k("How many weeks of parental leave do new parents get?", sections, k=2)
    assert [s.number for s, _ in top] == sorted(s.number for s, _ in top)
    best = max(top, key=lambda x: x[1])[0]
    assert "16 weeks" in best.text
    assert len(bm25_top_k("anything", sections, k=None)) == 3


@pytest.fixture
def qa(tmp_path):
    (tmp_path / "prompt.txt").write_text("Answer from the sections.")
    qa = DocQA(DocQASettings.model_validate(SETTINGS), tmp_path)
    qa.load_documents({"acme": TEXT})
    return qa


def fake_ollama(monkeypatch, content=None, exc=None):
    def post(url, json=None, timeout=None):
        if exc:
            raise exc
        return httpx.Response(200, json={"message": {"content": content}, "prompt_eval_count": 120,
                                         "eval_count": 15, "load_duration": 0, "prompt_eval_duration": 50_000_000},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(docqa_app.httpx, "post", post)


def test_found_answer(qa, monkeypatch):
    fake_ollama(monkeypatch, json.dumps({"answer": "16 weeks.", "citations": [2], "found": True}))
    a = qa.ask("acme", "How long is parental leave?")
    assert (a.status, a.answer, a.citations, a.found) == ("ok", "16 weeks.", [2], True)
    assert a.input_tokens == 120 and a.ttft_s == pytest.approx(0.05)


def test_not_found_gives_a_consistent_refusal(qa, monkeypatch):
    fake_ollama(monkeypatch, json.dumps({"answer": "Maybe 10 days?", "citations": [1], "found": False}))
    a = qa.ask("acme", "What is the bonus policy?")
    assert a.answer == NOT_FOUND_MESSAGE and a.citations == [] and a.found is False


def test_malformed_and_timeout(qa, monkeypatch):
    fake_ollama(monkeypatch, '{"answer": "16 we')
    assert qa.ask("acme", "Parental leave?").status == "malformed"
    fake_ollama(monkeypatch, exc=httpx.ReadTimeout("slow"))
    assert qa.ask("acme", "Parental leave?").status == "timeout"
    assert qa.ask("unknown-doc", "Parental leave?").status == "error"


def test_adapter_maps_citations_to_retrieved_passages(tmp_path, monkeypatch):
    (tmp_path / "prompt.txt").write_text("Answer from the sections.")
    adapter = DocQAAdapter(SETTINGS, tmp_path)
    adapter.prepare({"acme": Document(document_id="acme", title="t", text=TEXT)})
    fake_ollama(monkeypatch, json.dumps({"answer": "16 weeks.", "citations": [2, 7], "found": True}))
    case = TestCase.model_validate({"case_id": "q", "input": {"document_id": "acme", "question": "Parental leave weeks?"},
                                    "expected": {"answerable": True, "reference": "16 weeks"},
                                    "categories": ["answerable"]})
    r = adapter.run_case(case, timeout_s=5)
    assert [p.passage_id for p in r.passages] == [p.passage_id for p in sorted(r.passages, key=lambda p: p.passage_id)]
    assert r.citations[0].passage_id == "acme#s2"
    assert r.citations[1].passage_id is None             # section 7 does not exist: an invalid citation
    assert adapter.components()["prompt"]["sha256"]


def test_adapter_refuses_a_drifted_model(tmp_path, monkeypatch):
    (tmp_path / "prompt.txt").write_text("p")
    adapter = DocQAAdapter(SETTINGS, tmp_path)
    from docqa import adapter as adapter_module
    from evalops.adapters import AdapterNotReady
    monkeypatch.setattr(adapter_module, "installed_digest", lambda name: "ffffffffffff")
    with pytest.raises(AdapterNotReady, match="weights changed"):
        adapter.check_ready()


def test_service(tmp_path, monkeypatch):
    (tmp_path / "prompt.txt").write_text("p")
    config = tmp_path / "app.yaml"
    config.write_text(yaml.safe_dump({"app": "docqa", "version": "1.0.0", "adapter": "docqa.adapter:DocQAAdapter",
                                      "settings": SETTINGS}))
    docs = tmp_path / "documents.jsonl"
    docs.write_text(json.dumps({"document_id": "acme", "title": "Leave Policy", "text": TEXT}) + "\n")
    from docqa import service
    monkeypatch.setattr(service, "CONFIG", config)
    monkeypatch.setattr(service, "DOCUMENTS", docs)
    monkeypatch.setattr(service, "BASE_DIR", tmp_path)
    fake_ollama(monkeypatch, json.dumps({"answer": "16 weeks.", "citations": [2], "found": True}))
    with TestClient(service.app) as c:
        assert c.get("/v1/documents").json() == [{"document_id": "acme", "title": "Leave Policy"}]
        r = c.post("/v1/ask", json={"document_id": "acme", "question": "Parental leave?"}).json()
        assert r["answer"] == "16 weeks." and r["version"] == "1.0.0"
        assert c.post("/v1/ask", json={"document_id": "nope", "question": "Parental leave?"}).status_code == 404
        assert 'docqa_requests_total{status="ok",version="1.0.0"}' in c.get("/metrics").text


def test_service_serves_the_production_version_unless_told_otherwise(tmp_path, monkeypatch):
    from docqa import service
    release = tmp_path / "release.yaml"
    release.write_text(yaml.safe_dump({"production": {"config": "configs/apps/docqa-v1.2.yaml"}}))
    monkeypatch.setattr(service, "RELEASE", release)
    monkeypatch.delenv("DOCQA_CONFIG", raising=False)
    assert service.production_config() == Path("configs/apps/docqa-v1.2.yaml")
    monkeypatch.setenv("DOCQA_CONFIG", "configs/apps/docqa-v1.3.yaml")
    assert service.production_config() == Path("configs/apps/docqa-v1.3.yaml")
