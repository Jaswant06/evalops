"""Deterministic checks: everything that can be decided without a model.

These never call an LLM, so they are exact and reproducible:
  - response validity: status, and the raw output parsed by the app's declared schema
  - abstention: did the app decline exactly when the dataset says it should
  - citation validity: citations exist, resolve to passages the app retrieved,
    and point into the right document (whether they *support* the answer is a
    semantic question for the judge, kept separate on purpose)
  - retrieval: did the retrieved passages contain the gold evidence passage
  - reference overlap: token F1 against the reference answer, a judge-free
    signal used to sanity-check the judge
"""

from __future__ import annotations

import re
import string
from collections import Counter

from pydantic import BaseModel, ValidationError

from .config import ChecksConfig
from .schemas import AppResponse, Document, TestCase

_STOP = frozenset("""
a an and are as at be been but by for from has have he her his i if in into is it its of on or our she so
that the their them they this to was we were will with you your
""".split())
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation + "\u2019\u2018\u201c\u201d")   # curly quotes too


def normalize_answer(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, no punctuation or articles."""
    text = _ARTICLES.sub(" ", text.lower().translate(_PUNCT))
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def content_tokens(text: str) -> set[str]:
    return {t for t in normalize_answer(text) if t not in _STOP}


def schema_valid(raw_output: str | None, output_model: type[BaseModel] | None) -> bool | None:
    if output_model is None or raw_output is None:
        return None
    try:
        output_model.model_validate_json(raw_output)
        return True
    except ValidationError:
        return False


def run_checks(case: TestCase, response: AppResponse, documents: dict[str, Document],
               output_model: type[BaseModel] | None, config: ChecksConfig) -> dict:
    valid_schema = schema_valid(response.raw_output, output_model)
    status = response.status
    if status == "ok" and valid_schema is False:
        status = "malformed"  # the parser has the final say, not the adapter
    ok = status == "ok"
    answered = ok and not response.abstained
    answerable = case.expected.answerable

    # --- citations: existence and format only --------------------------------
    passages = {p.passage_id: p for p in response.passages}
    invalid = 0
    for c in response.citations:
        p = passages.get(c.passage_id) if c.passage_id else None
        if p is None or p.document_id not in documents or p.document_id != case.input.document_id:
            invalid += 1
    needs_citation = "citation_required" in case.categories
    citation_missing = (answered and not response.citations) if needs_citation else None

    # --- retrieval against the gold evidence passage --------------------------
    retrieval_recall = retrieval_precision = evidence_coverage = None
    if answerable and case.expected.evidence and ok:
        evidence = content_tokens(case.expected.evidence)
        if evidence and response.passages:
            covered: set[str] = set()
            relevant = 0
            for p in response.passages:
                tokens = content_tokens(p.text)
                covered |= evidence & tokens
                if len(evidence & tokens) / len(evidence) >= config.passage_relevance_threshold:
                    relevant += 1
            evidence_coverage = len(covered) / len(evidence)
            retrieval_recall = float(evidence_coverage >= config.evidence_coverage_threshold)
            retrieval_precision = relevant / len(response.passages)
        elif evidence:
            retrieval_recall, retrieval_precision, evidence_coverage = 0.0, None, 0.0

    reference_f1 = None
    if answerable and case.expected.reference:
        reference_f1 = token_f1(response.answer or "", case.expected.reference) if answered else 0.0

    return {
        "status": status,
        "schema_valid": valid_schema,
        "answered": answered,
        "abstained": ok and response.abstained,
        "abstention_correct": ok and (response.abstained == (not answerable)),
        "n_citations": len(response.citations),
        "citations_invalid": invalid if ok else None,
        "citation_missing": citation_missing,
        "retrieval_recall": retrieval_recall,
        "retrieval_precision": retrieval_precision,
        "evidence_coverage": evidence_coverage,
        "reference_f1": reference_f1,
    }
