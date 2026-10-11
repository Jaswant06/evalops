"""LLM judge for the semantic questions deterministic checks cannot answer.

Two judgments per answered case:
  correctness  answer vs the human reference answer: verdict + 1-5 quality
  claims       the answer split into claims, each checked against the cited
               passages (citation support) and against the whole document
               (groundedness), plus contradiction

Outputs are constrained by a JSON schema and validated with Pydantic. A
malformed output is retried with a different seed up to `max_attempts`, then
the case is marked `unevaluable`: it is excluded from judged metrics (never
given a score) and counted in `judge_unevaluable_rate`, which the gate limits,
so a failing judge can never make a candidate look better.

Raw judge outputs go through the version-aware cache; the parsing and the
metrics are recomputed on every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cache import ResponseCache
from .config import JudgeConfig
from .hashing import sha256_file, sha256_json
from .llm import LLMError, OllamaChat
from .schemas import AppResponse, Document, TestCase

MAX_CLAIMS = 6
VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


class CorrectnessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    verdict: Literal["correct", "partially_correct", "incorrect"]
    quality: int = Field(ge=1, le=5)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    supported_by_cited: bool
    supported_by_document: bool
    contradicted: bool


class ClaimsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim]


@dataclass
class Judgment:
    status: Literal["ok", "unevaluable", "skipped"]
    output: dict[str, Any] | None = None
    attempts: int = 0
    cache_hits: int = 0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Judge:
    def __init__(self, config: JudgeConfig, base_dir: Path, cache: ResponseCache, client: OllamaChat | None = None,
                 before_live_call: Callable[[], None] | None = None):
        self.config = config
        self.cache = cache
        self.client = client or OllamaChat(max_retries=1)
        # Called before any live judge call, e.g. to verify the judge model's digest.
        self.before_live_call = before_live_call
        self.live_calls = 0
        self.prompts = {name: (base_dir / path).read_text().strip() for name, path in config.prompts.items()}
        self.identity = {
            "model": config.model.model_dump(),
            "prompts": {name: sha256_file(base_dir / path) for name, path in config.prompts.items()},
            "options": config.options,
        }
        self.fingerprint = sha256_json(self.identity)

    # --- message builders -------------------------------------------------------

    def correctness_messages(self, case: TestCase, response: AppResponse) -> list[dict]:
        user = (
            f"QUESTION: {case.input.question}\n\n"
            f"REFERENCE ANSWER: {case.expected.reference}\n\n"
            f"SUPPORTING PASSAGE: {case.expected.evidence or '(not available)'}\n\n"
            f"CANDIDATE ANSWER: {response.answer}"
        )
        return [{"role": "system", "content": self.prompts["correctness"]}, {"role": "user", "content": user}]

    def claims_messages(self, case: TestCase, response: AppResponse, document: Document) -> list[dict]:
        by_id = {p.passage_id: p for p in response.passages}
        cited = [by_id[c.passage_id] for c in response.citations if c.passage_id in by_id]
        cited_text = "\n\n".join(f"[{p.passage_id}] {p.text}" for p in cited) or "(none)"
        # The document goes first so consecutive calls about the same document
        # share a prompt prefix that Ollama can reuse.
        user = (
            f"FULL DOCUMENT:\n{document.text}\n\n"
            f"CITED PASSAGES:\n{cited_text}\n\n"
            f"QUESTION: {case.input.question}\n\n"
            f"ANSWER: {response.answer}"
        )
        return [{"role": "system", "content": self.prompts["claims"]}, {"role": "user", "content": user}]

    # --- judging -------------------------------------------------------------------

    def _judge(self, item_id: str, messages: list[dict], schema: type[BaseModel]) -> Judgment:
        format_schema = schema.model_json_schema()
        judgment = Judgment(status="unevaluable")
        for attempt in range(1, self.config.max_attempts + 1):
            judgment.attempts = attempt
            # A different seed per attempt, so a retry is not the same malformed output again.
            options = {**self.config.options, "seed": self.config.options.get("seed", 0) + attempt - 1}
            input_hash = sha256_json({"messages": messages, "format": format_schema, "options": options})
            attempt_id = f"{item_id}:a{attempt}"
            entry = self.cache.get(self.fingerprint, attempt_id, input_hash, self.identity)
            if entry is not None:
                payload = entry.payload
                judgment.cache_hits += 1
            else:
                if self.before_live_call:
                    self.before_live_call()
                self.live_calls += 1
                try:
                    result = self.client.chat(self.config.model.name, messages, options=options,
                                              format=format_schema, timeout_s=self.config.timeout_s)
                except LLMError as exc:
                    judgment.error = f"judge call failed: {exc}"
                    continue
                payload = {"text": result.text, "input_tokens": result.input_tokens,
                           "output_tokens": result.output_tokens, "done_reason": result.done_reason}
                self.cache.put(self.fingerprint, attempt_id, input_hash, self.identity, payload)
            judgment.input_tokens += payload.get("input_tokens") or 0
            judgment.output_tokens += payload.get("output_tokens") or 0
            try:
                parsed = schema.model_validate_json(payload["text"])
            except ValidationError as exc:
                judgment.error = f"malformed judge output (attempt {attempt}): {str(exc)[:160]}"
                continue
            judgment.status, judgment.output, judgment.error = "ok", parsed.model_dump(), None
            return judgment
        return judgment

    def correctness(self, case: TestCase, response: AppResponse) -> Judgment:
        return self._judge(f"{case.case_id}:correctness", self.correctness_messages(case, response), CorrectnessOutput)

    def claims(self, case: TestCase, response: AppResponse, document: Document) -> Judgment:
        judgment = self._judge(f"{case.case_id}:claims", self.claims_messages(case, response, document), ClaimsOutput)
        if judgment.status == "ok":
            judgment.output["claims"] = judgment.output["claims"][:MAX_CLAIMS]
        return judgment


def structured_judgment(evaluation: str, j: Judgment) -> dict:
    """The stored form of one judge evaluation: same fields for every evaluation, validated outputs only."""
    record = {"evaluation": evaluation, "status": j.status, "attempts": j.attempts, "score": None,
              "label": None, "justification": None, "error": j.error}
    if j.status == "ok" and evaluation == "correctness":
        out = j.output
        record.update(score=VERDICT_SCORE[out["verdict"]], label=out["verdict"],
                      justification=out["reasoning"][:500])
    elif j.status == "ok" and evaluation == "claims":
        claims = j.output["claims"]
        supported = sum(c["supported_by_cited"] for c in claims)
        record.update(score=supported / len(claims) if claims else None,
                      label=f"{supported}/{len(claims)} claims supported by citations",
                      justification="; ".join(f"{'supported' if c['supported_by_document'] else 'UNSUPPORTED'}: "
                                               f"{c['claim']}" for c in claims)[:500])
    return record


def summarize_correctness(j: Judgment) -> dict:
    if j.status != "ok":
        return {"verdict": None, "correctness": None, "quality": None, "judge_reasoning": None}
    out = j.output
    return {"verdict": out["verdict"], "correctness": VERDICT_SCORE[out["verdict"]],
            "quality": out["quality"], "judge_reasoning": out["reasoning"]}


def summarize_claims(j: Judgment) -> dict:
    if j.status != "ok":
        return {"claims_total": None, "claims_supported_cited": None,
                "claims_supported_document": None, "claims_contradicted": None, "claims": None}
    claims = j.output["claims"]
    return {
        "claims_total": len(claims),
        "claims_supported_cited": sum(c["supported_by_cited"] for c in claims),
        "claims_supported_document": sum(c["supported_by_document"] for c in claims),
        "claims_contradicted": sum(c["contradicted"] for c in claims),
        "claims": claims,
    }
