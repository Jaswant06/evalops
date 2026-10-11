"""The DocQA application: retrieve sections, ask the model, return a cited answer."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .retrieval import Section, bm25_top_k, split_sections

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

NOT_FOUND_MESSAGE = "I could not find the answer to that in this document."


class ModelOutput(BaseModel):
    """The JSON object the model must return. Also used by EvalOps to check validity."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[int]
    found: bool


OUTPUT_SCHEMA = ModelOutput.model_json_schema()


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    digest: str


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "bm25"
    top_k: int | None = Field(4, ge=1)


class DocQASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelSettings
    prompt: Path
    options: dict[str, Any] = {"temperature": 0, "seed": 7}
    retrieval: RetrievalSettings = RetrievalSettings()


@dataclass
class Answer:
    status: str                       # ok | malformed | timeout | error
    answer: str | None = None
    found: bool | None = None
    citations: list[int] = field(default_factory=list)
    sections: list[tuple[Section, float]] = field(default_factory=list)
    raw_output: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None
    error: str | None = None


class DocQA:
    def __init__(self, settings: DocQASettings, base_dir: Path = Path("."), host: str = OLLAMA_HOST):
        self.settings = settings
        self.system_prompt = (base_dir / settings.prompt).read_text().strip()
        self.host = host.rstrip("/")
        self._sections: dict[str, list[Section]] = {}

    def load_documents(self, documents: dict[str, str]) -> None:
        self._sections = {doc_id: split_sections(text) for doc_id, text in documents.items()}

    def has_document(self, document_id: str) -> bool:
        return document_id in self._sections

    def retrieve(self, document_id: str, question: str) -> list[tuple[Section, float]]:
        return bm25_top_k(question, self._sections[document_id], self.settings.retrieval.top_k)

    @staticmethod
    def build_user_message(sections: list[tuple[Section, float]], question: str) -> str:
        context = "\n\n".join(f"[{s.number}] {s.text}" for s, _ in sections)
        return f"Document sections:\n\n{context}\n\nQuestion: {question}"

    def warm_up(self) -> None:
        """Load the model into memory with a one-token request."""
        try:
            httpx.post(f"{self.host}/api/generate", json={"model": self.settings.model.name, "prompt": "ok",
                                                         "stream": False, "options": {"num_predict": 1}}, timeout=120)
        except httpx.HTTPError:
            pass  # a real problem will show up, and be measured, on the first request

    def ask(self, document_id: str, question: str, timeout_s: float = 60) -> Answer:
        started = time.perf_counter()
        if document_id not in self._sections:
            return Answer(status="error", error=f"unknown document {document_id}")
        sections = self.retrieve(document_id, question)
        body = {
            "model": self.settings.model.name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_user_message(sections, question)},
            ],
            "stream": False,
            "think": False,
            "format": OUTPUT_SCHEMA,
            "options": self.settings.options,
        }
        try:
            r = httpx.post(f"{self.host}/api/chat", json=body, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
        except httpx.TimeoutException:
            return Answer(status="timeout", sections=sections, latency_s=time.perf_counter() - started,
                          error=f"model did not answer within {timeout_s}s")
        except httpx.HTTPError as exc:
            return Answer(status="error", sections=sections, latency_s=time.perf_counter() - started,
                          error=f"{type(exc).__name__}: {exc}")

        raw = data["message"]["content"]
        result = Answer(
            status="ok", sections=sections, raw_output=raw,
            input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"),
            latency_s=time.perf_counter() - started,
            ttft_s=(data.get("load_duration", 0) + data.get("prompt_eval_duration", 0)) / 1e9,
        )
        try:
            parsed = ModelOutput.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            result.status = "malformed"
            result.error = f"model output is not valid JSON for the schema: {str(exc)[:200]}"
            return result

        result.found = parsed.found
        if parsed.found:
            result.answer = parsed.answer.strip()
            result.citations = parsed.citations
        else:
            # A consistent refusal: users never see a half-answer flagged as not found.
            result.answer = NOT_FOUND_MESSAGE
        return result
