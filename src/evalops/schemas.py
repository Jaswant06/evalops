"""Data contracts shared by the engine, the adapters and the store.

These models are the boundary between EvalOps and any application it
evaluates. An adapter turns a `TestCase` into an `AppResponse`; nothing in the
engine knows how the application produced it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Unknown fields are an error: a typo in a dataset or config must not pass silently."""

    model_config = ConfigDict(extra="forbid")


# --- Dataset -----------------------------------------------------------------

class Document(Strict):
    document_id: str
    title: str
    text: str


class CaseInput(Strict):
    document_id: str
    question: str


class Expected(Strict):
    answerable: bool
    reference: str | None = None
    evidence: str | None = None


class TestCase(Strict):
    __test__ = False  # not a pytest test class

    case_id: str
    input: CaseInput
    expected: Expected
    categories: list[str] = Field(min_length=1)


# --- Application response (what an adapter returns) --------------------------

ResponseStatus = Literal["ok", "malformed", "timeout", "error"]


class Passage(Strict):
    """A piece of context the application retrieved and showed to its model."""

    passage_id: str
    document_id: str
    text: str
    score: float | None = None


class Citation(Strict):
    """A citation as the application emitted it, resolved to a passage when possible."""

    raw: str
    passage_id: str | None = None


class Usage(Strict):
    input_tokens: int | None = None
    output_tokens: int | None = None


class Timing(Strict):
    latency_s: float
    ttft_s: float | None = None


class AppResponse(Strict):
    status: ResponseStatus
    answer: str | None = None
    abstained: bool = False
    citations: list[Citation] = []
    passages: list[Passage] = []
    raw_output: str | None = None
    usage: Usage = Usage()
    timing: Timing | None = None
    error: str | None = None


class AppVersion(Strict):
    """Everything that determines an application's behaviour, hashed into a fingerprint.

    `components` is adapter-defined (model name and digest, prompt hash,
    generation parameters, retrieval settings, app code version). Two versions
    with the same fingerprint behave identically, so their cached responses are
    interchangeable; anything else is not.
    """

    app: str
    version: str
    adapter: str
    components: dict[str, Any]
    fingerprint: str


# --- Results -------------------------------------------------------------------

class CaseExecution(Strict):
    """One application response plus where it came from."""

    response: AppResponse
    cache_hit: bool
    recorded_at: str
    attempts: int = 1
