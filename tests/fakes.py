"""Test doubles: a scripted application and a scripted judge.

`ScriptedAdapter` is a second, completely different application plugged into
EvalOps through the same adapter interface as DocQA. Its behaviour per case is
set in the app config, so every failure mode (malformed JSON, timeouts,
crashes, bad citations, hallucinations) can be produced on demand.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from evalops.adapters import AdapterNotReady, AppAdapter
from evalops.llm import ChatResult
from evalops.schemas import AppResponse, Citation, Passage, TestCase, Timing, Usage


class FakeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[int]
    found: bool


class Interrupt(BaseException):
    """Stands in for Ctrl-C without killing pytest."""


class ScriptedAdapter(AppAdapter):
    output_model = FakeOutput
    calls: ClassVar[int] = 0
    interrupt_after: ClassVar[int | None] = None

    def __init__(self, settings, base_dir):
        super().__init__(settings, base_dir)
        self.attempts: dict[str, int] = {}

    def components(self) -> dict[str, Any]:
        return {"model": self.settings["model"], "prompt_version": self.settings.get("prompt_version", "p1"),
                "behaviour": self.settings.get("behaviour", {}), "usage_scale": self.settings.get("usage_scale", 1),
                "latency_s": self.settings.get("latency_s", {}), "sleep_s": self.settings.get("sleep_s", 0)}

    def check_ready(self) -> None:
        if self.settings.get("ready") is False:
            raise AdapterNotReady("scripted app is down")

    def run_case(self, case: TestCase, timeout_s: float) -> AppResponse:
        cls = type(self)
        if cls.interrupt_after is not None and cls.calls >= cls.interrupt_after:
            raise KeyboardInterrupt
        cls.calls += 1
        if self.settings.get("sleep_s"):
            time.sleep(self.settings["sleep_s"])     # real wall time, so a process can be interrupted mid-run
        behaviour = self.settings.get("behaviour", {})
        kind = behaviour.get("cases", {}).get(case.case_id, behaviour.get("default", "good"))
        if kind.startswith("flaky_timeout_"):       # times out N times, then answers
            seen = self.attempts.setdefault(case.case_id, 0)
            self.attempts[case.case_id] = seen + 1
            if seen < int(kind.rsplit("_", 1)[1]):
                return AppResponse(status="timeout", error="timed out (transient)")
            kind = "good"
        latency = self.settings.get("latency_s", {}).get(kind, self.settings.get("latency_s", {}).get("default", 1.0))
        doc = case.input.document_id
        passages = [Passage(passage_id=f"{doc}#s1", document_id=doc,
                            text=case.expected.evidence or "Nothing relevant here.")]
        scale = self.settings.get("usage_scale", 1)
        usage = Usage(input_tokens=int(100 * scale), output_tokens=int(20 * scale))

        if kind == "timeout":
            return AppResponse(status="timeout", passages=passages, error="timed out")
        if kind == "error":
            return AppResponse(status="error", error="connection refused")
        if kind == "crash":
            raise RuntimeError("adapter bug")

        found, answer, citations = True, case.expected.reference or "An invented answer.", [1]
        if kind == "good" and not case.expected.answerable:
            found, answer, citations = False, "", []
        elif kind == "abstain":
            found, answer, citations = False, "", []
        elif kind == "wrong":
            answer = "WRONG answer"
        elif kind == "unsupported":
            answer = "UNSUPPORTED claim"
        elif kind == "no_citation":
            citations = []
        elif kind == "bad_citation":
            citations = [99]
        elif kind in ("judge_garbage", "judge_flaky"):
            answer = f"{answer} {kind.upper()}"

        raw = json.dumps({"answer": answer, "citations": citations, "found": found})
        if kind == "malformed":
            raw = raw[: len(raw) // 2]
        cites = [Citation(raw=str(n), passage_id=f"{doc}#s{n}" if n == 1 else None) for n in citations]
        return AppResponse(status="ok", answer=answer if found else "Not found.", abstained=not found,
                           citations=cites if found else [], passages=passages, raw_output=raw, usage=usage,
                           timing=Timing(latency_s=latency, ttft_s=latency / 4))


class FakeJudgeClient:
    """Deterministic judge: reads markers in the answer instead of reasoning."""

    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, *, options=None, format=None, timeout_s=120) -> ChatResult:
        self.calls += 1
        user = messages[-1]["content"]
        title = (format or {}).get("title")
        seed = (options or {}).get("seed", 7)
        if title == "CorrectnessOutput":
            answer = user.split("CANDIDATE ANSWER:", 1)[1]
        else:
            answer = user.split("ANSWER:")[-1]
        if "JUDGE_GARBAGE" in answer or ("JUDGE_FLAKY" in answer and seed == 7):
            text = "I think the answer is fine."
        elif title == "CorrectnessOutput":
            wrong = "WRONG" in answer
            text = json.dumps({"reasoning": "scripted", "verdict": "incorrect" if wrong else "correct",
                               "quality": 2 if wrong else 5})
        else:
            cited = re.search(r"CITED PASSAGES:\n(.*?)\n\nQUESTION:", user, re.S).group(1)
            supported = "UNSUPPORTED" not in answer
            text = json.dumps({"claims": [{"claim": answer.strip()[:40], "supported_by_document": supported,
                                           "supported_by_cited": supported and cited != "(none)",
                                           "contradicted": False}]})
        return ChatResult(text=text, input_tokens=50, output_tokens=10, latency_s=0.01, ttft_s=0.0,
                          done_reason="stop")
