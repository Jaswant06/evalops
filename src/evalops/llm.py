"""Minimal Ollama client for the LLM judge.

Timeouts and bounded retries live here: transport errors and timeouts are
retried with backoff, HTTP 4xx errors are not (they will not fix themselves).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = f"http://{DEFAULT_HOST}"


class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    pass


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    ttft_s: float
    done_reason: str


def installed_digest(model: str, host: str = DEFAULT_HOST) -> str | None:
    """First 12 hex chars of the model's primary manifest digest, or None if not installed."""
    try:
        r = httpx.post(f"{host}/api/show", json={"model": model}, timeout=30)
    except httpx.HTTPError as exc:
        raise LLMError(f"Ollama not reachable at {host}: {exc}") from None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    manifests = data.get("manifests") or []
    if manifests:
        return manifests[0]["digest"].removeprefix("sha256:")[:12]
    # Older Ollama versions: fall back to the tag listing.
    tags = httpx.get(f"{host}/api/tags", timeout=30).json().get("models", [])
    for m in tags:
        if m["name"] == model:
            return m["digest"][:12]
    return None


def loaded_models(host: str = DEFAULT_HOST) -> list[dict]:
    """Models Ollama currently holds in memory, with their size: resource usage for live runs."""
    try:
        r = httpx.get(f"{host}/api/ps", timeout=10)
        r.raise_for_status()
    except httpx.HTTPError:
        return []
    return [{"name": m["name"], "bytes": m.get("size_vram") or m.get("size") or 0} for m in r.json().get("models", [])]


class OllamaChat:
    def __init__(self, host: str = DEFAULT_HOST, max_retries: int = 2, backoff_s: float = 1.0):
        self.host = host.rstrip("/")
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def chat(self, model: str, messages: list[dict], *, options: dict[str, Any] | None = None,
             format: dict | None = None, timeout_s: float = 120) -> ChatResult:
        body = {"model": model, "messages": messages, "stream": False, "think": False,
                "options": options or {}}
        if format is not None:
            body["format"] = format
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
            started = time.perf_counter()
            try:
                r = httpx.post(f"{self.host}/api/chat", json=body, timeout=timeout_s)
            except httpx.TimeoutException as exc:
                last = LLMTimeout(f"{model} timed out after {timeout_s}s")
                continue
            except httpx.TransportError as exc:
                last = LLMError(f"Ollama not reachable at {self.host}: {exc}")
                continue
            if r.status_code >= 500:
                last = LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
                continue
            if r.status_code >= 400:
                raise LLMError(f"Ollama rejected the request ({r.status_code}): {r.text[:200]}")
            data = r.json()
            wall = time.perf_counter() - started
            ns = 1e9
            return ChatResult(
                text=data["message"]["content"],
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency_s=wall,
                ttft_s=(data.get("load_duration", 0) + data.get("prompt_eval_duration", 0)) / ns,
                done_reason=data.get("done_reason", ""),
            )
        assert last is not None
        raise last
