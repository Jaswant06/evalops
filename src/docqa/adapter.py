"""EvalOps adapter for DocQA.

The only place where DocQA meets EvalOps: it maps a test case to a DocQA call
and the DocQA answer to the standard `AppResponse`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalops.adapters import AdapterNotReady, AppAdapter
from evalops.hashing import sha256_file
from evalops.llm import LLMError, installed_digest, loaded_models
from evalops.schemas import AppResponse, Citation, Document, Passage, TestCase, Timing, Usage

from . import __version__
from .app import DocQA, DocQASettings, ModelOutput


class DocQAAdapter(AppAdapter):
    output_model = ModelOutput

    def __init__(self, settings: dict[str, Any], base_dir: Path):
        super().__init__(settings, base_dir)
        self.config = DocQASettings.model_validate(settings)
        prompt_path = base_dir / self.config.prompt
        if not prompt_path.exists():
            raise FileNotFoundError(f"DocQA prompt not found: {prompt_path}")
        self._prompt_sha = sha256_file(prompt_path)
        self._app: DocQA | None = None

    def components(self) -> dict[str, Any]:
        return {
            "app_code": f"docqa {__version__}",
            "model": self.config.model.model_dump(),
            "prompt": {"path": str(self.config.prompt), "sha256": self._prompt_sha},
            "options": self.config.options,
            "retrieval": self.config.retrieval.model_dump(),
        }

    def prepare(self, documents: dict[str, Document]) -> None:
        self._app = DocQA(self.config, self.base_dir)
        self._app.load_documents({d.document_id: d.text for d in documents.values()})

    def check_ready(self) -> None:
        model = self.config.model
        try:
            digest = installed_digest(model.name)
        except LLMError as exc:
            raise AdapterNotReady(str(exc)) from None
        if digest is None:
            raise AdapterNotReady(f"Model {model.name} is not installed. Run: ollama pull {model.name}")
        if digest != model.digest:
            raise AdapterNotReady(
                f"Model {model.name} has digest {digest} but this app version pins {model.digest}. "
                f"The weights changed: create a new app version with the new digest instead of "
                f"reusing this one."
            )

    def resource_usage(self) -> list[dict]:
        return loaded_models()

    def warm_up(self) -> None:
        if self._app is not None:
            self._app.warm_up()

    def run_case(self, case: TestCase, timeout_s: float) -> AppResponse:
        if self._app is None:
            raise RuntimeError("prepare() must be called before run_case()")
        doc_id = case.input.document_id
        result = self._app.ask(doc_id, case.input.question, timeout_s=timeout_s)

        passages = [
            Passage(passage_id=f"{doc_id}#s{s.number}", document_id=doc_id, text=s.text, score=round(score, 4))
            for s, score in result.sections
        ]
        shown = {s.number for s, _ in result.sections}
        citations = [
            Citation(raw=str(n), passage_id=f"{doc_id}#s{n}" if n in shown else None)
            for n in result.citations
        ]
        return AppResponse(
            status=result.status,
            answer=result.answer,
            abstained=result.found is False,
            citations=citations,
            passages=passages,
            raw_output=result.raw_output,
            usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
            timing=Timing(latency_s=round(result.latency_s, 4),
                          ttft_s=round(result.ttft_s, 4) if result.ttft_s is not None else None),
            error=result.error,
        )
