"""The adapter interface between EvalOps and the application under test.

EvalOps never imports an application directly. An app config names an adapter
class by import path (for example `docqa.adapter:DocQAAdapter`) and the engine
talks to it only through this interface. Supporting a new application, a RAG
service or a prediction API, means writing one adapter; the engine, the checks,
the gate and the reports stay the same.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from .schemas import AppResponse, Document, TestCase


class AdapterNotReady(RuntimeError):
    """The application cannot serve live requests (model missing, digest drift, service down)."""


class AppAdapter(ABC):
    #: Pydantic model of the application's raw structured output, if it has one.
    #: The deterministic checks parse `AppResponse.raw_output` with it, so JSON
    #: validity is decided by a parser rather than trusted from the adapter.
    output_model: ClassVar[type[BaseModel] | None] = None

    def __init__(self, settings: dict[str, Any], base_dir: Path):
        self.settings = settings
        self.base_dir = base_dir

    @abstractmethod
    def components(self) -> dict[str, Any]:
        """Everything that determines behaviour: model name and digest, prompt hash,
        generation parameters, app code version. Must not need the network, so
        replay runs in CI can fingerprint a version without its model."""

    def prepare(self, documents: dict[str, Document]) -> None:
        """Load the corpus the test cases refer to. Called once per run."""

    def check_ready(self) -> None:
        """Raise AdapterNotReady if live requests cannot be served as versioned."""

    def warm_up(self) -> None:
        """Optional: load models before the first timed request, so cold starts do not skew latency."""

    @abstractmethod
    def run_case(self, case: TestCase, timeout_s: float) -> AppResponse:
        """Answer one test case. Application failures (bad output, timeouts, errors)
        are returned as a status, not raised: they are results to be measured."""

    def resource_usage(self) -> list[dict]:
        """Optional: memory held by the app's models, as [{"name": ..., "bytes": ...}], after live requests."""
        return []

    def model_name(self) -> str | None:
        """The model name used for cost estimates, if the app has a single model."""
        model = self.components().get("model")
        return model.get("name") if isinstance(model, dict) else None


def load_adapter_class(path: str) -> type[AppAdapter]:
    """'package.module:ClassName' -> the class, checked to be an AppAdapter."""
    module_name, _, class_name = path.partition(":")
    if not class_name:
        raise ValueError(f"Adapter path '{path}' must look like 'package.module:ClassName'")
    cls = getattr(importlib.import_module(module_name), class_name, None)
    if cls is None:
        raise ValueError(f"{module_name} has no attribute {class_name}")
    if not (isinstance(cls, type) and issubclass(cls, AppAdapter)):
        raise TypeError(f"{path} is not an evalops.adapters.AppAdapter")
    return cls
