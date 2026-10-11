"""A self-contained toy project: dataset, configs and fake app/judge, in a temp directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from evalops import engine as engine_module
from evalops.config import load_platform
from evalops.engine import Evaluator, app_from_config
from evalops.config import load_app
from evalops.hashing import sha256_file, sha256_json
from evalops.judge import Judge
from evalops.store import Store
from tests.fakes import FakeJudgeClient, ScriptedAdapter

JUDGE_DIGEST = "aaaaaaaaaaaa"
N_ANSWERABLE, N_UNANSWERABLE = 16, 4

GATE = {
    "version": 1,
    "bootstrap_samples": 300,
    "rules": [
        {"id": "correctness", "metric": "answer_correctness", "category": "answerable", "min": 0.8,
         "max_drop": 0.05, "min_cases": 1},
        {"id": "declines", "metric": "correct_abstention_rate", "category": "unanswerable", "min": 0.75,
         "min_cases": 1},
        {"id": "citations", "metric": "citation_missing_rate", "category": "citation_required", "max": 0.1,
         "min_cases": 1},
        {"id": "support", "metric": "citation_support_rate", "category": "citation_required", "min": 0.8,
         "min_cases": 1},
        {"id": "failures", "metric": "app_failure_rate", "max": 0.05, "min_cases": 1},
        {"id": "judge", "metric": "judge_unevaluable_rate", "max": 0.05, "min_cases": 1},
        {"id": "latency", "metric": "latency_p95_s", "max": 3.0, "max_increase_pct": 50, "min_cases": 1},
        {"id": "cost", "metric": "est_cost_per_1k_requests_usd", "severity": "advisory", "max_increase_pct": 20,
         "min_cases": 1},
    ],
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def build_dataset(root: Path, name: str = "toy", version: str = "v1") -> Path:
    d = root / "datasets" / name / version
    d.mkdir(parents=True)
    docs = [{"document_id": f"d{i}", "title": f"Policy {i}", "text": f"Policy {i}. Staff get {i + 10} days off."}
            for i in (1, 2)]
    cases = []
    for i in range(1, N_ANSWERABLE + N_UNANSWERABLE + 1):
        answerable = i <= N_ANSWERABLE
        cases.append({
            "case_id": f"c{i:02d}",
            "input": {"document_id": f"d{1 + i % 2}", "question": f"Question number {i}?"},
            "expected": {"answerable": answerable,
                         "reference": f"The answer is {i} days." if answerable else None,
                         "evidence": f"Staff get the answer is {i} days off in policy." if answerable else None},
            "categories": ["answerable", "citation_required"] if answerable else ["unanswerable"],
        })
    write_jsonl(d / "documents.jsonl", docs)
    write_jsonl(d / "test.jsonl", cases)
    files = {p.name: sha256_file(p) for p in sorted(d.glob("*.jsonl"))}
    manifest = {"name": name, "version": version, "schema_version": 1,
                "splits": {"test": {"file": "test.jsonl", "cases": len(cases)}},
                "files": files, "dataset_hash": sha256_json(files)}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


@dataclass
class Project:
    root: Path
    monkeypatch: pytest.MonkeyPatch

    def write_app(self, name: str, behaviour: dict | None = None, *, digest: str = "bbbbbbbbbbbb",
                  latency: dict | None = None, faults: dict | None = None, prompt_version: str = "p1",
                  version: str | None = None, usage_scale: float = 1, sleep_s: float = 0) -> Path:
        config = {
            "app": "fakeapp", "version": version or name, "adapter": "tests.fakes:ScriptedAdapter",
            "settings": {"model": {"name": "fake-model", "digest": digest}, "prompt_version": prompt_version,
                         "behaviour": behaviour or {"default": "good"}, "latency_s": latency or {"default": 1.0},
                         "usage_scale": usage_scale, "sleep_s": sleep_s},
        }
        if faults:
            config["fault_injection"] = faults
        path = Path("configs/apps") / f"{name}.yaml"
        (self.root / path).write_text(yaml.safe_dump(config))
        return path

    def handle(self, name: str):
        return app_from_config(load_app(self.root / "configs/apps" / f"{name}.yaml"), self.root)

    def evaluator(self, database: str | None = None) -> "FakeJudgeEvaluator":
        platform = load_platform(self.root / "configs/evalops.yaml")
        return FakeJudgeEvaluator(platform, self.root, Store(database or f"sqlite:///{self.root}/evalops.db"))


class FakeJudgeEvaluator(Evaluator):
    judge_client: FakeJudgeClient | None = None

    def make_judge(self, cache, before_live_call) -> Judge:
        self.judge_client = self.judge_client or FakeJudgeClient()
        return Judge(self.scoring.judge, self.base_dir, cache, client=self.judge_client,
                     before_live_call=before_live_call)


@pytest.fixture(autouse=True)
def reset_fake_app():
    ScriptedAdapter.calls = 0
    ScriptedAdapter.interrupt_after = None
    yield


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    root = tmp_path / "repo"
    (root / "configs/apps").mkdir(parents=True)
    (root / "prompts/judge").mkdir(parents=True)
    (root / "prompts/judge/correctness.txt").write_text("Grade the answer.")
    (root / "prompts/judge/claims.txt").write_text("Check the claims.")
    build_dataset(root)
    configs = {
        "evalops.yaml": {"dataset": "toy/v1:test", "scoring": "configs/scoring.yaml", "gate": "configs/gate.yaml",
                         "pricing": "configs/pricing.yaml", "release": "configs/release.yaml",
                         "cache_dir": "cache", "runs_dir": ".evalops/runs",
                         "execution": {"timeout_s": 5, "max_retries": 1, "concurrency": 1, "judge_concurrency": 1}},
        "scoring.yaml": {"judge": {"model": {"name": "fake-judge", "digest": JUDGE_DIGEST},
                                   "prompts": {"correctness": "prompts/judge/correctness.txt",
                                               "claims": "prompts/judge/claims.txt"},
                                   "options": {"temperature": 0, "seed": 7}, "max_attempts": 2}},
        "gate.yaml": GATE,
        "pricing.yaml": {"source": "test", "as_of": "2026-10",
                         "models": {"fake-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}},
    }
    for name, data in configs.items():
        (root / "configs" / name).write_text(yaml.safe_dump(data))
    # The judge model "is installed" with the pinned digest.
    monkeypatch.setattr(engine_module, "installed_digest", lambda name: JUDGE_DIGEST)
    return Project(root, monkeypatch)
