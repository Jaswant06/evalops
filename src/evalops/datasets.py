"""Versioned evaluation datasets.

A dataset version is a directory with a manifest.json listing the sha256 of
every file. Loading verifies the hashes and validates every row against the
schema before any case runs, so an evaluation can never silently use edited or
malformed test data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .hashing import sha256_file, sha256_json
from .schemas import Document, TestCase

SUPPORTED_SCHEMA_VERSIONS = {1}


class DatasetError(RuntimeError):
    """The dataset is missing, modified, or does not match the schema."""


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    split: str
    dataset_hash: str
    cases: list[TestCase]
    documents: dict[str, Document]

    @property
    def ref(self) -> str:
        return f"{self.name}/{self.version}:{self.split}"

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            for c in case.categories:
                counts[c] = counts.get(c, 0) + 1
        return dict(sorted(counts.items()))


def parse_ref(ref: str) -> tuple[str, str, str]:
    """'policies/v1:test' -> ('policies', 'v1', 'test')."""
    try:
        name_version, split = ref.split(":")
        name, version = name_version.split("/")
    except ValueError:
        raise DatasetError(f"Bad dataset reference '{ref}'. Expected name/version:split, e.g. policies/v1:test") from None
    return name, version, split


def verify(directory: Path) -> dict:
    """Check every file against the manifest. Returns the manifest."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise DatasetError(f"No manifest.json in {directory}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise DatasetError(f"{directory}: unsupported schema_version {manifest.get('schema_version')!r}")

    expected_files = manifest["files"]
    present = {p.name for p in directory.glob("*.jsonl")}
    if extra := present - set(expected_files):
        raise DatasetError(f"{directory}: files not in the manifest: {sorted(extra)}")
    for name, digest in expected_files.items():
        path = directory / name
        if not path.exists():
            raise DatasetError(f"{directory}: missing {name}")
        actual = sha256_file(path)
        if actual != digest:
            raise DatasetError(
                f"{directory / name} does not match its manifest hash "
                f"(expected {digest[:12]}, found {actual[:12]}). Published dataset versions are "
                f"immutable: revert the change or build a new version."
            )
    if sha256_json(expected_files) != manifest["dataset_hash"]:
        raise DatasetError(f"{directory}: dataset_hash does not match the file list")
    return manifest


def _read_rows(path: Path, model):
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"{path}:{lineno} does not match the schema:\n{exc}") from None
    return rows


def load(ref: str, root: Path = Path("datasets")) -> Dataset:
    name, version, split = parse_ref(ref)
    directory = root / name / version
    manifest = verify(directory)
    if split not in manifest["splits"]:
        raise DatasetError(f"{name}/{version} has no split '{split}'. Splits: {sorted(manifest['splits'])}")

    documents = {d.document_id: d for d in _read_rows(directory / "documents.jsonl", Document)}
    cases = _read_rows(directory / manifest["splits"][split]["file"], TestCase)

    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise DatasetError(f"{ref}: duplicate case_id {case.case_id}")
        seen.add(case.case_id)
        if case.input.document_id not in documents:
            raise DatasetError(f"{ref}: case {case.case_id} refers to unknown document {case.input.document_id}")
        if case.expected.answerable and not case.expected.reference:
            raise DatasetError(f"{ref}: answerable case {case.case_id} has no reference answer")

    return Dataset(name=name, version=version, split=split, dataset_hash=manifest["dataset_hash"],
                   cases=cases, documents=documents)
