"""Build the `policies` evaluation dataset from RepliQA.

RepliQA (ServiceNow, CC BY 4.0) pairs made-up documents with questions written
by human annotators. Because the documents are fictional, a model cannot answer
from memory: it has to read the document. About one question in five is
deliberately unanswerable ("The answer is not found in the document."), which
is what we need to measure unsupported claims.

We keep the "Company Policies" topic so the system under test is a coherent
product (a policy document assistant), sample whole documents with a fixed
seed, and split them so no document appears in two splits:

    test   50 documents, 250 questions   the release gate runs on this
    dev    15 documents,  75 questions   for prompt and judge iteration
    smoke  first 10 test documents       a fast subset of test

A published version is immutable. Every file is hashed into manifest.json and
`evalops dataset verify` fails if anything changes; edits go into a new
version directory instead.

Usage:
    python scripts/build_dataset.py --version v1
    python scripts/build_dataset.py --version v1 --parquet /path/to/repliqa_0.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from evalops.hashing import sha256_json

SOURCE = "ServiceNow/repliqa"
SOURCE_SPLIT = "repliqa_0"
# Pinned revision of the Hub's parquet conversion, so the build is reproducible.
SOURCE_REVISION = "7ba2f15e1b95dbece63e03935c63c791a20f00e6"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE}/resolve/{SOURCE_REVISION}/default/{SOURCE_SPLIT}/0000.parquet"
TOPIC = "Company Policies"
UNANSWERABLE = "The answer is not found in the document."

SEED = 2026
N_TEST_DOCS = 50
N_DEV_DOCS = 15
N_SMOKE_DOCS = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_source(parquet: str | None, cache_dir: Path) -> pd.DataFrame:
    if parquet:
        return pd.read_parquet(parquet)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"repliqa-{SOURCE_SPLIT}-{SOURCE_REVISION[:12]}.parquet"
    if not local.exists():
        print(f"Downloading {SOURCE_URL}", file=sys.stderr)
        urllib.request.urlretrieve(SOURCE_URL, local)
    return pd.read_parquet(local)


def build(df: pd.DataFrame, out: Path, version: str) -> dict:
    df = df[df.document_topic == TOPIC]
    doc_ids = sorted(df.document_id.unique())
    random.Random(SEED).shuffle(doc_ids)
    test_ids = doc_ids[:N_TEST_DOCS]
    dev_ids = doc_ids[N_TEST_DOCS:N_TEST_DOCS + N_DEV_DOCS]

    docs_by_id = df.drop_duplicates("document_id").set_index("document_id")
    documents = []
    for doc_id in sorted(test_ids + dev_ids):
        text = docs_by_id.loc[doc_id, "document_extracted"].strip()
        documents.append({"document_id": doc_id, "title": text.split("\n", 1)[0].strip(), "text": text})

    def cases_for(ids: list[str]) -> list[dict]:
        rows = []
        for doc_id in ids:
            for r in df[df.document_id == doc_id].sort_values("question_id").itertuples():
                answerable = r.answer.strip() != UNANSWERABLE
                # Every answerable question in RepliQA is answered from a passage of the
                # document, so the answer must cite it. The categories are kept separate
                # so later versions can add answerable questions that need no citation.
                categories = ["answerable", "citation_required"] if answerable else ["unanswerable"]
                rows.append({
                    "case_id": r.question_id,
                    "input": {"document_id": doc_id, "question": r.question.strip()},
                    "expected": {
                        "answerable": answerable,
                        "reference": r.answer.strip() if answerable else None,
                        # The supporting passage, close to verbatim from the document.
                        # Ground truth for retrieval metrics.
                        "evidence": r.long_answer.strip() if answerable else None,
                    },
                    "categories": categories,
                })
        return rows

    splits = {
        "test": cases_for(test_ids),
        "dev": cases_for(dev_ids),
        "smoke": cases_for(test_ids[:N_SMOKE_DOCS]),
    }

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "documents.jsonl", documents)
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)

    files = {p.name: sha256_file(p) for p in sorted(out.glob("*.jsonl"))}
    manifest = {
        "name": "policies",
        "version": version,
        "schema_version": 1,
        "description": "Questions about company policy documents, including unanswerable ones. "
                       "Built from RepliQA 'Company Policies' documents.",
        "created": date.today().isoformat(),
        "source": {
            "dataset": SOURCE,
            "split": SOURCE_SPLIT,
            "revision": SOURCE_REVISION,
            "topic": TOPIC,
            "license": "CC BY 4.0",
            "citation": "Monteiro et al., RepliQA: A Question-Answering Dataset for Benchmarking LLMs "
                        "on Unseen Reference Content, NeurIPS 2024 Datasets and Benchmarks.",
        },
        "build": {"script": "scripts/build_dataset.py", "seed": SEED},
        "splits": {
            name: {
                "file": f"{name}.jsonl",
                "documents": len({r["input"]["document_id"] for r in rows}),
                "cases": len(rows),
                "categories": {c: sum(c in r["categories"] for r in rows)
                               for c in ("answerable", "unanswerable", "citation_required")},
            }
            for name, rows in splits.items()
        },
        "files": files,
        # One hash for the whole version: any change to any file changes it.
        "dataset_hash": sha256_json(files),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="dataset version directory, e.g. v1")
    parser.add_argument("--parquet", help="local copy of the RepliQA parquet file (skips the download)")
    parser.add_argument("--root", default="datasets/policies", help="where dataset versions live")
    parser.add_argument("--force", action="store_true", help="overwrite an existing version (only before it is published)")
    args = parser.parse_args()

    out = Path(args.root) / args.version
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists. Published versions are immutable; build a new version instead.")
    manifest = build(load_source(args.parquet, Path(".cache")), out, args.version)
    print(json.dumps(manifest["splits"], indent=2))
    print("dataset_hash", manifest["dataset_hash"])


if __name__ == "__main__":
    main()
