"""Sectioning and BM25 retrieval within one document."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
of on or our over so than that the their them then there these they this to under was were what when where
which who whom why will with would you your
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Section:
    number: int          # 1-based position in the document; what the model cites
    text: str


def split_sections(text: str, heading_max_words: int = 12) -> list[Section]:
    """Paragraph blocks, with short heading lines merged into the block that follows."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    merged: list[str] = []
    pending_heading = ""
    for block in blocks:
        is_heading = len(block.split()) <= heading_max_words and not block.rstrip().endswith((".", "!", "?", ":"))
        if is_heading:
            pending_heading = f"{pending_heading} {block}".strip() if pending_heading else block
            continue
        merged.append(f"{pending_heading}\n{block}" if pending_heading else block)
        pending_heading = ""
    if pending_heading:
        merged.append(pending_heading)
    return [Section(number=i + 1, text=t) for i, t in enumerate(merged)]


def bm25_top_k(question: str, sections: list[Section], k: int | None,
               k1: float = 1.5, b: float = 0.75) -> list[tuple[Section, float]]:
    """Top-k sections by BM25 score, returned in document order. k=None returns all."""
    docs = [tokenize(s.text) for s in sections]
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / max(n, 1)
    df = Counter(t for d in docs for t in set(d))
    query = tokenize(question)

    scored = []
    for section, tokens in zip(sections, docs):
        tf = Counter(tokens)
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(tokens) / avg_len))
        scored.append((section, score))

    if k is not None and k < n:
        keep = sorted(scored, key=lambda x: (-x[1], x[0].number))[:k]
    else:
        keep = scored
    return sorted(keep, key=lambda x: x[0].number)
