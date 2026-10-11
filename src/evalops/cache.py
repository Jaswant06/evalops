"""Version-aware cache of model outputs.

The cache stores raw outputs (application responses and judge responses),
never scores or decisions, so cached outputs can always be rescored by newer
scoring logic.

An entry is keyed by the version fingerprint (model name and digest, prompt
hash, generation parameters, code version), the item id and a hash of the
exact input. On every read the entry's recorded identity is checked against
the requester's; an entry written by another model version is rejected, never
silently reused.

Storage is append-only JSON lines, one file per version fingerprint, so a
crash mid-run loses at most the line being written, and the files can be
committed to git for replay in CI.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .hashing import canonical_json, sha256_text

CacheMode = Literal["use", "refresh", "replay", "off"]


class CacheMiss(RuntimeError):
    """Replay mode needed an output that was never recorded."""


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    rejected: int = 0
    expired: int = 0
    writes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "rejected": self.rejected, "expired": self.expired,
                "writes": self.writes}


@dataclass
class CacheEntry:
    key: str
    fingerprint: str
    item_id: str
    input_hash: str
    identity: dict[str, Any]
    recorded_at: str
    payload: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_days(recorded_at: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(recorded_at)).total_seconds() / 86400


@dataclass
class ResponseCache:
    root: Path
    kind: str                      # "app" or "judge"
    mode: CacheMode = "use"
    write: bool = True             # False: never add recordings (e.g. a live run that must not change them)
    max_age_days: float | None = None  # entries older than this count as missing
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._loaded: dict[str, dict[str, CacheEntry]] = {}
        self.used: set[str] = set()   # keys served by get(), for pruning

    @staticmethod
    def make_key(fingerprint: str, item_id: str, input_hash: str) -> str:
        return sha256_text(f"{fingerprint}|{item_id}|{input_hash}")

    def _path(self, fingerprint: str) -> Path:
        return self.root / self.kind / f"{fingerprint[:16]}.jsonl"

    def _entries(self, fingerprint: str) -> dict[str, CacheEntry]:
        if fingerprint not in self._loaded:
            entries: dict[str, CacheEntry] = {}
            path = self._path(fingerprint)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        entry = CacheEntry(**raw)
                    except (json.JSONDecodeError, TypeError):
                        continue  # a torn final line from an interrupted write
                    entries[entry.key] = entry  # last write wins
            self._loaded[fingerprint] = entries
        return self._loaded[fingerprint]

    def get(self, fingerprint: str, item_id: str, input_hash: str, identity: dict[str, Any]) -> CacheEntry | None:
        """Return a valid entry, or None. In replay mode a miss raises CacheMiss."""
        if self.mode in ("off", "refresh"):
            return None
        key = self.make_key(fingerprint, item_id, input_hash)
        with self._lock:
            entry = self._entries(fingerprint).get(key)
            if entry is not None and not self._matches(entry, fingerprint, item_id, input_hash, identity):
                self.stats.rejected += 1
                entry = None
            if entry is not None and self.max_age_days is not None and _age_days(entry.recorded_at) > self.max_age_days:
                self.stats.expired += 1
                entry = None
            if entry is None:
                self.stats.misses += 1
            else:
                self.stats.hits += 1
                self.used.add(key)
        if entry is None and self.mode == "replay":
            raise CacheMiss(
                f"No recorded {self.kind} output for {item_id} under version {fingerprint[:12]}. "
                f"Evaluate it with --mode auto or --mode live to record it, then commit {self._path(fingerprint)}."
            )
        return entry

    @staticmethod
    def _matches(entry: CacheEntry, fingerprint: str, item_id: str, input_hash: str, identity: dict) -> bool:
        return (entry.fingerprint == fingerprint and entry.item_id == item_id
                and entry.input_hash == input_hash
                and canonical_json(entry.identity) == canonical_json(identity))

    def put(self, fingerprint: str, item_id: str, input_hash: str, identity: dict[str, Any],
            payload: dict[str, Any]) -> CacheEntry:
        entry = CacheEntry(
            key=self.make_key(fingerprint, item_id, input_hash),
            fingerprint=fingerprint, item_id=item_id, input_hash=input_hash,
            identity=identity, recorded_at=now_iso(), payload=payload,
        )
        if self.mode in ("off", "replay") or not self.write:
            return entry
        line = json.dumps(entry.__dict__, ensure_ascii=False, sort_keys=True)
        with self._lock:
            path = self._path(fingerprint)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            self._entries(fingerprint)[entry.key] = entry
            self.stats.writes += 1
        return entry


def prune(root: Path, kind: str, keep: set[str], keep_item_prefixes: tuple[str, ...] = ()) -> dict[str, int]:
    """Rewrite the cache files of one kind, keeping only the given keys (and items with the given prefixes)."""
    kept = removed = 0
    for path in sorted((root / kind).glob("*.jsonl")):
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue
            if raw.get("key") in keep or str(raw.get("item_id", "")).startswith(keep_item_prefixes):
                lines.append(line)
            else:
                removed += 1
        kept += len(lines)
        if lines:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            path.unlink()
    return {"kept": kept, "removed": removed}
