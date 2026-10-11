"""How much should the LLM judge be trusted?

Two independent checks:

1. Against human labels. `evalops label` shows a stratified sample of answers
   (question, reference, answer) without the judge's verdict and records a
   human verdict; `evalops judge-agreement` reports raw agreement, Cohen's
   kappa and the confusion matrix.
2. Against the judge-free signal. Token F1 with the reference answer is crude
   (a correct paraphrase can score low) but it never hallucinates, so large
   disagreements between it and the judge are worth reading. Available for
   every run, with no labelling.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .metrics import CaseRecord

VERDICTS = ("correct", "partially_correct", "incorrect")


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(records: list[CaseRecord], labels: dict[str, str]) -> dict:
    pairs = [(labels[r.case_id], r.verdict) for r in records if r.case_id in labels and r.verdict]
    if not pairs:
        return {"n": 0}
    confusion = {h: {j: sum(1 for a, b in pairs if a == h and b == j) for j in VERDICTS} for h in VERDICTS}
    # Binary view: is the answer acceptable (correct) or not?
    binary = [(a == "correct", b == "correct") for a, b in pairs]
    lenient = sum(1 for h, j in binary if j and not h)   # judge says correct, human does not
    strict = sum(1 for h, j in binary if h and not j)
    return {
        "n": len(pairs),
        "agreement": sum(a == b for a, b in pairs) / len(pairs),
        "kappa": cohen_kappa(pairs),
        "binary_agreement": sum(h == j for h, j in binary) / len(binary),
        "judge_more_lenient": lenient,
        "judge_more_strict": strict,
        "confusion_human_rows_judge_columns": confusion,
    }


def f1_consistency(records: list[CaseRecord], high: float = 0.6, low: float = 0.2) -> dict:
    judged = [r for r in records if r.verdict and r.reference_f1 is not None]
    hi = [r for r in judged if r.reference_f1 >= high]
    lo = [r for r in judged if r.reference_f1 < low]
    return {
        "judged": len(judged),
        "high_overlap": len(hi),
        "high_overlap_judged_correct": sum(r.verdict == "correct" for r in hi) / len(hi) if hi else None,
        "low_overlap": len(lo),
        "low_overlap_judged_correct": sum(r.verdict == "correct" for r in lo) / len(lo) if lo else None,
        "suspicious": [r.case_id for r in hi if r.verdict == "incorrect"],
    }


def sample_for_labelling(records: list[CaseRecord], n: int, seed: int = 0) -> list[CaseRecord]:
    """Stratified by the judge's verdict so rare verdicts are represented, shuffled so the order hides them."""
    rng = random.Random(seed)
    by_verdict: dict[str, list[CaseRecord]] = {}
    for r in records:
        if r.verdict:
            by_verdict.setdefault(r.verdict, []).append(r)
    picked: list[CaseRecord] = []
    per = max(1, n // max(1, len(by_verdict)))
    for group in by_verdict.values():
        picked += rng.sample(group, min(per, len(group)))
    rest = [r for r in records if r.verdict and r not in picked]
    picked += rng.sample(rest, min(n - len(picked), len(rest))) if len(picked) < n else []
    rng.shuffle(picked)
    return picked[:n]


def load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["case_id"]] = row["verdict"]
    return labels


def run_probes(evaluator, run_id: str, n: int = 30, seed: int = 0) -> dict:
    """Feed the judge answers that are wrong by construction and count how many it catches.

    For n answerable cases the app answered correctly (per the judge), three probes:
      cross_document   the answer to a question about a *different* document, with this
                       case's citations: its claims must be unsupported by this document
      wrong_question   the answer to a different question about the *same* document: it
                       is grounded, but it does not answer this question, so it must not
                       be judged correct
      uncited          the original answer with its citations removed: no claim can be
                       supported by the cited passages
    No human labels are needed: the right verdict is known from how the probe was built.
    """
    from .cache import ResponseCache
    from .datasets import load as load_dataset
    from .judge import Judge, summarize_claims
    from .llm import installed_digest
    from .schemas import CaseExecution

    store = evaluator.store
    run = store.get_run(run_id)
    if run is None or run["status"] != "completed":
        raise ValueError(f"Run {run_id} is not a completed run")
    dataset = load_dataset(run["dataset_ref"], evaluator.base_dir / "datasets")
    cases = {c.case_id: c for c in dataset.cases}
    rows = store.get_cases(run_id)
    responses = {cid: CaseExecution.model_validate(row["execution"]["execution"]).response for cid, row in rows.items()}
    records = {cid: CaseRecord.model_validate(row["record"]) for cid, row in rows.items()}

    clean = sorted(cid for cid, r in records.items()
                   if r.answerable and r.verdict == "correct" and r.claims_total
                   and r.claims_supported_document == r.claims_total and r.n_citations)
    rng = random.Random(seed)
    picked = rng.sample(clean, min(n, len(clean)))

    model = evaluator.scoring.judge.model

    def check_digest():
        if installed_digest(model.name) != model.digest:
            raise RuntimeError(f"judge model {model.name} does not match the pinned digest {model.digest}")

    cache = ResponseCache(evaluator.base_dir / evaluator.platform.cache_dir, "judge", "use")
    judge = Judge(evaluator.scoring.judge, evaluator.base_dir, cache, before_live_call=check_digest)
    out = {"cross_document": [], "wrong_question": [], "uncited": []}
    for cid in picked:
        case, response = cases[cid], responses[cid]
        doc = dataset.documents[case.input.document_id]
        others_doc = [o for o in clean if cases[o].input.document_id != case.input.document_id]
        same_doc = [o for o in records if o != cid and records[o].answered and records[o].answerable
                    and cases[o].input.document_id == case.input.document_id]

        donor = rng.choice(others_doc)
        swapped = response.model_copy(update={"answer": responses[donor].answer})
        j = summarize_claims(judge.claims(case.model_copy(update={"case_id": f"probe-x-{cid}"}), swapped, doc))
        if j["claims_total"]:
            out["cross_document"].append(j["claims_supported_document"] < j["claims_total"])

        if same_doc:
            donor = rng.choice(same_doc)
            off = response.model_copy(update={"answer": responses[donor].answer})
            j = judge.correctness(case.model_copy(update={"case_id": f"probe-q-{cid}"}), off)
            if j.status == "ok":
                out["wrong_question"].append(j.output["verdict"] != "correct")

        uncited = response.model_copy(update={"citations": []})
        j = summarize_claims(judge.claims(case.model_copy(update={"case_id": f"probe-u-{cid}"}), uncited, doc))
        if j["claims_total"]:
            out["uncited"].append(j["claims_supported_cited"] == 0)

    return {
        "run_id": run_id, "probes_per_kind": len(picked),
        **{f"{kind}_caught": f"{sum(v)}/{len(v)}" for kind, v in out.items()},
        **{f"{kind}_detection_rate": (sum(v) / len(v) if v else None) for kind, v in out.items()},
        "cache": cache.stats.as_dict(),
    }
