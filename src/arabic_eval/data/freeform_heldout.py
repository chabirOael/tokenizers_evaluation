"""Held-out free-form evaluation set drawn from CIDAR.

The free-form eval (``arabic_eval.tasks.freeform``) generates an answer for
every prompt of this set and scores it with reference-based metrics and LLM
judges. The set is declared as the ``freeform_prompts`` held-out set of
``configs/contamination/heldout_sets.yaml`` (``kind: file``): the contamination
scan then excludes the CIDAR training rows it came from (exact question hash)
and every other training record that overlaps it, and the pretraining-mix pool
filters against it by content.

``select_heldout`` is a pure function over records — testable on fixture rows;
``scripts/build_freeform_heldout.py`` wires the real corpora and the E5
embedder. Every filter reports its count in the manifest written next to the
JSONL, and the JSONL carries the CIDAR record id of every row so the set can be
traced back without re-running.

Why an embedding gate on top of the n-gram check: CIDAR and Bactrian-X both
descend from Alpaca. Measured 2026-09-17 on the real rows, 749 of 8 294 CIDAR
instructions are verbatim twins of a Bactrian-X instruction after
normalization and 1 797 share a 6-gram — the n-gram check catches those. The
same Alpaca instruction translated *differently* shares no n-gram and is
invisible to it, hence the multilingual-embedding near-duplicate gate
(``intfloat/multilingual-e5-large``, cosine between ``query:``-prefixed
mean-pooled embeddings), calibrated on the verbatim twins: the threshold is
the 5th percentile of the twins' nearest-training-neighbour cosine, so a gate
that would have caught 95 % of the known twins on its own is applied to the
paraphrase twins it is there for.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contamination import normalize_text
from .finetune_corpora import QARecord
from ..tokenizers.utils.arabic_text import contains_latin_letters

logger = logging.getLogger(__name__)

STRATA_LABELS = ("short", "medium", "long")

# texts -> [n, d] float32, L2-normalized rows
Embedder = Callable[[Sequence[str]], Any]


@dataclass
class SelectionConfig:
    n_examples: int = 250
    seed: int = 42
    min_ref_chars: int = 20
    max_ref_chars: int = 1500
    min_prompt_chars: int = 8
    prompt_ngram: int = 6
    reference_ngram: int = 8
    # None → calibrate on the n-gram twins (see module docstring); a float
    # fixes it. The calibrated value is clamped to the bounds.
    embed_threshold: Optional[float] = None
    embed_threshold_bounds: Tuple[float, float] = (0.80, 0.97)
    embed_calibration_quantile: float = 0.05
    n_strata: int = 3


@dataclass
class HeldoutRow:
    id: str
    prompt: str
    context: str
    reference: str
    stratum: str
    ref_chars: int
    max_train_cosine: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        d = {"id": self.id, "prompt": self.prompt}
        if self.context:
            d["context"] = self.context
        d["reference"] = self.reference
        d["stratum"] = self.stratum
        d["ref_chars"] = self.ref_chars
        if self.max_train_cosine is not None:
            d["max_train_cosine"] = round(float(self.max_train_cosine), 4)
        return d


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def prompt_key(rec: QARecord) -> str:
    """The normalized prompt side of a record: instruction plus its input."""
    return normalize_text(f"{rec.context} {rec.question}" if rec.context else rec.question)


def word_ngrams(normalized: str, n: int) -> set:
    w = normalized.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {f"p{int(q * 100):02d}": (None if (v := _quantile(values, q)) is None else round(v, 4))
            for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)}


def _largest_remainder(total: int, weights: Sequence[float]) -> List[int]:
    if total <= 0 or not weights:
        return [0] * len(weights)
    s = sum(weights)
    raw = [total * w / s for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def select_heldout(
    candidates: Sequence[QARecord],
    training: Sequence[QARecord],
    cfg: SelectionConfig,
    embedder: Optional[Embedder] = None,
    excluded_ids: Sequence[str] = (),
) -> Tuple[List[HeldoutRow], Dict[str, Any]]:
    """Pick ``cfg.n_examples`` rows of ``candidates`` (the CIDAR train split)
    that leak nothing into ``training`` (the other free-form corpora) or into
    the rest of ``candidates`` (which stays in training).

    Filters, in order, each counted in the returned manifest:
      1. ids in ``excluded_ids`` (the committed contamination list);
      2. Latin letters in prompt or reference (``clean_latin_rows`` predicate);
      3. a non-empty context (CIDAR has none; keeps the prompt shape uniform);
      4. reference length outside ``[min_ref_chars, max_ref_chars]``, prompt
         shorter than ``min_prompt_chars``;
      5. n-gram twins: the normalized prompt equals another record's, or shares
         a ``prompt_ngram`` with one, or the reference shares a
         ``reference_ngram`` with another record's answer — over training
         *and* the other candidates;
      6. embedding near-duplicates (only with an ``embedder``): the prompt's
         cosine to its nearest *other* training prompt is at or above the
         threshold (fixed, or calibrated on the twins of step 5).
    Survivors are stratified by reference length (``n_strata`` quantile
    bins) and drawn per stratum by largest remainder in a seeded permutation
    of the id-sorted list, a short stratum spilling its deficit to the others.
    """
    t0 = time.time()
    counts: Dict[str, int] = {"candidates": len(candidates), "training_records": len(training)}
    excluded = set(map(str, excluded_ids))
    by_id: Dict[str, QARecord] = {}
    for r in candidates:
        if r.id in by_id:
            raise ValueError(f"duplicate candidate id {r.id!r}")
        by_id[r.id] = r

    # -- 1-4: cheap per-record filters -----------------------------------
    alive: List[QARecord] = []
    drop = Counter()
    for r in candidates:
        if r.id in excluded:
            drop["excluded_id"] += 1
        elif contains_latin_letters(r.question) or contains_latin_letters(r.answer) or contains_latin_letters(r.context):
            drop["latin"] += 1
        elif r.context:
            drop["context_nonempty"] += 1
        elif not (cfg.min_ref_chars <= len(r.answer) <= cfg.max_ref_chars):
            drop["reference_length"] += 1
        elif len(r.question) < cfg.min_prompt_chars:
            drop["prompt_length"] += 1
        else:
            alive.append(r)
    counts.update({k: drop.get(k, 0) for k in ("excluded_id", "latin", "context_nonempty", "reference_length", "prompt_length")})

    # -- 5: n-gram / exact twins over training + every candidate -----------
    pool = list(training) + list(candidates)
    prompt_norm = {r.id: prompt_key(r) for r in candidates}
    prompt_exact = Counter(prompt_key(r) for r in pool)
    pgram = Counter()
    rgram = Counter()
    for r in pool:
        pgram.update(word_ngrams(prompt_key(r), cfg.prompt_ngram))
        rgram.update(word_ngrams(normalize_text(r.answer), cfg.reference_ngram))
    twins_prompt: set = set()
    twins_reference: set = set()
    for r in candidates:            # flagged over all candidates: the calibration positives
        pk = prompt_norm[r.id]
        if prompt_exact[pk] >= 2 or any(pgram[g] >= 2 for g in word_ngrams(pk, cfg.prompt_ngram)):
            twins_prompt.add(r.id)
        if any(rgram[g] >= 2 for g in word_ngrams(normalize_text(r.answer), cfg.reference_ngram)):
            twins_reference.add(r.id)
    counts["ngram_prompt_twin"] = sum(1 for r in alive if r.id in twins_prompt)
    counts["ngram_reference_twin"] = sum(1 for r in alive if r.id in twins_reference and r.id not in twins_prompt)
    alive = [r for r in alive if r.id not in twins_prompt and r.id not in twins_reference]

    # -- 6: embedding near-duplicates ---------------------------------------
    embed_info: Dict[str, Any] = {"applied": embedder is not None}
    max_cos: Dict[str, float] = {}
    nearest: Dict[str, str] = {}
    if embedder is not None:
        import numpy as np
        train_texts = [prompt_key(r) for r in pool]
        train_ids = [r.id for r in pool]
        cand_ids = [r.id for r in candidates]
        emb = np.asarray(embedder(train_texts), dtype=np.float32)
        if emb.shape[0] != len(pool):
            raise ValueError(f"embedder returned {emb.shape[0]} rows for {len(pool)} texts")
        pos_of = {rid: i for i, rid in enumerate(train_ids) if rid in by_id}   # candidates are the tail of pool
        cand_rows = np.stack([emb[pos_of[c]] for c in cand_ids])
        mc, nn_ = _max_cosine_excluding_self(cand_rows, emb, [pos_of[c] for c in cand_ids])
        for cid, m, j in zip(cand_ids, mc, nn_):
            max_cos[cid] = float(m)
            nearest[cid] = train_ids[j]
        positives = [max_cos[c] for c in twins_prompt]
        if cfg.embed_threshold is None:
            lo, hi = cfg.embed_threshold_bounds
            cal = _quantile(positives, cfg.embed_calibration_quantile)
            threshold = hi if cal is None else min(max(cal, lo), hi)
            embed_info["calibrated"] = cal is not None
        else:
            threshold = float(cfg.embed_threshold)
            embed_info["calibrated"] = False
        embed_info.update({
            "threshold": round(threshold, 4),
            "twins_for_calibration": len(positives),
            "twin_max_cosine_quantiles": _quantiles(positives),
            "alive_max_cosine_quantiles": _quantiles([max_cos[r.id] for r in alive]),
        })
        before = len(alive)
        dropped_emb = [r for r in alive if max_cos[r.id] >= threshold]
        alive = [r for r in alive if max_cos[r.id] < threshold]
        counts["embedding_near_duplicate"] = before - len(alive)

        def _pair(r: QARecord) -> Dict[str, Any]:
            n = nearest[r.id]
            nrec = next((x for x in pool if x.id == n), None)
            return {"id": r.id, "cosine": round(max_cos[r.id], 4), "prompt": r.question,
                    "nearest_id": n, "nearest_prompt": (nrec.question if nrec else None)}
        embed_info["kept_closest"] = [_pair(r) for r in sorted(alive, key=lambda r: -max_cos[r.id])[:8]]
        embed_info["dropped_farthest"] = [_pair(r) for r in sorted(dropped_emb, key=lambda r: max_cos[r.id])[:8]]
    else:
        counts["embedding_near_duplicate"] = 0
    counts["survivors"] = len(alive)

    # -- stratify by reference length, draw ---------------------------------
    lengths = sorted(len(r.answer) for r in alive)
    bounds = [_quantile(lengths, (i + 1) / cfg.n_strata) for i in range(cfg.n_strata - 1)] if lengths else []
    labels = list(STRATA_LABELS[:cfg.n_strata]) if cfg.n_strata <= len(STRATA_LABELS) else [f"s{i}" for i in range(cfg.n_strata)]

    def stratum_of(n: int) -> str:
        for i, b in enumerate(bounds):
            if n <= b:
                return labels[i]
        return labels[-1]

    strata: Dict[str, List[QARecord]] = {lab: [] for lab in labels}
    for r in alive:
        strata[stratum_of(len(r.answer))].append(r)
    quota = dict(zip(labels, _largest_remainder(min(cfg.n_examples, len(alive)), [1.0] * len(labels))))
    # water-fill: a stratum short of its quota hands the deficit to the others
    take: Dict[str, int] = {}
    remaining = min(cfg.n_examples, len(alive))
    open_labels = list(labels)
    while remaining > 0 and open_labels:
        share = _largest_remainder(remaining, [1.0] * len(open_labels))
        progressed = False
        for lab, s in zip(list(open_labels), share):
            room = len(strata[lab]) - take.get(lab, 0)
            got = min(s, room)
            if got:
                take[lab] = take.get(lab, 0) + got
                remaining -= got
                progressed = True
            if take.get(lab, 0) >= len(strata[lab]):
                open_labels.remove(lab)
        if not progressed:
            break
    rows: List[HeldoutRow] = []
    rng = random.Random(cfg.seed)
    for lab in labels:
        ordered = sorted(strata[lab], key=lambda r: r.id)
        rng.shuffle(ordered)
        for r in ordered[: take.get(lab, 0)]:
            rows.append(HeldoutRow(id=r.id, prompt=r.question, context=r.context, reference=r.answer,
                                   stratum=lab, ref_chars=len(r.answer), max_train_cosine=max_cos.get(r.id)))
    rows.sort(key=lambda x: x.id)
    counts["selected"] = len(rows)
    if len(rows) < cfg.n_examples:
        logger.warning("free-form held-out: only %d of the requested %d survive the filters", len(rows), cfg.n_examples)

    manifest = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(cfg),
        "counts": counts,
        "embedding": embed_info,
        "strata": {
            "boundaries_ref_chars": [None if b is None else round(b, 1) for b in bounds],
            "available": {lab: len(strata[lab]) for lab in labels},
            "equal_quota": quota,
            "selected": {lab: sum(1 for x in rows if x.stratum == lab) for lab in labels},
        },
        "selected_ids": [x.id for x in rows],
        "seconds": round(time.time() - t0, 1),
    }
    return rows, manifest


def _max_cosine_excluding_self(cand: Any, train: Any, self_pos: Sequence[int], chunk: int = 2048):
    """Per candidate row: max cosine over ``train`` rows except its own copy,
    and that row's index. Uses torch on the GPU when available."""
    import numpy as np
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(np.ascontiguousarray(train)).to(dev)
        best = np.empty(len(cand), dtype=np.float32)
        arg = np.empty(len(cand), dtype=np.int64)
        for s in range(0, len(cand), chunk):
            C = torch.from_numpy(np.ascontiguousarray(cand[s:s + chunk])).to(dev)
            sim = C @ T.T
            idx = torch.arange(sim.shape[0], device=dev)
            sim[idx, torch.tensor(self_pos[s:s + chunk], device=dev)] = -2.0
            m, a = sim.max(dim=1)
            best[s:s + chunk] = m.float().cpu().numpy()
            arg[s:s + chunk] = a.cpu().numpy()
        return best, arg
    except ImportError:                      # numpy fallback (tests without torch)
        best = np.empty(len(cand), dtype=np.float32)
        arg = np.empty(len(cand), dtype=np.int64)
        for s in range(0, len(cand), chunk):
            sim = cand[s:s + chunk] @ train.T
            for i, p in enumerate(self_pos[s:s + chunk]):
                sim[i, p] = -2.0
            best[s:s + chunk] = sim.max(axis=1)
            arg[s:s + chunk] = sim.argmax(axis=1)
        return best, arg


# --------------------------------------------------------------------------
# E5 embedder (the real one; lazy torch / transformers)
# --------------------------------------------------------------------------

class E5Embedder:
    """``intfloat/multilingual-e5-large`` mean-pooled, L2-normalized, with the
    ``query:`` prefix on every text (the symmetric-task convention)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: Optional[str] = None,
                 batch_size: int = 128, max_length: int = 128) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self._model.to(self.device).eval()

    def __call__(self, texts: Sequence[str]):
        import numpy as np
        import torch
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-sorted batches
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                batch = self._tok(["query: " + texts[i] for i in idx], padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self._model(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                out.append((idx, pooled.cpu().numpy()))
        emb = np.empty((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, arr in out:
            emb[idx] = arr
        return emb


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def write_heldout_jsonl(rows: Sequence[HeldoutRow], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def load_freeform_heldout(path: Path | str) -> List[Dict[str, Any]]:
    """The raw rows (``id``, ``prompt``, ``context``, ``reference``, ``stratum``)
    exactly as written — no normalization; this is what the task generates from."""
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("prompt"):
                raise ValueError(f"{path}:{lineno}: no 'prompt'")
            rid = str(obj.get("id", lineno))
            if rid in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {rid!r}")
            seen.add(rid)
            rows.append({"id": rid, "prompt": obj["prompt"], "context": obj.get("context") or "",
                         "reference": obj.get("reference") or "", "stratum": obj.get("stratum") or "",
                         "ref_chars": obj.get("ref_chars")})
    return rows
