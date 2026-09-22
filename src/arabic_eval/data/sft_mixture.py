"""Ratio-controlled composition of a QA phase's training set (``PhaseConfig.mixture``).

A phase with a ``mixture`` trains on exactly ``total_examples`` records
drawn from its ``datasets``:

1. **Category quotas.** ``shares`` (over ``extractive`` / ``mcq`` /
   ``free_form`` — the category of a corpus is fixed by its loader, see
   ``CORPUS_CATEGORY``) are turned into integer counts by largest
   remainder, so they sum to ``total_examples`` exactly.
2. **Within a category.** The quota is split across the listed corpora by
   *water-filling*: every corpus gets its weighted share, a corpus whose
   pool is smaller than its share is capped at the pool and the remainder
   is re-split among the others. ``within_category: equal`` weights every
   corpus 1, ``proportional`` weights it by pool size, and an explicit
   ``weights`` map overrides both. A category whose pools cannot fill the
   quota is an error naming the numbers and the largest ``total_examples``
   these shares admit — unless ``upsample: true``, in which case the
   residual is re-split by the same weights and those corpora repeat a
   further seeded permutation.
3. **The draw.** Every corpus is walked in a seeded permutation and
   tokenized as it goes; a record the answer-only masking drops (truncation
   ate the answer) does not count, and with ``drop_truncated_answers`` a
   record whose full text hits ``max_length`` (its answer is cut) is
   skipped too. The walk stops when the quota is met, so quotas count
   **kept** records and the ratio is exact after truncation; a corpus
   that runs dry (its whole pool was allocated and some of it was
   dropped) hands its deficit to the other corpora of the category. Only
   the records used are tokenized.

The composed set is shuffled (seeded) and handed to the phase's collator.
``compose_mixture`` returns the encodings and a manifest with, per corpus
and per category, the pool sizes, quota, drawn / dropped / kept counts,
loss tokens (the answer-token share is what the loss actually weights —
a free-form answer is one to two orders of magnitude longer than an MCQ
letter) and the pinned revision; the ids drawn are listed so the set can
be reconstructed without re-running.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from torch.utils.data import DataLoader

from ..config import CORPUS_CATEGORY, MixtureConfig
from ..tokenizers.base import BaseTokenizer
from .collation import get_collator
from .finetune_corpora import (
    PINNED_REVISIONS,
    TEMPLATE_VERSION,
    QARecord,
    _QATokenizedDataset,
    filter_latin_records,
    load_corpus,
    tokenize_record,
)

logger = logging.getLogger(__name__)


class MixtureShortfallError(ValueError):
    """A category's corpora cannot supply its quota (and ``upsample`` is off)."""


# --------------------------------------------------------------------------
# Quota arithmetic
# --------------------------------------------------------------------------

def largest_remainder(targets: Mapping[str, float], total: int) -> Dict[str, int]:
    """Integer counts summing to ``total`` that are closest to the real-valued
    ``targets`` (Hamilton / largest-remainder apportionment). ``targets``
    are rescaled to ``total``; ties break on key order for determinism."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    keys = list(targets)
    if not keys:
        if total:
            raise ValueError("no keys to apportion a positive total over")
        return {}
    weight_sum = float(sum(targets[k] for k in keys))
    if weight_sum <= 0:
        raise ValueError("targets must have a positive sum")
    raw = {k: total * targets[k] / weight_sum for k in keys}
    floors = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(floors.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - floors[k]), keys.index(k)))
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def category_quotas(shares: Mapping[str, float], total_examples: int) -> Dict[str, int]:
    """``shares × total_examples`` rounded so the quotas sum exactly."""
    return largest_remainder(dict(shares), total_examples)


def water_fill(quota: int, capacities: Mapping[str, int], weights: Mapping[str, float]) -> Tuple[Dict[str, int], int]:
    """Split ``quota`` over corpora by ``weights``, capping each at its
    capacity and re-splitting the remainder among the uncapped ones.

    Returns ``(allocation, shortfall)``: ``shortfall`` is what no corpus
    could absorb (0 when the quota fits).
    """
    alloc = {k: 0 for k in capacities}
    remaining = quota
    active = [k for k in capacities if weights.get(k, 0) > 0]
    while remaining > 0 and active:
        tentative = largest_remainder({k: weights[k] for k in active}, remaining)
        capped = [k for k in active if tentative[k] >= capacities[k] - alloc[k]]
        if not capped:
            for k in active:
                alloc[k] += tentative[k]
            remaining = 0
            break
        for k in capped:
            room = capacities[k] - alloc[k]
            alloc[k] += room
            remaining -= room
            active.remove(k)
    return alloc, remaining


def within_category_weights(mixture: MixtureConfig, names: Sequence[str], capacities: Mapping[str, int]) -> Dict[str, float]:
    """Weights the within-category split uses: explicit ``weights`` for the
    corpora that have one, else 1 (``equal``) or the pool size (``proportional``)."""
    out: Dict[str, float] = {}
    for n in names:
        if mixture.weights is not None and n in mixture.weights:
            out[n] = float(mixture.weights[n])
        elif mixture.within_category == "proportional":
            out[n] = float(capacities[n])
        else:
            out[n] = 1.0
    return out


def max_total_at_shares(shares: Mapping[str, float], category_capacity: Mapping[str, int], batch_size: int = 1) -> int:
    """Largest ``total_examples`` (a multiple of ``batch_size``) these shares
    admit without upsampling: ``min_cat(capacity_cat / share_cat)``."""
    bound = min(int(math.floor(category_capacity[c] / shares[c])) for c in shares)
    return (bound // batch_size) * batch_size


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

def plan_mixture(
    mixture: MixtureConfig,
    datasets: Sequence[str],
    capacities: Mapping[str, int],
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Quotas per category and per corpus for the given pool sizes — no
    tokenization. Raises ``MixtureShortfallError`` unless ``upsample``.

    Returns ``{"category_quotas", "allocation", "residual", "weights",
    "max_total_examples_at_these_shares"}`` where ``residual[name]`` is the
    part of a corpus' allocation beyond its pool (only with ``upsample``).
    """
    by_cat: Dict[str, List[str]] = {}
    for n in datasets:
        by_cat.setdefault(CORPUS_CATEGORY[n], []).append(n)
    quotas = category_quotas(mixture.shares, mixture.total_examples)
    cat_capacity = {c: sum(capacities[n] for n in by_cat.get(c, [])) for c in mixture.shares}
    max_total = max_total_at_shares(mixture.shares, cat_capacity, batch_size)

    allocation: Dict[str, int] = {}
    residual: Dict[str, int] = {}
    weights_all: Dict[str, float] = {}
    for cat, quota in quotas.items():
        names = by_cat.get(cat, [])
        weights = within_category_weights(mixture, names, capacities)
        weights_all.update(weights)
        alloc, short = water_fill(quota, {n: capacities[n] for n in names}, weights)
        if short > 0:
            if not mixture.upsample:
                pools = ", ".join(f"{n} {capacities[n]}" for n in names)
                raise MixtureShortfallError(
                    f"mixture: category {cat!r} needs {quota} examples "
                    f"({mixture.shares[cat]:.0%} of {mixture.total_examples}) but its corpora hold "
                    f"{cat_capacity[cat]} ({pools}); lower total_examples to at most {max_total} "
                    f"at these shares, change the shares, or set mixture.upsample: true"
                )
            extra = largest_remainder(weights, short)
            for n, k in extra.items():
                alloc[n] += k
                residual[n] = k
        allocation.update(alloc)
    return {
        "category_quotas": quotas,
        "allocation": allocation,
        "residual": residual,
        "weights": weights_all,
        "max_total_examples_at_these_shares": max_total,
        "category_capacity": cat_capacity,
    }


def _permutation(records: Sequence[QARecord], seed: int, name: str, pass_index: int) -> List[QARecord]:
    """A seeded, corpus-specific permutation; sorted by id first so the
    draw does not depend on the loader's row order."""
    ordered = sorted(records, key=lambda r: r.id)
    random.Random(f"{seed}:{name}:{pass_index}").shuffle(ordered)
    return ordered


class CorpusDrawer:
    """Resumable walk over seeded permutations of one corpus, tokenizing
    as it goes. ``draw(n)`` keeps up to ``n`` more encodings and returns
    how many it kept; it comes back short only when the corpus is
    exhausted (``exhausted`` is then set) — with ``upsample`` a further
    permutation is started instead, and a pass that keeps nothing raises.
    """

    def __init__(self, name: str, records: Sequence[QARecord], tokenizer: BaseTokenizer,
                 max_length: int, loss_target: str, mixture: MixtureConfig) -> None:
        self.name = name
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_target = loss_target
        self.mixture = mixture
        self.kept: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.drawn = 0
        self.dropped_truncation = 0
        self.dropped_cut_answer = 0
        self.loss_tokens = 0
        self.passes = 0
        self.exhausted = not self.records
        self._perm: List[QARecord] = []
        self._pos = 0
        self._kept_this_pass = 0

    def _next_pass(self) -> None:
        if self.passes > 0 and self._kept_this_pass == 0:
            raise MixtureShortfallError(
                f"mixture: no record of corpus {self.name!r} survives tokenization at max_length "
                f"{self.max_length} ({self.dropped_truncation} dropped: truncation ate the answer, "
                f"{self.dropped_cut_answer} dropped: answer cut); raise max_length"
            )
        self._perm = _permutation(self.records, self.mixture.seed, self.name, self.passes)
        self._pos = 0
        self._kept_this_pass = 0
        self.passes += 1

    def draw(self, n: int) -> int:
        got = 0
        while got < n and not self.exhausted:
            if self._pos >= len(self._perm):
                if self.passes > 0 and not self.mixture.upsample:
                    self.exhausted = True
                    break
                self._next_pass()
                if not self._perm:
                    self.exhausted = True
                    break
            rec = self._perm[self._pos]
            self._pos += 1
            self.drawn += 1
            entry, loss_tokens, truncated = tokenize_record(rec, self.tokenizer, self.max_length, self.loss_target)
            if entry is None:
                self.dropped_truncation += 1
                continue
            if truncated and self.mixture.drop_truncated_answers:
                self.dropped_cut_answer += 1
                continue
            self.kept.append(entry)
            self.ids.append(rec.id)
            self.loss_tokens += loss_tokens
            self._kept_this_pass += 1
            got += 1
        if not self.exhausted and self._pos >= len(self._perm) and not self.mixture.upsample:
            self.exhausted = True
        return got

    def stats(self) -> Dict[str, Any]:
        return {
            "drawn": self.drawn,
            "dropped_truncation": self.dropped_truncation,
            "dropped_cut_answer": self.dropped_cut_answer,
            "kept": len(self.kept),
            "repeated": len(self.kept) - len(set(self.ids)),
            "loss_tokens": self.loss_tokens,
            "passes": self.passes,
            "ids": list(self.ids),
        }


def draw_category(
    cat: str,
    quota: int,
    allocation: Mapping[str, int],
    weights: Mapping[str, float],
    drawers: Mapping[str, CorpusDrawer],
    mixture: MixtureConfig,
) -> None:
    """Fill a category's quota. Each corpus first draws its planned
    allocation; a corpus that runs out (its pool lost records to
    truncation) hands the deficit to the corpora that still have records,
    re-split by the same weights, until the quota is met. With every
    corpus exhausted the shortfall is an error unless ``upsample``."""
    names = list(allocation)
    want = dict(allocation)
    while True:
        for n in names:
            need = want[n] - len(drawers[n].kept)
            if need > 0:
                drawers[n].draw(need)
        kept = sum(len(drawers[n].kept) for n in names)
        deficit = quota - kept
        if deficit <= 0:
            for n in names:
                if allocation[n] > 0 and not drawers[n].kept:
                    logger.warning(
                        "mixture: corpus %r contributes nothing to category %r — every one of its %d records was "
                        "dropped at max_length %d; the other corpora covered its %d examples",
                        n, cat, len(drawers[n].records), drawers[n].max_length, allocation[n],
                    )
            return
        open_ = [n for n in names if not drawers[n].exhausted and weights.get(n, 0) > 0]
        if not open_:
            detail = ", ".join(
                f"{n} kept {len(drawers[n].kept)} of {len(drawers[n].records)} "
                f"({drawers[n].dropped_truncation} truncation drops"
                f"{', ' + str(drawers[n].dropped_cut_answer) + ' cut-answer drops' if mixture.drop_truncated_answers else ''})"
                for n in names
            )
            raise MixtureShortfallError(
                f"mixture: category {cat!r} kept {kept} of its {quota} examples after tokenization — {detail}; "
                f"lower total_examples, raise max_length, or set mixture.upsample: true"
            )
        extra = largest_remainder({n: weights[n] for n in open_}, deficit)
        for n in open_:
            want[n] += extra[n]


def compose_mixture(
    mixture: MixtureConfig,
    datasets: Sequence[str],
    pools: Mapping[str, Sequence[QARecord]],
    tokenizer: BaseTokenizer,
    max_length: int,
    loss_target: str,
    batch_size: int = 1,
    pool_sizes_before_filter: Optional[Mapping[str, int]] = None,
    clean_latin_rows: bool = False,
    attach_category: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Draw the phase's training set from per-corpus record pools.

    ``pools`` maps each of ``datasets`` to its (already Latin-filtered, if
    requested) records. Returns ``(encodings, manifest)``; the encodings are
    shuffled with ``mixture.seed``.

    ``attach_category`` adds a ``"_category"`` key to every encoding — used by
    the early-stop eval mixture, which scores each category separately. The
    collators read only the keys they know, but the training path leaves it off
    so its entries stay exactly what they were.
    """
    capacities = {n: len(pools[n]) for n in datasets}
    plan = plan_mixture(mixture, datasets, capacities, batch_size)
    drawers = {
        n: CorpusDrawer(n, pools[n], tokenizer, max_length, loss_target, mixture) for n in datasets
    }
    for cat, quota in plan["category_quotas"].items():
        members = [n for n in datasets if CORPUS_CATEGORY[n] == cat]
        draw_category(
            cat, quota, {n: plan["allocation"][n] for n in members},
            {n: plan["weights"][n] for n in members}, drawers, mixture,
        )

    encodings: List[Dict[str, Any]] = []
    per_dataset: Dict[str, Dict[str, Any]] = {}
    for name in datasets:
        stats = drawers[name].stats()
        kept = drawers[name].kept
        if attach_category:
            for entry in kept:
                entry["_category"] = CORPUS_CATEGORY[name]
        encodings.extend(kept)
        per_dataset[name] = {
            "category": CORPUS_CATEGORY[name],
            "available": int((pool_sizes_before_filter or {}).get(name, capacities[name])),
            "after_latin_filter": capacities[name],
            "weight": plan["weights"][name],
            "planned": plan["allocation"][name],
            **{k: stats[k] for k in ("drawn", "dropped_truncation", "dropped_cut_answer", "kept", "repeated", "loss_tokens", "passes")},
            "revision": PINNED_REVISIONS.get(name),
            "ids": stats["ids"],
        }
        logger.info(
            "mixture: %s (%s) planned %d → kept %d of %d drawn (%d truncation drops%s, %d loss tokens%s)",
            name, CORPUS_CATEGORY[name], plan["allocation"][name], stats["kept"], stats["drawn"], stats["dropped_truncation"],
            f", {stats['dropped_cut_answer']} cut-answer drops" if mixture.drop_truncated_answers else "",
            stats["loss_tokens"], f", {stats['passes']} passes / {stats['repeated']} repeated" if stats["passes"] > 1 else "",
        )

    total_loss_tokens = sum(d["loss_tokens"] for d in per_dataset.values())
    categories: Dict[str, Dict[str, Any]] = {}
    for cat, quota in plan["category_quotas"].items():
        members = [n for n in datasets if CORPUS_CATEGORY[n] == cat]
        kept = sum(per_dataset[n]["kept"] for n in members)
        loss_tokens = sum(per_dataset[n]["loss_tokens"] for n in members)
        categories[cat] = {
            "share": mixture.shares[cat],
            "quota": quota,
            "kept": kept,
            "example_share": round(kept / mixture.total_examples, 4) if mixture.total_examples else 0.0,
            "loss_tokens": loss_tokens,
            "loss_token_share": round(loss_tokens / total_loss_tokens, 4) if total_loss_tokens else 0.0,
            "capacity": plan["category_capacity"][cat],
            "datasets": members,
        }
    logger.info(
        "mixture: %d examples — %s",
        len(encodings),
        "; ".join(
            f"{cat} {c['example_share']:.0%} of examples / {c['loss_token_share']:.0%} of loss tokens"
            for cat, c in categories.items()
        ),
    )
    assert len(encodings) == mixture.total_examples, (len(encodings), mixture.total_examples)
    random.Random(mixture.seed).shuffle(encodings)

    manifest = {
        "total_examples": mixture.total_examples,
        "shares": dict(mixture.shares),
        "within_category": mixture.within_category,
        "weights": dict(mixture.weights) if mixture.weights else None,
        "upsample": mixture.upsample,
        "drop_truncated_answers": mixture.drop_truncated_answers,
        "seed": mixture.seed,
        "max_length": max_length,
        "loss_target": loss_target,
        "clean_latin_rows": clean_latin_rows,
        "template_version": TEMPLATE_VERSION,
        "max_total_examples_at_these_shares": plan["max_total_examples_at_these_shares"],
        "loss_tokens": total_loss_tokens,
        "categories": categories,
        "datasets": per_dataset,
    }
    return encodings, manifest


def manifest_summary(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The manifest without the per-corpus id lists (what goes into
    ``all_metrics.json``; the full manifest is written beside the cell)."""
    out = dict(manifest)
    out["datasets"] = {n: {k: v for k, v in d.items() if k != "ids"} for n, d in manifest["datasets"].items()}
    return out


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------

def load_mixture_pools(
    datasets: Sequence[str],
    corpus_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    clean_latin_rows: bool = False,
    exclusions: Any = None,
    split: str = "train",
) -> Tuple[Dict[str, List[QARecord]], Dict[str, int]]:
    """Load every corpus' ``split`` (with its ``training.corpus_params`` and
    the contamination ``exclusions``) and apply the phase's Latin filter.
    Returns ``(pools, sizes_before_filter)``. ``split="dev"`` builds the pools
    of an early-stop eval mixture; ``validation`` is never a mixture pool."""
    corpus_params = corpus_params or {}
    pools: Dict[str, List[QARecord]] = {}
    before: Dict[str, int] = {}
    for name in datasets:
        kw: Dict[str, Any] = dict(corpus_params.get(name, {}))
        if exclusions is not None:
            kw["exclusions"] = exclusions
        recs = load_corpus(name, split, **kw)
        before[name] = len(recs)
        if clean_latin_rows:
            recs = filter_latin_records(recs)
            logger.info("mixture: clean_latin_rows on %s/%s: %d → %d records", name, split, before[name], len(recs))
        pools[name] = recs
    return pools, before


def build_mixture_dataloader(
    mixture: MixtureConfig,
    datasets: Sequence[str],
    tokenizer: BaseTokenizer,
    batch_size: int,
    max_length: int,
    loss_target: str,
    corpus_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    clean_latin_rows: bool = False,
    exclusions: Any = None,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """Load, compose and wrap the phase's mixture in a DataLoader with the
    tokenizer's collator (``embedding_type`` dispatch as ``build_qa_dataloader``)."""
    pools, before = load_mixture_pools(datasets, corpus_params, clean_latin_rows, exclusions)
    encodings, manifest = compose_mixture(
        mixture, datasets, pools, tokenizer, max_length, loss_target,
        batch_size=batch_size, pool_sizes_before_filter=before, clean_latin_rows=clean_latin_rows,
    )
    collator = get_collator(
        tokenizer.embedding_type,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        max_length=max_length,
    )
    loader = DataLoader(
        _QATokenizedDataset(encodings),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    return loader, manifest
