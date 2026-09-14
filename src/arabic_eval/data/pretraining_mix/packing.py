"""Stage B — per-cell token-budget fill + packing.

Given a built pool and the *active* tokenizer:

1. ``fill_token_budget`` — for each source, walk its pool documents in a
   fixed seeded order, tokenize, and take documents until
   ``share × token_budget`` tokens are reached (overshoot ≤ one document).
   For a given source every cell's selection is a *prefix of the same
   order*, so a low-fertility tokenizer sees a superset of what a
   high-fertility one sees: same distribution, nested cutoffs.
2. ``pack_blocks`` — shuffle the selected documents (seeded), concatenate
   with an EOS separator between documents, chunk into ``block_size``
   blocks, drop the final partial block.

``build_packed_corpus`` caches the result under ``{cell_output_dir}/data/
pretraining_mix/`` keyed on (pool fingerprint, tokenizer identity, budget,
block size, seed) and returns a ``PackedCorpus`` whose ``take`` hands each
phase a contiguous block range (``consume_sequentially``: the next phase
continues where the previous one stopped).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from ...config import PretrainingMixConfig, TrainingConfig
from ...tokenizers.base import BaseTokenizer
from .pool import build_pool, iter_pool_docs, load_pool_manifest, pool_fingerprint

logger = logging.getLogger(__name__)

PACKED_MANIFEST = "packed_manifest.json"
INPUT_IDS_FILE = "packed_input_ids.npy"
CHAR_IDS_FILE = "packed_char_ids.npy"


# --------------------------------------------------------------------------
# Budget / identity
# --------------------------------------------------------------------------

def resolve_token_budget(training_cfg: TrainingConfig) -> int:
    """Explicit ``token_budget`` or Σ(steps × batch_size) × block_size over
    the enabled phases that consume the mix (micro-steps: ``run_phase``
    draws one batch per step)."""
    mix = training_cfg.pretraining_mix
    if mix is None:
        raise ValueError("training.pretraining_mix is not configured")
    if mix.token_budget is not None:
        return int(mix.token_budget)
    phases = training_cfg.phases_using_mix()
    blocks = sum(getattr(training_cfg.phases, p).steps * getattr(training_cfg.phases, p).batch_size
                 for p in phases)
    if blocks == 0:
        raise ValueError(
            "pretraining_mix.token_budget is null and no enabled phase lists "
            "datasets: ['pretraining_mix'] — nothing to size the budget from"
        )
    return blocks * mix.block_size


def tokenizer_identity(tokenizer: BaseTokenizer, tokenizer_type: str) -> Dict[str, Any]:
    return {
        "type": tokenizer_type,
        "class": type(tokenizer).__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "embedding_type": tokenizer.embedding_type,
        "special_tokens": {k: int(v) for k, v in tokenizer.special_tokens.items()},
    }


def packed_fingerprint(pool_fp: str, tok_id: Dict[str, Any], token_budget: int,
                       block_size: int, seed: int, shares: Dict[str, float]) -> str:
    payload = json.dumps(
        {"pool": pool_fp, "tokenizer": tok_id, "token_budget": token_budget,
         "block_size": block_size, "seed": seed, "shares": shares},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Fill
# --------------------------------------------------------------------------

@dataclass
class EncodedDoc:
    source: str
    ids: np.ndarray                          # int64 [n_tokens]
    char_ids: Optional[np.ndarray] = None    # int64 [n_tokens, max_char_len] (character_cnn only)


@dataclass
class SourceFillStats:
    name: str
    share: float
    target_tokens: int
    docs_available: int = 0
    docs_taken: int = 0
    tokens: int = 0
    words: int = 0
    wall_sec: float = 0.0

    @property
    def fertility(self) -> float:
        return self.tokens / self.words if self.words else 0.0

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "share": self.share, "target_tokens": self.target_tokens,
                "docs_available": self.docs_available, "docs_taken": self.docs_taken,
                "tokens": self.tokens, "words": self.words, "fertility": round(self.fertility, 4),
                "wall_sec": self.wall_sec}


def source_doc_order(n_docs: int, seed: int, source_name: str) -> np.ndarray:
    """Fixed permutation of a source's pool — identical for every cell."""
    rng = np.random.default_rng([int(seed), zlib.crc32(source_name.encode("utf-8"))])
    return rng.permutation(n_docs)


def encode_document(tokenizer: BaseTokenizer, text: str, eos_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Tokenize one document without truncation; guarantee a trailing EOS."""
    enc = tokenizer.encode(text, max_length=None, truncation=False)
    ids = list(enc.input_ids)
    char_ids = enc.char_ids
    if not ids or ids[-1] != eos_id:
        if char_ids is not None:
            raise ValueError(
                f"{type(tokenizer).__name__}.encode() returned char_ids without a trailing EOS "
                f"row — cannot append a separator consistently"
            )
        ids.append(eos_id)
    ids_arr = np.asarray(ids, dtype=np.int64)
    char_arr = np.asarray(char_ids, dtype=np.int64) if char_ids is not None else None
    if char_arr is not None and char_arr.shape[0] != ids_arr.shape[0]:
        raise ValueError(
            f"char_ids rows ({char_arr.shape[0]}) != input_ids length ({ids_arr.shape[0]})"
        )
    return ids_arr, char_arr


def fill_token_budget(
    pool_dir: Path,
    mix_cfg: PretrainingMixConfig,
    tokenizer: BaseTokenizer,
    token_budget: int,
    log_every: int = 1000,
) -> Tuple[List[EncodedDoc], List[SourceFillStats]]:
    eos_id = int(tokenizer.special_tokens["eos_token"])
    docs: List[EncodedDoc] = []
    stats: List[SourceFillStats] = []
    for src in mix_cfg.sources:
        target = int(round(src.share * token_budget))
        st = SourceFillStats(name=src.name, share=src.share, target_tokens=target)
        t0 = time.perf_counter()
        pool_docs = list(iter_pool_docs(pool_dir, src.name))
        st.docs_available = len(pool_docs)
        order = source_doc_order(len(pool_docs), mix_cfg.seed, src.name)
        for k, idx in enumerate(order):
            if st.tokens >= target:
                break
            _, text = pool_docs[idx]
            ids, char_ids = encode_document(tokenizer, text, eos_id)
            docs.append(EncodedDoc(source=src.name, ids=ids, char_ids=char_ids))
            st.docs_taken += 1
            st.tokens += int(ids.shape[0])
            st.words += len(text.split())
            if log_every and st.docs_taken % log_every == 0:
                logger.info("[pack:%s] docs=%d tokens=%d/%d", src.name, st.docs_taken, st.tokens, target)
        st.wall_sec = round(time.perf_counter() - t0, 1)
        if st.tokens < target:
            raise ValueError(
                f"pretraining_mix: source {src.name!r} pool is exhausted at {st.tokens} tokens "
                f"({st.docs_taken} docs) but the token budget needs {target} — raise "
                f"pretraining_mix.pool.total_words (currently {mix_cfg.pool.total_words}) and rebuild "
                f"the pool, or lower token_budget"
            )
        logger.info(
            "[pack:%s] took %d/%d docs → %d tokens (target %d, fertility %.2f) in %.0fs",
            src.name, st.docs_taken, st.docs_available, st.tokens, target, st.fertility, st.wall_sec,
        )
        stats.append(st)
    return docs, stats


# --------------------------------------------------------------------------
# Pack
# --------------------------------------------------------------------------

def pack_blocks(
    docs: List[EncodedDoc], block_size: int, seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Shuffle docs, concatenate, chunk. Returns ``(input_ids [n, L] int32,
    char_ids [n, L, C] int16 | None, n_dropped_tail_tokens)``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    ids = np.concatenate([docs[i].ids for i in order]) if docs else np.zeros(0, dtype=np.int64)
    n_blocks = ids.shape[0] // block_size
    if n_blocks == 0:
        raise ValueError(f"pretraining_mix: {ids.shape[0]} tokens < one block of {block_size}")
    tail = int(ids.shape[0] - n_blocks * block_size)
    input_ids = ids[: n_blocks * block_size].reshape(n_blocks, block_size).astype(np.int32)

    char_ids = None
    has_char = any(d.char_ids is not None for d in docs)
    if has_char:
        if not all(d.char_ids is not None for d in docs):
            raise ValueError("pretraining_mix: mixed char_ids / no-char_ids documents")
        cat = np.concatenate([docs[i].char_ids for i in order])
        if cat.max(initial=0) > np.iinfo(np.int16).max:
            raise ValueError("char id exceeds int16 range")
        char_ids = cat[: n_blocks * block_size].reshape(n_blocks, block_size, -1).astype(np.int16)
    return input_ids, char_ids, tail


# --------------------------------------------------------------------------
# Corpus object handed to the phases
# --------------------------------------------------------------------------

class PackedBlockDataset(Dataset):
    """Contiguous block range ``[start, end)`` of a packed corpus; indices
    beyond ``n_blocks`` wrap around (the phase then sees an epoch > 1)."""

    def __init__(self, input_ids: np.ndarray, char_ids: Optional[np.ndarray], start: int, end: int) -> None:
        if end <= start:
            raise ValueError(f"empty block range [{start}, {end})")
        self._ids = input_ids
        self._chars = char_ids
        self._start = start
        self._end = end
        self._n = int(input_ids.shape[0])

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = (self._start + i) % self._n
        ex: Dict[str, Any] = {"input_ids": np.asarray(self._ids[idx], dtype=np.int64)}
        if self._chars is not None:
            ex["char_ids"] = np.asarray(self._chars[idx], dtype=np.int64)
        return ex


@dataclass
class PackedCorpus:
    input_ids: np.ndarray
    char_ids: Optional[np.ndarray]
    manifest: Dict[str, Any]
    consume_sequentially: bool = True
    cursor: int = 0
    consumption: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.input_ids.shape[1])

    def take(self, phase_name: str, n_blocks_needed: int) -> Tuple[PackedBlockDataset, Dict[str, Any]]:
        """Hand ``phase_name`` its next ``n_blocks_needed`` blocks."""
        start = self.cursor if self.consume_sequentially else 0
        end = start + n_blocks_needed
        epochs = n_blocks_needed / self.n_blocks
        wrapped = end > self.n_blocks
        if wrapped:
            logger.warning(
                "[%s] pretraining_mix: needs %d blocks from offset %d but the corpus has %d — "
                "wrapping around (%.2f epochs of the packed corpus)",
                phase_name, n_blocks_needed, start, self.n_blocks, epochs,
            )
        if self.consume_sequentially:
            self.cursor = end % self.n_blocks
        info = {
            "dataset": "pretraining_mix",
            "block_start": start, "block_end": end, "n_blocks": n_blocks_needed,
            "n_tokens": n_blocks_needed * self.block_size,
            "corpus_blocks": self.n_blocks, "epochs": round(epochs, 4), "wrapped": wrapped,
        }
        self.consumption[phase_name] = info
        return PackedBlockDataset(self.input_ids, self.char_ids, start, end), info


# --------------------------------------------------------------------------
# Build / cache
# --------------------------------------------------------------------------

def build_packed_corpus(
    training_cfg: TrainingConfig,
    tokenizer: BaseTokenizer,
    tokenizer_type: str,
    out_dir: Path,
    pool_dir: Optional[Path] = None,
) -> PackedCorpus:
    """Build (or load) the packed corpus for one experiment cell."""
    mix = training_cfg.pretraining_mix
    if mix is None:
        raise ValueError("training.pretraining_mix is not configured")
    out_dir = Path(out_dir)
    pool_dir = Path(pool_dir) if pool_dir is not None else build_pool(mix)
    pool_manifest = load_pool_manifest(pool_dir) or {}
    pool_fp = pool_manifest.get("fingerprint") or pool_fingerprint(mix)

    token_budget = resolve_token_budget(training_cfg)
    tok_id = tokenizer_identity(tokenizer, tokenizer_type)
    shares = {s.name: s.share for s in mix.sources}
    fp = packed_fingerprint(pool_fp, tok_id, token_budget, mix.block_size, mix.seed, shares)

    manifest_path = out_dir / PACKED_MANIFEST
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("fingerprint") == fp and (out_dir / INPUT_IDS_FILE).exists():
            logger.info("pretraining_mix: reusing packed corpus at %s", out_dir)
            input_ids = np.load(out_dir / INPUT_IDS_FILE, mmap_mode="r")
            char_ids = np.load(out_dir / CHAR_IDS_FILE, mmap_mode="r") if (out_dir / CHAR_IDS_FILE).exists() else None
            return PackedCorpus(input_ids, char_ids, existing, mix.consume_sequentially)

    logger.info(
        "pretraining_mix: packing for %s (budget %d tokens, block %d) from pool %s",
        tokenizer_type, token_budget, mix.block_size, pool_dir,
    )
    t0 = time.perf_counter()
    docs, fill_stats = fill_token_budget(pool_dir, mix, tokenizer, token_budget)
    input_ids, char_ids, tail = pack_blocks(docs, mix.block_size, mix.seed)
    total_tokens = sum(s.tokens for s in fill_stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / INPUT_IDS_FILE, input_ids)
    if char_ids is not None:
        np.save(out_dir / CHAR_IDS_FILE, char_ids)
    manifest = {
        "fingerprint": fp,
        "pool_dir": str(pool_dir),
        "pool_fingerprint": pool_fp,
        "tokenizer": tok_id,
        "token_budget": token_budget,
        "block_size": mix.block_size,
        "seed": mix.seed,
        "n_blocks": int(input_ids.shape[0]),
        "n_tokens_packed": int(input_ids.shape[0] * mix.block_size),
        "tail_tokens_dropped": tail,
        "phases_using_mix": training_cfg.phases_using_mix(),
        "sources": [s.to_json() | {"achieved_share": round(s.tokens / total_tokens, 4)} for s in fill_stats],
        "wall_sec": round(time.perf_counter() - t0, 1),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(
        "pretraining_mix: packed %d blocks × %d (%s) in %.0fs → %s",
        manifest["n_blocks"], mix.block_size,
        ", ".join(f"{s['name']}={s['achieved_share']:.3f}" for s in manifest["sources"]),
        manifest["wall_sec"], out_dir,
    )
    return PackedCorpus(input_ids, char_ids, manifest, mix.consume_sequentially)
