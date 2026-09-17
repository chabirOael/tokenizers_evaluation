"""Stage B — pack the whole pool once per tokenizer; phases draw disjoint slices.

Given a built pool and a tokenizer:

1. ``tokenize_pool`` — tokenize *every* pool document of every source, in
   a fixed seeded order (``source_doc_order``; identical for every
   tokenizer, so a smaller pool is always a prefix of a larger one).
2. ``exact_share_budget`` — the largest token budget at which the source
   shares hold exactly: ``min_i(available_i / share_i)``. The pool is split
   by share in *words*, and per-source fertility differs slightly under any
   given tokenizer, so a few percent of the over-supplied sources stay
   unused rather than bending the ratio.
3. ``select_docs`` + ``pack_blocks`` — take each source's prefix up to its
   share of the budget, shuffle the selected documents (seeded),
   concatenate with an EOS separator, chunk into ``block_size`` blocks.

The result is cached at ``<pool>/packed/<tokenizer_fingerprint>/`` and
shared by every experiment — the fingerprint includes a *content hash* of
the tokenizer (ids of a fixed probe paragraph), so two BPE-32K tokenizers
with different learned vocabs never share an entry. Block order is fixed
once per (pool, tokenizer): ``PackedCorpus.take`` hands each phase the
next ``mix_tokens / block_size`` blocks, strictly consecutive and
non-overlapping, and raises (never wraps) when the pool is too small.

A phase may additionally *blend* SFT-format QA text (``qa_blend``):
``pack_qa_blend`` renders the QA train records with the Phase 3 / eval
surface form, packs them into blocks the same way (own cache, once per
tokenizer, at ``<cache_dir>/qa_blend/<fp>/``), and ``BlendedBlockDataset``
interleaves the phase's raw-text slice with its QA slice through a seeded
permutation — every block read once, batches are random mixtures.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from ...config import PretrainingMixConfig, QABlendConfig
from ...tokenizers.base import BaseTokenizer
from .pool import build_pool, iter_pool_docs, load_pool_manifest, pool_fingerprint

logger = logging.getLogger(__name__)

PACKED_MANIFEST = "packed_manifest.json"
INPUT_IDS_FILE = "packed_input_ids.npy"
CHAR_IDS_FILE = "packed_char_ids.npy"
PACKED_SUBDIR = "packed"
QA_BLEND_SUBDIR = "qa_blend"
QA_BLEND_MANIFEST = "qa_blend_manifest.json"

# Fixed probe for the tokenizer content hash: diacritics, digits, Latin,
# punctuation and a rare word so vocab / merge differences show up.
_PROBE_TEXT = (
    "تُعَدُّ اللغةُ العربيةُ من أكثرِ اللغاتِ انتشارًا؛ يتحدث بها نحو 400 مليون شخص. "
    "قال المهندس: «سنطلق النسخة 2.0 من التطبيق Alpha غدًا»، فاستبشر الجميع خيرًا. "
    "والمستشرقون المتخصصون بالبلاغة لم يستطيعوا تفسير الاستعارة المكنية بسهولة."
)


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def tokenizer_identity(tokenizer: BaseTokenizer, tokenizer_type: str) -> Dict[str, Any]:
    enc = tokenizer.encode(_PROBE_TEXT)
    h = hashlib.sha256(json.dumps(list(enc.input_ids)).encode("utf-8"))
    if enc.char_ids is not None:
        h.update(json.dumps([list(r) for r in enc.char_ids]).encode("utf-8"))
    return {
        "type": tokenizer_type,
        "class": type(tokenizer).__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "embedding_type": tokenizer.embedding_type,
        "special_tokens": {k: int(v) for k, v in tokenizer.special_tokens.items()},
        "content_hash": h.hexdigest()[:16],
    }


def packed_fingerprint(pool_fp: str, tok_id: Dict[str, Any], block_size: int,
                       seed: int, shares: Dict[str, float]) -> str:
    payload = json.dumps(
        {"pool": pool_fp, "tokenizer": tok_id, "block_size": block_size, "seed": seed, "shares": shares},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def packed_dir(pool_dir: Path, fingerprint: str) -> Path:
    return Path(pool_dir) / PACKED_SUBDIR / fingerprint


# --------------------------------------------------------------------------
# Tokenize the whole pool
# --------------------------------------------------------------------------

@dataclass
class EncodedDoc:
    source: str
    ids: np.ndarray                          # int32 [n_tokens]
    char_ids: Optional[np.ndarray] = None    # int16 [n_tokens, max_char_len] (character_cnn only)


@dataclass
class SourceTokenStats:
    name: str
    share: float
    docs_available: int = 0
    available_tokens: int = 0
    words: int = 0
    target_tokens: int = 0
    docs_taken: int = 0
    tokens: int = 0
    wall_sec: float = 0.0

    @property
    def fertility(self) -> float:
        return self.available_tokens / self.words if self.words else 0.0

    def to_json(self, total_taken: int) -> Dict[str, Any]:
        return {
            "name": self.name, "share": self.share,
            "docs_available": self.docs_available, "available_tokens": self.available_tokens,
            "words": self.words, "fertility": round(self.fertility, 4),
            "target_tokens": self.target_tokens, "docs_taken": self.docs_taken, "tokens": self.tokens,
            "unused_tokens": self.available_tokens - self.tokens,
            "achieved_share": round(self.tokens / total_taken, 4) if total_taken else 0.0,
            "wall_sec": self.wall_sec,
        }


def source_doc_order(n_docs: int, seed: int, source_name: str) -> np.ndarray:
    """Fixed permutation of a source's pool — identical for every tokenizer."""
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
    ids_arr = np.asarray(ids, dtype=np.int32)
    char_arr = None
    if char_ids is not None:
        char_arr = np.asarray(char_ids, dtype=np.int64)
        if char_arr.shape[0] != ids_arr.shape[0]:
            raise ValueError(f"char_ids rows ({char_arr.shape[0]}) != input_ids length ({ids_arr.shape[0]})")
        if char_arr.size and char_arr.max() > np.iinfo(np.int16).max:
            raise ValueError("char id exceeds int16 range")
        char_arr = char_arr.astype(np.int16)
    return ids_arr, char_arr


def tokenize_pool(
    pool_dir: Path,
    mix_cfg: PretrainingMixConfig,
    tokenizer: BaseTokenizer,
    log_every: int = 5000,
) -> Tuple[Dict[str, List[EncodedDoc]], Dict[str, SourceTokenStats]]:
    """Tokenize every pool document, per source, in the fixed seeded order."""
    eos_id = int(tokenizer.special_tokens["eos_token"])
    docs: Dict[str, List[EncodedDoc]] = {}
    stats: Dict[str, SourceTokenStats] = {}
    for src in mix_cfg.sources:
        st = SourceTokenStats(name=src.name, share=src.share)
        t0 = time.perf_counter()
        pool_docs = list(iter_pool_docs(pool_dir, src.name))
        st.docs_available = len(pool_docs)
        out: List[EncodedDoc] = []
        for k, idx in enumerate(source_doc_order(len(pool_docs), mix_cfg.seed, src.name), start=1):
            _, text = pool_docs[idx]
            ids, char_ids = encode_document(tokenizer, text, eos_id)
            out.append(EncodedDoc(source=src.name, ids=ids, char_ids=char_ids))
            st.available_tokens += int(ids.shape[0])
            st.words += len(text.split())
            if log_every and k % log_every == 0:
                logger.info("[pack:%s] tokenized %d/%d docs (%d tokens)", src.name, k, len(pool_docs), st.available_tokens)
        st.wall_sec = round(time.perf_counter() - t0, 1)
        logger.info(
            "[pack:%s] %d docs → %d tokens (%d words, fertility %.2f) in %.0fs",
            src.name, len(out), st.available_tokens, st.words, st.fertility, st.wall_sec,
        )
        docs[src.name] = out
        stats[src.name] = st
    return docs, stats


# --------------------------------------------------------------------------
# Exact-share selection
# --------------------------------------------------------------------------

def exact_share_budget(available_tokens: Dict[str, int], shares: Dict[str, float]) -> int:
    """Largest budget B such that every source can supply share_i × B."""
    return int(min(available_tokens[n] / shares[n] for n in shares))


def select_docs(
    docs: Dict[str, List[EncodedDoc]],
    stats: Dict[str, SourceTokenStats],
    budget: int,
) -> List[EncodedDoc]:
    """Per source, the prefix of the fixed order reaching share × budget
    tokens (overshoot ≤ one document, capped at what is available)."""
    selected: List[EncodedDoc] = []
    for name, st in stats.items():
        st.target_tokens = int(round(st.share * budget))
        for d in docs[name]:
            if st.tokens >= st.target_tokens:
                break
            selected.append(d)
            st.docs_taken += 1
            st.tokens += int(d.ids.shape[0])
    return selected


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
    ids = np.concatenate([docs[i].ids for i in order]) if docs else np.zeros(0, dtype=np.int32)
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
        char_ids = cat[: n_blocks * block_size].reshape(n_blocks, block_size, -1).astype(np.int16)
    return input_ids, char_ids, tail


# --------------------------------------------------------------------------
# Corpus object handed to the phases
# --------------------------------------------------------------------------

class PackedBlockDataset(Dataset):
    """Contiguous block range ``[start, end)`` of a packed corpus."""

    def __init__(self, input_ids: np.ndarray, char_ids: Optional[np.ndarray], start: int, end: int) -> None:
        if end <= start:
            raise ValueError(f"empty block range [{start}, {end})")
        if end > input_ids.shape[0]:
            raise ValueError(f"block range [{start}, {end}) exceeds corpus of {input_ids.shape[0]} blocks")
        self._ids = input_ids
        self._chars = char_ids
        self._start = start
        self._end = end

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self._start + i
        ex: Dict[str, Any] = {"input_ids": np.asarray(self._ids[idx], dtype=np.int64)}
        if self._chars is not None:
            ex["char_ids"] = np.asarray(self._chars[idx], dtype=np.int64)
        return ex


class BlendedBlockDataset(Dataset):
    """Two block datasets read through one seeded permutation, so a phase
    that blends QA text into the raw-text mix sees random mixtures in every
    batch while still reading each block exactly once."""

    def __init__(self, parts: List[Dataset], seed: int) -> None:
        self._parts = [p for p in parts if len(p)]
        if not self._parts:
            raise ValueError("BlendedBlockDataset needs at least one non-empty part")
        self._offsets = np.cumsum([0] + [len(p) for p in self._parts])
        self._order = np.random.default_rng(seed).permutation(int(self._offsets[-1]))

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        j = int(self._order[i])
        part = int(np.searchsorted(self._offsets, j, side="right") - 1)
        return self._parts[part][j - int(self._offsets[part])]


@dataclass
class PackedCorpus:
    input_ids: np.ndarray
    char_ids: Optional[np.ndarray]
    manifest: Dict[str, Any]
    cursor: int = 0
    consumption: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    kind: str = "pretraining_mix"   # "pretraining_mix" | "qa_blend"

    @property
    def n_blocks(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.input_ids.shape[1])

    def take(self, phase_name: str, n_blocks_needed: int) -> Tuple[PackedBlockDataset, Dict[str, Any]]:
        """Hand ``phase_name`` the next ``n_blocks_needed`` blocks — strictly
        after every block handed out so far. Raises when the packed corpus
        cannot supply them; nothing is ever repeated."""
        start = self.cursor
        end = start + n_blocks_needed
        if end > self.n_blocks and self.kind == "qa_blend":
            raise ValueError(
                f"[{phase_name}] qa_blend: needs {n_blocks_needed} blocks starting at block {start} "
                f"but the packed QA corpus has only {self.n_blocks} ({self.n_blocks * self.block_size} "
                f"tokens over {self.manifest.get('n_records')} records of {self.manifest.get('datasets')}). "
                f"Lower qa_blend.share or add a corpus"
            )
        if end > self.n_blocks:
            raise ValueError(
                f"[{phase_name}] pretraining_mix: needs {n_blocks_needed} blocks starting at block "
                f"{start} but the packed corpus has {self.n_blocks} ({self.n_blocks * self.block_size} "
                f"tokens at the exact-share budget). Phases draw disjoint ranges — lower mix_tokens "
                f"or raise pretraining_mix.pool.total_words (this tokenizer needs ≥ "
                f"{end * self.block_size} tokens ≈ {int(end * self.block_size / max(self.manifest.get('fertility_min', 1.0), 1e-9))} pool words) "
                f"and rebuild the pool"
            )
        self.cursor = end
        info = {
            "dataset": self.kind,
            "block_start": start, "block_end": end, "n_blocks": n_blocks_needed,
            "n_tokens": n_blocks_needed * self.block_size,
            "corpus_blocks": self.n_blocks,
            "corpus_share": round(n_blocks_needed / self.n_blocks, 4),
        }
        self.consumption[phase_name] = info
        return PackedBlockDataset(self.input_ids, self.char_ids, start, end), info


# --------------------------------------------------------------------------
# Build / cache
# --------------------------------------------------------------------------

def _publish_packed(directory: Path, manifest_name: str, manifest: Dict[str, Any],
                    input_ids: np.ndarray, char_ids: Optional[np.ndarray]) -> None:
    """Atomic publish: another process packing the same tokenizer must
    never observe a half-written entry."""
    tmp = directory.with_name(f"{directory.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / INPUT_IDS_FILE, input_ids)
    if char_ids is not None:
        np.save(tmp / CHAR_IDS_FILE, char_ids)
    with open(tmp / manifest_name, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(tmp, directory)


def _load_packed(directory: Path, manifest: Dict[str, Any]) -> PackedCorpus:
    input_ids = np.load(directory / INPUT_IDS_FILE, mmap_mode="r")
    char_path = directory / CHAR_IDS_FILE
    char_ids = np.load(char_path, mmap_mode="r") if char_path.exists() else None
    return PackedCorpus(input_ids, char_ids, manifest)


def build_packed_corpus(
    mix_cfg: PretrainingMixConfig,
    tokenizer: BaseTokenizer,
    tokenizer_type: str,
    cell_data_dir: Optional[Path] = None,
    pool_dir: Optional[Path] = None,
) -> PackedCorpus:
    """Pack (or load) the whole pool for this tokenizer.

    The packed corpus lives under the pool (``<pool>/packed/<fp>/``) and is
    shared across experiments; ``cell_data_dir``, when given, receives a
    copy of the manifest pointing at it so each cell stays self-describing.
    """
    pool_dir = Path(pool_dir) if pool_dir is not None else build_pool(mix_cfg)
    pool_manifest = load_pool_manifest(pool_dir) or {}
    pool_fp = pool_manifest.get("fingerprint") or pool_fingerprint(mix_cfg)
    tok_id = tokenizer_identity(tokenizer, tokenizer_type)
    shares = {s.name: s.share for s in mix_cfg.sources}
    fp = packed_fingerprint(pool_fp, tok_id, mix_cfg.block_size, mix_cfg.seed, shares)
    directory = packed_dir(pool_dir, fp)
    manifest_path = directory / PACKED_MANIFEST

    corpus: Optional[PackedCorpus] = None
    if manifest_path.exists() and (directory / INPUT_IDS_FILE).exists():
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("fingerprint") == fp:
            logger.info("pretraining_mix: reusing packed corpus %s (%d blocks)", directory, existing["n_blocks"])
            corpus = _load_packed(directory, existing)

    if corpus is None:
        logger.info(
            "pretraining_mix: packing the whole pool %s for %s (block %d)",
            pool_dir, tokenizer_type, mix_cfg.block_size,
        )
        t0 = time.perf_counter()
        docs, stats = tokenize_pool(pool_dir, mix_cfg, tokenizer)
        budget = exact_share_budget({n: s.available_tokens for n, s in stats.items()}, shares)
        selected = select_docs(docs, stats, budget)
        input_ids, char_ids, tail = pack_blocks(selected, mix_cfg.block_size, mix_cfg.seed)
        del docs, selected
        total_taken = sum(s.tokens for s in stats.values())
        manifest = {
            "fingerprint": fp,
            "pool_dir": str(pool_dir),
            "pool_fingerprint": pool_fp,
            "tokenizer": tok_id,
            "block_size": mix_cfg.block_size,
            "seed": mix_cfg.seed,
            "exact_share_budget": budget,
            "n_blocks": int(input_ids.shape[0]),
            "n_tokens_packed": int(input_ids.shape[0] * mix_cfg.block_size),
            "tail_tokens_dropped": tail,
            "fertility_min": round(min(s.fertility for s in stats.values()), 4),
            "sources": [s.to_json(total_taken) for s in stats.values()],
            "wall_sec": round(time.perf_counter() - t0, 1),
        }
        _publish_packed(directory, PACKED_MANIFEST, manifest, input_ids, char_ids)
        logger.info(
            "pretraining_mix: packed %d blocks × %d = %d tokens (budget %d; %s) in %.0fs → %s",
            manifest["n_blocks"], mix_cfg.block_size, manifest["n_tokens_packed"], budget,
            ", ".join(f"{s['name']}={s['achieved_share']:.3f}" for s in manifest["sources"]),
            manifest["wall_sec"], directory,
        )
        corpus = _load_packed(directory, manifest)

    if cell_data_dir is not None:
        cell_data_dir = Path(cell_data_dir)
        cell_data_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_data_dir / PACKED_MANIFEST, "w", encoding="utf-8") as f:
            json.dump({**corpus.manifest, "packed_dir": str(directory)}, f, ensure_ascii=False, indent=2)
    return corpus


# --------------------------------------------------------------------------
# QA blend — SFT-format QA text packed like the pool
# --------------------------------------------------------------------------

def qa_blend_fingerprint(tok_id: Dict[str, Any], qa_cfg: QABlendConfig, block_size: int,
                         clean_latin_rows: bool, corpus_params: Optional[Dict[str, Any]] = None) -> str:
    """``corpus_params`` (``training.corpus_params`` restricted to the blend's
    datasets) only enters the payload when non-empty, so entries packed
    before it existed keep their fingerprint."""
    payload_dict: Dict[str, Any] = {
        "tokenizer": tok_id, "datasets": list(qa_cfg.datasets), "split": qa_cfg.split,
        "block_size": block_size, "seed": qa_cfg.seed, "clean_latin_rows": clean_latin_rows,
    }
    relevant = {n: p for n, p in (corpus_params or {}).items() if n in qa_cfg.datasets and p}
    if relevant:
        payload_dict["corpus_params"] = relevant
    payload = json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pack_qa_blend(
    qa_cfg: QABlendConfig,
    block_size: int,
    cache_dir: Path,
    tokenizer: BaseTokenizer,
    tokenizer_type: str,
    clean_latin_rows: bool = False,
    cell_data_dir: Optional[Path] = None,
    corpus_params: Optional[Dict[str, Any]] = None,
) -> PackedCorpus:
    """Pack (or load) the *whole* QA train split(s) for this tokenizer.

    Records are rendered with the Phase 3 / eval surface form
    (``_format_qa_full``), encoded without truncation (BOS/EOS as the
    tokenizer emits them, trailing EOS guaranteed), shuffled with
    ``qa_cfg.seed``, concatenated and chunked into ``block_size`` blocks.
    Cached at ``<cache_dir>/qa_blend/<fp>/`` and shared across experiments;
    a phase takes the prefix it needs via ``PackedCorpus.take`` (strict, no
    wrap). The blend does not depend on the pool, so it is cached beside
    it rather than under it.
    """
    from ..finetune_corpora import _format_qa_full, filter_latin_records, load_corpora

    tok_id = tokenizer_identity(tokenizer, tokenizer_type)
    fp = qa_blend_fingerprint(tok_id, qa_cfg, block_size, clean_latin_rows, corpus_params)
    directory = Path(cache_dir) / QA_BLEND_SUBDIR / fp
    manifest_path = directory / QA_BLEND_MANIFEST

    corpus: Optional[PackedCorpus] = None
    if manifest_path.exists() and (directory / INPUT_IDS_FILE).exists():
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("fingerprint") == fp:
            logger.info("qa_blend: reusing packed QA corpus %s (%d blocks)", directory, existing["n_blocks"])
            corpus = _load_packed(directory, existing)

    if corpus is None:
        t0 = time.perf_counter()
        records = load_corpora(list(qa_cfg.datasets), splits=qa_cfg.split, corpus_params=corpus_params)
        n_loaded = len(records)
        if clean_latin_rows:
            records = filter_latin_records(records)
        if not records:
            raise ValueError(f"qa_blend: no records left from {list(qa_cfg.datasets)}/{qa_cfg.split}")
        eos_id = int(tokenizer.special_tokens["eos_token"])
        docs: List[EncodedDoc] = []
        per_source: Dict[str, Dict[str, int]] = {}
        words = 0
        for rec in records:
            text = _format_qa_full(rec)
            ids, char_ids = encode_document(tokenizer, text, eos_id)
            docs.append(EncodedDoc(source=rec.source, ids=ids, char_ids=char_ids))
            words += len(text.split())
            ps = per_source.setdefault(rec.source, {"records": 0, "tokens": 0})
            ps["records"] += 1
            ps["tokens"] += int(ids.shape[0])
        input_ids, char_ids, tail = pack_blocks(docs, block_size, qa_cfg.seed)
        n_tokens = int(sum(d.ids.shape[0] for d in docs))
        del docs
        manifest = {
            "fingerprint": fp,
            "tokenizer": tok_id,
            "datasets": list(qa_cfg.datasets),
            "split": qa_cfg.split,
            "clean_latin_rows": clean_latin_rows,
            "records_loaded": n_loaded,
            "n_records": len(records),
            "words": words,
            "n_tokens": n_tokens,
            "fertility": round(n_tokens / words, 4) if words else 0.0,
            "tokens_per_record": round(n_tokens / len(records), 1),
            "block_size": block_size,
            "seed": qa_cfg.seed,
            "n_blocks": int(input_ids.shape[0]),
            "n_tokens_packed": int(input_ids.shape[0] * block_size),
            "tail_tokens_dropped": tail,
            "per_source": per_source,
            "wall_sec": round(time.perf_counter() - t0, 1),
        }
        _publish_packed(directory, QA_BLEND_MANIFEST, manifest, input_ids, char_ids)
        logger.info(
            "qa_blend: packed %d records (%s/%s) → %d blocks × %d (fertility %.2f, %.0f tokens/record) in %.0fs → %s",
            len(records), "+".join(qa_cfg.datasets), qa_cfg.split, manifest["n_blocks"], block_size,
            manifest["fertility"], manifest["tokens_per_record"], manifest["wall_sec"], directory,
        )
        corpus = _load_packed(directory, manifest)
    corpus.kind = "qa_blend"

    if cell_data_dir is not None:
        cell_data_dir = Path(cell_data_dir)
        cell_data_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_data_dir / QA_BLEND_MANIFEST, "w", encoding="utf-8") as f:
            json.dump({**corpus.manifest, "packed_dir": str(directory)}, f, ensure_ascii=False, indent=2)
    return corpus
