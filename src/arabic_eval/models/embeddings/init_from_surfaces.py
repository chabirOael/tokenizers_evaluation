"""Informed initialisation of a swapped vocabulary's embedding rows (``model.embedding_init``).

Why this exists (2026-09-22). ``resize_token_embeddings(model, N)`` with ``N`` below
the base vocabulary keeps the base model's **first N pretrained rows** — so a
from-scratch tokenizer's id 5 inherits whatever the base tokenizer's id 5 meant
(an ASCII character, a byte, an English piece). The mapping is arbitrary, the
literature treats it as random initialisation, and the AraRooPat arm B v2 that
trained on it for 37 M tokens never recovered: raw-text loss 1.34 nats/char
after Phase 2 against 0.82 for the untouched native model (see CLAUDE.md,
*Reinitialization behavior*). The remedies in the literature share one idea —
start each new token where the base model already represents what it stands
for (Fast Vocabulary Transfer, arXiv 2402.09977; "Beyond Initialization Loss",
arXiv 2608.03494: subword-composition init cuts CPT steps ~6×).

Methods (``EmbeddingInitConfig.method``):

* ``legacy``      — nothing beyond the resize; the pre-2026-09-22 behaviour.
* ``random``      — every new row N(0, 0.02²), the HF convention for added rows.
* ``mean``        — every new row = the global mean of the base matrix.
* ``surface_avg`` — for every token, ``BaseTokenizer.token_surfaces()`` gives the
  surface strings it stands for with weights; each surface is tokenized with the
  *base* HF tokenizer, the pretrained rows of its pieces are averaged (uniform,
  or ``char_len``: each piece weighted by its decoded character count), and the
  token's row is the weight-average over its surfaces. A token without any
  surface gets the global mean. ``norm: base_mean`` then rescales every new row
  to the mean L2 norm of the base rows (an average of unit-ish vectors is
  shorter than its parts).

The base matrix must be captured **before** the resize (``capture_base_embeddings``):
after it, only its first N rows survive. Qwen3-4B-Base and Llama-3.2-1B have
**tied** embeddings — one matrix serves as input embedding and output head — so
the rows are written in place into that one tensor and the tie survives; with
untied weights the same rows go into ``lm_head`` too. The input/output
asymmetry of arXiv 2608.03494 (a different average for the output side) is a
follow-up that needs an untied model; it is noted, not implemented.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("arabic_eval.models.embeddings.init_from_surfaces")

METHODS = ("legacy", "random", "mean", "surface_avg")
WEIGHTINGS = ("uniform", "char_len")
NORMS = ("none", "base_mean")


@dataclass
class EmbeddingInitReport:
    """What the initialiser did — recorded under ``all_metrics.json["training"]["embedding_init"]``."""
    method: str
    weighting: str
    norm: str
    base_tokenizer: Optional[str]
    rows_total: int
    rows_from_surfaces: int
    rows_fallback_mean: int
    rows_random: int
    surfaces_total: int
    surfaces_unique: int
    base_rows: int
    base_row_norm_mean: Optional[float]
    new_row_norm_mean_before: Optional[float]
    new_row_norm_mean_after: Optional[float]
    tied_embeddings: Optional[bool]
    wall_sec: float
    families: Dict[str, Dict[str, int]] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def capture_base_embeddings(model) -> torch.Tensor:
    """A float32 CPU copy of the model's input-embedding matrix, taken before any resize."""
    weight = model.get_input_embeddings().weight
    return weight.detach().to(device="cpu", dtype=torch.float32).clone()


def _mean_norm(rows: torch.Tensor) -> Optional[float]:
    if rows.numel() == 0:
        return None
    return round(float(rows.float().norm(dim=1).mean().item()), 4)


def _family_of(token_string: str) -> str:
    """Bracketed token families (``[ROOT_x]`` → ``ROOT``); anything else → ``other``."""
    if token_string.startswith("[") and "_" in token_string:
        return token_string[1:token_string.index("_")]
    if token_string.startswith("<") and token_string.endswith(">"):
        return "special"
    return "other"


def surface_vectors(
    surfaces: Sequence[str],
    base_matrix: torch.Tensor,
    hf_tokenizer,
    weighting: str = "uniform",
    batch_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One vector per surface: the (weighted) mean of the base rows of its pieces.

    Returns ``(vectors [S, D], ok [S])`` — ``ok`` is False for a surface whose
    base tokenization is empty (nothing to average). BOS / EOS are never
    added (``add_special_tokens=False``).
    """
    vecs = torch.zeros((len(surfaces), base_matrix.shape[1]), dtype=torch.float32, device=base_matrix.device)
    ok = torch.zeros(len(surfaces), dtype=torch.bool)
    specials = set(getattr(hf_tokenizer, "all_special_ids", []) or [])
    char_len_cache: Dict[int, float] = {}

    def piece_len(pid: int) -> float:
        if pid not in char_len_cache:
            try:
                char_len_cache[pid] = float(max(len(hf_tokenizer.decode([pid])), 1))
            except Exception:  # noqa: BLE001
                char_len_cache[pid] = 1.0
        return char_len_cache[pid]

    for start in range(0, len(surfaces), batch_size):
        chunk = list(surfaces[start:start + batch_size])
        enc = hf_tokenizer(chunk, add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            ids = [i for i in ids if i not in specials and 0 <= i < base_matrix.shape[0]]
            if not ids:
                continue
            idx = torch.tensor(ids, dtype=torch.long, device=base_matrix.device)
            rows = base_matrix.index_select(0, idx)
            if weighting == "char_len":
                w = torch.tensor([piece_len(i) for i in ids], dtype=torch.float32, device=base_matrix.device)
                vecs[start + j] = (rows * w[:, None]).sum(0) / w.sum()
            else:
                vecs[start + j] = rows.mean(0)
            ok[start + j] = True
    return vecs, ok


def build_init_matrix(
    method: str,
    base_matrix: torch.Tensor,
    new_vocab_size: int,
    token_surfaces: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    hf_tokenizer=None,
    weighting: str = "uniform",
    norm: str = "none",
    seed: int = 42,
    token_strings: Optional[Dict[int, str]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """The ``[new_vocab_size, D]`` matrix for ``method`` and the bookkeeping behind it.

    Pure: no model is touched. ``base_matrix`` is the *full* pre-resize matrix
    (float32). ``token_surfaces`` / ``hf_tokenizer`` are needed for ``surface_avg``
    only. ``token_strings`` (id → token string) drives the per-family coverage
    counts in the report and is optional.
    """
    if method not in METHODS:
        raise ValueError(f"embedding_init.method must be one of {METHODS}, got {method!r}")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"embedding_init.weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    if norm not in NORMS:
        raise ValueError(f"embedding_init.norm must be one of {NORMS}, got {norm!r}")
    dim = base_matrix.shape[1]
    global_mean = base_matrix.mean(0)
    stats: Dict[str, Any] = {"rows_from_surfaces": 0, "rows_fallback_mean": 0, "rows_random": 0,
                             "surfaces_total": 0, "surfaces_unique": 0, "families": {}}

    if method == "legacy":
        # The rows the resize left behind: the base matrix's first N rows (or the
        # whole base matrix padded with the HF random rows when N is larger).
        out = base_matrix[:new_vocab_size].clone()
        if new_vocab_size > base_matrix.shape[0]:
            gen = torch.Generator().manual_seed(seed)
            extra = torch.randn((new_vocab_size - base_matrix.shape[0], dim), generator=gen) * 0.02
            out = torch.cat([out, extra.to(out.device)], 0)
        return out, stats

    if method == "random":
        gen = torch.Generator().manual_seed(seed)
        out = (torch.randn((new_vocab_size, dim), generator=gen) * 0.02).to(base_matrix.device)
        stats["rows_random"] = new_vocab_size
        return out, stats

    out = global_mean.expand(new_vocab_size, dim).clone()
    if method == "mean":
        stats["rows_fallback_mean"] = new_vocab_size
        return out, stats

    # surface_avg
    if token_surfaces is None or hf_tokenizer is None:
        raise ValueError("surface_avg needs token_surfaces and a base HF tokenizer")
    unique: Dict[str, int] = {}
    for items in token_surfaces.values():
        for s, _w in items:
            if s and s not in unique:
                unique[s] = len(unique)
    stats["surfaces_total"] = sum(len(v) for v in token_surfaces.values())
    stats["surfaces_unique"] = len(unique)
    vecs, ok = surface_vectors(list(unique.keys()), base_matrix, hf_tokenizer, weighting)
    fam_cov: Dict[str, Dict[str, int]] = {}
    for tid in range(new_vocab_size):
        items = [(s, float(w)) for s, w in token_surfaces.get(tid, []) if s and s in unique and ok[unique[s]]]
        fam = _family_of(token_strings.get(tid, "")) if token_strings else "all"
        cov = fam_cov.setdefault(fam, {"tokens": 0, "with_surface": 0})
        cov["tokens"] += 1
        if not items:
            stats["rows_fallback_mean"] += 1
            continue
        idx = torch.tensor([unique[s] for s, _ in items], dtype=torch.long, device=base_matrix.device)
        w = torch.tensor([w for _, w in items], dtype=torch.float32, device=base_matrix.device)
        w = w / w.sum() if float(w.sum()) > 0 else torch.full_like(w, 1.0 / len(items))
        out[tid] = (vecs.index_select(0, idx) * w[:, None]).sum(0)
        stats["rows_from_surfaces"] += 1
        cov["with_surface"] += 1
    stats["families"] = fam_cov

    if norm == "base_mean":
        target = base_matrix.norm(dim=1).mean()
        norms = out.norm(dim=1, keepdim=True).clamp_min(1e-8)
        out = out * (target / norms)
    return out, stats


def apply_embedding_init(
    model,
    tokenizer,
    method: str,
    base_matrix: torch.Tensor,
    base_tokenizer_name: Optional[str],
    weighting: str = "uniform",
    norm: str = "none",
    seed: int = 42,
) -> EmbeddingInitReport:
    """Overwrite the (already resized) embedding rows of ``model`` in place.

    Writes into the existing weight tensors — never replaces the Parameter —
    so tied input/output embeddings stay tied. With untied weights the same
    rows are written into ``lm_head`` (no in/out asymmetry; see module doc).
    """
    t0 = time.perf_counter()
    emb = model.get_input_embeddings()
    new_vocab = int(emb.weight.shape[0])
    lm_head = getattr(model, "lm_head", None)
    tied = bool(lm_head is not None and lm_head.weight.data_ptr() == emb.weight.data_ptr())
    before = _mean_norm(emb.weight.detach()[:new_vocab])

    hf_tok = None
    surfaces = None
    token_strings: Optional[Dict[int, str]] = None
    if method == "surface_avg":
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(base_tokenizer_name, trust_remote_code=True)
        surfaces = tokenizer.token_surfaces()
        rev = getattr(tokenizer, "_reverse_vocab", None)
        if isinstance(rev, dict):
            token_strings = {int(k): str(v) for k, v in rev.items()}

    matrix, stats = build_init_matrix(
        method, base_matrix, new_vocab, surfaces, hf_tok, weighting=weighting, norm=norm, seed=seed,
        token_strings=token_strings,
    )
    if method != "legacy":
        with torch.no_grad():
            emb.weight.data.copy_(matrix.to(device=emb.weight.device, dtype=emb.weight.dtype))
            if lm_head is not None and not tied and lm_head.weight.shape[0] == new_vocab:
                lm_head.weight.data.copy_(matrix.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype))
    after = _mean_norm(emb.weight.detach()[:new_vocab])

    report = EmbeddingInitReport(
        method=method, weighting=weighting, norm=norm, base_tokenizer=base_tokenizer_name,
        rows_total=new_vocab,
        rows_from_surfaces=int(stats.get("rows_from_surfaces", 0)),
        rows_fallback_mean=int(stats.get("rows_fallback_mean", 0)),
        rows_random=int(stats.get("rows_random", 0)),
        surfaces_total=int(stats.get("surfaces_total", 0)),
        surfaces_unique=int(stats.get("surfaces_unique", 0)),
        base_rows=int(base_matrix.shape[0]),
        base_row_norm_mean=_mean_norm(base_matrix),
        new_row_norm_mean_before=before,
        new_row_norm_mean_after=after,
        tied_embeddings=tied,
        wall_sec=round(time.perf_counter() - t0, 2),
        families=stats.get("families", {}),
        note=("legacy: the resize's first-N pretrained rows, untouched" if method == "legacy" else
              "tied embeddings: one matrix serves input and output; no in/out asymmetry" if tied else
              "untied: the same rows written into lm_head"),
    )
    logger.info(
        "embedding_init %s (weighting=%s, norm=%s): %d rows — %d from surfaces, %d global-mean fallback, "
        "%d random; mean row norm base %.3f, new %s → %s; %.1fs",
        method, weighting, norm, report.rows_total, report.rows_from_surfaces, report.rows_fallback_mean,
        report.rows_random, report.base_row_norm_mean or 0.0, before, after, report.wall_sec,
    )
    return report
