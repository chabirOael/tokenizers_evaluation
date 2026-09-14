"""Packed raw-text pretraining mix for Phase 1 (embedding alignment) / Phase 2.

Two stages:

* **Stage A — pool** (``pool.py``): tokenizer-independent. Stream each
  source (``sources.py``), normalize + quality-filter + dialect-gate
  (``filters.py``), dedup across sources (``dedup.py``), keep
  ``share × pool.total_words`` words per source. Cached under
  ``cache_dir/<fingerprint>/`` and shared by every experiment cell.
* **Stage B — pack** (``packing.py``): per cell. Walk each source's pool in
  a fixed seeded order, tokenize with the *active* tokenizer, take documents
  until ``share × token_budget`` tokens, shuffle, concatenate with EOS
  separators, chunk into ``block_size`` blocks.

Imports here stay light so ``arabic_eval.data`` keeps working without
``datasets`` / ``datasketch`` installed; heavy modules are imported inside
the functions that need them.
"""
from .filters import (
    DialectScore,
    dialect_marker_score,
    normalize_document,
    quality_reason,
    sample_sentences,
    strip_wikipedia_sections,
    truncate_words,
)

__all__ = [
    "DialectScore",
    "dialect_marker_score",
    "normalize_document",
    "quality_reason",
    "sample_sentences",
    "strip_wikipedia_sections",
    "truncate_words",
]
