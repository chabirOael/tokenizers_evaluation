"""Two-level cross-source deduplication for the pretraining-mix pool.

* ``ParagraphDeduper`` — exact hash of each normalized paragraph (line).
  A paragraph seen anywhere before in the pool is removed *from* the
  document; this is what kills shared boilerplate (nav bars, footers,
  Wikipedia lead sentences quoted on web pages). Paragraphs shorter than
  ``min_words`` are left alone — they are the short-line rule's business.
* ``MinHashDeduper`` — document-level MinHash + LSH over word shingles
  (datasketch). A near-duplicate of an already-kept document is dropped.
  Because ``pool.build_pool`` walks sources in ``dedup.priority`` order,
  "already kept" means "from a higher-priority source or earlier in this
  one", so the surviving copy is deterministic.

Both hash with ``hashlib.blake2b`` (stable across processes, unlike
``hash()`` under ``PYTHONHASHSEED``).
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Set, Tuple


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


class ParagraphDeduper:
    """Exact-hash paragraph dedup across every document seen so far."""

    def __init__(self, min_words: int = 3) -> None:
        self.min_words = min_words
        self._seen: Set[bytes] = set()

    def __len__(self) -> int:
        return len(self._seen)

    def filter(self, text: str) -> Tuple[str, int]:
        """Return ``(text_without_repeated_paragraphs, n_removed)``.
        Paragraphs of this document are registered as they are kept, so a
        paragraph repeated *within* the document is also collapsed."""
        kept: List[str] = []
        removed = 0
        for line in text.split("\n"):
            if len(line.split()) < self.min_words:
                kept.append(line)
                continue
            d = _digest(line)
            if d in self._seen:
                removed += 1
                continue
            self._seen.add(d)
            kept.append(line)
        return "\n".join(kept), removed


class MinHashDeduper:
    """Document-level near-duplicate detection (MinHash LSH)."""

    def __init__(
        self,
        num_perm: int = 128,
        shingle_words: int = 5,
        threshold: float = 0.80,
        seed: int = 1,
    ) -> None:
        from datasketch import MinHashLSH  # lazy: optional for tokenizer-only installs

        self.num_perm = num_perm
        self.shingle_words = max(1, shingle_words)
        self.threshold = threshold
        self.seed = seed
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def minhash(self, text: str):
        from datasketch import MinHash

        words = text.split()
        k = self.shingle_words
        if len(words) >= k:
            shingles = {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
        else:
            shingles = {" ".join(words)}
        m = MinHash(num_perm=self.num_perm, seed=self.seed)
        m.update_batch([s.encode("utf-8") for s in shingles])
        return m

    def check_and_add(self, key: str, text: str) -> Optional[str]:
        """Return the key of an already-kept near-duplicate, or ``None`` and
        register ``key`` as kept."""
        m = self.minhash(text)
        hits = self._lsh.query(m)
        if hits:
            return sorted(hits)[0]
        self._lsh.insert(key, m)
        self._n += 1
        return None
