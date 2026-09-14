"""Opt-in CAMeL dialect-ID signal for the MSA filter.

Runs ``DIDModel26`` in ``.venv-camel`` through the araroopat bridge. The
model is MADAR-trained and mislabels encyclopedic MSA, so the document
decision aggregates several sentences and uses a lenient threshold: a
document is flagged only when fewer than ``min_msa_share`` of its sampled
sentences are labeled MSA. Documents with no sentence long enough to
sample are *not* flagged (unmeasurable ≠ dialect).
"""
from __future__ import annotations

from typing import List, Optional

from ...config import MixCamelDidConfig
from .filters import sample_sentences


class CamelDialectScorer:
    def __init__(self, cfg: MixCamelDidConfig, bridge=None) -> None:
        self.cfg = cfg
        self._bridge = bridge          # injectable for tests
        self.n_sentences_scored = 0

    def _get_bridge(self):
        if self._bridge is None:
            from ...tokenizers.araroopat_bridge import get_shared_bridge
            self._bridge = get_shared_bridge()
        return self._bridge

    def msa_share(self, text: str) -> Optional[float]:
        """Share of sampled sentences labeled MSA, or ``None`` if nothing
        could be sampled."""
        sents = sample_sentences(text, self.cfg.sentences_per_doc, self.cfg.min_sentence_words)
        if not sents:
            return None
        preds = self._get_bridge().dialect_id(sents)
        self.n_sentences_scored += len(sents)
        n_msa = sum(1 for p in preds if p.get("top") == "MSA")
        return n_msa / len(preds)

    def is_dialect(self, text: str) -> bool:
        share = self.msa_share(text)
        return share is not None and share < self.cfg.min_msa_share
