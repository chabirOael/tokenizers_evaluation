"""BERTScore (Zhang et al., 2020) on plain ``transformers``.

The ``bert_score`` package breaks under transformers 5 (its empty-string path
calls a tokenizer method that no longer exists), so the metric is implemented
here: contextual embeddings from one layer of an encoder, greedy cosine
matching, precision / recall / F1. No IDF weighting, no baseline rescaling —
raw scores, comparable across tokenizers because every candidate is *decoded
text*. Like the reference implementation, every token (special ones
included) is a match target but only non-special tokens are averaged (their
IDF weight is 0 there).

Default model: ``xlm-roberta-large`` layer 17, the multilingual default the
reference implementation recommends. Verified 2026-09-18 against
``bert_score`` 0.3.12 on non-empty Arabic strings: max |Δ| on P / R / F1 of
1e-4 (the empty-candidate path is ours: it scores 0).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class BertScorer:
    def __init__(self, model_name: str = "xlm-roberta-large", layer: int = 17, device: Optional[str] = None,
                 batch_size: int = 32, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name, self.layer, self.batch_size, self.max_length = model_name, layer, batch_size, max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self._model.to(self.device).eval()
        n_layers = self._model.config.num_hidden_layers
        if not 0 <= layer <= n_layers:
            raise ValueError(f"layer {layer} out of range for {model_name} ({n_layers} layers)")

    def _embed(self, texts: Sequence[str]) -> List[Tuple["torch.Tensor", "torch.Tensor"]]:
        """Per text: (``[n_tokens, d]`` L2-normalized embeddings of every real
        token, specials included; ``[n_tokens]`` bool mask of the non-special ones)."""
        import torch
        out: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                batch = self._tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length,
                                  return_tensors="pt", return_special_tokens_mask=True)
                special = batch.pop("special_tokens_mask").bool()
                batch = batch.to(self.device)
                hs = self._model(**batch).hidden_states[self.layer]
                hs = torch.nn.functional.normalize(hs.float(), dim=-1).cpu()
                real = batch["attention_mask"].bool().cpu()
                for row, i in enumerate(idx):
                    out[i] = (hs[row][real[row]], ~special[row][real[row]])
        return out  # type: ignore[return-value]

    def score(self, candidates: Sequence[str], references: Sequence[str]) -> Tuple[List[float], List[float], List[float]]:
        """Precision, recall, F1 per pair. An empty candidate scores 0."""
        if len(candidates) != len(references):
            raise ValueError("candidates and references differ in length")
        nonempty = [i for i, c in enumerate(candidates) if c.strip()]
        P = [0.0] * len(candidates)
        R = [0.0] * len(candidates)
        F = [0.0] * len(candidates)
        if not nonempty:
            return P, R, F
        ce = self._embed([candidates[i] for i in nonempty])
        re_ = self._embed([references[i] for i in nonempty])
        for k, i in enumerate(nonempty):
            (c, c_keep), (r, r_keep) = ce[k], re_[k]
            if int(c_keep.sum()) == 0 or int(r_keep.sum()) == 0:
                continue
            sim = c @ r.T
            p = float(sim.max(dim=1).values[c_keep].mean())
            rr = float(sim.max(dim=0).values[r_keep].mean())
            P[i], R[i] = p, rr
            F[i] = 0.0 if p + rr == 0 else 2 * p * rr / (p + rr)
        return P, R, F
