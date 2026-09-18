"""Batched greedy generation with a tokenizer-independent budget.

Fixed decoding across every tokenizer means the *same* rules, not the same
token count: a 256-token budget is 250 characters for char-JABER and 800 for
BPE. So the budget is ``max_output_chars`` of decoded text, turned into a
per-tokenizer ``max_new_tokens`` from the tokenizer's measured
characters-per-token on the reference answers (with a margin), and every
generation is cut to the character budget afterwards. Pure greedy: no
sampling, no repetition penalty and no n-gram blocking — any of those is
granularity-dependent (a character-level tokenizer repeats characters by
nature) and would be a confound. Repetition loops are left for the metrics
and the judge to penalize.

Stops: the tokenizer's EOS, or a stop marker in the decoded text (the model
starting a new ``السؤال:`` / ``السياق:`` block is the common failure of a
small SFT'd model), checked every ``marker_check_every`` steps by decoding
the unfinished sequences; the cap otherwise.

Generation is possible for the ``standard`` and ``char_jaber`` embedding
families only. CharacterBERT's word-level output head and Charformer's
non-causal GBST cannot generate; ``generation_supported`` says so and the task
records a typed status instead of a number.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType

logger = logging.getLogger(__name__)

DEFAULT_STOP_MARKERS: Tuple[str, ...] = ("\nالسؤال:", "\nالسياق:", "\nالإجابة:", "\n###")


@dataclass
class DecodingConfig:
    max_output_chars: int = 1200
    max_prompt_tokens: int = 512
    batch_size: int = 16
    token_cap_margin: float = 1.15
    token_cap_floor: int = 32
    token_cap_ceiling: int = 4096
    stop_markers: Tuple[str, ...] = DEFAULT_STOP_MARKERS
    marker_check_every: int = 16
    seed: int = 42

    def to_json(self) -> dict:
        return {"max_output_chars": self.max_output_chars, "max_prompt_tokens": self.max_prompt_tokens,
                "batch_size": self.batch_size, "token_cap_margin": self.token_cap_margin,
                "token_cap_floor": self.token_cap_floor, "token_cap_ceiling": self.token_cap_ceiling,
                "stop_markers": list(self.stop_markers), "marker_check_every": self.marker_check_every,
                "seed": self.seed, "sampling": "greedy", "repetition_penalty": 1.0}


@dataclass
class GenerationRow:
    prompt_tokens: int
    gen_tokens: int
    generation_raw: str
    generation: str
    stop_reason: str            # "eos" | "marker" | "cap"
    hit_cap: bool
    char_truncated: bool
    gen_time_sec: float         # the batch's wall time divided by its size


def generation_supported(tokenizer: BaseTokenizer) -> Tuple[bool, str]:
    et = tokenizer.embedding_type
    if et == EmbeddingType.CHARACTER_CNN:
        return False, "character_cnn: the output head indexes a word/morpheme vocabulary; no autoregressive decoding"
    if et == EmbeddingType.CHARFORMER:
        return False, "charformer: GBST pools blocks X[i:i+b], position i sees i+M-1; non-causal, no decoding"
    return True, ""


def measure_chars_per_token(tokenizer: BaseTokenizer, texts: Sequence[str]) -> float:
    """Pooled characters per token id over ``texts`` (special ids included,
    which only makes the cap a little more generous)."""
    chars = tokens = 0
    for t in texts:
        if not t:
            continue
        chars += len(t)
        tokens += len(tokenizer.encode(t).input_ids)
    return chars / tokens if tokens else 1.0


def derive_token_cap(cfg: DecodingConfig, chars_per_token: float) -> int:
    cap = math.ceil(cfg.max_output_chars / max(chars_per_token, 1e-6) * cfg.token_cap_margin) + 8
    return int(min(max(cap, cfg.token_cap_floor), cfg.token_cap_ceiling))


def strip_trailing_eos(ids: List[int], eos_id: Optional[int]) -> List[int]:
    """Every from-scratch tokenizer appends ``</s>`` to a standalone encoding
    (the LCP masking helper exists for the same reason); a prompt must not end
    with it or the model is asked to continue past an end-of-text."""
    if eos_id is not None and ids and ids[-1] == eos_id:
        return ids[:-1]
    return ids


def truncate_at_marker(text: str, markers: Sequence[str]) -> Tuple[str, bool]:
    cut = len(text)
    for m in markers:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return (text[:cut], True) if cut < len(text) else (text, False)


def _marker_stopping(tokenizer: BaseTokenizer, markers: Sequence[str], prompt_len: int, every: int):
    import torch
    from transformers import StoppingCriteria

    class _MarkerStopping(StoppingCriteria):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
            self.step += 1
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            if not markers or self.step % every:
                return done
            for i in range(input_ids.shape[0]):
                text = tokenizer.decode(input_ids[i, prompt_len:].tolist())
                if any(m in text for m in markers):
                    done[i] = True
            return done

    return _MarkerStopping()


def generate_freeform(
    adapter: BaseModelAdapter,
    tokenizer: BaseTokenizer,
    prompts: Sequence[str],
    cfg: DecodingConfig,
    token_cap: int,
) -> List[GenerationRow]:
    """Greedy-decode every prompt under the shared rules; rows in input order."""
    import torch
    from transformers import StoppingCriteriaList

    specials = tokenizer.special_tokens
    pad_id = int(specials.get("pad_token", 0))
    eos_id = specials.get("eos_token")
    eos_id = None if eos_id is None else int(eos_id)

    encoded: List[List[int]] = []
    for p in prompts:
        ids = list(tokenizer.encode(p, max_length=cfg.max_prompt_tokens, truncation=True).input_ids)
        encoded.append(strip_trailing_eos(ids, eos_id))
    order = sorted(range(len(prompts)), key=lambda i: -len(encoded[i]))   # long first: least padding per batch

    model = adapter.model
    was_training = model.training
    model.eval()
    device = adapter.device
    torch.manual_seed(cfg.seed)     # greedy is seed-free; recorded for the manifest all the same
    rows: List[Optional[GenerationRow]] = [None] * len(prompts)
    try:
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            width = max(len(encoded[i]) for i in idx)
            input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):                   # left padding
                ids = encoded[i]
                input_ids[r, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attn[r, width - len(ids):] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            stopping = StoppingCriteriaList([_marker_stopping(tokenizer, cfg.stop_markers, width, cfg.marker_check_every)])
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = adapter.generate(
                    input_ids, attention_mask=attn, max_new_tokens=token_cap, do_sample=False, num_beams=1,
                    repetition_penalty=1.0, pad_token_id=pad_id, eos_token_id=eos_id,
                    stopping_criteria=stopping, use_cache=True,
                )
            dt = (time.perf_counter() - t0) / len(idx)
            for r, i in enumerate(idx):
                new = out[r, width:].tolist()
                reason = "cap"
                if eos_id is not None and eos_id in new:
                    new = new[: new.index(eos_id)]
                    reason = "eos"
                raw = tokenizer.decode(new)
                text, hit_marker = truncate_at_marker(raw, cfg.stop_markers)
                if hit_marker:
                    reason = "marker"
                elif reason == "cap" and len(new) < token_cap:
                    reason = "eos"            # HF ended it (pad-only tail); treat as a natural stop
                text = text.strip()
                char_truncated = len(text) > cfg.max_output_chars
                if char_truncated:
                    text = text[: cfg.max_output_chars].rstrip()
                rows[i] = GenerationRow(
                    prompt_tokens=len(encoded[i]), gen_tokens=len(new), generation_raw=raw, generation=text,
                    stop_reason=reason, hit_cap=(reason == "cap"), char_truncated=char_truncated, gen_time_sec=dt,
                )
            logger.info("free-form generation: %d/%d prompts (batch of %d, %d new tokens max, %.1fs)",
                        min(s + cfg.batch_size, len(order)), len(order), len(idx), token_cap, dt * len(idx))
    finally:
        if was_training:
            model.train()
    return rows  # type: ignore[return-value]
