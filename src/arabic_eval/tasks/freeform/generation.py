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

Two token-level knobs exist for **measurement only** (added 2026-09-23 for the
decoding ablation, ``DecodingConfig.repetition_penalty`` /
``no_repeat_ngram_size``; defaults 1.0 / 0 = the greedy rule above, byte for
byte): how much of a cell's loop rate is a greedy attractor that a penalty
removes. They are granularity-dependent by construction — one AraRooPat word
is ``[ROOT] [PAT]`` plus clitic tokens, so ``[CLITICP_و]`` or ``[CLITICE_ة]``
is "repeated" in every other word, while a BPE word is one or two pieces — so
a cell scored with them is not comparable with a greedy cell of another
tokenizer. The repetition penalty is HF's formula (CTRL: a logit of a token
already in the context is divided by the penalty when positive, multiplied
when negative; the prompt counts, as in HF) applied over the context
**minus each row's left padding**: HF's own processor penalizes every id in
``input_ids``, pads included, and the native Qwen / Llama wrappers pad with
the EOS id — every padded row of a batch would have its EOS penalized, i.e.
the output would depend on the batch composition. ``no_repeat_ngram_size``
is HF's processor unchanged (a pad n-gram can only ban a token after a
finished row's padding).

Stops: the tokenizer's EOS; a stop marker in the decoded text (the model
starting a new prompt block — a template section label or a header line, in
the templates' exact words, is the common failure of a small SFT'd model); a
**repetition loop** in the decoded text —
``metrics.detect_loop``: the text *ends* in enough contiguous copies of one
unit of up to ``LOOP_MAX_PERIOD`` words (a periodic tail; 5 copies of a
single word, 4 of a unit of ≤ 3 words, 3 beyond, a trailing partial copy
counted once it covers half the unit), or one letter or digit twenty times
in a row — tokenizer-agnostic (words and characters of decoded text,
not tokens — a repetition *penalty* would charge char-JABER for spelling), and
every stopped text is one ``degenerate_rate`` flags; the cap otherwise.
Markers and loops are checked every ``marker_check_every`` steps by decoding
the unfinished sequences. A loop-stopped generation is cut right after the
*first* copy of the repeated unit (``stop_reason="loop"``, ``loop_rule`` /
``loop_period`` on the row); ``generation_raw`` keeps the whole text and the
degeneration flag is computed on it, so ``degenerate_rate`` stays comparable
with runs that had no loop stop. (The cuDNN SDPA backend, whose
per-shape kernel compilation made the first and the odd-sized last batch cost
~300 s of host time each, is disabled at adapter load —
``models.llama_adapter.configure_sdpa_backends``.)

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
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

from arabic_eval.data import finetune_corpora as templates
from arabic_eval.data.answer_only_masking import strip_trailing_eos  # noqa: F401 — re-exported
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tasks.freeform.metrics import Loop, detect_loop
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType

logger = logging.getLogger(__name__)

STOP_REASONS: Tuple[str, ...] = ("eos", "marker", "loop", "cap")

# Stop markers = the model starting a new prompt block, in the exact words of the
# templates: the three block labels of the flat v1 template, the five ``### ``
# section labels of the sectioned v2 templates, and a header restart (the first
# three words of each v2 header — ``فيما يلي تعليمات`` / ``فيما يلي نص``). Derived
# from the constants of ``data.finetune_corpora`` so a template change moves the
# markers with it. Nothing looser: under the v2 template ``### `` is both the
# prompt's section syntax and ordinary markdown, and a bare ``\n###`` cut every
# sub-header the model opened inside its own answer (``### ملاحظات:``,
# ``### مثال متكامل:`` — 5 of the 5 marker stops of the untrained Qwen3-4B
# control, 2026-09-21); ``\nفيما يلي`` alone would cut a ``فيما يلي قائمة …``
# line. A marker is looked for anywhere in the decoded text (line start = the
# leading newline).
V1_STOP_MARKERS: Tuple[str, ...] = ("\nالسؤال:", "\nالسياق:", "\nالإجابة:")
HEADER_MARKER_WORDS = 3


def header_restart_marker(header: str) -> str:
    """``"\n"`` + the first ``HEADER_MARKER_WORDS`` words of a template header."""
    return "\n" + " ".join(header.split()[:HEADER_MARKER_WORDS])


LABEL_STOP_MARKERS: Tuple[str, ...] = tuple("\n" + label for label in (
    templates.INSTRUCTION_LABEL, templates.INPUT_LABEL, templates.CONTEXT_LABEL, templates.QUESTION_LABEL,
    templates.ANSWER_LABEL,
))
HEADER_STOP_MARKERS: Tuple[str, ...] = tuple(dict.fromkeys(header_restart_marker(h) for h in (
    templates.INSTRUCTION_HEADER, templates.INSTRUCTION_HEADER_WITH_INPUT, templates.QA_HEADER,
)))
DEFAULT_STOP_MARKERS: Tuple[str, ...] = V1_STOP_MARKERS + LABEL_STOP_MARKERS + HEADER_STOP_MARKERS


@dataclass
class DecodingConfig:
    max_output_chars: int = 2400     # 1200 until 2026-09-21: it cut 11 finished answers and capped 12 of the control's 250
    max_prompt_tokens: int = 512
    batch_size: int = 16
    token_cap_margin: float = 1.15
    token_cap_floor: int = 32
    token_cap_ceiling: int = 4096
    stop_markers: Tuple[str, ...] = DEFAULT_STOP_MARKERS
    marker_check_every: int = 16
    loop_stop: bool = True      # stop a sequence whose decoded text contains a repetition loop
    seed: int = 42
    # Token-level, granularity-dependent — for the decoding ablation, not for a cross-tokenizer comparison.
    repetition_penalty: float = 1.0   # HF's formula over the context minus left padding; 1.0 = off
    no_repeat_ngram_size: int = 0     # HF's n-gram ban; 0 = off

    def to_json(self) -> dict:
        return {"max_output_chars": self.max_output_chars, "max_prompt_tokens": self.max_prompt_tokens,
                "batch_size": self.batch_size, "token_cap_margin": self.token_cap_margin,
                "token_cap_floor": self.token_cap_floor, "token_cap_ceiling": self.token_cap_ceiling,
                "stop_markers": list(self.stop_markers), "marker_check_every": self.marker_check_every,
                "loop_stop": self.loop_stop, "seed": self.seed, "sampling": "greedy",
                "repetition_penalty": self.repetition_penalty, "no_repeat_ngram_size": self.no_repeat_ngram_size}


@dataclass
class GenerationRow:
    prompt_tokens: int
    gen_tokens: int
    generation_raw: str
    generation: str
    stop_reason: str            # one of STOP_REASONS
    hit_cap: bool
    hit_loop: bool              # a repetition loop ended it; ``generation`` keeps the first copy
    char_truncated: bool
    gen_time_sec: float         # the batch's wall time divided by its size
    loop_rule: Optional[str] = None      # "period" | "char" when hit_loop
    loop_period: Optional[int] = None    # the unit's length in words when hit_loop (1 for the char rule)


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


def truncate_at_marker(text: str, markers: Sequence[str]) -> Tuple[str, bool]:
    cut = len(text)
    for m in markers:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return (text[:cut], True) if cut < len(text) else (text, False)


class TextStop(NamedTuple):
    """The text-level stop applied to a decoded generation: ``text`` after the
    cut, ``reason`` (``"loop"`` / ``"marker"`` / ``""`` when nothing fired)
    and the ``Loop`` when a loop ended it."""
    text: str
    reason: str
    loop: Optional[Loop] = None


def text_stop(text: str, markers: Sequence[str], loop_stop: bool) -> TextStop:
    """Apply the text-level stops to a decoded generation: cut at the first
    stop marker, then (when ``loop_stop``) look for a repetition loop at the
    tail of what is left — a loop the model ran before starting a new section
    is still a loop — and keep the first copy of its unit. The reason is
    ``"loop"`` when one fired, else ``"marker"``, else ``""``."""
    text_m, hit_marker = truncate_at_marker(text, markers)
    loop = detect_loop(text_m) if loop_stop else None
    if loop is not None:
        return TextStop(text_m[: loop.cut], "loop", loop)
    if hit_marker:
        return TextStop(text_m, "marker")
    return TextStop(text, "")


def _context_repetition_penalty(penalty: float, pad_lens: Sequence[int]):
    """HF's ``RepetitionPenaltyLogitsProcessor`` formula over each row's context
    from its first non-pad position on (left padding: row ``r`` is padded on
    ``[0, pad_lens[r])``) — so a pad id that equals the EOS id is never penalized."""
    import torch
    from transformers import LogitsProcessor

    class _ContextRepetitionPenalty(LogitsProcessor):
        def __init__(self) -> None:
            self.penalty = float(penalty)
            self.pad_lens = torch.tensor(list(pad_lens), dtype=torch.long)

        def __call__(self, input_ids, scores):  # noqa: D401
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            valid = (pos >= self.pad_lens.to(input_ids.device).unsqueeze(1)).to(scores.dtype)
            seen = torch.zeros_like(scores).scatter_add_(1, input_ids, valid) > 0
            penalized = torch.where(scores < 0, scores * self.penalty, scores / self.penalty)
            return torch.where(seen, penalized, scores)

    return _ContextRepetitionPenalty()


def decoding_kwargs(cfg: DecodingConfig, pad_lens: Sequence[int]) -> dict:
    """The ``generate`` kwargs of the decoding knobs. At the defaults this is exactly
    the pre-2026-09-23 call (``repetition_penalty=1.0``, nothing else)."""
    kw: dict = {"repetition_penalty": 1.0}
    if cfg.repetition_penalty != 1.0:
        from transformers import LogitsProcessorList
        kw["logits_processor"] = LogitsProcessorList([_context_repetition_penalty(cfg.repetition_penalty, pad_lens)])
    if cfg.no_repeat_ngram_size > 0:
        kw["no_repeat_ngram_size"] = int(cfg.no_repeat_ngram_size)
    return kw


def _marker_stopping(tokenizer: BaseTokenizer, markers: Sequence[str], prompt_len: int, every: int,
                     loop_stop: bool = True):
    import torch
    from transformers import StoppingCriteria

    class _MarkerStopping(StoppingCriteria):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
            self.step += 1
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            if (not markers and not loop_stop) or self.step % every:
                return done
            for i in range(input_ids.shape[0]):
                text = tokenizer.decode(input_ids[i, prompt_len:].tolist())
                if text_stop(text, markers, loop_stop).reason:
                    done[i] = True
            return done

    return _MarkerStopping()


def generate_freeform(
    adapter: BaseModelAdapter,
    tokenizer: BaseTokenizer,
    prompts: Sequence[str],
    cfg: DecodingConfig,
    token_cap: int,
    warmup: bool = True,
) -> List[GenerationRow]:
    """Greedy-decode every prompt under the shared rules; rows in input order.
    ``warmup`` runs one untimed 8-token generate on the first batch first."""
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

    def _batch(idx: List[int]):
        width = max(len(encoded[i]) for i in idx)
        input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):                   # left padding
            ids = encoded[i]
            input_ids[r, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn[r, width - len(ids):] = 1
        return input_ids.to(device), attn.to(device), width

    def _pad_lens(idx: List[int], width: int) -> List[int]:
        return [width - len(encoded[i]) for i in idx]

    try:
        if order and warmup:
            # Untimed warm-up on the first batch (mirrors the tokenizer warm-up
            # of the pipeline): transformers' first generate call pays a large
            # one-off CPU-side cost that must not be billed to gen_chars_per_sec.
            idx = order[: cfg.batch_size]
            input_ids, attn, width = _batch(idx)
            t0 = time.perf_counter()
            with torch.inference_mode():
                adapter.generate(input_ids, attention_mask=attn, max_new_tokens=8, do_sample=False, num_beams=1,
                                 pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True,
                                 **decoding_kwargs(cfg, _pad_lens(idx, width)))
            logger.info("free-form generation: warm-up batch of %d × 8 tokens in %.1fs (not timed)",
                        len(idx), time.perf_counter() - t0)
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            input_ids, attn, width = _batch(idx)
            stopping = StoppingCriteriaList([_marker_stopping(tokenizer, cfg.stop_markers, width, cfg.marker_check_every,
                                                              loop_stop=cfg.loop_stop)])
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = adapter.generate(
                    input_ids, attention_mask=attn, max_new_tokens=token_cap, do_sample=False, num_beams=1,
                    pad_token_id=pad_id, eos_token_id=eos_id,
                    stopping_criteria=stopping, use_cache=True, **decoding_kwargs(cfg, _pad_lens(idx, width)),
                )
            batch_wall = time.perf_counter() - t0
            dt = batch_wall / len(idx)
            for r, i in enumerate(idx):
                new = out[r, width:].tolist()
                reason = "cap"
                if eos_id is not None and eos_id in new:
                    new = new[: new.index(eos_id)]
                    reason = "eos"
                raw = tokenizer.decode(new)
                text, text_reason, loop = text_stop(raw, cfg.stop_markers, cfg.loop_stop)
                if text_reason:
                    reason = text_reason
                elif reason == "cap" and len(new) < token_cap:
                    reason = "eos"            # HF ended it (pad-only tail); treat as a natural stop
                text = text.strip()
                char_truncated = len(text) > cfg.max_output_chars
                if char_truncated:
                    text = text[: cfg.max_output_chars].rstrip()
                rows[i] = GenerationRow(
                    prompt_tokens=len(encoded[i]), gen_tokens=len(new), generation_raw=raw, generation=text,
                    stop_reason=reason, hit_cap=(reason == "cap"), hit_loop=(reason == "loop"),
                    char_truncated=char_truncated, gen_time_sec=dt,
                    loop_rule=loop.rule if reason == "loop" and loop is not None else None,
                    loop_period=loop.period if reason == "loop" and loop is not None else None,
                )
            n_new = sum(rows[i].gen_tokens for i in idx)   # type: ignore[union-attr]
            logger.info("free-form generation: %d/%d prompts (batch of %d × ≤%d tokens, width %d, %d new tokens, %.1fs = %.1f tok/s)",
                        min(s + cfg.batch_size, len(order)), len(order), len(idx), token_cap, width,
                        n_new, batch_wall, n_new / batch_wall if batch_wall > 0 else 0.0)
    finally:
        if was_training:
            model.train()
    return rows  # type: ignore[return-value]
