"""Answer-only label masking via longest-common-prefix.

Used by the 3-phase finetune corpora (Phase 2 and Phase 3, when
``loss_target='answer_only'``) and — through ``continuation_start`` — by the
LightEval MCQ scorer, so training and scoring locate an answer / continuation
with one rule.

The naive approach — mask ``labels[:len(prompt_enc)] = -100`` — is wrong
for tokenizers that auto-append ``</s>`` to standalone encodings: the
prompt encoding ends with EOS at index ``len(prompt) - 1``, while the
same index in the full encoding is the *first answer token*. Length-based
masking eats the first answer token. The LCP approach tolerates any
end-of-string artifacts because it walks the prefix of the two encodings
in lockstep until they diverge.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def common_prefix_len(a: Sequence, b: Sequence) -> int:
    """Length of the longest common prefix of two id sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def strip_trailing_eos(ids: List[int], eos_id: Optional[int]) -> List[int]:
    """Drop one trailing ``eos_id`` from ``ids``.

    Every from-scratch tokenizer appends ``</s>`` to a standalone encoding.
    A prompt must not end with it (the model would be asked to continue past
    an end-of-text), and a scored continuation must not include it (the
    scorer would credit P(``</s>``) to the answer). The native wrappers append
    nothing, so for them this is the identity."""
    if eos_id is not None and ids and ids[-1] == eos_id:
        return ids[:-1]
    return ids


def continuation_start(
    context_ids: Sequence[int],
    full_ids: Sequence[int],
    eos_id: Optional[int] = None,
) -> Tuple[List[int], int]:
    """Where the continuation begins in ``full_ids``: ``(full, k)``.

    ``full`` is ``full_ids`` with one trailing ``eos_id`` stripped and ``k``
    the longest common prefix of the two encodings after stripping the same
    from ``context_ids``; the continuation is ``full[k:]``, empty when
    truncation left no room for it (``k >= len(full)``). Stripping first is
    what makes the prefix meet the continuation: the context encoding ends in
    ``</s>`` exactly where the full encoding has the first continuation token.
    """
    ctx = strip_trailing_eos(list(context_ids), eos_id)
    full = strip_trailing_eos(list(full_ids), eos_id)
    return full, common_prefix_len(ctx, full)


def compute_answer_only_labels(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
) -> Optional[List[int]]:
    """Return labels for ``full_ids`` with prompt span masked to -100.

    Computes the longest common prefix length ``lcp`` between
    ``prompt_ids`` and ``full_ids``, then returns
    ``[-100] * lcp + full_ids[lcp:]``.

    Returns ``None`` if ``lcp >= len(full_ids)`` — meaning truncation cut
    the answer span entirely; the example should be skipped (zero gradient
    contribution otherwise).
    """
    lcp = common_prefix_len(prompt_ids, full_ids)
    full_len = len(full_ids)
    if lcp >= full_len:
        return None
    labels = list(full_ids)
    for i in range(lcp):
        labels[i] = -100
    return labels
