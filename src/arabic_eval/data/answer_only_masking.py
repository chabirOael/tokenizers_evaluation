"""Answer-only label masking via longest-common-prefix.

Used by both the 3-phase finetune corpora (Phase 2 and Phase 3, when
``loss_target='answer_only'``) and the LightEval SFT dataloader.

The naive approach — mask ``labels[:len(prompt_enc)] = -100`` — is wrong
for tokenizers that auto-append ``</s>`` to standalone encodings: the
prompt encoding ends with EOS at index ``len(prompt) - 1``, while the
same index in the full encoding is the *first answer token*. Length-based
masking eats the first answer token. The LCP approach tolerates any
end-of-string artifacts because it walks the prefix of the two encodings
in lockstep until they diverge.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


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
    lcp = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        lcp += 1
    full_len = len(full_ids)
    if lcp >= full_len:
        return None
    labels = list(full_ids)
    for i in range(lcp):
        labels[i] = -100
    return labels
