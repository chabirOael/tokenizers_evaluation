"""Parameter freezing for the 3-phase training pipeline.

Phase 1 (embedding alignment) freezes the transformer body and trains only
``embed_tokens`` + ``lm_head``; Phases 2 and 3 unfreeze everything. Both
cases are expressed as a list of substrings matched against
``model.named_parameters()`` — see ``apply_trainable_filter``.

The substring-match semantics mirror the freezing pattern in the task
spec:

    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

The wildcard ``"*"`` (alone) means "all parameters trainable".
"""
from __future__ import annotations

import logging
from typing import List

import torch.nn as nn

logger = logging.getLogger(__name__)

WILDCARD = "*"


def apply_trainable_filter(model: nn.Module, substrings: List[str]) -> List[str]:
    """Set ``requires_grad`` on every parameter, return the names that are trainable.

    A parameter is trainable iff ``substrings == ["*"]`` (wildcard) or its
    full name (per ``model.named_parameters()``) contains any of the
    substrings.

    Validation:
      - ``substrings`` must be a non-empty list of non-empty strings;
      - ``WILDCARD`` cannot be mixed with other entries;
      - if NO substring matches any parameter, raise (typo guard);
      - if SOME substrings match nothing while others do, log a warning —
        this is the legitimate Llama-3.2-1B tied-embeddings case, where
        ``lm_head.weight is model.embed_tokens.weight`` so ``lm_head`` is
        absent from ``named_parameters()`` even though we semantically
        want to train it (training embed_tokens IS training lm_head);
      - assert every trainable param contains at least one substring (the
        spec's output-direction assertion).
    """
    if not isinstance(substrings, list) or not substrings:
        raise ValueError(f"substrings must be a non-empty list, got {substrings!r}")
    if any(not isinstance(s, str) or not s for s in substrings):
        raise ValueError(f"substrings must all be non-empty strings, got {substrings!r}")
    if WILDCARD in substrings and len(substrings) > 1:
        raise ValueError(
            f"cannot mix '{WILDCARD}' with other entries; use ['{WILDCARD}'] alone for 'all parameters'"
        )

    all_trainable = (substrings == [WILDCARD])
    matches = {s: 0 for s in substrings}
    trainable_names: List[str] = []

    for name, param in model.named_parameters():
        if all_trainable:
            param.requires_grad = True
            trainable_names.append(name)
            continue
        hit = False
        for s in substrings:
            if s in name:
                matches[s] += 1
                hit = True
        param.requires_grad = hit
        if hit:
            trainable_names.append(name)

    if not all_trainable:
        zero_match = [s for s, n in matches.items() if n == 0]
        if zero_match and not trainable_names:
            available = sorted({n.split(".")[0] for n, _ in model.named_parameters()})
            raise AssertionError(
                f"substring(s) {zero_match} matched no parameters AND no other "
                f"substring matched anything either. Likely typo. Top-level "
                f"modules available: {available[:15]}"
            )
        if zero_match:
            logger.warning(
                "freezing: substring(s) %s matched no parameters (other "
                "substrings did match). Common cause: tied embeddings — e.g. "
                "Llama-3.2-1B has lm_head.weight is embed_tokens.weight, so "
                "'lm_head' is absent from named_parameters(). Continuing.",
                zero_match,
            )
        # Spec assertion: every trainable name contains at least one substring.
        for n in trainable_names:
            assert any(s in n for s in substrings), (
                f"unexpected trainable parameter {n!r} for substrings={substrings}"
            )

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_trainable_params / max(n_total_params, 1)
    logger.info(
        "freezing: substrings=%s -> %d/%d trainable params (%.2f%%); first 5 names: %s",
        substrings, n_trainable_params, n_total_params, pct, trainable_names[:5],
    )
    return trainable_names


def unfreeze_all(model: nn.Module) -> None:
    """Mark every parameter trainable — used between phases that re-grow the trainable set."""
    for param in model.parameters():
        param.requires_grad = True
