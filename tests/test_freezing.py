"""Tests for ``arabic_eval.training.freezing``.

Uses a tiny synthetic ``nn.Module`` whose named parameters mirror the
Llama path-shape (``model.embed_tokens.weight``, ``model.layers.<i>....``,
``lm_head.weight``) so the substring matcher is tested under the exact
strings the production model produces, without needing to download Llama.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.training.freezing import (
    WILDCARD,
    apply_trainable_filter,
    unfreeze_all,
)


class TinyLlamaLike(nn.Module):
    """Mirror of Llama's named-parameter shape; weights are random."""
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, 16)
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.model.layers:
            layer.self_attn = nn.Linear(16, 16, bias=False)
            layer.mlp = nn.Linear(16, 16, bias=False)
            layer.input_layernorm = nn.LayerNorm(16)
        self.model.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 64, bias=False)


def _all_param_names(m: nn.Module) -> list[str]:
    return [n for n, _ in m.named_parameters()]


def _trainable_names(m: nn.Module) -> list[str]:
    return [n for n, p in m.named_parameters() if p.requires_grad]


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------

def test_phase1_freeze_body_trains_embed_and_head():
    m = TinyLlamaLike()
    out = apply_trainable_filter(m, ["embed_tokens", "lm_head"])
    trainable = _trainable_names(m)
    # Returned list matches what the model agrees is trainable
    assert sorted(out) == sorted(trainable)
    # Exactly two trainable names: embed_tokens.weight and lm_head.weight
    assert sorted(trainable) == sorted([
        "model.embed_tokens.weight",
        "lm_head.weight",
    ])
    # Body is frozen
    body_names = [n for n in _all_param_names(m) if n not in trainable]
    assert all(not dict(m.named_parameters())[n].requires_grad for n in body_names)
    assert len(body_names) > 0  # sanity


def test_wildcard_unfreezes_all():
    m = TinyLlamaLike()
    # Pre-freeze something to make sure the wildcard *replaces* state, not unions
    m.lm_head.weight.requires_grad = False
    out = apply_trainable_filter(m, [WILDCARD])
    trainable = _trainable_names(m)
    assert sorted(out) == sorted(trainable)
    assert sorted(trainable) == sorted(_all_param_names(m))


def test_unfreeze_all_helper():
    m = TinyLlamaLike()
    apply_trainable_filter(m, ["embed_tokens", "lm_head"])
    assert len(_trainable_names(m)) == 2
    unfreeze_all(m)
    assert sorted(_trainable_names(m)) == sorted(_all_param_names(m))


def test_substring_matches_multiple_params():
    """``self_attn`` matches every layer's attention; check the count."""
    m = TinyLlamaLike()
    apply_trainable_filter(m, ["self_attn"])
    trainable = _trainable_names(m)
    # 2 layers × 1 weight each
    assert len(trainable) == 2
    assert all("self_attn" in n for n in trainable)


def test_substrings_union_semantics():
    """Multiple substrings → union of matches."""
    m = TinyLlamaLike()
    apply_trainable_filter(m, ["embed_tokens", "self_attn"])
    trainable = _trainable_names(m)
    # 1 (embed) + 2 (self_attn) = 3
    assert len(trainable) == 3


# --------------------------------------------------------------------------
# Validation failures
# --------------------------------------------------------------------------

def test_empty_substrings_rejected():
    m = TinyLlamaLike()
    with pytest.raises(ValueError, match="non-empty list"):
        apply_trainable_filter(m, [])


def test_non_string_entry_rejected():
    m = TinyLlamaLike()
    with pytest.raises(ValueError, match="non-empty strings"):
        apply_trainable_filter(m, ["embed_tokens", ""])


def test_wildcard_mixed_with_other_rejected():
    m = TinyLlamaLike()
    with pytest.raises(ValueError, match="cannot mix"):
        apply_trainable_filter(m, [WILDCARD, "embed_tokens"])


def test_typo_substring_caught_when_alone():
    """Sole substring matches nothing → AssertionError (no params trainable)."""
    m = TinyLlamaLike()
    with pytest.raises(AssertionError, match="matched no parameters"):
        apply_trainable_filter(m, ["embeed_tokens"])  # double-e typo


def test_partial_miss_warns_but_does_not_raise(caplog):
    """One substring matches nothing while another DOES match → warn, not raise.

    This is the Llama-3.2-1B tied-embeddings case: ``lm_head.weight is
    model.embed_tokens.weight``, so ``lm_head`` doesn't appear in
    ``named_parameters()`` even though we semantically want to train it.
    """
    import logging
    m = TinyLlamaLike()
    with caplog.at_level(logging.WARNING):
        trainable = apply_trainable_filter(m, ["embed_tokens", "frobnicate"])
    assert trainable == ["model.embed_tokens.weight"]
    assert any("frobnicate" in r.message for r in caplog.records)
    assert any("matched no parameters" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Spec compliance: re-applying after wildcard reproduces the canonical pattern
# --------------------------------------------------------------------------

def test_spec_loop_equivalent_after_wildcard():
    """Apply ['*'] then ['embed_tokens', 'lm_head']; result must equal the spec's loop."""
    m = TinyLlamaLike()
    apply_trainable_filter(m, [WILDCARD])
    apply_trainable_filter(m, ["embed_tokens", "lm_head"])
    # Exactly the spec's expectation
    for name, param in m.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
