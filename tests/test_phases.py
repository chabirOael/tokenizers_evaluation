"""Tests for ``arabic_eval.training.phases``.

Uses a tiny synthetic model + minimal BaseModelAdapter so phase logic can
be exercised without downloading Llama. The key invariants tested:

  - Phase 1 (frozen body): only embed_tokens + lm_head accumulate gradients
  - Phase 2 (full unfreeze): every parameter accumulates gradients
  - Step budget is honored even when train_loader has fewer batches
  - Early stopping triggers on a plateaued eval loss
  - Checkpoint is saved when ``save_checkpoint=true``
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import EarlyStoppingConfig, PhaseConfig
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tokenizers.base import BaseTokenizer
from arabic_eval.training.phases import run_phase


# --------------------------------------------------------------------------
# Tiny synthetic Llama-like model + adapter
# --------------------------------------------------------------------------

class TinyLlama(nn.Module):
    """Small transformer-shaped model with the right named-parameter layout."""
    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.model.layers:
            layer.self_attn = nn.Linear(hidden, hidden, bias=False)
            layer.mlp = nn.Linear(hidden, hidden, bias=False)
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = x + layer.self_attn(x)
            x = x + layer.mlp(x)
        x = self.model.norm(x)
        return self.lm_head(x)


class _TinyAdapter(BaseModelAdapter):
    """Minimal BaseModelAdapter for tests."""
    def __init__(self, model_name_or_path: str = "tiny", device: str = "cpu", **kwargs):
        self._model = TinyLlama()
        self._device = torch.device(device)
        self._model.to(self._device)

    def adapt_to_tokenizer(self, tokenizer: BaseTokenizer) -> None:  # pragma: no cover
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self._model(batch["input_ids"])
        labels = batch["labels"]
        # Causal LM loss with shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": logits}

    def generate(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def get_trainable_parameters(self):
        return [p for p in self._model.parameters() if p.requires_grad]

    def save_checkpoint(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), Path(path) / "model.pt")

    def load_checkpoint(self, path) -> None:  # pragma: no cover
        self._model.load_state_dict(torch.load(Path(path) / "model.pt"))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        return self._model


class _ToyDataset(Dataset):
    """Random integer sequences with identity labels (causal LM style)."""
    def __init__(self, vocab: int, seq_len: int, n: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab, (n, seq_len), generator=g)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _make_loader(vocab=32, seq_len=8, n=8, batch=4, seed=0) -> DataLoader:
    return DataLoader(_ToyDataset(vocab, seq_len, n, seed=seed),
                      batch_size=batch, shuffle=False)


def _phase_cfg(**kwargs) -> PhaseConfig:
    base = dict(
        datasets=["arabic_squad"],
        trainable_parameters=["*"],
        steps=10,
        learning_rate=1e-3,
        batch_size=4,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        max_length=64,
        loss_target="full_sequence",
        lr_scheduler="constant",
        warmup_steps=0,
        max_grad_norm=1.0,
        save_checkpoint=False,
    )
    base.update(kwargs)
    return PhaseConfig(**base)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def _snapshot_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _changed_params(before: Dict[str, torch.Tensor], model: nn.Module) -> List[str]:
    """Names whose weight tensor differs from the snapshot (any element changed)."""
    changed = []
    for n, p in model.named_parameters():
        if not torch.equal(before[n], p.detach()):
            changed.append(n)
    return changed


def test_phase1_only_embed_and_head_change(tmp_path):
    """Frozen body must NOT update — verify by weight-delta snapshot."""
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(
        trainable_parameters=["embed_tokens", "lm_head"],
        steps=4, learning_rate=1e-2,
    )
    loader = _make_loader()
    snap = _snapshot_params(adapter.model)
    res = run_phase(
        phase_name="embedding_alignment",
        adapter=adapter, phase_cfg=cfg, train_loader=loader,
        output_dir=tmp_path, bf16=False, fp16=False, logging_steps=10,
    )
    assert res.steps_completed == 4
    changed = _changed_params(snap, adapter.model)
    # Exactly the two trainable tensors must change
    assert sorted(changed) == sorted([
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]), f"unexpected change set: {changed}"


def test_phase2_all_params_change(tmp_path):
    """Wildcard unfreezes everything — every weight tensor must be touched."""
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(trainable_parameters=["*"], steps=4, learning_rate=1e-2)
    loader = _make_loader()
    snap = _snapshot_params(adapter.model)
    run_phase(
        phase_name="warmup",
        adapter=adapter, phase_cfg=cfg, train_loader=loader,
        output_dir=tmp_path, bf16=False, fp16=False, logging_steps=10,
    )
    changed = _changed_params(snap, adapter.model)
    all_names = [n for n, _ in adapter.model.named_parameters()]
    # Every parameter tensor whose gradient flows must change with LR=1e-2 over 4 steps
    missing = set(all_names) - set(changed)
    assert not missing, f"params that did not change: {missing}"


def test_step_budget_honored_when_loader_shorter_than_steps(tmp_path):
    """8 examples / batch 4 = 2 batches per epoch; steps=10 must keep going."""
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(steps=10)
    loader = _make_loader(n=8, batch=4)
    res = run_phase(
        phase_name="embedding_alignment",
        adapter=adapter, phase_cfg=cfg, train_loader=loader,
        output_dir=tmp_path, bf16=False, fp16=False, logging_steps=100,
    )
    assert res.steps_completed == 10
    assert len(res.train_losses) == 10


def test_early_stop_triggers_on_plateau(tmp_path):
    """Patience=2 + tiny min_delta + no real improvement → must early-stop."""
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(
        steps=200,
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            metric="eval_loss",
            eval_every_n_steps=2,
            patience=2,
            min_delta=0.5,           # huge — model never beats
            min_steps_before_stop=2,
            restore_best_at_end=True,
            eval_splits={"tydiqa_arabic": "validation"},
        ),
    )
    train = _make_loader(n=16, batch=4, seed=0)
    eval_loader = _make_loader(n=8, batch=4, seed=1)
    res = run_phase(
        phase_name="sft",
        adapter=adapter, phase_cfg=cfg, train_loader=train,
        eval_loader=eval_loader, output_dir=tmp_path, bf16=False, fp16=False,
        logging_steps=100,
    )
    assert res.early_stopped, "should have early-stopped on plateaued eval loss"
    assert res.steps_completed < 200
    # We must have recorded at least 'patience+1' eval points before stopping
    assert len(res.eval_losses) >= cfg.early_stopping.patience + 1


def test_early_stop_requires_eval_loader(tmp_path):
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(
        steps=10,
        early_stopping=EarlyStoppingConfig(
            enabled=True, eval_every_n_steps=5,
            patience=2, min_steps_before_stop=0,
        ),
    )
    train = _make_loader()
    with pytest.raises(ValueError, match="no eval_loader"):
        run_phase(
            phase_name="sft",
            adapter=adapter, phase_cfg=cfg, train_loader=train,
            eval_loader=None, output_dir=tmp_path, bf16=False, fp16=False,
        )


def test_checkpoint_saved_when_enabled(tmp_path):
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(steps=2, save_checkpoint=True)
    loader = _make_loader()
    res = run_phase(
        phase_name="embedding_alignment",
        adapter=adapter, phase_cfg=cfg, train_loader=loader,
        output_dir=tmp_path, bf16=False, fp16=False, logging_steps=100,
    )
    assert res.checkpoint_path is not None
    assert (Path(res.checkpoint_path) / "model.pt").exists()


def test_loss_decreases_on_simple_task(tmp_path):
    """Tiny sanity: 50 steps of LR=1e-2 should decrease train loss on identity-LM."""
    adapter = _TinyAdapter(device="cpu")
    cfg = _phase_cfg(steps=50, learning_rate=1e-2, lr_scheduler="constant")
    loader = _make_loader(n=8, batch=4)
    res = run_phase(
        phase_name="warmup",
        adapter=adapter, phase_cfg=cfg, train_loader=loader,
        output_dir=tmp_path, bf16=False, fp16=False, logging_steps=100,
    )
    first_5 = sum(l for _, l in res.train_losses[:5]) / 5
    last_5 = sum(l for _, l in res.train_losses[-5:]) / 5
    assert last_5 < first_5, f"loss did not decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
