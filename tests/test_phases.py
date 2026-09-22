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


class TestSchedulerHorizonInUpdates:
    """``steps`` / ``warmup_steps`` are micro-steps; the scheduler steps once per
    optimizer update (every ``gradient_accumulation_steps`` micro-steps), so its
    horizon must be ``ceil(steps / accum)`` updates. Before the fix the cosine
    got the micro-step count and at accumulation 4 decayed a quarter of the way
    by the end of the phase (``lr ≈ 0.85 × peak`` instead of ≈ 0)."""

    @staticmethod
    def _lr_trace(**cfg_kwargs):
        """Run a phase with an optimizer-hooked LR log: one entry per update."""
        from torch.optim import AdamW as _AdamW
        seen: List[float] = []
        orig_step = _AdamW.step

        def spy(self, *a, **k):
            seen.append(self.param_groups[0]["lr"])
            return orig_step(self, *a, **k)

        _AdamW.step = spy
        try:
            adapter = _TinyAdapter(device="cpu")
            cfg = _phase_cfg(**cfg_kwargs)
            loader = _make_loader(n=32, batch=4)
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                res = run_phase(
                    phase_name="warmup", adapter=adapter, phase_cfg=cfg,
                    train_loader=loader, output_dir=Path(td),
                    bf16=False, fp16=False, logging_steps=1000,
                )
        finally:
            _AdamW.step = orig_step
        return seen, res

    def test_cosine_reaches_zero_with_accumulation(self):
        """(a) steps=8, accum=4, cosine, no warmup → 2 updates over a 2-update
        horizon: the LR used by the last update is at progress 1/2 (cos → 0.5×peak)
        and the LR *after* the phase is ≈ 0. Under the old horizon (8) the last
        update ran at progress 1/8 and the post-phase LR was ≈ 0.85 × peak."""
        peak = 1e-3
        seen, _ = self._lr_trace(steps=8, gradient_accumulation_steps=4,
                                 lr_scheduler="cosine", warmup_steps=0,
                                 learning_rate=peak)
        assert len(seen) == 2, seen
        assert seen[0] == pytest.approx(peak)                       # progress 0
        assert seen[1] == pytest.approx(0.5 * peak, rel=1e-6)       # progress 1/2
        # The scheduler has now stepped twice: the LR that a *next* update would
        # see is the end of the cosine, i.e. 0.
        assert self._final_lr(steps=8, gradient_accumulation_steps=4,
                              lr_scheduler="cosine", warmup_steps=0,
                              learning_rate=peak) == pytest.approx(0.0, abs=1e-12)

    def _final_lr(self, **cfg_kwargs) -> float:
        """LR left in the optimizer after the phase (= what the next update would use)."""
        from torch.optim import AdamW as _AdamW
        holder: Dict[str, Any] = {}
        orig_init = _AdamW.__init__

        def spy_init(self, *a, **k):
            orig_init(self, *a, **k)
            holder["opt"] = self

        _AdamW.__init__ = spy_init
        try:
            adapter = _TinyAdapter(device="cpu")
            cfg = _phase_cfg(**cfg_kwargs)
            loader = _make_loader(n=32, batch=4)
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                run_phase(
                    phase_name="warmup", adapter=adapter, phase_cfg=cfg,
                    train_loader=loader, output_dir=Path(td),
                    bf16=False, fp16=False, logging_steps=1000,
                )
        finally:
            _AdamW.__init__ = orig_init
        return holder["opt"].param_groups[0]["lr"]

    def test_warmup_counts_micro_steps(self):
        """(b) warmup_steps=4 at accum=4 = one warm-up *update*: the first update
        runs at ramp position 0/1 (LR 0) and the second at full peak. Under the
        old semantics warmup covered 4 updates and the second ran at 1/4 × peak."""
        peak = 1e-3
        seen, _ = self._lr_trace(steps=12, gradient_accumulation_steps=4,
                                 lr_scheduler="constant", warmup_steps=4,
                                 learning_rate=peak)
        assert len(seen) == 3, seen
        assert seen[0] == pytest.approx(0.0)
        assert seen[1] == pytest.approx(peak)
        assert seen[2] == pytest.approx(peak)

    def test_accumulation_one_unchanged(self):
        """(c) accum=1: micro-steps == updates, horizon and warmup are untouched."""
        peak = 1e-3
        seen, res = self._lr_trace(steps=8, gradient_accumulation_steps=1,
                                   lr_scheduler="cosine", warmup_steps=2,
                                   learning_rate=peak)
        assert len(seen) == 8
        assert seen[0] == pytest.approx(0.0)              # warmup 0/2
        assert seen[1] == pytest.approx(0.5 * peak)       # warmup 1/2
        assert seen[2] == pytest.approx(peak)             # progress 0/6
        # progress 5/6 at the 8th update
        import math
        assert seen[7] == pytest.approx(peak * 0.5 * (1 + math.cos(math.pi * 5 / 6)), rel=1e-6)
        assert res.steps_completed == 8

    def test_mixture_reference_shape(self):
        """The reference SFT arm: 7 500 micro-steps, accumulation 4, warmup 400
        → 1 875 updates with a 100-update warm-up; LR at the last update is the
        cosine tail, not 0.85 × peak."""
        from arabic_eval.training.phases import _build_lr_scheduler
        import math as _m
        m = nn.Linear(2, 2)
        opt = torch.optim.AdamW(m.parameters(), lr=1.0)
        accum, steps, warm = 4, 7500, 400
        sched = _build_lr_scheduler(opt, "cosine", _m.ceil(warm / accum), _m.ceil(steps / accum))
        lrs = []
        for _ in range(steps // accum):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step(); sched.step()
        assert lrs[100] == pytest.approx(1.0)              # end of the 100-update warm-up
        assert lrs[-1] < 1e-5                               # last update ≈ 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


class TestPhaseConfigCleanLatinRows:
    """The clean_latin_rows flag is a per-phase opt-in; the field must
    round-trip through pydantic with a False default and accept True."""

    def test_default_is_false(self):
        cfg = _phase_cfg()
        assert cfg.clean_latin_rows is False

    def test_explicit_true_roundtrips(self):
        cfg = _phase_cfg(clean_latin_rows=True)
        assert cfg.clean_latin_rows is True

    def test_explicit_false_roundtrips(self):
        cfg = _phase_cfg(clean_latin_rows=False)
        assert cfg.clean_latin_rows is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------
# Token-weighted per-category early-stop eval (added 2026-09-22)
# --------------------------------------------------------------------------

class _MaskedDataset(Dataset):
    """Sequences whose first ``n_prompt`` label positions are masked, so the
    loss sees exactly ``seq_len - n_prompt`` answer tokens per record."""
    def __init__(self, vocab: int, seq_len: int, n: int, n_prompt: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab, (n, seq_len), generator=g)
        self.n_prompt = n_prompt

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        labels = ids.clone()
        labels[: self.n_prompt] = -100
        return {"input_ids": ids, "labels": labels}


def _masked_loader(vocab=32, seq_len=8, n=4, n_prompt=6, batch=2, seed=0) -> DataLoader:
    return DataLoader(_MaskedDataset(vocab, seq_len, n, n_prompt, seed=seed),
                      batch_size=batch, shuffle=False)


def _hand_nll(adapter, loader):
    """Σ NLL over shifted, unmasked label positions, and their count."""
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            logits = adapter.model(batch["input_ids"])
            sl = logits[..., :-1, :].contiguous()
            lab = batch["labels"][..., 1:].contiguous()
            nll = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1),
                                  ignore_index=-100, reduction="sum")
            total += float(nll)
            tokens += int((lab != -100).sum())
    return total, tokens


class TestTokenWeightedEval:
    def test_matches_a_hand_computed_sum_over_two_categories(self):
        from arabic_eval.training.phases import _eval_loss_token_weighted
        adapter = _TinyAdapter(device="cpu")
        loaders = {
            "extractive": _masked_loader(n=4, n_prompt=6, seed=1),   # 4 records × 1 answer token
            "free_form": _masked_loader(n=4, n_prompt=2, seed=2),    # 4 records × 5 answer tokens
        }
        got = _eval_loss_token_weighted(adapter, loaders, bf16=False, fp16=False)

        exp_nll, exp_tok = 0.0, 0
        for cat, loader in loaders.items():
            nll, tok = _hand_nll(adapter, loader)
            assert got["per_category"][cat]["tokens"] == tok
            assert got["per_category"][cat]["loss"] == pytest.approx(nll / tok, rel=1e-5)
            assert got["per_category"][cat]["records"] == 4
            exp_nll += nll
            exp_tok += tok
        assert got["tokens"] == exp_tok
        assert got["loss"] == pytest.approx(exp_nll / exp_tok, rel=1e-5)

    def test_weights_by_tokens_not_by_batches(self):
        """The whole point: a category of one-token answers must not weigh the
        same as a category of five-token answers."""
        from arabic_eval.training.phases import _eval_loss, _eval_loss_token_weighted
        adapter = _TinyAdapter(device="cpu")
        short = _masked_loader(n=4, n_prompt=6, seed=1)
        long = _masked_loader(n=4, n_prompt=2, seed=2)
        tw = _eval_loss_token_weighted(adapter, {"a": short, "b": long}, bf16=False, fp16=False)
        s_nll, s_tok = _hand_nll(adapter, short)
        l_nll, l_tok = _hand_nll(adapter, long)
        assert l_tok >= s_tok * 3                      # the long category dominates the tokens
        batch_mean = (_eval_loss(adapter, short, bf16=False, fp16=False)
                      + _eval_loss(adapter, long, bf16=False, fp16=False)) / 2
        assert tw["loss"] == pytest.approx((s_nll + l_nll) / (s_tok + l_tok), rel=1e-5)
        assert abs(tw["loss"] - batch_mean) > 1e-6     # and it differs from the per-batch mean

    def test_run_phase_uses_the_mixture_metric_and_logs_the_breakdown(self, caplog, tmp_path):
        cfg = _phase_cfg(steps=4, early_stopping=EarlyStoppingConfig(
            enabled=True, eval_every_n_steps=2, patience=10, min_steps_before_stop=0,
            restore_best_at_end=False, eval_splits={},
        ))
        adapter = _TinyAdapter(device="cpu")
        loaders = {"extractive": _masked_loader(n=4, n_prompt=6, seed=1),
                   "free_form": _masked_loader(n=4, n_prompt=2, seed=2)}
        with caplog.at_level("INFO"):
            result = run_phase(phase_name="sft", adapter=adapter, phase_cfg=cfg,
                               train_loader=_make_loader(n=16), eval_loaders=loaders,
                               output_dir=tmp_path, bf16=False, fp16=False, logging_steps=100)
        assert result.eval_loss_definition == "token_weighted_mixture"
        assert result.eval_history and set(result.eval_history[0]) == {"step", "loss", "tokens", "per_category"}
        assert set(result.eval_history[0]["per_category"]) == {"extractive", "free_form"}
        assert result.eval_losses[0][1] == pytest.approx(result.eval_history[0]["loss"], rel=1e-6)

        line = next(m for m in caplog.messages if "eval_loss=" in m)
        assert "extractive=" in line and "free_form=" in line and "tokens=" in line
        # the console's progress parser still matches
        from arabic_eval.tools.experiment_console import _RE_EVAL_LOSS
        m = _RE_EVAL_LOSS.search(line)
        assert m and m.group(1) == "sft"

    def test_the_old_path_is_unchanged(self, caplog, tmp_path):
        cfg = _phase_cfg(steps=4, early_stopping=EarlyStoppingConfig(
            enabled=True, eval_every_n_steps=2, patience=10, min_steps_before_stop=0,
            restore_best_at_end=False,
        ))
        adapter = _TinyAdapter(device="cpu")
        with caplog.at_level("INFO"):
            result = run_phase(phase_name="sft", adapter=adapter, phase_cfg=cfg,
                               train_loader=_make_loader(n=16), eval_loader=_make_loader(n=8),
                               output_dir=tmp_path, bf16=False, fp16=False, logging_steps=100)
        assert result.eval_loss_definition == "batch_mean"
        assert all("per_category" not in e for e in result.eval_history)
        line = next(m for m in caplog.messages if "eval_loss=" in m)
        assert "extractive=" not in line and line.rstrip().endswith(")")

    def test_early_stop_fires_on_the_mixture_metric(self, tmp_path):
        cfg = _phase_cfg(steps=20, learning_rate=0.0, early_stopping=EarlyStoppingConfig(
            enabled=True, eval_every_n_steps=2, patience=2, min_steps_before_stop=0,
            restore_best_at_end=False, eval_splits={},
        ))
        adapter = _TinyAdapter(device="cpu")
        loaders = {"extractive": _masked_loader(n=4, n_prompt=6, seed=1)}
        result = run_phase(phase_name="sft", adapter=adapter, phase_cfg=cfg,
                           train_loader=_make_loader(n=80), eval_loaders=loaders,
                           output_dir=tmp_path, bf16=False, fp16=False, logging_steps=100)
        assert result.early_stopped                      # LR 0 → a flat metric → patience runs out
        assert result.steps_completed < 20

    def test_missing_both_eval_inputs_still_raises(self, tmp_path):
        cfg = _phase_cfg(steps=2, early_stopping=EarlyStoppingConfig(enabled=True))
        with pytest.raises(ValueError, match="no eval_loader"):
            run_phase(phase_name="sft", adapter=_TinyAdapter(device="cpu"), phase_cfg=cfg,
                      train_loader=_make_loader(), output_dir=tmp_path, bf16=False, fp16=False)
