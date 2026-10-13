"""Gemma-3 model adapter — the text model of a (multimodal) Gemma-3 checkpoint under :class:`LlamaAdapter`.

``google/gemma-3-4b-pt`` is a vision-language checkpoint (``Gemma3ForConditionalGeneration``: a SigLIP vision
tower, a projector and a Gemma-3 text model). The campaign uses the text model only. Two loader traps, both
measured on a tiny random checkpoint with transformers 5.7 (tests/test_gemma3_support.py):

* ``AutoModelForCausalLM`` maps the multimodal config to ``Gemma3ForConditionalGeneration``, whose text model sits
  at ``model.language_model`` and whose config has no ``hidden_size`` — not the attribute surface
  ``LlamaAdapter`` relies on.
* ``Gemma3ForCausalLM.from_pretrained`` on that checkpoint *silently* initialises every text weight at random:
  the checkpoint stores them as ``language_model.model.*`` (the release layout; newer saves use
  ``model.language_model.*``), neither of which ``Gemma3ForCausalLM`` expects — a warning table, no error.

So ``_load_model`` builds ``Gemma3ForCausalLM`` from the *text* config with a ``key_mapping`` that sends both
layouts to ``model.*``, and refuses to continue unless every text weight landed: missing keys may only be the tied
``lm_head.weight``, unexpected keys only the vision tower and the projector. On the tiny checkpoint the result's
logits equal the multimodal model's on text input exactly. A text-only checkpoint (``model_type: gemma3_text`` —
what this pipeline's phase checkpoints save) loads as a plain ``Gemma3ForCausalLM``, with the same check.

Gemma-3 facts that matter for the pipeline (verify on the real checkpoint, never assume):

* **Scaled embeddings.** ``Gemma3TextScaledWordEmbedding`` multiplies rows by ``√hidden_size`` in the forward.
  ``resize_token_embeddings`` keeps the class and the scale (tested); ``surface_avg`` averages the raw rows,
  which the forward scales like every other row — consistent. The replaced-embedding branches (CharCNN,
  char-JABER, Charformer) feed ``inputs_embeds`` and skip the scale; they train their modules from scratch.
* **Tied** input / output embeddings (``tie_word_embeddings``), like Qwen3-4B and Llama-3.2.
* **Padded vocabulary**: the 4B text config has 262 208 rows, the tokenizer 262 144 entries — ``native_gemma``
  reports the config's count so the native resize stays a no-op.
* Alternating local (sliding window) and global attention layers, inside the standard forward;
  ``final_logit_softcapping`` is ``None`` on Gemma-3 (it was set on Gemma-2).
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from arabic_eval.models.llama_adapter import LlamaAdapter
from arabic_eval.registry import model_registry

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-3-4b-pt"
# both checkpoint layouts of the text model → Gemma3ForCausalLM's own names
KEY_MAPPING = {
    r"^language_model\.model": "model",
    r"^model\.language_model": "model",
    r"^language_model\.lm_head": "lm_head",
}
NON_TEXT_PREFIXES = ("vision_tower", "multi_modal_projector", "model.vision_tower", "model.multi_modal_projector")


class Gemma3LoadError(RuntimeError):
    pass


def load_gemma3_text_model(model_name_or_path: str, torch_dtype: torch.dtype = torch.bfloat16, device_map: Any = None):
    """``Gemma3ForCausalLM`` with every text weight of the checkpoint, or an exception — never a partly random model."""
    from transformers import AutoConfig, Gemma3ForCausalLM

    cfg = AutoConfig.from_pretrained(model_name_or_path)
    text_cfg = cfg.get_text_config()
    multimodal = text_cfg is not cfg
    # The mapping is applied to every load: a no-op on plain ``model.*`` keys, and it also reads a text-only
    # checkpoint that transformers wrote in the original layout (save_pretrained reverses a load-time mapping
    # unless save_original_format=False — Gemma3Adapter.save_checkpoint passes False).
    kw: Any = dict(config=text_cfg, torch_dtype=torch_dtype, output_loading_info=True, key_mapping=KEY_MAPPING)
    if device_map is not None:
        kw["device_map"] = device_map
    model, info = Gemma3ForCausalLM.from_pretrained(model_name_or_path, **kw)
    tied = bool(getattr(text_cfg, "tie_word_embeddings", True))
    missing = sorted(k for k in info.get("missing_keys", []) if not (tied and k == "lm_head.weight"))
    unexpected = sorted(info.get("unexpected_keys", []))
    stray = [k for k in unexpected if not k.startswith(NON_TEXT_PREFIXES)]
    if missing or stray or info.get("mismatched_keys"):
        raise Gemma3LoadError(
            f"{model_name_or_path}: the text model did not load whole — missing {missing[:5]} ({len(missing)}), "
            f"unexpected non-vision {stray[:5]} ({len(stray)}), mismatched {list(info.get('mismatched_keys') or [])[:3]}")
    if tied and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.tie_weights()
    logger.info("Gemma-3 text model from %s (%s checkpoint): %d vision / projector tensors left out, lm_head %s",
                model_name_or_path, "multimodal" if multimodal else "text-only",
                len(unexpected), "tied" if tied else "untied")
    return model


@model_registry.register("gemma3")
class Gemma3Adapter(LlamaAdapter):
    """Adapter for Gemma-3 causal LMs (the text model; see the module docstring)."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "bfloat16",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, device=device, dtype=dtype, **kwargs)

    def _load_model(self, model_name_or_path: str, torch_dtype: torch.dtype, device: str):
        return load_gemma3_text_model(model_name_or_path, torch_dtype=torch_dtype,
                                      device_map=device if device != "cpu" else None)

    def save_checkpoint(self, path) -> None:
        """Plain ``model.*`` keys (``save_original_format=False``): by default transformers would write the
        multimodal layout back (``language_model.model.*``) under a text-only config, which a plain
        ``from_pretrained`` and ``scripts/dump_eval_rows.load_state_into`` cannot read."""
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path, save_original_format=False)
        logger.info("Checkpoint saved to %s (plain model.* keys)", path)

    def load_checkpoint(self, path) -> None:
        self._model = load_gemma3_text_model(str(path), torch_dtype=self._model.dtype)
        logger.info("Checkpoint loaded from %s", path)

    def adapt_to_tokenizer(self, tokenizer) -> None:
        """As ``LlamaAdapter``, plus one Gemma-3 detail: ``Gemma3ForCausalLM`` computes its loss with
        ``self.vocab_size`` — an attribute fixed at construction — not ``config.vocab_size``, so after a resize or
        a replaced output head (CharCNN, char-JABER, Charformer) the loss would reshape the logits with the old
        size. The attribute follows the output head here."""
        super().adapt_to_tokenizer(tokenizer)
        if hasattr(self._model, "vocab_size"):
            self._model.vocab_size = int(tokenizer.vocab_size)
