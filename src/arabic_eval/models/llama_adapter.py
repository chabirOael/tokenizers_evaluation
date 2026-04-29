"""LLaMA model adapter with support for multiple embedding types."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.models.embeddings.standard import resize_token_embeddings
from arabic_eval.models.embeddings.character_cnn import CharacterCNNEmbedding
from arabic_eval.models.embeddings.char_jaber_embed import CharJaberEmbedding, CharJaberOutputHead
from arabic_eval.registry import model_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType

logger = logging.getLogger("arabic_eval.models.llama")


@model_registry.register("llama")
class LlamaAdapter(BaseModelAdapter):
    """Adapter for LLaMA models (or any AutoModelForCausalLM-compatible model).

    Handles three embedding integration modes:
      - STANDARD: resize nn.Embedding for subword tokenizers
      - CHARACTER_CNN: replace embedding with CharCNN for CharacterBERT
      - CHAR_JABER: replace embedding with character embedding for char-JABER
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-1B",
        device: str = "auto",
        dtype: str = "bfloat16",
        **kwargs: Any,
    ) -> None:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info("Loading model: %s (dtype=%s, device=%s)", model_name_or_path, dtype, device)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._model = self._model.to("cpu")

        self._device_str = device
        self._embedding_type: Optional[str] = None
        self._char_output_head: Optional[nn.Module] = None
        logger.info("Model loaded — parameters: %d", sum(p.numel() for p in self._model.parameters()))

    def adapt_to_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        """Adapt model's embedding/output layers for the given tokenizer."""
        emb_type = tokenizer.embedding_type
        self._embedding_type = emb_type

        if emb_type == EmbeddingType.STANDARD:
            self._adapt_standard(tokenizer)
        elif emb_type == EmbeddingType.CHARACTER_CNN:
            self._adapt_character_cnn(tokenizer)
        elif emb_type == EmbeddingType.CHAR_JABER:
            self._adapt_char_jaber(tokenizer)
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}")

        # Update model config
        self._model.config.pad_token_id = tokenizer.special_tokens.get("pad_token", 0)
        self._model.config.bos_token_id = tokenizer.special_tokens.get("bos_token", 1)
        self._model.config.eos_token_id = tokenizer.special_tokens.get("eos_token", 2)

        logger.info("Model adapted for embedding type: %s", emb_type)

    def _adapt_standard(self, tokenizer: BaseTokenizer) -> None:
        """Standard subword tokenizer: just resize embeddings."""
        resize_token_embeddings(self._model, tokenizer.vocab_size)
        self._model.config.vocab_size = tokenizer.vocab_size

    def _adapt_character_cnn(self, tokenizer: BaseTokenizer) -> None:
        """CharacterBERT: replace embedding with CharCNN."""
        emb_config = tokenizer.get_embedding_config()
        hidden_size = self._model.config.hidden_size

        # Capture device/dtype from existing layers before replacing embed_tokens
        ref_param = next(self._model.model.layers.parameters())

        char_cnn = CharacterCNNEmbedding(
            char_vocab_size=emb_config["char_vocab_size"],
            char_embed_dim=emb_config.get("char_embed_dim", 16),
            cnn_filters=emb_config.get("cnn_filters"),
            num_highway_layers=emb_config.get("num_highway_layers", 2),
            output_dim=hidden_size,
            max_char_len=emb_config.get("max_char_len", 50),
        )
        char_cnn = char_cnn.to(device=ref_param.device, dtype=ref_param.dtype)

        # Replace the embedding layer
        self._model.model.embed_tokens = char_cnn

        # Resize output head for word-level vocabulary
        output_vocab_size = emb_config.get("output_vocab_size", tokenizer.vocab_size)
        lm_head = nn.Linear(hidden_size, output_vocab_size, bias=False)
        nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)
        self._model.lm_head = lm_head.to(device=ref_param.device, dtype=ref_param.dtype)
        self._model.config.vocab_size = output_vocab_size

        logger.info(
            "Replaced embedding with CharCNN (char_vocab=%d, output_vocab=%d)",
            emb_config["char_vocab_size"], output_vocab_size,
        )

    def _adapt_char_jaber(self, tokenizer: BaseTokenizer) -> None:
        """char-JABER: replace embedding with character embedding."""
        emb_config = tokenizer.get_embedding_config()
        hidden_size = self._model.config.hidden_size
        char_vocab_size = emb_config["char_vocab_size"]
        downsample_factor = emb_config.get("downsample_factor", 1)

        # Capture device/dtype from existing layers before replacing embed_tokens
        ref_param = next(self._model.model.layers.parameters())

        char_embed = CharJaberEmbedding(
            char_vocab_size=char_vocab_size,
            output_dim=hidden_size,
            downsample_factor=downsample_factor,
        )
        char_embed = char_embed.to(device=ref_param.device, dtype=ref_param.dtype)

        # Output head must upsample back to full sequence length when downsampling
        # is active, so use CharJaberOutputHead rather than a plain Linear.
        output_head = CharJaberOutputHead(
            hidden_dim=hidden_size,
            char_vocab_size=char_vocab_size,
            upsample_factor=downsample_factor,
        )
        nn.init.normal_(output_head.output_projection.weight, mean=0.0, std=0.02)
        output_head = output_head.to(device=ref_param.device, dtype=ref_param.dtype)

        self._model.model.embed_tokens = char_embed
        self._model.lm_head = output_head
        self._model.config.vocab_size = char_vocab_size

        logger.info(
            "Replaced embedding with char-JABER (char_vocab=%d, downsample_factor=%d)",
            char_vocab_size, downsample_factor,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass handling both standard and character-level inputs."""
        if self._embedding_type == EmbeddingType.CHARACTER_CNN:
            return self._forward_character_cnn(batch)
        else:
            # Standard forward (works for both STANDARD and CHAR_JABER)
            outputs = self._model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

    def _forward_character_cnn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for CharacterBERT: compute CharCNN embeddings, then use
        the standard model forward via inputs_embeds so that RoPE, causal masking,
        and lm_head are all handled by the existing HF implementation.
        """
        char_ids = batch["char_ids"]  # [B, S, max_char_len]

        # Compute word-level embeddings with our CharCNN
        inputs_embeds = self._model.model.embed_tokens(char_ids)  # [B, S, D]

        # Pass through the full LlamaForCausalLM forward (skips embed_tokens lookup
        # because inputs_embeds is provided; uses our replaced lm_head for output)
        outputs = self._model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate tokens. Only works for STANDARD and CHAR_JABER modes."""
        if self._embedding_type == EmbeddingType.CHARACTER_CNN:
            raise NotImplementedError(
                "Auto-regressive generation is not supported for CharacterBERT. "
                "Use evaluation metrics that don't require generation."
            )
        return self._model.generate(input_ids=input_ids, **kwargs)

    def get_trainable_parameters(self) -> list:
        """Return all parameters (entire model is trainable after adaptation)."""
        return list(self._model.parameters())

    def save_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        self._model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        logger.info("Checkpoint loaded from %s", path)

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    @property
    def model(self):
        return self._model
