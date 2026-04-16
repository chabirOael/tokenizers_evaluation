"""Text generation task: perplexity evaluation with sliding window."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

from arabic_eval.data.collation import get_collator
from arabic_eval.data.preprocessing import normalize_arabic
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.registry import task_registry
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tokenizers.base import BaseTokenizer

logger = logging.getLogger("arabic_eval.tasks.text_generation")


class TextDataset(Dataset):
    """Simple dataset wrapping tokenized texts."""

    def __init__(self, encodings: List[Dict[str, Any]]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.encodings[idx]


@task_registry.register("text_generation")
class TextGenerationTask(BaseTask):
    """Evaluate language model perplexity on Arabic text.

    Uses the same ArabicText-Large dataset (eval split) or a specified dataset.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.max_length = config.get("max_length", 512)
        self.stride = config.get("stride", 256)
        self.dataset_name = config.get("dataset_name", "Jr23xd23/ArabicText-Large")
        self.dataset_config = config.get("dataset_config", None)
        self.cache_dir = config.get("cache_dir", "outputs/data_cache")

    def _load_texts(self, split: str, max_samples: Optional[int] = None) -> List[str]:
        """Load raw texts for the given split, falling back to 'train' if unavailable."""
        try:
            ds = load_dataset(
                self.dataset_name, self.dataset_config,
                cache_dir=self.cache_dir, split=split,
            )
        except ValueError:
            logger.warning("Split '%s' not found in dataset, falling back to 'train'", split)
            ds = load_dataset(
                self.dataset_name, self.dataset_config,
                cache_dir=self.cache_dir, split="train",
            )

        # Find text column
        text_col = "text"
        for candidate in ("text", "content", "sentence"):
            if candidate in ds.column_names:
                text_col = candidate
                break

        texts = ds[text_col]
        texts = [normalize_arabic(t) for t in texts if t and len(t.strip()) >= 10]

        if max_samples and len(texts) > max_samples:
            texts = texts[:max_samples]

        return texts

    def _tokenize_texts(
        self, texts: List[str], tokenizer: BaseTokenizer
    ) -> List[Dict[str, Any]]:
        """Tokenize texts into fixed-length chunks for LM evaluation."""
        encodings = []
        for text in tqdm(texts, desc="Tokenizing texts", unit="text", leave=False):
            enc = tokenizer.encode(text)
            ids = enc.input_ids

            # Sliding window chunking
            for begin in range(0, len(ids), self.stride):
                end = min(begin + self.max_length, len(ids))
                chunk_ids = ids[begin:end]
                if len(chunk_ids) < 2:
                    continue

                entry = {"input_ids": chunk_ids}
                if enc.char_ids is not None:
                    entry["char_ids"] = enc.char_ids[begin:end]

                encodings.append(entry)

                if end == len(ids):
                    break

        return encodings

    def get_dataloader(
        self,
        tokenizer: BaseTokenizer,
        split: str = "train",
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        texts = self._load_texts(split, max_samples)
        encodings = self._tokenize_texts(texts, tokenizer)

        collator = get_collator(
            tokenizer.embedding_type,
            pad_token_id=tokenizer.pad_token_id,
            max_length=self.max_length,
        )

        return DataLoader(
            TextDataset(encodings),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        split: str = "test",
        max_samples: Optional[int] = 5000,
    ) -> Dict[str, float]:
        """Compute perplexity on the eval set."""
        logger.info("Evaluating perplexity on '%s' split", split)

        dataloader = self.get_dataloader(
            tokenizer, split=split, batch_size=4, max_samples=max_samples
        )

        model.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(dataloader, desc="Perplexity eval", unit="batch"):
            # Move to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model.forward(batch)
            if outputs["loss"] is not None:
                # Count non-padded tokens
                labels = batch.get("labels", batch.get("input_ids"))
                n_tokens = (labels != -100).sum().item()
                total_loss += outputs["loss"].item() * n_tokens
                total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        metrics = {
            "perplexity": round(perplexity, 4),
            "avg_loss": round(avg_loss, 6),
            "total_tokens": total_tokens,
        }
        logger.info("Text generation metrics: %s", metrics)
        return metrics

    @property
    def name(self) -> str:
        return "text_generation"

    @property
    def metric_names(self) -> List[str]:
        return ["perplexity", "avg_loss"]
