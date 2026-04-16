"""HuggingFace dataset loading and caching."""
from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from arabic_eval.data.preprocessing import preprocess_dataset

logger = logging.getLogger("arabic_eval.data")


def load_arabic_dataset(
    dataset_name: str = "Jr23xd23/ArabicText-Large",
    dataset_config: Optional[str] = None,
    cache_dir: str = "outputs/data_cache",
    train_split: str = "train",
    eval_split: Optional[str] = "test",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    preprocessing_config: Optional[dict] = None,
    seed: int = 42,
) -> DatasetDict:
    """Load the Arabic dataset, preprocess, and return train/eval splits.

    If the dataset only has a single split, it is automatically divided
    into train (90%) and eval (10%).
    """
    logger.info("Loading dataset %s (config=%s)", dataset_name, dataset_config)
    raw = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)

    # Handle single-split datasets
    if isinstance(raw, Dataset):
        raw = raw.train_test_split(test_size=0.1, seed=seed)
        train_split, eval_split = "train", "test"
    elif isinstance(raw, DatasetDict):
        if eval_split and eval_split not in raw:
            if "validation" in raw:
                eval_split = "validation"
            else:
                # Create eval split from train
                split_ds = raw[train_split].train_test_split(test_size=0.1, seed=seed)
                raw[train_split] = split_ds["train"]
                raw["test"] = split_ds["test"]
                eval_split = "test"

    # Detect the text column
    text_column = _detect_text_column(raw[train_split])
    logger.info("Using text column: '%s'", text_column)

    # Preprocess
    if preprocessing_config:
        raw = preprocess_dataset(raw, text_column=text_column, **preprocessing_config)

    # Subsample
    result = DatasetDict()
    train_ds = raw[train_split]
    if max_train_samples and len(train_ds) > max_train_samples:
        train_ds = train_ds.shuffle(seed=seed).select(range(max_train_samples))
    result["train"] = train_ds

    if eval_split and eval_split in raw:
        eval_ds = raw[eval_split]
        if max_eval_samples and len(eval_ds) > max_eval_samples:
            eval_ds = eval_ds.shuffle(seed=seed).select(range(max_eval_samples))
        result["eval"] = eval_ds

    logger.info("Dataset loaded — train: %d, eval: %d", len(result["train"]),
                len(result.get("eval", [])))
    return result


def _detect_text_column(dataset: Dataset) -> str:
    """Find the text column in a dataset."""
    for candidate in ("text", "content", "sentence", "document", "passage"):
        if candidate in dataset.column_names:
            return candidate
    # Fallback: first string column
    for col in dataset.column_names:
        if dataset.features[col].dtype == "string":
            return col
    raise ValueError(
        f"Cannot detect text column. Available columns: {dataset.column_names}"
    )


def extract_texts(dataset: Dataset, text_column: Optional[str] = None) -> list[str]:
    """Extract raw text strings from a dataset split."""
    col = text_column or _detect_text_column(dataset)
    return dataset[col]
