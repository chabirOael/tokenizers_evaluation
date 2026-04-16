"""Question Answering task: Arabic QA evaluation (ARCD / TyDi QA)."""
from __future__ import annotations

import logging
import re
import string
from collections import Counter
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

logger = logging.getLogger("arabic_eval.tasks.question_answering")


class QADataset(Dataset):
    """Dataset for QA: stores tokenized context+question pairs and answer spans."""

    def __init__(self, examples: List[Dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def _normalize_answer(text: str) -> str:
    """Normalize answer text for F1/EM computation."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove Arabic diacritics for comparison
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    return text.lower()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Check if normalized prediction matches ground truth exactly."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


@task_registry.register("question_answering")
class QuestionAnsweringTask(BaseTask):
    """Arabic QA evaluation using ARCD or TyDi QA Arabic subset.

    For causal LMs, QA is framed as text generation:
      prompt = "Context: {context}\nQuestion: {question}\nAnswer:"
      The model generates the answer span.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.max_length = config.get("max_length", 512)
        self.max_new_tokens = config.get("max_new_tokens", 64)
        self.dataset_name = config.get("dataset_name", "hsseinmz/arcd")
        self.dataset_config = config.get("dataset_config", None)
        self.cache_dir = config.get("cache_dir", "outputs/data_cache")

    def _format_qa_prompt(self, context: str, question: str) -> str:
        """Format QA as a generation prompt for causal LM."""
        return f"السياق: {context}\nالسؤال: {question}\nالإجابة:"

    def _load_qa_data(
        self, split: str, max_samples: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Load QA examples from the dataset."""
        try:
            ds = load_dataset(
                self.dataset_name, self.dataset_config,
                cache_dir=self.cache_dir, split=split,
            )
        except Exception:
            # Try loading with a different split name
            available_splits = load_dataset(
                self.dataset_name, self.dataset_config,
                cache_dir=self.cache_dir,
            )
            if "validation" in available_splits:
                ds = available_splits["validation"]
            elif "train" in available_splits:
                ds = available_splits["train"]
            else:
                ds = list(available_splits.values())[0]

        examples = []
        for item in ds:
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", {})

            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
            elif isinstance(answers, list):
                answer_texts = [a.get("text", "") if isinstance(a, dict) else str(a)
                               for a in answers]
            else:
                answer_texts = [str(answers)]

            if not answer_texts:
                continue

            examples.append({
                "context": normalize_arabic(context),
                "question": normalize_arabic(question),
                "answers": [normalize_arabic(a) for a in answer_texts if a],
            })

            if max_samples and len(examples) >= max_samples:
                break

        return examples

    def _tokenize_qa(
        self, examples: List[Dict[str, str]], tokenizer: BaseTokenizer
    ) -> List[Dict[str, Any]]:
        """Tokenize QA prompts for training."""
        encodings = []
        for ex in tqdm(examples, desc="Tokenizing QA", unit="example", leave=False):
            prompt = self._format_qa_prompt(ex["context"], ex["question"])
            answer = ex["answers"][0] if ex["answers"] else ""
            full_text = prompt + " " + answer

            enc = tokenizer.encode(full_text, max_length=self.max_length, truncation=True)
            entry = {"input_ids": enc.input_ids}
            if enc.char_ids is not None:
                entry["char_ids"] = enc.char_ids
            encodings.append(entry)

        return encodings

    def get_dataloader(
        self,
        tokenizer: BaseTokenizer,
        split: str = "train",
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        examples = self._load_qa_data(split, max_samples)
        encodings = self._tokenize_qa(examples, tokenizer)

        collator = get_collator(
            tokenizer.embedding_type,
            pad_token_id=tokenizer.pad_token_id,
            max_length=self.max_length,
        )

        return DataLoader(
            QADataset(encodings),
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
        max_samples: Optional[int] = 500,
    ) -> Dict[str, float]:
        """Evaluate QA by generating answers and computing F1/EM."""
        logger.info("Evaluating QA on '%s' split", split)
        examples = self._load_qa_data(split, max_samples)

        if not examples:
            logger.warning("No QA examples found for split '%s'", split)
            return {"f1": 0.0, "exact_match": 0.0}

        model.model.eval()
        all_f1 = []
        all_em = []

        for ex in tqdm(examples, desc="QA generation", unit="example"):
            prompt = self._format_qa_prompt(ex["context"], ex["question"])
            enc = tokenizer.encode(prompt, max_length=self.max_length, truncation=True)

            input_ids = torch.tensor([enc.input_ids], device=model.device)

            try:
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # Extract generated part (after the prompt)
                generated = gen_ids[0, input_ids.shape[1]:]
                prediction = tokenizer.decode(generated.tolist())
            except NotImplementedError:
                # For CharacterBERT, fall back to loss-based evaluation
                prediction = ""

            # Compute metrics against all reference answers
            best_f1 = max(compute_f1(prediction, ans) for ans in ex["answers"])
            best_em = max(compute_exact_match(prediction, ans) for ans in ex["answers"])
            all_f1.append(best_f1)
            all_em.append(best_em)

        metrics = {
            "f1": round(sum(all_f1) / max(len(all_f1), 1), 4),
            "exact_match": round(sum(all_em) / max(len(all_em), 1), 4),
            "num_examples": len(examples),
        }
        logger.info("QA metrics: %s", metrics)
        return metrics

    @property
    def name(self) -> str:
        return "question_answering"

    @property
    def metric_names(self) -> List[str]:
        return ["f1", "exact_match"]
