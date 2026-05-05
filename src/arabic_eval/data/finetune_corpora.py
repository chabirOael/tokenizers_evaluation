"""QA corpora used by the 3-phase training pipeline.

Three datasets, all share a uniform ``QARecord`` schema. Resolution table:

    "arabic_squad"  -> Mostafa3zazi/Arabic_SQuAD          # MT-translated SQuAD-v1
    "tydiqa_arabic" -> google-research-datasets/tydiqa     # secondary_task, Arabic-only
    "arcd"          -> hsseinmz/arcd                       # plain_text

Phase 1 + Phase 2 use ``arabic_squad`` (translated, Phase 2 spec calls for
"a large translated Arabic dataset"). Phase 3 uses
``tydiqa_arabic + arcd`` (native Arabic QA).

Prompt format — implemented fresh in this module rather than imported from
``tasks/question_answering.py`` (which is being removed). Surface form:

    ### السياق: {context}
    ### السؤال: {question}
    ### الإجابة: {answer}

For ``loss_target='answer_only'`` the prompt span ends just before
``{answer}`` and is masked to -100 via ``answer_only_masking``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from torch.utils.data import DataLoader, Dataset

from .answer_only_masking import compute_answer_only_labels
from .collation import get_collator
from ..tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Uniform record schema
# --------------------------------------------------------------------------

@dataclass
class QARecord:
    """A single question-answering example after normalization across corpora."""
    id: str
    question: str
    context: str
    answer: str
    source: str  # corpus name: arabic_squad | tydiqa_arabic | arcd


# --------------------------------------------------------------------------
# Prompt format (single source of truth — do NOT import from tasks/)
# --------------------------------------------------------------------------

def _format_qa_prompt(record: QARecord) -> str:
    """Prompt text up to (but excluding) the answer. Used for LCP masking."""
    return (
        f"### السياق: {record.context}\n"
        f"### السؤال: {record.question}\n"
        f"### الإجابة:"
    )


def _format_qa_full(record: QARecord) -> str:
    """Full text including the answer. Used for tokenizing training examples."""
    return f"{_format_qa_prompt(record)} {record.answer}"


# --------------------------------------------------------------------------
# Per-corpus loaders
# --------------------------------------------------------------------------

def _load_arabic_squad(split: str) -> List[QARecord]:
    """Load Mostafa3zazi/Arabic_SQuAD (flat schema, train-only)."""
    if split != "train":
        raise ValueError(
            f"arabic_squad has no '{split}' split (only 'train' is available)"
        )
    from datasets import load_dataset
    ds = load_dataset("Mostafa3zazi/Arabic_SQuAD", split="train")
    records: List[QARecord] = []
    for ex in ds:
        question = ex["question"]
        context = ex["context"]
        answer = ex["text"]
        if not (question and context and answer):
            continue
        records.append(QARecord(
            id=str(ex["index"]),
            question=question,
            context=context,
            answer=answer,
            source="arabic_squad",
        ))
    logger.info("arabic_squad/%s: loaded %d records", split, len(records))
    return records


def _load_tydiqa_arabic(split: str) -> List[QARecord]:
    """Load TyDiQA-Arabic from secondary_task (filter id starts with 'arabic-')."""
    if split not in {"train", "validation"}:
        raise ValueError(f"tydiqa_arabic split must be 'train' or 'validation', got {split!r}")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split=split)
    records: List[QARecord] = []
    for ex in ds:
        ex_id = ex["id"]
        if not ex_id.startswith("arabic-"):
            continue
        answers = ex["answers"]
        texts = answers.get("text") or []
        if not texts:
            continue
        # Take the first reference answer (TyDiQA train rows ship one).
        answer = texts[0]
        if not (ex["question"] and ex["context"] and answer):
            continue
        records.append(QARecord(
            id=ex_id,
            question=ex["question"],
            context=ex["context"],
            answer=answer,
            source="tydiqa_arabic",
        ))
    logger.info("tydiqa_arabic/%s: loaded %d records", split, len(records))
    return records


def _load_arcd(split: str) -> List[QARecord]:
    """Load hsseinmz/arcd (plain_text config)."""
    if split not in {"train", "validation"}:
        raise ValueError(f"arcd split must be 'train' or 'validation', got {split!r}")
    from datasets import load_dataset
    ds = load_dataset("hsseinmz/arcd", "plain_text", split=split)
    records: List[QARecord] = []
    for ex in ds:
        answers = ex["answers"]
        texts = answers.get("text") or []
        if not texts:
            continue
        answer = texts[0]
        if not (ex["question"] and ex["context"] and answer):
            continue
        records.append(QARecord(
            id=str(ex["id"]),
            question=ex["question"],
            context=ex["context"],
            answer=answer,
            source="arcd",
        ))
    logger.info("arcd/%s: loaded %d records", split, len(records))
    return records


_LOADERS = {
    "arabic_squad": _load_arabic_squad,
    "tydiqa_arabic": _load_tydiqa_arabic,
    "arcd": _load_arcd,
}


def load_corpus(name: str, split: str) -> List[QARecord]:
    """Resolve a registry name + split to a list of normalized QARecord."""
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise KeyError(
            f"unknown corpus name {name!r}; known names: {sorted(_LOADERS)}"
        ) from None
    return loader(split)


def load_corpora(
    names: Sequence[str],
    splits: Mapping[str, str] | str,
) -> List[QARecord]:
    """Load and concatenate multiple corpora.

    ``splits`` may be a single string (applied to all names) or a mapping
    ``{corpus_name: split_name}`` (per-corpus override; useful when Phase 3
    SFT trains on TyDiQA train + ARCD train but evaluates on TyDiQA val +
    ARCD val).
    """
    if isinstance(splits, str):
        splits = {n: splits for n in names}
    out: List[QARecord] = []
    for n in names:
        s = splits.get(n)
        if s is None:
            raise KeyError(f"no split provided for corpus {n!r}")
        out.extend(load_corpus(n, s))
    return out


# --------------------------------------------------------------------------
# Tokenization + dataloader
# --------------------------------------------------------------------------

class _QATokenizedDataset(Dataset):
    """In-memory list of pre-tokenized QA examples."""
    def __init__(self, encodings: List[Dict[str, Any]]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.encodings[idx]


def tokenize_records(
    records: Sequence[QARecord],
    tokenizer: BaseTokenizer,
    max_length: int,
    loss_target: str,
) -> List[Dict[str, Any]]:
    """Tokenize records into per-example encodings ready for the collator.

    ``loss_target='full_sequence'``: store ``input_ids`` only — the
    StandardCollator's default fallback computes labels = input_ids
    (with pad masked to -100).

    ``loss_target='answer_only'``: store ``input_ids`` + ``labels`` with
    the prompt span masked to -100 via the LCP technique.
    """
    if loss_target not in {"full_sequence", "answer_only"}:
        raise ValueError(f"unknown loss_target {loss_target!r}")

    encodings: List[Dict[str, Any]] = []
    n_dropped = 0
    n_completion_tokens = 0

    for rec in records:
        full_text = _format_qa_full(rec)
        full_enc = tokenizer.encode(full_text, max_length=max_length, truncation=True)
        entry: Dict[str, Any] = {"input_ids": full_enc.input_ids}
        if full_enc.char_ids is not None:
            entry["char_ids"] = full_enc.char_ids

        if loss_target == "answer_only":
            prompt_text = _format_qa_prompt(rec)
            prompt_enc = tokenizer.encode(
                prompt_text, max_length=max_length, truncation=True
            )
            labels = compute_answer_only_labels(prompt_enc.input_ids, full_enc.input_ids)
            if labels is None:
                n_dropped += 1
                continue
            entry["labels"] = labels
            n_completion_tokens += sum(1 for l in labels if l != -100)

        encodings.append(entry)

    if loss_target == "answer_only":
        n_kept = len(encodings)
        avg_comp = n_completion_tokens / n_kept if n_kept else 0.0
        logger.info(
            "tokenize_records (answer_only): kept=%d, dropped=%d (truncation), "
            "avg %.2f answer tokens/example",
            n_kept, n_dropped, avg_comp,
        )
    else:
        logger.info("tokenize_records (full_sequence): %d examples", len(encodings))

    return encodings


def build_qa_dataloader(
    records: Sequence[QARecord],
    tokenizer: BaseTokenizer,
    batch_size: int,
    max_length: int,
    loss_target: str,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader over QA records with the right collator.

    Uses ``tokenizer.embedding_type`` to dispatch to the right collator
    so character_cnn / char_jaber / charformer tokenizers slot in
    without changes (they propagate ``char_ids`` from
    ``TokenizerOutput`` automatically).
    """
    encodings = tokenize_records(records, tokenizer, max_length, loss_target)
    collator = get_collator(
        tokenizer.embedding_type,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        max_length=max_length,
    )
    return DataLoader(
        _QATokenizedDataset(encodings),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
