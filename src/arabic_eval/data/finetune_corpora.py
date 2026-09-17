"""QA corpora used by the 3-phase training pipeline.

Every corpus is normalized to the uniform ``QARecord`` schema. Resolution
table (``CORPUS_CATEGORY`` in ``config.py`` fixes each corpus' category):

    "arabic_squad"     -> Mostafa3zazi/Arabic_SQuAD          # MT-translated SQuAD-v1     (extractive)
    "tydiqa_arabic"    -> google-research-datasets/tydiqa     # secondary_task, Arabic-only (extractive)
    "arcd"             -> hsseinmz/arcd                       # plain_text                 (extractive)
    "arabic_squad_mcq" -> synthetic 4-way MCQ from Arabic-SQuAD (synthetic_mcq.py)         (mcq)
    "cidar"            -> arbml/CIDAR                         # 10 K culturally-aligned instructions (free_form)
    "bactrian_x_ar"    -> MBZUAI/Bactrian-X data/ar.json.gz   # 67 K translated Alpaca + Dolly       (free_form)
    "aya_ar"           -> CohereForAI/aya_collection_language_split standard_arabic,
                          sub-dataset allowlist (default Aya-Dataset + Dolly-v2 (T))       (free_form)

Phase 1 + Phase 2 use ``arabic_squad`` (translated, Phase 2 spec calls for
"a large translated Arabic dataset"). Phase 3 uses ``tydiqa_arabic + arcd``
(native Arabic QA) + ``arabic_squad_mcq``, optionally ratio-controlled with
the free-form corpora through ``PhaseConfig.mixture`` (``sft_mixture.py``).

Prompt formats (all end in the LightEval-official ``الإجابة:`` prefix so
train/eval share the same anchor tokens). Extractive (``qa``):

    السياق: {context}
    السؤال: {question}
    الإجابة: {answer}

Free-form (``instruction``) is the same shape with the ``السياق:`` line only
when the record has an input/context; MCQ (``mcq_letter``) is the
LightEval letter prompt. For ``loss_target='answer_only'`` the prompt span
ends just before ``{answer}`` and is masked to -100 via
``answer_only_masking``.

The three Hub-hosted free-form corpora are read at a **pinned revision**
(``PINNED_REVISIONS``) so a corpus silently edited upstream never changes
a training set; the revision is recorded in the mixture manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from torch.utils.data import DataLoader, Dataset

from .answer_only_masking import compute_answer_only_labels
from .collation import get_collator
from ..config import CORPUS_CATEGORY
from ..tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)

# Dataset revisions (git SHAs on the Hub) the free-form loaders read at.
# Bump deliberately; the mixture manifest records the value in use.
PINNED_REVISIONS: Dict[str, str] = {
    "cidar": "bc2f9d7a9de34b534b126c32b8cce98e098138df",
    "bactrian_x_ar": "3698480a001cb3f62c61c314d4acb181eb31e983",
    "aya_ar": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
}

# Where filtered Hub subsets are cached as Parquet (``aya_ar``).
SFT_CORPORA_CACHE_DIR = Path(os.environ.get("ARABIC_EVAL_SFT_CORPORA_CACHE", "outputs/data_cache/sft_corpora"))

# Label of the instruction line of the free-form template. ``السؤال:`` keeps
# the exact anchors the LightEval prompts use; ``التعليمات:`` is the
# one-line alternative for imperative instructions.
INSTRUCTION_LABEL = "السؤال:"


# --------------------------------------------------------------------------
# Uniform record schema
# --------------------------------------------------------------------------

@dataclass
class QARecord:
    """A single question-answering example after normalization across corpora.

    ``prompt_template`` selects the surface form:
      * ``"qa"`` (default): extractive QA — ``السياق: …\\nالسؤال: …\\nالإجابة:``
        followed by the answer text. Used by Arabic-SQuAD, TyDiQA-Arabic, ARCD.
      * ``"mcq_letter"``: 4-way MCQ — LightEval-official letter prompt
        (instruction + question + ``أ.`` / ``ب.`` / ``ج.`` / ``د.`` listing +
        ``الإجابة:``) followed by a single Arabic letter. Used by the
        synthetic ``arabic_squad_mcq`` corpus to teach the eval-time MCQ
        format. ``choices`` must be set when ``prompt_template == "mcq_letter"``.
      * ``"instruction"``: free-form instruction following — the ``qa``
        shape with ``question`` = the instruction, ``answer`` = the model
        output and the ``السياق:`` line only when ``context`` (the
        instruction's input) is non-empty. Used by CIDAR, Bactrian-X and
        the Aya collection.

    ``category`` (extractive / mcq / free_form) is derived from the template
    and is what ``PhaseConfig.mixture`` shares refer to.
    """
    id: str
    question: str
    context: str
    answer: str
    source: str  # corpus name: a key of CORPUS_CATEGORY
    prompt_template: str = "qa"  # "qa" | "mcq_letter" | "instruction"
    choices: Optional[List[str]] = None  # required when prompt_template == "mcq_letter"

    @property
    def category(self) -> str:
        return TEMPLATE_CATEGORY[self.prompt_template]


TEMPLATE_CATEGORY: Dict[str, str] = {
    "qa": "extractive",
    "mcq_letter": "mcq",
    "instruction": "free_form",
}


# --------------------------------------------------------------------------
# Prompt format (single source of truth — do NOT import from tasks/)
# --------------------------------------------------------------------------

def _format_qa_prompt(record: QARecord) -> str:
    """Prompt text up to (but excluding) the answer. Used for LCP masking.

    Dispatches on ``record.prompt_template``:

      * ``"qa"`` (default): ``السياق: …\\nالسؤال: …\\nالإجابة:`` — matches the
        LightEval-official eval prefix (no ``### `` markers) so train/eval
        share the same prefix tokens.
      * ``"mcq_letter"``: LightEval-official MCQ letter prompt — instruction
        line + question + ``أ.`` / ``ب.`` / ``ج.`` / ``د.`` listing +
        ``الإجابة:``. ``record.choices`` must be set; when ``record.context``
        is non-empty it's prepended as ``السياق: …\\n`` (mirrors
        ``ArabicExamTask._format_eval_context``).
      * ``"instruction"``: the ``qa`` shape with the ``السياق:`` line only
        when the record carries an input (``_format_instruction_prompt``).
    """
    if record.prompt_template == "mcq_letter":
        return _format_mcq_letter_prompt(record)
    if record.prompt_template == "instruction":
        return _format_instruction_prompt(record)
    if record.prompt_template != "qa":
        raise ValueError(
            f"unknown prompt_template {record.prompt_template!r} (record id={record.id!r}); "
            f"known: {sorted(TEMPLATE_CATEGORY)}"
        )
    return (
        f"السياق: {record.context}\n"
        f"السؤال: {record.question}\n"
        f"الإجابة:"
    )


def _format_instruction_prompt(record: QARecord) -> str:
    """Free-form instruction prompt: the extractive shape, context optional.

    ``[السياق: {input}\n]{INSTRUCTION_LABEL} {instruction}\nالإجابة:`` — the
    ``السياق:`` line only when the record carries an input.
    """
    head = f"السياق: {record.context}\n" if record.context else ""
    return f"{head}{INSTRUCTION_LABEL} {record.question}\nالإجابة:"


def _format_mcq_letter_prompt(record: QARecord) -> str:
    """LightEval-official letter MCQ prompt for synthetic ``arabic_squad_mcq``."""
    # Lazy import to keep the layering intentional: data/ knows about
    # tasks/lighteval/utils as a generic helper module, not vice-versa.
    from arabic_eval.tasks.lighteval.utils import format_mcq_context_letter_official

    if not record.choices:
        raise ValueError(
            f"prompt_template='mcq_letter' requires record.choices "
            f"(record id={record.id!r})"
        )
    base = format_mcq_context_letter_official(record.question, record.choices)
    if record.context:
        return f"السياق: {record.context}\n{base}"
    return base


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


def _load_arabic_squad_mcq(split: str) -> List[QARecord]:
    """Lazy-import wrapper for the synthetic MCQ corpus (avoids cycling
    through ``synthetic_mcq.py`` at module-import time)."""
    from .synthetic_mcq import load_arabic_squad_mcq
    return load_arabic_squad_mcq(split)


# --------------------------------------------------------------------------
# Free-form instruction corpora (CIDAR, Bactrian-X, Aya collection)
# --------------------------------------------------------------------------

def _clean_text(text: Optional[str]) -> str:
    """Windows line endings → ``\n``; surrounding whitespace stripped."""
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _instruction_record(source: str, rec_id: str, instruction: Optional[str],
                        context: Optional[str], output: Optional[str]) -> Optional[QARecord]:
    instruction, context, output = _clean_text(instruction), _clean_text(context), _clean_text(output)
    if not (instruction and output):
        return None
    return QARecord(
        id=rec_id, question=instruction, context=context, answer=output,
        source=source, prompt_template="instruction",
    )


def _load_cidar(split: str) -> List[QARecord]:
    """Load arbml/CIDAR (10 000 rows, ``instruction`` / ``output`` / ``index``, train only).

    Outputs ship with ``\r\n`` line endings, normalized here. Some
    instructions embed the text they act on (summaries); there is no
    separate input field, so ``context`` is always empty. The pinned
    revision repeats 33 exact ``(instruction, output)`` pairs and reuses
    18 ``index`` values: exact duplicates are dropped (first kept) and a
    repeated index gets its row position appended so record ids stay unique.
    """
    if split != "train":
        raise ValueError(f"cidar has no '{split}' split (only 'train' is available)")
    from datasets import load_dataset
    ds = load_dataset("arbml/CIDAR", split="train", revision=PINNED_REVISIONS["cidar"])
    records: List[QARecord] = []
    seen_pairs = set()
    seen_ids = set()
    n_dup = 0
    for pos, ex in enumerate(ds):
        key = (_clean_text(ex["instruction"]), _clean_text(ex["output"]))
        if key in seen_pairs:
            n_dup += 1
            continue
        seen_pairs.add(key)
        rec_id = f"cidar-{ex['index']}"
        if rec_id in seen_ids:
            rec_id = f"{rec_id}-{pos}"
        rec = _instruction_record("cidar", rec_id, ex["instruction"], "", ex["output"])
        if rec is not None:
            seen_ids.add(rec_id)
            records.append(rec)
    logger.info("cidar/%s: loaded %d records (of %d rows; %d exact duplicates dropped)",
                split, len(records), len(ds), n_dup)
    return records


def _load_bactrian_x_ar(split: str) -> List[QARecord]:
    """Load the Arabic split of MBZUAI/Bactrian-X (67 017 rows: 52 002 Alpaca
    + 15 015 Dolly instructions machine-translated to Arabic, outputs
    generated in Arabic).

    The repo is a loading-script dataset, which ``datasets`` refuses since
    3.0, so ``data/ar.json.gz`` is fetched directly (one gzip JSON list of
    ``{instruction, input, id, output}``). 641 rows carry ``input: null``
    (treated as empty); ``input`` becomes the record's ``context``.
    """
    if split != "train":
        raise ValueError(f"bactrian_x_ar has no '{split}' split (only 'train' is available)")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "MBZUAI/Bactrian-X", "data/ar.json.gz", repo_type="dataset",
        revision=PINNED_REVISIONS["bactrian_x_ar"],
    )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = json.load(f)
    records: List[QARecord] = []
    for ex in rows:
        rec = _instruction_record(
            "bactrian_x_ar", f"bactrian-{ex['id']}", ex.get("instruction"), ex.get("input"), ex.get("output"),
        )
        if rec is not None:
            records.append(rec)
    logger.info("bactrian_x_ar/%s: loaded %d records (of %d rows)", split, len(records), len(rows))
    return records


AYA_REPO = "CohereForAI/aya_collection_language_split"
AYA_CONFIG = "standard_arabic"
AYA_TRAIN_FILES = tuple(f"{AYA_CONFIG}/train-0000{i}-of-00003.parquet" for i in range(3))
# The genuinely free-form sub-datasets of the 5.86 M-row standard_arabic
# split (the rest is templated Wiki-split simplification, SODA dialogue,
# event linking and translated HotpotQA / NQ / CNN-DM …).
AYA_DEFAULT_INCLUDE = ("Aya-Dataset", "Dolly-v2 (T)")
_AYA_CONTEXT_SPLIT = re.compile(r"\n\s*Context:\s*", re.IGNORECASE)


def _load_aya_ar(split: str, include_datasets: Sequence[str] = AYA_DEFAULT_INCLUDE) -> List[QARecord]:
    """Load an allowlist of sub-datasets of the Aya collection, ``standard_arabic`` config.

    Facts the parser encodes (verified on the pinned revision, 2026-09-17):
    ``inputs`` / ``targets`` / ``dataset_name`` / ``script`` columns; half
    of ``Dolly-v2 (T)`` (14 808 of 29 616 rows) is ``script == "Latn"`` —
    English or ``<unk>`` garbage — so only ``Arab``-script rows are kept;
    4 522 Dolly rows embed their passage after a literal ``\nContext:``
    label, split here into ``context`` so the English label never reaches
    the prompt. ``Aya-Dataset`` (4 995 rows) is the human-written Standard
    Arabic part of CohereForAI/aya_dataset.

    The three train shards (1.34 GB) are downloaded once into the HF cache
    and read with pyarrow filters; the filtered subset is cached as Parquet
    under ``SFT_CORPORA_CACHE_DIR/aya_ar/<fingerprint>.parquet`` keyed on
    revision + allowlist, so later runs skip the shards entirely.
    """
    if split != "train":
        raise ValueError(f"aya_ar has no '{split}' split (only 'train' is available)")
    include = list(include_datasets)
    if not include:
        raise ValueError("aya_ar: include_datasets must name at least one Aya sub-dataset")
    fp = hashlib.sha256(json.dumps(
        {"revision": PINNED_REVISIONS["aya_ar"], "config": AYA_CONFIG, "include": sorted(include), "script": "Arab"},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()[:16]
    cache_path = SFT_CORPORA_CACHE_DIR / "aya_ar" / f"{fp}.parquet"
    table = _aya_cached_subset(cache_path, include)
    records: List[QARecord] = []
    seen_names: Dict[str, int] = {}
    for ex in table.to_pylist():
        inputs = ex["inputs"] or ""
        context = ""
        m = _AYA_CONTEXT_SPLIT.search(inputs)
        if m:
            context = inputs[m.end():]
            inputs = inputs[:m.start()]
        rec = _instruction_record("aya_ar", f"aya-{ex['id']}", inputs, context, ex["targets"])
        if rec is not None:
            records.append(rec)
            seen_names[ex["dataset_name"]] = seen_names.get(ex["dataset_name"], 0) + 1
    missing = [n for n in include if n not in seen_names]
    if missing:
        raise ValueError(
            f"aya_ar: include_datasets {missing} matched no Arab-script row of {AYA_REPO}/{AYA_CONFIG} "
            f"(present: {sorted(seen_names)}); check the sub-dataset names (dataset_name column)"
        )
    logger.info("aya_ar/%s: loaded %d records %s", split, len(records), seen_names)
    return records


def _aya_cached_subset(cache_path: Path, include: Sequence[str]):
    """The allowlisted Arab-script rows as a pyarrow table, from the local
    Parquet cache or (once) from the pinned Hub shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ["id", "inputs", "targets", "dataset_name", "script"]
    if cache_path.exists():
        return pq.read_table(cache_path, columns=columns)
    from huggingface_hub import hf_hub_download
    tables = []
    for name in AYA_TRAIN_FILES:
        path = hf_hub_download(AYA_REPO, name, repo_type="dataset", revision=PINNED_REVISIONS["aya_ar"])
        tables.append(pq.read_table(
            path, columns=columns,
            filters=[("dataset_name", "in", list(include)), ("script", "=", "Arab")],
        ))
    table = pa.concat_tables(tables)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp)
    os.replace(tmp, cache_path)
    logger.info("aya_ar: cached %d filtered rows → %s", table.num_rows, cache_path)
    return table


_LOADERS = {
    "arabic_squad": _load_arabic_squad,
    "tydiqa_arabic": _load_tydiqa_arabic,
    "arcd": _load_arcd,
    "arabic_squad_mcq": _load_arabic_squad_mcq,
    "cidar": _load_cidar,
    "bactrian_x_ar": _load_bactrian_x_ar,
    "aya_ar": _load_aya_ar,
}

# Every loader has a category and vice versa (the config validators rely on it).
assert set(_LOADERS) == set(CORPUS_CATEGORY), (set(_LOADERS) ^ set(CORPUS_CATEGORY))


def load_corpus(name: str, split: str, **params: Any) -> List[QARecord]:
    """Resolve a registry name + split to a list of normalized QARecord.

    ``params`` are the corpus' entry of ``training.corpus_params`` (today
    only ``aya_ar`` takes any: ``include_datasets``).
    """
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise KeyError(
            f"unknown corpus name {name!r}; known names: {sorted(_LOADERS)}"
        ) from None
    return loader(split, **params)


def load_corpora(
    names: Sequence[str],
    splits: Mapping[str, str] | str,
    corpus_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[QARecord]:
    """Load and concatenate multiple corpora.

    ``splits`` may be a single string (applied to all names) or a mapping
    ``{corpus_name: split_name}`` (per-corpus override; useful when Phase 3
    SFT trains on TyDiQA train + ARCD train but evaluates on TyDiQA val +
    ARCD val). ``corpus_params`` is ``training.corpus_params``.
    """
    if isinstance(splits, str):
        splits = {n: splits for n in names}
    corpus_params = corpus_params or {}
    out: List[QARecord] = []
    for n in names:
        s = splits.get(n)
        if s is None:
            raise KeyError(f"no split provided for corpus {n!r}")
        out.extend(load_corpus(n, s, **dict(corpus_params.get(n, {}))))
    return out


# --------------------------------------------------------------------------
# Latin-row filtering (shared predicate with the eval-side LightEval flag)
# --------------------------------------------------------------------------

def filter_latin_records(records: Sequence[QARecord]) -> List[QARecord]:
    """Drop records whose question, context, answer, or any choice contains
    a Latin-script letter. Mirrors ``LightEvalBenchmarkTask._row_has_latin``
    so the per-phase ``clean_latin_rows`` flag uses the same predicate as
    the eval-side one.
    """
    from ..tokenizers.utils.arabic_text import contains_latin_letters

    def _has_latin(rec: QARecord) -> bool:
        fields = [rec.question, rec.context, rec.answer]
        if rec.choices:
            fields.extend(rec.choices)
        return any(contains_latin_letters(t) for t in fields)

    return [r for r in records if not _has_latin(r)]


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


def tokenize_record(
    rec: QARecord,
    tokenizer: BaseTokenizer,
    max_length: int,
    loss_target: str,
) -> "tuple[Optional[Dict[str, Any]], int, bool]":  # noqa: UP037
    """Tokenize one record into a collator entry.

    Returns ``(entry, loss_tokens, truncated)``: ``entry`` is ``None`` when
    ``loss_target='answer_only'`` and truncation ate the whole answer (the
    LCP masking finds no answer token); ``loss_tokens`` is the number of
    positions the loss sees (unmasked labels, or every token under
    ``full_sequence``); ``truncated`` is whether the full text reached
    ``max_length`` (its tail — usually the answer — was cut).
    """
    full_text = _format_qa_full(rec)
    full_enc = tokenizer.encode(full_text, max_length=max_length, truncation=True)
    entry: Dict[str, Any] = {"input_ids": full_enc.input_ids}
    if full_enc.char_ids is not None:
        entry["char_ids"] = full_enc.char_ids
    truncated = len(full_enc.input_ids) >= max_length
    if loss_target == "answer_only":
        prompt_text = _format_qa_prompt(rec)
        prompt_enc = tokenizer.encode(prompt_text, max_length=max_length, truncation=True)
        labels = compute_answer_only_labels(prompt_enc.input_ids, full_enc.input_ids)
        if labels is None:
            return None, 0, truncated
        entry["labels"] = labels
        return entry, sum(1 for l in labels if l != -100), truncated
    return entry, len(full_enc.input_ids), truncated


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
        entry, loss_tokens, _truncated = tokenize_record(rec, tokenizer, max_length, loss_target)
        if entry is None:
            n_dropped += 1
            continue
        if loss_target == "answer_only":
            n_completion_tokens += loss_tokens
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
