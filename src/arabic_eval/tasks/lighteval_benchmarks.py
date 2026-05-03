"""LightEval-based evaluation tasks for Arabic benchmark datasets.

Four benchmarks: ACVA, Alghafa, Culture-Arabic-MMLU, Arabic-Exam.

Data split strategy:
  - 10 % of total benchmark data  → supervised fine-tuning (SFT)
  - 90 % of total benchmark data  → LightEval log-likelihood evaluation

Evaluation follows LightEval's multiple-choice methodology: for each answer
option compute log P(option | question_context) via the fine-tuned model, then
predict the option with highest log-likelihood.  Accuracy is the reported metric.

The ``LightEvalModelWrapper`` class exposes the same ``loglikelihood`` interface
that LightEval expects from custom model adapters, so it can be plugged into a
full LightEval pipeline if needed.
"""
from __future__ import annotations

import logging
import zlib
from abc import abstractmethod
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from arabic_eval.data.collation import get_collator
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.registry import task_registry
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType
from arabic_eval.utils.io import write_failure_csv

logger = logging.getLogger("arabic_eval.tasks.lighteval_benchmarks")

# Optional LightEval import — presence is checked so the package loads without it.
try:
    import lighteval  # noqa: F401
    LIGHTEVAL_AVAILABLE = True
    logger.info("LightEval detected; evaluations will use its log-likelihood methodology.")
except ImportError:
    LIGHTEVAL_AVAILABLE = False
    logger.warning(
        "lighteval not installed. Benchmark evaluation will use the built-in "
        "log-likelihood implementation (same methodology). "
        "Install with: pip install lighteval>=0.6.0"
    )

# Latin letters — used ONLY for dataset field-name lookups and answer-key parsing
# (datasets store choices as columns "A"/"B"/"C"/"D" and answer keys as "A"–"D").
CHOICE_LETTERS: List[str] = ["A", "B", "C", "D", "E"]

# Arabic letters — used in all user-facing prompt text (SFT training + evaluation).
ARABIC_CHOICE_LETTERS: List[str] = ["أ", "ب", "ج", "د", "هـ"]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _format_mcq_context(question: str, choices: List[str]) -> str:
    """Build the context string for a multiple-choice question (LightEval convention)."""
    lines = [f"السؤال: {question}", ""]
    for i, choice in enumerate(choices):
        letter = ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i)
        lines.append(f"{letter}. {choice}")
    lines.append("الإجابة:")
    return "\n".join(lines)


def _format_mcq_full(question: str, choices: List[str], answer_idx: int) -> str:
    """Full MCQ text including the correct answer letter — used for SFT."""
    context = _format_mcq_context(question, choices)
    letter = ARABIC_CHOICE_LETTERS[answer_idx] if answer_idx < len(ARABIC_CHOICE_LETTERS) else str(answer_idx)
    return f"{context} {letter}"


# ---------------------------------------------------------------------------
# Core log-likelihood computation (LightEval methodology)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_loglikelihood(
    model: BaseModelAdapter,
    tokenizer: BaseTokenizer,
    context: str,
    continuation: str,
    max_length: int = 512,
) -> float:
    """
    Compute log P(continuation | context) following LightEval's approach.

    The full sequence ``context + continuation`` is encoded once; the model's
    forward pass yields logits; we sum the log-probabilities of the continuation
    tokens only (the tokens after ``len(context_tokens)``).

    CharacterBERT (``character_cnn``) uses word-level logits: the batch is built
    from ``char_ids`` instead of ``input_ids``, but continuation scoring follows
    the same causal-LM approach using word vocabulary indices.
    """
    full_text = context + continuation
    ctx_enc = tokenizer.encode(context, max_length=max_length, truncation=True, padding=False)
    full_enc = tokenizer.encode(full_text, max_length=max_length, truncation=True, padding=False)

    ctx_len = len(ctx_enc.input_ids)
    full_len = len(full_enc.input_ids)

    if full_len <= ctx_len:
        # Continuation was completely truncated — should not happen for single letters.
        return -1e9

    attention_mask = torch.tensor([full_enc.attention_mask], device=model.device)

    if tokenizer.embedding_type == EmbeddingType.CHARACTER_CNN:
        # CharacterBERT: input is 3-D char_ids; output logits are over word vocab.
        char_ids = torch.tensor([full_enc.char_ids], device=model.device)
        batch = {"char_ids": char_ids, "attention_mask": attention_mask}
    else:
        input_ids = torch.tensor([full_enc.input_ids], device=model.device)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "labels": input_ids.clone()}

    output = model.forward(batch)
    logits = output["logits"]                          # [1, seq_len, vocab_size]
    log_probs = F.log_softmax(logits[0], dim=-1)       # [seq_len, vocab_size]

    # Causal LM: position i predicts position i+1.
    # Sum log P(token_{ctx_len} … token_{full_len-1}) given their left contexts.
    total_ll = 0.0
    for pos in range(ctx_len - 1, full_len - 1):
        next_tok = full_enc.input_ids[pos + 1]
        total_ll += log_probs[pos, next_tok].item()

    return total_ll


# ---------------------------------------------------------------------------
# LightEval model wrapper
# ---------------------------------------------------------------------------

class LightEvalModelWrapper:
    """
    Wraps ``BaseModelAdapter`` to expose LightEval's ``loglikelihood`` interface.

    This class can be used as-is inside our evaluation loop or as the basis
    for a custom ``LightevalModel`` subclass in a full LightEval pipeline.
    """

    def __init__(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[float]:
        """
        Compute log P(continuation | context) for each ``(context, continuation)`` pair.

        Matches LightEval's ``loglikelihood`` request/response protocol so this
        wrapper can be swapped for a ``LightevalModel`` subclass transparently.
        """
        return [
            _compute_loglikelihood(
                self.model, self.tokenizer, ctx, cont, self.max_length
            )
            for ctx, cont in requests
        ]

    def evaluate_mcq(
        self,
        examples: List[Dict[str, Any]],
        collect_failures: bool = False,
        task: Optional["LightEvalBenchmarkTask"] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Run LightEval-style multiple-choice accuracy evaluation.

        For each example: build one ``(context, continuation)`` pair per choice,
        call ``loglikelihood``, predict the argmax, compare to ground truth.

        Continuation building delegates to the active task via
        ``task._format_eval_context(ex)`` and ``task._build_continuations(ex)``.
        When ``task`` is ``None`` the wrapper falls back to the legacy
        Arabic-letter scoring (used by direct callers and by the test suite
        that exercises this wrapper standalone).

        When ``collect_failures`` is True, also returns a list of failure
        records (one per wrong-answer example) with per-choice log-likelihoods
        for downstream CSV reporting. When False, the second tuple element is
        an empty list. The ``gold_letter`` / ``pred_letter`` fields hold the
        winning continuation string (without leading space) — for ACVA these
        are the Arabic words ``صح`` / ``خطأ``; for letter-based benchmarks
        they remain ``أ`` / ``ب`` / ``ج`` / ``د``.
        """
        correct = 0
        total = 0
        failures: List[Dict[str, Any]] = []

        for idx, ex in enumerate(tqdm(examples, desc="LightEval MCQ", unit="example")):
            if task is not None:
                context = task._format_eval_context(ex)
                continuations = task._build_continuations(ex)
            else:
                context = _format_mcq_context(ex["question"], ex["choices"])
                continuations = [
                    " " + (ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i))
                    for i in range(len(ex["choices"]))
                ]
            requests: List[Tuple[str, str]] = [(context, c) for c in continuations]
            log_likelihoods = self.loglikelihood(requests)
            predicted = int(np.argmax(log_likelihoods))
            gold = ex["answer"]
            if predicted == gold:
                correct += 1
            elif collect_failures:
                # The "letter" column holds the winning continuation string —
                # for ACVA this is the Arabic word, for letter-based benchmarks
                # it's still the Arabic letter. Strip the leading space.
                gold_token = continuations[gold].lstrip()
                pred_token = continuations[predicted].lstrip()
                record: Dict[str, Any] = {
                    "index": idx,
                    "question": ex["question"],
                    "gold_idx": gold,
                    "gold_letter": gold_token,
                    "pred_idx": predicted,
                    "pred_letter": pred_token,
                    "ll_margin": round(
                        float(log_likelihoods[predicted]) - float(log_likelihoods[gold]), 6
                    ),
                }
                for i, choice in enumerate(ex["choices"]):
                    record[f"choice_{i}"] = choice
                    record[f"ll_{i}"] = round(float(log_likelihoods[i]), 6)
                failures.append(record)
            total += 1

        metrics = {
            "accuracy": round(correct / max(total, 1), 4),
            "num_samples": total,
        }
        return metrics, failures


# ---------------------------------------------------------------------------
# Dataset helper for SFT dataloader
# ---------------------------------------------------------------------------

class MCQTokenizedDataset(Dataset):
    """Tokenized multiple-choice examples for supervised fine-tuning."""

    def __init__(self, encodings: List[Dict[str, Any]]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.encodings[idx]


# ---------------------------------------------------------------------------
# Base benchmark task
# ---------------------------------------------------------------------------

class LightEvalBenchmarkTask(BaseTask):
    """
    Base class for LightEval-based Arabic multiple-choice benchmarks.

    Subclasses must implement:
      * ``_default_dataset_name()``  — default HuggingFace dataset path
      * ``_parse_example(raw)``      — normalise a raw dict into
                                       ``{"question", "choices", "answer"}``
      * ``name``                     — registry key string

    The 10/90 split uses a fixed RNG seed (configurable via ``seed`` param)
    so every tokenizer variant in a sweep sees the identical split.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.dataset_name: str = config.get("dataset_name", self._default_dataset_name())
        self.dataset_config: Optional[str] = config.get("dataset_config", None)
        self.cache_dir: str = config.get("cache_dir", "outputs/data_cache")
        self.max_length: int = config.get("max_length", 512)
        self.train_split_ratio: float = config.get("train_split_ratio", 0.10)
        self.seed: int = config.get("seed", 42)
        self._cached_splits: Optional[Tuple[List[Dict], List[Dict]]] = None

    @abstractmethod
    def _default_dataset_name(self) -> str:
        """Default HuggingFace dataset identifier."""
        ...

    def _parse_mcq_generic(
        self,
        raw: Dict[str, Any],
        question_keys: Tuple[str, ...] = ("question",),
        answer_keys: Tuple[str, ...] = ("answer", "label", "answerKey"),
    ) -> Optional[Dict[str, Any]]:
        """
        Generic parser that handles the most common MCQ dataset schemas:

        * Separate A/B/C/D columns + answer letter/index
        * ``choices`` or ``options`` list + answer index
        * PIQA-style ``sol1``/``sol2`` + label
        """
        question = ""
        for key in question_keys:
            question = str(raw.get(key, "")).strip()
            if question:
                break
        if not question:
            return None

        choices: List[str] = []
        if "A" in raw and "B" in raw:
            for letter in CHOICE_LETTERS:
                val = raw.get(letter, "")
                if val:
                    choices.append(str(val).strip())
        elif "choices" in raw:
            choices = [str(c).strip() for c in raw["choices"]]
        elif "options" in raw:
            choices = [str(c).strip() for c in raw["options"]]
        elif "sol1" in raw and "sol2" in raw:
            choices = [str(raw["sol1"]).strip(), str(raw["sol2"]).strip()]

        if not choices:
            return None

        answer_raw = None
        for key in answer_keys:
            answer_raw = raw.get(key)
            if answer_raw is not None:
                break
        if answer_raw is None:
            return None

        if isinstance(answer_raw, str) and answer_raw.strip().upper() in CHOICE_LETTERS:
            answer_idx = CHOICE_LETTERS.index(answer_raw.strip().upper())
        elif isinstance(answer_raw, (int, np.integer)):
            answer_idx = int(answer_raw)
        else:
            try:
                answer_idx = int(answer_raw)
            except (ValueError, TypeError):
                return None

        if answer_idx < 0 or answer_idx >= len(choices):
            return None

        return {"question": question, "choices": choices, "answer": answer_idx}

    @abstractmethod
    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse one raw dataset row into ``{"question": str, "choices": List[str],
        "answer": int}``.  Return ``None`` to skip malformed rows.
        """
        ...

    # ------------------------------------------------------------------
    # Prompt / continuation hooks (overridable by subclasses)
    # ------------------------------------------------------------------
    # The default implementations match LightEval's letter-based MCQ protocol:
    # the prompt lists choices as "أ. <choice_0>", "ب. <choice_1>", …, ends
    # with "الإجابة:", and the model is scored on the continuation " أ" / " ب"
    # / etc. Subclasses (e.g. ``ACVATask``) can override these hooks to score
    # words instead of letters when single-letter continuations carry too
    # little signal.

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        """Context string fed to the model at eval time (everything before
        the continuation). Default: standard MCQ prompt with letter-labelled
        choices."""
        return _format_mcq_context(ex["question"], ex["choices"])

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        """Continuation strings (one per choice, in answer-index order). Each
        starts with a leading space to match LightEval's tokenisation
        convention. Default: Arabic letters " أ", " ب", " ج", " د", …"""
        n = len(ex["choices"])
        return [
            " " + (ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i))
            for i in range(n)
        ]

    def _format_sft_text(self, ex: Dict[str, Any]) -> str:
        """Full text used for supervised fine-tuning (context + correct
        continuation). Default: ``_format_mcq_full`` (letter-based)."""
        return _format_mcq_full(ex["question"], ex["choices"], ex["answer"])

    # ------------------------------------------------------------------
    # Data loading and splitting
    # ------------------------------------------------------------------

    def _load_all_examples(self) -> List[Dict[str, Any]]:
        """Load and concatenate all available splits, then parse each row.

        If the dataset requires a config name (e.g. ``OALL/ACVA`` has 58 topic
        sub-configs), ``dataset_config=None`` triggers a ``ValueError``.  In that
        case we enumerate all configs via ``get_dataset_config_names`` and merge
        them automatically — no manual listing of configs needed.
        """
        try:
            raw_ds = load_dataset(
                self.dataset_name,
                self.dataset_config,
                cache_dir=self.cache_dir,
            )
        except ValueError as exc:
            if "Config name is missing" in str(exc) and self.dataset_config is None:
                # Multi-config dataset: load every sub-config and merge.
                return self._load_all_configs_merged()
            raise RuntimeError(
                f"Failed to load '{self.dataset_name}' "
                f"(config={self.dataset_config!r}): {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load '{self.dataset_name}' "
                f"(config={self.dataset_config!r}): {exc}"
            ) from exc

        # Merge all predefined splits so the 10/90 split is our own, not the
        # dataset author's, preventing unintentional data leakage.
        combined = concatenate_datasets(list(raw_ds.values()))
        # Single-config case: use a sentinel group name. Stratified split in
        # _get_splits then degenerates to one group → behaviour matches the
        # pre-stratification code-path modulo the floor→ceil rounding change.
        return self._parse_combined(
            combined,
            context=f"{self.dataset_name} (config={self.dataset_config!r})",
            source_config=self.dataset_config or "_default",
        )

    def _load_all_configs_merged(self) -> List[Dict[str, Any]]:
        """Enumerate every sub-config of a multi-config dataset and merge examples.

        Used automatically when ``load_dataset(name, None)`` raises
        ``ValueError: Config name is missing``.
        """
        try:
            config_names = get_dataset_config_names(self.dataset_name)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot enumerate configs for '{self.dataset_name}': {exc}. "
                "Set dataset_config to a specific config name in your task params."
            ) from exc

        logger.info(
            "Multi-config dataset '%s': merging %d sub-configs (%s … %s)",
            self.dataset_name, len(config_names),
            config_names[0] if config_names else "",
            config_names[-1] if config_names else "",
        )
        all_examples: List[Dict[str, Any]] = []
        skipped: List[str] = []

        for config_name in config_names:
            try:
                raw_ds = load_dataset(self.dataset_name, config_name, cache_dir=self.cache_dir)
                combined = concatenate_datasets(list(raw_ds.values()))
                n_before = len(all_examples)
                all_examples.extend(
                    self._parse_combined(
                        combined,
                        context=config_name,
                        source_config=config_name,
                        quiet=True,
                    )
                )
                logger.debug("Config '%s': +%d examples", config_name, len(all_examples) - n_before)
            except Exception as exc:
                logger.warning("Skipping config '%s' of '%s': %s", config_name, self.dataset_name, exc)
                skipped.append(config_name)

        if skipped:
            logger.warning(
                "Skipped %d/%d sub-configs of '%s': %s%s",
                len(skipped), len(config_names), self.dataset_name,
                skipped[:3], " …" if len(skipped) > 3 else "",
            )

        if not all_examples:
            raise RuntimeError(
                f"No valid MCQ examples found across all {len(config_names)} "
                f"sub-configs of '{self.dataset_name}'. "
                "Check that _parse_example matches the dataset schema."
            )

        logger.info(
            "Loaded %d total examples from %d/%d sub-configs of '%s'",
            len(all_examples), len(config_names) - len(skipped), len(config_names), self.dataset_name,
        )
        return all_examples

    def _parse_combined(
        self,
        combined,
        context: str = "",
        source_config: str = "_default",
        quiet: bool = False,
    ) -> List[Dict[str, Any]]:
        """Parse a concatenated dataset into validated MCQ dicts.

        Each parsed dict is stamped with a private ``_source_config`` key
        recording the originating sub-config name, used by
        ``_get_splits`` to stratify the 10/90 partition so every sub-config
        is represented in the SFT split. Single-config datasets use the
        sentinel ``"_default"``.
        """
        examples: List[Dict[str, Any]] = []
        for raw in combined:
            parsed = self._parse_example(dict(raw))
            if parsed is not None:
                parsed["_source_config"] = source_config
                examples.append(parsed)

        if not examples and not quiet:
            raise RuntimeError(
                f"No valid MCQ examples found in {context!r}. "
                "Check that _parse_example matches the dataset schema."
            )

        if not quiet:
            logger.info("Loaded %d valid MCQ examples from %s", len(examples), context)
        return examples

    def _get_splits(self) -> Tuple[List[Dict], List[Dict]]:
        """Return ``(finetune_split, eval_split)`` with stratified sub-config
        coverage, memoised per instance.

        For multi-config benchmarks (ACVA's 58 cultural topics, Alghafa's
        sub-tasks, Arabic_Exam's school subjects), random global sampling can
        leave entire sub-configs unrepresented in the small SFT split. This
        method instead splits **within each sub-config** using a per-group
        seed derived from ``self.seed`` and a CRC32 hash of the config name,
        guaranteeing ≥ 1 SFT example per sub-config while staying fully
        deterministic.

        Single-config datasets degenerate to one ``_default`` group →
        behaviour matches the unstratified code-path modulo the floor→ceil
        rounding change for tiny groups.

        Note: the rounding change (``int`` → ``max(1, ceil(...))``) shifts
        the SFT total up by ~one example per sub-config compared to the
        pre-stratification implementation. This is the fix that guarantees
        coverage; comparison-table consumers should be aware that historic
        run totals will not match exactly.
        """
        if self._cached_splits is None:
            all_examples = self._load_all_examples()

            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for ex in all_examples:
                groups[ex.get("_source_config", "_default")].append(ex)

            train_ex: List[Dict[str, Any]] = []
            eval_ex: List[Dict[str, Any]] = []
            per_config_log: List[Tuple[str, int, int]] = []

            for config_name in sorted(groups.keys()):
                group = groups[config_name]
                # Per-config seed: deterministic, depends on master seed and
                # config name (so renaming the user-set seed shifts every
                # group consistently, while different configs get distinct
                # permutations within a single run).
                grp_seed = (self.seed + zlib.crc32(config_name.encode("utf-8"))) & 0x7FFFFFFF
                rng = np.random.default_rng(grp_seed)
                perm = rng.permutation(len(group))
                # ceil + max(1, ...) so even a 1-row group lands in SFT.
                n_train = max(1, ceil(len(group) * self.train_split_ratio))
                n_train = min(n_train, len(group))  # never empty the eval side trivially
                train_ex.extend(group[i] for i in perm[:n_train])
                eval_ex.extend(group[i] for i in perm[n_train:])
                per_config_log.append((config_name, n_train, len(group) - n_train))

            # Final shuffle so the SFT and eval streams aren't ordered by
            # config (matters for fine-tuning batches drawn without
            # additional shuffling).
            master_rng = np.random.default_rng(self.seed)
            master_rng.shuffle(train_ex)
            master_rng.shuffle(eval_ex)

            self._cached_splits = (train_ex, eval_ex)
            logger.info(
                "%s split (stratified across %d sub-config(s)): %d SFT (%.0f%%) | %d eval (%.0f%%)",
                self.name, len(per_config_log),
                len(train_ex), self.train_split_ratio * 100,
                len(eval_ex), (1 - self.train_split_ratio) * 100,
            )
            # DEBUG-level per-config breakdown — handy when investigating
            # coverage but too noisy at INFO for 58 ACVA configs.
            for cfg, ntr, nev in per_config_log:
                logger.debug("  [%s] SFT=%d eval=%d", cfg, ntr, nev)
        return self._cached_splits

    def get_fine_tune_examples(self) -> List[Dict[str, Any]]:
        """Return the 10 % fine-tuning split."""
        return self._get_splits()[0]

    def get_eval_examples(self) -> List[Dict[str, Any]]:
        """Return the 90 % evaluation split."""
        return self._get_splits()[1]

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        tokenizer: BaseTokenizer,
        split: str = "train",
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Return a DataLoader over the **10 % fine-tune split**, tokenised as
        causal-LM text (full question + correct answer letter).

        Both ``split="train"`` and ``split="test"`` use the same fine-tune
        partition so that the 90 % evaluation set remains unseen during training.
        """
        examples = self.get_fine_tune_examples()
        if max_samples:
            examples = examples[:max_samples]

        # SFT text formatting is delegated to the task so subclasses can
        # train on word-based continuations (ACVA) without overriding the
        # whole dataloader path.
        texts = [self._format_sft_text(ex) for ex in examples]

        encodings: List[Dict[str, Any]] = []
        for text in tqdm(texts, desc=f"Tokenising {self.name}", unit="ex", leave=False):
            enc = tokenizer.encode(text, max_length=self.max_length, truncation=True)
            entry: Dict[str, Any] = {"input_ids": enc.input_ids}
            if enc.char_ids is not None:
                entry["char_ids"] = enc.char_ids
            encodings.append(entry)

        collator = get_collator(
            tokenizer.embedding_type,
            pad_token_id=tokenizer.pad_token_id,
            max_length=self.max_length,
        )
        return DataLoader(
            MCQTokenizedDataset(encodings),
            batch_size=batch_size,
            shuffle=(shuffle and split == "train"),
            collate_fn=collator,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        split: str = "test",
        max_samples: Optional[int] = None,
        failure_report_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on the **90 % held-out split** using LightEval's log-likelihood
        multiple-choice methodology.

        CharacterBERT (``character_cnn``) is scored using word-level logits;
        the continuation word IDs from the word vocabulary are used directly.

        If ``failure_report_dir`` is given, a ``<task_name>_accuracy_failures.csv``
        is written there containing one row per wrong-answer example with the
        question, choices, gold/pred letters, per-choice log-likelihoods, and
        the ``ll_margin = ll_pred - ll_gold`` (positive = model preferred wrong
        over right).
        """
        if not LIGHTEVAL_AVAILABLE:
            logger.warning(
                "lighteval package not found. Running built-in log-likelihood "
                "evaluation (identical methodology to LightEval)."
            )

        examples = self.get_eval_examples()
        if max_samples:
            examples = examples[:max_samples]

        logger.info(
            "%s: evaluating %d examples (90 %% LightEval split)", self.name, len(examples)
        )
        model.model.eval()
        wrapper = LightEvalModelWrapper(model, tokenizer, max_length=self.max_length)
        collect = failure_report_dir is not None
        # Pass `self` so the wrapper uses this task's prompt/continuation
        # hooks (ACVA scores صح/خطأ; the rest score letters).
        metrics, failures = wrapper.evaluate_mcq(
            examples, collect_failures=collect, task=self
        )
        logger.info("%s metrics: %s", self.name, metrics)

        if collect:
            max_choices = max((len(ex["choices"]) for ex in examples), default=0)
            fieldnames: List[str] = [
                "index", "question",
                *[f"choice_{i}" for i in range(max_choices)],
                "gold_idx", "gold_letter", "pred_idx", "pred_letter",
                *[f"ll_{i}" for i in range(max_choices)],
                "ll_margin",
            ]
            csv_path = Path(failure_report_dir) / f"{self.name}_accuracy_failures.csv"
            n_written = write_failure_csv(csv_path, failures, fieldnames)
            logger.info(
                "%s: wrote %d failure rows to %s", self.name, n_written, csv_path
            )

        return metrics

    @property
    def metric_names(self) -> List[str]:
        return ["accuracy", "num_samples"]


# ---------------------------------------------------------------------------
# ACVA — Arabic Culture and Values Assessment
# ---------------------------------------------------------------------------

@task_registry.register("acva")
class ACVATask(LightEvalBenchmarkTask):
    """Arabic Culture and Values Assessment (OALL/ACVA).

    Schema: id, question, answer — where answer is "صح" (True) or "خطأ" (False).
    This is a True/False dataset; choices are presented as ["صح", "خطأ"].

    ACVA-specific scoring: instead of LightEval's standard letter-based
    multiple-choice scoring (where the model is asked to emit " أ" or " ب"),
    ACVA scores the continuation strings ``" صح"`` / ``" خطأ"`` directly.
    Single-letter continuations carry almost no signal — every Arabic letter
    has roughly the same unigram prior, so the choice is dominated by noise.
    Scoring the actual T/F words gives the model lexically meaningful
    options. SFT formatting matches: training examples end with the full
    word, not a letter.
    """

    # Single source of truth for the dataset's True/False label strings.
    # Used by `_parse_example` (matching incoming raw rows), by
    # `_build_continuations` (eval-time scoring), and by `_format_sft_text`
    # (training-time supervision). To switch to e.g. صحيح/خاطئ, change here only.
    LABEL_TRUE = "صح"
    LABEL_FALSE = "خطأ"
    LABELS = (LABEL_TRUE, LABEL_FALSE)  # index 0 = TRUE, 1 = FALSE

    def _default_dataset_name(self) -> str:
        return "OALL/ACVA"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = str(raw.get("question", "")).strip()
        if not question:
            return None
        answer = str(raw.get("answer", "")).strip()
        if answer == self.LABEL_TRUE:
            answer_idx = 0
        elif answer == self.LABEL_FALSE:
            answer_idx = 1
        else:
            return None
        return {
            "question": question,
            "choices": list(self.LABELS),
            "answer": answer_idx,
        }

    # ---- ACVA-specific prompt / continuation hooks ----

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        # ACVA is True/False — no need to display "أ. صح / ب. خطأ" choice
        # lines because the answer is the word itself, not a letter.
        return f"السؤال: {ex['question']}\nالإجابة:"

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        # Score the words themselves, in the same index order as `choices`
        # (so argmax over log-likelihoods → answer index, unchanged).
        return [f" {label}" for label in self.LABELS]

    def _format_sft_text(self, ex: Dict[str, Any]) -> str:
        # SFT supervision must match eval scoring: train on the word, not
        # the letter, otherwise the model never sees " صح"/" خطأ" during
        # fine-tuning.
        return f"{self._format_eval_context(ex)} {self.LABELS[ex['answer']]}"

    @property
    def name(self) -> str:
        return "acva"


# ---------------------------------------------------------------------------
# Alghafa — AlGhafa Native Arabic benchmark
# ---------------------------------------------------------------------------

@task_registry.register("alghafa")
class AlghafaTask(LightEvalBenchmarkTask):
    """AlGhafa Native Arabic benchmark (OALL/AlGhafa-Arabic-LLM-Benchmark-Native).

    Schema: query, sol1, sol2, sol3, sol4, label — where label is 1-indexed ("1"–"4").
    The dataset has multiple sub-configs; all are merged automatically.
    """

    def _default_dataset_name(self) -> str:
        return "OALL/AlGhafa-Arabic-LLM-Benchmark-Native"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = str(raw.get("query", "")).strip()
        if not question:
            return None
        choices: List[str] = []
        for key in ("sol1", "sol2", "sol3", "sol4"):
            val = raw.get(key)
            if val is not None and str(val).strip():
                choices.append(str(val).strip())
        if not choices:
            return None
        label_raw = raw.get("label")
        if label_raw is None:
            return None
        try:
            answer_idx = int(label_raw) - 1  # 1-indexed → 0-indexed
        except (ValueError, TypeError):
            return None
        if answer_idx < 0 or answer_idx >= len(choices):
            return None
        return {"question": question, "choices": choices, "answer": answer_idx}

    @property
    def name(self) -> str:
        return "alghafa"


# ---------------------------------------------------------------------------
# Culture Arabic MMLU
# ---------------------------------------------------------------------------

@task_registry.register("culture_arabic_mmlu")
class CultureArabicMMLUTask(LightEvalBenchmarkTask):
    """Arabic MMLU benchmark (OALL/Arabic_MMLU — AceGPT Arabic MMLU).

    Schema: question, A, B, C, D, answer (letter), subject.
    Single-config dataset; no sub-config enumeration needed.
    """

    def _default_dataset_name(self) -> str:
        return "OALL/Arabic_MMLU"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._parse_mcq_generic(
            raw,
            question_keys=("question",),
            answer_keys=("answer", "label"),
        )

    @property
    def name(self) -> str:
        return "culture_arabic_mmlu"


# ---------------------------------------------------------------------------
# Arabic Exam
# ---------------------------------------------------------------------------

@task_registry.register("arabic_exam")
class ArabicExamTask(LightEvalBenchmarkTask):
    """Arabic MMLU by MBZUAI — school exam MCQ questions (MBZUAI/ArabicMMLU).

    Schema: Question, Option 1–4, Answer Key (letter A–D).
    Multi-config (one per subject); all configs are merged automatically.
    """

    def _default_dataset_name(self) -> str:
        return "MBZUAI/ArabicMMLU"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = str(raw.get("Question", raw.get("question", ""))).strip()
        if not question:
            return None
        choices: List[str] = []
        for key in ("Option 1", "Option 2", "Option 3", "Option 4"):
            val = raw.get(key)
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                choices.append(str(val).strip())
        if not choices:
            return None
        answer_key = str(raw.get("Answer Key", raw.get("answer", ""))).strip().upper()
        if answer_key not in CHOICE_LETTERS:
            return None
        answer_idx = CHOICE_LETTERS.index(answer_key)
        if answer_idx >= len(choices):
            return None
        return {"question": question, "choices": choices, "answer": answer_idx}

    @property
    def name(self) -> str:
        return "arabic_exam"
