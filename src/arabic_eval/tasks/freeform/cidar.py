"""Free-form instruction-following eval on the held-out CIDAR set.

Eval-only task (``BaseTask``): for every row of the held-out JSONL the model
greedy-decodes an answer to the *exact* Phase 3 instruction prompt
(``_format_qa_prompt`` on an ``instruction`` record — the string the
mixture trained on, so generation continues where training left off), under
the shared decoding rules of ``generation.py`` (EOS / stop marker / text-level
repetition-loop stop — a periodic tail, cut after its first copy — /
character-budget cap; ``stop_reason`` per row). Reference-based (chrF,
BERTScore) and reference-free metrics are computed inline and every row goes
to ``eval_rows/freeform_cidar.parquet`` (prompt, reference, generation, per-
row metrics). LLM-judge scoring is a separate stage
(``scripts/judge/judge_freeform.py``) that reads that file — a 31B judge does not
share the GPU with a training run, and judges get swapped.

Tokenizers that cannot generate (CharacterBERT, Farasa-CharacterBERT,
Charformer) get a typed status record, the pattern MEI uses — never a zero.
MEI itself skips this task (``task_not_mcq``).

Only meaningful for configs whose Phase 3 saw free-form corpora (the
``*_sft_mixture`` YAMLs): a model that never learned the instruction format
answers every prompt with noise and the judge cannot discriminate.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.data.finetune_corpora import QARecord, _format_qa_prompt
from arabic_eval.data.freeform_heldout import load_freeform_heldout
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.params_spec import ParamSpec
from arabic_eval.registry import task_registry
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tasks.freeform import metrics as M
from arabic_eval.tasks.freeform.generation import (
    DEFAULT_STOP_MARKERS, DecodingConfig, derive_token_cap, generate_freeform, generation_supported,
    measure_chars_per_token,
)
from arabic_eval.tokenizers.base import BaseTokenizer
from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters
from arabic_eval.utils.io import write_report_table

logger = logging.getLogger(__name__)

DEFAULT_HELDOUT_PATH = "configs/contamination/freeform_cidar_heldout_v1.jsonl"
TASK_NAME = "freeform_cidar"
# BERTScore knobs (``tasks/freeform/bertscore.py``): xlm-roberta-large layer 17, the layer the
# reference implementation uses for that model; ``None`` skips BERTScore altogether.
DEFAULT_BERTSCORE_MODEL: Optional[str] = "xlm-roberta-large"
DEFAULT_BERTSCORE_LAYER = 17
DEFAULT_BERTSCORE_BATCH_SIZE = 32
ROW_FIELDS = [
    "id", "stratum", "instruction", "context", "prompt_text", "reference", "generation", "generation_raw",
    "prompt_tokens", "gen_tokens", "gen_chars", "ref_chars", "stop_reason", "hit_cap", "hit_loop", "loop_rule",
    "loop_period", "char_truncated",
    "empty", "degenerate", "latin", "arabic_letter_ratio", "chrf",
    "bertscore_p", "bertscore_r", "bertscore_f1", "reference_roundtrip_chrf", "gen_time_sec",
]
METRIC_NAMES = [
    "chrf", "chrf_corpus", "bertscore_f1", "bertscore_p", "bertscore_r",
    "empty_rate", "degenerate_rate", "latin_rate", "arabic_letter_ratio",
    "hit_cap_rate", "loop_stop_rate", "marker_stop_rate", "eos_rate", "char_truncated_rate",
    "mean_gen_chars", "mean_gen_tokens", "mean_ref_chars", "gen_chars_per_sec", "gen_tokens_per_sec",
    "generation_wall_sec", "reference_roundtrip_chrf", "chars_per_token", "token_cap", "max_output_chars", "num_samples",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_heldout_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    alt = _repo_root() / path
    return alt if alt.exists() else path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@task_registry.register(TASK_NAME)
class FreeformCidarTask(BaseTask):
    """Params (``sweep.tasks[].params``): exactly the entries of :meth:`param_spec` —
    the ``DecodingConfig`` fields (budget / stops / misc), ``heldout_path`` and the
    three BERTScore knobs. An absent key means the code default (the pipeline hands a
    task only the experiment YAML's params; ``configs/tasks/freeform_cidar.yaml`` is
    generated *from* the spec and never read by a run). Unknown keys are ignored
    here and warned about at validation and run start.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.heldout_path = resolve_heldout_path(cfg.get("heldout_path") or DEFAULT_HELDOUT_PATH)
        # Every default is the ``DecodingConfig`` field default — never retyped here, so the
        # dataclass, this constructor and ``param_spec`` cannot drift (a test pins all three).
        d = DecodingConfig()
        self.decoding = DecodingConfig(
            max_output_chars=int(cfg.get("max_output_chars", d.max_output_chars)),
            max_prompt_tokens=int(cfg.get("max_prompt_tokens", d.max_prompt_tokens)),
            batch_size=int(cfg.get("batch_size", d.batch_size)),
            token_cap_margin=float(cfg.get("token_cap_margin", d.token_cap_margin)),
            token_cap_floor=int(cfg.get("token_cap_floor", d.token_cap_floor)),
            token_cap_ceiling=int(cfg.get("token_cap_ceiling", d.token_cap_ceiling)),
            stop_markers=tuple(cfg.get("stop_markers") or d.stop_markers),
            marker_check_every=int(cfg.get("marker_check_every", d.marker_check_every)),
            loop_stop=bool(cfg.get("loop_stop", d.loop_stop)),
            seed=int(cfg.get("seed", d.seed)),
            repetition_penalty=float(cfg.get("repetition_penalty", d.repetition_penalty)),
            no_repeat_ngram_size=int(cfg.get("no_repeat_ngram_size", d.no_repeat_ngram_size)),
        )
        self.bertscore_model: Optional[str] = cfg.get("bertscore_model", DEFAULT_BERTSCORE_MODEL)
        self.bertscore_layer = int(cfg.get("bertscore_layer", DEFAULT_BERTSCORE_LAYER))
        self.bertscore_batch_size = int(cfg.get("bertscore_batch_size", DEFAULT_BERTSCORE_BATCH_SIZE))
        self._rows: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        """Every parameter the task reads, defaults taken from ``DecodingConfig()`` and the
        module constants (never retyped). Groups: *budget* (the character budget and how
        it becomes a per-tokenizer token cap), *stops* (markers / loop stop), *scoring*
        (BERTScore), *misc*."""
        d = DecodingConfig()
        return [
            ParamSpec("max_output_chars", "int", d.max_output_chars, min=32, group="budget",
                      help="Budget of decoded text per answer, in CHARACTERS — the same amount of text for every "
                           "tokenizer; it becomes a per-tokenizer max_new_tokens from the measured chars/token "
                           "(2400 since 2026-09-21; 1200 capped 12 of the control's 250 answers)."),
            ParamSpec("token_cap_margin", "float", d.token_cap_margin, min=1.0, advanced=True, group="budget",
                      help="Safety factor on the derived token cap: max_new_tokens = max_output_chars / chars_per_token "
                           "× margin + 8, before clamping."),
            ParamSpec("token_cap_floor", "int", d.token_cap_floor, min=1, advanced=True, group="budget",
                      help="Lower clamp of the derived token cap (a tokenizer with huge chars/token still gets this many tokens)."),
            ParamSpec("token_cap_ceiling", "int", d.token_cap_ceiling, min=1, advanced=True, group="budget",
                      help="Upper clamp of the derived token cap (char-JABER lands near 2 800, well under it)."),
            ParamSpec("max_prompt_tokens", "int", d.max_prompt_tokens, min=16, group="budget",
                      help="Longest prompt encoding kept (tokenizer truncation beyond it) before generation."),
            ParamSpec("stop_markers", "list[str]", list(d.stop_markers), group="stops",
                      help="Decoded-text stop strings: the model starting a new prompt block, in the templates' exact "
                           "words (derived from finetune_corpora.py; never a bare \\n### — that cut the model's own "
                           "markdown sub-headers). One marker per line in the form."),
            ParamSpec("marker_check_every", "int", d.marker_check_every, min=1, advanced=True, group="stops",
                      help="Decode the unfinished sequences and look for a marker / loop every this many generation steps."),
            ParamSpec("loop_stop", "bool", d.loop_stop, group="stops",
                      help="Stop a sequence whose decoded text ends in a repetition loop (periodic tail or a letter ×20), "
                           "cut right after the first copy; tokenizer-agnostic, unlike a token repetition penalty."),
            ParamSpec("repetition_penalty", "float", d.repetition_penalty, min=1.0, advanced=True, group="decoding",
                      help="MEASUREMENT ONLY (decoding ablation): HF's token-level repetition penalty over the context "
                           "(prompt included, left padding excluded); 1.0 = pure greedy. Granularity-dependent — an "
                           "AraRooPat clitic token recurs in every other word — so a cell scored with it is not "
                           "comparable with another tokenizer's greedy cell."),
            ParamSpec("no_repeat_ngram_size", "int", d.no_repeat_ngram_size, min=0, advanced=True, group="decoding",
                      help="MEASUREMENT ONLY (decoding ablation): HF's ban on repeating any token n-gram of the context; "
                           "0 = off. Token-level and granularity-dependent like repetition_penalty."),
            ParamSpec("bertscore_model", "str", DEFAULT_BERTSCORE_MODEL, nullable=True, group="scoring",
                      help="Encoder of the in-house BERTScore (tasks/freeform/bertscore.py); null skips BERTScore."),
            ParamSpec("bertscore_layer", "int", DEFAULT_BERTSCORE_LAYER, min=0, advanced=True, group="scoring",
                      help="Hidden layer of the encoder whose token vectors BERTScore matches (17 for xlm-roberta-large)."),
            ParamSpec("bertscore_batch_size", "int", DEFAULT_BERTSCORE_BATCH_SIZE, min=1, advanced=True, group="scoring",
                      help="Sentence pairs per BERTScore forward pass."),
            ParamSpec("batch_size", "int", d.batch_size, min=1, group="misc",
                      help="Prompts generated per batch (left-padded); memory only, the greedy output does not depend on it."),
            ParamSpec("seed", "int", d.seed, min=0, group="misc",
                      help="Recorded in the row dump; greedy decoding itself is deterministic."),
            ParamSpec("heldout_path", "path", DEFAULT_HELDOUT_PATH, group="misc",
                      help="The held-out JSONL of prompts (id, prompt, reference, stratum); the committed v1 set of 250 "
                           "CIDAR rows, also the freeform_prompts held-out set of the contamination check."),
        ]

    # ---- BaseTask ---------------------------------------------------------
    @property
    def name(self) -> str:
        return TASK_NAME

    @property
    def metric_names(self) -> List[str]:
        return list(METRIC_NAMES)

    # ---- data --------------------------------------------------------------
    def load_examples(self) -> List[Dict[str, Any]]:
        if self._rows is None:
            self._rows = load_freeform_heldout(self.heldout_path)
        return self._rows

    @staticmethod
    def build_prompt(row: Dict[str, Any]) -> str:
        """The training-time prompt string, byte for byte."""
        rec = QARecord(id=row["id"], question=row["prompt"], context=row.get("context") or "",
                       answer=row.get("reference") or "", source="cidar", prompt_template="instruction")
        return _format_qa_prompt(rec)

    # ---- eval ----------------------------------------------------------------
    def _unsupported(self, tokenizer: BaseTokenizer, reason: str) -> Dict[str, Any]:
        logger.warning("%s: skipped — %s", TASK_NAME, reason)
        out: Dict[str, Any] = {k: None for k in METRIC_NAMES}
        out.update({"status": "generation_unsupported", "reason": reason, "num_samples": 0,
                    "embedding_type": tokenizer.embedding_type})
        return out

    def evaluate(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        split: str = "test",
        max_samples: Optional[int] = None,
        row_dump_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        supported, reason = generation_supported(tokenizer)
        if not supported:
            return self._unsupported(tokenizer, reason)

        rows = self.load_examples()
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        prompts = [self.build_prompt(r) for r in rows]
        references = [r["reference"] for r in rows]

        logger.info("%s: decoding config %s", TASK_NAME, self.decoding.to_json())
        cpt = measure_chars_per_token(tokenizer, references)
        token_cap = derive_token_cap(self.decoding, cpt)
        logger.info("%s: %d prompts, %.2f chars/token → token cap %d for %d chars (batch %d)",
                    TASK_NAME, len(rows), cpt, token_cap, self.decoding.max_output_chars, self.decoding.batch_size)

        t0 = time.perf_counter()
        gens = generate_freeform(model, tokenizer, prompts, self.decoding, token_cap)
        total_wall = time.perf_counter() - t0
        # The timed wall is the sum of the per-batch generate calls (each row carries its
        # batch's share); the untimed warm-up and the encode / decode bookkeeping between
        # batches stay out of gen_chars_per_sec, mirroring the tokenizer warm-up of the pipeline.
        gen_wall = sum(g.gen_time_sec for g in gens)
        logger.info("%s: generation wall %.1fs timed (%.1fs incl. warm-up and bookkeeping)", TASK_NAME, gen_wall, total_wall)

        records: List[Dict[str, Any]] = []
        for row, prompt, g in zip(rows, prompts, gens):
            ref = row["reference"]
            rt = M.roundtrip(tokenizer, ref) if ref else ""
            records.append({
                "id": row["id"], "stratum": row.get("stratum") or "", "instruction": row["prompt"],
                "context": row.get("context") or "", "prompt_text": prompt, "reference": ref,
                "generation": g.generation, "generation_raw": g.generation_raw,
                "prompt_tokens": g.prompt_tokens, "gen_tokens": g.gen_tokens, "gen_chars": len(g.generation),
                "ref_chars": len(ref), "stop_reason": g.stop_reason, "hit_cap": g.hit_cap, "hit_loop": g.hit_loop,
                "loop_rule": g.loop_rule, "loop_period": g.loop_period,
                "char_truncated": g.char_truncated, "empty": not g.generation.strip(),
                # On the raw text: a loop-stopped generation keeps only the first copy of the
                # repeated unit, so the flag has to look at what the model actually produced
                # for degenerate_rate to stay comparable with runs that had no loop stop.
                "degenerate": M.is_degenerate(g.generation_raw), "latin": contains_latin_letters(g.generation),
                "arabic_letter_ratio": M.arabic_letter_ratio(g.generation),
                "chrf": round(M.chrf_sentence(g.generation, ref), 4),
                "bertscore_p": None, "bertscore_r": None, "bertscore_f1": None,
                "reference_roundtrip_chrf": round(M.chrf_sentence(rt, ref), 4) if ref else None,
                "gen_time_sec": round(g.gen_time_sec, 4),
            })

        bertscore_info: Optional[Dict[str, Any]] = None
        if self.bertscore_model and records:
            try:
                from arabic_eval.tasks.freeform.bertscore import BertScorer
                scorer = BertScorer(self.bertscore_model, self.bertscore_layer, device=str(model.device),
                                    batch_size=self.bertscore_batch_size)
                P, R, F = scorer.score([r["generation"] for r in records], [r["reference"] for r in records])
                for r, p, rr, f in zip(records, P, R, F):
                    r["bertscore_p"], r["bertscore_r"], r["bertscore_f1"] = round(p, 4), round(rr, 4), round(f, 4)
                bertscore_info = {"model": self.bertscore_model, "layer": self.bertscore_layer}
                del scorer
            except Exception as e:  # noqa: BLE001 — the eval must not die on the scorer
                logger.warning("%s: BERTScore failed (%s: %s); reported as None", TASK_NAME, type(e).__name__, e)

        metrics = M.summarize(records, gen_wall)
        metrics.update({"chars_per_token": round(cpt, 4), "token_cap": token_cap,
                        "max_output_chars": self.decoding.max_output_chars, "status": "ok"})

        if row_dump_dir is not None:
            path = Path(row_dump_dir) / f"{TASK_NAME}.parquet"
            write_report_table(path, records, ROW_FIELDS, metadata={
                "task": TASK_NAME, "kind": "freeform_generations", "schema_version": 3,
                "heldout_path": str(self.heldout_path), "heldout_sha256": _sha256(self.heldout_path),
                "tokenizer_class": type(tokenizer).__name__, "embedding_type": tokenizer.embedding_type,
                "decoding": self.decoding.to_json(), "token_cap": token_cap, "chars_per_token": round(cpt, 4),
                "bertscore": bertscore_info, "prompt_template": "instruction", "num_rows": len(records),
            })
            logger.info("%s: wrote %d rows → %s", TASK_NAME, len(records), path)
        return metrics
