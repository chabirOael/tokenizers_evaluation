"""Free-form instruction-following eval on the held-out CIDAR set.

Eval-only task (``BaseTask``): for every row of the held-out JSONL the model
greedy-decodes an answer to the *exact* Phase 3 instruction prompt
(``_format_qa_prompt`` on an ``instruction`` record — the string the
mixture trained on, so generation continues where training left off), under
the shared decoding rules of ``generation.py`` (EOS / stop marker / text-level
repetition-loop stop / character-budget cap; ``stop_reason`` per row). Reference-based (chrF,
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
ROW_FIELDS = [
    "id", "stratum", "instruction", "context", "prompt_text", "reference", "generation", "generation_raw",
    "prompt_tokens", "gen_tokens", "gen_chars", "ref_chars", "stop_reason", "hit_cap", "hit_loop", "char_truncated",
    "empty", "degenerate", "latin", "arabic_letter_ratio", "chrf",
    "bertscore_p", "bertscore_r", "bertscore_f1", "reference_roundtrip_chrf", "gen_time_sec",
]
METRIC_NAMES = [
    "chrf", "chrf_corpus", "bertscore_f1", "bertscore_p", "bertscore_r",
    "empty_rate", "degenerate_rate", "latin_rate", "arabic_letter_ratio",
    "hit_cap_rate", "loop_stop_rate", "marker_stop_rate", "eos_rate", "char_truncated_rate",
    "mean_gen_chars", "mean_gen_tokens", "mean_ref_chars", "gen_chars_per_sec", "gen_tokens_per_sec",
    "generation_wall_sec", "reference_roundtrip_chrf", "chars_per_token", "token_cap", "num_samples",
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
    """Params (``sweep.tasks[].params``; unknown keys such as the injected
    ``num_fewshot`` are ignored):

    ``heldout_path`` — the JSONL (default the committed v1 set);
    ``max_output_chars`` 1200, ``max_prompt_tokens`` 512, ``batch_size`` 16,
    ``token_cap_margin`` 1.15, ``token_cap_floor`` 32, ``token_cap_ceiling``
    4096, ``stop_markers``, ``marker_check_every`` 16, ``seed`` 42 — see
    ``DecodingConfig``; ``bertscore_model`` (``null`` skips BERTScore),
    ``bertscore_layer`` 17, ``bertscore_batch_size`` 32.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.heldout_path = resolve_heldout_path(cfg.get("heldout_path") or DEFAULT_HELDOUT_PATH)
        self.decoding = DecodingConfig(
            max_output_chars=int(cfg.get("max_output_chars", 1200)),
            max_prompt_tokens=int(cfg.get("max_prompt_tokens", 512)),
            batch_size=int(cfg.get("batch_size", 16)),
            token_cap_margin=float(cfg.get("token_cap_margin", 1.15)),
            token_cap_floor=int(cfg.get("token_cap_floor", 32)),
            token_cap_ceiling=int(cfg.get("token_cap_ceiling", 4096)),
            stop_markers=tuple(cfg.get("stop_markers") or DEFAULT_STOP_MARKERS),
            marker_check_every=int(cfg.get("marker_check_every", 16)),
            loop_stop=bool(cfg.get("loop_stop", True)),
            seed=int(cfg.get("seed", 42)),
        )
        self.bertscore_model: Optional[str] = cfg.get("bertscore_model", "xlm-roberta-large")
        self.bertscore_layer = int(cfg.get("bertscore_layer", 17))
        self.bertscore_batch_size = int(cfg.get("bertscore_batch_size", 32))
        self._rows: Optional[List[Dict[str, Any]]] = None

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
        metrics.update({"chars_per_token": round(cpt, 4), "token_cap": token_cap, "status": "ok"})

        if row_dump_dir is not None:
            path = Path(row_dump_dir) / f"{TASK_NAME}.parquet"
            write_report_table(path, records, ROW_FIELDS, metadata={
                "task": TASK_NAME, "kind": "freeform_generations", "schema_version": 2,
                "heldout_path": str(self.heldout_path), "heldout_sha256": _sha256(self.heldout_path),
                "tokenizer_class": type(tokenizer).__name__, "embedding_type": tokenizer.embedding_type,
                "decoding": self.decoding.to_json(), "token_cap": token_cap, "chars_per_token": round(cpt, 4),
                "bertscore": bertscore_info, "prompt_template": "instruction", "num_rows": len(records),
            })
            logger.info("%s: wrote %d rows → %s", TASK_NAME, len(records), path)
        return metrics
