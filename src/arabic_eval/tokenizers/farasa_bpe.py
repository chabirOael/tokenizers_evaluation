"""Farasa-BPE — a byte-level BPE whose merges never cross a Farasa morpheme boundary (2026-11-08).

"Farasa-BPE-16K (ours, MorphBPE-style)": the arm that differs from ``bpe`` (BPE-16K)
in exactly one step. Same corpus, preprocessing, ``BpeTrainer`` settings
(``min_frequency 2``), special tokens, ByteLevel decoder and post-processor as
:class:`BPETokenizer`; what changes is the **training-time pre-tokenization**:

    Sequence([Split(JOINER, "removed"), ByteLevel(add_prefix_space=False)])

over training text that carries ``JOINER`` (U+2060, absent from the corpus — checked
at build time) at every Farasa boundary inside a word (``farasa_segmentation``:
Farasa per distinct word, aligned to the raw word, the raw text never changed). BPE
never merges across a pre-tokenization boundary, so no learned piece spans a Farasa
boundary of the training text; a word-initial morpheme keeps its ``Ġ`` form, the
following ones have none.

Two inference modes (``inference_segmentation``):

* ``none`` — **variant B, "training-time constraint"** (the default; MorphBPE-style):
  the saved ``tokenizer.json`` carries the plain ``ByteLevel`` pre-tokenizer; raw text
  is tokenized with the learned merges, no Farasa, no joiner; decode is exact
  (ByteLevel). Learned sub-morpheme merges may still chain across a boundary at
  inference — what the gate (``morpheme_integrity_rate`` ≥ 0.90) measures.
* ``farasa`` — **variant A, "Farasa at inference"** (the fallback): the training
  pre-tokenizer is kept and ``encode`` inserts joiners from Farasa's segmentation of
  the input (the corpus table first, Farasa for an unseen word). Decode is exact too
  (the joiner is removed by the pre-tokenizer and is never a piece).

The mode is saved in ``farasa_bpe.json`` next to ``tokenizer.json``; ``load`` restores
it, and a constructor value that disagrees with the saved one is an error — ``encode``
never changes mode silently.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from arabic_eval.params_spec import ParamSpec
from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers import farasa_segmentation as fs
from arabic_eval.tokenizers.base import TokenizerOutput
from arabic_eval.tokenizers.bpe import DEFAULT_MIN_FREQUENCY, SPECIAL_TOKENS, BPETokenizer

logger = logging.getLogger("arabic_eval.tokenizers.farasa_bpe")

MODES = ("none", "farasa")
DEFAULT_MODE = "none"
DEFAULT_CACHE_ROOT = "outputs/data_cache/farasa_segmented"
DEFAULT_WORKERS = 12
SIDE_FILE = "farasa_bpe.json"


def _training_pre_tokenizer(joiner: str):
    return pre_tokenizers.Sequence([pre_tokenizers.Split(joiner, behavior="removed"),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False)])


@tokenizer_registry.register("farasa_bpe")
class FarasaBPETokenizer(BPETokenizer):
    """BPE trained on Farasa-joined text; inference with (A) or without (B) Farasa."""

    def __init__(self, inference_segmentation: Optional[str] = None, cache_root: str = DEFAULT_CACHE_ROOT,
                 workers: int = DEFAULT_WORKERS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if inference_segmentation is not None and inference_segmentation not in MODES:
            raise ValueError(f"inference_segmentation must be one of {MODES}, got {inference_segmentation!r}")
        self._requested_mode = inference_segmentation
        self._mode = inference_segmentation or DEFAULT_MODE
        self._cache_root = cache_root
        self._workers = int(workers)
        self._joiner = fs.JOINER
        self._build: Dict[str, Any] = {}
        self._table: Optional[Dict[str, Tuple[int, ...]]] = None   # variant A: word → cuts (tables + memo)

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        return BPETokenizer.param_spec() + [
            ParamSpec("inference_segmentation", "str", DEFAULT_MODE, choices=MODES,
                      help="none = variant B (Farasa constrains training only; raw text at inference); "
                           "farasa = variant A (Farasa joiners inserted on every encode)."),
            ParamSpec("cache_root", "path", DEFAULT_CACHE_ROOT, advanced=True,
                      help="Root of the cached Farasa segmentation (scripts/farasa_segment_corpus.py); looked up by "
                           "the training texts' sha256 at train time."),
            ParamSpec("workers", "int", DEFAULT_WORKERS, min=1, advanced=True,
                      help="Farasa JVM worker processes when the segmentation has to be built inside train()."),
        ]

    # ------------------------------------------------------------------ build
    def _find_cache(self, digest: str) -> Optional[Path]:
        root = Path(self._cache_root)
        if not root.exists():
            return None
        versions = fs.farasa_versions()
        for man in sorted(root.glob("*/manifest.json")):
            m = json.loads(man.read_text(encoding="utf-8"))
            if (m.get("texts_sha256") == digest and m.get("segmenter_jar_sha256") == versions["segmenter_jar_sha256"]
                    and m.get("farasapy") == versions["farasapy"]):
                return man.parent
        return None

    def _build_table_inline(self, texts: List[str], digest: str) -> Path:
        versions = fs.farasa_versions()
        fp = fs.corpus_fingerprint(digest, None, versions)
        out = Path(self._cache_root) / fp
        out.mkdir(parents=True, exist_ok=True)
        counts = fs.collect_word_counts(texts)
        words = sorted(counts)
        farasa_out, seg_stats = fs.segment_words(words, workers=self._workers)
        table, stats = fs.build_boundary_table(counts, farasa_out, words)
        fs.save_table(out / "boundary_table.parquet", table)
        man = {"fingerprint": fp, "texts_sha256": digest, "num_texts": len(texts), **versions,
               "preprocessing": "unknown (built inside farasa_bpe.train)",
               "joiner": "U+2060", "stats": stats, "segmentation": seg_stats}
        (out / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    def train(self, texts: List[str], vocab_size: int, **kwargs: Any) -> None:
        min_frequency = kwargs.get("min_frequency", DEFAULT_MIN_FREQUENCY)
        t0 = time.time()
        digest = fs.texts_digest(texts)
        cache = self._find_cache(digest)
        if cache is None:
            logger.info("farasa_bpe: no cached segmentation for texts sha256 %s… — building it now", digest[:12])
            cache = self._build_table_inline(texts, digest)
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        self._joiner = fs.JOINER if manifest.get("joiner", "U+2060") == "U+2060" else fs.FALLBACK_JOINER
        table = fs.load_table(cache / "boundary_table.parquet")
        if any(self._joiner in t for t in texts[:1000]):
            raise ValueError("farasa_bpe: the joiner occurs in the training texts")
        logger.info("farasa_bpe: %d words with boundaries from %s", len(table), cache)

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = _training_pre_tokenizer(self._joiner)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                      special_tokens=SPECIAL_TOKENS, show_progress=True)
        t1 = time.time()
        joiner = self._joiner
        tokenizer.train_from_iterator((fs.insert_joiners(t, table, joiner) for t in texts), trainer=trainer,
                                      length=len(texts))
        t_bpe = time.time() - t1
        bos_id, eos_id = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s>:0 $A:0 </s>:0", pair="<s>:0 $A:0 </s>:0 <s>:1 $B:1 </s>:1",
            special_tokens=[("<s>", bos_id), ("</s>", eos_id)])
        joiner_bytes = pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(joiner)[0][0]
        joiner_pieces = [p for p in tokenizer.get_vocab() if joiner in p or joiner_bytes in p]
        self._tokenizer = tokenizer
        self._build_special_token_map()
        self._apply_mode()
        self._build = {
            "segmentation_cache": str(cache), "segmentation_fingerprint": manifest.get("fingerprint"),
            "joiner": manifest.get("joiner", "U+2060"), "farasapy": manifest.get("farasapy"),
            "segmenter_jar_sha256": manifest.get("segmenter_jar_sha256"),
            "segmentation_stats": manifest.get("stats"), "context_check": manifest.get("context_check"),
            "segmentation_wall_sec": manifest.get("wall_sec"),
            "bpe": {"vocab_size_arg": vocab_size, "min_frequency": min_frequency, "num_texts": len(texts),
                    "vocab_size": tokenizer.get_vocab_size(), "train_wall_sec": round(t_bpe, 1),
                    "pieces_containing_joiner": len(joiner_pieces)},
            "total_wall_sec": round(time.time() - t0, 1),
        }
        logger.info("farasa_bpe trained — %d pieces in %.0f s (mode %s)", self.vocab_size, t_bpe, self._mode)

    # ------------------------------------------------------------------ modes
    def _apply_mode(self) -> None:
        if self._tokenizer is None:
            return
        if self._mode == "none":
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        else:
            self._tokenizer.pre_tokenizer = _training_pre_tokenizer(self._joiner)
        self._tokenizer.decoder = decoders.ByteLevel()

    @property
    def inference_segmentation(self) -> str:
        return self._mode

    # Variant A: a word's cuts come from the per-word tables saved with the tokenizer
    # (``segment_tables/*.parquet``: every corpus word, plus the words of any text packed for
    # training, e.g. a pool), else from Farasa on the unknown words of the text in one batch.
    # Farasa segments a word on its own (verified in and out of context, 500 / 500 texts), so the
    # tables only save time; the cuts are those of ``farasa_segmentation.align_word`` either way.
    def _ensure_known(self) -> Dict[str, Tuple[int, ...]]:
        if self._table is None:
            self._table = {}
        return self._table

    def _segment_text(self, text: str) -> str:
        known = self._ensure_known()
        words = set(text.split())
        unknown = sorted(w for w in words if w not in known)
        if unknown:
            outs, _ = fs._seg_lines(unknown)
            for w, o in zip(unknown, outs):
                known[w] = tuple(fs.align_word(w, o)[0])
        return fs.insert_joiners(text, {w: known[w] for w in words if known[w]}, self._joiner)

    def build_segment_tables(self, path: Path | str, extra_words: Optional[List[str]] = None,
                             name: str = "extra", workers: int = DEFAULT_WORKERS) -> Dict[str, Any]:
        """Write ``segment_tables/corpus.parquet`` (every training-corpus word with its cuts, from the
        segmentation cache) and, with ``extra_words``, ``segment_tables/<name>.parquet`` (Farasa on the
        words the corpus lacks, ``workers`` JVMs). Pure speed-up for variant A; returns counts."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = Path(path)
        tdir = path / "segment_tables"
        tdir.mkdir(parents=True, exist_ok=True)
        stats: Dict[str, Any] = {}
        corpus_file = tdir / "corpus.parquet"
        known = self._ensure_known()
        if not corpus_file.exists():
            cache = Path(self._build["segmentation_cache"])
            raw = pq.read_table(cache / "farasa_raw.parquet", columns=["word", "farasa"]).to_pydict()
            cuts = [list(fs.align_word(w, o)[0]) for w, o in zip(raw["word"], raw["farasa"])]
            pq.write_table(pa.table({"word": raw["word"], "cuts": cuts}), corpus_file, compression="zstd")
            stats["corpus_words"] = len(raw["word"])
        for f in sorted(tdir.glob("*.parquet")):
            t = pq.read_table(f).to_pydict()
            known.update({w: tuple(c) for w, c in zip(t["word"], t["cuts"])})
        if extra_words:
            todo = sorted({w for w in extra_words if w not in known})
            outs, seg_stats = fs.segment_words(todo, workers=workers) if todo else ([], {})
            cuts = [list(fs.align_word(w, o)[0]) for w, o in zip(todo, outs)]
            pq.write_table(pa.table({"word": todo, "cuts": cuts}), tdir / f"{name}.parquet", compression="zstd")
            known.update({w: tuple(c) for w, c in zip(todo, cuts)})
            stats.update({f"{name}_words_requested": len(set(extra_words)), f"{name}_words_segmented": len(todo),
                          f"{name}_segmentation": seg_stats})
        stats["known_words"] = len(known)
        return stats

    def encode(self, text: str, max_length: Optional[int] = None, padding: bool = False,
               truncation: bool = False) -> TokenizerOutput:
        if self._mode == "farasa":
            text = self._segment_text(text)
        return super().encode(text, max_length=max_length, padding=padding, truncation=truncation)

    # ------------------------------------------------------------------ io
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # tokenizer.json always carries the variant-B pre-tokenizer; the mode lives in the side file.
        saved_pt = self._tokenizer.pre_tokenizer
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        try:
            self._tokenizer.save(str(path / "tokenizer.json"))
        finally:
            self._tokenizer.pre_tokenizer = saved_pt
        side = {"inference_segmentation": self._mode, "joiner": "U+2060" if self._joiner == fs.JOINER else "U+E000",
                "build": self._build}
        (path / SIDE_FILE).write_text(json.dumps(side, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def load(self, path: Path | str) -> None:
        path = Path(path)
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        side = json.loads((path / SIDE_FILE).read_text(encoding="utf-8")) if (path / SIDE_FILE).exists() else {}
        saved = side.get("inference_segmentation", DEFAULT_MODE)
        if self._requested_mode is not None and self._requested_mode != saved:
            raise ValueError(f"farasa_bpe: {path} was saved with inference_segmentation={saved!r}, the config asks "
                             f"{self._requested_mode!r} — re-save the tokenizer in that mode instead of switching silently")
        self._mode = saved
        self._joiner = fs.JOINER if side.get("joiner", "U+2060") == "U+2060" else fs.FALLBACK_JOINER
        self._build = side.get("build", {})
        if self._mode == "farasa":
            known = self._ensure_known()
            import pyarrow.parquet as pq
            for f in sorted((path / "segment_tables").glob("*.parquet")):
                t = pq.read_table(f).to_pydict()
                known.update({w: tuple(c) for w, c in zip(t["word"], t["cuts"])})
        self._build_special_token_map()
        self._apply_mode()

    def set_inference_segmentation(self, mode: str) -> None:
        """Explicit mode switch (used to re-save the same vocabulary as variant A)."""
        if mode not in MODES:
            raise ValueError(mode)
        self._mode = mode
        self._requested_mode = mode
        self._apply_mode()
