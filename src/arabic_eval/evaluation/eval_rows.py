"""Per-eval-row Parquet dump — every scored benchmark row, exactly as the model saw it.

The LightEval MCQ loop scores one ``(context, continuation)`` pair per choice
and then throws everything away except an aggregate accuracy and (optionally)
a wrong-answer report. This module keeps the whole thing: for every eval row,
the **exact prompt string** that was fed to the scorer, the continuations that
were scored against it, every per-choice log-likelihood and normalised score,
the gold and predicted answers, and the diagnostics that say whether the
decision was real or an artifact of truncation.

Why Parquet rather than CSV or JSONL (measured on 8 220 real rows, extrapolated
to the 54 333 rows of one sweep cell across the four benchmarks):

    parquet+zstd   17.7 MB/cell     filter columns load in   2.2 ms
    parquet+snappy 26.1 MB/cell
    csv (padded)   67.5 MB/cell
    jsonl          99.7 MB/cell     full scan in           102.7 ms

Column projection makes a sidecar index unnecessary: reading the dozen small
filter columns of a 21 144-row task costs a few milliseconds, and a page of 50
filtered rows touches one row group. Per-choice data is stored in native list
columns, so the ragged 2-to-5-way shape of AlGhafa needs no padding and no
``max_choices`` pre-scan.

One property to know: **a Parquet file is not readable until its footer is
written at close**. While a task is being evaluated its ``.parquet`` does not
exist yet as far as a reader is concerned, so the writer also maintains a tiny
``<task>.progress.json`` (rows done, running accuracy) that a live view can
poll. It is removed when the file closes cleanly.

``pyarrow`` is imported lazily so tokenizer-only workflows (and the console at
startup) do not need it.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("arabic_eval.evaluation.eval_rows")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: 2 (2026-09-24): ``cont_tokens`` per choice, and ``hit_cap`` also set when the
#: cap reached a continuation's full encoding (1: the prompt's only). Readers
#: take both versions — a missing column is simply absent from the page.
SCHEMA_VERSION = 2

#: Value ``_compute_loglikelihood`` returns when truncation left no room for
#: the continuation (``full_len <= ctx_len``). Every choice scoring this means
#: the row's argmax is an artifact of the ``max_length`` cap, not a decision.
#: Single source of truth — ``tasks.lighteval.base`` imports it.
SENTINEL_LL = -1e9

#: A decision whose top-1 minus top-2 score gap is below this is effectively a
#: coin flip. Keyed off the *decision* margin, never off ``margin`` (which is
#: pred-minus-gold and therefore exactly 0 on every correct row).
NEAR_TIE_MARGIN = 0.01

DEFAULT_ROW_GROUP_SIZE = 2000
PROGRESS_EVERY = 500
PARQUET_COMPRESSION = "zstd"
METADATA_KEY = b"arabic_eval"
# Superseded cells are *moved*, never deleted, into ``<experiment>/_superseded/`` (a
# README there names the rule set and budget of each). The console browsers walk
# ``outputs/experiments`` recursively and skip anything under such a folder, so a
# superseded cell is not listed as a live one; the one-level cell discoveries (the
# judge, the rating tab, ``compare_results.py``) never see it in the first place.
SUPERSEDED_DIR = "_superseded"

#: What one unit of ``prompt_units`` means, per embedding family. Recorded in
#: the file metadata so a reader can label the column honestly: a "token" is a
#: whole word for CharacterBERT and a single byte for Charformer.
UNIT_BY_EMBEDDING: Dict[str, str] = {
    "standard": "tokens",
    "character_cnn": "words",
    "char_jaber": "chars",
    "charformer": "bytes",
}

#: Small, fixed-width columns — everything the browser needs to filter,
#: count and paginate without touching the text.
FILTER_COLUMNS: Sequence[str] = (
    "row_index", "source_config", "n_choices",
    "gold_idx", "pred_idx", "pred_idx_char", "pred_idx_pmi",
    "correct", "correct_char", "correct_pmi",
    "margin", "margin_ll", "decision_margin", "prompt_units",
    "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree",
)

#: The boolean per-row flags ``select`` accepts. Deliberately a closed set:
#: the index also holds availability booleans (``has_correct_pmi``) that are
#: not row masks and must not be usable as filters.
FLAG_COLUMNS: Sequence[str] = (
    "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree",
)

#: Columns a table page renders (the text lives here).
PAGE_COLUMNS: Sequence[str] = (
    "row_index", "source_config", "prompt", "question", "context",
    "choices", "continuations", "gold_idx", "gold_text",
    "pred_idx", "pred_text", "pred_idx_char", "pred_idx_pmi",
    "correct", "correct_char", "correct_pmi",
    "ll", "score_char", "score_pmi", "uncond_ll", "cont_tokens",
    "margin", "margin_ll", "decision_margin", "prompt_units", "n_choices",
    "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree",
)


def _require_pyarrow():
    """Import pyarrow on demand, with an actionable message when it is absent."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "pyarrow is required to read/write eval-row dumps. "
            "Install it with `pip install pyarrow` (it already ships as a "
            "dependency of `datasets`)."
        ) from e
    return pa, pq


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def eval_row_schema(metadata: Optional[Dict[str, Any]] = None):
    """The Parquet schema of one eval-row dump.

    ``metadata`` is stored as a single JSON blob under the ``arabic_eval``
    key: task, tokenizer, model, scoring mode, ``max_length``, few-shot K and
    so on. Row counts and accuracy are deliberately *not* stored — both are
    derivable from the file itself (``metadata.num_rows``, and the correctness
    columns), so nothing can go stale.
    """
    pa, _ = _require_pyarrow()
    dict_str = pa.dictionary(pa.int32(), pa.string())
    fields = [
        # identity
        pa.field("row_index", pa.int32(), nullable=False),
        pa.field("source_config", dict_str),
        # what the model received, verbatim
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("continuations", pa.list_(pa.string()), nullable=False),
        # the raw row behind the prompt
        pa.field("question", pa.string()),
        pa.field("context", pa.string()),
        pa.field("choices", pa.list_(pa.string())),
        # expected vs received
        pa.field("gold_idx", pa.int8(), nullable=False),
        pa.field("gold_text", pa.string()),
        pa.field("pred_idx", pa.int8(), nullable=False),
        pa.field("pred_text", pa.string()),
        pa.field("pred_idx_char", pa.int8()),
        pa.field("pred_idx_pmi", pa.int8()),
        pa.field("correct", pa.bool_(), nullable=False),
        pa.field("correct_char", pa.bool_()),
        pa.field("correct_pmi", pa.bool_()),
        # scores, one entry per choice
        pa.field("ll", pa.list_(pa.float32()), nullable=False),
        pa.field("score_char", pa.list_(pa.float32())),
        pa.field("score_pmi", pa.list_(pa.float32())),
        pa.field("uncond_ll", pa.list_(pa.float32())),
        # Continuation tokens the scorer summed, per choice (0 = sentinel):
        # the audit of the scoring window (schema 2). Null when unknown.
        pa.field("cont_tokens", pa.list_(pa.int16())),
        pa.field("margin", pa.float32()),
        pa.field("margin_ll", pa.float32()),
        pa.field("decision_margin", pa.float32()),
        # diagnostics
        # Length of the prompt as the scorer encoded it, i.e. *after* the
        # ``max_length`` cap — it never exceeds the cap, so ``hit_cap`` says
        # "reached it", not by how much.
        pa.field("prompt_units", pa.int32()),
        pa.field("n_choices", pa.int8(), nullable=False),
        pa.field("sentinel", pa.bool_(), nullable=False),
        pa.field("all_sentinel", pa.bool_(), nullable=False),
        pa.field("hit_cap", pa.bool_(), nullable=False),
        pa.field("near_tie", pa.bool_(), nullable=False),
        pa.field("disagree", pa.bool_(), nullable=False),
    ]
    meta = None
    if metadata is not None:
        meta = {METADATA_KEY: json.dumps(metadata, ensure_ascii=False).encode("utf-8")}
    return pa.schema(fields, metadata=meta)


# --------------------------------------------------------------------------
# Record construction
# --------------------------------------------------------------------------

def build_row_record(
    *,
    row_index: int,
    example: Dict[str, Any],
    prompt: str,
    continuations: Sequence[str],
    log_likelihoods: Sequence[float],
    scores_char: Optional[Sequence[float]],
    scores_pmi: Optional[Sequence[float]],
    unconditioned_log_likelihoods: Optional[Sequence[float]],
    gold_idx: int,
    pred_idx: int,
    pred_idx_char: Optional[int],
    pred_idx_pmi: Optional[int],
    prompt_units: Optional[int] = None,
    max_length: Optional[int] = None,
    cont_tokens: Optional[Sequence[int]] = None,
    cont_truncated: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Build one dump record.

    Pure and pyarrow-free, so it is cheap (it runs 54 333 times per cell) and
    testable on its own. ``prompt`` must be the string the scorer actually
    concatenated each continuation onto — i.e. the return of
    ``_format_eval_context_with_fewshot`` — otherwise the dump lies about what
    the model received.

    Two margins are recorded and they answer different questions. ``margin``
    is ``score[pred] - score[gold]``: how far the model landed from the right
    answer, and therefore 0 whenever it was right (the convention the failure
    reports already use). ``decision_margin`` is ``top1 - top2``: how sure the
    decision was, regardless of correctness. ``near_tie`` keys off the latter.

    ``cont_tokens`` / ``cont_truncated`` are the scorer's per-choice window
    (``ScoredLogLikelihood.n_tokens`` / ``.truncated``): the first is stored,
    the second only feeds ``hit_cap`` — the cap touched the row when the
    prompt reached it *or* when it reached a continuation's full encoding.
    """
    conts = [str(c) for c in continuations]
    lls = [float(v) for v in log_likelihoods]
    n = len(conts)
    primary = list(scores_pmi) if scores_pmi is not None else (
        list(scores_char) if scores_char is not None else lls
    )
    in_range = 0 <= gold_idx < n
    margin = (
        float(primary[pred_idx]) - float(primary[gold_idx])
        if in_range and 0 <= pred_idx < n else None
    )
    margin_ll = (
        lls[pred_idx] - lls[gold_idx]
        if in_range and 0 <= pred_idx < n else None
    )
    ordered = sorted(primary, reverse=True)
    decision_margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else None
    sentinel_flags = [v <= SENTINEL_LL for v in lls]
    return {
        "row_index": int(row_index),
        "source_config": example.get("_source_config", "_default"),
        "prompt": prompt,
        "continuations": conts,
        "question": str(example.get("question", "")),
        "context": str(example.get("context", "") or ""),
        "choices": [str(c) for c in (example.get("choices") or [])],
        "gold_idx": int(gold_idx),
        "gold_text": conts[gold_idx].lstrip() if in_range else "",
        "pred_idx": int(pred_idx),
        "pred_text": conts[pred_idx].lstrip() if 0 <= pred_idx < n else "",
        "pred_idx_char": None if pred_idx_char is None else int(pred_idx_char),
        "pred_idx_pmi": None if pred_idx_pmi is None else int(pred_idx_pmi),
        "correct": bool(pred_idx == gold_idx),
        "correct_char": None if pred_idx_char is None else bool(pred_idx_char == gold_idx),
        "correct_pmi": None if pred_idx_pmi is None else bool(pred_idx_pmi == gold_idx),
        "ll": lls,
        "score_char": None if scores_char is None else [float(v) for v in scores_char],
        "score_pmi": None if scores_pmi is None else [float(v) for v in scores_pmi],
        "uncond_ll": (
            None if unconditioned_log_likelihoods is None
            else [float(v) for v in unconditioned_log_likelihoods]
        ),
        "cont_tokens": None if cont_tokens is None else [int(v) for v in cont_tokens],
        "margin": margin,
        "margin_ll": margin_ll,
        "decision_margin": decision_margin,
        "prompt_units": None if prompt_units is None else int(prompt_units),
        "n_choices": n,
        "sentinel": any(sentinel_flags),
        "all_sentinel": bool(sentinel_flags) and all(sentinel_flags),
        "hit_cap": bool(
            (prompt_units is not None and max_length is not None
             and prompt_units >= max_length)
            or (cont_truncated is not None and any(cont_truncated))
        ),
        "near_tie": decision_margin is not None and decision_margin < NEAR_TIE_MARGIN,
        "disagree": (
            pred_idx_char is not None and pred_idx_pmi is not None
            and pred_idx_char != pred_idx_pmi
        ),
    }


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

class EvalRowWriter:
    """Streaming Parquet writer for eval rows.

    Buffers ``row_group_size`` records, flushes them as one row group, and
    keeps a ``<stem>.progress.json`` up to date so a live view has something
    to show before the footer exists. Usable as a context manager; ``close``
    is idempotent.

    An eval pass that produced no rows still leaves a valid, empty Parquet
    file, mirroring the header-only CSV guarantee the old reports gave.
    """

    def __init__(
        self,
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        compression: str = PARQUET_COMPRESSION,
        progress_every: int = PROGRESS_EVERY,
    ) -> None:
        pa, pq = _require_pyarrow()
        self._pa = pa
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(metadata or {})
        meta.setdefault("schema_version", SCHEMA_VERSION)
        meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.metadata = meta
        self.schema = eval_row_schema(meta)
        self.row_group_size = max(1, int(row_group_size))
        self.progress_every = max(1, int(progress_every))
        self.progress_path = self.path.with_suffix(".progress.json")
        self._writer = pq.ParquetWriter(self.path, self.schema, compression=compression)
        self._buffer: List[Dict[str, Any]] = []
        self._n = 0
        self._n_correct = 0
        self._started = time.time()
        self._closed = False
        self._write_progress(done=False)

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "EvalRowWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- api ------------------------------------------------------------
    def write(self, record: Dict[str, Any]) -> None:
        """Append one record (the dict ``build_row_record`` returns)."""
        self._buffer.append(record)
        self._n += 1
        if record.get("correct"):
            self._n_correct += 1
        if len(self._buffer) >= self.row_group_size:
            self._flush()
        if self._n % self.progress_every == 0:
            self._write_progress(done=False)

    @property
    def n_rows(self) -> int:
        return self._n

    def close(self) -> None:
        """Flush the tail, write the footer, drop the progress file."""
        if self._closed:
            return
        self._closed = True
        try:
            self._flush()
        finally:
            self._writer.close()
        try:
            self.progress_path.unlink()
        except OSError:
            pass
        logger.info(
            "eval rows: wrote %d rows to %s (%.1f KB)",
            self._n, self.path,
            self.path.stat().st_size / 1024 if self.path.exists() else 0.0,
        )

    # -- internals ------------------------------------------------------
    def _flush(self) -> None:
        if not self._buffer:
            return
        table = self._pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self._buffer.clear()
        if not self._closed:
            self._write_progress(done=False)

    def _write_progress(self, done: bool) -> None:
        payload = {
            "rows": self._n,
            "correct": self._n_correct,
            "done": done,
            "started_at": self._started,
            "updated_at": time.time(),
            "task": self.metadata.get("task"),
        }
        try:
            self.progress_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:  # pragma: no cover - disk-full / permissions
            logger.debug("could not write %s", self.progress_path)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------

def is_superseded(path: str | Path, base: str | Path) -> bool:
    """True when ``path`` lies under a ``SUPERSEDED_DIR`` folder below ``base``."""
    try:
        parts = Path(path).resolve().relative_to(Path(base).resolve()).parts
    except ValueError:
        parts = Path(path).parts
    return SUPERSEDED_DIR in parts


def read_progress(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read the ``<stem>.progress.json`` of an in-flight dump, if any.

    Present only while a task is being evaluated (the writer removes it on a
    clean close), so its presence is how a live view distinguishes "still
    running" from "crashed before the footer was written".
    """
    p = Path(path)
    p = p if p.name.endswith(".progress.json") else p.with_suffix(".progress.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class EvalRowFile:
    """Read-side of a dump: metadata, filtering, aggregates, paging.

    The small filter columns are read once and cached (a few milliseconds for
    a 21 144-row task); text columns are read only when a search needs them;
    a page reads just the row groups its rows land in. Nothing is loaded
    eagerly, so opening a file to show its header costs one footer read.
    """

    def __init__(self, path: str | Path) -> None:
        _, pq = _require_pyarrow()
        self.path = Path(path)
        self._pf = pq.ParquetFile(self.path)
        self._index: Optional[Dict[str, Any]] = None
        self._text: Dict[bool, List[str]] = {}
        self._rg_offsets: Optional[List[int]] = None

    # -- header ---------------------------------------------------------
    @property
    def metadata(self) -> Dict[str, Any]:
        raw = (self._pf.schema_arrow.metadata or {}).get(METADATA_KEY)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}

    @property
    def n_rows(self) -> int:
        return int(self._pf.metadata.num_rows)

    @property
    def columns(self) -> List[str]:
        return list(self._pf.schema_arrow.names)

    # -- filter index ---------------------------------------------------
    def index(self) -> Dict[str, Any]:
        """Load (once) the fixed-width columns every filter works on."""
        if self._index is not None:
            return self._index
        import numpy as np

        have = set(self.columns)
        cols = [c for c in FILTER_COLUMNS if c in have]
        tbl = self._pf.read(columns=cols)
        n = tbl.num_rows
        idx: Dict[str, Any] = {"n": n}

        def _bool(name: str):
            if name not in have:
                return None, False
            col = tbl.column(name)
            available = col.null_count < n
            return col.fill_null(False).to_numpy(zero_copy_only=False).astype(bool), available

        def _int(name: str, fill: int = -1):
            if name not in have:
                return None
            return (
                tbl.column(name).fill_null(fill)
                .to_numpy(zero_copy_only=False).astype("int32")
            )

        def _float(name: str):
            if name not in have:
                return None
            return (
                tbl.column(name).cast("float64").fill_null(float("nan"))
                .to_numpy(zero_copy_only=False)
            )

        idx["row_index"] = _int("row_index")
        idx["source_config"] = np.array(
            tbl.column("source_config").to_pylist() if "source_config" in have
            else ["_default"] * n,
            dtype=object,
        )
        for name in ("gold_idx", "pred_idx", "pred_idx_char", "pred_idx_pmi",
                     "n_choices", "prompt_units"):
            idx[name] = _int(name)
        for name in ("margin", "margin_ll", "decision_margin"):
            idx[name] = _float(name)
        for name in ("correct", "correct_char", "correct_pmi",
                     "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree"):
            arr, available = _bool(name)
            idx[name] = arr
            idx[f"has_{name}"] = available
        self._index = idx
        return idx

    def _text_column(self, include_prompt: bool) -> List[str]:
        """Lower-cased haystack per row for the free-text filter (cached)."""
        key = bool(include_prompt)
        if key in self._text:
            return self._text[key]
        cols = ["question"]
        if include_prompt:
            cols.append("prompt")
        cols = [c for c in cols if c in self.columns]
        tbl = self._pf.read(columns=cols)
        parts = [tbl.column(c).to_pylist() for c in cols]
        hay = [
            " ".join(p[i] or "" for p in parts).lower()
            for i in range(tbl.num_rows)
        ]
        self._text[key] = hay
        return hay

    # -- selection ------------------------------------------------------
    def select(
        self,
        outcome: str = "all",
        scoring: str = "primary",
        source_configs: Optional[Sequence[str]] = None,
        flags: Optional[Dict[str, bool]] = None,
        gold_idx: Optional[int] = None,
        pred_idx: Optional[int] = None,
        query: str = "",
        search_prompt: bool = False,
        sort: str = "row",
    ):
        """Return the row positions matching the filters, in display order."""
        import numpy as np

        idx = self.index()
        n = idx["n"]
        mask = np.ones(n, dtype=bool)

        correct = self._correct_column(scoring)
        if outcome == "correct":
            mask &= correct
        elif outcome == "wrong":
            mask &= ~correct
        elif outcome not in ("all", ""):
            raise ValueError(f"unknown outcome {outcome!r}")

        if source_configs:
            mask &= np.isin(idx["source_config"], np.array(list(source_configs), dtype=object))
        for name, want in (flags or {}).items():
            arr = idx.get(name) if name in FLAG_COLUMNS else None
            if arr is None:
                raise ValueError(
                    f"unknown flag {name!r} (known: {', '.join(FLAG_COLUMNS)})"
                )
            mask &= arr if want else ~arr
        if gold_idx is not None and idx.get("gold_idx") is not None:
            mask &= idx["gold_idx"] == int(gold_idx)
        if pred_idx is not None:
            pred = self._pred_column(scoring)
            if pred is not None:
                mask &= pred == int(pred_idx)
        q = (query or "").strip().lower()
        if q:
            hay = self._text_column(search_prompt)
            mask &= np.fromiter((q in h for h in hay), dtype=bool, count=n)

        positions = np.nonzero(mask)[0]
        if sort == "row" or not len(positions):
            return positions
        def _vals(name: str):
            arr = idx.get(name)
            if arr is None:
                return None
            v = arr[positions]
            return np.where(np.isnan(v), 0.0, v)

        if sort in ("margin_asc", "margin_desc"):
            vals = _vals("margin")
            if vals is None:
                return positions
            order = np.argsort(vals if sort == "margin_asc" else -vals, kind="stable")
        elif sort in ("uncertain", "confident"):
            # Decision margin: how close the top two scores were. This is the
            # coin-flip axis; ``margin`` cannot answer it (it is 0 when correct).
            vals = _vals("decision_margin")
            if vals is None:
                return positions
            order = np.argsort(vals if sort == "uncertain" else -vals, kind="stable")
        else:
            raise ValueError(f"unknown sort {sort!r}")
        return positions[order]

    def _correct_column(self, scoring: str):
        idx = self.index()
        if scoring in ("char", "pmi"):
            arr = idx.get(f"correct_{scoring}")
            if arr is not None and idx.get(f"has_correct_{scoring}"):
                return arr
            raise ValueError(
                f"this dump has no {scoring}-normalised scores "
                f"(score_normalization={self.metadata.get('score_normalization')!r})"
            )
        return idx["correct"]

    def _pred_column(self, scoring: str):
        idx = self.index()
        if scoring in ("char", "pmi"):
            return idx.get(f"pred_idx_{scoring}")
        return idx.get("pred_idx")

    # -- aggregates -----------------------------------------------------
    def summary(self, positions=None, scoring: str = "primary") -> Dict[str, Any]:
        """Counts, accuracy and the prediction histogram over ``positions``.

        ``accuracy`` is measured under ``scoring`` (the same column the
        selection was filtered on); ``accuracy_char`` and ``accuracy_pmi``
        report the same rows under each normalization.

        The histogram is the cheap way to see class collapse: a model that
        answers the same slot on most rows is riding a prior, not reading the
        question.
        """
        import numpy as np

        idx = self.index()
        pos = np.arange(idx["n"]) if positions is None else np.asarray(positions)
        out: Dict[str, Any] = {"n_rows": int(len(pos))}
        if not len(pos):
            return {**out, "accuracy": None, "pred_hist": [], "gold_hist": [],
                    "sentinel": 0, "all_sentinel": 0, "hit_cap": 0,
                    "near_tie": 0, "disagree": 0}

        # The headline follows the scoring the caller selected, so it always
        # agrees with the selection: filtering to "wrong under char-norm" and
        # then reading a non-zero accuracy would be nonsense. The named
        # breakdowns stay available for comparing the two.
        out["accuracy"] = round(float(self._correct_column(scoring)[pos].mean()), 6)
        for mode in ("char", "pmi"):
            try:
                col = self._correct_column(mode)
            except ValueError:
                continue
            out[f"accuracy_{mode}"] = round(float(col[pos].mean()), 6)

        max_choices = int(np.nanmax(idx["n_choices"][pos])) if idx.get("n_choices") is not None else 0
        pred = self._pred_column(scoring)
        gold = idx.get("gold_idx")
        out["pred_hist"] = (
            [int((pred[pos] == k).sum()) for k in range(max_choices)]
            if pred is not None else []
        )
        out["gold_hist"] = (
            [int((gold[pos] == k).sum()) for k in range(max_choices)]
            if gold is not None else []
        )
        for name in ("sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree"):
            arr = idx.get(name)
            out[name] = int(arr[pos].sum()) if arr is not None else 0
        for name, key in (("margin", "margin_median"),
                          ("decision_margin", "decision_margin_median")):
            arr = idx.get(name)
            if arr is None:
                continue
            vals = arr[pos]
            vals = vals[~np.isnan(vals)]
            out[key] = round(float(np.median(vals)), 6) if len(vals) else None
        units = idx.get("prompt_units")
        if units is not None and (units >= 0).any():
            u = units[pos]
            u = u[u >= 0]
            if len(u):
                out["prompt_units_mean"] = int(u.mean())
                out["prompt_units_max"] = int(u.max())
        return out

    def source_configs(self) -> List[str]:
        """Sub-configs present in the file, with row counts, most common first."""
        import numpy as np

        arr = self.index()["source_config"]
        names, counts = np.unique(arr, return_counts=True)
        pairs = sorted(zip(names.tolist(), counts.tolist()), key=lambda kv: (-kv[1], kv[0]))
        return [{"name": k, "n_rows": int(v)} for k, v in pairs]

    # -- paging ---------------------------------------------------------
    def _row_group_offsets(self) -> List[int]:
        if self._rg_offsets is None:
            offs, acc = [], 0
            for i in range(self._pf.num_row_groups):
                offs.append(acc)
                acc += self._pf.metadata.row_group(i).num_rows
            self._rg_offsets = offs
        return self._rg_offsets

    def rows(self, positions, columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Materialise the given row positions, in the order given.

        Only the row groups the positions fall into are read, so a page of 50
        costs one or two row-group reads however deep into the file it sits.
        """
        pa, _ = _require_pyarrow()
        import bisect

        positions = [int(p) for p in positions]
        if not positions:
            return []
        n = self.n_rows
        bad = [p for p in positions if not (0 <= p < n)]
        if bad:
            raise ValueError(
                f"row position(s) outside the file (0..{n - 1}): {bad[:5]}"
            )
        have = set(self.columns)
        cols = [c for c in (columns or PAGE_COLUMNS) if c in have]
        offs = self._row_group_offsets()
        by_group: Dict[int, List[int]] = {}
        for p in positions:
            g = bisect.bisect_right(offs, p) - 1
            by_group.setdefault(g, []).append(p)
        materialised: Dict[int, Dict[str, Any]] = {}
        for g, group_positions in by_group.items():
            tbl = self._pf.read_row_group(g, columns=cols)
            local = pa.array([p - offs[g] for p in group_positions], type=pa.int32())
            for pos, rec in zip(group_positions, tbl.take(local).to_pylist()):
                rec["position"] = pos
                materialised[pos] = rec
        return [materialised[p] for p in positions if p in materialised]

    def describe(self) -> Dict[str, Any]:
        """Header for a picker: metadata, row count, sub-configs, aggregates."""
        return {
            "path": str(self.path),
            "n_rows": self.n_rows,
            "metadata": self.metadata,
            "source_configs": self.source_configs(),
            "summary": self.summary(),
        }
