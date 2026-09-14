"""Stage A — the tokenizer-independent document pool.

``build_pool`` streams each source in ``dedup.priority`` order and, per
document, applies (cheap first):

    normalize → [wikipedia: strip tail sections] → [strip Latin parentheticals]
    → quality rules →
    dialect-marker gate → [opt-in CAMeL dialect ID] → MinHash near-dup →
    paragraph dedup (→ re-check min_words) → truncate to max_words

MinHash runs on the quality-passed text *before* paragraph stripping so a
lightly edited copy (one word changed per paragraph) is judged as a whole
document; the exact-paragraph pass then removes verbatim repeats —
boilerplate and verbatim article copies alike (a web page that is
nothing but a Wikipedia article ends as ``too_short_after_dedup``).
Two pages sharing a disclaimer paragraph are not near-duplicates (their
Jaccard is dominated by their own content once the short-line and
min-words rules have run), so this order does not merge them.

until the source has ``share × pool.total_words`` kept words. Output:

    <cache_dir>/<fingerprint>/
        manifest.json            counts per source per stage + pinned revisions
        <source>.parquet         id, source, text, n_words, url
        dropped_samples.jsonl    ≤ SAMPLE_PER_REASON docs per (source, reason)
        dropped_dialect.csv      every dialect-gate drop with its markers

The fingerprint hashes only the Stage A fields of ``PretrainingMixConfig``
(``STAGE_A_FIELDS``) so changing ``block_size`` / ``token_budget`` /
``consume_sequentially`` never invalidates a cached pool.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ...config import PretrainingMixConfig
from .dedup import MinHashDeduper, ParagraphDeduper
from .filters import (
    count_words,
    dialect_marker_score,
    normalize_document,
    quality_reason,
    strip_latin_parentheticals,
    strip_wikipedia_sections,
    truncate_words,
)
from .sources import BaseSource, make_source

logger = logging.getLogger(__name__)

STAGE_A_FIELDS = frozenset({"seed", "normalization", "sources", "pool", "dedup", "msa_filter", "quality"})
MANIFEST_NAME = "manifest.json"
SAMPLE_PER_REASON = 30
DIALECT_CSV_FIELDS = ["source", "id", "n_words", "n_markers", "per_1k_words", "top_markers", "snippet"]


# --------------------------------------------------------------------------
# Fingerprint / paths
# --------------------------------------------------------------------------

def stage_a_config(cfg: PretrainingMixConfig) -> Dict[str, Any]:
    return cfg.model_dump(include=set(STAGE_A_FIELDS), mode="json")


def pool_fingerprint(cfg: PretrainingMixConfig) -> str:
    payload = json.dumps(stage_a_config(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pool_dir(cfg: PretrainingMixConfig) -> Path:
    return Path(cfg.cache_dir) / pool_fingerprint(cfg)


def pool_source_path(directory: Path, source_name: str) -> Path:
    return Path(directory) / f"{source_name}.parquet"


def load_pool_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    p = Path(directory) / MANIFEST_NAME
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pool_is_complete(directory: Path) -> bool:
    m = load_pool_manifest(directory)
    return bool(m and m.get("complete"))


def iter_pool_docs(directory: Path, source_name: str, batch_rows: int = 1024) -> Iterator[Tuple[str, str]]:
    """Yield ``(id, text)`` in on-disk order for one source of a built pool."""
    import pyarrow.parquet as pq

    path = pool_source_path(directory, source_name)
    if not path.exists():
        raise FileNotFoundError(f"pool source {source_name!r} missing at {path}")
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["id", "text"]):
        ids = batch.column("id").to_pylist()
        texts = batch.column("text").to_pylist()
        for i, t in zip(ids, texts):
            yield i, t


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

@dataclass
class SourceBuildStats:
    name: str
    target_words: int
    streamed: int = 0
    kept: int = 0
    kept_words: int = 0
    dropped: Counter = field(default_factory=Counter)
    paragraphs_removed: int = 0
    docs_with_paragraphs_removed: int = 0
    docs_truncated: int = 0
    wiki_sections_cut: int = 0
    latin_parentheticals_removed: int = 0
    source_prefilter_skipped: Dict[str, int] = field(default_factory=dict)
    reached_target: bool = False
    stop_reason: str = ""
    wall_sec: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dropped"] = dict(sorted(self.dropped.items()))
        d["dropped_total"] = sum(self.dropped.values())
        d["keep_rate"] = (self.kept / self.streamed) if self.streamed else 0.0
        return d


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

class _ParquetSink:
    def __init__(self, path: Path, batch_rows: int = 512) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self._schema = pa.schema([
            ("id", pa.string()), ("source", pa.string()), ("text", pa.string()),
            ("n_words", pa.int32()), ("url", pa.string()),
        ])
        self._writer = pq.ParquetWriter(str(path), self._schema, compression="zstd")
        self._rows: List[Dict[str, Any]] = []
        self._batch_rows = batch_rows

    def write(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._batch_rows:
            self.flush()

    def flush(self) -> None:
        if self._rows:
            table = self._pa.Table.from_pylist(self._rows, schema=self._schema)
            self._writer.write_table(table)
            self._rows = []

    def close(self) -> None:
        self.flush()
        self._writer.close()


def build_pool(
    cfg: PretrainingMixConfig,
    *,
    out_dir: Optional[Path] = None,
    force: bool = False,
    doc_limit_per_source: Optional[int] = None,
    stop_at_target: bool = True,
    did_scorer=None,
    sources_override: Optional[Dict[str, BaseSource]] = None,
    log_every: int = 2000,
) -> Path:
    """Build (or reuse) the pool. Returns its directory.

    ``doc_limit_per_source`` + ``stop_at_target=False`` is the calibration
    mode used by ``scripts/build_pretraining_mix.py --calibrate``: stream a
    fixed number of raw docs per source and report every drop reason
    without stopping at the word target. ``did_scorer`` (a
    ``CamelDialectScorer``) is required when ``msa_filter.camel_did.enabled``.
    ``sources_override`` injects in-memory sources (tests).
    """
    out_dir = Path(out_dir) if out_dir is not None else pool_dir(cfg)
    if out_dir.exists() and not force and pool_is_complete(out_dir):
        logger.info("pretraining_mix pool: reusing %s", out_dir)
        return out_dir
    if out_dir.exists():
        logger.info("pretraining_mix pool: rebuilding %s (force=%s)", out_dir, force)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.msa_filter.camel_did.enabled and did_scorer is None:
        from .dialect_id import CamelDialectScorer
        did_scorer = CamelDialectScorer(cfg.msa_filter.camel_did)

    norm = cfg.normalization.model_dump()
    quality = cfg.quality
    heur = cfg.msa_filter.heuristic
    para_dedup = ParagraphDeduper() if cfg.dedup.paragraph_exact else None
    mh = cfg.dedup.minhash
    minhash = MinHashDeduper(mh.num_perm, mh.shingle_words, mh.threshold) if mh.enabled else None

    samples: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    dialect_csv = open(out_dir / "dropped_dialect.csv", "w", encoding="utf-8-sig", newline="")
    dialect_writer = csv.DictWriter(dialect_csv, fieldnames=DIALECT_CSV_FIELDS)
    dialect_writer.writeheader()

    def _sample(source: str, reason: str, doc_id: str, text: str, **extra: Any) -> None:
        bucket = samples.setdefault((source, reason), [])
        if len(bucket) < SAMPLE_PER_REASON:
            bucket.append({"source": source, "reason": reason, "id": doc_id,
                           "snippet": text[:300].replace("\n", " ⏎ "), **extra})

    manifest_sources: List[Dict[str, Any]] = []
    t_all = time.perf_counter()
    all_targets = True

    for src_cfg in cfg.source_order():
        source = (sources_override or {}).get(src_cfg.name) or make_source(src_cfg)
        target_words = int(round(src_cfg.share * cfg.pool.total_words))
        stats = SourceBuildStats(name=src_cfg.name, target_words=target_words)
        revision = None if sources_override else source.resolve_revision()
        sink = _ParquetSink(pool_source_path(out_dir, src_cfg.name))
        t0 = time.perf_counter()
        logger.info(
            "[pool:%s] streaming %s (target %d words%s)",
            src_cfg.name, source.repo_id or "in-memory", target_words,
            f", doc limit {doc_limit_per_source}" if doc_limit_per_source else "",
        )
        # Keep a handle on the generator: breaking out of a HF streaming
        # iterator without closing it leaves pyarrow/fsspec threads alive
        # and the interpreter aborts at exit (PyGILState_Release). Explicit
        # close() in the finally block below is what prevents that.
        doc_iter = source.iter_docs(seed=cfg.seed, shuffle_buffer=cfg.pool.stream_shuffle_buffer)
        try:
            for doc in doc_iter:
                stats.streamed += 1
                text = normalize_document(doc.text, **norm)

                if doc.kind == "wikipedia" and quality.wikipedia_strip_sections:
                    text, cut = strip_wikipedia_sections(text, quality.wikipedia_strip_sections)
                    stats.wiki_sections_cut += int(cut)

                if quality.strip_latin_parentheticals:
                    text, n_glosses = strip_latin_parentheticals(text)
                    stats.latin_parentheticals_removed += n_glosses

                reason = quality_reason(text, quality)
                if reason is None and heur.enabled:
                    score = dialect_marker_score(text)
                    if (score.n_markers >= heur.min_markers
                            and len(score.matched) >= heur.min_distinct_markers
                            and score.per_1k_words > heur.max_markers_per_1k_words):
                        reason = "dialect_markers"
                        top = " ".join(f"{m}×{c}" for m, c in score.top(5))
                        dialect_writer.writerow({
                            "source": src_cfg.name, "id": doc.id, "n_words": score.n_words,
                            "n_markers": score.n_markers, "per_1k_words": round(score.per_1k_words, 2),
                            "top_markers": top, "snippet": text[:200].replace("\n", " ⏎ "),
                        })
                        _sample(src_cfg.name, reason, doc.id, text, markers=top)
                if reason is None and did_scorer is not None and did_scorer.is_dialect(text):
                    reason = "dialect_camel"
                if reason is None and minhash is not None:
                    # Positional key: source ids may repeat or be empty (101B).
                    dup_of = minhash.check_and_add(f"{src_cfg.name}#{stats.streamed}:{doc.id}", text)
                    if dup_of is not None:
                        reason = "near_duplicate"
                        _sample(src_cfg.name, reason, doc.id, text, duplicate_of=dup_of)
                if reason is None and para_dedup is not None:
                    text, n_removed = para_dedup.filter(text)
                    if n_removed:
                        stats.paragraphs_removed += n_removed
                        stats.docs_with_paragraphs_removed += 1
                        if count_words(text) < quality.min_words:
                            reason = "too_short_after_dedup"

                if reason is not None:
                    stats.dropped[reason] += 1
                    if reason not in ("dialect_markers", "near_duplicate"):
                        _sample(src_cfg.name, reason, doc.id, text)
                else:
                    text, truncated = truncate_words(text, quality.max_words)
                    stats.docs_truncated += int(truncated)
                    n_words = count_words(text)
                    sink.write({"id": doc.id, "source": src_cfg.name, "text": text,
                                "n_words": n_words, "url": doc.meta.get("url")})
                    stats.kept += 1
                    stats.kept_words += n_words

                if log_every and stats.streamed % log_every == 0:
                    logger.info(
                        "[pool:%s] streamed=%d kept=%d (%.1f%%) words=%d/%d",
                        src_cfg.name, stats.streamed, stats.kept,
                        100.0 * stats.kept / stats.streamed, stats.kept_words, target_words,
                    )

                if stop_at_target and stats.kept_words >= target_words:
                    stats.reached_target = True
                    stats.stop_reason = "target_reached"
                    break
                if doc_limit_per_source and stats.streamed >= doc_limit_per_source:
                    stats.stop_reason = "doc_limit"
                    stats.reached_target = stats.kept_words >= target_words
                    break
                if stats.streamed >= cfg.pool.max_stream_docs_per_source:
                    stats.stop_reason = "max_stream_docs"
                    logger.warning(
                        "[pool:%s] hit max_stream_docs_per_source=%d with %d/%d words — "
                        "filters reject too much; raise the cap or loosen thresholds",
                        src_cfg.name, cfg.pool.max_stream_docs_per_source,
                        stats.kept_words, target_words,
                    )
                    break
            else:
                stats.stop_reason = "source_exhausted"
                stats.reached_target = stats.kept_words >= target_words
        finally:
            close = getattr(doc_iter, "close", None)
            if close is not None:
                close()
            sink.close()

        stats.wall_sec = round(time.perf_counter() - t0, 1)
        stats.source_prefilter_skipped = dict(source.skipped)
        all_targets = all_targets and stats.reached_target
        logger.info(
            "[pool:%s] done: streamed=%d kept=%d words=%d/%d dropped=%s wall=%.0fs",
            src_cfg.name, stats.streamed, stats.kept, stats.kept_words, target_words,
            dict(stats.dropped), stats.wall_sec,
        )
        entry = source.describe()
        entry.update({"revision": revision, "share": src_cfg.share, "stats": stats.to_json()})
        manifest_sources.append(entry)

    dialect_csv.close()
    with open(out_dir / "dropped_samples.jsonl", "w", encoding="utf-8") as f:
        for bucket in samples.values():
            for row in bucket:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "fingerprint": pool_fingerprint(cfg),
        "complete": True,
        "all_targets_reached": all_targets,
        "calibration": bool(doc_limit_per_source) or not stop_at_target,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_sec": round(time.perf_counter() - t_all, 1),
        "stage_a_config": stage_a_config(cfg),
        "camel_did_sentences_scored": getattr(did_scorer, "n_sentences_scored", 0),
        "paragraph_hashes": len(para_dedup) if para_dedup is not None else None,
        "minhash_docs": len(minhash) if minhash is not None else None,
        "sources": manifest_sources,
        "totals": {
            "streamed": sum(s["stats"]["streamed"] for s in manifest_sources),
            "kept": sum(s["stats"]["kept"] for s in manifest_sources),
            "kept_words": sum(s["stats"]["kept_words"] for s in manifest_sources),
        },
    }
    with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("pretraining_mix pool: built %s (%s)", out_dir, manifest["totals"])
    return out_dir


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def format_manifest(manifest: Dict[str, Any]) -> str:
    """Human-readable summary of a pool manifest (used by the CLI)."""
    lines = [
        f"pool fingerprint: {manifest.get('fingerprint')}  complete={manifest.get('complete')}  "
        f"all_targets_reached={manifest.get('all_targets_reached')}  "
        f"calibration={manifest.get('calibration')}  wall={manifest.get('wall_sec')}s",
    ]
    for src in manifest.get("sources", []):
        st = src["stats"]
        lines.append(
            f"\n[{src['name']}] {src.get('repo_id')} rev={str(src.get('revision'))[:10]} share={src.get('share')}"
        )
        lines.append(
            f"  streamed={st['streamed']}  kept={st['kept']} ({100 * st['keep_rate']:.1f}%)  "
            f"words={st['kept_words']}/{st['target_words']}  stop={st['stop_reason']}"
        )
        if st.get("source_prefilter_skipped"):
            lines.append(f"  source prefilter skipped: {st['source_prefilter_skipped']}")
        dropped = st.get("dropped") or {}
        if dropped:
            total = max(st["streamed"], 1)
            lines.append("  drop reasons:")
            for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {reason:<24} {n:>8}  ({100.0 * n / total:5.1f}% of streamed)")
        lines.append(
            f"  paragraphs removed={st['paragraphs_removed']} (in {st['docs_with_paragraphs_removed']} docs)  "
            f"truncated={st['docs_truncated']}  wiki_sections_cut={st['wiki_sections_cut']}  "
            f"latin_glosses_removed={st.get('latin_parentheticals_removed', 0)}"
        )
    t = manifest.get("totals", {})
    lines.append(f"\ntotals: streamed={t.get('streamed')} kept={t.get('kept')} words={t.get('kept_words')}")
    return "\n".join(lines)
