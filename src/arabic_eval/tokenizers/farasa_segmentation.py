"""Farasa morpheme boundaries over a corpus, aligned to the raw text (2026-11-08).

Used by ``farasa_bpe`` (Farasa-BPE-16K, "ours, MorphBPE-style"). Farasa rewrites
some surfaces — it strips diacritics, turns Arabic-Indic digits into ASCII, writes
the لِ + الـ contraction with its alef (``للشركة`` → ``ل+ال+شرك+ة``), restores a
dropped ي (``ثان`` → ``ثاني``), folds ``ٱ`` and splits punctuation off — so its
output is never used as training text. Only its **boundaries** are kept, mapped
back onto the raw word, and **the raw text is never changed** — a boundary is
marked by inserting ``JOINER`` between two raw characters.

Alignment rule, per raw whitespace word: join Farasa's segments of that word
without ``+``; if the join equals the raw word the boundaries are the cumulative
segment lengths; otherwise the join is aligned to the raw word character by
character (``difflib.SequenceMatcher``, autojunk off) and a boundary is kept only
when the two characters on either side of it lie in the same ``equal`` block (a
word without an ``equal`` block keeps none). Only boundaries strictly inside the
raw word are kept.

Segmentation is done **per distinct word, one word per line**: Farasa's segmenter
scores a word on its own features, so a word gets the same segmentation in and
out of context (verified per build on sampled corpus texts, ``context_check`` in
the manifest), and one line per word gives an exact word ↔ output correspondence.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("arabic_eval.tokenizers.farasa_segmentation")

JOINER = "⁠"            # WORD JOINER; verified absent from the corpus at build time (fallback U+E000)
FALLBACK_JOINER = ""
BATCH_WORDS = 4000           # words per Farasa call (one per line)


# --------------------------------------------------------------------------- versions / fingerprint
def farasa_versions() -> Dict[str, str]:
    import farasa
    from importlib.metadata import version
    base = Path(farasa.__file__).resolve().parent
    jar = base / "farasa_bin" / "lib" / "FarasaSegmenterJar.jar"
    h = hashlib.sha256()
    if jar.exists():
        with open(jar, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return {"farasapy": version("farasapy"), "segmenter_jar": str(jar),
            "segmenter_jar_sha256": h.hexdigest() if jar.exists() else "missing"}


def texts_digest(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def corpus_fingerprint(digest: str, preprocessing: Optional[dict], versions: Dict[str, str]) -> str:
    payload = {"texts_sha256": digest, "preprocessing": preprocessing or None,
               "farasapy": versions["farasapy"], "jar_sha256": versions["segmenter_jar_sha256"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- alignment
def farasa_segments(farasa_out: str) -> List[str]:
    """Farasa's output for one raw word → its segments (a literal '+' token stays text)."""
    segs: List[str] = []
    for tok in farasa_out.split():
        if tok.strip("+") == "":
            segs.append(tok)                 # a literal plus sign (or several)
            continue
        segs.extend(p for p in tok.split("+") if p != "")
    return segs


def align_word(raw: str, farasa_out: str) -> Tuple[List[int], int, bool]:
    """Boundaries (character offsets strictly inside ``raw``) kept from Farasa's segmentation.

    Returns ``(kept, dropped, rewritten)``: ``rewritten`` is True when the join of the
    segments differs from the raw word (a Farasa rewrite).
    """
    segs = farasa_segments(farasa_out)
    if len(segs) <= 1:
        return [], 0, ("".join(segs) != raw)
    join = "".join(segs)
    cuts, pos = [], 0
    for s in segs[:-1]:
        pos += len(s)
        cuts.append(pos)
    if join == raw:
        return [c for c in cuts if 0 < c < len(raw)], 0, False
    sm = difflib.SequenceMatcher(None, join, raw, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    kept: List[int] = []
    for c in cuts:
        for b in blocks:
            # chars c-1 and c of the join both inside block b
            if b.a <= c - 1 and c < b.a + b.size:
                r = c - b.a + b.b
                if 0 < r < len(raw):
                    kept.append(r)
                break
    return kept, len(cuts) - len(kept), True


def insert_joiners(text: str, table: Dict[str, Tuple[int, ...]], joiner: str = JOINER) -> str:
    """The raw text with ``joiner`` at every kept boundary of every word (whitespace untouched)."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            out.append(text[i:j])
            i = j
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        word = text[i:j]
        cuts = table.get(word)
        if cuts:
            prev = 0
            for c in cuts:
                out.append(word[prev:c])
                out.append(joiner)
                prev = c
            out.append(word[prev:])
        else:
            out.append(word)
        i = j
    return "".join(out)


# --------------------------------------------------------------------------- Farasa over word types
def _seg_lines(words: List[str], depth: int = 0) -> Tuple[List[str], int]:
    """Farasa on one word per line; a batch whose line count comes back wrong (a word Farasa
    turns into several lines or none) is bisected down to the offending word, which is then
    segmented alone with its newlines flattened. Returns (outputs, words segmented alone)."""
    global _SEG
    from farasa.segmenter import FarasaSegmenter
    last = None
    for _ in range(3):
        try:
            if _SEG is None:
                _SEG = FarasaSegmenter(interactive=True)
            out = _SEG.segment("\n".join(words)).split("\n")
            break
        except Exception as e:  # noqa: BLE001 — restart the Java process and retry
            _SEG, out, last = None, None, e
    if out is None:
        if len(words) == 1:
            return [words[0]], 1           # unsegmentable: kept whole (no boundary)
        raise RuntimeError(f"Farasa failed 3 times on a batch of {len(words)}: {last}")
    if len(out) == len(words):
        return out, 0
    if len(words) == 1:
        return [" ".join(out)], 1
    mid = len(words) // 2
    a, na = _seg_lines(words[:mid], depth + 1)
    b, nb = _seg_lines(words[mid:], depth + 1)
    return a + b, na + nb


def _segment_batch_worker(args):
    idx, words = args
    try:
        out, alone = _seg_lines(list(words))
        return idx, out, (f"{alone} words segmented alone (line-count mismatch)" if alone else None)
    except Exception as e:  # noqa: BLE001
        return idx, list(words), f"FAILED, words kept unsegmented: {type(e).__name__}: {e}"


_SEG = None


def segment_words(words: Sequence[str], workers: int = 8, batch: int = BATCH_WORDS,
                  log_every: int = 50) -> Tuple[List[str], Dict[str, object]]:
    """Farasa output for each word (same order). One Farasa JVM per worker process."""
    batches = [(k, list(words[s:s + batch])) for k, s in enumerate(range(0, len(words), batch))]
    results: List[Optional[List[str]]] = [None] * len(batches)
    errors: List[str] = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, workers)) as pool:
        for done, (idx, out, err) in enumerate(pool.imap_unordered(_segment_batch_worker, batches), 1):
            results[idx] = out
            if err:
                errors.append(f"batch {idx}: {err}")
            if done % log_every == 0 or done == len(batches):
                el = time.time() - t0
                logger.info("farasa: %d / %d batches, %.0f s, %.0f words/s", done, len(batches), el,
                            done * batch / max(el, 1e-9))
    flat = [o for r in results for o in r]  # type: ignore[union-attr]
    return flat, {"wall_sec": round(time.time() - t0, 1), "workers": workers, "batch_words": batch,
                  "batches": len(batches), "errors": errors}


def build_boundary_table(word_counts: Counter, farasa_out: Sequence[str], words: Sequence[str]
                         ) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, object]]:
    """``word → kept boundaries`` plus the occurrence-weighted counts of the manifest."""
    table: Dict[str, Tuple[int, ...]] = {}
    occ_words = occ_kept = occ_dropped = occ_rewrite_words = occ_dropped_words = 0
    types_kept = types_dropped = 0
    rewrites: Counter = Counter()
    for w, fo in zip(words, farasa_out):
        n = word_counts[w]
        kept, dropped, rewritten = align_word(w, fo)
        if kept:
            table[w] = tuple(kept)
        occ_words += n
        occ_kept += n * len(kept)
        occ_dropped += n * dropped
        types_kept += len(kept)
        types_dropped += dropped
        if rewritten:
            occ_rewrite_words += n
            rewrites[("+".join(farasa_segments(fo)), w)] += n
        if dropped:
            occ_dropped_words += n
    stats = {
        "word_types": len(words), "word_occurrences": occ_words,
        "boundaries_kept": occ_kept, "boundaries_dropped": occ_dropped,
        "boundaries_per_100_words": round(100 * occ_kept / max(occ_words, 1), 2),
        "words_with_dropped_boundary": occ_dropped_words,
        "words_with_dropped_boundary_share": round(occ_dropped_words / max(occ_words, 1), 5),
        "farasa_rewrite_words": occ_rewrite_words,
        "farasa_rewrite_share": round(occ_rewrite_words / max(occ_words, 1), 5),
        "types_with_boundary": len(table), "type_boundaries_kept": types_kept, "type_boundaries_dropped": types_dropped,
        "top_rewrites": [{"farasa": f, "raw": r, "occurrences": c} for (f, r), c in rewrites.most_common(10)],
        "unit": "occurrence-weighted over whitespace words of the training texts",
    }
    return table, stats


def collect_word_counts(texts: Iterable[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(t.split())
    return c


def save_table(path: Path, table: Dict[str, Tuple[int, ...]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    words = list(table)
    pq.write_table(pa.table({"word": words, "cuts": [list(table[w]) for w in words]}), path, compression="zstd")


def load_table(path: Path) -> Dict[str, Tuple[int, ...]]:
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pydict()
    return {w: tuple(c) for w, c in zip(t["word"], t["cuts"])}
