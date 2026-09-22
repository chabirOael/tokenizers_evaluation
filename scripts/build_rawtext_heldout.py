#!/usr/bin/env python
"""Build a raw-text held-out set the pretraining-mix loss diagnostic can trust.

``scripts/diag_heldout_loss.py`` used to read the **last 120 documents of the
cell's pool** ``fineweb2_arb.parquet``. That is not held out: the packer walks
the whole pool in ``source_doc_order`` and the phases draw a *prefix of the
packed blocks*, so for the adapted arms of the 2026-09-22 campaign each of
those documents was training text with probability 0.88 (AraRooPat v3), 0.98
(BPE-16K) and 0.62 (v2), while the native cells were measured on a pool they
had never seen — an in-training number compared with a held-out one.

The rule here, instead. A candidate is a FineWeb-2 document of the source pool
(default: the 80 M-word pool, the largest built) that

  1. sits at a position **>= the largest ``docs_taken``** of any packed entry
     under that pool, in ``source_doc_order(n_docs, seed, "fineweb2_arb")`` —
     i.e. no packer of this pool ever reached it; and
  2. whose raw text sha1 appears in **no other pool directory** under
     ``outputs/data_cache/pretraining_mix/`` (all three sources) — so no run on
     another pool trained on it either; and
  3. (belt and braces, beyond the rule) whose sha1 is not also one of the
     *taken* documents of the source pool.

``--verify`` then re-checks the draw with the contamination normalizer: the
drawn documents are indexed and every pool document is scanned against them,
so a near-copy that shares a long window is caught even when the hashes differ.

    .venv/bin/python scripts/build_rawtext_heldout.py                   # build + verify
    .venv/bin/python scripts/build_rawtext_heldout.py --n-docs 150 --seed 42

The file is **not** registered in ``configs/contamination/heldout_sets.yaml``:
doing so changes the held-out fingerprint, hence the pretraining-mix pool
fingerprint and ``exclusions.json``, which would confound a comparison against
cells trained on the current pools. It must be registered before the next pool
is built, so that future pools drop it by content at Stage A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.data.pretraining_mix.packing import source_doc_order  # noqa: E402

log = logging.getLogger("build_rawtext_heldout")

MIX_CACHE = REPO_ROOT / "outputs/data_cache/pretraining_mix"
DEFAULT_POOL = "51e0a74a3d05b6cf"
DEFAULT_OUT = REPO_ROOT / "configs/contamination/rawtext_heldout_v1.jsonl"
SOURCE = "fineweb2_arb"
SOURCES = ("fineweb2_arb", "wikipedia_ar", "arabicweb24")
N_DOCS = 150
SEED = 42


# --------------------------------------------------------------------------
# pure helpers (tested)
# --------------------------------------------------------------------------

def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (tests write to tmp)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def largest_docs_taken(pool_dir: Path, source: str) -> Tuple[int, List[Dict[str, Any]]]:
    """The largest ``docs_taken`` over every packed entry of a pool, and what
    each entry reported (a pool with no packed entry yields 0)."""
    entries: List[Dict[str, Any]] = []
    best = 0
    for man in sorted(pool_dir.glob("packed/*/packed_manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        srcs = m.get("sources") or []
        taken = next((int(s.get("docs_taken", 0)) for s in srcs if s.get("name") == source), 0)
        entries.append({
            "entry": man.parent.name,
            "tokenizer": (m.get("tokenizer") or {}).get("type") if isinstance(m.get("tokenizer"), dict) else m.get("tokenizer"),
            "docs_taken": taken,
        })
        best = max(best, taken)
    return best, entries


def untaken_positions(n_docs: int, seed: int, source: str, docs_taken: int) -> List[int]:
    """Pool-row indices the packer never reached, in permutation order."""
    order = source_doc_order(n_docs, seed, source)
    return [int(i) for i in order[docs_taken:]]


def ngram_clean(candidates: Sequence[Tuple[int, str, str]]) -> Tuple[Set[str], Dict[str, Any]]:
    """sha1s of the candidates that share a long window with any pool document.

    The criterion is the meaningful one for raw web text: an exact duplicate, a
    shared run of >= ``run_words`` (20) words, or >= ``coverage`` (0.5) of the
    candidate's 8-grams. The index's exact-**paragraph** tier is deliberately
    not a criterion here: in FineWeb-2 a "paragraph" is a line, and hundreds of
    those lines are short headings ("مقدمه اذاعه عن صحه الاسنان") that recur
    verbatim across thousands of unrelated pages — 20 661 such matches on the
    first draw, none of them document overlap.
    """
    from arabic_eval.data.contamination import HeldoutIndex, HeldoutRecord, normalize_text, split_paragraphs

    records = []
    for _pos, doc_id, text in candidates:
        norm = normalize_text(text)
        records.append(HeldoutRecord(
            set_name="rawtext", rec_id=text_sha1(text), question="", passage=norm,
            paragraphs=[normalize_text(p) for p in split_paragraphs(text)], text=norm,
        ))
    index = HeldoutIndex(records, {"rawtext": 8})
    thr = index.thresholds
    log.info("n-gram filter: indexing %d candidates (run_words>=%s, coverage>=%s)",
             len(records), thr.run_words, thr.coverage)

    # A candidate is a document *of* the source pool, so the scan meets its own
    # row: a hit whose held-out id equals the scanned row's text hash is that
    # self-match and is skipped.
    dirty: Set[str] = set()
    worst: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    paragraph_only = 0
    for d in sorted(MIX_CACHE.iterdir()):
        if not d.is_dir() or d.name == "qa_blend":
            continue
        for src in SOURCES:
            for row in read_pool_source(d, src, ["id", "text"]):
                scanned += 1
                row_hash = text_sha1(row["text"])
                for hit in index.scan({"passage": row["text"]}):
                    if hit.heldout_id == row_hash:
                        continue          # the candidate's own row in its own pool
                    if hit.coverage >= thr.coverage or hit.run_words >= thr.run_words \
                       or hit.exact in ("passage", "text"):
                        dirty.add(hit.heldout_id)
                        prev = worst.get(hit.heldout_id)
                        if prev is None or hit.run_words > prev["run_words"]:
                            worst[hit.heldout_id] = {
                                "pool": d.name, "source": src, "doc": row["id"],
                                "exact": hit.exact, "coverage": round(hit.coverage, 3),
                                "run_words": hit.run_words,
                            }
                    elif hit.exact == "paragraph":
                        paragraph_only += 1
    log.info("n-gram filter: %d pool documents scanned, %d/%d candidates share a long window "
             "(%d short-heading paragraph matches ignored)",
             scanned, len(dirty), len(records), paragraph_only)
    return dirty, {"pool_documents_scanned": scanned, "candidates_indexed": len(records),
                   "candidates_with_long_window": len(dirty),
                   "short_heading_matches_ignored": paragraph_only,
                   "criterion": "exact duplicate, or a shared run >= 20 words, or coverage >= 0.5 of the candidate's 8-grams",
                   "examples": [dict(heldout_id=k, **v) for k, v in list(worst.items())[:10]]}


def select_heldout(candidates: Sequence[Tuple[int, str, str]], excluded_hashes: Set[str],
                   n_docs: int, seed: int) -> List[Tuple[int, str, str]]:
    """``(position, doc_id, text)`` triples that survive the hash filter, drawn
    with a seeded permutation. ``position`` is the index in the packer's order
    (not the pool row) so the manifest says how far past ``docs_taken`` it sat."""
    import numpy as np

    kept = [c for c in candidates if text_sha1(c[2]) not in excluded_hashes]
    if len(kept) < n_docs:
        raise SystemExit(f"only {len(kept)} candidates survive the hash filter, {n_docs} requested")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(kept))[:n_docs]
    return [kept[int(i)] for i in sorted(idx)]


# --------------------------------------------------------------------------
# pool reading
# --------------------------------------------------------------------------

def read_pool_source(pool_dir: Path, source: str, columns: Sequence[str]) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    path = pool_dir / f"{source}.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path, columns=list(columns))
    return tbl.to_pylist()


def pool_text_hashes(pool_dir: Path) -> Set[str]:
    """sha1 of every document text of every source of one pool."""
    out: Set[str] = set()
    for src in SOURCES:
        for row in read_pool_source(pool_dir, src, ["text"]):
            out.add(text_sha1(row["text"]))
    return out


def build(pool_fp: str, out_path: Path, n_docs: int, seed: int,
          ngram_filter: bool = True) -> Dict[str, Any]:
    pool_dir = MIX_CACHE / pool_fp
    if not pool_dir.is_dir():
        raise SystemExit(f"no pool at {pool_dir}")

    rows = read_pool_source(pool_dir, SOURCE, ["id", "text", "n_words"])
    if not rows:
        raise SystemExit(f"{pool_dir}/{SOURCE}.parquet is missing or empty")
    mix_seed = _pool_seed(pool_dir)
    docs_taken, entries = largest_docs_taken(pool_dir, SOURCE)
    log.info("pool %s: %d %s docs, packed entries %s, largest docs_taken %d",
             pool_fp, len(rows), SOURCE, [e["entry"] for e in entries], docs_taken)

    positions = untaken_positions(len(rows), mix_seed, SOURCE, docs_taken)
    candidates = [(pos, rows[pos]["id"], rows[pos]["text"]) for pos in positions]
    log.info("candidates beyond docs_taken: %d", len(candidates))

    # hashes of every other pool (all sources) + this pool's taken docs
    other_pools: Dict[str, int] = {}
    other_hashes: Set[str] = set()
    for d in sorted(MIX_CACHE.iterdir()):
        if not d.is_dir() or d.name == pool_fp or d.name == "qa_blend":
            continue
        h = pool_text_hashes(d)
        other_pools[d.name] = len(h)
        other_hashes |= h
    taken_positions = source_doc_order(len(rows), mix_seed, SOURCE)[:docs_taken].tolist()
    same_pool_taken = {text_sha1(rows[int(p)]["text"]) for p in taken_positions}
    excluded = other_hashes | same_pool_taken

    cand_hashes = [text_sha1(c[2]) for c in candidates]
    dropped_other = sum(1 for h in cand_hashes if h in other_hashes)
    dropped_same = sum(1 for h in cand_hashes if h in same_pool_taken and h not in other_hashes)
    surviving = [c for c, h in zip(candidates, cand_hashes) if h not in excluded]
    log.info("dropped by other-pool hash %d, by same-pool taken hash %d, surviving %d",
             dropped_other, dropped_same, len(surviving))

    n_hash_surviving = len(surviving)
    if ngram_filter:
        dirty, scan_meta = ngram_clean(surviving)
        excluded |= dirty
        surviving = [c for c in surviving if text_sha1(c[2]) not in dirty]
        log.info("after the n-gram filter: %d clean candidates", len(surviving))
    else:
        scan_meta = {"skipped": True}

    drawn = select_heldout(candidates, excluded, n_docs, seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pos, doc_id, text in drawn:
            f.write(json.dumps({
                "id": text_sha1(text),
                "source": SOURCE,
                "pool_fingerprint": pool_fp,
                "position": pos,
                "words": len(text.split()),
                "text": text,
            }, ensure_ascii=False) + "\n")

    words = [len(t.split()) for _, _, t in drawn]
    manifest = {
        "file": _rel(out_path),
        "sha256": file_sha256(out_path),
        "n_docs": len(drawn),
        "seed": seed,
        "source": SOURCE,
        "rule": (
            "FineWeb-2 documents of the source pool at a position >= the largest docs_taken of any "
            "packed entry of that pool in source_doc_order(n_docs, pool_seed, source), whose raw-text "
            "sha1 is absent from every other pool directory (all sources) and from the taken documents "
            "of the source pool; drawn with a seeded permutation."
        ),
        "source_pool": {
            "fingerprint": pool_fp, "seed": mix_seed, "docs": len(rows),
            "packed_entries": entries, "docs_taken_max": docs_taken,
        },
        "other_pools": other_pools,
        "candidates": {
            "beyond_docs_taken": len(candidates),
            "dropped_other_pool_hash": dropped_other,
            "dropped_same_pool_taken_hash": dropped_same,
            "surviving_hash_filter": n_hash_surviving,
            "clean_after_ngram_filter": len(surviving),
        },
        "ngram_filter": scan_meta,
        "words": {
            "total": sum(words), "median": sorted(words)[len(words) // 2],
            "min": min(words), "max": max(words),
        },
        "registered_in_heldout_sets": False,
        "registration_note": (
            "Deliberately not in configs/contamination/heldout_sets.yaml: registering it changes the "
            "held-out fingerprint, hence the pretraining-mix pool fingerprint and exclusions.json, which "
            "would confound comparisons with cells trained on the current pools. Register it before the "
            "next pool build so Stage A drops it by content."
        ),
    }
    man_path = out_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s (%d docs) and %s", out_path, len(drawn), man_path)
    return manifest


def _pool_seed(pool_dir: Path) -> int:
    m = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = m.get("stage_a_config") or {}
    return int(cfg.get("seed", SEED))


# --------------------------------------------------------------------------
# verification with the contamination normalizer
# --------------------------------------------------------------------------

def verify(out_path: Path, pool_fp: str) -> Dict[str, Any]:
    """Index the drawn documents and scan every pool document against them.

    The decision criterion is the one the brief states and the one that means
    something for raw web text: a pool document is a **real** overlap when it
    shares a run of >= ``run_words`` (20) words with a drawn document, or covers
    >= ``coverage`` (0.5) of its 8-grams, or duplicates it exactly.

    The index's exact-**paragraph** tier is not that criterion. It is built for
    QA passages, where a paragraph is substantial; in FineWeb-2 a "paragraph" is
    a line, and 764 of the drawn documents' 3 000-odd lines are short headings
    ("مقدمه اذاعه عن صحه الاسنان") that recur verbatim across thousands of
    unrelated pages. Those are counted and reported separately, never as a hit.
    """
    from arabic_eval.data.contamination import HeldoutIndex, HeldoutRecord, normalize_text, split_paragraphs

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = []
    for r in rows:
        norm = normalize_text(r["text"])
        records.append(HeldoutRecord(
            set_name="rawtext", rec_id=r["id"], question="", passage=norm,
            paragraphs=[normalize_text(p) for p in split_paragraphs(r["text"])], text=norm,
        ))
    index = HeldoutIndex(records, {"rawtext": 8})
    thr = index.thresholds
    log.info("verification index: %d documents (run_words>=%s, coverage>=%s)",
             len(index), thr.run_words, thr.coverage)

    real: List[Dict[str, Any]] = []
    short_paragraph = 0
    long_paragraph: List[Dict[str, Any]] = []
    scanned = 0
    para_words = {r.rec_id: {p: len(p.split()) for p in r.paragraphs} for r in records}
    for d in sorted(MIX_CACHE.iterdir()):
        if not d.is_dir() or d.name == "qa_blend":
            continue
        for src in SOURCES:
            for row in read_pool_source(d, src, ["id", "text"]):
                scanned += 1
                row_hash = text_sha1(row["text"])
                for hit in index.scan({"passage": row["text"]}):
                    if hit.heldout_id == row_hash:
                        continue          # the drawn document's own row in its own pool
                    rec = {"pool": d.name, "source": src, "doc": row["id"],
                           "heldout_id": hit.heldout_id, "exact": hit.exact,
                           "coverage": round(hit.coverage, 3), "run_words": hit.run_words}
                    if hit.coverage >= thr.coverage or hit.run_words >= thr.run_words \
                       or hit.exact in ("passage", "text"):
                        real.append(rec)
                    elif hit.exact == "paragraph":
                        # a heading that recurs across the web, or a real shared block?
                        norm = normalize_text(row["text"])
                        shared = [p for p, n in para_words.get(hit.heldout_id, {}).items()
                                  if n >= thr.run_words and p and p in norm]
                        if shared:
                            rec["shared_paragraph_words"] = max(len(p.split()) for p in shared)
                            long_paragraph.append(rec)
                        else:
                            short_paragraph += 1
    log.info("scanned %d pool documents: %d real overlaps, %d long shared paragraphs, "
             "%d short-heading paragraph matches (ignored)",
             scanned, len(real), len(long_paragraph), short_paragraph)
    return {"scanned": scanned, "hits": real + long_paragraph,
            "real_overlaps": len(real), "long_shared_paragraphs": len(long_paragraph),
            "short_heading_matches": short_paragraph}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=DEFAULT_POOL, help="source pool fingerprint (default: the 80 M-word pool)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-docs", type=int, default=N_DOCS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-ngram-filter", action="store_true",
                    help="draw from the hash-filtered candidates without the n-gram pass (faster, less clean)")
    ap.add_argument("--verify-only", action="store_true", help="re-verify the existing file without rebuilding it")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    if not args.verify_only:
        man = build(args.pool, out, args.n_docs, args.seed, ngram_filter=not args.no_ngram_filter)
        print(json.dumps({k: v for k, v in man.items() if k != "other_pools"}, ensure_ascii=False, indent=2))
    if not args.no_verify:
        res = verify(out, args.pool)
        man_path = out.with_suffix(".manifest.json")
        m = json.loads(man_path.read_text(encoding="utf-8"))
        m["verification"] = {
            "pool_documents_scanned": res["scanned"],
            "real_overlaps": res["real_overlaps"],
            "long_shared_paragraphs": res["long_shared_paragraphs"],
            "short_heading_matches": res["short_heading_matches"],
            "criterion": "coverage >= 0.5, or a shared run >= 20 words, or an exact passage duplicate",
            "hits": res["hits"][:20],
        }
        man_path.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nverification: {res['scanned']} pool documents scanned — "
              f"{res['real_overlaps']} real overlaps, {res['long_shared_paragraphs']} long shared paragraphs, "
              f"{res['short_heading_matches']} short-heading matches (ignored)")
        for h in res["hits"][:10]:
            print("  ", h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
