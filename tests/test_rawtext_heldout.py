"""The raw-text held-out set for the LM-loss diagnostic (added 2026-09-22).

Two halves: the selection rule on a synthetic pool (positions beyond the
packer's ``docs_taken``, the cross-pool hash filter, the seeded draw, the
manifest), and the committed file itself (150 rows, unique ids, Arabic).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.data.pretraining_mix.packing import source_doc_order  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_rawtext_heldout", REPO / "scripts" / "build_rawtext_heldout.py")
brh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brh)

COMMITTED = REPO / "configs" / "contamination" / "rawtext_heldout_v1.jsonl"


# --------------------------------------------------------------------------
# the selection rule
# --------------------------------------------------------------------------

def _pool(tmp_path: Path, n_docs: int, taken: int, texts=None, name="poolfp", seed=42):
    """A synthetic pool directory with a fineweb parquet and one packed entry."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path / name
    (d / "packed" / "tok1").mkdir(parents=True)
    texts = texts or [f"وثيقة رقم {i} " + " ".join(["كلمة"] * 60) for i in range(n_docs)]
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(n_docs)], "source": ["fineweb2_arb"] * n_docs,
        "text": texts, "n_words": [len(t.split()) for t in texts], "url": [""] * n_docs,
    }), d / "fineweb2_arb.parquet")
    (d / "manifest.json").write_text(json.dumps({"stage_a_config": {"seed": seed}}), encoding="utf-8")
    (d / "packed" / "tok1" / "packed_manifest.json").write_text(json.dumps({
        "tokenizer": {"type": "bpe"},
        "sources": [{"name": "fineweb2_arb", "docs_taken": taken, "docs_available": n_docs}],
    }), encoding="utf-8")
    return d


def test_largest_docs_taken_takes_the_max_over_entries(tmp_path):
    d = _pool(tmp_path, 50, 30)
    (d / "packed" / "tok2").mkdir()
    (d / "packed" / "tok2" / "packed_manifest.json").write_text(json.dumps({
        "tokenizer": {"type": "araroopat"},
        "sources": [{"name": "fineweb2_arb", "docs_taken": 44, "docs_available": 50}],
    }), encoding="utf-8")
    best, entries = brh.largest_docs_taken(d, "fineweb2_arb")
    assert best == 44
    assert sorted(e["docs_taken"] for e in entries) == [30, 44]


def test_untaken_positions_are_exactly_the_tail_of_the_packer_order():
    n, taken = 100, 80
    order = source_doc_order(n, 42, "fineweb2_arb")
    pos = brh.untaken_positions(n, 42, "fineweb2_arb", taken)
    assert pos == [int(i) for i in order[taken:]]
    assert len(pos) == n - taken
    # and none of them is one the packer took
    assert not (set(pos) & set(int(i) for i in order[:taken]))


def test_select_drops_hash_excluded_and_is_seed_deterministic():
    cands = [(i, f"d{i}", f"نص فريد رقم {i}") for i in range(40)]
    excluded = {brh.text_sha1(t) for _, _, t in cands[:10]}
    a = brh.select_heldout(cands, excluded, 8, 42)
    b = brh.select_heldout(cands, excluded, 8, 42)
    c = brh.select_heldout(cands, excluded, 8, 7)
    assert a == b
    assert a != c
    assert len(a) == 8
    assert all(brh.text_sha1(t) not in excluded for _, _, t in a)


def test_select_raises_when_too_few_candidates_survive():
    cands = [(i, f"d{i}", f"نص {i}") for i in range(5)]
    with pytest.raises(SystemExit, match="survive the hash filter"):
        brh.select_heldout(cands, set(), 10, 42)


def test_build_excludes_documents_present_in_another_pool(tmp_path, monkeypatch):
    shared = "نص مشترك بين البركتين " + " ".join(["كلمة"] * 60)
    texts = [f"وثيقة {i} " + " ".join(["كلمة"] * 60) for i in range(60)]
    order = source_doc_order(60, 42, "fineweb2_arb")
    untaken = [int(i) for i in order[40:]]
    texts[untaken[0]] = shared                       # an untaken doc that another pool also holds
    src = _pool(tmp_path, 60, 40, texts=texts, name="srcpool")
    _pool(tmp_path, 3, 0, texts=[shared, "غير ذي صلة أ", "غير ذي صلة ب"], name="otherpool")

    monkeypatch.setattr(brh, "MIX_CACHE", tmp_path)
    out = tmp_path / "heldout.jsonl"
    man = brh.build("srcpool", out, n_docs=5, seed=42, ngram_filter=False)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert all(r["text"] != shared for r in rows)
    assert man["candidates"]["beyond_docs_taken"] == 20
    assert man["candidates"]["dropped_other_pool_hash"] == 1
    assert man["candidates"]["surviving_hash_filter"] == 19
    assert man["source_pool"]["docs_taken_max"] == 40
    assert man["registered_in_heldout_sets"] is False
    assert (tmp_path / "heldout.manifest.json").exists()
    # ids are the text hashes, positions are pool rows past the packer's reach
    for r in rows:
        assert r["id"] == brh.text_sha1(r["text"])
        assert r["position"] in untaken
        assert r["words"] == len(r["text"].split())


def test_build_manifest_records_sha256_and_word_stats(tmp_path, monkeypatch):
    _pool(tmp_path, 40, 20, name="srcpool")
    monkeypatch.setattr(brh, "MIX_CACHE", tmp_path)
    out = tmp_path / "h.jsonl"
    man = brh.build("srcpool", out, n_docs=6, seed=1, ngram_filter=False)
    assert man["sha256"] == brh.file_sha256(out)
    assert man["n_docs"] == 6 and man["seed"] == 1
    assert man["words"]["min"] <= man["words"]["median"] <= man["words"]["max"]


def test_ngram_clean_skips_the_candidates_own_row_but_catches_a_copy(tmp_path, monkeypatch):
    """A candidate meets itself in its own pool — that self-match is not a hit;
    a near-copy in another pool is."""
    body = " ".join(f"كلمة{i}" for i in range(120))
    cand_text = "بداية الوثيقة " + body
    _pool(tmp_path, 4, 0, texts=[cand_text, "غير ذي صلة", "نص آخر", "نص ثالث"], name="srcpool")
    _pool(tmp_path, 2, 0, texts=["مقدمة مختلفة " + body, "لا علاقة"], name="otherpool")
    monkeypatch.setattr(brh, "MIX_CACHE", tmp_path)

    dirty, meta = brh.ngram_clean([(0, "d0", cand_text)])
    assert brh.text_sha1(cand_text) in dirty          # the other pool's near-copy is a real overlap
    assert meta["candidates_with_long_window"] == 1

    unique = "وثيقة لا نظير لها " + " ".join(f"فريدة{i}" for i in range(120))
    _pool(tmp_path, 1, 0, texts=[unique], name="srcpool2")
    dirty2, _ = brh.ngram_clean([(0, "u0", unique)])
    assert dirty2 == set()                            # only its own row matched


# --------------------------------------------------------------------------
# the committed file
# --------------------------------------------------------------------------

@pytest.mark.skipif(not COMMITTED.exists(), reason="the held-out file is not built")
def test_committed_file_shape():
    rows = [json.loads(l) for l in COMMITTED.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 150
    assert len({r["id"] for r in rows}) == 150
    for r in rows:
        assert set(r) == {"id", "source", "pool_fingerprint", "position", "words", "text"}
        assert r["source"] == "fineweb2_arb"
        assert r["id"] == brh.text_sha1(r["text"])
        assert r["words"] >= 50


@pytest.mark.skipif(not COMMITTED.exists(), reason="the held-out file is not built")
def test_committed_file_is_arabic_not_latin():
    rows = [json.loads(l) for l in COMMITTED.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        letters = [c for c in r["text"] if c.isalpha()]
        latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
        assert latin / max(len(letters), 1) <= 0.05, f"{r['id']} is Latin-heavy"


@pytest.mark.skipif(not COMMITTED.exists(), reason="the held-out file is not built")
def test_committed_manifest_matches_the_file():
    man = json.loads(COMMITTED.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert man["sha256"] == brh.file_sha256(COMMITTED)
    assert man["n_docs"] == 150
    assert man["registered_in_heldout_sets"] is False
    assert man["verification"]["real_overlaps"] == 0
