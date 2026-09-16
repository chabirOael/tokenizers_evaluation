"""Tests for the corpus-scale AraRooPat explorer (tab 03).

Tier A — pure: wazn / gloss / describe_pattern, request validation, the JS
wazn map in the page mirrors the Python one, the page splice is additive.
Tier B — live CAMeL bridge (skipped without ``.venv-camel``): a full
``CorpusTraceJob`` in train mode on an injected mini corpus (fresh → cache
written → verify hits it), a second run that hits the cache, the search
index, save + saved-mode load.
"""
from __future__ import annotations

import json
import pickle
import re
import shutil
import time
from pathlib import Path

import pytest

from arabic_eval.tokenizers import araroopat_corpus_trace as C

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "araroopat_train_explorer.html"
_CAMEL_VENV = REPO / ".venv-camel" / "bin" / "python"

# ---------------------------------------------------------------------------
# Tier A — pure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern, wazn", [
    ("مَ1ْ2َ3َ", "مَفْعَلَ"),
    ("ٱِسْتِ1ْ2ا3", "ٱِسْتِفْعال"),
    ("يُ1َ2ِّ3", "يُفَعِّل"),
    ("1ا2ِ3", "فاعِل"),
    ("1َ2ْ3َ4", "فَعْلَل"),          # quadriliteral: second ل by convention
    ("1ا3", "فال"),                  # hollow: slot 2 absent, ا is template material
    ("أَ1ْ2ا3", "أَفْعال"),
])
def test_wazn_of(pattern, wazn):
    assert C.wazn_of(pattern) == wazn


def test_wazn_skeleton_folds_diacritics_and_hamza():
    assert C.wazn_skeleton("ٱِسْتِ1ْ2ا3") == "استفعال"
    assert C.wazn_skeleton("أَ1ْ2ا3") == "افعال"
    assert C.wazn_skeleton("إِ1ْ2ا3") == "افعال"


def test_gloss_exact_then_skeleton_then_none():
    gloss, tier = C.gloss_for("مَ1ْ2ُو3")
    assert tier == "exact" and "passive participle" in gloss
    gloss, tier = C.gloss_for("مَ1ْ2ُو3ِ")        # case vowel → no exact key, skeleton مفعول matches
    assert tier == "skeleton" and "مَفْعُول" in gloss
    assert C.gloss_for("1َوْ3") == (None, None)      # فَوْل: hollow, not in the closed list


def test_describe_pattern_weak_slots_and_filled_example():
    d = C.describe_pattern("1ا3", [["ق#ل", "قال"]])
    assert d["wazn"] == "فال"
    assert d["slots_present"] == [1, 3] and d["weak_slots"] == [2]
    assert d["filled"]["stem"] == "قال"
    assert [m["slot"] for m in d["filled"]["marks"]] == [1, 0, 3]
    d2 = C.describe_pattern("مَ1ْ2َ3َ", [["درس", "المَدْرَسَة"]])
    assert d2["filled"]["stem"] == "مَدْرَسَ" and d2["weak_slots"] == []
    assert C.describe_pattern("1َ2ْ3")["filled"] is None


def test_gloss_dicts_are_well_formed():
    # every exact key must be a wazn made of ف ع ل + template material (no slot digits)
    for k in list(C.WAZN_GLOSSES_EXACT) + list(C.WAZN_GLOSSES_SKELETON):
        assert not re.search(r"[1234]", k), k
        assert k.strip() == k and k, k
    for k in C.WAZN_GLOSSES_SKELETON:
        assert C.wazn_skeleton(k) == k, f"skeleton key {k!r} is not in skeleton form"


def test_js_wazn_map_mirrors_python():
    src = PAGE.read_text(encoding="utf-8")
    m = re.search(r"const CX_WAZN = (\{[^}]*\});", src)
    assert m, "CX_WAZN literal not found in the page"
    assert json.loads(m.group(1)) == C.WAZN_SLOT_LETTERS


def test_page_has_tab_03_and_keeps_tabs_01_02():
    src = PAGE.read_text(encoding="utf-8")
    assert 'data-panel="p-corpus"' in src and 'id="p-corpus"' in src
    assert 'data-panel="p-train"' in src and 'data-panel="p-explainer"' in src
    # the tab-03 module never touches tab-01 element ids
    module = src.split("/* TAB 3: corpus training.")[1]
    for forbidden in ("'#steps'", "'#pg-", "'#rail'", "'#playbar'", "'#text'", "'#status'"):
        assert forbidden not in module, forbidden


def test_validate_request_train_and_saved(tmp_path, monkeypatch):
    req = C.validate_request({"mode": "train", "params": {"max_patterns": "50", "use_diacritized_surface": 1},
                              "probe_text": "نص", "max_train_samples": "", "cache_policy": "auto"})
    assert req["params"] == {"max_patterns": 50, "use_diacritized_surface": True}
    assert req["max_train_samples"] is None and req["write_cache"] is True and req["verify"] is True
    with pytest.raises(ValueError):
        C.validate_request({"mode": "train", "probe_text": ""})
    with pytest.raises(ValueError):
        C.validate_request({"mode": "train", "probe_text": "x", "cache_policy": "sometimes"})
    with pytest.raises(ValueError):
        C.validate_request({"mode": "saved", "probe_text": "x", "saved_dir": "../etc"})
    with pytest.raises(ValueError):
        C.validate_request({"mode": "saved", "probe_text": "x", "saved_dir": "does_not_exist_xyz"})


def test_categorize_mirrors_emit_alpha_rules():
    vocab = {"[ROOT_كتب]": 1, "[PAT_1ِ2ا3]": 2, "[CLITICP_ال]": 3, "[CLITICE_ة]": 4, "[PREP_في]": 5}
    cat = lambda *a, **k: C.categorize(vocab, *a, **k)[0]  # noqa: E731
    assert cat(False, None, None, None, (), ()) == "lit_no_analysis"
    assert cat(True, None, "كتب", "1ِ2ا3", ("ال",), ()) == "root_pat"
    assert cat(True, None, "كتب", "1ِ2ا3", ("ال",), (), peeled=True) == "root_pat_peeled"
    assert cat(True, None, "درس", "1ِ2ا3", (), ()) == "lit_root_cut"
    assert cat(True, None, "كتب", "مَ1ْ2َ3", (), ()) == "lit_pattern_cut"
    assert cat(True, None, "كتب", "1ِ2ا3", ("و",), ()) == "lit_clitic_missing"
    assert cat(True, None, "كتب", "1ِ2ا3", (), ("ه",)) == "lit_clitic_missing"
    assert cat(True, "في", None, None, (), ()) == "prep"
    assert cat(True, "في", None, None, (), ("ه",)) == "lit_clitic_missing"
    assert cat(True, "على", None, None, (), ()) == "lit_clitic_missing"
    # the func group has its own prefix and category; a func surface is never looked up as [PREP_*]
    vocab["[FUNC_هذا]"] = 6
    assert cat(True, "هذا", None, None, (), (), particle_kind="func") == "func"
    assert cat(True, "هذا", None, None, (), ()) == "lit_clitic_missing"
    assert cat(True, "في", None, None, (), (), particle_kind="func") == "lit_clitic_missing"
    # clitic-only words: proclitic + pronoun tokens, nothing else
    vocab["[CLITICP_ل]"] = 7; vocab["[CLITICE_ه]"] = 8
    assert cat(True, None, None, None, ("ل",), ("ه",), clitic_only=True) == "clitic"
    assert cat(True, None, None, None, ("ل",), ("هم",), clitic_only=True) == "lit_clitic_missing"
    assert cat(True, None, None, None, ("ل",), (), clitic_only=True) == "lit_clitic_missing"
    assert cat(True, None, "", "", (), ()) == "lit_no_analysis"
    assert [k for k, _ in C.CATEGORIES] == list(C.CATEGORY_LABEL)


def test_list_sources_shape():
    s = C.list_sources()
    assert {"saved", "caches", "dataset", "tiers", "default_cache"} <= set(s)
    assert s["tiers"]["balanced"] == {"max_roots": 10000, "max_patterns": 4000}
    for d in s["saved"]:
        assert {"name", "vocab_size", "roots", "patterns", "config"} <= set(d)



def _entries():
    E = C.CorpusEntry
    return [
        E(word="الكتاب", analyzed=True, root="كتب", pattern="1ِ2ا3", pattern_raw="ال1ِ2ا3", stem="كِتاب", surface="الكِتاب", proclitics=("ال",)),
        E(word="كتب", analyzed=True, root="كتب", pattern="1َ2َ3َ", pattern_raw="1َ2َ3َ", stem="كَتَبَ", surface="كَتَبَ"),
        E(word="قال", analyzed=True, root="ق#ل", pattern="1ا3َ", pattern_raw="1ا3َ", stem="قالَ", surface="قالَ"),
        E(word="مدرسته", analyzed=True, root="درس", pattern="مَ1ْ2َ3َ", pattern_raw="مَ1ْ2َ3َت", stem="مَدْرَسَ", surface="مَدْرَسَتِهِ", enclitics=("ة", "ه")),
        E(word="في", analyzed=True, particle="في", surface="في"),
        E(word="وعليه", analyzed=True, particle="على", surface="وعليه", proclitics=("و",), enclitics=("ه",)),
        E(word="وهم", analyzed=True, particle="هم", particle_kind="func", surface="وهم", proclitics=("و",)),
        E(word="له", analyzed=True, clitic_only=True, surface="له", proclitics=("ل",), enclitics=("ه",)),
        E(word="أتكتب", analyzed=True, root="كتب", pattern="تَ1ْ2ُ3", pattern_raw="تَ1ْ2ُ3", stem="تَكْتُب", surface="أَتَكْتُب", proclitics=("أ",), peeled=True),
        E(word="مايكروسوفت", analyzed=False),
        E(word="جوجل", analyzed=False),
    ]


def _fake_tok(vocab, max_roots=100, max_patterns=100, min_root_freq=1, min_pattern_freq=1):
    from types import SimpleNamespace
    return SimpleNamespace(_vocab=vocab, max_roots=max_roots, max_patterns=max_patterns,
                           min_root_freq=min_root_freq, min_pattern_freq=min_pattern_freq)


def test_word_index_paths_records_and_query():
    from collections import Counter
    counts = Counter({"الكتاب": 50, "كتب": 5, "قال": 40, "مدرسته": 3, "في": 900, "وعليه": 7, "وهم": 6, "له": 4, "أتكتب": 2, "مايكروسوفت": 20, "جوجل": 1})
    W = C.WordCategoryIndex(_entries(), counts)
    # exclusive pre-pass paths that sum to the number of unique chunks; a peeled particle would count as peeled
    assert W.path_totals() == {"ROOT+PAT": 4, "PREP": 2, "FUNC": 1, "CLITIC": 1, "peeled": 1, "LIT": 2} and W.analyzed_count == 9
    assert [r[W.W] for r in W.rows[:3]] == ["في", "الكتاب", "قال"]          # sorted by occurrences
    rec = W.record("مدرسته")
    assert rec == {**rec, "path": "root_pat", "path_label": "ROOT+PAT", "analyzed": True, "root": "درس", "pattern": "مَ1ْ2َ3َ",
                   "pattern_raw": "مَ1ْ2َ3َت", "stem": "مَدْرَسَ", "surface": "مَدْرَسَتِهِ", "proclitics": [], "enclitics": ["ة", "ه"],
                   "category": None, "count": 3}
    assert W.record("زرافة") is None and W.cached("في")["path"] == "prep" and "surface" not in W.cached("في")
    assert W.cached("وهم")["path"] == "func" and W.record("وهم")["particle_kind"] == "func" and W.record("في")["particle_kind"] == "prep"
    assert W.cached("له")["path"] == "clitic" and W.record("له")["clitic_only"] is True and W.record("في")["clitic_only"] is False
    # path filters, aliases, full records, paging, substring
    assert W.query(path="lit")["total"] == 2 and [r["word"] for r in W.query(path="lit")["rows"]] == ["مايكروسوفت", "جوجل"]
    assert W.query(path="analyzed")["total"] == 9 and W.query(path="rejected")["total"] == 2
    assert W.query(path="func")["total"] == 1 and W.query(path="clitic")["total"] == 1
    assert W.query(path="peeled", full=True)["rows"][0]["proclitics"] == ["أ"]
    q = W.query(path="analyzed", limit=3, offset=3)
    assert q["total"] == 9 and [r["word"] for r in q["rows"]] == ["وعليه", "وهم", "كتب"]
    assert [r["word"] for r in W.query(q="كتب", path="root_pat")["rows"]] == ["كتب"]   # peeled أتكتب is on another path
    assert W.query(q="كتب")["total"] == 2 and W._filter_cache[0] == (None, None, "كتب", None)
    with pytest.raises(ValueError):
        W.query(path="nope")
    # categories need the vocab: cut the pattern of قال and keep everything else
    vocab = {"[ROOT_كتب]": 1, "[ROOT_ق#ل]": 2, "[ROOT_درس]": 3, "[PAT_1ِ2ا3]": 4, "[PAT_1َ2َ3َ]": 5, "[PAT_مَ1ْ2َ3َ]": 6,
             "[PAT_تَ1ْ2ُ3]": 7, "[CLITICP_ال]": 8, "[CLITICP_و]": 9, "[CLITICP_أ]": 10, "[CLITICE_ة]": 11, "[CLITICE_ه]": 12,
             "[PREP_في]": 13, "[PREP_على]": 14, "[FUNC_هم]": 15, "[CLITICP_ل]": 16}
    root_freq, pat_freq = Counter({"كتب": 3, "درس": 1, "ق#ل": 1}), Counter({"1ِ2ا3": 1, "1َ2َ3َ": 1, "1ا3َ": 1, "مَ1ْ2َ3َ": 1, "تَ1ْ2ُ3": 1})
    W.attach_vocab(_fake_tok(vocab), root_freq, pat_freq)
    assert W.counts == {"root_pat": 3, "root_pat_peeled": 1, "prep": 2, "func": 1, "clitic": 1, "lit_no_analysis": 2, "lit_root_cut": 0, "lit_pattern_cut": 1, "lit_clitic_missing": 0}
    assert W.cached("قال")["category"] == "lit_pattern_cut" and W.cached("قال")["path"] == "root_pat"
    s = W.summary()
    assert [p["unique"] for p in s["paths"]] == [4, 2, 1, 1, 1, 2] and s["analyzed"] == 9 and s["path_aliases"]["analyzed"] == ["root_pat", "prep", "func", "clitic", "peeled"]
    assert W.query(category="lit_pattern_cut", path="analyzed")["total"] == 1 and W.query(category="lit_pattern_cut", path="lit")["total"] == 0
    assert "never produced" in W.budget_reason("pattern", "1ُ2ُو3", vocab)["reason"]


def test_freq_index_search_kept_and_paging():
    from collections import Counter
    vocab = {"[ROOT_كتب]": 1, "[ROOT_ق#ل]": 2, "[PAT_مَ1ْ2ُو3]": 3, "[PAT_1ا2ِ3]": 4, "[CLITICP_ال]": 5, "[CLITICE_ه]": 6, "[PREP_في]": 7}
    F = C.FreqIndex.from_counters(
        Counter({"كتب": 9, "ق#ل": 4, "درس": 4, "س#ر": 1}), Counter({"مَ1ْ2ُو3": 7, "1ا2ِ3": 7, "1َ2َ3َ": 2, "مَ1ْ2ُو3َة": 1}),
        Counter({"ال": 20, "و": 3}), Counter({"ه": 5}), Counter({"في": 12}),
        {"root": {"كتب": ["الكتاب", "كتب"], "ق#ل": ["قال"]}, "pat": {"مَ1ْ2ُو3": ["مكتوب"]}}, vocab)
    assert F.sizes == {"root": 4, "pat": 4, "prc": 2, "enc": 1, "prep": 1, "func": 0} and F.kept == {"root": 2, "pat": 2, "prc": 1, "enc": 1, "prep": 1, "func": 0}
    page = F.query("root")
    assert [(r["rank"], r["key"], r["freq"], r["in_vocab"]) for r in page["rows"]] == [(1, "كتب", 9, True), (2, "درس", 4, False), (3, "ق#ل", 4, True), (4, "س#ر", 1, False)]
    assert page["max_freq"] == 9 and page["candidates"] == 4 and page["rows"][0]["words"] == ["الكتاب", "كتب"] and page["from_metadata"] is False
    # root search: a weak letter or # matches a masked radical
    assert [r["key"] for r in F.query("root", q="قول")["rows"]] == ["ق#ل"]
    assert F.query("root", q="#")["total"] == 4 and [r["key"] for r in F.query("root", q="س#ر")["rows"]] == ["س#ر"]   # # = any radical
    assert F.query("root", q="سير")["rows"][0]["key"] == "س#ر" and F.query("root", q="زرف")["total"] == 0
    # pattern search: CAMeL slots or wazn, tashkeel ignored; private match keys never leave the index
    hits = F.query("pat", q="مفعول")["rows"]
    assert [r["key"] for r in hits] == ["مَ1ْ2ُو3", "مَ1ْ2ُو3َة"] and hits[0]["wazn"] == "مَفْعُول" and not any(k.startswith("_") for k in hits[0])
    assert [r["key"] for r in F.query("pat", q="1ا2")["rows"]] == ["1ا2ِ3"] and [r["key"] for r in F.query("pat", q="فَعَلَ")["rows"]] == ["1َ2َ3َ"]
    # kept / cut filter, paging, other kinds
    assert [r["key"] for r in F.query("pat", kept=False)["rows"]] == ["1َ2َ3َ", "مَ1ْ2ُو3َة"] and F.query("pat", kept=True)["total"] == 2
    assert [r["rank"] for r in F.query("root", limit=2, offset=1)["rows"]] == [2, 3] and F.query("root", limit=2, offset=1)["total"] == 4
    assert [r["key"] for r in F.query("prc", q="و")["rows"]] == ["و"] and F.query("prep")["rows"][0]["in_vocab"] is True
    with pytest.raises(ValueError):
        F.query("nope")
    # saved mode: from vocab_metadata.json (examples are [rank?, word] pairs or bare words)
    meta = {"roots": {"كتب": {"freq": 9, "example_words": ["الكتاب"]}}, "patterns": {"مَ1ْ2ُو3": {"freq": 7, "examples": [["x", "مكتوب"], "مطلوب"]}},
            "proclitic_freq": {"ال": 20}, "enclitic_freq": {}, "prepositions": {"في": {"freq": 12}}}
    G = C.FreqIndex.from_metadata(meta, vocab)
    assert G.from_metadata and G.sizes == {"root": 1, "pat": 1, "prc": 1, "enc": 0, "prep": 1, "func": 0}
    H = C.FreqIndex.from_metadata({**meta, "func_words": {"هذا": {"freq": 30}}}, {**vocab, "[FUNC_هذا]": 8})
    assert H.sizes["func"] == 1 and H.query("func")["rows"][0] == {**H.query("func")["rows"][0], "key": "هذا", "freq": 30, "in_vocab": True}
    assert G.query("pat")["rows"][0]["words"] == ["مكتوب", "مطلوب"] and G.query("root")["rows"][0]["words"] == ["الكتاب"]


def test_page_nav_and_browsers_stay_out_of_the_step_rule():
    src = PAGE.read_text(encoding="utf-8")
    assert 'id="cx-nav"' in src and 'id="cx-nav-list"' in src and 'id="cx-setup"' in src and 'id="cx-words-mount"' in src
    # the global .step rule collapses any element with class "step" — nav entries must not use it
    assert "'nv-step'" in src and "'nv-sec'" in src and "#cx-nav a.step" not in src
    for fn in ("function cxBrowser(", "function cxRecordsBrowser(", "function cxNavBuild(", "function cxNavSync(", "RX.freq = d =>", "/api/corpus/freq?"):
        assert fn in src, fn

# ---------------------------------------------------------------------------
# Tier B — live bridge
# ---------------------------------------------------------------------------

TEXTS = [
    "والكتاب مفيد جداً، للولد الصغير.",
    "قال المعلم: الكتاب جديد وسيدرسه الطلاب في 2024!",
    "كتابهم على الطاولة وكتابك في الحقيبة، فهل رأيتهما؟ أتكتب سلمتكها",
    "تعمل شركة مايكروسوفت مع جوجل على الذكاء الاصطناعي.",
    "درس الطالب الدرس، والمدرس يدرّس في المدرسة، والدروس مفيدة.",
] * 3
PROBE = "قال المعلم: الكتاب جديد وسيدرسه الطلاب.\nللولد كتابهم."
STEP_IDS = ["config", "nfkc", "split", "chunks", "dedup", "cache", "ipc", "validate", "peel", "entries",
            "freq", "vocab", "reconstruction", "metadata", "encode", "decode", "verify"]


def _run(req):
    job = C.CorpusTraceJob(req)
    job.start()
    deadline = time.time() + 600
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.2)
    assert job.status == "done", "\n".join(job.log)
    return job


@pytest.mark.skipif(not _CAMEL_VENV.exists(), reason="needs .venv-camel with camel-tools")
def test_corpus_job_train_cache_search_save_load():
    cache_name = "_test_corpus_trace_cache"
    saved_name = "_test_corpus_trace_saved"
    cache_dir = C.TOKENIZERS_DIR / cache_name
    saved_dir = C.TOKENIZERS_DIR / saved_name
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(saved_dir, ignore_errors=True)
    base = dict(mode="train", params={"max_roots": 100, "max_patterns": 100, "min_root_freq": 1, "min_pattern_freq": 1},
                probe_text=PROBE, sample_size=10, seed=42, verify=True, dataset_name="injected",
                max_train_samples=None, cache_dir=cache_name, cache_policy="auto", write_cache=True, texts=TEXTS)
    jobs = []
    try:
        j1 = _run(base)
        jobs.append(j1)
        t = j1.trace
        assert [s["id"] for s in t["steps"]] == STEP_IDS
        by = {s["id"]: s["data"] for s in t["steps"]}
        assert by["cache"]["decision"] == "miss" and (cache_dir / "corpus_analysis.pkl").exists()
        assert [b["source"] for b in by["ipc"]["batches"]] == ["live pre-pass batch", "sample replay (live, for the cards below)"]
        assert by["validate"]["total"] == by["entries"]["totals"]["entries"] > 0
        assert by["validate"]["replay_disagreements"] == 0
        assert by["verify"] == {**by["verify"], "vocab_equal": True, "reconstruction_equal": True}
        assert by["encode"]["ipc_calls"] == 0 and by["decode"]["match_ignoring_spacing"] is True
        assert all("_scale" in d for d in by.values())
        assert any(v.get("wazn") for v in by["vocab"]["vocab"] if v["family"] == "pat")
        # every explained step agrees with the real helper — incl. the double-enclitic peeled
        # word سلمتكها (ة realized as ت before the pronoun), explained by _trace_enclitic_stack
        mism = [(s["id"], r.get("word")) for s in t["steps"] for k in ("words", "pass1", "pass2", "pass3")
                for r in (s["data"].get(k) or []) if isinstance(r, dict) and r.get("matches_real") is False]
        assert mism == [] and t["all_ok"] is True

        jp = _run({**base, "texts": TEXTS + ["زرافة جميلة في الحديقة"]})  # three chunks the cache lacks
        jobs.append(jp)
        byp = {s["id"]: s["data"] for s in jp.trace["steps"]}
        assert byp["cache"]["decision"] == "partial" and byp["cache"]["coverage"]["missing"] == 3
        assert byp["cache"]["coverage"]["missing_examples"] == ["زرافة", "جميلة", "الحديقة"]
        assert byp["ipc"]["total_unique"] == byp["validate"]["total"] and byp["ipc"]["total_batches"] == 1
        assert byp["ipc"]["batches"][0]["words"] == ["زرافة", "جميلة", "الحديقة"]     # only the missing chunks went over the pipe
        assert byp["verify"]["vocab_equal"] and byp["verify"]["reconstruction_equal"]   # the real train() agrees
        assert jp.words.record("زرافة") is not None and jp.words.record("الكتاب") is not None

        j2 = _run(base)  # the union cache written by jp still covers the original corpus
        jobs.append(j2)
        by2 = {s["id"]: s["data"] for s in j2.trace["steps"]}
        assert by2["cache"]["decision"] == "hit"
        assert [b["source"] for b in by2["ipc"]["batches"]] == ["sample replay (live, for the cards below)"]
        assert j2.tokenizer._vocab == j1.tokenizer._vocab

        idx = j2.index
        r = idx.search("كتب")
        assert r["hits"][0]["token"] == "[ROOT_كتب]" and r["hits"][0]["why"] == "exact"
        assert [s["token"] for s in r["encode"]["stream"]][1:-1] == ["[LIT_BEGIN]", "[CHAR_ك]", "[CHAR_ت]", "[CHAR_ب]", "[LIT_END]"]
        r = idx.search("المدرسة")
        assert [s["token"] for s in r["encode"]["stream"]] == ["<s>", "[CLITICP_ال]", "[ROOT_درس]", "[PAT_مَ1ْ2َ3َ]", "[CLITICE_ة]", "</s>"]
        assert any(h["token"] == "[PAT_مَ1ْ2َ3َ]" for h in r["hits"])
        r = idx.search("قول")  # weak letter matches the masked radical
        assert any(h["token"] == "[ROOT_ق#ل]" for h in r["hits"])
        r = idx.search("فاعل", family="pat")
        assert r["hits"] and all(h["family"] == "pat" for h in r["hits"]) and r["hits"][0]["why"] == "wazn"
        rid = j2.tokenizer._vocab["[ROOT_درس]"]
        assert idx.search(str(rid))["hits"][0]["token"] == "[ROOT_درس]"
        assert idx.search("", family="root", limit=5)["total"] == idx.families["root"]

        # word categories: every corpus chunk's category must agree with the tokens encode() emits
        W = j2.words
        assert W is not None and W.summary()["unique"] == len(W.rows) > 0
        # the validate / peel / entries counters are the store's exclusive path totals
        assert by2["validate"]["path_totals"] == W.path_totals() and by2["validate"]["analyzed"] == W.analyzed_count
        assert by2["peel"]["rescued"] == W.path_counts["peeled"] and by2["entries"]["browsable"] is True
        assert W.query(path="analyzed")["total"] + W.query(path="lit")["total"] == by2["validate"]["total"]
        full = W.query(path="peeled", full=True)["rows"]
        assert full and all(r["analyzed"] and r["peeled"] and r["surface"] for r in full)
        # the freq index reproduces the freq card's top rows and knows what the budget kept
        F = j2.freq
        assert F is not None and not F.from_metadata
        assert [(r["key"], r["freq"], r["words"]) for r in F.query("root", limit=5)["rows"]] == [(r["key"], r["freq"], r["words"]) for r in by2["freq"]["root_freq"][:5]]
        assert [r["key"] for r in F.query("pat", limit=5)["rows"]] == [r["key"] for r in by2["freq"]["pat_freq"][:5]]
        assert F.sizes["root"] == by2["freq"]["sizes"]["root_freq"] and F.query("root", q="قول")["rows"][0]["key"] == "ق#ل"
        assert F.query("pat", kept=True)["total"] == by2["vocab"]["budget_counts"]["pattern"]["kept"]
        for i, row in enumerate(W.rows):
            w = row[W.W]
            toks = [j2.tokenizer._reverse_vocab[i_] for i_ in j2.tokenizer.encode(w).input_ids]
            cat = C.CATEGORIES[W.cats[i]][0]
            is_lit, is_prep, is_rp = "[LIT_BEGIN]" in toks, any(t.startswith("[PREP_") for t in toks), any(t.startswith("[ROOT_") for t in toks)
            is_func = any(t.startswith("[FUNC_") for t in toks)
            is_clitic = bool(toks) and all(t.startswith(("[CLITICP_", "[CLITICE_")) for t in toks if t not in ("<s>", "</s>"))
            assert (cat.startswith("lit") and is_lit and not is_rp) or (cat == "prep" and is_prep) or (cat == "func" and is_func) \
                or (cat == "clitic" and is_clitic) or (cat.startswith("root_pat") and is_rp), (w, cat, toks)
        assert W.query("prep")["rows"] and all(r["category"] == "prep" for r in W.query("prep")["rows"])
        q = W.query(q="كتاب", limit=2); assert q["total"] >= 3 and len(q["rows"]) == 2 and W.query(q="كتاب", limit=2, offset=2)["rows"]
        with pytest.raises(ValueError):
            W.query("no_such_category")
        enc = C.encode_text(j2, "قال المعلم: الكتاب جديد وسيدرسه الطلاب في 2024! مايكروسوفت")
        assert enc["ipc_calls"] == 0 and enc["match_ignoring_spacing"] is True
        bycat = {c["chunk"]: c["category"] for c in enc["chunks"]}
        assert bycat["في"] == "prep" and bycat["مايكروسوفت"] == "lit_no_analysis" and bycat["قال"] == "root_pat"
        assert all(c["corpus"] for c in enc["chunks"] if c["chunk"] != "الطلاب" or True)  # every chunk of a corpus sentence has a corpus record
        tr_ = C.trace_word_process(j2, "مايكروسوفت")
        assert tr_["category"] == "lit_no_analysis" and tr_["peel"] is not None and tr_["stream"][1]["token"] == "[LIT_BEGIN]" and tr_["agrees_with_corpus"] is True
        assert tr_["prepass"]["path"] == "lit" and tr_["agrees_with_prepass"] is True
        tr_ = C.trace_word_process(j2, "بيتنا")   # not in the corpus: root never a candidate
        assert tr_["category"] == "lit_root_cut" and tr_["cached"] is None and "never produced" in tr_["vocab_check"]["root"]["reason"]
        assert tr_["prepass"] is None and tr_["agrees_with_prepass"] is None
        tr_ = C.trace_word_process(j2, "المدرسة")
        assert tr_["category"] == "root_pat" and tr_["vocab_check"]["pattern"]["in_vocab"] and [s["token"] for s in tr_["stream"]][1:-1] == ["[CLITICP_ال]", "[ROOT_درس]", "[PAT_مَ1ْ2َ3َ]", "[CLITICE_ة]"]
        with pytest.raises(ValueError):
            C.trace_word_process(j2, "2024")

        saved = C.save_trained(j2.tokenizer, saved_name)
        assert set(saved["files"]) >= {"config.json", "vocab.json", "reconstruction.pkl", "vocab_metadata.json"}
        with pytest.raises(FileExistsError):
            C.save_trained(j2.tokenizer, saved_name)

        j3 = _run(dict(mode="saved", saved_dir=saved_name, params={}, probe_text=PROBE, sample_size=10, seed=42, verify=True))
        jobs.append(j3)
        by3 = {s["id"]: s["data"] for s in j3.trace["steps"]}
        assert [s["id"] for s in j3.trace["steps"]] == STEP_IDS
        assert {k for k, d in by3.items() if d.get("unavailable")} == {"nfkc", "split", "chunks", "dedup", "ipc", "validate", "peel", "entries"}
        assert by3["vocab"]["vocab_size"] == j2.tokenizer.vocab_size and by3["vocab"]["from_metadata"] is True
        assert by3["verify"]["vocab_equal"] and by3["reconstruction"]["size"] == len(j2.tokenizer._reconstruction)
        assert j3.trace["all_ok"] is True
        assert j3.index.search("المدرسة")["encode"]["decoded"] == j2.index.search("المدرسة")["encode"]["decoded"]
        assert j3.words is None   # saved mode carries no per-word records
        assert j3.freq is not None and j3.freq.from_metadata and all(r["in_vocab"] for r in j3.freq.query("root", limit=500)["rows"])
        assert j3.freq.query("root")["rows"][0]["key"] == j2.freq.query("root")["rows"][0]["key"]
        tr3 = C.trace_word_process(j3, "المدرسة")
        assert tr3["category"] == "root_pat" and tr3["cached"] is None and "loaded vocab" in tr3["vocab_check"]["root"]["reason"]
    finally:
        for j in jobs:
            j.close()
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(saved_dir, ignore_errors=True)


@pytest.mark.skipif(not _CAMEL_VENV.exists(), reason="needs .venv-camel with camel-tools")
def test_real_prepass_partial_cache_reuse(tmp_path, monkeypatch):
    from arabic_eval.tokenizers import araroopat as A
    from arabic_eval.tokenizers.araroopat_backend import MorphAnalyzer
    sent: list = []
    real = MorphAnalyzer.analyze_many

    def spy(self, words, batch_size=256):
        sent.append(list(words))
        return real(self, words, batch_size=batch_size)

    monkeypatch.setattr(MorphAnalyzer, "analyze_many", spy)
    params = {"max_roots": 100, "max_patterns": 100, "min_root_freq": 1, "min_pattern_freq": 1}
    # (the tokenizers share the process-wide CamelBridge, reaped at exit — nothing to close here)
    t1 = A.AraRooPatTokenizer(**params)
    t1.train(TEXTS, cache_path=str(tmp_path))
    n_first = sum(len(b) for b in sent)
    assert n_first > 0 and (tmp_path / "corpus_analysis.pkl").exists()
    extra = TEXTS + ["زرافة جميلة في الحديقة"]
    sent.clear()
    t2 = A.AraRooPatTokenizer(**params)
    t2.train(extra, cache_path=str(tmp_path))
    assert sent == [["زرافة", "جميلة", "الحديقة"]]          # only the three chunks the cache lacked
    cached = pickle.load((tmp_path / "corpus_analysis.pkl").open("rb"))["entries"]
    assert {"زرافة", "جميلة", "الكتاب"} <= {e.word for e in cached}   # the union was written back
    sent.clear()
    t3 = A.AraRooPatTokenizer(**params, cache_corpus_analysis=False)
    t3.train(extra, cache_path=str(tmp_path))
    assert sum(len(b) for b in sent) == n_first + 3               # a fresh run analyzes everything
    assert t3._vocab == t2._vocab and t3._reconstruction == t2._reconstruction and t3._metadata["roots"] == t2._metadata["roots"]
    sent.clear()
    t4 = A.AraRooPatTokenizer(**params)
    t4.train(extra, cache_path=str(tmp_path))
    assert sent == [] and t4._vocab == t2._vocab                  # now a full hit


@pytest.mark.skipif(not _CAMEL_VENV.exists(), reason="needs .venv-camel with camel-tools")
def test_corpus_job_cancel_and_no_cache_write():
    cache_name = "_test_corpus_trace_cache_cancel"
    cache_dir = C.TOKENIZERS_DIR / cache_name
    shutil.rmtree(cache_dir, ignore_errors=True)
    job = C.CorpusTraceJob(dict(mode="train", params={}, probe_text=PROBE, sample_size=5, seed=1, verify=False,
                                dataset_name="injected", max_train_samples=None, cache_dir=cache_name,
                                cache_policy="ignore", write_cache=False, texts=TEXTS * 40))
    job.cancel()   # cancelled before the first stage check fires
    job.start()
    for _ in range(300):
        if job.status not in ("queued", "running"):
            break
        time.sleep(0.1)
    try:
        assert job.status == "cancelled", job.log
        assert not cache_dir.exists()
    finally:
        job.close()
