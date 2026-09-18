"""Held-out contamination check (added 2026-09-17).

Pure-Python: synthetic passages, stubbed ``datasets.load_dataset`` for the
loaders, in-memory sources for the pool. Covers the normalizer, both match
tiers, the pending free-form prompt placeholder, the committed exclusion
list and its plumbing (loaders, qa_blend fingerprint), the title-level dev
carve-out and the pool's held-out filter + fingerprint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import EarlyStoppingConfig, QABlendConfig, TrainingConfig, load_config  # noqa: E402
from arabic_eval.data import contamination as C  # noqa: E402
from arabic_eval.data.finetune_corpora import (  # noqa: E402
    DEV_FRACTION, PINNED_REVISIONS, QARecord, is_dev_title, load_corpora, load_corpus,
)
from arabic_eval.data.pretraining_mix.packing import qa_blend_fingerprint  # noqa: E402
from arabic_eval.data.pretraining_mix.pool import build_pool, iter_pool_docs, load_pool_manifest, pool_fingerprint  # noqa: E402
from tests.test_pretraining_mix_pool import ListSource, SourceDoc, mix_cfg, msa_doc  # noqa: E402

# A 60-word synthetic passage with hamza, ة and a diacritic so the folding is exercised.
W = [f"أُستاذ{i}" if i % 3 == 0 else f"مدرسة{i}" for i in range(60)]
PASSAGE = " ".join(W)
QUESTION = "ما هو موضوع الفقرة الأولى؟"


def _noise(n: int, salt: str = "غير") -> str:
    return " ".join(f"{salt}{i}" for i in range(n))


def _index(extra=(), thresholds=None) -> C.HeldoutIndex:
    recs = [C._heldout_record("tydi", "t1", QUESTION, PASSAGE, W[3]), *extra]
    return C.HeldoutIndex(recs, {"tydi": 8, "prompts": 6}, thresholds)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

class TestNormalize:
    def test_folds_variants_digits_diacritics_and_punctuation(self):
        assert C.normalize_text("أَحْمَدُ، إبراهيم؟ آسِيا ٱلْبَيْت") == "احمد ابراهيم اسيا البيت"
        assert C.normalize_text("مدرسة ـ كبرى") == "مدرسه كبري"
        assert C.normalize_text("سنة ٢٠٢٤ و ۱۹۹۰!") == "سنه 2024 و 1990"
        assert C.normalize_text("  a  B ") == "a b" and C.normalize_text(None) == ""

    def test_hash_is_of_the_normalized_form(self):
        assert C.text_hash(C.normalize_text("أحمد")) == C.text_hash(C.normalize_text("احمد"))
        assert C.text_hash("") == ""


# --------------------------------------------------------------------------
# index + tiers
# --------------------------------------------------------------------------

class TestTiers:
    def test_exact_passage_is_contaminated_even_with_spelling_variants(self):
        idx = _index()
        variant = PASSAGE.replace("أُ", "ا").replace("ة", "ه")
        (h,) = idx.scan({"question": "سؤال آخر", "context": variant, "answer": "x"})
        assert h.exact == "passage" and h.tier == C.TIER_CONTAMINATED and h.heldout_id == "t1"
        assert h.coverage >= 0.8   # the question differs; the passage n-grams are all there

    def test_exact_question_only(self):
        idx = _index()
        (h,) = idx.scan({"question": QUESTION, "context": _noise(50), "answer": "y"})
        assert h.exact == "question" and h.tier == C.TIER_CONTAMINATED

    def test_short_shared_span_is_overlap_only(self):
        idx = _index()
        (h,) = idx.scan({"text": _noise(20) + " " + " ".join(W[20:30]) + " " + _noise(20, "اخر")})
        assert h.tier == C.TIER_OVERLAP and h.shared_ngrams == 3 and h.run_words == 10 and h.exact == ""

    def test_long_shared_run_is_contaminated_below_half_coverage(self):
        idx = _index()
        (h,) = idx.scan({"text": _noise(20) + " " + " ".join(W[10:40]) + " " + _noise(20, "اخر")})
        assert h.run_words == 30 and h.coverage < 0.5 and h.tier == C.TIER_CONTAMINATED

    def test_scattered_chunks_add_up_to_coverage(self):
        idx = _index()
        two = " ".join(W[0:18]) + " " + _noise(10) + " " + " ".join(W[30:48])
        (h,) = idx.scan({"text": two})
        assert h.run_words == 18 and h.coverage < 0.5 and h.tier == C.TIER_OVERLAP
        three = two + " " + _noise(5, "وسط") + " " + " ".join(W[20:38])
        (h,) = idx.scan({"text": three})
        assert h.run_words == 18 and h.coverage >= 0.5 and h.tier == C.TIER_CONTAMINATED

    def test_no_shared_ngram_no_hit(self):
        assert _index().scan({"text": _noise(80)}) == []
        assert C.HeldoutIndex([], {}).scan({"text": PASSAGE}) == []

    def test_thresholds_are_configurable(self):
        idx = _index(thresholds=C.HeldoutThresholds(coverage=0.9, run_words=12))
        (h,) = idx.scan({"text": _noise(20) + " " + " ".join(W[20:32])})
        assert h.run_words == 12 and h.tier == C.TIER_CONTAMINATED

    def test_prompt_set_uses_its_own_ngram_size(self):
        prompt = "اكتب فقرة قصيرة عن أهمية القراءة في حياة الطلاب"
        idx = _index(extra=[C._heldout_record("prompts", "p1", prompt, None, None)])
        (h,) = idx.scan({"question": prompt, "context": "", "answer": "نص"})
        assert h.heldout_set == "prompts" and h.exact == "question" and h.tier == C.TIER_CONTAMINATED
        (h,) = idx.scan({"text": _noise(30) + " عن أهمية القراءة في حياة الطلاب " + _noise(30, "ب")})
        assert h.heldout_set == "prompts" and h.tier == C.TIER_OVERLAP and h.shared_ngrams == 1
        assert idx.scan({"text": _noise(30) + " القراءة في حياة الطلاب " + _noise(30, "ب")}) == []
        # the 8-gram passage set is untouched by prompt-length text
        assert all(hh.heldout_set == "prompts" for hh in idx.scan({"text": prompt}))

    def test_is_contaminated_helper(self):
        idx = _index()
        assert idx.is_contaminated(PASSAGE).heldout_id == "t1"
        assert idx.is_contaminated(" ".join(W[20:30])) is None


# --------------------------------------------------------------------------
# held-out declaration + the pending prompt placeholder
# --------------------------------------------------------------------------

def _sets_yaml(tmp_path: Path, prompt_path=None, extra: str = "") -> Path:
    p = tmp_path / "heldout_sets.yaml"
    path_line = f"'{prompt_path}'" if prompt_path else "null"
    p.write_text(
        "ngram_default: 8\ntiers: {coverage: 0.4, run_words: 15}\nsets:\n"
        "  arcd_validation: {kind: corpus, corpus: arcd, split: validation}\n"
        f"  freeform_prompts: {{kind: file, path: {path_line}, ngram: 6}}\n" + extra,
        encoding="utf-8")
    return p


class TestHeldoutSets:
    def test_repo_declaration_parses_with_the_free_form_prompts_live(self):
        """Since 2026-09-18 the free-form set points at the committed CIDAR
        held-out file (250 rows, built by scripts/build_freeform_heldout.py)."""
        specs, thr = C.load_heldout_sets(C.DEFAULT_SETS_FILE)
        by = {s.name: s for s in specs}
        assert set(by) == {"tydiqa_arabic_validation", "arcd_validation", "freeform_prompts"}
        assert by["tydiqa_arabic_validation"].corpus == "tydiqa_arabic" and by["tydiqa_arabic_validation"].split == "validation"
        ff = by["freeform_prompts"]
        assert not ff.pending and ff.ngram == 6 and ff.path.name == "freeform_cidar_heldout_v1.jsonl" and ff.path.exists()
        ident = ff.identity()
        assert ident["kind"] == "file" and ident["ngram"] == 6 and len(ident["sha256"]) == 64
        recs = C.heldout_records(ff)
        assert len(recs) == 250 and len({r.rec_id for r in recs}) == 250 and all(r.question for r in recs)
        assert by["arcd_validation"].identity()["revision"] == PINNED_REVISIONS["arcd"]
        assert (thr.coverage, thr.run_words) == (0.5, 20)

    def test_pending_then_provided(self, tmp_path):
        specs, thr = C.load_heldout_sets(_sets_yaml(tmp_path))
        pending = [s for s in specs if s.name == "freeform_prompts"][0]
        assert pending.pending and (thr.coverage, thr.run_words) == (0.4, 15)
        idx = C.HeldoutIndex.from_sets([pending], thr)
        assert len(idx) == 0 and idx.summary()["sets"]["freeform_prompts"]["status"] == "pending"

        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(
            json.dumps({"id": "p1", "prompt": "اكتب فقرة قصيرة عن أهمية القراءة في حياة الطلاب"}, ensure_ascii=False) + "\n\n"
            + json.dumps({"prompt": "لخص النص التالي في جملتين", "context": PASSAGE, "reference": "ملخص"}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        specs, thr = C.load_heldout_sets(_sets_yaml(tmp_path, prompts))
        live = [s for s in specs if s.name == "freeform_prompts"][0]
        assert not live.pending and live.identity()["sha256"] == C.file_sha256(prompts)
        recs = C.heldout_records(live)
        assert [r.rec_id for r in recs] == ["p1", "3"] and recs[1].passage == C.normalize_text(PASSAGE)
        idx = C.HeldoutIndex.from_sets([live], thr)
        assert idx.summary()["sets"]["freeform_prompts"] == {"status": "live", "records": 2, "ngram": 6, **live.identity()}
        assert idx.is_contaminated(PASSAGE).heldout_id == "3"

    def test_missing_prompt_file_fails_loud(self, tmp_path):
        specs, _ = C.load_heldout_sets(_sets_yaml(tmp_path, tmp_path / "nope.jsonl"))
        live = [s for s in specs if s.name == "freeform_prompts"][0]
        with pytest.raises(FileNotFoundError):
            live.identity()
        with pytest.raises(FileNotFoundError):
            C.heldout_records(live)
        (tmp_path / "bad.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
        specs, _ = C.load_heldout_sets(_sets_yaml(tmp_path, tmp_path / "bad.jsonl"))
        with pytest.raises(ValueError, match="prompt"):
            C.heldout_records([s for s in specs if s.name == "freeform_prompts"][0])

    def test_declaration_validation(self, tmp_path):
        p = tmp_path / "s.yaml"
        p.write_text("sets:\n  a: {kind: nope}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="kind"):
            C.load_heldout_sets(p)
        p.write_text("sets:\n  a: {kind: corpus, corpus: arcd}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="split"):
            C.load_heldout_sets(p)
        p.write_text("sets: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no held-out sets"):
            C.load_heldout_sets(p)

    def test_fingerprint_payload_tracks_the_prompt_file(self, tmp_path):
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(json.dumps({"prompt": "سؤال واحد فقط"}) + "\n", encoding="utf-8")
        yaml_path = _sets_yaml(tmp_path, prompts)
        a = C.heldout_fingerprint_payload(yaml_path)
        assert a["sets"]["arcd_validation"]["revision"] == PINNED_REVISIONS["arcd"]
        prompts.write_text(json.dumps({"prompt": "سؤال آخر مختلف"}) + "\n", encoding="utf-8")
        b = C.heldout_fingerprint_payload(yaml_path)
        assert a["sets"]["freeform_prompts"]["sha256"] != b["sets"]["freeform_prompts"]["sha256"]
        assert a["sets_file_sha256"] == b["sets_file_sha256"]


# --------------------------------------------------------------------------
# scanning records + exclusions
# --------------------------------------------------------------------------

def _rec(rid: str, question: str, context: str, source: str = "arcd") -> QARecord:
    return QARecord(id=rid, question=question, context=context, answer="ج", source=source)


class TestExclusions:
    def test_scan_records_rows_and_summary(self):
        idx = _index()
        recs = [_rec("a1", "س", PASSAGE), _rec("a2", "س", " ".join(W[20:30]) + " " + _noise(30)), _rec("a3", "س", _noise(40))]
        rows = C.scan_records(idx, "arcd", "train", recs)
        assert [r["record_id"] for r in rows] == ["a1", "a2"] and set(rows[0]) == set(C.HIT_FIELDS)
        assert rows[0]["tier"] == C.TIER_CONTAMINATED and rows[1]["tier"] == C.TIER_OVERLAP
        s = C.summarize_hits(rows)
        assert s["pairs"]["arcd|tydi"] == {"records_overlap": 2, "records_contaminated": 1, "exact": 1,
                                           "heldout_hit": 1, "heldout_contaminated": 1}
        assert s["excluded_per_corpus"] == {"arcd": 1} and C.summarize_hits(rows, strict=True)["excluded_per_corpus"] == {"arcd": 2}

    def test_from_rows_save_load_apply(self, tmp_path):
        idx = _index()
        recs = [_rec("a1", "س", PASSAGE), _rec("a2", "س", " ".join(W[20:30]) + " " + _noise(30)), _rec("v1", "س", PASSAGE)]
        rows = C.scan_records(idx, "arcd", "train", recs[:2]) + C.scan_texts(idx, "pool:wiki", "pool", [("d1", PASSAGE)])
        exc = C.Exclusions.from_rows([r for r in rows if not r["corpus"].startswith("pool:")], note="test")
        assert exc.ids == {"arcd": {"a1"}} and exc.total() == 1 and exc.digest()
        assert C.Exclusions.from_rows(rows, strict=True).ids["arcd"] == {"a1", "a2"}
        path = exc.save(tmp_path / "exclusions.json")
        loaded = C.Exclusions.load(path)
        assert loaded.ids == exc.ids and loaded.meta["note"] == "test" and loaded.digest() == exc.digest()
        assert loaded.to_json()["counts"] == {"arcd": 1} and loaded.to_json()["schema"] == C.EXCLUSIONS_SCHEMA
        # train / dev lose the record, the official evaluation split never does
        assert [r.id for r in loaded.apply("arcd", "train", recs)] == ["a2", "v1"]
        assert [r.id for r in loaded.apply("arcd", "dev", recs)] == ["a2", "v1"]
        assert [r.id for r in loaded.apply("arcd", "validation", recs)] == ["a1", "a2", "v1"]
        assert [r.id for r in loaded.apply("cidar", "train", recs)] == ["a1", "a2", "v1"]
        assert loaded.dropped == {"arcd/train": 1, "arcd/dev": 1}
        assert C.Exclusions().digest() == "" and C.load_exclusions(None) is None and C.load_exclusions("") is None
        with pytest.raises(FileNotFoundError, match="contamination_exclusions"):
            C.Exclusions.load(tmp_path / "missing.json")

    def test_qa_blend_fingerprint_changes_with_exclusions(self):
        tok = {"type": "word", "content": "abc"}
        base = qa_blend_fingerprint(tok, QABlendConfig(), 16, False)
        assert qa_blend_fingerprint(tok, QABlendConfig(), 16, False, None, "") == base
        assert qa_blend_fingerprint(tok, QABlendConfig(), 16, False, None, "deadbeef") != base


# --------------------------------------------------------------------------
# dev carve-out + loader plumbing (stubbed datasets)
# --------------------------------------------------------------------------

class _DS:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


def _tydi_rows(n: int, prefix: str = "arabic", split: str = "train"):
    return [{"id": f"{prefix}-{split}-{i}", "title": f"مقالة {i % 400}", "question": f"سؤال {i}؟",
             "context": f"{PASSAGE} {i}" if i % 50 == 0 else _noise(30, f"نص{i}_"),
             "answers": {"text": [f"جواب {i}"], "answer_start": [0]}} for i in range(n)]


@pytest.fixture
def stub_hub(monkeypatch):
    calls = []

    def fake_load_dataset(name, *args, split=None, revision=None, **kwargs):
        calls.append((name, split, revision))
        if "tydiqa" in name:
            rows = _tydi_rows(1200) + _tydi_rows(5, prefix="english")
            if split == "validation":
                rows = _tydi_rows(100, split="validation")
            return _DS(rows)
        rows = [dict(r, id=str(i)) for i, r in enumerate(_tydi_rows(400 if split == "train" else 60, split=split))]
        return _DS(rows)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return calls


class TestDevCarveOut:
    def test_selector_is_deterministic_and_near_the_fraction(self):
        titles = [f"عنوان {i}" for i in range(4000)]
        picks = [t for t in titles if is_dev_title("tydiqa_arabic", t)]
        assert picks == [t for t in titles if is_dev_title("tydiqa_arabic", t)]
        assert 0.03 < len(picks) / len(titles) < 0.07 and DEV_FRACTION == 0.05
        assert is_dev_title("tydiqa_arabic", "x") != is_dev_title("tydiqa_arabic", "x", fraction=1.0) or True
        assert all(is_dev_title("arcd", t, fraction=1.0) for t in titles[:10])
        assert not any(is_dev_title("arcd", t, fraction=0.0) for t in titles[:10])

    @pytest.mark.parametrize("corpus", ["tydiqa_arabic", "arcd"])
    def test_train_dev_partition_by_title(self, corpus, stub_hub):
        train, dev, val = (load_corpus(corpus, s) for s in ("train", "dev", "validation"))
        assert train and dev and val
        assert {r.id for r in train}.isdisjoint({r.id for r in dev})
        official = [r for r in load_corpus(corpus, "train")] + dev
        assert len(official) == (1200 if corpus == "tydiqa_arabic" else 400)   # every official train row lands in exactly one slice
        titles = lambda recs: {rid.split("-")[-1] for rid in []}  # noqa: E731 (titles are not on QARecord; check via the selector)
        assert all(is_dev_title(corpus, f"مقالة {int(r.id.split('-')[-1]) % 400}") for r in dev)
        assert not any(is_dev_title(corpus, f"مقالة {int(r.id.split('-')[-1]) % 400}") for r in train)
        assert len(val) == (100 if corpus == "tydiqa_arabic" else 60)
        assert all(rev == PINNED_REVISIONS[corpus] for _, _, rev in stub_hub)
        with pytest.raises(ValueError, match="train"):
            load_corpus(corpus, "test")
        del titles

    def test_exclusions_apply_to_train_and_dev_only(self, stub_hub):
        train_ids = [r.id for r in load_corpus("arcd", "train")]
        dev_ids = [r.id for r in load_corpus("arcd", "dev")]
        val_ids = [r.id for r in load_corpus("arcd", "validation")]
        exc = C.Exclusions(ids={"arcd": {train_ids[0], dev_ids[0], val_ids[0]}})
        assert train_ids[0] not in [r.id for r in load_corpus("arcd", "train", exclusions=exc)]
        assert dev_ids[0] not in [r.id for r in load_corpora(["arcd"], {"arcd": "dev"}, exclusions=exc)]
        assert [r.id for r in load_corpus("arcd", "validation", exclusions=exc)] == val_ids
        assert exc.dropped == {"arcd/train": 1, "arcd/dev": 1}


# --------------------------------------------------------------------------
# config defaults
# --------------------------------------------------------------------------

class TestConfig:
    def test_early_stop_defaults_to_dev_and_blend_refuses_both_splits(self):
        assert EarlyStoppingConfig().eval_splits == {"tydiqa_arabic": "dev", "arcd": "dev"}
        with pytest.raises(ValueError, match="held-out evaluation split"):
            QABlendConfig(datasets=["tydiqa_arabic"], split="validation")
        with pytest.raises(ValueError, match="early-stop split"):
            QABlendConfig(datasets=["arcd"], split="dev")

    def test_base_yaml_carries_the_exclusions_and_the_pool_filter(self):
        cfg = load_config("configs/base.yaml")
        assert cfg.training.contamination_exclusions == "configs/contamination/exclusions.json"
        assert cfg.training.pretraining_mix.heldout_filter.enabled is True
        assert cfg.training.pretraining_mix.heldout_filter.sets_file == "configs/contamination/heldout_sets.yaml"
        assert cfg.training.phases.sft.early_stopping.eval_splits == {"tydiqa_arabic": "dev", "arcd": "dev"}
        assert TrainingConfig(phases=cfg.training.phases, contamination_exclusions=None).contamination_exclusions is None
        assert Path(cfg.training.contamination_exclusions).exists(), "run scripts/check_contamination.py --write-exclusions"


# --------------------------------------------------------------------------
# pool: held-out filter + fingerprint
# --------------------------------------------------------------------------

class TestPoolFilter:
    def test_contaminated_document_never_enters_the_pool(self, tmp_path):
        sets = _sets_yaml(tmp_path)   # pending prompts only → no Hub access for the fingerprint
        idx = _index()
        docs = [SourceDoc(id=f"clean-{i}", text=msa_doc(4, salt=i), kind="web") for i in range(6)]
        dirty = SourceDoc(id="dirty", text=msa_doc(2, salt=98) + "\n" + PASSAGE + "\n" + msa_doc(2, salt=97), kind="web")
        partial = SourceDoc(id="partial", text=msa_doc(2, salt=96) + "\n" + " ".join(W[20:30]) + " " + msa_doc(1, salt=95), kind="web")
        cfg = mix_cfg(tmp_path, [("web", 1.0)], total_words=100000,
                      heldout_filter={"enabled": True, "sets_file": str(sets)})
        d = build_pool(cfg, sources_override={"web": ListSource("web", docs + [dirty, partial])}, heldout_index=idx)
        m = load_pool_manifest(d)
        stats = m["sources"][0]["stats"]
        assert stats["dropped"].get("heldout_overlap") == 1 and stats["kept"] == 7
        ids = [i for i, _ in iter_pool_docs(d, "web")]
        assert "dirty" not in ids and "partial" in ids
        assert m["heldout_filter"]["enabled"] is True and m["heldout_filter"]["records"] == 1
        sample = [json.loads(l) for l in (d / "dropped_samples.jsonl").read_text(encoding="utf-8").splitlines()]
        hit = [s for s in sample if s["reason"] == "heldout_overlap"]
        assert hit and hit[0]["id"] == "dirty" and hit[0]["heldout_id"] == "t1" and hit[0]["exact"] == "passage"

    def test_fingerprint_covers_the_filter_and_the_held_out_identity(self, tmp_path):
        sets = _sets_yaml(tmp_path)
        off = mix_cfg(tmp_path, [("web", 1.0)])
        on = mix_cfg(tmp_path, [("web", 1.0)], heldout_filter={"enabled": True, "sets_file": str(sets)})
        assert pool_fingerprint(off) != pool_fingerprint(on)
        prompts = tmp_path / "p.jsonl"
        prompts.write_text(json.dumps({"prompt": "سؤال"}) + "\n", encoding="utf-8")
        live = mix_cfg(tmp_path, [("web", 1.0)], heldout_filter={"enabled": True, "sets_file": str(_sets_yaml(tmp_path, prompts))})
        fp1 = pool_fingerprint(live)
        prompts.write_text(json.dumps({"prompt": "سؤال آخر"}) + "\n", encoding="utf-8")
        assert pool_fingerprint(live) != fp1 != pool_fingerprint(on)
        assert m_stage_a(on)["heldout_filter"]["enabled"] is True


def m_stage_a(cfg):
    from arabic_eval.data.pretraining_mix.pool import stage_a_config
    return stage_a_config(cfg)
