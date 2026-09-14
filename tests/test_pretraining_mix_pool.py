"""Stage A of the pretraining mix: filters, dedup, pool builder, config.

No network: sources are injected through ``build_pool(sources_override=)``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import (  # noqa: E402
    MixQualityConfig, MixSourceConfig, PretrainingMixConfig, PhaseConfig, PhasesConfig,
    TrainingConfig, EarlyStoppingConfig,
)
from arabic_eval.data.pretraining_mix.dedup import MinHashDeduper, ParagraphDeduper  # noqa: E402
from arabic_eval.data.pretraining_mix.filters import (  # noqa: E402
    DIALECT_MARKERS, dialect_marker_score, normalize_document, quality_reason,
    sample_sentences, strip_wikipedia_sections, truncate_words,
)
from arabic_eval.data.pretraining_mix.pool import (  # noqa: E402
    build_pool, iter_pool_docs, load_pool_manifest, pool_fingerprint,
)
from arabic_eval.data.pretraining_mix.sources import BaseSource, SourceDoc  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: synthetic Arabic prose
# --------------------------------------------------------------------------

MSA_SENTENCES = [
    "تعتبر مدينة القاهرة من أكبر المدن في العالم العربي من حيث عدد السكان والمساحة.",
    "وقد أشار الوزير في تصريح صحفي إلى أن الحكومة تعتزم إطلاق برنامج جديد لدعم الشركات الناشئة.",
    "يقع نهر النيل في شمال شرق أفريقيا ويعد من أطول الأنهار في العالم على الإطلاق.",
    "تأسست الجامعة في مطلع القرن العشرين وتضم اليوم عشرات الكليات والمعاهد المتخصصة.",
    "شهدت المنطقة خلال العقود الماضية تحولات اقتصادية واجتماعية عميقة أثرت في بنية المجتمع.",
    "تعد اللغة العربية إحدى اللغات الأكثر انتشارا في العالم ويتحدث بها مئات الملايين.",
]

EGYPTIAN = "عايز اروح البيت دلوقتي بس مش عارف ازاي، وكده يعني احنا هنروح فين علشان نجيب الحاجات بتاعت البيت"
LEVANTINE = "شو بدك تعمل بكرا يا زلمة؟ والله ما بعرف، هيك الدنيا كتير صعبة هلق ورح نشوف"
GULF = "شنو رايك بهالموضوع؟ اني ما اعرف شي عنه، شلون نسويها وايد صعبة يا هسه"
MAGHREBI = "واش كاين شي حاجة جديدة؟ بزاف ديال الناس غادي يجيو دابا كيفاش نديرو"


def msa_doc(n_paragraphs: int, salt: int = 0) -> str:
    """Distinct MSA-ish document: rotate sentences and stamp a salt so
    MinHash never sees two identical docs by accident."""
    paras = []
    for i in range(n_paragraphs):
        s = MSA_SENTENCES[(i + salt) % len(MSA_SENTENCES)]
        paras.append(f"{s} الفقرة رقم {salt}-{i} في هذه الوثيقة التجريبية المخصصة للاختبار.")
    return "\n".join(paras)


class ListSource(BaseSource):
    """In-memory source; ``iter_docs`` ignores the shuffle (deterministic order)."""

    def __init__(self, name: str, docs: List[SourceDoc], kind: str = "web") -> None:
        super().__init__({})
        self.name = name
        self.kind = kind
        self._docs = docs

    def iter_docs(self, seed: int, shuffle_buffer: int) -> Iterator[SourceDoc]:
        yield from self._docs


def mix_cfg(tmp_path: Path, sources, total_words=1000, **overrides) -> PretrainingMixConfig:
    base = dict(
        cache_dir=str(tmp_path / "cache"),
        sources=[MixSourceConfig(name=n, share=s) for n, s in sources],
        pool={"total_words": total_words, "stream_shuffle_buffer": 0},
        quality={"min_words": 20, "max_words": 200},
    )
    base.update(overrides)
    return PretrainingMixConfig(**base)


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

def test_normalize_document_keeps_paragraph_lines():
    raw = "  الفقرة   الأولى \r\n\r\n\n الفقرة\tالثانية ـــ  \n"
    out = normalize_document(raw)
    assert out == "الفقرة الأولى\nالفقرة الثانية"


def test_normalize_document_alef_and_diacritics_opt_in():
    raw = "إِنَّ الأَمْرَ"
    assert normalize_document(raw) == raw
    assert normalize_document(raw, remove_diacritics=True, normalize_alef=True) == "ان الامر"


@pytest.mark.parametrize("text,reason", [
    ("كلمة " * 5, "too_short"),
    ("Epistle Reading Document Actions Home Page\n" * 3 + msa_doc(3), "latin_heavy"),
    ("\n".join(["قائمة", "رئيسية", "تسجيل", "دخول"] * 10 + [msa_doc(2)]), "short_lines"),
    ("\n".join([MSA_SENTENCES[0]] * 8 + [msa_doc(2)]), "dup_lines"),
    (" | ".join(["•", "►"] * 40) + "\n" + msa_doc(3), "symbol_heavy"),
    (msa_doc(4), None),
])
def test_quality_reason(text, reason):
    cfg = MixQualityConfig(min_words=20)
    assert quality_reason(normalize_document(text), cfg) == reason


def test_strip_wikipedia_sections_cuts_at_header_not_first_line():
    text = "المراجع\nنص المقال هنا بالتفصيل.\nفقرة ثانية.\nالمراجع:\nمرجع 1\nمرجع 2"
    out, cut = strip_wikipedia_sections(text, ["المراجع"])
    assert cut and out == "المراجع\nنص المقال هنا بالتفصيل.\nفقرة ثانية."
    out, cut = strip_wikipedia_sections("أ\nب", ["المراجع"])
    assert not cut and out == "أ\nب"


def test_truncate_words_at_paragraph_boundary():
    text = "a b c\nd e f\ng h"
    assert truncate_words(text, 6) == ("a b c\nd e f", True)
    assert truncate_words(text, 4) == ("a b c", True)
    assert truncate_words(text, 100) == (text, False)
    assert truncate_words("a b c d e", 2) == ("a b", True)  # single huge paragraph → hard cut


@pytest.mark.parametrize("text", [EGYPTIAN, LEVANTINE, GULF, MAGHREBI])
def test_dialect_marker_score_flags_dialect(text):
    s = dialect_marker_score(text)
    assert s.n_markers >= 3 and s.per_1k_words > 100


def test_dialect_marker_score_clean_on_msa_with_fold_collisions():
    # إلى→الي, آية→ايه, البدو, لكان, هول, كيما, توًّا must NOT match.
    text = ("وقد أشار إلى أن الآية الكريمة تدل على ذلك، وكان البدو يعيشون في الصحراء، "
            "فلو كان ذلك لكان أفضل، وهول الموقف كيما يقال، وتوًّا وصل الخبر " + " ".join(MSA_SENTENCES))
    assert dialect_marker_score(text).n_markers == 0


def test_dialect_marker_peels_proclitic():
    s = dialect_marker_score("وعشان كده فمش هنروح")
    assert s.matched["عشان"] == 1 and s.matched["مش"] == 1 and s.matched["كده"] == 1


def test_dialect_marker_table_has_no_duplicates_across_regions():
    seen = set()
    for entries in DIALECT_MARKERS.values():
        for e in entries:
            assert e not in seen, e
            seen.add(e)


def test_sample_sentences_is_deterministic_and_bounded():
    text = ". ".join(MSA_SENTENCES * 3)
    a = sample_sentences(text, 4, 5)
    assert len(a) == 4 and a == sample_sentences(text, 4, 5)
    assert sample_sentences("قصير جدا", 4, 5) == []


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------

def test_paragraph_deduper_removes_repeats_across_and_within_docs():
    d = ParagraphDeduper(min_words=3)
    assert d.filter("a b c d\nx y z w") == ("a b c d\nx y z w", 0)
    assert d.filter("a b c d\nnew line here\nnew line here") == ("new line here", 2)
    assert d.filter("ab\nab") == ("ab\nab", 0)  # short lines untouched


def test_minhash_deduper_detects_near_duplicates_only():
    d = MinHashDeduper(num_perm=128, shingle_words=3, threshold=0.8)
    base = msa_doc(6)
    assert d.check_and_add("wiki:1", base) is None
    assert d.check_and_add("web:1", base + "\nسطر إضافي واحد فقط") == "wiki:1"
    assert d.check_and_add("web:2", msa_doc(6, salt=3)) is None
    assert len(d) == 2


# --------------------------------------------------------------------------
# pool builder
# --------------------------------------------------------------------------

def _docs(prefix: str, n: int, paragraphs: int = 4, kind: str = "web", salt0: int = 0) -> List[SourceDoc]:
    return [SourceDoc(id=f"{prefix}{i}", text=msa_doc(paragraphs, salt=salt0 + i), kind=kind) for i in range(n)]


def test_build_pool_reaches_word_targets_in_priority_order_and_writes_artifacts(tmp_path):
    cfg = mix_cfg(tmp_path, [("web_a", 0.7), ("wiki_b", 0.3)], total_words=1000,
                  dedup={"priority": ["wiki_b"]})
    sources = {
        "web_a": ListSource("web_a", _docs("a", 60, salt0=100)),
        "wiki_b": ListSource("wiki_b", _docs("b", 60, kind="wikipedia", salt0=500)),
    }
    out = build_pool(cfg, sources_override=sources)
    manifest = load_pool_manifest(out)
    assert manifest["complete"] and manifest["all_targets_reached"]
    assert [s["name"] for s in manifest["sources"]] == ["wiki_b", "web_a"]  # priority first
    for s in manifest["sources"]:
        st = s["stats"]
        assert st["reached_target"] and st["kept_words"] >= st["target_words"]
        assert st["kept_words"] - st["target_words"] < 200  # overshoot ≤ one doc
    a_words = sum(len(t.split()) for _, t in iter_pool_docs(out, "web_a"))
    b_words = sum(len(t.split()) for _, t in iter_pool_docs(out, "wiki_b"))
    assert a_words >= 700 and b_words >= 300
    assert (out / "dropped_dialect.csv").exists() and (out / "dropped_samples.jsonl").exists()
    # reuse without rebuild
    assert build_pool(cfg, sources_override=sources) == out


def test_build_pool_drop_reasons_and_cross_source_dedup(tmp_path):
    # Long paragraphs, each edited by one word in the web copy: exact
    # paragraph hashes all differ (paragraph dedup can't catch it) but the
    # shingle Jaccard stays ~0.87, so it's MinHash's job. LSH is
    # probabilistic near its threshold, so the fixture uses 0.7 to stay
    # clear of the band edge (the default 0.8 misses ~1/3 of 0.87 pairs).
    long_paras = [
        MSA_SENTENCES[i] + " " + " ".join(f"مفردة{i}{j}" for j in range(60)) for i in range(6)
    ]
    shared = "\n".join(long_paras)
    near_dup = "\n".join(p + " تعديل" for p in long_paras)
    cfg = mix_cfg(tmp_path, [("wiki", 0.5), ("web", 0.5)], total_words=2000,
                  dedup={"priority": ["wiki", "web"], "minhash": {"threshold": 0.7}})
    sources = {
        "wiki": ListSource("wiki", [
            SourceDoc(id="w0", text=shared + "\nالمراجع\nمرجع أول\nمرجع ثان", kind="wikipedia"),
            *_docs("w", 10, kind="wikipedia", salt0=20),
        ], kind="wikipedia"),
        "web": ListSource("web", [
            SourceDoc(id="dup", text=near_dup, kind="web"),                       # near-dup of wiki w0
            SourceDoc(id="dia", text="\n".join([EGYPTIAN, LEVANTINE, GULF, MAGHREBI]), kind="web"),
            SourceDoc(id="short", text="نص قصير جدا", kind="web"),
            SourceDoc(id="latin", text="Hello world this is english " * 20, kind="web"),
            *_docs("x", 20, paragraphs=5, salt0=40),
        ]),
    }
    out = build_pool(cfg, sources_override=sources)
    manifest = load_pool_manifest(out)
    by = {s["name"]: s["stats"] for s in manifest["sources"]}
    assert by["wiki"]["wiki_sections_cut"] == 1
    dropped = by["web"]["dropped"]
    assert dropped["near_duplicate"] == 1
    assert dropped["dialect_markers"] == 1
    assert dropped["too_short"] == 1
    assert dropped["latin_heavy"] == 1
    kept_ids = [i for i, _ in iter_pool_docs(out, "web")]
    assert "dup" not in kept_ids and "dia" not in kept_ids
    samples = [json.loads(l) for l in (out / "dropped_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [i for i, _ in iter_pool_docs(out, "wiki")][0] == "w0"   # the wiki copy survived
    dup_sample = [s for s in samples if s["reason"] == "near_duplicate" and s["source"] == "web"][0]
    assert dup_sample["id"] == "dup" and dup_sample["duplicate_of"].startswith("wiki#1:w0")
    rows = (out / "dropped_dialect.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 2 and rows[1].startswith("web,dia,")
    assert {s["reason"] for s in samples} >= {"near_duplicate", "dialect_markers", "too_short", "latin_heavy"}


def test_build_pool_paragraph_dedup_strips_boilerplate_from_docs(tmp_path):
    boiler = "جميع الحقوق محفوظة لموقعنا الإلكتروني الرسمي هذا العام"
    cfg = mix_cfg(tmp_path, [("web", 1.0)], total_words=300)
    docs = [SourceDoc(id=f"d{i}", text=boiler + "\n" + msa_doc(4, salt=i), kind="web") for i in range(10)]
    out = build_pool(cfg, sources_override={"web": ListSource("web", docs)})
    st = load_pool_manifest(out)["sources"][0]["stats"]
    assert st["paragraphs_removed"] >= 1
    texts = [t for _, t in iter_pool_docs(out, "web")]
    assert texts[0].startswith(boiler) and all(boiler not in t for t in texts[1:])


def test_build_pool_calibration_mode_streams_fixed_docs(tmp_path):
    cfg = mix_cfg(tmp_path, [("web", 1.0)], total_words=50)
    out = build_pool(cfg, out_dir=tmp_path / "calib", sources_override={"web": ListSource("web", _docs("d", 30))},
                     doc_limit_per_source=12, stop_at_target=False)
    st = load_pool_manifest(out)["sources"][0]["stats"]
    assert st["streamed"] == 12 and st["stop_reason"] == "doc_limit"
    assert load_pool_manifest(out)["calibration"] is True


def test_build_pool_camel_did_uses_injected_scorer(tmp_path):
    class FakeScorer:
        n_sentences_scored = 0
        def is_dialect(self, text):
            return "سنتينل" in text
    cfg = mix_cfg(tmp_path, [("web", 1.0)], total_words=200,
                  msa_filter={"camel_did": {"enabled": True}})
    docs = _docs("d", 6) + [SourceDoc(id="c", text=msa_doc(4, salt=99) + "\n" + "سنتينل " * 5, kind="web")]
    docs = [docs[-1]] + docs[:-1]
    out = build_pool(cfg, sources_override={"web": ListSource("web", docs)}, did_scorer=FakeScorer())
    assert load_pool_manifest(out)["sources"][0]["stats"]["dropped"]["dialect_camel"] == 1


def test_build_pool_source_exhausted_marks_target_unreached(tmp_path):
    cfg = mix_cfg(tmp_path, [("web", 1.0)], total_words=100_000)
    out = build_pool(cfg, sources_override={"web": ListSource("web", _docs("d", 5))})
    m = load_pool_manifest(out)
    assert m["complete"] and not m["all_targets_reached"]
    assert m["sources"][0]["stats"]["stop_reason"] == "source_exhausted"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_fingerprint_ignores_stage_b_fields(tmp_path):
    a = mix_cfg(tmp_path, [("web", 1.0)])
    b = mix_cfg(tmp_path, [("web", 1.0)], block_size=1024, token_budget=5, consume_sequentially=False)
    c = mix_cfg(tmp_path, [("web", 1.0)], quality={"min_words": 21})
    assert pool_fingerprint(a) == pool_fingerprint(b) != pool_fingerprint(c)


def test_config_rejects_bad_shares_and_priority(tmp_path):
    with pytest.raises(ValueError, match="sum to 1.0"):
        mix_cfg(tmp_path, [("a", 0.5), ("b", 0.3)])
    with pytest.raises(ValueError, match="duplicate"):
        mix_cfg(tmp_path, [("a", 0.5), ("a", 0.5)])
    with pytest.raises(ValueError, match="priority"):
        mix_cfg(tmp_path, [("a", 1.0)], dedup={"priority": ["zzz"]})


def _phase(**kw) -> PhaseConfig:
    base = dict(datasets=["arabic_squad"], trainable_parameters=["*"], steps=10, learning_rate=1e-4,
                batch_size=2, loss_target="full_sequence", max_length=512)
    base.update(kw)
    return PhaseConfig(**base)


def test_training_config_mix_phase_rules(tmp_path):
    mix = mix_cfg(tmp_path, [("web", 1.0)])
    sft = _phase(datasets=["arcd"], loss_target="answer_only", early_stopping=EarlyStoppingConfig())
    ok = TrainingConfig(phases=PhasesConfig(
        embedding_alignment=_phase(datasets=["pretraining_mix"]),
        warmup=_phase(datasets=["pretraining_mix"]), sft=sft), pretraining_mix=mix)
    assert ok.phases_using_mix() == ["embedding_alignment", "warmup"]
    with pytest.raises(ValueError, match="not configured"):
        TrainingConfig(phases=PhasesConfig(embedding_alignment=_phase(datasets=["pretraining_mix"]),
                                           warmup=_phase(), sft=sft))
    with pytest.raises(ValueError, match="sole dataset"):
        TrainingConfig(phases=PhasesConfig(embedding_alignment=_phase(datasets=["pretraining_mix", "arcd"]),
                                           warmup=_phase(), sft=sft), pretraining_mix=mix)
    with pytest.raises(ValueError, match="full_sequence"):
        TrainingConfig(phases=PhasesConfig(embedding_alignment=_phase(datasets=["pretraining_mix"], loss_target="answer_only"),
                                           warmup=_phase(), sft=sft), pretraining_mix=mix)
    with pytest.raises(ValueError, match="block_size"):
        TrainingConfig(phases=PhasesConfig(embedding_alignment=_phase(datasets=["pretraining_mix"], max_length=256),
                                           warmup=_phase(), sft=sft), pretraining_mix=mix)
