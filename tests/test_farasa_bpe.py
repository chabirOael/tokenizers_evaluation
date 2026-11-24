"""``farasa_bpe`` — BPE whose merges never cross a Farasa boundary (2026-11-08).

The core tests run without Java: the segmentation cache is written by hand (a
boundary table for a tiny corpus + a manifest keyed on the texts' sha256), which is
exactly what ``farasa_bpe.train`` looks up. The Farasa-dependent tests (variant A on
unseen words, the real segmenter on ``للشركة``) skip when Java / farasapy is absent.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.registry import tokenizer_registry  # noqa: E402
from arabic_eval.tokenizers import farasa_segmentation as fs  # noqa: E402
from arabic_eval.tokenizers.farasa_bpe import SIDE_FILE, FarasaBPETokenizer, _training_pre_tokenizer  # noqa: E402

ROUNDTRIP = [
    "ذهب الطالب إلى المدرسة في الصباح الباكر.",
    "قالَ المعلّمُ: إنّ الكتابَ مفيدٌ جدًّا!",
    "وُلد عام ١٩٤٥ في القاهرة، وتوفي سنة 2001.",
    "استخدم برنامج Python لتحليل البيانات (الإصدار 3.10).",
    "هل تعرف أين مكتبة الجامعة؟ نعم، إنها قرب المسجد الكبير",
]

# A tiny training corpus with a hand-written segmentation (Farasa-like).
CUTS = {"والكتاب": (1, 3), "الكتاب": (2,), "بالقلم": (1, 3), "القلم": (2,), "كتبوها": (4, 6), "المدرسة": (2, 6),
        "وقال": (1,), "للطالب": (2,), "الطالب": (2,), "مدرسته": (4, 5)}
CORPUS = [
    "والكتاب على الطاولة والكتاب جديد",
    "كتب الطالب بالقلم والقلم أزرق",
    "وقال المعلم إن الطلاب كتبوها في المدرسة",
    "ذهب الطالب إلى مدرسته وقال للطالب الكتاب مفيد",
] * 40 + ROUNDTRIP * 5      # the round-trip sentences are training text, so every byte is in the alphabet (as on the real corpus)


def _java_ok() -> bool:
    if shutil.which("java") is None:
        return False
    try:
        import farasa  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _fake_cache(root: Path, texts) -> Path:
    digest = fs.texts_digest(texts)
    v = fs.farasa_versions() if _java_ok() else {"farasapy": "x", "segmenter_jar": "x", "segmenter_jar_sha256": "x"}
    d = root / "fp_test"
    d.mkdir(parents=True)
    fs.save_table(d / "boundary_table.parquet", CUTS)
    (d / "manifest.json").write_text(json.dumps({"fingerprint": "fp_test", "texts_sha256": digest, "joiner": "U+2060",
                                                 "farasapy": v["farasapy"], "segmenter_jar_sha256": v["segmenter_jar_sha256"],
                                                 "stats": {"note": "hand-written"}}))
    return d


@pytest.fixture()
def trained(tmp_path, monkeypatch):
    if not _java_ok():
        monkeypatch.setattr(fs, "farasa_versions",
                            lambda: {"farasapy": "x", "segmenter_jar": "x", "segmenter_jar_sha256": "x"})
    _fake_cache(tmp_path / "cache", CORPUS)
    tok = FarasaBPETokenizer(cache_root=str(tmp_path / "cache"))
    tok.train(CORPUS, vocab_size=400)
    return tok, tmp_path


def test_registry_and_spec():
    assert tokenizer_registry.get("farasa_bpe") is FarasaBPETokenizer
    names = [p.name for p in FarasaBPETokenizer.param_spec()]
    assert names[:2] == ["min_frequency", "inference_segmentation"]


def test_no_piece_contains_the_joiner_and_training_boundaries_are_never_spanned(trained):
    tok, _ = trained
    raw = tok._tokenizer
    joiner_bytes = __import__("tokenizers").pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(fs.JOINER)[0][0]
    assert not any(fs.JOINER in p or joiner_bytes in p for p in raw.get_vocab())
    assert tok._build["bpe"]["pieces_containing_joiner"] == 0
    # Encode the training sentences as training saw them (joiners + training pre-tokenizer).
    raw.pre_tokenizer = _training_pre_tokenizer(fs.JOINER)
    try:
        for text in CORPUS[:4]:
            joined = fs.insert_joiners(text, CUTS)
            # character positions of the joiners in `joined`
            jpos = {i for i, ch in enumerate(joined) if ch == fs.JOINER}
            enc = raw.encode(joined, add_special_tokens=False)
            for (a, b) in enc.offsets:
                assert not any(a < j < b for j in jpos), f"a piece spans a training boundary in {text!r}"
    finally:
        tok._apply_mode()


def test_variant_b_round_trip_and_raw_pretokenizer(trained):
    tok, _ = trained
    assert tok.inference_segmentation == "none"
    assert type(tok._tokenizer.pre_tokenizer).__name__ == "ByteLevel"
    for s in ROUNDTRIP + CORPUS[:4]:
        assert tok.decode(tok.encode(s).input_ids) == s


def test_save_load_mode_and_mismatch(trained):
    tok, tmp = trained
    tok.save(tmp / "saved")
    side = json.loads((tmp / "saved" / SIDE_FILE).read_text())
    assert side["inference_segmentation"] == "none" and side["joiner"] == "U+2060"
    assert "ByteLevel" in json.dumps(json.loads((tmp / "saved" / "tokenizer.json").read_text())["pre_tokenizer"])
    again = FarasaBPETokenizer(inference_segmentation="none")
    again.load(tmp / "saved")
    assert [again.encode(s).input_ids for s in ROUNDTRIP] == [tok.encode(s).input_ids for s in ROUNDTRIP]
    with pytest.raises(ValueError, match="switching silently"):
        FarasaBPETokenizer(inference_segmentation="farasa").load(tmp / "saved")
    # an explicit re-save as variant A restores variant A
    tok.set_inference_segmentation("farasa")
    tok.save(tmp / "saved_a")
    a = FarasaBPETokenizer()
    a.load(tmp / "saved_a")
    assert a.inference_segmentation == "farasa" and type(a._tokenizer.pre_tokenizer).__name__ == "Sequence"


def test_variant_a_uses_the_table_and_round_trips(trained):
    tok, _ = trained
    tok.set_inference_segmentation("farasa")
    tok._table = {w: CUTS.get(w, ()) for t in CORPUS for w in t.split()}
    for s in CORPUS[:4]:     # every Arabic word is in the table → no Farasa call
        ids = tok.encode(s).input_ids
        assert tok.decode(ids) == s
    if not _java_ok():
        pytest.skip("no Java: unseen words need Farasa")
    for s in ROUNDTRIP:
        assert tok.decode(tok.encode(s).input_ids) == s


def test_alignment_rule():
    assert fs.align_word("الكتاب", "ال+كتاب") == ([2], 0, False)                 # clean word
    kept, dropped, rewritten = fs.align_word("للشركة", "ل+ال+شرك+ة")              # Farasa inserts an alef
    assert rewritten and dropped == 1 and kept == [2, 5]
    assert fs.insert_joiners("للشركة", {"للشركة": tuple(kept)}).replace(fs.JOINER, "") == "للشركة"
    assert fs.align_word("جدًّا", "جد+ا") == ([], 1, True)                          # cut between two equal blocks: dropped
    assert fs.align_word("a+b", "a + b") == ([1, 2], 0, False)                       # a literal plus stays text


@pytest.mark.skipif(not _java_ok(), reason="needs Java + farasapy")
def test_real_farasa_on_lilsharika():
    out, stats = fs.segment_words(["للشركة", "الكتاب"], workers=1)
    assert out[0].replace("+", "") != "للشركة"           # the rewrite is real
    table, st = fs.build_boundary_table(__import__("collections").Counter({"للشركة": 1, "الكتاب": 1}), out,
                                        ["للشركة", "الكتاب"])
    assert st["farasa_rewrite_words"] == 1 and table["الكتاب"] == (2,)
