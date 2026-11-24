"""``hf_pretrained`` — a released tokenizer evaluated as shipped (2026-11-08).

On the two real folders staged in ``outputs/tokenizers/_incoming/`` (skipped when
absent): ids equal the raw ``tokenizers`` encode, exact round trip, the special
ids measured on 2026-11-08, ``vocab_size`` = ``vocab_rows``, save → load gives
identical ids. A tiny fixture tokenizer covers the no-network / no-file path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizers import Tokenizer  # noqa: E402

from arabic_eval.registry import tokenizer_registry  # noqa: E402
from arabic_eval.tokenizers.hf_pretrained import MARKER, HFPretrainedTokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
JAIS = REPO / "outputs/tokenizers/_incoming/jais2_8b_chat_da0e1639"
FANAR = REPO / "outputs/tokenizers/_incoming/fanar1_9b_d33480ed"

# The five sentences of tests/test_tokenizer_roundtrip.py's style: hamza seats, ة, diacritics, digits, Latin.
SENTENCES = [
    "ذهب الطالب إلى المدرسة في الصباح الباكر.",
    "قالَ المعلّمُ: إنّ الكتابَ مفيدٌ جدًّا!",
    "وُلد عام ١٩٤٥ في القاهرة، وتوفي سنة 2001.",
    "استخدم برنامج Python لتحليل البيانات (الإصدار 3.10).",
    "هل تعرف أين مكتبة الجامعة؟ نعم، إنها قرب المسجد الكبير",
]

CASES = [
    # folder, vocab_rows, len, bos, eos, pad, unk (None = key omitted)
    pytest.param(JAIS, 150272, 150222, 0, 150024, 1, None, id="jais2"),
    pytest.param(FANAR, 128256, 128256, 2, 1, 0, 3, id="fanar1"),
]


@pytest.mark.parametrize("folder,rows,n,bos,eos,pad,unk", CASES)
def test_released_tokenizer_facts(folder, rows, n, bos, eos, pad, unk, tmp_path):
    if not (folder / "tokenizer.json").exists():
        pytest.skip(f"{folder} not staged")
    tok = HFPretrainedTokenizer(path=str(folder), vocab_rows=rows, label="x")
    raw = Tokenizer.from_file(str(folder / "tokenizer.json"))
    assert raw.get_vocab_size() == n
    assert tok.vocab_size == rows
    st = tok.special_tokens
    assert (st["bos_token"], st["eos_token"], st["pad_token"]) == (bos, eos, pad)
    assert st.get("unk_token") == unk and (("unk_token" in st) == (unk is not None))
    assert tok.reuses_model_embeddings is False
    for s in SENTENCES:
        out = tok.encode(s)
        assert out.input_ids == raw.encode(s).ids
        assert out.input_ids[0] == bos, "the file's post-processor adds the BOS"
        assert eos not in out.input_ids, "no EOS on encode"
        assert tok.decode(out.input_ids) == s, "exact round trip"
    # truncation / padding go through the inherited BPE path
    t = tok.encode(SENTENCES[0], max_length=4, truncation=True)
    assert len(t.input_ids) == 4
    p = tok.encode("نص", max_length=8, padding=True)
    assert len(p.input_ids) == 8 and p.attention_mask.count(0) > 0
    # save → load
    tok.save(tmp_path / "saved")
    marker = json.loads((tmp_path / "saved" / MARKER).read_text())
    assert marker["vocab_rows"] == rows and marker["label"] == "x" and len(marker["tokenizer_json_sha256"]) == 64
    again = HFPretrainedTokenizer()
    again.load(tmp_path / "saved")
    assert again.vocab_size == rows and again.label == "x" and again.special_tokens == st
    for s in SENTENCES:
        assert again.encode(s).input_ids == tok.encode(s).input_ids


def _fixture_folder(tmp_path: Path) -> Path:
    from tokenizers import models, pre_tokenizers, decoders, processors
    t = Tokenizer(models.WordLevel({"<bos>": 0, "<eos>": 1, "<pad>": 2, "a": 3, "b": 4}, unk_token=None))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.decoder = decoders.WordPiece(prefix="##")
    t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 0)])
    folder = tmp_path / "fixture"
    folder.mkdir()
    t.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}))
    return folder


def test_fixture_tokenizer_and_registry(tmp_path):
    assert tokenizer_registry.get("hf_pretrained") is HFPretrainedTokenizer
    folder = _fixture_folder(tmp_path)
    tok = HFPretrainedTokenizer(path=str(folder), vocab_rows=None, label="tiny")
    assert tok.vocab_size == 5 and tok.special_tokens == {"pad_token": 2, "bos_token": 0, "eos_token": 1}
    assert tok.encode("a b").input_ids == [0, 3, 4]
    tok2 = HFPretrainedTokenizer(path=str(folder), vocab_rows=8)
    assert tok2.vocab_size == 8, "vocab_rows pads the table"
    tok.train(["a b"], vocab_size=100)        # a no-op
    assert tok.encode("a b").input_ids == [0, 3, 4]
    tok.save(tmp_path / "out")
    (tmp_path / "out" / "tokenizer.json").write_text((tmp_path / "out" / "tokenizer.json").read_text().replace('"a"', '"c"'))
    with pytest.raises(ValueError, match="sha256"):
        HFPretrainedTokenizer().load(tmp_path / "out")


def test_missing_special_token_is_an_error(tmp_path):
    folder = _fixture_folder(tmp_path)
    (folder / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<nope>", "pad_token": "<pad>"}))
    with pytest.raises(ValueError, match="eos_token"):
        HFPretrainedTokenizer(path=str(folder))
