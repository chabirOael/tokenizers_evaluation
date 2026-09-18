"""decode(encode(text)) must give the text back for the subword tokenizers
that can (byte-level BPE exactly; WordPiece up to punctuation spacing) — the
free-form eval decodes, and a missing HF decoder went unnoticed while the
pipeline only scored log-likelihoods (found 2026-09-18 by
``reference_roundtrip_chrf``: BPE returned byte-level surrogates, WordPiece
``##`` pieces joined with spaces)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.tokenizers.bpe import BPETokenizer  # noqa: E402
from arabic_eval.tokenizers.char_jaber import CharJaberTokenizer  # noqa: E402
from arabic_eval.tokenizers.wordpiece import WordPieceTokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEXTS = [
    "المجاز اللغوي هو استعمال اللفظ في معنى مخالف للمعنى الذي وضع له",
    "ذهب الولد إلى المدرسة صباحا وعاد في المساء",
    "الكتاب على الطاولة، والقلم في الحقيبة.",
    "يدرس الطلاب اللغة العربية في الجامعة",
    "هل الجملة السابقة صحيحة؟",
] * 8
PROBE = "المجاز اللغوي هو استعمال اللفظ في معنى مخالف، والقلم في الحقيبة؟"


def test_bytelevel_bpe_round_trips_exactly(tmp_path):
    tok = BPETokenizer()
    tok.train(TEXTS, vocab_size=400)
    assert tok.decode(tok.encode(PROBE).input_ids) == PROBE
    tok.save(tmp_path / "bpe")
    fresh = BPETokenizer()
    fresh.load(tmp_path / "bpe")
    assert fresh.decode(fresh.encode(PROBE).input_ids) == PROBE


def test_wordpiece_rejoins_pieces(tmp_path):
    tok = WordPieceTokenizer()
    tok.train(TEXTS, vocab_size=300)
    out = tok.decode(tok.encode(PROBE).input_ids)
    assert "##" not in out
    # whole words come back intact; Whitespace pre-tokenization may detach punctuation
    for w in ("المجاز", "اللغوي", "استعمال", "الحقيبة"):
        assert w in out.split() or w in out
    tok.save(tmp_path / "wp")
    fresh = WordPieceTokenizer()
    fresh.load(tmp_path / "wp")
    assert fresh.decode(fresh.encode(PROBE).input_ids) == out


def test_char_jaber_round_trips_known_characters():
    tok = CharJaberTokenizer()
    tok.train(TEXTS, vocab_size=0)
    assert tok.decode(tok.encode(PROBE).input_ids) == PROBE


@pytest.mark.parametrize("name, cls", [("bpe_32k", BPETokenizer), ("wordpiece_32k", WordPieceTokenizer)])
def test_archived_tokenizers_decode_after_load(name, cls):
    """Tokenizers saved before the fix carry no decoder in tokenizer.json; load() sets it."""
    path = REPO / "outputs" / "tokenizers" / name
    if not (path / "tokenizer.json").exists():
        pytest.skip(f"{name} not trained on this machine")
    tok = cls()
    tok.load(path)
    out = tok.decode(tok.encode(PROBE).input_ids)
    assert "Ø" not in out and "Ù" not in out and "##" not in out
    assert "المجاز" in out and "الحقيبة" in out
