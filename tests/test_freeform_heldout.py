"""Held-out free-form set selection (``data/freeform_heldout.py``): every
filter on fixture rows, the embedding gate with an injected embedder, the
calibration, stratified drawing, determinism and the JSONL round trip
through both readers (the task's and the contamination scanner's)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.data import contamination as C  # noqa: E402
from arabic_eval.data.finetune_corpora import QARecord  # noqa: E402
from arabic_eval.data.freeform_heldout import (  # noqa: E402
    SelectionConfig, _largest_remainder, load_freeform_heldout, prompt_key, select_heldout,
    word_ngrams, write_heldout_jsonl,
)

WORDS = ("كتاب", "مدرسة", "بحر", "جبل", "نهر", "سماء", "قمر", "شمس", "طريق", "مدينة", "قرية", "حديقة",
         "طائر", "شجرة", "زهرة", "ريح", "مطر", "ثلج", "صحراء", "غابة", "نجم", "سفينة", "قلم", "ورقة")


def _rec(i: int, question: str, answer: str, context: str = "", source: str = "cidar") -> QARecord:
    return QARecord(id=f"{source}-{i}", question=question, context=context, answer=answer,
                    source=source, prompt_template="instruction")


def _candidates(n: int = 40):
    """n distinct candidates: prompt = 5 distinct words rotated, reference length grows with i."""
    out = []
    for i in range(n):
        ws = [f"{WORDS[(i * 3 + k) % len(WORDS)]}{i}" for k in range(5)]     # words carry i: no shared 6-gram
        q = f"اكتب فقرة عن {ws[0]} و{ws[1]} و{ws[2]} و{ws[3]} و{ws[4]} رقم {i}"
        ans = " ".join(f"{WORDS[(i + k) % len(WORDS)]}{i}" for k in range(8 + i))    # 8..47 words, unique
        out.append(_rec(i, q, ans))
    return out


def _bag_embedder(dim: int = 64):
    """Deterministic bag-of-words embedder: paraphrases (same words, different order) → cosine 1."""
    def embed(texts):
        m = np.zeros((len(texts), dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.split():
                m[r, hash(w) % dim] += 1.0
            n = np.linalg.norm(m[r])
            if n:
                m[r] /= n
        return m
    return embed


class TestFilters:
    def test_every_filter_counts_and_drops(self):
        cands = _candidates(40)
        # 1 excluded id; 2 latin; 3 context; 4 short ref; 5 long ref; 6 short prompt
        cands[2] = _rec(2, cands[2].question + " Alpaca", cands[2].answer)
        cands[3] = _rec(3, cands[3].question, cands[3].answer, context="سياق")
        cands[4] = _rec(4, cands[4].question, "قصير")
        cands[5] = _rec(5, cands[5].question, "طويل " * 400)
        cands[6] = _rec(6, "قل", cands[6].answer)
        training = [
            _rec(100, cands[7].question, "جواب مختلف تماما", source="bactrian"),                 # exact prompt twin of 7
            _rec(101, cands[8].question.replace("رقم 8", "رقم ثمانية اليوم"), "جواب اخر", source="bactrian"),  # 6-gram twin of 8
            _rec(102, "سؤال لا يشبه شيئا", cands[9].answer, source="aya"),                       # reference twin of 9
        ]
        cfg = SelectionConfig(n_examples=12, seed=1, embed_threshold=0.95)
        rows, man = select_heldout(cands, training, cfg, embedder=None, excluded_ids=["cidar-1"])
        c = man["counts"]
        assert (c["excluded_id"], c["latin"], c["context_nonempty"], c["reference_length"], c["prompt_length"]) == (1, 1, 1, 2, 1)
        assert c["ngram_prompt_twin"] == 2 and c["ngram_reference_twin"] == 1
        assert c["survivors"] == 40 - 9 and c["selected"] == 12
        ids = {r.id for r in rows}
        for i in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            assert f"cidar-{i}" not in ids

    def test_internal_candidate_twins_are_dropped_too(self):
        cands = _candidates(20)
        cands[11] = _rec(11, cands[10].question, cands[11].answer)      # 10 and 11 share a prompt: both leak
        rows, man = select_heldout(cands, [], SelectionConfig(n_examples=5, seed=0))
        assert man["counts"]["ngram_prompt_twin"] == 2
        assert not {"cidar-10", "cidar-11"} & {r.id for r in rows}

    def test_short_prompt_twins_use_exact_match(self):
        cands = _candidates(10)
        cands[0] = _rec(0, "اكتب حكمة عن الحياة", cands[0].answer)              # 4 words: no 6-gram
        training = [_rec(200, "اكتب حكمة عن الحياة", "حكمة", source="bactrian")]
        _, man = select_heldout(cands, training, SelectionConfig(n_examples=3, seed=0))
        assert man["counts"]["ngram_prompt_twin"] == 1


class TestEmbeddingGate:
    def test_paraphrase_twin_dropped_with_fixed_threshold(self):
        cands = _candidates(20)
        q = cands[0].question.split()
        training = [_rec(300, " ".join(reversed(q)), "شيء", source="bactrian")]   # same words, no shared 6-gram
        cfg = SelectionConfig(n_examples=5, seed=0, embed_threshold=0.9)
        rows, man = select_heldout(cands, training, cfg, embedder=_bag_embedder())
        assert man["counts"]["ngram_prompt_twin"] == 0
        assert man["counts"]["embedding_near_duplicate"] == 1
        assert "cidar-0" not in {r.id for r in rows}
        assert man["embedding"]["threshold"] == 0.9 and man["embedding"]["calibrated"] is False
        assert all(r.max_train_cosine is not None and r.max_train_cosine < 0.9 for r in rows)
        assert man["embedding"]["dropped_farthest"][0]["id"] == "cidar-0"
        assert man["embedding"]["dropped_farthest"][0]["nearest_id"] == "bactrian-300"

    def test_self_copy_in_training_is_not_a_neighbour(self):
        cands = _candidates(8)
        rows, man = select_heldout(cands, [], SelectionConfig(n_examples=4, seed=0, embed_threshold=0.99),
                                   embedder=_bag_embedder())
        assert man["counts"]["embedding_near_duplicate"] == 0 and len(rows) == 4

    def test_calibration_uses_the_ngram_twins(self):
        cands = _candidates(30)
        training = [_rec(400 + i, cands[i].question, "x", source="bactrian") for i in range(5)]   # 5 exact twins, cos 1.0
        cfg = SelectionConfig(n_examples=5, seed=0, embed_threshold=None, embed_threshold_bounds=(0.80, 0.97))
        _, man = select_heldout(cands, training, cfg, embedder=_bag_embedder())
        e = man["embedding"]
        assert e["calibrated"] is True and e["twins_for_calibration"] == 5
        assert e["threshold"] == 0.97            # twins sit at 1.0 → clamped to the upper bound
        assert e["twin_max_cosine_quantiles"]["p50"] == 1.0

    def test_no_twins_falls_back_to_upper_bound(self):
        cands = _candidates(10)
        _, man = select_heldout(cands, [], SelectionConfig(n_examples=3, embed_threshold=None), embedder=_bag_embedder())
        assert man["embedding"]["calibrated"] is False and man["embedding"]["threshold"] == 0.97


class TestDraw:
    def test_stratified_equal_and_deterministic(self):
        cands = _candidates(40)
        cfg = SelectionConfig(n_examples=12, seed=7)
        rows1, man1 = select_heldout(cands, [], cfg)
        rows2, _ = select_heldout(list(reversed(cands)), [], cfg)          # input order is irrelevant
        assert [r.to_json() for r in rows1] == [r.to_json() for r in rows2]
        assert man1["strata"]["selected"] == {"short": 4, "medium": 4, "long": 4}
        assert sorted(r.id for r in rows1) == [r.id for r in rows1]         # id-sorted output
        rows3, _ = select_heldout(cands, [], SelectionConfig(n_examples=12, seed=8))
        assert {r.id for r in rows3} != {r.id for r in rows1}

    def test_short_stratum_spills(self):
        cands = _candidates(30)
        for i in range(29):                                # 29 refs of 8 words, one of 40 words
            cands[i] = _rec(i, cands[i].question, " ".join(f"{WORDS[(i + k) % 24]}{i}" for k in range(8)))
        cands[29] = _rec(29, cands[29].question, " ".join(f"كلمة{k}" for k in range(40)))
        rows, man = select_heldout(cands, [], SelectionConfig(n_examples=9, seed=0))
        assert len(rows) == 9 and sum(man["strata"]["selected"].values()) == 9

    def test_requesting_more_than_survivors_returns_all(self):
        cands = _candidates(6)
        rows, man = select_heldout(cands, [], SelectionConfig(n_examples=50, seed=0))
        assert len(rows) == 6 and man["counts"]["selected"] == 6

    def test_largest_remainder(self):
        assert _largest_remainder(10, [1, 1, 1]) == [4, 3, 3]
        assert _largest_remainder(0, [1, 1]) == [0, 0]


class TestIO:
    def test_jsonl_round_trip_through_both_readers(self, tmp_path):
        rows, _ = select_heldout(_candidates(12), [], SelectionConfig(n_examples=6, seed=0))
        p = tmp_path / "heldout.jsonl"
        write_heldout_jsonl(rows, p)
        back = load_freeform_heldout(p)
        assert [b["id"] for b in back] == [r.id for r in rows]
        assert all(b["context"] == "" and b["reference"] and b["stratum"] for b in back)
        first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
        assert "context" not in first                                   # empty context omitted
        # the contamination reader (the repo's field mapping) sees the same prompts
        recs = C.read_prompt_file(p, {"id": "id", "prompt": "prompt", "context": "context", "reference": "reference"}, "freeform")
        assert [r.rec_id for r in recs] == [r.id for r in rows]
        assert recs[0].question == C.normalize_text(rows[0].prompt)
        assert C.normalize_text(rows[0].reference) in recs[0].text

    def test_duplicate_id_rejected(self, tmp_path):
        p = tmp_path / "dup.jsonl"
        p.write_text('{"id":"a","prompt":"س","reference":"ج"}\n{"id":"a","prompt":"س2","reference":"ج"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate id"):
            load_freeform_heldout(p)

    def test_helpers(self):
        assert word_ngrams("a b c d", 3) == {"a b c", "b c d"}
        assert word_ngrams("a b", 3) == set()
        r = _rec(1, "سؤال", "ج", context="سياق")
        assert prompt_key(r) == C.normalize_text("سياق سؤال")
