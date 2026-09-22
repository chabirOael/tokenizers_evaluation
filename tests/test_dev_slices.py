"""Record-level ``dev`` slices for the corpora that have no title to group by
(added 2026-09-22, for the free-form early-stop signal).

``arabic_squad``, ``arabic_squad_mcq``, ``cidar``, ``bactrian_x_ar`` and
``aya_ar`` used to refuse everything but ``train``. They now carve a 5 % dev
slice by record id, ``validation`` stays refused, and the MCQ corpus is hashed
on the Arabic-SQuAD row it was built from so a passage is dev in both corpora
or in neither.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.data.contamination import Exclusions  # noqa: E402
from arabic_eval.data.finetune_corpora import (  # noqa: E402
    DEV_FRACTION, dev_key, is_dev_id, is_dev_record, load_corpus,
)

ID_DEV_CORPORA = ("arabic_squad", "arabic_squad_mcq", "cidar", "bactrian_x_ar", "aya_ar")


# --------------------------------------------------------------------------
# the selector
# --------------------------------------------------------------------------

def test_is_dev_id_is_deterministic_and_hits_the_fraction():
    ids = [f"rec-{i}" for i in range(40000)]
    picks = [i for i in ids if is_dev_id("cidar", i)]
    assert picks == [i for i in ids if is_dev_id("cidar", i)]
    assert 0.045 <= len(picks) / len(ids) <= 0.055
    assert all(is_dev_id("cidar", i, fraction=1.0) for i in ids[:50])
    assert not any(is_dev_id("cidar", i, fraction=0.0) for i in ids[:50])


def test_the_corpus_is_part_of_the_hash():
    ids = [f"rec-{i}" for i in range(4000)]
    a = {i for i in ids if is_dev_id("cidar", i)}
    b = {i for i in ids if is_dev_id("aya_ar", i)}
    assert a != b


def test_mcq_is_hashed_on_the_squad_row_it_came_from():
    assert dev_key("arabic_squad_mcq", "sq_mcq_1234") == ("arabic_squad", "1234")
    assert dev_key("cidar", "cidar-7") == ("cidar", "cidar-7")
    for i in range(500):
        assert is_dev_record("arabic_squad_mcq", f"sq_mcq_{i}") == is_dev_record("arabic_squad", str(i))


def test_an_mcq_id_without_the_prefix_falls_back_to_its_own_key():
    assert dev_key("arabic_squad_mcq", "weird-id") == ("arabic_squad_mcq", "weird-id")


# --------------------------------------------------------------------------
# the loaders (stubbed sources)
# --------------------------------------------------------------------------

class _DS:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


@pytest.fixture
def stub_sources(monkeypatch, tmp_path):
    squad = [{"index": i, "question": f"سؤال {i}؟",
              "context": " ".join(f"كلمة{i}_{j}" for j in range(40)),
              "text": f"كلمة{i}_3"} for i in range(600)]
    cidar = [{"index": i, "instruction": f"تعليمة {i}", "output": f"جواب {i} مفصل"} for i in range(600)]

    def fake_load_dataset(name, *args, split=None, revision=None, **kwargs):
        if "Arabic_SQuAD" in name:
            return _DS(squad)
        if "CIDAR" in name:
            return _DS(cidar)
        raise AssertionError(f"unexpected dataset {name}")
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    gz = tmp_path / "ar.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        json.dump([{"id": f"b{i}", "instruction": f"تعليمة {i}", "input": "", "output": f"جواب {i}"}
                   for i in range(600)], f)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(gz))
    return {"squad": squad, "cidar": cidar}


@pytest.mark.parametrize("corpus", ["arabic_squad", "arabic_squad_mcq", "cidar", "bactrian_x_ar"])
def test_train_and_dev_partition_the_corpus(corpus, stub_sources):
    train, dev = load_corpus(corpus, "train"), load_corpus(corpus, "dev")
    assert train and dev
    tr, dv = {r.id for r in train}, {r.id for r in dev}
    assert tr.isdisjoint(dv)
    total = len(tr) + len(dv)
    assert 0.01 <= len(dv) / total <= 0.12, f"{corpus}: dev is {len(dv)}/{total}"
    assert all(is_dev_record(corpus, r.id) for r in dev)
    assert not any(is_dev_record(corpus, r.id) for r in train)


@pytest.mark.parametrize("corpus", ["arabic_squad", "arabic_squad_mcq", "cidar", "bactrian_x_ar"])
def test_dev_is_deterministic(corpus, stub_sources):
    assert [r.id for r in load_corpus(corpus, "dev")] == [r.id for r in load_corpus(corpus, "dev")]


@pytest.mark.parametrize("corpus", ID_DEV_CORPORA)
def test_validation_is_still_refused(corpus, stub_sources):
    with pytest.raises(ValueError, match="only 'train' is available"):
        load_corpus(corpus, "validation")


@pytest.mark.parametrize("corpus", ID_DEV_CORPORA)
def test_an_unknown_split_is_refused(corpus, stub_sources):
    with pytest.raises(ValueError, match="only 'train' is available"):
        load_corpus(corpus, "test")


def test_exclusions_apply_to_train_and_dev(stub_sources):
    all_ids = [r.id for r in load_corpus("cidar", "train")] + [r.id for r in load_corpus("cidar", "dev")]
    dev_ids = [r.id for r in load_corpus("cidar", "dev")]
    drop = {all_ids[0], dev_ids[0]}
    exc = Exclusions(ids={"cidar": drop}, meta={})
    train = load_corpus("cidar", "train", exclusions=exc)
    dev = load_corpus("cidar", "dev", exclusions=exc)
    assert drop.isdisjoint({r.id for r in train})
    assert drop.isdisjoint({r.id for r in dev})


def test_mcq_dev_mirrors_squad_dev_and_the_corpus_is_unchanged(stub_sources):
    """The MCQ corpus is derived from Arabic-SQuAD: carving out dev must not
    change which MCQ records exist (they would draw other distractors)."""
    squad_dev = {r.id for r in load_corpus("arabic_squad", "dev")}
    mcq_dev = {r.id[len("sq_mcq_"):] for r in load_corpus("arabic_squad_mcq", "dev")}
    assert mcq_dev == squad_dev

    mcq_all = {r.id for r in load_corpus("arabic_squad_mcq", "train")} | \
              {r.id for r in load_corpus("arabic_squad_mcq", "dev")}
    squad_all = {r.id for r in load_corpus("arabic_squad", "train")} | squad_dev
    assert mcq_all == {f"sq_mcq_{i}" for i in squad_all}


def test_mcq_records_are_built_from_the_full_squad_corpus(stub_sources, monkeypatch):
    """Regression: building from the *train* slice would silently shrink the
    corpus and redraw every distractor (48 344 → 45 936 on the real data)."""
    from arabic_eval.data import synthetic_mcq
    seen = {}
    real = synthetic_mcq._arabic_squad_records

    def spy():
        out = real()
        seen["n"] = len(out)
        return out
    monkeypatch.setattr(synthetic_mcq, "_arabic_squad_records", spy)
    load_corpus("arabic_squad_mcq", "train")
    assert seen["n"] == 600                    # every stub row, not the train slice
