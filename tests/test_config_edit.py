"""Tests for ``arabic_eval.config_edit`` — the text-level rewriters that
edit an experiment YAML's ``experiment:`` keys without touching the rest
of the file (clone renames, ``created_at`` stamps, ``runs`` entries)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import EXPERIMENT_KEYS, load_config  # noqa: E402
from arabic_eval.config_edit import (  # noqa: E402
    append_run,
    get_created_at,
    now_iso,
    rewrite_experiment_keys,
    stamp_created_at,
    strip_runs,
)

UP = {"name": "bar", "output_dir": "outputs/experiments/bar", "description": "new: desc # kept"}
E1 = {"started_at": "2026-09-20T14:02:10+03:00", "run_id": "20260920-140210_x", "source": "console"}
E2 = {"started_at": "2026-09-21T09:15:00+03:00", "source": "cli"}


def _runs(text: str):
    d = yaml.safe_load(text)
    return (d.get("experiment") or d).get("runs")


def test_rewrite_experiment_keys_layouts():
    # top-level keys, bare scalars, trailing comment, description missing
    out = rewrite_experiment_keys("name: foo   # run\noutput_dir: outputs/experiments/foo\ntokenizer:\n  type: bpe\n", UP)
    assert out == ('name: bar   # run\noutput_dir: outputs/experiments/bar\ndescription: "new: desc # kept"\n'
                   "tokenizer:\n  type: bpe\n")
    # single quotes keep their style, '' escaping
    out = rewrite_experiment_keys("experiment:\n  name: 'foo'\n  description: 'it''s'\n  output_dir: 'x'\n", UP)
    assert out == "experiment:\n  name: 'bar'\n  description: 'new: desc # kept'\n  output_dir: 'outputs/experiments/bar'\n"
    # a plain scalar continued on the next line is folded into the one new line
    out = rewrite_experiment_keys("experiment:\n  name: foo\n  description: one two\n    three\n  output_dir: x\n  seed: 1\n", UP)
    assert out == 'experiment:\n  name: bar\n  description: "new: desc # kept"\n  output_dir: outputs/experiments/bar\n  seed: 1\n'
    # keys the rewriter does not own, and nested `name:` keys, are untouched
    out = rewrite_experiment_keys("experiment:\n  name: foo\n  seed: 1\nmodel:\n  name: keep\n", {"name": "bar"})
    assert out == "experiment:\n  name: bar\n  seed: 1\nmodel:\n  name: keep\n"
    # unrecognised layouts → None (the caller renders instead)
    assert rewrite_experiment_keys("experiment: {name: foo}\n", UP) is None
    assert rewrite_experiment_keys("experiment:\n  name: foo\n  description: |\n    block\n", UP) is None
    assert rewrite_experiment_keys("output_dir: x\n", UP) is None
    with pytest.raises(ValueError):
        rewrite_experiment_keys("name: foo\n", {"seed": "3"})


def test_stamp_created_at_inserts_or_replaces():
    text = "# head\nexperiment:\n  name: x\n  output_dir: o   # where\n  seed: 42\ntokenizer:\n  type: bpe\n"
    assert get_created_at(text) is None
    out = stamp_created_at(text, "2026-09-20T13:00:00+03:00")
    assert out == ('# head\nexperiment:\n  name: x\n  output_dir: o   # where\n  created_at: "2026-09-20T13:00:00+03:00"\n'
                   "  seed: 42\ntokenizer:\n  type: bpe\n")
    assert get_created_at(out) == "2026-09-20T13:00:00+03:00"
    again = stamp_created_at(out, "2027-01-01T00:00:00+00:00")
    assert again.count("created_at") == 1 and get_created_at(again) == "2027-01-01T00:00:00+00:00"
    assert stamp_created_at(text).count("created_at") == 1 and get_created_at(stamp_created_at(text))[:2] == "20"


@pytest.mark.parametrize("label, text", [
    ("block, no runs", "# head\nexperiment:\n  name: \"x\"   # n\n  output_dir: \"o\"\n  seed: 42\ntokenizer:\n  type: bpe\n"),
    ("top-level, no runs", "name: x\noutput_dir: o\ntokenizer:\n  type: bpe\n"),
    ("empty list with comment", "experiment:\n  name: x\n  runs: []  # none yet\n  seed: 1\n"),
    ("bare key", "experiment:\n  name: x\n  runs:\n  seed: 1\n"),
    ("pyyaml-style items", "experiment:\n  name: x\n  runs:\n  - started_at: '2026-01-01T00:00:00+03:00'\n    source: cli\n  seed: 1\n"),
    ("indented flow items + comment", "experiment:\n  name: x\n  runs:\n    - {started_at: a, source: cli}\n    # a note\n    - {started_at: b, source: cli}\n  seed: 1\n"),
])
def test_append_run_every_layout_keeps_text_and_order(label, text):
    before = _runs(text) or []
    one = append_run(text, E1)
    assert one is not None, label
    assert _runs(one) == before + [E1], label
    two = append_run(one, E2)
    assert _runs(two) == before + [E1, E2], label
    # only lines were added: every original line is still there, in order
    orig = [ln for ln in text.split("\n") if ln.strip() and not ln.lstrip().startswith("runs:")]
    kept = [ln for ln in two.split("\n") if ln in orig]
    assert kept == orig, label
    # stripping the runs gives back the file without them (an empty `runs:` line of the original may stay)
    no_key = lambda t: [ln for ln in t.split("\n") if not ln.lstrip().startswith("runs:")]
    assert no_key(strip_runs(two)) == no_key(strip_runs(text)) and _runs(strip_runs(two)) in (None, []), label
    # the rest of the file is still what it was
    flat = yaml_flat(two)
    assert flat["name"] == "x" and flat.get("tokenizer") == yaml_flat(text).get("tokenizer")


def yaml_flat(text: str) -> dict:
    d = yaml.safe_load(text)
    exp = d.pop("experiment", {}) or {}
    return {**exp, **{k: v for k, v in d.items() if k in EXPERIMENT_KEYS or k in ("tokenizer",)}}


def test_append_run_refuses_inline_lists_and_bad_entries():
    assert append_run("experiment:\n  name: x\n  runs: [{started_at: a, source: cli}]\n", E1) is None
    assert append_run("experiment: {name: x}\n", E1) is None
    with pytest.raises(ValueError):
        append_run("name: x\n", {"source": "cli"})
    assert strip_runs("experiment: {name: x}\n") is None
    assert strip_runs("name: x\n") == "name: x\n"
    assert strip_runs("name: x\nruns: []  # kept\n") == "name: x\nruns: []  # kept\n"


def test_fields_round_trip_through_load_config(tmp_path: Path):
    base = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("experiment:\n  name: x\n  created_at: \"2026-09-20T13:00:00+03:00\"\n  runs:\n"
                   "    - {started_at: \"2026-09-20T14:02:10+03:00\", run_id: r1, source: console}\n"
                   "    - {started_at: \"2026-09-21T09:15:00+03:00\", source: cli}\n", encoding="utf-8")
    c = load_config(cfg, base_path=base)
    assert c.created_at == "2026-09-20T13:00:00+03:00"
    assert [(r.started_at, r.run_id, r.source) for r in c.runs] == [
        ("2026-09-20T14:02:10+03:00", "r1", "console"), ("2026-09-21T09:15:00+03:00", None, "cli")]
    plain = load_config(tmp_path / "plain.yaml", base_path=base) if (tmp_path / "plain.yaml").write_text("name: p\n") else None
    assert plain.created_at is None and plain.runs == []
    cfg.write_text("name: x\nruns:\n  - {started_at: t, source: cron}\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(cfg, base_path=base)
    assert now_iso()[10] == "T" and now_iso()[-6] in "+-"
