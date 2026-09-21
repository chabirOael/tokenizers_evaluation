"""The declared tokenizer parameter specs (``BaseTokenizer.param_spec()`` on every registered tokenizer).

* every tokenizer declares a spec whose defaults are the code's own constants — pinned by
  constructing the tokenizer with no kwargs (native_* excluded: they download a tokenizer)
  and by the train-time constants;
* ``configs/tokenizers/<type>.yaml`` are PRESETS the form copies into experiments (not
  generated documentation like the task files): every key must be declared and typed, and
  the only value allowed to differ from the code default is araroopat's ``max_patterns``
  (Balanced tier 4000 vs the class default 500 — a documented, deliberate divergence);
* the console bundle serves ``tokenizer_params``; ``param_warnings`` covers
  ``tokenizer.params`` and ``sweep.tokenizers[].params``; the pipeline warns at run start.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arabic_eval.tokenizers  # noqa: E402, F401
from arabic_eval.params_spec import spec_defaults, tokenizer_param_specs, validate_params  # noqa: E402
from arabic_eval.registry import tokenizer_registry  # noqa: E402
from arabic_eval.tools.experiment_console import ConsolePaths, param_warnings, parse_yaml, schema_bundle, validate_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REAL = ConsolePaths(REPO)
NATIVE = {"native_llama", "native_qwen3"}
FROM_SCRATCH = ["bpe", "wordpiece", "morpho_bpe", "character_bert", "farasa_character_bert", "char_jaber", "charformer", "araroopat"]


def test_every_registered_tokenizer_declares_a_spec():
    specs = tokenizer_param_specs()
    assert set(FROM_SCRATCH) | NATIVE <= set(specs)
    for tok, spec in specs.items():
        if tok.startswith("test_"):          # stubs other test modules register in the shared registry
            continue
        assert spec, tok
        names = [s.name for s in spec]
        assert len(names) == len(set(names)), tok
        for s in spec:
            assert validate_params(spec, {s.name: s.default}, owner=tok) == [], (tok, s.name)


def test_constructor_defaults_equal_the_spec():
    """Instances built with no kwargs carry the spec defaults (the train-time knobs are
    pinned against the module constants the trainers read)."""
    from arabic_eval.tokenizers import bpe, character_bert, morpho_bpe, wordpiece
    from arabic_eval.tokenizers.araroopat import DEFAULTS as ARAROOPAT_DEFAULTS, FUNC_INVENTORY, PREPOSITION_INVENTORY, AraRooPatTokenizer
    from arabic_eval.tokenizers.character_bert import CharacterBERTTokenizer
    from arabic_eval.tokenizers.char_jaber import CharJaberTokenizer
    from arabic_eval.tokenizers.charformer import CharformerTokenizer
    from arabic_eval.tokenizers.farasa_character_bert import DEFAULT_MORPHEME_MAX_CHAR_LEN, FarasaCharacterBERTTokenizer

    d = spec_defaults(AraRooPatTokenizer.param_spec())
    a = AraRooPatTokenizer()                                   # the CAMeL bridge is lazy: no subprocess here
    for k in ARAROOPAT_DEFAULTS:
        assert getattr(a, k) == d[k] == ARAROOPAT_DEFAULTS[k], k
    assert d["max_patterns"] == 500                            # the stale class default, kept on purpose
    assert list(a.prepositions) == d["prepositions"] == list(PREPOSITION_INVENTORY)
    assert list(a.func_words) == d["func_words"] == list(FUNC_INVENTORY)
    assert [s.name for s in AraRooPatTokenizer.param_spec() if s.name == "proper_nouns"] and \
        AraRooPatTokenizer.param_spec()[4].choices == ("unrooted", "all")

    d = spec_defaults(CharacterBERTTokenizer.param_spec())
    assert CharacterBERTTokenizer().max_char_len == d["max_char_len"] == character_bert.DEFAULT_MAX_CHAR_LEN == 50
    assert d["max_word_vocab"] == character_bert.DEFAULT_MAX_WORD_VOCAB == 50_000
    d = spec_defaults(FarasaCharacterBERTTokenizer.param_spec())
    assert d["max_char_len"] == DEFAULT_MORPHEME_MAX_CHAR_LEN == 25 and d["max_word_vocab"] == 50_000
    assert CharJaberTokenizer().downsample_factor == spec_defaults(CharJaberTokenizer.param_spec())["downsample_factor"] == 1
    cf = CharformerTokenizer()
    d = spec_defaults(CharformerTokenizer.param_spec())
    emb = cf.get_embedding_config()
    for k in ("max_block_size", "downsample_rate", "conv_kernel_size", "block_attention"):
        assert emb[k] == d[k] == CharformerTokenizer.GBST_DEFAULTS[k], k
    assert d == {"max_block_size": 4, "downsample_rate": 2, "conv_kernel_size": 5, "block_attention": False}
    for mod, cls_name in ((bpe, "BPETokenizer"), (wordpiece, "WordPieceTokenizer"), (morpho_bpe, "MorphoBPETokenizer")):
        cls = getattr(mod, cls_name)
        assert spec_defaults(cls.param_spec()) == {"min_frequency": mod.DEFAULT_MIN_FREQUENCY} == {"min_frequency": 2}
    from arabic_eval.tokenizers.native_llama import DEFAULT_MODEL as LLAMA, NativeLlamaTokenizer
    from arabic_eval.tokenizers.native_qwen3 import DEFAULT_MODEL as QWEN, NativeQwen3Tokenizer
    assert spec_defaults(NativeLlamaTokenizer.param_spec()) == {"model_name_or_path": LLAMA}
    assert spec_defaults(NativeQwen3Tokenizer.param_spec()) == {"model_name_or_path": QWEN} and LLAMA != QWEN


def test_tokenizer_presets_are_declared_typed_and_differ_only_where_documented():
    """``configs/tokenizers/*.yaml`` are presets (recommended values the form copies), so they
    are not pinned to the defaults — but every key must be declared with the right type, and
    the set of preset values that differ from the code default is pinned: araroopat's
    ``max_patterns`` (Balanced tier 4000 vs class default 500) and nothing else."""
    specs = tokenizer_param_specs()
    diverging = {}
    files = sorted((REPO / "configs" / "tokenizers").glob("*.yaml"))
    assert files
    for f in files:
        block = yaml.safe_load(f.read_text(encoding="utf-8"))["tokenizer"]
        tok = block["type"]
        assert tok in specs, f.name
        params = block.get("params") or {}
        assert validate_params(specs[tok], params, owner=tok) == [], f.name
        d = spec_defaults(specs[tok])
        for k, v in params.items():
            if v != d[k]:
                diverging[(tok, k)] = (v, d[k])
    assert diverging == {("araroopat", "max_patterns"): (4000, 500)}


MINI = """\
experiment:
  name: "mini"
  output_dir: "outputs/experiments/mini"
tokenizer:
  type: "bpe"
  vocab_size: 32000
  params: {min_frequency: "2", vocab_size: 32000}
sweep:
  tokenizers:
    - type: "bpe"
      vocab_sizes: [32000]
      params: {min_frequency: 2}
    - type: "araroopat"
      vocab_sizes: [null]
      params: {max_patterns: 4000, proper_nouns: "some"}
  tasks:
    - type: "acva"
      params: {}
"""


@pytest.fixture
def tmp_repo(tmp_path: Path) -> ConsolePaths:
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "base.yaml", tmp_path / "configs" / "base.yaml")
    return ConsolePaths(tmp_path)


def test_console_warnings_cover_tokenizer_params(tmp_repo: ConsolePaths):
    v = validate_config(tmp_repo, parse_yaml(MINI), file="mini.yaml")
    assert v["ok"]
    pw = [(w["loc"], w["task"], w["key"]) for w in v["param_warnings"]]
    assert ("tokenizer.params", "bpe", "min_frequency") in pw and ("tokenizer.params", "bpe", "vocab_size") in pw
    assert ("sweep.tokenizers.1.params", "araroopat", "proper_nouns") in pw
    assert not any(loc == "sweep.tokenizers.0.params" for loc, _, _ in pw)
    msgs = {w["key"]: w["msg"] for w in v["param_warnings"]}
    assert 'bpe.min_frequency should be int, got "2" (str)' == msgs["min_frequency"]
    assert msgs["vocab_size"].startswith("bpe does not declare a parameter 'vocab_size'")
    assert 'araroopat.proper_nouns must be one of "unrooted", "all", got "some"' == msgs["proper_nouns"]
    assert sum(1 for w in v["warnings"] if w.startswith("tokenizer.params:")) == 2
    b = schema_bundle(REAL)
    assert {s["name"] for s in b["tokenizer_params"]["charformer"]} == {"max_block_size", "downsample_rate", "conv_kernel_size", "block_attention"}
    assert b["tokenizer_params"]["araroopat"][1]["name"] == "max_patterns" and b["tokenizer_params"]["araroopat"][1]["default"] == 500


def test_repo_configs_produce_no_tokenizer_param_warnings():
    from arabic_eval.config import ExperimentConfig
    from arabic_eval.tools.experiment_console import read_config
    for p in sorted((REPO / "configs" / "experiments").glob("*.yaml")):
        rc = read_config(REAL, p.name)
        assert rc["valid"], p.name
        assert param_warnings(ExperimentConfig(**rc["resolved"])) == [], p.name


def test_pipeline_warns_on_tokenizer_params_at_run_start():
    from arabic_eval.config import ExperimentConfig, _deep_merge, load_yaml
    from arabic_eval.pipeline.experiment import _warn_task_params
    from arabic_eval.tools.experiment_console import flatten_experiment
    raw = flatten_experiment(load_yaml(REPO / "configs" / "base.yaml"))
    _deep_merge(raw, parse_yaml(MINI))
    findings = _warn_task_params(ExperimentConfig(**raw))
    assert findings == ['tokenizer.params: bpe.min_frequency should be int, got "2" (str)',
                        "tokenizer.params: bpe does not declare a parameter 'vocab_size' — it is ignored at run time (declared: min_frequency)"]


def test_assistant_reference_lists_declared_tokenizer_params():
    from arabic_eval.tools import config_assistant as ca
    text = ca._presets_text(schema_bundle(REAL))
    assert "declares max_patterns (int, code default 500)" in text and "preset: max_roots: 10000, max_patterns: 4000" in text
    assert "declares min_frequency (int, code default 2)" in text
