"""The declared task parameter specs (``arabic_eval.params_spec`` + ``<Task>.param_spec()``).

* the helpers: type / range / choice / unknown-key findings, resolution, non-default filtering;
* the pin tests: constructing each task with ``{}`` yields attributes equal to the spec
  defaults (the constructors load no data, so this needs no network), and the free-form
  spec equals ``DecodingConfig()`` field for field;
* the generated ``configs/tasks/<type>.yaml`` equal what the spec renders (replaces the
  one-task ``test_console_preset_equals_the_code_defaults``);
* the pipeline: warnings at run start, ``num_fewshot`` injected only where declared;
* the console: ``task_params`` in the schema bundle, ``param_warnings`` with ``loc`` in
  ``validate_config``, the shared-output_dir warning, the assistant reference.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.params_spec import (  # noqa: E402
    ParamSpec, non_default_params, render_task_preset, resolve_params, spec_defaults, task_param_specs,
    validate_params,
)
from arabic_eval.registry import task_registry  # noqa: E402
from arabic_eval.tasks.freeform.cidar import (  # noqa: E402
    DEFAULT_BERTSCORE_BATCH_SIZE, DEFAULT_BERTSCORE_LAYER, DEFAULT_BERTSCORE_MODEL, DEFAULT_HELDOUT_PATH,
    FreeformCidarTask,
)
from arabic_eval.tasks.freeform.generation import DecodingConfig  # noqa: E402
from arabic_eval.tasks.lighteval.acva import ACVATask  # noqa: E402
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask  # noqa: E402
from arabic_eval.tools.experiment_console import (  # noqa: E402
    ConsolePaths, param_warnings, parse_yaml, read_config, schema_bundle, validate_config,
)

REPO = Path(__file__).resolve().parents[1]
REAL = ConsolePaths(REPO)
LIGHTEVAL_TASKS = ("acva", "alghafa", "arabic_exam", "culture_arabic_mmlu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SPEC = [
    ParamSpec("n", "int", 4, "an int", min=1, max=10),
    ParamSpec("f", "float", 1.5, "a float", min=0),
    ParamSpec("b", "bool", True, "a bool"),
    ParamSpec("s", "str", "x", "a str", choices=("x", "y")),
    ParamSpec("m", "str", "model", "nullable str", nullable=True),
    ParamSpec("l", "list[str]", ["a", "b"], "a list"),
    ParamSpec("p", "path", "some/where", "a path", advanced=True, group="g"),
]


def test_validate_params_findings_name_key_and_owner():
    bad = {"n": "4", "f": True, "b": 1, "s": "z", "m": 3, "l": ["a", 1], "zzz": 1, "p": None}
    out = validate_params(SPEC, bad, owner="acva")
    joined = "\n".join(out)
    assert "acva.n should be int" in joined and "acva.f should be float" in joined
    assert "acva.b should be bool" in joined and "acva.s must be one of" in joined
    assert "acva.m should be str or null" in joined and "acva.l should be list[str]" in joined
    assert "acva does not declare a parameter 'zzz'" in joined and "ignored at run time" in joined
    assert "acva.p should be path" in joined                   # not nullable
    assert len(out) == len(bad)


def test_validate_params_ranges_and_ok_values():
    assert validate_params(SPEC, {"n": 0}, owner="t") == ["t.n must be ≥ 1, got 0"]
    assert validate_params(SPEC, {"n": 11}, owner="t") == ["t.n must be ≤ 10, got 11"]
    assert validate_params(SPEC, {"n": 7, "f": 2, "b": False, "s": "y", "m": None, "l": [], "p": "q"}) == []
    assert validate_params(SPEC, {}) == [] and validate_params(SPEC, None) == []
    assert validate_params([], {"anything": 1}) == []           # no spec → nothing to check against
    assert validate_params(SPEC, ["not", "a", "dict"], owner="t") == ["t: params must be a mapping, got list"]


def test_param_spec_rejects_unknown_type_and_freezes_lists():
    with pytest.raises(ValueError):
        ParamSpec("x", "tuple", 1, "h")
    s = ParamSpec("l", "list[str]", ("a", "b"), "h", choices=["a"])
    assert s.default == ["a", "b"] and s.choices == ("a",)
    assert s.to_dict()["choices"] == ["a"] and s.to_dict()["default"] == ["a", "b"]


def test_resolve_and_non_default():
    assert spec_defaults(SPEC)["l"] == ["a", "b"]
    r = resolve_params(SPEC, {"n": 9, "extra": 1})
    assert r["n"] == 9 and r["f"] == 1.5 and r["extra"] == 1 and list(r)[:2] == ["n", "f"]
    assert non_default_params(SPEC, {"n": 4, "f": 2.0, "l": ["a", "b"], "extra": 1}) == {"f": 2.0, "extra": 1}


def test_render_task_preset_reloads_to_the_defaults_and_marks_groups():
    text = render_task_preset("demo", "DemoTask", SPEC)
    data = yaml.safe_load(text)
    assert data == {"task": {"type": "demo", "params": spec_defaults(SPEC)}}
    assert "# — g —" in text and "[path; advanced]" in text and "range 1–10" in text
    assert "A run NEVER reads this file" in text and "DemoTask.param_spec()" in text


# ---------------------------------------------------------------------------
# the specs on the real tasks
# ---------------------------------------------------------------------------

def test_every_registered_task_has_a_spec_with_unique_names():
    specs = task_param_specs()
    assert set(LIGHTEVAL_TASKS) | {"freeform_cidar"} <= set(specs)
    for task, spec in specs.items():
        names = [s.name for s in spec]
        assert len(names) == len(set(names)), task
        for s in spec:
            assert s.help.endswith(".") or s.help.endswith(")"), (task, s.name, s.help)
            assert validate_params(spec, {s.name: s.default}, owner=task) == [], (task, s.name)   # the default is valid


@pytest.mark.parametrize("task", LIGHTEVAL_TASKS)
def test_lighteval_task_built_with_empty_params_equals_the_spec_defaults(task):
    cls = task_registry.get(task)
    assert issubclass(cls, LightEvalBenchmarkTask)
    spec = cls.param_spec()
    d = spec_defaults(spec)
    assert list(d) == ["dataset_name", "dataset_config", "cache_dir", "max_length", "seed", "clean_latin_rows", "num_fewshot"]
    inst = cls({})                                   # no data is loaded in __init__
    assert inst.dataset_name == d["dataset_name"] == cls._default_dataset_name()
    assert inst.dataset_config == d["dataset_config"] is None
    assert inst.cache_dir == d["cache_dir"] and inst.max_length == d["max_length"]
    assert inst.seed == d["seed"] and inst.clean_latin_rows == d["clean_latin_rows"]
    assert inst.num_fewshot == d["num_fewshot"] == 0
    assert inst._cached_examples is None


def test_acva_rewords_only_num_fewshot():
    from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
    base = {s.name: s for s in AlghafaTask.param_spec()}      # the inherited list, on a sibling task
    acva = {s.name: s for s in ACVATask.param_spec()}
    assert set(base) == set(acva)
    for name in base:
        if name == "num_fewshot":
            assert "word-scored" in acva[name].help and "pin 0" in acva[name].help
            assert acva[name].default == base[name].default == 0
        elif name == "dataset_name":
            assert acva[name].default == "OALL/ACVA" and base[name].default != acva[name].default
        else:
            assert acva[name] == base[name], name


def test_dataset_name_is_readable_at_class_level():
    for task in LIGHTEVAL_TASKS:
        cls = task_registry.get(task)
        assert isinstance(cls._default_dataset_name(), str) and "/" in cls._default_dataset_name()


def test_lighteval_spec_tolerates_an_instance_method_dataset_name():
    class Stub(LightEvalBenchmarkTask):            # the shape of the test stubs elsewhere in the suite
        name = "stub"
        metric_names = ["accuracy"]

        def _default_dataset_name(self) -> str:
            return "stub/x"

        def _parse_example(self, raw): return raw
        def load_examples(self): return []
        def _format_eval_context(self, ex): return ""
        def _build_continuations(self, ex): return []
        def _aggregate_scores(self, ex, c, l, u=None, normalization="char"): return l

    spec = {s.name: s for s in Stub.param_spec()}
    assert spec["dataset_name"].default is None and spec["dataset_name"].nullable
    assert Stub({}).dataset_name == "stub/x"


def test_freeform_task_built_with_empty_params_equals_the_spec_and_the_dataclass():
    spec = FreeformCidarTask.param_spec()
    d = spec_defaults(spec)
    dc = DecodingConfig()
    for f in ("max_output_chars", "max_prompt_tokens", "batch_size", "token_cap_margin", "token_cap_floor",
              "token_cap_ceiling", "marker_check_every", "loop_stop", "seed"):
        assert d[f] == getattr(dc, f), f
    assert d["stop_markers"] == list(dc.stop_markers)
    assert d["heldout_path"] == DEFAULT_HELDOUT_PATH
    assert d["bertscore_model"] == DEFAULT_BERTSCORE_MODEL == "xlm-roberta-large"
    assert d["bertscore_layer"] == DEFAULT_BERTSCORE_LAYER and d["bertscore_batch_size"] == DEFAULT_BERTSCORE_BATCH_SIZE
    task = FreeformCidarTask({})                     # loads nothing (rows are read lazily by evaluate)
    assert task.decoding == dc and task._rows is None
    assert task.bertscore_model == d["bertscore_model"] and task.bertscore_layer == d["bertscore_layer"]
    assert task.bertscore_batch_size == d["bertscore_batch_size"]
    assert str(task.heldout_path).endswith(DEFAULT_HELDOUT_PATH)
    groups = {s.name: s.group for s in spec}
    assert groups["max_output_chars"] == "budget" and groups["stop_markers"] == "stops"
    assert groups["bertscore_model"] == "scoring" and groups["heldout_path"] == "misc"
    advanced = {s.name for s in spec if s.advanced}
    assert advanced == {"token_cap_margin", "token_cap_floor", "token_cap_ceiling", "marker_check_every",
                        "bertscore_layer", "bertscore_batch_size"}
    assert {s.name for s in spec if s.nullable} == {"bertscore_model"}
    # the spec accepts what the two reeval configs pin, and flags the injected num_fewshot
    assert validate_params(spec, {"max_output_chars": 2400}, owner="freeform_cidar") == []
    assert validate_params(spec, {"num_fewshot": 3}, owner="freeform_cidar")[0].startswith(
        "freeform_cidar does not declare a parameter 'num_fewshot'")


# ---------------------------------------------------------------------------
# generated presets
# ---------------------------------------------------------------------------

def test_configs_tasks_files_equal_the_rendered_specs():
    """``configs/tasks/<type>.yaml`` is generated from the spec — every file must equal
    what ``scripts/render_task_presets.py`` writes (the stale ``train_split_ratio`` of the
    3-phase migration and the 2400-vs-1200 drift of 2026-09-21 are what this prevents)."""
    specs = task_param_specs()
    files = {p.stem for p in (REPO / "configs" / "tasks").glob("*.yaml")}
    with_spec = {t for t, s in specs.items() if s and not t.startswith("test_")}   # stubs other test modules register
    assert files == with_spec, (files ^ with_spec)
    for task in with_spec:
        expected = render_task_preset(task, task_registry.get(task).__name__, specs[task])
        text = (REPO / "configs" / "tasks" / f"{task}.yaml").read_text(encoding="utf-8")
        assert text == expected, f"configs/tasks/{task}.yaml is stale — run scripts/render_task_presets.py"
        data = yaml.safe_load(text)
        assert data["task"]["type"] == task and data["task"]["params"] == spec_defaults(specs[task])
        assert "train_split_ratio" not in data["task"]["params"]


def test_render_script_check_mode(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "render_task_presets.py"), "--check"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "render_task_presets.py"), "--check",
                        "--out-dir", str(tmp_path)], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 1 and "STALE" in r.stdout


# ---------------------------------------------------------------------------
# pipeline: run-start warnings, num_fewshot injection
# ---------------------------------------------------------------------------

def _cfg(tasks):
    from arabic_eval.config import ExperimentConfig, _deep_merge, load_yaml
    from arabic_eval.tools.experiment_console import flatten_experiment
    raw = flatten_experiment(load_yaml(REPO / "configs" / "base.yaml"))
    _deep_merge(raw, {"name": "t", "output_dir": "outputs/experiments/t", "tokenizer": {"type": "native_llama", "vocab_size": None},
                      "sweep": {"tokenizers": [{"type": "native_llama", "vocab_sizes": [None]}], "tasks": tasks},
                      "evaluation": {"num_fewshot": 3}})
    return ExperimentConfig(**raw)


def test_pipeline_warns_at_run_start_and_never_fails(caplog):
    from arabic_eval.pipeline.experiment import _warn_task_params
    cfg = _cfg([{"type": "acva", "params": {"train_split_ratio": 0.1, "max_length": "512"}},
                {"type": "freeform_cidar", "params": {"max_output_chars": 2400}}])
    with caplog.at_level(logging.WARNING, logger="arabic_eval.pipeline.experiment"):
        findings = _warn_task_params(cfg)
    assert len(findings) == 2
    assert findings[0].startswith("sweep.tasks[0].params: acva does not declare a parameter 'train_split_ratio'")
    assert findings[1] == 'sweep.tasks[0].params: acva.max_length should be int, got "512" (str)'
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("task params: sweep.tasks[0].params: acva does not declare a parameter 'train_split_ratio'" in m for m in logged)
    assert _warn_task_params(_cfg([{"type": "acva", "params": {"num_fewshot": 0}}])) == []


def test_num_fewshot_is_injected_only_where_declared():
    from arabic_eval.pipeline.experiment import _task_params
    cfg = _cfg([{"type": "acva", "params": {}}])
    assert _task_params(ACVATask, {}, cfg) == {"num_fewshot": 3}
    assert _task_params(ACVATask, {"num_fewshot": 0}, cfg) == {"num_fewshot": 0}
    assert _task_params(FreeformCidarTask, {"max_output_chars": 2400}, cfg) == {"max_output_chars": 2400}


# ---------------------------------------------------------------------------
# console: bundle, warnings with loc, shared output_dir, assistant reference
# ---------------------------------------------------------------------------

MINI = """\
experiment:
  name: "mini"
  output_dir: "outputs/experiments/mini"
tokenizer:
  type: "native_llama"
  vocab_size: null
sweep:
  tokenizers:
    - type: "native_llama"
      vocab_sizes: [null]
  tasks:
    - type: "acva"
      params: {num_fewshot: 0, train_split_ratio: 0.1}
    - type: "freeform_cidar"
      params: {max_output_chars: "2400"}
"""


@pytest.fixture
def tmp_repo(tmp_path: Path) -> ConsolePaths:
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "base.yaml", tmp_path / "configs" / "base.yaml")
    (tmp_path / "configs" / "experiments" / "mini.yaml").write_text(MINI, encoding="utf-8")
    (tmp_path / "configs" / "experiments" / "twin.yaml").write_text(MINI.replace('name: "mini"', 'name: "twin"'), encoding="utf-8")
    return ConsolePaths(tmp_path)


def test_schema_bundle_serves_task_params():
    b = schema_bundle(REAL)
    tp = b["task_params"]
    assert set(LIGHTEVAL_TASKS) | {"freeform_cidar"} <= set(tp)
    ff = {s["name"]: s for s in tp["freeform_cidar"]}
    assert ff["max_output_chars"]["default"] == DecodingConfig().max_output_chars and ff["max_output_chars"]["group"] == "budget"
    assert ff["stop_markers"]["type"] == "list[str]" and ff["bertscore_model"]["nullable"] is True
    assert "tasks" not in b["presets"] and "tokenizers" in b["presets"] and "models" in b["presets"]


def test_validate_config_reports_param_warnings_with_loc(tmp_repo: ConsolePaths):
    v = validate_config(tmp_repo, parse_yaml(MINI), file="mini.yaml")
    assert v["ok"]
    pw = v["param_warnings"]
    assert [(w["loc"], w["task"], w["key"]) for w in pw] == [
        ("sweep.tasks.0.params", "acva", "train_split_ratio"), ("sweep.tasks.1.params", "freeform_cidar", "max_output_chars")]
    assert "does not declare a parameter 'train_split_ratio'" in pw[0]["msg"]
    assert 'should be int, got "2400" (str)' in pw[1]["msg"]
    assert any(w.startswith("sweep.tasks.0.params: acva does not declare") for w in v["warnings"])
    # the same output_dir as twin.yaml → warned; excluded when the config *is* twin.yaml
    assert any("also the output_dir of twin.yaml" in w for w in v["warnings"])
    v2 = validate_config(tmp_repo, parse_yaml(MINI.replace('name: "mini"', 'name: "twin"')), file="twin.yaml")
    assert any("also the output_dir of mini.yaml" in w for w in v2["warnings"])
    assert not any("twin.yaml" in w for w in v2["warnings"])
    rc = read_config(tmp_repo, "mini.yaml")
    assert rc["param_warnings"] == pw and rc["warnings"] == v["warnings"]


def test_repo_configs_produce_no_param_warnings():
    for p in sorted((REPO / "configs" / "experiments").glob("*.yaml")):
        rc = read_config(REAL, p.name)
        assert rc["valid"], p.name
        from arabic_eval.config import ExperimentConfig
        assert param_warnings(ExperimentConfig(**rc["resolved"])) == [], p.name


def test_assistant_reference_lists_the_task_params():
    from arabic_eval.tools import config_assistant as ca
    text = ca._presets_text(schema_bundle(REAL))
    assert "### task params" in text
    assert "max_output_chars (int, default 2400)" in text
    assert "- freeform_cidar:" in text and "stop_markers (list[str], default a list of 10 strings" in text
    assert "num_fewshot (int, default 0): In-context demonstrations" in text
    assert "generated documentation" in ca.PRIMER
