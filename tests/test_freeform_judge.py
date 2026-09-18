"""Judge stage (``judge/freeform_judge.py``): config validation, prompt
building, verdict parsing (fenced / plain / embedded JSON, prose fallback,
malformed), the paired statistics, and ``run`` on a fake two-cell sweep with
a fake backend — judge files, all_metrics merge, baseline deltas, agreement
between two judges, file reuse, overwrite, limit, report regeneration."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.judge import freeform_judge as J  # noqa: E402
from arabic_eval.utils.io import write_report_table  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class TestConfig:
    def test_committed_yamls_load(self):
        g = J.JudgeConfig.from_yaml(REPO / "configs/judges/gemma4_31b_local.yaml")
        o = J.JudgeConfig.from_yaml(REPO / "configs/judges/gpt56_terra_api.yaml")
        assert g.backend == "vllm" and g.structured_json and g.max_model_len == 4096
        assert o.backend == "openai" and o.api_key_env == "OPENAI_API_KEY" and o.model == "gpt-5.6-terra"
        # verified 2026-09-18: the model rejects an explicit temperature; low reasoning effort keeps it fast
        assert o.temperature is None and o.extra_body == {"reasoning_effort": "low"} and o.max_tokens == 1024

    def test_validation(self, tmp_path):
        with pytest.raises(ValueError, match="backend"):
            J.JudgeConfig(name="x", backend="hf", model="m")
        with pytest.raises(ValueError, match="rubric"):
            J.JudgeConfig(name="x", backend="vllm", model="m", rubric="nope")
        with pytest.raises(ValueError, match="file-safe"):
            J.JudgeConfig(name="a b", backend="vllm", model="m")
        p = tmp_path / "j.yaml"
        p.write_text("judge: {name: j, backend: openai, model: m, temperture: 0}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown judge keys"):
            J.JudgeConfig.from_yaml(p)

    def test_openai_backend_needs_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            J.OpenAICompatibleBackend(J.JudgeConfig(name="j", backend="openai", model="m"))

    def test_openai_body_shape(self, monkeypatch):
        monkeypatch.setenv("MYKEY", "k")
        cfg = J.JudgeConfig(name="j", backend="openai", model="m", api_key_env="MYKEY", temperature=None, seed=None,
                            extra_body={"reasoning_effort": "low"})
        b = J.OpenAICompatibleBackend(cfg)._body([{"role": "user", "content": "x"}])
        assert b["max_completion_tokens"] == 256 and "temperature" not in b and "seed" not in b
        assert b["response_format"]["json_schema"]["strict"] is True and b["reasoning_effort"] == "low"
        cfg2 = J.JudgeConfig(name="j", backend="openai", model="m", api_key_env="MYKEY", structured_json=False)
        assert "response_format" not in J.OpenAICompatibleBackend(cfg2)._body([])


class TestPromptAndParsing:
    def test_messages(self):
        m = J.build_messages("اكتب", "", "مرجع", "جواب")
        assert len(m) == 1 and m[0]["role"] == "user"
        assert "### Instruction\nاكتب\n\n### Reference answer\nمرجع\n\n### Candidate answer\nجواب" in m[0]["content"]
        assert "### Input" not in m[0]["content"]
        m2 = J.build_messages("لخص", "نص", "مرجع", "   ")
        assert "### Input\nنص" in m2[0]["content"] and J.EMPTY_CANDIDATE in m2[0]["content"]

    @pytest.mark.parametrize("text", [
        '{"score": 4, "correctness": 4, "fluency": 5, "instruction_following": 3, "flags": [], "rationale": "ok"}',
        '```json\n{"score": 4, "correctness": 4, "fluency": 5, "instruction_following": 3, "flags": [], "rationale": "ok"}\n```',
        'Sure. Here is my verdict:\n{"score": 4, "correctness": 4, "fluency": 5, "instruction_following": 3, "flags": ["bogus"], "rationale": "ok"}\nDone.',
    ])
    def test_json_variants(self, text):
        v = J.parse_verdict(text)
        assert v.parse_ok and (v.score, v.correctness, v.fluency, v.instruction_following) == (4, 4, 5, 3)
        assert v.flags == [] and v.rationale == "ok"

    def test_prose_fallback(self):
        v = J.parse_verdict("The answer is a repetition loop. Score: 1. correctness: 1, fluency = 2, instruction_following: 1")
        assert v.parse_ok and v.score == 1 and v.fluency == 2 and "repetition" in v.flags

    def test_malformed(self):
        v = J.parse_verdict("no verdict here")
        assert not v.parse_ok and v.score is None
        v = J.parse_verdict('{"score": 9, "flags": "x"}')
        assert not v.parse_ok and v.score is None and v.flags == []
        assert not J.parse_verdict("").parse_ok

    def test_schema_is_strict_and_closed(self):
        assert J.VERDICT_SCHEMA["additionalProperties"] is False
        assert set(J.VERDICT_SCHEMA["required"]) == set(J.VERDICT_SCHEMA["properties"])


class TestStats:
    def test_paired_bootstrap(self):
        base = {f"r{i}": 3.0 for i in range(40)}
        other = {f"r{i}": (4.0 if i % 2 else 3.0) for i in range(40)}
        s = J.paired_bootstrap(base, other, n_boot=500, seed=1)
        assert s["n"] == 40 and s["delta_mean"] == 0.5 and s["win_rate"] == 0.5 and s["tie_rate"] == 0.5
        assert s["ci_low"] > 0 and s["ci_high"] < 1
        assert J.paired_bootstrap({}, {"a": 1})["n"] == 0

    def test_agreement_measures(self):
        assert J.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
        assert J.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
        assert J.spearman([1, 1, 2], [1, 1, 2]) == 1.0
        assert J.quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
        assert J.quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0
        assert J.exact_agreement([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------
# fake sweep + fake backend
# --------------------------------------------------------------------------

class FakeBackend:
    """Scores by candidate text: 'good' → 5, 'meh' → 3, loop → 1; counts calls."""
    def __init__(self, fenced: bool = False) -> None:
        self.calls = 0
        self.fenced = fenced

    def complete(self, conversations):
        self.calls += len(conversations)
        out = []
        for c in conversations:
            text = c[0]["content"].split("### Candidate answer\n", 1)[1]
            if "good" in text:
                v = {"score": 5, "correctness": 5, "fluency": 5, "instruction_following": 5, "flags": [], "rationale": "fine"}
            elif "meh" in text:
                v = {"score": 3, "correctness": 3, "fluency": 4, "instruction_following": 3, "flags": [], "rationale": "so-so"}
            else:
                v = {"score": 1, "correctness": 1, "fluency": 1, "instruction_following": 1, "flags": ["repetition"], "rationale": "loop"}
            s = json.dumps(v)
            out.append(f"```json\n{s}\n```" if self.fenced else s)
        return out


def _cell(sweep: Path, name: str, gens: List[Dict[str, str]]) -> Path:
    cell = sweep / name
    cell.mkdir(parents=True)
    (cell / "all_metrics.json").write_text(json.dumps({
        "downstream": {"freeform_cidar": {"chrf": 10.0, "num_samples": len(gens), "status": "ok"}, "acva": {"accuracy": 0.5}},
        "intrinsic": {"fertility": 1.0}}), encoding="utf-8")
    rows = [{"id": f"cidar-{i}", "stratum": "short" if i % 2 else "long", "instruction": f"سؤال {i}", "context": "",
             "prompt_text": f"السؤال: سؤال {i}\nالإجابة:", "reference": "مرجع", "generation": g}
            for i, g in enumerate(gens)]
    write_report_table(cell / "eval_rows" / "freeform_cidar.parquet", rows,
                       ["id", "stratum", "instruction", "context", "prompt_text", "reference", "generation"],
                       metadata={"task": "freeform_cidar", "kind": "freeform_generations", "heldout_sha256": "abc"})
    return cell


@pytest.fixture
def sweep(tmp_path) -> Path:
    s = tmp_path / "sweep"
    _cell(s, "native_llama", ["good"] * 6 + ["meh"] * 2)
    _cell(s, "bpe_32k", ["good"] * 2 + ["meh"] * 4 + ["الطبيب الطبيب الطبيب"] * 2)
    (s / "no_gens").mkdir()
    (s / "no_gens" / "all_metrics.json").write_text(json.dumps({"downstream": {"freeform_cidar": {"status": "generation_unsupported"}}}), encoding="utf-8")
    return s


class TestRun:
    def test_end_to_end(self, sweep):
        cfg = J.JudgeConfig(name="fake_a", backend="vllm", model="fake/a")
        be = FakeBackend(fenced=True)
        rep = J.run(sweep, [cfg], {"fake_a": be}, regenerate_report=True)
        assert be.calls == 16 and rep["baseline"] == "native_llama"
        assert rep["cells_with_generations"] == ["bpe_32k", "native_llama"]
        # judge files + all_metrics merge
        for cell in ("native_llama", "bpe_32k"):
            assert (sweep / cell / "freeform_judge" / "fake_a.parquet").exists()
            am = json.loads((sweep / cell / "all_metrics.json").read_text(encoding="utf-8"))
            s = am["downstream"]["freeform_cidar"]["judge"]["fake_a"]
            assert s["n"] == 8 and s["parse_fail_rate"] == 0.0 and s["model"] == "fake/a"
            assert am["downstream"]["freeform_cidar"]["chrf"] == 10.0          # untouched
        base = rep["judges"]["fake_a"]["native_llama"]
        other = rep["judges"]["fake_a"]["bpe_32k"]
        assert base["score_mean"] == 4.5 and base["vs_baseline"] is None and base["baseline"] == "native_llama"
        assert other["score_mean"] == 3.0 and other["flag_rates"]["repetition"] == 0.25 and other["any_flag_rate"] == 0.25
        assert other["vs_baseline"]["delta_mean"] == -1.5 and other["vs_baseline"]["ci_high"] < 0
        assert other["vs_baseline"]["win_rate"] == 0.0 and other["score_by_stratum"] == {"long": 3.0, "short": 3.0}
        assert other["score_hist"] == {"1": 2, "2": 0, "3": 4, "4": 0, "5": 2}
        # report regenerated with the judge section
        txt = (sweep / "comparison_report.txt").read_text(encoding="utf-8")
        assert "LLM judge: fake_a" in txt and "baseline" in txt and "judge.fake_a" not in txt
        assert "no_gens" in txt                                                   # other cells still reported
        assert J.format_table(rep).count("fake_a") >= 1

    def test_reuse_overwrite_limit_and_agreement(self, sweep):
        cfg_a = J.JudgeConfig(name="fake_a", backend="vllm", model="fake/a")
        cfg_b = J.JudgeConfig(name="fake_b", backend="openai", model="fake/b")
        be_a, be_b = FakeBackend(), FakeBackend()
        J.run(sweep, [cfg_a], {"fake_a": be_a}, regenerate_report=False)
        assert be_a.calls == 16
        J.run(sweep, [cfg_a], {"fake_a": be_a}, regenerate_report=False)           # files reused
        assert be_a.calls == 16
        J.run(sweep, [cfg_a], {"fake_a": be_a}, overwrite=True, regenerate_report=False)
        assert be_a.calls == 32
        rep = J.run(sweep, [cfg_b], {"fake_b": be_b}, limit=4, regenerate_report=False)
        assert be_b.calls == 8 and rep["judges"]["fake_b"]["bpe_32k"]["n"] == 4
        # two judge files per cell → agreement recorded (on the 4 shared ids)
        am = json.loads((sweep / "bpe_32k" / "all_metrics.json").read_text(encoding="utf-8"))
        ag = am["downstream"]["freeform_cidar"]["judge_agreement"]["fake_a|fake_b"]
        assert ag["n"] == 4 and ag["exact"] == 1.0 and ag["qwk"] == 1.0
        assert set(am["downstream"]["freeform_cidar"]["judge"]) == {"fake_a", "fake_b"}

    def test_single_cell_and_explicit_baseline(self, sweep):
        cfg = J.JudgeConfig(name="fake_a", backend="vllm", model="fake/a")
        rep = J.run(sweep / "bpe_32k", [cfg], {"fake_a": FakeBackend()})
        assert rep["cells"] == ["bpe_32k"] and rep["baseline"] == "bpe_32k" and "comparison_report" not in rep
        rep2 = J.run(sweep, [cfg], {"fake_a": FakeBackend()}, baseline="bpe_32k", regenerate_report=False)
        assert rep2["judges"]["fake_a"]["native_llama"]["vs_baseline"]["delta_mean"] == 1.5
        with pytest.raises(ValueError, match="baseline cell"):
            J.run(sweep, [cfg], {"fake_a": FakeBackend()}, baseline="nope")

    def test_parse_failures_are_counted_not_fatal(self, sweep):
        class Broken:
            def complete(self, convs):
                return ["garbage"] * len(convs)
        cfg = J.JudgeConfig(name="broken", backend="vllm", model="x")
        rep = J.run(sweep, [cfg], {"broken": Broken()}, regenerate_report=False)
        s = rep["judges"]["broken"]["native_llama"]
        assert s["parse_fail_rate"] == 1.0 and s["score_mean"] is None and s["vs_baseline"] is None
