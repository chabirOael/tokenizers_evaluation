"""Console back-ends for the free-form eval: the rows browser
(``tools/freeform_rows_browser.py``: discovery incl. unsupported cells, the
judge join, filters / sorts / paging / summary, one row across cells) and the
blind rating set (``tools/freeform_rating.py``: build, hidden cells, balanced
variants, disagreement half, submit + progress, agreement vs judges)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.tools import freeform_rating as R  # noqa: E402
from arabic_eval.tools import freeform_rows_browser as B  # noqa: E402
from arabic_eval.utils.io import write_report_table  # noqa: E402

GEN_FIELDS = ["id", "stratum", "instruction", "context", "prompt_text", "reference", "generation", "generation_raw",
              "prompt_tokens", "gen_tokens", "gen_chars", "ref_chars", "stop_reason", "hit_cap", "char_truncated",
              "empty", "degenerate", "latin", "arabic_letter_ratio", "chrf", "bertscore_p", "bertscore_r", "bertscore_f1",
              "reference_roundtrip_chrf", "gen_time_sec"]
JUDGE_FIELDS = ["id", "stratum", "score", "correctness", "fluency", "instruction_following", "flags", "rationale", "parse_ok", "raw"]


def _gen_rows(n: int, quality: float) -> List[Dict]:
    rows = []
    for i in range(n):
        gen = "" if i == 0 else ("الطبيب " * 8 if i == 1 else f"جواب رقم {i} " * (1 + i % 3))
        rows.append({"id": f"cidar-{i}", "stratum": ("short", "medium", "long")[i % 3], "instruction": f"سؤال {i}", "context": "",
                     "prompt_text": f"السؤال: سؤال {i}\nالإجابة:", "reference": f"مرجع {i}", "generation": gen.strip(),
                     "generation_raw": gen, "prompt_tokens": 5, "gen_tokens": len(gen.split()), "gen_chars": len(gen.strip()),
                     "ref_chars": 6, "stop_reason": ("eos", "marker", "cap")[i % 3], "hit_cap": i % 3 == 2, "char_truncated": False,
                     "empty": i == 0, "degenerate": i == 1, "latin": False, "arabic_letter_ratio": 1.0,
                     "chrf": round(quality * 100 * (i + 1) / n, 2), "bertscore_p": 0.8, "bertscore_r": 0.8, "bertscore_f1": 0.8,
                     "reference_roundtrip_chrf": 100.0, "gen_time_sec": 0.1})
    return rows


def _judge_rows(n: int, offset: int, judge_bias: int = 0) -> List[Dict]:
    return [{"id": f"cidar-{i}", "stratum": ("short", "medium", "long")[i % 3],
             "score": max(1, min(5, 1 + (i + offset) % 5 + judge_bias)), "correctness": 3, "fluency": 4, "instruction_following": 3,
             "flags": "repetition" if i == 1 else "", "rationale": f"r{i}", "parse_ok": True, "raw": "{}"} for i in range(n)]


def _cell(root: Path, sweep: str, name: str, tokenizer: str, n: int = 12, quality: float = 1.0, judges=(), unsupported=False) -> Path:
    cell = root / "outputs" / "experiments" / sweep / name if sweep else root / "outputs" / "experiments" / name
    cell.mkdir(parents=True)
    (cell / "config.json").write_text(json.dumps({"tokenizer": {"type": tokenizer, "vocab_size": 32000}, "model": {"name_or_path": "m"}}), encoding="utf-8")
    task = {"status": "generation_unsupported", "reason": "no decoding"} if unsupported else {"status": "ok", "chrf": 42.0, "num_samples": n}
    (cell / "all_metrics.json").write_text(json.dumps({"downstream": {"freeform_cidar": task}}), encoding="utf-8")
    if not unsupported:
        write_report_table(cell / "eval_rows" / "freeform_cidar.parquet", _gen_rows(n, quality), GEN_FIELDS,
                           metadata={"task": "freeform_cidar", "kind": "freeform_generations", "token_cap": 300, "decoding": {"sampling": "greedy"}})
        for jname, offset, bias in judges:
            write_report_table(cell / "freeform_judge" / f"{jname}.parquet", _judge_rows(n, offset, bias), JUDGE_FIELDS,
                               metadata={"judge": {"model": f"model/{jname}"}})
    return cell


@pytest.fixture
def repo(tmp_path) -> Path:
    _cell(tmp_path, "sweep", "native_llama", "native_llama", judges=[("ja", 0, 0), ("jb", 0, 1)])
    _cell(tmp_path, "sweep", "bpe_32k", "bpe", quality=0.5, judges=[("ja", 2, 0), ("jb", 2, 0)])
    _cell(tmp_path, "sweep", "charformer", "charformer", unsupported=True)
    _cell(tmp_path, "", "single_exp", "native_qwen3")          # a single experiment: cell dir == experiment dir
    return tmp_path


class TestBrowser:
    def test_discover_lists_generated_and_unsupported_cells(self, repo):
        d = B.discover(repo)
        exps = {e["experiment"]: e for e in d["experiments"]}
        assert set(exps) == {"sweep", "single_exp"} and exps["single_exp"]["single"]
        sw = exps["sweep"]
        assert [c["cell"] for c in sw["cells"]] == ["bpe_32k", "native_llama", "charformer"]   # unsupported last
        cf = sw["cells"][2]
        assert cf["has_generations"] is False and cf["status"] == "generation_unsupported" and cf["reason"]
        nl = sw["cells"][1]
        assert nl["n_rows"] == 12 and nl["tokenizer"] == "native_llama" and nl["token_cap"] == 300
        assert sw["judges"] == [] and nl["judges"] == []       # judges from all_metrics (merged by the judge stage), none here

    def test_query_joins_judges_and_filters(self, repo):
        cell = "outputs/experiments/sweep/native_llama"
        d = B.query(repo, cell, {})
        assert d["total"] == 12 and d["judge"] == "ja" and [j["name"] for j in d["judges"]] == ["ja", "jb"]
        r0 = d["rows"][0]
        assert r0["judges"]["ja"]["score"] == 1 and r0["judges"]["jb"]["score"] == 2 and r0["disagreement"] == 1.0
        s = d["summary"]
        assert s["n_rows"] == 12 and s["empty"] == 1 and s["degenerate"] == 1 and s["stop"] == {"eos": 4, "marker": 4, "cap": 4}
        assert s["judges"]["ja"]["hist"] == [3, 3, 2, 2, 2] and s["judges"]["ja"]["flagged"] == 1
        assert s["judges"]["ja"]["unparsed"] == 0 and s["judges"]["ja"]["missing"] == 0
        assert B.query(repo, cell, {"score_min": 4, "judge": "ja"})["total"] == 4
        assert B.query(repo, cell, {"jflag": "repetition"})["total"] == 1
        assert B.query(repo, cell, {"jflag": "any", "judge": "jb"})["total"] == 1
        assert B.query(repo, cell, {"stop": "cap"})["total"] == 4
        assert B.query(repo, cell, {"degenerate": "1"})["rows"][0]["id"] == "cidar-1"
        assert B.query(repo, cell, {"empty": "0"})["total"] == 11
        assert B.query(repo, cell, {"stratum": "long"})["total"] == 4
        assert B.query(repo, cell, {"q": "سؤال 7"})["total"] == 1
        assert [r["id"] for r in B.query(repo, cell, {"sort": "chrf_desc", "page_size": 2})["rows"]] == ["cidar-11", "cidar-10"]
        top = B.query(repo, cell, {"sort": "score_desc", "judge": "jb"})["rows"][0]
        assert top["judges"]["jb"]["score"] == 5
        p = B.query(repo, cell, {"page_size": 5, "page": 3})
        assert p["pages"] == 3 and len(p["rows"]) == 2 and p["page"] == 3
        with pytest.raises(B.FreeformError, match="unknown judge"):
            B.query(repo, cell, {"judge": "nope"})
        with pytest.raises(B.FreeformError):
            B.query(repo, "outputs/experiments/sweep/charformer", {})
        # a judge file covering only some rows (judged with --limit) → missing, not unparsed
        import pyarrow.parquet as pq
        from arabic_eval.utils.io import write_report_table
        jp = repo / "outputs/experiments/sweep/native_llama/freeform_judge/ja.parquet"
        rows = pq.read_table(jp).to_pylist()[:4]
        write_report_table(jp, rows, JUDGE_FIELDS, metadata={"judge": {"model": "model/ja"}})
        s2 = B.query(repo, cell, {})["summary"]["judges"]["ja"]
        assert s2["n"] == 4 and s2["missing"] == 8 and s2["unparsed"] == 0

    def test_row_across_cells(self, repo):
        d = B.row(repo, "outputs/experiments/sweep/native_llama", "cidar-3")
        assert d["row"]["prompt_text"].endswith("الإجابة:") and d["row"]["judges"]["ja"]["rationale"] == "r3"
        assert [a["cell"] for a in d["across"]] == ["bpe_32k"]
        assert d["across"][0]["tokenizer"] == "bpe" and d["across"][0]["judges"] == {"ja": 1, "jb": 1}
        with pytest.raises(B.FreeformError, match="no row"):
            B.row(repo, "outputs/experiments/sweep/native_llama", "cidar-99")

    def test_paths_are_confined(self, repo):
        with pytest.raises(B.FreeformError):
            B.query(repo, "configs", {})
        with pytest.raises(B.FreeformError):
            B.query(repo, "../..", {})


class TestRating:
    def test_build_is_blind_balanced_and_half_disagreement(self, repo):
        info = R.build_set(repo, "outputs/experiments/sweep", name="v1", n_prompts=6, variants_per_prompt=2, seed=1)
        assert info["baseline"] == "native_llama" and info["n_items"] == 12 and info["judges"] == ["ja", "jb"]
        assert len(info["selection"]["disagreement"]) == 3 and len(info["selection"]["random"]) == 3
        assert not set(info["selection"]["disagreement"]) & set(info["selection"]["random"])
        items = R.get_items(repo, "outputs/experiments/sweep", "v1", "alice")["items"]
        assert len(items) == 12 and all("hidden" not in it and it["rating"] is None for it in items)
        doc = json.loads((repo / "outputs/experiments/sweep/freeform_rating/v1.json").read_text(encoding="utf-8"))
        cells = [it["hidden"]["cell"] for it in doc["items"]]
        assert cells.count("native_llama") == 6 and cells.count("bpe_32k") == 6
        assert [it["position"] for it in doc["items"]] == list(range(12))
        with pytest.raises(R.FreeformError, match="exists"):
            R.build_set(repo, "outputs/experiments/sweep", name="v1", n_prompts=6)
        sets = R.list_sets(repo, "outputs/experiments/sweep")
        assert sets["sets"][0]["name"] == "v1" and sets["sets"][0]["raters"] == [] and "charformer" not in sets["cells"]

    def test_submit_progress_and_agreement(self, repo):
        R.build_set(repo, "outputs/experiments/sweep", name="v1", n_prompts=6, variants_per_prompt=2, seed=1)
        doc = json.loads((repo / "outputs/experiments/sweep/freeform_rating/v1.json").read_text(encoding="utf-8"))
        # alice copies judge ja exactly; bob rates everything 3
        ja = {c: {r["id"]: r["score"] for r in _judge_rows(12, 0 if c == "native_llama" else 2)} for c in ("native_llama", "bpe_32k")}
        for it in doc["items"]:
            score = ja[it["hidden"]["cell"]][it["prompt_id"]]
            out = R.submit(repo, "outputs/experiments/sweep", "v1", "alice", it["item_id"], score, ["repetition"] if score == 1 else [], "n")
            R.submit(repo, "outputs/experiments/sweep", "v1", "bob", it["item_id"], 3)
        assert out["progress"] == {"rated": 12, "total": 12}
        got = R.get_items(repo, "outputs/experiments/sweep", "v1", "alice")
        assert got["progress"]["rated"] == 12 and got["items"][0]["rating"]["score"] in range(1, 6)
        ag = R.agreement(repo, "outputs/experiments/sweep", "v1")
        assert ag["raters"]["alice"]["vs_judge"]["ja"]["spearman"] == 1.0 and ag["raters"]["alice"]["vs_judge"]["ja"]["exact"] == 1.0
        assert ag["raters"]["alice"]["vs_judge"]["ja"]["n"] == 12
        assert ag["raters"]["bob"]["vs_judge"]["ja"]["spearman"] is None or ag["raters"]["bob"]["vs_judge"]["ja"]["exact"] < 1
        assert "alice|bob" in ag["rater_vs_rater"] and ag["judge_vs_judge_on_items"]["ja|jb"]["n"] == 12
        assert set(ag["per_cell"]) == {"native_llama", "bpe_32k"} and ag["per_cell"]["bpe_32k"]["human_n"] == 12
        assert (repo / "outputs/experiments/sweep/freeform_rating/v1.agreement.json").exists()
        assert R.list_sets(repo, "outputs/experiments/sweep")["sets"][0]["raters"] == [{"rater": "alice", "n_rated": 12}, {"rater": "bob", "n_rated": 12}]

    def test_validation(self, repo):
        R.build_set(repo, "outputs/experiments/sweep", name="v1", n_prompts=4, variants_per_prompt=2)
        with pytest.raises(R.FreeformError, match="1–5"):
            R.submit(repo, "outputs/experiments/sweep", "v1", "alice", "cidar-0::native_llama", 7)
        with pytest.raises(R.FreeformError, match="unknown item"):
            R.submit(repo, "outputs/experiments/sweep", "v1", "alice", "nope", 3)
        with pytest.raises(R.FreeformError, match="rater"):
            R.submit(repo, "outputs/experiments/sweep", "v1", "a b/c", "cidar-0::native_llama", 3)
        with pytest.raises(R.FreeformError, match="no rating set"):
            R.get_items(repo, "outputs/experiments/sweep", "v9", "alice")

    def test_single_cell_experiment_without_judges(self, repo):
        info = R.build_set(repo, "outputs/experiments/single_exp", name="v1", n_prompts=5, variants_per_prompt=3)
        assert info["variants_per_prompt"] == 1 and info["n_items"] == 5 and info["selection"]["disagreement"] == []
