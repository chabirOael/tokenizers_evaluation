"""``arabic_eval.distill``: the teacher chat turn, the strata, the P1.6 drop rules and the
preamble counter, the eval-row post-processing of a pseudo-cell, the seeded subset rule
(an extended subset keeps the earlier records) and the held-out overlap check."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.data import finetune_corpora as T  # noqa: E402
from arabic_eval.data.finetune_corpora import QARecord  # noqa: E402
from arabic_eval.distill import postprocess as PP  # noqa: E402
from arabic_eval.distill import prompts as PR  # noqa: E402
from arabic_eval.distill import teacher as TE  # noqa: E402

spec = importlib.util.spec_from_file_location("build_pseudo_cell", REPO / "scripts" / "distill" / "build_pseudo_cell.py")
BPC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BPC)  # type: ignore[union-attr]

LONG = "هذه إجابة عربية كاملة تكفي لتجاوز الحد الأدنى من الطول المطلوب."


def test_conversation_with_and_without_a_system_role():
    conv = TE.build_conversation("اكتب قصيدة", "")
    assert conv == [{"role": "system", "content": TE.SYSTEM_PROMPT}, {"role": "user", "content": "اكتب قصيدة"}]
    conv = TE.build_conversation("لخص النص", "النص هنا", system_role=False)
    assert conv == [{"role": "user", "content": f"{TE.SYSTEM_PROMPT}\n\nلخص النص\n\nالنص هنا"}]
    assert TE.user_turn(" س ", " ") == "س"
    assert len(TE.sha256_text(TE.SYSTEM_PROMPT)) == 64


def test_strata_follow_the_heldout_bounds():
    assert [TE.stratum_of(n) for n in (25, 141, 142, 300, 301, 1500)] == ["short", "short", "medium", "medium", "long", "long"]
    bounds = json.loads((REPO / "configs/contamination/freeform_cidar_heldout_v1.manifest.json").read_text())["strata"]["boundaries_ref_chars"]
    assert tuple(bounds) == TE.DEFAULT_STRATA_BOUNDS


def test_candidates_yaml_loads_and_excludes_judge_and_qwen_families():
    _d, cands = TE.load_candidates(REPO / "configs/distill/teacher_candidates.yaml")
    slugs = [c.slug for c in cands]
    assert slugs == ["jais2_8b", "llama33_70b_awq", "falcon_h1_34b", "cmdr7b_arabic", "aya_expanse_32b", "allam_7b"]
    assert all(len(c.revision) == 40 and c.params_b for c in cands)
    assert not any(k in c.repo.lower() for c in cands for k in ("gemma", "fanar", "qwen"))
    assert TE.candidate(REPO / "configs/distill/teacher_candidates.yaml", "aya_expanse_32b")[1].model == "CohereForAI/aya-expanse-32b"


@pytest.mark.parametrize("text,finish,instruction,reason", [
    (LONG, "length", "س", "length"),
    ("ا" * 1200 + " " + "ب" * 1250, "stop", "س", "char_truncated"),
    ("مقدمة قصيرة ثم " + " ".join(["نفس الكلمة"] * 12), "stop", "س", "loop"),
    ("قصير", "stop", "س", "empty"),
    ("", "stop", "س", "empty"),
    (LONG + " LinkedIn", "stop", "س", "latin"),
    (LONG + "\n" + T.ANSWER_LABEL, "stop", "س", "echo"),
    (LONG, "stop", LONG, "echo"),
    (LONG, "stop", "سؤال آخر", None),
])
def test_drop_rules_in_order(text, finish, instruction, reason):
    assert PP.drop_reason(text, finish, instruction) == reason


def test_drop_order_puts_length_before_everything():
    looped = "مقدمة " + " ".join(["نفس الكلمة"] * 12)
    assert PP.drop_reason(looped, "length", "س") == "length"
    assert PP.drop_reason("HTML " + looped, "stop", "س") == "loop"      # loop before latin
    assert PP.drop_reason(looped + " HTML", "stop", "س") == "degenerate"  # a broken tail: the density rule


def test_preamble_counter():
    assert PP.has_preamble("بالتأكيد! إليك الإجابة")
    assert PP.has_preamble("إليك قائمة بالأفكار")
    assert PP.has_preamble("فيما يلي خمس نقاط")
    assert PP.has_preamble("حسنا، سأبدأ")
    assert not PP.has_preamble("نعمة الله كبيرة")                   # a word that starts with نعم is not the preamble
    assert not PP.has_preamble("القاهرة عاصمة مصر")


def test_eval_row_applies_the_eval_text_rules_without_markers():
    loop_text = "بداية الجواب ثم " + " ".join(["العالقين :"] * 6)
    ev = PP.eval_row(loop_text, "length")
    assert ev["stop_reason"] == "loop" and ev["hit_loop"] and ev["loop_rule"] == "period" and ev["loop_period"] == 2
    assert ev["generation"] == "بداية الجواب ثم العالقين :" and ev["generation_raw"] == loop_text and ev["degenerate"]
    ev = PP.eval_row("جواب\n### السؤال:\nنص بعده", "stop")        # our markers are NOT applied to a teacher
    assert ev["stop_reason"] == "eos" and "### السؤال" in ev["generation"]
    ev = PP.eval_row("ب" * 3000, "length")
    assert ev["stop_reason"] == "loop"                                 # a letter ×20 is the char rule
    ev = PP.eval_row("كلمة " * 10 + "x" * 3, "length")
    assert ev["latin"] and ev["stop_reason"] in ("cap", "loop")
    ev = PP.eval_row(LONG + " " + "ج" * 2500, "stop")
    assert ev["char_truncated"] or ev["stop_reason"] == "loop"


def test_build_rows_of_a_pseudo_cell():
    prompts = [{"id": "a", "instruction": "س1", "context": "", "reference": "مرجع", "stratum": "short"},
               {"id": "b", "prompt": "س2", "reference": "مرجع طويل", "stratum": "long"}]
    raw = {"a": {"id": "a", "text": LONG, "finish_reason": "stop", "n_tokens": 12, "prompt_tokens": 40, "prompt_text": "<chat>"},
           "b": {"id": "b", "text": "نص", "finish_reason": "length", "n_tokens": 7, "prompt_tokens": 30, "prompt_text": "<chat2>"}}
    rows = BPC.build_rows(prompts, raw, gen_wall=4.0)
    assert [r["id"] for r in rows] == ["a", "b"] and rows[1]["instruction"] == "س2"
    assert rows[0]["stop_reason"] == "eos" and rows[1]["stop_reason"] == "cap" and rows[1]["hit_cap"]
    assert rows[0]["prompt_text"] == "<chat>" and rows[0]["gen_time_sec"] == 2.0
    assert rows[0]["reference_roundtrip_chrf"] is None and rows[0]["gen_chars"] == len(LONG)
    from arabic_eval.tasks.freeform.cidar import ROW_FIELDS
    assert set(ROW_FIELDS) <= set(rows[0])
    with pytest.raises(SystemExit, match="no teacher answer"):
        BPC.build_rows(prompts, {"a": raw["a"]}, gen_wall=1.0)


def test_seeded_subset_is_deterministic_and_extends_by_prefix():
    recs = [QARecord(id=f"r{i:03d}", question="س", context="", answer="ج", source="cidar", prompt_template="instruction")
            for i in range(50)]
    a = [r.id for r in PR.seeded_subset(recs, 10)]
    b = [r.id for r in PR.seeded_subset(list(reversed(recs)), 10)]
    assert a == b and len(set(a)) == 10                                  # loader order is irrelevant
    assert [r.id for r in PR.seeded_subset(recs, 20)][:10] == a          # extending keeps the first N
    assert len(PR.seeded_subset(recs, None)) == 50


def test_heldout_overlap_catches_ids_and_normalized_text(tmp_path):
    held = tmp_path / "held.jsonl"
    held.write_text(json.dumps({"id": "cidar-1", "prompt": "ما هي عاصمة مصر؟", "reference": "القاهرة"}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    ok = PR.heldout_overlap([{"id": "cidar-2", "instruction": "اكتب قصيدة", "context": ""}], held)
    assert ok["passed"] and ok["heldout_rows"] == 1
    bad = PR.heldout_overlap([{"id": "cidar-1", "instruction": "x", "context": ""},
                              {"id": "cidar-3", "instruction": "ما هى عاصمة مصر", "context": ""}], held)   # ى / ي and ? folded
    assert not bad["passed"] and bad["id_overlap"] == ["cidar-1"] and bad["normalized_text_matches"] == ["cidar-3"]
