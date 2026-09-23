"""Free-form generation task (``tasks/freeform``): prompt equality with the
training template, the character budget → token cap, EOS stripping, marker
truncation, the degeneration detector, the typed status for non-generative
tokenizers, and an end-to-end ``evaluate`` on a tiny random Llama with a
word-level stub tokenizer (real HF greedy generation on CPU, row dump, metrics)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from arabic_eval.data import finetune_corpora as fc  # noqa: E402
from arabic_eval.data.finetune_corpora import QARecord, _format_qa_full, _format_qa_prompt  # noqa: E402
from arabic_eval.data.freeform_heldout import HeldoutRow, write_heldout_jsonl  # noqa: E402
from arabic_eval.models.llama_adapter import LlamaAdapter  # noqa: E402
from arabic_eval.tasks.freeform import metrics as M  # noqa: E402
from arabic_eval.tasks.freeform.cidar import METRIC_NAMES, ROW_FIELDS, FreeformCidarTask  # noqa: E402
from arabic_eval.tasks.freeform import generation as G  # noqa: E402
from arabic_eval.tasks.freeform.generation import (  # noqa: E402
    STOP_REASONS, DecodingConfig, derive_token_cap, generate_freeform, generation_supported,
    measure_chars_per_token, strip_trailing_eos, text_stop, truncate_at_marker,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

VOCAB = 96
PAD, BOS, EOS, UNK = 0, 1, 2, 3
TEXTS = ["الكتاب على الطاولة", "ذهب الولد إلى المدرسة", "يدرس الطلاب اللغة العربية", "كتبت الطالبة رسالة طويلة",
         "في المكتبة كتب كثيرة", "قرأت مقالة عن التاريخ", "السؤال: اكتب جملة الإجابة: نعم"]


class _WordTokenizer(BaseTokenizer):
    """Word-level tokenizer that appends BOS/EOS to every encoding — the
    from-scratch convention the generator has to cope with."""

    def __init__(self, embedding_type: str = EmbeddingType.STANDARD) -> None:
        self._et = embedding_type
        self._w2i: Dict[str, int] = {}
        self._i2w: Dict[int, str] = {}

    def train(self, texts, vocab_size, **kw):
        for t in texts:
            for w in t.split():
                if w not in self._w2i and len(self._w2i) < VOCAB - 4:
                    self._w2i[w] = len(self._w2i) + 4
                    self._i2w[self._w2i[w]] = w

    def encode(self, text, max_length=None, padding=False, truncation=False):
        ids = [BOS] + [self._w2i.get(w, UNK) for w in text.split()] + [EOS]
        if truncation and max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())

    def decode(self, ids):
        return " ".join(self._i2w.get(i, "" if i in (PAD, BOS, EOS) else "<unk>") for i in ids).strip()

    def save(self, path): ...
    def load(self, path): ...

    @property
    def vocab_size(self): return VOCAB
    @property
    def embedding_type(self): return self._et
    @property
    def special_tokens(self): return {"pad_token": PAD, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK}


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory) -> Path:
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama")
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


@pytest.fixture
def adapter(tiny_llama) -> LlamaAdapter:
    return LlamaAdapter(str(tiny_llama), device="cpu", dtype="float32")


@pytest.fixture
def tokenizer() -> _WordTokenizer:
    t = _WordTokenizer()
    t.train(TEXTS, VOCAB)
    return t


@pytest.fixture
def heldout(tmp_path) -> Path:
    rows = [HeldoutRow(id=f"cidar-{i}", prompt=f"اكتب جملة عن {TEXTS[i].split()[0]}", context="",
                       reference=TEXTS[i], stratum="short", ref_chars=len(TEXTS[i])) for i in range(6)]
    p = tmp_path / "heldout.jsonl"
    write_heldout_jsonl(rows, p)
    return p


# --------------------------------------------------------------------------

class TestPromptAndBudget:
    def test_prompt_is_the_training_prompt(self, heldout):
        """The eval prompt is the Phase 3 training prompt of the same instruction,
        byte for byte: the prompt span of the full training text (what the LCP
        masking hides from the loss) equals what the model is asked to continue."""
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "num_fewshot": 3})
        row = task.load_examples()[0]
        rec = QARecord(id=row["id"], question=row["prompt"], context="", answer=row["reference"],
                       source="cidar", prompt_template="instruction")
        p = task.build_prompt(row)
        assert p == _format_qa_prompt(rec)
        assert _format_qa_full(rec) == p + row["reference"]                  # training text = prompt + answer
        assert p.startswith(fc.INSTRUCTION_HEADER + "\n\n" + fc.INSTRUCTION_LABEL + "\n" + row["prompt"])
        assert p.endswith("\n\n" + fc.ANSWER_LABEL + "\n")
        assert fc.INPUT_LABEL not in p and "السياق:" not in p and "السؤال:" not in p

    def test_context_row_gets_the_input_section(self):
        p = FreeformCidarTask.build_prompt({"id": "x", "prompt": "لخص", "context": "نص", "reference": ""})
        assert p == (f"{fc.INSTRUCTION_HEADER_WITH_INPUT}\n\n{fc.INSTRUCTION_LABEL}\nلخص\n\n"
                     f"{fc.INPUT_LABEL}\nنص\n\n{fc.ANSWER_LABEL}\n")
        assert p == _format_qa_prompt(QARecord(id="x", question="لخص", context="نص", answer="",
                                               source="cidar", prompt_template="instruction"))

    def test_token_cap_from_character_budget(self):
        cfg = DecodingConfig(max_output_chars=1200, token_cap_margin=1.15)
        assert derive_token_cap(cfg, 4.0) == 353            # ceil(1200/4*1.15)+8
        assert derive_token_cap(cfg, 1.0) == 1388           # char-level: ~4x the BPE cap
        assert derive_token_cap(cfg, 100.0) == cfg.token_cap_floor
        assert derive_token_cap(cfg, 0.01) == cfg.token_cap_ceiling
        # the default budget (2400 since 2026-09-21) still fits char-JABER under the 4096 ceiling
        default = DecodingConfig()
        assert default.max_output_chars == 2400
        assert derive_token_cap(default, 2.299) == 1209     # native Qwen3 on the references (2.299 chars/token): the measured cap
        assert derive_token_cap(default, 1.0) == 2768 < default.token_cap_ceiling

    def test_chars_per_token_measured_on_the_tokenizer(self, tokenizer):
        cpt = measure_chars_per_token(tokenizer, ["الكتاب على الطاولة"])   # 18 chars / (3 words + BOS + EOS)
        assert cpt == pytest.approx(18 / 5)

    def test_strip_trailing_eos_and_markers(self):
        assert strip_trailing_eos([1, 5, 6, 2], 2) == [1, 5, 6]
        assert strip_trailing_eos([1, 5, 6], 2) == [1, 5, 6]
        assert strip_trailing_eos([1, 5, 6, 2], None) == [1, 5, 6, 2]
        assert truncate_at_marker("جواب\nالسؤال: آخر", ("\nالسؤال:",)) == ("جواب", True)
        assert truncate_at_marker("جواب طويل", ("\nالسؤال:",)) == ("جواب طويل", False)
        assert truncate_at_marker("أ\n### الإجابة:\nب\nالسؤال:ج", ("\nالسؤال:", "\n### الإجابة:")) == ("أ", True)

    def test_default_markers_are_the_templates_exact_labels(self):
        """A stop marker is the model starting a new prompt block in the templates'
        exact words. A bare ``\\n###`` cut every markdown sub-header the model opened
        inside its answer (5 of the 5 marker stops of the untrained Qwen3-4B control were
        ``### ملاحظات:``-style headings, 2026-09-21) and ``\\nفيما يلي`` would cut a
        ``فيما يلي قائمة …`` line, so neither is a marker any more."""
        markers = G.DEFAULT_STOP_MARKERS
        assert "\n###" not in markers and "\nفيما يلي" not in markers
        for label in (fc.INSTRUCTION_LABEL, fc.INPUT_LABEL, fc.CONTEXT_LABEL, fc.QUESTION_LABEL, fc.ANSWER_LABEL):
            assert "\n" + label in markers, label
        for header in (fc.INSTRUCTION_HEADER, fc.INSTRUCTION_HEADER_WITH_INPUT, fc.QA_HEADER):
            assert G.header_restart_marker(header) in markers, header
        assert set(G.HEADER_STOP_MARKERS) == {"\nفيما يلي تعليمات", "\nفيما يلي نص"}
        assert markers[:3] == ("\nالسؤال:", "\nالسياق:", "\nالإجابة:")          # the flat v1 blocks stay
        assert len(markers) == len(set(markers)) == 10
        # a section label or a header restart stops …
        assert truncate_at_marker("جواب.\n### الإجابة:\nآخر", markers) == ("جواب.", True)
        assert truncate_at_marker("جواب.\n### التعليمات:\nآخر", markers) == ("جواب.", True)
        assert truncate_at_marker("جواب.\nفيما يلي تعليمات تصف مهمة.", markers) == ("جواب.", True)
        assert truncate_at_marker("جواب.\nفيما يلي نص وسؤال عنه.", markers) == ("جواب.", True)
        # … the model's own markdown sub-header or list intro does not
        for text in ("النقطة الأولى.\n### مثال متكامل:\nمثال", "شرح.\n### ملاحظات:\nملاحظة",
                     "جواب.\nفيما يلي قائمة بالمتطلبات:\n- أ", "جواب فيما يلي أمثلة", "أ\n#### عنوان\nب"):
            assert truncate_at_marker(text, markers) == (text, False), text
        assert DecodingConfig().stop_markers == markers

    def test_empty_params_build_the_dataclass_defaults(self):
        """An absent key means the ``DecodingConfig`` default (the pipeline hands a task only
        ``sweep.tasks[].params``). ``configs/tasks/freeform_cidar.yaml`` is now generated from
        ``param_spec()`` and pinned by ``tests/test_task_param_specs.py``."""
        assert FreeformCidarTask({}).decoding == DecodingConfig()
        assert FreeformCidarTask({"max_output_chars": 1200}).decoding.max_output_chars == 1200

    def test_generation_support_by_embedding_family(self):
        for et in (EmbeddingType.STANDARD, EmbeddingType.CHAR_JABER):
            assert generation_supported(_WordTokenizer(et))[0]
        for et in (EmbeddingType.CHARACTER_CNN, EmbeddingType.CHARFORMER):
            ok, why = generation_supported(_WordTokenizer(et))
            assert not ok and why


class TestMetrics:
    def test_degeneration_detector(self):
        assert M.is_degenerate("الطبيب الطبيب الطبيب الطبيب الطبيب الطبيب الطبيب")
        assert M.is_degenerate("هذا نص عادي " * 6)                              # repeated 4-grams
        assert M.is_degenerate("ن" * 25)
        assert M.is_degenerate("الطبيب" * 7)                                        # space-less loop, 42 chars
        assert not M.is_degenerate("البومة طائر ليلي بعينين كبيرتين ووجه مستدير أما الصقر فطائر نهاري حاد البصر")
        assert not M.is_degenerate("")

    def test_detect_loop_keeps_the_first_copy(self):
        """The generation-time loop stop and the degeneration metric share one
        detector (a periodic tail, ``tests/test_loop_detector.py`` has the
        rule in full): every text ``detect_loop`` flags is degenerate, and the
        cut keeps exactly the first copy of the repeated unit."""
        cycle = "أ ب ج د ه و"
        t = "مقدمة أولا ثم " + " ".join([cycle] * 3)
        loop = M.detect_loop(t)
        assert loop is not None and loop.rule == "period" and loop.unit == cycle and loop.period == 6
        assert t[: loop.cut] == "مقدمة أولا ثم " + cycle
        # one word five times in a row → keep one
        t = "قال الطبيب الطبيب الطبيب الطبيب الطبيب الطبيب"
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == 1 and t[: loop.cut] == "قال الطبيب"
        # one character twenty times in a row → keep one
        t = "نعم " + "ه" * 30
        loop = M.detect_loop(t)
        assert loop is not None and loop.rule == "char" and t[: loop.cut] == "نعم ه"
        # two copies only, or four distinct words repeated twice: not a loop
        assert M.detect_loop(" ".join([cycle] * 2)) is None
        assert M.detect_loop("الطبيب الطبيب الطبيب الطبيب") is None
        assert M.detect_loop("") is None and M.detect_loop("   ") is None
        for text in ("هذا نص عادي " * 6, "ن" * 25, "قال الطبيب الطبيب الطبيب الطبيب الطبيب"):
            assert M.detect_loop(text) is not None and M.is_degenerate(text)
        assert M.LOOP_MAX_PERIOD == 60 and M.LOOP_CHAR_RUN == 20 and M.loop_min_copies(1) == 5

    def test_generation_uses_the_metrics_detector(self):
        """One function for both: the generator's loop stop is ``metrics.detect_loop``."""
        assert G.detect_loop is M.detect_loop
        assert "loop" in STOP_REASONS
        # text_stop: earliest of marker and loop wins; nothing → ""
        cycle = "أ ب ج د ه و"
        looping = " ".join([cycle] * 3) + "\nالسؤال: تالي"
        out = text_stop(looping, ("\nالسؤال:",), True)
        assert (out.text, out.reason) == (cycle, "loop") and out.loop.period == 6
        assert text_stop(looping, ("\nالسؤال:",), False) == (" ".join([cycle] * 3), "marker", None)
        assert text_stop("جواب\nالسؤال: " + " ".join([cycle] * 3), ("\nالسؤال:",), True) == ("جواب", "marker", None)
        assert text_stop("جواب سليم قصير", ("\nالسؤال:",), True) == ("جواب سليم قصير", "", None)

    def test_arabic_ratio_and_chrf(self):
        assert M.arabic_letter_ratio("نص عربي") == 1.0
        assert M.arabic_letter_ratio("نص Latin") == pytest.approx(2 / 7)
        assert M.arabic_letter_ratio("123 ...") is None
        assert M.chrf_sentence("الحياة رحلة قصيرة", "الحياة رحلة قصيرة") == pytest.approx(100.0)
        assert M.chrf_sentence("", "مرجع") == 0.0
        assert 0 < M.chrf_sentence("الحياة رحلة", "الحياة رحلة قصيرة جدا") < 100

    def test_summary_shape(self):
        rows = [{"gen_chars": 10, "gen_tokens": 4, "ref_chars": 12, "chrf": 50.0, "generation": "أ ب", "reference": "أ ب ج",
                 "bertscore_f1": 0.8, "bertscore_p": 0.8, "bertscore_r": 0.8, "empty": False, "degenerate": False,
                 "latin": False, "arabic_letter_ratio": 1.0, "hit_cap": False, "hit_loop": False, "stop_reason": "eos",
                 "char_truncated": False, "reference_roundtrip_chrf": 100.0}]
        s = M.summarize(rows, gen_wall_sec=2.0)
        assert s["num_samples"] == 1 and s["chrf"] == 50.0 and s["eos_rate"] == 1.0 and s["gen_chars_per_sec"] == 5.0
        assert s["loop_stop_rate"] == 0.0
        rows.append({**rows[0], "hit_loop": True, "stop_reason": "loop", "degenerate": True})
        s = M.summarize(rows, gen_wall_sec=2.0)
        assert s["loop_stop_rate"] == 0.5 and s["degenerate_rate"] == 0.5 and s["eos_rate"] == 0.5


class TestEvaluate:
    def test_unsupported_family_returns_typed_status_without_touching_the_model(self, heldout):
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None})
        for et in (EmbeddingType.CHARACTER_CNN, EmbeddingType.CHARFORMER):
            out = task.evaluate(model=None, tokenizer=_WordTokenizer(et))
            assert out["status"] == "generation_unsupported" and out["num_samples"] == 0
            assert out["chrf"] is None and out["embedding_type"] == et and out["reason"]

    def test_prompt_fed_to_the_model_has_no_trailing_eos(self, adapter, tokenizer, heldout, monkeypatch):
        seen: List[torch.Tensor] = []

        def fake_generate(input_ids, **kw):
            seen.append(input_ids.clone())
            cont = torch.full((input_ids.shape[0], 3), tokenizer._w2i["الكتاب"], dtype=torch.long)
            return torch.cat([input_ids, cont, torch.full((input_ids.shape[0], 1), EOS)], dim=1)

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 4})
        out = task.evaluate(adapter, tokenizer)
        assert seen and all((ids[:, -1] != EOS).all() for ids in seen)          # no prompt ends with EOS
        assert all((ids[:, -1] != PAD).all() for ids in seen)                   # left padding
        assert out["status"] == "ok" and out["num_samples"] == 6
        assert out["eos_rate"] == 1.0 and out["hit_cap_rate"] == 0.0 and out["mean_gen_tokens"] == 3.0

    def test_marker_truncation_and_stop_reason(self, adapter, tokenizer, heldout, monkeypatch):
        w = tokenizer._w2i

        def fake_generate(input_ids, **kw):
            # "نعم\nالسؤال: ..." cannot be produced by a word tokenizer (no newline) — emit the marker words
            cont = torch.tensor([[w["نعم"], w["السؤال:"], w["اكتب"]]] * input_ids.shape[0])
            return torch.cat([input_ids, cont], dim=1)

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None,
                                  "stop_markers": [" السؤال:"], "batch_size": 8})
        out = task.evaluate(adapter, tokenizer)
        assert out["marker_stop_rate"] == 1.0 and out["mean_gen_chars"] == len("نعم")

    def test_loop_stop_keeps_the_first_copy(self, adapter, tokenizer, heldout, monkeypatch):
        """A synthetic looping continuation: the stopping criterion fires on the
        decoded text (checked every ``marker_check_every`` steps), the row gets
        ``stop_reason="loop"`` / ``hit_loop``, ``generation`` keeps the first copy,
        ``generation_raw`` the whole text, and ``degenerate`` is judged on the raw
        text. A non-looping continuation is untouched."""
        w = tokenizer._w2i
        cycle = ["الكتاب", "على", "الطاولة", "ذهب", "الولد"]
        calls: List[Dict] = []

        def fake_generate(input_ids, **kw):
            if "stopping_criteria" not in kw:                 # the untimed warm-up call
                return input_ids
            calls.append(kw)
            stopping = kw["stopping_criteria"]
            cont = torch.tensor([[w[x] for x in cycle * 4]] * input_ids.shape[0])   # 20 tokens: 4 copies
            # replay the criterion step by step as HF would; it must ask for a stop
            seq = input_ids
            fired_at = None
            for t in range(cont.shape[1]):
                seq = torch.cat([seq, cont[:, t:t + 1]], dim=1)
                if stopping(seq, None).all() and fired_at is None:
                    fired_at = t + 1
            calls[-1]["fired_at"] = fired_at
            return seq

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 8,
                                  "marker_check_every": 4, "stop_markers": []})
        out = task.evaluate(adapter, tokenizer, max_samples=3)
        assert calls and calls[-1]["fired_at"] == 16          # 3 copies = 15 tokens; first check at a multiple of 4 after that
        assert out["loop_stop_rate"] == 1.0 and out["degenerate_rate"] == 1.0 and out["hit_cap_rate"] == 0.0
        assert out["eos_rate"] == 0.0 and out["marker_stop_rate"] == 0.0
        assert out["mean_gen_chars"] == len(" ".join(cycle))   # exactly the first copy
        # untouched non-looping generation
        calls.clear()

        def plain_generate(input_ids, **kw):
            cont = torch.tensor([[w["قرأت"], w["مقالة"], w["عن"], w["التاريخ"]]] * input_ids.shape[0])
            return torch.cat([input_ids, cont, torch.full((input_ids.shape[0], 1), EOS)], dim=1)

        monkeypatch.setattr(adapter, "generate", plain_generate)
        out = task.evaluate(adapter, tokenizer, max_samples=3)
        assert out["loop_stop_rate"] == 0.0 and out["degenerate_rate"] == 0.0 and out["eos_rate"] == 1.0
        assert out["mean_gen_chars"] == len("قرأت مقالة عن التاريخ")

    def test_untimed_warmup_generate_precedes_the_timed_batches(self, adapter, tokenizer, heldout, monkeypatch):
        """The first generate call of a run is an 8-token warm-up on the first
        batch without stopping criteria; its wall time is not in the rows'
        ``gen_time_sec`` nor in ``generation_wall_sec``."""
        calls: List[Dict] = []

        def fake_generate(input_ids, **kw):
            calls.append({"n": input_ids.shape[0], **kw})
            if "stopping_criteria" not in kw:
                import time
                time.sleep(0.2)                                # a slow warm-up must not be billed
                return input_ids
            cont = torch.full((input_ids.shape[0], 2), tokenizer._w2i["الكتاب"], dtype=torch.long)
            return torch.cat([input_ids, cont, torch.full((input_ids.shape[0], 1), EOS)], dim=1)

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 4})
        out = task.evaluate(adapter, tokenizer)
        assert calls[0]["max_new_tokens"] == 8 and calls[0]["n"] == 4 and "stopping_criteria" not in calls[0]
        assert all("stopping_criteria" in c for c in calls[1:]) and len(calls) == 3      # warm-up + 2 batches of 4 / 2
        assert out["generation_wall_sec"] < 0.15                                          # the 0.2 s warm-up is excluded
        rows = generate_freeform(adapter, tokenizer, ["س"], task.decoding, 4, warmup=False)
        assert len(rows) == 1 and calls[-1]["max_new_tokens"] == 4                       # warmup=False: no extra call

    def test_loop_stop_row_fields(self, adapter, tokenizer, heldout, monkeypatch, tmp_path):
        w = tokenizer._w2i
        cycle = ["الكتاب", "على", "الطاولة", "ذهب"]

        def fake_generate(input_ids, **kw):
            # three copies of the 4-word cycle and half of a fourth: the model is mid-loop at the cap
            cont = torch.tensor([[w[x] for x in cycle * 3 + cycle[:2]]] * input_ids.shape[0])
            return torch.cat([input_ids, cont], dim=1)

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 8,
                                  "token_cap_floor": 14, "token_cap_ceiling": 14})
        task.evaluate(adapter, tokenizer, max_samples=2, row_dump_dir=tmp_path)
        import pyarrow.parquet as pq
        t = pq.read_table(tmp_path / "freeform_cidar.parquet")
        assert "hit_loop" in t.column_names and list(t.column_names) == ROW_FIELDS
        rows = t.to_pylist()
        for r in rows:
            assert r["stop_reason"] == "loop" and r["hit_loop"] and not r["hit_cap"] and r["degenerate"]
            assert r["generation"] == " ".join(cycle)
            assert r["generation_raw"] == " ".join(cycle * 3 + cycle[:2])
            assert r["loop_rule"] == "period" and r["loop_period"] == 4
        meta = json.loads(t.schema.metadata[b"arabic_eval"].decode("utf-8"))
        assert meta["decoding"]["loop_stop"] is True and meta["schema_version"] == 3
        # loop_stop off: the same continuation runs to the cap and is flagged degenerate only
        task_off = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 8,
                                      "token_cap_floor": 14, "token_cap_ceiling": 14, "loop_stop": False})
        out = task_off.evaluate(adapter, tokenizer, max_samples=2)
        assert out["loop_stop_rate"] == 0.0 and out["degenerate_rate"] == 1.0 and out["hit_cap_rate"] == 1.0

    def test_end_to_end_real_greedy_generation_with_row_dump(self, adapter, tokenizer, heldout, tmp_path):
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 4,
                                  "max_output_chars": 60, "token_cap_floor": 6, "token_cap_ceiling": 12})
        out = task.evaluate(adapter, tokenizer, max_samples=5, row_dump_dir=tmp_path / "eval_rows")
        assert out["status"] == "ok" and out["num_samples"] == 5
        assert set(METRIC_NAMES) <= set(out) and out["bertscore_f1"] is None
        assert out["token_cap"] == 12 and out["max_output_chars"] == 60 and out["generation_wall_sec"] > 0
        assert 0.0 <= out["degenerate_rate"] <= 1.0 and out["reference_roundtrip_chrf"] == pytest.approx(100.0)
        import pyarrow.parquet as pq
        t = pq.read_table(tmp_path / "eval_rows" / "freeform_cidar.parquet")
        assert t.num_rows == 5 and list(t.column_names) == ROW_FIELDS
        meta = json.loads(t.schema.metadata[b"arabic_eval"].decode("utf-8"))
        assert meta["kind"] == "freeform_generations" and meta["decoding"]["sampling"] == "greedy"
        assert meta["token_cap"] == 12 and meta["embedding_type"] == "standard" and meta["bertscore"] is None
        rows = t.to_pylist()
        assert all(r["prompt_text"].endswith(fc.ANSWER_LABEL + "\n") for r in rows)
        assert all(r["gen_tokens"] <= 12 and len(r["generation"]) <= 60 for r in rows)
        assert all(r["stop_reason"] in STOP_REASONS for r in rows)

    def test_determinism(self, adapter, tokenizer, heldout):
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "token_cap_floor": 6, "token_cap_ceiling": 8})
        a = task.evaluate(adapter, tokenizer, max_samples=4)
        b = task.evaluate(adapter, tokenizer, max_samples=4)
        for k in ("chrf", "mean_gen_tokens", "degenerate_rate", "eos_rate"):
            assert a[k] == b[k]


# --------------------------------------------------------------------------
# decoding knobs for the ablation (2026-09-23): repetition_penalty / no_repeat_ngram_size
# --------------------------------------------------------------------------

class _PadIsEosTokenizer(_WordTokenizer):
    """The native wrappers' convention: pad id == EOS id."""

    @property
    def special_tokens(self): return {"pad_token": EOS, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK}


def _capture_generate(adapter, monkeypatch) -> List[Dict]:
    calls: List[Dict] = []
    real = adapter.generate

    def spy(input_ids, **kw):
        out = real(input_ids, **kw)
        calls.append({"input_ids": input_ids.clone(), "out": out.clone(), **kw})
        return out

    monkeypatch.setattr(adapter, "generate", spy)
    return calls


class TestRepetitionPenaltyKnob:
    def test_default_call_is_the_greedy_call(self):
        assert G.decoding_kwargs(DecodingConfig(), [0, 3]) == {"repetition_penalty": 1.0}
        assert FreeformCidarTask({}).decoding.repetition_penalty == 1.0
        assert DecodingConfig().to_json()["repetition_penalty"] == 1.0
        kw = G.decoding_kwargs(DecodingConfig(repetition_penalty=1.2), [0, 3])
        assert kw["repetition_penalty"] == 1.0 and len(kw["logits_processor"]) == 1   # HF's own stays off

    def test_processor_is_hfs_formula_without_the_left_padding(self):
        from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
        torch.manual_seed(0)
        scores = torch.randn(2, 12)
        ids = torch.tensor([[EOS, EOS, 5, 6, 5], [7, 5, 6, 8, 9]])           # row 0 left-padded with pad == EOS
        ours = G._context_repetition_penalty(1.3, [2, 0])(ids, scores.clone())
        hf = RepetitionPenaltyLogitsProcessor(1.3)(ids, scores.clone())
        assert torch.equal(ours[1], hf[1])                                    # no padding: identical to HF
        assert ours[0, EOS] == scores[0, EOS] and hf[0, EOS] != scores[0, EOS]  # the pad (= EOS) is not penalized
        for t in (5, 6):
            assert ours[0, t] == hf[0, t] != scores[0, t]
        untouched = [t for t in range(12) if t not in (5, 6)]
        assert torch.equal(ours[0, untouched], scores[0, untouched])

    def test_real_generation_is_batch_independent_with_pad_equal_eos(self, adapter, heldout):
        tok = _PadIsEosTokenizer()
        tok.train(TEXTS, VOCAB)
        adapter.adapt_to_tokenizer(tok)
        with torch.no_grad():                        # near-flat logits (tied head), so a penalty can flip the argmax
            adapter.model.get_input_embeddings().weight.mul_(0.05)
        params = {"heldout_path": str(heldout), "bertscore_model": None, "token_cap_floor": 12,
                  "token_cap_ceiling": 12, "loop_stop": False, "repetition_penalty": 1.5}
        prompts = [FreeformCidarTask.build_prompt(r) for r in FreeformCidarTask(params).load_examples()]
        one = generate_freeform(adapter, tok, prompts, FreeformCidarTask({**params, "batch_size": 1}).decoding, 12)
        six = generate_freeform(adapter, tok, prompts, FreeformCidarTask({**params, "batch_size": 6}).decoding, 12)
        assert [r.generation_raw for r in one] == [r.generation_raw for r in six]
        greedy = generate_freeform(adapter, tok, prompts, FreeformCidarTask({**params, "repetition_penalty": 1.0,
                                                                            "batch_size": 6}).decoding, 12)
        assert [r.generation_raw for r in greedy] != [r.generation_raw for r in six]   # the knob does something

    def test_both_calls_carry_it_and_the_dump_records_it(self, adapter, tokenizer, heldout, monkeypatch, tmp_path):
        adapter.adapt_to_tokenizer(tokenizer)
        calls = _capture_generate(adapter, monkeypatch)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 4,
                                  "token_cap_floor": 6, "token_cap_ceiling": 6, "repetition_penalty": 1.2})
        task.evaluate(adapter, tokenizer, max_samples=5, row_dump_dir=tmp_path)
        assert calls[0]["max_new_tokens"] == 8 and len(calls) == 3                      # warm-up + 2 batches
        assert all(len(c["logits_processor"]) == 1 and c["repetition_penalty"] == 1.0 for c in calls)
        import pyarrow.parquet as pq
        meta = json.loads(pq.read_table(tmp_path / "freeform_cidar.parquet").schema.metadata[b"arabic_eval"])
        assert meta["decoding"]["repetition_penalty"] == 1.2 and meta["decoding"]["sampling"] == "greedy"


class TestNoRepeatNgramKnob:
    def test_default_off_and_forwarded_when_set(self):
        assert "no_repeat_ngram_size" not in G.decoding_kwargs(DecodingConfig(), [0])
        assert FreeformCidarTask({}).decoding.no_repeat_ngram_size == 0
        assert G.decoding_kwargs(DecodingConfig(no_repeat_ngram_size=3), [0])["no_repeat_ngram_size"] == 3
        assert DecodingConfig(no_repeat_ngram_size=3).to_json()["no_repeat_ngram_size"] == 3

    def test_real_generation_repeats_no_ngram(self, adapter, tokenizer, heldout, monkeypatch):
        adapter.adapt_to_tokenizer(tokenizer)
        base = {"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 6, "loop_stop": False,
                "token_cap_floor": 20, "token_cap_ceiling": 20}

        def bigrams_repeated(calls) -> int:
            n = 0
            for c in calls[1:]:                                                   # skip the warm-up
                width = c["input_ids"].shape[1]
                for row in c["out"][:, width:].tolist():
                    row = row[: row.index(EOS)] if EOS in row else row
                    grams = list(zip(row, row[1:]))
                    n += len(grams) - len(set(grams))
            return n

        calls = _capture_generate(adapter, monkeypatch)
        FreeformCidarTask(base).evaluate(adapter, tokenizer)
        assert bigrams_repeated(calls) > 0                                        # the random tiny model does loop
        calls.clear()
        FreeformCidarTask({**base, "no_repeat_ngram_size": 2}).evaluate(adapter, tokenizer)
        assert all(c["no_repeat_ngram_size"] == 2 for c in calls)
        assert bigrams_repeated(calls) == 0
