"""The generation-time loop stop of the free-form eval (``metrics.detect_loop``):
a *periodic tail* — the decoded text ends in enough contiguous copies of one
unit of up to ``LOOP_MAX_PERIOD`` words — plus the non-blank character run.
The old rule ("a word 4-gram occurs 3 times anywhere", cut before its second
occurrence) truncated ordinary numbered lists, markdown tables and answers
that restate their question (2026-09-20); these tests pin the shapes it got
wrong, the exact cut (first copy kept, everything before it untouched), the
partial-copy rule, the stop ⊂ degenerate invariant, and the real rows."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.tasks.freeform import metrics as M  # noqa: E402
from arabic_eval.tasks.freeform.generation import TextStop, text_stop  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CELLS = REPO / "outputs" / "experiments" / "qwen_native_vs_araroopat"

# 600+ characters of ordinary Arabic prose without a repeated 4-gram
PROSE = (
    "تُعدّ المكتبات العامة من أهم المؤسسات الثقافية في المدن الحديثة، فهي تتيح للقراء من مختلف الأعمار "
    "الوصول إلى الكتب والدوريات والمصادر الرقمية دون مقابل. وقد تطورت خدماتها في العقود الأخيرة لتشمل "
    "ورش العمل والمحاضرات ونوادي القراءة، كما أصبحت فضاءً للتعلم المستمر والعمل الجماعي. ويرى كثير من "
    "الباحثين أن حضور المكتبة في الحي يرفع معدلات القراءة لدى الأطفال ويقوي الروابط الاجتماعية بين "
    "السكان. ومع انتشار الإنترنت لم تفقد المكتبة دورها، بل أعادت تعريفه: فهي اليوم تقدم الإرشاد في "
    "البحث عن المعلومات الموثوقة وتوفر أجهزة الحاسوب لمن لا يملكها، وتحتضن الأرشيفات المحلية التي تحفظ "
    "ذاكرة المكان. ولهذا تواصل البلديات الاستثمار في مبانيها ومجموعاتها وموظفيها المؤهلين."
)
assert len(PROSE) >= 600

UNITS = {
    1: "والاتفاقيات",
    2: "السلام العادل",
    7: "تحدث تسونامي نتيجة لحدث طبيعي مثل زلزال",
    30: " ".join(f"كلمة{i}" for i in range(30)),
}


def _loop_text(period: int, copies: int, prefix: str = "") -> str:
    unit = UNITS[period]
    body = " ".join([unit] * copies)
    return (prefix + " " + body) if prefix else body


class TestThresholds:
    def test_constants(self):
        assert M.LOOP_MAX_PERIOD == 60 and M.LOOP_CHAR_RUN == 20
        assert M.loop_min_copies(1) == 5
        assert M.loop_min_copies(2) == 4 and M.loop_min_copies(3) == 4
        assert M.loop_min_copies(4) == 3 and M.loop_min_copies(30) == 3 and M.loop_min_copies(60) == 3

    @pytest.mark.parametrize("period", [1, 2, 7, 30])
    def test_periodic_tail_fires_at_threshold_and_keeps_the_first_copy(self, period):
        need = M.loop_min_copies(period)
        t = _loop_text(period, need)
        loop = M.detect_loop(t)
        assert loop is not None and loop.rule == "period" and loop.period == period
        assert loop.copies == need and loop.partial is False and loop.unit == UNITS[period]
        assert t[: loop.cut] == UNITS[period]                     # exactly the first copy
        # one copy short, no partial: not a loop
        assert M.detect_loop(_loop_text(period, need - 1)) is None

    @pytest.mark.parametrize("period", [1, 2, 7, 30])
    def test_prefix_of_ordinary_text_is_kept(self, period):
        need = M.loop_min_copies(period)
        t = _loop_text(period, need, prefix=PROSE)
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == period
        assert t[: loop.cut] == PROSE + " " + UNITS[period]
        assert loop.cut > 600 and M.detect_loop(PROSE) is None

    def test_earliest_cut_wins_the_fundamental_period(self):
        # a period-2 loop also satisfies period 4; the period-2 first copy ends first
        t = "مقدمة " + " ".join(["أ ب"] * 8)
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == 2 and t[: loop.cut] == "مقدمة أ ب"
        # a period-2 stretch that ends in a single-word run: the run is the loop, the stretch stays
        t = "أ ب أ ب أ ب أ ب ب ب ب ب ب"
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == 1 and t[: loop.cut] == "أ ب أ ب أ ب أ ب"

    def test_period_beyond_max_is_not_a_loop(self):
        unit = " ".join(f"ك{i}" for i in range(M.LOOP_MAX_PERIOD + 1))
        assert M.detect_loop(" ".join([unit] * 3)) is None
        unit = " ".join(f"ك{i}" for i in range(M.LOOP_MAX_PERIOD))
        loop = M.detect_loop(" ".join([unit] * 3))
        assert loop is not None and loop.period == M.LOOP_MAX_PERIOD


class TestPartialCopy:
    def test_partial_copy_counts_once_it_covers_half_the_unit(self):
        unit = UNITS[7]
        words = unit.split()
        # 2 full copies (threshold − 1) + 4 of 7 words (≥ ceil(7/2) = 4) → fires, partial
        t = " ".join([unit] * 2 + words[:4])
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == 7 and loop.copies == 2 and loop.partial is True
        assert t[: loop.cut] == unit
        # 3 of 7 words: not enough
        assert M.detect_loop(" ".join([unit] * 2 + words[:3])) is None
        # 1 full copy + a partial: never (threshold − 1 full copies are required)
        assert M.detect_loop(" ".join([unit] * 1 + words[:6])) is None

    def test_unfinished_last_word_belongs_to_the_partial_copy(self):
        # the check runs mid-word: the last word only has to be a prefix of the word one period back
        t = "والنزاعات والتفاوضات " + " ".join(["والاتفاقيات"] * 4) + " والاتف"
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == 1 and loop.copies == 4 and loop.partial is True
        assert t[: loop.cut] == "والنزاعات والتفاوضات والاتفاقيات"
        # 3 full + an unfinished one: the unfinished word is not a full copy
        assert M.detect_loop("والنزاعات " + " ".join(["والاتفاقيات"] * 3) + " والاتف") is None
        # a last word that is longer than the unit's word is not a prefix → no periodic tail
        unit = UNITS[7]
        assert M.detect_loop(" ".join([unit] * 2 + unit.split()[:3] + ["زلزالان"])) is None
        # period 7 mid-word: 2 full copies + 3 whole words + the 4th cut short
        t = " ".join([unit] * 2 + unit.split()[:3]) + " لحد"
        loop = M.detect_loop(t)
        assert loop is not None and loop.period == 7 and loop.copies == 2 and loop.partial

    def test_single_word_partial_needs_four_full_copies(self):
        assert M.detect_loop(" ".join(["نعم"] * 4) + " نع") is not None
        assert M.detect_loop(" ".join(["نعم"] * 3) + " نع") is None
        assert M.detect_loop(" ".join(["نعم"] * 4)) is None


class TestLegitimateRepetition:
    """The shapes the old 4-gram rule cut (measured on the untrained control)."""

    def test_numbered_list_whose_headings_share_a_4gram(self):
        t = ("لتمييز بين صيغة المبالغة والصفة المشبهة (على وزن فَعيل وفُعال) وبين اسم الفاعل والصفة المشبهة "
             "(على وزن فاعل)، يمكن اتباع الخطوات التالية:\n\n"
             "1. **صيغة المبالغة**:\n   - تستخدم لتعزيز صفة أو فعل.\n   - مثال: غفّار.\n\n"
             "2. **الصفة المشبهة (على وزن فَعيل وفُعال)**:\n   - تدل على صفة ثابتة.\n   - مثال: كريم.\n\n"
             "3. **اسم الفاعل والصفة المشبهة (على وزن فاعل)**:\n   - تدل على من قام بالفعل.\n   - مثال: كاتب.")
        assert M.detect_loop(t) is None and not M.is_degenerate(t)
        assert text_stop(t, ("\n###",), True) == TextStop(t, "")

    def test_markdown_table_with_repeated_bold_cells(self):
        rows = ["| **السنة** | **الحدث** |", "| **1946** | **الميلاد** |", "| **1967** | **البكالوريوس** |",
                "| **1999** | **جائزة نوبل** |", "| **2016** | **الوفاة** |"]
        t = "**جدول زمني:**\n\n" + "\n".join(rows)
        assert M.detect_loop(t) is None and not M.is_degenerate(t)
        # column padding: a run of 65 spaces is not a character loop
        padded = "| **السنة** | **الحدث**" + " " * 65 + "| **التفاصيل** |"
        assert M.detect_loop(padded) is None

    def test_answer_that_restates_the_question_twice(self):
        q = "ما هي عاصمة الأردن وما هو نظام الحكم فيها"
        t = (f"سؤالك: {q}؟ الجواب: عاصمة الأردن هي عمّان، ونظام الحكم فيها ملكي دستوري. "
             f"إذن، بالنسبة إلى سؤال {q}، فإن الإجابة هي عمّان والملكية الدستورية. "
             f"وللتأكيد مرة أخرى: {q}؟ عمّان، ملكي دستوري.")
        assert M.detect_loop(t) is None

    def test_same_4gram_three_times_at_distance(self):
        g = "في هذا السياق نجد"
        t = (f"{g} أن الزراعة كانت أساس الاقتصاد القديم. ثم تطورت الصناعة تدريجياً. "
             f"{g} أن التجارة ربطت المدن بعضها ببعض عبر القوافل. "
             f"{g} أن العلوم ازدهرت في المراكز الحضرية الكبرى.")
        assert M.detect_loop(t) is None
        assert text_stop(t, ("\n###",), True).reason == ""

    def test_two_copies_only_and_short_runs(self):
        cycle = "أ ب ج د ه و"
        assert M.detect_loop(" ".join([cycle] * 2)) is None
        assert M.detect_loop("الطبيب الطبيب الطبيب الطبيب") is None
        assert M.detect_loop("") is None and M.detect_loop("   ") is None and M.detect_loop("كلمة") is None


class TestCharRule:
    def test_letter_run_fires_and_keeps_one(self):
        t = "نعم " + "ه" * 30
        loop = M.detect_loop(t)
        assert loop is not None and loop.rule == "char" and loop.unit == "ه" and loop.period == 1
        assert loop.copies == 30 and t[: loop.cut] == "نعم ه"
        assert M.detect_loop("نعم " + "ه" * 19) is None
        assert M.detect_loop("!" * 20) is None                     # punctuation is formatting (see below)
        assert M.is_degenerate("!" * 40)                            # …but the density rule still flags a text of it

    def test_whitespace_and_punctuation_runs_are_formatting_not_loops(self):
        assert M.detect_loop("أ" + " " * 40 + "ب") is None
        assert M.detect_loop("أ" + "\n" * 40 + "ب") is None
        assert M.detect_loop("\t" * 25) is None
        # a markdown table's separator row, horizontal rules, dot leaders, a tatweel stretch
        assert M.detect_loop("| **السنة** | **الحدث** |\n|" + "-" * 40 + "|" + "-" * 40 + "|\n| 1946 | الميلاد |") is None
        for ch in "-=_*.!ـ":
            assert M.detect_loop("عنوان\n" + ch * 30 + "\nنص") is None, ch
        # letters and digits are the loop material
        assert M.detect_loop("عنوان\n" + "ه" * 20).rule == "char"
        assert M.detect_loop("الرقم " + "0" * 20).rule == "char"
        assert M.detect_loop("word " + "a" * 20).rule == "char"

    def test_earlier_cut_wins_between_rules(self):
        # both rules fire on a periodic tail of run-words: the run's cut (one character) is the earlier
        t = " ".join(["ه" * 25] * 6)
        loop = M.detect_loop(t)
        assert loop.rule == "char" and t[: loop.cut] == "ه"
        # a run before a periodic tail of ordinary words: the run is the earlier cut
        t = "ه" * 25 + " " + " ".join(["نعم"] * 6)
        loop = M.detect_loop(t)
        assert loop.rule == "char" and t[: loop.cut] == "ه"
        # a run-free periodic tail after a run-free prefix: the period rule
        t = "مقدمة " + " ".join(["نعم"] * 6)
        loop = M.detect_loop(t)
        assert loop.rule == "period" and t[: loop.cut] == "مقدمة نعم"


class TestDegenerateInvariant:
    def test_every_stop_is_degenerate_on_a_fuzz(self):
        rng = random.Random(20260920)
        vocab = ["أ", "ب", "ج", "د", "ه", "و", "ز", "ح"]
        fired = 0
        for _ in range(200):
            n = rng.randint(1, 80)
            words = [rng.choice(vocab) for _ in range(n)]
            if rng.random() < 0.5:                                    # append a periodic tail of some period
                p = rng.randint(1, 12)
                unit = [rng.choice(vocab) for _ in range(p)]
                words += unit * rng.randint(1, 6) + unit[: rng.randint(0, p)]
            if rng.random() < 0.2:
                words[-1] = words[-1][: max(1, len(words[-1]) - 1)]     # an unfinished last word
            text = " ".join(words)
            loop = M.detect_loop(text)
            if loop is not None:
                fired += 1
                assert 0 < loop.cut <= len(text) and text[: loop.cut].strip()
                assert M.is_degenerate(text)
                assert loop.rule in ("period", "char") and 1 <= loop.period <= M.LOOP_MAX_PERIOD
        assert fired > 20

    def test_density_rules_survive(self):
        assert M.is_degenerate("هذا نص عادي " * 6)                       # duplicate 4-grams ≥ 50 %
        assert M.is_degenerate("الطبيب" * 7)                               # space-less: char 8-grams
        assert M.is_degenerate("ن" * 25)
        assert not M.is_degenerate("البومة طائر ليلي بعينين كبيرتين ووجه مستدير أما الصقر فطائر نهاري حاد البصر")
        assert not M.is_degenerate("")
        # a loop that ended before the tail is caught by density, not by the stop rule
        t = " ".join(["أ ب ج د"] * 6) + " ثم انتهى النص بجملة عادية."
        assert M.detect_loop(t) is None and M.is_degenerate(t)

    def test_text_stop_reports_the_loop(self):
        cycle = "أ ب ج د ه و"
        looping = " ".join([cycle] * 3) + "\nالسؤال: تالي"
        out = text_stop(looping, ("\nالسؤال:",), True)          # the loop sits before the marker: still a loop
        assert out == TextStop(cycle, "loop", out.loop) and out.loop.period == 6 and out.loop.copies == 3
        out = text_stop(" ".join([cycle] * 3), (), True)
        assert out.text == cycle and out.reason == "loop" and out.loop.rule == "period"
        assert text_stop(looping, ("\nالسؤال:",), False) == TextStop(" ".join([cycle] * 3), "marker")
        assert text_stop("جواب\nالسؤال: " + " ".join([cycle] * 3), ("\nالسؤال:",), True) == TextStop("جواب", "marker")
        assert text_stop("جواب سليم قصير", ("\nالسؤال:",), True) == TextStop("جواب سليم قصير", "")


def _load_rows(cell: str):
    p = CELLS / cell / "eval_rows" / "freeform_cidar.parquet"
    if not p.exists():
        pytest.skip(f"no dump at {p}")
    import pandas as pd

    return pd.read_parquet(p, columns=["id", "generation_raw", "stop_reason"]).set_index("id")


class TestRealRows:
    """The rows of the 2026-09-20 measurement (``generation_raw`` of the archived
    cells); skipped when the dumps are not on this machine."""

    FALSE_POSITIVES = ("cidar-10013", "cidar-4688", "cidar-5043", "cidar-1351", "cidar-3923")

    def test_false_positives_of_the_old_rule_do_not_fire(self):
        rows = _load_rows("native_qwen3_base")
        for cid in self.FALSE_POSITIVES:
            assert rows.loc[cid, "stop_reason"] == "loop"                # what the old rule did
            assert M.detect_loop(rows.loc[cid, "generation_raw"]) is None, cid

    def test_the_true_loop_fires_with_the_right_cut(self):
        rows = _load_rows("native_qwen3_sft")
        raw = rows.loc["cidar-3388", "generation_raw"]
        loop = M.detect_loop(raw)
        assert loop is not None and loop.rule == "period" and loop.period == 1 and loop.unit == "والاتفاقيات"
        assert raw[: loop.cut].endswith("والنزاعات والتفاوضات والاتفاقيات")
        assert M.is_degenerate(raw)
