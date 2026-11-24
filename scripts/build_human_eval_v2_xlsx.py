#!/usr/bin/env python3
"""Excel versions of the blind rating sheets of human-evaluation campaign v2, for annotators who rate offline.

Reads the blind CSVs written by scripts/build_human_eval_v2.py and writes, next to them, one .xlsx per set:
a ``Ratings`` sheet with the same columns (item_id, position, instruction, context, reference, generation, score,
flags, note — the import script reads them by name), Arabic cells right-to-left and wrapped, the three cells to fill
in yellow, a 1–5 dropdown on ``score`` and a dropdown of the five flags on ``flags`` (free text such as
``repetition|truncated`` is still accepted), a progress counter, and a ``Guide`` sheet with the rubric in English and
Arabic, the legend and one example row. No cell names the arm.

    .venv/bin/python scripts/build_human_eval_v2_xlsx.py
    .venv/bin/python scripts/human_eval_v2_recalc.py  # not needed: the only formula is the progress counter, which
                                                       # recalc.py of the xlsx skill evaluates (run after this script)
"""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/_human_eval_v2"
SETS = {"qwen": "Qwen3-4B-Base", "llama3b": "Llama-3.2-3B"}
COLUMNS = ["item_id", "position", "instruction", "context", "reference", "generation", "score", "flags", "note"]
WIDTHS = {"item_id": 11, "position": 9, "instruction": 42, "context": 30, "reference": 52, "generation": 60, "score": 8, "flags": 22, "note": 30}
FLAGS = ["repetition", "wrong_language", "empty", "off_topic", "truncated"]

FONT = Font(name="Arial", size=11)
BOLD = Font(name="Arial", size=11, bold=True)
TITLE = Font(name="Arial", size=14, bold=True)
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(name="Arial", size=11, bold=True, color="FFFFFF")
FILL_ME = PatternFill("solid", fgColor="FFFF00")
DONE = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
RTL = Alignment(horizontal="right", vertical="top", wrap_text=True, readingOrder=2)
LTR = Alignment(horizontal="left", vertical="top", wrap_text=True)
CENTER = Alignment(horizontal="center", vertical="top")

RUBRIC_EN = [
    ("5", "correct, complete, fluent Modern Standard Arabic, does exactly what the instruction asks"),
    ("4", "minor omissions or small language errors; still clearly useful"),
    ("3", "partially correct or only partly follows the instruction; noticeable errors"),
    ("2", "mostly wrong, off-topic or barely follows the instruction, but contains some relevant Arabic content"),
    ("1", "empty, not Arabic, a repetition loop, gibberish, or entirely off-topic"),
]
RUBRIC_AR = [
    ("٥", "إجابة صحيحة وكاملة بعربية فصحى سليمة، تنفّذ ما تطلبه التعليمات تمامًا"),
    ("٤", "نواقص طفيفة أو أخطاء لغوية بسيطة، وما زالت مفيدة بوضوح"),
    ("٣", "صحيحة جزئيًا أو تتّبع التعليمات جزئيًا فقط، مع أخطاء ملحوظة"),
    ("٢", "خاطئة في معظمها أو خارج الموضوع أو بالكاد تتّبع التعليمات، لكنها تحوي بعض المحتوى العربي ذي الصلة"),
    ("١", "فارغة، أو ليست بالعربية، أو تكرار متواصل، أو كلام غير مفهوم، أو خارج الموضوع تمامًا"),
]
FLAGS_EN = [("repetition", "loops or repeats phrases"), ("wrong_language", "mostly not Arabic"), ("empty", "no answer"),
            ("off_topic", "does not address the instruction"), ("truncated", "cut off mid-sentence")]


def guide_sheet(wb: Workbook, model: str, n: int) -> None:
    ws = wb.create_sheet("Guide")
    ws.sheet_view.rightToLeft = False
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 95
    r = 1
    ws.cell(r, 1, f"Human evaluation v2 — {model}").font = TITLE; r += 2
    lines = [
        ("What to do", f"On the sheet “Ratings”, read the instruction (and the context, when there is one), the reference answer and the candidate answer, "
                       f"then fill the three yellow cells of that row: score (1–5, required), flags (optional) and note (optional). There are {n} rows. "
                       f"Every row needs a score. The counter at the top of “Ratings” shows how many are done."),
        ("Blind", "You are not told which system wrote an answer, and the file does not contain it. Please do not discuss the items with the other annotators while rating."),
        ("The reference", "The reference is a guide to what a correct and complete answer contains. It is not the only acceptable answer: a different but correct, complete and fluent answer in Modern Standard Arabic also deserves 5."),
        ("Length", "Do not reward length. A short complete answer beats a long padded one. Judge meaning, not wording."),
        ("Flags", "Pick one from the dropdown, or type several separated by | (for example repetition|truncated). Leave empty when none applies."),
        ("When finished", "Save the file as human_v2_<model>_<YourName>.xlsx and return it. Do not rename or reorder the columns, and do not change the item_id column."),
    ]
    for k, v in lines:
        ws.cell(r, 1, k).font = BOLD
        c = ws.cell(r, 2, v); c.font = FONT; c.alignment = LTR
        ws.row_dimensions[r].height = 48 if len(v) > 150 else 32
        r += 1
    r += 1
    ws.cell(r, 1, "Score").font = BOLD; ws.cell(r, 2, "Meaning").font = BOLD; r += 1
    for s, text in RUBRIC_EN:
        ws.cell(r, 1, s).font = BOLD; ws.cell(r, 1).alignment = CENTER
        c = ws.cell(r, 2, text); c.font = FONT; c.alignment = LTR
        r += 1
    r += 1
    ws.cell(r, 1, "الدرجة").font = BOLD; ws.cell(r, 1).alignment = RTL
    ws.cell(r, 2, "المعنى").font = BOLD; ws.cell(r, 2).alignment = RTL; r += 1
    for s, text in RUBRIC_AR:
        ws.cell(r, 1, s).font = BOLD; ws.cell(r, 1).alignment = CENTER
        c = ws.cell(r, 2, text); c.font = FONT; c.alignment = RTL
        r += 1
    c = ws.cell(r, 2, "الإجابة المرجعية دليل وليست الإجابة الوحيدة المقبولة. لا تكافئوا الطول؛ الإجابة القصيرة الكاملة أفضل من الطويلة المحشوّة. احكموا على المعنى لا على الصياغة.")
    c.font = FONT; c.alignment = RTL; ws.row_dimensions[r].height = 32; r += 2
    ws.cell(r, 1, "Flag").font = BOLD; ws.cell(r, 2, "Meaning").font = BOLD; r += 1
    for f, text in FLAGS_EN:
        ws.cell(r, 1, f).font = FONT
        c = ws.cell(r, 2, text); c.font = FONT; c.alignment = LTR
        r += 1
    r += 1
    ws.cell(r, 1, "Legend").font = BOLD
    c = ws.cell(r, 2, "Yellow cells are the ones to fill (score, flags, note). A row turns light green once its score is in. Everything else is read-only material."); c.font = FONT; c.alignment = LTR
    ws.cell(r, 2).fill = FILL_ME; r += 2
    ws.cell(r, 1, "Example row").font = BOLD; r += 1
    hdr = ["item_id", "score", "flags", "note"]
    ex = ["h2x-000", 4, "", "complete and correct but one agreement error"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(r, j, h); c.font = HEAD_FONT; c.fill = HEAD_FILL; c.border = BORDER
    r += 1
    for j, v in enumerate(ex, 1):
        c = ws.cell(r, j, v); c.font = FONT; c.border = BORDER
        if j > 1:
            c.fill = FILL_ME
    r += 1
    c = ws.cell(r, 2, "(the example is here on the guide only; the Ratings sheet holds the real items)"); c.font = Font(name="Arial", size=9, italic=True)


def build(model: str) -> Path:
    src = OUT / f"human_v2_{model}_blind.csv"
    dst = OUT / f"human_v2_{model}_blind.xlsx"
    with src.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [c for c in rows[0]] == COLUMNS, list(rows[0])
    wb = Workbook()
    ws = wb.active
    ws.title = "Ratings"
    n = len(rows)
    # row 1: title + progress; row 2: headers; data from row 3
    ws.cell(1, 1, f"Human evaluation v2 — {SETS[model]} — {n} items. Fill the yellow cells: score (1–5), flags, note. See the Guide sheet.").font = BOLD
    ws.cell(1, 7, "done:").font = BOLD
    ws.cell(1, 7).alignment = Alignment(horizontal="right")
    first, last = 3, 2 + n
    prog = ws.cell(1, 8, f'=COUNT(G{first}:G{last})&" / {n}"')
    prog.font = BOLD
    for j, col in enumerate(COLUMNS, 1):
        c = ws.cell(2, j, col); c.font = HEAD_FONT; c.fill = HEAD_FILL; c.border = BORDER; c.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = WIDTHS[col]
    for i, row in enumerate(rows):
        r = first + i
        for j, col in enumerate(COLUMNS, 1):
            v = row[col]
            if col == "position":
                v = int(v)
            if col == "score":
                v = None
            c = ws.cell(r, j, v if v != "" else None)
            c.font = FONT; c.border = BORDER
            if col in ("instruction", "context", "reference", "generation"):
                c.alignment = RTL
            elif col in ("score", "position"):
                c.alignment = CENTER
            else:
                c.alignment = LTR
            if col in ("score", "flags", "note"):
                c.fill = FILL_ME
    ws.freeze_panes = "C3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(COLUMNS))}{last}"
    dv_score = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True, showErrorMessage=True,
                              errorTitle="Score", error="Enter a whole number from 1 to 5.")
    dv_score.add(f"G{first}:G{last}")
    dv_flags = DataValidation(type="list", formula1='"' + ",".join(FLAGS) + '"', allow_blank=True, showErrorMessage=False)
    dv_flags.add(f"H{first}:H{last}")
    ws.add_data_validation(dv_score)
    ws.add_data_validation(dv_flags)
    # a row turns green once its score is in (the yellow fill of G is replaced by the rule when the cell is filled)
    ws.conditional_formatting.add(f"A{first}:I{last}", FormulaRule(formula=[f"AND(ISNUMBER($G{first}),$G{first}>=1,$G{first}<=5)"], fill=DONE))
    ws.print_title_rows = "2:2"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    guide_sheet(wb, SETS[model], n)
    wb.calculation.fullCalcOnLoad = True      # the progress counter is computed by Excel / LibreOffice on open
    wb.save(dst)
    return dst


if __name__ == "__main__":
    for m in SETS:
        p = build(m)
        print("wrote", p.relative_to(ROOT))
