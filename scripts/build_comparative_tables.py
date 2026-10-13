#!/usr/bin/env python
"""The supervisor's comparative tables (brief v6 §9): tokenizers side by side, per base model, from files only.

    .venv/bin/python scripts/build_comparative_tables.py            # → docs/comparative_tables_2026-10.md
                                                                    #   + outputs/experiments/_comparative_tables/<date>/

Reads only files (``arabic_eval.analysis`` over ``outputs/experiments``, the per-cell JSONs, the timing / speed /
tokenizer-basics JSONs, the judge files) and never a hand-typed number: every number of the document goes through
``Registry.num`` with the file it comes from, ``numbers_sources.md`` lists them all, and the build stops when the
document holds a number the registry did not produce (small integers ≤ 10, dates and § references excepted).
Regenerated, never hand-edited. Missing inputs (a cell not run yet, a judge not chosen yet) are written
"pending" — never a zero.

Tables, per model (one column per tokenizer — native, AraRooPat, BPE-16K, BPE-128K — the teacher and the
untrained model as reference rows where they apply): T1 vocabulary and intrinsic; T2 writing (judge scores; answer
shape and overlap), T2b paired differences, T2c judges against each other and against the human rating; T3
recognition, T3b paired differences, T3c letter-fair numbers; T4 response time; T5 tokenizer speed; T6 compute;
T7 across models; T8 budget actually spent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import arabic_eval.analysis as ae  # noqa: E402
from arabic_eval.judge.freeform_judge import paired_bootstrap  # noqa: E402

EXP_ROOT = REPO / "outputs" / "experiments"
TABLES_ROOT = EXP_ROOT / "_comparative_tables"
DOC = REPO / "docs" / "comparative_tables_2026-10.md"
SELECTION = TABLES_ROOT / "judge_selection.json"         # written when S3 / S4 of docs/report.md §3.14 is applied
BASICS = TABLES_ROOT / "m5" / "tokenizer_basics.json"
COUNTS = TABLES_ROOT / "m5" / "adaptation_token_counts.json"          # scripts/count_adaptation_tokens.py
PHASE3 = TABLES_ROOT / "m10" / "phase3_tokens.json"                   # scripts/count_phase3_tokens.py
SPEED_GLOB = "_tokenizer_speed/*/speed_v1.json"
N_BOOT, SEED = 10000, 0

# --------------------------------------------------------------------------------------------------------------
# what is compared
# --------------------------------------------------------------------------------------------------------------

ARM_ORDER = ("native", "araroopat", "bpe16k", "bpe128k")
ARM_LABEL = {"native": "native (the model's own)", "araroopat": "AraRooPat (roots and patterns)",
             "bpe16k": "BPE, 16 000 pieces", "bpe128k": "BPE, 128 000 pieces"}
ARM_SHORT = {"native": "native", "araroopat": "AraRooPat", "bpe16k": "BPE-16K", "bpe128k": "BPE-128K"}


@dataclass
class Arm:
    key: str
    ff: Optional[str]                    # free-form cell (v5 checkpoint)
    mcq: Optional[str]                   # MCQ cell at max_length 4096
    letter: Optional[str]                # arm name in _letter_slot/letter_slot.json
    basics: Optional[str]                # name in the tokenizer-basics JSON
    adapt: Tuple[str, ...] = ()          # cells whose training phases make up the arm (Phase 1/2 cell first)
    tokenizer_dir: Optional[str] = None


@dataclass
class ModelSpec:
    key: str
    label: str
    hf_id: str
    experiment: str
    arms: Dict[str, Arm]
    teacher: Optional[str]
    base_ff: Optional[str]
    base_mcq: Optional[str]
    base_letter: Optional[str]
    letter_slot: Optional[str] = None


E = "qwen_native_vs_araroopat"
MODELS: List[ModelSpec] = [
    ModelSpec(
        key="qwen3_4b", label="Qwen3-4B-Base", hf_id="Qwen/Qwen3-4B-Base", experiment=E,
        arms={
            "native": Arm("native", "native_qwen3_sft_v5_distill", "native_qwen3_sft_v5_distill_mcq4096", "native_v5",
                          "native_qwen3", adapt=("native_qwen3_sft_v5_distill",)),
            "araroopat": Arm("araroopat", "araroopat_3phase_v5_distill", "araroopat_3phase_v5_distill_mcq4096",
                             "araroopat_v5", "araroopat", adapt=("araroopat_3phase_v3", "araroopat_3phase_v5_distill"),
                             tokenizer_dir="outputs/tokenizers/araroopat_maxpat40k_v3"),
            "bpe16k": Arm("bpe16k", "bpe_16k_3phase_v5_distill", "bpe_16k_3phase_v5_distill_mcq4096", "bpe16k_v5",
                          "bpe_16k", adapt=("bpe_16k_3phase_v3", "bpe_16k_3phase_v5_distill"),
                          tokenizer_dir="outputs/tokenizers/bpe_16k_v3"),
            "bpe128k": Arm("bpe128k", "bpe_128k_3phase_v5_distill", "bpe_128k_3phase_v5_distill_mcq4096", "bpe128k_v5",
                           "bpe_128k", adapt=("bpe_128k_3phase_v5_distill",),
                           tokenizer_dir="outputs/tokenizers/bpe_128k_v1"),
        },
        teacher="teacher_aya_expanse_32b_m2_c2400", base_ff="native_qwen3_base_m2_c2400",
        base_mcq="native_qwen3_base_mcq4096", base_letter="native_base",
        letter_slot=f"{E}/_letter_slot/letter_slot_bpe128k.json"),   # the five-arm run (letter_slot.json + bpe128k_v5; common values identical)
    *[ModelSpec(
        key=key, label=label, hf_id=hf, experiment=exp,
        arms={
            "native": Arm("native", "native_sft_v5_distill", "native_sft_v5_distill_mcq4096", "native_v5", native_basics,
                          adapt=("native_sft_v5_distill",)),
            "araroopat": Arm("araroopat", "araroopat_3phase_v5_distill", "araroopat_3phase_v5_distill_mcq4096",
                             "araroopat_v5", "araroopat", adapt=("araroopat_3phase_v5_distill",),
                             tokenizer_dir="outputs/tokenizers/araroopat_maxpat40k_v3"),
            "bpe16k": Arm("bpe16k", "bpe_16k_3phase_v5_distill", "bpe_16k_3phase_v5_distill_mcq4096", "bpe16k_v5",
                          "bpe_16k", adapt=("bpe_16k_3phase_v5_distill",), tokenizer_dir="outputs/tokenizers/bpe_16k_v3"),
            "bpe128k": Arm("bpe128k", "bpe_128k_3phase_v5_distill", "bpe_128k_3phase_v5_distill_mcq4096", "bpe128k_v5",
                           "bpe_128k", adapt=("bpe_128k_3phase_v5_distill",), tokenizer_dir="outputs/tokenizers/bpe_128k_v1"),
        },
        # the teacher answered the same 250 prompts whatever the student model: its cell lives in the Qwen folder
        teacher=f"{E}/teacher_aya_expanse_32b_m2_c2400", base_ff="native_base_m2_c2400", base_mcq="native_base_mcq4096",
        base_letter="native_base",
        # P4 (chains/m16_queue.sh): read when the file exists, "pending" until then
        letter_slot=f"{exp}/_letter_slot/letter_slot.json")
      for key, label, hf, exp, native_basics in (
          ("llama3b", "Llama-3.2-3B", "meta-llama/Llama-3.2-3B", "llama3b_tokenizers", "native_llama3b"),
          ("gemma3_4b", "Gemma-3-4B", "google/gemma-3-4b-pt", "gemma3_4b_tokenizers", "native_gemma3"))],
]

JUDGES: Dict[str, Dict[str, str]] = {
    # name → label, role. The candidates of docs/report.md §3.14 first, then the two older judges.
    "gptoss_120b": {"label": "gpt-oss-120b", "role": "candidate C1 (OpenAI)"},
    "mistral_small_32": {"label": "Mistral-Small-3.2-24B", "role": "dropped 2026-10-10 (cannot load in the judge environment)"},
    "falcon_h1_34b": {"label": "Falcon-H1-34B", "role": "candidate C3 (TII)"},
    "phi4_14b": {"label": "phi-4", "role": "candidate C4 (Microsoft)"},
    "gemma4_31b": {"label": "Gemma-4-31B", "role": "earlier judge (history; Gemma is now a base model)"},
    "gpt56_terra": {"label": "GPT-5.6 (API)", "role": "earlier second judge, five cells only (same vendor as gpt-oss)"},
}
RETIRED_JUDGES = ("gemma4_31b", "gpt56_terra")         # brief v6 §3.6: not run again, not extended to new cells
CANDIDATES = ("gptoss_120b", "falcon_h1_34b", "phi4_14b")   # Mistral-Small-3.2 dropped 2026-10-10 (docs/report.md §3.14)
DEFAULT_PRIMARY = "gptoss_120b"                           # S4: C1 until the human rating names one


# --------------------------------------------------------------------------------------------------------------
# number registry
# --------------------------------------------------------------------------------------------------------------

class Registry:
    """Every number of the document passes through ``num``; ``check`` refuses a document holding another."""

    def __init__(self) -> None:
        self.entries: List[Tuple[str, str, str, str]] = []      # (where, text, source, note)
        self.strings: set = set()

    def num(self, value: Any, fmt: str | Callable[[Any], str], source: str, where: str = "", note: str = "") -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        text = fmt(value) if callable(fmt) else fmt.format(value)
        text = text.replace("-", "−") if re.match(r"^-\d", text) else text
        self.entries.append((where, text, source, note))
        self.strings.add(text)
        return text

    _NUM = re.compile(r"(?<![\w.§/-])[−+]?\d+(?:[.,  ]\d+)*%?(?![\w/])")
    _ALLOW = re.compile(r"^\d$|^10$|^20\d\d$")

    def check(self, text: str) -> List[str]:
        bad = []
        for line in text.splitlines():
            if line.startswith("<!--"):
                continue
            for m in self._NUM.finditer(line):
                tok = m.group(0)
                if self._ALLOW.match(tok) or any(tok in s for s in self.strings):
                    continue
                bad.append(f"{tok!r} in: {line[:160]}")
        return bad


R = Registry()


def D(text: str, source: str = "definition (fixed by the method)") -> str:
    """A number that is part of a definition (a resample count, a unit) — registered like any other."""
    R.entries.append(("definition", text, source, ""))
    R.strings.add(text)
    return text


BOOT = None  # set in build(): D(thou(N_BOOT))


def f2(v): return f"{v:.2f}"
def f3(v): return f"{v:.3f}"
def pct1(v): return f"{100 * v:.1f} %"
def pct0(v): return f"{100 * v:.0f} %"
def sgn3(v): return f"{v:+.3f}".replace("-", "−")
def thou(v): return f"{int(round(v)):,}".replace(",", " ")


def ci3(d: Dict[str, Any], src: str, where: str, keys=("delta_mean", "ci_low", "ci_high")) -> str:
    if not d or d.get(keys[0]) is None:
        return "—"
    a, lo, hi = (R.num(d[k], sgn3, src, where) for k in keys)
    return f"{a} [{lo}, {hi}]"


# --------------------------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------------------------

@dataclass
class Table:
    tid: str
    title: str
    how: str
    source: str
    columns: List[str]
    rows: List[Tuple[str, List[str]]] = field(default_factory=list)
    shows: str = ""
    notes: List[str] = field(default_factory=list)

    def add(self, label: str, cells: Sequence[str]) -> None:
        self.rows.append((label, list(cells)))

    def md(self) -> str:
        out = [f"### {self.tid} — {self.title}", "", f"*How to read:* {self.how}", ""]
        out.append("| " + " | ".join([""] + self.columns) + " |")
        out.append("|" + "---|" * (len(self.columns) + 1))
        for label, cells in self.rows:
            if any("|" in c for c in [label] + list(cells)):
                raise ValueError(f"{self.tid}: a cell holds '|', which breaks the Markdown table: {label!r}")
            out.append("| " + " | ".join([label] + [c if c else "—" for c in cells]) + " |")
        out.append("")
        if self.shows:
            out += [f"**What it shows.** {self.shows[:1].upper()}{self.shows[1:]}", ""]
        for n in self.notes:
            out.append(f"- {n}")
        if self.notes:
            out.append("")
        out += [f"*Source:* {self.source}", ""]
        return "\n".join(out)

    def csv(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + self.columns)
            for label, cells in self.rows:
                w.writerow([label] + cells)


def pending(what: str = "pending") -> str:
    return what


# --------------------------------------------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------------------------------------------

def cell_dir(spec: ModelSpec, cell: Optional[str]) -> Optional[Path]:
    if not cell:
        return None
    p = EXP_ROOT / cell if "/" in cell else EXP_ROOT / spec.experiment / cell
    return p if (p / "all_metrics.json").exists() else None


def cell_id(spec: ModelSpec, cell: str) -> str:
    """The ``arabic_eval.analysis`` id of a cell: ``<experiment>/<cell>``, or the cell as given when it already names
    its experiment (the teacher cell, shared by every model)."""
    return cell if "/" in cell else f"{spec.experiment}/{cell}"


def jload(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def rel(p: Path) -> str:
    return str(Path(p).relative_to(REPO))


def judge_file(d: Optional[Path], judge: str) -> Optional[Path]:
    if d is None:
        return None
    p = d / "freeform_judge" / f"{judge}.parquet"
    return p if p.exists() else None


def judge_scores(d: Path, judge: str) -> Dict[str, float]:
    import pyarrow.parquet as pq
    t = pq.read_table(d / "freeform_judge" / f"{judge}.parquet", columns=["id", "score", "parse_ok"]).to_pylist()
    return {str(r["id"]): float(r["score"]) for r in t if r["score"] is not None and r["parse_ok"] is not False}


def gen_chars(d: Path) -> Dict[str, int]:
    import pyarrow.parquet as pq
    t = pq.read_table(d / "eval_rows" / "freeform_cidar.parquet", columns=["id", "generation"]).to_pylist()
    return {str(r["id"]): len(r["generation"] or "") for r in t}


def primary_judge() -> Tuple[Optional[str], str]:
    sel = jload(SELECTION)
    if sel and sel.get("primary"):
        return sel["primary"], f"named by rule {sel.get('rule')} on {sel.get('date')} ({rel(SELECTION)})"
    return None, "not named yet: the human rating is not complete (docs/report.md §3.14, S0 / S4)"


def judges_present(spec: ModelSpec, cells: Sequence[Optional[str]]) -> List[str]:
    out = []
    for j in JUDGES:
        if any(judge_file(cell_dir(spec, c), j) for c in cells):
            out.append(j)
    return out


# --------------------------------------------------------------------------------------------------------------
# T1 — vocabulary and intrinsic
# --------------------------------------------------------------------------------------------------------------

def t1(spec: ModelSpec, basics: Optional[Dict[str, Any]]) -> Table:
    t = Table("T1", f"{spec.label}: the four tokenizers before any training",
              "one column per tokenizer; each row is a property of the tokenizer alone, measured on fixed Arabic text "
              "(no model involved). Arrows say which direction is better where there is one.",
              f"{rel(BASICS)} (scripts/measure_tokenizer_basics.py) and each arm's intrinsic_metrics.json "
              "(the pipeline's intrinsic step)",
              [ARM_SHORT[a] for a in ARM_ORDER])
    bsrc = rel(BASICS)
    toks = (basics or {}).get("tokenizers", {})

    def b(arm: str, path: Sequence[str], fmt, where: str) -> str:
        name = spec.arms[arm].basics
        v: Any = toks.get(name) if name else None
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        return R.num(v, fmt, bsrc, where) if v is not None else pending()

    def intr(arm: str, key: str, fmt, where: str) -> str:
        d = cell_dir(spec, spec.arms[arm].ff)
        im = jload(d / "intrinsic_metrics.json") if d else None
        if not im or im.get(key) is None:
            return pending()
        return R.num(im[key], fmt, rel(d / "intrinsic_metrics.json"), where)

    rows = [
        ("vocabulary size (pieces)", lambda a: b(a, ["vocab_size"], thou, f"T1 vocab {a}")),
        ("pieces per word ↓ (fertility, ArabicText-Large eval texts)", lambda a: b(a, ["eval_split", "fertility"], f2, f"T1 fertility {a}")),
        ("characters per piece ↑ (same texts)", lambda a: b(a, ["eval_split", "chars_per_token"], f2, f"T1 cpt {a}")),
        ("characters per piece on the free-form reference answers", lambda a: b(a, ["references", "chars_per_token"], f2, f"T1 cpt refs {a}")),
        ("pieces per word of web text (the training pool)", lambda a: b(a, ["pool", "tokens_per_word"], f2, f"T1 pool tpw {a}")),
        ("unknown-piece rate ↓", lambda a: intr(a, "unk_rate", lambda v: f"{100 * v:.2f} %", f"T1 unk {a}")),
        ("words written without an unknown piece ↑", lambda a: intr(a, "vocab_coverage", pct1, f"T1 coverage {a}")),
        ("text given back exactly after encode → decode: reference answers, character overlap (chrF, 0–100) ↑",
         lambda a: b(a, ["references", "roundtrip_chrf"], lambda v: f"{v:.1f}", f"T1 rt chrf {a}")),
        ("the same with the answers' diacritics removed first ↑",
         lambda a: b(a, ["references", "roundtrip_chrf_diacritics_stripped"], lambda v: f"{v:.1f}", f"T1 rt chrf strip {a}")),
        (f"single words given back exactly (of {D('2 000', 'outputs/data_cache/araroopat_roundtrip_words_2000_42.json (2 000 words)')} corpus words) ↑", lambda a: b(a, ["words", "exact"], pct1, f"T1 word exact {a}")),
        ("roots kept inside one piece, where possible ↑ (root conservation, attainable)",
         lambda a: intr(a, "root_conservation_attainable", f3, f"T1 rps {a}")),
        ("share of words whose root can be kept at all (the ceiling)", lambda a: intr(a, "root_measurable_pct", lambda v: f"{v:.1f} %", f"T1 rps ceiling {a}")),
        ("pieces per morpheme (≈ 1 = cut at morpheme size; > 1 = cut finer)",
         lambda a: intr(a, "semantic_fragmentation_ratio", f2, f"T1 sfr {a}")),
        ("the letter أ written as a choice answer, in pieces", lambda a: b(a, ["letter_alef", "n_tokens"], "{}", f"T1 letter {a}")),
    ]
    for label, fn in rows:
        t.add(label, [fn(a) for a in ARM_ORDER])
    counts = jload(COUNTS) or {}
    csrc = rel(COUNTS)

    def seen(arm: str, key: str, where: str) -> str:
        if arm == "native":
            return "not applicable (no new pieces)"
        cell = spec.arms[arm].adapt[0] if spec.arms[arm].adapt else None
        r = counts.get(f"outputs/experiments/{spec.experiment}/{cell}") if cell else None
        return R.num(r[key], pct1, f"{csrc} ({cell})", where) if r else pending()

    t.add("pieces the adaptation never showed the model (stages 1–2)", [seen(a, "share_seen_0", f"T1 unseen {a}") for a in ARM_ORDER])
    t.add("pieces it showed fewer than ten times", [seen(a, "share_seen_1_9", f"T1 rare {a}") for a in ARM_ORDER])
    t.shows = ("how differently the four tokenizers cut the same Arabic text. A larger vocabulary cuts words into fewer, "
               "longer pieces; AraRooPat cuts every word into a root and a pattern, so it needs more pieces per word.")
    t.notes = [
        "AraRooPat's root conservation is high *by construction* on the words it writes as root + pattern; the number "
        "is a property of its design, not a measured skill (docs/report.md, the intrinsic metrics).",
        "A larger vocabulary trained on the same text sees each of its pieces less often during the model's "
        "adaptation; the last two rows say how many pieces the two adaptation stages never showed the model, or "
        "showed it fewer than ten times (scripts/count_adaptation_tokens.py, over the blocks the stages consumed).",
        "The native vocabularies are much larger than the from-scratch ones (see the first row); "
        "BPE-128K is the from-scratch tokenizer of native size.",
    ]
    return t


# --------------------------------------------------------------------------------------------------------------
# T2 — writing
# --------------------------------------------------------------------------------------------------------------

def ff_rows(spec: ModelSpec) -> List[Tuple[str, str, Optional[str]]]:
    rows = [(a, ARM_LABEL[a], spec.arms[a].ff) for a in ARM_ORDER]
    rows.append(("teacher", "teacher model (Aya-Expanse-32B), for reference", spec.teacher))
    rows.append(("base", "the model before any training, for reference", spec.base_ff))
    return rows


def t2(spec: ModelSpec, primary: Optional[str]) -> Tuple[Table, Table]:
    rows = ff_rows(spec)
    present = judges_present(spec, [r[2] for r in rows])
    order = ([primary] if primary in present else []) + [j for j in present if j != primary]
    cols = []
    for j in order:
        tag = " — main judge" if j == primary else ""
        cols += [f"{JUDGES[j]['label']}{tag}: score (1–5)", f"{JUDGES[j]['label']}: share of the teacher's score"]
    ta = Table("T2", f"{spec.label}: how well each version answers 250 open questions (judge scores)",
               "each row is one trained version of the model, answering the same 250 held-out Arabic instructions. "
               "A judge model scores every answer from 1 (useless) to 5 (correct, complete, fluent); the number is the "
               "mean ± its standard error. 'Share of the teacher' divides by the teacher model's own mean under the "
               "same judge.",
               "each cell's freeform_judge/<judge>.parquet (scores) and all_metrics.json (summaries)", cols or ["judge"])
    teacher_mean: Dict[str, Optional[float]] = {}
    for j in order:
        d = cell_dir(spec, spec.teacher)
        if judge_file(d, j):
            s = judge_scores(d, j)
            teacher_mean[j] = sum(s.values()) / len(s)
    for key, label, cell in rows:
        d = cell_dir(spec, cell)
        cells = []
        for j in order:
            f = judge_file(d, j)
            if not f:
                cells += [pending() if d is None else "not judged", ""]
                continue
            s = judge_scores(d, j)
            m = sum(s.values()) / len(s)
            sd = math.sqrt(sum((x - m) ** 2 for x in s.values()) / len(s))
            se = sd / math.sqrt(len(s))
            cells.append(f"{R.num(m, f2, rel(f), f'T2 {key} {j} mean')} ± {R.num(se, f2, rel(f), f'T2 {key} {j} se')}")
            tm = teacher_mean.get(j)
            cells.append(R.num(m / tm, f2, f"{rel(f)} ÷ the teacher's mean under the same judge", f"T2 {key} {j} share")
                         if tm and key != "teacher" else "")
        if not order:                       # no judge has scored any cell of this model yet
            cells = [pending() if d is None else "not judged yet"]
        ta.add(label, cells)
    ta.shows = ("the judges' verdict on open writing. Every version trained on the same teacher's answers, so the "
                "teacher's own score is the ceiling the versions are read against.")
    ta.notes = [
        "The distilled versions write at about the teacher's length (mean answer length in T2, continued), and judges "
        "tend to prefer longer answers; T2c checks the main comparison on answers of equal length.",
        "Gemma-4-31B was the judge of the thesis deck of 26 September; it is a Gemma model and Gemma is now one of "
        "the base models, so it cannot be the main judge any more (docs/report.md §3.14). Its scores stay here for "
        "the record.",
        "GPT-5.6 judged only the five cells of the 26 September check; it comes from the same company as gpt-oss.",
    ]

    tb = Table("T2 (continued)", f"{spec.label}: the shape of the answers and their overlap with the reference",
               "the same 250 answers, measured without a judge. Lower is better for loops, the cap and the two "
               "'surprise' rows; higher is better for ended-on-its-own and the two overlap rows.",
               "each cell's all_metrics.json (downstream.freeform_cidar) and diag_heldout_rawtext_v1.json "
               "(scripts/diag_heldout_loss.py)",
               ["answers stuck in a loop ↓", "answers that ended on their own ↑", "answers cut at the length cap ↓",
                "mean answer length (characters)", "character overlap with the reference (chrF, 0–100) ↑",
                "meaning overlap with the reference (BERTScore-F1) ↑",
                "surprise on the reference answers, nats per character ↓",
                "surprise on held-out web text, nats per character ↓",
                "the same, each text block opened with the start-of-text marker ↓"])
    bos_model = spec.key != "qwen3_4b"           # Llama-3 / Gemma-3 tokenizers have a BOS; Qwen3's has none
    for key, label, cell in rows:
        d = cell_dir(spec, cell)
        if d is None:
            tb.add(label, [pending()] * 9)
            continue
        am = jload(d / "all_metrics.json") or {}
        ff = (am.get("downstream") or {}).get("freeform_cidar") or {}
        src = rel(d / "all_metrics.json")
        stem = "diag_heldout_rawtext_v1_base" if key == "base" and (d / "diag_heldout_rawtext_v1_base.json").exists() \
            else "diag_heldout_rawtext_v1"
        diag = jload(d / f"{stem}.json")
        dsrc = rel(d / f"{stem}.json")
        if diag is None and key == "base":
            p = EXP_ROOT / spec.experiment / "native_qwen3_base" / "diag_heldout_rawtext_v1_base.json"
            diag, dsrc = jload(p), rel(p) if p.exists() else dsrc
        bdiag = jload(d / f"{stem}_bosblocks.json") if bos_model else None
        bsrc = rel(d / f"{stem}_bosblocks.json")
        tb.add(label, [
            R.num(ff.get("loop_stop_rate"), pct1, src, f"T2c {key} loops"),
            R.num(ff.get("eos_rate"), pct1, src, f"T2c {key} eos"),
            R.num(ff.get("hit_cap_rate"), pct1, src, f"T2c {key} cap"),
            R.num(ff.get("mean_gen_chars"), thou, src, f"T2c {key} chars"),
            R.num(ff.get("chrf"), lambda v: f"{v:.1f}", src, f"T2c {key} chrf"),
            R.num(ff.get("bertscore_f1"), f3, src, f"T2c {key} bertscore"),
            R.num(((diag or {}).get("heldout_answers") or {}).get("nll_per_answer_char"), f3, dsrc, f"T2c {key} nll")
            if diag else ("not applicable" if key == "teacher" else pending()),
            R.num(((diag or {}).get("mix") or {}).get("loss_per_char"), f3, dsrc, f"T2c {key} raw")
            if diag else ("not applicable" if key == "teacher" else pending()),
            ("not applicable (no start marker)" if not bos_model else
             "not applicable" if key == "teacher" else
             R.num(((bdiag or {}).get("mix") or {}).get("loss_per_char"), f3, bsrc, f"T2c {key} raw bos")
             if bdiag else pending()),
        ])
    tb.shows = ("whether the answers loop, stop and stay close to the reference. The two 'surprise' rows measure how "
                "unexpected a fixed text is to the model, per character, so they compare across tokenizers.")
    tb.notes = ["The teacher is a separate, larger model whose answers were collected once; the surprise columns do "
                "not apply to it."]
    if bos_model:
        tb.notes.append("Llama and Gemma always see a start-of-text marker before any text, so the web-text "
                        "surprise is shown both without it (the original measurement, which suited Qwen, whose "
                        "tokenizer has no such marker) and with it. Without the marker Gemma's own tokenizer looks "
                        "far worse than it is; with it, it reads the web text best. The answer column always "
                        "included the marker (docs/report.md §3.17).")
    return ta, tb


def t2b(spec: ModelSpec, primary: Optional[str]) -> Table:
    present = judges_present(spec, [spec.arms[a].ff for a in ARM_ORDER])
    judges = ([primary] if primary in present else []) + [j for j in ("gemma4_31b", "gpt56_terra") if j in present and j != primary]
    if primary is None:
        judges += [j for j in CANDIDATES if j in present and j not in judges]
    cols = [f"{JUDGES[j]['label']}{' — main judge' if j == primary else ''}: difference [95 % range]" for j in judges]
    cols += [f"{JUDGES[j]['label']}: better / same / worse" for j in judges[:1]]
    t = Table("T2b", f"{spec.label}: paired differences in judge score",
              "each row compares two versions on the same 250 instructions: the mean of (first − second) with its "
              f"95 % range from {D(thou(N_BOOT), 'N_BOOT of this script = the judge stage and mcq_compare.py')} resamples. A range that does not include zero means the difference is "
              "unlikely to be chance; a range that includes zero means the two cannot be told apart — not that "
              "they are equal.",
              f"scripts/judge/paired_compare.py through arabic_eval.analysis.judge_compare ({D(thou(N_BOOT))} resamples, seed 0) "
              "over each cell's freeform_judge/<judge>.parquet", cols or ["judge"])
    pairs = [("araroopat", "native"), ("bpe16k", "native"), ("bpe128k", "native"), ("araroopat", "bpe16k"),
             ("bpe128k", "bpe16k"), ("araroopat", "teacher"), ("bpe16k", "teacher"), ("bpe128k", "teacher"),
             ("native", "teacher")]
    cellof = {**{a: spec.arms[a].ff for a in ARM_ORDER}, "teacher": spec.teacher}
    name = {**ARM_SHORT, "teacher": "teacher"}
    for a, b in pairs:
        da, db = cell_dir(spec, cellof[a]), cell_dir(spec, cellof[b])
        row = []
        wtl = ""
        for k, j in enumerate(judges):
            if not (judge_file(da, j) and judge_file(db, j)):
                row.append("not judged (judge retired before this version existed)"
                           if j in RETIRED_JUDGES and da is not None and db is not None else pending())
                continue
            res = ae.judge_compare(cell_id(spec, cellof[b]), cell_id(spec, cellof[a]), judge=j,
                                   n_boot=N_BOOT, seed=SEED)
            s = res["score"]
            src = f"judge_compare({cellof[b]}, {cellof[a]}, judge={j})"
            row.append(ci3(s, src, f"T2b {a}-{b} {j}"))
            if k == 0:
                wtl = " / ".join(R.num(s[x], pct0, src, f"T2b {a}-{b} {x}") for x in ("win_rate", "tie_rate", "loss_rate"))
        row += [wtl or pending()] if judges else ["not judged yet"]
        t.add(f"{name[a]} − {name[b]}", row)
    t.shows = "which differences in T2 survive pairing on the same instructions."
    return t


def t2c(spec: ModelSpec, primary: Optional[str]) -> Table:
    t = Table("T2c", f"{spec.label}: the judges against each other, against a human, and on answers of equal length",
              "agreement is the quadratic-weighted kappa (1 = perfect, 0 = chance) between two sets of 1–5 scores on "
              "the same answers. The last rows repeat the main comparison of T2b on the instruction pairs whose two "
              "answers differ in length by at most 100 characters, so that length cannot explain the result.",
              "each cell's freeform_judge/<judge>.parquet; the blind rating set freeform_rating/v5_blind.json and its "
              "ratings; eval_rows/freeform_cidar.parquet for the answer lengths",
              ["value"])
    rows = [r for r in ff_rows(spec) if r[2]]
    present = judges_present(spec, [r[2] for r in rows])
    from arabic_eval.judge.freeform_judge import quadratic_weighted_kappa
    pairs = [(present[i], present[k]) for i in range(len(present)) for k in range(i + 1, len(present))]
    for ja, jb in pairs:
        xs, ys = [], []
        srcs = []
        for _, _, cell in rows:
            d = cell_dir(spec, cell)
            fa, fb = judge_file(d, ja), judge_file(d, jb)
            if fa and fb:
                sa, sb = judge_scores(d, ja), judge_scores(d, jb)
                ids = sorted(set(sa) & set(sb))
                xs += [int(sa[i]) for i in ids]
                ys += [int(sb[i]) for i in ids]
                srcs.append(cell)
        if not xs:
            continue
        q = quadratic_weighted_kappa(xs, ys)
        src = f"freeform_judge/{{{ja},{jb}}}.parquet in {', '.join(srcs)}"
        t.add(f"{JUDGES[ja]['label']} vs {JUDGES[jb]['label']}: agreement over the cells both judged "
              f"({R.num(len(xs), thou, src, f'T2c n {ja}-{jb}')} answers)", [R.num(q, f3, src, f"T2c qwk {ja}-{jb}")])
    rating = EXP_ROOT / spec.experiment / "freeform_rating" / "v5_blind.ratings"
    n_rated = 0
    if rating.is_dir():
        for p in rating.glob("*.json"):
            n_rated = max(n_rated, len(json.loads(p.read_text(encoding="utf-8"))))
    if spec.key == "qwen3_4b":
        sel = jload(SELECTION)
        if sel and sel.get("agreement"):
            for j, a in sel["agreement"].items():
                t.add(f"human rater vs {JUDGES.get(j, {'label': j})['label']} (200 answers)",
                      [R.num(a.get("qwk"), f3, rel(SELECTION), f"T2c human {j}")])
        else:
            t.add("human rater vs each judge (200 blind answers)",
                  [f"waiting for the blind rating: {R.num(n_rated, '{}', rel(rating), 'T2c n_rated')} of "
                   f"{R.num(200, '{}', 'freeform_rating/v5_blind.json (n_items)', 'T2c n_items')} rated"])
    # length-matched check (docs/report.md §3.13 J4 → §3.14 J7)
    for a, b in (("araroopat", "bpe16k"), ("araroopat", "native")):
        da, db = cell_dir(spec, spec.arms[a].ff), cell_dir(spec, spec.arms[b].ff)
        if da is None or db is None:
            continue
        La, Lb = gen_chars(da), gen_chars(db)
        ids = {i for i in set(La) & set(Lb) if abs(La[i] - Lb[i]) <= 100}
        for j in present:
            if not (judge_file(da, j) and judge_file(db, j)):
                continue
            sa, sb = judge_scores(da, j), judge_scores(db, j)
            res = paired_bootstrap({i: sb[i] for i in ids if i in sb}, {i: sa[i] for i in ids if i in sa},
                                   n_boot=N_BOOT, seed=SEED)
            src = f"paired_bootstrap over the pairs within 100 characters ({rel(da)}, {rel(db)}, {j})"
            t.add(f"{ARM_SHORT[a]} − {ARM_SHORT[b]} on answers of equal length "
                  f"({R.num(res['n'], '{}', src, f'T2c lm n {a}-{b} {j}')} pairs), {JUDGES[j]['label']}",
                  [ci3(res, src, f"T2c lm {a}-{b} {j}")])
    t.shows = ("whether the judges agree with each other and with a human reader, and whether the main writing "
               "difference holds once answer length is matched.")
    t.notes = ["The human rating is blind: the rater does not know which version wrote an answer. Its agreement "
               "table also decides which candidate becomes the main judge (docs/report.md §3.14, rule S3)."]
    return t


# --------------------------------------------------------------------------------------------------------------
# T3 — recognition
# --------------------------------------------------------------------------------------------------------------

TASK_LABEL = {"arabic_exam": "Arabic school exams (ArabicMMLU)", "culture_arabic_mmlu": "translated MMLU (Culture-Arabic-MMLU)",
              "alghafa": "AlGhafa", "acva": "ACVA"}


def mcq_cells(spec: ModelSpec) -> List[Tuple[str, str, Optional[str]]]:
    rows = [(a, ARM_LABEL[a], spec.arms[a].mcq) for a in ARM_ORDER]
    rows.append(("base", "the model before any training, for reference", spec.base_mcq))
    return rows


def t3(spec: ModelSpec) -> Tuple[Table, Table, Optional[Dict[str, Any]]]:
    rows = mcq_cells(spec)
    have = [(k, l, c) for k, l, c in rows if cell_dir(spec, c)]
    res = None
    if len(have) >= 2:
        ids = [f"{spec.experiment}/{c}" for _, _, c in have]
        name = {c: k for k, _, c in have}
        pair_keys = [("araroopat", "native"), ("bpe16k", "native"), ("bpe128k", "native"), ("araroopat", "bpe16k"),
                     ("bpe128k", "bpe16k"), ("araroopat", "bpe128k")]
        cof = {k: c for k, _, c in have}
        pairs = [(f"{spec.experiment}/{cof[a]}", f"{spec.experiment}/{cof[b]}") for a, b in pair_keys if a in cof and b in cof]
        res = ae.mcq_compare(ids, tasks=("arabic_exam", "culture_arabic_mmlu", "alghafa", "acva"), pairs=pairs,
                             n_boot=N_BOOT, seed=SEED)
    cols = []
    for task in ("arabic_exam", "culture_arabic_mmlu"):
        cols += [f"{TASK_LABEL[task]}: plain score", f"{TASK_LABEL[task]}: prior-corrected score"]
    cols += ["AlGhafa, Modern Standard Arabic questions: plain", "AlGhafa, MSA: prior-corrected",
             "AlGhafa, dialect questions (out of scope): plain", "ACVA (listed only)", "AlGhafa, fixed labels (listed only)"]
    t = Table("T3", f"{spec.label}: recognition — multiple-choice accuracy",
              "each number is the share of questions answered correctly (0–1). The model is not asked to write: it "
              "only ranks the given answers by likelihood. 'Plain' divides each answer's likelihood by its length; "
              "'prior-corrected' subtracts how likely the answer is without the question (the PMI rule). Only questions "
              "that no version had to cut for length are counted, so every version sees the same questions.",
              "scripts/mcq_compare.py through arabic_eval.analysis.mcq_compare over each cell's eval_rows/<task>.parquet "
              "(the rows no cell truncated)", cols)
    for k, label, c in rows:
        if res is None or cell_dir(spec, c) is None:
            t.add(label, [pending()] * len(cols))
            continue
        cid = f"{spec.experiment}/{c}"
        src = f"mcq_compare: {c}"
        vals = []
        for task in ("arabic_exam", "culture_arabic_mmlu"):
            cc = res["tasks"][task]["cells"][cid]
            vals += [R.num(cc["acc_char_int"], f3, src, f"T3 {k} {task} char"), R.num(cc["acc_pmi_int"], f3, src, f"T3 {k} {task} pmi")]
        g = res["tasks"]["alghafa"]["cells"][cid]["groups"]
        vals += [R.num(g["choice_text_msa"]["acc_char_int"], f3, src, f"T3 {k} msa char"),
                 R.num(g["choice_text_msa"]["acc_pmi_int"], f3, src, f"T3 {k} msa pmi"),
                 R.num(g["choice_text_dialect"]["acc_char_int"], f3, src, f"T3 {k} dialect char"),
                 R.num(res["tasks"]["acva"]["cells"][cid]["acc_char_int"], f3, src, f"T3 {k} acva"),
                 R.num(g["fixed_label"]["acc_char_int"], f3, src, f"T3 {k} fixed")]
        t.add(label, vals)
    if res:
        n = {task: res["tasks"][task]["intersection_rows"] for task in ("arabic_exam", "culture_arabic_mmlu", "alghafa", "acva")}
        t.notes.append("Questions counted: " + "; ".join(
            f"{TASK_LABEL[k]} {R.num(v, thou, 'mcq_compare intersection_rows', f'T3 n {k}')}" for k, v in n.items()) + ".")
    t.notes += [
        "ACVA's answer keys contain known errors and the fixed-label AlGhafa questions mostly measure which label a "
        "model favours, not understanding; they are listed, not read (docs/report.md §3.10).",
        "The campaign is about Modern Standard Arabic, so AlGhafa's dialect questions are reported but out of scope "
        "(docs/report.md §3.11).",
        "The questions counted are those no version in this table had to cut at the length limit. The main table of "
        "docs/report.md §3.10 also dropped the questions an older, shorter-context version had cut, so its numbers "
        "differ from these in the third decimal; its sensitivity table without that version (_mcq_compare/"
        "mcq_compare_4096) uses the same questions as here.",
    ]
    t.shows = "the recognition side, which the writing tables do not measure. The order of the versions here need not match T2."

    tb = Table("T3b", f"{spec.label}: paired differences in multiple-choice accuracy",
               f"first − second on the same questions, with the 95 % range from {D(thou(N_BOOT))} resamples; 'only first / only "
               "second right' counts the questions exactly one of the two answered correctly.",
               "scripts/mcq_compare.py (pairs block) through arabic_eval.analysis.mcq_compare",
               ["Arabic exams, plain", "Arabic exams, prior-corrected", "translated MMLU, plain",
                "translated MMLU, prior-corrected", "AlGhafa MSA, plain", "only first / only second right (Arabic exams, plain)"])
    if res:
        for task in ("arabic_exam",):
            pass
        pairs = res["tasks"]["arabic_exam"]["pairs"]
        for i, p in enumerate(pairs):
            a, b = p["a"].split("/")[-1], p["b"].split("/")[-1]
            ka = next(k for k, _, c in rows if c == a)
            kb = next(k for k, _, c in rows if c == b)
            src = f"mcq_compare pairs {a} − {b}"
            vals = []
            for task, key in (("arabic_exam", "char"), ("arabic_exam", "pmi"), ("culture_arabic_mmlu", "char"),
                              ("culture_arabic_mmlu", "pmi"), ("alghafa", "choice_text_msa_char")):
                pp = res["tasks"][task]["pairs"][i]
                vals.append(ci3(pp.get(key) or {}, src, f"T3b {ka}-{kb} {task} {key}", ("delta", "ci_low", "ci_high")))
            c0 = pairs[i]["char"]
            vals.append(f"{R.num(c0['only_a'], thou, src, 'T3b only_a')} / {R.num(c0['only_b'], thou, src, 'T3b only_b')}")
            tb.add(f"{ARM_SHORT[ka]} − {ARM_SHORT[kb]}", vals)
    for a, b in (("bpe128k", "native"), ("bpe128k", "bpe16k")):
        if not any(r[0].startswith(ARM_SHORT[a]) and r[0].endswith(ARM_SHORT[b]) for r in tb.rows):
            tb.add(f"{ARM_SHORT[a]} − {ARM_SHORT[b]}", [pending()] * 6)
    tb.shows = "which recognition differences survive pairing."
    return t, tb, res


def t3c(spec: ModelSpec) -> Table:
    t = Table("T3c", f"{spec.label}: letter-fair recognition (a proposal, applied to nothing)",
              "the same exam questions asked four times with the answer letters rotated (أ ب ج د moved round), and "
              "each question decided by the average over the four. This removes a model's taste for one letter. The "
              "'أ gap' is the difference on the questions whose right answer carries the letter أ in the ordinary "
              "order, where AraRooPat lost most.",
              f"{spec.letter_slot or 'pending'} (scripts/mcq_letter_slot.py; {D('1 000', 'configs/experiments/mcq_letter_slot/rows_*.json (1 000 rows each)')}-question subsets)",
              ["Arabic exams: letter-fair accuracy", "translated MMLU: letter-fair accuracy",
               "Arabic exams: difference to BPE-16K, letter-fair", "translated MMLU: difference to BPE-16K, letter-fair",
               "Arabic exams: أ gap to BPE-16K, ordinary order", "translated MMLU: أ gap to BPE-16K, ordinary order"])
    ls = jload(REPO / "outputs" / "experiments" / spec.letter_slot) if spec.letter_slot else None
    src = f"outputs/experiments/{spec.letter_slot}" if spec.letter_slot else ""
    for key, label, name in [(a, ARM_LABEL[a], spec.arms[a].letter) for a in ARM_ORDER] + [("base", "the model before any training", spec.base_letter)]:
        if not ls or not name or name not in ls["tasks"]["arabic_exam"]["arms"]:
            t.add(label, [pending()] * 6)
            continue
        vals = []
        for task in ("arabic_exam", "culture_arabic_mmlu"):
            ra = ls["tasks"][task]["arms"][name]["rotation_averaged"]["char"]
            acc = ra.get("acc") if isinstance(ra, dict) else ra
            acc = acc.get("est") if isinstance(acc, dict) else acc
            vals.append(R.num(acc, f3, src, f"T3c {key} {task} acc"))
        for task in ("arabic_exam", "culture_arabic_mmlu"):
            pk = f"{name}-bpe16k_v5"
            p = ls["tasks"][task]["pairs"].get(pk)
            vals.append(ci3((p or {}).get("rotation_averaged", {}).get("char", {}).get("delta", {}), src, f"T3c {key} {task} ra",
                            ("est", "lo", "hi")) if p else "")
        for task in ("arabic_exam", "culture_arabic_mmlu"):
            pk = f"{name}-bpe16k_v5"
            p = ls["tasks"][task]["pairs"].get(pk)
            vals.append(ci3((p or {}).get("char", {}).get("by_letter", {}).get("أ", {}), src, f"T3c {key} {task} alef",
                            ("est", "lo", "hi")) if p else "")
        t.add(label, vals)
    t.shows = ("that most of AraRooPat's exam deficit comes from how it scores the letter أ, not from the questions: "
               "once the letters are rotated, the Arabic-exam gap to BPE-16K disappears and a smaller gap remains on the "
               "translated MMLU." if spec.key == "qwen3_4b" else
               "the same letter-fair reading on this base model, to set beside Qwen3-4B-Base's T3c.")
    t.notes = ["Letter-fair scoring is a proposal recorded in docs/report.md §3.11–§3.13; no headline number of the "
               "campaign uses it."]
    return t


# --------------------------------------------------------------------------------------------------------------
# T6 / T8 — compute and budget
# --------------------------------------------------------------------------------------------------------------

def model_shape(hf_id: str) -> Optional[Dict[str, Any]]:
    """hidden, layers, attention width, vocab and the parameter count of the cached checkpoint (bf16 bytes / 2)."""
    from huggingface_hub import snapshot_download
    try:
        snap = Path(snapshot_download(hf_id, local_files_only=True, allow_patterns=["config.json", "*.index.json"]))
    except Exception:  # noqa: BLE001
        return None
    cfg = json.loads((snap / "config.json").read_text())
    if "text_config" in cfg:
        # a vision-language checkpoint (Gemma-3): its config.json leaves text fields at their class defaults, and the
        # safetensors total includes the vision tower — so resolve the text config and count the text model's own
        # parameters on the meta device (the model the campaign trains; models/gemma3_adapter.py)
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        tc = AutoConfig.from_pretrained(str(snap)).get_text_config()
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(tc)
        total = sum(p.numel() for p in m.parameters())
        cfg = tc.to_dict()
        source = f"{hf_id} text config (resolved) + text-model parameters counted on the meta device (vision tower excluded)"
    else:
        idx = next(iter(snap.glob("*.index.json")), None)
        total = json.loads(idx.read_text())["metadata"]["total_size"] // 2 if idx else None
        source = f"{hf_id} config.json + safetensors index (cached)"
    heads = cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // heads)
    return {"hidden": cfg["hidden_size"], "layers": cfg["num_hidden_layers"], "attn_width": heads * head_dim,
            "vocab": cfg["vocab_size"], "params_total": total, "tied": cfg.get("tie_word_embeddings", True),
            "source": source}


def t6(spec: ModelSpec, basics: Optional[Dict[str, Any]]) -> Table:
    shape = model_shape(spec.hf_id)
    t = Table("T6", f"{spec.label}: the computation each tokenizer makes the model spend",
              "the model body is the same in every column; what changes is the size of the vocabulary tables and how "
              "many pieces the same text becomes. Operations are counted for one forward pass; 'G' = 10⁹.",
              f"{(shape or {}).get('source', 'model files (pending)')}; {rel(BASICS)}; each arm's all_metrics.json "
              "(training wall times) and config.json (training budgets)",
              [ARM_SHORT[a] for a in ARM_ORDER])
    if not shape or not shape.get("params_total"):
        t.add("model shape", [pending()] * 4)
        return t
    ssrc = shape["source"]
    h, L, aw = shape["hidden"], shape["layers"], shape["attn_width"]
    non_emb = shape["params_total"] - shape["vocab"] * h
    toks = (basics or {}).get("tokenizers", {})
    vocab = {}
    for a in ARM_ORDER:
        name = spec.arms[a].basics
        v = (toks.get(name) or {}).get("vocab_size") if name else None
        if a == "native":
            v = shape["vocab"]
        vocab[a] = v
    t.add("vocabulary rows in the model", [R.num(vocab[a], thou, ssrc if a == "native" else rel(BASICS), f"T6 vocab {a}") for a in ARM_ORDER])
    emb = {a: (vocab[a] * h if vocab[a] else None) for a in ARM_ORDER}
    t.add("vocabulary-table parameters (millions; input and output share one table)",
          [R.num(emb[a] / 1e6 if emb[a] else None, lambda v: f"{v:.1f}", f"{ssrc}: vocab × hidden", f"T6 emb {a}") for a in ARM_ORDER])
    t.add("model-body parameters (millions; the same in every column)",
          [R.num(non_emb / 1e6, lambda v: f"{v:.1f}", f"{ssrc}: total − vocab × hidden", f"T6 body {a}") for a in ARM_ORDER])
    t.add("share of all parameters in the vocabulary table",
          [R.num(emb[a] / (emb[a] + non_emb) if emb[a] else None, pct1, ssrc, f"T6 emb share {a}") for a in ARM_ORDER])
    cpt = {a: ((toks.get(spec.arms[a].basics) or {}).get("references") or {}).get("chars_per_token") if spec.arms[a].basics else None
           for a in ARM_ORDER}
    tpk = {a: (1000.0 / cpt[a] if cpt[a] else None) for a in ARM_ORDER}
    t.add(f"pieces per {D('1 000')} characters of answer text ↓",
          [R.num(tpk[a], thou, f"{rel(BASICS)}: 1 000 ÷ characters per piece on the references", f"T6 tpk {a}") for a in ARM_ORDER])
    for ctx in (512, 4096):
        vals = []
        for a in ARM_ORDER:
            if not (tpk[a] and vocab[a]):
                vals.append(pending())
                continue
            per_tok = 2 * non_emb + 2 * vocab[a] * h + 2 * L * ctx * aw
            vals.append(R.num(tpk[a] * per_tok / 1e9, thou,
                              f"pieces per 1 000 characters × (2 × body + 2 × vocab × hidden + 2 × layers × {ctx} × attention width)",
                              f"T6 gflops {ctx} {a}"))
        t.add(f"forward operations per {D('1 000')} characters, G, at a context of {R.num(ctx, thou, 'definition', f'T6 ctx {ctx}')} pieces ↓", vals)
    # training budget and time
    phase_rows = {"embedding_alignment": "stage 1 (vocabulary tables only)", "warmup": "stage 2 (whole model, web text)",
                  "sft": "stage 3 (whole model, instructions)"}
    p3 = jload(PHASE3) or {}
    params = {a: (non_emb + emb[a]) if emb[a] else None for a in ARM_ORDER}
    pf_total = {a: 0.0 for a in ARM_ORDER}
    pf_ok = {a: True for a in ARM_ORDER}
    for ph, lab in phase_rows.items():
        tok_vals, wall_vals, flop_vals = [], [], []
        for a in ARM_ORDER:
            arm = spec.arms[a]
            src_cell = None
            for c in arm.adapt:
                d = cell_dir(spec, c)
                am = jload(d / "all_metrics.json") if d else None
                v = ((am or {}).get("training") or {}).get(ph) or {}
                if v.get("wall_time_sec"):
                    src_cell = (d, v)
            if src_cell is None:
                skip = "not run" if a == "native" and ph != "sft" and arm.adapt and cell_dir(spec, arm.adapt[0]) else pending()
                tok_vals.append(skip); wall_vals.append(skip); flop_vals.append(skip)
                if skip != "not run":
                    pf_ok[a] = False
                continue
            d, v = src_cell
            cfg = jload(d / "config.json") or {}
            pc = (((cfg.get("training") or {}).get("phases") or {}).get(ph)) or {}
            n_tok, src = pc.get("mix_tokens"), rel(d / "config.json")
            if ph == "sft":
                rec = p3.get(rel(d))
                n_tok, src = (rec["input_tokens_processed_est"], f"{rel(PHASE3)} ({d.name})") if rec else (None, "")
            if n_tok:
                tok_vals.append(R.num(n_tok / 1e6, lambda x: f"{x:.1f}", src, f"T6 tokens {ph} {a}"))
                per_tok = (2 * params[a] + 4 * emb[a]) if ph == "embedding_alignment" else 6 * params[a]
                pf = per_tok * n_tok / 1e15
                pf_total[a] += pf
                flop_vals.append(R.num(pf, thou, f"{src}; {'(2 × params + 4 × table) × pieces' if ph == 'embedding_alignment' else '6 × params × pieces'}",
                                       f"T6 pflops {ph} {a}"))
            else:
                tok_vals.append(pending()); flop_vals.append(pending()); pf_ok[a] = False
            wall_vals.append(R.num(v["wall_time_sec"] / 3600, f2, rel(d / "all_metrics.json"), f"T6 wall {ph} {a}"))
        t.add(f"{lab}: training pieces (millions)", tok_vals)
        t.add(f"{lab}: training operations, P (10¹⁵)", flop_vals)
        t.add(f"{lab}: GPU hours (measured)", wall_vals)
    t.add("all stages: training operations, P (10¹⁵)",
          [R.num(pf_total[a], thou, "sum of the rows above", f"T6 pflops total {a}") if pf_ok[a] and pf_total[a] else pending()
           for a in ARM_ORDER])
    # equal text: the words of Arabic web text the stage 1 + 2 pieces covered (pieces ÷ pool pieces per word)
    toks = (basics or {}).get("tokenizers", {})
    tpw = {a: ((toks.get(spec.arms[a].basics) or {}).get("pool") or {}).get("tokens_per_word") if spec.arms[a].basics else None
           for a in ARM_ORDER}
    mix = {}
    for a in ARM_ORDER:
        tot = 0
        for c in spec.arms[a].adapt:
            d = cell_dir(spec, c)
            for ph in ("embedding_alignment", "warmup"):
                pc = (((jload(d / "config.json") or {}).get("training") or {}).get("phases") or {}).get(ph) or {} if d else {}
                if pc.get("enabled") and pc.get("mix_tokens"):
                    tot += pc["mix_tokens"]
        mix[a] = tot or None
    words = {a: (mix[a] / tpw[a]) if (mix[a] and tpw[a]) else None for a in ARM_ORDER}
    t.add("stages 1–2: millions of words of web text those pieces covered",
          ["not applicable" if a == "native" else R.num(words[a] / 1e6 if words[a] else None, lambda x: f"{x:.0f}",
                                                         f"stage 1 + 2 pieces ÷ {rel(BASICS)} pool pieces per word", f"T6 words {a}")
           for a in ARM_ORDER])
    ref_words = words.get("bpe128k")
    t.add("pieces it would need to read the words BPE-128K read (millions)",
          ["not applicable" if a == "native" else R.num(ref_words * tpw[a] / 1e6 if (ref_words and tpw[a]) else None,
                                                         lambda x: f"{x:.0f}", "BPE-128K's words × this tokenizer's pool pieces per word",
                                                         f"T6 equal-text {a}")
           for a in ARM_ORDER])
    # inference: T4 (timing_v1.json) and T5 (speed_v1.json)
    tim = {a: jload(cell_dir(spec, spec.arms[a].ff) / "timing_v1.json") if cell_dir(spec, spec.arms[a].ff) else None
           for a in ARM_ORDER}

    def trow(label, fn, fmt, where):
        vals = []
        for a in ARM_ORDER:
            d = tim.get(a)
            v = fn(d) if d else None
            vals.append(R.num(v, fmt, rel(cell_dir(spec, spec.arms[a].ff) / "timing_v1.json"), f"{where} {a}") if v is not None else pending())
        t.add(label, vals)

    trow("characters written per second, one answer at a time ↑",
         lambda d: 1e5 / d["batch1"]["repeat_1_all_prompts"]["ms_per_100_generated_chars"], thou, "T6 b1 cps")
    trow("characters written per second, 16 answers at a time ↑", lambda d: d["batch16"]["chars_per_sec"], thou, "T6 b16 cps")
    trow("energy per 1 000 answers, Wh ↓", lambda d: d["batch1"]["repeat_1_all_prompts"].get("wh_per_1000_answers"),
         lambda v: f"{v:.0f}", "T6 wh")
    trow("multiple-choice, ms per Arabic-exam question ↓", lambda d: (d.get("mcq") or {}).get("arabic_exam", {}).get("ms_per_row_median"),
         lambda v: f"{v:.0f}", "T6 mcq ms")
    sp_files = sorted(EXP_ROOT.glob(SPEED_GLOB))
    sp = jload(sp_files[-1]) if sp_files else None
    sp_name = {"native": {"qwen3_4b": "native_qwen3", "llama3b": "native_llama3b", "gemma3_4b": "native_gemma3"}.get(spec.key),
               "araroopat": "araroopat", "bpe16k": "bpe_16k", "bpe128k": "bpe_128k"}
    vals = []
    for a in ARM_ORDER:
        c = next((c for c in (sp or {}).get("cases", []) if c.get("name") == sp_name[a] and c.get("mode") == "encode"
                  and not c.get("error")), None)
        ev = ((c or {}).get("sets") or {}).get("eval_split") or {}
        v = (ev.get("steady") or {}).get("us_per_word") if a == "araroopat" else (ev.get("encode") or {}).get("us_per_word")
        vals.append(R.num(v, lambda x: f"{x:.1f}", rel(sp_files[-1]) if sp_files else "", f"T6 tok us {a}") if v else pending())
    t.add("the tokenizer's processor time, µs per word (AraRooPat: steady) ↓", vals)
    t.shows = ("that the vocabulary tables are a small part of the model at 16 000 pieces and a large part at "
               "128 000 or more, and that a tokenizer cutting text finer makes the model do more work for the same text.")
    t.notes = ["Stage 3 trains on a fixed number of instruction examples (the same number for every version, set by the mixture), not a fixed "
               "number of pieces; its piece count is the examples the run processed × the mixture's mean pieces per "
               "example (scripts/count_phase3_tokens.py).",
               "The native model needs no stages 1 and 2: its vocabulary is the one it was pretrained with.",
               "Equal compute is not equal text: the same 131 M pieces cover very different amounts of Arabic under "
               "different tokenizers (the two 'words' rows).",
               "Peak memory was not recorded for the Qwen3-4B runs; the Llama and Gemma runs record it (chains/*peak_mem.jsonl)."]
    return t


def t8(spec: ModelSpec) -> Table:
    t = Table("T8", f"{spec.label}: GPU time actually spent per version",
              "measured wall-clock hours on one H100 GPU, from the runs' own records; loading times are not included.",
              "each cell's all_metrics.json (training wall times, free-form generation time, multiple-choice "
              "inference time) and freeform_judge/<judge>.parquet metadata (judge seconds)",
              ["training", "writing eval (generation)", "multiple-choice eval", "judging (all judges)", "total"])
    for a in ARM_ORDER:
        arm = spec.arms[a]
        tr = ev = mc = jd = 0.0
        srcs = []
        found = False
        for c in arm.adapt:
            d = cell_dir(spec, c)
            am = jload(d / "all_metrics.json") if d else None
            if not am:
                continue
            found = True
            for ph, v in (am.get("training") or {}).items():
                if isinstance(v, dict) and v.get("wall_time_sec") and (c == arm.adapt[-1] or ph != "sft"):
                    tr += v["wall_time_sec"]
            srcs.append(rel(d / "all_metrics.json"))
        d = cell_dir(spec, arm.ff)
        if d:
            ff = ((jload(d / "all_metrics.json") or {}).get("downstream") or {}).get("freeform_cidar") or {}
            ev += ff.get("generation_wall_sec") or 0.0
            import pyarrow.parquet as pq
            for f in (d / "freeform_judge").glob("*.parquet"):
                md = pq.read_schema(f).metadata or {}
                meta = json.loads(md.get(b"arabic_eval", b"{}"))
                jd += float(meta.get("seconds") or 0.0)
        dm = cell_dir(spec, arm.mcq)
        if dm:
            for task, v in ((jload(dm / "all_metrics.json") or {}).get("downstream") or {}).items():
                if isinstance(v, dict) and v.get("inference_time_sec"):
                    mc += v["inference_time_sec"]
        if not found and not d:
            t.add(ARM_LABEL[a], [pending()] * 5)
            continue
        src = "; ".join(srcs) or "—"
        t.add(ARM_LABEL[a], [R.num(tr / 3600, f2, src, f"T8 train {a}"), R.num(ev / 3600, f2, src, f"T8 ff {a}"),
                             R.num(mc / 3600, f2, src, f"T8 mcq {a}"), R.num(jd / 3600, f2, src, f"T8 judge {a}"),
                             R.num((tr + ev + mc + jd) / 3600, f2, src, f"T8 total {a}")])
    t.shows = "where the GPU budget went."
    t.notes = ["AraRooPat and BPE-16K share their stages 1 and 2 between the earlier and the distilled versions; "
               "the stage-3 hours are those of the distilled version only."]
    return t



# --------------------------------------------------------------------------------------------------------------
# T4 — response time (scripts/measure_response_time.py → <cell>/timing_v1.json)
# --------------------------------------------------------------------------------------------------------------

def t4(spec: ModelSpec) -> Optional[Table]:
    rows = [(a, ARM_LABEL[a], spec.arms[a].ff) for a in ARM_ORDER] + [("base", "the model before any training", spec.base_ff)]
    have = {k: jload(cell_dir(spec, c) / "timing_v1.json") for k, _, c in rows if cell_dir(spec, c) and
            (cell_dir(spec, c) / "timing_v1.json").exists()}
    if not have:
        return None
    cols = ["seconds per answer (median, one at a time)", "milliseconds per 100 characters written ↓",
            "time to the first piece, ms ↓", "tokenizer's share of answer time", "characters per second, 16 at a time ↑",
            "energy per 1 000 answers, Wh ↓", "multiple-choice: ms per question, Arabic exams ↓",
            "multiple-choice: ms per question, AlGhafa ↓", "answers identical to the evaluated ones"]
    t = Table("T4", f"{spec.label}: how long the model takes to answer under each tokenizer",
              "every version answers the same 250 instructions one at a time on an otherwise idle GPU, under the "
              "evaluation's own rules; the multiple-choice rows score the same 500 questions per benchmark. Fewer "
              "pieces per character usually means fewer steps per answer.",
              "each cell's timing_v1.json (scripts/measure_response_time.py)", cols)
    for k, label, c in rows:
        d = have.get(k)
        if not d:
            t.add(label, [pending()] * len(cols))
            continue
        src = rel(cell_dir(spec, c) / "timing_v1.json")
        b1 = d["batch1"]["repeat_1_all_prompts"]
        mcq = d.get("mcq") or {}
        t.add(label, [R.num(b1["sec_per_answer_median"], f2, src, f"T4 {k} spa"),
                      R.num(b1["ms_per_100_generated_chars"], lambda v: f"{v:.0f}", src, f"T4 {k} ms100"),
                      R.num(b1["ttft_ms_median"], lambda v: f"{v:.0f}", src, f"T4 {k} ttft"),
                      R.num(b1["tokenizer_share_of_answer_time"], pct1, src, f"T4 {k} tokshare"),
                      R.num(d["batch16"]["chars_per_sec"], thou, src, f"T4 {k} b16"),
                      R.num(b1.get("wh_per_1000_answers"), lambda v: f"{v:.1f}", src, f"T4 {k} wh"),
                      R.num((mcq.get("arabic_exam") or {}).get("ms_per_row_median"), lambda v: f"{v:.0f}", src, f"T4 {k} mcq exam"),
                      R.num((mcq.get("alghafa") or {}).get("ms_per_row_median"), lambda v: f"{v:.0f}", src, f"T4 {k} mcq alghafa"),
                      R.num(b1["equal_to_dump_share"], pct0, src, f"T4 {k} eq")])
    t.shows = "the cost of an answer in time and energy, per tokenizer, on the same hardware."
    t.notes = ["One answer at a time is the latency a user sees; sixteen at a time is the throughput of the evaluation. "
               "Answers differ in length between versions, so 'per 100 characters' is the fair comparison.",
               "AraRooPat's encoding goes through a morphological analyser in a second process; its share of the "
               "answer time is an implementation cost, measured apart in T5.",
               "'Answers identical to the evaluated ones' checks that one-at-a-time decoding reproduces the "
               "evaluated answers (small differences come from batching arithmetic)."]
    return t


# --------------------------------------------------------------------------------------------------------------
# T5 — the tokenizers' own speed (scripts/measure_tokenizer_speed.py → _tokenizer_speed/<date>/speed_v1.json)
# --------------------------------------------------------------------------------------------------------------

SPEED_LABEL = {"native_qwen3": "native Qwen3", "native_llama3b": "native Llama-3.2", "native_gemma3": "native Gemma-3",
               "bpe_16k": "BPE-16K", "bpe_128k": "BPE-128K", "araroopat": "AraRooPat"}


def t5() -> Optional[Table]:
    files = sorted(EXP_ROOT.glob(SPEED_GLOB))
    if not files:
        return None
    f = files[-1]
    d = jload(f)
    src = rel(f)
    cols = ["encode, µs per word (one text at a time) ↓", "encode, µs per word (a batch, all cores) ↓",
            "decode, µs per word ↓", "encode latency of one answer-length text, p95 ms ↓",
            "whole texts given back exactly (eval texts)", "load time, s", "memory added by loading, MB"]
    t = Table("T5", "the tokenizers' own speed (processor only, no model)",
              f"the same {R.num(d['sets']['eval_split']['texts'], thou, src, 'T5 n eval')} Arabic texts, "
              "encoded and decoded on the processor; AraRooPat is shown cold (first contact, its analyser starting), "
              "warm (the same texts again) and steady (new texts once it is running).",
              f"{src} (scripts/measure_tokenizer_speed.py); machine: {d['machine'].get('cpu')}", cols)
    by = {}
    for c in d["cases"]:
        by.setdefault(c["name"], {})[c.get("mode")] = c
    for name, modes in by.items():
        base = modes.get("encode") or {}
        if base.get("error") or not base.get("sets"):
            t.add(SPEED_LABEL.get(name, name), [pending("not measured")] * len(cols))
            continue
        ev = base["sets"]["eval_split"]
        refs = base["sets"]["references"]
        batch = ((modes.get("batch_parallel_default") or {}).get("sets") or {}).get("eval_split", {}).get("encode_batch")
        load_mb = (base.get("rss_mb_after_load") or 0) - (base.get("rss_mb_before_load") or 0)
        if name == "araroopat":
            for phase in ("cold", "warm", "steady"):
                e = ev[phase]
                t.add(f"AraRooPat — {phase}", [
                    R.num(e["us_per_word"], lambda v: f"{v:.1f}", src, f"T5 ara {phase} enc"), "not applicable",
                    R.num(ev["decode"]["us_per_word"], lambda v: f"{v:.1f}", src, "T5 ara dec") if phase == "warm" else "",
                    R.num(refs[phase]["latency_ms_p95"], f2, src, f"T5 ara {phase} p95"),
                    R.num(ev["roundtrip_exact"], pct1, src, "T5 ara rt") if phase == "warm" else "",
                    R.num(base["load_sec"], f2, src, "T5 ara load") if phase == "cold" else "",
                    R.num(load_mb + (base.get("camel_server_rss_mb") or 0), thou, src, "T5 ara mem") if phase == "cold" else ""])
            continue
        t.add(SPEED_LABEL.get(name, name), [
            R.num(ev["encode"]["us_per_word"], f2, src, f"T5 {name} enc"),
            R.num(batch["us_per_word"], f2, src, f"T5 {name} batch") if batch else "",
            R.num(ev["decode"]["us_per_word"], f2, src, f"T5 {name} dec"),
            R.num(refs["encode"]["latency_ms_p95"], f3, src, f"T5 {name} p95"),
            R.num(ev["roundtrip_exact"], pct1, src, f"T5 {name} rt"),
            R.num(base["load_sec"], f2, src, f"T5 {name} load"),
            R.num(load_mb, thou, src, f"T5 {name} mem")])
    t.shows = ("how much processor time the tokenizer itself costs. The native and BPE tokenizers run compiled code; "
               "AraRooPat runs Python plus a morphological analyser in a second process, so its numbers describe "
               "this implementation, not the method's limit.")
    t.notes = ["Measured on a machine also running a GPU training job; the load is recorded in the source file.",
               "AraRooPat rarely gives a whole text back character for character: decoding re-spaces punctuation, writes "
               "Latin-script words as '?' (it has no Latin pieces), and restores a hamza the writer dropped (ا → أ, its "
               "canonical spelling, docs/report.md §3.5). Single Arabic words come back exactly far more often (T1).",
               "AraRooPat's cold row includes its analyser warming up on unseen words; the steady row is new texts once "
               "the analyser runs, the case of real use."]
    return t

# --------------------------------------------------------------------------------------------------------------
# pending tables and the document
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# T7 — across models (the pre-registered readings G1–G5 and R-128-1..4 of brief v6; J6 / J10 in T7b)
# --------------------------------------------------------------------------------------------------------------

BAKEOFF = "teacher_bakeoff"
BAKEOFF_LABEL = {"aya_expanse_32b": "Aya-Expanse-32B", "allam_7b": "ALLaM-7B", "cmdr7b_arabic": "Command-R7B-Arabic",
                 "falcon_h1_34b": "Falcon-H1-34B", "llama33_70b_awq": "Llama-3.3-70B (AWQ)", "jais2_8b": "Jais-2-8B"}
JUDGE_OWN_BAKEOFF = {"falcon_h1_34b": "falcon_h1_34b"}     # J10: a judge never ranks its own answers


def n_rated() -> int:
    rating = EXP_ROOT / E / "freeform_rating" / "v5_blind.ratings"
    n = 0
    if rating.is_dir():
        for p in rating.glob("*.json"):                     # the count only (rule S0)
            n = max(n, len(json.loads(p.read_text(encoding="utf-8"))))
    return n


def side(d: Dict[str, Any], keys=("delta_mean", "ci_low", "ci_high")) -> str:
    lo, hi = d.get(keys[1]), d.get(keys[2])
    if lo is None or hi is None:
        return ""
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "includes zero")


def judge_mean(d: Optional[Path], j: str) -> Optional[float]:
    if not judge_file(d, j):
        return None
    s = judge_scores(d, j)
    return sum(s.values()) / len(s) if s else None


def writing_diff(spec: ModelSpec, a: str, b: str, j: str, where: str) -> str:
    ca, cb = spec.arms[a].ff, spec.arms[b].ff
    da, db = cell_dir(spec, ca), cell_dir(spec, cb)
    if da is None or db is None:
        return pending()
    if not (judge_file(da, j) and judge_file(db, j)):
        return "not judged yet"
    s = ae.judge_compare(cell_id(spec, cb), cell_id(spec, ca), judge=j, n_boot=N_BOOT, seed=SEED)["score"]
    return f"{ci3(s, f'judge_compare({cb}, {ca}, judge={j})', where)}: {side(s)}"


def mcq_order(spec: ModelSpec, res: Optional[Dict[str, Any]], task: str, norm: str) -> str:
    """The four arms by accuracy, best first; '>' where the paired 95 % range of the two neighbours excludes zero,
    '≈' where it includes zero (cannot be told apart)."""
    if res is None:
        return pending()
    cid = {a: f"{spec.experiment}/{spec.arms[a].mcq}" for a in ARM_ORDER}
    cells = res["tasks"][task]["cells"]
    if any(cid[a] not in cells for a in ARM_ORDER):
        return pending()
    acc = {a: cells[cid[a]][f"acc_{norm}_int"] for a in ARM_ORDER}
    order = sorted(ARM_ORDER, key=lambda a: -acc[a])
    pairs = {(p["a"], p["b"]): p for p in res["tasks"][task]["pairs"]}
    out = ARM_SHORT[order[0]]
    for x, y in zip(order, order[1:]):
        p = pairs.get((cid[x], cid[y])) or pairs.get((cid[y], cid[x]))
        sep = None
        if p and p.get(norm) and p[norm].get("ci_low") is not None:
            sep = p[norm]["ci_low"] > 0 or p[norm]["ci_high"] < 0
        out += (" > " if sep else (" ≈ " if sep is False else " ? ")) + ARM_SHORT[y]
    return out


def letter_pair(spec: ModelSpec, a: str, kind: str, where: str) -> str:
    """kind 'alef': the gold-letter أ gap to BPE-16K in the ordinary order; 'fair': the rotation-averaged difference."""
    ls = jload(REPO / "outputs" / "experiments" / spec.letter_slot) if spec.letter_slot else None
    name = spec.arms[a].letter
    if not ls or not name:
        return "pending (rotation cells, P4)"
    src = f"outputs/experiments/{spec.letter_slot}"
    parts = []
    for task, tl in (("arabic_exam", "exams"), ("culture_arabic_mmlu", "MMLU")):
        p = ls["tasks"][task]["pairs"].get(f"{name}-bpe16k_v5")
        if not p:
            return pending()
        d = p["char"]["by_letter"]["أ"] if kind == "alef" else p["rotation_averaged"]["char"]["delta"]
        parts.append(f"{tl} {ci3(d, src, f'{where} {task}', ('est', 'lo', 'hi'))}")
    return "; ".join(parts)


def t7(res_by_model: Dict[str, Optional[Dict[str, Any]]], primary: Optional[str]) -> Tuple[Table, Table]:
    j = primary or DEFAULT_PRIMARY
    jl = JUDGES[j]["label"]
    cols = [f"G1 AraRooPat − BPE-16K, writing ({jl})", f"G2 AraRooPat − native, writing ({jl})",
            "G3 recognition order, Arabic exams (plain; prior-corrected)",
            "G3 recognition order, translated MMLU (plain; prior-corrected)",
            "G4 AraRooPat − BPE-16K on questions whose answer is أ (ordinary order)",
            f"G5 share of the teacher's score ({jl})",
            f"R-128-1 BPE-128K − BPE-16K, writing ({jl})",
            "R-128-2 BPE-128K − BPE-16K, recognition (plain; prior-corrected; letter-fair)",
            "R-128-3 BPE-128K − BPE-16K on questions whose answer is أ",
            "R-128-4 BPE-128K pieces never seen in adaptation; used in the answers"]
    t = Table("T7", "across models — do the findings replicate?",
              "one row per base model. Each reading was written down before its numbers existed (brief v6; docs/report.md "
              "§3.14–§3.17). A difference is first − second with its 95 % range and where that range lies: 'above zero' "
              "and 'below zero' mean the difference is unlikely to be chance, 'includes zero' means the two cannot be "
              "told apart. In an order, '>' separates versions whose paired range excludes zero and '≈' joins versions "
              "that cannot be told apart.",
              "the same files as T2b, T3, T3b and T3c of each model, plus _comparative_tables/m5/"
              "{adaptation_token_counts,unseen_in_answers}.json", cols)
    counts = jload(COUNTS) or {}
    unseen_ans = jload(TABLES_ROOT / "m5" / "unseen_in_answers.json") or {}
    for spec in MODELS:
        if not any(cell_dir(spec, a.ff) for a in spec.arms.values()):
            t.add(spec.label, [pending()] * len(cols))
            continue
        res = res_by_model.get(spec.key)
        row = [writing_diff(spec, "araroopat", "bpe16k", j, f"T7 {spec.key} G1"),
               writing_diff(spec, "araroopat", "native", j, f"T7 {spec.key} G2")]
        for task in ("arabic_exam", "culture_arabic_mmlu"):
            row.append(f"{mcq_order(spec, res, task, 'char')}; {mcq_order(spec, res, task, 'pmi')}"
                       if res else pending())
        row.append(letter_pair(spec, "araroopat", "alef", f"T7 {spec.key} G4"))
        tm = judge_mean(cell_dir(spec, spec.teacher), j)
        shares = []
        for a in ARM_ORDER:
            d = cell_dir(spec, spec.arms[a].ff)
            m = judge_mean(d, j)
            if m is None or not tm:
                shares.append(f"{ARM_SHORT[a]} {pending() if d is None else 'not judged yet'}")
            else:
                shares.append(f"{ARM_SHORT[a]} {R.num(m / tm, f2, f'{rel(judge_file(d, j))} ÷ the teacher mean ({j})', f'T7 {spec.key} G5 {a}')}")
        row.append(" · ".join(shares))
        row.append(writing_diff(spec, "bpe128k", "bpe16k", j, f"T7 {spec.key} R1"))
        if res and all(f"{spec.experiment}/{spec.arms[a].mcq}" in res["tasks"]["arabic_exam"]["cells"] for a in ("bpe128k", "bpe16k")):
            ia, ib = f"{spec.experiment}/{spec.arms['bpe128k'].mcq}", f"{spec.experiment}/{spec.arms['bpe16k'].mcq}"
            bits = []
            for task, tl in (("arabic_exam", "exams"), ("culture_arabic_mmlu", "MMLU")):
                p = next(p for p in res["tasks"][task]["pairs"] if (p["a"], p["b"]) == (ia, ib))
                src = f"mcq_compare pairs {spec.arms['bpe128k'].mcq} − {spec.arms['bpe16k'].mcq}"
                bits.append(f"{tl} {ci3(p['char'], src, f'T7 {spec.key} R2 {task} char', ('delta', 'ci_low', 'ci_high'))} "
                            f"and {ci3(p['pmi'], src, f'T7 {spec.key} R2 {task} pmi', ('delta', 'ci_low', 'ci_high'))}")
            fair = letter_pair(spec, "bpe128k", "fair", f"T7 {spec.key} R2 fair")
            row.append(" — ".join(bits) + f" — letter-fair: {fair}")
        else:
            row.append(pending())
        row.append(letter_pair(spec, "bpe128k", "alef", f"T7 {spec.key} R3"))
        ad = spec.arms["bpe128k"].adapt[0]
        ck = counts.get(f"outputs/experiments/{spec.experiment}/{ad}")
        ans = unseen_ans.get(spec.arms["bpe128k"].ff if spec.key == "qwen3_4b" else f"{spec.experiment}/{spec.arms['bpe128k'].ff}")
        if ck:
            src = f"{rel(COUNTS)} (outputs/experiments/{spec.experiment}/{ad}: the packed blocks its stages 1–2 read)"
            never = R.num(ck["seen_0"] / ck["regular_ids"], lambda v: f"{100 * v:.2f} %", src, f"T7 {spec.key} R4 never")
            used = (f"{R.num(ans['unseen_pieces'], thou, 'm5/unseen_in_answers.json', f'T7 {spec.key} R4 used')} of "
                    f"{R.num(ans['pieces'], thou, 'm5/unseen_in_answers.json', f'T7 {spec.key} R4 pieces')} answer pieces"
                    if ans else pending())
            row.append(f"{never} never seen; used: {used}")
        else:
            row.append(pending())
        t.add(spec.label, row)
    t.shows = ("where each finding of the campaign replicates across the three base models and where it does not.")
    t.notes = [f"The writing readings use {jl}{' (the main judge)' if primary else ' (candidate C1, the main judge by default until the human rating names one — docs/report.md §3.14, S4)'}; "
               "T2b of each model has the other judges' columns.",
               "G4 and R-128-3 need the four letter-rotation cells of each version (optional stage P4); they exist "
               "for all three base models.",
               "'≈' and 'includes zero' mean the two versions cannot be told apart with these questions, not that they are equal."]

    # T7b — the judge readings J6 (the three headline differences under every judge) and J10 (the bake-off order)
    q = MODELS[0]
    judges = [x for x in (j, *[c for c in CANDIDATES if c != j], "gemma4_31b", "gpt56_terra")
              if any(judge_file(cell_dir(q, q.arms[a].ff), x) for a in ARM_ORDER)]
    tb = Table("T7b", "the judge readings: do the judges agree on the headline differences and on the teacher choice?",
               "J6 repeats three paired differences of Qwen3-4B-Base (T2b) under every judge that scored them. J10 "
               "orders the six teacher candidates of the bake-off by their mean score under each judge; the teacher "
               "(Aya-Expanse-32B) was chosen under Gemma-4-31B, and a different order under the new judge is a "
               "finding, not a re-choice. A judge that was itself a candidate is never used to rank its own answers.",
               "judge_compare over the Qwen3-4B v5 cells; outputs/experiments/teacher_bakeoff/<candidate>/freeform_judge/<judge>.parquet",
               [JUDGES[x]["label"] for x in judges] or ["judge"])
    for a, b in (("araroopat", "bpe16k"), ("araroopat", "native"), ("bpe16k", "native")):
        tb.add(f"J6 {ARM_SHORT[a]} − {ARM_SHORT[b]}, writing (Qwen3-4B)",
               [writing_diff(q, a, b, x, f"T7b J6 {a}-{b} {x}") for x in judges])
    orders = []
    for x in judges:
        means = {}
        for c in BAKEOFF_LABEL:
            if JUDGE_OWN_BAKEOFF.get(x) == c:
                continue
            m = judge_mean(EXP_ROOT / BAKEOFF / c if (EXP_ROOT / BAKEOFF / c).is_dir() else None, x)
            if m is not None:
                means[c] = m
        if len(means) < len(BAKEOFF_LABEL) - (1 if x in JUDGE_OWN_BAKEOFF else 0):
            orders.append("not judged (judge retired)" if x in RETIRED_JUDGES else
                          "not judged (runs only if the human rating names this judge, rule S3)")
            continue
        src = f"outputs/experiments/{BAKEOFF}/<candidate>/freeform_judge/{x}.parquet"
        orders.append(" > ".join(f"{BAKEOFF_LABEL[c]} {R.num(m, f2, src, f'T7b J10 {x} {c}')}"
                                 for c, m in sorted(means.items(), key=lambda kv: -kv[1]))
                      + (f" (its own cell, {BAKEOFF_LABEL[JUDGE_OWN_BAKEOFF[x]]}, left out)" if x in JUDGE_OWN_BAKEOFF else ""))
    tb.add("J10 the teacher candidates, best first (mean score on the bake-off prompts)", orders)
    tb.shows = "whether the campaign's two judge-dependent conclusions survive the change of judge."
    return t, tb


def pending_table(tid: str, title: str, what: str) -> str:
    return f"### {tid} — {title}\n\n*Pending.* {what}\n"


def model_status(spec: ModelSpec, j: str) -> str:
    """Where a base model stands, read from the files (no number in the sentence)."""
    ff = [cell_dir(spec, c) for c in [spec.arms[a].ff for a in ARM_ORDER] + [spec.base_ff]]
    if not any(ff):
        return "not started"
    if not all(ff):
        return "training"
    mcq = all(cell_dir(spec, c) for c in [spec.arms[a].mcq for a in ARM_ORDER] + [spec.base_mcq])
    judged = all(judge_file(d, j) for d in ff)
    timed = all((d / "timing_v1.json").exists() for d in ff)
    return "complete" if (mcq and judged and timed) else "trained, evaluations running"


def preface(primary: Optional[str], primary_note: str, n_models_run: int) -> str:
    pj = JUDGES[primary]["label"] if primary else "not chosen yet"
    j = primary or DEFAULT_PRIMARY
    status = {m.key: model_status(m, j) for m in MODELS}
    models_line = "; ".join(f"{m.label} ({status[m.key]})" for m in MODELS)
    pend = [f"{m.label} ({status[m.key]})" for m in MODELS if status[m.key] != "complete"]
    nr = n_rated()
    if nr < 200:
        pend.append(f"the human rating ({R.num(nr, '{}', 'freeform_rating/v5_blind.ratings (count only)', 'preface n_rated')} "
                    f"of {R.num(200, '{}', 'freeform_rating/v5_blind.json (n_items)', 'preface n_items')} answers rated)")
    if any(m.letter_slot and not (REPO / "outputs" / "experiments" / m.letter_slot).exists() for m in MODELS):
        pend.append("the letter-rotation cells of the two new base models (optional)")
    if not pend:
        pend.append("nothing")
    return f"""# Comparing Arabic tokenizers — tables for the supervisor

*Generated by `scripts/build_comparative_tables.py` on {R.num(time.strftime('%Y-%m-%d'), '{}', 'build date', 'date')} from the result files; do not edit by hand.*

**The question.** A tokenizer cuts text into the pieces a language model reads and writes. We keep everything else
fixed — the base model, the training data, the training steps, the evaluation — and change only the tokenizer, to see
what the tokenizer alone changes. Four tokenizers are compared: the model's **own** (native) tokenizer; **AraRooPat**,
which writes each Arabic word as a root and a pattern; and two ordinary **BPE** tokenizers trained on the same Arabic
text, one with 16 000 pieces and one with 128 000 (about the size of the native ones). Three base models, to see
whether the findings hold for other model families: {models_line}.

**Equal treatment.** Every version goes through the same training: the from-scratch tokenizers first get two stages
that teach the model its new vocabulary on Arabic web text, then all versions get the same final stage on Arabic
instructions answered by one teacher model (Aya-Expanse-32B). Nothing is tuned for one version alone.

**Two kinds of evaluation.** *Writing*: each version answers the same 250 held-out Arabic instructions, and a separate
judge model scores each answer from 1 to 5 (T2). *Recognition*: four multiple-choice benchmarks, where the model
only ranks given answers (T3).

**The judge.** Gemma-4-31B judged the earlier results, including the deck of 26 September. Gemma is now one of the base
models, so it cannot stay the main judge. The new main judge is an open model run locally, chosen among candidates
from different companies (gpt-oss-120b from OpenAI and Falcon-H1-34B from TII; a third candidate could not be loaded
and was dropped) by its agreement with a blind human rating; until that rating is complete, every candidate's column
is shown. Main judge: **{pj}** — {primary_note}.

**What is pending.** {'; '.join(pend)}. Pending cells say "pending" — they are never zero. T7 at the end
reads the pre-registered findings across the three base models.
"""


def build(args) -> int:
    basics = jload(BASICS)
    primary, primary_note = primary_judge()
    parts: List[str] = []
    tables: List[Table] = []
    run_models = [m for m in MODELS if any(cell_dir(m, a.ff) for a in m.arms.values())]
    res_by_model: Dict[str, Optional[Dict[str, Any]]] = {}
    parts.append(preface(primary, primary_note, len(run_models)))
    for spec in MODELS:
        parts.append(f"\n## {spec.label}\n")
        if spec not in run_models:
            parts.append(f"*Pending: no cell of {spec.label} has run yet.*\n")
            continue
        tt = [t1(spec, basics)]
        ta, tb = t2(spec, primary)
        tt += [ta, tb, t2b(spec, primary), t2c(spec, primary)]
        t3a, t3b_, res_by_model[spec.key] = t3(spec)
        tt += [t3a, t3b_, t3c(spec)]
        for t in tt:
            parts.append(t.md())
        t4_ = t4(spec)
        if t4_ is not None:
            parts.append(t4_.md())
            tt.append(t4_)
        else:
            parts.append(pending_table("T4", f"{spec.label}: response time per tokenizer",
                                       f"Seconds per answer, time to the first piece, milliseconds per {D('100')} characters, "
                                       f"energy per {D('1 000')} answers and multiple-choice time per question, all measured "
                                       "one answer at a time on an otherwise idle GPU (scripts/measure_response_time.py)."))
        if spec is MODELS[0]:
            t5_ = t5()
            if t5_ is not None:
                parts.append(t5_.md())
                tt.append(t5_)
            else:
                parts.append(pending_table("T5", "the tokenizers' own speed",
                                           "Encoding and decoding speed on fixed texts, cold and warm, with memory "
                                           "(scripts/measure_tokenizer_speed.py)."))
        t6_ = t6(spec, basics)
        t8_ = t8(spec)
        parts.append(t6_.md())
        parts.append(t8_.md())
        tt += [t6_, t8_]
        tables += [(spec.key, t) for t in tt]
    parts.append("\n## Across models\n")
    t7a, t7b = t7(res_by_model, primary)
    parts += [t7a.md(), t7b.md()]
    tables += [("all", t7a), ("all", t7b)]
    doc = "\n".join(parts)
    shape_bad = [f"{mk} {t.tid}: row {lab!r} has {len(c)} cells for {len(t.columns)} columns"
                 for mk, t in tables for lab, c in t.rows if len(c) != len(t.columns)]
    if shape_bad:
        print("REFUSED — table rows of the wrong width:\n  " + "\n  ".join(shape_bad[:20]))
        return 3
    bad = R.check(doc)
    if bad:
        print("REFUSED — numbers without a source:\n  " + "\n  ".join(bad[:40]))
        return 2
    out_dir = TABLES_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    for mk, t in tables:
        t.csv(out_dir / f"{mk}_{t.tid.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    srcs = ["# Where every number comes from", "", f"Built {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; "
            f"{len(R.entries)} numbers.", "", "| where | value | source |", "|---|---|---|"]
    srcs += [f"| {w} | {v} | {s} |" for w, v, s, _n in R.entries]
    (out_dir / "numbers_sources.md").write_text("\n".join(srcs) + "\n", encoding="utf-8")
    try:
        import openpyxl  # noqa: F401
        import pandas as pd
        with pd.ExcelWriter(out_dir / "comparative_tables.xlsx") as xw:
            pd.DataFrame({"README": ["Generated by scripts/build_comparative_tables.py; one sheet per table; "
                                     "sources in numbers_sources.md; the document is docs/comparative_tables_2026-10.md."]}
                         ).to_excel(xw, sheet_name="README", index=False)
            for mk, t in tables:
                df = pd.DataFrame([[lab] + cells for lab, cells in t.rows], columns=[""] + t.columns)
                df.to_excel(xw, sheet_name=f"{mk}_{t.tid}".replace(" ", "")[:31].replace("(", "").replace(")", ""), index=False)
        xlsx = "written"
    except ImportError:
        xlsx = "pending (openpyxl is not installed — network step 3)"
    doc += f"\n\n---\n*Files: CSV per table, numbers_sources.md{', comparative_tables.xlsx' if xlsx == 'written' else ''} " \
           f"in `{rel(out_dir)}`. xlsx: {xlsx}.*\n"
    DOC.write_text(doc, encoding="utf-8")
    print(f"→ {rel(DOC)} ({len(doc)} chars, {len(R.entries)} sourced numbers), {rel(out_dir)}/ (xlsx {xlsx})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
