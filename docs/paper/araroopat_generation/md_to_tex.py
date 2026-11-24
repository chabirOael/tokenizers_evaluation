#!/usr/bin/env python3
"""Regenerate introduction.tex from introduction.md (same text, LaTeX conventions of the 2026-11-08 draft).

    .venv/bin/python docs/paper/araroopat_generation/md_to_tex.py            # writes introduction.tex
    .venv/bin/python docs/paper/araroopat_generation/md_to_tex.py --check    # exit 1 when a citation has no key

Conventions reproduced: \\textarabic{} for Arabic runs, \\tok{FAMILY}{arabic} for AraRooPat tokens, \\citep / \\citet
from the author–year strings of the Markdown (the table CITES below maps every "Name et al., Year" to its .bib key),
\\, in thousands, \\,\\% for percentages, signed numbers and intervals in math mode, LaTeX quotes and dashes, 120-column
wrapping. The .tex is generated: edit the .md and re-run.
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD, TEX, BIB = HERE / "introduction.md", HERE / "introduction.tex", HERE / "references.bib"

# in-text form (as written in the Markdown, without the year) -> bib key
CITES = {
    "Sennrich et al.": "sennrich-etal-2016-neural",
    "Kudo and Richardson": "kudo-richardson-2018-sentencepiece",
    "Mielke et al.": "mielke-2021-tokenization",
    "Rust et al.": "rust-etal-2021-good",
    "Petrov et al.": "petrov-2023-unfairness",
    "Hofmann et al.": "hofmann-2021-superbizarre",
    "Bostrom and Durrett": "bostrom-2020-bpe",
    "Park et al.": "park-2021-morphology",
    "Nzeyimana and Niyongabo Rubungo": "nzeyimana-2022-kinyabert",
    "Toraman et al.": "toraman-2023-tokenization",
    "Jabbar": "jabbar-2023-morphpiece",
    "Bauwens and Delobelle": "bauwens-2024-bpeknockout",
    "Antoun et al.": "antoun-2020-arabert",
    "Alyafeai et al., 2023": "alyafeai-2023-tokenizers",
    "Alyafeai et al., 2024": "alyafeai2024cidar",
    "Alrefaie et al., 2024": "alrefaie2024exploring",
    "Asgari et al., 2025": "asgari2025morphbpe",
    "Asgari et al., 2026": "asgari-2026-morphbpe",
    "Alakeel et al.": "alakeel2026morphemes",
    "Reda et al.": "reda2026yallamorph",
    "Arnett and Bergen": "arnett-2025-morphology",
    "Ali et al.": "ali-2024-tokenizer",
    "Dagan et al.": "dagan-2024-tokenizer",
    "Alqahtani et al.": "alqahtani-2026-tokenizers",
    "Goldman et al.": "goldman-2024-unpacking",
    "Schmidt et al.": "schmidt-2024-compression",
    "Batsuren et al.": "batsuren-2024-alien",
    "Di Marco and Fraser": "dimarco-2024-subword",
    "Libovický and Helcl": "libovicky-2024-lexically",
    "Minixhofer et al.": "minixhofer-etal-2022-wechsel",
    "Dobler and de Melo": "dobler-de-melo-2023-focus",
    "Mundra et al.": "mundra2024empirical",
    "Li et al.": "li-2025-tokalign",
    "Goddard and Fernandes Neto": "goddard-2025-omp",
    "Andriushchenko et al.": "andriushchenko-2025-tokenizer-russian",
    "Bari et al.": "bari-2025-allam",
    "Zhu et al.": "zhu-2025-arallama",
    "Kashirskiy et al.": "kashirskiy-2025-aratoken",
    "Zheng et al., 2023": "zheng2023judging",
    "Zheng et al., 2024": "zheng2024large",
    "El Filali et al.": "elfilali-2024-aragen",
    "Sanz-Guerrero et al.": "sanz-guerrero-2025-mindthegap",
    "Balepur et al.": "balepur-2025-mcqa",
    "Lyu et al.": "lyu-2024-beyondprobabilities",
    "Wei et al.": "wei-2024-rethinking",
    "Myrzakhan et al.": "myrzakhan-2024-openllm",
    "Chandak et al.": "chandak-2025-answermatching",
    "Altıntaş et al.": "altintas-2025-toksuite",
    "Lotz et al.": "lotz-2025-beyondcompression",
    "Obeid et al.": "obeid-etal-2020-camel",
    "Dang et al.": "dang2024ayaexpanse",
    "Kim and Rush": "kim-rush-2016-sequence",
    "Yang et al.": "yang2025qwen3",
    "Grattafiori et al.": "grattafiori2024llama3",
    "Gemma Team": "gemmateam2025gemma3",
    "Holtzman et al.": "holtzman2020curious",
    "Xu et al.": "xu2022learning",
    "Anwar et al.": "anwar2026jais2",
    "Fanar Team": "fanarteam2025fanar",
    "Abdelali et al.": "abdelali-etal-2016-farasa",
    "Uzan et al.": "uzan-2024-greed",
}

ARABIC = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
ARABIC_RUN = re.compile(rf"{ARABIC}+(?:-{ARABIC}+)*")
CITE_ITEM = re.compile(r"^(?P<name>.+?),\s*(?P<year>\d{4}[a-z]?)$")


def key_for(name: str, year: str) -> str | None:
    return CITES.get(f"{name}, {year}") or CITES.get(name)


missing: list[str] = []


def cite_group(m: re.Match) -> str:
    """'(Sennrich et al., 2016; Kudo and Richardson, 2018)' -> \\citep{...}; a leading part without a year is a note."""
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(";")]
    keys, note = [], None
    for p in parts:
        cm = CITE_ITEM.match(p)
        if not cm:
            if keys:  # not a citation group after all
                return m.group(0)
            note = p
            continue
        name, years = cm.group("name"), [cm.group("year")]
        two = CITE_ITEM.match(name)          # "Asgari et al., 2025, 2026" -> two keys
        if two and two.group("year").isdigit():
            name, years = two.group("name"), [two.group("year")] + years
        for y in years:
            k = key_for(name, y)
            if k is None:
                missing.append(p)
                return m.group(0)
            keys.append(k)
    if not keys:
        return m.group(0)
    if note:
        return f"\\citep[{note};][]{{{','.join(keys)}}}"
    return f"\\citep{{{','.join(keys)}}}"


def cite_textual(m: re.Match) -> str:
    """'Alrefaie et al. (2024)' -> \\citet{key}."""
    name, year = m.group(1), m.group(2)
    k = key_for(name, year)
    if k is None:
        missing.append(f"{name} ({year})")
        return m.group(0)
    return f"\\citet{{{k}}}"


def convert_inline(t: str) -> str:
    # 1. AraRooPat tokens -> placeholders, before any Arabic wrapping
    toks: list[str] = []

    def tok(m):
        toks.append(f"\\tok{{{m.group(1)}}}{{{m.group(2)}}}")
        return f"\x00{len(toks) - 1}\x00"

    t = re.sub(rf"\[([A-Z]+)_({ARABIC}+)\]", tok, t)
    # 2. citations
    t = re.sub(r"\(([^()]*\b\d{4}[a-z]?(?:;[^()]*)?)\)", cite_group, t)
    t = re.sub(r"\b([A-Z][\w'’.\- ]*?(?: et al\.| and [A-Z][\w'’.\- ]+?)?) \((\d{4}[a-z]?)\)", cite_textual, t)
    # 3. Arabic runs
    t = ARABIC_RUN.sub(lambda m: f"\\textarabic{{{m.group(0)}}}", t)
    # 4. markup
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"\*(.+?)\*", r"\\textit{\1}", t)
    t = re.sub(r"`([^`]+)`", r"\\texttt{\1}", t)
    # 5. typography
    t = t.replace("‘", "`").replace("’", "'").replace("“", "``").replace("”", "''")
    t = t.replace("—", "---").replace("–", "--").replace("…", "\\ldots{}")
    t = t.replace("κ", "$\\kappa$").replace("×", "$\\times$").replace("≤", "$\\le$").replace("≥", "$\\ge$").replace("±", "$\\pm$")
    t = re.sub(r"(\d) %", r"\1\\,\\%", t)
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", r"\\,", t)          # 16 000 -> 16\,000 (also 150 272, 2 400)
    t = re.sub(r"(?<![\w$])([+−-])(\d+\.\d+)", lambda m: f"${'-' if m.group(1) == '−' else m.group(1)}{m.group(2)}$", t)
    t = t.replace("%", "\\%") if "\\%" not in t else re.sub(r"(?<!\\)%", r"\\%", t)
    t = t.replace("&", "\\&")
    # 6. restore tokens
    t = re.sub(r"\x00(\d+)\x00", lambda m: toks[int(m.group(1))], t)
    return t


def wrap(s: str, indent: str = "") -> str:
    return "\n".join(textwrap.wrap(s, width=120, initial_indent=indent, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False))


def convert(md: str) -> str:
    out = []
    comment = re.search(r"<!--(.*?)-->", md, re.S)
    if comment:
        out.append("\n".join("% " + ln.strip() for ln in comment.group(1).strip().splitlines()))
    out.append("% Generated from introduction.md by md_to_tex.py — edit the Markdown, not this file.")
    out.append("% Requirements of the main file (not set here): natbib (ACL style), XeLaTeX/LuaLaTeX with polyglossia and an Arabic")
    out.append("% font, e.g.  \\usepackage{polyglossia} \\setotherlanguage{arabic} \\newfontfamily\\arabicfont[Script=Arabic]{Amiri}")
    out.append("% so that \\textarabic{...} is defined. \\tok below prints an AraRooPat token such as [ROOT_كتب].")
    out.append("\\providecommand{\\tok}[2]{\\texttt{[#1\\_}\\textarabic{#2}\\texttt{]}}\n")
    body = re.sub(r"<!--.*?-->", "", md, flags=re.S).strip()
    for block in re.split(r"\n\s*\n", body):
        lines = [ln.rstrip() for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("# "):
            continue  # the paper title lives in the main file
        if lines[0].startswith("## "):
            out.append("\\section{Introduction}\n\\label{sec:intro}\n")
            continue
        if all(ln.startswith("- ") for ln in lines):
            out.append("\\begin{itemize}")
            for ln in lines:
                out.append(wrap("\\item " + convert_inline(ln[2:].strip()), "  "))
            out.append("\\end{itemize}\n")
            continue
        out.append(wrap(convert_inline(" ".join(lines))) + "\n")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    tex = convert(MD.read_text(encoding="utf-8"))
    keys_used = set(re.findall(r"\\cite[pt](?:\[[^\]]*\]\[[^\]]*\])?\{([^}]*)\}", tex))
    used = {k for group in keys_used for k in group.split(",")}
    bib_keys = set(re.findall(r"^@\w+\{([^,\s]+),", BIB.read_text(encoding="utf-8"), re.M))
    unknown = sorted(used - bib_keys)
    uncited = sorted(bib_keys - used)
    if "--check" in sys.argv:
        print("citations without a key:", missing or "none")
        print("keys missing from references.bib:", unknown or "none")
        print("bib entries not cited:", uncited or "none")
        sys.exit(1 if (missing or unknown) else 0)
    TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {TEX} ({len(tex.splitlines())} lines, {len(used)} keys cited)")
    if missing:
        print("WARNING citations without a key:", missing)
    if unknown:
        print("WARNING keys missing from references.bib:", unknown)
    if uncited:
        print("note: bib entries not cited:", uncited)


if __name__ == "__main__":
    main()
