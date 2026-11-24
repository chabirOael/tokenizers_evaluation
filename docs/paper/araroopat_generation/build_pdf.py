#!/usr/bin/env python3
"""Build a printable PDF of the introduction together with its reference list.

Reads ``introduction.md`` and ``references.bib`` from this folder, renders them to one HTML page (Latin text in
DejaVu Serif, Arabic in Amiri, every ``[TBD: …]`` placeholder shaded) and prints that page to ``introduction.pdf``
with the headless Chromium of the Playwright cache — this machine has neither LaTeX nor pandoc, and Chromium shapes
Arabic (ligatures, diacritics) correctly.  The reference list is formatted from the .bib file in an author–year
style close to ACL's, sorted by first author.

    .venv/bin/python docs/paper/araroopat_generation/build_pdf.py            # writes introduction.pdf next to the sources
    .venv/bin/python docs/paper/araroopat_generation/build_pdf.py --html-out /tmp/x.html   # keep the HTML too

Fonts: Amiri from ``~/.local/share/fonts/amiri/`` (Google Fonts copy, OFL); DejaVu from the system.
Chromium: ``~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`` (needs libgbm.so.1 — the script adds
``~/.local/lib/chromium-deps/extract/usr/lib/x86_64-linux-gnu`` to ``LD_LIBRARY_PATH`` when that folder exists).
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
FONT_DIR = HOME / ".local/share/fonts/amiri"
DEPS_DIR = HOME / ".local/lib/chromium-deps/extract/usr/lib/x86_64-linux-gnu"
CHROME_CANDIDATES = [
    *sorted(glob.glob(str(HOME / ".cache/ms-playwright/chromium-*/chrome-linux64/chrome")), reverse=True),
    *sorted(glob.glob(str(HOME / ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell")), reverse=True),
]

ARABIC = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
# A maximal Arabic run; hyphens *between* Arabic letters stay inside the run (a root written ق-ل-م), spaces and
# slashes do not (قال / يقول keeps the left-to-right order of its English gloss, as the LaTeX source does).
ARABIC_RUN = re.compile(rf"{ARABIC}+(?:-{ARABIC}+)*")

# ----------------------------------------------------------------------------------------------- markdown → HTML


def inline(text: str) -> str:
    """Escape, then mark up the inline features the introduction uses."""
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    t = ARABIC_RUN.sub(lambda m: f'<span class="ar">{m.group(0)}</span>', t)
    # AraRooPat tokens such as [ROOT_كتب] — keep the bracket, prefix and Arabic part together, left to right.
    t = re.sub(r"\[([A-Z]+(?:_[A-Z]+)*)_(<span class=\"ar\">[^<]+</span>)\]", r'<span class="tok">[\1_\2]</span>', t)
    t = re.sub(r"\[TBD(?::[^\]]*)?\]", lambda m: f'<span class="tbd">{m.group(0)}</span>', t)
    return t


def md_to_html(md: str) -> tuple[str, str]:
    """Return (title, body_html) for the subset of Markdown the introduction uses: #/## headings, paragraphs, - lists."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S).strip()
    title = ""
    parts: list[str] = []
    for block in re.split(r"\n\s*\n", md):
        lines = [ln.rstrip() for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("# ") and len(lines) == 1:
            title = lines[0][2:].strip()
            continue
        if lines[0].startswith("## ") and len(lines) == 1:
            parts.append(f"<h2>{inline(lines[0][3:].strip())}</h2>")
            continue
        if all(ln.startswith("- ") for ln in lines):
            items = "".join(f"<li>{inline(ln[2:].strip())}</li>" for ln in lines)
            parts.append(f"<ul>{items}</ul>")
            continue
        parts.append(f"<p>{inline(' '.join(lines))}</p>")
    return title, "\n".join(parts)


# ----------------------------------------------------------------------------------------------- .bib → HTML

ACCENTS = {"'": "\u0301", '"': "\u0308", "`": "\u0300", "^": "\u0302", "~": "\u0303", "=": "\u0304", ".": "\u0307",
           "v": "\u030C", "c": "\u0327", "H": "\u030B", "u": "\u0306"}


def strip_bib_comments(src: str) -> str:
    """Drop ``%`` comments that sit outside field values.

    Depth 0 is between entries, depth 1 is inside an entry but outside any ``{value}`` (where the file carries its
    ``% verify`` notes after some fields); a ``%`` inside a value (depth >= 2) is kept.
    """
    out = []
    depth = 0
    for line in src.splitlines():
        buf = []
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "%" and depth <= 1:
                break
            buf.append(ch)
        out.append("".join(buf))
    return "\n".join(out)


def parse_bib(src: str) -> list[dict]:
    src = strip_bib_comments(src)
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", src):
        etype, key = m.group(1).lower(), m.group(2)
        i = m.end()
        depth = 1
        fields: dict[str, str] = {}
        while i < len(src) and depth > 0:
            while i < len(src) and src[i] in " \t\r\n,":
                i += 1
            if i >= len(src) or src[i] == "}":
                break
            name_start = i
            while i < len(src) and src[i] not in "=":
                i += 1
            name = src[name_start:i].strip().lower()
            i += 1
            while i < len(src) and src[i] in " \t\r\n":
                i += 1
            if src[i] == "{":
                d = 1
                j = i + 1
                while j < len(src) and d > 0:
                    if src[j] == "{":
                        d += 1
                    elif src[j] == "}":
                        d -= 1
                    j += 1
                value = src[i + 1:j - 1]
                i = j
            elif src[i] == '"':
                j = src.index('"', i + 1)
                value = src[i + 1:j]
                i = j + 1
            else:
                j = i
                while j < len(src) and src[j] not in ",}\n":
                    j += 1
                value = src[i:j]
                i = j
            fields[name] = " ".join(value.split())
        entries.append({"type": etype, "key": key, **fields})
    return entries


def latex_to_html(value: str) -> str:
    t = html.escape(value, quote=False)
    t = re.sub(r"\\url\{([^{}]*)\}", r"<code>\1</code>", t)
    t = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", t)
    t = re.sub(r"\\(['\"`^~=.vcHu])\{?(\w)\}?",
               lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]), t)
    for cmd, ch in (("\\i", "ı"), ("\\l", "ł"), ("\\o", "ø"), ("\\ss", "ß"), ("\\ae", "æ")):
        t = re.sub(re.escape(cmd) + r"(?![A-Za-z])", ch, t)
    t = t.replace("---", "—").replace("--", "–").replace("~", "\u00a0")
    t = t.replace("{", "").replace("}", "")
    return t.strip()


def split_authors(field: str) -> list[tuple[str, str]]:
    """Return (last, first) per author; a corporate author ({Gemma Team}) is (name, ''); 'others' is ('et al.', '')."""
    out = []
    for raw in re.split(r"\s+and\s+", field.strip()):
        n = raw.strip()
        if n == "others":
            out.append(("et al.", ""))
        elif n.startswith("{") and n.endswith("}"):
            out.append((n[1:-1], ""))
        elif "," in n:
            last, first = (x.strip() for x in n.split(",", 1))
            out.append((last, first))
        else:
            toks = n.split()
            out.append((toks[-1], " ".join(toks[:-1])))
    return out


def format_authors(field: str) -> str:
    names = split_authors(field)
    rendered = []
    for i, (last, first) in enumerate(names):
        if last == "et al." or not first:
            rendered.append(last)
        elif i == 0:
            rendered.append(f"{last}, {first}")
        else:
            rendered.append(f"{first} {last}")
    if len(rendered) == 1:
        return rendered[0]
    if rendered[-1] == "et al.":
        return ", ".join(rendered)
    if len(rendered) == 2:
        return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", latex_to_html(s)) if not unicodedata.combining(c)).lower()


def sort_key(e: dict) -> tuple:
    """First author's surname, then first name, then year, then title — the usual bibliography order."""
    last, first = split_authors(e.get("author", ""))[0]
    return (_fold(last), _fold(first), e.get("year", ""), e.get("title", "").lower())


def format_entry(e: dict) -> str:
    authors = latex_to_html(format_authors(e["author"]))
    if not authors.endswith("."):          # "… et al." already ends in a period
        authors += "."
    bits = [f"{authors} {e.get('year', 'n.d.')}. {latex_to_html(e['title'])}."]
    if e["type"] == "inproceedings":
        venue = f"In <em>{latex_to_html(e['booktitle'])}</em>"
        if e.get("pages"):
            venue += f", pages {latex_to_html(e['pages'])}"
        if e.get("address"):
            venue += f", {latex_to_html(e['address'])}"
        venue += "."
        if e.get("publisher"):
            venue += f" {latex_to_html(e['publisher'])}."
        bits.append(venue)
        if e.get("eprint"):
            bits.append(f"arXiv:{e['eprint']}.")
    elif e["type"] == "article":
        venue = f"<em>{latex_to_html(e['journal'])}</em>"
        if e.get("volume"):
            venue += f", {latex_to_html(e['volume'])}"
            if e.get("number"):
                venue += f"({latex_to_html(e['number'])})"
        if e.get("pages"):
            venue += f":{latex_to_html(e['pages'])}"
        bits.append(venue + ".")
        if e.get("doi"):
            bits.append(f"doi:{e['doi']}.")
        if e.get("eprint"):
            bits.append(f"arXiv:{e['eprint']}.")
    else:
        if e.get("eprint"):
            bits.append(f"arXiv preprint arXiv:{e['eprint']}.")
        elif e.get("howpublished"):
            bits.append(latex_to_html(e["howpublished"]) + ".")
    if e.get("note"):     # the .bib's own notes (checkpoints, release status), set apart from the venue
        bits.append(f'<span class="bibnote">{latex_to_html(e["note"]).rstrip(".")}.</span>')
    return " ".join(bits)


def check_citations(entries: list[dict], md_text: str) -> list[str]:
    """Every entry's first author (or corporate name) must appear in the text within a few words of its year."""
    missing = []
    for e in entries:
        last, _ = split_authors(e["author"])[0]
        surname = latex_to_html(last)
        year = e.get("year", "")
        if not re.search(re.escape(surname) + r"[^()]{0,60}?\b" + re.escape(year) + r"\b", md_text):
            missing.append(f"{e['key']} ({surname}, {year})")
    return missing


# ----------------------------------------------------------------------------------------------- page

CSS = """
@font-face { font-family: "Amiri"; font-weight: normal; font-style: normal;
  src: local("Amiri"), url("file://FONT_DIR/Amiri-Regular.ttf"); }
@font-face { font-family: "Amiri"; font-weight: bold; font-style: normal;
  src: local("Amiri Bold"), url("file://FONT_DIR/Amiri-Bold.ttf"); }
@font-face { font-family: "Amiri"; font-weight: normal; font-style: italic;
  src: local("Amiri Italic"), url("file://FONT_DIR/Amiri-Italic.ttf"); }
@font-face { font-family: "Amiri"; font-weight: bold; font-style: italic;
  src: local("Amiri Bold Italic"), url("file://FONT_DIR/Amiri-BoldItalic.ttf"); }

@page { size: A4; margin: 22mm 20mm 24mm 20mm;
  @bottom-center { content: counter(page); font-family: "DejaVu Serif", serif; font-size: 9pt; color: #555; } }

html { font-size: 10.5pt; }
body { margin: 0; color: #111; font-family: "DejaVu Serif", serif; line-height: 1.45; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 6pt; }
.meta { font-size: 9.5pt; color: #555; margin: 0 0 16pt; padding-bottom: 8pt; border-bottom: 0.6pt solid #999; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; page-break-after: avoid; }
p { margin: 0 0 8pt; text-align: justify; hyphens: auto; -webkit-hyphens: auto; orphans: 2; widows: 2; }
ul { margin: 0 0 8pt; padding-left: 1.4em; }
li { margin: 0 0 4pt; text-align: justify; hyphens: auto; -webkit-hyphens: auto; }
em { font-style: italic; }
/* Arabic runs: Amiri at 1.2em (its Naskh body is small next to DejaVu), with line-height 1 so the larger
   inline box never stretches the line it sits in. */
.ar { font-family: "Amiri", "DejaVu Sans", sans-serif; font-size: 1.2em; line-height: 1; direction: rtl;
  unicode-bidi: isolate; }
.tok { font-family: "DejaVu Sans Mono", monospace; font-size: 0.88em; direction: ltr; unicode-bidi: isolate;
  white-space: nowrap; }
.tok .ar { font-size: 1.3em; }
.tbd { background: #e9e9e9; color: #444; font-style: italic; padding: 0 2pt; border-radius: 2pt; }
.bibnote { color: #555; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 0.86em; }
.refs p { text-align: left; font-size: 9.8pt; line-height: 1.38; margin: 0 0 5pt; padding-left: 1.6em;
  text-indent: -1.6em; hyphens: manual; -webkit-hyphens: manual; }
.note { font-size: 9pt; color: #555; margin-top: 10pt; }
"""


def build_html(md_text: str, bib_text: str) -> tuple[str, dict]:
    title, body = md_to_html(md_text)
    entries = sorted(parse_bib(bib_text), key=sort_key)
    refs = "\n".join(f"<p>{format_entry(e)}</p>" for e in entries)
    unverified = [e for e in entries if e["key"] in ("holtzman2020curious", "zheng2024large")]
    note = ""
    if unverified:
        labels = "; ".join(f"{latex_to_html(split_authors(e['author'])[0][0])} et al., {e['year']}" for e in unverified)
        note = (f'<p class="note">Bibliography note: every entry was checked against the ACL Anthology, arXiv or the '
                f'publisher\'s page (the first 24 on 8 November 2026, the 29 added on 10 November 2026) except {labels}, '
                f'whose OpenReview pages could not be fetched; their venues agree with the campaign report '
                f'(docs/report.md §7).</p>')
    has_tbd = "[TBD" in md_text
    meta = ("Introduction — draft v2 of 10 November 2026, for review, with the results of the tokenizer benchmark. "
            + ("Shaded <span class=\"tbd\">[TBD: …]</span> marks a result not yet available. " if has_tbd else "")
            + "Every number is listed with its source in <code>numbers_sources.md</code>; every reference was verified "
            "at its ACL Anthology, arXiv or publisher page.")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS.replace("FONT_DIR", str(FONT_DIR))}</style></head>
<body>
<h1>{inline(title)}</h1>
<p class="meta">{meta}</p>
{body}
<div class="refs">
<h2>References</h2>
{refs}
{note}
</div>
</body></html>
"""
    info = {"title": title, "entries": len(entries), "missing_citations": check_citations(entries, md_text),
            "words": len(re.sub(r"<!--.*?-->", "", md_text, flags=re.S).split())}
    return page, info


def print_pdf(html_path: Path, pdf_path: Path, chrome: str) -> None:
    env = dict(os.environ)
    if DEPS_DIR.is_dir():
        env["LD_LIBRARY_PATH"] = f"{DEPS_DIR}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
           "--allow-file-access-from-files", "--no-pdf-header-footer", "--virtual-time-budget=10000",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    if res.returncode != 0 or not pdf_path.exists():
        sys.exit(f"Chromium failed (exit {res.returncode}):\n{res.stderr[-2000:]}")


def count_pages(pdf_path: Path) -> int:
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?![s/])", data))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "introduction.pdf")
    ap.add_argument("--html-out", type=Path, default=None, help="also keep the intermediate HTML here")
    ap.add_argument("--chrome", default=next(iter(CHROME_CANDIDATES), None))
    args = ap.parse_args()
    if not args.chrome:
        sys.exit("No Chromium found under ~/.cache/ms-playwright; pass --chrome")

    md_text = (HERE / "introduction.md").read_text(encoding="utf-8")
    bib_text = (HERE / "references.bib").read_text(encoding="utf-8")
    page, info = build_html(md_text, bib_text)

    html_path = args.html_out or (Path(os.environ.get("TMPDIR", "/tmp")) / "araroopat_introduction.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    print_pdf(html_path, args.out, args.chrome)

    print(f"title            : {info['title']}")
    print(f"introduction     : {info['words']} words (markdown, title and comment excluded)")
    print(f"references       : {info['entries']} entries from references.bib")
    if info["missing_citations"]:
        print(f"NOT CITED IN TEXT: {', '.join(info['missing_citations'])}")
    else:
        print("citation check   : every entry's first author and year appear in the text")
    print(f"pdf              : {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {count_pages(args.out)} pages)")
    print(f"html             : {html_path}")


if __name__ == "__main__":
    main()
