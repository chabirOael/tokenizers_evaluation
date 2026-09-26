"""Search the campaign record (``docs/report.md``) and the platform notes (``CLAUDE.md``).

The report is ~235 KB — too large to read whole; these return the paragraphs
that match, with the heading path they sit under. Use them for definitions,
rules and what a cell is; compute numbers from the data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from arabic_eval.analysis._root import repo_root

SOURCES = {"report": "docs/report.md", "claude": "CLAUDE.md"}
_WORD = re.compile(r"[\w؀-ۿ.+-]+", re.UNICODE)


@dataclass
class Passage:
    source: str
    heading: str
    text: str
    score: float = 0.0

    def __repr__(self) -> str:
        return f"[{self.source} › {self.heading}]\n{self.text}"


class Passages(list):
    def __repr__(self) -> str:
        if not self:
            return "(no match)"
        return "\n\n".join(repr(p) for p in self)


_PARSED: Dict[Tuple[str, float], List[Passage]] = {}


def _parse(source: str) -> List[Passage]:
    path = repo_root() / SOURCES[source]
    if not path.exists():
        return []
    key = (str(path), path.stat().st_mtime)
    if key in _PARSED:
        return _PARSED[key]
    out: List[Passage] = []
    stack: List[Tuple[int, str]] = []
    buf: List[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            out.append(Passage(source, " › ".join(h for _, h in stack) or "(top)", text))
        buf.clear()

    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, m.group(2).strip())]
        elif not line.strip() and not in_code:
            flush()
        else:
            buf.append(line)
    flush()
    _PARSED[key] = out
    return out


def report_search(query: str, k: int = 5, *, source: str = "both", max_chars: int = 1200) -> Passages:
    """The ``k`` paragraphs of the report (and/or CLAUDE.md) that best match ``query``.

    ``source``: "report", "claude" or "both". Scoring: occurrences of the query's
    words (case-insensitive), heading matches count double. Each passage is cut to
    ``max_chars``."""
    terms = [t.lower() for t in _WORD.findall(query) if len(t) > 1]
    if not terms:
        return Passages()
    srcs = list(SOURCES) if source == "both" else [source]
    scored: List[Passage] = []
    for s in srcs:
        for p in _parse(s):
            body, head = p.text.lower(), p.heading.lower()
            score = sum(body.count(t) + 2 * head.count(t) for t in terms)
            covered = sum(1 for t in terms if t in body or t in head)
            if covered:
                score *= covered / len(terms)
                text = p.text if len(p.text) <= max_chars else p.text[:max_chars] + " …"
                scored.append(Passage(p.source, p.heading, text, round(score, 2)))
    scored.sort(key=lambda p: -p.score)
    return Passages(scored[:k])


def report_section(title: str, *, source: str = "report", max_chars: int = 8000) -> str:
    """The text of the first section whose heading contains ``title`` (e.g. "3.11", "Letter or slot"),
    sub-sections included, cut to ``max_chars``."""
    path = repo_root() / SOURCES[source]
    lines = path.read_text(encoding="utf-8").splitlines()
    start, level = None, None
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        if start is None and title.lower() in m.group(2).lower():
            start, level = i, len(m.group(1))
        elif start is not None and len(m.group(1)) <= level:
            text = "\n".join(lines[start:i])
            return text if len(text) <= max_chars else text[:max_chars] + "\n… (cut; ask for a narrower section)"
    if start is None:
        heads = [l for l in lines if l.startswith("#")][:60]
        raise KeyError(f"no heading contains {title!r}; headings: {heads}")
    text = "\n".join(lines[start:])
    return text if len(text) <= max_chars else text[:max_chars] + "\n… (cut)"
