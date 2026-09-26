"""Which numbers of an Analysis-tab answer did no step of the session print?

The agent is told that every number in its final answer must come from an
output of the session. ``check_numbers`` holds it to that, heuristically: it
extracts the numbers of the answer (not identifiers like ``v5`` / ``bpe_16k``,
dates, section numbers ``§3.11`` or small counts ≤ 10) and looks each one up
among the numbers of the evidence (step outputs, tables, figure data, code,
the prompt), allowing the rounding the answer's own precision implies, the
fraction ↔ percent conversion and a dropped sign (``0.07 below`` from −0.0701).
Anything left is returned for the page to highlight — flagged, never blocked:
a derived figure the model computed in its head is exactly what should be
flagged, and a false alarm costs a glance.
"""
from __future__ import annotations

import base64
import bisect
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_MINUS = "−"
_THIN = "   "

#: A number in prose: optional sign, digits with optional space/comma thousands groups, decimals, percent.
_ANSWER_NUM = re.compile(
    r"(?<![\w.\-/#§])"
    r"(?P<sign>[+\-−])?"
    r"(?P<int>\d{1,3}(?:[,    ]\d{3})+|\d+)"
    r"(?P<frac>\.\d+)?"
    r"(?P<pct>\s?%)?"
    r"(?![\w.]*[A-Za-z_])"                     # not the head of an identifier (16k, 4096_x)
    r"(?!\.\d)"
)
_EVIDENCE_NUM = re.compile(r"[+\-−]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+\-]?\d+)?")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_SECTION = re.compile(r"§\s?\d+(?:\.\d+)*")


def _clean_answer(text: str) -> str:
    """Blank out what is not prose (code, inline code, dates, § refs) keeping offsets."""
    def blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())
    for rx in (_CODE_FENCE, _INLINE_CODE, _DATE, _SECTION):
        text = rx.sub(blank, text)
    return text.translate(_ARABIC_DIGITS)


def answer_numbers(text: str) -> List[Dict[str, Any]]:
    """The numbers of an answer worth checking, with their offsets in ``text``."""
    out: List[Dict[str, Any]] = []
    clean = _clean_answer(text)
    for m in _ANSWER_NUM.finditer(clean):
        raw_int = m.group("int")
        frac = m.group("frac") or ""
        digits = re.sub(r"[,\s" + _THIN + "]", "", raw_int)
        # "3 114" is only a thousands group when it is not two separate small numbers
        if not frac and not m.group("pct") and len(digits) <= 2:
            value = int(digits)
            if value <= 10:
                continue
        try:
            value = float(digits + frac)
        except ValueError:
            continue
        if not frac and not m.group("pct") and value <= 10:
            continue                              # counts, k-shot, step numbers, the 1–5 scale
        if not frac and 1990 <= value <= 2100:
            continue                              # years
        sign = m.group("sign")
        if sign in ("-", _MINUS):
            value = -value
        out.append({"text": m.group(0).strip(), "start": m.start(), "end": m.end(), "value": value,
                    "decimals": len(frac) - 1 if frac else 0, "percent": bool(m.group("pct"))})
    return out


def evidence_numbers(texts: Iterable[str]) -> List[float]:
    vals: set = set()
    for t in texts:
        if not t:
            continue
        t = t.translate(_ARABIC_DIGITS).replace(_MINUS, "-")
        for m in _EVIDENCE_NUM.finditer(t):
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            if math.isfinite(v):
                vals.add(v)
                vals.add(abs(v))
    return sorted(vals)


def plotly_numbers(path: Path, limit: int = 20000) -> List[float]:
    """The numeric data of a saved Plotly figure (plain arrays and Plotly 6's base64 typed arrays)."""
    try:
        fig = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    import numpy as np
    out: List[float] = []

    def walk(node: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            if "bdata" in node and "dtype" in node:
                try:
                    arr = np.frombuffer(base64.b64decode(node["bdata"]), dtype=np.dtype(node["dtype"]))
                    out.extend(float(x) for x in arr[: limit - len(out)] if np.isfinite(x))
                except (ValueError, TypeError):
                    pass
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and math.isfinite(node):
            out.append(float(node))

    walk(fig.get("data"))
    return out


def _supported(n: Dict[str, Any], ev: Sequence[float]) -> bool:
    x = n["value"]
    tol = 0.5 * 10 ** (-n["decimals"]) + 1e-9
    candidates = [(x, tol), (abs(x), tol)]
    # 55.1 % / "55.1 points" from a printed 0.551 — only for a percent or a value that can be one
    if n["percent"] or 1.0 < abs(x) <= 100.0:
        candidates += [(x / 100.0, tol / 100.0), (abs(x) / 100.0, tol / 100.0)]
    # 0.551 from a printed 55.1 % — only for a fraction (2.5 must not match a printed 250)
    if not n["percent"] and abs(x) <= 1.0:
        candidates += [(x * 100.0, tol * 100.0), (abs(x) * 100.0, tol * 100.0)]
    for target, t in candidates:
        i = bisect.bisect_left(ev, target - t)
        if i < len(ev) and ev[i] <= target + t:
            return True
    return False


def check_numbers(answer: str, evidence_texts: Iterable[str], extra_numbers: Optional[Iterable[float]] = None) -> Dict[str, Any]:
    """``{"checked": n, "unverified": [{text, start, end, value}]}`` for an answer against its evidence."""
    nums = answer_numbers(answer)
    ev = evidence_numbers(evidence_texts)
    if extra_numbers:
        ev = sorted(set(ev) | {float(v) for v in extra_numbers} | {abs(float(v)) for v in extra_numbers})
    unverified = [{k: n[k] for k in ("text", "start", "end", "value")} for n in nums if not _supported(n, ev)]
    return {"checked": len(nums), "unverified": unverified}
