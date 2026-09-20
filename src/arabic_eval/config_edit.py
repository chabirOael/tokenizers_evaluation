"""Text-level edits of experiment YAMLs that keep the file's own bytes.

An experiment config is written by hand as often as by the console, and its
comments explain the choices. Every edit the tooling makes to such a file —
renaming a clone, stamping ``created_at`` when a file is first written,
appending a ``runs`` entry at every start — therefore rewrites only the lines
it owns inside the ``experiment:`` block (or the same keys at the top level)
and leaves everything else untouched: comments, key order, quoting style.

The rewriters return ``None`` when a file's layout is not one they can edit
safely (a flow-mapped ``experiment: {…}``, a block scalar, a ``runs`` written
inline). Callers then either render a fresh copy (clone) or skip the
bookkeeping with a warning (run stamps) — never guess.

Every result is verified by re-parsing before it is returned.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from arabic_eval.config import EXPERIMENT_KEYS

log = logging.getLogger(__name__)

OUTPUT_DIR_TEMPLATE = "outputs/experiments/{name}"
SCALAR_KEYS = ("name", "output_dir", "description", "created_at")
RUNS_KEY = "runs"
_RE_EXP_BLOCK = re.compile(r"^experiment:\s*(#.*)?$")
_RE_KEY_LINE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>" + "|".join(EXPERIMENT_KEYS) + r"):(?P<rest>.*)$")
_RE_BARE_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
_YAML_RESERVED = frozenset({"null", "true", "false", "yes", "no", "on", "off", "~", ""})
DQ = '"'


def default_output_dir(name: str) -> str:
    """The convention every config in the repo follows unless it is a cell
    folder of a larger campaign: ``outputs/experiments/<name>``."""
    return OUTPUT_DIR_TEMPLATE.format(name=name)


def now_iso() -> str:
    """Local time with its UTC offset, seconds precision — the format of every
    ``created_at`` / ``started_at`` stamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def is_repo_experiment_config(path: Path, repo_root: Path) -> bool:
    """True for a file inside ``configs/experiments`` of *repo_root* — the
    only files that receive run stamps (a run started from a snapshot under
    ``outputs/runs`` is stamped on its source by the console instead)."""
    try:
        return Path(path).resolve().parent == (Path(repo_root).resolve() / "configs" / "experiments")
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

def _split_scalar_comment(rest: str) -> Optional[Tuple[str, str, str]]:
    """Split the text after ``key:`` into ``(leading_ws, scalar, trailing)``
    where *trailing* is the ``  # comment`` (or empty). ``None`` for a shape
    the rewriter does not handle (block scalars ``|`` / ``>``, flow nodes,
    anchors) — the caller then falls back."""
    lead = rest[: len(rest) - len(rest.lstrip())]
    body = rest.strip()
    if body.startswith(("|", ">", "{", "[", "&", "*", "!")):
        return None
    if body.startswith('"'):
        i, n = 1, len(body)
        while i < n:
            if body[i] == "\\":
                i += 2
                continue
            if body[i] == '"':
                break
            i += 1
        if i >= n:
            return None
        return lead, body[: i + 1], body[i + 1:]
    if body.startswith("'"):
        i, n = 1, len(body)
        while i < n:
            if body[i] == "'":
                if i + 1 < n and body[i + 1] == "'":
                    i += 2
                    continue
                break
            i += 1
        if i >= n:
            return None
        return lead, body[: i + 1], body[i + 1:]
    m = re.search(r"\s+#", body)
    if m:
        return lead, body[: m.start()], body[m.start():]
    return lead, body, ""


def _render_scalar(value: str, like: str) -> str:
    """*value* in the quoting style of the scalar it replaces."""
    if like.startswith('"'):
        return json.dumps(value, ensure_ascii=False)
    if like.startswith("'"):
        return "'" + value.replace("'", "''") + "'"
    if _RE_BARE_SAFE.match(value) and value.lower() not in _YAML_RESERVED:
        return value
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Locating the experiment keys
# ---------------------------------------------------------------------------

def _indent_of(ln: str) -> int:
    return len(ln) - len(ln.lstrip())


def _locate(lines: List[str]) -> Optional[Tuple[int, int, str, Dict[str, Tuple[int, int]]]]:
    """``(start, end, indent, found)`` — the line range of the ``experiment:``
    block (or the whole file when the keys sit at the top level), the key
    indent, and for every experiment key present ``(line index, span)`` where
    *span* counts the continuation lines (more-indented, non-comment) that
    belong to it. ``None`` when there is no ``name`` line to anchor on."""
    start, end, indent = None, len(lines), ""
    for i, ln in enumerate(lines):
        if _RE_EXP_BLOCK.match(ln):
            start = i + 1
            break
    if start is not None:
        for j in range(start, len(lines)):
            ln = lines[j]
            if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith("#"):
                end = j
                break
        first = next((lines[j] for j in range(start, end) if lines[j].strip() and not lines[j].lstrip().startswith("#")), None)
        if first is None:
            return None
        indent = first[: len(first) - len(first.lstrip())]
    else:
        start = 0
    found: Dict[str, Tuple[int, int]] = {}
    for j in range(start, end):
        m = _RE_KEY_LINE.match(lines[j])
        if not (m and m.group("indent") == indent and m.group("key") not in found):
            continue
        span = 1
        while j + span < end and lines[j + span].strip() and _indent_of(lines[j + span]) > len(indent) \
                and not lines[j + span].lstrip().startswith("#"):
            span += 1
        found[m.group("key")] = (j, span)
    if "name" not in found:
        return None
    return start, end, indent, found


def _experiment_of(text: str) -> Optional[dict]:
    """The flattened experiment keys of *text* (``None`` when it does not parse)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    out = {k: v for k, v in data.items() if k in EXPERIMENT_KEYS}
    exp = data.get("experiment")
    if isinstance(exp, dict):
        for k in EXPERIMENT_KEYS:
            if k in exp:
                out.setdefault(k, exp[k])
    return out


# ---------------------------------------------------------------------------
# Rewriters
# ---------------------------------------------------------------------------

def rewrite_experiment_keys(text: str, updates: Dict[str, str]) -> Optional[str]:
    """Rewrite scalar experiment keys (``name`` / ``output_dir`` /
    ``description`` / ``created_at``) in a config's own text — inside its
    ``experiment:`` block when it has one, else at the top level — keeping
    every other byte (comments, order, quoting). A scalar that continues on
    more-indented lines is folded into the one new line; a key the file lacks
    is inserted below the block's own scalar keys. Returns ``None`` when the
    layout is not one of those two (flow mapping, block scalar, no ``name``
    line) or the result does not re-parse to the requested values."""
    unknown = set(updates) - set(SCALAR_KEYS)
    if unknown:
        raise ValueError(f"not scalar experiment keys: {sorted(unknown)}")
    lines = text.split("\n")
    loc = _locate(lines)
    if loc is None:
        return None
    _, _, indent, found = loc
    replace: Dict[int, str] = {}
    skip: set = set()
    for key in SCALAR_KEYS:
        if key not in updates or key not in found:
            continue
        j, span = found[key]
        m = _RE_KEY_LINE.match(lines[j])
        rest = " ".join([m.group("rest")] + [lines[j + k].strip() for k in range(1, span)])
        parts = _split_scalar_comment(rest)
        if parts is None:
            return None
        lead, scalar, trailing = parts
        replace[j] = f"{indent}{key}:{lead or ' '}{_render_scalar(updates[key], scalar)}{trailing}"
        skip.update(range(j + 1, j + span))
    last = max(j + span - 1 for k, (j, span) in found.items() if k in SCALAR_KEYS)   # missing keys go right below
    out: List[str] = []
    for j, ln in enumerate(lines):
        if j in skip:
            continue
        out.append(replace.get(j, ln))
        if j == last:
            for key in SCALAR_KEYS:
                if key in updates and key not in found:
                    out.append(f"{indent}{key}: {_render_scalar(updates[key], DQ)}")
    result = "\n".join(out)
    exp = _experiment_of(result)
    if exp is None or any(exp.get(k) != v for k, v in updates.items()):
        return None
    return result


def get_created_at(text: str) -> Optional[str]:
    exp = _experiment_of(text)
    v = exp.get("created_at") if exp else None
    return str(v) if v else None


def stamp_created_at(text: str, when: Optional[str] = None) -> Optional[str]:
    """*text* with ``created_at`` set to *when* (default: now)."""
    return rewrite_experiment_keys(text, {"created_at": when or now_iso()})


def _render_run(entry: Dict[str, Any]) -> str:
    """One ``runs`` item as a single flow-mapped line body: ``{started_at: "…", run_id: "…", source: console}``."""
    parts = []
    for k, v in entry.items():
        if v is None:
            continue
        parts.append(f"{k}: {v if _RE_BARE_SAFE.match(str(v)) and str(v).lower() not in _YAML_RESERVED and not str(v)[0].isdigit() else json.dumps(str(v), ensure_ascii=False)}")
    return "{" + ", ".join(parts) + "}"


def _runs_extent(lines: List[str], j: int, end: int, indent: str) -> Optional[Tuple[int, Optional[int]]]:
    """For a ``runs:`` key line at *j*: ``(last line of its items, item indent)``
    — items may sit at the key's own indent (PyYAML's style) or deeper (ours);
    a comment between items is skipped, a blank line ends the list. ``None``
    when the key's own line carries anything but ``[]`` or a comment."""
    rest = _RE_KEY_LINE.match(lines[j]).group("rest").strip()
    rest_no_comment = "" if rest.startswith("#") else re.split(r"\s+#", rest)[0].strip()
    if rest_no_comment not in ("", "[]"):
        return None
    if rest_no_comment == "[]":
        return j, None
    last, item_indent, k = j, None, j + 1
    while k < end:
        ln = lines[k]
        if not ln.strip():
            break
        ind, stripped = _indent_of(ln), ln.lstrip()
        if stripped.startswith("- ") and ind >= len(indent) and (item_indent is None or ind == item_indent):
            item_indent, last = ind, k
        elif item_indent is not None and ind > item_indent and not stripped.startswith("#"):
            last = k                                           # a nested line of a block-mapped item
        elif item_indent is not None and stripped.startswith("#") and ind >= item_indent:
            pass                                               # a comment between items
        else:
            break
        k += 1
    return last, item_indent


def append_run(text: str, entry: Dict[str, Any]) -> Optional[str]:
    """*text* with *entry* appended to the experiment's ``runs`` list, as one
    block-sequence item on its own line. Creates the key (below the block's
    other keys) when absent, replaces an empty ``runs: []``, appends after
    the last item of a block sequence at the items' own indent. ``None`` for
    an inline non-empty ``runs: [...]`` or when the result does not re-parse
    with the entry last."""
    if "started_at" not in entry:
        raise ValueError("a run entry needs started_at")
    lines = text.split("\n")
    loc = _locate(lines)
    if loc is None:
        return None
    _, end, indent, found = loc
    body = _render_run(entry)
    out = list(lines)
    if RUNS_KEY in found:
        j = found[RUNS_KEY][0]
        ext = _runs_extent(lines, j, end, indent)
        if ext is None:
            return None
        last, item_indent = ext
        if item_indent is None:                                # `runs:` with nothing under it, or `runs: []`
            rest = _RE_KEY_LINE.match(lines[j]).group("rest").strip()
            comment = rest[2:].strip() if rest.startswith("[]") else rest
            out[j] = f"{indent}{RUNS_KEY}:" + (f"  {comment}" if comment else "")
            out.insert(j + 1, f"{indent}  - {body}")
        else:
            out.insert(last + 1, " " * item_indent + f"- {body}")
    else:
        last = max(j + span - 1 for j, span in found.values())
        out.insert(last + 1, f"{indent}  - {body}")
        out.insert(last + 1, f"{indent}{RUNS_KEY}:")
    result = "\n".join(out)
    exp = _experiment_of(result)
    runs = exp.get(RUNS_KEY) if exp else None
    if not isinstance(runs, list) or not runs or runs[-1] != {k: v for k, v in entry.items() if v is not None}:
        return None
    return result


def strip_runs(text: str) -> Optional[str]:
    """*text* without its ``runs`` key and items (a clone has not run).
    Unchanged when the key is absent or its list is empty; ``None`` when the
    layout cannot be located or the items cannot be delimited."""
    lines = text.split("\n")
    loc = _locate(lines)
    if loc is None:
        return None
    _, end, indent, found = loc
    if RUNS_KEY not in found:
        return text
    j = found[RUNS_KEY][0]
    ext = _runs_extent(lines, j, end, indent)
    if ext is None:
        return None
    if ext[1] is None:                       # `runs: []` / bare `runs:` — nothing to strip, keep the line (and its comment)
        return text
    result = "\n".join(lines[:j] + lines[ext[0] + 1:])
    exp = _experiment_of(result)
    if exp is None or exp.get(RUNS_KEY):
        return None
    return result


# ---------------------------------------------------------------------------
# File-level operations
# ---------------------------------------------------------------------------

def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def record_run_start(path: Path, *, source: str, run_id: Optional[str] = None,
                     when: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Append a run stamp to the config file at *path*; returns the entry, or
    ``None`` (with a warning) when the file's layout cannot take it. Never
    raises for a layout problem — bookkeeping must not stop a run."""
    path = Path(path)
    entry: Dict[str, Any] = {"started_at": when or now_iso()}
    if run_id:
        entry["run_id"] = run_id
    entry["source"] = source
    text = path.read_text(encoding="utf-8")
    new = append_run(text, entry)
    if new is None:
        log.warning("could not record the run start in %s (unrecognised experiment block layout)", path)
        return None
    _write_atomic(path, new)
    return entry
