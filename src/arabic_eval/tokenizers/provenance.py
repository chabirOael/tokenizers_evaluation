"""Training provenance sidecar for saved tokenizers.

A saved tokenizer directory says nothing about the *text* it was trained
on. That mattered on 2026-09-16: the pipeline applied ``base.yaml``
preprocessing (alef folding included, at the time) while the
``train_tokenizer.py`` CLI applied none, so the same tokenizer type had a
different vocabulary depending on which script trained it, and nothing on
disk recorded which. Both entry points now write ``training_provenance.json``
next to the tokenizer files so the question "what text produced this
vocab?" is answerable without re-running anything.

The file is informational — ``load()`` never reads it.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

from arabic_eval.utils.io import save_json

PROVENANCE_FILENAME = "training_provenance.json"


def write_training_provenance(
    save_path: str | Path,
    *,
    dataset_name: str,
    preprocessing: Optional[Dict[str, Any]],
    num_texts: int,
    entry_point: str,
    tokenizer_type: Optional[str] = None,
    tokenizer_params: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``<save_path>/training_provenance.json`` and return its path.

    ``preprocessing`` is the resolved ``data.preprocessing`` dict (``None``
    or ``{}`` means the texts were used raw — recorded as such, not omitted).
    """
    record: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "preprocessing": dict(preprocessing) if preprocessing else {},
        "preprocessing_applied": bool(preprocessing),
        "num_texts": int(num_texts),
        "entry_point": entry_point,
        "tokenizer_type": tokenizer_type,
        "tokenizer_params": dict(tokenizer_params or {}),
        "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        record.update(extra)
    out = Path(save_path) / PROVENANCE_FILENAME
    save_json(record, out)
    return out


def read_training_provenance(save_path: str | Path) -> Optional[Dict[str, Any]]:
    """Return the sidecar's contents, or ``None`` when the directory has none."""
    p = Path(save_path) / PROVENANCE_FILENAME
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)
