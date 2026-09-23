"""Training-prompt selection for the teacher (main venv): the seeded subset rule and
the held-out overlap check both prompt dumps run (``scripts/distill/dump_bakeoff_prompts.py``,
``scripts/distill/dump_teacher_prompts.py``).

Rule 2 of the distillation brief — the teacher never sees the test: every prompt file
is checked against the 250 held-out CIDAR prompts by id and by normalized text
(``contamination.normalize_text``, the one normalization every contamination test
uses) and must have zero hits. The committed exclusions already drop those prompts
from ``cidar/train`` by exact hash; the check makes the property explicit per file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from arabic_eval.data.contamination import normalize_text
from arabic_eval.data.freeform_heldout import load_freeform_heldout
from arabic_eval.distill.teacher import file_sha256, user_turn

SEED = 42


def seeded_subset(records: Sequence[Any], n: Optional[int], seed: int = SEED) -> List[Any]:
    """Sort by id, ``numpy.random.default_rng(seed).permutation``, the first ``n``
    (every record, permuted, when ``n`` is None). Extending ``n`` later keeps the
    earlier records: the first ``n`` of a larger ``n`` are the same ones."""
    import numpy as np
    ordered = sorted(records, key=lambda r: r.id)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    picked = [ordered[i] for i in perm]
    return picked if n is None else picked[:n]


def heldout_overlap(rows: Sequence[Dict[str, Any]], heldout_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Id overlap and normalized-text matches of ``rows`` (``id``, ``instruction``,
    ``context``) against the held-out prompts; ``passed`` when both are empty."""
    held = load_freeform_heldout(heldout_path)
    held_ids = {h["id"] for h in held}
    held_norm = {normalize_text(user_turn(h["prompt"], h.get("context") or "")) for h in held}
    held_norm |= {normalize_text(h["prompt"]) for h in held}
    id_hits = sorted(r["id"] for r in rows if r["id"] in held_ids)
    text_hits = sorted(r["id"] for r in rows
                       if normalize_text(r["instruction"]) in held_norm
                       or normalize_text(user_turn(r["instruction"], r.get("context") or "")) in held_norm)
    shown = heldout_path
    if repo_root is not None and heldout_path.is_absolute():
        try:
            shown = heldout_path.relative_to(repo_root)
        except ValueError:
            pass
    return {"heldout_file": str(shown), "heldout_sha256": file_sha256(heldout_path), "heldout_rows": len(held),
            "checked_rows": len(rows), "id_overlap": id_hits, "normalized_text_matches": text_hits,
            "passed": not id_hits and not text_hits}
