"""CAMeL Tools subprocess server for the AraRooPat tokenizer.

Runs INSIDE `.venv-camel` (the isolated env that holds camel-tools and
its incompatible numpy<2 / transformers<4.54 pins). The main process
(`.venv`) talks to this server over stdin/stdout NDJSON via the
`CamelBridge` client in `araroopat_bridge.py`.

This file MUST NOT be imported from the main `.venv` — `camel_tools`
isn't installed there, and importing this module would crash. The
client spawns it via subprocess only.

Wire format (one JSON object per line, both directions):

  client → server:
    {"id": <int>, "op": "analyze",  "words": ["wordA", "wordB", ...]}
    {"id": <int>, "op": "generate", "root": "...", "pattern": "..."}
    {"id": <int>, "op": "shutdown"}

  server → client:
    {"id": <int>, "ok": true,  "results": [[<analysis-dict>|null, ...], ...]}   # analyze
    {"id": <int>, "ok": true,  "result": "<stem>" | null}                       # generate
    {"id": <int>, "ok": true,  "result": "shutdown"}                            # shutdown
    {"id": <int>, "ok": false, "error": "<reason>"}                             # any op

Each `analyze` result is a list-of-candidate-analyses for that word
(MLE-disambiguated, in score order). Empty list ⇒ no analysis. The
client takes [0] to keep MLE semantics. Each candidate is a trimmed
dict containing only the ~11 fields araroopat consumes (root, pattern,
stem, diac, lex, pos, prc3, prc2, prc1, prc0, enc0). All values are
JSON-safe strings (or empty string for missing fields).

Fail-loud policy:
- Fatal init errors (missing DB, broken install) → print to stderr,
  exit non-zero. The client's `Popen` will see EOF on stdout.
- Per-request errors (analyzer raised on a single word) → return
  `{"ok": false, "error": "..."}` and continue. Client raises.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List, Optional

# These imports MUST succeed — if camel-tools isn't installed in this
# venv, fail loud at startup so the client sees an immediate EOF and
# raises, instead of silently producing useless output.
try:
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.generator import Generator
except Exception as e:  # noqa: BLE001 — startup must catch everything
    print(f"FATAL: camel-tools import failed: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)


# Fields the client actually consumes. We trim to keep wire size small
# and avoid serializing CAMeL-internal types.
_TRIMMED_FIELDS = (
    "root", "pattern", "stem", "diac", "lex", "pos",
    "prc3", "prc2", "prc1", "prc0", "enc0",
)


def _trim(analysis: Dict[str, Any]) -> Dict[str, str]:
    """Pick the JSON-safe subset of a CAMeL analysis dict."""
    return {k: (analysis.get(k) or "") for k in _TRIMMED_FIELDS}


def _init_backends() -> tuple:
    """Load analyzer/disambiguator/generator. Fail loud on error."""
    try:
        # Disambiguator wraps an Analyzer internally — no need for a
        # separate analyzer instance for the analyze op.
        disambig = MLEDisambiguator.pretrained()
        gen_db = MorphologyDB.builtin_db(flags="g")
        generator = Generator(gen_db)
        return disambig, generator
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: CAMeL backend init failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(3)


def _op_analyze(disambig, words: List[str]) -> List[List[Dict[str, str]]]:
    """Disambiguate a batch of words. One sublist per word, top-scored first.

    Empty sublist means no analysis. We pass the full list to
    `disambiguate(...)` in one call — CAMeL handles batching internally.
    """
    if not words:
        return []
    disambig_results = disambig.disambiguate(list(words))
    out: List[List[Dict[str, str]]] = []
    for dr in disambig_results:
        if not dr.analyses:
            out.append([])
            continue
        out.append([_trim(scored.analysis) for scored in dr.analyses])
    return out


def _op_generate(disambig, root: str, pattern: str) -> Optional[str]:
    """Tier-2 reconstruction: naive-fill, re-analyze, return matching stem.

    The client passes the bare-stem (clitic-stripped) pattern; we mirror
    the original `MorphAnalyzer._generate_cached` logic exactly so the
    behavior is unchanged from before the bridge.

    Returns None when no usable surface can be produced — the client
    falls back to its naive-substitution tier 3.
    """
    # Naive fill happens client-side too, but doing it here saves a
    # round-trip on the disambig call's input.
    naive = _naive_fill(root, pattern)
    if not naive:
        return None
    try:
        disambig_results = disambig.disambiguate([naive])
    except Exception:
        return naive  # the naive form is still usable as a fallback
    if not disambig_results or not disambig_results[0].analyses:
        return naive
    for scored in disambig_results[0].analyses:
        a = scored.analysis
        a_root = (a.get("root") or "").replace("_", "").replace(".", "").replace("#", "")
        a_pat_raw = a.get("pattern") or ""
        a_pat_bare = _normalize_pattern(
            a_pat_raw,
            a.get("prc3"), a.get("prc2"), a.get("prc1"), a.get("prc0"),
            a.get("enc0"),
        )
        if a_root == root and a_pat_bare == pattern:
            return a.get("stem") or naive
    return naive


# ---------------------------------------------------------------------------
# Client-side post-processing helpers, duplicated here so the server can
# do the bare-pattern match in `_op_generate` without an extra round-trip.
# Kept in sync with `araroopat_backend.py`.
# ---------------------------------------------------------------------------

_PATTERN_DIACRITICS = set("ًٌٍَُِّْٰٕٓٔ")

# Same table as `CAMEL_CLITIC_SURFACE` in `araroopat_backend.py`.
# Duplicated (not imported) because that module lives in the main venv,
# not here. Keep the two in sync if you add tags.
_CLITIC_SURFACE: Dict[str, str] = {
    "AAA_quest": "أ", "Aa_quest": "أ",
    "wa_conj": "و", "wa_part": "و", "wa_prep": "و", "wa_sub": "و",
    "fa_conj": "ف", "fa_rc": "ف", "fa_conn": "ف", "fa_sub": "ف",
    "fa_part": "ف",
    "bi_prep": "ب", "bi_part": "ب",
    "ka_prep": "ك",
    "li_prep": "ل", "li_jus": "ل", "li_sub": "ل",
    "sa_fut": "س",
    "ta_prep": "ت",
    "Al_det": "ال",
    "lA_neg": "لا",
    "mA_neg": "ما", "mA_part": "ما", "mA_rel": "ما", "ma_rel": "ما",
    "1s_dobj": "ي", "1s_poss": "ي", "1s_pron": "ي",
    "2ms_dobj": "ك", "2ms_poss": "ك", "2ms_pron": "ك",
    "2fs_dobj": "ك", "2fs_poss": "ك", "2fs_pron": "ك",
    "3ms_dobj": "ه", "3ms_poss": "ه", "3ms_pron": "ه",
    "3fs_dobj": "ها", "3fs_poss": "ها", "3fs_pron": "ها",
    "1p_dobj": "نا", "1p_poss": "نا", "1p_pron": "نا",
    "2mp_dobj": "كم", "2mp_poss": "كم", "2mp_pron": "كم",
    "2fp_dobj": "كن", "2fp_poss": "كن", "2fp_pron": "كن",
    "3mp_dobj": "هم", "3mp_poss": "هم", "3mp_pron": "هم",
    "3fp_dobj": "هن", "3fp_poss": "هن", "3fp_pron": "هن",
    "2d_dobj": "كما", "2d_poss": "كما", "2d_pron": "كما",
    "3d_dobj": "هما", "3d_poss": "هما", "3d_pron": "هما",
}


def _surface(tag: Optional[str]) -> str:
    if not tag or tag in ("0", "na"):
        return ""
    return _CLITIC_SURFACE.get(tag, tag)


def _strip_start(pat: str, clitic: str) -> str:
    if not clitic:
        return pat
    consumed, i = 0, 0
    while i < len(pat) and consumed < len(clitic):
        ch = pat[i]
        if ch in _PATTERN_DIACRITICS:
            i += 1
            continue
        if ch == clitic[consumed]:
            consumed += 1
            i += 1
        else:
            return pat
    if consumed != len(clitic):
        return pat
    while i < len(pat) and pat[i] in _PATTERN_DIACRITICS:
        i += 1
    return pat[i:]


def _strip_end(pat: str, clitic: str) -> str:
    if not clitic:
        return pat
    consumed, j = 0, len(pat)
    target = clitic[::-1]
    while j > 0 and consumed < len(target):
        ch = pat[j - 1]
        if ch in _PATTERN_DIACRITICS:
            j -= 1
            continue
        if ch == target[consumed]:
            consumed += 1
            j -= 1
        else:
            return pat
    if consumed != len(target):
        return pat
    while j > 0 and pat[j - 1] in _PATTERN_DIACRITICS:
        j -= 1
    return pat[:j]


def _normalize_pattern(
    pattern_raw: str,
    prc3_tag: Optional[str], prc2_tag: Optional[str],
    prc1_tag: Optional[str], prc0_tag: Optional[str],
    enc0_tag: Optional[str],
) -> str:
    pat = pattern_raw
    for tag in (prc3_tag, prc2_tag, prc1_tag, prc0_tag):
        s = _surface(tag)
        if s:
            pat = _strip_start(pat, s)
    enc_s = _surface(enc0_tag)
    if enc_s:
        pat = _strip_end(pat, enc_s)
    return pat


def _naive_fill(root: str, pattern: str) -> str:
    if not root or not pattern:
        return ""
    out: List[str] = []
    for ch in pattern:
        if ch in "1234":
            idx = int(ch) - 1
            if idx < len(root):
                out.append(root[idx])
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _send(payload: Dict[str, Any]) -> None:
    """Write one NDJSON line and flush. ensure_ascii=False keeps Arabic raw."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _serve(disambig, generator) -> int:  # noqa: ARG001 (generator unused for now)
    """Read NDJSON from stdin until EOF or shutdown. Return exit code."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"id": -1, "ok": False, "error": f"invalid JSON: {e}"})
            continue

        req_id = req.get("id", -1)
        op = req.get("op")
        try:
            if op == "analyze":
                results = _op_analyze(disambig, req.get("words") or [])
                _send({"id": req_id, "ok": True, "results": results})
            elif op == "generate":
                result = _op_generate(
                    disambig, req.get("root") or "", req.get("pattern") or ""
                )
                _send({"id": req_id, "ok": True, "result": result})
            elif op == "shutdown":
                _send({"id": req_id, "ok": True, "result": "shutdown"})
                return 0
            else:
                _send({"id": req_id, "ok": False, "error": f"unknown op: {op!r}"})
        except Exception as e:  # noqa: BLE001
            # Per-request error — keep the loop alive so a single bad
            # word doesn't kill a long sweep. Client decides whether to
            # raise (current policy: yes, fail-loud).
            _send({"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


def main() -> int:
    disambig, generator = _init_backends()
    # Signal "ready" to the client by emitting a banner line. The client
    # reads exactly one banner before sending the first request.
    _send({"id": 0, "ok": True, "result": "ready"})
    return _serve(disambig, generator)


if __name__ == "__main__":
    sys.exit(main())
