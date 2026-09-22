"""Instrumented replay of ``AraRooPatTokenizer.train()`` for the local explorer.

Runs the *real* training code path on a small user-supplied corpus and
records every intermediate value — the NFKC normalization, the character
classification, the literal NDJSON lines exchanged with the CAMeL
subprocess, each validation sub-step of ``_dict_to_analysis``, the
frequency tables, the vocab assembly with its ID ranges and budget cuts,
the three passes of ``_build_reconstruction``, the provenance metadata,
and finally an encode → decode round-trip through the vocab just built.

Design rule: nothing here re-implements tokenizer logic. Every step calls
the real helper (``_extract_alpha_chunks``, ``_strip_clitic_from_start``,
``_build_vocab``, ...) and, where a step is *explained* by walking the
same sub-helpers the real code walks, the explained result is asserted
equal to the real result and the mismatch (if any) is reported in the
trace rather than hidden.

Consumed by ``debugger/serve_araroopat_explorer.py`` (JSON over HTTP) and
rendered by ``docs/araroopat_train_explorer.html``.
"""
from __future__ import annotations

import difflib
import inspect
import json
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from arabic_eval.tokenizers import araroopat as A
from arabic_eval.tokenizers import araroopat_backend as B
from arabic_eval.tokenizers.araroopat import (
    CHAR_INVENTORY,
    DIGIT_INVENTORY,
    PFX_CHAR,
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_DIGIT,
    PFX_PAT,
    PFX_FUNC,
    PFX_PREP,
    PFX_PUNCT,
    PFX_ROOT,
    PUNCT_INVENTORY,
    SFX,
    SPECIAL_TOKENS_ORDERED,
    TOK_LIT_BEGIN,
    TOK_LIT_END,
    TOK_PROP_BEGIN,
    TOK_PROP_END,
    AraRooPatTokenizer,
    _classify_char,
    _extract_alpha_chunks,
    _split_runs,
    _strip_clitic_surfaces,
    entry_realization,
    join_word,
    pick_realization,
)
from arabic_eval.tokenizers.araroopat_bridge import _resolve_camel_python
from arabic_eval.tokenizers.araroopat_backend import (
    CorpusEntry,
    MorphAnalyzer,
    _dict_to_analysis,
    _clitic_only_analysis,
    _particle_analysis,
    _proper_analysis,
    is_db_proper,
    PROPER_POS,
    canonical_particle,
    strip_fem_from_pattern,
    _is_arabic_root,
    _norm_clitic,
    _strip_clitic_from_end,
    _strip_clitic_from_start,
    clitic_surface,
    naive_pattern_fill,
    normalize_pattern,
    strip_proclitics_from_start,
)
from arabic_eval.tokenizers.utils.arabic_text import strip_diacritics

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Real defaults, echoed to the page so the user sees what the explorer
# overrides (min_*_freq default to 1 in the explorer; see the server).
REAL_DEFAULTS: Dict[str, Any] = {
    "max_roots": 10000,
    "max_patterns": 500,
    "min_root_freq": 2,
    "min_pattern_freq": 2,
    "use_diacritized_surface": False,
    "cache_corpus_analysis": True,
    "add_bos_eos": True,
}

_BATCH_SIZE = 256  # mirrors the literal in _corpus_prepass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def ref(obj: Any) -> str:
    """``path:line`` of a function/method, relative to the repo root."""
    try:
        src = inspect.getsourcefile(obj)
        _, line = inspect.getsourcelines(obj)
        return f"{Path(src).resolve().relative_to(_REPO_ROOT)}:{line}"
    except Exception:  # noqa: BLE001
        return getattr(obj, "__qualname__", repr(obj))


def _cp(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def _cp_name(ch: str) -> str:
    return unicodedata.name(ch, "?")


class _WireTap:
    """Record the literal NDJSON lines the bridge writes and reads.

    Wraps ``_send`` / ``_read_response`` on the shared ``CamelBridge``
    instance for the lifetime of the trace, so the page can show the
    exact bytes that crossed the subprocess boundary.
    """

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self.lines: List[Dict[str, Any]] = []
        self._orig_send = bridge._send
        self._orig_read = bridge._read_response

    def __enter__(self) -> "_WireTap":
        tap = self

        def send(payload: Dict[str, Any]) -> None:
            tap.lines.append({
                "dir": "→ server",
                "t": time.perf_counter(),
                "line": json.dumps(payload, ensure_ascii=False),
            })
            tap._orig_send(payload)

        def read(timeout_s: Optional[float] = None) -> Dict[str, Any]:
            resp = tap._orig_read(timeout_s)
            tap.lines.append({
                "dir": "← server",
                "t": time.perf_counter(),
                "line": json.dumps(resp, ensure_ascii=False),
            })
            return resp

        self.bridge._send = send
        self.bridge._read_response = read
        return self

    def __exit__(self, *exc: Any) -> None:
        self.bridge._send = self._orig_send
        self.bridge._read_response = self._orig_read

    def take(self) -> List[Dict[str, Any]]:
        out, self.lines = self.lines, []
        for i, rec in enumerate(out):
            rec["n"] = i
            rec.pop("t", None)
        return out


class _Trace:
    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.t0 = time.perf_counter()

    def step(self, sid: str, title: str, func: Any, data: Dict[str, Any],
             started: float, notes: Optional[List[str]] = None,
             wire: Optional[List[Dict[str, Any]]] = None) -> None:
        self.steps.append({
            "id": sid,
            "num": len(self.steps),
            "title": title,
            "ref": ref(func) if not isinstance(func, str) else func,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "notes": notes or [],
            "wire": wire or [],
            "data": data,
        })


def _split_key(t: str) -> str:
    """Token → family, mirroring the prefixes in araroopat.py."""
    if t in SPECIAL_TOKENS_ORDERED:
        return "special"
    if t in (TOK_LIT_BEGIN, TOK_LIT_END):
        return "lit"
    if t in (TOK_PROP_BEGIN, TOK_PROP_END):
        return "prop"
    for pfx, fam in ((PFX_CLITICP, "cliticp"), (PFX_CLITICE, "clitice"),
                     (PFX_PREP, "prep"), (PFX_FUNC, "func"),
                     (PFX_CHAR, "char"), (PFX_DIGIT, "digit"),
                     (PFX_PUNCT, "punct"), (PFX_ROOT, "root"), (PFX_PAT, "pat")):
        if t.startswith(pfx):
            return fam
    return "other"


def _inner(t: str) -> str:
    for pfx in (PFX_CLITICP, PFX_CLITICE, PFX_PREP, PFX_FUNC, PFX_CHAR, PFX_DIGIT, PFX_PUNCT, PFX_ROOT, PFX_PAT):
        if t.startswith(pfx):
            return t[len(pfx):-len(SFX)]
    return t


# ---------------------------------------------------------------------------
# Traced explanation of _dict_to_analysis (asserted against the real one)
# ---------------------------------------------------------------------------

def _trace_strip_start(pat: str, clitic: str) -> Dict[str, Any]:
    out = _strip_clitic_from_start(pat, clitic)
    return {"clitic": clitic, "before": pat, "after": out, "stripped": out != pat}


def _trace_proclitic_stack(text: str, proclitics: Tuple[Optional[str], ...]) -> Dict[str, Any]:
    """Walk ``strip_proclitics_from_start`` one clitic at a time."""
    ops: List[Dict[str, Any]] = []
    prev: Optional[str] = None
    cur = text
    for clitic in proclitics:
        if not clitic:
            continue
        op = _trace_strip_start(cur, clitic)
        op["contraction"] = False
        if op["after"] == cur and clitic == B._AL_DET and prev == B._LI_PREP:
            op2 = _trace_strip_start(cur, B._LI_PREP)
            op = {
                "clitic": clitic, "before": cur, "after": op2["after"],
                "stripped": op2["stripped"], "contraction": True,
                "note": "لِ + الـ contraction: the article's surface is a single ل "
                        "after the li preposition, so strip one lam instead of ال",
            }
        prev = clitic
        cur = op["after"]
        ops.append(op)
    real = strip_proclitics_from_start(text, proclitics)
    return {"input": text, "ops": ops, "output": cur, "matches_real": cur == real}


def _trace_enclitic_stack(text: str, enclitics: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Walk ``strip_enclitics_from_end`` one clitic at a time (outermost first).

    ``enclitics`` is in emission order (innermost first: ``(ة, ه)``); the real
    helper peels them in reverse, and the ة suffix strips as ت when a pronoun
    followed it (``مَرْكَبَته`` → ``مَرْكَبَت`` → ``مَرْكَب``). The result is
    asserted against the real helper by the callers.
    """
    ops: List[Dict[str, Any]] = []
    cur = text
    for clitic in reversed([c for c in enclitics if c]):
        before = cur
        note = None
        if clitic == B.TAA_MARBUTA:
            after = _strip_clitic_from_end(cur, B.TAA_MARBUTA)
            if after == cur:
                after = _strip_clitic_from_end(cur, B._TAA)
                if after != cur:
                    note = "ة is realized as ت before a pronoun — stripped as ت"
        else:
            after = _strip_clitic_from_end(cur, clitic)
        op = {"clitic": clitic, "before": before, "after": after, "stripped": after != before}
        if note:
            op["note"] = note
        ops.append(op)
        cur = after
    return ops


def _trace_dict_to_analysis(d: Dict[str, str], particles: frozenset,
                            word: Optional[str] = None,
                            func_words: frozenset = frozenset()) -> Dict[str, Any]:
    """Explain every gate of ``_dict_to_analysis`` for one candidate dict."""
    real = _dict_to_analysis(d, particles, word, func_words)
    t: Dict[str, Any] = {"candidate": d, "gates": [], "accepted": real is not None}

    def gate(name: str, ok: bool, detail: Dict[str, Any]) -> bool:
        t["gates"].append({"name": name, "ok": ok, **detail})
        return ok

    # Closed-class words short-circuit every other gate: no root to
    # validate, one [PREP_*] / [FUNC_*] token. The check is the inverse of
    # the decoder join (input == proclitics + join_particle_enclitic(p, enc0)),
    # modulo alef variants. Prepositions are tried first.
    lemma_bare = strip_diacritics(d.get("lex") or "")
    for gate_name, inventory, kind in (("preposition intercept", particles, "prep"),
                                       ("function-word intercept", func_words, "func")):
        if kind == "func" and not func_words:
            continue
        part = _particle_analysis(d, inventory, word, kind=kind)
        if gate(gate_name, part is not None, {
            "lemma": lemma_bare, "in_inventory": canonical_particle(lemma_bare, inventory) is not None,
            "particle": part.particle if part else None, "kind": kind,
            "proclitics": [c for c in (part.prc3, part.prc2, part.prc1, part.prc0) if c] if part else [],
            "enclitic": part.enc0 if part else None,
        }):
            t["analysis"] = {
                "root": "", "pattern": "", "pattern_raw": d.get("pattern") or "", "stem": "",
                "surface": word or d.get("diac") or "", "lemma": d.get("lex", ""), "pos": d.get("pos", ""),
                "prc3": part.prc3, "prc2": part.prc2, "prc1": part.prc1, "prc0": part.prc0,
                "enc0": part.enc0, "particle": part.particle, "particle_kind": kind, "fem": None,
            }
            t["matches_real"] = real is not None and all(
                getattr(real, k) == v for k, v in t["analysis"].items()
            )
            return t

    # Pronoun-hosted prepositions (له, بها): POS prep, lemma لِ/بِ/كَ, a
    # pronoun in enc0 — emitted as clitic tokens only, no root to validate.
    cw = _clitic_only_analysis(d, word)
    if gate("clitic-only word (proclitic + pronoun)", cw is not None, {
        "pos": d.get("pos", ""), "lemma": lemma_bare,
        "proclitics": [c for c in (cw.prc3, cw.prc2, cw.prc1, cw.prc0) if c] if cw else [],
        "enclitic": cw.enc0 if cw else None,
    }):
        t["analysis"] = {
            "root": "", "pattern": "", "pattern_raw": d.get("pattern") or "", "stem": "",
            "surface": word or d.get("diac") or "", "lemma": d.get("lex", ""), "pos": "prep",
            "prc3": cw.prc3, "prc2": cw.prc2, "prc1": cw.prc1, "prc0": None,
            "enc0": cw.enc0, "particle": None, "clitic_only": True, "fem": None,
        }
        t["matches_real"] = real is not None and all(
            getattr(real, k) == v for k, v in t["analysis"].items()
        )
        return t

    # Proper nouns: a *database* noun_prop reading is always accepted — with
    # its root when the gates below pass, as a rootless [PROP_*] word when
    # they fail. CAMeL's NOAN_PROP backoff stamps noun_prop on every unknown
    # word (root "O", pattern "backoff"); those are not names to us.
    proper = is_db_proper(d)
    if (d.get("pos") or "") == PROPER_POS:
        gate("proper noun: database noun_prop (not a backoff guess)", proper, {
            "pos": d.get("pos", ""), "root": d.get("root") or "", "pattern": d.get("pattern") or "",
            "backoff": not proper,
            "note": ("a failing root gate below does not reject the word: it becomes a rootless "
                     "proper noun ([PROP_BEGIN] chars [PROP_END])" if proper else
                     "backoff guess for an out-of-vocabulary word — treated like any unknown word"),
        })

    def reject(reason: str) -> Dict[str, Any]:
        if proper:
            pa = _proper_analysis(d, word)
            t["proper_fallback"] = reason
            t["analysis"] = {
                "root": "", "pattern": "", "pattern_raw": d.get("pattern") or "", "stem": "",
                "surface": pa.surface, "lemma": d.get("lex", ""), "pos": PROPER_POS,
                "prc3": pa.prc3, "prc2": pa.prc2, "prc1": pa.prc1, "prc0": pa.prc0, "enc0": pa.enc0,
                "particle": None, "fem": pa.fem, "proper": True,
            }
            t["matches_real"] = real is not None and all(
                getattr(real, k) == v for k, v in t["analysis"].items()
            )
        else:
            t["reject_reason"] = reason
            t["matches_real"] = real is None
        return t

    root_raw = d.get("root") or ""
    pattern_raw = d.get("pattern") or ""
    if not gate("has root & pattern", bool(root_raw and pattern_raw),
                {"root": root_raw, "pattern": pattern_raw}):
        return reject("missing root or pattern")

    radicals = [r for r in root_raw.replace("_", ".").split(".") if r]
    unseparated = False
    if len(radicals) == 1 and len(radicals[0]) >= 3:
        radicals = list(radicals[0])
        unseparated = True
    root = "".join(radicals)
    gate("split radicals (keep '#')", True, {
        "root_raw": root_raw, "radicals": radicals, "root": root,
        "weak_radicals": [i + 1 for i, r in enumerate(radicals) if r == B.WEAK_RADICAL_MARK],
        "unseparated_fallback": unseparated,
    })
    if not gate("≥ 3 radicals", len(radicals) >= 3, {"count": len(radicals)}):
        return reject(f"only {len(radicals)} radical(s)")

    is_ntws = root in ("NTWS", "FOREIGN") or "NTWS" in pattern_raw or "FOREIGN" in pattern_raw
    if not gate("not NTWS / FOREIGN", not is_ntws, {"root": root, "pattern": pattern_raw}):
        return reject("loanword / non-Arabic source (NTWS or FOREIGN)")

    if not gate("Arabic-letter root guard", _is_arabic_root(root),
                {"root": root, "codepoints": [_cp(c) for c in root]}):
        return reject("root contains non-Arabic characters")

    clitics: Dict[str, Dict[str, Any]] = {}
    for slot in ("prc3", "prc2", "prc1", "prc0", "enc0"):
        raw = d.get(slot)
        norm = _norm_clitic(raw)
        surf = clitic_surface(norm)
        clitics[slot] = {"tag": raw, "normalized": norm, "surface": surf}
    gate("clitic tag → surface", True, {"slots": clitics})

    prc = tuple(clitics[s]["surface"] for s in ("prc3", "prc2", "prc1", "prc0"))
    enc0 = clitics["enc0"]["surface"]
    pro = _trace_proclitic_stack(pattern_raw, prc)
    pat_after_pro = pro["output"]
    enc_op = None
    pat_bare = pat_after_pro
    if enc0:
        after = _strip_clitic_from_end(pat_after_pro, enc0)
        enc_op = {"clitic": enc0, "before": pat_after_pro, "after": after,
                  "stripped": after != pat_after_pro}
        pat_bare = after
    real_bare = normalize_pattern(pattern_raw, *prc, enc0)
    gate("normalize_pattern → bare-stem template", True, {
        "pattern_raw": pattern_raw, "proclitic_strip": pro, "enclitic_strip": enc_op,
        "pattern_bare": pat_bare, "matches_real": pat_bare == real_bare,
    })

    pat_with_fem = pat_bare
    pat_bare, fem = strip_fem_from_pattern(
        pat_bare, prc, enc0, d.get("stem") or "", d.get("pos") or "", d.get("diac") or "",
    )
    gate("ة suffix → [CLITICE_ة]", True, {
        "before": pat_with_fem, "after": pat_bare, "fem": fem,
        "realized_as": ("ت" if enc0 else "ة") if fem else None,
    })

    stem_from_db = d.get("stem", "") or ""
    stem = stem_from_db or naive_pattern_fill(root, pat_bare)
    gate("stem", True, {"from_db": bool(stem_from_db), "stem": stem,
                        "naive_fill_used": not stem_from_db})

    t["analysis"] = {
        "root": root, "pattern": pat_bare, "pattern_raw": pattern_raw, "stem": stem,
        "surface": d.get("diac") or "", "lemma": d.get("lex", ""), "pos": d.get("pos", ""),
        "prc3": prc[0], "prc2": prc[1], "prc1": prc[2], "prc0": prc[3], "enc0": enc0,
        "particle": None, "fem": fem, "proper": proper,
    }
    t["matches_real"] = real is not None and all(
        getattr(real, k) == v for k, v in t["analysis"].items()
    )
    return t


def trace_validate_words(
    unique_words: List[str], raw_by_word: Dict[str, List[Dict[str, str]]],
    backend: MorphAnalyzer,
) -> Tuple[Dict[str, Optional[B.Analysis]], List[Dict[str, Any]]]:
    """Step 1h: walk every candidate of every word through the traced gates.

    Returns ``(analyses, rows)`` and populates the backend's native /
    analyze caches exactly like ``analyze_many()`` would. Shared by the
    small-text tracer and the corpus tracer (which runs it on a sample).
    """
    analyses: Dict[str, Optional[B.Analysis]] = {}
    val_rows = []
    for w in unique_words:
        cands = raw_by_word[w]
        top = MorphAnalyzer._first_valid(cands, w, backend.particles, backend.func_words)
        # The spelling-faithful walk of _native_many (2026-09-22): a rooted top
        # reading that does not spell the word as written is checked against
        # every ranked candidate; failing that its ة/ه slot is reconciled.
        walk: Optional[Dict[str, Any]] = None
        real = top
        if B.needs_spelling_walk(top, w):
            more = backend._candidates_many([w])[0]
            real = B.prefer_faithful(top, more, w)
            faithful = next((c for c in more if B.is_rooted_analysis(c) and B.spelling_faithful(c, w)), None)
            walk = {
                "top_surface": top.surface, "candidates": len(more),
                "faithful_index": more.index(faithful) if faithful is not None else None,
                "outcome": ("faithful candidate" if faithful is not None else
                            "ة/ه slot reconciled" if real is not top else "kept top reading"),
                "chosen": {"surface": real.surface, "root": real.root, "pattern": real.pattern,
                           "fem": real.fem, "enc0": real.enc0},
            }
        backend._native_cache[w] = real   # same side effect as analyze_many()
        backend._analyze_cache[w] = real  # (overwritten below if the peeler rescues it)
        analyses[w] = real
        traced = []
        for ci, c in enumerate(cands):
            tc = _trace_dict_to_analysis(c, backend.particles, w, backend.func_words)
            tc["index"] = ci
            traced.append(tc)
            if tc["accepted"]:
                break
        val_rows.append({
            "word": w, "num_candidates": len(cands), "candidates": traced,
            "accepted_index": next((c["index"] for c in traced if c["accepted"]), None),
            "spelling_walk": walk,
            "analyzed": real is not None,
            "path": (("FUNC" if real.particle_kind == "func" else "PREP") if real is not None and real.particle else
                     "CLITIC" if real is not None and real.clitic_only else
                     "PROP" if real is not None and real.proper and not real.root else
                     "ROOT+PAT" if real is not None else "LIT (character fallback)"),
            "proper": real is not None and real.proper,
            "surface_fallback": real is not None and real.particle is not None and not any(
                c.get("accepted") for c in traced),
            "matches_real": all(c.get("matches_real", True) for c in traced),
        })
    return analyses, val_rows


def trace_peel_words(
    unique_words: List[str], analyses: Dict[str, Optional[B.Analysis]],
    backend: MorphAnalyzer,
) -> List[Dict[str, Any]]:
    """Step 1h': replay the clitic peeler on the words CAMeL rejected natively.

    Mutates ``analyses`` (and the backend analyze cache) with the peeled
    result, exactly like ``analyze_many()``. Words already analyzed are
    skipped; ``trace_validate_words`` must have run on them first.
    """
    peel_rows = []
    if backend.enable_peeler:
        for w in unique_words:
            if analyses[w] is not None:
                continue
            cands = B.peel_candidates(w, backend.peel_bare_alef)
            spellings: List[str] = []
            for c in cands:
                for sp in MorphAnalyzer._residual_spellings(c):
                    if sp not in spellings:
                        spellings.append(sp)
            # Every valid reading of each residual (server `top` > 1): for
            # an unseen word the MLE scores tie at 1.0, so rank 1 is just
            # database order and the reading we need may sit further down.
            readings = dict(zip(spellings, backend._candidates_many(spellings)))
            tried = []
            accepted = None
            for c in cands:
                for sp in MorphAnalyzer._residual_spellings(c):
                    reads = readings.get(sp, [])
                    if not reads:
                        tried.append({"proclitics": list(c.proclitics), "residual": sp,
                                      "enclitics": list(c.enclitics), "peeled_len": c.peeled_len,
                                      "residual_analyzed": False,
                                      "verdict": "residual has no analysis"})
                        continue
                    for rank, res in enumerate(reads):
                        row = {"proclitics": list(c.proclitics), "residual": sp,
                               "enclitics": list(c.enclitics), "peeled_len": c.peeled_len,
                               "residual_analyzed": True, "reading": rank + 1,
                               "of_readings": len(reads),
                               "residual_analysis": {"pos": res.pos, "surface": res.surface,
                                                     "proclitics": list(res.proclitics),
                                                     "enclitics": list(res.enclitics),
                                                     "root": res.root, "pattern": res.pattern}}
                        verdict = MorphAnalyzer.residual_verdict(c, sp, res)
                        row["verdict"] = verdict or "accepted"
                        tried.append(row)
                        if verdict is None:
                            accepted = B.merge_peeled(w, c, res)
                            break
                    if accepted is not None:
                        break
                if accepted is not None:
                    break
            real = backend._peel(w)
            backend._analyze_cache[w] = real
            analyses[w] = real
            matches = (real is None) == (accepted is None) and (
                real is None or (real.proclitics, real.root, real.pattern, real.enclitics)
                == (accepted.proclitics, accepted.root, accepted.pattern, accepted.enclitics))
            peel_rows.append({
                "word": w, "num_candidates": len(cands), "candidates": tried,
                "accepted": accepted is not None,
                "result": None if accepted is None else {
                    "proclitics": list(accepted.proclitics), "root": accepted.root,
                    "pattern": accepted.pattern, "enclitics": list(accepted.enclitics),
                    "surface": accepted.surface},
                "path": "ROOT+PAT (peeled)" if accepted is not None else "LIT (character fallback)",
                "matches_real": matches,
            })
    return peel_rows


def trace_probe_roundtrip(
    tr: "_Trace", tok: AraRooPatTokenizer, backend: MorphAnalyzer, tap: "_WireTap",
    texts: List[str], unique_words: Optional[List[str]],
    analyses: Optional[Dict[str, Optional[B.Analysis]]], real_reco: Dict[Tuple[int, int], str],
) -> None:
    """Step 6: encode → decode ``texts`` through the vocab just built (two steps).

    ``unique_words`` / ``analyses`` may be ``None``: they are then derived
    from the probe text and the backend's analyze cache after the encode
    call (which fills it). The small-text tracer passes both explicitly.
    """
    vocab = tok._vocab
    if unique_words is None:
        counts: Counter = Counter()
        for t_ in texts:
            for w in unicodedata.normalize("NFKC", t_).split():
                for chunk in _extract_alpha_chunks(w):
                    counts[chunk] += 1
        unique_words = list(counts.keys())
    # ---- Step 6: encode → decode round trip -------------------------
    t0 = time.perf_counter()
    full_text = "\n".join(texts)
    out = tok.encode(full_text)
    if analyses is None:
        analyses = {w: backend._analyze_cache.get(w) for w in unique_words}
    enc_lines = tap.take()
    stream = []
    for i, (tid, metric) in enumerate(zip(out.input_ids, out.tokens)):
        t_ = tok._reverse_vocab.get(tid, "?")
        stream.append({"pos": i, "id": tid, "token": t_, "family": _split_key(t_),
                       "inner": _inner(t_), "metric_string": metric})
    # Per-chunk path annotation from the analysis cache.
    chunk_paths = []
    for w in unique_words:
        a = analyses[w]
        if a is None:
            chunk_paths.append({"chunk": w, "path": "LIT", "why": "no valid analysis"})
            continue
        if a.particle:
            pfx = PFX_FUNC if a.particle_kind == "func" else PFX_PREP
            needed = [f"{pfx}{a.particle}{SFX}"] + \
                [f"{PFX_CLITICP}{c}{SFX}" for c in (a.prc3, a.prc2, a.prc1, a.prc0) if c] + \
                ([f"{PFX_CLITICE}{a.enc0}{SFX}"] if a.enc0 else [])
            missing = [t_ for t_ in needed if t_ not in vocab]
            chunk_paths.append({"chunk": w, "path": ("FUNC" if pfx == PFX_FUNC else "PREP") if not missing else "LIT",
                                "why": "" if not missing else f"{missing[0]} not in vocab"})
            continue
        if a.clitic_only:
            needed = [f"{PFX_CLITICP}{c}{SFX}" for c in (a.prc3, a.prc2, a.prc1, a.prc0) if c] + \
                ([f"{PFX_CLITICE}{a.enc0}{SFX}"] if a.enc0 else [])
            missing = [t_ for t_ in needed if t_ not in vocab]
            chunk_paths.append({"chunk": w, "path": "CLITIC" if not missing else "LIT",
                                "why": "" if not missing else f"{missing[0]} not in vocab"})
            continue
        if tok._routes_to_prop(a.proper, a.root):
            chunk_paths.append({"chunk": w, "path": "PROP",
                                "why": "proper noun" + ("" if a.root else " without a root (NTWS)")})
            continue
        rt, pt = f"{PFX_ROOT}{a.root}{SFX}", f"{PFX_PAT}{a.pattern}{SFX}"
        if rt in vocab and pt in vocab:
            chunk_paths.append({"chunk": w, "path": "ROOT+PAT", "why": ""})
        elif a.proper:
            chunk_paths.append({"chunk": w, "path": "PROP",
                                "why": f"{rt if rt not in vocab else pt} cut by the budget — a name keeps the [PROP_*] path"})
        else:
            chunk_paths.append({"chunk": w, "path": "LIT",
                                "why": f"{rt if rt not in vocab else pt} cut by the budget"})
    tr.step("encode", "encode(): the corpus through the vocab just built", A.AraRooPatTokenizer.encode, {
        "input": full_text, "num_tokens": len(out.input_ids), "stream": stream,
        "chunk_paths": chunk_paths, "ipc_calls": len([l for l in enc_lines if l["dir"] == "→ server"]),
    }, t0, wire=enc_lines, notes=[
        "Zero IPC calls expected: every alpha chunk was analyzed in the pre-pass and sits in "
        "MorphAnalyzer's cache. That is what the shared _classify_char chunking buys.",
        "metric_string is the cleaned Arabic surface the morphological metrics see for each token "
        "(root letters for [ROOT_*], the cleaned inflected stem for [PAT_*], the clitic-stripped "
        "surface for [PREP_*] / [FUNC_*] — علي for عليه, so the tokens still concatenate to the word).",
    ])

    t0 = time.perf_counter()
    decoded = tok.decode(out.input_ids)
    dec_lines = tap.take()
    tiers = []
    pending: Optional[Tuple[int, str]] = None
    for s in stream:
        if s["family"] == "root":
            pending = (s["id"], s["inner"])
        elif s["family"] == "pat" and pending is not None:
            key = (pending[0], s["id"])
            if key in real_reco:
                tiers.append({"root": pending[1], "pattern": s["inner"], "tier": 1, "value": real_reco[key]})
            else:
                gen = backend._generate_cache.get((pending[1], s["inner"]))
                tiers.append({"root": pending[1], "pattern": s["inner"], "tier": 2 if gen else 3,
                              "value": strip_diacritics(gen) if gen else
                              strip_diacritics(naive_pattern_fill(pending[1], s["inner"]))})
            pending = None
    src_words = unicodedata.normalize("NFKC", full_text).split()
    dec_words = decoded.split()
    src_nodiac = [strip_diacritics(w) for w in src_words]
    dec_nodiac = [strip_diacritics(w) for w in dec_words]
    # Align source vs decoded words (decode emits punctuation/digits as
    # separate units, so a positional zip would misalign everything).
    diff_ops = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src_nodiac, dec_nodiac, autojunk=False).get_opcodes():
        diff_ops.append({"op": op, "src": src_words[i1:i2], "dec": dec_words[j1:j2]})
    tr.step("decode", "decode(): three-tier reconstruction back to Arabic", A.AraRooPatTokenizer.decode, {
        "decoded": decoded, "tiers": tiers,
        "source_words": src_words, "decoded_words": dec_words,
        "exact_match": " ".join(src_words) == decoded,
        "match_ignoring_diacritics": " ".join(src_nodiac) == strip_diacritics(decoded),
        "match_ignoring_spacing": "".join(src_nodiac) == "".join(dec_nodiac),
        "word_diff": diff_ops,
        "length_mismatch": len(src_words) != len(dec_words),
    }, t0, wire=dec_lines, notes=[
        "Tier 1 = lookup table built in step 4 (O(1)); tier 2 = CAMeL generator; tier 3 = naive fill.",
        "Decode joins proclitics with join_proclitics — the inverse of strip_proclitics_from_start, "
        "including the لِ + الـ contraction.",
        "Punctuation and digits become separate whitespace-joined words, so exact_match is usually "
        "false on punctuated input even when every Arabic word round-trips.",
    ])



# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def trace_training(text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replay ``train()`` on ``text`` and return the full step trace as JSON-able dict."""
    return trace_training_with_tokenizer(text, params)[0]


def trace_training_with_tokenizer(
    text: str, params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], AraRooPatTokenizer]:
    """Like ``trace_training`` but also hands back the trained instance.

    The explorer server keeps that instance alive so the decode playground
    can run the real ``decode()`` on arbitrary id sequences afterwards.
    """
    params = dict(params or {})
    params.setdefault("min_root_freq", 1)
    params.setdefault("min_pattern_freq", 1)
    params["cache_corpus_analysis"] = False  # never touch the on-disk corpus cache

    tr = _Trace()
    tok = AraRooPatTokenizer(**params)
    backend: MorphAnalyzer = tok._ensure_backend()
    bridge = backend._bridge
    bridge._ensure_started()

    # Every line of the textarea is one "text" of the corpus.
    texts = [ln for ln in text.splitlines() if ln.strip()]
    if not texts:
        raise ValueError("Empty input.")

    # ---- Step 0: config ------------------------------------------------
    t0 = time.perf_counter()
    effective = {
        "max_roots": tok.max_roots, "max_patterns": tok.max_patterns,
        "min_root_freq": tok.min_root_freq, "min_pattern_freq": tok.min_pattern_freq,
        "use_diacritized_surface": tok.use_diacritized_surface,
        "cache_corpus_analysis": tok.cache_corpus_analysis, "add_bos_eos": tok.add_bos_eos,
    }
    tr.step("config", "Tokenizer instance & parameters", AraRooPatTokenizer.__init__, {
        "effective": effective, "real_defaults": REAL_DEFAULTS,
        "overridden": [k for k, v in effective.items() if REAL_DEFAULTS.get(k) != v],
        "num_texts": len(texts),
        "camel_python": str(bridge._camel_python or _resolve_camel_python()),
    }, t0, notes=[
        "vocab_size is ignored by araroopat — the final size is max_roots + max_patterns + fixed slots.",
        "The explorer disables the on-disk corpus cache so every request hits CAMeL for real.",
    ])

    with _WireTap(bridge) as tap:
        # ---- Step 1a: NFKC ----------------------------------------------
        t0 = time.perf_counter()
        norm_rows = []
        for i, t in enumerate(texts):
            n = unicodedata.normalize("NFKC", t)
            changes = []
            if n != t:
                # Character-level diff is enough for the typical NFKC cases
                # (presentation forms, ligatures, compatibility digits).
                for ch in sorted(set(t) - set(n)):
                    changes.append({"before": ch, "before_cp": _cp(ch), "before_name": _cp_name(ch),
                                    "after": unicodedata.normalize("NFKC", ch)})
            norm_rows.append({"i": i, "before": t, "after": n, "changed": n != t, "changes": changes})
        tr.step("nfkc", "Unicode NFKC normalization", A.AraRooPatTokenizer._corpus_prepass, {
            "rows": norm_rows,
        }, t0, notes=["Run once per text inside _corpus_prepass, before whitespace splitting."])

        # ---- Step 1b: split ---------------------------------------------
        t0 = time.perf_counter()
        split_rows = [{"i": r["i"], "words": r["after"].split()} for r in norm_rows]
        tr.step("split", "Whitespace split", "builtins.str.split", {
            "rows": split_rows,
            "total_words": sum(len(r["words"]) for r in split_rows),
        }, t0, notes=["str.split() with no argument: any run of Unicode whitespace is a boundary."])

        # ---- Step 1c: classify + alpha chunks ---------------------------
        t0 = time.perf_counter()
        word_rows = []
        word_counts: Counter = Counter()
        for r in split_rows:
            for w in r["words"]:
                chars = [{"ch": c, "cls": _classify_char(c), "cp": _cp(c)} for c in w]
                runs = [{"text": chunk, "cls": cls, "kept": cls == "alpha",
                         "ends_with_fem": chunk.endswith("ة")}
                        for cls, chunk in _split_runs(w)]
                chunks = _extract_alpha_chunks(w)
                for c in chunks:
                    word_counts[c] += 1
                word_rows.append({"line": r["i"], "word": w, "chars": chars, "runs": runs,
                                  "chunks": chunks})
        tr.step("chunks", "Character classes → Arabic alpha chunks", _extract_alpha_chunks, {
            "words": word_rows,
            "class_legend": {
                "alpha": "ARABIC_LETTERS ∪ ARABIC_DIACRITICS (minus ة)",
                "fem": "ة — closes the alpha run it follows; never a [CHAR_*], always [CLITICE_ة]",
                "digit": "0-9 and ٠-٩ (DIGIT_INVENTORY)",
                "punct": "ASCII punctuation + Arabic/typographic (PUNCT_INVENTORY)",
                "space": "whitespace (already removed by split)",
                "other": "anything else: Latin letters, emoji, … → UNK at encode time",
            },
        }, t0, notes=[
            "Only 'alpha' runs go to the analyzer. Digit/punct runs are handled by fixed "
            "inventories at encode time and never enter the pre-pass.",
            "An alpha run absorbs a directly following ة and ends there (_split_runs), so a ة "
            "can never sit mid-chunk: مدرسةكبيرة becomes two chunks.",
            "_classify_char is shared with _encode_word, so the pre-pass and encode() agree "
            "on the exact strings sent to CAMeL — this is what makes the analysis cache hit.",
        ])

        # ---- Step 1d: dedup ---------------------------------------------
        t0 = time.perf_counter()
        unique_words = list(word_counts.keys())
        tr.step("dedup", "Deduplicate → word_counts", "collections.Counter", {
            "word_counts": [{"word": w, "count": c} for w, c in word_counts.items()],
            "unique": len(unique_words),
            "occurrences": sum(word_counts.values()),
        }, t0, notes=[
            "Counter keeps first-seen order; unique_words = list(word_counts.keys()).",
            "The analyzer is called once per *unique* chunk — this is the whole point of the pre-pass.",
        ])

        # ---- Step 1e: cache decision ------------------------------------
        t0 = time.perf_counter()
        default_cache = Path("outputs/tokenizers/araroopat_cache") / "corpus_analysis.pkl"
        tr.step("cache", "On-disk cache check (bypassed here)", A.AraRooPatTokenizer._corpus_prepass, {
            "cache_corpus_analysis": tok.cache_corpus_analysis,
            "default_cache_file": str(default_cache),
            "default_cache_exists": (_REPO_ROOT / default_cache).exists(),
            "rule": "if cached_words ⊇ set(unique_words): reuse (filtered to current words) else re-run",
        }, t0, notes=[
            "In a real train() the pickle is reused when it covers every current word — a superset "
            "test, so training on a bigger corpus re-runs the whole pre-pass.",
        ])

        # ---- Step 1f/1g: batches over the bridge ------------------------
        t0 = time.perf_counter()
        batches = []
        raw_by_word: Dict[str, List[Dict[str, str]]] = {}
        for start in range(0, len(unique_words), _BATCH_SIZE):
            batch = unique_words[start:start + _BATCH_SIZE]
            tb = time.perf_counter()
            results = bridge.analyze(batch)
            lines = tap.take()
            for w, cands in zip(batch, results):
                raw_by_word[w] = cands
            batches.append({
                "index": len(batches), "size": len(batch), "words": batch,
                "duration_ms": round((time.perf_counter() - tb) * 1000, 2),
                "request": next((l["line"] for l in lines if l["dir"] == "→ server"), ""),
                "response": next((l["line"] for l in lines if l["dir"] == "← server"), ""),
                "candidates_per_word": [len(raw_by_word[w]) for w in batch],
            })
        tr.step("ipc", "Batched NDJSON round-trip to the CAMeL subprocess", bridge.analyze, {
            "batch_size": _BATCH_SIZE, "batches": batches,
            "server_module": "src/arabic_eval/tools/araroopat_camel_server.py",
            "server_pipeline": [
                "MLEDisambiguator.disambiguate(words) — analyzer + MLE ranking, one call per batch",
                "for each word: keep every scored analysis, top-scored first",
                "_trim(analysis) — keep only the 12 fields the client consumes (incl. asp for the peeler)",
                "json.dumps(ensure_ascii=False) → one line on stdout",
            ],
        }, t0, notes=[
            "One request per 256 unique words. Each response sublist is the MLE-ranked candidate list "
            "for one word; an empty sublist means CAMeL has no analysis at all.",
            "Values are raw CAMeL strings: root radicals dot-separated ('ك.ت.ب'), '#' for a masked "
            "weak radical, clitics as feature tags ('wa_conj', 'Al_det'), '0' for 'none'.",
        ])

        # ---- Step 1h: validate each candidate ---------------------------
        t0 = time.perf_counter()
        analyses, val_rows = trace_validate_words(unique_words, raw_by_word, backend)
        tr.step("validate", "Client-side validation: _first_valid → _dict_to_analysis", _dict_to_analysis, {
            "words": val_rows,
            "analyzed": sum(1 for r in val_rows if r["analyzed"]),
            "total": len(val_rows),
        }, t0, notes=[
            "Candidates are walked in MLE order; the first one that survives every gate wins.",
            "'#' is kept in the root token. It is a radical whose surface letter lives in the pattern "
            "(قال → root ق#ل, pattern 1ا3َ). Deleting it would renumber the slot digits.",
            "Clitic tags become Arabic surfaces here; then normalize_pattern strips those surfaces from "
            "the raw pattern (outermost proclitic first: prc3 → prc2 → prc1 → prc0, then enc0) so the "
            "[PAT_*] token is a bare-stem template.",
        ])

        # ---- Step 1h': clitic peeler on the native misses -----------------
        # Only words CAMeL could not analyse reach this step. Each closed-list
        # slicing is tried least-peeled-first; the residual goes through the
        # same native call + validation as a whole word (its rows are in the
        # bridge trace as extra requests), then `peel_compatible` decides.
        t0 = time.perf_counter()
        peel_rows = trace_peel_words(unique_words, analyses, backend)
        tr.step("peel", "Clitic peeler: closed-list slicing of the words CAMeL rejected",
                B.peel_candidates, {
            "enabled": backend.enable_peeler,
            "words": peel_rows,
            "rescued": sum(1 for r in peel_rows if r["accepted"]),
            "total": len(peel_rows),
        }, t0, notes=[
            "calima-msa-r13 has no interrogative-أ prefix row, no second enclitic slot (enc1) and no "
            "lengthened كمو/همو — any word carrying one of those has no analysis at all.",
            "The peeler may only remove what CAMeL cannot represent (أ, a second pronoun, كمو/همو); "
            "a peel that removes only و/ب/ال/… or a single pronoun is refused, because CAMeL models "
            "those natively and its refusal of the whole word was informative.",
            "Least-peeled candidate first; the residual's own CAMeL clitics are kept and merged; the "
            "residual's canonical spelling must equal the slice so the word decodes byte-identically.",
            "Each residual is looked up with top=32: the MLE model scores every reading of an unseen "
            "word 1.0, so rank 1 is database order (ألزمنا: noun + نا first, the PV + SUBJ:1P we need "
            "sixth). Readings are walked in order and the first one the peeled clitics fit wins.",
        ])

        # ---- Step 1i: CorpusEntry ---------------------------------------
        t0 = time.perf_counter()
        entries: List[CorpusEntry] = [CorpusEntry.from_analysis(w, analyses[w]) for w in unique_words]
        tr.step("entries", "CorpusEntry records (the pre-pass output)", CorpusEntry.from_analysis, {
            "entries": [e.to_dict() for e in entries],
        }, t0, notes=[
            "proclitics = (prc3, prc2, prc1, prc0) minus empties, enclitics = (ة, enc0) minus empties "
            "in emission order. This is exactly what _corpus_prepass returns (and pickles when caching is on).",
        ])

        # ---- Step 2: frequency tables -----------------------------------
        t0 = time.perf_counter()
        root_freq: Counter = Counter()
        pat_freq: Counter = Counter()
        proclitic_freq: Counter = Counter()
        enclitic_freq: Counter = Counter()
        particle_freq: Counter = Counter()
        func_freq: Counter = Counter()
        contrib: Dict[str, Dict[str, List[str]]] = {"root": {}, "pat": {}, "prc": {}, "enc": {}, "prep": {}, "func": {}}
        for e in entries:
            if not e.analyzed:
                continue
            if e.particle:
                kind = "func" if e.particle_kind == "func" else "prep"
                (func_freq if kind == "func" else particle_freq)[e.particle] += 1
                contrib[kind].setdefault(e.particle, []).append(e.word)
            elif e.root and e.pattern:
                root_freq[e.root] += 1
                contrib["root"].setdefault(e.root, []).append(e.word)
                pat_freq[e.pattern] += 1
                contrib["pat"].setdefault(e.pattern, []).append(e.word)
            else:
                continue
            for c in e.proclitics:
                proclitic_freq[c] += 1
                contrib["prc"].setdefault(c, []).append(e.word)
            for c in e.enclitics:
                enclitic_freq[c] += 1
                contrib["enc"].setdefault(c, []).append(e.word)

        def table(cnt: Counter, kind: str) -> List[Dict[str, Any]]:
            return [{"key": k, "freq": v, "words": contrib[kind][k]}
                    for k, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]

        tr.step("freq", "Frequency tables (one count per unique word, not per occurrence)",
                A.AraRooPatTokenizer.train, {
                    "root_freq": table(root_freq, "root"), "pat_freq": table(pat_freq, "pat"),
                    "proclitic_freq": table(proclitic_freq, "prc"),
                    "enclitic_freq": table(enclitic_freq, "enc"),
                    "preposition_freq": table(particle_freq, "prep"),
                    "func_freq": table(func_freq, "func"),
                }, t0, notes=[
                    "Prepositions ([PREP_*]) and function words ([FUNC_*]) contribute their clitics to the "
                    "clitic tables but no root or pattern — they have none to preserve.",
                    "Note the unit: entries are unique words, so a root seen in 3 distinct words "
                    "has freq 3 even if one of them occurred 50 times.",
                ])

        # ---- Step 3: vocab ----------------------------------------------
        t0 = time.perf_counter()
        tok._build_vocab(root_freq, pat_freq, proclitic_freq, enclitic_freq)
        vocab = tok._vocab
        ranges: List[Dict[str, Any]] = []
        for t_, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            fam = _split_key(t_)
            if ranges and ranges[-1]["family"] == fam:
                ranges[-1]["end"] = i
                ranges[-1]["count"] += 1
            else:
                ranges.append({"family": fam, "start": i, "end": i, "count": 1})

        def budget(items: Counter, max_k: int, min_f: int) -> List[Dict[str, Any]]:
            rows = []
            kept = 0
            stopped = False
            for k, f in sorted(items.items(), key=lambda kv: (-kv[1], kv[0])):
                if stopped:
                    rows.append({"key": k, "freq": f, "kept": False, "reason": "after break"})
                    continue
                if kept >= max_k:
                    stopped = True
                    rows.append({"key": k, "freq": f, "kept": False, "reason": f"budget max={max_k} reached"})
                    continue
                if f < min_f:
                    stopped = True
                    rows.append({"key": k, "freq": f, "kept": False, "reason": f"freq {f} < min_freq {min_f}"})
                    continue
                rows.append({"key": k, "freq": f, "kept": True, "reason": ""})
                kept += 1
            return rows

        clitic_order = lambda cnt: [k for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if k]  # noqa: E731
        tr.step("vocab", "Assemble the vocab in deterministic ID order", A.AraRooPatTokenizer._build_vocab, {
            "vocab_size": len(vocab),
            "ranges": ranges,
            "layout": [
                {"family": "special", "source": "SPECIAL_TOKENS_ORDERED", "items": SPECIAL_TOKENS_ORDERED},
                {"family": "lit", "source": "TOK_LIT_BEGIN / TOK_LIT_END", "items": [TOK_LIT_BEGIN, TOK_LIT_END]},
                {"family": "prop", "source": "TOK_PROP_BEGIN / TOK_PROP_END", "items": [TOK_PROP_BEGIN, TOK_PROP_END]},
                {"family": "cliticp", "source": "proclitic_freq sorted by (-freq, surface)", "items": clitic_order(proclitic_freq)},
                {"family": "clitice", "source": "enclitic_freq sorted by (-freq, surface)", "items": clitic_order(enclitic_freq)},
                {"family": "prep", "source": "tok.prepositions (fixed order, corpus-independent)", "items": list(tok.prepositions)},
                {"family": "func", "source": "tok.func_words (fixed order, corpus-independent)", "items": list(tok.func_words)},
                {"family": "char", "source": "CHAR_INVENTORY (sorted letters + sorted diacritics)", "items": CHAR_INVENTORY},
                {"family": "digit", "source": "DIGIT_INVENTORY", "items": DIGIT_INVENTORY},
                {"family": "punct", "source": "PUNCT_INVENTORY", "items": PUNCT_INVENTORY},
                {"family": "root", "source": f"root_freq top-{tok.max_roots}, freq ≥ {tok.min_root_freq}", "items": None},
                {"family": "pat", "source": f"pat_freq top-{tok.max_patterns}, freq ≥ {tok.min_pattern_freq}", "items": None},
            ],
            "root_budget": budget(root_freq, tok.max_roots, tok.min_root_freq),
            "pattern_budget": budget(pat_freq, tok.max_patterns, tok.min_pattern_freq),
            "vocab": [{"id": i, "token": t_, "family": _split_key(t_), "inner": _inner(t_)}
                      for t_, i in sorted(vocab.items(), key=lambda kv: kv[1])],
            "special_token_map": tok._special_token_map,
        }, t0, notes=[
            "Fixed slots (specials, LIT and PROP markers, PREP, FUNC, CHAR, DIGIT, PUNCT) are always present so the "
            "character fallback can encode any Arabic string; only clitics/roots/patterns depend on the corpus.",
            "The budget loop uses `break`, not `continue`: because items are sorted by frequency, the "
            "first item below min_freq ends the loop for everything after it too.",
        ])

        # ---- Step 4: reconstruction -------------------------------------
        t0 = time.perf_counter()
        tok._build_reconstruction(entries, word_counts)
        gen_lines = tap.take()
        real_reco = dict(tok._reconstruction)

        pass1 = []
        written_real: Dict[Tuple[str, str], Counter] = {}
        diac_real: Dict[Tuple[str, str], Counter] = {}
        all_pairs: set = set()
        for e in entries:
            row: Dict[str, Any] = {"word": e.word}
            if not (e.analyzed and e.root and e.pattern):
                row["skipped"] = "not analyzed" if not e.analyzed else "missing root/pattern"
                pass1.append(row)
                continue
            root_tok = f"{PFX_ROOT}{e.root}{SFX}"
            pat_tok = f"{PFX_PAT}{e.pattern}{SFX}"
            if root_tok not in vocab or pat_tok not in vocab:
                row["skipped"] = (f"{root_tok} not in vocab" if root_tok not in vocab
                                  else f"{pat_tok} not in vocab")
                pass1.append(row)
                continue
            all_pairs.add((e.root, e.pattern))
            surface = e.surface or ""
            pro = _trace_proclitic_stack(surface, e.proclitics)
            enc_ops = _trace_enclitic_stack(pro["output"], e.enclitics)
            s = enc_ops[-1]["after"] if enc_ops else pro["output"]
            inflected = _strip_clitic_surfaces(surface, e.proclitics, e.enclitics)
            # The written form: the same strip on the chunk as the corpus wrote it,
            # accepted only when the decoder's joins give the chunk back.
            bare = strip_diacritics(e.word)
            written_stem = _strip_clitic_surfaces(bare, e.proclitics, e.enclitics)
            rejoined = join_word(e.proclitics, written_stem, e.enclitics) if written_stem else ""
            real = entry_realization(e, tok.use_diacritized_surface)
            row.update({
                "root": e.root, "pattern": e.pattern, "surface": surface,
                "camel_stem": e.stem, "proclitic_strip": pro, "enclitic_strip": enc_ops,
                "inflected": inflected,
                "written": {"chunk": bare, "stem": written_stem, "rejoined": rejoined,
                            "reproduces": rejoined == bare},
                "form": real[0] if real else None, "source": real[1] if real else None,
                "weight": word_counts.get(e.word, 1),
                "matches_real": inflected == s and (
                    real is None or real[0] == (written_stem if real[1] == "written" else
                                                (inflected if tok.use_diacritized_surface else strip_diacritics(inflected)))),
            })
            if real is not None:
                (written_real if real[1] == "written" else diac_real).setdefault(
                    (e.root, e.pattern), Counter())[real[0]] += word_counts.get(e.word, 1)
            pass1.append(row)

        pass2 = []
        unresolved = []
        for (root, pat) in sorted(all_pairs):
            rid = vocab[f"{PFX_ROOT}{root}{SFX}"]
            pid = vocab[f"{PFX_PAT}{pat}{SFX}"]
            w_cnt = written_real.get((root, pat))
            d_cnt = diac_real.get((root, pat))
            value = pick_realization(w_cnt, d_cnt)
            if value is not None:
                pass2.append({
                    "root": root, "pattern": pat, "root_id": rid, "pat_id": pid,
                    "realizations": [{"form": f, "count": n, "source": "written"} for f, n in (w_cnt or Counter()).most_common()]
                                    + [{"form": f, "count": n, "source": "diac"} for f, n in (d_cnt or Counter()).most_common()],
                    "chosen": value, "value": value, "tier": 1,
                    "source": "written" if w_cnt else "diac",
                    "matches_real": real_reco.get((rid, pid)) == value,
                })
            else:
                unresolved.append((root, pat, rid, pid))

        pass3 = []
        for (root, pat, rid, pid) in unresolved:
            gen = backend.generate(root, pat)  # cached — populated by the real call above
            naive = naive_pattern_fill(root, pat)
            s = gen or naive
            value = (s if tok.use_diacritized_surface else strip_diacritics(s)) if s else None
            pass3.append({
                "root": root, "pattern": pat, "root_id": rid, "pat_id": pid,
                "generator_result": gen, "naive_fill": naive,
                "tier": 2 if gen else (3 if naive else None), "value": value,
                "matches_real": real_reco.get((rid, pid)) == value,
            })
        tr.step("reconstruction", "Reconstruction lookup: (root_id, pat_id) → inflected stem",
                A.AraRooPatTokenizer._build_reconstruction, {
                    "pass1": pass1, "pass2": pass2, "pass3": pass3,
                    "table": [{"root_id": k[0], "pat_id": k[1], "value": v,
                               "root": _inner(tok._reverse_vocab[k[0]]),
                               "pattern": _inner(tok._reverse_vocab[k[1]])}
                              for k, v in sorted(real_reco.items())],
                    "size": len(real_reco),
                }, t0, wire=gen_lines, notes=[
                    "Pass 1 strips the clitic surfaces from the chunk AS WRITTEN (diacritics removed) and keeps that "
                    "form when the decoder's own joins give the chunk back (source 'written'); otherwise from CAMeL's "
                    "diac — NOT from its stem field, so inflection (the ي of يدرس) survives (source 'diac'). "
                    "With use_diacritized_surface the diac form is the only source.",
                    "Pass 2 keeps the most frequent WRITTEN form per pair, weighted by corpus occurrences (ties on the "
                    "string); diac forms are consulted only for a pair with no written form. That is why الإعرابية "
                    "decodes with the writers' إ although CAMeL's reading spells أَعْرابِيَّة (2026-09-22).",
                    "Pass 3 only fires for pairs with no usable surface: CAMeL generator (tier 2), "
                    "then naive slot substitution (tier 3).",
                ])

        # ---- Step 5: metadata -------------------------------------------
        t0 = time.perf_counter()
        tok._build_metadata(root_freq, pat_freq, proclitic_freq, enclitic_freq, entries, particle_freq, func_freq)
        tr.step("metadata", "Provenance metadata (vocab_metadata.json)", A.AraRooPatTokenizer._build_metadata, {
            "roots": tok._metadata["roots"], "patterns": tok._metadata["patterns"],
            "proclitic_freq": tok._metadata["proclitic_freq"],
            "enclitic_freq": tok._metadata["enclitic_freq"], "config": tok._metadata["config"],
        }, t0, notes=["Answers 'where did this token come from?' without re-running the pre-pass."])

        trace_probe_roundtrip(tr, tok, backend, tap, texts, unique_words, analyses, real_reco)

    # ---- Consistency: compare with a fresh, un-instrumented train() -------
    t0 = time.perf_counter()
    fresh = AraRooPatTokenizer(**params)
    with tempfile.TemporaryDirectory() as tmp:
        fresh.train(texts, cache_path=tmp)
    same_vocab = fresh._vocab == tok._vocab
    same_reco = fresh._reconstruction == tok._reconstruction
    tr.step("verify", "Cross-check against an un-instrumented train()", A.AraRooPatTokenizer.train, {
        "vocab_equal": same_vocab, "reconstruction_equal": same_reco,
        "fresh_vocab_size": fresh.vocab_size, "traced_vocab_size": tok.vocab_size,
    }, t0, notes=["Proves the trace above is the real algorithm, not a paraphrase of it."])

    trace = {
        "input": text,
        "texts": texts,
        "params": effective,
        "total_ms": round((time.perf_counter() - tr.t0) * 1000, 1),
        "steps": tr.steps,
        "all_ok": same_vocab and same_reco and all(
            r.get("matches_real", True) for st in tr.steps
            for r in (st["data"].get("words") or st["data"].get("pass1") or
                      st["data"].get("pass2") or [])
            if isinstance(r, dict)
        ),
    }
    return trace, tok


# ---------------------------------------------------------------------------
# Decode playground: an arbitrary id sequence, "as emitted by an LLM"
# ---------------------------------------------------------------------------

MAX_PLAYGROUND_TOKENS = 512

_SPECIAL_SKIPPED = (A.TOK_PAD, A.TOK_BOS, A.TOK_EOS)


def parse_emission(tok: AraRooPatTokenizer, items: List[Any]) -> Dict[str, Any]:
    """Turn a list of ids / token strings into ids, reporting every item.

    Statuses: ``ok`` (resolved to a vocab id), ``id_not_in_vocab`` (integer
    outside the vocab — allowed, because the real ``decode()`` silently
    drops such ids and that is worth seeing), ``unknown_token`` (a string
    that is not a vocab entry — rejected, there is no id to emit).
    """
    parsed: List[Dict[str, Any]] = []
    ids: List[int] = []
    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        rec: Dict[str, Any] = {"item": item}
        if item.lstrip("-").isdigit():
            tid = int(item)
            t = tok._reverse_vocab.get(tid)
            rec.update({"id": tid, "token": t, "family": _split_key(t) if t else None,
                        "status": "ok" if t is not None else "id_not_in_vocab"})
            ids.append(tid)
        elif item in tok._vocab:
            tid = tok._vocab[item]
            rec.update({"id": tid, "token": item, "family": _split_key(item), "status": "ok"})
            ids.append(tid)
        else:
            rec.update({"id": None, "token": item, "family": None, "status": "unknown_token"})
        parsed.append(rec)
    return {"parsed": parsed, "ids": ids,
            "unknown": [r["item"] for r in parsed if r["status"] == "unknown_token"]}


def trace_decode(tok: AraRooPatTokenizer, items: List[Any]) -> Dict[str, Any]:
    """Run the real ``decode()`` on an arbitrary emission and explain it.

    Ground truth is ``decode(ids)`` plus ``decode(ids[:k])`` for every
    prefix (the streaming view — what an LLM consumer would see as tokens
    arrive). The per-token *annotation* is an interpretation derived from
    the token family with a light shadow of the state machine; it is
    labelled as such in the UI and never used to compute the output.
    """
    if not tok._vocab:
        raise RuntimeError("Tokenizer not trained.")
    parsed = parse_emission(tok, items)
    if parsed["unknown"]:
        raise ValueError("unknown token string(s): " + ", ".join(parsed["unknown"]))
    ids = parsed["ids"]
    if not ids:
        raise ValueError("empty emission")
    if len(ids) > MAX_PLAYGROUND_TOKENS:
        raise ValueError(f"emission too long (> {MAX_PLAYGROUND_TOKENS} tokens)")

    backend: MorphAnalyzer = tok._ensure_backend()
    bridge = backend._bridge
    bridge._ensure_started()
    t_all = time.perf_counter()

    # Full decode first so tier-2 generator calls (and their wire lines)
    # happen once; every prefix decode below then hits the generate cache.
    with _WireTap(bridge) as tap:
        t0 = time.perf_counter()
        decoded = tok.decode(ids)
        full_ms = round((time.perf_counter() - t0) * 1000, 2)
        wire = tap.take()

    prefixes: List[str] = [tok.decode(ids[:k]) for k in range(1, len(ids) + 1)]

    # ---- interpretation layer (shadow state, family-driven) ---------------
    rows: List[Dict[str, Any]] = []
    in_lit = False
    pending: Optional[Tuple[str, int]] = None  # (root, root_id)
    flushed = 0  # words appended to decode()'s ``out`` so far (shadow count)
    counters = Counter()
    buffered_procs = False   # proclitics waiting for a host (shadow of decode()'s clitic_prefix)
    _last_flushed = 0
    tiers: List[Dict[str, Any]] = []
    prev_out = ""
    for pos, tid in enumerate(ids):
        if flushed > _last_flushed:
            buffered_procs = False   # a flushed word consumed the buffer
            _last_flushed = flushed
        t = tok._reverse_vocab.get(tid)
        fam = _split_key(t) if t is not None else None
        inner = _inner(t) if t is not None else None
        note = ""
        tier = None
        if t is None:
            note = "ignored — id is not in the vocab, decode() drops it silently"
            counters["ignored"] += 1
        elif t in _SPECIAL_SKIPPED:
            note = f"ignored — {t} is skipped by decode()"
            counters["ignored"] += 1
        elif t == A.TOK_UNK:
            if pending:
                note = "orphan root dumped as a bare word, then '?' emitted"; counters["orphan_root"] += 1; pending = None; flushed += 1
            else:
                note = "'?' emitted for <unk>"
            counters["unk"] += 1; flushed += 1
        elif t in (TOK_LIT_BEGIN, TOK_PROP_BEGIN):
            what = "literal" if t == TOK_LIT_BEGIN else "proper noun"
            if pending:
                note = f"orphan root dumped as a bare word, then {what} opened"; counters["orphan_root"] += 1; pending = None; flushed += 1
            else:
                note = f"{what} opened — following [CHAR_*] accumulate"
            in_lit = True
        elif t in (TOK_LIT_END, TOK_PROP_END):
            note = (f"{'literal' if t == TOK_LIT_END else 'proper noun'} closed → buffered chars flushed as one word "
                    "(buffered proclitics prepended; either END closes either BEGIN)") if in_lit else \
                f"{t} without an open literal — flushes an empty literal"
            in_lit = False; flushed += 1
        elif in_lit:
            note = "char appended to the open literal" if fam == "char" else f"{fam} token inside an open literal — ignored until the closing marker"
            if fam != "char":
                counters["ignored_in_lit"] += 1
        elif fam == "cliticp":
            note = "proclitic buffered — attaches to the front of the next word"; buffered_procs = True
        elif fam == "clitice":
            if buffered_procs:
                note = "enclitic after buffered proclitics → closed into a clitic-only word (له = ل + ه)"; flushed += 1; buffered_procs = False
            elif flushed:
                note = "enclitic attached to the last flushed word (particle join / ة→ت rules apply)"
            else:
                note = "enclitic with no flushed word yet — emitted standalone (buffered proclitics are NOT consumed)"
                counters["leading_enclitic"] += 1; flushed += 1
        elif fam in ("prep", "func"):
            what = "preposition" if fam == "prep" else "function word"
            if pending:
                note = f"orphan root dumped, then {what} flushed"; counters["orphan_root"] += 1; pending = None; flushed += 1
            else:
                note = f"{what} flushed as a word — a following enclitic joins with particle rules"
            flushed += 1
        elif fam == "root":
            if pending:
                note = "previous root was orphaned (no PAT) → dumped as a bare word; this root now pending"; counters["orphan_root"] += 1; flushed += 1
            else:
                note = "root pending — nothing is emitted until a [PAT_*] arrives"
            pending = (inner, tid)
        elif fam == "pat":
            if pending is None:
                note = "PAT without a pending root → naive fill with an empty root (slots vanish, template letters remain)"
                counters["pat_without_root"] += 1; flushed += 1
            else:
                root, rid = pending
                key = (rid, tid)
                if key in tok._reconstruction:
                    tier, value = 1, tok._reconstruction[key]
                else:
                    gen = backend._generate_cache.get((root, inner))
                    surf = (gen if tok.use_diacritized_surface else strip_diacritics(gen)) if gen else ""
                    if surf:
                        tier, value = 2, surf
                    else:
                        tier, value = 3, (strip_diacritics(naive_pattern_fill(root, inner)) or root)
                note = f"reconstruct ({root}, {inner}) → tier {tier} → {value}; word flushed with buffered proclitics"
                tiers.append({"pos": pos, "root": root, "pattern": inner, "root_id": rid, "pat_id": tid,
                              "tier": tier, "value": value})
                counters[f"tier{tier}"] += 1
                pending = None; flushed += 1
        elif fam == "digit":
            if pending:
                note = "orphan root dumped, then digit flushed"; counters["orphan_root"] += 1; pending = None; flushed += 1
            else:
                note = "digit — glued onto a preceding number, else flushed as a word"
            flushed += 1
        elif fam == "punct":
            if pending:
                note = "orphan root dumped, then punctuation flushed"; counters["orphan_root"] += 1; pending = None; flushed += 1
            else:
                note = "punctuation flushed as its own word"
            flushed += 1
        elif fam == "char":
            note = "bare [CHAR_*] outside a literal — tolerated, flushed as its own word"
            counters["bare_char"] += 1; flushed += 1
        else:
            note = "no rule matched — decode() ignores it"
        out_k = prefixes[pos]
        rows.append({"pos": pos, "id": tid, "token": t, "family": fam, "inner": inner,
                     "note": note, "tier": tier, "output": out_k, "changed": out_k != prev_out})
        prev_out = out_k
    if pending:
        counters["orphan_root_at_end"] += 1
    if in_lit:
        counters["unclosed_lit"] += 1

    return {
        "items": [r["item"] for r in parsed["parsed"]],
        "parsed": parsed["parsed"],
        "ids": ids,
        "decoded": decoded,
        "decode_ms": full_ms,
        "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
        "ref": ref(A.AraRooPatTokenizer.decode),
        "reconstruct_ref": ref(A.AraRooPatTokenizer._reconstruct),
        "rows": rows,
        "tiers": tiers,
        "counters": dict(counters),
        "wire": wire,
        "prefix_matches_final": prefixes[-1] == decoded,
    }
