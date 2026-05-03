"""AraRooPat — Arabic Roots & Patterns tokenizer.

Each Arabic content word becomes two consecutive tokens:
``[ROOT_x] [PAT_y]`` where ``x`` is the trilateral/quadrilateral root and
``y`` is the morphological pattern (CAMeL Tools' positional notation,
e.g. ``"1a2a3"``). Clitics are emitted as separate ``[CLITIC_*]`` tokens
around the stem. Words that fail morphological analysis fall through to
a ``[LIT_BEGIN] [CHAR_*]... [LIT_END]`` byte-safe path so coverage stays
high on loanwords, proper nouns, and dialectal text.

Reconstruction (decode) walks a small state machine and applies a
three-tier resolver: lookup table built at training time → CAMeL
``Generator`` for unseen ``(root, pattern)`` pairs → naive slot
substitution as last resort.

``embedding_type = "standard"`` so this slots into the existing
``LlamaAdapter`` and ``StandardCollator`` without changes.
"""
from __future__ import annotations

import json
import logging
import pickle
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.araroopat_backend import (
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    _strip_clitic_from_end,
    _strip_clitic_from_start,
    naive_pattern_fill,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput
from arabic_eval.tokenizers.utils.arabic_text import (
    ARABIC_DIACRITICS,
    ARABIC_LETTERS,
    strip_diacritics,
)

logger = logging.getLogger("arabic_eval.tokenizers.araroopat")

# ---------------------------------------------------------------------------
# Vocabulary prefixes — these are stored verbatim in vocab.json. The metric
# pipeline strips brackets / non-Arabic chars via ``clean_token_string``,
# but for *its* purposes we populate ``TokenizerOutput.tokens`` with cleaned
# Arabic surface forms instead (see _surface_for_metric in encode()).
# ---------------------------------------------------------------------------

PFX_ROOT = "[ROOT_"
PFX_PAT = "[PAT_"
PFX_CLITICP = "[CLITICP_"  # proclitic (article, conjunction, preposition, ...)
PFX_CLITICE = "[CLITICE_"  # enclitic (object/possessive pronouns)
PFX_CHAR = "[CHAR_"
PFX_PUNCT = "[PUNCT_"
PFX_DIGIT = "[DIGIT_"
SFX = "]"

TOK_PAD = "<pad>"
TOK_BOS = "<s>"
TOK_EOS = "</s>"
TOK_UNK = "<unk>"
TOK_LIT_BEGIN = "[LIT_BEGIN]"
TOK_LIT_END = "[LIT_END]"

SPECIAL_TOKENS_ORDERED = [TOK_PAD, TOK_BOS, TOK_EOS, TOK_UNK]

# Punctuation we recognize. Anything else falls into LIT or UNK.
PUNCT_INVENTORY = (
    list(string.punctuation)
    + ["،", "؛", "؟", "«", "»", "…", "—", "–", "ـ"]  # Arabic + typographic
)

# Digits we recognize: ASCII 0-9 + Arabic-Indic ٠-٩.
DIGIT_INVENTORY = list("0123456789") + list("٠١٢٣٤٥٦٧٨٩")

# Arabic character inventory for the CHAR fallback. We include letters and
# diacritics; long vowels are already in ``ARABIC_LETTERS``.
CHAR_INVENTORY = sorted(ARABIC_LETTERS) + sorted(ARABIC_DIACRITICS)


def _clean_arabic(text: str) -> str:
    """Keep only Arabic letters, long vowels, and diacritics."""
    return "".join(c for c in text if c in ARABIC_LETTERS or c in ARABIC_DIACRITICS)


def _is_arabic_alpha(ch: str) -> bool:
    return ch in ARABIC_LETTERS or ch in ARABIC_DIACRITICS


def _classify_char(ch: str) -> str:
    """Return one of {'alpha', 'digit', 'punct', 'space', 'other'}."""
    if _is_arabic_alpha(ch):
        return "alpha"
    if ch in DIGIT_INVENTORY:
        return "digit"
    if ch in PUNCT_INVENTORY:
        return "punct"
    if ch.isspace():
        return "space"
    return "other"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@tokenizer_registry.register("araroopat")
class AraRooPatTokenizer(BaseTokenizer):
    """Arabic Roots & Patterns tokenizer (see module docstring)."""

    def __init__(self, **kwargs: Any) -> None:
        # Configurable params (also exposed via configs/tokenizers/araroopat.yaml).
        self.max_roots: int = int(kwargs.get("max_roots", 10000))
        self.max_patterns: int = int(kwargs.get("max_patterns", 500))
        self.min_root_freq: int = int(kwargs.get("min_root_freq", 2))
        self.min_pattern_freq: int = int(kwargs.get("min_pattern_freq", 2))
        self.generator_timeout_ms: int = int(kwargs.get("generator_timeout_ms", 50))
        self.use_diacritized_surface: bool = bool(kwargs.get("use_diacritized_surface", False))
        self.cache_corpus_analysis: bool = bool(kwargs.get("cache_corpus_analysis", True))
        self.add_bos_eos: bool = bool(kwargs.get("add_bos_eos", True))

        # State.
        self._backend: Optional[MorphAnalyzer] = None
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._special_token_map: Dict[str, int] = {}
        # (root_id, pattern_id) -> surface form (cleaned or diacritized per config)
        self._reconstruction: Dict[Tuple[int, int], str] = {}
        # Provenance: per-token-string metadata.
        self._metadata: Dict[str, Any] = {"roots": {}, "patterns": {}, "config": {}}

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> MorphAnalyzer:
        if self._backend is None:
            self._backend = MorphAnalyzer(generator_timeout_ms=self.generator_timeout_ms)
        return self._backend

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> Dict[str, int]:
        return self._special_token_map

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        cache_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Build vocab from corpus.

        ``vocab_size`` is ignored — final vocab size is determined by
        ``max_roots + max_patterns + fixed-slot tokens``. We accept the
        argument so the platform's ``train_tokenizer.py`` CLI works
        uniformly across all tokenizers.

        ``cache_path`` (optional): directory where the per-corpus analysis
        is persisted. If absent, we use ``./outputs/tokenizers/araroopat_cache/``.
        """
        if vocab_size is not None:
            logger.info(
                "vocab_size=%d ignored — araroopat sizes the vocab via max_roots (%d) "
                "+ max_patterns (%d) + fixed slots.",
                vocab_size, self.max_roots, self.max_patterns,
            )

        # ---- Step 1: corpus pre-pass (with on-disk cache) ----
        cache_dir = Path(cache_path) if cache_path else Path("outputs/tokenizers/araroopat_cache")
        entries = self._corpus_prepass(texts, cache_dir)

        # ---- Step 2: frequency tables ----
        root_freq: Counter = Counter()
        pat_freq: Counter = Counter()
        proclitic_freq: Counter = Counter()
        enclitic_freq: Counter = Counter()
        for e in entries:
            if e.analyzed and e.root and e.pattern:
                root_freq[e.root] += 1
                pat_freq[e.pattern] += 1
                for c in e.proclitics:
                    proclitic_freq[c] += 1
                for c in e.enclitics:
                    enclitic_freq[c] += 1

        logger.info(
            "Pre-pass stats: %d analyzed words, %d unique roots, %d unique patterns, "
            "%d proclitic surfaces, %d enclitic surfaces.",
            sum(1 for e in entries if e.analyzed),
            len(root_freq), len(pat_freq),
            len(proclitic_freq), len(enclitic_freq),
        )

        # ---- Step 3: assemble flat vocab in deterministic ID order ----
        self._build_vocab(root_freq, pat_freq, proclitic_freq, enclitic_freq)

        # ---- Step 4: reconstruction lookup from observed (root, pattern) ----
        self._build_reconstruction(entries)

        # ---- Step 5: provenance metadata ----
        self._build_metadata(root_freq, pat_freq, proclitic_freq, enclitic_freq, entries)

        logger.info("AraRooPat trained — vocab size: %d", self.vocab_size)
        logger.info(
            "  roots: %d, patterns: %d, proclitics: %d, enclitics: %d, "
            "fixed slots: %d, reconstruction entries: %d",
            sum(1 for t in self._vocab if t.startswith(PFX_ROOT)),
            sum(1 for t in self._vocab if t.startswith(PFX_PAT)),
            sum(1 for t in self._vocab if t.startswith(PFX_CLITICP)),
            sum(1 for t in self._vocab if t.startswith(PFX_CLITICE)),
            sum(1 for t in self._vocab
                if t.startswith((PFX_CHAR, PFX_PUNCT, PFX_DIGIT, "<", "[L"))),
            len(self._reconstruction),
        )

    def _corpus_prepass(self, texts: List[str], cache_dir: Path) -> List[CorpusEntry]:
        """Analyze every distinct *alpha chunk* in the corpus once. Cache to disk.

        We chunk words by character class (matching what ``_encode_word`` does
        at runtime) so the cache keys match the analyze() calls made during
        encoding — no analyzer hits at runtime if the cache covers the corpus.
        """
        backend = self._ensure_backend()

        word_counts: Counter = Counter()
        for t in texts:
            t = unicodedata.normalize("NFKC", t)
            for w in t.split():
                for chunk in _extract_alpha_chunks(w):
                    word_counts[chunk] += 1
        unique_words = list(word_counts.keys())

        cache_file = cache_dir / "corpus_analysis.pkl"
        if self.cache_corpus_analysis and cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    cached = pickle.load(f)
                cached_words = {e.word for e in cached}
                if cached_words >= set(unique_words):
                    logger.info("Loaded cached corpus analysis (%d entries) from %s",
                                len(cached), cache_file)
                    # Filter to current vocabulary universe; expand counts.
                    return [e for e in cached if e.word in word_counts]
                logger.info("Cache exists but doesn't cover this corpus — re-running pre-pass.")
            except Exception as e:
                logger.warning("Cache load failed (%s) — re-running pre-pass.", e)

        # Run analyzer in batches via the bridge — saves one IPC round-trip
        # per cache miss vs the old per-word loop.
        entries: List[CorpusEntry] = []
        analyzed_count = 0
        batch_size = 256
        with tqdm(total=len(unique_words), desc="CAMeL pre-pass", unit="word") as pbar:
            for start in range(0, len(unique_words), batch_size):
                batch = unique_words[start:start + batch_size]
                analyses = backend.analyze_many(batch, batch_size=batch_size)
                for word, a in zip(batch, analyses):
                    entries.append(CorpusEntry.from_analysis(word, a))
                    if a is not None:
                        analyzed_count += 1
                pbar.update(len(batch))

        logger.info("Pre-pass: analyzed %d / %d words (%.1f%%)",
                    analyzed_count, len(unique_words),
                    100.0 * analyzed_count / max(len(unique_words), 1))

        if self.cache_corpus_analysis:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                pickle.dump(entries, f)
            # JSON view for quick inspection.
            with (cache_dir / "corpus_analysis.json").open("w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in entries[:10000]], f,
                          ensure_ascii=False, indent=2)
            logger.info("Cached corpus analysis to %s (+ first-10k JSON view)", cache_file)

        return entries

    def _build_vocab(
        self,
        root_freq: Counter,
        pat_freq: Counter,
        proclitic_freq: Counter,
        enclitic_freq: Counter,
    ) -> None:
        """Assemble the flat vocab dict with deterministic ID order."""
        vocab: Dict[str, int] = {}

        def add(tok: str) -> None:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        # 0–3: specials.
        for s in SPECIAL_TOKENS_ORDERED:
            add(s)
        # 4–5: literal markers.
        add(TOK_LIT_BEGIN)
        add(TOK_LIT_END)
        # 6+: proclitics (CLITICP), then enclitics (CLITICE) — sorted by
        # freq desc, ties broken alphabetically. Distinct ranges so the
        # decoder never has to disambiguate prc vs enc from surface alone.
        for clitic, freq in sorted(proclitic_freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < 1 or not clitic:
                continue
            add(f"{PFX_CLITICP}{clitic}{SFX}")
        for clitic, freq in sorted(enclitic_freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < 1 or not clitic:
                continue
            add(f"{PFX_CLITICE}{clitic}{SFX}")
        # Then: chars (Arabic letters + diacritics), in fixed order.
        for ch in CHAR_INVENTORY:
            add(f"{PFX_CHAR}{ch}{SFX}")
        # Then: digits.
        for d in DIGIT_INVENTORY:
            add(f"{PFX_DIGIT}{d}{SFX}")
        # Then: punctuation.
        for p in PUNCT_INVENTORY:
            add(f"{PFX_PUNCT}{p}{SFX}")
        # Then: roots (top-K by freq, freq >= min_root_freq).
        root_items = sorted(root_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        kept_roots = 0
        for root, freq in root_items:
            if kept_roots >= self.max_roots:
                break
            if freq < self.min_root_freq:
                break
            add(f"{PFX_ROOT}{root}{SFX}")
            kept_roots += 1
        # Then: patterns (top-K by freq, freq >= min_pattern_freq).
        pat_items = sorted(pat_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        kept_pats = 0
        for pat, freq in pat_items:
            if kept_pats >= self.max_patterns:
                break
            if freq < self.min_pattern_freq:
                break
            add(f"{PFX_PAT}{pat}{SFX}")
            kept_pats += 1

        self._vocab = vocab
        self._reverse_vocab = {i: t for t, i in vocab.items()}
        self._special_token_map = {
            "pad_token": vocab[TOK_PAD],
            "bos_token": vocab[TOK_BOS],
            "eos_token": vocab[TOK_EOS],
            "unk_token": vocab[TOK_UNK],
        }

    def _build_reconstruction(self, entries: List[CorpusEntry]) -> None:
        """Build (root_id, pat_id) -> *inflected-stem* surface for every distinct pair.

        Important nuance: CAMeL's ``stem`` field gives the **lexical** stem
        (root letters in their pattern slots only — no inflectional prefixes
        like the present-tense ي of يدرس). What we want for reconstruction
        is the **inflected stem** — diacritized surface with clitics stripped
        but inflection retained. We compute that by stripping clitic surface
        chars from CAMeL's ``diac`` field (the full surface).
        """
        reco: Dict[Tuple[int, int], str] = {}
        backend = self._ensure_backend()

        # Pass 1: collect inflected-stem realizations per (root, pattern_bare).
        corpus_realizations: Dict[Tuple[str, str], Counter] = {}
        all_pairs: set = set()
        for e in entries:
            if not (e.analyzed and e.root and e.pattern and e.surface):
                continue
            root_tok = f"{PFX_ROOT}{e.root}{SFX}"
            pat_tok = f"{PFX_PAT}{e.pattern}{SFX}"
            if root_tok not in self._vocab or pat_tok not in self._vocab:
                continue
            all_pairs.add((e.root, e.pattern))
            inflected = _strip_clitic_surfaces(e.surface, e.proclitics, e.enclitics)
            if inflected:
                corpus_realizations.setdefault((e.root, e.pattern), Counter())[inflected] += 1

        # Pass 2: prefer the most-frequent corpus realization.
        unresolved: List[Tuple[str, str]] = []
        for (root, pat) in all_pairs:
            if (root, pat) in corpus_realizations:
                inflected = corpus_realizations[(root, pat)].most_common(1)[0][0]
            else:
                unresolved.append((root, pat))
                continue
            inflected = inflected if self.use_diacritized_surface else strip_diacritics(inflected)
            root_id = self._vocab[f"{PFX_ROOT}{root}{SFX}"]
            pat_id = self._vocab[f"{PFX_PAT}{pat}{SFX}"]
            reco[(root_id, pat_id)] = inflected

        # Pass 3: resolve the rest via CAMeL generator (returns bare stem) +
        # naive substitution as last resort.
        if unresolved:
            logger.info("Generating %d (root, pattern) pairs unseen in corpus...",
                        len(unresolved))
            for (root, pat) in tqdm(unresolved, desc="Generator-fill", unit="pair"):
                s = backend.generate(root, pat) or naive_pattern_fill(root, pat)
                if not s:
                    continue
                inflected = s if self.use_diacritized_surface else strip_diacritics(s)
                root_id = self._vocab[f"{PFX_ROOT}{root}{SFX}"]
                pat_id = self._vocab[f"{PFX_PAT}{pat}{SFX}"]
                reco[(root_id, pat_id)] = inflected

        self._reconstruction = reco

    def _build_metadata(
        self,
        root_freq: Counter,
        pat_freq: Counter,
        proclitic_freq: Counter,
        enclitic_freq: Counter,
        entries: List[CorpusEntry],
    ) -> None:
        """Provenance: which roots/patterns came from which words, with examples."""
        # Build root → example words map (up to 5 each).
        root_examples: Dict[str, List[str]] = {}
        pat_examples: Dict[str, List[Tuple[str, str]]] = {}
        for e in entries:
            if not (e.analyzed and e.root and e.pattern):
                continue
            if e.root in self._vocab_root_set():
                lst = root_examples.setdefault(e.root, [])
                if len(lst) < 5 and e.word not in lst:
                    lst.append(e.word)
            if e.pattern in self._vocab_pattern_set():
                lst2 = pat_examples.setdefault(e.pattern, [])
                if len(lst2) < 5 and (e.root, e.surface or e.word) not in lst2:
                    lst2.append((e.root, e.surface or e.word))

        roots_meta: Dict[str, Any] = {}
        for tok, tid in self._vocab.items():
            if not tok.startswith(PFX_ROOT):
                continue
            root = tok[len(PFX_ROOT):-len(SFX)]
            roots_meta[root] = {
                "id": tid,
                "freq": root_freq.get(root, 0),
                "source": "corpus" if root_freq.get(root, 0) > 0 else "camel_db_only",
                "example_words": root_examples.get(root, [])[:5],
            }
        patterns_meta: Dict[str, Any] = {}
        for tok, tid in self._vocab.items():
            if not tok.startswith(PFX_PAT):
                continue
            pat = tok[len(PFX_PAT):-len(SFX)]
            patterns_meta[pat] = {
                "id": tid,
                "freq": pat_freq.get(pat, 0),
                "source": "corpus" if pat_freq.get(pat, 0) > 0 else "camel_db_only",
                "examples": [list(p) for p in pat_examples.get(pat, [])[:5]],
            }
        self._metadata = {
            "roots": roots_meta,
            "patterns": patterns_meta,
            "proclitic_freq": dict(proclitic_freq),
            "enclitic_freq": dict(enclitic_freq),
            "config": {
                "max_roots": self.max_roots,
                "max_patterns": self.max_patterns,
                "min_root_freq": self.min_root_freq,
                "min_pattern_freq": self.min_pattern_freq,
                "use_diacritized_surface": self.use_diacritized_surface,
                "generator_timeout_ms": self.generator_timeout_ms,
            },
        }

    def _vocab_root_set(self) -> set:
        return {t[len(PFX_ROOT):-len(SFX)] for t in self._vocab if t.startswith(PFX_ROOT)}

    def _vocab_pattern_set(self) -> set:
        return {t[len(PFX_PAT):-len(SFX)] for t in self._vocab if t.startswith(PFX_PAT)}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        if not self._vocab:
            raise RuntimeError("Tokenizer not trained or loaded.")

        ids: List[int] = []
        toks: List[str] = []  # cleaned Arabic surface strings, for the metrics

        if self.add_bos_eos:
            ids.append(self._special_token_map["bos_token"])
            toks.append("")  # BOS contributes no Arabic content

        text = unicodedata.normalize("NFKC", text)
        for raw_word in text.split():
            self._encode_word(raw_word, ids, toks)

        if self.add_bos_eos:
            ids.append(self._special_token_map["eos_token"])
            toks.append("")

        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
            toks = toks[:max_length]

        attention_mask = [1] * len(ids)

        if padding and max_length and len(ids) < max_length:
            pad = self._special_token_map["pad_token"]
            need = max_length - len(ids)
            ids = ids + [pad] * need
            attention_mask = attention_mask + [0] * need
            toks = toks + [""] * need

        return TokenizerOutput(input_ids=ids, attention_mask=attention_mask, tokens=toks)

    def _encode_word(self, word: str, ids: List[int], toks: List[str]) -> None:
        """Walk a whitespace word, emitting tokens for runs of alpha/digit/punct."""
        if not word:
            return
        # Split into runs by character class.
        i = 0
        n = len(word)
        while i < n:
            cls = _classify_char(word[i])
            j = i + 1
            while j < n and _classify_char(word[j]) == cls:
                j += 1
            chunk = word[i:j]
            if cls == "alpha":
                self._emit_alpha(chunk, ids, toks)
            elif cls == "digit":
                for ch in chunk:
                    self._emit_atom(f"{PFX_DIGIT}{ch}{SFX}", ch, ids, toks, arabic=False)
            elif cls == "punct":
                for ch in chunk:
                    self._emit_atom(f"{PFX_PUNCT}{ch}{SFX}", ch, ids, toks, arabic=False)
            elif cls == "space":
                pass  # outer split already removed
            else:
                # Unknown char (Latin letter, emoji, ...) — UNK.
                ids.append(self._special_token_map["unk_token"])
                toks.append("")
            i = j

    def _emit_alpha(self, chunk: str, ids: List[int], toks: List[str]) -> None:
        """Emit tokens for an Arabic alphabetic chunk: try analyzer, fall back to LIT."""
        backend = self._ensure_backend()
        a: Optional[Analysis] = backend.analyze(chunk)

        if a is not None and a.root and a.pattern:
            root_tok = f"{PFX_ROOT}{a.root}{SFX}"
            pat_tok = f"{PFX_PAT}{a.pattern}{SFX}"
            if root_tok in self._vocab and pat_tok in self._vocab:
                for c in (a.prc3, a.prc2, a.prc1, a.prc0):
                    if c:
                        self._emit_clitic(c, ids, toks, kind="p")
                ids.append(self._vocab[root_tok])
                toks.append(_clean_arabic(a.root))
                ids.append(self._vocab[pat_tok])
                # PAT metric-string = cleaned inflected stem (full surface
                # minus clitic chars). Contains root letters in their
                # pattern-positioned context plus inflectional morphology
                # (e.g. ي of present-tense verbs). Needed for
                # pattern_conservation_rate.
                proc = tuple(c for c in (a.prc3, a.prc2, a.prc1, a.prc0) if c)
                enc = tuple(c for c in (a.enc0,) if c)
                inflected = _strip_clitic_surfaces(a.surface or chunk, proc, enc)
                toks.append(_clean_arabic(strip_diacritics(inflected or chunk)))
                if a.enc0:
                    self._emit_clitic(a.enc0, ids, toks, kind="e")
                return
            # Either root or pattern is OOV — fall through to LIT.

        self._emit_lit(chunk, ids, toks)

    def _emit_clitic(self, clitic: str, ids: List[int], toks: List[str],
                     kind: str) -> None:
        """Emit a clitic with explicit kind ('p'=proclitic, 'e'=enclitic)."""
        prefix = PFX_CLITICP if kind == "p" else PFX_CLITICE
        tok = f"{prefix}{clitic}{SFX}"
        if tok in self._vocab:
            ids.append(self._vocab[tok])
            toks.append(_clean_arabic(clitic))
        else:
            # Unknown clitic surface for that kind — emit as literal chars.
            self._emit_lit(clitic, ids, toks)

    def _emit_lit(self, chunk: str, ids: List[int], toks: List[str]) -> None:
        """Emit chunk as [LIT_BEGIN] CHAR... [LIT_END]."""
        ids.append(self._vocab[TOK_LIT_BEGIN])
        toks.append("")
        for ch in chunk:
            tok = f"{PFX_CHAR}{ch}{SFX}"
            if tok in self._vocab:
                ids.append(self._vocab[tok])
                toks.append(ch if ch in ARABIC_LETTERS else "")
            else:
                ids.append(self._special_token_map["unk_token"])
                toks.append("")
        ids.append(self._vocab[TOK_LIT_END])
        toks.append("")

    def _emit_atom(self, tok: str, ch: str, ids: List[int], toks: List[str],
                   arabic: bool) -> None:
        if tok in self._vocab:
            ids.append(self._vocab[tok])
            toks.append(ch if arabic else "")
        else:
            ids.append(self._special_token_map["unk_token"])
            toks.append("")

    # ------------------------------------------------------------------
    # Decoding (three-tier reconstruction)
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """Walk tokens with a small state machine.

        Distinct prefixes ``[CLITICP_*]`` (proclitic) and ``[CLITICE_*]``
        (enclitic) make the prc-vs-enc decision unambiguous from the token
        type alone. Proclitics buffer until the next ROOT+PAT (or LIT/PUNCT)
        word; enclitics attach to the just-emitted word.
        """
        if not self._vocab:
            raise RuntimeError("Tokenizer not trained or loaded.")
        backend = self._ensure_backend()

        out: List[str] = []                  # finalized "words" in output order
        clitic_prefix: List[str] = []        # buffered proclitics for next word
        pending_root: Optional[str] = None
        pending_root_id: Optional[int] = None
        in_lit = False
        lit_buffer: List[str] = []

        def attach_enclitic(s: str) -> None:
            """Append enclitic surface to the just-emitted word, or drop it."""
            if out:
                out[-1] = out[-1] + s
            else:
                out.append(s)

        def flush_word(word: str) -> None:
            """Emit a content-bearing word, prepending any buffered proclitics."""
            out.append("".join(clitic_prefix) + word)
            clitic_prefix.clear()

        def dump_orphan_root() -> None:
            nonlocal pending_root, pending_root_id
            if pending_root is None:
                return
            flush_word(pending_root)
            pending_root = None
            pending_root_id = None

        for tid in ids:
            tok = self._reverse_vocab.get(tid)
            if tok is None or tok in (TOK_PAD, TOK_BOS, TOK_EOS):
                continue

            if tok == TOK_LIT_BEGIN:
                dump_orphan_root()
                in_lit = True
                lit_buffer.clear()
                continue

            if tok == TOK_LIT_END:
                lit_str = "".join(lit_buffer)
                if clitic_prefix or lit_str:
                    flush_word(lit_str)
                in_lit = False
                lit_buffer.clear()
                continue

            if in_lit:
                if tok.startswith(PFX_CHAR):
                    lit_buffer.append(tok[len(PFX_CHAR):-len(SFX)])
                continue

            if tok.startswith(PFX_CLITICP):
                clitic_prefix.append(tok[len(PFX_CLITICP):-len(SFX)])
                continue

            if tok.startswith(PFX_CLITICE):
                attach_enclitic(tok[len(PFX_CLITICE):-len(SFX)])
                continue

            if tok.startswith(PFX_ROOT):
                dump_orphan_root()
                pending_root = tok[len(PFX_ROOT):-len(SFX)]
                pending_root_id = tid
                continue

            if tok.startswith(PFX_PAT):
                pat = tok[len(PFX_PAT):-len(SFX)]
                if pending_root is None:
                    flush_word(naive_pattern_fill("", pat))
                    continue
                stem = self._reconstruct(pending_root_id, tid, pending_root, pat, backend)
                flush_word(stem)
                pending_root = None
                pending_root_id = None
                continue

            if tok.startswith(PFX_DIGIT):
                dump_orphan_root()
                ch = tok[len(PFX_DIGIT):-len(SFX)]
                # Glue consecutive digits into one number.
                if out and out[-1] and out[-1][-1].isdigit() and not clitic_prefix:
                    out[-1] = out[-1] + ch
                else:
                    flush_word(ch)
                continue

            if tok.startswith(PFX_PUNCT):
                dump_orphan_root()
                flush_word(tok[len(PFX_PUNCT):-len(SFX)])
                continue

            if tok.startswith(PFX_CHAR):
                # Bare CHAR outside LIT — tolerate during generation.
                flush_word(tok[len(PFX_CHAR):-len(SFX)])
                continue

            if tok == TOK_UNK:
                dump_orphan_root()
                flush_word("?")
                continue

        # Final flushes.
        dump_orphan_root()
        if clitic_prefix:
            out.append("".join(clitic_prefix))

        return " ".join(s for s in out if s)

    def _reconstruct(
        self,
        root_id: int,
        pat_id: int,
        root: str,
        pattern: str,
        backend: MorphAnalyzer,
    ) -> str:
        """Three-tier: lookup → CAMeL generator → naive substitution."""
        # Tier 1: lookup.
        surf = self._reconstruction.get((root_id, pat_id))
        if surf:
            return surf
        # Tier 2: CAMeL generator (also returns naive on its own internal failures).
        gen = backend.generate(root, pattern)
        if gen:
            surface = gen if self.use_diacritized_surface else strip_diacritics(gen)
            if surface:
                return surface
        # Tier 3: naive slot fill.
        logger.debug("Tier-3 (naive) reconstruction for (%s, %s)", root, pattern)
        return strip_diacritics(naive_pattern_fill(root, pattern)) or root

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # vocab.json — token → id, sorted by id for diffability.
        with (path / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(
                {t: i for t, i in sorted(self._vocab.items(), key=lambda kv: kv[1])},
                f, ensure_ascii=False, indent=2,
            )
        # reverse_vocab.json — convenience.
        with (path / "reverse_vocab.json").open("w", encoding="utf-8") as f:
            json.dump(
                {str(i): t for i, t in sorted(self._reverse_vocab.items())},
                f, ensure_ascii=False, indent=2,
            )

        # reconstruction.pkl (fast) + reconstruction.json (human-readable).
        with (path / "reconstruction.pkl").open("wb") as f:
            pickle.dump(self._reconstruction, f)
        # JSON form: keys are "root_id,pat_id" strings since JSON can't have tuple keys.
        reco_json = {
            f"{r},{p}": {
                "root": self._reverse_vocab.get(r, "?")[len(PFX_ROOT):-len(SFX)]
                        if self._reverse_vocab.get(r, "").startswith(PFX_ROOT) else "?",
                "pattern": self._reverse_vocab.get(p, "?")[len(PFX_PAT):-len(SFX)]
                            if self._reverse_vocab.get(p, "").startswith(PFX_PAT) else "?",
                "surface": surf,
            }
            for (r, p), surf in sorted(self._reconstruction.items())
        }
        with (path / "reconstruction.json").open("w", encoding="utf-8") as f:
            json.dump(reco_json, f, ensure_ascii=False, indent=2)

        # vocab_metadata.json — provenance.
        with (path / "vocab_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        # config.json — round-trip params.
        with (path / "config.json").open("w", encoding="utf-8") as f:
            json.dump({
                "tokenizer_class": "AraRooPatTokenizer",
                "max_roots": self.max_roots,
                "max_patterns": self.max_patterns,
                "min_root_freq": self.min_root_freq,
                "min_pattern_freq": self.min_pattern_freq,
                "generator_timeout_ms": self.generator_timeout_ms,
                "use_diacritized_surface": self.use_diacritized_surface,
                "add_bos_eos": self.add_bos_eos,
            }, f, indent=2)

    def load(self, path: Path | str) -> None:
        path = Path(path)

        with (path / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.max_roots = cfg.get("max_roots", self.max_roots)
        self.max_patterns = cfg.get("max_patterns", self.max_patterns)
        self.min_root_freq = cfg.get("min_root_freq", self.min_root_freq)
        self.min_pattern_freq = cfg.get("min_pattern_freq", self.min_pattern_freq)
        self.generator_timeout_ms = cfg.get("generator_timeout_ms", self.generator_timeout_ms)
        self.use_diacritized_surface = cfg.get("use_diacritized_surface", self.use_diacritized_surface)
        self.add_bos_eos = cfg.get("add_bos_eos", self.add_bos_eos)

        with (path / "vocab.json").open("r", encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._reverse_vocab = {i: t for t, i in self._vocab.items()}
        self._special_token_map = {
            "pad_token": self._vocab[TOK_PAD],
            "bos_token": self._vocab[TOK_BOS],
            "eos_token": self._vocab[TOK_EOS],
            "unk_token": self._vocab[TOK_UNK],
        }

        # Prefer pickle; fall back to JSON if missing/corrupt.
        reco_pkl = path / "reconstruction.pkl"
        reco_json = path / "reconstruction.json"
        if reco_pkl.exists():
            try:
                with reco_pkl.open("rb") as f:
                    self._reconstruction = pickle.load(f)
            except Exception as e:
                logger.warning("Pickle reconstruction load failed (%s) — falling back to JSON.", e)
                self._reconstruction = self._load_reco_json(reco_json)
        else:
            self._reconstruction = self._load_reco_json(reco_json)

        meta_path = path / "vocab_metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def _load_reco_json(self, path: Path) -> Dict[Tuple[int, int], str]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[Tuple[int, int], str] = {}
        for key, val in raw.items():
            r_str, p_str = key.split(",")
            out[(int(r_str), int(p_str))] = val["surface"]
        return out


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _strip_clitic_surfaces(
    surface: str,
    proclitics: Tuple[str, ...],
    enclitics: Tuple[str, ...],
) -> str:
    """Strip clitic surface chars from a diacritized surface to get the inflected stem.

    Mirrors what ``normalize_pattern`` does for patterns, but on the diac
    string. Used to compute the reconstruction value (clitic-free *inflected*
    stem) from CAMeL's full ``diac`` field.
    """
    s = surface
    for c in proclitics:
        s = _strip_clitic_from_start(s, c)
    for c in enclitics:
        s = _strip_clitic_from_end(s, c)
    return s


def _extract_alpha_chunks(word: str) -> List[str]:
    """Return contiguous Arabic-alpha runs from a whitespace word.

    Matches what ``_encode_word`` does at runtime — used during the corpus
    pre-pass so the analyzer cache keys agree with the analyze() calls
    made during encoding.
    """
    out: List[str] = []
    i, n = 0, len(word)
    while i < n:
        if _classify_char(word[i]) == "alpha":
            j = i + 1
            while j < n and _classify_char(word[j]) == "alpha":
                j += 1
            out.append(word[i:j])
            i = j
        else:
            i += 1
    return out
