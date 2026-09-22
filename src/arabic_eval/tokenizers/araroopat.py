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

from arabic_eval.params_spec import ParamSpec
from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.araroopat_backend import (
    FUNC_INVENTORY,
    PREPOSITION_INVENTORY,
    TAA_MARBUTA,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    join_particle_enclitic,
    naive_pattern_fill,
    needs_spelling_walk,
    strip_enclitics_from_end,
    strip_proclitics_from_start,
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
PFX_PREP = "[PREP_"        # closed-class preposition/particle, one token each
PFX_FUNC = "[FUNC_"        # closed-class function word (pronoun, conjunction, ...), one token each
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
# Proper nouns: the same character path, with its own markers so the model
# can tell "a name CAMeL knows" from "a word CAMeL does not know".
TOK_PROP_BEGIN = "[PROP_BEGIN]"
TOK_PROP_END = "[PROP_END]"
# Which database proper nouns take the [PROP_*] path: "unrooted" = only the
# names CAMeL has no root for (باريس, كوريا); "all" = every name, including
# those with a root (محمد, مصر, القاهرة), which then no longer feed the
# root / pattern tables.
PROPER_NOUN_MODES = ("unrooted", "all")

# Constructor defaults of the configurable params — one table read by ``__init__`` and
# ``param_spec`` so the two cannot drift. NOTE ``max_patterns`` 500 is the class default
# the code has always had; the Balanced tier every experiment uses is 4000, set in
# configs/tokenizers/araroopat.yaml (a changed default here would change every future vocab).
DEFAULTS: Dict[str, Any] = {
    "max_roots": 10000,
    "max_patterns": 500,
    "min_root_freq": 2,
    "min_pattern_freq": 2,
    "generator_timeout_ms": 50,
    "use_diacritized_surface": False,
    "cache_corpus_analysis": True,
    "add_bos_eos": True,
    "clitic_peeler": True,
    "peel_bare_alef": False,
    "proper_nouns": "unrooted",
}

SPECIAL_TOKENS_ORDERED = [TOK_PAD, TOK_BOS, TOK_EOS, TOK_UNK]

# Punctuation we recognize. Anything else falls into LIT or UNK.
PUNCT_INVENTORY = (
    list(string.punctuation)
    + ["،", "؛", "؟", "«", "»", "…", "—", "–", "ـ"]  # Arabic + typographic
)

# Digits we recognize: ASCII 0-9 + Arabic-Indic ٠-٩.
DIGIT_INVENTORY = list("0123456789") + list("٠١٢٣٤٥٦٧٨٩")

# Arabic character inventory for the CHAR fallback. We include letters and
# diacritics; long vowels are already in ``ARABIC_LETTERS``. The tāʾ marbūṭa
# is NOT a char here: it is always word-final and always emitted as the
# enclitic [CLITICE_ة] (on every path, LIT included), never as [CHAR_ة].
CHAR_INVENTORY = sorted(ARABIC_LETTERS - {TAA_MARBUTA}) + sorted(ARABIC_DIACRITICS)


def _clean_arabic(text: str) -> str:
    """Keep only Arabic letters, long vowels, and diacritics."""
    return "".join(c for c in text if c in ARABIC_LETTERS or c in ARABIC_DIACRITICS)


def _is_format_char(ch: str) -> bool:
    """Unicode ``Cf`` — zero-width space / joiners, bidi marks, BOM, soft hyphen, ALM."""
    return unicodedata.category(ch) == "Cf"


def normalize_text(text: str) -> str:
    """The one text normalisation of the tokenizer: NFKC, then format characters dropped.

    Shared by ``encode`` and the corpus pre-pass so the chunk universe (and
    therefore the analyzer cache keys) agree. Format characters carry no
    text: U+200B is the most common one in web Arabic (70 occurrences in the
    250 held-out CIDAR references, always at a word start) and used to
    reach ``_encode_word`` as an unknown character → ``<unk>`` → ``?`` on
    decode. NFKC already folds U+FEFF-style compatibility forms of letters;
    it does not touch the ``Cf`` class.
    """
    text = unicodedata.normalize("NFKC", text)
    if any(_is_format_char(c) for c in text):
        text = "".join(c for c in text if not _is_format_char(c))
    return text


def _is_arabic_alpha(ch: str) -> bool:
    return ch in ARABIC_LETTERS or ch in ARABIC_DIACRITICS


def _classify_char(ch: str) -> str:
    """Return one of {'alpha', 'fem', 'digit', 'punct', 'space', 'other'}.

    'fem' is the tāʾ marbūṭa: it closes the alpha run it follows (see
    ``_split_runs``) so it can never sit in the middle of a chunk.
    """
    if ch == TAA_MARBUTA:
        return "fem"
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

def _check_closed_inventories(prepositions: Tuple[str, ...], func_words: Tuple[str, ...]) -> None:
    """The two closed groups must be duplicate-free and must not shadow each other.

    Exact duplicates (within or across the groups) are an error: one
    surface, one token. Across the groups a collision *modulo alef
    variants* is an error too: the intercept checks [PREP_*] first, so an
    alef-folded input would never reach the [FUNC_*] entry. Within one
    group a fold collision is allowed (أن and إن are different words that
    both belong here); an exact spelling always wins, and a folded input
    (ان) resolves to the entry that sorts first by code point (أ < إ < آ).
    """
    from arabic_eval.tokenizers.araroopat_backend import _alef_norm
    exact: Dict[str, str] = {}
    folded: Dict[str, Tuple[str, str]] = {}
    for group, items in (("prepositions", prepositions), ("func_words", func_words)):
        for w in items:
            if w in exact:
                raise ValueError(
                    f"araroopat: {group!r} entry {w!r} is already listed under {exact[w]!r}; "
                    "each closed-class surface may be listed once."
                )
            exact[w] = group
            key = _alef_norm(w)
            prev = folded.get(key)
            if prev and prev[0] != group:
                raise ValueError(
                    f"araroopat: {group!r} entry {w!r} collides with {prev[0]}:{prev[1]!r} modulo alef "
                    "variants; [PREP_*] is matched first and would shadow the [FUNC_*] entry."
                )
            folded.setdefault(key, (group, w))


@tokenizer_registry.register("araroopat")
class AraRooPatTokenizer(BaseTokenizer):
    """Arabic Roots & Patterns tokenizer (see module docstring)."""

    def __init__(self, **kwargs: Any) -> None:
        # Configurable params (also exposed via configs/tokenizers/araroopat.yaml).
        self.max_roots: int = int(kwargs.get("max_roots", DEFAULTS["max_roots"]))
        self.max_patterns: int = int(kwargs.get("max_patterns", DEFAULTS["max_patterns"]))
        self.min_root_freq: int = int(kwargs.get("min_root_freq", DEFAULTS["min_root_freq"]))
        self.min_pattern_freq: int = int(kwargs.get("min_pattern_freq", DEFAULTS["min_pattern_freq"]))
        self.generator_timeout_ms: int = int(kwargs.get("generator_timeout_ms", DEFAULTS["generator_timeout_ms"]))
        self.use_diacritized_surface: bool = bool(kwargs.get("use_diacritized_surface", DEFAULTS["use_diacritized_surface"]))
        self.cache_corpus_analysis: bool = bool(kwargs.get("cache_corpus_analysis", DEFAULTS["cache_corpus_analysis"]))
        self.add_bos_eos: bool = bool(kwargs.get("add_bos_eos", DEFAULTS["add_bos_eos"]))
        # Closed-class words emitted as a single [PREP_*] token. Order is
        # the vocab ID order. See PREPOSITION_INVENTORY in the backend.
        self.prepositions: Tuple[str, ...] = tuple(
            kwargs.get("prepositions") or PREPOSITION_INVENTORY
        )
        # Closed-class function words emitted as a single [FUNC_*] token
        # (pronouns, demonstratives, relatives, conjunctions, interrogatives,
        # particles). Fixed order = vocab ID order; the range follows the
        # [PREP_*] range. See FUNC_INVENTORY in the backend.
        self.func_words: Tuple[str, ...] = tuple(
            kwargs.get("func_words") or FUNC_INVENTORY
        )
        _check_closed_inventories(self.prepositions, self.func_words)
        # Closed-list clitic peeler for clitic *combinations* the CAMeL
        # database has no row for (interrogative أ, double object pronouns,
        # classical كمو/همو). Runs only on a native CAMeL miss. See
        # ``peel_candidates`` in the backend.
        self.clitic_peeler: bool = bool(kwargs.get("clitic_peeler", DEFAULTS["clitic_peeler"]))
        # Also accept a bare ا as the interrogative (alef-normalised text).
        # Off by default — see BARE_ALEF_INTERROGATIVE in the backend.
        self.peel_bare_alef: bool = bool(kwargs.get("peel_bare_alef", DEFAULTS["peel_bare_alef"]))
        # Proper nouns (CAMeL database noun_prop) between [PROP_BEGIN] /
        # [PROP_END]: "unrooted" (default) or "all". Applied at vocab-build
        # and encode time, never in the pre-pass cache, so switching it
        # re-runs only the vocab build.
        self.proper_nouns: str = str(kwargs.get("proper_nouns") or DEFAULTS["proper_nouns"])
        if self.proper_nouns not in PROPER_NOUN_MODES:
            raise ValueError(
                f"araroopat: proper_nouns must be one of {PROPER_NOUN_MODES}, got {self.proper_nouns!r}"
            )

        # State.
        self._backend: Optional[MorphAnalyzer] = None
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._special_token_map: Dict[str, int] = {}
        # (root_id, pattern_id) -> surface form (cleaned or diacritized per config)
        self._reconstruction: Dict[Tuple[int, int], str] = {}
        # Provenance: per-token-string metadata.
        self._metadata: Dict[str, Any] = {"roots": {}, "patterns": {}, "config": {}}

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        d = DEFAULTS
        return [
            ParamSpec("max_roots", "int", d["max_roots"], min=1, group="budget",
                      help="Cap on [ROOT_*] tokens (most frequent roots first). Never binding on ArabicText-Large: only "
                           "~4 100 roots clear min_root_freq."),
            ParamSpec("max_patterns", "int", d["max_patterns"], min=1, group="budget",
                      help="Cap on [PAT_*] tokens — the binding budget (keeping # in roots moves the weak letter into the "
                           "pattern). The class default 500 is a stale legacy value; the Balanced tier every experiment uses "
                           "is 4000 (configs/tokenizers/araroopat.yaml), Compact 1000, Max 6076."),
            ParamSpec("min_root_freq", "int", d["min_root_freq"], min=1, group="budget",
                      help="A root needs this many corpus occurrences to enter the vocabulary."),
            ParamSpec("min_pattern_freq", "int", d["min_pattern_freq"], min=1, group="budget",
                      help="A pattern needs this many corpus occurrences to enter the vocabulary."),
            ParamSpec("proper_nouns", "str", d["proper_nouns"], choices=PROPER_NOUN_MODES, group="analysis",
                      help="Names CAMeL's database knows: 'unrooted' sends only the rootless ones between [PROP_BEGIN]/"
                           "[PROP_END] (rooted names keep ROOT+PAT), 'all' sends every name there (+2.7 % fertility)."),
            ParamSpec("clitic_peeler", "bool", d["clitic_peeler"], group="analysis",
                      help="On a native CAMeL miss, strip clitic combinations the database lacks (interrogative أ, double "
                           "object pronouns, كمو/همو) from a closed list and re-analyse the residual."),
            ParamSpec("peel_bare_alef", "bool", d["peel_bare_alef"], group="analysis",
                      help="Also accept a bare ا as the interrogative prefix (for alef-normalised text); 1.1 % false peels "
                           "when on, keep off with raw hamza text."),
            ParamSpec("prepositions", "list[str]", list(PREPOSITION_INVENTORY), group="analysis",
                      help="Closed-class prepositions emitted as one [PREP_*] token each (fixed order = vocab id order); "
                           "matched alef-insensitively on CAMeL's lemma."),
            ParamSpec("func_words", "list[str]", list(FUNC_INVENTORY), group="analysis",
                      help="Closed-class function words emitted as one [FUNC_*] token each (pronouns, demonstratives, "
                           "relatives, conjunctions, particles); a surface may not also be a preposition."),
            ParamSpec("use_diacritized_surface", "bool", d["use_diacritized_surface"], group="decode",
                      help="Store the diacritized stem in the reconstruction table instead of the cleaned one."),
            ParamSpec("generator_timeout_ms", "int", d["generator_timeout_ms"], min=1, advanced=True, group="decode",
                      help="Per-call timeout of CAMeL's Generator (tier 2 of the decode resolver, unseen root/pattern pairs)."),
            ParamSpec("add_bos_eos", "bool", d["add_bos_eos"], advanced=True, group="misc",
                      help="Wrap every encoding in <s> … </s>."),
            ParamSpec("cache_corpus_analysis", "bool", d["cache_corpus_analysis"], advanced=True, group="misc",
                      help="Reuse outputs/tokenizers/araroopat_cache/corpus_analysis.pkl for chunks the pre-pass already "
                           "analysed (hours of CAMeL otherwise)."),
        ]

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> MorphAnalyzer:
        if self._backend is None:
            self._backend = MorphAnalyzer(
                generator_timeout_ms=self.generator_timeout_ms,
                particles=frozenset(self.prepositions),
                enable_peeler=self.clitic_peeler,
                peel_bare_alef=self.peel_bare_alef,
                func_words=frozenset(self.func_words),
            )
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
        entries, chunk_counts = self._corpus_prepass(texts, cache_dir)

        # ---- Step 2: frequency tables ----
        root_freq: Counter = Counter()
        pat_freq: Counter = Counter()
        proclitic_freq: Counter = Counter()
        enclitic_freq: Counter = Counter()
        particle_freq: Counter = Counter()
        func_freq: Counter = Counter()
        proper_count = 0
        for e in entries:
            if not e.analyzed:
                continue
            if e.particle:
                (func_freq if e.particle_kind == "func" else particle_freq)[e.particle] += 1
            elif self._routes_to_prop(e.proper, e.root):
                proper_count += 1   # characters between [PROP_*]: no root / pattern to count
            elif e.root and e.pattern:
                root_freq[e.root] += 1
                pat_freq[e.pattern] += 1
            elif e.clitic_only:
                pass   # nothing but clitics: they count into the two tables below
            else:
                continue
            for c in e.proclitics:
                proclitic_freq[c] += 1
            for c in e.enclitics:
                enclitic_freq[c] += 1

        logger.info(
            "Pre-pass stats: %d analyzed words (%d prepositions, %d function words, %d proper nouns on "
            "[PROP_*], mode %s), %d unique roots, %d unique patterns, %d proclitic surfaces, "
            "%d enclitic surfaces.",
            sum(1 for e in entries if e.analyzed), sum(particle_freq.values()), sum(func_freq.values()),
            proper_count, self.proper_nouns, len(root_freq), len(pat_freq),
            len(proclitic_freq), len(enclitic_freq),
        )

        # ---- Step 3: assemble flat vocab in deterministic ID order ----
        self._build_vocab(root_freq, pat_freq, proclitic_freq, enclitic_freq)

        # ---- Step 4: reconstruction lookup from observed (root, pattern) ----
        self._build_reconstruction(entries, chunk_counts)

        # ---- Step 5: provenance metadata ----
        self._build_metadata(root_freq, pat_freq, proclitic_freq, enclitic_freq,
                             entries, particle_freq, func_freq)

        logger.info("AraRooPat trained — vocab size: %d", self.vocab_size)
        logger.info(
            "  roots: %d, patterns: %d, proclitics: %d, enclitics: %d, "
            "fixed slots: %d, reconstruction entries: %d",
            sum(1 for t in self._vocab if t.startswith(PFX_ROOT)),
            sum(1 for t in self._vocab if t.startswith(PFX_PAT)),
            sum(1 for t in self._vocab if t.startswith(PFX_CLITICP)),
            sum(1 for t in self._vocab if t.startswith(PFX_CLITICE)),
            sum(1 for t in self._vocab
                if t.startswith((PFX_PREP, PFX_FUNC, PFX_CHAR, PFX_PUNCT, PFX_DIGIT, "<", "[L"))),
            len(self._reconstruction),
        )

    def _routes_to_prop(self, proper: bool, root: Optional[str]) -> bool:
        """Does a proper-noun analysis take the [PROP_*] path under ``proper_nouns``?"""
        return bool(proper) and (self.proper_nouns == "all" or not root)

    # Bump when the post-processing in araroopat_backend changes shape
    # (_dict_to_analysis, normalize_pattern, strip_proclitics_from_start, ...).
    # 5: [FUNC_*] group + alef-insensitive matching; 6: clitic-only words; 7: no
    # peel onto them; 8: database proper nouns kept as `proper` entries (2026-09-16);
    # 9: spelling-faithful candidate walk + ة/ه slot reconciliation; 10: an alef-inexact
    # closed-class match needs a closed-class POS (كان is not كأن) — both 2026-09-22
    _CACHE_FORMAT = 10
    # Formats a cache can be *migrated* from instead of discarded: the entries the
    # newer rules can change are re-analysed (``_stale_cache_entry``), every other
    # entry is reused as is. 8 → 10 touches the rooted entries whose surface does
    # not spell the word (~92 K of the corpus' 1.15 M chunks) plus the particle
    # entries matched modulo alef (~600); 9 → 10 only the latter. Minutes over the
    # bridge against hours for a full pre-pass.
    _MIGRATABLE_FORMATS = (8, 9)

    def _cache_key(self) -> Tuple[Any, ...]:
        return (self._CACHE_FORMAT, tuple(self.prepositions), self.clitic_peeler,
                self.peel_bare_alef, tuple(self.func_words))

    def _reusable_cache_entries(self, payload: Any) -> Tuple[List[CorpusEntry], str]:
        """The entries of a loaded cache payload this tokenizer may reuse, and why.

        Returns ``(entries, "match")`` when the key is ours, ``(entries minus
        the ones the newer rule can change, "migrate")`` when only the format
        differs and is in ``_MIGRATABLE_FORMATS``; raises ``ValueError`` for
        anything else (a different inventory or peeler setting, a stale format).
        Shared with the explorer's cache card so both report the same decision.
        """
        if not isinstance(payload, dict):
            raise ValueError("cache format/key mismatch")
        key = payload.get("key")
        mine = self._cache_key()
        if key == mine:
            return list(payload["entries"]), "match"
        if (isinstance(key, tuple) and len(key) == len(mine) and key[1:] == mine[1:]
                and key[0] in self._MIGRATABLE_FORMATS):
            kept = [e for e in payload["entries"] if not _stale_cache_entry(e, key[0])]
            return kept, "migrate"
        raise ValueError("cache format/key mismatch")

    def _corpus_prepass(self, texts: List[str], cache_dir: Path) -> Tuple[List[CorpusEntry], Counter]:
        """Analyze every distinct *alpha chunk* in the corpus once. Cache to disk.

        We chunk words by character class (matching what ``_encode_word`` does
        at runtime) so the cache keys match the analyze() calls made during
        encoding — no analyzer hits at runtime if the cache covers the corpus.

        Returns the entries (corpus order) and the chunk → occurrence counter
        the reconstruction table weights its written surfaces by.
        """
        backend = self._ensure_backend()

        word_counts: Counter = Counter()
        for t in texts:
            t = normalize_text(t)
            for w in t.split():
                for chunk in _extract_alpha_chunks(w):
                    word_counts[chunk] += 1
        unique_words = list(word_counts.keys())

        cache_file = cache_dir / "corpus_analysis.pkl"
        # A cache whose key matches is reused for every chunk it holds; only
        # the chunks it lacks go to CAMeL, and the union is written back. An
        # analysis is per word, so a partial cache is never stale — only
        # incomplete. (Before 2026-09-16 one missing chunk re-ran the whole
        # pre-pass: hours for the corpus after a preprocessing tweak.)
        cached_by_word: Dict[str, CorpusEntry] = {}
        to_analyze = unique_words
        if self.cache_corpus_analysis and cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    payload = pickle.load(f)
                # The cache stores *post-processed* entries, so it is only
                # valid for the analysis logic + preposition inventory that
                # produced it. A migratable older format keeps every entry
                # the newer rule cannot change; anything else re-runs the
                # pre-pass rather than silently training on stale analyses.
                cached, decision = self._reusable_cache_entries(payload)
                if decision == "migrate":
                    logger.info("Cache is format %s — migrating: %d of %d entries reused, the rest re-analysed.",
                                payload["key"][0], len(cached), len(payload["entries"]))
                cached_by_word = {e.word: e for e in cached}
                missing = [w for w in unique_words if w not in cached_by_word]
                if not missing:
                    logger.info("Loaded cached corpus analysis (%d entries) from %s",
                                len(cached), cache_file)
                    # Filter to current vocabulary universe; expand counts.
                    return [e for e in cached if e.word in word_counts], word_counts
                logger.info("Cache covers %d of %d unique chunks — analyzing only the %d missing ones.",
                            len(unique_words) - len(missing), len(unique_words), len(missing))
                to_analyze = missing
            except Exception as e:
                logger.warning("Cache load failed (%s) — re-running pre-pass.", e)
                cached_by_word = {}
                to_analyze = unique_words

        # Run analyzer in batches via the bridge — saves one IPC round-trip
        # per cache miss vs the old per-word loop.
        new_entries: List[CorpusEntry] = []
        analyzed_count = 0
        batch_size = 256
        with tqdm(total=len(to_analyze), desc="CAMeL pre-pass", unit="word") as pbar:
            for start in range(0, len(to_analyze), batch_size):
                batch = to_analyze[start:start + batch_size]
                analyses = backend.analyze_many(batch, batch_size=batch_size)
                for word, a in zip(batch, analyses):
                    new_entries.append(CorpusEntry.from_analysis(word, a))
                    if a is not None:
                        analyzed_count += 1
                pbar.update(len(batch))

        logger.info("Pre-pass: analyzed %d / %d words (%.1f%%)",
                    analyzed_count, len(to_analyze),
                    100.0 * analyzed_count / max(len(to_analyze), 1))
        # Corpus order (unique_words) regardless of what came from the cache,
        # so a partial reuse yields exactly the entries a fresh run would.
        merged = {**cached_by_word, **{e.word: e for e in new_entries}}
        entries: List[CorpusEntry] = [merged[w] for w in unique_words]
        if self.clitic_peeler:
            logger.info(
                "Pre-pass: clitic peeler rescued %d words after a native CAMeL miss "
                "(%d exhausted -> character path). Per-word detail at DEBUG level.",
                backend.peel_stats["peeled"], backend.peel_stats["exhausted"],
            )

        if self.cache_corpus_analysis:
            # Union of what the cache held and what was analyzed now, so the
            # file keeps growing towards a superset of every corpus seen.
            to_store = list(merged.values())
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                pickle.dump({"key": self._cache_key(), "entries": to_store}, f)
            # JSON view for quick inspection.
            with (cache_dir / "corpus_analysis.json").open("w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in entries[:10000]], f,
                          ensure_ascii=False, indent=2)
            logger.info("Cached corpus analysis to %s (%d entries, + first-10k JSON view)",
                        cache_file, len(to_store))

        return entries, word_counts

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
        # 4–5: literal markers; 6–7: proper-noun markers.
        add(TOK_LIT_BEGIN)
        add(TOK_LIT_END)
        add(TOK_PROP_BEGIN)
        add(TOK_PROP_END)
        # 6+: proclitics (CLITICP), then enclitics (CLITICE) — sorted by
        # freq desc, ties broken alphabetically. Distinct ranges so the
        # decoder never has to disambiguate prc vs enc from surface alone.
        for clitic, freq in sorted(proclitic_freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < 1 or not clitic:
                continue
            add(f"{PFX_CLITICP}{clitic}{SFX}")
        # [CLITICE_ة] is a fixed slot at the head of the enclitic range: the
        # LIT path relies on it for every ة-final word, corpus or not.
        add(f"{PFX_CLITICE}{TAA_MARBUTA}{SFX}")
        for clitic, freq in sorted(enclitic_freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < 1 or not clitic:
                continue
            add(f"{PFX_CLITICE}{clitic}{SFX}")
        # Then: prepositions / particles, in the configured (fixed) order —
        # not frequency-sorted, so IDs don't depend on the corpus.
        for prep in self.prepositions:
            add(f"{PFX_PREP}{prep}{SFX}")
        # Then: function words, same fixed-order rule.
        for w in self.func_words:
            add(f"{PFX_FUNC}{w}{SFX}")
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

    def _build_reconstruction(self, entries: List[CorpusEntry],
                              word_counts: Optional[Counter] = None) -> None:
        """Build (root_id, pat_id) -> *inflected-stem* surface for every distinct pair.

        Important nuance: CAMeL's ``stem`` field gives the **lexical** stem
        (root letters in their pattern slots only — no inflectional prefixes
        like the present-tense ي of يدرس). What we want for reconstruction
        is the **inflected stem** — the surface with clitics stripped but
        inflection retained.

        Where that surface comes from (``entry_realization``): with
        ``use_diacritized_surface`` off, from the chunk **as the corpus wrote
        it** — clitics stripped from the written word, accepted only when the
        decoder's own joins give the word back — so a pair decodes to the
        spelling the writers used (الإعرابية, not CAMeL's أَعْرابِيَّة; see
        the backend's *spelling-faithful* section). CAMeL's ``diac`` is the
        fallback for an entry whose written form does not reproduce, and the
        only source when the diacritized surface is requested. The most
        frequent written form wins, weighted by corpus occurrences when
        ``word_counts`` is given (chunk types otherwise); ``diac`` forms are
        consulted only for pairs with no written form at all, then the
        generator, then naive slot substitution.
        """
        reco: Dict[Tuple[int, int], str] = {}
        backend = self._ensure_backend()
        weight = (lambda w: word_counts.get(w, 1)) if word_counts else (lambda w: 1)

        # Pass 1: collect inflected-stem realizations per (root, pattern_bare),
        # written forms and diac-derived forms in separate counters.
        written: Dict[Tuple[str, str], Counter] = {}
        diac_forms: Dict[Tuple[str, str], Counter] = {}
        all_pairs: set = set()
        for e in entries:
            if not (e.analyzed and e.root and e.pattern):
                continue
            root_tok = f"{PFX_ROOT}{e.root}{SFX}"
            pat_tok = f"{PFX_PAT}{e.pattern}{SFX}"
            if root_tok not in self._vocab or pat_tok not in self._vocab:
                continue
            all_pairs.add((e.root, e.pattern))
            real = entry_realization(e, self.use_diacritized_surface)
            if real is None:
                continue
            form, source = real
            (written if source == "written" else diac_forms).setdefault(
                (e.root, e.pattern), Counter())[form] += weight(e.word)

        # Pass 2: the most frequent written form, else the most frequent diac form.
        unresolved: List[Tuple[str, str]] = []
        for (root, pat) in all_pairs:
            form = pick_realization(written.get((root, pat)), diac_forms.get((root, pat)))
            if form is None:
                unresolved.append((root, pat))
                continue
            root_id = self._vocab[f"{PFX_ROOT}{root}{SFX}"]
            pat_id = self._vocab[f"{PFX_PAT}{pat}{SFX}"]
            reco[(root_id, pat_id)] = form

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
        n_written = sum(1 for k in all_pairs if k in written)
        logger.info("Reconstruction: %d pairs — %d from written corpus forms, %d from CAMeL diac, %d generated",
                    len(all_pairs), n_written, len(all_pairs) - n_written - len(unresolved), len(unresolved))

    def _build_metadata(
        self,
        root_freq: Counter,
        pat_freq: Counter,
        proclitic_freq: Counter,
        enclitic_freq: Counter,
        entries: List[CorpusEntry],
        particle_freq: Optional[Counter] = None,
        func_freq: Optional[Counter] = None,
    ) -> None:
        """Provenance: which roots/patterns came from which words, with examples."""
        particle_freq = particle_freq or Counter()
        func_freq = func_freq or Counter()
        # Build root → example words map (up to 5 each).
        # Hoist the two vocab-derived sets out of the loop: they were being
        # rebuilt per entry, which is O(entries x vocab). At 506k analyzed
        # entries and an 11k vocab that is ~11e9 string operations — hours.
        vocab_roots = self._vocab_root_set()
        vocab_patterns = self._vocab_pattern_set()
        root_examples: Dict[str, List[str]] = {}
        pat_examples: Dict[str, List[Tuple[str, str]]] = {}
        for e in entries:
            if not (e.analyzed and e.root and e.pattern):
                continue
            if e.root in vocab_roots:
                lst = root_examples.setdefault(e.root, [])
                if len(lst) < 5 and e.word not in lst:
                    lst.append(e.word)
            if e.pattern in vocab_patterns:
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
            "prepositions": {
                prep: {"id": self._vocab[f"{PFX_PREP}{prep}{SFX}"],
                       "freq": particle_freq.get(prep, 0)}
                for prep in self.prepositions
            },
            "func_words": {
                w: {"id": self._vocab[f"{PFX_FUNC}{w}{SFX}"],
                    "freq": func_freq.get(w, 0)}
                for w in self.func_words
            },
            # Proper nouns: database noun_prop readings, by path under the mode.
            "proper_nouns": {
                "mode": self.proper_nouns,
                "count": sum(1 for e in entries if e.proper),
                "unrooted": sum(1 for e in entries if e.proper and not e.root),
                "rooted": sum(1 for e in entries if e.proper and e.root),
                "on_prop_path": sum(1 for e in entries if self._routes_to_prop(e.proper, e.root)),
                "examples": [
                    {"word": e.word, "root": e.root, "pattern": e.pattern,
                     "proclitics": list(e.proclitics), "enclitics": list(e.enclitics)}
                    for e in [x for x in entries if x.proper][:50]
                ],
            },
            "config": {
                "prepositions": list(self.prepositions),
                "func_words": list(self.func_words),
                "proper_nouns": self.proper_nouns,
                "max_roots": self.max_roots,
                "max_patterns": self.max_patterns,
                "min_root_freq": self.min_root_freq,
                "min_pattern_freq": self.min_pattern_freq,
                "use_diacritized_surface": self.use_diacritized_surface,
                "generator_timeout_ms": self.generator_timeout_ms,
                "clitic_peeler": self.clitic_peeler,
                "peel_bare_alef": self.peel_bare_alef,
            },
            # Pronoun-hosted prepositions emitted as clitic tokens only.
            "clitic_only": {
                "count": sum(1 for e in entries if e.clitic_only),
                "examples": [
                    {"word": e.word, "proclitics": list(e.proclitics), "enclitics": list(e.enclitics)}
                    for e in [x for x in entries if x.clitic_only][:50]
                ],
            },
            # Provenance for the clitic peeler: how many corpus words only
            # analyze because of it, with examples for auditing false peels.
            "peeled": {
                "count": sum(1 for e in entries if e.peeled),
                "examples": [
                    {"word": e.word, "proclitics": list(e.proclitics),
                     "enclitics": list(e.enclitics), "root": e.root,
                     "pattern": e.pattern, "particle": e.particle,
                     "particle_kind": e.particle_kind}
                    for e in [x for x in entries if x.peeled][:200]
                ],
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

        text = normalize_text(text)
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
        # Split into runs by character class (shared with the pre-pass).
        for cls, chunk in _split_runs(word):
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

    def _emit_alpha(self, chunk: str, ids: List[int], toks: List[str]) -> None:
        """Emit tokens for an Arabic alphabetic chunk: try analyzer, fall back to LIT."""
        backend = self._ensure_backend()
        a: Optional[Analysis] = backend.analyze(chunk)

        if a is not None and a.particle:
            pfx = PFX_FUNC if a.particle_kind == "func" else PFX_PREP
            prep_tok = f"{pfx}{a.particle}{SFX}"
            proc = a.proclitics
            enc = a.pronoun_enclitics
            # Commit to the PREP/FUNC path only if every token it needs exists;
            # a half-tokenized word (LIT clitic + PREP) would decode with a
            # space in it, so an OOV clitic sends the whole chunk to LIT.
            needed = [prep_tok] + [f"{PFX_CLITICP}{c}{SFX}" for c in proc] \
                + [f"{PFX_CLITICE}{c}{SFX}" for c in enc]
            if all(t in self._vocab for t in needed):
                for c in proc:
                    self._emit_clitic(c, ids, toks, kind="p")
                ids.append(self._vocab[prep_tok])
                # Metric string = the clitic-stripped *surface* (علي for
                # عليه, not the lemma على) so the cleaned tokens still
                # concatenate back into the word for aligned_token_offsets.
                core = _strip_clitic_surfaces(strip_diacritics(a.surface or chunk), proc, enc)
                toks.append(_clean_arabic(core) or _clean_arabic(a.particle))
                for c in enc:
                    self._emit_clitic(c, ids, toks, kind="e")
                return
            self._emit_lit(chunk, ids, toks)
            return

        if a is not None and a.clitic_only:
            # Pronoun-hosted preposition (له, بها, ولهم): the clitic tokens
            # and nothing else. Same all-or-nothing rule — an OOV clitic
            # token sends the whole chunk to LIT.
            proc = a.proclitics
            enc = a.pronoun_enclitics
            needed = [f"{PFX_CLITICP}{c}{SFX}" for c in proc] + [f"{PFX_CLITICE}{c}{SFX}" for c in enc]
            if proc and enc and all(t in self._vocab for t in needed):
                for c in proc:
                    self._emit_clitic(c, ids, toks, kind="p")
                for c in enc:
                    self._emit_clitic(c, ids, toks, kind="e")
                return
            self._emit_lit(chunk, ids, toks)
            return

        if a is not None and self._routes_to_prop(a.proper, a.root):
            # Proper noun (database noun_prop): characters between the
            # [PROP_*] markers, clitics outside. Never falls to LIT — the
            # markers are fixed slots.
            self._emit_prop(chunk, ids, toks, a.proclitics, a.pronoun_enclitics)
            return

        if a is not None and a.root and a.pattern:
            root_tok = f"{PFX_ROOT}{a.root}{SFX}"
            pat_tok = f"{PFX_PAT}{a.pattern}{SFX}"
            proc = a.proclitics
            enc = a.enclitics
            # Same all-or-nothing rule as the PREP path: an OOV clitic token
            # would be emitted as a literal, which decodes with a space in
            # it — so the whole chunk goes to LIT instead (still reversible).
            needed = [root_tok, pat_tok] \
                + [f"{PFX_CLITICP}{c}{SFX}" for c in proc] \
                + [f"{PFX_CLITICE}{c}{SFX}" for c in enc]
            if all(t in self._vocab for t in needed):
                for c in proc:
                    self._emit_clitic(c, ids, toks, kind="p")
                ids.append(self._vocab[root_tok])
                toks.append(_clean_arabic(a.root))
                ids.append(self._vocab[pat_tok])
                # PAT metric-string = cleaned inflected stem (full surface
                # minus clitic chars). Contains root letters in their
                # pattern-positioned context plus inflectional morphology
                # (e.g. ي of present-tense verbs). Needed for
                # pattern_conservation_rate.
                inflected = _strip_clitic_surfaces(a.surface or chunk, proc, enc)
                toks.append(_clean_arabic(strip_diacritics(inflected or chunk)))
                pronouns = a.pronoun_enclitics
                if a.fem:
                    # Metric string = the surface realization (ت before a
                    # pronoun) so the tokens still concatenate to the word.
                    self._emit_clitic(a.fem, ids, toks, kind="e",
                                      metric="ت" if pronouns else TAA_MARBUTA)
                for c in pronouns:
                    self._emit_clitic(c, ids, toks, kind="e")
                return
            # Root, pattern or a clitic is OOV — fall through to LIT, or
            # to PROP when the word is a name whose root the budget cut.
            if a.proper:
                self._emit_prop(chunk, ids, toks, proc, a.pronoun_enclitics)
                return

        self._emit_lit(chunk, ids, toks)

    def _emit_prop(self, chunk: str, ids: List[int], toks: List[str],
                   proclitics: Tuple[str, ...] = (), pronouns: Tuple[str, ...] = ()) -> None:
        """Emit a proper noun: clitic tokens outside, its characters between [PROP_*].

        ``بمكة`` → ``[CLITICP_ب] [PROP_BEGIN] [CHAR_م] [CHAR_ك] [PROP_END] [CLITICE_ة]``.
        The split is accepted only when the decoder's own joins reproduce
        the chunk exactly (proclitics joined with the لِ+الـ contraction +
        core + pronouns) and every clitic token exists; otherwise the whole
        chunk goes between the markers unsplit — still a PROP, still
        reversible. A tokenizer saved before the markers existed falls
        back to the plain literal.
        """
        if TOK_PROP_BEGIN not in self._vocab or TOK_PROP_END not in self._vocab:
            self._emit_lit(chunk, ids, toks)
            return
        proc: Tuple[str, ...] = tuple(c for c in proclitics if c)
        enc: Tuple[str, ...] = tuple(c for c in pronouns if c)
        core = _strip_clitic_surfaces(chunk, proc, enc)
        clitic_toks = [f"{PFX_CLITICP}{c}{SFX}" for c in proc] + [f"{PFX_CLITICE}{c}{SFX}" for c in enc]
        if (proc or enc) and not (
            core and join_proclitics(list(proc)) + core + "".join(enc) == chunk
            and all(t in self._vocab for t in clitic_toks)
        ):
            proc, enc, core = (), (), chunk
        for c in proc:
            self._emit_clitic(c, ids, toks, kind="p")
        fem = core.endswith(TAA_MARBUTA) and f"{PFX_CLITICE}{TAA_MARBUTA}{SFX}" in self._vocab
        if fem:
            core = core[:-1]
        ids.append(self._vocab[TOK_PROP_BEGIN])
        toks.append("")
        for ch in core:
            tok = f"{PFX_CHAR}{ch}{SFX}"
            if tok in self._vocab:
                ids.append(self._vocab[tok])
                toks.append(ch if ch in ARABIC_LETTERS else "")
            else:
                ids.append(self._special_token_map["unk_token"])
                toks.append("")
        ids.append(self._vocab[TOK_PROP_END])
        toks.append("")
        if fem:
            # ة is written ت before a pronoun; the enclitic token is the same.
            self._emit_clitic(TAA_MARBUTA, ids, toks, kind="e", metric="ت" if enc else TAA_MARBUTA)
        for c in enc:
            self._emit_clitic(c, ids, toks, kind="e")

    def _emit_clitic(self, clitic: str, ids: List[int], toks: List[str],
                     kind: str, metric: Optional[str] = None) -> None:
        """Emit a clitic with explicit kind ('p'=proclitic, 'e'=enclitic)."""
        prefix = PFX_CLITICP if kind == "p" else PFX_CLITICE
        tok = f"{prefix}{clitic}{SFX}"
        if tok in self._vocab:
            ids.append(self._vocab[tok])
            toks.append(_clean_arabic(metric if metric is not None else clitic))
        else:
            # Unknown clitic surface for that kind — emit as literal chars.
            self._emit_lit(clitic, ids, toks)

    def _emit_lit(self, chunk: str, ids: List[int], toks: List[str]) -> None:
        """Emit chunk as [LIT_BEGIN] CHAR... [LIT_END], then [CLITICE_ة] if it ends in ة.

        There is no [CHAR_ة]: the tāʾ marbūṭa is the enclitic on every path.
        """
        # A tokenizer saved before the fixed [CLITICE_ة] slot has [CHAR_ة]
        # instead; keep the ة inside the literal there, or the OOV-clitic
        # fallback below would recurse (_emit_clitic → _emit_lit("ة") → ...).
        fem = chunk.endswith(TAA_MARBUTA) and f"{PFX_CLITICE}{TAA_MARBUTA}{SFX}" in self._vocab
        if fem:
            chunk = chunk[:-1]
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
        if fem:
            self._emit_clitic(TAA_MARBUTA, ids, toks, kind="e")

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
        # Bare surface of the [PREP_*] / [FUNC_*] just flushed as out[-1] (None once
        # anything else is emitted). Enclitics attach to prepositions with
        # orthographic adjustments (إلى+ه → إليه, من+ما → مما) that must
        # not fire on nouns (مستشفى+ه → مستشفاه, a different rule).
        last_particle: Optional[str] = None

        def attach_enclitic(s: str) -> None:
            """Append enclitic surface to the just-emitted word, or drop it."""
            nonlocal last_particle
            if clitic_prefix:
                # Proclitics with no host yet + an enclitic = a clitic-only
                # word (له = ل + ه, ولهم = و + ل + هم). Close them into a
                # word of their own rather than attaching the pronoun to
                # the previous word.
                out.append(join_proclitics(clitic_prefix) + s)
                clitic_prefix.clear()
                last_particle = None
                return
            if out and last_particle is not None:
                prefix = out[-1][: len(out[-1]) - len(last_particle)]
                out[-1] = prefix + join_particle_enclitic(last_particle, s)
            elif out:
                # A ة-final word takes a pronoun with ة → ت (مدرسة + ه → مدرسته).
                if out[-1].endswith(TAA_MARBUTA) and s != TAA_MARBUTA:
                    out[-1] = out[-1][:-1] + "ت"
                out[-1] = out[-1] + s
            else:
                out.append(s)
            last_particle = None

        def flush_word(word: str) -> None:
            """Emit a content-bearing word, prepending any buffered proclitics."""
            nonlocal last_particle
            out.append(join_proclitics(clitic_prefix) + word)
            clitic_prefix.clear()
            last_particle = None

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

            # [PROP_*] markers decode exactly like the literal markers: the
            # characters between them are one word (either END closes
            # either BEGIN, so a malformed stream still flushes).
            if tok in (TOK_LIT_BEGIN, TOK_PROP_BEGIN):
                dump_orphan_root()
                in_lit = True
                lit_buffer.clear()
                continue

            if tok in (TOK_LIT_END, TOK_PROP_END):
                # Always flush, even when empty: a lone ة encodes as an empty
                # literal + [CLITICE_ة], and the enclitic must attach to it
                # rather than to the previous word. Empty entries are dropped
                # at the join below.
                flush_word("".join(lit_buffer))
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

            if tok.startswith((PFX_PREP, PFX_FUNC)):
                # Both closed groups decode the same way: the word itself,
                # then a following enclitic joins with the particle rules.
                dump_orphan_root()
                pfx = PFX_PREP if tok.startswith(PFX_PREP) else PFX_FUNC
                prep = tok[len(pfx):-len(SFX)]
                flush_word(prep)
                last_particle = prep
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
            out.append(join_proclitics(clitic_prefix))

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
    # Surfaces per token (embedding initialisation)
    # ------------------------------------------------------------------

    def token_surfaces(self) -> Dict[int, List[Tuple[str, float]]]:
        """What each id stands for, for ``model.embedding_init: surface_avg``.

        * ``[ROOT_r]`` → the reconstruction surface of every (r, p) pair in the
          table, weighted by the pattern's corpus frequency (``vocab_metadata``;
          1 when unknown): the words the root appears in, in proportion.
        * ``[PAT_p]`` → the surface of every (r, p) pair, weighted by the root's
          frequency.
        * ``[CLITICP_*]`` / ``[CLITICE_*]`` / ``[PREP_*]`` / ``[FUNC_*]`` /
          ``[CHAR_*]`` / ``[DIGIT_*]`` / ``[PUNCT_*]`` → the surface string itself.
        * Word-initial Arabic surfaces (roots, patterns, prepositions, function
          words, proclitics) also get the space-prefixed variant with the same
          weight — the base tokenizers are byte-level with leading-space pieces.
        * ``<pad>`` / ``<s>`` / ``</s>`` / ``<unk>`` and the LIT / PROP markers →
          empty (the initialiser's global-mean fallback).
        """
        root_freq = {r: int(m.get("freq") or 0) for r, m in (self._metadata.get("roots") or {}).items()}
        pat_freq = {p_: int(m.get("freq") or 0) for p_, m in (self._metadata.get("patterns") or {}).items()}
        by_root: Dict[int, List[Tuple[str, float]]] = {}
        by_pat: Dict[int, List[Tuple[str, float]]] = {}
        for (rid, pid), surf in self._reconstruction.items():
            if not surf:
                continue
            root = self._reverse_vocab.get(rid, "")[len(PFX_ROOT):-len(SFX)]
            pat = self._reverse_vocab.get(pid, "")[len(PFX_PAT):-len(SFX)]
            by_root.setdefault(rid, []).append((surf, float(max(pat_freq.get(pat, 0), 1))))
            by_pat.setdefault(pid, []).append((surf, float(max(root_freq.get(root, 0), 1))))

        def with_space(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
            return items + [(" " + s_, w) for s_, w in items]

        out: Dict[int, List[Tuple[str, float]]] = {}
        for tok, tid in self._vocab.items():
            if tok in SPECIAL_TOKENS_ORDERED or tok in (TOK_LIT_BEGIN, TOK_LIT_END, TOK_PROP_BEGIN, TOK_PROP_END):
                out[tid] = []
            elif tok.startswith(PFX_ROOT):
                out[tid] = with_space(by_root.get(tid, []))
            elif tok.startswith(PFX_PAT):
                out[tid] = with_space(by_pat.get(tid, []))
            elif tok.startswith((PFX_CLITICP, PFX_PREP, PFX_FUNC)):
                pfx = next(p_ for p_ in (PFX_CLITICP, PFX_PREP, PFX_FUNC) if tok.startswith(p_))
                out[tid] = with_space([(tok[len(pfx):-len(SFX)], 1.0)])
            elif tok.startswith((PFX_CLITICE, PFX_CHAR, PFX_DIGIT, PFX_PUNCT)):
                pfx = next(p_ for p_ in (PFX_CLITICE, PFX_CHAR, PFX_DIGIT, PFX_PUNCT) if tok.startswith(p_))
                inner = tok[len(pfx):-len(SFX)]
                out[tid] = [(inner, 1.0)] if inner.strip() else []
            else:
                out[tid] = []
        return out

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
                "prepositions": list(self.prepositions),
                "func_words": list(self.func_words),
                "clitic_peeler": self.clitic_peeler,
                "peel_bare_alef": self.peel_bare_alef,
                "proper_nouns": self.proper_nouns,
            }, f, ensure_ascii=False, indent=2)

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
        # Tokenizers saved before the [PREP_*] range have no inventory and
        # no such tokens in vocab.json; keep them decodable/encodable as
        # they were (encode falls through to LIT when the token is absent).
        self.prepositions = tuple(cfg.get("prepositions") or ())
        # Likewise for the [FUNC_*] range (added 2026-09-16).
        self.func_words = tuple(cfg.get("func_words") or ())
        # Pre-peeler tokenizers have no [CLITICE_كمو]-style tokens; the
        # all-or-nothing clitic check in _emit_alpha keeps them consistent.
        self.clitic_peeler = bool(cfg.get("clitic_peeler", True))
        self.peel_bare_alef = bool(cfg.get("peel_bare_alef", False))
        # A tokenizer saved before the [PROP_*] markers has none in its
        # vocab; _emit_prop falls back to the plain literal there.
        self.proper_nouns = str(cfg.get("proper_nouns") or "unrooted")
        self._backend = None  # rebuilt lazily with the loaded inventory

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

def join_proclitics(clitics: List[str]) -> str:
    """Concatenate buffered proclitics, applying the لِ + الـ contraction.

    Arabic writes one lam, not two: لِ + الوَلَد → لِلوَلَد. Joining naively
    produces "لالولد". Mirrors ``strip_proclitics_from_start`` on the encode
    side — the two must stay inverse to each other or round trips break on
    every ``لل...`` word.
    """
    out: List[str] = []
    for c in clitics:
        out.append("ل" if (c == "ال" and out and out[-1] == "ل") else c)
    return "".join(out)


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
    s = strip_proclitics_from_start(surface, proclitics)
    return strip_enclitics_from_end(s, enclitics)


def join_word(proclitics: Tuple[str, ...], stem: str, enclitics: Tuple[str, ...]) -> str:
    """Rebuild a ROOT+PAT word from its parts exactly as ``decode`` does.

    Proclitics through ``join_proclitics`` (the لِ+الـ contraction), then the
    enclitics in emission order (innermost first) with the decoder's ة → ت
    rule before a pronoun (مدرس + ة + ه → مدرسته). The inverse of
    ``_strip_clitic_surfaces`` when the strip was clean — which is exactly
    what ``entry_realization`` uses it to check.
    """
    word = stem
    for enc in enclitics:
        if not enc:
            continue
        if word.endswith(TAA_MARBUTA) and enc != TAA_MARBUTA:
            word = word[:-1] + "ت"
        word += enc
    return join_proclitics(list(proclitics)) + word


def entry_realization(e: CorpusEntry, use_diacritized_surface: bool) -> Optional[Tuple[str, str]]:
    """The inflected-stem surface a corpus entry contributes to its (root, pattern) pair.

    Returns ``(form, source)`` with ``source`` ``"written"`` — the chunk as the
    corpus wrote it, diacritics and clitics stripped, accepted only when
    ``join_word`` gives the (undiacritized) chunk back — or ``"diac"`` — the
    clitic-stripped CAMeL ``diac``, the pre-2026-09-22 rule and the fallback
    when the written strip does not reproduce (a ه pronoun read on a written
    ة the reconciliation could not fix, a peel the joins do not cover).
    With ``use_diacritized_surface`` the answer is always the diac form: the
    corpus does not carry the diacritics a diacritized table needs. ``None``
    when neither source yields a form.
    """
    if not use_diacritized_surface and e.word:
        bare = strip_diacritics(e.word)
        stem = _strip_clitic_surfaces(bare, e.proclitics, e.enclitics)
        if stem and join_word(e.proclitics, stem, e.enclitics) == bare:
            return stem, "written"
    if e.surface:
        inflected = _strip_clitic_surfaces(e.surface, e.proclitics, e.enclitics)
        if inflected:
            return (inflected if use_diacritized_surface else strip_diacritics(inflected)), "diac"
    return None


def pick_realization(written: Optional[Counter], diac_forms: Optional[Counter]) -> Optional[str]:
    """Pass 2 of ``_build_reconstruction``: the most frequent written form, else diac form.

    Ties break on the form string so the table is a function of the corpus,
    not of dict order.
    """
    for counter in (written, diac_forms):
        if counter:
            return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return None


def _stale_under_spelling_walk(e: CorpusEntry) -> bool:
    """Would the 2026-09-22 spelling-faithful walk re-decide this cached entry?

    Only a rooted, natively analysed entry whose surface does not spell the
    word is a candidate for a different reading (see the backend's
    ``needs_spelling_walk``); particles, clitic-only words, rootless names,
    peeled words and unanalysed chunks are untouched by the rule.
    """
    if not (e.analyzed and e.root and e.pattern) or e.peeled or e.particle or e.clitic_only:
        return False
    return needs_spelling_walk(
        Analysis(root=e.root, pattern=e.pattern, pattern_raw=e.pattern_raw or "", stem=e.stem or "",
                 surface=e.surface or "", lemma="", pos=""),
        e.word,
    )


def _stale_particle_match(e: CorpusEntry) -> bool:
    """A cached closed-class entry whose word does not contain the particle's own
    spelling — matched modulo alef, the case the format-10 POS gate re-decides."""
    return bool(e.analyzed and e.particle and e.particle not in strip_diacritics(e.word))


def _stale_cache_entry(e: CorpusEntry, from_format: int) -> bool:
    """Which entries of an older cache format the current rules may change."""
    if from_format <= 8 and _stale_under_spelling_walk(e):
        return True
    return from_format <= 9 and _stale_particle_match(e)


def _split_runs(word: str) -> List[Tuple[str, str]]:
    """Split a whitespace word into ``(class, chunk)`` runs.

    The single chunking convention shared by the corpus pre-pass and
    ``_encode_word`` (so analyzer cache keys agree). An alpha run absorbs a
    directly following ة and ends there: ة is word-final by definition, so
    a ة followed by more letters starts a new chunk (مدرسةكبيرة → مدرسة |
    كبيرة). A ة with no letters before it is a chunk of its own.
    """
    runs: List[Tuple[str, str]] = []
    i, n = 0, len(word)
    while i < n:
        cls = _classify_char(word[i])
        j = i + 1
        if cls == "alpha":
            while j < n and _classify_char(word[j]) == "alpha":
                j += 1
            if j < n and _classify_char(word[j]) == "fem":
                j += 1
        elif cls == "fem":
            cls = "alpha"  # a lone ة still goes down the alpha path
        else:
            while j < n and _classify_char(word[j]) == cls:
                j += 1
        runs.append((cls, word[i:j]))
        i = j
    return runs


def _extract_alpha_chunks(word: str) -> List[str]:
    """Return the Arabic-alpha chunks of a whitespace word (see ``_split_runs``)."""
    return [chunk for cls, chunk in _split_runs(word) if cls == "alpha"]
