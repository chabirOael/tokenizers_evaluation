---
name: araroopat
description: Use this skill for any work on the AraRooPat (Arabic Roots & Patterns) tokenizer — adding features, debugging encode/decode, tuning vocab budget, fixing CAMeL Tools integration, interpreting reconstruction failures, or extending pattern/clitic handling. TRIGGER on edits to `src/arabic_eval/tokenizers/araroopat.py`, `src/arabic_eval/tokenizers/araroopat_backend.py`, `configs/tokenizers/araroopat.yaml`, `scripts/smoke_araroopat.py`, or when the user mentions araroopat / root-pattern / wazn-based tokenizer / CAMeL Tools / morphological generator / pattern reconstruction / clitic stripping / [ROOT_*] [PAT_*] tokens.
---

# AraRooPat — Skill

Companion to `CLAUDE.md` (architecture overview) and `arabic-token-eval` (project-wide skill). This file captures the design rationale, hard-won gotchas, and the alternatives we rejected — the things a fresh contributor would re-discover painfully.

## Anchor: what this tokenizer is

Each Arabic content word becomes two consecutive tokens:

```
الكتاب → [CLITICP_ال] [ROOT_كتب] [PAT_1ِ2ا3ِ]
وكتابه → [CLITICP_و] [ROOT_كتب] [PAT_1ِ2ا3ِ] [CLITICE_ه]
يدرس   → [ROOT_درس] [PAT_يَ1ْ2ُ3ُ]                    ← inflectional ي is part of pattern, NOT a clitic
تلفزيون → [LIT_BEGIN] [CHAR_ت] [CHAR_ل] ... [LIT_END]   ← loanword fallback
```

`embedding_type = "standard"`. Same `nn.Embedding` + `lm_head` resize path as BPE/WordPiece/MorphoBPE — slots into `LlamaAdapter` and `StandardCollator` unchanged.

The architectural commitment: **roots carry semantic content, patterns carry morphology, clitics are decoupled atoms**. The LLM emits `(root, pattern)` and the tokenizer reconstructs via a lookup table.

## Vocab layout (deterministic ID order)

```
0–3:    specials              <pad> <s> </s> <unk>
4–5:    literal markers       [LIT_BEGIN] [LIT_END]
6+:     proclitics            [CLITICP_ال] [CLITICP_و] ...     (sorted by corpus freq)
        enclitics             [CLITICE_ه] [CLITICE_ها] ...
        chars                 [CHAR_ا]...[CHAR_ي] + diacritics
        digits                [DIGIT_0]...[DIGIT_9] [DIGIT_٠]...[DIGIT_٩]
        punctuation           [PUNCT_.] [PUNCT_،] ...
        roots                 [ROOT_كتب] ... (top-K by freq, freq ≥ min_root_freq)
        patterns              [PAT_*] (top-K by freq, freq ≥ min_pattern_freq)
```

**ID-range layout matters.** The decoder dispatches on the prefix of the token string. Future work like constrained decoding or factored embeddings can use the ID range to mask token classes without changing storage.

## CAMeL Tools API — the contract that bites

These are the things I learned the hard way. Verify before changing the backend.

### Disambiguator API

```python
disambig = MLEDisambiguator.pretrained()
out = disambig.disambiguate([word])           # list of DisambiguatedWord
out[0].word                                    # original word
out[0].analyses                                # list of ScoredAnalysis  ← USE THIS
# out[0].scored_analyses                        # ← DOES NOT EXIST. Don't use.
out[0].analyses[0].analysis                    # the dict with root, pattern, diac, ...
out[0].analyses[0].score                       # disambiguation probability
```

**The API uses `.analyses`, not `.scored_analyses`.** Earlier docs / examples sometimes said the latter; do not trust them.

### Analyzer fields we care about

```python
{
  'root':    'ك.ت.ب',           # ← dot-separated. Strip both '.' and '_' (and '#' for defective).
  'pattern': 'وَال1ُ2ّا3ِ',      # ← positional template; clitic surface chars BAKED IN.
  'diac':    'وَالكُتّابِ',      # ← full diacritized surface (with clitics).
  'stem':    'كُتّاب',           # ← lexical bare stem (no clitics, NO inflectional prefixes).
  'lex':     'كاتِب',            # lemma / dictionary form
  'pos':     'noun',
  'prc3':    '0',                # question proclitic feature TAG (not surface!)
  'prc2':    'wa_conj',          # conjunction proclitic feature TAG → translate via clitic_surface()
  'prc1':    '0',                # preposition / future proclitic
  'prc0':    'Al_det',           # article proclitic
  'enc0':    '0',                # pronominal enclitic feature TAG
}
```

Three traps in those fields:

1. **`root` uses `.` separators** (`'ك.ت.ب'`) and sometimes `_`. Some defective entries use `#` as a placeholder for a missing/weak letter (`'ش#ق'`). Strip all three. After stripping, reject roots shorter than 3 letters.

2. **`prc*` and `enc0` are FEATURE TAGS, not Arabic surface strings.** `'wa_conj'` not `'و'`. Translate them via `clitic_surface()` in `araroopat_backend.py` before they leave the analyzer wrapper. There's a hand-built `CAMEL_CLITIC_SURFACE` table covering the common ~60 tags — extend it (don't redo it) when you encounter unknown ones; unknown tags fall through verbatim with a debug log.

3. **`pattern` includes clitic surface chars** (e.g. `'ال1ِ2ا3ِ'` for `الكتاب` — the `ال` is *both* in `prc0='Al_det'` AND the pattern's leading `ال`). Without stripping, you double-count clitics at decode. See "Pattern normalization" below.

### Generator API

```python
db_gen = MorphologyDB.builtin_db(flags='g')   # 'g' = generation flags
gen = Generator(db_gen)
gen.generate(lemma, features_dict)            # ← takes lemma+features, NOT (root, pattern)
```

The generator does NOT take `(root, pattern)`. It takes a lemma and a feature dict. We don't have a lemma for unseen `(root, pattern)` pairs. The workaround in `MorphAnalyzer._generate_cached`:

1. Naive-fill the bare pattern with root letters → produces a phonologically rough surface.
2. Run that surface back through the analyzer + disambiguator.
3. Find an analysis whose `(root, pattern_bare)` matches the target.
4. Return its `stem` field (the morphologically correct bare-stem surface — handles weak roots, hamza, gemination via the calima-msa rules).

This is the tier-2 reconstruction in `_reconstruct()`. It works because CAMeL's analyzer is "lossy-invertible" — feed it a rough form, get back the canonical form.

### Database loading

`MorphologyDB.builtin_db()` defaults to `'a'` flags (analysis only). For generation, pass `flags='g'`. Loading both takes ~1–2s. Lazy-init in the backend; don't import at module-load time or you slow down tokenizer-only workflows.

## Design decisions — and why each alternative was rejected

The four decisions below are the ones a future contributor is most likely to question. Keep this section close.

### 1. Why we ship Path B (separate clitic tokens) and not Path A (clitic-baked-in patterns)

**Path B (chosen)**: emit `[CLITICP_*] / [CLITICE_*]` tokens; patterns are *bare-stem* templates.

**Path A (rejected)**: drop clitic tokens; let patterns carry clitic morphology too. Vocab shape: each `(base_pattern, clitic_combo)` becomes a distinct PAT token (~250 base × ~10 clitic combos = ~2500 patterns).

We chose B because:
- Roots and clitics are linguistically separable atoms. Path A buries the article inside the pattern token, which loses the LLM's ability to learn `[CLITICP_ال]` as a generic morpheme.
- Path A bloats the pattern vocab 5–10×. With our budget, that pushes either roots or patterns down a tier.
- The user's whole pitch was **"the LLM tells the tokenizer the root and the pattern"** — A doesn't honor that, B does.

**Don't switch to A** without revisiting the whole design philosophy. If you want to experiment, do it as a separate `araroopat_flat` tokenizer.

### 2. Why we use the *inflected stem* (`diac` minus clitics), not CAMeL's `stem` field

This was a real bug I shipped → unshipped. CAMeL's `stem` field is the **lexical** stem — it excludes inflectional prefixes/suffixes that are part of the conjugation, not just clitics. Specifically:

```
يدرس  → diac='يَدْرُسُ'  stem='دْرُس'    ← stem drops the present-tense ي!
كاتب → diac='كاتِبٌ'    stem='كاتِب'    ← stem keeps the form-I active participle ا
```

If we use `stem` for reconstruction, `يدرس` decodes as `درس` (loses the inflection). We instead compute the **inflected stem** = `diac` minus clitic surface chars (using `_strip_clitic_surfaces()`). This keeps inflection (the ي), drops clitics (the ال).

```python
inflected = _strip_clitic_surfaces(a.surface, a.proclitics, a.enclitics)
# 'وَكِتابِ' → strip 'و' (prc2) → 'َكِتابِ' → trim diacritic → 'كِتابِ'
# strip diacritics if config says so → 'كتاب'
```

**Don't switch back to `stem`** unless you also bake inflection into the pattern token (which collides with Path A above).

### 3. Why we use `[CLITICP_*]` and `[CLITICE_*]` distinct prefixes (not one `[CLITIC_*]`)

This was the second bug I shipped → unshipped. With a single `[CLITIC_*]` prefix, the decoder cannot disambiguate:

```
[ROOT_كتب] [PAT_كتب] [CLITIC_ال] [ROOT_طلب] [PAT_طلب]   ← is CLITIC_ال an enclitic of كتب or proclitic of الطالب?
```

The encoder's bigram structure doesn't distinguish them, and there's no marker between words. With distinct prefix prefixes, the decoder dispatches on token type:
- `CLITICP_*` → buffer as proclitic for next word
- `CLITICE_*` → attach to just-emitted word

Linguistically defensible too — same surface form can be different morphemes:
- `ك` as preposition (`ka_prep`, proclitic) vs `ك` as 2ms object pronoun (`2ms_dobj`, enclitic)
- Real linguistic ambiguity collapses if you give them the same token.

Vocab cost: ~5–10 extra tokens (each surface that appears in BOTH proclitic and enclitic positions). Negligible.

**Alternative I considered and rejected**: a `[WORD_END]` marker token after every word. Costs 1 extra token per content word — multiplies sequence length unnecessarily.

### 4. Why we use a hybrid (lookup → generator → naive) reconstruction, not pure lookup or pure rule-based

| Tier | Source | Coverage | Speed | Faithfulness |
|---|---|---|---|---|
| 1 | corpus lookup | ~99% of LLM emissions (LLM was trained on this distribution) | O(1) | exact |
| 2 | CAMeL `Generator` (re-analyze a naive form, return its stem) | unseen `(root, pattern)` from corpus | ~1ms cached | rule-based; correct on weak roots |
| 3 | naive slot substitution | always works | O(L) | wrong on weak roots, hamza |

**Pure lookup**: rejected — a (root, pattern) pair the LLM invents that we never saw produces garbage.
**Pure rule-based**: rejected — would require shipping all of Arabic phonology, weeks of linguistic engineering.
**Pure naive**: rejected — produces `قَوَلَ` instead of `قَالَ` for weak roots. Unacceptable.

In practice, tier-1 hits ~99%, tier-2 fires on rare LLM emissions, tier-3 is the safety net.

### 5. Why CAMeL Tools, not MADAMIRA

CAMeL: MIT-licensed, on PyPI, Python-native, ~95% MSA accuracy, fully offline after one-time `camel_data -i light` (~80MB). MADAMIRA: ~96% accuracy but Java subprocess, Columbia-registration license that blocks redistribution. MADAMIRA's marginal accuracy gain is not worth the licensing/build friction.

**The dependency conflict is the real cost** — see "Dep conflict" section below.

## Critical implementation gotchas

### Pattern normalization (the central trick)

CAMeL's `pattern` field bakes clitic surface chars in. We strip them at analyzer-wrapper time so each `[PAT_*]` represents a bare-stem template:

```
'وَال1ُ2ّا3ِ'  with prc2='و', prc0='ال'
   strip prc3 (none) → 'وَال1ُ2ّا3ِ'
   strip prc2 'و'    → 'ال1ُ2ّا3ِ'   (drops 'و' and following diacritic)
   strip prc1 (none) → 'ال1ُ2ّا3ِ'
   strip prc0 'ال'   → '1ُ2ّا3ِ'    ← bare pattern stored in vocab
   strip enc0 (none) → '1ُ2ّا3ِ'
```

The `_strip_clitic_from_start` / `_strip_clitic_from_end` helpers walk char-by-char skipping diacritics — necessary because the pattern interleaves diacritics with the clitic surface chars (`'وَ'` is `و` + fatha `َ`).

**If you change `normalize_pattern`, run the smoke test.** It catches doubling bugs immediately.

### Encoder/pre-pass cache key consistency

The corpus pre-pass builds an LRU-cached `analyze(word)` lookup. If the encoder calls `analyze()` with a different chunking convention than the pre-pass, every encode call hits the analyzer fresh — slow.

Both must use `_extract_alpha_chunks(word)` to get the contiguous Arabic-alpha runs. **If you change chunking, change it in both.**

### Cache invalidation across schema changes

The `corpus_analysis.pkl` cache holds `CorpusEntry` instances. Adding/removing fields on `CorpusEntry` makes old caches mismatch silently (pickle restores old shape, then field accesses fail or return garbage). Always:

```bash
rm -rf outputs/tokenizers/araroopat_*_cache outputs/tokenizers/araroopat_smoke
```

before re-running smoke tests after schema changes.

### NTWS detection

CAMeL marks loanwords / non-Arabic-source words with `root='NTWS'` ("Non-Triliteral Word Source"). Always reject these as analyses (return None) so they route to the LIT fallback path. Otherwise `[ROOT_NTWS]` pollutes the vocab. The check:

```python
if root == "NTWS" or "NTWS" in pattern_raw:
    return None
```

### Defective root handling (`#` placeholder)

CAMeL uses `#` as a placeholder for missing/weak letters in some entries (e.g. `'ش#ق'`, `'عط#'`). These break Arabic-letter-only checks downstream. Strip them along with `_` and `.`; if the resulting root is shorter than 3 letters, route to LIT.

### Generator timeout and signal handling

`_Timeout` uses `signal.SIGALRM` on POSIX. **It only works on the main thread.** From a worker thread, `signal.signal()` raises `ValueError` and we silently fall through with no timeout. That's intentional (better than crashing in a worker), but it means generator calls in dataloader workers are unbounded. If you see slow generator calls in training, consider an explicit `multiprocessing.Process` timeout or pre-compute everything at vocab time.

### Order matters: clitic stripping is outer-to-inner

```python
for clitic in (prc3, prc2, prc1, prc0):   # outermost first
    pat = _strip_clitic_from_start(pat, clitic)
```

CAMeL's clitic stack is `prc3 (question) > prc2 (conjunction) > prc1 (preposition/future) > prc0 (article)` — outermost to innermost. Strip in that order from the start. **Don't reorder** — `prc1` could begin with a substring that appears later in `prc0`'s position, and you'd partial-match.

For enclitics there's only one slot (`enc0`) so the order issue doesn't arise.

### Glue digits in decode

Without special handling, `[DIGIT_2] [DIGIT_0] [DIGIT_2] [DIGIT_4]` decodes to `"2 0 2 4"` (4 separate "words"). The decoder glues consecutive digit tokens into one number when the previous output ends with a digit and there are no buffered proclitics in between. Test case: `"في عام 2024"` → `"في عام 2024"`.

## Common workflows

### Adding a new clitic feature tag

If the smoke test or a real run logs `Unknown CAMeL clitic tag: 'foo_bar'`, extend `CAMEL_CLITIC_SURFACE` in `araroopat_backend.py` with the surface form. Don't change anything else — the unknown-tag path falls through verbatim, so it's a graceful degradation, but the surface won't be a valid Arabic clitic.

### Diagnosing a low `root_conservation_rate`

The metric uses qalsadi's `RootExtractor`, which extracts a root that may differ from CAMeL's. Three diagnostic steps:

1. **Count LIT-path words**: in your sample, how many words went through `[LIT_BEGIN]`? On a small smoke corpus this is ~30%; on real-scale corpus ~5–15%. If higher, CAMeL coverage is bad — investigate which words fail `analyze()`.
2. **Compare extractors**: pick a sample word, run `RootExtractor().extract(w)` and `MorphAnalyzer().analyze(w).root`. Mismatches are common (`مالك` → qalsadi `مول` vs CAMeL `ملك`); not a bug, just a measurement floor.
3. **Check vocab inclusion**: rare roots/patterns get cut by `min_root_freq` / `min_pattern_freq`. If a sample word's analyzed root isn't in vocab, the encoder routes it to LIT.

The *relative* numbers across tokenizers are what matter — araroopat's 0.54 on a 200-sentence smoke corpus is not the architectural ceiling, it's the corpus-coverage floor.

### Tracing a roundtrip failure

`scripts/smoke_araroopat.py` has 7 hand-picked test cases covering each branch. If a roundtrip breaks after a change, run it and look at the printed `tokens` list — that's the cleaned metric strings, but it also reveals the encode-time decisions:

- Empty `''` strings = special tokens (BOS/EOS/LIT_BEGIN/LIT_END/clitic-cleaned)
- `'كتب'` = root letters (root token)
- `'كتاب'` = inflected stem (pattern token)
- Single-char strings = LIT path or punct/digit

If you see `'كتب', 'كتب'` for a single word, that's `[ROOT_كتب] [PAT_*]` and the cleaned PAT string equals the root letters (which means the pattern has no template letters between the root chars — likely a perfect verb pattern like `1a2a3`).

### Inspecting provenance

`outputs/tokenizers/araroopat_<tier>/vocab_metadata.json` records:
- For each root: `{id, freq, source ("corpus" or "camel_db_only"), example_words: [up to 5]}`
- For each pattern: `{id, freq, source, examples: [(root, surface), ...]}`
- Full proclitic/enclitic frequency maps

Use `jq` to slice it: `jq '.roots["كتب"]' vocab_metadata.json`.

### Inspecting reconstruction

Both `reconstruction.pkl` (for runtime) and `reconstruction.json` (human-readable) are saved. The JSON has keys `"root_id,pat_id"` and values `{root, pattern, surface}`. To check what a specific (root, pattern) reconstructs to:

```bash
jq 'to_entries[] | select(.value.root == "كتب") | .value' outputs/tokenizers/araroopat_smoke/reconstruction.json
```

### Re-training without re-analyzing

The corpus pre-pass writes `outputs/tokenizers/araroopat_cache/corpus_analysis.pkl`. Subsequent runs (e.g. different `max_roots` / `max_patterns`) reuse it. Don't delete this between vocab-size sweeps — it's the expensive ~30-minute step on real corpora.

Cache invalidates if `set(unique_words) > set(cached_words)` (new corpus). If equal, it's a cache hit even on different vocab budgets.

### Changing vocab tier

Three pre-set tiers in `configs/tokenizers/araroopat.yaml` (Compact / Balanced / Max). Coverage product (analyzed-words × in-vocab fraction):

- Compact 5K+200: ~81%
- Balanced 10K+500 (default): ~94%
- Max 15K+1000: ~98%

Above Balanced, returns diminish — most extra roots are very rare (single-occurrence) and most extra patterns are inflectional variants that the LLM rarely emits. Stay at Balanced unless you have a specific reason.

## Things to push back on

### "Just use the pattern as-is — clitic-baked-in is fine"

This is Design Path A. It bloats the pattern vocab 5–10× and removes the linguistic decomposition of clitics. Rejected — see Decision §1. **Only revisit** if downstream training reveals that distinct prc/enc tokens are hurting more than helping, which would show up as worse perplexity than BPE in the v1 comparison.

### "Just use CAMeL's `stem` field directly for reconstruction — it's the bare stem, that's what we want"

`stem` excludes inflectional prefixes (the ي of present-tense verbs, etc.), not just clitics. Reconstruction would lose all conjugation. Rejected — see Decision §2. Use `_strip_clitic_surfaces(diac, ...)` instead.

### "Naive substitution is fine for tier-2 — skip the CAMeL Generator"

Naive fill on `(قول, 1َ2َ3َ)` gives `قَوَلَ` instead of `قَالَ`. For weak roots / hamza / gemination cases this is meaningfully wrong. The generator is ~1ms per call, cached, fires only on tier-1 misses (~1% of decodes). The cost is negligible; the correctness gain is real. Keep all three tiers.

### "Drop the LIT fallback — every Arabic word has a root, just compute it"

CAMeL's database covers ~85–95% of MSA on real corpora. The other 5–15% are loanwords (تلفزيون, إنترنت), proper nouns, dialectal text, and code-mixed fragments. They genuinely have no `(root, pattern)` decomposition. Without LIT, you'd lose them entirely or pollute the root vocab with garbage `NTWS`-style entries. Keep LIT.

### "Make `vocab_size` actually drive vocab construction like other tokenizers"

`vocab_size` is ignored on purpose — araroopat sizes vocab via `max_roots + max_patterns + fixed slots`. The platform's `train_tokenizer.py` CLI passes `--vocab-size` for uniformity; we accept and log-ignore it. **If you wire vocab_size to drive max_roots/max_patterns**, you'd have to pick a split (e.g., 95% to roots, 5% to patterns), which encodes a vocab-design choice into a number that should mean total vocabulary. Cleaner to keep them separate.

### "Bump `downsample_factor` (or whatever) to make sequences shorter"

There is no downsample mechanism in araroopat. Sequences are 2–3 tokens per content word (CLITICs + ROOT + PAT). That's the price of separable morphology — accept it. If you need shorter sequences, the right knob is fewer fine-tuning epochs or a smaller `max_length` in the collator.

### "Migrate to a newer CAMeL version that doesn't conflict with lighteval"

Solved differently — see "Bridge architecture" below. We no longer install `camel-tools` in the main `.venv`; it lives in an isolated `.venv-camel` and the main process talks to it via a subprocess bridge. If a future `camel-tools` release does pin `numpy>=2` and `transformers>=4.54`, you can collapse the two venvs back into one — but don't do it speculatively. The bridge is also more robust (per-request timeouts, subprocess isolation) than the in-process import was.

### "Initialize ROOT/PAT embeddings smartly from LLaMA's pretrained subword embeddings to speed convergence"

This is "Option D" from the design discussion. Tempting because araroopat's structural alternation likely needs more fine-tuning steps to converge than BPE. **Don't do it in v1** — it breaks the platform's "trained from scratch" frame and would be unfair unless applied uniformly to all tokenizers. Reserve for a follow-up experiment after we have comparable v1 numbers.

## Bridge architecture — how the dep conflict is solved

`camel-tools>=1.5` pins `numpy<2` and `transformers<4.54`, both of which break `lighteval>=0.11`. We don't choose — `camel-tools` lives in its own venv and the main process talks to it via a subprocess bridge.

**Layout:**
```
.venv         # main env: lighteval, transformers>=4.54, numpy>=2, NO camel-tools
.venv-camel   # isolated env: ONLY camel-tools + its old numpy/transformers pins
```

**Files:**
- `src/arabic_eval/tools/araroopat_camel_server.py` — runs *inside* `.venv-camel`. Imports `camel_tools` directly. Reads NDJSON from stdin, writes NDJSON to stdout. Three ops: `analyze` (batch words → list-of-lists of trimmed analysis dicts), `generate` (root + bare pattern → stem string or null), `shutdown`.
- `src/arabic_eval/tokenizers/araroopat_bridge.py` — runs in main `.venv`. `CamelBridge` class spawns the server subprocess lazily, correlates requests by integer `id`, enforces a per-request `select()` timeout (default 5s), and registers an `atexit` handler to send `shutdown` cleanly.
- `src/arabic_eval/tokenizers/araroopat_backend.py` — `MorphAnalyzer` wraps the bridge. The pure-Python post-processing (clitic surface translation, pattern normalization, NTWS rejection, `Analysis` dataclass) lives here, untouched by the move.

**Setup (one-time):**
```bash
python -m venv .venv-camel
.venv-camel/bin/pip install -e ".[araroopat-camel]"
.venv-camel/bin/camel_data -i light
```

**Override the interpreter path:** set `$ARAROOPAT_CAMEL_PYTHON` if `.venv-camel` lives elsewhere. Otherwise the bridge resolves `<repo_root>/.venv-camel/bin/python`.

**Fail-loud policy (no silent degradation):**
- Missing `.venv-camel` → `CamelBridgeError` with the exact setup commands.
- Server exited (EOF on stdout) → `CamelBridgeError`, with the server's stderr drained and attached to the exception message.
- Server returned `{"ok": false, "error": "..."}` → `CamelBridgeError` with the server-side error.
- `select()` timed out (default 5s, override via `CamelBridge(read_timeout_s=...)`) → `CamelBridgeError`.

There is **no** `MorphAnalyzer.is_available` and **no** "degraded mode" where missing camel routes everything to LIT. Using araroopat without camel is a configuration error, not a runtime branch.

**Wire format example:**
```
client → server:  {"id": 7, "op": "analyze", "words": ["والكتاب", "يدرس"]}
server → client:  {"id": 7, "ok": true, "results": [[{"root": "كتب", "pattern": "ال1ِ2ا3", ...}], [...]]}

client → server:  {"id": 8, "op": "generate", "root": "قول", "pattern": "1َا2َ"}
server → client:  {"id": 8, "ok": true, "result": "قال"}
```

`results` is a list-of-lists because each word can have multiple candidate analyses; the client takes `[0]` to keep MLE semantics, falling through to `[1]` only when `[0]` is rejected (e.g. NTWS or defective root).

**Banner handshake:** the server emits `{"id": 0, "ok": true, "result": "ready"}` once init succeeds. The bridge reads this with a 30s timeout (covers DB load), so init failures surface immediately rather than at the first real request.

**Batching:** `MorphAnalyzer.analyze_many(words, batch_size=256)` does one IPC round-trip per batch. The corpus pre-pass uses this — saves ~40k bridge calls on a 10k-word corpus.

**When the conflict goes away:** if `camel-tools` ships a release that pins `numpy>=2` and `transformers>=4.54`, you *can* collapse the two venvs and import camel directly. But: the bridge gives subprocess isolation (camel can't crash the main process), per-request timeouts, and clean cleanup via `atexit`. That's worth keeping even if the version constraints relax. Don't tear out the bridge purely to "simplify the architecture."

## Smoke test recipe

The minimum signal that nothing is broken:

```bash
rm -rf outputs/tokenizers/araroopat_smoke_cache outputs/tokenizers/araroopat_smoke
.venv/bin/python scripts/smoke_araroopat.py
```

What to check in the output:
- **All 7 roundtrips match exactly** (modulo punctuation spacing). Test cases exercise: regular triliteral, weak root (قال), proclitic stack (والكتاب), inflectional prefix (يدرس), enclitic (قرأته), loanword fallback (تلفزيون), digits (2024), punctuation (!).
- **`vocab_size` ≈ specials + literals + clitics + chars + digits + punct + roots + patterns** — sanity-check the breakdown line in the `[INFO] AraRooPat trained` log.
- **`reconstruction_entries` ≈ unique `(root, pattern)` pairs in the corpus** with both members in vocab.
- **`morpheme_integrity_rate ≈ 1.0`** (mechanical, expected — clitics are separate tokens by construction).
- **`root_conservation_rate` and `pattern_conservation_rate`** depend on corpus size and qalsadi/CAMeL agreement; on the 200-sentence smoke they're 0.54 and 0.66. On real corpora they should be higher.

If any roundtrip fails, the most likely causes (in rough order): (1) cache invalidation needed after a schema change, (2) a CAMeL clitic tag missing from the translation table, (3) a pattern-normalization edge case (especially when a clitic ends in a long vowel that overlaps with the stem), (4) the encoder/pre-pass chunking diverged.

## Pre-existing files this skill cares about

- [src/arabic_eval/tokenizers/araroopat.py](src/arabic_eval/tokenizers/araroopat.py) — main tokenizer
- [src/arabic_eval/tokenizers/araroopat_backend.py](src/arabic_eval/tokenizers/araroopat_backend.py) — CAMeL wrapper, pattern normalizer, clitic table
- [configs/tokenizers/araroopat.yaml](configs/tokenizers/araroopat.yaml) — vocab tiers + params
- [configs/experiments/araroopat_intrinsic.yaml](configs/experiments/araroopat_intrinsic.yaml) — smoke-test config
- [scripts/smoke_araroopat.py](scripts/smoke_araroopat.py) — 200-sentence end-to-end check with 7 roundtrip cases
