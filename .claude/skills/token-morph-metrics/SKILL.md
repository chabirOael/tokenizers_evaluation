---
name: token-morph-metrics
description: Use this skill when working on the Arabic morphological metrics that score tokenizers — `root_conservation_rate` (RPS), `pattern_conservation_rate` (PIS), `morpheme_integrity_rate`, `clitic_separation_accuracy` (CSA), `semantic_fragmentation_ratio` (SFR), `root_bearing_token_pct`, `pattern_bearing_token_pct`. TRIGGER on edits to `src/arabic_eval/evaluation/intrinsic_metrics.py`, `tests/test_morphological_metrics.py`, `scripts/smoke_morph_metrics.py`, `tokenizers/utils/arabic_text.py`, or when the user mentions RPS / PIS / CSA / SFR / clitic separation / morpheme integrity / fragmentation / Farasa boundary alignment / `aligned_token_offsets` / `_clitic_boundaries` / `_morpheme_metrics_for_word` / qalsadi root extraction / `clean_token_string` / mechanical extremes / fairness across tokenizer families. Also TRIGGER for "is this metric measuring what I think it's measuring?" questions and tokenizer comparison-table interpretation.
---

# Tokenizer Morphological Metrics — Skill

The seven Arabic morphological metrics are the *intrinsic* signal for comparing tokenizers without running expensive downstream training. This skill is the operating manual: how the metrics are wired, where they fail, and how to add or fix one without breaking the fairness story.

The companion skills are `arabic-token-eval` (the platform overall, the 7 tokenizer families) and `araroopat` (the AraRooPat tokenizer specifically). This skill is the deep-dive on the *metric panel itself*.

## The seven metrics in one table

All live in `src/arabic_eval/evaluation/intrinsic_metrics.py` (sole entry point: `compute_morphological_metrics`).

| Key | Reported as | What it measures | Alignment-dependent? | Source of `None` |
|---|---|---|---|---|
| `root_conservation_rate` | rate ∈ [0,1] | % of sample words whose 3/4-letter root is a *subsequence* inside one token | No (uses `contains_subsequence`) | qalsadi/tashaphyne can't extract root for any sampled word |
| `pattern_conservation_rate` | rate ∈ [0,1] | % of words whose stem-span pattern (root letters + immediate context, clitics trimmed by `stem_pattern_span`) is recoverable from one token | No | Same as RPS; or `derive_pattern` fails on every sampled word |
| `morpheme_integrity_rate` | rate ∈ [0,1] | % of Farasa morpheme boundaries that align with token boundaries (averaged over multi-morpheme words) | **Yes** (`aligned_token_offsets`) | Farasa unavailable; OR every sampled multi-morpheme word fails alignment; OR no multi-morpheme words sampled |
| `clitic_separation_accuracy` (CSA) | rate ∈ [0,1] | % of clitic-flanking boundaries (proclitic-end / enclitic-start) that align with token boundaries; pooled across the sample | **Yes** | Farasa unavailable; sample contains no clitic morphemes; alignment fails on every clitic-bearing word |
| `semantic_fragmentation_ratio` (SFR) | real ≥ 0 | (raw non-special token count) / (Farasa morpheme count), summed over the sample | **No** — counts are alignment-free | Farasa unavailable; OR sample yields zero morphemes |
| `root_bearing_token_pct` | percent ∈ [0,100] | % of all sampled tokens that contain a full root from the sample's root set | No | No tokens AND no roots; OR no roots extracted (raw_token_count=0 → `None`; raw_token_count>0 but cleaned tokens empty → **`0.0`** mechanical) |
| `pattern_bearing_token_pct` | percent ∈ [0,100] | % of tokens whose stem span matches a known pattern from the sample's pattern set | No | Same `None` vs `0.0` carve-out as the root-bearing version |

The aliases RPS / PIS / CSA / SFR are how these typically show up in literature; the keys above are the canonical names in the codebase.

## Mechanical extremes — flag them, don't suppress them

The same architectural property of a tokenizer often forces a metric to hit 0 or 1 *for free*. Reporting that mechanical value is correct; suppressing it would mask the architecture. The five families:

| Family | Tokenizers | Forced extremes |
|---|---|---|
| Plain subword | `bpe`, `wordpiece` | None — these are the discriminating tokenizers across every metric |
| Farasa-aware subword | `morpho_bpe`, `farasa_character_bert` | integrity ≈ 1, CSA ≈ 1 (Farasa pre-segments, so token boundaries trivially include morpheme boundaries — non-trivial because chosen by design) |
| Word-level CharCNN | `character_bert` | integrity = 0, CSA = 0 (no internal token boundaries → no morpheme/clitic boundary aligns); RCR high but not 1.0 (qalsadi extracts non-subsequence roots for irregular forms) |
| Character | `char_jaber` | RCR = 0, RBT% = 0, PBT% = 0 (one char per token → can't hold a 3-letter root); integrity = 1, CSA = 1, SFR ≈ avg-chars-per-morpheme (mechanical, every char boundary is a token boundary) |
| Byte (Charformer) | `charformer` | RCR = 0, PCR = 0; RBT% = `0.0` (mechanical zero, **not `None`**); integrity = `None` (alignment fails for byte tokens); CSA = `None`; SFR ≈ 5–6 (each Arabic char = 2 bytes, 2–5 chars per morpheme) |

The Charformer story is the one that bites people. Each byte cleans to an empty Arabic-letter string, so `aligned_token_offsets` always returns `None`, so the alignment-dependent metrics (integrity, CSA) are *not measurable*. They report `None`, not 1.0. The discriminator for Charformer is **SFR**, which is alignment-free.

Numbers from the 240-sentence smoke (`scripts/smoke_morph_metrics.py`):

| Tokenizer | RCR | PCR | Integ | CSA | SFR | RBT% | PBT% |
|---|---|---|---|---|---|---|---|
| BPE | 0.26 | 0.39 | 0.33 | 0.33 | 1.10 | 12.2 | 12.2 |
| MorphoBPE | 0.67 | 0.96 | **1.00** | **1.00** | 1.00 | 33.9 | 33.9 |
| CharBERT | 0.67 | 1.00 | 0.00 | 0.00 | 0.51 | 66.7 | 66.7 |
| char-JABER | 0.00 | 0.00 | 0.99 | 1.00 | 2.68 | 0.0 | 0.0 |
| Charformer | 0.00 | 0.00 | None | None | 5.36 | 0.0 | 0.0 |

CSA discriminates plain BPE (0.33) from Farasa-aware MorphoBPE (1.00) cleanly — that's the new signal CSA was added for.

## How to interpret — the three reading rules

1. **Pick the metric that matches the question.**
   - "Does the tokenizer respect Arabic morphology?" → `morpheme_integrity_rate` for the broad question, **CSA** specifically for clitic handling.
   - "Does the tokenizer over- or under-fragment?" → **SFR**.
   - "Does the tokenizer keep root letters together?" → RCR.
   - "Does the tokenizer preserve the wazn?" → PCR.
   - "How many of the tokenizer's vocabulary entries actually carry morphological content?" → `root_bearing_token_pct`, `pattern_bearing_token_pct`.

2. **Use the mechanical-extremes table to pre-filter conclusions.** A 1.0 from CharBERT on RCR is not "CharBERT preserves roots best"; it's "CharBERT never splits a word." Always check the family forces before reading the number as a quality signal.

3. **Among Farasa-aware tokenizers, integrity and CSA are mechanically maxed — break ties with RCR and downstream MCQ scores.** Among plain subword tokenizers, integrity / CSA / SFR are the discriminators.

## Architecture: where things live

```
src/arabic_eval/evaluation/intrinsic_metrics.py
├── compute_intrinsic_metrics(...)          # entry point: size + morph
├── compute_morphological_metrics(...)      # the seven-metric panel
├── _morpheme_metrics_for_word(...)         # per-word: integrity + CSA + SFR data
├── _clitic_boundaries(...)                 # proclitic/enclitic offset detection
├── _empty_morph_metrics()                  # shape when sample is empty
├── _sample_words(...)                      # deterministic shared sample
├── derive_pattern / stem_pattern_span / ...# pure helpers (no side effects)
├── aligned_token_offsets(...)              # cleaned-token → word char offset alignment
├── filter_content_tokens(...)              # specials + empty cleaned strings filter
├── RootExtractor                           # qalsadi → tashaphyne → consonant skeleton
└── MorphemeSegmenter                       # Farasa subprocess wrapper

src/arabic_eval/tokenizers/utils/arabic_text.py
├── ARABIC_LETTERS / ARABIC_LONG_VOWELS / ARABIC_DIACRITICS
├── strip_diacritics(text)
└── clean_token_string(token)               # ByteLevel BPE inverse + marker strip + diacritic strip

src/arabic_eval/tokenizers/araroopat_backend.py
├── CAMEL_CLITIC_SURFACE                    # tag → Arabic surface mapping (CAMeL)
├── _PROCLITIC_TAGS / _ENCLITIC_TAGS        # bucket assignments (canonical source)
├── PROCLITIC_SURFACES / ENCLITIC_SURFACES  # ← imported by intrinsic_metrics.py for CSA
└── ...

tests/test_morphological_metrics.py         # 33 unit tests (no Java, mock everything)
scripts/smoke_morph_metrics.py              # end-to-end with real Farasa + 5 tokenizer families
```

The clitic-surface inventory for CSA is **sourced from the AraRooPat backend**, not duplicated. AraRooPat already has the CAMeL Tools clitic table (`CAMEL_CLITIC_SURFACE`); the metric module imports `PROCLITIC_SURFACES` and `ENCLITIC_SURFACES` derived from that table at import time. Don't fork the inventory — extend the bucketing in `araroopat_backend.py` if you add a new CAMeL clitic tag.

## The fairness invariants — never break these

The whole point of the panel is to compare tokenizers fairly on the same sample. Five rules keep that working:

1. **Same word sample for every tokenizer.** Driven by `_sample_words(texts, n, seed)` with a fixed seed. Don't add per-tokenizer pre-filtering inside `compute_morphological_metrics`.

2. **Same Farasa segmentation for every tokenizer.** `MorphemeSegmenter` is constructed once per call and called once per word. Don't re-segment inside the loop or per-tokenizer.

3. **Same skip rule for "alignment failed."** `aligned_token_offsets` returns `None` → integrity and CSA are `None` for that word. SFR is **not** gated on alignment — it just uses raw counts. Don't introduce per-tokenizer skip variants.

4. **Same skip rule for "root extraction failed."** The outer loop has `if root is None: continue` — this skips every metric for that word, including SFR. This keeps the *sampled population* identical across metrics. Don't make SFR sample independently.

5. **Mechanical extremes are reported, not suppressed.** A `None` means "not measurable for this tokenizer family"; a `0.0` from a bearing-token metric means "tokens generated, but mechanical zero" (Charformer); a `1.0` from a Farasa-aware tokenizer means "by design." All three are valid signal for a reader; collapsing them is information loss.

The `0.0` vs `None` distinction on `*_bearing_token_pct` is implemented via a `raw_token_count` counter that runs in parallel to `all_token_strings`. **Don't replace it with a blanket `None → 0.0` fallback** — `None` still has a real meaning (e.g. when no tokens were generated at all).

## CSA in detail — the position-based clitic walk

`_clitic_boundaries(morphemes)` returns the set of char offsets within a word that should be clitic↔stem boundaries:

```python
# Walk proclitics from the LEFT, stop at first non-clitic (the stem):
for m in morphemes[:-1]:
    if strip_diacritics(m) in PROCLITIC_SURFACES:
        cum += len(strip_diacritics(m))
        bounds.add(cum)
    else: break

# Walk enclitics from the RIGHT, stop at first non-clitic:
suffix_idx = len(morphemes)
for i in range(len(morphemes) - 1, 0, -1):
    if strip_diacritics(morphemes[i]) in ENCLITIC_SURFACES:
        suffix_idx = i
    else: break
# Then add: cumulative length up to suffix_idx, plus boundaries between
# consecutive enclitics (rare).
```

Position-based disambiguation handles the `ك` ambiguity:
- `ك` as `ka_prep` "like" — position 0 → proclitic walk catches it
- `ك` as `2ms_pron` "your" — position last → enclitic walk catches it

The diacritic strip on both sides is essential: Farasa sometimes returns diacritized morphemes (`وَ`, `الْ`) and the surface inventory is bare. Forgetting either side silently breaks CSA on diacritic-emitting Farasa modes.

## SFR in detail — why it's alignment-free

`semantic_fragmentation_ratio = sum(raw_token_counts) / sum(morpheme_counts)`.

- **`raw_token_count`** comes from filtering specials only: `sum(1 for t in tokens if t not in SPECIAL_TOKEN_STRINGS)`. **Not** the cleaned count — Charformer's bytes would all clean to empty, giving SFR = 0/N which is wrong.
- **`morpheme_count`** is `len(morphemes)` from Farasa, regardless of whether the word is single- or multi-morpheme. Single-morpheme words contribute meaningfully to SFR (a tokenizer that fragments single-morpheme words is over-fragmenting).
- **No alignment dependency.** Even when `aligned_token_offsets` fails (ByteLevel BPE artifacts), token and morpheme counts are still computable. This is the fairness rule that lets SFR work uniformly across all tokenizer families including Charformer.

Don't gate SFR on alignment success — it's the one metric that *must* work even when alignment fails, because it's the only signal left for byte-level tokenizers.

## Mock-friendly testing recipe

The test suite (`tests/test_morphological_metrics.py`, 33 tests, ~1s) runs without Java or qalsadi by patching `RootExtractor` and `MorphemeSegmenter` at the module level:

```python
@pytest.fixture
def patched_root_extractor():
    table = {"والكتاب": "كتب", "كتاب": "كتب", ...}
    with patch("arabic_eval.evaluation.intrinsic_metrics.RootExtractor") as RE:
        RE.return_value.extract.side_effect = lambda w: table.get(w)
        yield

@pytest.fixture
def patched_segmenter():
    table = {"والكتاب": ["و","ال","كتاب"], ...}
    with patch("arabic_eval.evaluation.intrinsic_metrics.MorphemeSegmenter") as MS:
        MS.return_value.segment_word.side_effect = lambda w: table.get(w)
        yield
```

Use the same pattern when adding tests. Don't try to install Java in CI just to test a metric — patch the segmenter.

## Adding a new metric — checklist

1. Decide if the metric is alignment-dependent (uses `aligned_token_offsets`) or not. This determines whether it can work for byte-level tokenizers.
2. If it needs new per-word inputs, extend `_morpheme_metrics_for_word`'s return dict (don't add a parallel helper that re-segments — Farasa is the bottleneck).
3. Add an accumulator in `compute_morphological_metrics`. Gate alignment-dependent accumulators on `csa_total` / `integrity is not None`; gate alignment-free accumulators only on Farasa-success.
4. Append the new key to the return dict AND to `_empty_morph_metrics()`.
5. Update the docstring with the new key.
6. Add unit tests in `tests/test_morphological_metrics.py` (mocked Farasa). Always cover: empty sample, single-morpheme word, alignment-failure, mechanical-extreme tokenizer family.
7. Add an EXPECTED_BANDS entry in `scripts/smoke_morph_metrics.py` for each tokenizer family the metric should run for. Use `EXACT(None)` to assert "not measurable for this family" if applicable.
8. Update `CLAUDE.md` and `.claude/skills/arabic-token-eval/SKILL.md` with the new row.

## Common gotchas

### qalsadi has TWO APIs — only `Analex.check_word` returns roots

```python
# WRONG — returns the lemma (dictionary form), not the root.
from qalsadi.lemmatizer import Lemmatizer
Lemmatizer().lemmatize("والكتاب")  # → "كتاب"  (lemma)

# CORRECT — proper root.
from qalsadi.analex import Analex
Analex().check_word("والكتاب")[0].root  # → "كتب"
```

`RootExtractor` already gets this right. If you change root extraction, **keep the qalsadi → tashaphyne → consonant-skeleton fallback chain in that order** and verify on test words like `والكتاب` (root `كتب`), `مدرسة` (root `درس`), `يدرسون` (root `درس`).

**Note on CharBERT not hitting RCR ≈ 1.0**: even though the whole word is one token, qalsadi sometimes extracts roots that aren't literal subsequences of the word for weak/irregular forms. CharBERT's RCR ≈ 0.67 on the smoke run is a *qalsadi artifact*, not a tokenizer failure.

### ByteLevel BPE token strings are byte-encoded

The HF `ByteLevel` pre-tokenizer (used by `BPETokenizer`) emits tokens like `'Ø§ÙĦÙĥØªØ§Ø¨'` — `'الكتاب'` byte-encoded through GPT-2's `bytes_to_unicode`. `clean_token_string()` reverses this via `_BYTELEVEL_INV`. **If you write a new tokenizer that uses any byte-level encoding, make sure `clean_token_string` handles it** — otherwise every token cleans to empty, alignment fails, and the bearing-token metrics silently become `None` for your tokenizer.

### Farasa = Java subprocess

- Requires `java` on PATH. Without it, integrity / CSA / SFR all return `None` (and a warning is logged).
- ~2s init cost for the first segment call (Java startup), then ~20–50 words/sec.
- `farasapy` warns about "interactive mode" on long inputs — benign, but switch to `interactive=False` in `MorphemeSegmenter._ensure()` if it gets noisy in sweeps.

### `_morpheme_metrics_for_word` returns `None` only on Farasa failure

Per-field `None`s have specific meanings:
- `integrity = None` → no internal morpheme boundaries (single-morpheme word) OR alignment failed
- `csa_respected = None` AND `csa_total = None` → no clitic boundaries OR alignment failed
- `morpheme_count` and `raw_token_count` → never `None` if the helper returns at all

The helper-level `None` (whole return) means Farasa segmented zero morphemes for the word. That's a strong signal — log it if it happens often.

### Diacritic handling consistency

Three places strip diacritics, and they must agree:
1. `_morpheme_metrics_for_word` for cumulative offset math (`len(strip_diacritics(m))`).
2. `_clitic_boundaries` for both length math AND set membership (`strip_diacritics(m) in PROCLITIC_SURFACES`).
3. `aligned_token_offsets` for the alignment target (`strip_diacritics(original_word)`).

Forget any of the three and offsets drift by 1+ chars per diacritic, integrity and CSA become noise, and the bug is silent because outputs still look like floats in [0,1].

### char-JABER `downsample_factor > 1` would break alignment metrics

`CharJaberEmbedding` has a `downsample_factor` knob (default 1 = disabled). If you ever turn it on, the model-side sequence shortens but the tokenizer's `tokens` field still has full-length char tokens. Alignment computes against the full token list, which is correct for the metric — but be aware the *model* sees a shorter sequence. Don't conflate the two.

## Things to push back on

- **"CSA is just a less powerful version of `morpheme_integrity_rate` — drop it."** No: integrity covers ALL Farasa boundaries (including stem-internal ones); CSA isolates clitic-flanking boundaries. They satisfy `integrity == 1.0 ⇒ CSA == 1.0` but disagree in general. CSA is the discriminating signal among plain subword tokenizers (BPE/WordPiece) where stem-internal splits are common.

- **"SFR < 1.0 means the tokenizer is broken."** No: it means the tokenizer compresses below morpheme grain — one token spans multiple morphemes. CharBERT does this by construction (1 token per word, ~2 morphemes per word → SFR ≈ 0.5). Whether that's "broken" or "by design" depends on the tokenizer; the metric is descriptive.

- **"Make Charformer's CSA / integrity = 1.0 since byte boundaries trivially align."** No: alignment is the protocol for these metrics, and byte tokens don't align to Arabic-letter offsets via `aligned_token_offsets`. Reporting `None` correctly marks the metric as not-measurable for byte-level tokenizers. SFR is the alignment-free discriminator. Forcing 1.0 would falsely suggest Charformer perfectly preserves morpheme/clitic structure when in fact the metric simply can't be computed.

- **"Compute SFR with cleaned token count (`filter_content_tokens`)."** No: that breaks Charformer (all bytes clean to empty). SFR uses *raw* non-special token count specifically so byte-level tokenizers report a real (high) fragmentation rather than 0. Same philosophy as the `0.0`-vs-`None` carve-out on `*_bearing_token_pct`.

- **"Skip words where alignment fails — those are the tokenizer's fault."** Not for SFR. Token and morpheme counts are alignment-free; gating SFR on alignment penalizes tokenizers that occasionally fail to round-trip (ByteLevel BPE artifacts) without that being a fragmentation signal at all. Integrity and CSA correctly skip on alignment failure; SFR does not.

- **"Add a fallback that returns 0.0 instead of None when Farasa is unavailable."** Don't — `0.0` would be indistinguishable from a real "tokenizer respects no boundaries" result. `None` correctly signals "not measured." (The `*_bearing_token_pct` carve-out is *separate*: there `0.0` is the right answer when tokens were generated but cleaned to empty Arabic-letter strings — that's the byte-level mechanical-zero case, not the Farasa-unavailable case.)

- **"Maintain a hardcoded `PROCLITICS`/`ENCLITICS` set in `intrinsic_metrics.py` instead of importing from `araroopat_backend`."** No — that creates two sources of truth. The CAMeL Tools clitic inventory is curated by AraRooPat; if a future contributor adds a new CAMeL tag, the buckets update there and CSA picks it up automatically.

- **"Use `FarasaPOSTagger` instead of `MorphemeSegmenter` for CSA — it tags clitics directly."** Possible upgrade path, not a current requirement. POS tagging is slower (a second Java subprocess) and the surface-set-based detection agrees with POS tagging on ~95%+ of cases. Revisit if observed CSA looks suspicious; don't refactor preemptively.

## Composite metrics that build on these (MEI and friends)

Any composite metric that *multiplies* one of these morphological metrics inherits its mechanical-extreme problem. The current example is **MEI** (`compute_mei` in [src/arabic_eval/evaluation/metrics.py](src/arabic_eval/evaluation/metrics.py)):

```
MEI = (accuracy × RPS × compression × num_eval_rows) / inference_time_sec
    = (accuracy × RPS × compression) / (inference_time_sec / num_eval_rows)
```

A few facts about MEI that are load-bearing for any future composite in this family:

- **Per-row time normalization.** The `num_eval_rows` factor (= `downstream[<task>].num_samples`) makes MEI comparable across LightEval MCQ tasks that differ in eval-set size — e.g. ACVA (~7.3K rows) vs Alghafa (~18.6K rows). Without it, the time term scales with dataset size and depresses MEI on bigger benchmarks for reasons unrelated to per-example efficiency. Within a task all tokenizers share the same row count, so the factor is a constant scale and rankings stay invariant; across tasks it removes the size contamination. Time is normalized **per-row, not per-token**, on purpose: `compression` already captures sequence-length differences in the numerator, so dividing by per-token time would double-count length.
- **Scope guard is by class, not name.** `compute_mei` takes `is_lighteval_mcq: bool`; the pipeline computes that with `isinstance(task, LightEvalBenchmarkTask)` rather than a hardcoded set of registry keys. This stays correct as new LightEval benchmarks are added (the `arabic-token-eval` skill documents that as a supported extension path).
- **The mechanical-flag inventory has one home.** `RPS_MECHANICAL_FLAGS` in `src/arabic_eval/evaluation/reporter.py` is the canonical list of tokenizers whose RPS is mechanically forced. The reporter footnotes them in the MEI table. **Do not redefine this inventory** in a new file when you add another composite — import from `reporter` or refactor the constant up to a shared module.
- **Self-describing record shape.** MEI returns `{"mei": float|None, "status": str, "inputs": {accuracy, rps, compression, inference_time_sec, num_eval_rows}}` with typed status codes (`ok`, `task_not_mcq`, `missing_<input>`, `zero_time`, `zero_rows`). The saved JSON is readable without re-running, and `scripts/recompute_mei.py` exploits the echoed inputs to migrate archived MEI numbers in-place when the formula changes. Reuse this shape for any future composite — see `tests/test_mei.py` for the corner cases (real-zero is `status=ok`, missing-inputs are typed).
- **Tokenizer warmup before the inference timer.** The pipeline runs one throwaway `tokenizer.encode("نص قصير للإحماء")` before `time.perf_counter()` so AraRooPat's CAMeL bridge spawn (~1–2 s) and Farasa Java subprocess startup don't get billed to MEI's denominator. **If you write a new composite that uses inference time, copy this pattern**, not just the timing line — without warmup, the rate metric is systematically unfair to morphology-aware tokenizers.

### What MEI predicts across the seven tokenizers

The same mechanical-extreme story plays out, just multiplied through:

| Tokenizer | RPS | Compression | Time | Net MEI tendency |
|---|---|---|---|---|
| BPE / WordPiece | low–mid (frequency-driven) | high (~3–4 chars/tok) | fast | mid-high — efficiency frontier |
| MorphoBPE / FarasaCharBERT | mid (RPS varies, FarasaCharBERT ~0.78 mechanical-ish on stem morpheme) | mid | mid (Farasa adds latency) | mid — the family MEI is designed to reward |
| AraRooPat | ~1.0 mechanical (ROOT token IS the root letters) | mid (~2.5 chars/tok, 2 tokens per content word) | mid–slow (CAMeL bridge IPC per word) | **flag** — RPS multiplier maxed out by construction |
| CharacterBERT | ~0.95 mechanical (never splits a word) | very high (one whole word per token) | mid | **flag** — both RPS and compression ceilinged mechanically; can rank surprisingly high |
| char-JABER | ~0 mechanical | low (1 char/tok) | slow (4–6× sequence) | **floor** — RPS multiplier zeros it out |
| Charformer | 0 mechanical (1 byte/tok) | very low (~0.5 chars/byte for Arabic) | slowest | **floor** — same as char-JABER |

The MEI sweep report flags AraRooPat / CharBERT / char-JABER / Charformer with an asterisk + footnote so a reader doesn't conclude the mechanical extreme is a real ranking signal. **Don't strip this footnote** to make the table shorter — it's load-bearing for correct interpretation.

### When someone proposes a different RPS-using composite

The instinct is to reuse RPS straight. Push back if the proposal:
- Skips the warmup step (will produce systematically unfair numbers for AraRooPat).
- Uses raw wall-clock time without dividing by row count (depresses scores on larger eval sets for reasons unrelated to the tokenizer; per-row normalization is the standard).
- Hardcodes a tokenizer-name allowlist instead of using `isinstance(task, LightEvalBenchmarkTask)` (will silently miss new benchmarks).
- Returns a bare float-or-None (loses the "why is this None?" debugging signal).
- Redefines `RPS_MECHANICAL_FLAGS` instead of importing it (drift between two reports is inevitable).
- Suppresses the mechanical-extreme tokenizers entirely (loses information; the architectural ceiling is itself a useful signal — see the "flag, don't suppress" rule above).

## Common workflows

### Adding a new tokenizer-family metric (worked example: SFR)

1. Identified that integrity/CSA can't measure byte-level tokenizers because they need alignment. Need an alignment-free metric.
2. Decided on `tokens / morphemes` as a count ratio.
3. Extended `_morpheme_metrics_for_word`'s return to include `morpheme_count` and `raw_token_count` (already had Farasa segments and the token list — no extra work).
4. Added `sfr_tok_sum` and `sfr_morph_sum` accumulators in `compute_morphological_metrics`.
5. Crucially: SFR accumulates **regardless of alignment** — token and morpheme counts come from independent paths.
6. Tested with a Charformer-style mock (14 byte tokens cleaning to empty) → confirmed SFR returns 14/3 not 0.
7. Asserted band `(4.0, ∞)` for Charformer in the smoke; observed 5.36.

### Investigating a metric that looks wrong

1. Run `scripts/smoke_morph_metrics.py` first — that compares all five families on a fixed corpus and tells you whether the issue is family-specific or universal.
2. Reduce to a one-word repro: pick one word from the smoke output where the metric looks off.
3. Call `_morpheme_metrics_for_word(word, tokens, segmenter)` directly to see the per-word values.
4. If `integrity` looks wrong, check `aligned_token_offsets(content, word)` — does it return `None`? That's the most common silent failure (cleaned-token string doesn't round-trip).
5. If CSA looks wrong, call `_clitic_boundaries(morphemes)` directly — verify the boundary set matches your linguistic intuition.
6. If SFR looks wrong, check `raw_token_count` (specials filtered correctly?) and `morpheme_count` (Farasa output correct?).

### Diagnosing a fairness-comparison anomaly

If tokenizer A scores higher than tokenizer B on a metric and you don't believe it, check in this order:

1. **Did they use the same sample?** `_sample_words` is deterministic with `seed=42`; if either ran with a different seed, comparison is invalid.
2. **Did Farasa run on both?** If only one had Farasa available, integrity / CSA / SFR are `None` for the other → no comparison possible.
3. **Did `aligned_token_offsets` succeed for both?** Check by adding a counter for "alignment-success rate" — if A round-trips 100% of words and B only 60%, B's integrity / CSA aggregates are over only 60% of the sample, distorting comparison.
4. **Is one of them in a mechanical-extreme regime?** Use the family table at the top of this skill — if A is char-JABER and B is BPE, integrity ≈ 1.0 for A is mechanical, not a real signal.

## Where to look first when something breaks

| Symptom | First file to check |
|---|---|
| New tokenizer's RBT% / PBT% silently `None` | `clean_token_string` in `tokenizers/utils/arabic_text.py` (byte-encoding handling) |
| CSA off for a Farasa-aware tokenizer | `_clitic_boundaries` (diacritic strip on membership test) |
| SFR `None` for a tokenizer where Farasa worked | `_morpheme_metrics_for_word` (gating SFR on alignment by mistake) |
| Integrity number doesn't match by-hand expectation | `aligned_token_offsets` returning `None` silently |
| Test failing only with real Farasa | Java not on PATH, or `farasapy` interactive-mode warning eating output |
| RCR low for whole-word tokenizer | qalsadi extracting non-subsequence root (irregular form) — usually expected |
| Metric drift after refactor | Snapshot `outputs/smoke_morph_metrics.json` before, compare after |
