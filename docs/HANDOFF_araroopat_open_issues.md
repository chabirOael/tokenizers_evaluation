# Handoff — AraRooPat: fixed, and what's still open

**Written 2026-08-30. Self-contained: assumes no access to the session that produced it.**

You are picking up work on the AraRooPat tokenizer in the Arabic Tokenizers Evaluation
Platform. Three defects were fixed on 2026-08-30 (§3 — **do not redo these**), the measurement
instrument was corrected (§4), and six problems remain open (§6). Every number below is a real
measurement with the command to reproduce it; nothing is estimated.

---

## 1. Orientation — read this before running anything

**Repo:** `/home/s3user/tokenizers_evaluation` (not a git repository — no history to consult).

**Two virtualenvs, and this is load-bearing:**

| env | holds | why |
|---|---|---|
| `.venv` | the platform, `lighteval`, `transformers>=4.54`, `numpy>=2` | main env |
| `.venv-camel` | **only** `camel-tools` + its `numpy<2` / `transformers<4.54` pins | irreconcilable with lighteval |

`camel-tools` is never imported in the main env. AraRooPat reaches it through a subprocess
NDJSON bridge: client `src/arabic_eval/tokenizers/araroopat_bridge.py`, server
`src/arabic_eval/tools/araroopat_camel_server.py` (runs inside `.venv-camel`).

**Always invoke Python as `.venv/bin/python`** (or `.venv-camel/bin/python`). Never bare
`python` — it resolves to a system interpreter without the deps.

One-time camel setup if missing:
```bash
python -m venv .venv-camel
.venv-camel/bin/pip install -e ".[araroopat-camel]"
.venv-camel/bin/camel_data -i light
```

**Other environment facts:** `java` must be on PATH for Farasa (used by
`morpheme_integrity_rate` / `clitic_separation_accuracy` / `semantic_fragmentation_ratio`); a
gated-model HF token is needed for `meta-llama/Llama-3.2-1B`; an H100 is available.

**Fail-loud policy:** the bridge raises `CamelBridgeError` on any failure. There is no
degraded mode — araroopat without camel would route every word to the character fallback,
which is a configuration error, not a runtime branch. Do not add a fallback.

---

## 2. What AraRooPat is, in one page

Each analyzable Arabic content word becomes **two consecutive tokens** — a root and a pattern —
with clitics as separate tokens around them:

```
ذهب    → [ROOT_ذهب] [PAT_1َ2َ3َ]
يذهب   → [ROOT_ذهب] [PAT_يَ1ْ2َ3]          same root, different pattern
الولد  → [CLITICP_ال] [ROOT_#لد] [PAT_وَ2َ3]
تلفزيون → [LIT_BEGIN] [CHAR_ت][CHAR_ل]… [LIT_END]   loanword fallback
```

A pattern is a template with numbered slots: digits `1`–`4` are the n-th root consonant,
everything else is literal template material. `يَ1ْ2َ3` is the classical wazn `يَفْعَل`.

`embedding_type = "standard"` — a flat vocabulary and an ordinary `nn.Embedding`, exactly like
BPE. Nothing about the model is special-cased.

**Decode** is a left-to-right state machine over four pieces of state (`out`, `clitic_prefix`,
`pending_root`, `lit_buffer`) dispatching on the token's *string prefix*, plus a three-tier
resolver: `(root_id, pat_id)` lookup table → CAMeL generator → naive slot fill.

**Key files**

| path | role |
|---|---|
| `src/arabic_eval/tokenizers/araroopat.py` | encode / decode / vocab build / reconstruction |
| `src/arabic_eval/tokenizers/araroopat_backend.py` | CAMeL post-processing, pattern normalization, clitic table |
| `src/arabic_eval/tokenizers/araroopat_bridge.py` | subprocess client |
| `src/arabic_eval/tools/araroopat_camel_server.py` | server (in `.venv-camel`) — **duplicates** several helpers by necessity |
| `src/arabic_eval/evaluation/intrinsic_metrics.py` | the morphological metrics |
| `configs/tokenizers/araroopat.yaml` | vocab budget tiers |
| `scripts/smoke_araroopat.py` | 7 roundtrip cases |
| `tests/test_araroopat_normalization.py` | normalization regression tests |

Deeper background: `CLAUDE.md`, `.claude/skills/araroopat/SKILL.md`,
`docs/araroopat_end_to_end.md` (a full traced walkthrough of `يذهب`).

---

## 3. Already fixed on 2026-08-30 — do not redo

### 3.1 CAMeL's `#` masked radical is kept

`#` marks a radical whose surface realization is **not stable across the paradigm** — the weak
letters و/ي/ا and the hamza family. It is a *radical*, not a missing field: قال · يقول · قول ·
أقوال all analyse as `ق.#.ل`. It is complementary to the pattern — the masked slot's digit is
absent and the realized letter sits in the pattern as literal material (`1ا3َ`, `يَ1ُو3`).
**Verified on 457 of 457 masked radicals, zero counter-examples.**

The old code did `root.replace("#","")` then rejected roots under 3 characters. That was
destructive twice: the root fell below the bar, *and* the survivors were renumbered so the
pattern's slot digits indexed the wrong letters. It sent **49.3 % of word occurrences** to the
character path.

Now: radicals counted via `root.split(".")`, `#` retained in the token (`[ROOT_ق#ل]`),
`WEAK_RADICAL_MARK` whitelisted by `_is_arabic_root`, guarded in `naive_pattern_fill`.

**Known and accepted trade-off:** `#` is lossy. `[ROOT_س#ر]` covers both س-و-ر (سور, wall) and
س-ي-ر (سار, walk). The `(root, pattern)` *pair* stays unambiguous; the root token alone does
not. Canonicalizing `#` to a guessed radical was **rejected** — it merges exactly the same
pairs while hiding the ambiguity behind a plausible-looking root.

### 3.2 Pattern budget re-tiered 500 → 4000

Keeping `#` moves the weak letter's identity out of the root and into the pattern, so unique
patterns grew **8,634 → 21,578** while roots barely moved. At the old budget the `#` fix
delivered almost nothing (22.1 % → 26.1 %). `max_roots` is **never** binding — only 4,128 roots
clear `min_root_freq`.

### 3.3 The لِ + الـ contraction

Arabic writes one lam, not two: لِ + الوَلَد → لِلوَلَد. The article's surface is `ل`, not `ال`,
so a literal strip failed and left a stray lam in the bare pattern (`ل1ِ2ا3ِ`) and the stem
(`لكتاب`); `للولد` decoded as `لالولد`. **Pre-existing and independent of `#` — it hits sound
roots too.** It was wasting 2,608 of 21,578 patterns (12 % of the inventory).

Implemented as an inverse pair that **must stay inverse**:
`strip_proclitics_from_start` (encode, mirrored into the server's `_normalize_pattern`) and
`join_proclitics` (decode).

### 3.4 Smaller fixes

* `FOREIGN` rejected alongside `NTWS`; `_is_arabic_root` catch-all rejects ASCII fragments
  (was leaking `[ROOT_FOREIGN]` freq 61 and `[ROOT_Uٌٍ]` freq 3 into the vocabulary).
* Four untranslated CAMeL tags added to `CAMEL_CLITIC_SURFACE`: `la_emph`→`ل`, `la_rc`→`ل`,
  `>a_ques`→`أ` (Buckwalter `>` = أ), `mA_sub`→`ما`. They had been entering the vocabulary as
  literal Latin text (`[CLITICP_la_emph]`, 12,676 occurrences). Unknown-tag fallthrough is kept
  but raised from `logger.debug` to `logger.warning`.
* **Performance:** `_build_metadata` was calling `_vocab_root_set()` / `_vocab_pattern_set()`
  *inside* its per-entry loop — O(entries × vocab). At 506k entries × 8.3k vocab that is ~11e9
  string ops and it hung a training run. Hoisted. **Keep them hoisted.**
* `scripts/train_tokenizer.py` gained `--params '<json>'` to pass constructor kwargs.

### 3.5 Measured result

| | before | after |
|---|---|---|
| words on the ROOT+PAT path | 22.1 % | **48.2 %** |
| tokens that are `[CHAR_*]` | 52.3 % | 37.5 % |
| fertility | 5.7627 | **4.4688** |
| compression_ratio | 1.0218 | **1.3176** |
| root_conservation_rate | 0.1996 | **0.3541** |
| root_conservation_attainable | 0.2321 | **0.4071** |
| pattern_conservation_rate | 0.2296 | 0.3969 |
| semantic_fragmentation_ratio | 3.5820 | 2.9388 |
| analyzed types in pre-pass | 297,177 (27.5 %) | 506,101 (44.4 %) |
| reconstruction entries | 60,248 | 108,149 |

---

## 4. The measurement instrument was also wrong — corrected

`root_conservation_rate` (RPS) was silently doing three things at once. Five diagnostics were
added to `compute_morphological_metrics`; **all pre-existing fields are byte-identical** so
archived runs stay comparable.

| field | what it says | measured |
|---|---|---|
| `root_measurable_pct` | share of sampled words whose root is a subsequence of the *unsplit* word | ~84 % |
| `root_conservation_attainable` | RPS renormalized onto that population — **the comparable number** | — |
| `morph_alignment_coverage` | share of words `morpheme_integrity_rate`/`clitic_separation_accuracy` actually saw | araroopat 0.21 |
| `morph_alignment_ceiling` | what a whole-word tokenizer scores on the same sample | **0.6202** |
| `root_extractor_agreement` | qalsadi vs tashaphyne — noise in the ground truth | ~0.80 |

Two findings that change how results must be read:

**(a) RPS's denominator contained unwinnable words.** A weak root is not a subsequence of its
own surface — قال does not contain the و of قول — so ~16 % of the sample is unscorable by
*any* tokenizer. This is why `character_bert` reads 0.61 despite never splitting a word.

**(b) `morpheme_integrity_rate = 1.0` and `clitic_separation_accuracy = 1.0` for araroopat are
an artifact, not an architectural ceiling.** Both require `aligned_token_offsets` to succeed,
which needs the cleaned tokens to concatenate back into the word. araroopat's ROOT+PAT output
never does — measured **0 of 112** such words align, **488 of 488** character-path words do. The
two rates therefore describe only the *fallback* path. Older docs claimed "≈1.0 by
construction, clitics are separate tokens" — that was wrong and has been corrected.

The diagnostic earned its place immediately: after the fix `morph_alignment_coverage` **fell
0.43 → 0.21** while both rates kept printing 1.0.

`generate_report` now renders a *Morphological Metrics* section and footnotes any experiment
below `LOW_ALIGNMENT_COVERAGE` (0.5) — report-and-footnote, never suppression.

---

## 5. Current artifacts and vocabulary layout

| directory | what it is |
|---|---|
| `outputs/tokenizers/araroopat_balanced` | **before** everything (vocab 3,784). Pairs with the archived experiment — do not overwrite. |
| `outputs/tokenizers/araroopat_hashfix_p500` | ablation: `#` fix only, old budget (4,765) |
| `outputs/tokenizers/araroopat_hashfix` | **current** (8,265) |
| `outputs/tokenizers/araroopat_cache/` | CAMeL pre-pass cache — see the invalidation rule below |
| `outputs/tokenizers/araroopat_cache_pre_hashfix/`, `…_prelam/` | preserved older caches |
| `outputs/experiments/araroopat_3phase_with_sft/` | trained model + benchmark results, **paired with `araroopat_balanced`** |
| `outputs/metric_baseline/*.json` | intrinsic metrics under the corrected instrument |

**Contiguous ID ranges by class** (needed for §6.1):

```
                 araroopat_balanced      araroopat_hashfix
specials          0 ..    3               0 ..    3
LIT markers       4 ..    5               4 ..    5
CLITICP           6 ..   17               6 ..   15
CLITICE          18 ..   29              16 ..   28
CHAR             30 ..   76              29 ..   75
DIGIT            77 ..   96              76 ..   95
PUNCT            97 ..  137              96 ..  136
ROOT            138 .. 3283             137 .. 4264
PAT            3284 .. 3783            4265 .. 8264
```

**Derive these at runtime from the vocabulary — never hardcode them.** They shift with the
budget, as the two columns show.

### Cache invalidation rule (costly to get wrong)

`outputs/tokenizers/araroopat_cache/corpus_analysis.pkl` stores **post-processed** roots and
patterns. Any change to `_dict_to_analysis`, `normalize_pattern`, or
`strip_proclitics_from_start` invalidates it, and the tokenizer will silently train on stale
analyses. Move the directory aside and retrain:

```bash
mv outputs/tokenizers/araroopat_cache outputs/tokenizers/araroopat_cache_$(date +%s)
.venv/bin/python scripts/train_tokenizer.py --type araroopat \
    --params '{"max_patterns": 4000}' --output outputs/tokenizers/araroopat_hashfix
```

~22 min: ~7 min indexing 669k texts, ~9 min CAMeL pre-pass over 1.14M unique chunks, then
vocab + reconstruction. Watch for `Pre-pass stats:` and `AraRooPat trained` in the log.

---

## 6. Open problems

Ordered by value-for-effort. P1 is self-contained and has the clearest payoff.

### P1 — ROOT→PAT pairing is not enforced during generation (**highest priority**)

**Symptom.** The model emits one id per forward pass. `[ROOT_ذهب]` at step *t*,
`[PAT_1َ2َ3َ]` at *t+1* — two independent decisions. Nothing in the architecture, the sampling
loop, or the tokenizer requires a ROOT to be followed by a PAT. The pairing is a *learned soft
bigram* only.

**Evidence — this fires in practice.** Four greedy generations from the repo's own Phase-3
checkpoint:

```
ROOT followed by PAT : 2/3
PAT preceded by ROOT : 2/4

prompt 2 →  PAT LIT CHAR CHAR …      decoded: ''          ← entire generation vanished
prompt 4 →  ROOT CHAR CHAR LIT …     decoded: 'ن د عبر'
```

**Why it fails silently.** `naive_pattern_fill` returns `""` when the root is empty, and the
final `" ".join(s for s in out if s)` filters empties out. An orphan PAT is not an error, not a
`?`, not an `<unk>` — it is *nothing*. With BPE a stray subword still decodes to some text; here
the factorization means half a pair is a whole lost word, dropped without a trace.

```
decode([ROOT, PAT]) → 'يذهب'   ✓
decode([ROOT])      → 'ذهب'    ← orphan root: bare letters, inflection lost
decode([PAT])       → ''       ← orphan pattern: vanishes
```

**Fix.** A `LogitsProcessor` that makes the bigram structural:

* after a `[ROOT_*]` id → mask everything except the PAT range;
* otherwise → mask the PAT range (a PAT may only follow a ROOT).

Add a method on `AraRooPatTokenizer` returning the class→(lo,hi) map derived from `self._vocab`
prefixes (do not hardcode; see §5). Wire it through `LlamaAdapter.generate()` in
`src/arabic_eval/models/llama_adapter.py`, which currently forwards `**kwargs` straight to
`self._model.generate(...)` — HF accepts `logits_processor`. Prefer the platform's
signature-gating convention (`inspect.signature(...).parameters`) over widening
`BaseModelAdapter`.

Optionally also constrain `[LIT_BEGIN] … [LIT_END]` well-formedness.

**Verify.** Re-run the compliance measurement (§7) and expect 100 % on both lines, with
generated text no longer collapsing to `''`.

**Scope note.** Generation is *not* on the eval path — LightEval scoring is teacher-forced
log-likelihood — so no recorded benchmark number changes. This matters the moment anyone
generates text.

### P2 — `ى` and `ٱ` are missing from `ARABIC_LETTERS` — **FIXED 2026-09-12**

> Both letters were added to `ARABIC_LETTERS`; the ceiling test in `tests/test_morphological_metrics.py` was retargeted to an embedded-digit word and a regression test pins `مستشفى` as alignable. The smoke roundtrip (`ذهب الطفل إلى المدرسة`) now passes 9/9. **Still outstanding from the risk list below:** the araroopat retrain (`araroopat_hashfix` and the pre-pass cache predate the fix) and the re-baseline of all nine tokenizers. Original write-up kept for the record:

**Location:** `src/arabic_eval/tokenizers/utils/arabic_text.py`, ~line 40:

```python
ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآؤئة")   # 35 chars — no ى, no ٱ
```

**Symptom.** `_classify_char('ى')` returns `other`, which both **splits the word** and emits
`<unk>`:

```
encode("إلى المستشفى") → … [CHAR_إ][CHAR_ل] [LIT_END] <unk> … <unk>
decode(...)            → 'إل ? المستشف ?'
```

`normalize_arabic()` in `src/arabic_eval/data/preprocessing.py` normalizes alef *variants*
(`آأإٱ → ا`) but does **not** map `ى → ي`, so nothing upstream rescues it. Since `ى` ends إلى /
على / حتى and every defective-root noun, those words are truncated *before* CAMeL ever sees
them — it inflates the §6.5 rejection rate as well.

It is also what caps `morph_alignment_ceiling` at 0.6202 for **every** tokenizer:
`clean_token_string` drops `ى`, so the cleaned tokens cannot reconstruct the word.

**Fix.** Add `ى` and `ٱ` to `ARABIC_LETTERS`. They are Arabic letters; their absence is a bug.

**Risks — read before doing it.** That set feeds `clean_token_string`, `_clean_arabic` and
`_classify_char`, so it changes morphological metrics for **all nine tokenizers** and requires
retraining araroopat (§5 cache rule). Expect `morph_alignment_ceiling` to rise above 0.62 and
every tokenizer's alignment-dependent metrics to move. Re-run the baselines in §7 and record
before/after. Do **not** instead map `ى → ي` in preprocessing: that is lossy (the two letters
are distinct) and changes the corpus for every tokenizer.

### P3 — the decoder is not stream-safe

`decode()` assumes it receives the whole id list. Called per-token (a streamer, an incremental
UI), the ROOT arrives alone → `dump_orphan_root()` flushes bare root letters as a finished word;
the PAT arrives alone → vanishes. `يذهب` streams out as `ذهب`.

There is exactly one per-id decode in the repo —
`src/arabic_eval/evaluation/intrinsic_metrics.py:445`,
`[tokenizer.decode([i]) for i in out.input_ids]` — but it is a fallback guarded by
`if not out.tokens:` and araroopat always populates `.tokens`, so **it never fires today**. The
platform is safe; the tokenizer is not.

Two related edges:

* Truncation. `encode(truncation=True)` is a plain `ids[:max_length]` slice and can cut between
  a pair; so can `max_new_tokens`. The trailing orphan root decodes to bare letters, silently.
* `</s>` is skipped **without clearing `pending_root`**, so `ROOT </s> PAT` still pairs. If you
  concatenate two generations, a trailing orphan root in the first can capture the leading
  pattern of the second.

**Fix options.** A stream-safe decoder that buffers a pending root until its pattern arrives;
and/or make the orphan branches loud (emit the bare root *and* log) instead of silent. Clearing
`pending_root` on EOS is a one-line correctness improvement.

### P4 — downstream accuracy of the fix is unmeasured

Everything in §3.5 is intrinsic. Whether more words on the morphological path improves
ACVA / Alghafa / Culture-Arabic-MMLU / Arabic-Exam accuracy or MEI is **unknown**.

Run the fixed tokenizer through the same 3-phase pipeline and compare against
`outputs/experiments/araroopat_3phase_with_sft/` (which pairs with `araroopat_balanced`). ~5 h
on an H100. Keep every other variable fixed — that is the platform's core contract.

Note the confound to disclose in any writeup: the two tokenizers were built from pre-pass runs
over slightly different word universes (1,081,971 vs 1,140,803 unique chunks). The token-class
comparison on the fixed 300-question set is exact; the vocabulary-size comparison is not
perfectly controlled.

### P5 — the pattern budget still truncates

CAMeL analyses **66.6 %** of word occurrences; the tokenizer admits **48.2 %**. The gap is
budget truncation, with measured diminishing returns:

```
patterns   500 → 26.2 %   (vocab ~4.7K)      patterns  4000 → 48.2 %   (vocab ~8.2K)  ← current
patterns  1000 → 33.6 %   (vocab ~5.2K)      patterns  6076 → 52.1 %   (vocab ~10.3K)
patterns  2000 → 42.2 %   (vocab ~6.2K)
```

The remaining 33.4 % is CAMeL's **backoff analyzer** (`root='O'`, `pattern='backoff'`) firing
where the database has no entry — correctly rejected by the length rule, and no budget reaches
it. Treat 66.6 % as the ceiling for this approach.

Whether to spend vocabulary here is an experimental question, not a bug: a larger vocabulary
means a larger embedding matrix and a different comparison against the other tokenizers.

### P6 — alignment-dependent metrics on non-substring tokenizers (lower priority)

`morpheme_integrity_rate` and `clitic_separation_accuracy` are structurally inapplicable to any
tokenizer whose tokens are not substrings of the source word — araroopat today, and any future
factored or generative tokenizer. They currently report a number computed on the fallback
subset, qualified by `morph_alignment_coverage`.

Consider returning `None` when coverage is 0 over the tokenizer's primary path, the way
Charformer already reports `None`. **Do not** "fix" it by treating unaligned words as respected
— that is the `None → 1.0` fallback in a new costume, and the project has an explicit rule
against it.

---

## 6b. Clitic combinations the CAMeL database lacks — FIXED 2026-09-13 (clitic peeler)

### The failure

`أنلزمكموها` (Qurʾān 11:28, *shall we compel you to it?*) had no CAMeL analysis and went to
`[LIT_*]`. Diagnosis on `calima-msa-r13` (the only MSA DB we can load — `calima-msa-s31` ships as
`morphology.db.muddle` under an LDC licence and needs SAMA 3.1 to unmuddle):

| probe | result |
|---|---|
| `Analyzer(backoff='NONE')` | 0 analyses |
| `NOAN_ALL` / `ADD_ALL` | 66 analyses, all `root='O', pattern='backoff'` → rejected by `_dict_to_analysis` |
| `NOAN_PROP` (what `MLEDisambiguator.pretrained()` uses in the server) | 1 `NOUN_PROP` backoff → rejected → LIT |
| `db.suffix_hash` | 193 keys; `كموها`, `كمو`, `كمها`, `كها`, `نيها`, `هها` all **missing** |
| `db.defines['enc1']` | **`None` — the feature does not exist in r13** |
| prefix table | 60 keys; **no row carries `prc3`** — the interrogative أ is absent (`أتكتب` → 0 analyses) |

So the word fails for three independent reasons — no interrogative prefix, no second enclitic
slot, no classical lengthened form — and the class is *every* interrogative and *every*
double-object verb, not a Qurʾānic quirk. The apparent double-pronoun suffixes in the table
(`تموها`, `ناها`, `وها`) are subject-suffix + one object.

Stress set (37 words, `tests/test_araroopat_clitic_peeler.py`): **13/37 correct before** — every
control passed, every word with أ or two enclitics went to LIT. Corpus: the ArabicText-Large
pre-pass holds `أنلزمكموها` itself plus `أكفيكموه`, `يحدثكموه`, `فأسقيناكموه`, `لأحدثنهموه`, …
(rare by type count — this fix is about correctness on classical/QA text, not admission).

### What was rejected

* **Pre-normalisation (`كموها → كمها`)** — measured zero gain: the MSA target forms are absent too.
* **Extending the DB** — hundreds of new prefix/suffix rows plus `AB/BC/AC` compat lines, no
  `enc1` feature to put the second pronoun in, no MLE statistics for new rows, and r13 is GPL v2.
* **Qurʾānic gold lookup** — only helps verbatim Qurʾān tokens; does nothing for `سلمتكها`.

### What shipped — `peel_candidates` / `peel_compatible` / `merge_peeled` in `araroopat_backend.py`

Fallback chain in `MorphAnalyzer.analyze` / `analyze_many`, explicit:
**CAMeL native (whole word) → clitic peeler → bare-particle check → `[LIT_*]`.**

* Runs **only** on a native miss; a word CAMeL reads (أستكتبه as Form X) is never touched.
* Closed lists: proclitics `أ > (و|ف) > (س|ب|ل|ك) > ال` (with `لل` = `ل`+`ال`); enclitics = one
  pronoun, or a first pronoun (or lengthened `كمو`/`همو`) + a **3rd-person** second pronoun
  (Wright §187: the first object outranks the second, so `نا+ني` "us me" is impossible).
* Candidates least-peeled-first; the residual goes through the same native call + validation as
  a whole word; CAMeL's own inner clitics on the residual are kept and merged.
* **The governing gate:** the peel must remove something CAMeL *cannot* represent — أ, a second
  pronoun, or a lengthened form. Removing only و/ب/ال/… or a single pronoun is refused: CAMeL
  models those natively, so its refusal of the whole word was informative (`المجتهدي` is a nisba,
  not article + possessive; `المسا` is article + tanween). This one rule took false peels on
  3,000 CAMeL-rejected corpus types from **5.8 % → 0.8 %**; the person-rank rule, "a peeled
  conjunction does not license أ on a bare noun", and "reject a residual root with two masked
  radicals" took it to **0.6 %**. Accepting bare `ا` as the interrogative (required: the
  pipeline's `normalize_alef` rewrites `أ → ا`) re-opened one shape — `ا + ل` read as
  interrogative + preposition on `ال…` words CAMeL rejects (`الحيفي`, 150 of 10,000) — closed by
  refusing bare `ا` followed by a `ل` proclitic, peeled or CAMeL's own. Final: **122 of 10,000
  CAMeL-rejected types (1.2 %)**, every one round-tripping. What remains is أ-initial
  transliterations whose residual is a well-formed verb (`أوهارا` → `هارا`) — not separable
  without context.
* `س` attaches to imperfects only (`سلمناهموها` is `سلّمنا + …`, never `س + لمّنا`): CAMeL's `asp`
  is now in the server's `_TRIMMED_FIELDS` and on `Analysis.aspect`.
* **Reversibility by construction:** every peeled surface is a literal slice of the word, and the
  residual's CAMeL canonical spelling must equal the slice modulo alef variants (CAMeL normalises
  ة/ه, hamza, ى/ي on lookup — `قرضة` comes back as `قرض + ه`; accepting that would decode `القرضة`
  as `القرضه`, so ة/ه and ى/ي are strict; hamza is compared normalised because the native path
  already decodes to CAMeL's hamza spelling and the pipeline rewrites it upstream anyway).
  **No surface form is altered:** `كمو`/`همو` are their own `[CLITICE_*]` tokens, never rewritten
  to `كم`/`هم` (decode could not know when to re-lengthen, since MSA writes `كمها`).
* Data model: `Analysis.enc1` + `Analysis.peeled`; `CorpusEntry.peeled`; `_CACHE_FORMAT` 3 → 4
  (the key also carries the `clitic_peeler` flag); `vocab_metadata.json["peeled"]` lists count +
  200 examples for auditing; `config.json` persists `clitic_peeler`. YAML: `clitic_peeler: true`.
* `_emit_alpha` is now **all-or-nothing on clitic tokens** on the ROOT+PAT path (as the PREP path
  already was): an OOV `[CLITICE_كمو]` sends the whole word to LIT instead of emitting a literal
  clitic that decodes with a space. Also fixed on the way: a tokenizer saved before the
  `[CLITICE_ة]` slot recursed forever in `_emit_lit` on any ة-final word.
* Explorer: `araroopat_trace.py` gained a `peel` step (every candidate, its residual analysis
  and the verdict); the un-instrumented `train()` self-check still passes.

### Measured result

| | before | after |
|---|---|---|
| stress set (37 words) | 13 correct | **36 correct** (`أستكتبه` keeps its native Form X reading by design) |
| `أنلزمكموها` | LIT (12 tokens) | `[CLITICP_أ] [ROOT_لزم] [PAT_نَ1ْ2َ3] [CLITICE_كمو] [CLITICE_ها]` |
| false peels on 3,000 CAMeL-rejected corpus types | — | 21 (0.7 %) |
| existing suites + smoke | green | green (220 new tests, live tier included) |

### P7 — residual readings below rank 1 — **FIXED 2026-09-13** (server `top`, candidate walk)

`ألزمناهموها` and `وسأعطيكموه` stayed on LIT because the server built
`MLEDisambiguator.pretrained()` with the default **`top=1`** (a constructor argument, not a
per-call one), so every residual came back with one reading. For a word the MLE model has never
seen every reading scores 1.0, so rank 1 is just database order: `ألزمنا` → noun + `نا` ("our
necessity"), while the `أَلْزَمْنا` PV + `SUBJ:1P` we need is reading **6 of 7**; `وسأعطي` → three
passive `وَسَأُعْطَى` readings (surface guard rejects them — they'd decode with `ى`), the active
`وَسَأُعْطِي` is reading **4 of 5**.

Shipped: the server keeps `MAX_TOP = 32` candidates and each `analyze` request carries `top`
(default 1); the bridge forwards it; `MorphAnalyzer._candidates_many` fetches every valid reading
of a *residual* (`RESIDUAL_TOP = 32`, own cache) and `_resolve_peel` walks slicings
least-peeled-first and readings in order, accepting the first that passes
`residual_verdict` (the ة/surface/compatibility gate, now one shared helper the tracer also
uses). **The native whole-word path still requests one candidate**, so native admission is
unchanged (P7b below). False-peel rate under the walk: 0.6 % → 0.7 % (one extra, `أكابده`).

**Determinism bug found and fixed on the way.** The analyzer's candidate order behind an MLE tie
follows Python string hashing, i.e. it differed from process to process — the same word could get
a different top-1 reading on every run (`وَمالٍ` vs `وَمَآلٍ`, both alef-normalising to `ومال`),
and the peeler a different residual reading. `_op_analyze` now sorts by
`(-score, diac, bw, pos, prc*, enc0, asp)` so the order is a function of the word only. This is
the native path too — a latent non-reproducibility that predates the peeler.
`TestLiveCamel::test_live_reading_order_is_deterministic_across_processes` pins it.

### P7b — should the native whole-word path also see `top > 1`? — **PARTLY DONE 2026-09-22**

`_native_many` now asks for `top=32` for exactly the words whose top-1 reading does not spell
the word as written (`needs_spelling_walk`, ~8 % of the corpus chunks) and walks them for a
spelling-faithful rooted reading (`prefer_faithful`, see P8). A rank-1 NTWS / backoff reading
still goes straight to LIT — rescuing those with a rank-2 root reading remains open and needs
the admission benchmark first; the cost side is settled (one extra batched request for the
unfaithful words, 443 words/s over the bridge, the pre-pass cache migrates instead of rebuilding).

### Bare alef as the interrogative — opt-in (`peel_bare_alef`, default off)

The platform's preprocessing normalises `أ → ا`, so in pipeline text the interrogative surfaces as
a bare alef. Adding `ا` to the closed list unguarded **doubled** the false-peel rate on
CAMeL-rejected corpus types (0.6 % → 1.2 %): word-initial `ا` is also the hamzat-wasl of Forms
VII–X and imperatives (`اعرفوني` → `ا + عرفون + ي`, `استضف` → `ا+س + تضف`), `ابن/اسم`, and the
article (`المسا` → `ا + لمسا`). The article case is now excluded in the grammar (a lone `ا`
followed by `ل` is never the interrogative), which brings the mode to 1.1 % — still the
hamzat-wasl and transliteration classes. It is therefore a YAML knob (`peel_bare_alef: false`;
part of the cache key and `config.json`). Turn it on only for alef-normalised corpora, and
re-measure false peels first (`peel_bare_alef=True` in the measurement script gives you the list).

### Regenerating the test fixture

`tests/data/araroopat_camel_recorded.json` holds every *valid* reading (`top=32`, server order)
for every whole word and residual the suite consults. Regenerate after any camel-tools / DB /
`_dict_to_analysis` change — `TestLiveCamel::test_live_recorded_readings_are_current` tells you
when:

```python
# .venv/bin/python — from the repo root
import json, sys; sys.path.insert(0, "src")
from tests.test_araroopat_clitic_peeler import STRESS, NATIVE_AMBIGUOUS
from arabic_eval.tokenizers.araroopat_backend import (MorphAnalyzer, peel_candidates,
                                                       _dict_to_analysis, PREPOSITION_INVENTORY)
from arabic_eval.tokenizers.araroopat_bridge import get_shared_bridge
br, P = get_shared_bridge(), frozenset(PREPOSITION_INVENTORY)
words = [w for w, *_ in STRESS] + [NATIVE_AMBIGUOUS, "بيرنيني", "أندرسون", "المجتهدي", "القرضة",
         "عدناني", "أومالي", "أوديس", "أوهموه", "أعطيتني", "كتاب", "مدرسة", "مدرست", "مدرسته",
         "اتكتب", "المسا", "اعرفوني"]
need = list(dict.fromkeys(w for word in words for w in [word] + [
    sp for c in peel_candidates(word, bare_alef=True) for sp in MorphAnalyzer._residual_spellings(c)]))
rec = {}
for i in range(0, len(need), 64):
    for w, cands in zip(need[i:i + 64], br.analyze(need[i:i + 64], top=32)):
        valid = [d for d in cands if _dict_to_analysis(d, P, w) is not None]
        if valid:
            rec[w] = valid
json.dump(rec, open("tests/data/araroopat_camel_recorded.json", "w"), ensure_ascii=False, indent=1, sort_keys=True)
```

### P8 — native analyses decode to CAMeL's canonical spelling, not the input — **FIXED 2026-09-22**

Measured on the pre-pass cache before the fix: **91,355 of 519,330** natively analysed rooted
types (17.6 %) had `dediac(surface) != word` — hamza seat 50 228 (`الإعرابية` read as
`الأَعْرابِيَّة`, `الاسفل` → `الأسفل`), ة/ه 27 608 (21 160 of them a ة-final word read with a
3ms ه pronoun: `همزة` → `هَمَزَهُ` → `[CLITICE_ه]` → `همزه`), ى/ي 9 085, mixed 4 895. The cause
is CAMeL's input normalisation (آ/أ/إ/ٱ → ا, ى → ي, ة → ه before the database lookup) plus an
MLE disambiguator that never looks at the spelling. Three fixes, in the order they apply:

1. **Spelling-faithful candidate walk** (`prefer_faithful`, `araroopat_backend.py`): a rooted
   top reading that does not spell the word yields to the first ranked candidate (`top=32`,
   fetched only for those words) that does — `الإعراب` gets `إِعْراب` (rank 5) instead of
   `أَعْراب`, `همزة` the feminine noun `هَمْزَة` instead of the verb + pronoun. Faithful candidate
   exists for 22 % of the hamza/ى cases (the rest are hamza-dropping writers: `ابعد`) and 44 %
   of the ة/ه ones.
2. **ة/ه slot reconciliation** (`reconcile_final_taa`): no faithful candidate and the mismatch
   is exactly that slot → the slot follows the written letter (a ة-final word never carries a
   ه pronoun; a written ه is a ه). Covers the remaining 56 % of the ة/ه class (`المتضرره`,
   `تزعمتة`).
3. **Written-form reconstruction surfaces** (`entry_realization` / `_build_reconstruction`):
   the surface of a (root, pattern) pair is the most frequent *written* clitic-stripped stem
   (occurrence-weighted, accepted only when `join_word` — the decoder's own joins — gives the
   chunk back), CAMeL's `diac` only as a fallback (110 of 127 293 pairs on the full corpus).
   This is what makes `الإعرابية` (no faithful candidate in CAMeL's database) decode with the
   writers' إ.

Plus a closed-class fix found on the way (`_particle_analysis`): the alef-insensitive inventory
match let the verb `كان` ("was", CAMeL POS `verb`) become `[FUNC_كأن]` and decode as `كأن` (27
chunk types: `كان وكان فكان أكان كانك …`, plus `أفلام` → `أم`, `آن` → `أن`). An alef-*inexact*
match now needs a closed-class reading (not `verb` / `noun*` / `adj*`; `verb_pseudo` stays), and
an exactly-spelled inventory entry beats the alphabetically first fold (`إنما` = `إن + ما`).
A hamza the writer dropped still canonicalises (`الى` → `إلى`, `انها` → `أنها`).

`_CACHE_FORMAT` 8 → 10 with **migration** instead of a rebuild (`_reusable_cache_entries`):
only the entries the new rules can change are re-analysed (91 355 rooted + ~600 particle
entries, 4 min over the bridge). Format characters (U+200B, ZWNJ, bidi marks — 70 U+200B in
the 250 held-out references) are dropped at encode (`normalize_text`) instead of `<unk>`.
Measured with `scripts/measure_araroopat_roundtrip.py` (v2 → v3 tokenizer, same params):
reference round-trip chrF 89.11 → 90.85 raw, 96.71 → 98.68 with reference diacritics
stripped; exact match on 2 000 distinct corpus words 96.40 % → 99.15 % (ة/ه 33 → 0, hamza
29 → 16 — all hamza-dropped or hamza-added spellings canonicalised to the corpus majority —
ى/ي 7 → 1). Tests: `tests/test_araroopat_decode_canonical.py`. The v3 numbers are those of
the format-9 build; the format-10 rebuild (the كان fix) is measured in the 2026-09-22 report.

---

## 7. Verification commands

```bash
cd /home/s3user/tokenizers_evaluation

# full suite
.venv/bin/python -m pytest tests/ -q

# 7 roundtrip cases (clear the smoke cache first)
rm -rf outputs/tokenizers/araroopat_smoke outputs/tokenizers/araroopat_smoke_cache
.venv/bin/python scripts/smoke_araroopat.py

# intrinsic + morphological metrics under the corrected instrument
.venv/bin/python scripts/evaluate_intrinsic.py \
    --tokenizer-path outputs/tokenizers/araroopat_hashfix --type araroopat \
    --num-samples 3000 --output /tmp/after.json
```

**Roundtrip expectation.** 8 of 9 exact; the one failure is P2 (`ذهب الطفل إلى المدرسة` →
`ذهب الطفل إل ? المدرسة`). If anything else diverges, you broke something.

**Generation compliance (P1)** — this is the measurement to re-run after fixing it. Note it
must use `araroopat_balanced`, the vocabulary the trained checkpoint was built against:

```python
# .venv/bin/python
import sys, torch; sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM
from arabic_eval.tokenizers.araroopat import AraRooPatTokenizer

tok = AraRooPatTokenizer(); tok.load('outputs/tokenizers/araroopat_balanced')
m = AutoModelForCausalLM.from_pretrained(
    'outputs/experiments/araroopat_3phase_with_sft/training/sft',
    dtype=torch.bfloat16).to('cuda').eval()

def cls(i):
    s = tok._reverse_vocab.get(i, '')
    for p in ('[ROOT_', '[PAT_', '[CLITICP_', '[CLITICE_', '[CHAR_', '[LIT_', '[PUNCT_', '[DIGIT_'):
        if s.startswith(p):
            return p.strip('[_')
    return s

p = "السياق: ذهب الولد إلى المدرسة\nالسؤال: أين ذهب الولد\nالإجابة:"
ids = torch.tensor([tok.encode(p).input_ids], device='cuda')
g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
               max_new_tokens=40, do_sample=False, pad_token_id=0, eos_token_id=2)
new = g[0, ids.shape[1]:].tolist()
print([cls(i) for i in new])          # every ROOT must be followed by PAT
print(repr(tok.decode(new)))          # must not be ''
```

**Reference values for `araroopat_hashfix`** (3,000 eval texts, sample 466) — a regression check:

```
fertility 4.4688 · compression_ratio 1.3176 · unk_rate 0.0132 · vocab_coverage 0.9445
root_conservation_rate 0.3541 · root_conservation_attainable 0.4071 · root_measurable_pct 84.33
pattern_conservation_rate 0.3969 · semantic_fragmentation_ratio 2.9388
morph_alignment_coverage 0.2082 · morph_alignment_ceiling 0.6202 · root_extractor_agreement 0.7951
```

---

## 8. Traps — things that will cost you a day

* **Do not delete `#` from roots.** §3.1. It is a radical; deleting it renumbers the survivors
  and desynchronizes them from the pattern's slot digits.
* **Do not canonicalize `#` to a guessed radical.** It merges سور/سار anyway and hides the
  ambiguity. `#` is honest about what CAMeL discarded.
* **Keep `strip_proclitics_from_start` and `join_proclitics` inverse.** Break the symmetry and
  every `لل…` word round-trips wrong.
* **Three normalization sites must stay in sync**, two of them across the venv boundary:
  `_dict_to_analysis` and `normalize_pattern` (backend) and `_normalize_pattern` /
  `_op_generate` (server). If the client and server disagree on root normalization, tier-2
  reconstruction silently never matches and everything degrades to tier-3 naive fill.
* **Invalidate the pre-pass cache** after touching any normalization (§5), or you train on
  stale analyses with no warning.
* **Never call `self._vocab_root_set()` inside a loop over corpus entries.** §3.4.
* **Do not source `root_extractor_agreement` from CAMeL.** araroopat's ROOT token *is* CAMeL's
  root, so CAMeL-as-ground-truth drives its RPS to ~1.0 by construction and measures nothing.
  qalsadi must stay the independent judge.
* **Do not overwrite `outputs/tokenizers/araroopat_balanced`.** It pairs with the archived
  experiment; overwriting destroys the only baseline.
* **`vocab_size` is ignored by araroopat** by design — size comes from
  `max_roots + max_patterns + fixed slots`. The CLI accepts and logs it for uniformity.
* **qalsadi has two APIs and only one returns roots.** Use `analex.Analex().check_word(w).root`,
  never `Lemmatizer().lemmatize()` (that returns the lemma).

---

## 9. Suggested order

1. **P1** — self-contained, no retraining, removes a silent corruption. Highest value.
2. **P3** — small, and it hardens the same seam as P1.
3. **P2** — a one-line change with a wide blast radius; budget time for re-baselining all nine
   tokenizers and one araroopat retrain.
4. **P4** — the expensive experiment. Worth doing once P1–P3 have settled the tokenizer, so you
   measure one thing rather than four.
5. **P5 / P6** — judgement calls, not defects. Decide deliberately.
