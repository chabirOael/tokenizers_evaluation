---
name: arabic-token-eval
description: Use this skill for any work in the Arabic Tokenizers Evaluation Platform — adding/modifying tokenizers, models, tasks, or intrinsic/morphological metrics; running experiments and sweeps; debugging tokenizer→model integration; interpreting Arabic root/pattern/morpheme conservation results. TRIGGER on questions or code edits that touch `src/arabic_eval/**`, `configs/**`, `scripts/{train_tokenizer,run_experiment,evaluate_intrinsic,compare_results}.py`, or any of the 7 tokenizers (BPE, WordPiece, MorphoBPE, CharacterBERT, char-JABER, FarasaCharacterBERT, Charformer), or when the user mentions root_conservation_rate / pattern_conservation_rate / morpheme_integrity_rate / GBST / qalsadi / Farasa / wazn.
---

# Arabic Tokenizer Evaluation — Skill

Companion to `CLAUDE.md`. Read that first for the full architecture; this skill captures the gotchas that bite during edits and the conceptual frame for the morphological metrics.

## When working in this repo, anchor to these facts

### The 7 tokenizers split into 4 architectural families

| Family | Members | Embedding type | Unit |
|---|---|---|---|
| Subword | `bpe`, `wordpiece`, `morpho_bpe` | `standard` | learned subword piece |
| Word/morpheme + CharCNN | `character_bert`, `farasa_character_bert` | `character_cnn` | whole word OR Farasa morpheme; chars feed CharCNN |
| Character | `char_jaber` | `char_jaber` | single character |
| Learned-block byte | `charformer` | `charformer` | byte; GBST module *inside the model* enumerates blocks of size 1..M, scores them, soft-mixes per position, mean-pool downsamples by `d_s` |

Every dispatch in the platform branches on **`embedding_type`**, not on tokenizer name. When you add a tokenizer, decide which embedding family it belongs to first; only invent a new one if none of the existing four fits.

**Pre-segmentation and embedding family are orthogonal.** `farasa_character_bert` shows this: Farasa pre-segmentation (the front-end of MorphoBPE) can be paired with the CharCNN embedding (the back-end of CharacterBERT) by subclassing `CharacterBERTTokenizer` and overriding `train`/`encode` to apply `segment_with_farasa()` before delegating to `super()`. When the only thing you're changing is the pre-step, **subclass — don't copy the encoding logic**. The CharCNN dispatch and `CharacterCNNCollator` are reused unchanged.

**Where the work happens differs across families.** For the first three families the tokenizer *file* does most of the work (vocab building, segmentation, char-id encoding). **For Charformer the tokenizer file is essentially a no-op** — `train()` literally returns immediately, the vocab is a fixed 260 ids (256 bytes + 4 specials), and all the actual "subword learning" lives in `models/embeddings/charformer_embed.py` (the `GBSTEmbedding` module). When debugging or extending Charformer, look in the embedding module first; the tokenizer file has nothing interesting to change.

### Tokenizers "preserve morphology" in five different ways

These are not alternative implementations of the same idea; they are different commitments:

- **MorphoBPE** *exposes* morphology — Farasa pre-segments, BPE trains on segments → token boundaries align with Farasa morpheme boundaries by construction. Inherits Farasa's segmentation errors.
- **FarasaCharacterBERT** *exposes morphology and composes characters* — Farasa pre-segments (same as MorphoBPE), then each morpheme is encoded by a CharCNN over its characters (same as CharacterBERT) instead of as a learned BPE merge. Output head indexes a *morpheme* vocabulary, not a word vocabulary. Smaller `max_char_len` (default 25) suffices because morphemes are 2–5 chars typically.
- **CharacterBERT** *defers* morphology — never splits a whitespace word; CharCNN must learn morphological features from raw chars during training. Internal Arabic morpheme structure (و+ال+كتاب) is invisible in the token stream.
- **WordPiece / BPE** are agnostic — splits land wherever frequency drives them, sometimes coinciding with morpheme boundaries, often not.
- **Charformer** *learns* morphology end-to-end — no pre-segmentation, no fixed split. GBST enumerates byte blocks of size 1..M at every position, scores each candidate via a learned linear head, softmax-mixes per position, then mean-pool downsamples. Soft, position-wise, data-driven; the "subword inventory" is implicit in the block-scoring weights. Crucially the unit of learning is the *byte*, not the character — so for Arabic the model has to discover that 2 bytes form a letter before it can discover any sub-letter or sub-morpheme structure.

If a user is choosing between these, the right answer depends on what kind of commitment they want morphology to be: discrete-and-interpretable (MorphoBPE), discrete-with-compositional-units (FarasaCharBERT), implicitly-learned-from-chars (CharBERT), emergent-from-frequency (BPE/WordPiece), or learned-end-to-end-from-bytes (Charformer). FarasaCharBERT is still the only one that gives you Farasa's discrete boundaries *and* CharCNN's character composition; Charformer is the only one with no fixed boundaries at all.

## Morphological metrics — interpreting and not over-interpreting

Five metrics live in `src/arabic_eval/evaluation/intrinsic_metrics.py`:

| Metric | What it measures | Mechanical extremes (flag, don't fix) |
|---|---|---|
| `root_conservation_rate` (RPS) | root letters all inside one token | CharBERT high but not 1.0 (qalsadi extracts non-subsequence roots for irregular forms; observed 0.67); char-JABER ≈ 0.0 by construction; FarasaCharBERT typically high (~0.75–0.85, observed 0.775 on smoke test) — root usually sits inside a non-split stem morpheme but Farasa over-segmentation occasionally splits it |
| `pattern_conservation_rate` (PIS) | stem-span pattern (clitics trimmed) recoverable from one token | CharBERT high; char-JABER 0.0; FarasaCharBERT high (clitics already stripped by Farasa, stem morpheme not further split) |
| `morpheme_integrity_rate` | Farasa morpheme boundaries that align with token boundaries | MorphoBPE ≈ 1.0 (non-trivial — by design); CharBERT ≈ 0.0; char-JABER ≈ 1.0 (mechanical); FarasaCharBERT ≈ 1.0 (mechanical); Charformer = `None` (byte tokens never reconstruct to Arabic-letter offsets, so alignment fails uniformly — *not measurable*, not 1.0) |
| `clitic_separation_accuracy` (CSA) | clitic↔stem boundaries that align with token boundaries | char-JABER ≈ 1.0 (every char boundary is a token boundary, mechanical); MorphoBPE ≈ 1.0 (Farasa pre-segments clitics — non-trivial); CharBERT ≈ 0.0 (no internal boundaries); BPE / WordPiece varies — this is the discriminating signal among plain subword tokenizers; Charformer = `None` (alignment-dependent, see above) |
| `semantic_fragmentation_ratio` (SFR) | raw_tokens / Farasa morphemes — alignment-free | char-JABER ~2.7; Charformer ~5.4 (each Arabic char = 2 bytes); MorphoBPE ~1.0; BPE ~1.1; CharBERT ~0.5 (under-segments — 1 token per multi-morpheme word); SFR ≈ 1.0 = morpheme-aligned grain; >1 over-fragments; <1 under-segments |
| `root_bearing_token_pct` | % of tokens containing a full root from the sample | char-JABER ≈ 0% mechanical; Charformer ≈ 0% mechanical (one byte cannot hold a 3-letter root, and most Arabic letters span 2 bytes) |
| `pattern_bearing_token_pct` | % of tokens whose stem span matches a sample pattern | char-JABER ≈ 0% mechanical; Charformer ≈ 0% mechanical |

**Charformer is the most extreme mechanical case for Arabic.** Each token is one *byte*, not one character — and most Arabic letters are 2 bytes in UTF-8. Root and pattern conservation are mechanical zeros, bearing-token metrics are mechanical zeros (raw count > 0 but cleaned tokens are empty). Crucially, `morpheme_integrity_rate` and `clitic_separation_accuracy` are reported as **`None`** (not 1.0) — alignment-dependent metrics rely on `aligned_token_offsets` which fails uniformly for byte tokens (they don't reconstruct to Arabic-letter offsets). The discriminator for Charformer is `semantic_fragmentation_ratio`, which IS alignment-free (raw token count / Farasa morpheme count) and runs at ~5.4 — the highest in the panel. Don't try to "fix" the `None` to 1.0; it correctly reflects that the metric is not measurable for this tokenizer family.

**Don't suppress the mechanical extremes.** Reporting `~1.0` for CharBERT on root conservation and `~1.0` for char-JABER / FarasaCharBERT on morpheme integrity is correct and useful — it shows the architectural ceiling. Do flag them in comparison tables (e.g. with an asterisk + footnote) so a reader doesn't conclude char-JABER or FarasaCharBERT is the "best" tokenizer for morphology.

The metric that genuinely separates the field is **`morpheme_integrity_rate`** for *plain* subword tokenizers (BPE/WordPiece) vs *Farasa-aware* ones (MorphoBPE/FarasaCharBERT). Among the Farasa-aware ones, integrity is mechanical for both, so use **`root_conservation_rate`** + downstream task scores to break ties: MorphoBPE can lose root letters when BPE merges *within* a morpheme produce sub-pieces, while FarasaCharBERT cannot (each morpheme is one atomic unit feeding the CharCNN).

### `None` vs `0.0` on the bearing-token metrics — two distinct cases

The bearing-token metrics (`root_bearing_token_pct`, `pattern_bearing_token_pct`) can come back as either `None` or `0.0`, and they mean different things. The current implementation distinguishes them:

- **`None`** — no tokens were generated for the sample at all (or no roots could be extracted). Truly not measured.
- **`0.0`** — tokens *were* generated, but every one cleaned to an empty Arabic-letter string after `clean_token_string()`. This is the byte-level mechanical-zero case (Charformer) — single bytes can't carry a 3-letter root, mechanically. Reporting `0.0` here is correct, not a fallback hiding a bug.

This distinction is wired up via a `raw_token_count` counter that runs in parallel to the cleaned-token list. **Don't replace it with a blanket `None → 0.0` fallback** — `None` still has a real meaning (the Farasa-unavailable case keeps it).

## Critical implementation gotchas

### qalsadi has two APIs; only one returns roots

```python
# WRONG — returns the lemma (dictionary form), not the root.
from qalsadi.lemmatizer import Lemmatizer
Lemmatizer().lemmatize("والكتاب")  # → "كتاب"  (lemma, not root)

# CORRECT — proper root.
from qalsadi.analex import Analex
Analex().check_word("والكتاب")[0].root  # → "كتب"
```

`RootExtractor` already gets this right. If you change root extraction, **keep the qalsadi → tashaphyne → consonant-skeleton fallback chain in that order** and verify on test words like `والكتاب` (root `كتب`), `مدرسة` (root `درس`), `يدرسون` (root `درس`).

### ByteLevel BPE tokens are byte-encoded, not Arabic strings

The HF `tokenizers` `ByteLevel` pre-tokenizer (used by our `BPETokenizer`) emits token strings like `'Ø§ÙĦÙĥØªØ§Ø¨'`, which is `'الكتاب'` encoded byte-by-byte through GPT-2's `bytes_to_unicode` mapping. `clean_token_string()` reverses this via `_BYTELEVEL_INV`. **If you write a new tokenizer that uses any byte-level encoding, make sure cleaned token strings are Arabic before computing token-level metrics — otherwise everything cleans to empty and `root_bearing_token_pct` silently becomes `None`.**

### Farasa = Java subprocess

- `morpheme_integrity_rate` requires `java` on PATH. Without it the metric returns `None` and a warning is logged. Don't treat the `None` as a bug; check `which java` first.
- Farasa init takes ~2 s per word for the very first call (subprocess startup) then settles to ~20 words/sec. For a 500-word sample that's ~30 s overhead per experiment — don't be surprised by it on sweeps.
- `farasapy` warns about "interactive mode" on long lines; benign, but if it gets noisy in sweeps switch `MorphemeSegmenter._ensure()` to `interactive=False`.

### CharacterBERT input is 3D, generation is unsupported (but log-likelihood IS supported)

- Input shape: `[batch, seq_len, max_char_len]`. Don't try to feed `input_ids` to a model adapter expecting it; use `char_ids`.
- `LlamaAdapter.generate()` raises `NotImplementedError` for `CHARACTER_CNN`. QA evaluation falls back to empty predictions. Don't add a generation loop without redesigning the output head — the word-vocab lm_head can't reconstruct OOV words.
- LightEval log-likelihood scoring **IS** supported for `character_cnn`. The `character_cnn` branch in `_compute_loglikelihood` (`tasks/lighteval_benchmarks.py:118`) feeds `char_ids` through the CharCNN and scores continuations against the *word-vocabulary* `lm_head`. CharBERT returns a real accuracy in `[0, 1]` — not 0.0. (Older docs and tests claimed 0.0; that was true before this branch was added and is no longer accurate. If you see a comment or test asserting "CharBERT must return accuracy=0.0", it's stale.)
- **`farasa_character_bert` inherits the same paths** — same `embedding_type=character_cnn`, same adapter, same collator, same `generate()` NotImplementedError, same log-likelihood support. The only difference is that its `lm_head` indexes a *morpheme* vocab instead of a word vocab, so continuation scoring is over morpheme-vocab IDs. Real accuracy in `[0, 1]` on log-likelihood benchmarks.

### char-JABER sequence length

Each character is one token → sequences are 4–6× longer than subword. Default `CharJaberCollator` uses `max_length=2048` (vs 512 for subword). If you change this, also revisit the `downsample_factor` in `CharJaberEmbedding` (currently 1 = disabled).

**Latent issue: char-JABER `downsample_factor>1` is broken today.** `CharJaberEmbedding` will shrink the sequence by `downsample_factor` internally, but the `LlamaAdapter` standard forward path (used for `CHAR_JABER`) feeds the *full-length* `attention_mask` straight to `self._model(...)`. The mask length and the embedded length stop agreeing. The default of 1 hides this. If you ever want to revive char-JABER downsampling, mirror what `_forward_charformer` does: shrink the mask via `mask.view(B, -1, d_s).any(-1)` *before* the transformer, and use `CharJaberOutputHead` to upsample logits back. Don't just bump the default — that crashes silently.

### Charformer (GBST): the tokenizer is in the model, not the tokenizer file

Charformer's tokenizer file (`tokenizers/charformer.py`) is a no-op byte encoder with a fixed 260-id vocab — `train()` literally returns immediately. The actual learning lives in `models/embeddings/charformer_embed.py` as `GBSTEmbedding`. When debugging or extending Charformer, that's the file to open.

Things that bite when working with it:

- **GBST is non-causal within the block window.** A block of size `b` at position `i` pools `X[i:i+b]`, so position `i` sees up to position `i+M-1` where M is `max_block_size` (default 4). This is harmless for teacher-forced LM training and log-likelihood scoring (which is what our pipeline does), but it breaks naive autoregressive byte generation. `LlamaAdapter.generate()` raises `NotImplementedError` for `CHARFORMER`. **Don't try to fix this with a "causal GBST" variant** without also reading the paper carefully — the original Charformer is encoder-decoder, so causality was never an issue there.

- **The transformer operates on the downsampled sequence.** GBST output shape is `[B, ceil(L / d_s), D]`. So before the transformer runs, you must (a) shrink `attention_mask` to match — `_forward_charformer` does this via `mask.view(B, -1, d_s).any(-1)` after right-padding; (b) confirm the mask and the embedding agree on length (mean-pool's edge cases can drift by 1; `_forward_charformer` clips to `min(...)` to recover). Labels stay at byte length because `CharformerOutputHead` upsamples logits back via `ConvTranspose1d(stride=d_s)`. If you change `downsample_rate`, the output head's `upsample_factor` MUST be the same value or labels will misalign by a multiplicative factor (silent loss garbage).

- **Mechanical-zero on every Arabic morphology metric.** Each token = 1 byte; most Arabic letters need 2 bytes; therefore no token can hold a single Arabic letter, let alone a 3-letter root. Expect `root_conservation_rate = 0`, `pattern_conservation_rate = 0`, `root_bearing_token_pct = 0`, `pattern_bearing_token_pct = 0`. `morpheme_integrity_rate ≈ 1` mechanically (every byte boundary is a token boundary, so morpheme boundaries trivially align). Use downstream LightEval scores to compare Charformer against the others — the morphology metrics tell you nothing about it.

- **Not a "fix it" — a new metric path:** the `0.0`-vs-`None` distinction in §"Morphological metrics" was added specifically because Charformer hits the "tokens generated but cleaned-to-empty" case. A future contributor who "fixes" the metric to return `None` would mask Charformer's mechanical zero and make sweeps harder to interpret.

- **Generation isn't supported, and that's fine for the current task suite.** With text_generation and question_answering being phased out, Charformer's coverage is: intrinsic metrics ✓, perplexity (teacher-forced) ✓, LightEval log-likelihood MCQ ✓. No generation path needed.

### Optional eval features: opt-in via signature, not via abstract base

When adding an *optional* per-task evaluation feature (failure-case CSVs, per-example diagnostics, attention dumps, etc.), do **not** widen `BaseTask.evaluate()` in [tasks/base.py](src/arabic_eval/tasks/base.py). Instead:

1. Add the kwarg (e.g. `failure_report_dir: Optional[Path] = None`) to the *concrete* `evaluate()` of the task(s) that support it.
2. In [pipeline/experiment.py](src/arabic_eval/pipeline/experiment.py), gate the kwarg via `inspect.signature(task.evaluate).parameters` — only pass it when the active task accepts it; log-and-skip otherwise.

Why: the four LightEval MCQ tasks share `LightEvalBenchmarkTask.evaluate`, while QA / text_generation have entirely different evaluation shapes (generation, sliding-window perplexity). Forcing every task to accept (and ignore) the kwarg pollutes signatures and creates pressure to half-implement the feature for tasks where it doesn't make sense. Signature-based gating keeps the abstract contract minimal and lets each family opt in cleanly. Used today for `failure_reports` (LightEval-only); reuse this pattern for any future per-task evaluation switch.

The `failure_reports` CSV: one row per wrong-answer example; per-choice log-likelihoods + an `ll_margin = ll_pred - ll_gold`. The margin is the diagnostic — it separates "model was confidently wrong" from "model was nearly right." When users ask "why is accuracy bad?", the histogram of `ll_margin` answers it more usefully than the raw accuracy number does.

## Common workflows

### Adding a new tokenizer (checklist)

1. `src/arabic_eval/tokenizers/<name>.py` implementing `BaseTokenizer` (all abstract methods + properties), decorated with `@tokenizer_registry.register("<name>")`.
2. Add import in `src/arabic_eval/tokenizers/__init__.py` (must succeed without torch — keep imports minimal).
3. Make sure `encode()` populates `TokenizerOutput.tokens` (string list) — the morphological metrics depend on it. If your tokenizer uses byte-level encoding, also extend `clean_token_string` (or its `_try_decode_bytelevel` helper) to handle it.
4. Pick `embedding_type` from the existing 4 if possible; only add a new one with a matching adapter branch in `LlamaAdapter.adapt_to_tokenizer()` and a matching collator.
5. **If your embedding shrinks the sequence (any kind of pooling/striding/downsampling), you also need (a) a custom `_forward_<type>` that downsamples `attention_mask` to match the embedded length, and (b) an output head that upsamples logits back to the byte/char/token grid where labels live.** `_forward_charformer` + `CharformerOutputHead` is the working reference. Don't expect the standard HF forward to handle this for you — it won't.
6. `configs/tokenizers/<name>.yaml` with sane defaults.
7. Smoke-test the morphological metrics on it (see "Smoke test recipe" below) — make sure `root_bearing_token_pct` is `0.0` or a real number (not `None`). For byte-level tokenizers `0.0` is the correct mechanical answer; `None` only when no tokens were generated at all. If you see `None` despite real input, that's byte-encoding leakage in `clean_token_string`.

### Smoke test recipe (Farasa enabled, ~30 s)

```python
from arabic_eval.tokenizers.<your_tok> import YourTokenizer
from arabic_eval.evaluation.intrinsic_metrics import compute_morphological_metrics
texts = [...] * 10  # ~150–200 short Arabic sentences
tok = YourTokenizer(); tok.train(texts, vocab_size=500)
m = compute_morphological_metrics(tok, texts, sample_size=40, use_farasa=True)
# Expect: every metric is a float (not None) for subword tokenizers.
# For word-level / char-level expect the mechanical extremes documented above.
```

### Running an experiment

- Single: `python scripts/run_experiment.py --config configs/experiments/<name>.yaml`
- Sweep: add `--sweep` and use `configs/experiments/full_sweep.yaml` or `benchmark_sweep.yaml`.
- Intrinsic-only: `python scripts/evaluate_intrinsic.py --tokenizer-path outputs/tokenizers/<name> --type <type>`.

### Disabling the morphological metrics

For a fast turnaround experiment (e.g. iterating on training hyperparameters) set `evaluation.morphological_metrics: false` — the size/coverage metrics still run.

## Things to push back on

- **"Just split each token character by character to compute root conservation."** No — the metric is *per-token*; that would conflate the metric with the tokenizer's granularity. The right algorithm is `contains_subsequence(token, root)` over each *whole* token, which is what's implemented.
- **"CharBERT scores 100% on root conservation, so it must be the best Arabic tokenizer."** It hits the architectural ceiling because it doesn't split at all. Use `morpheme_integrity_rate` and downstream task scores to break ties.
- **"Just widen `BaseTask.evaluate()` to take the new optional eval flag — that's the cleanest way."** No — the LightEval, QA, and text_generation tasks have fundamentally different evaluation shapes; forcing every one to accept (and silently ignore) a new kwarg pollutes signatures and creates pressure to half-implement features where they don't fit. The established pattern is signature-based gating in the pipeline (`inspect.signature(task.evaluate).parameters`). The `failure_reports` flag does this today; new per-task eval flags should follow suit. See §"Optional eval features: opt-in via signature, not via abstract base".
- **"Add a fallback that returns 0.0 instead of None when Farasa is unavailable."** Don't — `0.0` would be indistinguishable from a real "tokenizer respects no boundaries" result. `None` correctly signals "not measured." (Note: this rule still holds for `morpheme_integrity_rate`. The bearing-token metrics have a *separate* carve-out where `0.0` is the right answer when tokens were generated but cleaned to empty Arabic-letter strings — that's the byte-level mechanical-zero case, see §"Morphological metrics".)

- **"Charformer's GBST is just a tokenizer — go put it in `src/arabic_eval/tokenizers/charformer.py`."** Half right — there is a tokenizer file, but it's a no-op byte encoder. The actual learning is `GBSTEmbedding` in `models/embeddings/charformer_embed.py`. Hyperparameter changes (M, d_s, conv_kernel) flow from the tokenizer YAML's `params` dict through `get_embedding_config()` to the embedding module's constructor; you almost never want to touch the tokenizer file itself.

- **"Charformer should be able to generate text — it's a 'character transformer' after all."** It can in the original paper *because* the original Charformer is encoder-decoder and the decoder is a normal byte-level decoder. In our decoder-only setup, GBST's block-pooling looks ahead within blocks of size up to M, which is incompatible with autoregressive decoding. Don't try to "make it causal" by changing the block-formation direction without reading the paper carefully — that's a different model.

- **"Bump char-JABER's `downsample_factor` to 2 to make it faster."** Latent bug: `LlamaAdapter` doesn't shrink the attention mask for `CHAR_JABER` (it uses the standard forward path). The default of 1 hides this. If you want char-JABER downsampling, add a `_forward_char_jaber` that mirrors `_forward_charformer`'s mask-shrinking logic — don't just bump the default.
- **"Bump qalsadi to a newer API to avoid the import inside `_ensure_backends`."** The lazy init is intentional — qalsadi loads SQLite databases on construction; eager-importing it slows down every tokenizer-only workflow. Keep it lazy.
- **"Copy the CharacterBERT class into a new file to add Farasa pre-segmentation."** Don't — `FarasaCharacterBERTTokenizer` shows the right pattern: subclass `CharacterBERTTokenizer`, override `train`/`encode` to apply `segment_with_farasa()` first, then `super()`. Copying the class duplicates ~150 lines (char vocab building, char-id encoding, save/load, special tokens) that have no reason to diverge.
- **"FarasaCharBERT scores ~1.0 on `morpheme_integrity_rate` so it must beat MorphoBPE on Arabic morphology."** Both score ~1.0 — for FarasaCharBERT it's mechanical (each morpheme is exactly one unit), for MorphoBPE it's by-design but still essentially baked in by the Farasa pre-step. Use `root_conservation_rate` and downstream tasks to break the tie, not integrity.
- **"CSA and `morpheme_integrity_rate` measure the same thing — drop one."** No: CSA restricts to clitic boundaries (proclitic-end / enclitic-start positions). Integrity covers all Farasa boundaries including stem-internal ones. They satisfy `integrity == 1.0 ⇒ CSA == 1.0` but can disagree in general. Integrity catches stem-internal splits (BPE chopping a root in half); CSA does not. Both are needed.
- **"Add a `None → 1.0` fallback for Charformer's CSA / integrity since the byte boundaries trivially align."** No: alignment is the protocol for these metrics, and byte tokens don't align to Arabic-letter offsets. Reporting `None` correctly marks the metric as not-measurable for this tokenizer family. SFR is the alignment-free discriminator; that's where Charformer comparisons should land. Forcing 1.0 would falsely suggest Charformer perfectly preserves morpheme/clitic structure when in fact the metric simply can't be computed for it.
