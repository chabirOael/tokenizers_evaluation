---
name: arabic-token-eval
description: Use this skill for any work in the Arabic Tokenizers Evaluation Platform — adding/modifying tokenizers, models, eval tasks, or intrinsic/morphological metrics; running experiments and sweeps; tuning the 3-phase training pipeline; debugging tokenizer→model integration; interpreting Arabic root/pattern/morpheme conservation results. TRIGGER on questions or code edits that touch `src/arabic_eval/**`, `configs/**`, `scripts/{train_tokenizer,run_experiment,evaluate_intrinsic,compare_results}.py`, or any of the 8 tokenizers (BPE, WordPiece, MorphoBPE, CharacterBERT, char-JABER, FarasaCharacterBERT, Charformer, AraRooPat) plus the `native_llama` wrapper, or when the user mentions root_conservation_rate / pattern_conservation_rate / morpheme_integrity_rate / GBST / qalsadi / Farasa / wazn / 3-phase pipeline / embedding alignment / Phase 1/2/3 / Arabic-SQuAD / TyDiQA / ARCD.
---

# Arabic Tokenizer Evaluation — Skill

Companion to `CLAUDE.md`. Read that first for the full architecture; this skill captures the gotchas that bite during edits and the conceptual frame for the morphological metrics.

## When working in this repo, anchor to these facts

### Training is a fixed 3-phase pipeline (the only training path)

There is no per-task SFT, no 10/90 benchmark split, no `completion_only_loss` flag. Every training run executes the same three phases regardless of tokenizer / model / eval task. The phases are independently toggleable via `enabled` flags but always run in fixed order when on.

| Phase | Trains | Frozen? | Dataset | Loss | Default budget |
|---|---|---|---|---|---|
| `embedding_alignment` (Phase 1) | `embed_tokens` + `lm_head` only | body frozen | `arabic_squad` | full-sequence causal LM | 1000 steps, LR=1e-3, BS=8 |
| `warmup` (Phase 2) | all params | none | `arabic_squad` | answer-only | 2000 steps, LR=2e-4, BS=4×4 |
| `sft` (Phase 3) | all params | none | `tydiqa_arabic + arcd` | answer-only | 2000 steps, LR=2e-4, BS=4×4 + early-stop |

After training, eval runs on **every task in `sweep.tasks`** (default ACVA + Alghafa + arabic_exam) using the **full benchmark** — no rows are reserved for SFT, since training is task-agnostic.

**Why this design.** The previous "10% SFT on benchmark + 90% eval" was destructive: each from-scratch tokenizer's vocab indices were silently mapped onto Llama's first N pretrained rows by `resize_token_embeddings`, and 10% benchmark-specific SFT couldn't drift those mappings far enough to find real signal. The 3-phase pipeline (a) deliberately aligns embeddings before any other training (Phase 1), (b) teaches QA format on a regular translated dataset before exposure to native Arabic complexity (Phase 2), and (c) does the decisive SFT on native Arabic QA in Phase 3.

**Two reference experiments:**
- `configs/experiments/native_llama_3phase_with_sft.yaml` — full pipeline (Phase 1 + 2 + 3)
- `configs/experiments/native_llama_3phase_no_sft.yaml` — Phase 1 + 2 only (sets `sft.enabled: false`)
- `configs/experiments/sample_full.yaml` — fully-documented template

### The 8 tokenizers split into 4 architectural families

| Family | Members | Embedding type | Unit |
|---|---|---|---|
| Subword | `bpe`, `wordpiece`, `morpho_bpe`, `araroopat` | `standard` | learned subword piece (or root+pattern token) |
| Word/morpheme + CharCNN | `character_bert`, `farasa_character_bert` | `character_cnn` | whole word OR Farasa morpheme; chars feed CharCNN |
| Character | `char_jaber` | `char_jaber` | single character |
| Learned-block byte | `charformer` | `charformer` | byte; GBST module *inside the model* enumerates blocks of size 1..M, scores them, soft-mixes per position, mean-pool downsamples by `d_s` |

Every dispatch in the platform branches on **`embedding_type`**, not on tokenizer name. When you add a tokenizer, decide which embedding family it belongs to first; only invent a new one if none of the existing four fits.

`native_llama` is a 9th wrapper that uses Llama-3.2-1B's pretrained tokenizer — `embedding_type=standard`, `train()` is a no-op, vocab=128256. Treat it as a normal tokenizer under the 3-phase pipeline (the previous "baseline-only" carve-out is gone — under the new pipeline every tokenizer runs the same phases).

**Pre-segmentation and embedding family are orthogonal.** `farasa_character_bert` shows this: Farasa pre-segmentation (the front-end of MorphoBPE) can be paired with the CharCNN embedding (the back-end of CharacterBERT) by subclassing `CharacterBERTTokenizer` and overriding `train`/`encode` to apply `segment_with_farasa()` before delegating to `super()`. When the only thing you're changing is the pre-step, **subclass — don't copy the encoding logic**. The CharCNN dispatch and `CharacterCNNCollator` are reused unchanged.

**Where the work happens differs across families.** For the first three families the tokenizer *file* does most of the work (vocab building, segmentation, char-id encoding). **For Charformer the tokenizer file is essentially a no-op** — `train()` literally returns immediately, the vocab is a fixed 260 ids (256 bytes + 4 specials), and all the actual "subword learning" lives in `models/embeddings/charformer_embed.py` (the `GBSTEmbedding` module). When debugging or extending Charformer, look in the embedding module first; the tokenizer file has nothing interesting to change.

### Tokenizers "preserve morphology" in five different ways

These are not alternative implementations of the same idea; they are different commitments:

- **MorphoBPE** *exposes* morphology — Farasa pre-segments, BPE trains on segments → token boundaries align with Farasa morpheme boundaries by construction. Inherits Farasa's segmentation errors.
- **AraRooPat** *encodes* morphology directly — each content word becomes `[ROOT_x] [PAT_y]` tokens via CAMeL Tools' analyzer; clitics emitted as separate `[CLITICP_*]` / `[CLITICE_*]` tokens.
- **FarasaCharacterBERT** *exposes morphology and composes characters* — Farasa pre-segments (same as MorphoBPE), then each morpheme is encoded by a CharCNN over its characters (same as CharacterBERT) instead of as a learned BPE merge. Output head indexes a *morpheme* vocabulary, not a word vocabulary.
- **CharacterBERT** *defers* morphology — never splits a whitespace word; CharCNN must learn morphological features from raw chars during training. Internal Arabic morpheme structure (و+ال+كتاب) is invisible in the token stream.
- **WordPiece / BPE** are agnostic — splits land wherever frequency drives them, sometimes coinciding with morpheme boundaries, often not.
- **Charformer** *learns* morphology end-to-end — no pre-segmentation, no fixed split. GBST enumerates byte blocks of size 1..M at every position, scores each candidate via a learned linear head, softmax-mixes per position, then mean-pool downsamples. Soft, position-wise, data-driven; the "subword inventory" is implicit in the block-scoring weights. Crucially the unit of learning is the *byte*, not the character — so for Arabic the model has to discover that 2 bytes form a letter before it can discover any sub-letter or sub-morpheme structure.

If a user is choosing between these, the right answer depends on what kind of commitment they want morphology to be: discrete-and-explicit (AraRooPat), discrete-and-interpretable (MorphoBPE), discrete-with-compositional-units (FarasaCharBERT), implicitly-learned-from-chars (CharBERT), emergent-from-frequency (BPE/WordPiece), or learned-end-to-end-from-bytes (Charformer).

## Morphological metrics — interpreting and not over-interpreting

Five metrics live in `src/arabic_eval/evaluation/intrinsic_metrics.py`:

| Metric | What it measures | Mechanical extremes (flag, don't fix) |
|---|---|---|
| `root_conservation_rate` (RPS) | root letters all inside one token | CharBERT high but not 1.0 (qalsadi extracts non-subsequence roots for irregular forms; observed 0.67); char-JABER ≈ 0.0 by construction; FarasaCharBERT typically high (~0.75–0.85, observed 0.775 on smoke test) — root usually sits inside a non-split stem morpheme but Farasa over-segmentation occasionally splits it; AraRooPat ~1.0 by construction |
| `pattern_conservation_rate` (PIS) | stem-span pattern (clitics trimmed) recoverable from one token | CharBERT high; char-JABER 0.0; FarasaCharBERT high (clitics already stripped by Farasa, stem morpheme not further split); AraRooPat ~1.0 |
| `morpheme_integrity_rate` | Farasa morpheme boundaries that align with token boundaries | MorphoBPE ≈ 1.0 (non-trivial — by design); CharBERT ≈ 0.0; char-JABER ≈ 1.0 (mechanical); FarasaCharBERT ≈ 1.0 (mechanical); Charformer = `None` (byte tokens never reconstruct to Arabic-letter offsets, so alignment fails uniformly — *not measurable*, not 1.0) |
| `clitic_separation_accuracy` (CSA) | clitic↔stem boundaries that align with token boundaries | char-JABER ≈ 1.0 (every char boundary is a token boundary, mechanical); MorphoBPE ≈ 1.0 (Farasa pre-segments clitics — non-trivial); CharBERT ≈ 0.0 (no internal boundaries); BPE / WordPiece varies — this is the discriminating signal among plain subword tokenizers; Charformer = `None` (alignment-dependent, see above) |
| `semantic_fragmentation_ratio` (SFR) | raw_tokens / Farasa morphemes — alignment-free | char-JABER ~2.7; Charformer ~5.4 (each Arabic char = 2 bytes); MorphoBPE ~1.0; BPE ~1.1; CharBERT ~0.5 (under-segments — 1 token per multi-morpheme word); SFR ≈ 1.0 = morpheme-aligned grain; >1 over-fragments; <1 under-segments |
| `root_bearing_token_pct` | % of tokens containing a full root from the sample | char-JABER ≈ 0% mechanical; Charformer ≈ 0% mechanical (one byte cannot hold a 3-letter root, and most Arabic letters span 2 bytes) |
| `pattern_bearing_token_pct` | % of tokens whose stem span matches a sample pattern | char-JABER ≈ 0% mechanical; Charformer ≈ 0% mechanical |

**Charformer is the most extreme mechanical case for Arabic.** Each token is one *byte*, not one character — and most Arabic letters are 2 bytes in UTF-8. Root and pattern conservation are mechanical zeros, bearing-token metrics are mechanical zeros (raw count > 0 but cleaned tokens are empty). Crucially, `morpheme_integrity_rate` and `clitic_separation_accuracy` are reported as **`None`** (not 1.0) — alignment-dependent metrics rely on `aligned_token_offsets` which fails uniformly for byte tokens (they don't reconstruct to Arabic-letter offsets). The discriminator for Charformer is `semantic_fragmentation_ratio`, which IS alignment-free (raw token count / Farasa morpheme count) and runs at ~5.4 — the highest in the panel. Don't try to "fix" the `None` to 1.0; it correctly reflects that the metric is not measurable for this tokenizer family.

**Don't suppress the mechanical extremes.** Reporting `~1.0` for CharBERT on root conservation and `~1.0` for char-JABER / FarasaCharBERT on morpheme integrity is correct and useful — it shows the architectural ceiling. Do flag them in comparison tables (e.g. with an asterisk + footnote) so a reader doesn't conclude char-JABER or FarasaCharBERT is the "best" tokenizer for morphology.

The metric that genuinely separates the field is **`morpheme_integrity_rate`** for *plain* subword tokenizers (BPE/WordPiece) vs *Farasa-aware* ones (MorphoBPE/FarasaCharBERT/AraRooPat). Among the morphology-aware ones, integrity is mechanical for all, so use **`root_conservation_rate`** + downstream task scores to break ties.

### `None` vs `0.0` on the bearing-token metrics — two distinct cases

The bearing-token metrics (`root_bearing_token_pct`, `pattern_bearing_token_pct`) can come back as either `None` or `0.0`, and they mean different things. The current implementation distinguishes them:

- **`None`** — no tokens were generated for the sample at all (or no roots could be extracted). Truly not measured.
- **`0.0`** — tokens *were* generated, but every one cleaned to an empty Arabic-letter string after `clean_token_string()`. This is the byte-level mechanical-zero case (Charformer) — single bytes can't carry a 3-letter root, mechanically. Reporting `0.0` here is correct, not a fallback hiding a bug.

This distinction is wired up via a `raw_token_count` counter that runs in parallel to the cleaned-token list. **Don't replace it with a blanket `None → 0.0` fallback** — `None` still has a real meaning (the Farasa-unavailable case keeps it).

## Critical implementation gotchas

### Llama-3.2-1B has tied embeddings

`lm_head.weight is model.embed_tokens.weight` — they are literally the same tensor. So `lm_head` does **not** appear in `model.named_parameters()`; only `model.embed_tokens.weight` does. The Phase 1 freeze list `["embed_tokens", "lm_head"]` is still correct (the freezing helper warns when `lm_head` matches no parameter and continues, recognizing the tied-weight case) — training `embed_tokens` IS training `lm_head`. Don't try to "fix" the warning by dropping `lm_head` from the YAML; if a future model with untied weights is used, `lm_head` will then match a real parameter and we want it trainable.

### Phase runner uses substring-match freezing with a wildcard

`PhaseConfig.trainable_parameters` is a **list of substrings** (matched against `model.named_parameters()`), or the sole sentinel `["*"]` meaning all parameters. The exact spec from the original task:

```python
for name, param in model.named_parameters():
    if "embed_tokens" in name or "lm_head" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

is faithfully implemented in `apply_trainable_filter` ([src/arabic_eval/training/freezing.py](src/arabic_eval/training/freezing.py)). Validation rejects empty lists, non-string entries, and `["*", "x"]`-style mixes. When some substrings match nothing while others do (the Llama tied-weight case), it warns and continues; only when **no** substring matches anything does it raise.

### Answer-only loss masking via LCP — shared helper

`compute_answer_only_labels` in [src/arabic_eval/data/answer_only_masking.py](src/arabic_eval/data/answer_only_masking.py) is the **single source of truth** for masking the prompt span to `-100`. The LCP (longest common prefix) approach is necessary because many tokenizers (e.g. Llama) auto-append `</s>` to standalone encodings:

- `prompt_enc = [BOS, ..., EOS]` length P
- `full_enc = [BOS, ..., answer, EOS]` length F > P
- so `prompt_enc[P-1] = EOS` but `full_enc[P-1] = first answer token`

Naive `labels[:len(prompt_enc)] = -100` would eat the first answer token. LCP walks the two lists in lockstep and stops at the first divergence. If `lcp >= len(full)` the example is dropped (truncation ate the answer). If you ever add a new training data path that needs answer-only loss, **call this helper, don't reimplement the masking**.

### Phase 3 prompt format: the `### السياق:` block

Phase 3 trains on TyDiQA-Arabic + ARCD (extractive QA), formatted as:

```
### السياق: {context}
### السؤال: {question}
### الإجابة: {answer}
```

The prompt span (everything up to but excluding `{answer}`) ends with `### الإجابة:` (no trailing space). The full text adds ` {answer}` after the colon. Format lives in [src/arabic_eval/data/finetune_corpora.py](src/arabic_eval/data/finetune_corpora.py) — `_format_qa_prompt` and `_format_qa_full`. **Don't import from any deleted `tasks/question_answering.py` — that path is gone.**

The LightEval per-task prompts (ACVA, Alghafa, etc.) are unchanged — they use their own format (e.g. `### السؤال: ... \n### الإجابة: {letter}`) defined in `_format_eval_context` / `_build_continuations`. Phase 3's QA prompt and the LightEval MCQ prompts are deliberately *different* shapes because the data is different (extractive QA vs MCQ).

### LightEval is now eval-only — no SFT methods

`LightEvalBenchmarkTask` no longer implements `_format_sft_text`, `get_dataloader`, `_build_sft_dataloader`, `_get_splits`, `train_split_ratio`, or `eval_full`. Under the 3-phase pipeline, training is task-agnostic so the benchmark contributes 0 rows to training and 100% to evaluation. The public API is now:

- `get_eval_examples()` — full list of parsed examples (after the optional `clean_latin_rows` filter)
- `evaluate(model, tokenizer, ...)` — runs LightEval log-likelihood scoring + char/PMI normalization + per-sub-config breakdown + optional failure CSV
- The 7 abstract hooks: `_default_dataset_name`, `_parse_example`, `load_examples`, `_format_eval_context`, `_build_continuations`, `_aggregate_scores`, `_unconditioned_query` (defaultable)

If you encounter old code or doc that references `_format_sft_text` / `get_dataloader` on a LightEval task, treat it as stale.

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
- `LlamaAdapter.generate()` raises `NotImplementedError` for `CHARACTER_CNN`. Generation paths are not exercised under the 3-phase pipeline (eval is log-likelihood MCQ), so this isn't a blocker.
- LightEval log-likelihood scoring **IS** supported for `character_cnn`. The `character_cnn` branch in `_compute_loglikelihood` ([tasks/lighteval/base.py](src/arabic_eval/tasks/lighteval/base.py)) feeds `char_ids` through the CharCNN and scores continuations against the *word-vocabulary* `lm_head`. CharBERT returns a real accuracy in `[0, 1]`.
- **`farasa_character_bert` inherits the same paths** — same `embedding_type=character_cnn`, same adapter, same collator, same `generate()` NotImplementedError, same log-likelihood support. The only difference is that its `lm_head` indexes a *morpheme* vocab instead of a word vocab.

### char-JABER sequence length

Each character is one token → sequences are 4–6× longer than subword. Default `CharJaberCollator` uses `max_length=2048` (vs 512 for subword). If you change this, also revisit the `downsample_factor` in `CharJaberEmbedding` (currently 1 = disabled).

**Latent issue: char-JABER `downsample_factor>1` is broken today.** `CharJaberEmbedding` will shrink the sequence by `downsample_factor` internally, but the `LlamaAdapter` standard forward path (used for `CHAR_JABER`) feeds the *full-length* `attention_mask` straight to `self._model(...)`. The mask length and the embedded length stop agreeing. The default of 1 hides this. If you ever want to revive char-JABER downsampling, mirror what `_forward_charformer` does: shrink the mask via `mask.view(B, -1, d_s).any(-1)` *before* the transformer, and use `CharJaberOutputHead` to upsample logits back. Don't just bump the default — that crashes silently.

### Charformer (GBST): the tokenizer is in the model, not the tokenizer file

Charformer's tokenizer file (`tokenizers/charformer.py`) is a no-op byte encoder with a fixed 260-id vocab — `train()` literally returns immediately. The actual learning lives in `models/embeddings/charformer_embed.py` as `GBSTEmbedding`. When debugging or extending Charformer, that's the file to open.

Things that bite when working with it:

- **GBST is non-causal within the block window.** A block of size `b` at position `i` pools `X[i:i+b]`, so position `i` sees up to position `i+M-1` where M is `max_block_size` (default 4). This is harmless for teacher-forced LM training and log-likelihood scoring (which is what our pipeline does), but it breaks naive autoregressive byte generation. `LlamaAdapter.generate()` raises `NotImplementedError` for `CHARFORMER`. **Don't try to fix this with a "causal GBST" variant** without also reading the paper carefully — the original Charformer is encoder-decoder, so causality was never an issue there.

- **The transformer operates on the downsampled sequence.** GBST output shape is `[B, ceil(L / d_s), D]`. So before the transformer runs, you must (a) shrink `attention_mask` to match — `_forward_charformer` does this via `mask.view(B, -1, d_s).any(-1)` after right-padding; (b) confirm the mask and the embedding agree on length (mean-pool's edge cases can drift by 1; `_forward_charformer` clips to `min(...)` to recover). Labels stay at byte length because `CharformerOutputHead` upsamples logits back via `ConvTranspose1d(stride=d_s)`. If you change `downsample_rate`, the output head's `upsample_factor` MUST be the same value or labels will misalign by a multiplicative factor (silent loss garbage).

- **Mechanical-zero on every Arabic morphology metric.** Each token = 1 byte; most Arabic letters need 2 bytes; therefore no token can hold a single Arabic letter, let alone a 3-letter root. Use downstream LightEval scores to compare Charformer against the others — the morphology metrics tell you nothing about it.

### Optional eval features: opt-in via signature, not via abstract base

When adding an *optional* per-task evaluation feature (failure-case CSVs, per-example diagnostics, attention dumps, etc.), do **not** widen `BaseTask.evaluate()` in [tasks/base.py](src/arabic_eval/tasks/base.py). Instead:

1. Add the kwarg (e.g. `failure_report_dir: Optional[Path] = None`) to the *concrete* `evaluate()` of the task(s) that support it.
2. In [pipeline/experiment.py](src/arabic_eval/pipeline/experiment.py), gate the kwarg via `inspect.signature(task.evaluate).parameters` — only pass it when the active task accepts it; log-and-skip otherwise.

Why: a future non-LightEval task family (perplexity, code-eval, etc.) could legitimately want a different evaluation shape. Forcing every one to accept (and silently ignore) a new kwarg pollutes signatures and creates pressure to half-implement features where they don't fit. Signature-based gating keeps the abstract contract minimal and lets each family opt in cleanly. Used today for `failure_reports` and `score_normalization` (LightEval-only); reuse this pattern for any future per-task evaluation switch.

The `failure_reports` CSV: one row per wrong-answer example; per-choice log-likelihoods + an `ll_margin = ll_pred - ll_gold`. After the 2026-05-03 char-norm change, two views are persisted: `ll_*` is the raw model log-likelihood (sum over continuation tokens), `score_*` is what the argmax saw after `_aggregate_scores` (default char-norm — divides by `len(continuation.lstrip())`). For 1-char letter MCQ they're identical; for ACVA (`صح`/`خطأ`) and word-scored Alghafa they differ. Use `score_margin` to interpret the model's actual decision and `ll_margin` to debug the unnormalized signal. The margin distinguishes "model was confidently wrong" from "model was nearly right" — histograms answer "why is accuracy bad?" better than the raw accuracy number does.

### Heterogeneous benchmarks: per-topic scoring dispatch (the Alghafa pattern)

Most benchmarks are uniform: ACVA is all T/F, Culture-MMLU is all 4-way, Arabic-Exam is all 4-way. **Alghafa is the exception** — 9 sub-configs spanning 2/3/4/5-way MCQ. The 4 binary/sentiment configs (T/F facts + 2-way binary sentiment + two 3-way sentiment) need **word-scored** prompts (mirroring ACVA's letter-prior fix); the 4-way and 5-way MCQ configs use **letter-scored** prompts (the inherited default). The scoring convention has to dispatch per-row, not per-task.

How it's wired (don't reinvent — extend if a future benchmark needs it):

- `_parse_combined` stamps `_source_config` onto every parsed example. Single-config datasets get the sentinel `"_default"`.
- `AlghafaTask.WORD_SCORED_CONFIGS` is a class-level frozenset (single source of truth for which sub-configs get word scoring). To add a new word-scored sub-config, edit that frozenset and nothing else — no hooks to touch.
- `_format_eval_context` / `_build_continuations` are overridden in `AlghafaTask`, each branching on `_is_word_scored(ex)`. Word-scored: drop the choice listing, score answer text directly. Letter-scored: inherit the base implementation.
- `_aggregate_scores(ex, continuations, log_likelihoods)` on the base class does **char-norm by default** (LightEval `LogProbCharNorm` equivalent — `ll / max(len(c.lstrip()), 1)`). 1-char letter continuations are unaffected (divide by 1); ACVA and word-scored Alghafa are normalized correctly. The hook is the override point if a future task wants token-norm or raw sum.
- `evaluate_mcq` buckets `correct/total` by `_source_config` and emits `per_subconfig_accuracy: {<config>: {accuracy, num_samples}}` alongside the aggregate. Visible in `all_metrics.json`; rendered as a separate "Per-sub-config breakdown" table in `comparison_report.txt` (filtered out of the main downstream-task table to keep its column count manageable).

Things that bite when extending:

- **Don't try to dispatch on the choice count alone** (e.g. "if `len(choices) == 2` use word-scoring"). T/F-shape rows in *future* benchmarks may legitimately want letter-scoring; the dispatch needs to be on the sub-config name, not on the data shape. The `WORD_SCORED_CONFIGS` frozenset makes the policy explicit and reviewable.
- **Char-norm is fairness, not preference.** If you switch a benchmark to word-scoring without char-norm, the model picks the shortest answer text on every row regardless of content (e.g. sol2=`"سلبي"` 4 chars wins over sol1=`"هو رأي ايجابي"` 12 chars on identical raw lls). Always keep `_aggregate_scores`'s char-norm in the path.
- **The base hook signature takes `ex`** (not just `continuations` and `log_likelihoods`) precisely so subclasses can dispatch on `_source_config` if needed for per-topic aggregation policy. Don't drop `ex` from the signature when refactoring.

### `OALL/AlGhafa-Native` schema gotcha (the bug that gave us all of this)

Before 2026-05-03, `AlghafaTask._parse_example` had two bugs that silently produced a corrupted eval set:

1. **Labels are 0-indexed** in the dataset, but the parser did `int(label) - 1` (assuming 1-indexed). This dropped every row with `label='0'` (~36 % of all rows; the first option being correct) and shifted gold answers by −1 across the rest.
2. **`sol5` was never enumerated** — only `sol1..sol4`. Two grounded-statement sub-configs (`multiple_choice_grounded_statement_soqal_task`, `multiple_choice_grounded_statement_xglue_mlqa_task`) ship five options and `label='4'`; their fifth option was always missing from `choices` and the gold sometimes pointed at a slot that no longer existed.

Both bugs are fixed; the regression tests in `tests/test_alghafa_parser.py` pin them down. **The lesson is to author test fixtures against the dataset's true convention** (and ideally cross-check against LightEval's reference adapter — `lighteval/tasks/multilingual/adapters.py:alghafa_adapter` would have caught both bugs at PR time). See `feedback_verify_dataset_schema_on_real_rows.md` in memory.

## Common workflows

### Adding a new tokenizer (checklist)

1. `src/arabic_eval/tokenizers/<name>.py` implementing `BaseTokenizer` (all abstract methods + properties), decorated with `@tokenizer_registry.register("<name>")`.
2. Add import in `src/arabic_eval/tokenizers/__init__.py` (must succeed without torch — keep imports minimal).
3. Make sure `encode()` populates `TokenizerOutput.tokens` (string list) — the morphological metrics depend on it. If your tokenizer uses byte-level encoding, also extend `clean_token_string` (or its `_try_decode_bytelevel` helper) to handle it.
4. Pick `embedding_type` from the existing 4 if possible; only add a new one with a matching adapter branch in `LlamaAdapter.adapt_to_tokenizer()` and a matching collator.
5. **If your embedding shrinks the sequence (any kind of pooling/striding/downsampling), you also need (a) a custom `_forward_<type>` that downsamples `attention_mask` to match the embedded length, and (b) an output head that upsamples logits back to the byte/char/token grid where labels live.** `_forward_charformer` + `CharformerOutputHead` is the working reference. Don't expect the standard HF forward to handle this for you — it won't.
6. `configs/tokenizers/<name>.yaml` with sane defaults.
7. Smoke-test the morphological metrics on it (see "Smoke test recipe" below) — make sure `root_bearing_token_pct` is `0.0` or a real number (not `None`). For byte-level tokenizers `0.0` is the correct mechanical answer; `None` only when no tokens were generated at all. If you see `None` despite real input, that's byte-encoding leakage in `clean_token_string`.

The 3-phase pipeline runs identically over any new tokenizer — no per-tokenizer training-config changes needed.

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

```bash
# Single experiment (one tokenizer, multiple eval tasks):
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/native_llama_3phase_with_sft.yaml

# Sweep (multiple tokenizer cells, shared eval task list):
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/<sweep_yaml>.yaml --sweep
```

`run_experiment` and `run_sweep` both train ONCE per (tokenizer, vocab_size) and then evaluate on every task in `sweep.tasks`. The pipeline auto-detects sweep mode based on whether more than one tokenizer cell is declared.

### Disabling phases / morphological metrics for fast iteration

Per-phase YAML overrides:
```yaml
training:
  phases:
    embedding_alignment: { enabled: false }     # skip Phase 1
    sft: { steps: 200 }                          # short SFT for iteration
evaluation:
  morphological_metrics: false                   # skip Farasa/qalsadi for speed
```

## Things to push back on

- **"Just split each token character by character to compute root conservation."** No — the metric is *per-token*; that would conflate the metric with the tokenizer's granularity. The right algorithm is `contains_subsequence(token, root)` over each *whole* token, which is what's implemented.

- **"CharBERT scores 100% on root conservation, so it must be the best Arabic tokenizer."** It hits the architectural ceiling because it doesn't split at all. Use `morpheme_integrity_rate` and downstream task scores to break ties.

- **"AraRooPat scores ~1.0 on root conservation, so it's the best."** Same architectural-ceiling reason — each ROOT token IS the root letters, by construction. Compare on downstream MCQ accuracy (the discriminating signal).

- **"Just widen `BaseTask.evaluate()` to take the new optional eval flag — that's the cleanest way."** No — task families have fundamentally different evaluation shapes; forcing every one to accept (and silently ignore) a new kwarg pollutes signatures. The established pattern is signature-based gating in the pipeline (`inspect.signature(task.evaluate).parameters`). The `failure_reports` and `score_normalization` flags do this today; new per-task eval flags should follow suit.

- **"Add a fallback that returns 0.0 instead of None when Farasa is unavailable."** Don't — `0.0` would be indistinguishable from a real "tokenizer respects no boundaries" result. `None` correctly signals "not measured." (Note: this rule still holds for `morpheme_integrity_rate`. The bearing-token metrics have a *separate* carve-out where `0.0` is the right answer when tokens were generated but cleaned to empty Arabic-letter strings — that's the byte-level mechanical-zero case, see §"Morphological metrics".)

- **"Charformer's GBST is just a tokenizer — go put it in `src/arabic_eval/tokenizers/charformer.py`."** Half right — there is a tokenizer file, but it's a no-op byte encoder. The actual learning is `GBSTEmbedding` in `models/embeddings/charformer_embed.py`. Hyperparameter changes (M, d_s, conv_kernel) flow from the tokenizer YAML's `params` dict through `get_embedding_config()` to the embedding module's constructor; you almost never want to touch the tokenizer file itself.

- **"Charformer should be able to generate text — it's a 'character transformer' after all."** It can in the original paper *because* the original Charformer is encoder-decoder and the decoder is a normal byte-level decoder. In our decoder-only setup, GBST's block-pooling looks ahead within blocks of size up to M, which is incompatible with autoregressive decoding. Generation isn't required by the 3-phase pipeline (eval is log-likelihood MCQ).

- **"Add LoRA / PEFT to speed up the SFT loop."** Today's loop is full-model — `LlamaAdapter.get_trainable_parameters()` plus the freezing helper covers it. Adding LoRA is a new model adapter (wraps the base with LoRA layers) plus a freeze-pattern that targets the LoRA params, not a flag on `PhaseConfig`. Confirm with the user that they want a new adapter (and accept the comparison-cleanliness trade-off — LoRA-tuned variants are no longer apples-to-apples vs full-FT) before starting.

- **"Drop `lm_head` from Phase 1's `trainable_parameters` since it doesn't appear in named_parameters() for Llama-3.2-1B."** No — the freezing helper warns and continues for tied-weight cases (Llama). Dropping `lm_head` would silently fail to train it on a model with **untied** weights (where `lm_head.weight` is a separate tensor). Keep the YAML semantically correct; let the helper handle the tied-weight reality.

- **"Reinit the embedding/lm_head when swapping tokenizers — that's what the doc says happens."** Read [models/embeddings/standard.py](src/arabic_eval/models/embeddings/standard.py): `resize_token_embeddings(new)` is a no-op when `new == old`, and HF's underlying resize keeps the **first N pretrained Llama rows** when `new < old`. Only newly-added rows when expanding (`new > old`) are reinitialized. Phase 1 (embedding alignment) is what actually drifts those rows under the 3-phase pipeline.

- **"Bump char-JABER's `downsample_factor` to 2 to make it faster."** Latent bug: `LlamaAdapter` doesn't shrink the attention mask for `CHAR_JABER` (it uses the standard forward path). The default of 1 hides this. If you want char-JABER downsampling, add a `_forward_char_jaber` that mirrors `_forward_charformer`'s mask-shrinking logic — don't just bump the default.

- **"Bump qalsadi to a newer API to avoid the import inside `_ensure_backends`."** The lazy init is intentional — qalsadi loads SQLite databases on construction; eager-importing it slows down every tokenizer-only workflow. Keep it lazy.

- **"Copy the CharacterBERT class into a new file to add Farasa pre-segmentation."** Don't — `FarasaCharacterBERTTokenizer` shows the right pattern: subclass `CharacterBERTTokenizer`, override `train`/`encode` to apply `segment_with_farasa()` first, then `super()`. Copying the class duplicates ~150 lines (char vocab building, char-id encoding, save/load, special tokens) that have no reason to diverge.

- **"FarasaCharBERT scores ~1.0 on `morpheme_integrity_rate` so it must beat MorphoBPE on Arabic morphology."** Both score ~1.0 — for FarasaCharBERT it's mechanical (each morpheme is exactly one unit), for MorphoBPE it's by-design but still essentially baked in by the Farasa pre-step. Use `root_conservation_rate` and downstream tasks to break the tie, not integrity.

- **"CSA and `morpheme_integrity_rate` measure the same thing — drop one."** No: CSA restricts to clitic boundaries (proclitic-end / enclitic-start positions). Integrity covers all Farasa boundaries including stem-internal ones. They satisfy `integrity == 1.0 ⇒ CSA == 1.0` but can disagree in general. Integrity catches stem-internal splits (BPE chopping a root in half); CSA does not. Both are needed.

- **"Add a `None → 1.0` fallback for Charformer's CSA / integrity since the byte boundaries trivially align."** No: alignment is the protocol for these metrics, and byte tokens don't align to Arabic-letter offsets. Reporting `None` correctly marks the metric as not-measurable for this tokenizer family. SFR is the alignment-free discriminator.

- **"The Alghafa results are surprisingly low — must be the from-scratch tokenizer / random row mapping / [insert model explanation here]."** Before reaching for a model-side explanation on Alghafa, **check the parser**. The 2026-05-03 fix uncovered two parser bugs (label off-by-one + sol5 truncation) that silently dropped 36 % of rows and shifted gold answers by −1. When numbers from a benchmark "look surprisingly low" for a strong baseline, inspect the parser against actual dataset rows from each sub-config first — it's the cheapest investigation step. Cross-check against LightEval's reference adapter when one exists for the dataset family.

- **"culture_arabic_mmlu accuracy is sub-30% — must be the from-scratch tokenizer / a small SFT split / a noisy benchmark."** Before reaching for any of those, check whether **`evaluation.score_normalization`** is set to `"pmi"` or `"char+pmi"`. Default char-norm is dominated by Llama's per-letter unigram prior on translated MMLU. The fix is `LogProbPMINorm` (subtract the unconditioned per-continuation ll), wired up as `score_normalization: "char+pmi"`.

- **"Just write the test fixture to match what `_parse_example` does so the tests pass."** No — that's how the Alghafa bug stayed live for months. Synthetic test fixtures must be authored against the dataset's true convention (and ideally cross-checked against LightEval's adapter), then the parser is implemented to match the fixture.

- **"Switch all benchmarks to word-scoring everywhere — letter-scoring is broken."** No — letter-scoring is fine for genuinely 4/5-way MCQ where each letter is rare in training and the prior matters less. The pathology is specifically on **2-way and 3-way** tasks where the unigram letter prior dominates the decision (the ACVA observation, replicated for Alghafa's binary/sentiment sub-configs). The dispatch should be per-sub-config (`AlghafaTask.WORD_SCORED_CONFIGS` is the single source of truth), not blanket.

- **"Bring back per-benchmark SFT — the 3-phase pipeline doesn't expose the model to the actual benchmark format during training."** That's by design. The previous 10/90 split silently mapped from-scratch tokenizer indices onto Llama's pretrained rows and 10% benchmark SFT couldn't drift them far enough to be useful. The 3-phase pipeline alternative — Phase 1 embedding alignment + Phase 2 translated QA + Phase 3 native QA — is task-agnostic precisely so every condition (native + 8 from-scratch tokenizers) trains on the *same* data and the only experimental variable is the tokenizer. If you want to compare a per-benchmark SFT regime against the 3-phase one, that's a new experiment, not a pipeline rewrite.
