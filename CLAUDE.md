# Arabic Tokenizers Evaluation Platform

## Project Overview

A universal platform for evaluating Arabic tokenizers by measuring LLM downstream performance. All external parameters (dataset, model architecture, training hyperparameters) are held fixed; only the tokenizer changes between experiments. Tokenizers are trained from scratch on the same Arabic dataset, integrated into the same LLM (with embedding layer replacement), fine-tuned, and evaluated on the same downstream tasks.

- **Primary dataset**: `Jr23xd23/ArabicText-Large` (HuggingFace)
- **Primary LLM**: LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`)
- **Downstream tasks**: Text Generation (perplexity), Question Answering (F1/EM on ARCD), and four LightEval benchmarks (ACVA, Alghafa, Culture-Arabic-MMLU, Arabic-Exam — accuracy via log-likelihood scoring)
- **Vocab sizes tested**: 16K, 32K, 50K for subword tokenizers; fixed char vocab for character-level; fixed 260-id byte vocab for Charformer

## Setup

```bash
pip install -e .          # installs all dependencies from pyproject.toml
# OR for tokenizer-only work (no GPU needed):
pip install pydantic pyyaml tokenizers tabulate numpy tqdm
```

Requires Python >= 3.10. GPU required for model training/evaluation. Farasa (morpho_bpe) requires Java runtime.

## Project Structure

```
src/arabic_eval/          # Main package
  registry.py             # Generic Registry class — all extensibility uses this
  config.py               # Pydantic config models + YAML loading/merging
  utils/                  # reproducibility.py, logging.py, io.py
  data/                   # loader.py, preprocessing.py, collation.py
  tokenizers/             # base.py + 5 implementations + intrinsic_metrics.py
  models/                 # base.py, llama_adapter.py, embeddings/{standard,character_cnn,char_jaber_embed}.py
  tasks/                  # base.py, text_generation.py, question_answering.py, lighteval_benchmarks.py
  training/               # trainer.py, callbacks.py
  evaluation/             # metrics.py, evaluator.py, reporter.py
  pipeline/               # experiment.py (end-to-end orchestrator)
configs/                  # YAML configs: base, tokenizers/, models/, tasks/, experiments/
scripts/                  # CLI entry points: train_tokenizer.py, run_experiment.py, evaluate_intrinsic.py, compare_results.py
outputs/                  # Gitignored: tokenizers/, checkpoints/, logs/, results/
```

## Architecture & Key Design Patterns

### Registry Pattern (`src/arabic_eval/registry.py`)

Three module-level singletons: `tokenizer_registry`, `model_registry`, `task_registry`. Every tokenizer/model/task class self-registers via decorator:

```python
from arabic_eval.registry import tokenizer_registry

@tokenizer_registry.register("my_tokenizer")
class MyTokenizer(BaseTokenizer):
    ...
```

Registration happens at import time. The `__init__.py` files in `tokenizers/`, `models/`, `tasks/` auto-import all implementations. Model and task imports are guarded with `try/except ImportError` so the package works without torch installed (for tokenizer-only workflows).

### Embedding Type Dispatch

The critical architectural pattern: each tokenizer declares an `embedding_type` property that tells the model adapter which embedding layer to use.

| Embedding Type | Tokenizers | What Happens in Model |
|---|---|---|
| `"standard"` | BPE, WordPiece, MorphoBPE | `model.resize_token_embeddings(vocab_size)` — standard nn.Embedding |
| `"character_cnn"` | CharacterBERT | Replace `embed_tokens` with `CharacterCNNEmbedding` (char IDs -> multi-width CNN -> highway -> projection). Input is 3D: `[batch, seq_len, max_char_len]`. Output head uses word-level vocabulary. |
| `"char_jaber"` | char-JABER | Replace `embed_tokens` with `CharJaberEmbedding` (simple char embedding). Sequences are 4-6x longer than subword. Small vocab (~300-500 chars). |
| `"charformer"` | Charformer | Replace `embed_tokens` with `GBSTEmbedding` (byte embed → optional pre-conv → enumerate blocks of size 1..M → score → softmax mix → mean-pool downsample by `d_s`). Input is 1D byte ids. The transformer operates on the *downsampled* sequence (length ~L/d_s); attention mask is shrunk to match in `_forward_charformer`. Output head (`CharformerOutputHead`) upsamples back to byte length so byte-level labels align. |

The dispatch happens in `LlamaAdapter.adapt_to_tokenizer()` (`src/arabic_eval/models/llama_adapter.py:61`).

### Collation Dispatch (`src/arabic_eval/data/collation.py`)

`get_collator(embedding_type)` returns the right collator:
- `StandardCollator`: pads 1D `input_ids`, builds `attention_mask` and `labels`
- `CharacterCNNCollator`: pads 3D `char_ids` tensors (`[batch, words, chars]`)
- `CharJaberCollator`: pads 1D char ID sequences (longer max_length=2048)
- `CharformerCollator`: pads 1D byte ID sequences (default max_length=2048; Arabic UTF-8 inflates ~2x over chars). Same shape as CharJaber; the GBST module inside the model does the downsampling, so the collator stays simple.

### Configuration System (`src/arabic_eval/config.py`)

Layered YAML with Pydantic validation:
1. `configs/base.yaml` — shared defaults
2. Experiment YAML overlaid on top (deep merge)
3. CLI overrides on top of that

`ExperimentConfig` is the top-level Pydantic model with nested `DataConfig`, `TokenizerConfig`, `ModelConfig`, `TaskConfig`, `TrainingConfig`, `EvaluationConfig`, `TrackingConfig`, and optional `SweepConfig`.

Key: experiment YAML files can nest top-level fields under `experiment:` key — the loader flattens this automatically.

## The 8 Tokenizers

| # | Name | Registry Key | File | Embedding | Notes |
|---|---|---|---|---|---|
| 1 | BPE | `bpe` | `tokenizers/bpe.py` | standard | HF `tokenizers` BpeTrainer, ByteLevel pre-tokenizer |
| 2 | WordPiece | `wordpiece` | `tokenizers/wordpiece.py` | standard | HF `tokenizers` WordPieceTrainer, Whitespace pre-tokenizer |
| 3 | Morphological BPE | `morpho_bpe` | `tokenizers/morpho_bpe.py` | standard | Farasa segmentation first, then BPE on morphemes. Requires Java. |
| 4 | CharacterBERT | `character_bert` | `tokenizers/character_bert.py` | character_cnn | Word-level split, each word -> fixed-length char ID vector. Builds both char vocab and word vocab (for output head). |
| 5 | char-JABER | `char_jaber` | `tokenizers/char_jaber.py` | char_jaber | Each character is a token. Small fixed vocab. Sequences ~4-6x longer. |
| 6 | Farasa-CharacterBERT | `farasa_character_bert` | `tokenizers/farasa_character_bert.py` | character_cnn | Farasa segmentation first (same as MorphoBPE), then each *morpheme* (instead of each word) -> fixed-length char ID vector via CharCNN. Subclasses `CharacterBERTTokenizer`. Output head indexes a morpheme vocab. Requires Java. Default `max_char_len=25` (morphemes are shorter than words). |
| 7 | Charformer | `charformer` | `tokenizers/charformer.py` | charformer | Byte-level UTF-8 tokenization (256 bytes + 4 specials = 260 ids). `train()` is a no-op — the actual "subword learning" happens inside the GBST module of the model (`models/embeddings/charformer_embed.py`). GBST enumerates candidate blocks of size 1..M, scores them with a learned linear head, softmax-mixes per position, then mean-pool downsamples by `d_s`. Generation is unsupported (GBST is non-causal within the block window). Sequences are ~2x longer than char-JABER on Arabic (each Arabic char = 2 bytes). |
| 8 | AraRooPat | `araroopat` | `tokenizers/araroopat.py` (+ `araroopat_backend.py`) | standard | Arabic Roots & Patterns. Each content word → `[ROOT_x] [PAT_y]` where root is the consonant skeleton and pattern is CAMeL Tools' positional template (e.g. `"1ُ2ُ3"`). Clitics emitted as separate `[CLITICP_*]` (proclitic) and `[CLITICE_*]` (enclitic) tokens — distinct prefix ranges remove the prc-vs-enc ambiguity at decode. Reconstruction is a three-tier resolver: lookup table built from corpus → CAMeL `Generator` for unseen pairs → naive slot substitution. Requires `camel-tools` and the `morphology-db-msa-r13` database (one-time `camel_data -i light` download). Generation is supported (unlike CharBERT/Charformer). |

All implement `BaseTokenizer` (`tokenizers/base.py`): `train()`, `encode()`, `decode()`, `save()`, `load()`, `vocab_size`, `embedding_type`, `special_tokens`, `get_embedding_config()`.

**Pre-segmentation and embedding family are orthogonal axes.** The Farasa-CharacterBERT case shows that you can pair MorphoBPE's front-end (Farasa morphological segmentation) with CharacterBERT's back-end (CharCNN over characters of each unit). When the only thing changing is the pre-step, **subclass the existing tokenizer and override `train`/`encode`** — don't copy the encoding logic. The CharCNN embedding dispatch in `LlamaAdapter.adapt_to_tokenizer()` and the `CharacterCNNCollator` are reused unchanged because both classes share `embedding_type=character_cnn`.

Special tokens for all: `<pad>` (0), `<s>` (1), `</s>` (2), `<unk>` (3).

## Experiment Pipeline Flow (`src/arabic_eval/pipeline/experiment.py`)

`run_single_experiment(config)` executes these steps:
1. **Set seed** and create output directory
2. **Load dataset** via HF `datasets`, apply Arabic preprocessing (normalization, optional diacritics removal), split train/eval
3. **Train tokenizer** from scratch on training texts (or load from `load_path` if set)
4. **Intrinsic evaluation** — compute fertility, compression ratio, UNK rate, vocab coverage
5. **Load LLM** and call `adapt_to_tokenizer()` — resizes/replaces embedding layers
6. **Fine-tune** — training loop with gradient accumulation, mixed precision (bf16), cosine LR schedule, early stopping
7. **Downstream evaluation** — perplexity for text generation, F1/EM for QA
8. **Save results** as JSON to output directory

`run_sweep(config)` iterates over the Cartesian product of (tokenizer types x vocab sizes x tasks) and generates a comparison report.

For **LightEval benchmark tasks** the pipeline flow adapts automatically:
- Steps 1–4 are unchanged (main Arabic dataset used for tokenizer training and intrinsic metrics).
- Step 6 fine-tunes on the **10 % benchmark split** (the task's `get_dataloader()` returns this).
- Step 7 evaluates on the **90 % benchmark split** via LightEval log-likelihood scoring.

## CLI Commands

```bash
# Train a single tokenizer
python scripts/train_tokenizer.py --type bpe --vocab-size 32000

# Run a single experiment
python scripts/run_experiment.py --config configs/experiments/bpe_32k_generation.yaml

# Run full sweep (all tokenizers x vocab sizes x tasks)
python scripts/run_experiment.py --config configs/experiments/full_sweep.yaml --sweep

# Run benchmark sweep (all tokenizers x ACVA/Alghafa/Culture-Arabic-MMLU/Arabic-Exam)
python scripts/run_experiment.py --config configs/experiments/benchmark_sweep.yaml --sweep

# Intrinsic-only evaluation of a saved tokenizer
python scripts/evaluate_intrinsic.py --tokenizer-path outputs/tokenizers/bpe_32k --type bpe

# Compare results across experiments
python scripts/compare_results.py outputs/experiments/*/
```

All scripts add `src/` to `sys.path`, so no install is needed for development. They auto-detect `configs/base.yaml` as the base config.

## How to Extend

### Adding a new tokenizer

1. Create `src/arabic_eval/tokenizers/my_tok.py`
2. Implement `BaseTokenizer` (all abstract methods + properties)
3. Decorate with `@tokenizer_registry.register("my_tok")`
4. Add import in `src/arabic_eval/tokenizers/__init__.py`
5. If it needs a custom embedding: add under `models/embeddings/`, add a new `EmbeddingType` constant, update `LlamaAdapter.adapt_to_tokenizer()` with a new branch
6. Create `configs/tokenizers/my_tok.yaml`

### Adding a new LLM

1. Create `src/arabic_eval/models/my_model_adapter.py`
2. Implement `BaseModelAdapter` (load, adapt_to_tokenizer, forward, generate, checkpointing)
3. Decorate with `@model_registry.register("my_model")`
4. Add import in `src/arabic_eval/models/__init__.py`
5. Must handle all `EmbeddingType` values in `adapt_to_tokenizer()`
6. Create `configs/models/my_model.yaml`

### Adding a new downstream task

1. Create `src/arabic_eval/tasks/my_task.py`
2. Implement `BaseTask` (get_dataloader, evaluate, name, metric_names)
3. Decorate with `@task_registry.register("my_task")`
4. Add import in `src/arabic_eval/tasks/__init__.py`
5. Create `configs/tasks/my_task.yaml`

### Adding a new LightEval benchmark task

1. In `src/arabic_eval/tasks/lighteval_benchmarks.py`, subclass `LightEvalBenchmarkTask`
2. Implement `_default_dataset_name()`, `_parse_example()`, and the `name` property
3. Decorate with `@task_registry.register("my_benchmark")`
4. Create `configs/tasks/my_benchmark.yaml` with `dataset_name`, `train_split_ratio: 0.10`, etc.

The 10/90 data split, SFT dataloader, and LightEval log-likelihood evaluation are all inherited from the base class — only the dataset-specific parsing needs to be provided.

## Key Technical Details

### Model Integration Approach
Tokenizers are trained from scratch. Then: load LLaMA with pretrained weights intact, replace/resize the embedding layer (`model.model.embed_tokens`) and output head (`model.lm_head`) to match the new tokenizer's vocab size, reinitialize those new weights, then fine-tune the full model.

### CharacterBERT Limitations
- Auto-regressive `generate()` is **not supported** — `LlamaAdapter.generate()` raises `NotImplementedError` for `CHARACTER_CNN`. QA evaluation falls back to empty predictions.
- The forward pass manually loops through transformer layers (`_forward_character_cnn`) because the standard HF forward expects `input_ids`, not `char_ids`.

### Charformer (GBST) Specifics
- Byte-level tokenization with a fixed 260-id vocab (256 bytes + 4 special tokens). `train()` is a no-op.
- All "subword learning" happens inside `GBSTEmbedding`: byte embed → optional pre-conv (k=5) → enumerate blocks of size 1..M (M=4 default) via mean-pool with stride=b → linear scoring (D→1, no bias) → repeat-interleave back to L → softmax across block sizes per position → weighted sum → final mean-pool with stride `d_s` (2 default).
- Optional position-wise score calibration (`block_attention=true`) implements `P̂ = softmax(P P^T) P` from §2.1.4 of the paper. The paper finds this helps in English and is neutral multilingually.
- The transformer operates on the *downsampled* sequence (length ~L/d_s). `_forward_charformer` shrinks the byte-level attention mask by OR-reducing windows of size `d_s`, then passes `inputs_embeds` (already downsampled by GBST) to the model. The replaced `lm_head` (`CharformerOutputHead`) upsamples back to byte length via `ConvTranspose1d` so byte-level labels align with logits.
- Auto-regressive `generate()` is **not supported** — GBST pools blocks `X[i:i+b]`, so position `i` sees up to position `i+M-1`. The original Charformer is encoder-decoder, sidestepping causality; in our decoder-only setup, only teacher-forced losses (LM perplexity, LightEval log-likelihood MCQ) are well-defined.
- Mechanical extremes on morphological metrics: each token is one byte, which cannot hold a 3-letter Arabic root (each Arabic letter is 2 bytes). Expect `root_conservation_rate ≈ 0`, `pattern_conservation_rate ≈ 0`, and `morpheme_integrity_rate ≈ 1` — same family of mechanical extremes as char-JABER. The token-level inventory metrics (`root_bearing_token_pct`, `pattern_bearing_token_pct`) are explicitly reported as `0.0` (not `None`) when the cleaned-token list is empty but raw tokens were generated; this distinguishes the byte-level mechanical zero from "not measured."

### AraRooPat (Arabic Roots & Patterns)
- The tokenizer file (`tokenizers/araroopat.py`) holds the encode/decode/state machinery; CAMeL Tools integration lives in `tokenizers/araroopat_backend.py` (analyzer + MLE disambiguator + generator + LRU caches + configurable timeout).
- Vocab layout (deterministic ID order): specials → `[LIT_BEGIN]` / `[LIT_END]` → `[CLITICP_*]` (proclitics) → `[CLITICE_*]` (enclitics) → `[CHAR_*]` → `[DIGIT_*]` → `[PUNCT_*]` → `[ROOT_*]` → `[PAT_*]`.
- **Pattern normalization is essential.** CAMeL's `pattern` field bakes clitic surface chars into the template (e.g. `"ال1ِ2ا3ِ"` for definite singular noun). We strip those clitic chars *out* of the pattern at vocab time so each `[PAT_*]` token represents a bare-stem template only — clitics live in their own tokens. See `normalize_pattern()` in the backend.
- **Reconstruction stores the *inflected* stem, not CAMeL's `stem` field.** CAMeL's `stem` excludes inflectional prefixes (e.g. the ي of present-tense `يدرس`), which would lose the inflection at decode time. We instead use `diac` minus clitic surfaces — keeps inflection, drops clitics. See `_strip_clitic_surfaces()` and `_build_reconstruction()`.
- **Three-tier reconstruction at decode**: (1) `(root_id, pat_id)` lookup table — covers ~99% of LLM emissions since the LLM was trained on this distribution; (2) CAMeL `Generator` for unseen pairs — handles weak roots, hamza placement, gemination via the database; (3) naive slot substitution as last resort (logged so you can audit how often tier 3 fires).
- **Distinct prefixes for proclitics vs enclitics** (`[CLITICP_*]` vs `[CLITICE_*]`) eliminate the prc-vs-enc ambiguity at decode time. Linguistically correct too — same surface form can be different morphemes (e.g. ك as preposition `ka_prep` vs ك as 2ms object pronoun).
- **Loanwords / proper nouns** route to the `[LIT_BEGIN] [CHAR_*]... [LIT_END]` fallback path. CAMeL marks these with `root='NTWS'` ("Non-Triliteral Word Source") which we detect and reject as analyses. ~15–30% of MSA goes through this path on a typical corpus; on a 200-sentence smoke test it was ~33% (most rare nouns and proper names lack CAMeL-DB entries — coverage rises with corpus scale).
- **Defective-root handling**: CAMeL uses `#` as a placeholder for missing/weak letters in some root entries (e.g. `'ش#ق'`). We strip `#` along with `_` and `.` from roots; if the resulting root is shorter than 3 letters, we route the word to LIT.
- **Generation is supported** (unlike CharBERT/Charformer): the LLM emits `[ROOT_x] [PAT_y]` and decode reconstructs the inflected stem in O(1) via the lookup table. Useful for QA evaluation.
- **Mechanical metrics ceiling**: `root_conservation_rate` and `pattern_conservation_rate` are ~1.0 by construction (each ROOT token IS the root letters; each PAT token's metric-string is the cleaned inflected stem). Like CharBERT's mechanical 1.0 on root conservation, this is the architectural ceiling — flag it in comparison tables. The metric headroom comes from words that CAMeL fails to analyze (LIT path with single-char tokens) and from cases where qalsadi (the metric's extractor) disagrees with CAMeL on the root.
- **Vocab budget tiers** (in `configs/tokenizers/araroopat.yaml`): Compact 5K+200 (~5.4K, ~81% coverage), **Balanced 10K+500** (~10.7K, ~94% coverage; default), Max 15K+1000 (~16.2K, ~98% coverage).
- **Provenance trail**: `vocab_metadata.json` records per-root and per-pattern `{id, freq, source, examples}` plus the full proclitic/enclitic frequency maps. Use it to answer "where did this token come from?" without re-running.
- **CAMeL Tools dep conflicts — solved via subprocess bridge.** `camel-tools>=1.5` pins `numpy<2` and `transformers<4.54`, which conflicts with `lighteval>=0.11`. Rather than force-choosing one, araroopat runs CAMeL in an isolated `.venv-camel` and the main `.venv` talks to it over stdin/stdout NDJSON.
  - **Setup (one-time):** `python -m venv .venv-camel && .venv-camel/bin/pip install -e ".[araroopat-camel]" && .venv-camel/bin/camel_data -i light`
  - **Files:** server runs in `.venv-camel` (`src/arabic_eval/tokenizers/araroopat_camel_server.py`); client runs in main `.venv` (`src/arabic_eval/tokenizers/araroopat_bridge.py`); the `MorphAnalyzer` in `araroopat_backend.py` wraps the bridge and exposes `analyze` / `analyze_many` / `generate`.
  - **Wire format:** one NDJSON line per request/response, integer `id` for correlation. Three ops: `analyze` (batch of words → list-of-lists of trimmed analysis dicts), `generate` (root + bare pattern → stem string or null), `shutdown`.
  - **Fail-loud:** missing `.venv-camel`, server crash (EOF), non-JSON, or per-request error all raise `CamelBridgeError`. There is no silent degradation — using araroopat without camel makes no sense (every word would route to `[LIT_*]`). Override the interpreter via `$ARAROOPAT_CAMEL_PYTHON` if `.venv-camel` lives elsewhere.
  - **Main env stays clean:** `.venv` no longer installs camel-tools at all. The `[morphological]` extras now contain only `qalsadi` + `pyarabic` (used by `morphological_utils.py` for the metrics, no version conflict). camel-tools moved to its own `[araroopat-camel]` extras.

### char-JABER Sequence Length
Character-level tokenization produces sequences ~4-6x longer. Default `max_length` for `CharJaberCollator` is 2048 (vs 512 for subword). This impacts memory and speed. The `CharJaberEmbedding` has an optional `downsample_factor` for strided convolution to reduce length, but it is set to 1 (disabled) by default.

### Arabic Preprocessing (`src/arabic_eval/data/preprocessing.py`)
- Unicode NFKC normalization
- Alef variant normalization (hamza forms -> bare alef)
- Optional diacritics (tashkeel) removal — controlled by `remove_diacritics` in config
- Tatweel (kashida) removal
- Whitespace collapsing

### Training Loop (`src/arabic_eval/training/trainer.py`)
- AdamW optimizer with cosine LR schedule + linear warmup
- Gradient accumulation (default: 4 steps)
- Mixed precision via `torch.cuda.amp` (bf16 by default)
- Gradient clipping at `max_grad_norm=1.0`
- Early stopping on `eval_loss` with patience=3
- Checkpoint management with `save_total_limit=2`

### Intrinsic Metrics (`src/arabic_eval/tokenizers/intrinsic_metrics.py`)

**Size / coverage metrics** (always on):
- **Fertility**: avg tokens per whitespace word
- **Compression ratio**: avg characters per token
- **UNK rate**: fraction of tokens that are `<unk>`
- **Vocab coverage**: fraction of unique words with no UNK tokens
- **Avg token count**: avg tokens per text

**Arabic morphological metrics** (controlled by `evaluation.morphological_metrics`, default `true`; helpers in `tokenizers/morphological_utils.py`):
- **`root_conservation_rate`** — % of sampled words whose 3/4-letter root appears as a subsequence inside a *single* token. Penalizes tokenizers that cut through a root.
- **`pattern_conservation_rate`** — % of words whose stem-span pattern (root letters + their immediate vowel context, clitics trimmed via `stem_pattern_span`) is recoverable from a single token. Distinguishes BPE/WordPiece splits that destroy the wazn from ones that preserve it.
- **`morpheme_integrity_rate`** — % of Farasa internal morpheme boundaries (e.g. `و|ال|كتاب`) that align with token boundaries. Averaged only over multi-morpheme words. Requires Java (Farasa subprocess); set to `None` if Farasa fails to load.
- **`root_bearing_token_pct`** — % of all tokens (across the sample) that contain at least one full root from the sample's root set. Token-level inventory metric.
- **`pattern_bearing_token_pct`** — % of all tokens whose stem span matches a known pattern from the sample's pattern set.

**Architectural reading of the metrics** (use this as a sanity check, not a bug):
- Each whitespace-bounded word is the unit for CharBERT and char-JABER, so two of the metrics hit a mechanical extreme on those tokenizers and should be reported but flagged: CharBERT trivially scores ~1.0 on `root_conservation_rate` (whole word never split → root never split) and ~0.0 on `morpheme_integrity_rate` (no internal token boundaries → none align with morpheme boundaries). char-JABER is the mirror: ~0.0 on `root_conservation_rate` (single-char tokens can't hold a 3-letter root) and ~1.0 on `morpheme_integrity_rate` (every char boundary is a token boundary, so morpheme boundaries are mechanically respected). MorphoBPE hits ~1.0 on `morpheme_integrity_rate` *non-trivially* — by design, since it pre-segments with Farasa before training BPE.
- **Farasa-CharacterBERT** (`farasa_character_bert`): each morpheme is exactly one input unit, so `morpheme_integrity_rate` ≈ 1.0 *mechanically* (Farasa boundaries are token boundaries by construction — same logic as char-JABER but at morpheme granularity). `root_conservation_rate` is high but not at the ceiling (~0.75–0.85 typical; observed 0.775 on a 200-sentence smoke test) because the root sits inside the unsplit stem morpheme, but Farasa occasionally over-segments the stem itself. Among Farasa-aware tokenizers (MorphoBPE vs FarasaCharBERT), `morpheme_integrity_rate` does *not* discriminate — use `root_conservation_rate` and downstream task scores to break the tie.
- **AraRooPat** (`araroopat`): `root_conservation_rate` and `pattern_conservation_rate` both → ~1.0 *by construction* — each ROOT token's metric-string IS the root letters and each PAT token's metric-string is the cleaned inflected stem (which contains the root letters in their pattern-positioned context). `morpheme_integrity_rate` ≈ 1.0 too (clitics are separate tokens). The metric headroom (smoke test on 200 sentences gave 0.54 root, 0.66 pattern — well below the ceiling) is *not* a bug in the tokenizer — it comes from (a) ~30% of words on the LIT fallback path because CAMeL's MSA database doesn't cover them (rare nouns, proper names — coverage rises with corpus scale to ~85–95%), and (b) qalsadi (the metric's extractor) disagreeing with CAMeL on the root for some words. Among morphology-aware tokenizers (MorphoBPE / FarasaCharBERT / AraRooPat), break ties with downstream MCQ scores rather than the conservation metrics.

**Root extraction backends** (in `RootExtractor`, falls through in order):
1. `qalsadi.analex.Analex.check_word(word)` — proper morphological roots. **Use this API, not `Lemmatizer.lemmatize()` which returns the lemma (dictionary form), not the root.**
2. `tashaphyne.stemming.ArabicLightStemmer.get_root()` — light stemmer; ~80% accurate, used as backup.
3. Consonant-skeleton heuristic — strips diacritics and matres lectionis (ا, و, ي). Rejected if length is outside 3–4 consonants.

**Token-string normalization** (`clean_token_string`):
- Reverses the GPT-2/ByteLevel BPE byte→char mapping (necessary for HF `ByteLevel` BPE, which encodes Arabic UTF-8 bytes as Latin-1 surrogates like `Ø§ÙĦÙĥØªØ§Ø¨` for `الكتاب`). Without this, every BPE token cleans to an empty string and `root_bearing_token_pct` becomes `None`.
- Strips `##` (WordPiece), `▁` (SentencePiece), `Ġ` (ByteLevel) prefix markers.
- Removes diacritics; keeps only Arabic letters + matres lectionis.

**Sampling** is controlled by `evaluation.morph_sample_size` (default 500). The sample is deterministic (fixed seed), distinct words only, length ≥ 3 chars.

### Downstream Metrics
- **Text generation** (`tasks/text_generation.py`): perplexity via sliding-window (stride=256) on held-out text
- **Question answering** (`tasks/question_answering.py`): F1 and Exact Match on ARCD dataset. QA is framed as generation (Arabic prompt: `السياق: ... \nالسؤال: ... \nالإجابة:`)
- **LightEval benchmarks** (`tasks/lighteval_benchmarks.py`): accuracy via log-likelihood multiple-choice scoring on ACVA, Alghafa, Culture-Arabic-MMLU, and Arabic-Exam. See [LightEval Benchmarks](#lighteval-benchmarks-acva-alghafa-culture-arabic-mmlu-arabic-exam) below.

## LightEval Benchmarks (ACVA, Alghafa, Culture-Arabic-MMLU, Arabic-Exam)

### Overview

These four multiple-choice benchmarks are used to evaluate the impact of tokenizer training choices on downstream Arabic understanding. All evaluation is conducted using the LightEval framework methodology.

| Registry Key | Class | Default Dataset | Metric |
|---|---|---|---|
| `acva` | `ACVATask` | `OALL/ACVA` | accuracy |
| `alghafa` | `AlghafaTask` | `OALL/AlGhafa-Native` | accuracy |
| `culture_arabic_mmlu` | `CultureArabicMMLUTask` | `acmc/arabic_culture_mmlu` | accuracy |
| `arabic_exam` | `ArabicExamTask` | `arabic_exam` | accuracy |

### Data Split Strategy

All examples across all predefined splits of each benchmark are pooled, then split with a fixed RNG seed:

- **10 %** → supervised fine-tuning (SFT) on formatted MCQ prompts
- **90 %** → reserved for LightEval evaluation (never seen during training)

The same seed is used across all tokenizer variants in a sweep so every experiment evaluates on the identical 90 % partition.

### Evaluation Methodology (LightEval)

For each multiple-choice question the `LightEvalModelWrapper` computes:

```
log P(" A" | context)  …  log P(" D" | context)
```

using the fine-tuned model's forward pass (`_compute_loglikelihood`), then predicts `argmax`. This matches LightEval's standard log-likelihood MCQ protocol exactly. The wrapper implements the same `loglikelihood(requests)` interface as LightEval's `LightevalModel`, so it can be substituted into a full LightEval pipeline if needed.

**CharacterBERT limitation**: `character_cnn` embedding does not support token-level log-likelihoods; accuracy will be reported as 0.0 for that embedding type.

### Fine-tuning Data Format

Each MCQ example is formatted as a plain-text causal-LM sequence:

```
السؤال: {question}

A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
الإجابة: {correct_letter}
```

This is tokenised and passed through the existing `Trainer` using the standard collator for the active embedding type.

### Key Classes (`src/arabic_eval/tasks/lighteval_benchmarks.py`)

| Class | Role |
|---|---|
| `LightEvalBenchmarkTask` | Abstract base — 10/90 split, SFT dataloader, `evaluate()` |
| `LightEvalModelWrapper` | Wraps `BaseModelAdapter` for LightEval's `loglikelihood` interface |
| `_compute_loglikelihood` | Core per-token log-likelihood sum (LightEval methodology) |
| `_format_mcq_context` | Formats question + choices as LightEval context string |
| `_format_mcq_full` | Formats complete MCQ + answer for SFT |

### Dataset Field Schemas

`_parse_mcq_generic()` in the base class handles three common formats automatically:

| Format | Fields |
|---|---|
| Separate columns | `question`, `A`, `B`, `C`, `D`, `answer` (letter or int) |
| Choices list | `question`, `choices` (list), `answer` (int) |
| Options list | `question`, `options` (list), `label` (int) |

Subclasses can override `_parse_example()` for non-standard schemas.

### Config Overrides

Dataset paths are configurable per-task via the `params` dict:

```yaml
task:
  type: "acva"          # or alghafa | culture_arabic_mmlu | arabic_exam
  params:
    dataset_name: "OALL/ACVA"
    dataset_config: null   # set to a specific subtask/config if needed
    train_split_ratio: 0.10
    max_length: 512
    seed: 42
```

> **Note:** `dataset_name` for `culture_arabic_mmlu` (`acmc/arabic_culture_mmlu`) and `arabic_exam` (`arabic_exam`) are best-effort defaults. Update them to the confirmed HuggingFace Hub paths before running.

## Config Reference

### Experiment YAML structure

```yaml
experiment:
  name: "my_experiment"
  output_dir: "outputs/experiments/my_experiment"
  seed: 42

data:
  dataset_name: "Jr23xd23/ArabicText-Large"
  max_train_samples: null    # null = use all
  max_eval_samples: 10000
  preprocessing:
    normalize_unicode: true
    remove_diacritics: false
    min_text_length: 10

tokenizer:
  type: "bpe"               # Registry key: bpe|wordpiece|morpho_bpe|character_bert|char_jaber
  vocab_size: 32000          # null for character-level tokenizers
  params: {}                 # Passed as **kwargs to constructor and train()
  save_path: "outputs/tokenizers/bpe_32k"
  load_path: null            # Set to skip training and load existing

model:
  type: "llama"              # Registry key
  name_or_path: "meta-llama/Llama-3.2-1B"
  dtype: "bfloat16"
  device: "auto"

task:
  # Registry key: text_generation | question_answering
  #               acva | alghafa | culture_arabic_mmlu | arabic_exam
  type: "text_generation"
  params:
    max_length: 512

training:
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  bf16: true
  early_stopping_patience: 3

evaluation:
  intrinsic_metrics: true
  morphological_metrics: true   # Arabic root/pattern/morpheme metrics
  morph_sample_size: 500        # # of distinct words to sample for them
  downstream_metrics: true
  num_eval_samples: 5000
```

### Sweep YAML structure (for full_sweep.yaml)

```yaml
sweep:
  tokenizers:
    - type: "bpe"
      vocab_sizes: [16000, 32000, 50000]
    - type: "character_bert"
      vocab_sizes: [null]        # N/A for char-level
  tasks:
    - type: "text_generation"
    - type: "question_answering"
```

### Benchmark sweep YAML structure (for benchmark_sweep.yaml)

```yaml
sweep:
  tokenizers:
    - type: "bpe"
      vocab_sizes: [16000, 32000, 50000]
    # ... other tokenizers ...
  tasks:
    - type: "acva"
      params:
        dataset_name: "OALL/ACVA"
        train_split_ratio: 0.10   # 10% SFT, 90% LightEval eval
        max_length: 512
        seed: 42
    - type: "alghafa"
      params:
        dataset_name: "OALL/AlGhafa-Native"
        train_split_ratio: 0.10
    - type: "culture_arabic_mmlu"
      params:
        dataset_name: "acmc/arabic_culture_mmlu"
        train_split_ratio: 0.10
    - type: "arabic_exam"
      params:
        dataset_name: "arabic_exam"
        train_split_ratio: 0.10
```

## Output Structure

Each experiment produces:
```
outputs/experiments/<name>/
  config.json               # Full resolved config
  intrinsic_metrics.json    # Fertility, compression, UNK rate, coverage
  all_metrics.json          # Combined intrinsic + downstream results
  training/
    train_results.json      # Loss, steps, time
    best/                   # Best checkpoint (early stopping)
    final/                  # Final checkpoint
    checkpoint-*/           # Periodic checkpoints
  experiment.log            # Full log file
```

Sweep mode additionally generates: `comparison_report.txt` and `comparison_report.json` in the sweep output directory.

## Dependencies

Core: `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate`, `farasapy`, `pydantic`, `pyyaml`, `numpy`, `tqdm`, `wandb`, `tabulate`, `matplotlib`, `lighteval>=0.6.0`

Tokenizer-only workflows (no GPU): `pydantic`, `pyyaml`, `tokenizers`, `tabulate`, `numpy`, `tqdm`

Optional `[morphological]` extras for full Arabic morphological metrics: `qalsadi`, `pyarabic` (Tashaphyne is pulled in transitively by qalsadi). Install via `pip install -e ".[morphological]"`. Without these, `RootExtractor` falls back to a consonant-skeleton heuristic and `morpheme_integrity_rate` still works via Farasa.

## Known Considerations

- LLaMA 3.2-1B requires HuggingFace access token (gated model). Set `HF_TOKEN` env var or `huggingface-cli login`.
- MorphoBPE (`morpho_bpe`) requires Java runtime for Farasa. The segmenter runs in interactive mode for efficiency.
- Character-level tokenizers (char-JABER) produce very long sequences; reduce `max_length` or `batch_size` if hitting OOM.
- The `models/__init__.py` and `tasks/__init__.py` use try/except on imports, so model and task registries will be empty if torch/transformers are not installed. The pipeline script (`pipeline/experiment.py`) force-imports them, so missing deps will surface at experiment runtime.
- `torch.cuda.amp.autocast` in the trainer uses `device_type="cuda"` — will need adjustment for non-CUDA accelerators (e.g., MPS).
- LightEval benchmark tasks (`acva`, `alghafa`, `culture_arabic_mmlu`, `arabic_exam`) fine-tune on the **10 % split only**; the 90 % eval split is never used during training. The `get_dataloader(split="test")` call from the Trainer also returns the 10 % split to avoid contamination.
- CharacterBERT (`character_cnn`) returns `accuracy=0.0` on LightEval benchmarks because log-likelihood scoring requires token-level logits over a standard vocabulary — not supported by the word-level CharCNN architecture. **`farasa_character_bert` inherits the same limitation** (shared `embedding_type=character_cnn`): no `generate()` for QA, accuracy=0.0 on log-likelihood benchmarks, output head is a morpheme vocab instead of a word vocab.
- The `dataset_name` defaults for `culture_arabic_mmlu` (`acmc/arabic_culture_mmlu`) and `arabic_exam` (`arabic_exam`) are best-effort; confirm the exact HuggingFace Hub paths before running and update the YAML configs accordingly.
