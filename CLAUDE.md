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
  tasks/                  # base.py, text_generation.py, question_answering.py
                          #   lighteval/    — abstract base + 4 dataset files (acva, alghafa,
                          #     culture_arabic_mmlu, arabic_exam) + utils.py (opt-in helpers)
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

## The 8 Tokenizers (+ 1 baseline-only wrapper)

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
| — | NativeLlama | `native_llama` | `tokenizers/native_llama.py` | standard | **Baseline-only wrapper, not part of the 8-tokenizer comparison.** Wraps `meta-llama/Llama-3.2-1B`'s pretrained tokenizer; `train()` is a no-op. `vocab_size = len(hf_tokenizer) = 128256` matches the model's embedding matrix → `resize_token_embeddings` is a no-op and pretrained embeddings stay byte-identical. Special tokens follow Llama: `bos=128000`, `eos=128001`, `pad=128001` (= eos, HF standard for causal LMs without dedicated pad — collator masks via attention_mask, not pad_id, so the collision is harmless), `unk=128002` (`<\|reserved_special_token_0\|>`, never emitted → UNK rate stays 0). Used by the `native_llama_{no,with}_sft_benchmark_sweep` baselines to anchor the from-scratch sweeps against the pretrained-tokenizer ceiling. |

All implement `BaseTokenizer` (`tokenizers/base.py`): `train()`, `encode()`, `decode()`, `save()`, `load()`, `vocab_size`, `embedding_type`, `special_tokens`, `get_embedding_config()`.

**Pre-segmentation and embedding family are orthogonal axes.** The Farasa-CharacterBERT case shows that you can pair MorphoBPE's front-end (Farasa morphological segmentation) with CharacterBERT's back-end (CharCNN over characters of each unit). When the only thing changing is the pre-step, **subclass the existing tokenizer and override `train`/`encode`** — don't copy the encoding logic. The CharCNN embedding dispatch in `LlamaAdapter.adapt_to_tokenizer()` and the `CharacterCNNCollator` are reused unchanged because both classes share `embedding_type=character_cnn`.

Special tokens for the 8 from-scratch tokenizers: `<pad>` (0), `<s>` (1), `</s>` (2), `<unk>` (3). NativeLlama deviates from this convention because IDs 0–3 are regular ASCII (`!`, `"`, `#`, `$`) in Llama's vocab — using (0,1,2,3) would have collided with pretrained character embeddings. The collator's label-masking is independent of pad_id value (it builds attention_mask from "real token positions"), so wrappers may set `pad = eos` without breaking the loss path.

## Experiment Pipeline Flow (`src/arabic_eval/pipeline/experiment.py`)

`run_single_experiment(config)` executes these steps:
1. **Set seed** and create output directory
2. **Load dataset** via HF `datasets`, apply Arabic preprocessing (normalization, optional diacritics removal), split train/eval
3. **Train tokenizer** from scratch on training texts (or load from `load_path` if set)
4. **Intrinsic evaluation** — compute fertility, compression ratio, UNK rate, vocab coverage
5. **Load LLM** and call `adapt_to_tokenizer()` — resizes/replaces embedding layers
6. **Fine-tune** — training loop with gradient accumulation, mixed precision (bf16), cosine LR schedule, early stopping
7. **Downstream evaluation** — perplexity for text generation, F1/EM for QA, accuracy for LightEval MCQ. Wall-clock wrapped via `time.perf_counter()` after a tokenizer warmup encode (so lazy backends — AraRooPat's CAMeL bridge, Farasa Java subprocess — don't get billed to the timed region). Result stored as `inference_time_sec` in the downstream block.
8. **Composite metric** — for LightEval MCQ tasks only, `compute_mei()` produces the Morphological Efficiency Index from the four inputs above. Stored at top-level `results["mei"]`. See *MEI* section below.
9. **Save results** as JSON to output directory

`run_sweep(config)` iterates over the Cartesian product of (tokenizer types x vocab sizes x tasks) and generates a comparison report.

For **LightEval benchmark tasks** the pipeline flow adapts automatically:
- Steps 1–4 are unchanged (main Arabic dataset used for tokenizer training and intrinsic metrics).
- Step 6 fine-tunes on the **10 % benchmark split** (the task's `get_dataloader()` returns this).
- Step 7 evaluates on the **90 % benchmark split** via LightEval log-likelihood scoring.
- Step 8 computes MEI; `compute_mei` short-circuits with `status="task_not_mcq"` for non-LightEval tasks.

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

1. Create `src/arabic_eval/tasks/lighteval/<my_benchmark>.py` and subclass `LightEvalBenchmarkTask`
2. Implement all 8 abstract hooks: `_default_dataset_name`, `name`, `_parse_example`, `load_examples`, `_format_eval_context`, `_build_continuations`, `_format_sft_text`, `_aggregate_scores`. The base class is intentionally opinion-free — every dataset declares its own prompt shape, continuations, SFT format, and aggregation policy. Most letter-MCQ datasets reuse `utils.format_mcq_context` / `utils.format_mcq_full` / `utils.char_norm_aggregator`; HF-loaded datasets call `utils.load_huggingface_mcq` from `load_examples`.
3. Decorate with `@task_registry.register("my_benchmark")`
4. Add the new module to the auto-import list in `src/arabic_eval/tasks/lighteval/__init__.py`
5. Create `configs/tasks/my_benchmark.yaml` with `dataset_name`, `train_split_ratio: 0.10`, etc.

The 10/90 stratified split, SFT dataloader, and LightEval log-likelihood evaluation are inherited from the base class — those operate on the unified example list returned by `load_examples` and are dataset-agnostic. A future benchmark loaded from a non-HF source (local files, S3, …) just implements its own `load_examples` without touching the base.

## Key Technical Details

### Model Integration Approach
Tokenizers are trained from scratch. Then: load LLaMA with pretrained weights intact, replace/resize the embedding layer (`model.model.embed_tokens`) and output head (`model.lm_head`) to match the new tokenizer's vocab size, then fine-tune the full model.

**Reinitialization behavior** ([models/embeddings/standard.py](src/arabic_eval/models/embeddings/standard.py)) — important to read correctly when interpreting cross-tokenizer comparisons:
- `new == old` (e.g. `native_llama` at 128256): early-return, **nothing changes**, pretrained embeddings stay byte-identical.
- `new < old` (every from-scratch tokenizer in our sweeps — 16K/32K/50K all ≤ 128256): HF's `resize_token_embeddings` keeps the **first N pretrained Llama rows** unchanged; **no reinitialization fires.** The from-scratch BPE-32K's token ID 5 is silently mapped onto Llama's pretrained ID 5 row. SFT then has to drift those associations to be useful.
- `new > old`: only the *newly added rows* `[old:new]` are reinitialized to N(0, 0.02²); the first old_vocab_size rows are preserved.

This is a notable correction: prior versions of this doc claimed reinit happens unconditionally on swap. It does not. The pretrained-row-preservation is what makes baseline (b) of the `native_llama` investigation a meaningful control for the existing sweep — both keep the first N pretrained rows; only the tokenizer differs.

For non-standard embedding types (CHARACTER_CNN / CHAR_JABER / CHARFORMER), the embedding layer and `lm_head` are *replaced* (not resized) and the new modules are explicitly reinitialized — see `_adapt_character_cnn` / `_adapt_char_jaber` / `_adapt_charformer` in [llama_adapter.py](src/arabic_eval/models/llama_adapter.py).

### CharacterBERT Limitations
- Auto-regressive `generate()` is **not supported** — `LlamaAdapter.generate()` raises `NotImplementedError` for `CHARACTER_CNN`. QA evaluation falls back to empty predictions.
- The forward pass manually loops through transformer layers (`_forward_character_cnn`) because the standard HF forward expects `input_ids`, not `char_ids`.

### Charformer (GBST) Specifics
- Byte-level tokenization with a fixed 260-id vocab (256 bytes + 4 special tokens). `train()` is a no-op.
- All "subword learning" happens inside `GBSTEmbedding`: byte embed → optional pre-conv (k=5) → enumerate blocks of size 1..M (M=4 default) via mean-pool with stride=b → linear scoring (D→1, no bias) → repeat-interleave back to L → softmax across block sizes per position → weighted sum → final mean-pool with stride `d_s` (2 default).
- Optional position-wise score calibration (`block_attention=true`) implements `P̂ = softmax(P P^T) P` from §2.1.4 of the paper. The paper finds this helps in English and is neutral multilingually.
- The transformer operates on the *downsampled* sequence (length ~L/d_s). `_forward_charformer` shrinks the byte-level attention mask by OR-reducing windows of size `d_s`, then passes `inputs_embeds` (already downsampled by GBST) to the model. The replaced `lm_head` (`CharformerOutputHead`) upsamples back to byte length via `ConvTranspose1d` so byte-level labels align with logits.
- Auto-regressive `generate()` is **not supported** — GBST pools blocks `X[i:i+b]`, so position `i` sees up to position `i+M-1`. The original Charformer is encoder-decoder, sidestepping causality; in our decoder-only setup, only teacher-forced losses (LM perplexity, LightEval log-likelihood MCQ) are well-defined.
- Mechanical extremes on morphological metrics: each token is one byte, which cannot hold a 3-letter Arabic root (each Arabic letter is 2 bytes). Expect `root_conservation_rate ≈ 0`, `pattern_conservation_rate ≈ 0`. Unlike char-JABER, however, `morpheme_integrity_rate` and `clitic_separation_accuracy` are reported as `None` (not ≈1.0): byte tokens never reconstruct to Arabic-letter offsets, so `aligned_token_offsets` always fails and integrity/CSA are *not measurable*. The discriminating metric for Charformer is `semantic_fragmentation_ratio` (alignment-free, observed ~5.4 on a 240-sentence smoke — the highest in the panel by construction). The token-level inventory metrics (`root_bearing_token_pct`, `pattern_bearing_token_pct`) are explicitly reported as `0.0` (not `None`) when the cleaned-token list is empty but raw tokens were generated; this distinguishes the byte-level mechanical zero from "not measured."

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
  - **Files:** server runs in `.venv-camel` (`src/arabic_eval/tools/araroopat_camel_server.py`); client runs in main `.venv` (`src/arabic_eval/tokenizers/araroopat_bridge.py`); the `MorphAnalyzer` in `araroopat_backend.py` wraps the bridge and exposes `analyze` / `analyze_many` / `generate`.
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
- **Full-model fine-tuning, no LoRA / PEFT.** `LlamaAdapter.get_trainable_parameters()` returns `list(self._model.parameters())` and AdamW updates every weight (transformer + replaced embedding + replaced lm_head). If you want LoRA, that's a new model adapter, not a flag.
- **`training.completion_only_loss`** (default `False`, opt-in). When true, prompt tokens are masked to `-100` in the SFT labels and only the answer span contributes to loss. Implemented in `LightEvalBenchmarkTask._build_sft_dataloader` via the longest-common-prefix between the prompt encoding and the full encoding (NOT `len(prompt_enc)` — many tokenizers auto-append `</s>`, so the naive length-based mask cuts the first answer token). Plumbed through `pipeline/experiment.py` via signature-gated kwarg — only LightEval MCQ tasks accept it; QA / text_generation log a warning and ignore. Per the ACVA ablation series (see "Known limitations: ACVA label quality"), this is *not* a fix for noisy short-answer 2-class collapse; treat it as opt-in for clean tasks where the answer-token signal is the binding constraint.
- **`training.ft.enabled`** (default `True`). Master switch for the fine-tuning step. When `false`, the pipeline skips dataloader construction (step 5) and the entire trainer call (step 6) and evaluates the pretrained model directly. Used by `native_llama_no_sft_benchmark_sweep.yaml` to measure the pretrained ceiling. The nested namespace (`training.ft.enabled` rather than `training.enabled`) leaves room for future fine-tuning sub-flags (`training.ft.freeze_layers`, etc.) without bloating `TrainingConfig`. The companion `training.lora.*` namespace is reserved for the same reason — adding LoRA is a new model adapter, not a flag, but if it ever lands the config space is ready.

### Intrinsic Metrics (`src/arabic_eval/evaluation/intrinsic_metrics.py`)

**Size / coverage metrics** (always on):
- **Fertility**: avg tokens per whitespace word
- **Compression ratio**: avg characters per token
- **UNK rate**: fraction of tokens that are `<unk>`
- **Vocab coverage**: fraction of unique words with no UNK tokens
- **Avg token count**: avg tokens per text

**Arabic morphological metrics** (controlled by `evaluation.morphological_metrics`, default `true`; backends `RootExtractor` + `MorphemeSegmenter` live at the bottom of `evaluation/intrinsic_metrics.py`; token-string normalization helpers in `tokenizers/utils/arabic_text.py`; clitic surface sets sourced from `araroopat_backend.PROCLITIC_SURFACES` / `ENCLITIC_SURFACES`):
- **`root_conservation_rate`** (RPS) — % of sampled words whose 3/4-letter root appears as a subsequence inside a *single* token. Penalizes tokenizers that cut through a root.
- **`pattern_conservation_rate`** (PIS) — % of words whose stem-span pattern (root letters + their immediate vowel context, clitics trimmed via `stem_pattern_span`) is recoverable from a single token. Distinguishes BPE/WordPiece splits that destroy the wazn from ones that preserve it.
- **`morpheme_integrity_rate`** — % of Farasa internal morpheme boundaries (e.g. `و|ال|كتاب`) that align with token boundaries. Averaged only over multi-morpheme words. Requires Java (Farasa subprocess); set to `None` if Farasa fails to load.
- **`clitic_separation_accuracy`** (CSA) — % of clitic↔stem boundaries (proclitic-end / enclitic-start positions) that align with token boundaries. Boundaries are detected by walking proclitics from the left of the Farasa segmentation and enclitics from the right, using the CAMeL Tools clitic surface inventory shared with AraRooPat. `ك` is correctly disambiguated by position (proclitic on the left, enclitic on the right). Pooled across the sample (not per-word averaged) so words with more clitics weigh proportionally. Returns `None` when Farasa is unavailable, alignment fails, or no clitic boundaries exist in the sample. Accuracy ceiling: Farasa is occasionally inconsistent on the `أ` question proclitic.
- **`semantic_fragmentation_ratio`** (SFR) — total *raw* (pre-clean) non-special tokens divided by total Farasa morphemes across the sample. Alignment-free — token and morpheme counts both work even when `aligned_token_offsets` fails (ByteLevel BPE artifacts, etc.) so byte-level tokenizers like Charformer report a real (high) fragmentation rather than `None`. SFR ≈ 1.0 means tokens align with morpheme grain; SFR > 1.0 means over-fragmentation; SFR < 1.0 means under-segmentation (one token spans multiple morphemes).
- **`root_bearing_token_pct`** — % of all tokens (across the sample) that contain at least one full root from the sample's root set. Token-level inventory metric.
- **`pattern_bearing_token_pct`** — % of all tokens whose stem span matches a known pattern from the sample's pattern set.

**Architectural reading of the metrics** (use this as a sanity check, not a bug):
- Each whitespace-bounded word is the unit for CharBERT and char-JABER, so two of the metrics hit a mechanical extreme on those tokenizers and should be reported but flagged: CharBERT trivially scores high (but not exactly 1.0 — qalsadi extracts roots that aren't literal subsequences for some weak/irregular forms; observed 0.67 on a 240-sentence smoke) on `root_conservation_rate` (whole word never split → root never split) and ~0.0 on both `morpheme_integrity_rate` and `clitic_separation_accuracy` (no internal token boundaries → no morpheme/clitic boundary aligns). char-JABER is the mirror: ~0.0 on `root_conservation_rate` (single-char tokens can't hold a 3-letter root) and ~1.0 on `morpheme_integrity_rate` AND `clitic_separation_accuracy` (every char boundary is a token boundary). MorphoBPE hits ~1.0 on `morpheme_integrity_rate` AND `clitic_separation_accuracy` *non-trivially* — by design, since it pre-segments with Farasa before training BPE.
- **`semantic_fragmentation_ratio`** is the new discriminator: char-JABER ~2.7, Charformer ~5.4, MorphoBPE ~1.0, BPE ~1.1, CharBERT ~0.5 (1 token per multi-morpheme word) on the same 240-sentence smoke. SFR > 1.0 = over-fragmentation; SFR < 1.0 = under-segmentation; SFR ≈ 1.0 = morpheme-aligned grain.
- **CSA vs `morpheme_integrity_rate`**: CSA restricts to clitic boundaries; integrity covers ALL Farasa boundaries. They satisfy the invariant `integrity == 1.0 ⇒ CSA == 1.0` (clitic boundaries ⊆ all boundaries). Among Farasa-aware tokenizers (MorphoBPE / FarasaCharBERT) both are ~1.0 by construction; among plain subword tokenizers (BPE / WordPiece) CSA isolates clitic-handling specifically, which is the actually-discriminating signal. Don't drop one in favor of the other — integrity catches stem-internal splits (BPE chopping a root in half), CSA does not.
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
- **LightEval benchmarks** (`tasks/lighteval/`): accuracy via log-likelihood multiple-choice scoring on ACVA, Alghafa, Culture-Arabic-MMLU, and Arabic-Exam. See [LightEval Benchmarks](#lighteval-benchmarks-acva-alghafa-culture-arabic-mmlu-arabic-exam) below.

### MEI — Morphological Efficiency Index (`compute_mei` in `evaluation/metrics.py`)

`MEI = (accuracy × RPS × compression × num_eval_rows) / inference_time_sec`, equivalently `(accuracy × RPS × compression) / (inference_time_sec / num_eval_rows)`. A composite per-experiment number that asks: *is good downstream accuracy aligned with high root preservation and high compression, per unit of **per-row** inference time?*

- **Scope**: defined only for LightEval MCQ tasks (`acva`, `alghafa`, `culture_arabic_mmlu`, `arabic_exam`). Detection in the pipeline uses `isinstance(task, LightEvalBenchmarkTask)` so adding a new LightEval benchmark requires no MEI changes.
- **Inputs**: `accuracy` (LightEval MCQ result), `RPS` = `root_conservation_rate` (intrinsic block), `compression` = `compression_ratio` (intrinsic block), `inference_time_sec` (wall-clock around `task.evaluate()`, persisted in the downstream block), `num_eval_rows` = `downstream[task].num_samples` (the row count the eval pass scored).
- **Why row-count normalization.** Without it, MEI's time term scales with eval-set size, so the same tokenizer scores ~2.5× lower on Alghafa (~18.6K rows) than on ACVA (~7.3K rows) for reasons unrelated to per-example efficiency. Per-row form is invariant under dataset size: within a task all tokenizers share the same `num_eval_rows` so rankings are preserved (constant scale); across tasks the time term becomes per-row-comparable. The `compression` factor in the numerator already captures sequence-length differences across tokenizers — dividing time per-row, not per-token, avoids double-counting length.
- **Record shape**: `{"mei": float|None, "status": "ok"|"task_not_mcq"|"missing_<input>"|"zero_time"|"zero_rows", "inputs": {accuracy, rps, compression, inference_time_sec, num_eval_rows, ...}}` at top-level `results["mei"]` in `all_metrics.json`. Inputs are echoed back so the JSON is self-describing — no need to re-read `intrinsic_metrics.json` to debug a `None` MEI, and migration scripts can recompute MEI in-place from `inputs` alone (see `scripts/recompute_mei.py`).
- **Warmup**: pipeline does one throwaway `tokenizer.encode("نص قصير للإحماء")` before `time.perf_counter()` starts. Required for fairness — without it, AraRooPat's CAMeL bridge spawn (~1–2 s) and Farasa Java subprocess startup get billed to the morphology-aware tokenizers' MEI denominator.
- **Mechanical-extreme flagging**: `RPS_MECHANICAL_FLAGS` in `evaluation/reporter.py` is the single source of truth for which tokenizers have a forced RPS (CharBERT and AraRooPat at the ceiling; char-JABER and Charformer at the floor). The sweep `comparison_report.txt` asterisks those tokenizers in the MEI table and prints a footnote. Don't strip the footnote — it's load-bearing for correct interpretation.
- **Tests**: `tests/test_mei.py` covers the typed-status logic, the per-row formula, the within-task-ranking-invariance property, and the `zero_rows` / `missing_num_eval_rows` branches.
- **Migrating archived MEI numbers**: pre-2026-05-04 runs used the per-pass formula. `scripts/recompute_mei.py` walks `outputs/experiments/` and recomputes MEI in place from `mei.inputs` (originals are preserved) plus `downstream.<task>.num_samples` (always present for LightEval MCQ runs). Idempotent — reads the same inputs and writes the same record on re-run. Also regenerates `comparison_report.{txt,json}` per sweep dir afterward.

## LightEval Benchmarks (ACVA, Alghafa, Culture-Arabic-MMLU, Arabic-Exam)

### Overview

These four multiple-choice benchmarks are used to evaluate the impact of tokenizer training choices on downstream Arabic understanding. All evaluation is conducted using the LightEval framework methodology.

| Registry Key | Class | Default Dataset | Metric |
|---|---|---|---|
| `acva` | `ACVATask` | `OALL/ACVA` | accuracy |
| `alghafa` | `AlghafaTask` | `OALL/AlGhafa-Arabic-LLM-Benchmark-Native` | accuracy |
| `culture_arabic_mmlu` | `CultureArabicMMLUTask` | `OALL/Arabic_MMLU` | accuracy |
| `arabic_exam` | `ArabicExamTask` | `MBZUAI/ArabicMMLU` | accuracy |

### Data Split Strategy

All examples across all predefined splits of each benchmark are pooled, then split with a fixed RNG seed:

- **10 %** → supervised fine-tuning (SFT) on formatted MCQ prompts
- **90 %** → reserved for LightEval evaluation (never seen during training)

The same seed is used across all tokenizer variants in a sweep so every experiment evaluates on the identical 90 % partition.

**Stratified per-sub-config split** (multi-config benchmarks). For ACVA (58 cultural topics), Alghafa (sub-tasks), and Arabic_Exam (school subjects), the 10/90 split is applied **within each sub-config** rather than globally. This guarantees ≥ 1 SFT example per sub-config — random global sampling at 10 % could otherwise leave small configs entirely unrepresented in fine-tuning. Per-config seeds are derived as `seed + crc32(config_name)` so the split is fully deterministic. Single-config benchmarks (Culture_Arabic_MMLU) degenerate to one `_default` group → behaviour matches a global split. Note: the floor→ceil rounding required to guarantee coverage shifts the SFT total up by ~one example per sub-config compared to the pre-stratification implementation, so historic run totals will not match exactly. Lives in `LightEvalBenchmarkTask._get_splits` ([tasks/lighteval/base.py](src/arabic_eval/tasks/lighteval/base.py)).

### Evaluation Methodology (LightEval)

For each multiple-choice question the `LightEvalModelWrapper` computes:

```
log P(" A" | context)  …  log P(" D" | context)
```

using the fine-tuned model's forward pass (`_compute_loglikelihood`), then predicts `argmax`. This matches LightEval's standard log-likelihood MCQ protocol exactly. The wrapper implements the same `loglikelihood(requests)` interface as LightEval's `LightevalModel`, so it can be substituted into a full LightEval pipeline if needed.

**CharacterBERT log-likelihood note**: `character_cnn` is scored using the *word-level* logits (the model's `lm_head` indexes the word vocabulary; the `char_ids` 3-D batch flows into the CharCNN, the transformer output is projected to the word vocab, and the continuation scoring sums log P(word) over the continuation's word-vocab IDs). This is implemented in the `character_cnn` branch of `_compute_loglikelihood`. The result is a real accuracy in `[0, 1]` — *not* a 0.0 fallback. **`farasa_character_bert` shares the same `embedding_type` and the same scoring path**; the only difference is that its `lm_head` indexes a *morpheme* vocabulary, so continuation scoring is over morpheme-vocab IDs rather than word-vocab IDs. Both produce real, comparable accuracies on `acva` / `alghafa` / `culture_arabic_mmlu` / `arabic_exam`.

**ACVA word-based scoring note**: ACVA is True/False, not 4-way MCQ, and its continuation pool is just `صح` / `خطأ` (rendered as letter `أ` / `ب` in vanilla LightEval). When scored with single-letter continuations, 99 % of model decisions ended up as near-tie log-likelihoods (differences < 1e-3) dominated by the per-letter unigram prior — accuracy clustered around the majority-class baseline regardless of tokenizer. ACVATask therefore overrides the default scoring hooks to score the words `" صح"` / `" خطأ"` directly: the prompt drops the `أ./ب.` choice listing, SFT supervision ends with the answer word, and `_build_continuations` returns `[" صح", " خطأ"]`. Letter scoring is preserved for the other three benchmarks (which are genuinely 4-way MCQ). The override is implemented via three hooks on `LightEvalBenchmarkTask` (`_format_eval_context`, `_build_continuations`, `_format_sft_text`) — same pattern any future task can use to swap continuations without touching the wrapper. The label strings are centralised on `ACVATask.LABELS` (a tuple, single source of truth) so switching to e.g. `صحيح` / `خاطئ` is a one-line change.

**PMI normalization (`evaluation.score_normalization`)**. The wrapper supports three score-normalization modes, mirroring LightEval's `LogProbNormalization`:

  * `"char"` (default) — divide each per-continuation ll by its character length (LightEval `LogProbCharNorm`). For 1-character letter continuations this is a no-op; for word-scored continuations (ACVA, Alghafa T/F + sentiment) it removes a length bias toward shorter answers. Backward-compatible with every existing run JSON in `outputs/` — the mode is the legacy default, the metrics dict is byte-identical (modulo timestamps).
  * `"pmi"` — score = `log P(c | full_context) − log P(c | unconditioned_context)` (LightEval `LogProbPMINorm`). The unconditioned context is supplied per task via `_unconditioned_query(ex)`, which defaults to `"الإجابة:"` (the bare answer prefix every current task's prompt ends with). PMI cancels the per-continuation prior — the dominant failure mode of letter-MCQ scoring on weak-signal Arabic benchmarks. Empirically: Llama-3.2-1B's unconditional log P( letter | empty MCQ context ) spans ~1.7 nats across ج/د/ب/أ, large enough to dominate the question-conditioned signal on translated MMLU. On the no-SFT failure CSV for `culture_arabic_mmlu` (9,723 wrong rows), recomputing argmax on `ll − prior` flipped 16.8 % of rows wrong → correct (estimated PMI accuracy ≈0.37 vs char-norm 0.2447); the actual lift after the bidirectional flip-back is more modest but consistently positive across all four tasks.
  * `"char+pmi"` — compute both. Metrics dict carries `accuracy_char_norm` and `accuracy_pmi`; the legacy `accuracy` field aliases char-norm so existing comparison-report consumers keep working. Failure CSV gains `score_pmi_*` and `score_pmi_margin` columns alongside the existing `score_*` / `score_margin`.

The flag is plumbed via the existing signature-gating pattern (`inspect.signature(task.evaluate).parameters`) — non-LightEval tasks log a warning and ignore. Unconditioned ll values are cached per `(unconditioned_query, tuple(continuations))`: for letter-MCQ this is one extra forward call total (cache hits every row); for word-scored sub-configs that vary continuations per row the cache misses and we pay one extra forward per example (~2× cost on those rows; acceptable). Default stays `"char"` so existing run JSONs remain reproducible — opt in per-experiment YAML (`evaluation.score_normalization: "char+pmi"`).

**MEI under PMI**. `compute_mei` prefers `accuracy_pmi` when present and records the source as `inputs.accuracy_source = "accuracy_pmi"`; under default char-only mode the field is omitted (preserves byte-identity).

> **Note: re-running the four-quadrant readout.** The existing `outputs/experiments/native_llama_with_sft_benchmark_sweep/four_quadrant_readout.md` was produced under `"char"` scoring. Once a sweep with `"char+pmi"` lands, the table — and the "ACVA cap is tokenizer-induced" finding it anchors — should be re-validated against PMI accuracies before any tokenizer-comparison conclusions are drawn from it. The `culture_arabic_mmlu` row (every cell at ~random) is the most likely to shift: that's the cell most contaminated by the letter-prior bias PMI corrects.

### Fine-tuning Data Format

For letter-scored MCQ benchmarks (`culture_arabic_mmlu` / `arabic_exam`, plus the 4-way and 5-way Alghafa sub-configs) each example is formatted as a plain-text causal-LM sequence:

```
السؤال: {question}

A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
الإجابة: {correct_letter}
```

For **ACVA** (True/False) and the **word-scored Alghafa sub-configs** (T/F facts + 2/3-way sentiment — see *Alghafa heterogeneity* below), the choice listing is dropped and the answer is the word itself:

```
السؤال: {question}
الإجابة: {صح|خطأ}                    # ACVA labels (fixed pair)
الإجابة: {sol_i for i = answer_idx}  # Alghafa word-scored: actual answer text
```

This is tokenised and passed through the existing `Trainer` using the standard collator for the active embedding type.

### Alghafa heterogeneity (per-topic scoring dispatch)

Alghafa is the only benchmark in the suite where one task class spans multiple MCQ shapes. The dataset has **9 sub-configs**:

| Topic class | Sub-configs | Choices | Rows | Scoring |
|---|---|---|---|---|
| 2-way T/F facts | `multiple_choice_facts_truefalse_balanced_task` | sol1-2 | 80 | **word** |
| 2-way binary sentiment | `multiple_choice_rating_sentiment_no_neutral_task` | sol1-2 | 8000 | **word** |
| 3-way sentiment | `multiple_choice_rating_sentiment_task`, `multiple_choice_sentiment_task` | sol1-3 | 6000+1725 | **word** |
| 4-way MCQ | `mcq_exams_test_ar`, `meta_ar_dialects`, `meta_ar_msa` | sol1-4 | 562+5400+900 | letter |
| 5-way grounded statement | `multiple_choice_grounded_statement_soqal_task`, `multiple_choice_grounded_statement_xglue_mlqa_task` | sol1-5 | 155+155 | letter |

The 4 binary/sentiment sub-configs use **word-scored prompts** (mirroring ACVA's fix for the letter-prior pathology — letter-based `أ`/`ب` decisions on a 2-way task collapse to the unigram letter prior). The 4-way and 5-way MCQ sub-configs keep the inherited letter-based default. Per-row dispatch keys on `ex["_source_config"]` populated by `_parse_combined`. The hooks are `_format_eval_context` / `_build_continuations` / `_format_sft_text` overridden in `AlghafaTask`; the dispatch list is `AlghafaTask.WORD_SCORED_CONFIGS` (single source of truth — to add a new sub-config to the word-scored set, edit that frozenset and nothing else).

**Char-normalization on the score aggregator is required for fairness.** When continuations vary in character length (e.g. sol1=`"هو رأي ايجابي"` 12 chars vs sol2=`"هو رأي سلبي"` 11 chars), summed log-probs systematically prefer the shorter answer. The base task class exposes `_aggregate_scores(ex, continuations, log_likelihoods)` (default: char-norm — divide each ll by `len(continuation.lstrip())`, the LightEval `LogProbCharNorm` equivalent). Letter-scored sub-configs all have 1-char continuations, so char-norm is mathematically a no-op for them; ACVA (`صح` 2 chars vs `خطأ` 3 chars) shifts slightly. The aggregator is the override point if a future task wants token-norm or sum-of-log-probs explicitly — same signature-hook pattern as the prompt/continuation overrides.

**Per-sub-config accuracy is emitted automatically.** `evaluate_mcq` buckets `correct/total` by `_source_config` and writes a `per_subconfig_accuracy: {<config>: {accuracy, num_samples}}` dict alongside the aggregate. Visible in `all_metrics.json` and rendered as a separate "Per-sub-config breakdown" section in `comparison_report.txt` (filtered out of the main downstream-task table to keep its column count manageable). Single-config benchmarks degenerate to a `_default` bucket and the breakdown section is suppressed.

**Schema gotcha that bit us once.** `label` in `OALL/AlGhafa-Arabic-LLM-Benchmark-Native` is **0-indexed** (matches LightEval's `alghafa_adapter`), and the two grounded-statement sub-configs ship `sol5`. The pre-2026-05-03 parser assumed 1-indexed labels and only iterated `sol1..sol4` — silently dropping 36% of rows and shifting the rest by −1 position. Verify dataset schemas against real rows from each sub-config when writing or modifying parsers; cross-check against LightEval's reference adapter when one exists.

### Known limitations: ACVA label quality

ACVA's gold labels were synthetically generated and ship with non-trivial noise. A direct inspection of the public dataset surfaced:

- **Wrong gold labels for factually contradicting claims.** Example: `الكبسة هي طبق وطني سعودي` (Kabsa is the Saudi national dish) is labelled `خطأ` (FALSE), and `الكبسة هي وجبة تقليدية في المطبخ السوري` (Kabsa is traditional in Syrian cuisine) is labelled `صح` (TRUE) — the opposite of the factual record. The dataset author's synthetic generator hallucinated cultural facts.
- **~30 % duplicate rate.** Because the 58 sub-configs share questions (e.g. an "Arabs discovered cosmic violet waves" claim appears in `Arab_Empire`, `Arabic_Astronomy`, `Arab_Achievement_Discovery`, …), the merged eval split has 5673 unique questions out of 8100 rows.
- **51 within-eval label conflicts.** The same question appears with both `صح` and `خطأ` golds in the same partition.
- **Pseudo-random model behaviour under letter-based scoring** (now mitigated by word-based scoring, see above).

The word-scoring override mitigates the pseudo-random behaviour but does **not** fix the upstream gold-label problem. Treat ACVA accuracy as label-noisy: cross-validate against the other three benchmarks before drawing tokenizer conclusions. The sweep `comparison_report.txt` flags ACVA with a dagger `†` and a footnote — keep the footnote (it's the equivalent of `RPS_MECHANICAL_FLAGS` for label-noisy tasks; the single source of truth is `LABEL_NOISY_TASKS` in [evaluation/reporter.py](src/arabic_eval/evaluation/reporter.py)).

**SFT-setup ablations don't break the discriminative ceiling.** A 2x2 ablation on `bpe_50k` over `train_split_ratio ∈ {0.10, 0.50}` × `completion_only_loss ∈ {F, T}` (artifacts: `outputs/experiments/ACVA_{top_tokenizers_benchmark_sweep, completion_loss_ablation, split_ratio_ablation, split_and_completion_ablation}/`):

| | mask=F | mask=T |
|---|---|---|
| split=0.10 | acc 0.5958 — 99.9% wrong-as-صح (PRIOR / majority-class baseline) | acc 0.5892 — 97.7% wrong-as-صح, margins compressed (98.1% < 1e-3) |
| split=0.50 | acc 0.5481 — **64.2%** wrong-as-صح (only cell to break the trap, but below baseline) | acc 0.5966 — 99.8% wrong-as-صح (reverted to always-class) |

Read together within the from-scratch tokenizer family: completion-only loss alone compresses margins toward the prior without breaking the trap; more SFT data alone breaks the trap but accuracy lands below baseline; combining them snaps back to majority-class (concentrated capacity + noisy 1–3 token answer = the model memorizes P(class) faster). **For from-scratch tokenizers in our pipeline, ACVA caps near 0.595–0.610.** Don't propose `completion_only_loss=true` as a fix for this ceiling — it's not the SFT setup, but the *tokenizer*. The native_llama baselines (see "Pretrained-tokenizer baselines (native_llama)" below) reveal that native tokenizer + same SFT pipeline reaches 0.7133 on ACVA, so the ceiling is from-scratch-tokenizer-induced, not label-noise-induced. The label-noise concerns (synthetic generation, ~30 % duplicates, 51 within-eval label conflicts) are still real and still warrant the dagger flag, but they are not the binding constraint on accuracy. For tokenizer comparisons on ACVA among from-scratch tokenizers, accept the ~0.6 cap.

### Pretrained-tokenizer baselines (native_llama)

Two reference baselines anchor the from-scratch sweeps against the pretrained-tokenizer ceiling. Both use the `native_llama` wrapper (which wraps `meta-llama/Llama-3.2-1B`'s pretrained tokenizer). With `vocab_size=128256` matching the model's embedding matrix, `resize_token_embeddings` is a no-op and the pretrained embeddings stay byte-identical — the only thing differing across (a)/(b)/existing-sweep is the tokenizer choice (and whether SFT runs).

| | no SFT | with SFT |
|---|---|---|
| **native_llama** (this section) | (a) — pretrained ceiling | (b) — pretrained + our SFT pipeline |
| **bpe_50k from-scratch** | not run (random reinit if forced — uninformative) | existing 50K sweep |

- **(a) vs majority-class on ACVA** confirms / refutes the "label noise is the ceiling" claim independent of any of our pipeline code paths
- **(a) vs (b)** shows whether our 10%-split SFT helps, washes, or hurts on these benchmarks
- **(b) vs the existing 50K sweep** shows the cost of using a 50K from-scratch tokenizer instead of the native 128K one, controlling for the SFT pipeline

Configs: [native_llama_no_sft_benchmark_sweep.yaml](configs/experiments/native_llama_no_sft_benchmark_sweep.yaml) (a) and [native_llama_with_sft_benchmark_sweep.yaml](configs/experiments/native_llama_with_sft_benchmark_sweep.yaml) (b). Wrapper at [tokenizers/native_llama.py](src/arabic_eval/tokenizers/native_llama.py); flag at `training.ft.enabled` (default `True`; baseline (a) sets it `false`).

**Results (run 2026-05-03)** — full readout at [outputs/experiments/native_llama_with_sft_benchmark_sweep/four_quadrant_readout.md](outputs/experiments/native_llama_with_sft_benchmark_sweep/four_quadrant_readout.md).

| Task | Random | bpe_50k+SFT (existing) | (a) native, no SFT | (b) native, +SFT | Δ (b) − existing |
|---|---|---|---|---|---|
| acva | 0.50 | 0.5958 (majority-class) | 0.6078 | **0.7133** | **+11.75 pp** |
| alghafa † | 0.25 | 0.2811 | 0.3275 | 0.5930 | (invalid; see footnote) |
| arabic_exam ‡ | 0.25 | **0.2679** (post-fix) | **0.3053** (post-fix) | **0.4067** (post-fix) | **+13.88 pp** |
| culture_arabic_mmlu | 0.25 | 0.2625 | 0.2447 | **0.2768** | +1.43 pp |

† **Alghafa pre-fix numbers are invalid.** The 2026-05-03 parser fix found two bugs (label off-by-one + `sol5` truncation) that silently dropped ~36 % of rows and shifted gold answers by −1 across the rest. The "+31.19 pp" delta was an artifact of the bug. A post-fix re-run of native, no-SFT (`configs/experiments/native_llama_no_sft_alghafa_postfix.yaml`) gives **0.4031** on **20,677 examples** (was 13,199 pre-fix). The per-sub-config breakdown shows real signal where the model has a chance: word-scored `multiple_choice_rating_sentiment_no_neutral_task` (7,200 rows) lands at **0.5726**, above the majority-class baseline ≈0.51; letter-scored 4/5-way MCQ sub-configs sit near random as expected for an un-fine-tuned pretrained Llama on Arabic letter-MCQ. The bpe_50k+SFT and native+SFT alghafa rows of the table need re-running before any cross-tokenizer claim can be made; do **not** cite the +31.19 pp number. See *Alghafa heterogeneity* above for the per-topic dispatch and *Known limitations: ACVA label quality* for the unrelated ACVA noise discussion (which is unaffected by the alghafa fix).

‡ **Arabic_Exam pre-fix numbers were on a contaminated eval set.** The 2026-05-04 parser fix found four bugs in [tasks/lighteval/arabic_exam.py](src/arabic_eval/tasks/lighteval/arabic_exam.py): (1) the `MBZUAI/ArabicMMLU` `All` config is a strict union of the other 40 subject configs, so the merger loaded every row twice and the stratified 10/90 split leaked SFT pairs into the eval split (both copies got independent permutations under different `_source_config` keys); (2) the parser dropped the `Context` field that ~5 % of rows depend on for a literal "based on the passage" answer; (3) iterating only Options 1–4 silently dropped 141 `Answer Key=E` rows and truncated the choice list of 344 5-option rows; (4) `is_few_shot=1` rows (the dataset's own dev-split demos) were not filtered. Post-fix eval pool drops from ~26 000 to ~13 000 (one copy each, no few-shot, with context). All three cells fell modestly: bpe_50k+SFT 0.2994→**0.2679**, native no-SFT 0.3125→**0.3053**, native+SFT 0.4217→**0.4067**. The directional findings hold (native+SFT >> native no-SFT >> bpe_50k+SFT ≈ random) but the absolute deltas shrank because the contaminated split was inflating bpe_50k disproportionately via SFT/eval leakage in the `All`-group cross-permutation. Mechanism is `ARABIC_EXAM_EXCLUDED_CONFIGS = frozenset({"All"})` defined in `arabic_exam.py` and passed inline to `utils.load_huggingface_mcq` from `ArabicExamTask.load_examples`. Configs: [arabic_exam_postfix_native_no_sft.yaml](configs/experiments/arabic_exam_postfix_native_no_sft.yaml) (a), [arabic_exam_postfix_native_with_sft.yaml](configs/experiments/arabic_exam_postfix_native_with_sft.yaml) (b), [arabic_exam_postfix_bpe_50k.yaml](configs/experiments/arabic_exam_postfix_bpe_50k.yaml) (existing). Tests: [tests/test_arabic_exam_parser.py](tests/test_arabic_exam_parser.py); end-to-end smoke: [scripts/smoke_arabic_exam_parser.py](scripts/smoke_arabic_exam_parser.py).

> **Side observation (not a bug).** After de-duplication, ~235 question strings still appear in 2+ subject configs (e.g. tagged in both `Islamic Studies` and `Islamic Studies (Middle School)`). These are inter-subject overlaps inherent to the dataset's taxonomy, not a merge artifact, and are kept as-is. ~1.6 % of the eval pool.

**Major finding — the previous ACVA-label-noise-ceiling claim was wrong.** The 2x2 ablation series (split_ratio × completion_only_loss on `bpe_50k`) plateaued at ~0.595 on ACVA, which we attributed to label noise. Native tokenizer + the same SFT pipeline reaches 0.7133 — there's a 12-point ceiling-headroom that the from-scratch tokenizer was hiding. **The binding constraint on the existing 50K sweep was the from-scratch tokenizer's randomly-aligned pretrained rows**, not data quality. The 2x2 ablation results remain internally consistent as a description of from-scratch-tokenizer SFT behavior; the misattribution was treating that ceiling as data-imposed rather than tokenizer-imposed.

**Native (a) > existing 50K+SFT on the 3 valid tasks.** A pretrained model with NO SFT outperforms the from-scratch BPE-50K + SFT on ACVA (0.6078 vs 0.5958), arabic_exam (post-fix: 0.3053 vs 0.2679), and culture_arabic_mmlu (0.2447 vs 0.2625 — the lone exception, native no-SFT *underperforms* on this one). The 10% SFT split isn't enough to overcome the random row-mapping (BPE-50K's token ID 5 → Llama's pretrained ID 5 row, etc.). The alghafa comparison was originally listed here too, but both numbers were on buggy data; that comparison is pending re-eval.

**Implications for the from-scratch sweep methodology.** Comparing 8 from-scratch tokenizers on these benchmarks doesn't cleanly measure tokenizer quality — it measures how each tokenizer's vocab indices happen to align with Llama's first N pretrained rows, modulated by what 10% SFT can drift. The native (b) result is the useful upper-bound reference for the same SFT pipeline; the (b) − existing gap is the cost of from-scratch-tokenizer + silent-row-preservation.

**Don't add native_llama to the 8-tokenizer comparison tables.** It's a baseline, not a competitor: different vocab regime (128K vs 16K/32K/50K), different special-token convention, `train()` short-circuited; intrinsic/MEI scores compute mechanically but aren't peer-comparable to the from-scratch tokenizers' numbers.

### Key Classes (`src/arabic_eval/tasks/lighteval/`)

| Symbol | Module | Role |
|---|---|---|
| `LightEvalBenchmarkTask` | `lighteval/base.py` | Abstract base — 8 abstract hooks; concrete 10/90 split + SFT dataloader + `evaluate()` |
| `LightEvalModelWrapper`  | `lighteval/base.py` | Wraps `BaseModelAdapter` for LightEval's `loglikelihood` interface |
| `_compute_loglikelihood` | `lighteval/base.py` | Core per-token log-likelihood sum (LightEval methodology) |
| `format_mcq_context`     | `lighteval/utils.py` | Formats question + choices as LightEval context string (opt-in) |
| `format_mcq_full`        | `lighteval/utils.py` | Formats complete MCQ + answer for SFT (opt-in) |
| `char_norm_aggregator`   | `lighteval/utils.py` | LightEval `LogProbCharNorm` equivalent (opt-in) |
| `parse_mcq_generic`      | `lighteval/utils.py` | A/B/C/D-style row parser (opt-in) |
| `load_huggingface_mcq`   | `lighteval/utils.py` | HF loader with multi-config auto-detection + exclusions (opt-in) |
| `ACVATask` / `AlghafaTask` / `CultureArabicMMLUTask` / `ArabicExamTask` | `lighteval/{acva,alghafa,culture_arabic_mmlu,arabic_exam}.py` | Concrete benchmark implementations |

### Dataset Field Schemas

`_parse_mcq_generic()` in the base class handles three common formats automatically:

| Format | Fields |
|---|---|
| Separate columns | `question`, `A`, `B`, `C`, `D`, `answer` (letter or int) |
| Choices list | `question`, `choices` (list), `answer` (int) |
| Options list | `question`, `options` (list), `label` (int) |

Subclasses can override `_parse_example()` for non-standard schemas.

**Arabic_Exam schema (`MBZUAI/ArabicMMLU`)** is non-standard and overrides `_parse_example()`. Fields: `Question`, `Context` (optional supporting passage, ~5 % of rows; **must be prepended to the prompt** — the question often refers to it explicitly), `Option 1` … `Option 5` (5 distractors max; ~344 rows ship with `Option 5`), `Answer Key` (Latin letter A–E), `is_few_shot` (dev-split demonstration flag — **filtered out**, ~120 rows), plus metadata (`Subject`, `Group`, `Level`, `Country`, `Source`). Multi-config: 41 configs ship, but the `All` config is a strict union of the other 40 (verified `|All| = sum(|other 40|) = 14575`); excluded via `EXCLUDED_CONFIGS = frozenset({"All"})` on `ArabicExamTask` to avoid 2× row duplication and SFT/eval leakage. After the 4 fixes documented at the ‡ footnote in the four-quadrant table below, the eval pool is ~13 000 (was ~26 000 contaminated).

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

> **Note:** Confirmed Hub defaults are: `acva` → `OALL/ACVA`; `alghafa` → `OALL/AlGhafa-Arabic-LLM-Benchmark-Native`; `culture_arabic_mmlu` → `OALL/Arabic_MMLU`; `arabic_exam` → `MBZUAI/ArabicMMLU`. Override via `params.dataset_name` if a benchmark moves on the Hub.

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
  failure_reports: false        # Per-task CSV of failing eval cases (LightEval MCQ only)
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
        dataset_name: "OALL/AlGhafa-Arabic-LLM-Benchmark-Native"
        train_split_ratio: 0.10
    - type: "culture_arabic_mmlu"
      params:
        dataset_name: "OALL/Arabic_MMLU"
        train_split_ratio: 0.10
    - type: "arabic_exam"
      params:
        dataset_name: "MBZUAI/ArabicMMLU"
        train_split_ratio: 0.10
```

## Output Structure

Each experiment produces:
```
outputs/experiments/<name>/
  config.json               # Full resolved config
  intrinsic_metrics.json    # Fertility, compression, UNK rate, coverage
  all_metrics.json          # Combined intrinsic + downstream + mei (top-level "mei" key for LightEval MCQ tasks)
  training/
    train_results.json      # Loss, steps, time
    best/                   # Best checkpoint (early stopping)
    final/                  # Final checkpoint
    checkpoint-*/           # Periodic checkpoints
  failure_reports/          # Only when evaluation.failure_reports=true
    <task_name>_accuracy_failures.csv   # LightEval MCQ tasks only
  experiment.log            # Full log file
```

Sweep mode additionally generates: `comparison_report.txt` and `comparison_report.json` in the sweep output directory. The text report includes a "Composite Metric: MEI" section with per-experiment rows + an asterisk-and-footnote on tokenizers with mechanical RPS extremes (`RPS_MECHANICAL_FLAGS` in `evaluation/reporter.py`). For multi-sub-config benchmarks (Alghafa today; any future heterogeneous benchmark), a "Per-sub-config breakdown" sub-section per task is emitted under the main downstream-task table — rows = experiments, columns = sub-configs, cells = `accuracy (n=N)`. Single-config benchmarks (`_default` bucket only) suppress this section; the main table strips `per_subconfig_accuracy.*` so its column count stays manageable.

### Failure-case CSV reports (opt-in)

When `evaluation.failure_reports: true`, LightEval MCQ tasks (`acva`, `alghafa`, `culture_arabic_mmlu`, `arabic_exam`) write one CSV per task with one row per wrong-answer example. Columns: `index, question, choice_0..N, gold_idx/letter, pred_idx/letter, ll_0..N, ll_margin, score_0..N, score_margin`. Both raw and aggregated views are persisted: `ll_*` is the raw model log-likelihood (sum over continuation tokens) and `ll_margin = ll_pred − ll_gold`; `score_*` is the value passed to `argmax` after `_aggregate_scores` (default char-norm, LightEval `LogProbCharNorm` equivalent), with `score_margin` defined the same way. They differ only when continuation lengths differ — letter-scored MCQ rows have `ll_* == score_*` (1-char continuations); ACVA and word-scored Alghafa rows have `score_*` divided by the character count of the answer text. The margins distinguish confident-wrong (large positive) from near-tie failures (~0); use `score_margin` to interpret the model's actual decision and `ll_margin` to debug the unnormalized signal. UTF-8-BOM encoding so Excel renders Arabic correctly.

QA and text_generation **do not** support failure reporting; the pipeline detects this via `inspect.signature(task.evaluate)` and silently skips. To opt a new task in, just add `failure_report_dir: Optional[Path] = None` to its `evaluate()` signature and handle it — no base-class change needed.

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
- CharacterBERT (`character_cnn`) **does** support LightEval log-likelihood scoring — the `character_cnn` branch in `_compute_loglikelihood` runs the CharCNN on `char_ids` and scores continuations against the word-vocabulary `lm_head`. Returns a real accuracy in `[0, 1]`. **`farasa_character_bert` shares the same path** (same `embedding_type=character_cnn`); its `lm_head` indexes a *morpheme* vocab instead of a word vocab, so scoring is over morpheme-vocab IDs. The shared remaining limitation is **autoregressive `generate()`**, which `LlamaAdapter.generate()` raises `NotImplementedError` for on `CHARACTER_CNN` — so QA evaluation falls back to empty predictions for both. (Older docs claimed `accuracy=0.0` on log-likelihood for these tokenizers; that was true before the `character_cnn` branch was added in `_compute_loglikelihood` and is no longer accurate.)
- The `dataset_name` defaults (in `configs/tasks/*.yaml` and `_default_dataset_name()` on each task class): `acva` → `OALL/ACVA`; `alghafa` → `OALL/AlGhafa-Arabic-LLM-Benchmark-Native`; `culture_arabic_mmlu` → `OALL/Arabic_MMLU`; `arabic_exam` → `MBZUAI/ArabicMMLU`. Override via the `params.dataset_name` field if a benchmark moves on the Hub.
