# Arabic Tokenizers Evaluation Platform

## Project Overview

A universal platform for evaluating Arabic tokenizers by measuring LLM downstream performance. All external parameters (training pipeline, datasets, model architecture, hyperparameters) are held fixed; only the tokenizer changes between experiments. Every condition (native Llama tokenizer + every from-scratch tokenizer variant) runs the same fixed 3-phase training pipeline, then is evaluated on the same downstream benchmarks.

- **Tokenizer-training corpus**: `Jr23xd23/ArabicText-Large` (HuggingFace) — used for tokenizer training + intrinsic eval
- **Phase 1 + 2 corpus**: `Mostafa3zazi/Arabic_SQuAD` (machine-translated SQuAD-v1, 48,344 train rows)
- **Phase 3 corpus**: TyDiQA-Arabic (`google-research-datasets/tydiqa` `secondary_task` filtered to Arabic — 14,805 train / 921 val rows) + ARCD (`hsseinmz/arcd` `plain_text` — 693 train / 702 val rows)
- **Primary LLM**: LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`)
- **Eval benchmarks**: four LightEval log-likelihood MCQ benchmarks — ACVA, Alghafa, Culture-Arabic-MMLU, Arabic-Exam. Eval is **full benchmark** (no SFT split — training is task-agnostic).
- **Vocab sizes tested**: 16K, 32K, 50K for subword tokenizers; fixed char vocab for character-level; fixed 260-id byte vocab for Charformer; 128256 for native_llama (matches model embedding matrix)

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
  data/                   # loader.py (main corpus), preprocessing.py, collation.py,
                          #   answer_only_masking.py (LCP helper),
                          #   finetune_corpora.py (Arabic-SQuAD + TyDiQA + ARCD)
  tokenizers/             # base.py + 8 implementations + native_llama wrapper
  models/                 # base.py, llama_adapter.py, embeddings/{standard,character_cnn,char_jaber_embed,charformer_embed}.py
  tasks/                  # base.py + lighteval/ (abstract base + 4 dataset files +
                          #   utils.py opt-in helpers). LightEval is eval-only.
  training/               # freezing.py, phases.py (3-phase runner)
  evaluation/             # metrics.py, evaluator.py, reporter.py, intrinsic_metrics.py
  pipeline/               # experiment.py (end-to-end orchestrator)
configs/                  # YAML configs: base, tokenizers/, models/, tasks/, experiments/
scripts/                  # CLI entry points: train_tokenizer.py, run_experiment.py, evaluate_intrinsic.py, compare_results.py
outputs/                  # Gitignored: tokenizers/, experiments/, logs/, data_cache/
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

## The 8 Tokenizers (+ NativeLlama wrapper)

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
| — | NativeLlama | `native_llama` | `tokenizers/native_llama.py` | standard | Wraps `meta-llama/Llama-3.2-1B`'s pretrained tokenizer; `train()` is a no-op. `vocab_size = 128256` matches the model's embedding matrix → `resize_token_embeddings` is a no-op and pretrained embeddings stay byte-identical. Llama uses **tied embeddings** — `lm_head.weight is model.embed_tokens.weight` — so `lm_head` is absent from `named_parameters()`; the freezing helper warns and continues (training `embed_tokens` IS training `lm_head` under tied weights). Special tokens follow Llama: `bos=128000`, `eos=128001`, `pad=128001` (= eos, HF standard; collator masks via `attention_mask`, not `pad_id`), `unk=128002` (`<\|reserved_special_token_0\|>`). Runs the same 3-phase pipeline as every other tokenizer. Used as the canonical reference experiment (`native_llama_3phase_with_sft.yaml` + `native_llama_3phase_no_sft.yaml`). |

All implement `BaseTokenizer` (`tokenizers/base.py`): `train()`, `encode()`, `decode()`, `save()`, `load()`, `vocab_size`, `embedding_type`, `special_tokens`, `get_embedding_config()`.

**Pre-segmentation and embedding family are orthogonal axes.** The Farasa-CharacterBERT case shows that you can pair MorphoBPE's front-end (Farasa morphological segmentation) with CharacterBERT's back-end (CharCNN over characters of each unit). When the only thing changing is the pre-step, **subclass the existing tokenizer and override `train`/`encode`** — don't copy the encoding logic. The CharCNN embedding dispatch in `LlamaAdapter.adapt_to_tokenizer()` and the `CharacterCNNCollator` are reused unchanged because both classes share `embedding_type=character_cnn`.

Special tokens for the 8 from-scratch tokenizers: `<pad>` (0), `<s>` (1), `</s>` (2), `<unk>` (3). NativeLlama deviates from this convention because IDs 0–3 are regular ASCII (`!`, `"`, `#`, `$`) in Llama's vocab — using (0,1,2,3) would have collided with pretrained character embeddings. The collator's label-masking is independent of pad_id value (it builds attention_mask from "real token positions"), so wrappers may set `pad = eos` without breaking the loss path.

## Experiment Pipeline Flow (`src/arabic_eval/pipeline/experiment.py`)

`run_experiment(config)` executes these steps once per (tokenizer, vocab_size) cell:

1. **Set seed** and create output directory
2. **Load main Arabic corpus** via HF `datasets`, apply Arabic preprocessing (normalization, optional diacritics removal), split train/eval. Used only for tokenizer training + intrinsic eval.
3. **Train tokenizer** from scratch on training texts (or load from `load_path` if set)
4. **Intrinsic evaluation** — compute fertility, compression ratio, UNK rate, vocab coverage, plus Arabic morphological metrics (root_conservation_rate, etc.) when `evaluation.morphological_metrics: true`.
5. **Load LLM** and call `adapt_to_tokenizer()` — resizes/replaces embedding layers
6. **Run training phases** — three independently-toggleable phases run in fixed order. See *3-Phase Training Pipeline* below for the per-phase contract.
7. **Downstream evaluation** — for each task in `sweep.tasks`, run LightEval log-likelihood scoring on the **full benchmark** (no SFT split — training was task-agnostic). Wall-clock wrapped via `time.perf_counter()` after a tokenizer warmup encode. Per-task `inference_time_sec` recorded.
8. **Composite metric** — `compute_mei()` per task produces the Morphological Efficiency Index. Stored at top-level `results["mei"][<task>]`.
9. **Save results** as `all_metrics.json`

`run_sweep(config)` iterates `run_experiment` over multiple (tokenizer, vocab_size) cells. The eval task list (`sweep.tasks`) is shared across all cells — training happens once per cell, eval iterates over tasks.

## 3-Phase Training Pipeline (`src/arabic_eval/training/phases.py`)

Every training run executes the same three phases regardless of tokenizer / model / eval task:

| Phase | YAML key | Trains | Body | Dataset | Loss | Default budget |
|---|---|---|---|---|---|---|
| 1 — Embedding alignment | `embedding_alignment` | `embed_tokens` + `lm_head` only | frozen | `arabic_squad` | full-sequence causal LM | 1000 steps, LR=1e-3, BS=8, constant LR |
| 2 — Warmup | `warmup` | all params | unfrozen | `arabic_squad` | answer-only | 2000 steps, LR=2e-4, BS=4×4, cosine + 100 warmup |
| 3 — SFT | `sft` | all params | unfrozen | `tydiqa_arabic + arcd` | answer-only | 2000 steps, LR=2e-4, BS=4×4, cosine + 100 warmup, early-stop |

Each phase is independently toggleable via its own `enabled` flag. Phase 3 additionally runs periodic eval on TyDiQA-val + ARCD-val for stagnation early-stop (patience=5, min_delta=5e-4, min_steps_before_stop=500, restore-best-at-end=true).

**Per-phase params (all adjustable):** `enabled`, `datasets` (registry keys: `arabic_squad`, `tydiqa_arabic`, `arcd`), `trainable_parameters` (substring list; `["*"]` = all), `steps`, `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `optimizer`, `weight_decay`, `max_length`, `loss_target` (`"full_sequence"` | `"answer_only"`), `lr_scheduler` (`"cosine"` | `"constant"` | `"linear"`), `warmup_steps`, `max_grad_norm`, `save_checkpoint`. Phase 3 also has `early_stopping`. Defaults in `configs/base.yaml`.

**Why this design.** The previous "10% SFT on benchmark + 90% eval" was empirically destructive: each from-scratch tokenizer's vocab indices were silently mapped onto Llama's first N pretrained rows by `resize_token_embeddings`, and 10% benchmark-specific SFT couldn't drift those mappings far enough to find real signal. The 3-phase pipeline (a) deliberately aligns embeddings before any other training (Phase 1), (b) teaches QA format on a regular translated dataset before exposure to native Arabic complexity (Phase 2), and (c) does the decisive SFT on native Arabic QA in Phase 3. The whole pipeline runs identically across all conditions so the only experimental variable is the tokenizer + its embedding/lm_head weights.

**Phase 3 prompt format** (`### السياق:` / `### السؤال:` / `### الإجابة:` block) lives in [src/arabic_eval/data/finetune_corpora.py](src/arabic_eval/data/finetune_corpora.py). LightEval per-task prompts are unchanged.

**Answer-only loss masking** uses an LCP (longest common prefix) helper at [src/arabic_eval/data/answer_only_masking.py](src/arabic_eval/data/answer_only_masking.py) — necessary because Llama auto-appends `</s>` to standalone encodings, so naive `labels[:len(prompt)] = -100` would eat the first answer token.

**Tied embeddings on Llama-3.2-1B** — `lm_head.weight is model.embed_tokens.weight`, so `lm_head` is absent from `named_parameters()`. The freezing helper ([src/arabic_eval/training/freezing.py](src/arabic_eval/training/freezing.py)) warns when a substring matches no parameter while others do (the tied-weight case) and continues — training `embed_tokens` IS training `lm_head`.

## CLI Commands

```bash
# Train a single tokenizer
.venv/bin/python scripts/train_tokenizer.py --type bpe --vocab-size 32000

# Run a single experiment (3-phase training + eval on every task in sweep.tasks)
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/native_llama_3phase_with_sft.yaml

# Same but Phase 3 disabled (Phase 1 + Phase 2 only)
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/native_llama_3phase_no_sft.yaml

# Sweep over multiple tokenizer cells (training happens per cell; eval task list shared)
.venv/bin/python scripts/run_experiment.py \
  --config configs/experiments/<sweep_yaml>.yaml --sweep

# Intrinsic-only evaluation of a saved tokenizer
.venv/bin/python scripts/evaluate_intrinsic.py --tokenizer-path outputs/tokenizers/bpe_32k --type bpe

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

### Adding a new eval task (non-LightEval)

Tasks are eval-only under the 3-phase pipeline. There's no per-task SFT contract.

1. Create `src/arabic_eval/tasks/my_task.py`
2. Implement `BaseTask` (just `evaluate`, `name`, `metric_names` — `get_dataloader` is gone)
3. Decorate with `@task_registry.register("my_task")`
4. Add import in `src/arabic_eval/tasks/__init__.py`
5. Create `configs/tasks/my_task.yaml`

If you need a task-specific evaluation flag, plumb it via the signature-gating pattern (see *Optional eval features: opt-in via signature* in the skill — `inspect.signature(task.evaluate).parameters`) rather than widening the abstract base.

### Adding a new LightEval benchmark task

1. Create `src/arabic_eval/tasks/lighteval/<my_benchmark>.py` and subclass `LightEvalBenchmarkTask`
2. Implement the 7 abstract hooks: `_default_dataset_name`, `name`, `_parse_example`, `load_examples`, `_format_eval_context`, `_build_continuations`, `_aggregate_scores`. The base class is intentionally opinion-free — every dataset declares its own prompt shape, continuations, and aggregation policy. Most letter-MCQ datasets reuse `utils.format_mcq_context` / `utils.char_norm_aggregator`; HF-loaded datasets call `utils.load_huggingface_mcq` from `load_examples`.
3. Decorate with `@task_registry.register("my_benchmark")`
4. Add the new module to the auto-import list in `src/arabic_eval/tasks/lighteval/__init__.py`
5. Create `configs/tasks/my_benchmark.yaml` with `dataset_name` and any task-specific overrides.

`get_eval_examples()` (returning the full list after the optional `clean_latin_rows` filter) and the LightEval log-likelihood evaluation are inherited from the base class. A future benchmark loaded from a non-HF source (local files, S3, …) just implements its own `load_examples` without touching the base.

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
- **لِ + الـ contraction**: Arabic writes one lam when the preposition لِ precedes the article (لِ + الوَلَد → لِلوَلَد), so the article's surface is `ل`, not `ال`. Stripping `ال` literally failed, leaving a stray lam in the bare pattern (`ل1ِ2ا3ِ`) and the reconstructed stem (`لكتاب`) — 2,608 of 21,578 unique patterns were duplicates of this shape, 12 % of the inventory, and `للولد` decoded as `لالولد`. Handled by `strip_proclitics_from_start` (encode, mirrored in the server's `_normalize_pattern`) and `join_proclitics` (decode); **the two must stay inverse**. Pre-existing and independent of the `#` work — it affects sound roots too.
- **Weak-radical handling (`#`)**: CAMeL marks a radical whose surface realization is not stable across the paradigm — the weak letters و/ي/ا and the hamza family — with `#` (e.g. `'ق.#.ل'` for every form of ق-و-ل: قال / يقول / قول / أقوال). It is a *radical*, not a missing field; the letter that actually surfaces sits in the pattern as literal template material (`'1ا3َ'`, `'يَ1ُو3'`). Verified on the corpus: for **457 of 457** masked radicals the corresponding slot digit is absent from the pattern. We therefore **keep `#` in the root token** (`[ROOT_ق#ل]`) and count radicals structurally (`root.split('.')`), rejecting only genuinely sub-trilateral roots. Deleting `#` — the behaviour before 2026-08-30 — was doubly destructive: it dropped the root below the 3-radical bar *and* renumbered the remaining radicals so the pattern's slot digits no longer indexed the right letters. It routed **49 % of word occurrences** in real eval text to the character fallback. Trade-off to know: `#` is lossy — `[ROOT_س#ر]` covers both س-و-ر (سور, wall) and س-ي-ر (سار, walk). The `(root, pattern)` *pair* stays unambiguous, the root token alone does not. Canonicalizing `#` to a guessed radical was rejected: it merges the same two roots while hiding the ambiguity behind a plausible-looking root.
- **Generation is supported** (unlike CharBERT/Charformer): the LLM emits `[ROOT_x] [PAT_y]` and decode reconstructs the inflected stem in O(1) via the lookup table. Useful for QA evaluation.
- **Measured effect of the 2026-08-30 `#` + budget fix** (same corpus, same 300 Arabic-Exam questions): words on the ROOT+PAT path 22.1 % → **48.2 %**, fertility 5.76 → **4.47**, compression 1.02 → **1.32**, `root_conservation_rate` 0.1996 → **0.3541**, `root_conservation_attainable` 0.2321 → **0.4071**, analyzed types 297,177 → 506,101, reconstruction entries 60,248 → 108,149. Artifacts: `araroopat_balanced` (before), `araroopat_hashfix_p500` (`#` fix only — 26.1 %, the ablation showing the budget was the real bottleneck), `araroopat_hashfix` (current). Downstream accuracy impact is **not yet measured** — that needs a 3-phase re-run.
- **Mechanical metrics ceiling**: the conservation metrics are ~1.0 by construction *for words that take the root+pattern path*; the measured value is dominated by how often that path fires. `morpheme_integrity_rate` and `clitic_separation_accuracy` reading 1.0 is an artifact of the measured population, not an architectural ceiling — araroopat's ROOT+PAT output never satisfies `aligned_token_offsets`, so those two rates describe only its character-fallback path (read them with `morph_alignment_coverage`).
- **Vocab budget tiers** (in `configs/tokenizers/araroopat.yaml`), re-tiered 2026-08-30 after the `#` fix: Compact 10K roots + 1K patterns (~5.2K vocab, 33.6 % admission), **Balanced 10K + 4K** (~8.2K vocab, 48.2 %; default), Max 10K + 6076 (~10.3K vocab, 52.1 %). *Admission* = share of word occurrences reaching the ROOT+PAT path on real eval text; CAMeL analyses 66.6 % of occurrences, which is the ceiling for any budget. `max_roots` is never binding (only 4,128 roots clear `min_root_freq`) — spend the budget on patterns, because keeping `#` moves the weak letter's identity out of the root and into the pattern (unique patterns 8,634 → 21,578).
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

### Training Loop (`src/arabic_eval/training/phases.py`)
- Step-driven loop (not epoch-driven) — each phase has its own `steps` budget
- AdamW optimizer with cosine / constant / linear LR schedule + optional linear warmup (per-phase)
- Gradient accumulation (per-phase; defaults: Phase 1 = 1, Phase 2/3 = 4)
- Mixed precision via `torch.amp.autocast` (bf16 by default; controlled by `training.bf16` / `training.fp16`)
- Gradient clipping at `max_grad_norm` (per-phase, default 1.0)
- Phase 3 only: stagnation early-stop with patience + min_delta + min_steps_before_stop + restore-best-at-end
- Checkpoint per phase: `{output_dir}/training/{phase_name}/`
- **Full-model fine-tuning, no LoRA / PEFT.** AdamW updates every parameter whose `requires_grad=True` after the freezing helper applies the substring filter from `phase_cfg.trainable_parameters`. If you want LoRA, that's a new model adapter (which would expose only the adapter weights via `requires_grad=True`).
- **Per-phase `enabled`** is the only on/off switch. There is no longer a `training.ft.enabled` master switch — set all three phase `enabled` flags to false to skip training entirely.

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
- **AraRooPat** (`araroopat`): the conservation metrics are at their architectural ceiling *for the words that take the root+pattern path* — each ROOT token's metric-string IS the root letters and each PAT token's is the cleaned inflected stem. The measured values are far below that ceiling (0.1996 root / 0.2296 pattern on ArabicText-Large) because most words do not take that path at all: RPS multiplies coverage by preservation, and coverage dominates. Use `root_conservation_attainable` (0.2321) against `root_measurable_pct` (84.12) to separate the two. **`morpheme_integrity_rate` = 1.0 and `clitic_separation_accuracy` = 1.0 are NOT the architectural ceiling** — they are an artifact of the measured population. Both require `aligned_token_offsets` to succeed, which needs the cleaned tokens to concatenate back into the word; araroopat's ROOT+PAT output never does (measured: 0 of 112 such words align, 488 of 488 LIT words align). So the two rates describe only the character-fallback path, where every boundary aligns trivially. Read them together with `morph_alignment_coverage` (0.4292) and `morph_alignment_ceiling` (0.6202). Among morphology-aware tokenizers (MorphoBPE / FarasaCharBERT / AraRooPat), break ties with downstream MCQ scores rather than the conservation metrics.

**Root extraction backends** (in `RootExtractor`, falls through in order):
1. `qalsadi.analex.Analex.check_word(word)` — proper morphological roots. **Use this API, not `Lemmatizer.lemmatize()` which returns the lemma (dictionary form), not the root.**
2. `tashaphyne.stemming.ArabicLightStemmer.get_root()` — light stemmer; ~80% accurate, used as backup.
3. Consonant-skeleton heuristic — strips diacritics and matres lectionis (ا, و, ي). Rejected if length is outside 3–4 consonants.

**Token-string normalization** (`clean_token_string`):
- Reverses the GPT-2/ByteLevel BPE byte→char mapping (necessary for HF `ByteLevel` BPE, which encodes Arabic UTF-8 bytes as Latin-1 surrogates like `Ø§ÙĦÙĥØªØ§Ø¨` for `الكتاب`). Without this, every BPE token cleans to an empty string and `root_bearing_token_pct` becomes `None`.
- Strips `##` (WordPiece), `▁` (SentencePiece), `Ġ` (ByteLevel) prefix markers.
- Removes diacritics; keeps only Arabic letters + matres lectionis.

**Diagnostics (added 2026-08-30) — the five fields that make the four headline metrics readable.** Each headline metric has a ceiling that is a property of the *sample*, not of the tokenizer; reporting the rate without its ceiling makes tokenizers look worse than they are and makes 1.0 unreachable.
- **`root_measurable_pct`** — % of sampled words whose root is a subsequence of the *unsplit* word. Weak roots are not (قال does not contain the و of قول), so no tokenizer can preserve them. Measured ~84 % on ArabicText-Large — i.e. a raw RPS of 1.0 is unreachable by construction.
- **`root_conservation_attainable`** — RPS renormalized onto that reachable population. This is the number to compare across tokenizers; 1.0 means "preserved the root everywhere it was possible to".
- **`morph_alignment_coverage`** — share of sampled words where `aligned_token_offsets` succeeded. `morpheme_integrity_rate` and `clitic_separation_accuracy` are computed **only** over these words, so a low value means those two rates describe a subset rather than the tokenizer.
- **`morph_alignment_ceiling`** — what a whole-word tokenizer would score on the same sample. Capped at ~0.62 by `clean_token_string`'s charset (`ى`, `ٱ`, embedded digits/Latin are dropped, so the word cannot reconstruct). `character_bert` and `char_jaber` both sit exactly on it; BPE reaches 0.52 (ByteLevel artifacts); araroopat 0.43.
- **`root_extractor_agreement`** — agreement between the primary (qalsadi) and secondary (tashaphyne) extractors on the same sample, ~0.80. That residual is noise in the ground truth, not in any tokenization; no tokenizer can be credited or blamed for it. Deliberately **not** sourced from CAMeL Tools: araroopat's ROOT token *is* CAMeL's root, so using CAMeL as the reference would drive its RPS to ~1.0 by construction.

The comparison report renders these in a dedicated *Morphological Metrics* section and footnotes any experiment whose `morph_alignment_coverage` is below `LOW_ALIGNMENT_COVERAGE` (0.5) — same report-and-footnote policy as `RPS_MECHANICAL_FLAGS`, never suppression.

**Sampling** is controlled by `evaluation.morph_sample_size` (default 500). The sample is deterministic (fixed seed), distinct words only, length ≥ 3 chars.

### Downstream Metrics
- **LightEval benchmarks** (`tasks/lighteval/`): accuracy via log-likelihood multiple-choice scoring on ACVA, Alghafa, Culture-Arabic-MMLU, and Arabic-Exam. See [LightEval Benchmarks](#lighteval-benchmarks-acva-alghafa-culture-arabic-mmlu-arabic-exam) below. Eval is full-benchmark — no rows reserved for SFT, since training is task-agnostic under the 3-phase pipeline.

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

**No SFT split — every benchmark row goes to evaluation.** Under the 3-phase pipeline, training is task-agnostic (Phase 3 SFT uses TyDiQA-Arabic + ARCD, not the benchmark itself), so every row of every benchmark is available for the final eval pass.

`get_eval_examples()` ([tasks/lighteval/base.py](src/arabic_eval/tasks/lighteval/base.py)) returns the full parsed list, after the optional `clean_latin_rows` filter. The previous 10/90 stratified split is gone — `_get_splits` / `train_split_ratio` / `eval_full` were removed in the 3-phase migration.

### Evaluation Methodology (LightEval)

For each multiple-choice question the `LightEvalModelWrapper` computes:

```
log P(" A" | context)  …  log P(" D" | context)
```

using the fine-tuned model's forward pass (`_compute_loglikelihood`), then predicts `argmax`. This matches LightEval's standard log-likelihood MCQ protocol exactly. The wrapper implements the same `loglikelihood(requests)` interface as LightEval's `LightevalModel`, so it can be substituted into a full LightEval pipeline if needed.

**CharacterBERT log-likelihood note**: `character_cnn` is scored using the *word-level* logits (the model's `lm_head` indexes the word vocabulary; the `char_ids` 3-D batch flows into the CharCNN, the transformer output is projected to the word vocab, and the continuation scoring sums log P(word) over the continuation's word-vocab IDs). This is implemented in the `character_cnn` branch of `_compute_loglikelihood`. The result is a real accuracy in `[0, 1]` — *not* a 0.0 fallback. **`farasa_character_bert` shares the same `embedding_type` and the same scoring path**; the only difference is that its `lm_head` indexes a *morpheme* vocabulary, so continuation scoring is over morpheme-vocab IDs rather than word-vocab IDs. Both produce real, comparable accuracies on `acva` / `alghafa` / `culture_arabic_mmlu` / `arabic_exam`.

**ACVA word-based scoring note**: ACVA is True/False, not 4-way MCQ, and its continuation pool is just `صح` / `خطأ` (rendered as letter `أ` / `ب` in vanilla LightEval). When scored with single-letter continuations, 99 % of model decisions ended up as near-tie log-likelihoods (differences < 1e-3) dominated by the per-letter unigram prior — accuracy clustered around the majority-class baseline regardless of tokenizer. ACVATask therefore overrides the default scoring hooks to score the words `" صح"` / `" خطأ"` directly: the prompt drops the `أ./ب.` choice listing and `_build_continuations` returns `[" صح", " خطأ"]`. Letter scoring is preserved for the other three benchmarks (which are genuinely 4-way MCQ). The override is implemented via two hooks on `LightEvalBenchmarkTask` (`_format_eval_context`, `_build_continuations`) — same pattern any future task can use to swap continuations without touching the wrapper. The label strings are centralised on `ACVATask.LABELS` (a tuple, single source of truth) so switching to e.g. `صحيح` / `خاطئ` is a one-line change.

**PMI normalization (`evaluation.score_normalization`)**. The wrapper supports three score-normalization modes, mirroring LightEval's `LogProbNormalization`:

  * `"char"` (default) — divide each per-continuation ll by its character length (LightEval `LogProbCharNorm`). For 1-character letter continuations this is a no-op; for word-scored continuations (ACVA, Alghafa T/F + sentiment) it removes a length bias toward shorter answers. Backward-compatible with every existing run JSON in `outputs/` — the mode is the legacy default, the metrics dict is byte-identical (modulo timestamps).
  * `"pmi"` — score = `log P(c | full_context) − log P(c | unconditioned_context)` (LightEval `LogProbPMINorm`). The unconditioned context is supplied per task via `_unconditioned_query(ex)`, which defaults to `"الإجابة:"` (the bare answer prefix every current task's prompt ends with). PMI cancels the per-continuation prior — the dominant failure mode of letter-MCQ scoring on weak-signal Arabic benchmarks. Empirically: Llama-3.2-1B's unconditional log P( letter | empty MCQ context ) spans ~1.7 nats across ج/د/ب/أ, large enough to dominate the question-conditioned signal on translated MMLU. On the no-SFT failure CSV for `culture_arabic_mmlu` (9,723 wrong rows), recomputing argmax on `ll − prior` flipped 16.8 % of rows wrong → correct (estimated PMI accuracy ≈0.37 vs char-norm 0.2447); the actual lift after the bidirectional flip-back is more modest but consistently positive across all four tasks.
  * `"char+pmi"` — compute both. Metrics dict carries `accuracy_char_norm` and `accuracy_pmi`; the legacy `accuracy` field aliases char-norm so existing comparison-report consumers keep working. Failure CSV gains `score_pmi_*` and `score_pmi_margin` columns alongside the existing `score_*` / `score_margin`.

The flag is plumbed via the existing signature-gating pattern (`inspect.signature(task.evaluate).parameters`) — non-LightEval tasks log a warning and ignore. Unconditioned ll values are cached per `(unconditioned_query, tuple(continuations))`: for letter-MCQ this is one extra forward call total (cache hits every row); for word-scored sub-configs that vary continuations per row the cache misses and we pay one extra forward per example (~2× cost on those rows; acceptable). Default stays `"char"` so existing run JSONs remain reproducible — opt in per-experiment YAML (`evaluation.score_normalization: "char+pmi"`).

**MEI under PMI**. `compute_mei` prefers `accuracy_pmi` when present and records the source as `inputs.accuracy_source = "accuracy_pmi"`; under default char-only mode the field is omitted (preserves byte-identity).

### Alghafa heterogeneity (per-topic scoring dispatch)

Alghafa is the only benchmark in the suite where one task class spans multiple MCQ shapes. The dataset has **9 sub-configs**:

| Topic class | Sub-configs | Choices | Rows | Scoring |
|---|---|---|---|---|
| 2-way T/F facts | `multiple_choice_facts_truefalse_balanced_task` | sol1-2 | 80 | **word** |
| 2-way binary sentiment | `multiple_choice_rating_sentiment_no_neutral_task` | sol1-2 | 8000 | **word** |
| 3-way sentiment | `multiple_choice_rating_sentiment_task`, `multiple_choice_sentiment_task` | sol1-3 | 6000+1725 | **word** |
| 4-way MCQ | `mcq_exams_test_ar`, `meta_ar_dialects`, `meta_ar_msa` | sol1-4 | 562+5400+900 | letter |
| 5-way grounded statement | `multiple_choice_grounded_statement_soqal_task`, `multiple_choice_grounded_statement_xglue_mlqa_task` | sol1-5 | 155+155 | letter |

The 4 binary/sentiment sub-configs use **word-scored prompts** (mirroring ACVA's fix for the letter-prior pathology — letter-based `أ`/`ب` decisions on a 2-way task collapse to the unigram letter prior). The 4-way and 5-way MCQ sub-configs keep the inherited letter-based default. Per-row dispatch keys on `ex["_source_config"]` populated by `_parse_combined`. The hooks are `_format_eval_context` / `_build_continuations` overridden in `AlghafaTask`; the dispatch list is `AlghafaTask.WORD_SCORED_CONFIGS` (single source of truth — to add a new sub-config to the word-scored set, edit that frozenset and nothing else).

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

### Arabic_Exam dataset gotchas (`MBZUAI/ArabicMMLU`)

Multi-config: 41 configs ship, but the `All` config is a strict union of the other 40 (verified `|All| = sum(|other 40|) = 14575`). Excluded via `EXCLUDED_CONFIGS = frozenset({"All"})` on `ArabicExamTask` to avoid 2× row duplication. Other parser nuances:

- **`Context` field** (~5 % of rows) supplies a passage the question refers to — must be prepended to the prompt.
- **`Option 5`** ships in ~344 rows (5-option MCQ); enumerate Options 1–5, not 1–4.
- **`Answer Key`** is a Latin letter A–E. Map back to 0-indexed integer.
- **`is_few_shot=1` rows** (~120) are dev-split demonstrations — filter them out.

After exclusion + filter, the eval pool is ~13,000 rows. ~235 question strings still appear in 2+ subject configs — these are inter-subject overlaps inherent to the dataset's taxonomy, not a merge artifact. Tests: [tests/test_arabic_exam_parser.py](tests/test_arabic_exam_parser.py).

### Reference experiments

The two canonical experiments under the 3-phase pipeline:

- **`configs/experiments/native_llama_3phase_with_sft.yaml`** — full pipeline (Phase 1 + Phase 2 + Phase 3). Reference for "what is the trained-model accuracy on ACVA / Alghafa / arabic_exam / culture_arabic_mmlu under task-agnostic SFT."
- **`configs/experiments/native_llama_3phase_no_sft.yaml`** — Phase 1 + Phase 2 only (`sft.enabled: false`). Isolates Phase 3's contribution: the (with_sft − no_sft) delta per benchmark.

Results land in `outputs/experiments/native_llama_3phase_{with,no}_sft/all_metrics.json`. When new tokenizers are added to the suite, run them through the same two configs (overriding `tokenizer.type`) for an apples-to-apples comparison against native_llama.

### Key Classes (`src/arabic_eval/tasks/lighteval/`)

| Symbol | Module | Role |
|---|---|---|
| `LightEvalBenchmarkTask` | `lighteval/base.py` | Abstract base — 7 abstract hooks; concrete `get_eval_examples()` + `evaluate()` |
| `LightEvalModelWrapper`  | `lighteval/base.py` | Wraps `BaseModelAdapter` for LightEval's `loglikelihood` interface |
| `_compute_loglikelihood` | `lighteval/base.py` | Core per-token log-likelihood sum (LightEval methodology) |
| `format_mcq_context`     | `lighteval/utils.py` | Formats question + choices as LightEval context string (opt-in) |
| `format_mcq_full`        | `lighteval/utils.py` | Formats complete MCQ + answer (opt-in helper) |
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

**Arabic_Exam schema (`MBZUAI/ArabicMMLU`)** is non-standard — see *Arabic_Exam dataset gotchas* above for the full rundown.

### Config Overrides

Dataset paths are configurable per-task via the `params` dict:

```yaml
sweep:
  tasks:
    - type: "acva"          # or alghafa | culture_arabic_mmlu | arabic_exam
      params:
        dataset_name: "OALL/ACVA"
        dataset_config: null   # set to a specific subtask/config if needed
        max_length: 512
        seed: 42
        clean_latin_rows: false   # filter Latin-script rows before eval
```

> **Note:** Confirmed Hub defaults are: `acva` → `OALL/ACVA`; `alghafa` → `OALL/AlGhafa-Arabic-LLM-Benchmark-Native`; `culture_arabic_mmlu` → `OALL/Arabic_MMLU`; `arabic_exam` → `MBZUAI/ArabicMMLU`. Override via `params.dataset_name` if a benchmark moves on the Hub.

## Config Reference

The full menu of every parameter (with defaults and comments) lives in [configs/experiments/sample_full.yaml](configs/experiments/sample_full.yaml). Real experiment YAMLs only need to override deltas — `configs/base.yaml` provides defaults for every field, including the 3-phase training block.

### Minimal experiment YAML (the with-SFT reference)

```yaml
experiment:
  name: "native_llama_3phase_with_sft"
  output_dir: "outputs/experiments/native_llama_3phase_with_sft"
  seed: 42

tokenizer:
  type: "native_llama"
  vocab_size: null
  load_path: null
  save_path: "outputs/tokenizers/native_llama"

model:
  type: "llama"
  name_or_path: "meta-llama/Llama-3.2-1B"
  dtype: "bfloat16"
  device: "auto"

# All training.phases.* defaults come from configs/base.yaml.

sweep:
  tokenizers:
    - type: "native_llama"
      vocab_sizes: [null]
  tasks:
    - type: "acva"
      params: {}
    - type: "alghafa"
      params: {}
    - type: "arabic_exam"
      params: {}

evaluation:
  intrinsic_metrics: true
  morphological_metrics: true
  morph_sample_size: 500
  downstream_metrics: true
  failure_reports: true
  intrinsic_unk_report: false   # dumps per-word UNK list to intrinsic_unks.csv
  downstream_unk_report: false  # dumps per-task UNK list to unk_reports/<task>_unks.csv
  score_normalization: "char+pmi"
  num_eval_samples: null
```

The "without SFT" variant adds one override:

```yaml
training:
  phases:
    sft:
      enabled: false       # Phase 3 skipped; Phase 1 + Phase 2 still run
```

### Phase block schema (in `training.phases.<phase>`)

Every phase shares the same fields; SFT additionally has `early_stopping`. See `PhaseConfig` / `EarlyStoppingConfig` in [src/arabic_eval/config.py](src/arabic_eval/config.py).

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | per-phase toggle |
| `datasets` | list of `DatasetName` | registry keys: `arabic_squad` \| `tydiqa_arabic` \| `arcd`. Single string is auto-coerced to a one-element list. |
| `trainable_parameters` | list of substrings | matched against `named_parameters()`. `["*"]` = all. Mixing `"*"` with other entries is rejected. |
| `steps`, `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `weight_decay`, `max_length`, `warmup_steps`, `max_grad_norm` | scalars | per-phase numeric params |
| `optimizer` | `"adamw"` | only AdamW supported |
| `lr_scheduler` | `"cosine"` \| `"constant"` \| `"linear"` | `constant` ignores `warmup_steps` (still has linear warmup but no decay after) |
| `loss_target` | `"full_sequence"` \| `"answer_only"` | full-seq for Phase 1; answer-only for Phase 2/3 |
| `save_checkpoint` | bool | writes to `{output_dir}/training/{phase}/` |
| `early_stopping` | nested config (SFT only) | required when `sft.enabled=true`; see below |

### Early-stopping schema (`training.phases.sft.early_stopping`)

| Field | Default | Notes |
|---|---|---|
| `enabled` | `true` | turn off to disable mid-Phase-3 eval entirely |
| `metric` | `"eval_loss"` | only `eval_loss` supported (answer-only causal LM loss on TyDiQA-val + ARCD-val) |
| `eval_every_n_steps` | `200` | how often to run the eval pass |
| `patience` | `5` | stop after this many consecutive non-improving evals |
| `min_delta` | `5e-4` | absolute improvement threshold |
| `min_steps_before_stop` | `500` | don't allow stop in the early LR-warmup region |
| `restore_best_at_end` | `true` | snapshot best, restore at end |
| `eval_splits` | `{tydiqa_arabic: validation, arcd: validation}` | which split per corpus |

### Sweep YAML structure (multiple tokenizer cells, shared eval task list)

```yaml
sweep:
  tokenizers:
    - type: "bpe"
      vocab_sizes: [16000, 32000, 50000]
    - type: "character_bert"
      vocab_sizes: [null]        # N/A for char-level
    - type: "native_llama"
      vocab_sizes: [null]
  tasks:
    - type: "acva"
      params: {}
    - type: "alghafa"
      params: {}
    - type: "arabic_exam"
      params: {}
    - type: "culture_arabic_mmlu"
      params: {}
```

Training happens once per (tokenizer_type, vocab_size) cell. Eval iterates over `sweep.tasks` per cell — each cell produces one subdirectory under `output_dir/`, each with its own `all_metrics.json` containing per-task downstream + per-task MEI.

## Output Structure

Each experiment produces:
```
outputs/experiments/<name>/
  config.json               # Full resolved config
  intrinsic_metrics.json    # Fertility, compression, UNK rate, coverage, morphological metrics
  all_metrics.json          # Combined: config + intrinsic + training (per-phase) + downstream (per-task) + mei (per-task)
  training/
    embedding_alignment/    # Phase 1 checkpoint
      model.pt
    warmup/                 # Phase 2 checkpoint
      model.pt
    sft/                    # Phase 3 checkpoint (best-by-eval-loss, restored)
      model.pt
  failure_reports/          # Only when evaluation.failure_reports=true
    <task_name>_accuracy_failures.csv   # one CSV per LightEval MCQ task
  intrinsic_unks.csv        # Only when evaluation.intrinsic_unk_report=true
  unk_reports/              # Only when evaluation.downstream_unk_report=true
    <task_name>_unks.csv    # one CSV per LightEval MCQ task
```

Per-phase histories (final loss, train-loss tail, eval losses, wall time, early-stop status, checkpoint path) live under `all_metrics.json["training"][<phase>]`. Per-task downstream metrics + MEI live under `all_metrics.json["downstream"][<task>]` and `all_metrics.json["mei"][<task>]`.

Sweep mode additionally generates: `comparison_report.txt` and `comparison_report.json` in the sweep output directory. The text report includes a "Composite Metric: MEI" section with per-experiment rows + an asterisk-and-footnote on tokenizers with mechanical RPS extremes (`RPS_MECHANICAL_FLAGS` in `evaluation/reporter.py`). For multi-sub-config benchmarks (Alghafa), a "Per-sub-config breakdown" sub-section per task is emitted under the main downstream-task table — rows = experiments, columns = sub-configs, cells = `accuracy (n=N)`. Single-config benchmarks suppress this section; the main table strips `per_subconfig_accuracy.*` so its column count stays manageable.

### Failure-case CSV reports (opt-in)

When `evaluation.failure_reports: true`, LightEval MCQ tasks (`acva`, `alghafa`, `culture_arabic_mmlu`, `arabic_exam`) write one CSV per task with one row per wrong-answer example. Columns: `index, question, choice_0..N, gold_idx/letter, pred_idx/letter, ll_0..N, ll_margin, score_0..N, score_margin`. Both raw and aggregated views are persisted: `ll_*` is the raw model log-likelihood (sum over continuation tokens) and `ll_margin = ll_pred − ll_gold`; `score_*` is the value passed to `argmax` after `_aggregate_scores` (default char-norm, LightEval `LogProbCharNorm` equivalent), with `score_margin` defined the same way. They differ only when continuation lengths differ — letter-scored MCQ rows have `ll_* == score_*` (1-char continuations); ACVA and word-scored Alghafa rows have `score_*` divided by the character count of the answer text. The margins distinguish confident-wrong (large positive) from near-tie failures (~0); use `score_margin` to interpret the model's actual decision and `ll_margin` to debug the unnormalized signal. UTF-8-BOM encoding so Excel renders Arabic correctly.

To opt a new task family into failure reporting, just add `failure_report_dir: Optional[Path] = None` to its `evaluate()` signature and handle it — no base-class change needed (signature gating in the pipeline picks it up automatically).

### UNK occurrence CSV reports (opt-in)

Two independent flags surface the per-word data underlying UNK detection, mirroring the failure-CSV pattern:

* **`evaluation.intrinsic_unk_report: true`** — during intrinsic evaluation, every whitespace-split word of the eval split is encoded individually (same loop that produces the scalar `unk_rate`). Words whose per-word encoding contains at least one UNK token id are aggregated into `<output_dir>/intrinsic_unks.csv` with columns `word, unk_token_count, total_token_count, example_context`. Sorted descending by `unk_token_count`. The scalar `unk_rate` and `vocab_coverage` are byte-identical with vs. without the flag — the CSV is a side-channel, not a recomputation.
* **`evaluation.downstream_unk_report: true`** — during LightEval MCQ evaluation, every prompt (via `_format_eval_context_with_fewshot`) and every continuation per example is scanned. Per-task output `<output_dir>/unk_reports/<task_name>_unks.csv` with columns `word, unk_token_count, source_fields, num_examples_seen_in, example_context`. `source_fields` is a pipe-joined sorted set (`"continuation_0|prompt"`); `num_examples_seen_in` counts each eval row at most once even if the word recurs across fields.

Both reports always write a CSV when the flag is on — even header-only when no UNK was seen, or when the tokenizer's `special_tokens` dict has no `unk_token` entry (byte-level Charformer; Llama under byte-fallback). The header-only file is intentional so every (tokenizer, task) pair in a sweep produces a comparable artifact. UTF-8-BOM encoding so Arabic columns render in Excel.

Helper module: [src/arabic_eval/evaluation/unk_reports.py](src/arabic_eval/evaluation/unk_reports.py) exposes `scan_text` / `aggregate_occurrences` / `records_to_rows` plus the two fieldname constants (`INTRINSIC_UNK_FIELDS`, `DOWNSTREAM_UNK_FIELDS`). Downstream wiring uses the existing signature-gating pattern (`unk_report_dir: Optional[Path] = None` kwarg on `LightEvalBenchmarkTask.evaluate`).

## Dependencies

Core: `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate`, `farasapy`, `pydantic`, `pyyaml`, `numpy`, `tqdm`, `wandb`, `tabulate`, `matplotlib`, `lighteval>=0.6.0`

Tokenizer-only workflows (no GPU): `pydantic`, `pyyaml`, `tokenizers`, `tabulate`, `numpy`, `tqdm`

Optional `[morphological]` extras for full Arabic morphological metrics: `qalsadi`, `pyarabic` (Tashaphyne is pulled in transitively by qalsadi). Install via `pip install -e ".[morphological]"`. Without these, `RootExtractor` falls back to a consonant-skeleton heuristic and `morpheme_integrity_rate` still works via Farasa.

## Known Considerations

- LLaMA 3.2-1B requires HuggingFace access token (gated model). Set `HF_TOKEN` env var or `huggingface-cli login`.
- LLaMA 3.2-1B uses **tied embeddings** — `lm_head.weight is model.embed_tokens.weight`. Phase 1's `trainable_parameters: ["embed_tokens", "lm_head"]` is correct; the freezing helper warns when `lm_head` matches no parameter (tied case) and continues. Training `embed_tokens` IS training `lm_head`.
- MorphoBPE (`morpho_bpe`) and FarasaCharacterBERT (`farasa_character_bert`) require Java runtime for Farasa. The segmenter runs in interactive mode for efficiency.
- AraRooPat (`araroopat`) requires `camel-tools` in its own `.venv-camel` (subprocess bridge to avoid `numpy<2` / `transformers<4.54` conflict). One-time setup: `python -m venv .venv-camel && .venv-camel/bin/pip install -e ".[araroopat-camel]" && .venv-camel/bin/camel_data -i light`.
- Character-level tokenizers (char-JABER) produce very long sequences; reduce `max_length` or `batch_size` per phase if hitting OOM.
- The `models/__init__.py` and `tasks/__init__.py` use try/except on imports, so model and task registries will be empty if torch/transformers are not installed. The pipeline script (`pipeline/experiment.py`) force-imports them, so missing deps will surface at experiment runtime.
- `torch.amp.autocast` in `phases.py` passes `device_type=adapter.device.type` — works for `cuda`, `cpu`, and most accelerators that PyTorch supports; no extra adjustment needed for MPS.
- CharacterBERT (`character_cnn`) and FarasaCharacterBERT support LightEval log-likelihood scoring via the `character_cnn` branch in `_compute_loglikelihood` (real accuracy in `[0, 1]`, not 0.0). The shared remaining limitation is autoregressive `generate()`, which `LlamaAdapter.generate()` raises `NotImplementedError` for on `CHARACTER_CNN` and `CHARFORMER` — but generation is not exercised by the 3-phase pipeline (eval is teacher-forced log-likelihood MCQ).
- The `dataset_name` defaults (per task class via `_default_dataset_name()`): `acva` → `OALL/ACVA`; `alghafa` → `OALL/AlGhafa-Arabic-LLM-Benchmark-Native`; `culture_arabic_mmlu` → `OALL/Arabic_MMLU`; `arabic_exam` → `MBZUAI/ArabicMMLU`. Override via `params.dataset_name` if a benchmark moves on the Hub.
- The Phase 1 + 2 corpus (`Mostafa3zazi/Arabic_SQuAD`) has **train-only** (no validation split). That's fine — Phases 1 and 2 don't run eval mid-phase. Phase 3 uses TyDiQA-Arabic-val + ARCD-val for stagnation early-stop.
- Full-step experiment wall time on H100: ~5h with SFT (Phase 1 + 2 + 3 + 4 benchmark evals), ~1.5h without SFT (Phase 1 + 2 only + 4 evals).
