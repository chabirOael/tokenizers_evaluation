# Arabic Tokenizers Evaluation Platform

## Project Overview

A universal platform for evaluating Arabic tokenizers by measuring LLM downstream performance. All external parameters (dataset, model architecture, training hyperparameters) are held fixed; only the tokenizer changes between experiments. Tokenizers are trained from scratch on the same Arabic dataset, integrated into the same LLM (with embedding layer replacement), fine-tuned, and evaluated on the same downstream tasks.

- **Primary dataset**: `Jr23xd23/ArabicText-Large` (HuggingFace)
- **Primary LLM**: LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`)
- **Downstream tasks**: Text Generation (perplexity), Question Answering (F1/EM on ARCD), and four LightEval benchmarks (ACVA, Alghafa, Culture-Arabic-MMLU, Arabic-Exam — accuracy via log-likelihood scoring)
- **Vocab sizes tested**: 16K, 32K, 50K for subword tokenizers; fixed char vocab for character-level

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

The dispatch happens in `LlamaAdapter.adapt_to_tokenizer()` (`src/arabic_eval/models/llama_adapter.py:61`).

### Collation Dispatch (`src/arabic_eval/data/collation.py`)

`get_collator(embedding_type)` returns the right collator:
- `StandardCollator`: pads 1D `input_ids`, builds `attention_mask` and `labels`
- `CharacterCNNCollator`: pads 3D `char_ids` tensors (`[batch, words, chars]`)
- `CharJaberCollator`: pads 1D char ID sequences (longer max_length=2048)

### Configuration System (`src/arabic_eval/config.py`)

Layered YAML with Pydantic validation:
1. `configs/base.yaml` — shared defaults
2. Experiment YAML overlaid on top (deep merge)
3. CLI overrides on top of that

`ExperimentConfig` is the top-level Pydantic model with nested `DataConfig`, `TokenizerConfig`, `ModelConfig`, `TaskConfig`, `TrainingConfig`, `EvaluationConfig`, `TrackingConfig`, and optional `SweepConfig`.

Key: experiment YAML files can nest top-level fields under `experiment:` key — the loader flattens this automatically.

## The 5 Tokenizers

| # | Name | Registry Key | File | Embedding | Notes |
|---|---|---|---|---|---|
| 1 | BPE | `bpe` | `tokenizers/bpe.py` | standard | HF `tokenizers` BpeTrainer, ByteLevel pre-tokenizer |
| 2 | WordPiece | `wordpiece` | `tokenizers/wordpiece.py` | standard | HF `tokenizers` WordPieceTrainer, Whitespace pre-tokenizer |
| 3 | Morphological BPE | `morpho_bpe` | `tokenizers/morpho_bpe.py` | standard | Farasa segmentation first, then BPE on morphemes. Requires Java. |
| 4 | CharacterBERT | `character_bert` | `tokenizers/character_bert.py` | character_cnn | Word-level split, each word -> fixed-length char ID vector. Builds both char vocab and word vocab (for output head). |
| 5 | char-JABER | `char_jaber` | `tokenizers/char_jaber.py` | char_jaber | Each character is a token. Small fixed vocab. Sequences ~4-6x longer. |

All implement `BaseTokenizer` (`tokenizers/base.py`): `train()`, `encode()`, `decode()`, `save()`, `load()`, `vocab_size`, `embedding_type`, `special_tokens`, `get_embedding_config()`.

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
- **Fertility**: avg tokens per whitespace word
- **Compression ratio**: avg characters per token
- **UNK rate**: fraction of tokens that are `<unk>`
- **Vocab coverage**: fraction of unique words with no UNK tokens
- **Avg token count**: avg tokens per text

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

## Known Considerations

- LLaMA 3.2-1B requires HuggingFace access token (gated model). Set `HF_TOKEN` env var or `huggingface-cli login`.
- MorphoBPE (`morpho_bpe`) requires Java runtime for Farasa. The segmenter runs in interactive mode for efficiency.
- Character-level tokenizers (char-JABER) produce very long sequences; reduce `max_length` or `batch_size` if hitting OOM.
- The `models/__init__.py` and `tasks/__init__.py` use try/except on imports, so model and task registries will be empty if torch/transformers are not installed. The pipeline script (`pipeline/experiment.py`) force-imports them, so missing deps will surface at experiment runtime.
- `torch.cuda.amp.autocast` in the trainer uses `device_type="cuda"` — will need adjustment for non-CUDA accelerators (e.g., MPS).
- LightEval benchmark tasks (`acva`, `alghafa`, `culture_arabic_mmlu`, `arabic_exam`) fine-tune on the **10 % split only**; the 90 % eval split is never used during training. The `get_dataloader(split="test")` call from the Trainer also returns the 10 % split to avoid contamination.
- CharacterBERT (`character_cnn`) returns `accuracy=0.0` on LightEval benchmarks because log-likelihood scoring requires token-level logits over a standard vocabulary — not supported by the word-level CharCNN architecture.
- The `dataset_name` defaults for `culture_arabic_mmlu` (`acmc/arabic_culture_mmlu`) and `arabic_exam` (`arabic_exam`) are best-effort; confirm the exact HuggingFace Hub paths before running and update the YAML configs accordingly.
