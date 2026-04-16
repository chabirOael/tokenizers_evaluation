# Arabic Tokenizers Evaluation Platform (`arabic-eval`)

Universal platform for evaluating **Arabic tokenizers** by measuring downstream LLM performance while holding the rest of the experimental setup fixed (dataset, model family, training loop, evaluation).

The pipeline:

- Train (or load) a tokenizer from scratch
- Compute **intrinsic tokenizer metrics** (fertility, compression ratio, UNK rate, vocab coverage)
- Load a causal LM (default: LLaMA) and **adapt embeddings/output head** to the tokenizer
- Fine-tune on a downstream task
- Evaluate and write results + a sweep comparison report

## What’s in this repo

- **Package**: `src/arabic_eval/`
- **Configs**: `configs/` (base defaults + experiments + per-tokenizer/model/task YAMLs)
- **CLIs**: `scripts/`
  - `train_tokenizer.py`
  - `run_experiment.py`
  - `evaluate_intrinsic.py`
  - `compare_results.py`
- **Outputs (gitignored)**: `outputs/` (tokenizers, experiments, cached datasets, reports)

## Requirements

- **Python**: >= 3.10
- **GPU**: required for model fine-tuning/evaluation (tokenizer-only workflows can run CPU-only)
- **HuggingFace access**:
  - Datasets are loaded via `datasets.load_dataset(...)`
  - The default model is `meta-llama/Llama-3.2-1B` (may be gated). You may need `HF_TOKEN`.
- **Java runtime**: required only for `morpho_bpe` (Farasa segmentation via `farasapy`)

## Install

From the repo root:

```bash
export HF_TOKEN="hf_..."
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional dev tools:

```bash
pip install -e ".[dev]"
```

If you already have a virtual environment, just activate it instead. If `.venv` already exists, you can skip the `python3 -m venv .venv` step.

### (Optional) Run tests in tmux

If you’re on a remote machine, running tests inside `tmux` keeps them running if your SSH session drops.

```bash
tmux new -s arabic-eval-tests
source .venv/bin/activate
pytest -q
```

Later, re-attach with:

```bash
tmux attach -t arabic-eval-tests
```

If you only want tokenizer training / intrinsic metrics (no torch/transformers), you can install a minimal set instead of `-e .`:

```bash
pip install pydantic pyyaml tokenizers tabulate numpy tqdm datasets
```

## Quickstart

### Train a tokenizer

```bash
python scripts/train_tokenizer.py --type bpe --vocab-size 32000
```

Examples:

```bash
python scripts/train_tokenizer.py --type wordpiece --vocab-size 16000
python scripts/train_tokenizer.py --type morpho_bpe --vocab-size 32000
python scripts/train_tokenizer.py --type character_bert
python scripts/train_tokenizer.py --type char_jaber
```

Saved tokenizers default to:

- `outputs/tokenizers/<type>_<vocab>` for subword tokenizers (e.g. `outputs/tokenizers/bpe_32k`)
- `outputs/tokenizers/<type>` for char-level tokenizers (e.g. `outputs/tokenizers/char_jaber`)

### Run a single end-to-end experiment

```bash
python scripts/run_experiment.py --config configs/experiments/bpe_32k_generation.yaml
```

This writes experiment artifacts to the `output_dir` specified in the YAML (and logs to `experiment.log` inside that directory).

### Run a sweep (all combinations)

```bash
python scripts/run_experiment.py --config configs/experiments/full_sweep.yaml --sweep
```

At the end of a sweep, you’ll get:

- `outputs/experiments/full_sweep/comparison_report.txt`
- `outputs/experiments/full_sweep/comparison_report.json`

### Intrinsic-only evaluation for a saved tokenizer

```bash
python scripts/evaluate_intrinsic.py --tokenizer-path outputs/tokenizers/bpe_32k --type bpe --num-samples 5000
```

### Compare multiple experiment directories

```bash
python scripts/compare_results.py outputs/experiments/*/
```

## Configuration

Experiments are driven by YAML. `scripts/run_experiment.py` loads:

- a base config (defaults to `configs/base.yaml` if present)
- then overlays your experiment config (e.g. `configs/experiments/bpe_32k_generation.yaml`)
- then applies optional CLI overrides (`--seed`, `--device`)

The config schema is defined in `src/arabic_eval/config.py` (`ExperimentConfig`).

### Key top-level fields

- **`name`**: experiment name
- **`description`**: free text
- **`output_dir`**: where to write artifacts
- **`seed`** / **`deterministic`**

You may also nest those under an `experiment:` block in YAML; the loader flattens it automatically.

### Data (`data:`)

Defaults (from `configs/base.yaml`):

- **`dataset_name`**: `Jr23xd23/ArabicText-Large`
- **`cache_dir`**: `outputs/data_cache`
- **`max_train_samples`**, **`max_eval_samples`**
- **`preprocessing`**:
  - `normalize_unicode`
  - `remove_diacritics`
  - `min_text_length`

### Tokenizer (`tokenizer:`)

- **`type`**: registry key (see “Tokenizers”)
- **`vocab_size`**: ignored for character-level tokenizers
- **`params`**: forwarded to the tokenizer constructor / `train(...)`
- **`save_path`**: where to save the trained tokenizer
- **`load_path`**: if set, skips training and loads from disk

### Model (`model:`)

- **`type`**: model adapter registry key (default: `llama`)
- **`name_or_path`**: HF model id or local path (default: `meta-llama/Llama-3.2-1B`)
- **`dtype`**: `float32|float16|bfloat16`
- **`device`**: `auto|cpu|cuda:0|...`
- **`params`**: extra kwargs passed to the adapter

### Task (`task:`)

- **`type`**: task registry key (see “Tasks”)
- **`params`**: task-specific params (e.g. `max_length`, `stride`, `dataset_name`)

### Training (`training:`)

Includes epochs, batch size, gradient accumulation, AdamW params, scheduler, clipping, mixed precision flags, checkpoint cadence, early stopping config.

### Evaluation (`evaluation:`)

- **`intrinsic_metrics`**: run tokenizer intrinsic eval
- **`downstream_metrics`**: run downstream evaluation
- **`num_eval_samples`**: number of texts/examples to evaluate on
- generation defaults used by some tasks: `generation_max_new_tokens`, `generation_temperature`, `generation_do_sample`

### Tracking (`tracking:`)

- **`use_wandb`**: enable Weights & Biases logging
- **`wandb_project`**, **`wandb_entity`**
- **`log_to_file`**

## Tokenizers

Tokenizers live in `src/arabic_eval/tokenizers/` and register themselves via `tokenizer_registry`.

Available registry keys:

- **`bpe`**: subword BPE (standard embedding)
- **`wordpiece`**: WordPiece (standard embedding)
- **`morpho_bpe`**: Farasa morphological segmentation + BPE (**requires Java**, standard embedding)
- **`character_bert`**: word-level tokens + per-word character IDs (**CharacterCNN embedding**)
- **`char_jaber`**: character-level tokenizer (**character embedding**, sequences are much longer)

### Embedding integration modes

Each tokenizer declares an `embedding_type` consumed by `LlamaAdapter.adapt_to_tokenizer(...)`:

- **`standard`**: resize `nn.Embedding` / output head to tokenizer vocab size
- **`character_cnn`**: replace embedding with a CharCNN; output head is over a **word vocabulary**
- **`char_jaber`**: replace embedding with a character embedding; output head is over the **char vocabulary**

## Tasks

Tasks live in `src/arabic_eval/tasks/` and register via `task_registry`.

- **`text_generation`**: perplexity evaluation with a sliding window (`max_length`, `stride`)
- **`question_answering`**: Arabic QA framed as generation with the prompt:
  - `السياق: ...`
  - `السؤال: ...`
  - `الإجابة:`

## Outputs

### Tokenizers

Saved under `outputs/tokenizers/...` (directory contains a `tokenizer.json` or equivalent).

### Experiments

For a run with `output_dir = outputs/experiments/<name>`, the pipeline writes:

- **`config.json`**: resolved config
- **`experiment.log`**: run log
- **`all_metrics.json`**: combined intrinsic + downstream metrics
- **`training/`**: training artifacts/checkpoints (created by the trainer)

Sweep runs create per-combination subdirectories under the sweep `output_dir` and a top-level comparison report.

## Extending the platform

The project uses a registry pattern (`src/arabic_eval/registry.py`):

- `tokenizer_registry`
- `model_registry`
- `task_registry`

### Add a new tokenizer

1. Create `src/arabic_eval/tokenizers/my_tokenizer.py`
2. Implement `BaseTokenizer` (`src/arabic_eval/tokenizers/base.py`)
3. Register it:

```python
from arabic_eval.registry import tokenizer_registry

@tokenizer_registry.register("my_tokenizer")
class MyTokenizer(...):
    ...
```

4. Import it in `src/arabic_eval/tokenizers/__init__.py` so it registers at import time
5. If it needs a new embedding integration mode, add an embedding module under `src/arabic_eval/models/embeddings/` and update `LlamaAdapter.adapt_to_tokenizer(...)`

### Add a new task

1. Create `src/arabic_eval/tasks/my_task.py`
2. Implement `BaseTask`
3. Register with `@task_registry.register("my_task")`
4. Add to `src/arabic_eval/tasks/__init__.py`

### Add a new model adapter

1. Create `src/arabic_eval/models/my_adapter.py`
2. Implement `BaseModelAdapter`
3. Register with `@model_registry.register("my_model")`
4. Import it from `src/arabic_eval/models/__init__.py` (guard imports if you want tokenizer-only installs to still work)

## Troubleshooting

- **HuggingFace gated model (LLaMA)**: set `HF_TOKEN` (or run `huggingface-cli login`) and ensure you have access to the model id in `model.name_or_path`.
- **Farasa / `morpho_bpe` fails**: install Java (JRE/JDK) so `farasapy` can run the segmenter.
- **OOM on `char_jaber`**: sequences are much longer; reduce `task.params.max_length`, reduce `training.batch_size`, increase gradient accumulation, or run on a larger GPU.
- **CharacterBERT generation**: `LlamaAdapter.generate(...)` raises `NotImplementedError` for `character_cnn` embedding type, so QA evaluation will fall back to empty predictions (F1/EM will be poor). Use tasks/metrics that do not require generation, or extend generation support for that mode.

## License

Add your license information here (none specified yet).
