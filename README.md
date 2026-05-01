# Arabic Tokenizers Evaluation Platform (`arabic-eval`)

Universal platform for evaluating **Arabic tokenizers** by measuring downstream LLM performance while holding the rest of the experimental setup fixed (dataset, model family, training loop, evaluation).

The pipeline:

- Train (or load) a tokenizer from scratch
- Compute **intrinsic tokenizer metrics**:
  - *Size / coverage* — fertility, compression ratio, UNK rate, vocab coverage, avg token count
  - *Arabic morphological* — root conservation, pattern (wazn) conservation, Farasa morpheme integrity, root-bearing & pattern-bearing token percentages
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
- **Java runtime**: required for `morpho_bpe` *and* for the `morpheme_integrity_rate` morphological metric (Farasa segmentation via `farasapy`). Other intrinsic metrics work without Java.

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

Optional **morphological metric backends** (recommended for the Arabic root/pattern metrics — without them, root extraction falls back to a consonant-skeleton heuristic):

```bash
pip install -e ".[morphological]"
```

This installs `qalsadi` (proper morphological root extraction via `Analex.check_word`) and `pyarabic`. Tashaphyne is pulled in transitively as a secondary backend.

If you already have a virtual environment, just activate it instead. If `.venv` already exists, you can skip the `python3 -m venv .venv` step.

### Optional: AraRooPat tokenizer (separate `.venv-camel`)

The `araroopat` tokenizer needs CAMeL Tools, which pins `numpy<2` and `transformers<4.54` — incompatible with `lighteval>=0.11`. To keep the main env lighteval-compatible, CAMeL runs in an **isolated venv** and the main process talks to it via a stdin/stdout NDJSON subprocess bridge.

Set up `.venv-camel` once (this does **not** affect your main `.venv`):

```bash
python3 -m venv .venv-camel
.venv-camel/bin/pip install -e ".[araroopat-camel]"
.venv-camel/bin/camel_data -i light
```

The `light` data bundle (~80MB) installs `disambig-mle-calima-msa-r13` + `morphology-db-msa-r13` and is enough for the analyzer + disambiguator + generator. Data is cached under `~/.camel_tools/` (shared across venvs).

The bridge auto-discovers `<repo_root>/.venv-camel/bin/python`. To point at a different interpreter, set:

```bash
export ARAROOPAT_CAMEL_PYTHON=/path/to/python   # interpreter must have camel-tools installed
```

If `araroopat` is invoked and neither path resolves to a valid interpreter, you'll get a `CamelBridgeError` with the exact setup commands above. There is **no silent fallback** — using `araroopat` without CAMeL is a configuration error.

## Testing

The test suite lives in `tests/`. It covers every tokenizer through the complete sweep pipeline — no internet access, GPU, or Java runtime is required.

```bash
pip install -e ".[dev]"   # installs pytest
pytest tests/ -v
```

### What’s covered

| Suite | Tests | Description |
|---|---|---|
| `TestTokenizerUnit` | 25 | `train`, `encode`, `decode`, `save`/`load` roundtrip, special tokens, intrinsic metrics |
| `TestCharacterBERTRegression` | 3 | EOW never truncated, BOS/EOS char IDs are distinct, `_next_word_id` restored on load |
| `TestCollation` | 5 | Correct batch tensor shapes and label presence per embedding type |
| `TestModelAdaptation` | 10 | Correct embedding module type after adaptation, no crash |
| `TestForwardPass` | 5 | Finite scalar loss for every tokenizer |
| `test_e2e_full_sweep` | 10 | `run_single_experiment` end-to-end for every (tokenizer × task) combination |

**Total: 63 tests, ~5 s.**

The full sweep matrix under test:

- **Tokenizers**: `bpe`, `wordpiece`, `morpho_bpe`, `character_bert`, `char_jaber`
- **Tasks**: `text_generation`, `question_answering`

### How it works without real infrastructure

- **Dataset**: `load_dataset` is patched in every module that imports it; a 12-text synthetic Arabic corpus is returned instead.
- **LLM**: `AutoModelForCausalLM.from_pretrained` is patched to return a 32-hidden-unit CPU model. The model has the same attribute structure as LLaMA (`model.embed_tokens`, `model.layers`, `model.norm`, `lm_head`) so all embedding-replacement logic is exercised exactly as in production.
- **Farasa (morpho_bpe)**: the Farasa segmenter is patched with a pass-through mock, so `morpho_bpe` exercises the full BPE-on-segmented-text path without Java.

### (Optional) Run tests in tmux

If you’re on a remote machine, running tests inside `tmux` keeps them running if your SSH session drops.

```bash
tmux new -s arabic-eval-tests
source .venv/bin/activate
pytest -v
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

- **`intrinsic_metrics`**: run tokenizer intrinsic eval (size + coverage)
- **`morphological_metrics`**: also run the Arabic morphological metrics (root / pattern / morpheme). Default: `true`.
- **`morph_sample_size`**: number of distinct words to sample for the morphological metrics. Default: `500`.
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
- **`acva`**, **`alghafa`**, **`culture_arabic_mmlu`**, **`arabic_exam`**: LightEval log-likelihood MCQ benchmarks. Each fine-tunes on its 10 % SFT split and evaluates on the 90 % held-out split.

## Intrinsic metrics

`compute_intrinsic_metrics()` (in `src/arabic_eval/tokenizers/intrinsic_metrics.py`) reports two groups of metrics, written to `intrinsic_metrics.json` (and merged into `all_metrics.json`).

### Size / coverage

| Metric | Meaning |
|---|---|
| `fertility` | avg tokens per whitespace word (lower = more efficient) |
| `compression_ratio` | avg characters per token |
| `unk_rate` | fraction of all tokens that are `<unk>` |
| `vocab_coverage` | fraction of unique words with no `<unk>` token |
| `avg_token_count` | avg tokens per text |
| `vocab_size` | tokenizer vocabulary size |

### Arabic morphological

These quantify how well a tokenizer respects Arabic root-and-pattern morphology. Sampled over `evaluation.morph_sample_size` distinct words (default 500). See [CLAUDE.md](CLAUDE.md) for definitions and the architectural-extremes caveat.

| Metric | Meaning |
|---|---|
| `root_conservation_rate` | % of words whose root letters all fall within a single token (root not split) |
| `pattern_conservation_rate` | % of words whose stem-span pattern (wazn — clitics trimmed) is recoverable from a single token |
| `morpheme_integrity_rate` | % of Farasa internal morpheme boundaries (e.g. `و\|ال\|كتاب`) that align with token boundaries |
| `root_bearing_token_pct` | % of tokens (across the sample) that contain at least one full root |
| `pattern_bearing_token_pct` | % of tokens whose stem span matches a known pattern |

**Backends** (best to worst, with automatic fallback):

1. `qalsadi.analex.Analex.check_word` — proper morphological roots (recommended; install via the `[morphological]` extras).
2. `tashaphyne.stemming.ArabicLightStemmer` — light stemmer; pulled in transitively by `qalsadi`.
3. Consonant-skeleton heuristic — strips diacritics + matres lectionis (ا/و/ي).

`morpheme_integrity_rate` requires Java + Farasa; if either is unavailable the metric is reported as `null` (not `0.0`) so it can't be confused with a real "no boundaries respected" result.

**How to read the numbers** (architectural ceilings to flag, not bugs):

- **CharacterBERT** ≈ 1.0 on root conservation by construction (whole word never split) and ≈ 0.0 on morpheme integrity (no internal token boundaries to honor).
- **char-JABER** ≈ 0.0 on root conservation (single-char tokens can't hold a 3-letter root) and ≈ 1.0 on morpheme integrity (every char boundary is a token boundary, so all morpheme boundaries are mechanically respected).
- **MorphoBPE** ≈ 1.0 on morpheme integrity *non-trivially* — by design, since Farasa pre-segments before BPE training.
- **BPE / WordPiece** sit in the middle; `morpheme_integrity_rate` is the cleanest discriminator among subword tokenizers.

Sample comparison from a smoke run on a small synthetic corpus:

| Tokenizer | root_cons | patt_cons | morph_int | root_bear% | patt_bear% |
|---|---:|---:|---:|---:|---:|
| BPE-2000      | 0.359 | 0.518 | 0.435 | 22.5 | 22.5 |
| WordPiece-500 | 0.692 | 1.000 | 0.000 | 74.4 | 74.4 |
| MorphoBPE-500 | 0.692 | 1.000 | **1.000** | 36.3 | 36.3 |
| CharacterBERT | 0.692 | 1.000 | 0.000 | 74.4 | 74.4 |
| char-JABER    | 0.000 | 0.000 | 1.000 | 0.0  | 0.0  |

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
- **`morpheme_integrity_rate: null` in results**: Farasa is unavailable (no Java, or `farasapy` failed to launch the segmenter). Install a JRE; the other morphological metrics still work without it.
- **`root_bearing_token_pct: null` in results**: every token cleaned to an empty string. Most often this is byte-level encoding leaking through — make sure new tokenizers either populate readable token strings in `TokenizerOutput.tokens` or extend `clean_token_string` / `_try_decode_bytelevel` in `tokenizers/morphological_utils.py`.
- **Roots look wrong (e.g. `لكتب` for `والكتاب`)**: the `[morphological]` extras aren't installed and the consonant-skeleton fallback is in use. Run `pip install -e ".[morphological]"` for proper qalsadi roots.
- **`CamelBridgeError: Camel subprocess interpreter not found ...`**: the `araroopat` tokenizer needs the separate `.venv-camel` (see the AraRooPat section under Install). Either run the 3 setup commands or point `$ARAROOPAT_CAMEL_PYTHON` at an interpreter that has `camel-tools`.
- **`CamelBridgeError: Camel subprocess exited unexpectedly (EOF on stdout)`**: the camel server crashed. The exception message includes the captured stderr — usually a missing data package (run `.venv-camel/bin/camel_data -i light`) or a corrupted `~/.camel_tools/` cache.
- **OOM on `char_jaber`**: sequences are much longer; reduce `task.params.max_length`, reduce `training.batch_size`, increase gradient accumulation, or run on a larger GPU.
- **CharacterBERT generation**: `LlamaAdapter.generate(...)` raises `NotImplementedError` for `character_cnn` embedding type, so QA evaluation will fall back to empty predictions (F1/EM will be poor). Use tasks/metrics that do not require generation, or extend generation support for that mode.
- **CharacterBERT on LightEval benchmarks**: reports `accuracy=0.0` — log-likelihood scoring is unsupported by the word-level CharCNN. Expected, not a bug.

## License

Add your license information here (none specified yet).
