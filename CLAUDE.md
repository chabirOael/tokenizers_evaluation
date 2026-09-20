# Arabic Tokenizers Evaluation Platform

## Project Overview

A universal platform for evaluating Arabic tokenizers by measuring LLM downstream performance. All external parameters (training pipeline, datasets, model architecture, hyperparameters) are held fixed; only the tokenizer changes between experiments. Every condition (native Llama tokenizer + every from-scratch tokenizer variant) runs the same fixed 3-phase training pipeline, then is evaluated on the same downstream benchmarks.

- **Tokenizer-training corpus**: `Jr23xd23/ArabicText-Large` (HuggingFace) — used for tokenizer training + intrinsic eval
- **Phase 1 + 2 corpus**: `Mostafa3zazi/Arabic_SQuAD` (machine-translated SQuAD-v1, 48,344 train rows)
- **Phase 3 corpus**: TyDiQA-Arabic (`google-research-datasets/tydiqa` `secondary_task` filtered to Arabic — 14,805 train / 921 val rows) + ARCD (`hsseinmz/arcd` `plain_text` — 693 train / 702 val rows) + the synthetic `arabic_squad_mcq` (48,344). Optional **free-form instruction corpora** (added 2026-09-17, pinned Hub revisions): CIDAR (`arbml/CIDAR`, 9,967 after exact-dup removal), Bactrian-X Arabic (`MBZUAI/Bactrian-X` `data/ar.json.gz`, 67,017) and the Aya collection (`CohereForAI/aya_collection_language_split` `standard_arabic`, sub-dataset allowlist, default `Aya-Dataset` + `Dolly-v2 (T)` = 19,803 Arab-script rows). A phase `mixture` block sets the total example count and the extractive / MCQ / free-form ratio — see *Phase 3 mixture*.
- **Primary LLM**: LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`). Also supported: Llama-3.2-3B (same adapter/tokenizer, `llama_3b_3phase_with_sft.yaml`) and **Qwen3-4B-Base** (`Qwen/Qwen3-4B-Base`, registry key `qwen3` + native tokenizer wrapper `native_qwen3`, `qwen3_4b_3phase_{with,no}_sft.yaml`)
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
                          #   finetune_corpora.py (Arabic-SQuAD + TyDiQA + ARCD),
                          #   pretraining_mix/ (packed raw-text mix for Phase 1/2: sources,
                          #     filters, dedup, dialect_id, pool [Stage A], packing [Stage B])
  tokenizers/             # base.py + 8 implementations + native_llama wrapper
  models/                 # base.py, llama_adapter.py, embeddings/{standard,character_cnn,char_jaber_embed,charformer_embed}.py
  tasks/                  # base.py + lighteval/ (abstract base + 4 dataset files +
                          #   utils.py opt-in helpers). LightEval is eval-only.
                          #   freeform/ (generation.py, metrics.py, bertscore.py, cidar.py: the
                          #   held-out CIDAR generation task)
  judge/                  # freeform_judge.py: LLM-judge stage over the dumps (vllm | openai backends)
  training/               # freezing.py, phases.py (3-phase runner)
  evaluation/             # metrics.py, evaluator.py, reporter.py, intrinsic_metrics.py
  pipeline/               # experiment.py (end-to-end orchestrator)
configs/                  # YAML configs: base, tokenizers/, models/, tasks/, experiments/, judges/,
                          #   contamination/ (held-out sets, exclusions, the free-form held-out JSONL)
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
| — | NativeQwen3 | `native_qwen3` | `tokenizers/native_qwen3.py` | standard | Subclass of `NativeLlamaTokenizer` wrapping `Qwen/Qwen3-4B-Base`'s pretrained tokenizer; `train()` is a no-op. Qwen quirks it encodes: **no BOS** (`Qwen2Tokenizer` adds no special tokens on encode — the pipeline is BOS-agnostic via LCP masking and `full_len − ctx_len` scoring), **no UNK** (byte fallback; the `unk_token` key is *omitted* from `special_tokens`, so UNK rate is 0 and the UNK CSVs are header-only), `pad = eos = bos = 151643` (`<\|endoftext\|>`). **Padded embedding matrix**: `len(tokenizer) = 151669` but the model ships 151936 rows, so `vocab_size` reports `AutoConfig.vocab_size` (151936) to keep `resize_token_embeddings` a true no-op. Qwen3-4B-Base has **tied embeddings** like Llama-3.2-1B (0.6B/1.7B/4B tied; 8B+ untied) — same freezing-helper warning. Use the **Base** checkpoint, not the post-trained `Qwen/Qwen3-4B` (chat template, thinking mode, `eos=<\|im_end\|>`). |

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

Each phase is independently toggleable via its own `enabled` flag. Phase 3 additionally runs periodic eval on the TyDiQA-AR + ARCD **`dev`** slices (a title-level 5 % of each official train split; the official `validation` splits are held out — see *Held-out evaluation sets and contamination*) for stagnation early-stop (patience=5, min_delta=5e-4, min_steps_before_stop=500, restore-best-at-end=true). Phases 1 and 2 can instead train on the packed raw-text **pretraining mix** (`datasets: ["pretraining_mix"]`, full-sequence loss) — see *Pretraining Mix* below.

**Per-phase params (all adjustable):** `enabled`, `datasets` (registry keys: `arabic_squad`, `tydiqa_arabic`, `arcd`, `arabic_squad_mcq`, `cidar`, `bactrian_x_ar`, `aya_ar`, `pretraining_mix`), `mixture` (see *Phase 3 mixture*), `trainable_parameters` (substring list; `["*"]` = all), `steps`, `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `optimizer`, `weight_decay`, `max_length`, `loss_target` (`"full_sequence"` | `"answer_only"`), `lr_scheduler` (`"cosine"` | `"constant"` | `"linear"`), `warmup_steps`, `max_grad_norm`, `save_checkpoint`. Phase 3 also has `early_stopping`. Defaults in `configs/base.yaml`.

**`steps` and `warmup_steps` are micro-steps** (one batch of `batch_size` sequences per loop iteration); the optimizer updates every `gradient_accumulation_steps` of them, so the effective batch is `batch_size × gradient_accumulation_steps` and a phase performs `steps / gradient_accumulation_steps` updates. `eval_every_n_steps`, `min_steps_before_stop`, `mix_tokens = steps × batch_size × block_size` and the mixture's `steps = total_examples / batch_size` all count the same unit. **Scheduler-horizon fix (2026-09-18):** `run_phase` used to hand the scheduler `phase_cfg.steps` as its horizon although `scheduler.step()` runs once per *update* — at accumulation 4 the cosine covered a quarter of the phase (`[sft] step 6800/7500 … lr=1.79e-04` for a 2e-4 peak; Phase 1 at accumulation 1 was unaffected). The horizon is now `ceil(steps / accum)` updates and the warm-up `ceil(warmup_steps / accum)` updates, logged at phase start; `warmup_steps: 400` at accumulation 4 = 100 updates. `tests/test_phases.py::TestSchedulerHorizonInUpdates` pins it. Every cosine-scheduled run before the fix effectively trained at a near-constant peak LR — read their logs with that in mind.

**Why this design.** The previous "10% SFT on benchmark + 90% eval" was empirically destructive: each from-scratch tokenizer's vocab indices were silently mapped onto Llama's first N pretrained rows by `resize_token_embeddings`, and 10% benchmark-specific SFT couldn't drift those mappings far enough to find real signal. The 3-phase pipeline (a) deliberately aligns embeddings before any other training (Phase 1), (b) teaches QA format on a regular translated dataset before exposure to native Arabic complexity (Phase 2), and (c) does the decisive SFT on native Arabic QA in Phase 3. The whole pipeline runs identically across all conditions so the only experimental variable is the tokenizer + its embedding/lm_head weights.

**Phase 3 prompt format** (`TEMPLATE_VERSION = 2`, 2026-09-18) lives in [src/arabic_eval/data/finetune_corpora.py](src/arabic_eval/data/finetune_corpora.py). The extractive (`qa`) and free-form (`instruction`) templates are sectioned Arabic-Alpaca prompts with a one-line header and `### ` labels that differ per template, so the model can tell a 3-word span task from a 40-word answer *before* the answer cue — extractive: `QA_HEADER` (`فيما يلي نص وسؤال عنه. أجب عن السؤال اعتمادًا على النص.`) + `### السياق:` / `### السؤال:` / `### الإجابة:`; instruction: `INSTRUCTION_HEADER` (`فيما يلي تعليمات تصف مهمة. اكتب إجابة تكمل الطلب بشكل مناسب.`) + `### التعليمات:` / `### الإجابة:`, and with an input the `INSTRUCTION_HEADER_WITH_INPUT` header plus a `### المدخل:` section. Sections are separated by blank lines, the prompt ends in `### الإجابة:\n` and the answer follows directly (`_format_qa_full` adds no separator). `mcq_letter` is untouched — the LightEval-official letter prompt with `" {letter}"` after `الإجابة:` — as are the LightEval per-task prompts. Version 1 was the flat `السياق: …\nالسؤال: …\nالإجابة: {answer}` for both templates: one shared answer cue, and 70 % of the 30 000-record mixture taught "≤ 4 words after it" (12 000 extractive records with a p50 answer of 3 words + 9 000 one-letter MCQ vs 9 000 free-form at p50 25–48 words); the Alpaca header was also the formulation that measured best on the untrained Qwen3-4B-Base (loops 23 % vs 44 %). `TEMPLATE_VERSION` enters the `qa_blend` cache fingerprint (blocks hold rendered text) and is recorded in the mixture manifest and `all_metrics.json["training"][<phase>]["data"]["template_version"]`; the contamination indexes hash raw passages / questions, not templates, so no rebuild. The free-form eval (`freeform_cidar`) builds its prompt with `_format_qa_prompt` on an `instruction` record, so it follows the training string by construction (a test pins the equality).

**Answer-only loss masking** uses an LCP (longest common prefix) helper at [src/arabic_eval/data/answer_only_masking.py](src/arabic_eval/data/answer_only_masking.py) — necessary because Llama auto-appends `</s>` to standalone encodings, so naive `labels[:len(prompt)] = -100` would eat the first answer token. **Every QA record ends in EOS** (`tokenize_record`, fixed 2026-09-18): the from-scratch tokenizers append `</s>` on encode, but the native Llama / Qwen3 wrappers add none (Qwen adds no special tokens at all, Llama-3 only BOS), so an SFT model on a native tokenizer never saw an answer end — the first native_qwen3 free-form run generated to the 1 200-char cap on 249 / 250 prompts, 99 % repetition loops, chrF 9. `tokenize_record` now appends the tokenizer's `eos_token` when the encoding lacks it (not to a truncated record — there is no answer end to mark — and never to a `char_ids` encoding, those tokenizers emit their own); the EOS is a loss target. The pretraining mix and `qa_blend` already guaranteed a trailing EOS (`encode_document`). Native-tokenizer SFT results from before the fix are not comparable on anything generative.

**Tied embeddings on Llama-3.2-1B** — `lm_head.weight is model.embed_tokens.weight`, so `lm_head` is absent from `named_parameters()`. The freezing helper ([src/arabic_eval/training/freezing.py](src/arabic_eval/training/freezing.py)) warns when a substring matches no parameter while others do (the tied-weight case) and continues — training `embed_tokens` IS training `lm_head`.

## Phase 3 mixture — ratio-controlled SFT set with free-form corpora (`src/arabic_eval/data/sft_mixture.py`, added 2026-09-17)

Without a `mixture`, a QA phase concatenates its `datasets` and shuffles: the ratio is whatever the pool sizes dictate (the default Phase 3 pool is 24 % extractive / 76 % synthetic MCQ, and 2 000 steps × batch 4 see 8 000 of its 63 842 records). With one, the phase trains on **exactly `total_examples` records at an exact per-category ratio**:

```yaml
training:
  corpus_params:                          # optional per-corpus loader params (only aya_ar takes any)
    aya_ar: {include_datasets: ["Aya-Dataset", "Dolly-v2 (T)"]}
  phases:
    sft:
      datasets: ["tydiqa_arabic", "arcd", "arabic_squad_mcq", "cidar", "bactrian_x_ar", "aya_ar"]
      mixture:
        total_examples: 30000
        shares: {extractive: 0.40, mcq: 0.30, free_form: 0.30}
        within_category: "equal"          # equal (capacity-aware) | proportional; weights: {corpus: w} overrides
        upsample: false                   # true = a short category repeats a further seeded permutation
        drop_truncated_answers: false     # true = skip records whose answer is cut at max_length, top up
        seed: 42
      steps: null                         # derived: total_examples / batch_size (7 500); validated if set
```

- **Categories are facts of the registry, not config.** `CORPUS_CATEGORY` in `config.py` maps every corpus to `extractive` (`qa` template: arabic_squad, tydiqa_arabic, arcd), `mcq` (`mcq_letter`: arabic_squad_mcq) or `free_form` (`instruction`: cidar, bactrian_x_ar, aya_ar); a test pins `DatasetName` ↔ `CORPUS_CATEGORY` ↔ `_LOADERS` in sync. `shares` must sum to 1 and name exactly the categories of the listed datasets (a listed corpus without a share, or a share without a corpus, is a validation error).
- **Quotas.** `shares × total_examples` by largest remainder; inside a category the quota is **water-filled** across the corpora — `equal` gives each the same count but a corpus smaller than its slice gives all it has and the rest re-splits (ARCD 693 + TyDiQA 11 307 for a 12 000 extractive quota); `proportional` weights by pool size; `weights` overrides per corpus. **Quotas count kept records**: each corpus is walked in a seeded permutation (sorted by id first, so loader row order is irrelevant) and tokenized as it goes, records the answer-only LCP masking drops (max_length ate the answer) do not count, and a corpus that runs dry hands its deficit to the others of its category. A category that still cannot fill its quota is a `MixtureShortfallError` naming the numbers and the largest `total_examples` these shares admit (`min_cat(capacity / share)`, 38 744 for the default six corpora at 40/30/30) — unless `upsample: true`. Only the records used are tokenized (the old path tokenized the whole 63 842-record pool every run).
- **`steps` mirrors `mix_tokens`:** derived as `total_examples / batch_size` when null (one exact pass; early-stop may end it sooner), an error when set inconsistently; `total_examples` must divide by `batch_size`. `mixture` is refused on a `pretraining_mix` phase and works on Phase 2 too (nothing is sft-specific). `clean_latin_rows` filters each pool before capacity is computed (Bactrian-X 67 K → ~50 K, CIDAR 11 % Latin rows).
- **Shares are in examples; the loss is in tokens.** Measured with the native Llama tokenizer at max_length 512 on the reference config: extractive 40 % of examples = **12 %** of loss tokens, MCQ 30 % = **0.7 %**, free-form 30 % = **87 %** (a free-form answer is ~100–145 tokens against 1 for an MCQ letter and ~13 for an extractive span). The manifest reports `loss_token_share` per category next to `example_share`; read both before interpreting a ratio ablation.
- **Provenance.** `all_metrics.json["training"]["sft"]["data"]["mixture"]` carries the per-category (`share`, `quota`, `kept`, `example_share`, `loss_tokens`, `loss_token_share`, `capacity`) and per-corpus (`available`, `after_latin_filter`, `weight`, `planned`, `drawn`, `dropped_truncation`, `dropped_cut_answer`, `kept`, `repeated`, `passes`, `revision`) numbers; `<cell>/data/sft_mixture_manifest.json` is the same plus the **record ids drawn** per corpus, so any set can be reconstructed without re-running. **Dry run first:** `scripts/plan_sft_mixture.py --config <yaml>` prints the plan (pool sizes, quotas, ceiling) and, with a tokenizer (`native_*` needs none; from-scratch ones `--tokenizer-path`), the drop counts and loss-token shares; `--plan-only` skips tokenization.
- **The free-form loaders** (`finetune_corpora.py`, `prompt_template="instruction"`: the sectioned `INSTRUCTION_HEADER` + `### التعليمات:` / `### الإجابة:` prompt, with `### المدخل:` and the with-input header when the record carries an input — see *Phase 3 prompt format*). Every one reads a **pinned revision** (`PINNED_REVISIONS`, recorded in the manifest). `cidar`: `\r\n` normalized, 33 exact `(instruction, output)` duplicates dropped, 18 repeated `index` values get the row position appended. `bactrian_x_ar`: the repo is a loading-script dataset that `datasets` ≥ 3 refuses, so `data/ar.json.gz` is fetched with `hf_hub_download` (52 002 Alpaca + 15 015 Dolly; 641 `input: null` → empty; `input` becomes `context`; 25 % of rows contain Latin letters). `aya_ar`: the `standard_arabic` split is 5.86 M rows, 84 % templated Wiki-split simplification plus SODA dialogue, event linking and translated HotpotQA / NQ / CNN-DM; only `Aya-Dataset` (4 995 human-written Standard Arabic rows) and `Dolly-v2 (T)` (29 616) are free-form, and **half of Dolly is `script: Latn`** (English / `<unk>` garbage) — the loader keeps `Arab`-script rows only (14 808) and splits the literal `\nContext:` label 4 522 of them embed into `context`. The three 1.34 GB shards are read once with pyarrow filters and the filtered subset is cached as Parquet under `outputs/data_cache/sft_corpora/aya_ar/<fp>.parquet` (fingerprint: revision + allowlist). `training.corpus_params` is threaded through every `load_corpus` call site (phases, early-stop splits, `qa_blend` — whose cache fingerprint includes the params only when non-empty, so existing packed entries keep their key).
- **Contamination scan** (`scripts/scan_sft_contamination.py`, one-off per new corpus): normalized exact-match of the corpus question and word-8-gram containment of every benchmark question against the four benchmarks. Run 2026-09-17 on the three free-form corpora: exact hits ≤ 12 per (corpus, benchmark) and 8-gram hits ≤ 14, all generic trivia (`ما هو أطول نهر في العالم`, `لماذا السماء زرقاء`, capital-of questions) or shared Wikipedia / hadith passages, against 9 000–23 000 benchmark rows — no systematic leakage; the ids are in `outputs/data_cache/sft_corpora/contamination_scan.json` if you want to exclude them.
- **Not changed:** the early-stop signal is still extractive (TyDiQA-AR + ARCD `dev` since 2026-09-17); Phase 1/2 defaults; the non-mixture path (byte-identical for every existing YAML). The contamination exclusions apply to the mixture pools like to every other train split. Reference config: [configs/experiments/native_llama_3phase_sft_mixture.yaml](configs/experiments/native_llama_3phase_sft_mixture.yaml). Tests: `tests/test_sft_mixture.py` (quota arithmetic, water-fill, spill-over, truncation top-up, upsample, determinism, manifest, every validator, the three loaders on fixture rows) and the mixture cell in `tests/test_e2e_sweep.py`.

## Held-out evaluation sets and contamination (`src/arabic_eval/data/contamination.py`, added 2026-09-17)

**Held out**: TyDiQA-AR `validation` (921 rows; the public dev split — TyDi's test is hidden, dev is what papers report), ARCD `validation` (702; the paper's test split) and, once it exists, the user's **free-form eval prompt file** (a *pending* placeholder today). Declared in [configs/contamination/heldout_sets.yaml](configs/contamination/heldout_sets.yaml) (`kind: corpus` with the pinned revision from `PINNED_REVISIONS`, or `kind: file` with a JSONL path — `id`, `prompt`, optional `context` / `reference`; `path: null` = pending, skipped everywhere, listed as such in every report; the template is `freeform_prompts.template.jsonl`). Nothing trains on them, and since 2026-09-17 nothing *selects* on them either: Phase 3 early-stops on **`dev`**, a deterministic title-level 5 % slice of each official *train* split (`is_dev_title`: sha1 of `salt:corpus:title`, so dev and train never share an article; `train` = official train minus dev; measured 861 TyDiQA-AR / 27 ARCD dev rows). `qa_blend` refuses both `validation` and `dev`.

**Two matches on one normalization** (`normalize_text`: NFKC, diacritics + tatweel out, alef / ة / ى folded, Arabic-Indic digits → ASCII, punctuation out): SHA-1 of the normalized passage / paragraph / question, and word n-gram containment (8 for passages, 6 for prompts) with, per (training record, held-out record), the *coverage* of the held-out record's n-grams and the longest shared *run*. Tiers: **`contaminated`** = exact hash, or coverage ≥ 0.5, or a run ≥ 20 words; anything else sharing an n-gram is **`overlap`** (reported only). Thresholds live in the YAML.

**Where it acts.** `scripts/check_contamination.py --write-exclusions` scans the seven QA corpora (TyDiQA / ARCD `train` *and* `dev`), the cached pretraining-mix pool by document, and opt-in the tokenizer corpus; writes `hits.parquet` + `report.json` under `outputs/data_cache/contamination/<run>/` and the committed **[configs/contamination/exclusions.json](configs/contamination/exclusions.json)** — training record ids in the contaminated tier. `training.contamination_exclusions` (default: that file; `null` = off) is applied by `load_corpus` to `train` / `dev` only — every phase, the mixture pools, the early-stop split, `qa_blend` (whose fingerprint gains the list's digest) — and `all_metrics.json["training"][<phase>]["data"]["contamination"]` records what was dropped. The pool does not use the id list: `pretraining_mix.heldout_filter` (on by default) re-runs the same index **by content** at Stage A (drop reason `heldout_overlap`, samples in `dropped_samples.jsonl`), and the pool fingerprint includes the held-out identity (YAML hash, pinned revisions, prompt-file hash), so a new prompt file rebuilds the pool on its own. `scan_sft_contamination.py` (corpus questions vs the four MCQ benchmarks) shares the normalizer.

**Measured (2026-09-17, first run, prompts pending; 103 s).** TyDiQA and ARCD both sample Arabic Wikipedia, so the same article introductions recur *across* the two datasets and across the ARCD split boundary as overlapping windows — verbatim runs of 20–74 words (الشيعة, جمال عبد الناصر, الهند), not artifacts. Excluded per corpus: `tydiqa_arabic` **356** of 14 805 (187 hit TyDiQA-AR dev, 177 of them by exact passage; 173 hit ARCD test), `arcd` **122** of 693 (92 vs ARCD test — 52 exact; 30 vs TyDiQA-AR dev), `aya_ar` 54, `bactrian_x_ar` 6, `cidar` 3, `arabic_squad` / `arabic_squad_mcq` 1 each (543 ids). The old 20 M-word pool held 24 contaminated documents; the rebuilt pool (fingerprint `ccf3fa895f9690be`, 248 s, every word target reached, 48 793 docs) dropped 25 as `heldout_overlap` (Wikipedia 8, FineWeb-2 14, ArabicWeb24 3) — the old pool `07f13d8a7d5e8ad1` and its packed entries are dead weight and can be deleted. ARCD train loses 18 % — the price of its own split design; raise `tiers.run_words` in the YAML if that is too much. Under the old protocol the early-stop signal saw 159 TyDiQA-AR dev rows whose passage was verbatim in train. Paraphrase overlap (Arabic-SQuAD is translated English Wikipedia) is invisible to hash / n-gram checks by construction. Tests: `tests/test_contamination.py`.

## Free-form eval — held-out CIDAR generation, LLM judges, human check (added 2026-09-18)

The four MCQ benchmarks only measure recognition through log-likelihood. The free-form eval is the first task that exercises the **decode** side of a tokenizer: every variant greedy-decodes an answer to the same 250 held-out CIDAR instructions under the same rules, the decoded text is scored with reference-based and reference-free metrics inline, LLM judges score it afterwards, and a blind human check calibrates the judges. Four pieces, in the order they run:

**1. Held-out set** — [configs/contamination/freeform_cidar_heldout_v1.jsonl](configs/contamination/freeform_cidar_heldout_v1.jsonl) (250 rows: `id` = the CIDAR record id, `prompt`, `reference`, `stratum`, `ref_chars`, `max_train_cosine`; manifest next to it) built by `scripts/build_freeform_heldout.py` ([data/freeform_heldout.py](src/arabic_eval/data/freeform_heldout.py), pure selection function + E5 embedder). It is the `freeform_prompts` held-out set of `heldout_sets.yaml` (`kind: file`, live since 2026-09-18), so `check_contamination.py --write-exclusions` dropped its 250 rows from `cidar/train` (exact question hash) plus 3 TyDiQA rows that overlap it by content, and the pool fingerprint changed (next run rebuilds the pool, ~4 min, and repacks per tokenizer). **Filters, measured on the real rows:** 9 967 candidates → excluded ids 3, Latin letters 1 134, reference outside 20–1 500 chars 298, n-gram twins 2 097 prompt + 592 reference (the normalized prompt equals or shares a word 6-gram with a Bactrian-X / Aya / other-CIDAR prompt; references by 8-gram), **E5 near-duplicates 4 706** → 1 137 survivors → 250 drawn equally from three reference-length terciles (≤ 141 / ≤ 300 / > 300 chars, seed 42). *The finding behind the E5 gate:* CIDAR is ~90 % Alpaca-derived and Bactrian-X holds all of Alpaca translated, so nearly every CIDAR instruction has a **paraphrase twin** (same Alpaca source, different translation) that shares no n-gram — `أنشئ قصة عن ببغاء` vs `قم بإنشاء قصة عن ببغاء` at cosine 0.970, `كيف بدأت رسوم الكهوف؟` vs `كيف بدأت رسم الكهوف؟` at 0.945 — with a reference that is a paraphrase of the twin's output too, i.e. genuine leakage for reference-based metrics and a reference-guided judge. The calibrated threshold (5th percentile of the verbatim twins' nearest-neighbour cosine, 0.947) still let such pairs through, so the committed set uses a fixed **0.93**: above it neighbours are the same instruction, below it they share a template with different content (`اشرح لماذا الجملة التالية غير صحيحة نحويًا` with a different input). Consequence, stated plainly: a leak-free set is confined to CIDAR's own material — 18 % of the 250 are grammar / إعراب questions, 40 % question-shaped, the rest imperative instructions; median nearest training cosine 0.91. Rebuilding with another threshold changes the file hash and therefore the exclusions and the pool.

**2. The task `freeform_cidar`** ([tasks/freeform/](src/arabic_eval/tasks/freeform/), a plain `BaseTask`; YAML [configs/tasks/freeform_cidar.yaml](configs/tasks/freeform_cidar.yaml); in the `sweep.tasks` of [native_llama_3phase_sft_mixture.yaml](configs/experiments/native_llama_3phase_sft_mixture.yaml)). The prompt is the exact Phase 3 instruction string (`_format_qa_prompt` on an `instruction` record — the sectioned `INSTRUCTION_HEADER` / `### التعليمات:` / `### الإجابة:` prompt of *Phase 3 prompt format*, a test pins the equality), so generation continues where the mixture's training left off — **only meaningful on configs whose Phase 3 saw free-form corpora**; a model that never learned the format answers noise and the judge cannot discriminate. **Fixed decoding = the same rules, not the same token count:** pure greedy (no sampling, no repetition penalty, no n-gram blocking — every one of those is granularity-dependent, a character tokenizer repeats characters by nature), a budget of **1 200 characters of decoded text** turned into a per-tokenizer `max_new_tokens` from the tokenizer's measured chars/token on the references × 1.15 (+ 8, clamped to 32–4 096: ~350 for BPE-32K, ~1 400 for char-JABER), stop at the tokenizer's EOS, at a stop marker in the decoded text (`\nالسؤال:` / `\nالسياق:` / `\nالإجابة:` / `\n###` / `\nفيما يلي` — the model starting a new block, section or header line) **or at a repetition loop in the decoded text** (added 2026-09-18, `loop_stop: true`: the three rules of `metrics.detect_loop` — a word 4-gram occurring 3 times, one word 5 times in a row, one character 20 times in a row — the same function `is_degenerate` uses, so a stopped generation is exactly one the metric would flag; tokenizer-agnostic because it reads words and characters of decoded text, unlike a token-level repetition penalty that would charge char-JABER for spelling and AraRooPat for `[CLITICP_ال]`), both checked every 16 steps by decoding the unfinished sequences; cut to the character budget afterwards. A loop-stopped row gets `stop_reason="loop"` / `hit_loop`, its `generation` is cut right before the *second* copy of the repeated unit (the first full copy is kept), `generation_raw` keeps everything, and **`degenerate` is judged on `generation_raw`** so `degenerate_rate` stays comparable with runs that had no loop stop; `loop_stop_rate` is the new summary metric. The first batch also gets an untimed 8-token warm-up `generate`, and `generation_wall_sec` / `gen_chars_per_sec` sum the timed per-batch walls, excluding the warm-up and the encode / decode bookkeeping; each batch logs its width, new tokens and tok/s. **The ~300 s CPU-bound first batch was PyTorch's cuDNN SDPA backend** (torch 2.11, the default on sm90): it host-compiles a kernel per new (batch, KV-length) shape, and decoding visits a new KV length at every step — measured 58 s vs 3.4 s for a repeated shape at 150 new tokens, 3.8 s with the backend off at the same steady-state throughput; the odd-sized last batch paid it again, and every distinct prompt length of an MCQ eval pays it too (the 30-row smoke ACVA took 161 s). `LlamaAdapter.__init__` now calls `configure_sdpa_backends()` ([models/llama_adapter.py](src/arabic_eval/models/llama_adapter.py)), which turns that backend off once per process (flash / mem-efficient / math stay — the same exact attention); `ARABIC_EVAL_CUDNN_SDP=1` keeps it. Free-form and MCQ timings from before 2026-09-18 include that compile cost. The prompt encoding's trailing `</s>` is stripped (every from-scratch tokenizer appends one to a standalone encoding). Batched, left-padded, `model.eval()`, restored afterwards. **Inline metrics** (`metrics.py`, all on decoded text so tokenizer-agnostic): `chrf` (mean sentence chrF, sacrebleu) + `chrf_corpus`, `bertscore_{p,r,f1}` (**in-house BERTScore**, [tasks/freeform/bertscore.py](src/arabic_eval/tasks/freeform/bertscore.py): `xlm-roberta-large` layer 17, greedy cosine matching, no IDF / rescaling; the `bert_score` package breaks under transformers 5 on empty strings and was verified equal to ours to 1e-6 on non-empty ones), `empty_rate`, `degenerate_rate` (repetition-loop detector: a word 4-gram ×3 or duplicate 4-grams ≥ 50 %, a word ×5 in a row, a character ×20, duplicate char 8-grams ≥ 60 % on 40+ chars), `latin_rate`, `arabic_letter_ratio`, `hit_cap_rate` / `loop_stop_rate` / `marker_stop_rate` / `eos_rate` / `char_truncated_rate`, `mean_gen_chars` / `mean_gen_tokens`, `gen_chars_per_sec` / `gen_tokens_per_sec` (generation speed is itself a tokenizer property), and **`reference_roundtrip_chrf`** — chrF of `decode(encode(reference))` against the reference, the decode-fidelity ceiling of a tokenizer (a lossy decoder caps every reference-based score before the model generates a word). `chars_per_token` and `token_cap` are recorded. **Non-generative tokenizers** (CharacterBERT, Farasa-CharacterBERT: word-level output head; Charformer: non-causal GBST) get `{"status": "generation_unsupported", "reason", "num_samples": 0}` with every metric `None` — the MEI pattern, never a zero; MEI itself skips the task (`task_not_mcq`). Covered variants: native_llama, native_qwen3, bpe, wordpiece, morpho_bpe, araroopat, char_jaber. **Row dump** `eval_rows/freeform_cidar.parquet` (one row per prompt: `instruction`, `context`, the exact `prompt_text`, `reference`, `generation` + `generation_raw`, token / char counts, `stop_reason`, the flags, every per-row metric; metadata carries the decoding config, cap, chars/token, held-out file hash).

**3. Judges** ([judge/freeform_judge.py](src/arabic_eval/judge/freeform_judge.py), torch-free; `scripts/judge/judge_freeform.py`; local wrapper `scripts/judge/run_judge.sh`) — a **separate stage over the dumps**: a 31B judge does not share the GPU with a training run, judges get swapped or added, and the human check reads the same files. Protocol: pointwise, reference-guided, **1–5 on a fixed English rubric** (`RUBRICS["default_v1"]`: the reference is a guide, not the only acceptable answer; do not reward length; judge meaning) plus sub-scores `correctness` / `fluency` / `instruction_following` and flags `repetition` / `wrong_language` / `empty` / `off_topic` / `truncated`, temperature 0, fixed seed, one user message (Gemma's template has no system role). Verdicts are JSON — fences stripped, first balanced object, then a prose regex fallback; `parse_ok=False` is counted, never fatal. **Two backends behind one YAML** ([configs/judges/](configs/judges/)): `vllm` (local; `gemma4_31b_local.yaml` = `google/gemma-4-31b-it`, structured JSON through vLLM's structured outputs) and `openai` (any OpenAI-compatible `/chat/completions` over plain urllib, `response_format` JSON schema; `gpt56_terra_api.yaml` = `gpt-5.6-terra`, **verified on your account 2026-09-18**: the model rejects an explicit `temperature`, so the YAML sets it `null` (field omitted) and `extra_body: {reasoning_effort: "low"}` with `max_tokens: 1024` — 4/4 probe verdicts identical to Gemma's, ~0.7 s per verdict at concurrency 8; export the key named by `api_key_env`). `structured_json` is a per-judge toggle with the prose parser as fallback (constrained decoding biases some models toward the first enum value). Per cell it writes `freeform_judge/<judge>.parquet` (score, sub-scores, flags, rationale, raw, `parse_ok`) and merges a compact summary into `all_metrics.json["downstream"]["freeform_cidar"]["judge"][<judge>]`: mean ± SE, histogram, sub-score means, flag rates, `score_by_stratum`, and — since every variant answers the same prompts — a **paired bootstrap** on the mean difference to the baseline cell (`--baseline`, default the `native_*` cell) with win / tie / loss rates; with ≥ 2 judge files per cell `judge_agreement` (Spearman, exact, quadratic-weighted κ). `comparison_report.txt` gains an *LLM judge* section per judge and a judge-agreement table (nested `judge*` blocks stay out of the main table like `per_subconfig_accuracy`). **Judge venv, verified 2026-09-18** (`.venv-judge`, gitignored): vLLM 0.29.0 **`+cu129` wheel** from the GitHub release + torch 2.13.0+cu129 from the PyTorch cu129 index (PyPI's default pulls a CUDA 13 torch that fails on the 570 driver); Triton's JIT needs `Python.h` → `apt-get download python3.10-dev libpython3.10-dev` (no root) + `dpkg -x` into `.local-pkgs/extracted/` and both include dirs on `CPATH` (what `run_judge.sh` sets); the FlashInfer sampler JIT needs ninja + nvcc → `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy judge). Measured: 59 GiB weights, 8.4 GiB KV cache at 4k context, **39 verdicts/s** at batch 64 (~1.5 min per 3 500 calls), cold start 441 s (compile + autotune caches now populated), 4/4 probe verdicts correct, structured JSON parsed. The transformers route (main venv) also fits (58 GiB) but decodes at ~27 tok/s — too slow.

**4. Human check** — the console's **Rate** tab ([tools/freeform_rating.py](src/arabic_eval/tools/freeform_rating.py)): build a set (default 50 prompts × 3 variants = 150 items: half the prompts at random, half where the judges disagree most — with one judge, where its scores vary most across variants; the baseline cell plus 2 others per prompt, round-robin so every cell is rated evenly; items shuffled, the cell kept only in `<experiment>/freeform_rating/<set>.json`), rate blind on the judge's rubric (instruction, reference, generation, 1–5 + flags + note; ratings in `<set>.ratings/<rater>.json`), then *agreement* reveals the cells and computes rater vs each judge (Spearman, exact, QWK, mean |Δ|), rater vs rater, judge vs judge on the same items, and the per-cell human mean next to the judge means. The **Free-form** tab ([tools/freeform_rows_browser.py](src/arabic_eval/tools/freeform_rows_browser.py)) browses the dumps joined with the judge files: filters on judge score / judge flag / stop reason / row flag / length stratum / text, sorts incl. *judges disagree most*, a summary strip (chrF, BERTScore, per-judge mean + 1–5 histogram, loops / empty / cap counts, each clickable), and a row detail with instruction, reference, generation, every judge's verdict and rationale, and **the same prompt in every other cell** (the cross-variant view a tokenizer comparison is read by). Unsupported cells are listed with their reason. **Judge runs start from the console too** (Free-form tab → *judge…*): `configs/judges/*.yaml` are listed with a readiness check (vLLM judge: `.venv-judge` + the extracted headers; API judge: its `api_key_env` exported in the server's environment), pick judges / baseline / limit, and `POST /api/judge/start` snapshots the YAMLs into `outputs/runs/<run_id>/judges/` and launches `run_judge.sh` (any GPU judge) or `judge_freeform.py` (API only) as a detached run with `kind: judge` — the Runs tab parses `[judge:<name>] <cell>: N verdicts` lines into per-cell chips, shows the per-cell judge means when the report exists, and a GPU judge conflicts with an active run exactly like an experiment (API-only judges start alongside). **Reproduce the judge environment with `scripts/judge/setup_judge_env.sh`** (venv + cu129 wheel + headers + verification; `VLLM_VERSION` / `CUDA_TAG` overridable). Verified 2026-09-18 end to end from the page with `gpt-5.6-terra` on a fixture sweep.

**Decode fidelity, measured 2026-09-18 on the archived tokenizers** (`reference_roundtrip_chrf`, the first thing this eval reads): **BPE decoded to byte-level surrogates** (`Ø§ÙĦ…`, round-trip chrF 0.9) and **WordPiece joined its `##` pieces with spaces** (52.6) — two pre-existing bugs, both HF `tokenizers` objects had no decoder set, unnoticed while the pipeline only scored log-likelihoods; fixed in `bpe.py` / `wordpiece.py` at train *and* load time (tokenizers saved before the fix carry no decoder in their JSON), BPE now 100.0 on the held-out references, WordPiece 77.2 (Whitespace pre-tokenization detaches punctuation and the archived tokenizer predates the alef-fold flip, so `أ`/`إ` are UNK). char-JABER 78–84 for the same alef / diacritics reason (48 UNKs on one reference) — the panel retrain already owed after the flip fixes it. **MorphoBPE decode is lossy by construction**: `segment_with_farasa` replaces Farasa's `+` with spaces before BPE, so word boundaries are gone and decode returns morphemes space-separated; a future retrain with an end-of-word marker would fix it, the metric says what it costs today. Tests: `tests/test_tokenizer_roundtrip.py`.

**Read the numbers with these in mind.** A native-tokenizer model trained before 2026-09-18 never learned to stop (no EOS in its SFT records — see *3-Phase Training Pipeline*); its `hit_cap_rate` ≈ 1 and `degenerate_rate` ≈ 1 are that bug, not the tokenizer. Shares of the held-out set are by reference length, not by task type. chrF / BERTScore against one reference are weak on open-ended instructions — the judge is primary, the reference-based numbers are the cheap, deterministic second opinion; `reference_roundtrip_chrf` says how much of a gap is the decoder's. A repetition loop or an empty answer scores 1 by the rubric and is also counted separately (`degenerate_rate`, `empty_rate`, flag rates) — do not let a fluent-but-wrong variant hide behind a mean. Judges favour fluent, longer text; the rubric says not to, and the human check is what tells you whether it listened. Tests: `tests/test_freeform_heldout.py`, `test_freeform_task.py` (tiny random Llama + a BOS/EOS word tokenizer, real HF greedy generation), `test_freeform_judge.py` (fake backend end to end), `test_freeform_console.py`; the console page was driven headlessly through both tabs on a fixture sweep (2026-09-18, no page errors).

## Pretraining Mix — packed raw-text corpus for Phase 1 / Phase 2 (`src/arabic_eval/data/pretraining_mix/`)

Any phase can train on a packed raw-text mix instead of QA records by setting `datasets: ["pretraining_mix"]` plus **`mix_tokens`** (the tokens the phase draws; must be the phase's sole dataset, `loss_target: full_sequence`, `max_length == pretraining_mix.block_size`, and `mix_tokens == steps × batch_size × block_size` — `steps` may be omitted and is derived; all validated in `TrainingConfig`). The reference config is [configs/experiments/all_tokenizers_sweep_pretrain_mix.yaml](configs/experiments/all_tokenizers_sweep_pretrain_mix.yaml): Phase 1 + Phase 2 on **70 % FineWeb-2 `arb_Arab` / 20 % Arabic Wikipedia (`20231101.ar`) / 10 % ArabicWeb24**, 4 096 000 tokens each; Phase 3 unchanged. Phase 2 there also carries a **`qa_blend`** (7 %; see below). The `warmup.datasets` line is the ablation switch — set it back to `["arabic_squad"]` + `answer_only` + `mix_tokens: null` + `qa_blend: null` to change only Phase 1; `qa_blend: null` alone gives a pure raw-text Phase 2 at identical compute. All knobs default in `configs/base.yaml` under `training.pretraining_mix`; the block is inert until a phase references it.

**Two stages.**
- **Stage A — pool** (`pool.py`, tokenizer-independent, cached under `cache_dir/<fingerprint>/`, shared by every cell and experiment; the fingerprint hashes only the Stage A fields, so `block_size` never invalidates it — `pool.total_words` does, and a change rebuilds from scratch, ~10 s per M words). Sources are streamed with a seeded *shard + buffer* shuffle (without the shard shuffle FineWeb-2 yields a 2013 snapshot) in `dedup.priority` order and each document goes through, cheapest first: `normalize_document` (NFKC + tatweel + whitespace, paragraph = line; alef/diacritic folding opt-in) → Wikipedia tail sections cut (`المراجع`, `وصلات خارجية`, …) → `(بالإنجليزية: …)` Latin glosses stripped → quality rules (`min_words` 50, Latin-letter ratio ≤ 5 %, Arabic-script ratio ≥ 70 %, short-line / duplicate-line / symbol ratios) → dialect-marker gate → opt-in CAMeL dialect ID → MinHash LSH near-dup (datasketch, 5-word shingles, 128 perms, Jaccard 0.8; the higher-priority source's copy survives) → exact-paragraph dedup (repeated normalized paragraphs removed *from* the doc — boilerplate and verbatim copies; re-checked against `min_words`) → truncate to `max_words` 3000 at a paragraph boundary. Stops per source at `share × pool.total_words` kept words (20 M default = 48 738 docs, 68 MB, 199 s; sufficient for Σ `mix_tokens` ≤ ~20 M under any tokenizer since every panel member has fertility ≥ 1 — CharacterBERT is the binding case). Artifacts: `manifest.json` (per-source counts per drop reason, pinned dataset revisions, source prefilter skips), `<source>.parquet`, `dropped_dialect.csv` (every dialect drop with its markers), `dropped_samples.jsonl` (≤30 per reason). `--calibrate N` runs the same pipeline on N raw docs per source without stopping or caching.
- **Stage B — pack** (`packing.py`, **once per tokenizer, shared across experiments**, at `<pool>/packed/<tokenizer_fingerprint>/`; each cell gets a `data/pretraining_mix/packed_manifest.json` pointing there). The *whole* pool is tokenized per source in a fixed seeded order (`source_doc_order`, identical for every tokenizer — a smaller pool is always a prefix of a larger one), then packed at the **exact-share budget** `min_i(available_tokens_i / share_i)`: the pool is 70/20/10 in *words* and per-source fertility differs slightly per tokenizer, so a few percent of the over-supplied sources stay unused rather than bending the ratio (`unused_tokens` per source in the manifest). Selected docs are shuffled (seeded), concatenated with an EOS separator, chunked into `block_size` blocks (CharBERT `char_ids` packed alongside as `int16`, ~2 GB). The tokenizer fingerprint includes a **content hash** (ids of a fixed probe paragraph), so two BPE-32K tokenizers with different learned vocabs never share an entry; publish is atomic (`tmp` dir + `os.replace`). Cost: ≤ 3 min per tokenizer for HF/char tokenizers, ~11 min araroopat (CAMeL bridge), ~14 min morpho_bpe (Farasa), paid once. **Phases draw consecutive, disjoint block ranges** — `PackedCorpus.take` hands each mix phase the next `mix_tokens / block_size` blocks and **raises (never wraps)** when the packed corpus cannot supply them, naming the `pool.total_words` to set. Because block order is fixed per (pool, tokenizer), Phase 1's slice `[0, n1)` is byte-identical across every experiment on that pool, and enlarging Phase 2 later never touches Phase 1's data. `all_metrics.json["training"][<phase>]["data"]` records the consumed block range and `corpus_share`; `["training"]["pretraining_mix"]` carries the packed manifest (`exact_share_budget`, per-source `available_tokens` / `docs_taken` / `fertility` / `achieved_share`). Measured on the 20 M pool: token shares 0.700/0.200/0.100 in every cell; the same 8.19 M tokens spanned 0.77 M words (Charformer, fertility 10.7) to 8.15 M words (CharacterBERT, 1.0) — equal *compute*, not equal *text*; equal-text is a per-cell `mix_tokens` choice if you want that ablation.

**QA blend (`qa_blend` on a mix phase).** `qa_blend: {datasets: [tydiqa_arabic, arcd], share: 0.07, split: train, seed: 42}` replaces `round(share × mix_tokens / block_size)` of the phase's blocks with SFT-format QA text: the train records are rendered by `_format_qa_full` (the exact Phase 3 / LightEval surface form `السياق: …\nالسؤال: …\nالإجابة: {answer}`, no chat template), encoded without truncation (BOS/EOS as the tokenizer emits them, trailing EOS guaranteed), shuffled, EOS-concatenated and chunked into `block_size` blocks exactly like the pool, then trained with the same full-sequence loss (no answer-only masking — "plain text" is the point). The whole train split is packed once per tokenizer (`pack_qa_blend`, ~1.55 M words = TyDiQA 14 805 + ARCD 693 records, ≈ 100 words each; seconds for HF tokenizers) under `cache_dir/qa_blend/<fp>/` (fingerprint: tokenizer content hash, datasets, split, block_size, seed, `clean_latin_rows` — which now filters the blend records and stays a no-op for the pool); a phase takes the prefix it needs (`PackedCorpus.take`, strict, no wrap — "Lower qa_blend.share" error) and `BlendedBlockDataset` interleaves it with the raw-text slice through a seeded permutation, so every block is read once and every batch is a random mixture. **The share comes out of `mix_tokens`, not on top:** in the sweep Phase 2 = 560 QA blocks + raw-text blocks `[8000, 15440)`, steps stay 2 000, Phase 1's `[0, 8000)` is untouched, and the with/without-blend arms spend identical compute (equal *compute*, not equal *text*: 560 blocks ≈ 2 800 records under CharacterBERT, ≈ 1 700 under BPE-32K, ≈ 260 under Charformer). `split: validation` of `tydiqa_arabic` / `arcd` is refused by the validator (Phase 3 early-stops on those splits); Phase 3 re-seeing the same *train* records with answer-only loss is intentional. Only on a `pretraining_mix` phase; `0 < share < 1`; the rounded block count must be ≥ 1 and leave raw-text blocks. Provenance: `all_metrics.json["training"]["warmup"]["data"]["qa_blend"]` (block range, `achieved_share`, `records_covered_approx`), `["training"]["pretraining_mix"]["qa_blend"]` (packed manifest + consumption), and `data/pretraining_mix/qa_blend_manifest.json` per cell.

**MSA filter — scope, honestly.** The default gate is a closed list of high-precision dialect function words / particles (`DIALECT_MARKERS` in `filters.py`, Egyptian / Levantine / Gulf-Iraqi / Maghrebi / pan-dialect), matched on whole tokens after diacritic-strip + alef/ة/ى folding with an optional و/ف proclitic peeled; a doc is dropped only when it has ≥ `min_markers` (3) markers of ≥ `min_distinct_markers` (2) *types* **and** > `max_markers_per_1k_words` (8). It catches documents *written in* dialect; it does not catch dialect that avoids every listed marker, or light code-switching. Entries that collide with MSA after folding are deliberately excluded (إلى→الي, آية→ايه, بدو, هول, لكان, كيما, تبع, توًّا, هلّا, هون, بس, ما, ليه, عم, زين, …) — a test pins that. **Calibration (2026-09-14, `--calibrate 2000`, i.e. 2 000 raw docs per source):** the gate fired on 0.05 % of Wikipedia, 0.05 % of FineWeb-2 and 0.9 % of ArabicWeb24 (FineWeb-2's GlotLID split already routes Egyptian/Levantine/Maghrebi to `arz_Arab` / `apc_Arab` / `ary_Arab`, so `arb_Arab` is mostly MSA). Every true positive mixed several marker types at ≥ 12/1k (an Idlib first-person narrative, a Moroccan interview, Egyptian song lyrics); every false positive was **one** type repeated — `شو` ×22 as the Japanese name Shū, `مش` ×4 in a quoted song title, `وش` ×3 as "wash" in a shampoo listing — which is why the distinct-type floor exists (regression test in `test_pretraining_mix_pool.py`). CAMeL `DIDModel26` is wired as an **opt-in** second signal (`msa_filter.camel_did.enabled`, `dialect_id` op on the araroopat bridge, lazy-loaded in `.venv-camel`) but **measured unusable as a gate on this domain**: it is MADAR-trained (short travel-domain sentences) and, even aggregated over ≤ 8 sentences with a lenient `min_msa_share` 0.30, it flags 16 % of Wikipedia, 3 % of FineWeb-2 and 8 % of ArabicWeb24 — every flagged sample inspected was plain MSA (biographies, a yacht, a plant, business news) with zero markers. Leave it off unless you have a dialect-heavy source. `DIDModel6` has no pretrained weights for Python 3.10.

**Gotchas.** ArabicWeb24 is gated (auto-approve) — export `HF_TOKEN`; `arabic_101b` (`ClusterlabAi/101_billion_arabic_words_dataset`) is the ungated alternative in `SOURCE_REGISTRY`. FineWeb-2 rows below `min_language_score` 0.80 are skipped at the source; the default was 0.90 until a 1 500-doc probe showed the 0.7–0.9 buckets have the same Latin ratio (0.027 vs 0.024), dialect-marker density (0.2 vs 0.13/1k) and length as ≥ 0.9 — the stricter cut discarded 39 % of the stream for nothing. Calibration keep rates: Wikipedia 36 % (stubs `too_short` 39 %, `short_lines` 16 %, `latin_heavy` 9 % after gloss stripping), FineWeb-2 90 %, ArabicWeb24 84 % (`latin_heavy` 14 % — product/brand pages). A HF streaming iterator that is abandoned on `break` without `close()` aborts the interpreter at exit (`PyGILState_Release`); `build_pool` closes it in its `finally`. Tests: `tests/test_pretraining_mix_pool.py` (filters, dedup, builder with injected sources, config validators incl. `mix_tokens`/`steps` derivation and mismatch) and `tests/test_pretraining_mix_packing.py` (whole-pool tokenization, exact-share budget under different fertilities, packing, strict disjoint slices + no-wrap error, shared cache keyed on tokenizer content, QA-blend packing / cache / interleave / too-small error, and mini 3-phase runs on the packed mix — with and without the blend — for the standard and CharCNN branches).

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

# Overlap between Phase 3 corpora and the four benchmarks (one-off per new corpus).
.venv/bin/python scripts/scan_sft_contamination.py --corpora cidar bactrian_x_ar aya_ar

# Training passages vs the held-out sets (TyDiQA-AR dev, ARCD test, the free-form prompts once
# their path is set in configs/contamination/heldout_sets.yaml); rewrites the committed exclusion
# list. Rerun after a new corpus, a new held-out set or a threshold change, then rebuild the pool.
.venv/bin/python scripts/check_contamination.py --write-exclusions
.venv/bin/python scripts/check_contamination.py --corpora tydiqa_arabic arcd --no-pool     # quick look, no write

# Free-form eval: (re)build the held-out CIDAR set (then rerun check_contamination.py --write-exclusions),
# judge a finished sweep's generations locally (Gemma under vLLM, from .venv-judge) or over an API,
# preview the judge prompt without calling anything.
.venv/bin/python scripts/build_freeform_heldout.py --embed-threshold 0.93
scripts/judge/run_judge.sh --experiment outputs/experiments/<sweep> --judge configs/judges/gemma4_31b_local.yaml
OPENAI_API_KEY=… .venv/bin/python scripts/judge/judge_freeform.py --experiment outputs/experiments/<sweep> --judge configs/judges/gpt56_terra_api.yaml
.venv/bin/python scripts/judge/judge_freeform.py --experiment outputs/experiments/<sweep> --judge configs/judges/gemma4_31b_local.yaml --dry-run

# Held-out loss diagnostic of a cell: raw-text LM loss on 120 pool documents + answer-only NLL,
# P(EOS) and EOS rank-1 on the 250 held-out CIDAR references under the current template
# (--base = the untouched model; --v1-template reproduces pre-2026-09-18 numbers). Standard-embedding
# tokenizers only. Prints a table, writes <cell>/diag_heldout_loss[_base][_v1].json.
.venv/bin/python scripts/diag_heldout_loss.py --cell outputs/experiments/qwen_native_vs_araroopat/native_qwen3_sft
.venv/bin/python scripts/diag_heldout_loss.py --cell outputs/experiments/qwen_native_vs_araroopat/native_qwen3_base --base

# Provenance backfill (one-off, idempotent): created_at from the commit that added each config,
# runs from the console's run.json records. New files and starts are stamped automatically.
.venv/bin/python scripts/backfill_config_provenance.py --dry-run

# Dry-run a phase's mixture (quotas, truncation drops, loss-token shares) without training.
.venv/bin/python scripts/plan_sft_mixture.py --config configs/experiments/native_llama_3phase_sft_mixture.yaml
.venv/bin/python scripts/plan_sft_mixture.py --config <yaml> --plan-only    # pool sizes + quotas only

# Rebuild the per-row eval dump of a finished experiment (eval only, no training).
# --max-rows gives a fast preview; --sweep walks every cell.
.venv/bin/python scripts/dump_eval_rows.py --cell outputs/experiments/<sweep>/<cell> --max-rows 300
.venv/bin/python scripts/dump_eval_rows.py --sweep outputs/experiments/<sweep>

# Migrate the CSV reports of older runs to Parquet (keeps the CSVs unless --delete-csv)
.venv/bin/python scripts/reports_to_parquet.py --dry-run
.venv/bin/python scripts/reports_to_parquet.py

# Pretraining mix (Phase 1/2 raw-text corpus): calibrate filters on N raw docs per
# source, build the cached pool, or print its manifest. Needs HF_TOKEN (ArabicWeb24).
.venv/bin/python scripts/build_pretraining_mix.py --config configs/experiments/all_tokenizers_sweep_pretrain_mix.yaml --calibrate 2000
.venv/bin/python scripts/build_pretraining_mix.py --config configs/experiments/all_tokenizers_sweep_pretrain_mix.yaml
.venv/bin/python scripts/build_pretraining_mix.py --config configs/experiments/all_tokenizers_sweep_pretrain_mix.yaml --report
```

```bash
# AraRooPat training explorer — local web page that replays train() on any typed text,
# step by step, against the real CAMeL bridge (needs .venv-camel). Opens on :8765.
.venv/bin/python debugger/serve_araroopat_explorer.py --open

# Experiment console — local web page to build / validate / save experiment YAMLs in a
# form, list the existing ones, start a run detached from the page and server, watch
# and cancel it. Opens on :8766. Export HF_TOKEN before starting it (runs inherit it)
# and OPENAI_API_KEY for the config assistant (✨ in the Configs tab).
.venv/bin/python debugger/serve_experiment_console.py --open
```

**Experiment console** (`debugger/serve_experiment_console.py` + `debugger/experiment_console.html`, logic in [src/arabic_eval/tools/experiment_console.py](src/arabic_eval/tools/experiment_console.py), added 2026-09-17). The form is generated from the `ExperimentConfig` JSON schema over the *resolved* config (file merged over `base.yaml`), with hand-built widgets for sweep cells / eval tasks / phase cards / model presets; **Validate** is the real `load_config` merge + Pydantic with `loc`-tagged errors painted on the fields; **Save** renders either *full* (every value explicit) or *delta* (differences from `base.yaml`) YAML — a test pins that every file in `configs/experiments/` round-trips identically in both styles. **Clone…** (toolbar or the `⧉` on a list row, added 2026-09-20) copies a saved file under a new name through `clone_config`: only the `name` / `output_dir` / `description` lines of the file's own text are rewritten (comments kept; a rendered delta with `comments_kept: false` when the layout defeats the rewriter), the dialog warns on an existing target, an `output_dir` another config already uses, or unsaved form edits. The `output_dir` row has an **infer from name** switch (`outputs/experiments/<name>`, re-derived on load so campaign cell configs open with it off); the clone dialog and *new from base* use the same convention (`default_output_dir`). **Provenance (added 2026-09-20):** `experiment.created_at` is stamped when a file is first saved (Save… under a new name, Clone…) and `experiment.runs` gains one `{started_at, run_id, source}` per start — the console appends to the source file after launching a run from it, `scripts/run_experiment.py` appends `source: cli` when its `--config` is under `configs/experiments/` (a console run passes the snapshot under `outputs/runs/`, which is skipped). Both are written through [src/arabic_eval/config_edit.py](src/arabic_eval/config_edit.py) — text-level edits of the `experiment:` block that keep comments (the clone rewriter lives there too; an unrecognised layout skips the stamp with a warning, never blocks a run) — shown read-only in the form and the list, and reserved for the tooling (the assistant is told not to touch them). The 18 pre-existing configs were backfilled with `scripts/backfill_config_provenance.py` (git add date; the 7 console `run.json` records). **Start** launches `scripts/run_experiment.py` from a snapshot of the config as a detached session leader (`setsid` + a `TERM`-trapping bash that records the exit code), so closing the page, the SSH session or the server never stops a run; `outputs/runs/<run_id>/{run.json,config.yaml,console.log,exit_code}` is the whole state and a restarted server rediscovers every run (liveness = pid + `/proc` start ticks, then a `/proc` scan of the process group that ignores zombies). `--sweep` is derived like the CLI requires (more than one tokenizer cell) — with a **single** `sweep.tokenizers` cell the run trains the top-level `tokenizer` block and ignores the sweep cell, so Validate / Start name the cell that actually runs and warn when the two blocks disagree; one run at a time unless *run concurrently* is ticked. The Runs tab parses `console.log` into cells / stage / phase step-loss / eval progress / traceback and tails it live; **Cancel** = SIGTERM to the group, SIGKILL after 15 s. The **step panel** (*☰ steps*, added 2026-09-18) is a floating card that shows the selected run's whole plan as an outline — cells, the six pipeline stages, the three phases under *training* and every task under *downstream eval* — with the current stage / sub-step highlighted, finished ones timed from the log's own timestamps (`parse_progress` records `stages` / `phase_times` / `task_times` / the tokenizer sub-step), pending ones greyed with their step budget; the plan (`plan_of`: enabled phases + steps, tasks, eval flags) is written into `run.json` at start and derived from the snapshot for older records. The run list is a floating card at the left edge (*☰ runs*) and the step panel one at the right, so the detail panel — and the log — take the whole width; `console.log` is coloured by Prism's `log` grammar (pinned from cdnjs with SRI; plain text when the CDN is unreachable) plus line tints for WARNING / ERROR heads, Python tracebacks, `Step N/7` headlines and tqdm bars, with a text filter, *hide progress bars* and a taller-log toggle above it. The **Eval rows** tab (added 2026-09-17, [tools/eval_rows_browser.py](src/arabic_eval/tools/eval_rows_browser.py)) browses the per-row dumps: pick experiment → tokenizer cell → benchmark, filter by outcome / scoring / sub-config / flag / free text, and page through every row with the exact prompt, the per-choice scores and the gold-vs-predicted answer. Filtering and paging run off Parquet column projection (no sidecar index); a page of 50 rows touches one row group. The summary strip shows the prediction histogram (class collapse at a glance) and the truncation shares, each clickable as a filter. Details in [debugger/README.md](debugger/README.md); tests in `tests/test_experiment_console.py`.

**Config assistant (added 2026-09-18, [tools/config_assistant.py](src/arabic_eval/tools/config_assistant.py)).** *✨ assistant* in the Configs tab opens a chat drawer: an LLM **creates** a config from a description, **modifies** the one in the form, or **answers** questions about the fields; it never saves or starts anything. Endpoints are `configs/assistant/*.yaml` — any OpenAI-compatible `/chat/completions`; the one shipped is `gpt56_terra_api.yaml` (`temperature: null` + `reasoning_effort: medium`) with a readiness chip (key exported; a localhost `base_url` would be pinged). A local model is possible through the same client (a hand-started `vllm serve` + a YAML with `base_url`), measured once on `google/gemma-4-31b-it`: the ~16 K-token prompt needs `--kv-cache-dtype fp8 --max-model-len 24576` on the H100 and the server takes the whole GPU — not shipped, by choice. Per turn the server builds the prompt (a hand-written platform **primer** with the validator's cross-field rules; the **field reference** rendered from [tools/config_hints.py](src/arabic_eval/tools/config_hints.py) — the `HINTS` table moved out of the page, one source for the form's `?` tooltips and the prompt, served by `/api/schema`; comment-stripped `base.yaml`; registries + tokenizer presets; every experiment YAML as a delta over base; then the **working config as a delta**, its validation state, the last `history_messages` messages; ≈ 12–14 K tokens, static part first for prompt caching) and streams the reply over SSE (`POST /api/assistant/chat`). The model ends a config-changing reply with one fenced **`edits`** block — a YAML mapping of **dotted config paths → values**, replace semantics, list indexes allowed (~10× cheaper than a whole config, exact for a modification; free text + fence rather than constrained JSON). The server applies it, runs the real `validate_config`, and on failure sends the errors back **once** for a repair (visible in the transcript); a valid result goes into the form immediately with **undo**, an invalid one shows its errors and *apply anyway*. *new config from base.yaml* ignores the form; *show context* dumps the exact messages. Verified 2026-09-18 headlessly through every flow on a scripted endpoint and live on `gpt-5.6-terra` (create / modify / question, all valid; one prompt needed the repair round before the `corpus_params` hint was clarified). Tests: `tests/test_config_assistant.py`.

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

**Shortcut for Llama-shaped HF causal LMs.** If the architecture exposes `model.model.embed_tokens`, `model.model.layers`, `model.lm_head`, `config.hidden_size` and an `inputs_embeds` forward (Llama, Qwen2/Qwen3, Mistral, Gemma, …), `LlamaAdapter` already works for every embedding branch — register a thin subclass with a different default checkpoint, the way [models/qwen3_adapter.py](src/arabic_eval/models/qwen3_adapter.py) does (`Qwen3Adapter(LlamaAdapter)` under key `qwen3`, ~20 lines). Pair it with a native-tokenizer wrapper subclassing `NativeLlamaTokenizer` that overrides only `special_tokens` (+ `vocab_size` if the embedding matrix is padded beyond `len(tokenizer)`), as [tokenizers/native_qwen3.py](src/arabic_eval/tokenizers/native_qwen3.py) does. Pin the attribute-surface claim with the offline tiny-model tests in [tests/test_qwen3_support.py](tests/test_qwen3_support.py) (random-init `<Arch>Config` → `save_pretrained` → real adapter; all four embedding branches + a mini 3-phase run + checkpoint round-trip).

For anything else:

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
Tokenizers are trained from scratch. Then: load the base LLM (Llama-3.2-1B/3B or Qwen3-4B-Base — same adapter code path) with pretrained weights intact, replace/resize the embedding layer (`model.model.embed_tokens`) and output head (`model.lm_head`) to match the new tokenizer's vocab size, then fine-tune the full model.

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
- `_forward_charformer` first pads `input_ids` / `attention_mask` / `labels` up to a multiple of `d_s` (pad id / 0 / −100). Without this, an odd byte length yields `L+1` upsampled logits against `L` labels and HF's loss raises `Expected input batch_size (2024) to match target batch_size (2020)` — this is what killed the `charformer` cell of `all_tokenizers_sweep` (fixed 2026-09-14; regression in `tests/test_qwen3_support.py::test_charformer_branch_forward`). The transformer operates on the *downsampled* sequence (length ~L/d_s). `_forward_charformer` then shrinks the byte-level attention mask by OR-reducing windows of size `d_s`, then passes `inputs_embeds` (already downsampled by GBST) to the model. The replaced `lm_head` (`CharformerOutputHead`) upsamples back to byte length via `ConvTranspose1d` so byte-level labels align with logits.
- Auto-regressive `generate()` is **not supported** — GBST pools blocks `X[i:i+b]`, so position `i` sees up to position `i+M-1`. The original Charformer is encoder-decoder, sidestepping causality; in our decoder-only setup, only teacher-forced losses (LM perplexity, LightEval log-likelihood MCQ) are well-defined.
- Mechanical extremes on morphological metrics: each token is one byte, which cannot hold a 3-letter Arabic root (each Arabic letter is 2 bytes). Expect `root_conservation_rate ≈ 0`, `pattern_conservation_rate ≈ 0`. Unlike char-JABER, however, `morpheme_integrity_rate` and `clitic_separation_accuracy` are reported as `None` (not ≈1.0): byte tokens never reconstruct to Arabic-letter offsets, so `aligned_token_offsets` always fails and integrity/CSA are *not measurable*. The discriminating metric for Charformer is `semantic_fragmentation_ratio` (alignment-free, observed ~5.4 on a 240-sentence smoke — the highest in the panel by construction). The token-level inventory metrics (`root_bearing_token_pct`, `pattern_bearing_token_pct`) are explicitly reported as `0.0` (not `None`) when the cleaned-token list is empty but raw tokens were generated; this distinguishes the byte-level mechanical zero from "not measured."

### AraRooPat (Arabic Roots & Patterns)
- The tokenizer file (`tokenizers/araroopat.py`) holds the encode/decode/state machinery; CAMeL Tools integration lives in `tokenizers/araroopat_backend.py` (analyzer + MLE disambiguator + generator + LRU caches + configurable timeout).
- Vocab layout (deterministic ID order): specials → `[LIT_BEGIN]` / `[LIT_END]` → `[PROP_BEGIN]` / `[PROP_END]` (proper-noun markers) → `[CLITICP_*]` (proclitics) → `[CLITICE_*]` (enclitics) → `[PREP_*]` (closed-class prepositions, fixed order) → `[FUNC_*]` (closed-class function words, fixed order) → `[CHAR_*]` → `[DIGIT_*]` → `[PUNCT_*]` → `[ROOT_*]` → `[PAT_*]`.
- **`ة` is an enclitic, never a character** (added 2026-09-13): `_classify_char('ة')` → `"fem"`, no `[CHAR_ة]` exists, and every word-final ة — ROOT+PAT or LIT path — is emitted as `[CLITICE_ة]` (a fixed first slot in the enclitic range). The ة is stripped from the bare pattern (`مَ1ْ2َ3َةِ` and `مَ1ْ2َ3َت…` collapse to one `[PAT_مَ1ْ2َ3َ]`), including the ت-before-pronoun realisation when `surface − clitics == stem + ت` on a nominal (`مدرسته`, `حياته`; not `بيته` / `كتبته`). An alpha chunk absorbs a trailing ة and ends there, so ة can never be mid-chunk (`مدرسةكبيرة` → `مدرسة كبيرة`). Decode rewrites ة→ت before an attached pronoun. Scoped to AraRooPat only — the shared `ARABIC_LETTERS` and the CSA clitic sets are untouched because they drive every tokenizer's metrics. Details in the `araroopat` skill.
- **Closed-class prepositions are one `[PREP_*]` token each** (added 2026-09-13): `من إلى عن على في حتى منذ مذ خلا عدا حاشا متى لعل كي لولا` (`PREPOSITION_INVENTORY`; YAML `prepositions:`; `رب` deliberately excluded). Matched on CAMeL's lemma so `عليه` / `منهم` / `وإلى` resolve to their base with clitics outside (`[CLITICP_و] [PREP_إلى] [CLITICE_ه]`); acceptance is the exact inverse of the decode-side `join_particle_enclitic` (ى→ي before pronouns, `مما/ممن/عما/عمن` assimilation), and it matches the *input* chunk so `علي` (Ali) / `إلي` stay verbatim on the ordinary path. Before this, `من`/`عن`/`في` failed the 3-radical gate and cost four `[LIT_*]`/`[CHAR_*]` tokens each, and `إلى`/`على`/`حتى`/`لعل` got a bogus root+wazn. Requires an araroopat retrain (IDs shift; the pre-pass cache is now keyed on `(_CACHE_FORMAT, prepositions)` and re-analyzes on mismatch). Details in the `araroopat` skill.
- **Closed-class function words are one `[FUNC_*]` token each** (added 2026-09-16): pronouns, demonstratives, relatives, conjunctions, subordinators, interrogatives and the negation / verbal / future particles (CAMeL POS `pron pron_dem pron_rel pron_interrog conj conj_sub adv_interrog part_neg part_verb part_interrog part_fut part`; note CAMeL spells the interrogative tags `pron_interrog` / `adv_interrog`, and tags `ثم` as `adv`, `كيف` as `adv_rel`, `إنّ` as `verb_pseudo`). Same mechanics as `[PREP_*]` — `Analysis.particle` + `particle_kind` (`prep` | `func`), intercepted in `_dict_to_analysis` *after* the prepositions, clitics outside (`[CLITICP_و] [FUNC_هم]`, `[FUNC_أن] [CLITICE_هم]` for `أنهم`, `[CLITICP_ل] [FUNC_قد]` for `لقد`), decode joins a following enclitic with the particle rules. The vocab range sits between `[PREP_*]` and `[CHAR_*]`; the inventory is a *surface* list (`هذه`, `الذين` are listed, not derived from `هذا` / `الذي`); `FUNC_INVENTORY` in the backend is a provisional seed, the curated list is YAML `func_words:` (a surface may not also be a preposition; changing it re-runs the pre-pass, `_CACHE_FORMAT` 5). Before this group these words cost four tokens on the LIT path (`هم هي ما لا الذي هذا قد هل لم لن كم`) or a bogus root+wazn (`أو` → `[ROOT_##ن]`, `إذا` → `[ROOT_#ذ#]`, `لكن`, `ثم`, `كيف`, `سوف`); on 3 000 corpus texts the requested POS set was 8.1 % of all word occurrences. Candidates come from `scripts/discover_araroopat_func_words.py` (CAMeL over every corpus chunk with `--top 3`, every closed-class tag with counts, lemma, clitics and the word's current path; writes `func_candidates.csv`, `func_by_lemma.csv`, `summary.json`). **Both closed groups now match alef-insensitively** (آ/أ/إ/ٱ ≡ ا on both sides; token and decode use the inventory's canonical spelling; ى/ي stay strict): the alef-folded training corpus of the pre-2026-09-16 pipeline never produced `[PREP_إلى]` (`الى` went to `[ROOT_#ل#]`) while raw eval prompts did. **Measured** (`scripts/measure_araroopat_admission.py`, same 300 Arabic-Exam questions, balanced tier, today's code on every tokenizer): `araroopat_hashfix` (2026-08-30) ROOT+PAT 49.7 % / LIT 50.3 %, fertility 4.74, compression 1.01 → `araroopat_func` (2026-09-16, raw hamza text, PREP + FUNC + ة-enclitic + peeler) ROOT+PAT **64.7 %** / PREP 8.5 % / FUNC **6.9 %** / LIT **20.0 %**, fertility **3.59**, compression **1.33**, vocab 8 359. The FUNC share comes straight out of LIT; the ROOT+PAT gain is the sum of everything since Aug 30 (ة collapse, peeler, PREP, no alef fold), not the func group alone. On the full corpus the requested POS set is 17.5 M of 220.7 M word occurrences (7.93 %, 915 surface types, 75 lemmas; 66 % of them were LIT, 33 % bogus ROOT+PAT). Downstream accuracy impact is not yet measured. Discovery artifacts: `outputs/tokenizers/araroopat_func_discovery/`. Tests: `tests/test_araroopat_func_words.py`.
- **Pronoun-hosted prepositions are clitic-only words** (added 2026-09-16): CAMeL reads `له به بها لها لهم لنا لك لي بهم بك` as POS `prep`, lemma `لِ`/`بِ`/`كَ`, pronoun in `enc0` — a preposition that is already a proclitic token plus a pronoun that is already an enclitic token — and every one of them went to LIT (no root to validate; CAMeL tags `prep` on 12.2 % of corpus word occurrences). `_clitic_only_analysis` (backend; `PRONOUN_HOSTING_PROCLITICS = (ل, ب, ك)`, always on) accepts the reading when the alef-folded input equals outer proclitics + lemma + pronoun and `prc1` is free; the lemma becomes `prc1`, `Analysis.clitic_only` / `CorpusEntry.clitic_only` carry the flag (`_CACHE_FORMAT` 6). `_emit_alpha` emits the clitic tokens and nothing else — `له` → `[CLITICP_ل] [CLITICE_ه]`, `ولهم` → `[CLITICP_و] [CLITICP_ل] [CLITICE_هم]` — so `لِ` has one token whether its host is a noun (`لكتاب`) or a pronoun, consistent with `بما` → `[CLITICP_ب] [FUNC_ما]`; the decoder closes buffered proclitics into a word of their own when an enclitic follows them directly (`attach_enclitic`). `بك` / `بي` reach the prep reading through the candidate walk (noun_prop / abbrev rank first). Same day the inventories grew: PREP += `مع تجاه خلال بلا` (CAMeL `prep`; `مع`/`معهم` were LIT, `تجاه`/`خلال` had a root+wazn), FUNC += `فيما` (CAMeL: fused subordinator, lemma `ما`, no clitic split) and `ذات` (CAMeL: noun, root NTWS; keeps its *possessive* — `ذاتها`, `بذاته` — via `POSSESSIVE_FUNC_WORDS`, the only carve-out of the intercept's noun guard). Spatial nouns (`عند بين حول أمام فوق تحت بعد قبل دون غير نحو ضد عبر حسب مثل`) are `noun` for CAMeL with a real root+pattern and stay there. The peeler never lands a clitic on a clitic-only residual (`أبي` is *my father*, not `أ + بِ + ي`; `_CACHE_FORMAT` 7). Explorer / admission script report the path as `CLITIC`. **Measured** (same 300 Arabic-Exam questions): `araroopat_prep` ROOT+PAT 64.6 % / PREP 8.7 % / FUNC 7.0 % / CLITIC 0.5 % / LIT 19.2 %, fertility 3.57, compression 1.34 — the MCQ text has few pronoun-hosted prepositions; on the corpus `ل`+pronoun is 566 K and `ب`+pronoun 362 K occurrences (0.4 %), `مع` 1.07 M, `خلال` 0.40 M. The full CAMeL-`prep` lemma list with counts: `outputs/tokenizers/araroopat_prep_discovery/func_by_lemma.csv` (the only ones left out are `طيلة` 7 K, `حوالى` 3 K — ي-spelled surfaces, `حيال` 1 K). Tests: `tests/test_araroopat_clitic_words.py`.
- **Proper nouns are characters between `[PROP_BEGIN]` / `[PROP_END]`** (added 2026-09-16): a word whose CAMeL reading is a *database* `noun_prop` is a name, and the model gets to know it. The two markers are fixed slots at ids 6–7 (right after the LIT markers); clitics ride outside exactly as on every other path (`بمكة` → `[CLITICP_ب] [PROP_BEGIN] [CHAR_م] [CHAR_ك] [PROP_END] [CLITICE_ة]`, the ة stays the enclitic), and the split is accepted only when the decoder's own joins reproduce the chunk, else the whole chunk goes between the markers unsplit. **CAMeL's backoff is not recognition**: the MLE disambiguator runs the analyzer with backoff `NOAN_PROP`, which stamps `noun_prop` (root `O`, pattern `backoff`) on *every* out-of-vocabulary word — on the corpus that is 150 K of the 162 K `noun_prop` surface types, plain words included (`معيلات`, `ترميز`, `المنقحة`); `is_db_proper` keeps those on the plain LIT path, so LIT now means "unknown to CAMeL" and PROP "a name CAMeL knows". Names without a root (NTWS: `بن كوريا أكتوبر فرنسا أبي إسرائيل`, 6.7 K surfaces, **3.0 %** of corpus word occurrences, all LIT before) become `Analysis.proper` with empty root; names with a root (`الله عبد محمد مصر ابن أبو القاهرة`, 4.6 K surfaces, 2.5 %) keep their root + pattern *and* the flag, and YAML `proper_nouns:` decides: `unrooted` (default) leaves them on ROOT+PAT, `all` sends every name to the markers (they then stop feeding the root / pattern tables; measured cost +1.25 M tokens per 13.2 M words ≈ +2.7 % fertility, `الله` / `محمد` go from two tokens to six). The mode is applied at vocab-build and encode time, never in the pre-pass cache (`CorpusEntry.proper` + root, `_CACHE_FORMAT` 8), so flipping it re-runs only the 8-minute vocab build. A name whose root or pattern the budget cut falls to PROP, never LIT. In `_first_valid` a rootless name yields to a later *closed-class* reading (`بك`: noun_prop "Bey" then بِ+ك) but not to a later root reading (`باريس` then بِ+أَرِيس) — moot on the native path, which fetches one candidate. `peel_compatible` refuses a proper residual (a name never carries an interrogative أ or two pronouns; `أندرسون` used to be tried as أ+ندرسون). Explorer: gate `proper noun: database noun_prop`, category / pre-pass path `prop` (`PROP` counter), `prop` token family; `scripts/measure_araroopat_admission.py` has a `PROP` column; `scripts/discover_araroopat_func_words.py --pos noun_prop` lists the candidates with a `backoff` column (the run behind the numbers above: 40 K texts, 13.16 M occurrences, `noun_prop` top reading on 10.06 % of them — 2.97 % NTWS, 2.50 % rooted, 4.59 % backoff). **Measured** (full corpus pre-pass at format 8: 532 482 analyzed unique chunks, 19 653 proper-noun chunks of which 11 632 rootless; same 300 Arabic-Exam questions): `araroopat_prop` (`unrooted`, vocab 8 365) ROOT+PAT 64.6 % / PREP 8.7 % / FUNC 7.0 % / CLITIC 0.4 % / **PROP 2.9 %** / LIT **16.4 %** (was 19.3 %), fertility 3.57, compression 1.34 — PROP comes straight out of LIT at no fertility cost; `araroopat_prop_all` (`all`, vocab 8 247 — 142 fewer roots, 223 fewer patterns are needed) ROOT+PAT 61.9 % / PROP 5.7 % / LIT 16.3 %, fertility 3.66, compression 1.31. Tests: `tests/test_araroopat_proper_nouns.py`.
- **Clitic combinations the CAMeL DB lacks go through a closed-list peeler** (added 2026-09-13): `calima-msa-r13` has no interrogative-أ prefix row, no second enclitic slot (`enc1` is not a feature) and no lengthened `كمو`/`همو`, so `أتكتب`, `سلمتكها`, `أنلزمكموها` had *no* analysis. `MorphAnalyzer` now runs **native → `peel_candidates` → bare particle → LIT**: on a native miss it strips clitics from a closed list (`أ > (و|ف) > (س|ب|ل|ك) > ال`; one pronoun, or pronoun + 3rd-person pronoun, incl. `كمو`/`همو`), sends the residual to CAMeL, and accepts only if the peel removes something CAMeL cannot represent, the residual's canonical spelling equals the slice (modulo alef variants — ة/ه and ى/ي stay strict), and the clitics fit the residual's POS/aspect (`س` only before an imperfect). Residual readings are fetched with `top=32` (server `MAX_TOP`; the native whole-word path still asks for one) and walked in order, because an unseen word's readings all score 1.0 and rank 1 is database order (`ألزمنا`'s PV+`SUBJ:1P` reading is 6th); the server sorts ties on content so the order is reproducible across processes (it followed string hashing before — a latent native non-determinism too). Lengthened forms are their own `[CLITICE_كمو]`/`[CLITICE_همو]` tokens — no surface is rewritten, so round-trips hold by construction. Stress set 13/37 → 36/37 (`أستكتبه` keeps its native Form X reading by design); false peels 0.7 % of CAMeL-rejected corpus types, every one still round-trips. YAML: `clitic_peeler: true`, `peel_bare_alef: false` (bare `ا` as the interrogative for alef-normalised text — 1.1 % false peels when on, opt-in); `_CACHE_FORMAT` 4. Details, rejected alternatives and open items (native `top>1`, native spelling drift) in `docs/HANDOFF_araroopat_open_issues.md` §6b and the `araroopat` skill.
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
- **Train explorer (local web page)**: `.venv/bin/python debugger/serve_araroopat_explorer.py --open` serves [docs/araroopat_train_explorer.html](docs/araroopat_train_explorer.html) on `localhost:8765`. Type any text → `POST /api/trace` replays `train()` on it via [tokenizers/araroopat_trace.py](src/arabic_eval/tokenizers/araroopat_trace.py) and returns every intermediate (NFKC, char classes → alpha chunks, dedup, the literal NDJSON lines on the CAMeL pipe, every `_dict_to_analysis` gate incl. per-clitic `normalize_pattern` strips and the لِ+الـ contraction, freq tables, `_build_vocab` ID ranges + budget cuts, the three `_build_reconstruction` passes, metadata, then an encode→decode round-trip). The tracer calls the real helpers and asserts its explanation equals the real result; a final step re-runs an un-instrumented `train()` and compares vocab + reconstruction. **LLM emission playground** (below the steps): the server keeps the trained instance (`trace_id`) and `POST /api/decode {trace_id, items}` runs the real `decode()` on any id / token-string sequence — malformed ones included (orphan root, PAT without root, unclosed LIT, unseen pair → tier 2) — returning the final string, `decode(ids[:k])` per prefix (the streaming view), per-pair tiers, generator wire lines and counters (`trace_decode` in `araroopat_trace.py`). **Save as standalone HTML** embeds the trace JSON into a copy of the page (`<!-- @@SNAPSHOT_INJECT@@ -->` anchor) and downloads it; the file opens from `file://` with no server/CAMeL, renders the same steps with playback, and is read-only (`window.__EMBEDDED_TRACE` → snapshot mode). Second tab iframes the older static encode explainer (`docs/araroopat_explainer.html`, untouched). Stdlib `http.server`, no new deps. Explorer defaults `min_root_freq=min_pattern_freq=1` (real default 2) so a short text yields a non-empty vocab — labelled on the page.
- **Corpus training tab (03, added 2026-09-15)**: the same 16 cards on the *real* corpus or a saved tokenizer, via [tokenizers/araroopat_corpus_trace.py](src/arabic_eval/tokenizers/araroopat_corpus_trace.py). `CorpusTraceJob` runs in a background thread on **its own `CamelBridge`** (tab 01 keeps the shared one), polled through `GET /api/corpus/status`. Mode `train` loads `Jr23xd23/ArabicText-Large` with the pipeline's loader + `base.yaml` preprocessing and applies the *real* cache rule (`corpus_analysis.pkl` key match; chunks the cache holds are reused, chunks it lacks go to CAMeL — decision `partial`, the union written back — since 2026-09-16; before, one missing chunk re-ran the whole pre-pass; ≈ 8 min on a hit — 6.5 of them the pure-Python chunk pass — vs hours for a fresh pre-pass; read policy and write flag are exposed separately), then the real `_build_vocab` / `_build_reconstruction` / `_build_metadata`; mode `saved` is `load()`. Every step's `data` carries `_scale` (totals + what is shown) and the cards are corpus-wide totals plus a seeded sample; on a cache hit the IPC / gates / peeler cards **replay the sample live** (`trace_validate_words` / `trace_peel_words`, extracted from the small tracer as shared helpers together with `trace_probe_roundtrip` — tab 01's output is byte-identical) and badge each word against its cached entry; encode → decode runs on a probe text; verify re-runs an un-instrumented `train()` (cache hit) or re-`load()`s. Steps a saved directory cannot reconstruct carry `unavailable`. **Human-readable patterns**: `describe_pattern` gives the wazn (`WAZN_SLOT_LETTERS` 1/2/3/4 → ف/ع/ل/ل, mirrored as `CX_WAZN` in the page — a test pins equality), an approximate gloss from the closed `WAZN_GLOSSES_EXACT` (diacritized wazn) → `WAZN_GLOSSES_SKELETON` (folded) lists, the missing-slot list (weak / `#` radicals) and the first corpus example filled with `naive_pattern_fill`; the page decorates every `.tok.pat` with a `.wz` chip and a three-way *patterns as* switch. **Search**: `TokenIndex.search` (`GET /api/corpus/search`) over id / token prefix / root letters with `#`-and-weak-letter wildcard / wazn / gloss / example words / reconstruction surfaces, plus a live `encode()` of a typed word. Playground (`POST /api/corpus/decode {corpus_id, items}`) and `POST /api/corpus/save` reuse `trace_decode` / `save()`. **Encode / word categories**: `POST /api/corpus/encode` runs the real `encode()` on plain text and classifies every alpha chunk with `categorize` — the all-or-nothing rule of `_emit_alpha` as seven categories (`root_pat`, `root_pat_peeled`, `prep`, `lit_no_analysis`, `lit_root_cut`, `lit_pattern_cut`, `lit_clitic_missing`; a test asserts the category agrees with the emitted tokens for every corpus chunk); `WordCategoryIndex` (train mode, one compact tuple per unique chunk plus root/pattern rank tables) backs `GET /api/corpus/words` (filter by category / substring / peeled, sorted by occurrences); `GET /api/corpus/word_trace?word=` replays one chunk live through `trace_validate_words` / `trace_peel_words`, then the vocab check with the budget loop's reason (`budget_reason`: rank vs max, freq vs min, never a candidate), then `encode()`. The page renders the process with tab 01's `candidateDict` / `gatesView`. **Every record, paged** (2026-09-15): `WordCategoryIndex` keeps every `CorpusEntry` field for every chunk (interned; ≈ 260 MB for the 1.09 M chunks of the full corpus, so the 2.6 GB of entry objects are still dropped) with an exclusive *pre-pass path* per row (`prepass_path`: lit / peeled / prep / root_pat — the validate, peel and entries counters, which therefore sum to the unique count) next to the encode-time category; the counters in those three cards open an in-card server-paged browser (`GET /api/corpus/words?path=&full=1`, aliases `analyzed` / `rejected`, substring filter, 25–200 per page, jump-to-page, `process ▾` per row); the last filtered index list is cached per query key. `FreqIndex` (both modes; `from_counters` in train, `from_metadata` in saved) backs `GET /api/corpus/freq?kind=root|pat|prc|enc|prep&q=&kept=`: every candidate with rank / kept-or-cut / contributing words, root search with the `_root_regex` wildcard, pattern search on slots or wazn. A fixed navigation panel (`#cx-nav`, scroll-spy, collapsible, tab 03 only) lists every card; its entries use `nv-step` / `nv-sec` because the global `.step` rule collapses anything classed `step`. The four browsers and the words card share one `cxBrowser` factory. Tabs 01/02 are untouched (the diff to the page is insert-only; the module never references tab-01 ids — pinned by a test). No snapshot support for tab 03. Tests: `tests/test_araroopat_corpus_trace.py`.
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
- **Alef variant normalization is OFF** (`normalize_alef: false`, flipped 2026-09-16). The fold (آ/أ/إ/ٱ → ا) rewrote 13.4 % of ArabicText-Large's words (`إلى` 65,533 vs `الى` 102 in the raw corpus) into spellings that the Phase 1–3 corpora, the pretraining mix and every eval benchmark never use — a train/eval skew for every from-scratch tokenizer (BPE fragments `إلى` at eval, CharacterBERT's word vocab has no row for it, AraRooPat's `[PREP_إلى]` never fired at training time) while the native Llama/Qwen tokenizers always saw raw text. Tokenizers trained before the flip are not comparable with tokenizers trained after it; retrain the panel before mixing numbers. The knob is now plumbed explicitly (it used to be swallowed by `**kwargs`, so a YAML override was silently ignored); `preprocess_dataset` rejects unknown keys.
- **Preprocessing applies to the tokenizer-training corpus only.** Phase 1–3 QA corpora, the pretraining mix (own `normalize_document`) and the four benchmarks are read raw. `scripts/train_tokenizer.py` applies the same `base.yaml` block as the pipeline (`--base-config`, `--no-preprocessing`); before 2026-09-16 it applied none, so the same tokenizer type came out with a different vocab per entry point. Both entry points write `training_provenance.json` (dataset, resolved preprocessing dict, text count, entry point, params) next to the saved tokenizer — [src/arabic_eval/tokenizers/provenance.py](src/arabic_eval/tokenizers/provenance.py).
- Optional diacritics (tashkeel) removal — controlled by `remove_diacritics` in config
- Tatweel (kashida) removal
- **Lone-و join** (`join_lone_waw`, added 2026-09-16, on by default in `base.yaml`): a و written as a word of its own is glued to the Arabic word after it — `و القمر` → `والقمر` (its tashkeel kept; nothing Arabic may precede it, an Arabic *letter* must follow, so `هو القمر` / `أبو` / `نحو` / `و 2024` / `و HD` are untouched; `و و القمر` → `ووالقمر`). Measured on ArabicText-Large: 351 joins in 330 of 743 K rows, 105 new unique chunks — the shape is rare there (≈ 2 per 1 000 words in the web-text pretraining mix, which has its own `normalize_document` and does **not** apply it). The far more frequent lone-و chunk in this corpus is و glued to a digit or Latin word (`1889 و1988`, `وHD`, ≈ 400 K occurrences): the chunker splits it off and AraRooPat emits it as `[LIT_BEGIN] [CHAR_و] [LIT_END]` — a tokenizer-side matter, not preprocessing. `preprocess_dataset` plumbs every key explicitly and raises on an unknown one. Tests: `tests/test_preprocessing.py`.
- Whitespace collapsing

### Training Loop (`src/arabic_eval/training/phases.py`)
- Step-driven loop (not epoch-driven) — each phase has its own `steps` budget
- AdamW optimizer with cosine / constant / linear LR schedule + optional linear warmup (per-phase); the scheduler steps once per optimizer update, so its horizon is `ceil(steps / gradient_accumulation_steps)` updates (fixed 2026-09-18 — see *3-Phase Training Pipeline*)
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

Results land in `outputs/experiments/native_llama_3phase_{with,no}_sft/all_metrics.json`. **Native vs AraRooPat on Qwen3-4B-Base** (2026-09-18): `qwen_native_sft_only.yaml` (Phase 3 only, mixture 40/30/30 with `drop_truncated_answers`, **LR 2e-5** — full fine-tuning of a 4B model; the 2e-4 of `base.yaml` is a LoRA-scale rate that pushed the raw-text loss from 2.27 to 3.4 nats — warm-up 400 micro-steps = 100 updates, cell `native_qwen3_sft`), `qwen_native_no_presteps.yaml` (the untrained control, free-form eval only, cell `native_qwen3_base` — it used to share the SFT arm's folder and overwrote it) and `qwen_araroopat_3phase.yaml` (Balanced tier, Phase 1 + 2 on the pretraining mix with the 7 % QA blend, the same Phase 3) — single-cell runs whose `output_dir`s are cell folders under `outputs/experiments/qwen_native_vs_araroopat/`, i.e. the sweep layout, so the console's Free-form tab, the judge (`--baseline native_qwen3_sft`; the default picks the first `native_*` cell, name it explicitly now that there are two) and `scripts/compare_results.py outputs/experiments/qwen_native_vs_araroopat/*/` read them as one experiment. `scripts/diag_heldout_loss.py` gives the held-out loss / EOS numbers an arm is accepted on. A per-cell phase toggle does not exist: `run_sweep` overrides only `tokenizer.*` per cell, so arms that differ in *phases* are separate configs. When new tokenizers are added to the suite, run them through the same two configs (overriding `tokenizer.type`) for an apples-to-apples comparison against native_llama.

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
  created_at: "2026-05-05T19:51:30+03:00"   # provenance, written by the tooling (first save); runs: [...] likewise, one per start
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
  eval_row_dump: true           # every scored row -> eval_rows/<task>.parquet (console Eval-rows tab)
  intrinsic_unk_report: false   # dumps per-word UNK list to intrinsic_unks.parquet
  downstream_unk_report: false  # dumps per-task UNK list to unk_reports/<task>_unks.parquet
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
| `datasets` | list of `DatasetName` | registry keys: `arabic_squad` \| `tydiqa_arabic` \| `arcd` \| `arabic_squad_mcq` \| `cidar` \| `bactrian_x_ar` \| `aya_ar` (\| `pretraining_mix`, alone). Single string is auto-coerced to a one-element list. |
| `mixture` | nested config (QA phases only) | `total_examples`, `shares` per category, `within_category`, `weights`, `upsample`, `drop_truncated_answers`, `seed`; derives `steps`. See *Phase 3 mixture*. |
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
| `eval_splits` | `{tydiqa_arabic: dev, arcd: dev}` | which split per corpus; `dev` = title-level 5 % of the official train split, `validation` = the held-out evaluation split (never for selection) |

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
  eval_rows/                # Only when evaluation.eval_row_dump=true (default in base.yaml)
    freeform_cidar.parquet  # free-form task: one row per held-out prompt (instruction, exact prompt,
                            #   reference, generation, stop reason, flags, chrF / BERTScore / round-trip)
    <task_name>.parquet     # EVERY scored eval row: exact prompt, continuations, per-choice
                            #   ll / char / pmi scores, gold + prediction, truncation flags.
                            #   ~18 MB per cell across the four benchmarks. Read by the
                            #   experiment console's Eval-rows tab; pd.read_parquet elsewhere.
  freeform_judge/<judge>.parquet   # written by scripts/judge/judge_freeform.py: one verdict per generation
                            #   (summary merged into all_metrics.json downstream.freeform_cidar.judge.<judge>)
  data/sft_mixture_manifest.json   # Only when Phase 3 has a mixture: quotas, per-corpus drop / kept counts,
                            #   loss-token shares and the record ids drawn (summary without ids in all_metrics.json)
  data/pretraining_mix/     # Only when a phase uses the packed mix: packed_manifest.json pointing at the
                            #   shared entry outputs/data_cache/pretraining_mix/<pool>/packed/<tokenizer_fp>/
                            #   (+ qa_blend_manifest.json → outputs/data_cache/pretraining_mix/qa_blend/<fp>/ when blended)
  training/
    embedding_alignment/    # Phase 1 checkpoint
      model.pt
    warmup/                 # Phase 2 checkpoint
      model.pt
    sft/                    # Phase 3 checkpoint (best-by-eval-loss, restored)
      model.pt
  failure_reports/          # Only when failure_reports=true AND eval_row_dump=false
    <task_name>_accuracy_failures.parquet   # one file per LightEval MCQ task
  intrinsic_unks.parquet    # Only when evaluation.intrinsic_unk_report=true
  unk_reports/              # Only when evaluation.downstream_unk_report=true
    <task_name>_unks.parquet   # one file per LightEval MCQ task
```

Per-phase histories (final loss, train-loss tail, eval losses, wall time, early-stop status, checkpoint path) live under `all_metrics.json["training"][<phase>]`. Per-task downstream metrics + MEI live under `all_metrics.json["downstream"][<task>]` and `all_metrics.json["mei"][<task>]`.

Sweep mode additionally generates: `comparison_report.txt` and `comparison_report.json` in the sweep output directory. The text report includes a "Composite Metric: MEI" section with per-experiment rows + an asterisk-and-footnote on tokenizers with mechanical RPS extremes (`RPS_MECHANICAL_FLAGS` in `evaluation/reporter.py`). For multi-sub-config benchmarks (Alghafa), a "Per-sub-config breakdown" sub-section per task is emitted under the main downstream-task table — rows = experiments, columns = sub-configs, cells = `accuracy (n=N)`. Single-config benchmarks suppress this section; the main table strips `per_subconfig_accuracy.*` so its column count stays manageable.

### Per-row eval dumps (`evaluation.eval_row_dump`, default **true** in `base.yaml`)

Every scored row of every LightEval MCQ benchmark is written to `<output_dir>/eval_rows/<task>.parquet` — **the artifact the experiment console's Eval-rows tab reads**, and a `pd.read_parquet` away from any analysis. One record per row: the **exact prompt string the scorer was handed** (the return of `_format_eval_context_with_fewshot`, few-shot demos included), the continuations, per-choice `ll` / `score_char` / `score_pmi` / `uncond_ll` as native list columns, gold and predicted index and text under each normalization, `margin` (`score[pred] − score[gold]`, hence 0 on every correct row) and `decision_margin` (`top1 − top2`, the coin-flip axis), `prompt_units` measured in the tokenizer's own unit (tokens / words / chars / bytes, named in the file metadata) against `max_length`, and the flags `sentinel` / `all_sentinel` / `hit_cap` / `near_tie` / `disagree`.

`all_sentinel` is the one to watch: when truncation leaves no room for the continuation, `_compute_loglikelihood` returns `SENTINEL_LL` (−1e9) for *every* choice and the row's argmax is an artifact of the cap. Measured on a 400-row sample of the archived `all_tokenizers_sweep_pretrain_mix/charformer` cell: **48.8 % of culture_arabic_mmlu rows**, all 195 of them predicting the same slot. The same cell's median `decision_margin` on ACVA is 5e-6 against 1.6 for `native_llama` — near-uniform continuation scores, a second and independent pathology. Neither is visible in an accuracy number.

Writing is streamed (a row group every 2 000 records) via `EvalRowWriter` in [evaluation/eval_rows.py](src/arabic_eval/evaluation/eval_rows.py), plumbed through the established signature-gating pattern (`row_dump_dir` on `evaluate`, `row_sink` on `evaluate_mcq`; `row_sink=None` keeps the loop byte-identical). A Parquet file has no footer until it closes, so while a benchmark is being evaluated its rows are unreadable — the writer keeps a `<task>.progress.json` beside it (row count + running accuracy) that the console shows instead. Cost: ~18 MB per sweep cell for all four benchmarks, against the ~20 GB of checkpoints the same cell writes.

**Backfill**: experiments that pre-date the flag have no dump. `scripts/dump_eval_rows.py --cell <dir> [--max-rows N]` rebuilds the tokenizer + phase checkpoint and re-runs *only* the scoring (no training), then compares the recomputed accuracy against `all_metrics.json` and reports whether it reproduces. It deliberately does **not** use `LlamaAdapter.load_checkpoint`: that calls `from_pretrained` on the checkpoint dir, which rebuilds a vanilla architecture from its `config.json`, so for CharacterBERT / char-JABER / Charformer cells the custom embedding and output-head weights become unexpected keys and the replaced modules stay randomly initialised. The script adapts to the tokenizer first, then loads the state dict strictly, allowing only a tied `lm_head.weight` to be missing.

### Failure-case reports (opt-in, superseded by the row dump)

When `evaluation.failure_reports: true` **and no row dump is being written**, LightEval MCQ tasks write one Parquet file per task with one row per wrong-answer example. With `eval_row_dump` on, the failure report is skipped and logged — it is the dump filtered to `correct == False`, with the prompt missing. Columns: `index, question, choice_0..N, gold_idx/letter, pred_idx/letter, ll_0..N, ll_margin, score_0..N, score_margin`. Both raw and aggregated views are persisted: `ll_*` is the raw model log-likelihood (sum over continuation tokens) and `ll_margin = ll_pred − ll_gold`; `score_*` is the value passed to `argmax` after `_aggregate_scores` (default char-norm, LightEval `LogProbCharNorm` equivalent), with `score_margin` defined the same way. They differ only when continuation lengths differ — letter-scored MCQ rows have `ll_* == score_*` (1-char continuations); ACVA and word-scored Alghafa rows have `score_*` divided by the character count of the answer text. The margins distinguish confident-wrong (large positive) from near-tie failures (~0); use `score_margin` to interpret the model's actual decision and `ll_margin` to debug the unnormalized signal.

To opt a new task family into failure reporting, just add `failure_report_dir: Optional[Path] = None` to its `evaluate()` signature and handle it — no base-class change needed (signature gating in the pipeline picks it up automatically).

### UNK occurrence reports (opt-in)

Two independent flags surface the per-word data underlying UNK detection, mirroring the failure-CSV pattern:

* **`evaluation.intrinsic_unk_report: true`** — during intrinsic evaluation, every whitespace-split word of the eval split is encoded individually (same loop that produces the scalar `unk_rate`). Words whose per-word encoding contains at least one UNK token id are aggregated into `<output_dir>/intrinsic_unks.parquet` with columns `word, unk_token_count, total_token_count, example_context`. Sorted descending by `unk_token_count`. The scalar `unk_rate` and `vocab_coverage` are byte-identical with vs. without the flag — the CSV is a side-channel, not a recomputation.
* **`evaluation.downstream_unk_report: true`** — during LightEval MCQ evaluation, every prompt (via `_format_eval_context_with_fewshot`) and every continuation per example is scanned. Per-task output `<output_dir>/unk_reports/<task_name>_unks.parquet` with columns `word, unk_token_count, source_fields, num_examples_seen_in, example_context`. `source_fields` is a pipe-joined sorted set (`"continuation_0|prompt"`); `num_examples_seen_in` counts each eval row at most once even if the word recurs across fields.

Both reports always write a file when the flag is on — an empty one carrying the full schema when no UNK was seen, or when the tokenizer's `special_tokens` dict has no `unk_token` entry (byte-level Charformer; Llama under byte-fallback). The empty file is intentional so every (tokenizer, task) pair in a sweep produces a comparable artifact.

**Report file format.** Every row-level report is Parquet, written through `write_report_table` in [utils/io.py](src/arabic_eval/utils/io.py) (column types inferred per column; an empty report still writes the full schema). `write_failure_csv` survives in the same module for *export* paths only — the console's "export this view" button and `scripts/reports_to_parquet.py --to-csv` — where a spreadsheet is the destination and the UTF-8 BOM matters. Measured on this repo's 166 archived CSV reports: 1 047.7 MB → 243.9 MB, 4.3× smaller, and the filter columns of a 21 144-row dump load in ~2 ms against ~103 ms for the equivalent JSONL scan. Migrate older runs with `scripts/reports_to_parquet.py` (keeps the CSVs unless `--delete-csv`).

Helper module: [src/arabic_eval/evaluation/unk_reports.py](src/arabic_eval/evaluation/unk_reports.py) exposes `scan_text` / `aggregate_occurrences` / `records_to_rows` plus the two fieldname constants (`INTRINSIC_UNK_FIELDS`, `DOWNSTREAM_UNK_FIELDS`). Downstream wiring uses the existing signature-gating pattern (`unk_report_dir: Optional[Path] = None` kwarg on `LightEvalBenchmarkTask.evaluate`).

## Dependencies

Core: `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate`, `farasapy`, `pydantic`, `pyyaml`, `numpy`, `tqdm`, `wandb`, `tabulate`, `matplotlib`, `lighteval>=0.6.0`, `datasketch` (MinHash LSH for the pretraining-mix dedup), `pyarrow` (row-level reports; already a `datasets` dependency, imported lazily so tokenizer-only workflows are unaffected)

Tokenizer-only workflows (no GPU): `pydantic`, `pyyaml`, `tokenizers`, `tabulate`, `numpy`, `tqdm`

Optional `[morphological]` extras for full Arabic morphological metrics: `qalsadi`, `pyarabic` (Tashaphyne is pulled in transitively by qalsadi). Install via `pip install -e ".[morphological]"`. Without these, `RootExtractor` falls back to a consonant-skeleton heuristic and `morpheme_integrity_rate` still works via Farasa.

## Known Considerations

- LLaMA 3.2-1B requires HuggingFace access token (gated model). Set `HF_TOKEN` env var or `huggingface-cli login`.
- The free-form judge stage runs from its own venv (`.venv-judge`, vLLM 0.29.0 cu129; setup in *Free-form eval*) through `scripts/judge/run_judge.sh`, after a sweep — `google/gemma-4-31b-it` takes 59 GiB of the H100 and cannot share it with training. The `sacrebleu` dependency (chrF) is required in the main venv; `bert-score` is deliberately not (in-house implementation).
- Qwen3-4B-Base (`Qwen/Qwen3-4B-Base`) is **not** gated; ~8 GB bf16. Full-FT AdamW in bf16 needs ~32 GB for weights+grads+states plus activations — fits an 80 GB H100 at the default 4×4 @ 512 batch (verified by `_smoke_qwen3_4b.yaml`). Expect ~3× Llama-3.2-1B per-step cost, ~15–20 h for a with-SFT run. Smoke-test first: `.venv/bin/python scripts/run_experiment.py --config configs/experiments/_smoke_qwen3_4b.yaml --sweep` (native_qwen3 + bpe@8K cells, 30 steps/phase, 30 eval rows).
- LLaMA 3.2-1B uses **tied embeddings** — `lm_head.weight is model.embed_tokens.weight`. Phase 1's `trainable_parameters: ["embed_tokens", "lm_head"]` is correct; the freezing helper warns when `lm_head` matches no parameter (tied case) and continues. Training `embed_tokens` IS training `lm_head`.
- MorphoBPE (`morpho_bpe`) and FarasaCharacterBERT (`farasa_character_bert`) require Java runtime for Farasa. The segmenter runs in interactive mode for efficiency.
- AraRooPat (`araroopat`) requires `camel-tools` in its own `.venv-camel` (subprocess bridge to avoid `numpy<2` / `transformers<4.54` conflict). One-time setup: `python -m venv .venv-camel && .venv-camel/bin/pip install -e ".[araroopat-camel]" && .venv-camel/bin/camel_data -i light`.
- Character-level tokenizers (char-JABER) produce very long sequences; reduce `max_length` or `batch_size` per phase if hitting OOM.
- The `models/__init__.py` and `tasks/__init__.py` use try/except on imports, so model and task registries will be empty if torch/transformers are not installed. The pipeline script (`pipeline/experiment.py`) force-imports them, so missing deps will surface at experiment runtime.
- PyTorch's **cuDNN SDPA backend is disabled at adapter load** (`configure_sdpa_backends`, see *Free-form eval*): on the H100 it recompiles per (batch, sequence-length) shape on the host and made every first generation batch and every variable-length eval pass minutes slower. `ARABIC_EVAL_CUDNN_SDP=1` re-enables it.
- `torch.amp.autocast` in `phases.py` passes `device_type=adapter.device.type` — works for `cuda`, `cpu`, and most accelerators that PyTorch supports; no extra adjustment needed for MPS.
- CharacterBERT (`character_cnn`) and FarasaCharacterBERT support LightEval log-likelihood scoring via the `character_cnn` branch in `_compute_loglikelihood` (real accuracy in `[0, 1]`, not 0.0). The shared remaining limitation is autoregressive `generate()`, which `LlamaAdapter.generate()` raises `NotImplementedError` for on `CHARACTER_CNN` and `CHARFORMER` — but generation is not exercised by the 3-phase pipeline (eval is teacher-forced log-likelihood MCQ).
- The `dataset_name` defaults (per task class via `_default_dataset_name()`): `acva` → `OALL/ACVA`; `alghafa` → `OALL/AlGhafa-Arabic-LLM-Benchmark-Native`; `culture_arabic_mmlu` → `OALL/Arabic_MMLU`; `arabic_exam` → `MBZUAI/ArabicMMLU`. Override via `params.dataset_name` if a benchmark moves on the Hub.
- The Phase 1 + 2 corpus (`Mostafa3zazi/Arabic_SQuAD`) has **train-only** (no validation split). That's fine — Phases 1 and 2 don't run eval mid-phase. Phase 3 uses TyDiQA-Arabic-val + ARCD-val for stagnation early-stop.
- Full-step experiment wall time on H100: ~5h with SFT (Phase 1 + 2 + 3 + 4 benchmark evals), ~1.5h without SFT (Phase 1 + 2 only + 4 evals).
