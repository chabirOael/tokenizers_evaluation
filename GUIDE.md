# Developer Guide: Arabic Tokenizers Evaluation Platform

## What Is This Project?

This platform answers the question: **does your choice of Arabic tokenizer matter for LLM performance?**

To find out, it trains several tokenizers from scratch, plugs each one into the same language model (LLaMA 3.2-1B), fine-tunes the model, and measures downstream task performance. Because everything else is held constant — same dataset, same model architecture, same training hyperparameters — any difference in results is caused by the tokenizer alone.

```
┌─────────────────────────────────────────────────────────────────┐
│                     The Core Hypothesis                         │
│                                                                 │
│   Fixed:  dataset · model architecture · hyperparameters        │
│   Varies: tokenizer                                             │
│                                                                 │
│   ∴  any difference in downstream score = tokenizer effect      │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Big Picture: One Experiment, End to End

Every run follows the same 7-step pipeline, defined in [src/arabic_eval/pipeline/experiment.py](src/arabic_eval/pipeline/experiment.py):

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Jr23xd23/ArabicText-Large  (HuggingFace)                    │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                 ┌───────▼────────────────────────────────────────┐
          Step 1 │  Load & preprocess dataset                     │
                 │  normalize · strip diacritics · split          │
                 └───────┬────────────────────────────────────────┘
                         │  train_texts / eval_texts
                 ┌───────▼────────────────────────────────────────┐
          Step 2 │  Train tokenizer from scratch                  │
                 │  (or load from load_path)                      │
                 └───────┬────────────────────────────────────────┘
                         │  trained tokenizer
              ┌──────────┴──────────┐
              │                     │
      ┌───────▼───────┐     ┌───────▼────────┐
Step 3│Intrinsic eval │     │  Step 4: Load  │
      │fertility, UNK │     │  LLaMA 3.2-1B  │
      │compression… │     │  + adapt embeds │
      └───────────────┘     └───────┬────────┘
                                    │  adapted model
                            ┌───────▼────────┐
                     Step 5 │  Build task    │
                            │  dataloaders   │
                            └───────┬────────┘
                                    │  train / eval batches
                            ┌───────▼────────┐
                     Step 6 │  Fine-tune     │
                            │  (full model)  │
                            └───────┬────────┘
                                    │  fine-tuned model
                            ┌───────▼────────┐
                     Step 7 │  Downstream    │
                            │  evaluation    │
                            └───────┬────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   all_metrics.json  │
                         └─────────────────────┘
```

A **sweep** repeats this pipeline for every combination of (tokenizer × vocab size × task):

```
  tokenizers: [bpe_16k, bpe_32k, bpe_50k, wordpiece_32k, char_jaber, …]
       ×
  tasks:      [text_generation, question_answering, acva, alghafa, …]
       │
       └──► N experiments run sequentially → comparison_report.txt
```

---

## Project Layout

```
tokenizers_evaluation/
│
├── src/arabic_eval/               ← all library code
│   ├── registry.py                ← plug-in system (3 registries)
│   ├── config.py                  ← Pydantic config + YAML loader
│   │
│   ├── pipeline/
│   │   └── experiment.py          ← run_single_experiment() · run_sweep()
│   │
│   ├── tokenizers/                ← 5 implementations + base + metrics
│   │   ├── base.py
│   │   ├── bpe.py
│   │   ├── wordpiece.py
│   │   ├── morpho_bpe.py
│   │   ├── character_bert.py
│   │   ├── char_jaber.py
│   │   └── intrinsic_metrics.py
│   │
│   ├── models/                    ← LLaMA adapter + embedding layers
│   │   ├── base.py
│   │   ├── llama_adapter.py
│   │   └── embeddings/
│   │       ├── standard.py
│   │       ├── character_cnn.py
│   │       └── char_jaber_embed.py
│   │
│   ├── data/                      ← load · preprocess · collate
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── collation.py
│   │
│   ├── tasks/                     ← evaluation tasks
│   │   ├── base.py
│   │   ├── text_generation.py
│   │   ├── question_answering.py
│   │   └── lighteval_benchmarks.py
│   │
│   ├── training/
│   │   ├── trainer.py             ← fine-tuning loop
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── reporter.py
│   │
│   └── utils/
│       ├── reproducibility.py     ← seed setting
│       ├── logging.py
│       └── io.py
│
├── configs/                       ← YAML experiment configs
│   ├── base.yaml                  ← shared defaults
│   ├── tokenizers/
│   ├── models/
│   ├── tasks/
│   └── experiments/
│       ├── bpe_32k_generation.yaml
│       ├── full_sweep.yaml
│       └── benchmark_sweep.yaml
│
├── scripts/                       ← CLI entry points
│   ├── train_tokenizer.py
│   ├── run_experiment.py
│   ├── evaluate_intrinsic.py
│   └── compare_results.py
│
└── outputs/                       ← generated at runtime (gitignored)
    ├── tokenizers/
    ├── experiments/
    └── data_cache/
```

---

## Core Concept: The Registry

Every extensible component — tokenizers, models, tasks — uses the same plug-in pattern defined in [src/arabic_eval/registry.py](src/arabic_eval/registry.py).

```
  ┌────────────────────────────────────────────────────────────┐
  │  At import time (class definition)                         │
  │                                                            │
  │  @tokenizer_registry.register("bpe")                       │
  │  class BPETokenizer(BaseTokenizer): ...                    │
  │                                    │                       │
  │                          stored in registry dict           │
  │                          { "bpe": BPETokenizer, … }        │
  └──────────────────────────────────────────────────────────┬─┘
                                                             │
  ┌──────────────────────────────────────────────────────────▼─┐
  │  At runtime (pipeline)                                     │
  │                                                            │
  │  config.tokenizer.type = "bpe"          ← from YAML        │
  │  cls = tokenizer_registry.get("bpe")    ← dict lookup      │
  │  tok = cls(**config.tokenizer.params)   ← instantiate      │
  └────────────────────────────────────────────────────────────┘
```

There are three independent registries:

```
  tokenizer_registry   →   bpe · wordpiece · morpho_bpe · character_bert · char_jaber
  model_registry       →   llama
  task_registry        →   text_generation · question_answering · acva · alghafa · …
```

Adding a new component never requires touching the pipeline code — register the class, import it in `__init__.py`, point a YAML at it.

---

## The 5 Tokenizers

All tokenizers share the same interface (`BaseTokenizer`) and differ only in their algorithm and the `embedding_type` they declare.

```
  ┌──────────────────┬────────────────────────────────┬────────────────┬──────────────┐
  │  Registry key    │  Algorithm                     │ Embedding type │  Vocab size  │
  ├──────────────────┼────────────────────────────────┼────────────────┼──────────────┤
  │  bpe             │  Byte-Pair Encoding (HF)        │  standard      │  16K/32K/50K │
  │  wordpiece       │  WordPiece (HF)                 │  standard      │  16K/32K/50K │
  │  morpho_bpe      │  Farasa segments → BPE          │  standard      │  16K/32K/50K │
  │  character_bert  │  Word-level, chars per word     │  character_cnn │  N/A (fixed) │
  │  char_jaber      │  One character = one token      │  char_jaber    │  ~300-500    │
  └──────────────────┴────────────────────────────────┴────────────────┴──────────────┘
```

What `morpho_bpe` does differently before BPE:

```
  Arabic text:  "كتابات الطلاب"
                      │
               Farasa segmenter (Java)
                      │
  Morphemes:   "كتاب + ات  ال + طلاب"
                      │
               BPE on morpheme stream
                      │
  Tokens:      ["كتاب", "##ات", "ال", "##طلاب"]
```

The `BaseTokenizer` contract every implementation must satisfy:

```
  train(texts, vocab_size)          ← learn vocabulary
  encode(text) → TokenizerOutput    ← text → token IDs  (+char_ids if needed)
  decode(ids)  → str                ← token IDs → text
  save(path) / load(path)           ← persistence
  vocab_size   (property)
  embedding_type (property)         ← tells the model which layer to use
  special_tokens (property)         ← {pad:0, bos:1, eos:2, unk:3}
```

---

## Embedding Types: The Critical Link Between Tokenizer and Model

The `embedding_type` property on a tokenizer is the single value that drives which embedding layer the model uses and which collator batches the data. The dispatch lives in `LlamaAdapter.adapt_to_tokenizer()` ([src/arabic_eval/models/llama_adapter.py:61](src/arabic_eval/models/llama_adapter.py#L61)).

```
  tokenizer.embedding_type
         │
         ├── "standard"       ──► resize nn.Embedding to new vocab_size
         │                        input: [B, seq_len]
         │
         ├── "character_cnn"  ──► replace embed_tokens with CharCNN
         │                        input: [B, seq_len, max_char_len]
         │                        ⚠ generate() NOT supported
         │
         └── "char_jaber"     ──► replace embed_tokens with CharJaberEmbedding
                                  input: [B, long_seq_len]  (4-6× longer)
```

### `standard` — subword tokenizers (BPE, WordPiece, MorphoBPE)

```
  "مرحبا بالعالم"
        │  tokenize
        ▼
  [1, 428, 97, 2]        input_ids  [batch, seq_len]
        │
        ▼
  ┌─────────────────────┐
  │  nn.Embedding       │  (resized to new vocab_size)
  │  table: V × hidden  │
  └──────────┬──────────┘
             │  [batch, seq_len, hidden_size]
             ▼
  LLaMA transformer layers (unchanged)
             │
             ▼
  lm_head  →  logits over vocabulary
```

### `character_cnn` — CharacterBERT

```
  "مرحبا"   →   word-level split   →   ["مرحبا"]
                                              │
                              char IDs per word (max 50 chars)
                              [م=5, ر=8, ح=12, ب=7, ا=3, PAD…]
                                              │
                              char_ids: [batch, seq_len, max_char_len]
                                              │
                                              ▼
                           ┌──────────────────────────────┐
                           │  CharCNN embedding           │
                           │  ┌──────────────────────┐   │
                           │  │ char embedding table  │   │
                           │  └──────────┬───────────┘   │
                           │             │                │
                           │  ┌──────────▼───────────┐   │
                           │  │ parallel CNNs         │   │
                           │  │ widths: [1,2,3,4,5,6] │   │
                           │  └──────────┬───────────┘   │
                           │             │  concat        │
                           │  ┌──────────▼───────────┐   │
                           │  │ Highway network (×2)  │   │
                           │  └──────────┬───────────┘   │
                           │             │  projection    │
                           └─────────────┼────────────────┘
                                         │  [batch, seq_len, hidden_size]
                                         ▼
                              Manual layer loop (bypasses HF forward)
                                         │
                                         ▼
                              lm_head  →  logits over word vocab
```

### `char_jaber` — character-level

```
  "مرحبا"
     │  one token per character
     ▼
  [م=5, ر=8, ح=12, ب=7, ا=3]     input_ids  [batch, long_seq_len]
     │                             (4-6× longer than subword)
     ▼
  ┌──────────────────────┐
  │  CharJaberEmbedding  │  small table ~300-500 chars
  └──────────┬───────────┘
             │  [batch, long_seq_len, hidden_size]
             ▼
  LLaMA transformer layers (standard HF forward)
             │
             ▼
  CharJaberOutputHead  →  logits over char vocab
```

---

## Data Flow: From Text to Batch

```
  HuggingFace dataset (Jr23xd23/ArabicText-Large)
           │
           ▼
  ┌─────────────────────────────────────────┐
  │  data/preprocessing.py                  │
  │                                         │
  │  NFKC normalization                     │
  │  Alef variants  →  bare alef (ا)        │
  │  optional: strip diacritics (تشكيل)     │
  │  remove tatweel (ـ)                      │
  │  collapse whitespace                    │
  └──────────────────────┬──────────────────┘
                         │  cleaned texts
           ┌─────────────┴──────────────┐
           │  tokenizer.encode(text)    │
           └─────────────┬──────────────┘
                         │  TokenizerOutput
                         │  .input_ids  [seq_len]
                         │  .char_ids   [seq_len, max_char_len]  (char_bert only)
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  data/collation.py — get_collator(embedding_type)        │
  │                                                          │
  │  embedding_type == "standard"                            │
  │    StandardCollator:     pad input_ids to batch max_len  │
  │                          build attention_mask, labels     │
  │                                                          │
  │  embedding_type == "character_cnn"                       │
  │    CharacterCNNCollator: pad char_ids [B, words, chars]  │
  │                          build word-level attention_mask  │
  │                                                          │
  │  embedding_type == "char_jaber"                          │
  │    CharJaberCollator:    pad long input_ids (max=2048)   │
  └──────────────────────────────────────────────────────────┘
                         │  padded batch dict
                         ▼
              model.forward(batch)
```

---

## Configuration System

Configs are layered YAML files validated by Pydantic models in [src/arabic_eval/config.py](src/arabic_eval/config.py).

### How layers stack

```
  ┌─────────────────────────────────────┐   lowest priority
  │  configs/base.yaml                  │
  │  dataset · model · training · eval  │
  └──────────────────┬──────────────────┘
                     │  deep merge (nested dicts merged, not replaced)
  ┌──────────────────▼──────────────────┐
  │  configs/experiments/my.yaml        │
  │  name · tokenizer · task overrides  │
  └──────────────────┬──────────────────┘
                     │  deep merge
  ┌──────────────────▼──────────────────┐
  │  CLI --overrides                    │   highest priority
  │  e.g. training.learning_rate=1e-4   │
  └──────────────────┬──────────────────┘
                     │  Pydantic validation
                     ▼
             ExperimentConfig
```

### `ExperimentConfig` structure

```
  ExperimentConfig
  ├── name, output_dir, seed
  │
  ├── data         ─── dataset_name, max_train_samples, preprocessing flags
  ├── tokenizer    ─── type (registry key), vocab_size, save_path, load_path
  ├── model        ─── type (registry key), name_or_path, dtype, device
  ├── task         ─── type (registry key), params {}
  ├── training     ─── epochs, batch_size, lr, scheduler, early_stopping…
  ├── evaluation   ─── intrinsic_metrics, downstream_metrics, num_eval_samples
  ├── tracking     ─── use_wandb, wandb_project
  └── sweep        ─── (optional) tokenizers × tasks for grid runs
```

### Example: minimal experiment config

```yaml
# configs/experiments/bpe_32k_generation.yaml
experiment:
  name: "bpe_32k_text_generation"
  output_dir: "outputs/experiments/bpe_32k_generation"

tokenizer:
  type: "bpe"           # ← registry key
  vocab_size: 32000
  save_path: "outputs/tokenizers/bpe_32k"

task:
  type: "text_generation"   # ← registry key
  params:
    max_length: 512
    stride: 256
# everything else comes from configs/base.yaml
```

### Example: sweep config

```yaml
# configs/experiments/full_sweep.yaml
sweep:
  tokenizers:
    - type: "bpe"
      vocab_sizes: [16000, 32000, 50000]
    - type: "wordpiece"
      vocab_sizes: [32000]
    - type: "char_jaber"
      vocab_sizes: [null]        # fixed char vocab, N/A
  tasks:
    - type: "text_generation"
    - type: "question_answering"
```

This generates **3 + 1 + 1 = 5 tokenizer configs × 2 tasks = 10 experiments**.

---

## Evaluation Tasks

```
  task_registry
  │
  ├── "text_generation"     ─── perplexity via sliding-window LM
  │                              lower perplexity = better language model
  │
  ├── "question_answering"  ─── ARCD dataset, generative QA
  │                              scores: F1 · Exact Match
  │
  ├── "acva"                ─┐
  ├── "alghafa"             │├── LightEval multiple-choice benchmarks
  ├── "culture_arabic_mmlu" │   score: accuracy via log-likelihood
  └── "arabic_exam"         ─┘
```

### LightEval benchmark data split

```
  Full benchmark dataset
  │
  ├──  10%  ──►  SFT fine-tuning
  │             (formatted as: السؤال / A. / B. / C. / D. / الإجابة:)
  │
  └──  90%  ──►  Evaluation (never seen during training)
                │
                ▼  log-likelihood scoring
                for each choice X in {A, B, C, D}:
                  score(X) = Σ log P(token | context)
                prediction = argmax over choices
```

### LightEval evaluation flow

```
  MCQ example
  ┌───────────────────────────────────────────┐
  │  السؤال: ما عاصمة المملكة العربية السعودية؟ │
  │  A. دبي    B. الرياض    C. القاهرة    D. بغداد│
  └────────────────────────────────────────────┘
           │                │                │               │
     context+"A"      context+"B"      context+"C"     context+"D"
           │                │                │               │
       log P(A)         log P(B)         log P(C)        log P(D)
           │                │                │               │
           └────────────────┴────────────────┴───────────────┘
                                    │
                              argmax → B  ✓  (correct)
```

---

## Training Loop

[src/arabic_eval/training/trainer.py](src/arabic_eval/training/trainer.py):

```
  for epoch in range(num_epochs):
  │
  ├── for batch in train_dataloader:
  │   │
  │   ├── model.forward(batch)             ← bf16 autocast
  │   ├── loss / gradient_accumulation_steps
  │   ├── loss.backward()
  │   │
  │   └── every 4 steps:
  │       ├── clip_grad_norm(max=1.0)
  │       ├── optimizer.step()             ← AdamW
  │       ├── lr_scheduler.step()          ← cosine + warmup
  │       └── optimizer.zero_grad()
  │
  └── every eval_steps:
      ├── evaluate on eval_dataloader
      ├── log eval_loss
      └── early stopping check (patience=3)
          ├── improved  →  save best checkpoint
          └── no improvement × 3  →  stop training

  LR schedule:
  ┌─────────────────────────────────────────────────┐
  │  warmup (10%)         cosine decay (90%)        │
  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁                               │
  │  ←──10%──→←───────────── 90% ──────────────────→│
  └─────────────────────────────────────────────────┘
```

---

## Intrinsic Tokenizer Metrics

Computed before any model training in [src/arabic_eval/tokenizers/intrinsic_metrics.py](src/arabic_eval/tokenizers/intrinsic_metrics.py):

```
  Arabic text sample: "كتب الطالب المقالة"   (3 whitespace words)
                              │
                       tokenizer.encode()
                              │
  Tokens: ["كتب", "ال", "##طالب", "ال", "##مقال", "##ة"]   (6 tokens)
                              │
  ┌─────────────────────────────────────────────────────────────┐
  │  Fertility         = tokens / words = 6 / 3 = 2.0          │
  │                      lower is more efficient                │
  │                                                             │
  │  Compression ratio = chars / token = 18 / 6 = 3.0          │
  │                      higher = more info per token           │
  │                                                             │
  │  UNK rate          = <unk> tokens / total tokens            │
  │                      lower is better                        │
  │                                                             │
  │  Vocab coverage    = words with zero <unk> / unique words   │
  │                      higher is better                       │
  │                                                             │
  │  Avg token count   = mean tokens per text sample            │
  └─────────────────────────────────────────────────────────────┘
```

---

## Output Files

```
  outputs/experiments/<name>/
  │
  ├── config.json               ← full resolved config snapshot
  ├── intrinsic_metrics.json    ← fertility, compression, UNK rate, coverage
  ├── all_metrics.json          ← intrinsic + downstream combined
  ├── experiment.log            ← full text log
  │
  └── training/
      ├── train_results.json    ← loss curve, step count, wall time
      ├── best/                 ← best checkpoint (early stopping winner)
      ├── final/                ← last checkpoint
      └── checkpoint-*/         ← periodic checkpoints

  outputs/experiments/<sweep_name>/   (sweep mode only)
  ├── bpe_32k_text_generation/        ← one sub-directory per run
  ├── bpe_32k_question_answering/
  ├── wordpiece_32k_text_generation/
  ├── …
  ├── comparison_report.txt           ← human-readable table
  └── comparison_report.json          ← machine-readable
```

---

## CLI Entry Points

All scripts in [scripts/](scripts/) add `src/` to `sys.path` — no install required for development.

```bash
# Train only a tokenizer (no GPU needed)
python scripts/train_tokenizer.py --type bpe --vocab-size 32000

# Run one experiment
python scripts/run_experiment.py --config configs/experiments/bpe_32k_generation.yaml

# Run a full grid sweep
python scripts/run_experiment.py --config configs/experiments/full_sweep.yaml --sweep

# Run benchmark sweep (ACVA, Alghafa, etc.)
python scripts/run_experiment.py --config configs/experiments/benchmark_sweep.yaml --sweep

# Evaluate a saved tokenizer's intrinsic metrics only
python scripts/evaluate_intrinsic.py --tokenizer-path outputs/tokenizers/bpe_32k --type bpe

# Compare results from multiple experiments
python scripts/compare_results.py outputs/experiments/*/
```

---

## How to Add Something New

### New tokenizer

```
  1. src/arabic_eval/tokenizers/my_tok.py
     └── class MyTok(BaseTokenizer):
             embedding_type = "standard"   # or new type
             train() / encode() / decode() / save() / load()

  2. @tokenizer_registry.register("my_tok")  ← add decorator

  3. src/arabic_eval/tokenizers/__init__.py
     └── from .my_tok import MyTok           ← triggers registration

  4. configs/tokenizers/my_tok.yaml          ← defaults for this tokenizer

  5. If new embedding type needed:
     ├── add models/embeddings/my_embed.py
     ├── add EmbeddingType.MY_TYPE constant in tokenizers/base.py
     ├── add branch in LlamaAdapter.adapt_to_tokenizer()
     └── add branch in data/collation.py get_collator()
```

### New task

```
  1. src/arabic_eval/tasks/my_task.py
     └── class MyTask(BaseTask):
             name = "my_task"
             metric_names = ["accuracy"]
             get_dataloader(tokenizer, split, …) → DataLoader
             evaluate(model, tokenizer, …) → dict

  2. @task_registry.register("my_task")

  3. src/arabic_eval/tasks/__init__.py
     └── from .my_task import MyTask

  4. configs/tasks/my_task.yaml
```

### New LightEval benchmark (multiple-choice)

```
  Subclass LightEvalBenchmarkTask — 10/90 split, SFT loop, and
  log-likelihood eval are all inherited. Only provide:

  ├── _default_dataset_name() → str
  ├── _parse_example(example) → (question, choices, answer_idx)
  └── name (property) → str
```

---

## Setup and Requirements

```bash
# Full install (GPU training + evaluation)
pip install -e .

# Tokenizer-only work (no GPU needed)
pip install pydantic pyyaml tokenizers tabulate numpy tqdm
```

```
  Requirements
  ├── Python ≥ 3.10
  ├── GPU (CUDA)          — model training & evaluation
  ├── HF_TOKEN env var    — LLaMA 3.2-1B is a gated model
  └── Java runtime        — morpho_bpe only (Farasa uses JVM)
```

---

## Known Limitations

```
  ┌──────────────────────┬────────────────────────────────────────────────┐
  │  Component           │  Limitation                                    │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │  character_bert      │  generate() raises NotImplementedError.        │
  │                      │  QA → empty predictions. LightEval → 0.0.      │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │  char_jaber          │  Sequences 4-6× longer → higher memory usage.  │
  │                      │  Reduce max_length or batch_size if OOM.       │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │  Non-CUDA devices    │  Trainer uses device_type="cuda" in autocast.  │
  │                      │  Needs adjustment for MPS or other hardware.   │
  ├──────────────────────┼────────────────────────────────────────────────┤
  │  Dataset paths       │  culture_arabic_mmlu and arabic_exam names are │
  │                      │  best-effort — verify on HuggingFace Hub first.│
  └──────────────────────┴────────────────────────────────────────────────┘
```
