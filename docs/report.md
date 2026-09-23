# Qwen3-4B-Base: native tokenizer vs AraRooPat — campaign record (2026-09-18 → 2026-09-22)

Working notes for the comprehensive report. Every number below was measured on this machine (H100, `Qwen/Qwen3-4B-Base`); the artifact that holds it is named. Free-form numbers are comparable only across cells scored under the same decoding rules — the rule set is stated per table.

## 0. Set-up

- **Question.** Does a morphological tokenizer (AraRooPat: `[ROOT_x] [PAT_y]` + clitic tokens) beat the native Qwen3 vocabulary on Arabic generation when the 3-phase pipeline is held fixed?
- **Arms.** A = native Qwen3 tokenizer (vocab 151 936, no training or Phase 3 only); B = AraRooPat (from-scratch vocab, Phase 1 + 2 on the pretraining mix, Phase 3). From 2026-09-22: C = from-scratch BPE-16K, the fair comparator.
- **Phase 3.** Mixture of 30 000 records at 40 % extractive (TyDiQA-AR + ARCD) / 30 % MCQ (synthetic Arabic-SQuAD) / 30 % free-form (CIDAR, Bactrian-X ar, Aya ar); in loss tokens the free-form share is 80–87 %. Early stop on the extractive `dev` slice until 2026-09-22; since then (§3.6) on a 1 000-record `dev` mixture at the training ratio, scored per token.
- **Free-form eval.** 250 held-out CIDAR instructions (E5 near-duplicate gate 0.93), pure greedy decoding, stop at EOS / template marker / text-level loop, character budget per tokenizer. Metrics: chrF, in-house BERTScore, loop / cap / EOS rates, and a 1–5 judge (`google/gemma-4-31b-it` under vLLM, reference-guided rubric, paired bootstrap against a baseline cell). Held-out loss diagnostic (`scripts/diag_heldout_loss.py`): raw-text LM loss on 150 held-out FineWeb-2 documents (`rawtext_heldout_v1.jsonl`; until 2026-09-22 the last 120 documents of each cell's own pool — not held out, see §3.6) and answer NLL / P(EOS) on the 250 references; answer NLL per character is Σ NLL ÷ the references' 72 705 characters — since 2026-09-23 the script emits it itself (`heldout_answers.nll_per_answer_char`, Σ NLL ÷ the characters *as encoded*: 72 636 for AraRooPat, whose normalization drops 70 zero-width format characters and NFKC-expands one, 72 705 for native and BPE; `nll_per_answer_char_raw_chars` divides by 72 705 for every tokenizer), and `scripts/diag_backfill_per_char.py` added both fields to the 19 earlier JSONs (§3.7).
- **Experiment folder.** `outputs/experiments/qwen_native_vs_araroopat/` (cells listed in §9); superseded cells under `_superseded/`.

## 1. Timeline

| date | step | outcome |
|---|---|---|
| 09-18 | first native SFT arm | 249/250 answers hit the cap, 99 % loops, chrF 9 — no EOS in SFT records (bug, fixed) |
| 09-18 | loop investigation on the untrained base | 44.4 % degenerate under greedy; format moves the rate (16–49 %), does not remove it |
| 09-18 | training fixes | scheduler horizon, LR 2e-4 → 2e-5, template v2, cuDNN SDPA off, `drop_truncated_answers`, diagnostic script |
| 09-18 | native control + SFT (v2 template, first loop stop) | base loops 27.2 %, SFT 23.2 % — the loop stop itself was wrong |
| 09-18 | AraRooPat v1 | degenerate 50.8 %, chrF 7.6, judge 1.06 |
| 09-20 | loop stop rewritten (periodic tail) | control loops 27 % → 6.8 %; 56 of 68 old stops were false |
| 09-20/21 | control re-run | byte-identical (greedy eval is deterministic) |
| 09-21 | exact stop markers + 2 400-char budget | 5 of 5 marker stops had been false; reference cells `*_m2_c2400` |
| 09-21 | console: typed params, re-eval, compare | found arm B's config mismatch (10 000 patterns, LR 2e-4) |
| 09-21 | AraRooPat v2 | judge 1.07, chrF 9.8, LM loss 1.34 nats/char |
| 09-22 | deep analysis + literature | cause: arbitrary embedding init + 37 M-token budget |
| 09-22 | decode canonicalization, `surface_avg` init, 131 M-token v3, BPE-16K comparator | v3 judge 2.00 (7/8 thresholds), BPE-16K 1.99 — a tie |
| 09-22 | held-out raw-text set for the loss diagnostic | the old "held-out" loss was each arm's own training pool; LM gap to native 0.13 → 0.03 nats/char |
| 09-22 | free-form dev slices + token-weighted mixture early stop; Phase 3 re-run of both arms | AraRooPat 5 600 → 24 000 examples, judge 2.00 → **2.15** (Δ +0.15, CI excludes 0); BPE unchanged (+0.004) — the control |

## 2. The native arm

### 2.1 First SFT run — the EOS bug (2026-09-18)
`tokenize_record` never appended EOS and the native Qwen3 wrapper emits no special tokens, so the SFT model never saw an answer end: 249/250 generations ran to the 1 200-char cap, 99 % repetition loops, chrF 9. Fix: every QA record ends in the tokenizer's EOS (not truncated records; the EOS is a loss target). Native SFT results before this fix are not comparable on anything generative.

### 2.2 Loop investigation on the untrained base (2026-09-18)
Same 250 prompts, pipeline greedy rules, v1 flat template (`السؤال: … الإجابة:`), cap 609 tokens: **degenerate 44.4 %**, cap 38.4 %, chrF 17.6 (cell `native_qwen3`).

Prompt-format ablation (untrained base, loop rate): `السؤال:/الإجابة:` 44 % · `التعليمات:` 30 % · one-line hint 41 % · **Arabic-Alpaca `### التعليمات / ### الإجابة` 23 % (EOS 87 %, chrF 22.7, best)** · 3-shot 16 % (but 71 % end by writing the next block) · ChatML on the base 49 % (27 % Latin/Chinese drift). 7 of 250 prompts loop under all six formats, 44 under none: the loop is a greedy attractor, not prompt-intrinsic. Same prompt with `repetition_penalty=1.2`: 8 %; nucleus p 0.9 / T 0.7: 35 % — both rejected as decoding rules because they are tokenizer-granularity-dependent (a character tokenizer repeats characters by nature). Self-reinforcement measured: P(cycle continues) 0.51 → 0.77 → 0.96 → 0.98 → 0.99 over repetitions 1–5, median lock at step 20. Template ambiguity: all three Phase-3 templates ended in `الإجابة:` and 70 % of the mixture taught ≤ 4 words after it.

Literature used: Xu et al. 2022 (self-reinforcement of repetition, DITTO), Ivgi et al. 2024 (repetition as an uncertainty fallback), Holtzman et al. 2019 (greedy degeneration), Fu et al. 2021, Li et al. 2023, Hiraoka & Inui 2025, Sclar et al. 2023 (format sensitivity); the Qwen3 model card ("greedy → endless repetitions"); the CIDAR authors decoded with T 0.2 + repetition penalty 1.2.

### 2.3 Training fixes (2026-09-18)
- **Scheduler horizon**: `steps` are micro-steps but the scheduler stepped per update, so at accumulation 4 the cosine covered a quarter of the phase (lr 1.79e-4 at step 6 800 of 7 500 for a 2e-4 peak). Horizon now `ceil(steps / accum)`.
- **Learning rate** 2e-4 → **2e-5** for full fine-tuning of the 4B model: at 2e-4 the native model's raw-text loss went 2.27 → 3.4 nats/token; the AraRooPat v1 SFT model's to 3.86. Warm-up 400 micro-steps (100 updates).
- **Prompt templates v2**: Arabic-Alpaca header + `### التعليمات:` / `### المدخل:` / `### الإجابة:` (free-form) and `### السياق:` / `### السؤال:` / `### الإجابة:` (extractive); the eval prompt is built by the same function.
- **cuDNN SDPA backend off**: per-shape host compile made the first generation batch ~300 s (58 s vs 3.8 s measured at 150 new tokens).
- **`drop_truncated_answers: true`** in the mixture: a truncated record has no EOS (8.3 % of free-form records under AraRooPat at 1.9 chars/token, 2.4 % native).
- `scripts/diag_heldout_loss.py` as the acceptance instrument.

### 2.4 Native control and SFT under the v2 template, first loop stop (2026-09-18, runs 20260918-183901 / -184653)
Untrained base: mix loss 2.248 nats/token = 0.824 nats/char, answer NLL 1.840/token, P(EOS at end) 0.42, EOS rank-1 76 %; free-form loops 27.2 %, EOS 71.6 %, chrF 22.9, BERTScore-F1 0.857, 348 chars. SFT at 2e-5 (early stop at 4 800 of 7 500, best dev 0.1553 at 3 800): mix 2.281 (no damage), answer NLL 1.748, P(EOS) 0.64, rank-1 91 %; loops 23.2 %, EOS 76.8 %, chrF 18.2, 154 chars; MCQ (PMI) ACVA 0.560 / AlGhafa 0.591 / Arabic-Exam 0.555 / Culture-MMLU 0.469. The free-form micro-batch loss stayed flat (~1.63) through the phase.

### 2.5 The loop stop was measured wrong and rewritten (2026-09-20)
The 09-18 rule ("a word 4-gram occurs 3× anywhere, cut before the second copy") truncated ordinary answers: 56 of the control's 68 loop stops and 31 of the SFT arm's 57 were numbered lists, markdown tables, question restatements or إعراب templates (15 inspected by hand: 13 legitimate). Canonical case `cidar-10013`: a 632-char answer cut to 78. New rule: a periodic tail (the text ends in ≥ 5 / 4 / 3 contiguous copies of a unit of period 1 / ≤ 3 / ≥ 4 words, up to 60 words) or one letter/digit 20× in a row; cut after the first copy; `degenerate` judged on the raw text. Corrected numbers (1 200-char budget): control degenerate 9.6 %, loop stop 6.8 %, EOS 86.4 %, cap 4.8 %, chrF 23.14, 441 chars; SFT degenerate 14.8 %, loop 12.4 %, EOS 86.4 %, chrF 18.62, 196 chars, 5.2 % of answers ≤ 3 words. Non-loop rows reproduced byte for byte.

### 2.6 Determinism (2026-09-20/21)
A re-run of the control was identical on all 250 generations, stop reasons and metrics. Greedy evaluation is deterministic here and served as the regression oracle for every later rule change (245/250 rows identical after the marker change — the 5 marker rows —, 223/250 after the budget change — the 27 cap/truncated rows).

### 2.7 Exact stop markers and the 2 400-char budget (2026-09-21)
All 5 marker stops of the control were the model opening its own markdown sub-header (`### مثال متكامل:` …), not a new prompt block; the bare `\n###` marker was replaced by the templates' exact labels and header openings. Three freed rows gained chrF (cidar-10013 29.4 → 37.3). Budget 1 200 → 2 400 chars: of the control's 12 cap rows 6 finish with EOS at 2 400 and the 11 finished-but-cut rows are no longer cut; wall ×1.38. Trap found on the way: `configs/tasks/*.yaml` are console presets a run never reads (decoding params live in `sweep.tasks[].params`); two GPU runs with an edited preset were byte-identical to the control. Reference cells: `native_qwen3_base_m2_c2400` (run 20260921-091458) and `native_qwen3_sft_m2_c2400` (20260921-093120).

| cell (2 400 chars, exact markers, periodic loop stop) | judge | chrF | loop stop | EOS | cap | mean chars | LM loss nats/char | answer NLL nats/char |
|---|---|---|---|---|---|---|---|---|
| native base (untrained) | 2.42 ± 0.09 | 22.94 | 6.8 % | 90.8 % | 2.4 % | 511 | 0.82 | 0.81 |
| native SFT | 2.28 ± 0.09 | 18.59 | 12.4 % | 87.2 % | 0.4 % | 207 | 0.84 | 0.77 |

Paired judge SFT − base: −0.14 [−0.30, +0.02]. Register: the base writes markdown-assistant style (44 % bold, 22 % numbered lists vs 1 % in the references); SFT removed it entirely and learned the references' terse length (207 chars) but not their content (18.8 % of answers ≤ 40 chars).

### 2.8 Tooling (2026-09-21)
Typed task and tokenizer parameter specs in the console (the preset trap cannot recur: only `sweep.tasks[].params` is read, presets are generated from the specs and pinned by a test), re-eval-from-checkpoint and config compare. Compare exposed arm B v1's real config: 10 000 patterns (top-level block) instead of the sweep cell's 4 000, Phase 3 at LR 2e-4 / warm-up 100 / `drop_truncated_answers: false`.

## 3. The AraRooPat arm

### 3.1 v1 (2026-09-18, cell `araroopat`)
Config as run: max_patterns 10 000 (vocab 14 365, fertility 3.27, 1.92 chars/token), Phase 1 4.1 M mix tokens at 5e-4 (235 s), Phase 2 4.1 M at 2e-4 with 7 % QA blend (349 s), Phase 3 mixture 30 000 at 2e-4 (early stop at 6 800, 2 011 s). Results (v1 rules): degenerate 50.8 %, cap 46 %, EOS 54 %, chrF 7.6, 585 chars, reference round-trip chrF 89.2; judge (scored 2026-09-22) 1.06, repetition 48 %, off-topic 54 %. Diagnostic: mix loss Phase 1 3.44, Phase 2 2.67 (1.35 nats/char), SFT 3.86 nats/token; answer NLL 2.66 → 2.69; P(EOS) 0.08 → 0.35. Loop trace: ~20 uncertain steps then a 3–16-token cycle locked with top-1 ≥ 0.98; `? ? ?` loops are period-1 `<unk>`; 15–22 % of answers contained `<unk>`.

### 3.2 v2 (2026-09-21, cell `araroopat_3phase_v2`)
Recipe: 40 000 patterns (every pattern of frequency ≥ 2; vocab 17 074), Phase 1 16.4 M tokens at 5e-4 (loss 2.78, 942 s), Phase 2 20.5 M at 2e-5 + 7 % blend (loss 2.48, 1 756 s), Phase 3 as the native arm plus `drop_truncated_answers` and `upsample` (the first attempt raised `MixtureShortfallError`: extractive kept 11 836 of 12 000 — 1 984 + 101 truncation drops, 239 cut answers; Phase 3 resumed from the Phase 2 checkpoint; early stop at 6 400, best dev 0.176 at 5 400).

Results (current rules): judge **1.07 ± 0.02** (235 of 250 scored 1; off-topic 72 %, repetition 12 %; paired Δ vs native base −1.34 [−1.54, −1.17], wins 0.8 %), chrF 9.77, loop stop 27.6 % (periods 1–3 dominate), EOS 72.4 %, cap 0, **mean answer 54 chars** (53.6 % ≤ 40 chars), 32 % of generated words copied from the instruction (native 14 %). Diagnostic: mix loss **1.29 nats/char after Phase 2, 1.34 after SFT** (native 0.82); answer NLL 2.56/token = **1.27/char** (native SFT 0.76); P(EOS) 0.06 → 0.49; SFT slightly damaged the LM (2.615 → 2.708/token). Phase 2 loss flat from step ~2 000 of 10 000. Token families of the generations match the references (ROOT+PAT 21 % vs 23 %, literal-path words 5.5 % vs 6.4 %): the tokenizer emitted well-formed sequences that decoded to grammatical, empty text. Decode ceiling: reference round-trip chrF 89.1, 95.8 with the references' diacritics stripped (rest: U+200B → `?`, hamza-seat and ة/ه drift in the reconstruction surfaces); chrF ignores whitespace, so the space-before-punctuation decode costs nothing.

### 3.3 Judge over every cell (2026-09-22, gemma-4-31b, baseline `native_qwen3_base_m2_c2400`)

| cell | rules | judge ± SE | histogram 1/2/3/4/5 | off-topic | repetition | paired Δ vs baseline |
|---|---|---|---|---|---|---|
| `native_qwen3` (untrained, v1 template) | v1 | 1.94 ± 0.09 | 148/36/24/18/24 | 11 % | 40 % | −0.48 [−0.65, −0.32] |
| `native_qwen3_base` | 09-18 | 2.38 ± 0.09 | 88/77/31/11/43 | 14 % | 4 % | −0.04 [−0.10, +0.02] |
| `native_qwen3_base_m2_c2400` | 09-21 | 2.42 ± 0.09 | 92/69/29/13/47 | 13 % | 10 % | baseline |
| `native_qwen3_sft` | 09-18 | 2.26 ± 0.09 | 103/66/29/18/34 | 14 % | 9 % | −0.16 [−0.33, 0.00] |
| `native_qwen3_sft_m2_c2400` | 09-21 | 2.28 ± 0.09 | 105/60/31/18/36 | 14 % | 16 % | −0.14 [−0.30, +0.02] |
| `araroopat` (v1) | 09-18 | 1.06 ± 0.02 | 239/7/4/0/0 | 54 % | 48 % | −1.36 [−1.54, −1.18] |
| `araroopat_3phase_v2` | 09-21 | 1.07 ± 0.02 | 235/14/0/0/1 | 72 % | 12 % | −1.34 [−1.54, −1.17] |

### 3.4 Diagnosis (2026-09-22) and the literature behind the fix
The failure was the language model, not the tokenizer or the eval: `resize_token_embeddings` keeps the base model's first N rows (ASCII, bytes, English pieces) as the new vocabulary's embeddings — an arbitrary mapping equivalent to random initialization — and 37 M tokens of adaptation is two to three orders of magnitude below what the literature spends after a vocabulary swap.

| paper | what we took from it |
|---|---|
| Mundra et al. 2024, *An Empirical Comparison of Vocabulary Expansion and Initialization Approaches* (arXiv 2407.05841) | random initialization produces "gibberish"; LLaMA-2 7B adapted with 2.5 B CPT tokens; initializations inside the convex hull of the base rows preserve behaviour |
| TokAlign (ACL 2025, arXiv 2506.03523) | full vocabulary replacement; aligned init perplexity 120 vs 340 (FOCUS, ZeTT); ~5 000 steps × 2 M tokens ≈ 10 B tokens to restore vanilla performance; progressive unfreezing (embeddings first) |
| Model-Aware Tokenizer Transfer (arXiv 2510.21954) | recovers a swapped tokenizer with ~50 M tokens per language only by distilling attention patterns from the original model; > 50 % of the gain within the first 5 M tokens |
| *Beyond Initialization Loss* (arXiv 2608.03494) | subword-composition init with asymmetric input/output weighting cuts CPT steps 6×; step-0 loss is a poor predictor — a 50-step CPT probe ranks methods reliably |
| Fast Vocabulary Transfer (Gee et al., arXiv 2402.09977) | new token = average of the old tokenizer's embeddings of its pieces; competitive and cheap |
| *Morphemes Without Borders* (LREC 2026, arXiv 2603.15773) | tokenizer morphological alignment is neither necessary nor sufficient for morphological generation in Arabic LLMs |
| *Exploring Tokenization Strategies and Vocabulary Sizes for Arabic LMs* (arXiv 2403.11130) | BPE over Farasa segmentation best of four; vocabulary size has limited impact at fixed model size |
| ReTok (Gu et al. 2024) | tokenizer replacement by retraining only the embedding and output layers |

### 3.5 v3 (2026-09-22): the three fixes and the run
**Decode canonicalization** (commit `e06910e`). CAMeL folds hamza, ى and ة before its lookup and its disambiguator ignores spelling: 91 355 of 519 330 rooted corpus chunks (17.6 %) had a surface that did not spell the word (`الإعرابية` → the Bedouin adjective `الأعرابية`; `همزة` → the verb `همزه` + a 3ms pronoun). Fix: prefer the first candidate that spells the word, reconcile a lone ة/ه slot with the written letter, reconstruction surfaces = the most frequent written stem validated by the decoder's own joins, drop format characters at encode; a pre-existing `كان` → `[FUNC_كأن]` bug fixed in the same commit; cache format 8 → 10 with a 4-minute migration. Measured on `araroopat_maxpat40k_v2 → _v3` (vocab 17 074 → 17 184): reference round-trip chrF 89.11 → **91.10** raw, 96.71 → **98.97** diacritics-stripped; exact `decode(encode(w)) == w` on 2 000 corpus words 96.40 % → **99.30 %** (ة/ه 33 → 0, hamza 29 → 13, ى/ي 7 → 1).

**Informed embedding initialization** (commit `b204a54`). `model.embedding_init.method = surface_avg`: each new token's row is the weighted average of the base model's embeddings of the surfaces the token stands for (root / pattern tokens → the reconstruction surfaces sharing that root / pattern, closed-class tokens → their surface, plus space-prefixed variants); 17 172 of 17 184 rows from surfaces, 12 fallbacks (specials and markers). Probe (`scripts/probe_embedding_init.py`; step-0 held-out loss, then 50 Phase-1 updates, nats/token):

| method | step 0 | step 50 |
|---|---|---|
| legacy (first-N rows, v1/v2) | 8.32 | 5.28 |
| random N(0, 0.02²) | 12.95 | 5.80 |
| mean of the base matrix | 9.75 | 5.65 |
| surface_avg, uniform | 9.11 | 3.62 |
| **surface_avg, char_len** | 8.92 | **3.57** |
| surface_avg, char_len + norm calibration | 14.07 | 4.02 |

Step-0 values are uninformative under tied embeddings (all within 1.4 nats of ln 17 184 = 9.75; the rows are also the output head); the 50-update result separated the methods by 1.7 nats, as the 2026 study predicts.

**Run** `20260922-045152_qwen_araroopat_3phase_v3`, cell `araroopat_3phase_v3`: v3 tokenizer loaded (`load_path`), `surface_avg / char_len`, 50 M-word pool (fingerprint `d9d1b696d93e6bd9`, 148.7 M tokens of this vocab), Phase 1 32.8 M tokens (8 000 updates, embeddings only, LR 5e-4, 31 min; loss 3.34 → 1.96 — v2's legacy init: 5.72 at step 50, 2.78 at 4 000), Phase 2 98.3 M (48 000 micro-steps, LR 2e-5 cosine, 5 % QA blend, 140 min; window means 1.80 → 1.68, flat from ~12 000), Phase 3 as v2 (early stop at 2 400 of 7 500, best dev 0.1115 at 1 400, 8.6 min), eval 6.8 min. Wall 3 h 17 min (+ a first attempt lost at Phase 2 start because 7 % of 98 M tokens asked for 13 440 QA-blend blocks and the packed QA corpus holds 10 396 for this tokenizer).

**BPE-16K comparator** (run `20260922-091654`, cell `bpe_16k_3phase_v3`, 3 h 25 min): same pipeline, budget and initialization with a from-scratch BPE-16K; its own 80 M-word pool (BPE packs 1.68 tokens per pool word vs AraRooPat 2.97) and a 3 % blend (same record exposure; a 5 % attempt died the same way, 5 919 QA blocks under BPE). Phase 3 ran all 7 500 steps (best dev 0.3581 at 6 800).

| | AraRooPat v2 | **AraRooPat v3** | **BPE-16K v3** | native base | native SFT |
|---|---|---|---|---|---|
| vocab / chars per token | 17 074 / 1.96 | 17 184 / 1.92 | 16 000 / 3.33 | 151 936 / 2.30 | 151 936 / 2.30 |
| LM loss after Phase 2 / SFT (nats/char) ¹ | 1.30 / 1.35 | **0.86 / 0.88** | 0.85 / 0.86 | 0.83 | 0.84 |
| answer NLL after SFT (nats/char) ² | 1.31 | **0.81** | 0.85 | 0.81 | 0.77 |
| P(EOS at end) / EOS rank-1 | 0.49 / 0.88 | 0.68 / 0.88 | 0.60 / 0.86 | 0.42 / 0.76 | 0.64 / 0.91 |
| judge ± SE | 1.07 ± 0.02 | **2.00 ± 0.08** | 1.99 ± 0.08 | 2.42 ± 0.09 | 2.28 ± 0.09 |
| paired Δ vs native base | −1.34 [−1.54, −1.17] | −0.41 [−0.59, −0.24] | −0.42 [−0.60, −0.26] | — | −0.14 [−0.30, +0.02] |
| judge off-topic / repetition | 72 % / 12 % | 25 % / 18 % | 19 % / 19 % | 13 % / 10 % | 14 % / 16 % |
| chrF / BERTScore-F1 | 9.77 / 0.840 | 15.48 / 0.853 | 16.65 / 0.852 | 22.94 / 0.853 | 18.59 / 0.859 |
| loop stop / EOS / cap | 27.6 / 72.4 / 0 % | **31.2** / 67.6 / 1.2 % | 20.8 / 77.6 / 1.6 % | 6.8 / 90.8 / 2.4 % | 12.4 / 87.2 / 0.4 % |
| mean answer chars / ≤ 40-char share | 54 / 54 % | 176 / 28 % | 190 / 20 % | 511 / 4 % | 207 / 19 % |
| generated chars per second | 199 | 111 | 153 | 285 | 213 |
| round-trip chrF (raw) | 89.11 | 91.10 | 100.0 | 99.42 | 99.42 |
| RPS (root conservation) | 0.379 | 0.379 | 0.048 | — | — |

Head to head, AraRooPat v3 − BPE-16K v3 on the same 250 prompts (paired bootstrap, 10 000 resamples): **Δ +0.012 [−0.136, +0.156]**, win / tie / loss 24 / 54 / 22 %.

¹ Re-measured 2026-09-22 on the held-out `configs/contamination/rawtext_heldout_v1.jsonl` (§3.6). As first reported the row read 1.29 / 1.34 · **0.95 / 0.97** · 0.90 / 0.91 · 0.82 · 0.84, from the last 120 documents of *each cell's own pool* — three different document sets, and for the adapted arms mostly their own training text. The correction closes most of the LM gap to the native model: 0.13 → **0.03** nats/char for AraRooPat v3 after Phase 2, 0.08 → **0.02** for BPE-16K.

² Σ NLL over the 250 held-out references ÷ their 72 705 characters (corrected 2026-09-23, §3.6 *Verification*). As first reported the row read 1.27 · 0.81 · **0.64** · 0.80 · 0.76: the BPE figure divided the per-token NLL by the wrong chars-per-token number. Under the comparable definition AraRooPat v3 (0.81) beats BPE-16K (0.85) on answer NLL.

**Acceptance (thresholds set before the run):** LM loss ≤ 1.00 nats/char after Phase 2 (0.95 ✓) and ≤ 1.05 after SFT (0.97 ✓); answer NLL ≤ 0.90 (0.81 ✓); judge ≥ 2.0 (2.004 ✓, at the line); paired CI vs native base not entirely below −0.5 (✓); **loop stop ≤ 12 % (31.2 % ✗)**; mean answer 150–400 chars (176 ✓); round-trip chrF ≥ 92 diacritics-stripped (98.97 ✓). Seven of eight.

### 3.6 Free-form dev early stop and the held-out loss correction (2026-09-22/23)

**The exposure asymmetry that prompted this.** Phase 3 early-stopped on the extractive dev loss
(`eval_splits: {tydiqa_arabic: dev, arcd: dev}`), but extractive records are 12 % of the mixture's loss tokens and
free-form answers 87 %. The two arms were treated very differently:

| arm | best extractive dev @ step | stop | restored = examples seen | free-form examples | Phase 3 wall | loop stop |
|---|---|---|---|---|---|---|
| `araroopat_3phase_v3` | 0.1115 @ 1 400 | 2 400 | 1 400 × 4 = **5 600** of 30 000 | ~1 700 | 514 s | 31.2 % |
| `bpe_16k_3phase_v3` | 0.3581 @ 6 800 | none (7 500) | 6 800 × 4 = **27 200** | ~8 200 | 1 526 s | 20.8 % |

A per-token loss averaged per batch is also not comparable across tokenizers: 0.11 against 0.36 says nothing about
quality.

**P0 — the raw-text loss was measured on the wrong documents.** `scripts/diag_heldout_loss.py` read *the last 120
documents of each cell's own pool*. The packer walks the whole pool in `source_doc_order` and a phase draws a prefix
of the packed blocks, so those documents were training text with probability 0.88 (v3), 0.98 (BPE) and 0.62 (v2) —
and each arm was scored on a **different** pool (native on the 20 M, v3 on the 50 M, BPE on the 80 M), i.e. on
different text. `scripts/build_rawtext_heldout.py` now writes
`configs/contamination/rawtext_heldout_v1.jsonl` (sha256 `c6522a4f0050702b…`): FineWeb-2 documents of the 80 M pool
past every packer's `docs_taken` (2 294 of 131 048), absent by text hash from every other pool (→ 875), sharing no
exact duplicate, no ≥ 20-word run and no ≥ 50 % 8-gram coverage with any pool document (→ 812); 150 drawn with seed 42,
median 232 words. Verification re-scans all 372 123 pool documents: **0 real overlaps**. Two rules the filter had to
learn: a candidate meets **itself** in its own pool (skip the self-match), and the exact-**paragraph** tier is
meaningless on web text — a FineWeb "paragraph" is a line and hundreds are headings that recur across thousands of
pages (19 508 such matches, none document overlap).

| cell / checkpoint | old nats/char | new nats/char | Δ | old pool |
|---|---|---|---|---|
| native_qwen3_base / base | 0.8235 | 0.8260 | +0.003 | 20 M |
| native_qwen3_sft / sft | 0.8358 | 0.8380 | +0.002 | 20 M |
| araroopat_3phase_v2 / warmup | 1.2934 | 1.3021 | +0.009 | 20 M |
| araroopat_3phase_v2 / sft | 1.3394 | 1.3471 | +0.008 | 20 M |
| araroopat_3phase_v3 / warmup | 0.9518 | **0.8595** | −0.092 | 50 M |
| araroopat_3phase_v3 / sft | 0.9747 | **0.8794** | −0.095 | 50 M |
| bpe_16k_3phase_v3 / warmup | 0.8985 | **0.8461** | −0.052 | 80 M |
| bpe_16k_3phase_v3 / sft | 0.9123 | **0.8577** | −0.055 | 80 M |

Exposure was the *smaller* half. The native cells never trained on a pool and move by +0.002 (so the new set is of
equal difficulty for an unadapted model); v2 saw 62 % of its pool and moves by +0.008, which is all the memorisation
the old rule hid. The dominant error was comparing arms across different document sets, and the *last* documents of
a pool parquet are a shard-ordered tail rather than a random sample. **The ranking does not change** (native <
BPE < AraRooPat v3 < v2) but the magnitudes do: the LM gap to the untouched native model closes from 0.13 to
**0.034** nats/char for AraRooPat v3 after Phase 2 and from 0.076 to **0.020** for BPE-16K. §4 finding 3 is amended
accordingly: the arms' language models are much closer to native than first reported, so the remaining judge gap is
*less* explainable by LM quality.

**P1 — dev slices and a stop signal that mirrors the objective.** `cidar`, `bactrian_x_ar`, `aya_ar`,
`arabic_squad` and `arabic_squad_mcq` now carve a 5 % `dev` slice by record id (`is_dev_id`); `validation` stays
refused. The MCQ corpus is hashed on the Arabic-SQuAD row it was built from, so a passage is dev in both corpora or
in neither, and `build_synthetic_mcq_corpus` reads the *unpartitioned* SQuAD list — building it from the train slice
would have shrunk it 48 344 → 45 936 and redrawn every distractor. Dev rows after exclusions: cidar 487,
bactrian_x_ar 3 375, aya_ar 1 011, arabic_squad / arabic_squad_mcq 2 407 each (train 9 227 / 63 636 / 18 738 /
45 936). The contamination scan now reads `train` + `dev` for all seven corpora and the excluded-id set is
byte-identical to the committed one (cidar 253, aya_ar 54, bactrian_x_ar 6). `early_stopping.eval_mixture`
(`{total_examples: 1000, seed: 42}`) composes the eval set from those dev slices at the phase mixture's ratio and
scores Σ NLL / Σ answer tokens per category; one pass is 12 s (1 000 records at batch 4), 33 % overhead at
`eval_every_n_steps: 200`. The composed eval set mirrors the training mixture: 400 / 300 / 300 records =
18 % / 3 % / 79 % of loss tokens against the training set's 17 % / 3 % / 80 %.

**P2 — the two runs.** Phase 3 only, from each arm's Phase 2 checkpoint, everything else v3's.

| | AraRooPat ffstop | AraRooPat v3 | BPE ffstop | BPE v3 | native base | native SFT |
|---|---|---|---|---|---|---|
| stop / restored step | 7 000 / **6 000** | 2 400 / 1 400 | 7 500 / **6 600** | 7 500 / 6 800 | — | — |
| mixture examples restored | **24 000** | 5 600 | **26 400** | 27 200 | — | — |
| judge ± SE | **2.152 ± 0.088** | 2.004 ± 0.085 | 1.996 ± 0.079 | 1.992 ± 0.081 | 2.416 ± 0.094 | 2.280 ± 0.090 |
| hist 1/2/3/4/5 | 112/67/26/11/34 | 130/56/24/13/27 | 117/71/30/10/22 | 123/66/26/10/25 | 92/69/29/13/47 | 105/60/31/18/36 |
| judge off-topic / repetition | 18 % / 17 % | 25 % / 18 % | 13 % / 18 % | 19 % / 19 % | 13 % / 10 % | 14 % / 16 % |
| chrF / BERTScore-F1 | 17.70 / 0.858 | 15.48 / 0.853 | 16.60 / 0.847 | 16.65 / 0.852 | 22.94 / 0.853 | 18.59 / 0.859 |
| loop stop / EOS / cap | **21.2** / 77.6 / 1.2 % | 31.2 / 67.6 / 1.2 % | 19.6 / 75.6 / 4.8 % | 20.8 / 77.6 / 1.6 % | 6.8 / 90.8 / 2.4 % | 12.4 / 87.2 / 0.4 % |
| mean chars / ≤ 40-char share | 190 / 23 % | 176 / 28 % | 211 / 21 % | 190 / 20 % | 511 / 4 % | 207 / 19 % |
| raw-text loss (P0 set) | 0.8751 | 0.8794 | 0.8589 | 0.8577 | 0.8260 | 0.8380 |
| answer NLL (nats/char, Σ NLL ÷ 72 705 reference chars — corrected 2026-09-23, see *Verification*) | **0.787** | 0.814 | 0.845 | 0.847 | 0.807 | 0.766 |
| P(EOS) / EOS rank-1 | 0.646 / 0.872 | 0.682 / 0.876 | 0.622 / 0.860 | 0.601 / 0.860 | 0.420 / 0.760 | 0.644 / 0.908 |

Paired bootstraps (10 000 resamples, same 250 prompts):

| pair | Δ | 95 % CI | win / tie / loss |
|---|---|---|---|
| AraRooPat ffstop − AraRooPat v3 | **+0.148** | [+0.036, +0.260] | 22 / 67 / 12 % |
| BPE ffstop − BPE v3 | +0.004 | [−0.128, +0.136] | 18 / 66 / 16 % |
| AraRooPat ffstop − BPE ffstop | +0.156 | [−0.000, +0.312] | 29 / 50 / 21 % |
| AraRooPat ffstop − native SFT | −0.128 | [−0.276, +0.020] | 21 / 53 / 26 % |
| BPE ffstop − native SFT | −0.284 | [−0.440, −0.132] | 17 / 54 / 29 % |
| AraRooPat ffstop − native base | −0.264 | [−0.440, −0.088] | 18 / 50 / 31 % |
| BPE ffstop − native base | −0.420 | [−0.588, −0.252] | 15 / 49 / 36 % |

**What this establishes, stated carefully.** The arm the old rule stopped early gained a significant +0.148 judge
points and improved on every free-form measure; the arm it never stopped gained +0.004. That contrast is the
control: the gain is *exposure*, not the new eval set. But the old rule was not inherently early-stopping — it
stopped v3 because *that cell's* extractive dev loss rose after step 1 400 (0.1115 → 0.1135 → 0.1165 → 0.1168 →
0.1148 → 0.1195) while BPE's fell monotonically to 0.3581. The counterfactual on the new run confirms it: an
extractive-only rule applied to its own extractive component would have restored step 5 600 (22 400 examples),
close to the mixture's 6 000. So the honest claim is not "the extractive signal stops too early" but "a signal
carrying 12 % of the loss tokens, measured on 888 rows, can wander the wrong way and take the other 87 % with it".
The mixture signal's case is that it measures what the phase optimises, and being per token it is comparable across
tokenizers — 0.996 (AraRooPat) against 1.836 (BPE) are on one scale, where 0.11 against 0.36 were not.

**Stop curves** (mixture metric, per category; both arms improved monotonically to their plateau):

| step | AraRooPat total | ext | mcq | free_form | BPE total | ext | mcq | free_form |
|---|---|---|---|---|---|---|---|---|
| 600 | 1.0422 | 0.1285 | 0.1471 | 1.2787 | 1.9096 | 0.4230 | 0.2844 | 2.1819 |
| 2 400 | 1.0169 | 0.1004 | 0.1242 | 1.2540 | 1.8621 | 0.3229 | 0.1812 | 2.1440 |
| 4 800 | 0.9976 | 0.0918 | 0.1168 | 1.2318 | 1.8368 | 0.2926 | 0.1690 | 2.1192 |
| 6 000 | **0.9958** | 0.0918 | 0.1140 | 1.2297 | 1.8357 | 0.2941 | 0.1619 | 2.1179 |
| 6 600 | 0.9958 | 0.0915 | 0.1135 | 1.2297 | **1.8356** | 0.2940 | 0.1613 | 2.1177 |
| 7 000 | 0.9958 | 0.0913 | 0.1135 | 1.2298 | 1.8359 | 0.2940 | 0.1623 | 2.1181 |

AraRooPat stopped at 7 000 (patience exhausted after the plateau at 6 000); BPE ran its full 7 500 and restored
6 600. Counterfactual stops on the same runs, same patience rule: an **extractive-only** signal would have stopped
AraRooPat at 6 600 restoring 5 600 (22 400 examples) and BPE at 5 800 restoring 4 800 (19 200) — i.e. on these runs
it would have cut *both* arms slightly shorter than the mixture rule, not reproduced v3's 1 400. A **free-form-only**
signal restores 6 000 for both. The mixture metric sits between them, which is what a token-weighted average of the
three should do.

**Acceptance.** AraRooPat ffstop: judge ≥ 2.004 with the paired CI vs v3 not below 0 — **2.152, [+0.036, +0.260] ✓**;
answer NLL ≤ 0.814 — **0.787 ✓**; raw-text within +0.03 of its Phase 2 checkpoint (0.8595) — **0.8751, +0.016 ✓**;
mean answer 150–400 chars — **190 ✓**; EOS ≥ 67.6 % — **77.6 % ✓**; loop stop below v3's 31.2 % — **21.2 % ✓** but
**≤ 12 % ✗**. Six of seven. BPE ffstop: judge ≥ 1.992, CI not below 0 — **1.996, [−0.128, +0.136] ✓**; answer NLL
≤ 0.847 — **0.845 ✓**; raw-text within +0.03 of 0.8461 — **0.8589, +0.013 ✓**; mean 150–400 — **211 ✓**; loop below
20.8 % — **19.6 % ✓**; ≤ 12 % — **✗**; EOS ≥ 77.6 % — **75.6 % ✗**. Five of seven, and it moved on nothing, which is
what the control was for.

**Loops.** AraRooPat ffstop 53 loop stops (period 1: 5, 2–3: 17, 4–10: 21, > 10: 10; no char-rule stops), 28 % of
its 112 score-1 rows (31 rows); BPE ffstop 49 (6 / 11 / 21 / 11; recounted 2026-09-23 from the dump — the run report
had 5 / 18 / 19 / 10), 21 % of 117 (24 rows). By period bucket the **shape did not change**: v3's 78 loops were
13 / 23 / 35 / 7, so the ffstop cell has fewer loops of the same kinds, with period 4–10 still the largest bucket
(40 % vs 45 %). The ten rows read by hand were mostly short **و-coordination runaways** — the model opens a
coordinated list and cannot close it (`والفلفل الأسود والفلفل الأحمر والفلفل الأخضر والفلفل الأصفر`, `الصيد والصيد`,
`الرقص الرقص`) — but the buckets say those (period 1–3, 22 of 53) are not the majority; the run report's claim that
the loops had turned from phrase loops into coordination runaways is not supported by the counts. The cut keeps the
first copy, so these rows end short.

**Examples.** AraRooPat: 15 rows gained ≥ 2 judge points, 7 lost ≥ 2. The gains are answers that became complete
(`cidar-9423`: *الإجابة صحيحة* → *الإجابة الصحيحة : الأشخاص التالية أسماؤهم*, 1 → 5; `cidar-1952`: an 18-character
fragment → a finished sentence, 1 → 5; `cidar-4481`: a character sheet that stays on topic, 2 → 5). The losses are
mostly longer answers that add wrong detail (`cidar-273`: a 54-character correct fact → a 308-character description
with three wrong borders, 5 → 2) or flipped grammar verdicts (`cidar-10118`, 5 → 1 in *both* arms). BPE: 14 gained,
13 lost — a wash, as its judge mean says.

**Artifacts.** `configs/contamination/rawtext_heldout_v1.jsonl` + `.manifest.json`;
`<cell>/diag_heldout_rawtext_v1[_warmup][_base].json` for eight checkpoints;
`outputs/experiments/qwen_native_vs_araroopat/{araroopat_3phase_v3_ffstop,bpe_16k_3phase_v3_ffstop}/`
(`all_metrics.json` with `training.sft.eval_history` + `eval_loss_definition`, `data/sft_eval_mixture_manifest.json`,
`eval_rows/freeform_cidar.parquet`, `freeform_judge/gemma4_31b.parquet`).

**Verification (2026-09-23, Fable session, against the outputs).** Reproduced from the artifacts: every raw-text loss
of the P0 table (four decimals); the 150 held-out documents absent by exact hash from all nine pool tables and all
150 beyond the packer's `docs_taken` in the 80 M pool's permutation; the dev sizes and the MCQ ↔ SQuAD dev-id
equality; the unchanged exclusion list; the eval-mixture composition (400 / 300 / 300 records, 18.1 / 2.5 / 79.4 % of
loss tokens); restored steps 6 000 / 6 600 and the resize no-op in both logs; judge means, histograms and all seven
paired bootstraps to three decimals; the counterfactual stops (extractive-only 5 600 / 4 800, free-form-only 6 000 /
6 000); 105 new and changed tests passing. **One derived column was wrong.** The diagnostic records NLL *per answer
token*; the run report (and §3.5 before it, for the BPE cell) converted it to nats per character by dividing by a
chars-per-token figure — the raw-text block's (3.63 for BPE, 2.73 for native, 2.00 for AraRooPat) or the free-form
task's — which is not the references' (BPE 2.49, native 2.30, AraRooPat 1.96), so BPE and native were flattered.
The comparable quantity is Σ NLL over the 250 references ÷ their 72 705 characters (the same text, the same
characters, under each tokenizer's own tokenization): v2 1.306, AraRooPat v3 0.814, AraRooPat ffstop 0.787, BPE v3
0.847, BPE ffstop 0.845, native base 0.807, native SFT 0.766 (reported: 0.803 / 0.776 / 0.588 / 0.587 / 0.673 /
0.639; §3.5 had BPE 0.64). The ranking on this measure is native SFT < AraRooPat ffstop < native base < AraRooPat v3
< BPE ffstop ≈ BPE v3 — AraRooPat beats BPE on answer NLL, the opposite of what §3.5 and §4 first said, and the
ffstop checkpoint edges the untouched native model. The tables and acceptance lines above carry the corrected
numbers; the acceptance verdicts do not change. Open: the diagnostic should emit `nll_per_answer_char` and
`reference_chars` itself (§5 item 10). Also corrected here: the BPE loop-period buckets and the loop-shape claim
(see *Loops*). The five commits of this stage carry the `Claude Fable 5.1` co-author trailer because the brief
prescribed it; the code was written by an Opus 5 session.

## 4. What the campaign established

1. The native model's loops were partly measurement (the first loop rule) and partly a greedy attractor of the base model; under the corrected rules the untrained base loops on 6.8 % of prompts and the SFT arm on 12.4 %. SFT on the 30 000-record mixture learns the references' register and length but not their content: judge 2.28 vs 2.42, CI including zero.
2. AraRooPat v1/v2 failed for a reason unrelated to morphology: an arbitrary embedding initialization and a 37 M-token budget left the language model at 1.35 nats/char (native 0.83). Informed initialization plus 131 M tokens brought it to 0.88 and the judge from 1.07 to 2.00; answer NLL now equals the untrained native model's. (Loss figures corrected in §3.6; as first measured, 1.34 → 0.97 against a native 0.82.)
3. At equal adaptation a plain BPE-16K vocabulary ties AraRooPat on the judge and beats it on LM loss (0.86 vs 0.88), loops (20.8 vs 31.2 %) and generation speed (153 vs 111 chars/s), with 1.7× more text seen per token budget; AraRooPat has the lower answer NLL on the held-out references (0.81 vs 0.85 nats/char — the 0.64 first reported for BPE divided by the wrong chars-per-token figure, corrected 2026-09-23, §3.6 *Verification*). **Amended by §3.6:** once both arms train on the free-form-aware stop signal, AraRooPat leads the judge (2.152 vs 1.996, paired Δ +0.156 [+0.000, +0.312]), closes the loop gap (21.2 vs 19.6 %) and widens its answer-NLL lead (0.787 vs 0.845); BPE keeps the LM-loss and speed advantages. The tie was partly an artefact of AraRooPat having trained on a fifth of the mixture. AraRooPat's measurable advantages are decode fidelity to the written word (99.3 % word round trip) and root conservation (0.379 vs 0.048). Both arms sit 0.4 judge points under the untouched native model; that gap is the cost of re-learning any vocabulary in 131 M tokens — and it is **not** explained by language-model quality, which after the §3.6 correction is within 0.03 nats/char of native for both.
4. Phase 2 loss is flat from its first fifth in both arms; more raw text at this learning rate is not the next lever.
5. (§3.6) Two measurement faults were corrected. The raw-text "held-out" loss was read from each cell's *own* pool — training text for the adapted arms, and a different document set per arm; on a properly held-out set the LM gap to native is 0.03 nats/char, not 0.13. And the Phase 3 stop signal carried 12 % of the loss tokens: on one arm it wandered upward at step 1 400 and stopped training at a fifth of the mixture. Neither fault changed a ranking; both changed magnitudes enough to change what the next step should be.

## 5. Open items and next steps (in the order proposed)

1. ~~Free-form dev slice for Phase 3 early stopping~~ — **done, §3.6.** Cells `araroopat_3phase_v3_ffstop` and `bpe_16k_3phase_v3_ffstop`.
2. **Register `configs/contamination/rawtext_heldout_v1.jsonl` in `heldout_sets.yaml` before the next pool is built**, so Stage A drops its 150 documents by content. It was deliberately left unregistered during this campaign because registering it moves the held-out fingerprint, hence the pool fingerprint and `exclusions.json`, which would have confounded the Phase 3 re-runs.
3. Loop-aware decoding ablation on the existing checkpoints (eval only) to measure how much of the loop gap is decoding rather than the model. Still open, and now the *only* acceptance threshold both v3 arms miss (21.2 / 19.6 % against 12 %).
4. Equal-text ablation: AraRooPat at ~227 M tokens (1.73× the token budget) so both arms see the same Arabic.
5. Distillation from the native model (attention / logit) for both arms.
6. A pre-flight check of `qa_blend.share` against the tokenizer's packed QA block count (two Phase 1 runs, 62 min, were lost to it).
7. The probe's step-0 gate is uninformative under tied embeddings; read step-N only, or untie the head.
8. 13 hamza and 1 ى round-trip mismatches remain (corpus-majority canonicalisations).
9. The free-form eval mixture costs 12 s per pass (33 % overhead at `eval_every_n_steps: 200`). Acceptable, but `eval_every_n_steps: 400` would halve it if a longer phase needs the time.
10. `scripts/diag_heldout_loss.py` should emit `nll_per_answer_char` (Σ NLL ÷ Σ reference characters) and `reference_chars` next to `nll_per_answer_token`: two reports in a row derived the per-character figure by dividing by a chars-per-token number and got the BPE ranking wrong (§3.6 *Verification*).

## 6. GPU time (2026-09-22 alone)
Four training runs 8 h 52 min (1 h 15 lost attempt + 3 h 17 v3 + 0 h 56 lost attempt + 3 h 25 BPE), probe ~35 min, six diagnostics ~6 min, judge passes ~10 min — ≈ 9 h 43 min. Earlier: v1 ~1 h, v2 ~1.5 h (+ 45 min resume), native runs and re-evals ~2 h.

§3.6 (2026-09-22 evening): AraRooPat ffstop 42 min end to end (Phase 3 26.9 min incl. 33 evals, free-form eval 5 min), BPE ffstop 36 min (Phase 3 25.9 min, 35 evals, eval 7.6 min), ten raw-text diagnostics ~12 min, judge 2 × 250 verdicts ~1 min plus a 7-minute cold start — **≈ 1 h 40 min**. Non-GPU: the held-out build + two verification scans ~13 min, the two mixture dry runs ~35 min (AraRooPat over the CAMeL bridge).

## 7. References
- Xu, J. et al. (2022). *Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation* (DITTO). NeurIPS.
- Ivgi, M. et al. (2024). *From Loops to Oops: Fallback Behaviors of Language Models Under Uncertainty*.
- Holtzman, A. et al. (2019). *The Curious Case of Neural Text Degeneration*.
- Fu, Z. et al. (2021). *A Theoretical Analysis of the Repetition Problem in Text Generation*.
- Sclar, M. et al. (2023). *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*.
- Mundra, N. et al. (2024). *An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models*. arXiv:2407.05841.
- *TokAlign: Efficient Vocabulary Adaptation via Token Alignment*. ACL 2025, arXiv:2506.03523.
- *Model-Aware Tokenizer Transfer* (2025). arXiv:2510.21954.
- *Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension* (2026). arXiv:2608.03494.
- Gee, L. et al. (2022/2024). *Fast Vocabulary Transfer for Language Model Compression*. arXiv:2402.09977.
- Gu, S. et al. (2024). *ReTok: Replacing Tokenizer to Enhance Representation Efficiency in Large Language Model*. (arXiv id to verify)
- *Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs* (LREC 2026). arXiv:2603.15773.
- *Exploring Tokenization Strategies and Vocabulary Sizes for Enhanced Arabic Language Models* (2024). arXiv:2403.11130.
- Dobler, K. & de Melo, G. (2023). *FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models*. Minixhofer, B. et al. (2022). *WECHSEL*. (cited through the papers above)

## 8. Where the numbers live
`outputs/experiments/qwen_native_vs_araroopat/<cell>/all_metrics.json` (metrics, judge summary, training histories), `eval_rows/freeform_cidar.parquet` (every generation), `freeform_judge/gemma4_31b.parquet` (verdicts), `diag_heldout_loss*.json` (LM loss / answer NLL), `embedding_init_probe.json`, `_p3_decode/roundtrip_v{2,3}_*.json`, `_report_2026-09-22/report_draft.md` (the 2026-09-22 hand-off report), `araroopat_3phase_v2/phase12_run_20260921-115522/` (v2's Phase 1/2 log). Configs: `configs/experiments/qwen_native_{no_presteps_after_LOOP_fix,sft_only,sft_reeval}.yaml`, `qwen_araroopat_3phase{,_v2,_v2_resume_sft,_v3}.yaml`, `qwen_bpe16k_3phase_v3.yaml`. Commits: `6a1d3db` (scheduler, templates, loop stop, SDPA), `6eee412` (periodic loop stop), `a2c671d` (markers + budget), `a879d73`/`f1ee2f5`/`3885dda` (console), `e06910e` (decode), `b204a54` (embedding init), `3ddf26c`…`5c91854` (v3 + comparator), `562da5e` (this record + the v2 configs), `cb68d34` (held-out raw-text set + the corrected loss numbers), `2615c37` (dev slices + `eval_mixture`), `77fded5` (the two ffstop configs + `paired_compare.py`).

§3.6 artifacts: `configs/contamination/rawtext_heldout_v1.jsonl` (+ `.manifest.json`, sha256 `c6522a4f0050702b…`), `<cell>/diag_heldout_rawtext_v1[_warmup][_base].json` (eight checkpoints of the earlier cells, two of the new ones), `<cell>/data/sft_eval_mixture_manifest.json`, `all_metrics.json["training"]["sft"]["eval_history"]`, `scripts/build_rawtext_heldout.py`, `scripts/judge/paired_compare.py`. The pre-2026-09-22 `diag_heldout_loss*.json` files are kept; they record what the old pool-tail rule measured.

## 9. Cells of `outputs/experiments/qwen_native_vs_araroopat/`
Current: `native_qwen3_base_m2_c2400`, `native_qwen3_sft_m2_c2400`, `araroopat_3phase_v2`, `araroopat_3phase_v3`, `bpe_16k_3phase_v3`, **`araroopat_3phase_v3_ffstop`**, **`bpe_16k_3phase_v3_ffstop`** (§3.6: the same recipe with the free-form-aware early stop; these are the current best arms of each vocabulary). Older rule sets, kept: `native_qwen3` (untrained base, v1 template), `native_qwen3_base`, `native_qwen3_sft`, `araroopat` (v1). Under `_superseded/`: `native_qwen3_base_after_loop_fix`, `native_qwen3_sft_reeval`, `native_qwen3_{base,sft}_m2_c1200`, the two lost v3 attempts.
