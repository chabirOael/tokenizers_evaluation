# Qwen3-4B-Base: native tokenizer vs AraRooPat — campaign record (2026-09-18 → 2026-09-22)

Working notes for the comprehensive report. Every number below was measured on this machine (H100, `Qwen/Qwen3-4B-Base`); the artifact that holds it is named. Free-form numbers are comparable only across cells scored under the same decoding rules — the rule set is stated per table.

## 0. Set-up

- **Question.** Does a morphological tokenizer (AraRooPat: `[ROOT_x] [PAT_y]` + clitic tokens) beat the native Qwen3 vocabulary on Arabic generation when the 3-phase pipeline is held fixed?
- **Arms.** A = native Qwen3 tokenizer (vocab 151 936, no training or Phase 3 only); B = AraRooPat (from-scratch vocab, Phase 1 + 2 on the pretraining mix, Phase 3). From 2026-09-22: C = from-scratch BPE-16K, the fair comparator.
- **Phase 3.** Mixture of 30 000 records at 40 % extractive (TyDiQA-AR + ARCD) / 30 % MCQ (synthetic Arabic-SQuAD) / 30 % free-form (CIDAR, Bactrian-X ar, Aya ar); in loss tokens the free-form share is 80–87 %. Early stop on the extractive `dev` slice.
- **Free-form eval.** 250 held-out CIDAR instructions (E5 near-duplicate gate 0.93), pure greedy decoding, stop at EOS / template marker / text-level loop, character budget per tokenizer. Metrics: chrF, in-house BERTScore, loop / cap / EOS rates, and a 1–5 judge (`google/gemma-4-31b-it` under vLLM, reference-guided rubric, paired bootstrap against a baseline cell). Held-out loss diagnostic (`scripts/diag_heldout_loss.py`): raw-text LM loss on 120 pool documents and answer NLL / P(EOS) on the 250 references.
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
| native base (untrained) | 2.42 ± 0.09 | 22.94 | 6.8 % | 90.8 % | 2.4 % | 511 | 0.82 | 0.80 |
| native SFT | 2.28 ± 0.09 | 18.59 | 12.4 % | 87.2 % | 0.4 % | 207 | 0.84 | 0.76 |

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
| LM loss after Phase 2 / SFT (nats/char) | 1.29 / 1.34 | **0.95 / 0.97** | 0.90 / 0.91 | 0.82 | 0.84 |
| answer NLL after SFT (nats/char) | 1.27 | **0.81** | 0.64 | 0.80 | 0.76 |
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

**Acceptance (thresholds set before the run):** LM loss ≤ 1.00 nats/char after Phase 2 (0.95 ✓) and ≤ 1.05 after SFT (0.97 ✓); answer NLL ≤ 0.90 (0.81 ✓); judge ≥ 2.0 (2.004 ✓, at the line); paired CI vs native base not entirely below −0.5 (✓); **loop stop ≤ 12 % (31.2 % ✗)**; mean answer 150–400 chars (176 ✓); round-trip chrF ≥ 92 diacritics-stripped (98.97 ✓). Seven of eight.

## 4. What the campaign established

1. The native model's loops were partly measurement (the first loop rule) and partly a greedy attractor of the base model; under the corrected rules the untrained base loops on 6.8 % of prompts and the SFT arm on 12.4 %. SFT on the 30 000-record mixture learns the references' register and length but not their content: judge 2.28 vs 2.42, CI including zero.
2. AraRooPat v1/v2 failed for a reason unrelated to morphology: an arbitrary embedding initialization and a 37 M-token budget left the language model at 1.34 nats/char (native 0.82). Informed initialization plus 131 M tokens brought it to 0.97 and the judge from 1.07 to 2.00; answer NLL now equals the untrained native model's.
3. At equal adaptation a plain BPE-16K vocabulary ties AraRooPat on the judge and beats it on LM loss (0.91 vs 0.97), answer NLL (0.64 vs 0.81 nats/char), loops (20.8 vs 31.2 %) and generation speed (153 vs 111 chars/s), with 1.7× more text seen per token budget. AraRooPat's measurable advantages are decode fidelity to the written word (99.3 % word round trip) and root conservation (0.379 vs 0.048). Both arms sit 0.4 judge points under the untouched native model; that gap is the cost of re-learning any vocabulary in 131 M tokens.
4. Phase 2 loss is flat from its first fifth in both arms; more raw text at this learning rate is not the next lever.

## 5. Open items and next steps (in the order proposed)

1. Free-form dev slice for Phase 3 early stopping (v3 stopped at 2 400 of 7 500 steps on the extractive signal; BPE ran to the end) — re-run Phase 3 of both arms from their Phase 2 checkpoints.
2. Loop-aware decoding ablation on the existing checkpoints (eval only) to measure how much of the loop gap is decoding rather than the model.
3. Equal-text ablation: AraRooPat at ~227 M tokens (1.73× the token budget) so both arms see the same Arabic.
4. Distillation from the native model (attention / logit) for both arms.
5. A pre-flight check of `qa_blend.share` against the tokenizer's packed QA block count (two Phase 1 runs, 62 min, were lost to it).
6. The probe's step-0 gate is uninformative under tied embeddings; read step-N only, or untie the head.
7. 13 hamza and 1 ى round-trip mismatches remain (corpus-majority canonicalisations).

## 6. GPU time (2026-09-22 alone)
Four training runs 8 h 52 min (1 h 15 lost attempt + 3 h 17 v3 + 0 h 56 lost attempt + 3 h 25 BPE), probe ~35 min, six diagnostics ~6 min, judge passes ~10 min — ≈ 9 h 43 min. Earlier: v1 ~1 h, v2 ~1.5 h (+ 45 min resume), native runs and re-evals ~2 h.

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
`outputs/experiments/qwen_native_vs_araroopat/<cell>/all_metrics.json` (metrics, judge summary, training histories), `eval_rows/freeform_cidar.parquet` (every generation), `freeform_judge/gemma4_31b.parquet` (verdicts), `diag_heldout_loss*.json` (LM loss / answer NLL), `embedding_init_probe.json`, `_p3_decode/roundtrip_v{2,3}_*.json`, `_report_2026-09-22/report_draft.md` (the 2026-09-22 hand-off report), `araroopat_3phase_v2/phase12_run_20260921-115522/` (v2's Phase 1/2 log). Configs: `configs/experiments/qwen_native_{no_presteps_after_LOOP_fix,sft_only,sft_reeval}.yaml`, `qwen_araroopat_3phase{,_v2,_v2_resume_sft,_v3}.yaml`, `qwen_bpe16k_3phase_v3.yaml`. Commits: `6a1d3db` (scheduler, templates, loop stop, SDPA), `6eee412` (periodic loop stop), `a2c671d` (markers + budget), `a879d73`/`f1ee2f5`/`3885dda` (console), `e06910e` (decode), `b204a54` (embedding init), `3ddf26c`…`5c91854` (v3 + comparator).

## 9. Cells of `outputs/experiments/qwen_native_vs_araroopat/`
Current: `native_qwen3_base_m2_c2400`, `native_qwen3_sft_m2_c2400`, `araroopat_3phase_v2`, `araroopat_3phase_v3`, `bpe_16k_3phase_v3`. Older rule sets, kept: `native_qwen3` (untrained base, v1 template), `native_qwen3_base`, `native_qwen3_sft`, `araroopat` (v1). Under `_superseded/`: `native_qwen3_base_after_loop_fix`, `native_qwen3_sft_reeval`, `native_qwen3_{base,sft}_m2_c1200`, the two lost v3 attempts.
