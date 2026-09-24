# Qwen3-4B-Base: native tokenizer vs AraRooPat — campaign record (2026-09-18 → 2026-09-24)

Working notes for the comprehensive report. Every number below was measured on this machine (H100, `Qwen/Qwen3-4B-Base`); the artifact that holds it is named. Free-form numbers are comparable only across cells scored under the same decoding rules — the rule set is stated per table.

## 0. Set-up

- **Question.** Does a morphological tokenizer (AraRooPat: `[ROOT_x] [PAT_y]` + clitic tokens) beat the native Qwen3 vocabulary on Arabic generation when the 3-phase pipeline is held fixed?
- **Arms.** A = native Qwen3 tokenizer (vocab 151 936, no training or Phase 3 only); B = AraRooPat (from-scratch vocab, Phase 1 + 2 on the pretraining mix, Phase 3). From 2026-09-22: C = from-scratch BPE-16K, the fair comparator.
- **Phase 3.** Mixture of 30 000 records at 40 % extractive (TyDiQA-AR + ARCD) / 30 % MCQ (synthetic Arabic-SQuAD) / 30 % free-form (CIDAR, Bactrian-X ar, Aya ar); in loss tokens the free-form share is 80–87 %. Early stop on the extractive `dev` slice until 2026-09-22; since then (§3.6) on a 1 000-record `dev` mixture at the training ratio, scored per token.
- **Free-form eval.** 250 held-out CIDAR instructions (E5 near-duplicate gate 0.93), pure greedy decoding, stop at EOS / template marker / text-level loop, character budget per tokenizer. Metrics: chrF, in-house BERTScore, loop / cap / EOS rates, and a 1–5 judge (`google/gemma-4-31b-it` under vLLM, reference-guided rubric, paired bootstrap against a baseline cell). Held-out loss diagnostic (`scripts/diag_heldout_loss.py`): raw-text LM loss on 150 held-out FineWeb-2 documents (`rawtext_heldout_v1.jsonl`; until 2026-09-22 the last 120 documents of each cell's own pool — not held out, see §3.6) and answer NLL / P(EOS) on the 250 references; answer NLL per character is Σ NLL ÷ the references' 72 705 characters — since 2026-09-23 the script emits it itself (`heldout_answers.nll_per_answer_char`, Σ NLL ÷ the characters *as encoded*: 72 636 for AraRooPat, whose normalization drops 70 zero-width format characters and NFKC-expands one, 72 705 for native and BPE; `nll_per_answer_char_raw_chars` divides by 72 705 for every tokenizer), and `scripts/diag_backfill_per_char.py` added both fields to the 19 earlier JSONs (§3.7).
- **Experiment folder.** `outputs/experiments/qwen_native_vs_araroopat/` (cells listed in §9); superseded cells under `_superseded/`. The decoding-ablation cells of §3.7 (eval-only re-runs of existing checkpoints under a repetition penalty) live in `outputs/experiments/qwen_decoding_ablation/`.

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
| 09-23 | the diagnostic emits answer NLL per character; decoding ablation (repetition penalty 1.2 on four checkpoints) | loops 0–1.2 % in every cell, judge not up (AraRooPat −0.30, native base −0.29, CIs below 0); AraRooPat's loop prompts hide no good answer and the penalty halves its article tokens |
| 09-23 | AraRooPat v4: Phase 3 at 45 000 × 25/15/60, max_length 1 024 | answer NLL 0.788 → **0.766** (= native SFT), judge 2.15 → 2.18 (Δ +0.03, CI spans 0), loops 21.2 → 23.2 %; gate failed, P4 not run |
| 09-23 | uncertainty diagnostic on five checkpoints | the flatness hypothesis fails: at equal answer NLL AraRooPat v4's reference entropy is 0.744 vs native SFT 0.764 nats/char; low margin before loop onset in every cell, native included |
| 09-23/24 | neutral-teacher distillation: 6-candidate bake-off (Aya-Expanse-32B, 4.64 on dev), teacher ceiling 3.65 on the test, 35 066 filtered teacher answers, v4's Phase 3 on them for all three arms | judge AraRooPat **2.43**, native **2.52**, BPE **2.13** (retained share 0.67 / 0.69 / 0.58); AraRooPat − BPE **+0.30 [+0.15, +0.45]**, AraRooPat − native −0.09 [−0.24, +0.07]; loops 6.8 / 8.4 / 15.6 % |
| 09-24 | tokenization audit on the eval texts + MCQ scorer check | AraRooPat: 67–81 % of Arabic words ROOT+PAT, 3.8–14.9 % character path (mostly the choice letters; dialect on Alghafa), `<unk>` 0.5–1.1 % on exam / Alghafa text; native Qwen3: 42–52 % of words contain a single-letter piece; AraRooPat needs 1.3–1.5× native's tokens (26 % of Alghafa prompts > 1 024). **The MCQ scorer skips the first continuation token and scores `</s>` for every EOS-appending tokenizer** (BPE letter MCQ scored on `</s>` alone); native numbers unaffected; fix before the v5 MCQ run (§3.9, §5 items 18–21) |

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

### 3.7 Decoding ablation and AraRooPat v4 (2026-09-23)

**The hypothesis.** The AraRooPat model has seen too few and too short free-form answers. Two measured facts behind
it. (1) Exposure was the lever of §3.6: 5 600 → 24 000 mixture examples moved the judge 2.00 → 2.15 and loops
31.2 → 21.2 %, but 30 000 records at 40/30/30 hold only 9 000 *distinct* free-form records (3 000 per corpus under
`within_category: equal`) and the dev loss plateaued on one pass over them. The free-form pools after dev slices and
exclusions are cidar 9 227, bactrian_x_ar 63 636, aya_ar 18 738. (2) At `max_length: 512` the mixture drops
AraRooPat's long answers: in ffstop's manifest 938 of 10 514 free-form draws (8.9 %) were dropped because 512 tokens
cut the answer, plus 576 whose prompt left no room for any answer; native SFT dropped 408 of 9 685 (4.2 %) + 277, BPE
ffstop 268 of 9 446 (2.8 %) + 178. (The brief quoted 8.3 % vs 2.4 %; the manifest numbers above are the ones this
record uses.) Both are Phase 3 data facts, so v4 resumes from v3's Phase 2 checkpoint and changes only Phase 3.
Before it, the decoding ablation (P1) and the answer-NLL field (P0).

**P0 — the diagnostic states answer NLL per character itself.** `scripts/diag_heldout_loss.py` now writes, in
`heldout_answers`, `answer_nll_total` (Σ NLL over the answer tokens, EOS included), `reference_chars` (Σ `len` of
the scored references *as encoded* — after the tokenizer's own normalization: AraRooPat's NFKC + format-character
drop gives 72 636, native's NFC and BPE's no-normalizer give 72 705), `reference_chars_raw` (72 705),
`nll_per_answer_char` = total ÷ as-encoded chars, and `nll_per_answer_char_raw_chars` = total ÷ raw chars; the
printed table shows both. `scripts/diag_backfill_per_char.py` (no GPU, idempotent) rebuilt the total of the 19
earlier `diag_heldout_*.json` files of the experiment folder from the recorded per-token NLL (4 decimals) × answer
tokens and added the fields; a second pass printed `=` for all 19. Over the raw 72 705 characters the backfill
reproduces §3.6 *Verification* exactly — AraRooPat v3 0.8141, ffstop 0.7873, BPE v3 0.8466, BPE ffstop 0.8450,
native base 0.8066, native SFT 0.7662, v2 1.3056. Over the as-encoded count AraRooPat's values are 0.001 higher (v3
0.8149, ffstop 0.7880, v2 1.3068), because 70 zero-width characters it never models leave the denominator; native
and BPE are unchanged. The rankings do not move. From here on the tables carry the as-encoded field, and the v4
acceptance line compares like with like (ffstop 0.7880).

**P1 — decoding ablation on the existing checkpoints.** `freeform_cidar` declares two knobs,
`repetition_penalty` (default 1.0) and `no_repeat_ngram_size` (default 0), in a `decoding` group of its
`param_spec()`; both generate calls receive them and the dump metadata records them. They are token-level and so
granularity-dependent — measurement knobs, not a comparison rule. The penalty is HF's formula (CTRL: a logit of a
token already in the context is divided by the penalty when positive, multiplied when negative; the prompt counts, as
in HF) applied over each row's context **minus its left padding**: HF's own `RepetitionPenaltyLogitsProcessor`
penalises every id in `input_ids`, and the native Qwen wrapper pads with the EOS id (151 643), so every padded row of
a batch would have had its EOS logit penalised and the output would depend on the batch composition (unit test:
HF's processor changes the EOS logit of a padded row, ours does not; on the tiny test model neither flipped an
argmax). **Reproduction check:** the AraRooPat ffstop checkpoint re-evaluated under the defaults through the new code
(`qwen_decoding_ablation/_repro/araroopat_3phase_v3_ffstop_greedy`) equals the twin cell in all 250 rows
(`generation_raw`, `generation`, stop reason, token count, prompt) — the defaults are byte-identical, and
`training/sft` is the model the twin scored in-run (restore-best precedes the save), so pairing the ablation cells
with their twins is valid. Four cells at `repetition_penalty: 1.2` (the value that took the untrained base from 44 %
to 8 % loops under the v1 template on 2026-09-18), everything else the reference rule set, output
`outputs/experiments/qwen_decoding_ablation/`, judge `gemma4_31b`:

| cell | greedy: loop / EOS / cap | mean chars | chrF | judge ± SE | off-topic | rp 1.2: loop / EOS / cap | mean chars | chrF | judge ± SE | off-topic | Δ rp − greedy [95 % CI] | win / tie / loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AraRooPat ffstop | 21.2 / 77.6 / 1.2 % | 190 | 17.70 | 2.15 ± 0.09 | 18.0 % | 0.8 / 99.2 / 0.0 % | 157 | 14.26 | 1.85 ± 0.08 | 42.0 % | **−0.30 [−0.46, −0.16]** | 15 / 54 / 31 % |
| BPE ffstop | 19.6 / 75.6 / 4.8 % | 211 | 16.60 | 2.00 ± 0.08 | 13.2 % | 0.0 / 100.0 / 0.0 % | 166 | 16.47 | 1.97 ± 0.08 | 38.4 % | −0.02 [−0.17, +0.12] | 22 / 55 / 24 % |
| native SFT | 12.4 / 87.2 / 0.4 % | 207 | 18.59 | 2.28 ± 0.09 | 14.0 % | 0.8 / 99.2 / 0.0 % | 232 | 18.93 | 2.21 ± 0.09 | 26.8 % | −0.07 [−0.23, +0.08] | 21 / 53 / 26 % |
| native base | 6.8 / 90.8 / 2.4 % | 511 | 22.94 | 2.42 ± 0.09 | 12.8 % | 1.2 / 96.8 / 0.4 % | 562 | 21.94 | 2.13 ± 0.09 | 28.4 % | **−0.29 [−0.43, −0.14]** | 13 / 57 / 30 % |

BERTScore-F1 greedy → rp 1.2: 0.858 → 0.838, 0.847 → 0.849, 0.859 → 0.852, 0.853 → 0.848. The judge's repetition
flag falls to 0.4–2.4 % in every cell; the sub-scores fall with the mean (AraRooPat correctness −0.38, instruction
following −0.42; native base fluency −0.43).

*Where the change lands.* Split by whether the greedy row was loop-stopped: AraRooPat's 53 loop rows go 1.62 →
1.72 (+0.09) while its 197 other rows go 2.29 → 1.88 (−0.41); BPE's 49 loop rows 1.69 → 2.12 (+0.43), other rows
−0.13; native SFT's 31 loop rows 2.16 → 2.42 (+0.26), other rows −0.12; the base's 17 loop rows −0.24, other rows
−0.29. On AraRooPat's own 53 loop prompts the other cells score at their usual level (BPE 2.00 greedy / 2.08 rp,
native SFT 2.25 / 2.32, native base 2.83 / 2.30), so those prompts are not especially hard — AraRooPat scores 1.62
there, and 1.72 once the loop is broken.

*The granularity confound, measured.* The share of generated words carrying the article (ال / وال / بال / لل) falls
0.388 → 0.182 for AraRooPat under the penalty (و-prefixed words 0.127 → 0.059) and stays flat for BPE (0.336 →
0.317), native SFT (0.314 → 0.299) and native base (0.313 → 0.328). One AraRooPat word is `[CLITICP_ال] [ROOT]
[PAT]`-shaped, so after the first definite noun every later article is a "repeated token"; the decoded nouns lose it
(`… 2 - الإرسال … 3 - لغة هي …`). That is most of AraRooPat's −0.41 on its non-loop rows.

**Decoding or model?** Both, but the part that matters is the model. The loop *rate* gap is a greedy attractor
every cell shares: under the penalty it is gone (0.8 % against 0.0 / 0.8 / 1.2 %). But removing it does not recover
answers for AraRooPat — the rows where it looped hold no good answer behind the loop (1.72 once broken, against 2.08 /
2.32 for BPE / native SFT on the same prompts), and the penalty itself damages AraRooPat's grammar. For BPE and native
SFT the loops did hide recoverable answers (+0.43 / +0.26 on those rows) but the penalty costs as much elsewhere, so
their means do not move. A repetition penalty is therefore neither a fix nor a fair comparison rule; the greedy rule
stays, and the lever for AraRooPat's loops is the model — P2 is the right bet. Cost: five eval-only runs 19.6 min
(greedy check 5.4, AraRooPat 2.8, BPE 1.7, native SFT 3.4, native base 6.4), one judge pass 5.9 min incl. cold start.

**P2 — AraRooPat v4: Phase 3 on more and longer free-form data.** `configs/experiments/qwen_araroopat_3phase_v4.yaml`
→ cell `araroopat_3phase_v4`: v3's Phase 2 checkpoint (`…/araroopat_3phase_v3/training/warmup`), phases 1/2 off, no
`embedding_init` (the log shows `Vocab size unchanged (17184), skipping resize`), tokenizer
`araroopat_maxpat40k_v3` loaded, the free-form eval of ffstop. Only the Phase 3 block differs from ffstop:

| | ffstop | v4 |
|---|---|---|
| mixture | 30 000 at 40/30/30 | **45 000 at 25/15/60** (11 250 / 6 750 / 27 000; 9 000 per free-form corpus) |
| max_length / batch × accumulation | 512 / 4 × 4 | **1 024 / 2 × 8** (16 sequences per update in both) |
| steps (micro) / updates / warm-up | 7 500 / 1 875 / 400 = 100 updates | 22 500 / 2 813 / 800 = 100 updates |
| early stop | eval_mixture 1 000, every 200, min 500 | eval_mixture 1 000, **every 500, min 1 500** |
| unchanged | LR 2e-5 cosine, `within_category: equal`, `upsample`, `drop_truncated_answers`, seed 42, patience 5, min_delta 5e-4, restore best | |

*Dry run* (`scripts/plan_sft_mixture.py --config <v4> --tokenizer-path outputs/tokenizers/araroopat_maxpat40k_v3`,
6 min 38 s on CPU; the run's own mixture log is identical line for line). Drops are *truncation* (the prompt left no
room for the answer) / *cut answer* (512 or 1 024 tokens cut the answer; `drop_truncated_answers` skips it and tops
up); the 512 columns are ffstop's manifest, for its smaller quotas:

| corpus | available | planned | drawn | drops at 1 024 | drops at 512 (ffstop) | repeated (1 024 / 512) |
|---|---|---|---|---|---|---|
| tydiqa_arabic | 13 610 | 10 700 | 10 920 | 205 / 15 | 1 998 / 230 | 0 / 60 |
| arcd | 550 | 550 | 560 | 10 / 0 | 135 / 21 | 9 / 118 |
| arabic_squad_mcq | 45 936 | 6 750 | 6 789 | 38 / 1 | 3 982 / 109 | 0 / 0 |
| cidar | 9 227 | 9 000 | 9 033 | 0 / 33 | 1 / 138 | 0 / 0 |
| bactrian_x_ar | 63 636 | 9 000 | 9 151 | 78 / 73 | 112 / 454 | 0 / 0 |
| aya_ar | 18 738 | 9 000 | 9 637 | 325 / 312 | 463 / 346 | 0 / 0 |

Free-form drop rates: 1.45 % truncation + 1.50 % cut answer of 27 821 draws at 1 024, against 5.48 % + 8.92 % of
10 514 at 512. **cidar did not run dry** (9 000 kept of 9 033 drawn), so nothing spilled to Bactrian-X / Aya — the
brief's expectation of a few hundred did not materialise; ARCD repeated 9 records under `upsample`. Loss tokens
4 503 629 (ffstop 1 431 883): extractive 5.1 %, MCQ 0.6 %, **free-form 94.3 %** (ffstop 17.1 / 2.5 / 80.4 %); per
answer cidar 145.8, Bactrian-X 196.3, Aya 129.7 tokens. The eval mixture is 250 / 150 / 600 records = 5 554 / 600 /
93 844 answer tokens (5.6 / 0.6 / 93.8 %). *Memory smoke* (`total_examples: 400`, 200 micro-steps, no stop, no eval,
no checkpoint; scratchpad config, output `outputs/experiments/_smoke/araroopat_v4_smoke/`): peak **46.5 GB** of 80 GB
(`nvidia-smi`, 0.5 s sampling), 0.165 s per micro-step — batch 2 × 1 024 fits, the batch-1 fallback was not needed.

*Run* `scripts/run_experiment.py --config configs/experiments/qwen_araroopat_3phase_v4.yaml`, 09:11:02 → 10:32:48
UTC (1 h 22 min): corpus load + intrinsic metrics 6 min (CPU), mixture build 4.5 min, **Phase 3 62.6 min** (3 754.9 s = 48.9 min
training + 38 eval passes × 21.6 s = 13.7 min), free-form eval 8.4 min (ffstop ≈ 5 min: longer raw generations, see
P3). Training cost 73.3 s per 1 000 examples at 1 024 tokens against 43.4 s at 512 (93.9 against 57.6 s with the
evals). **Stop:** patience ran out at micro-step 20 000; best 1.0979 at **17 500 → 35 000 of the 45 000 examples**
(≈ 21 000 free-form records, ≈ 2.9 × ffstop's ≈ 7 200). Stop curve (mixture metric per category; this eval set is
25/15/60 at 1 024 tokens and is not comparable with ffstop's 0.9958):

| micro-step | total | extractive | mcq | free_form |
|---|---|---|---|---|
| 1 500 | 1.1525 | 0.1379 | 0.1444 | 1.2190 |
| 3 000 | 1.1420 | 0.1527 | 0.1128 | 1.2072 |
| 6 000 | 1.1242 | 0.1338 | 0.0985 | 1.1894 |
| 10 000 | 1.1069 | 0.1256 | 0.0905 | 1.1715 |
| 14 000 | 1.1001 | 0.1241 | 0.0807 | 1.1644 |
| 16 000 | 1.0985 | 0.1201 | 0.0802 | 1.1629 |
| **17 500** | **1.0979** | 0.1190 | 0.0798 | 1.1623 |
| 20 000 | 1.0977 | 0.1189 | 0.0799 | 1.1621 |

The free-form loss fell monotonically and flattened from ~14 000 (the last 6 000 micro-steps gained 0.0023); the
improvements after 17 500 were below `min_delta`.

*Measured* (`scripts/diag_heldout_loss.py --cell …/araroopat_3phase_v4`, 42 s — the first live run of P0's fields;
judge `gemma4_31b`, 250 verdicts in 27 s, 4.5 min with the cold start; rule set of the reference cells):

| | **AraRooPat v4** | AraRooPat ffstop | BPE ffstop | native base | native SFT |
|---|---|---|---|---|---|
| mixture examples restored | **35 000** of 45 000 | 24 000 of 30 000 | 26 400 of 30 000 | — | — |
| judge ± SE | **2.18 ± 0.09** | 2.15 ± 0.09 | 2.00 ± 0.08 | 2.42 ± 0.09 | 2.28 ± 0.09 |
| hist 1/2/3/4/5 | 108/66/31/12/33 | 112/67/26/11/34 | 117/71/30/10/22 | 92/69/29/13/47 | 105/60/31/18/36 |
| judge off-topic / repetition | 16.8 % / 11.6 % | 18.0 % / 16.8 % | 13.2 % / 17.6 % | 12.8 % / 9.6 % | 14.0 % / 15.6 % |
| chrF / BERTScore-F1 | 18.61 / 0.86 | 17.70 / 0.86 | 16.60 / 0.85 | 22.94 / 0.85 | 18.59 / 0.86 |
| loop stop / EOS / cap | **23.2** / 76.4 / 0.4 % | 21.2 / 77.6 / 1.2 % | 19.6 / 75.6 / 4.8 % | 6.8 / 90.8 / 2.4 % | 12.4 / 87.2 / 0.4 % |
| mean chars / ≤ 40-char share | 198 / 19.6 % | 190 / 23.2 % | 211 / 20.8 % | 511 / 4.0 % | 207 / 18.8 % |
| raw-text loss (nats/char, held-out set) | 0.875 | 0.875 | 0.859 | 0.826 | 0.838 |
| answer NLL (nats/char, `nll_per_answer_char`, as encoded) | **0.766** | 0.788 | 0.845 | 0.807 | 0.766 |
| P(EOS) / EOS rank-1 | 0.62 / 0.83 | 0.65 / 0.87 | 0.62 / 0.86 | 0.42 / 0.76 | 0.64 / 0.91 |

Losses are given at three decimals because two would hide the 0.022 answer-NLL gain (v4 0.7664, native SFT
0.7662, ffstop 0.7880).

Paired bootstraps (10 000 resamples, same 250 prompts):

| pair | Δ | 95 % CI | win / tie / loss |
|---|---|---|---|
| AraRooPat v4 − AraRooPat ffstop | +0.03 | [−0.09, +0.16] | 18 / 65 / 17 % |
| AraRooPat v4 − BPE ffstop | **+0.19** | **[+0.03, +0.35]** | 33 / 46 / 21 % |
| AraRooPat v4 − native SFT | −0.10 | [−0.25, +0.06] | 21 / 55 / 24 % |
| AraRooPat v4 − native base | −0.23 | [−0.41, −0.05] | 19 / 49 / 32 % |

Sub-scores vs ffstop: correctness +0.02 [−0.11, +0.16], fluency +0.02 [−0.11, +0.16], **instruction following
+0.18 [+0.03, +0.34]**. The +0.19 over BPE is the first AraRooPat-vs-BPE interval above zero, but it is **not a
tokenizer finding**: AraRooPat had the v4 Phase 3 and BPE did not (P4 below).

**Acceptance (P2.4).** Loop stop ≤ 12 % and in any case < 21.2 % — **23.2 % ✗** (both); judge ≥ 2.152 with the
paired CI vs ffstop not entirely below 0 — **2.18, [−0.09, +0.16] ✓**; answer NLL per character ≤ ffstop's
(0.788 as encoded; 0.787 over the raw 72 705) — **0.766 (raw 0.766) ✓**; raw-text loss within +0.03 of the Phase 2
checkpoint's 0.8595 — **0.8747, +0.015 ✓**; mean answer 150–500 chars — **198 ✓**; EOS ≥ 77.6 % — **76.4 % ✗**. Four
of six. **Gate for P4** (paired CI vs ffstop above 0, or loop stop ≤ 15 %): **not passed** — the CI spans zero and
loops rose. P4 was not run; its two configs are validated and committed.

**What the run says about the hypothesis.** Three times the distinct free-form records, the long-answer tail kept,
and 35 000 examples restored moved the teacher-forced numbers — answer NLL 0.788 → 0.766 nats/char, *level with
native SFT* (0.766) — and did not move generation: judge +0.03 (CI spans 0), loops 21.2 → 23.2 %, EOS 77.6 → 76.4 %,
P(EOS) at the reference end 0.65 → 0.62. So Phase 3 data was not the lever for what the judge scores. The gap to the
native arms is no longer in how likely the model finds a good answer; at equal answer NLL per character native SFT
loops on 12.4 % of prompts and AraRooPat on 23.2 %. It sits in free-running generation — the regime P1 probed, where
AraRooPat's loop prompts hold no recoverable answer behind the loop and a token-level penalty damages its clitics.
Per the brief the next step is decided from P1 or distillation; P1 rules out a token-level penalty as a fix, so the
candidates are sequence-level: distillation from the native model (the campaign's planned step 5), or a training-time
repetition objective of the DITTO kind (Xu et al. 2022, §7) — and, as a cheap measurement before either, a
decoded-text (word-level) no-repeat constraint that cannot hit clitic tokens.

**P3 — analysis.** *Loops by period bucket* (1 / 2–3 / 4–10 / > 10 words, all by the periodic rule, no char-rule
stops): v4 **11 / 15 / 16 / 16** (58), ffstop 5 / 17 / 21 / 10 (53), native SFT 4 / 8 / 13 / 6 (31), BPE ffstop
6 / 11 / 21 / 11 (49), native base 0 / 4 / 6 / 7 (17). v4 has more period-1 runaways and more long-period (> 10
words) loops, fewer 2–10. Loop rows among the judge-score-1 rows: v4 29 of 108 (27 %), ffstop 31 of 112 (28 %),
native SFT 13 of 105 (12 %). v4's loop rows run much further before the stop fires — 661 raw characters on average
against ffstop's 223 (234 kept against 140) — which is why its free-form eval took 8.4 min: 287 generated tokens per
prompt against 182, at a similar 145 vs 160 tokens/s.

*Loop rate by reference-length tercile* (84 / 83 / 83 prompts; reference median 80 / 219 / 485 chars):

| cell | short | medium | long |
|---|---|---|---|
| AraRooPat v4 | 14.3 % | 22.9 % | **32.5 %** |
| AraRooPat ffstop | 16.7 % | 19.3 % | 27.7 % |
| BPE ffstop | 10.7 % | 18.1 % | 30.1 % |
| native SFT | 7.1 % | 13.3 % | 16.9 % |
| native base | 4.8 % | 4.8 % | 10.8 % |

The long-answer hypothesis predicted v4's gain in the top tercile; the top tercile is where v4 got *worse* (+4.8
points), the short tercile improved (−2.4). In every cell the loop rate rises with the reference length.

*Answer length by tercile* (mean / median generated chars; ≤ 40-char share):

| cell | short | medium | long | ≤ 40 overall |
|---|---|---|---|---|
| AraRooPat v4 | 126 / 62 (31.0 %) | 210 / 111 (16.9 %) | 258 / 204 (10.8 %) | 19.6 % |
| AraRooPat ffstop | 100 / 57 (35.7 %) | 182 / 107 (22.9 %) | 289 / 222 (10.8 %) | 23.2 % |
| native base | 322 / 241 (8.3 %) | 574 / 346 (1.2 %) | 640 / 439 (2.4 %) | 4.0 % |

Judge by tercile, v4 / ffstop: short 2.18 / 2.20, medium 2.37 / 2.29, long 2.00 / 1.96.

*Examples* (16 rows gained ≥ 2 judge points, 15 lost ≥ 2 — a wash, as the mean says):

| id | instruction | ffstop answer (judge) | v4 answer (judge) |
|---|---|---|---|
| cidar-10118 | حدد ما إذا كانت كلمة المكاسب مبتدأ أو لا في الجملة التالية. شهدت المكاسب القانونية. | كلمة المكاسب مبتدأ في الجملة . (1) | لا ، كلمة المكاسب ليست مبتدأ في الجملة السابقة . (5) |
| cidar-202 | صف نظام الاقتصاد في سوريا حالياً. | الاقتصاد في سوريا هو نظام اقتصادي محلي . (2) | سوريا لديها نظام اقتصادي مختلط ، حيث يتم دمج القطاع العام والقطاع الخاص … (5) |
| cidar-246 | اشرح ما هي القوة المركزية؟ | … وتواجه الجسم المتحرك في الدائرة … قوة كهرومغناطيسية أو قوة كهرومغناطيسية (2) | القوة المركزية هي القوة التي تعمل على جسم يتحرك في مدار دائري … وهي موجهة نحو المركز (5) |
| cidar-1152 | اذكر 4 ميزات لورقة بحث علمي. | four numbered features, complete (5) | two features, then a loop stop: «المحتوى الدقيق والدقيق» (2) |
| cidar-3388 | صف الأزمة الفلسطينية. | a dated factual account (5) | generic statements repeated in two forms: «واحدة من القضايا الأكثر …» (2) |
| cidar-3491 | أنشئ جملة مثالية تستخدم الفعل "يبتلع". | يبتلع الثعبان الأسماك الصغيرة في المياه . (5) | ابتلع الماء بسرعة للبقاء رطبا . — past tense, not the verb asked for (2) |

*GPU time (2026-09-23).* P1: five eval-only runs 19.6 min (greedy check 5.4, AraRooPat 2.8, BPE 1.7, native SFT
3.4, native base 6.4) + judge 5.9 min; smoke 1.3 min (+ a 4-second failed start: `sweep.tasks` may not be empty);
v4 run 1 h 22 min (Phase 3 62.6, free-form eval 8.4, the rest CPU-bound set-up with the model resident); diagnostic
0.7 min; judge 4.5 min — **≈ 1 h 54 min**. CPU only: mixture dry run 6.6 min, backfill < 1 min, tests ≈ 4 min.

**Artifacts.** `outputs/experiments/qwen_decoding_ablation/{araroopat_3phase_v3_ffstop,bpe_16k_3phase_v3_ffstop,
native_qwen3_sft,native_qwen3_base}_rp12/` (`all_metrics.json`, `eval_rows/freeform_cidar.parquet`,
`freeform_judge/gemma4_31b.parquet`), `qwen_decoding_ablation/_repro/araroopat_3phase_v3_ffstop_greedy/` (the
reproduction check), `qwen_decoding_ablation/comparison_report.txt`; `outputs/experiments/qwen_native_vs_araroopat/
araroopat_3phase_v4/` (`all_metrics.json` with `training.sft.eval_history`, `data/sft_mixture_manifest.json`,
`data/sft_eval_mixture_manifest.json`, `diag_heldout_rawtext_v1.json`, `eval_rows/freeform_cidar.parquet`,
`freeform_judge/gemma4_31b.parquet`, `training/sft/`); the 19 backfilled `diag_heldout_*.json`. Configs:
`configs/experiments/ablation_decoding/*.yaml` (five), `qwen_araroopat_3phase_v4.yaml`, `qwen_bpe16k_3phase_v4.yaml`,
`qwen_native_sft_v4.yaml`. Scripts: `scripts/diag_backfill_per_char.py`, `scripts/judge/paired_compare.py` (two
`--experiment`), `scripts/diag_heldout_loss.py` (P0 fields). Commits: `4efedb3` (the §3.6 verification edits),
`e10863c` (P0), `ab2e679` (decoding knobs + ablation configs), `a216601` (`paired_compare.py`), `4e702ce` (v4 + P4
configs), and the commit carrying this section.

**Verification (2026-09-23 evening, Fable session, against the outputs).** Reproduced from the artifacts: the
backfilled per-character answer NLL of all eight checkpoints (the raw-denominator values equal the §3.6 list to four
decimals; the "as encoded" values differ only by AraRooPat's 69 dropped format characters); the v4 history (stop at
20 000, restored 17 500, best 1.0979), the mixture manifest row by row including the 512-window column taken from
ffstop's manifest and the cut-answer rates 8.9 / 4.2 / 2.8 % (AraRooPat / native / BPE at 512), the resize no-op
and the config fields; the judge means, histograms and flag rates of v4 and the four penalty cells; all eight paired
bootstraps; the loop buckets, the loop rate by reference tercile, the answer lengths, the 16 / 15 gains and losses
and the six example rows; the ablation's loop-row analysis (AraRooPat's 53 loop prompts 1.62 → 1.72 under the
penalty while its other 197 rows lose 0.41; BPE 2.00 → 2.08, native SFT 2.25 → 2.32, native base 2.83 → 2.30 on
the same prompts); the greedy reproduction cell (250 of 250 generations identical to ffstop); the penalty processor
(HF's formula over each row's context with the left padding masked); the P4 configs load and their Phase 3 block is
byte-equal to v4's; 195 tests pass across the eight listed files. No number had to be corrected. One definition to
keep in mind: the share of words carrying the article depends on how a prefix is counted — with a regex that allows
one proclitic before ال it is 0.29 → 0.15 for AraRooPat under the penalty (the run report's count gives 0.39 →
0.18), BPE 0.32 → 0.30, native 0.30 → 0.28: the same halving for AraRooPat and no change for the others, which is
the finding.

*Reading, and the hypothesis it leaves.* Two arms with new vocabularies loop on about 20 % of prompts (AraRooPat
ffstop 21.2, v4 23.2, BPE ffstop 19.6 %) where native SFT loops on 12.4 % — at equal held-out likelihood of the
references (v4 and native SFT both 0.766 nats/char) and after the data levers (exposure, distinct records, the long
tail) and the decoding lever have each been tried and measured. What the two from-scratch arms share and the native
arm does not is a vocabulary whose rows were learned in 131 M tokens and which, under Qwen3's tied embeddings, is
also the output head. Ivgi et al. (2024) describe repetition as the fallback a model reaches for under uncertainty,
and the 2026-09-18 self-reinforcement curve showed the greedy attractor closing within a few repetitions once
entered. The testable form: the from-scratch arms decode from flatter next-token distributions (higher entropy,
smaller top-1 margin) than the native model on the same prompts, and their loops start where the margin is smallest.
That is an eval-only measurement on the existing checkpoints (the next-token entropy and top-1 margin along each
generated answer, per arm, and at loop onset) and it decides between two remedies. If the distributions are flatter,
the remedy is sequence-level and tokenizer-agnostic: **distillation by teacher-generated text** — the native base
model (judge 2.42, loops 6.8 %) answers the 27 000 free-form training prompts once (vLLM, the judge environment,
under an hour), and v4's Phase 3 trains on those answers in place of the references. Logit or attention distillation
in the usual sense is not available here: the student and the teacher do not share an output vocabulary or a token
alignment. If the distributions are not flatter, the loop is a property of the *text* the arms learned to write and
the DITTO-style objective (Xu et al. 2022) is the candidate. Either way the P4 replications stay unrun until a Phase 3
block is worth making the reference.


### 3.8 Distillation from a neutral Arabic teacher — bake-off and three arms (2026-09-23/24)

**The five rules** (set by the user on 2026-09-23, stated in every v5 config header). (1) *Same tutor for everyone*: BPE-16K and native get the same teacher answers, mixture, seed and draw rule as AraRooPat, and all three arms run unconditionally, in the order AraRooPat → BPE-16K → native; the residual asymmetry is equal *examples*, not equal tokens (`drop_truncated_answers` at 1 024 tokens drops more long teacher answers under AraRooPat's fertility). (2) *The teacher never sees the test*: teacher answers are generated for training prompts only (the `train` and `dev` slices of cidar / bactrian_x_ar / aya_ar after the committed exclusions); the teacher is chosen on 250 cidar-**dev** prompts; the only time it answers the 250 held-out prompts is the ceiling cell (P1.4), whose answers stay in its `eval_rows/`. (3) *Don't copy the teacher's mistakes*: every answer that lacks EOS, exceeds 2 400 characters, loops, is degenerate, is under 20 characters, carries a Latin letter or echoes our template / the instruction is dropped before the student sees it, and not regenerated. (4) *Be honest about the ceiling*: the result is how much of the teacher's judge score survives the vocabulary change, against the teacher's own cell and the native distilled arm; no judge score is a threshold, target or gate. (5) *One teacher, whole*: the winner's answers only; no blend, no per-prompt selection by the judge; the judge touches training material once — 250 dev prompts per candidate, to choose a model.

**Why a neutral teacher.** With `Qwen/Qwen3-4B-Base` as teacher the native arm would imitate its own greedy output under its own vocabulary. Excluded: the judge's family (Gemma-4, Fanar-2) and any Qwen model (the native arm's vocabulary).

**P0 — uncertainty diagnostic on the existing checkpoints** (`scripts/diag_uncertainty.py`, eval only; bf16; the reference pass reuses `diag_heldout_loss`'s encodings, order and batches and stops unless it reproduces the cell's `nll_per_answer_char` to three decimals — it did in all five cells, |Δ| ≤ 1.7e-5). Two teacher-forced passes record per answer token the entropy H (nats, full vocabulary), the top-1 probability p1, the margin p1 − p2 and −log p: (a) the 250 held-out references (answer + EOS), (b) the cell's own 250 generations re-encoded (`prompt_text` as the generation encoded it + `generation_raw`). The loop contrast takes the 32 answer tokens before the onset of each loop-stopped row (the first token that emits a character of the first copy of the repeated unit) against the same absolute token window in the EOS-terminated rows long enough to have it (averaged per looped row, then over rows); in-loop p1 averages the ≤ 64 tokens from the onset.

| cell | refs: H/char | H/token | p1 | margin | p1 < 0.5 | NLL/char (cross-check) | gens: H/char | p1 | margin | round-trip exact | before onset: H / p1 / margin | control: H / p1 / margin | in-loop p1 | looped rows (with control) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| native base | 0.766 | 1.747 | 0.611 | 0.514 | 41.6 % | 0.807 (0.807) | 0.317 | 0.791 | 0.719 | 250/250 | 1.394 / 0.674 / 0.585 | 1.037 / 0.746 / 0.661 | 0.751 | 17 of 17 (17) |
| native SFT | 0.764 | 1.742 | 0.612 | 0.514 | 41.2 % | 0.766 (0.766) | 0.439 | 0.745 | 0.672 | 250/250 | 1.683 / 0.622 / 0.529 | 1.433 / 0.676 / 0.588 | 0.753 | 28 of 31 (28) |
| BPE ffstop | 0.846 | 2.130 | 0.590 | 0.500 | 43.2 % | 0.845 (0.845) | 0.370 | 0.832 | 0.793 | 250/250 | 2.603 / 0.509 / 0.425 | 2.104 / 0.586 / 0.494 | 0.716 | 47 of 49 (47) |
| AraRooPat ffstop | 0.767 | 1.513 | 0.666 | 0.571 | 34.5 % | 0.788 (0.788) | 0.438 | 0.798 | 0.733 | 241/250 | 1.426 / 0.693 / 0.604 | 1.176 / 0.737 / 0.654 | 0.753 | 45 of 53 (45) |
| AraRooPat v4 | 0.744 | 1.467 | 0.674 | 0.579 | 33.4 % | 0.766 (0.766) | 0.341 | 0.836 | 0.781 | 237/250 | 1.367 / 0.696 / 0.604 | 1.047 / 0.757 / 0.675 | 0.793 | 52 of 58 (52) |

Skipped loop rows: the onset at the first answer token (native SFT 3, BPE 1, AraRooPat ffstop 8, v4 6) — no window before it; one BPE row char-truncated. The share of looped rows whose pre-onset margin is below its control: native base 82 %, native SFT 68 %, BPE 68 %, AraRooPat ffstop 60 %, v4 81 %. Decode round trip (`decode(encode(generation_raw)) == generation_raw`): exact on every row for native and BPE, 241 / 250 (ffstop) and 237 / 250 (v4) for AraRooPat — the re-encoded generation is the emitted sequence on those rows only up to the canonicalisations of §3.5.

Mean H by answer-position decile (references / generations):

| cell | references d1…d10 | generations d1…d10 |
|---|---|---|
| native base | 1.99 1.85 1.94 1.96 1.90 1.86 1.83 1.77 1.77 1.91 | 1.16 1.15 1.18 1.22 1.20 1.02 0.96 1.02 0.92 0.95 |
| native SFT | 2.06 1.87 1.94 1.95 1.87 1.85 1.81 1.73 1.73 1.66 | 1.83 1.48 1.55 1.55 1.49 1.47 1.35 1.23 1.04 1.13 |
| BPE ffstop | 2.64 2.48 2.52 2.54 2.55 2.39 2.29 2.33 2.26 2.17 | 2.37 2.12 2.14 2.11 2.04 1.88 1.71 1.58 1.36 1.28 |
| AraRooPat ffstop | 1.72 1.61 1.62 1.69 1.63 1.57 1.59 1.52 1.50 1.44 | 1.72 1.35 1.33 1.32 1.27 1.11 1.00 0.95 0.86 0.94 |
| AraRooPat v4 | 1.62 1.55 1.56 1.64 1.58 1.51 1.52 1.46 1.45 1.37 | 1.59 1.19 1.28 1.18 1.12 1.02 0.98 0.92 0.87 0.80 |

*Reading.* The hypothesis of §3.7 *Verification* predicted (i) higher entropy per character on the references for the from-scratch arms than for native SFT at equal answer NLL, and (ii) a lower margin before a loop onset than in the position-matched control. **(i) does not hold**: at equal answer NLL (0.766 nats/char both) AraRooPat v4's reference entropy is *lower* than native SFT's (0.744 against 0.764 nats/char); ffstop sits at native's level (0.767) with a higher NLL, and BPE's higher entropy (0.846) tracks its higher NLL (0.845) — in every cell entropy per character and NLL per character move together, so the from-scratch vocabularies are not flatter than native at equal likelihood. Along their own generations they are not flatter either (0.341–0.438 nats/char against native SFT's 0.439). **(ii) holds in every cell, the untrained native base included**: the 32 tokens before a loop onset have a lower margin than the same positions in the EOS-terminated rows (native base 0.585 against 0.661, native SFT 0.529 / 0.588, BPE 0.425 / 0.494, AraRooPat ffstop 0.604 / 0.654, v4 0.604 / 0.675), and p1 rises inside the loop (0.72–0.79 against 0.51–0.70 before the onset). So loops start where the model is least sure in every arm — the Ivgi et al. fallback — but that is not because the from-scratch arms decode from flatter distributions: per character they are as sharp as native or sharper. What differs between the arms is how often a generation reaches a low-margin point that opens into a copy, not the average sharpness. The per-token numbers are comparable within a cell only (AraRooPat's `[PAT_*]` after `[ROOT_*]` is predictable, hence its lower per-token entropy). P0 gates nothing; the distillation of P1–P3 runs as planned.

**P1.1–P1.3 — the bake-off.** *Prompts* (`scripts/distill/dump_bakeoff_prompts.py`): `load_corpus("cidar", "dev")` with the committed exclusions (500 dev records, 13 excluded — the held-out prompts that hash into the dev slice — 487 left), sorted ids, `numpy.random.default_rng(42).permutation`, the first 250 → `outputs/data_cache/distill/bakeoff/prompts_dev.jsonl` (sha256 `2116ff8274b9a38e…`; strata by the held-out set's boundaries 141 / 300 characters: 72 short / 78 medium / 100 long; mean reference 297 characters). Zero id overlap and zero normalized-text matches with the 250 held-out prompts (`bakeoff_manifest.json → heldout_check`). *Generation* (`scripts/distill/run_teacher.sh` → `generate_teacher_answers.py`, vLLM 0.29.0 in `.venv-judge`): greedy, seed 0, repetition penalty 1.0, `max_model_len` 4 096, the candidate's own chat template through `LLM.chat` with the fixed Arabic system prompt (sha256 `7252b4fc6a9f3f2e…`) — every template rendered a system turn natively —, `enable_thinking: false` passed to every template, stop at the model's own EOS, no stop markers, cap = `derive_token_cap` over `measure_chars_per_token` on the 250 dev references with the candidate's tokenizer (special ids included). *Determinism*: the first 200 prompts again in a separate process. *Pseudo-cells* (`scripts/distill/build_pseudo_cell.py`) under `outputs/experiments/teacher_bakeoff/<slug>/`, post-processed with the eval's text rules and no stop markers; the reference row `native_qwen3_base_dev` is the untouched base through the pipeline's own free-form task on the same file (`configs/experiments/teacher_bakeoff_native_base_dev.yaml`, cap pinned at 1 209 with `token_cap_floor = token_cap_ceiling` — the dev references alone would have derived 1 074). Judge `gemma4_31b`, one pass over the folder (1 750 verdicts; 628 s with the cold start), baseline `native_qwen3_base_dev`; paired bootstraps 10 000 resamples, seed 0.

| candidate | repo @ snapshot | loaded as | cap | load / gen wall (s) | tok/s | determinism | judge ± SE | hist 1/2/3/4/5 | Δ vs native_qwen3_base_dev [95 % CI] | win / tie / loss | kept | loop / cap / EOS | latin | mean chars | projected full wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jais2_8b | inceptionai/Jais-2-8B-Chat @ `da0e1639` | bf16 | 725 | 95.6 / 9.5 | 8 280 | 89.0 % | 4.21 ± 0.07 | 9/16/39/35/151 | +0.77 [+0.56, +0.98] | 45 / 38 / 17 % | **59.2 %** | 0.0 / 16.0 / 84.0 % | 28.8 % | 1 113 | 0.47 h |
| llama33_70b_awq | casperhansen/llama-3.3-70b-instruct-awq @ `64d25562` | AWQ int4 (fp16) | 1 094 | 168.9 / 33.8 | 1 234 | 87.5 % | 4.26 ± 0.08 | 14/18/25/25/168 | +0.82 [+0.64, +1.00] | 45 / 46 / 8 % | 82.4 % | 1.6 / 0.4 / 98.0 % | 12.0 % | 395 | 1.63 h |
| falcon_h1_34b | tiiuae/Falcon-H1-34B-Instruct @ `6e6890e2` | bf16, `max_num_seqs` 32 | 757 | 162.0 / 78.9 | 458 | 42.0 % | 4.32 ± 0.08 | 11/20/23/20/176 | +0.88 [+0.70, +1.06] | 46 / 47 / 8 % | **71.6 %** | 0.4 / 0.4 / 99.2 % | 26.0 % | 512 | 3.74 h |
| cmdr7b_arabic | CohereLabs/c4ai-command-r7b-arabic-02-2025 @ `2d5440c7` | bf16 | 1 025 | 97.3 / 7.6 | 7 152 | 53.5 % | 4.43 ± 0.08 | 14/16/16/7/197 | +0.98 [+0.81, +1.17] | 46 / 50 / 4 % | 88.4 % | 0.0 / 0.0 / 100.0 % | 8.4 % | 573 | 0.38 h |
| **aya_expanse_32b** | CohereLabs/aya-expanse-32b @ `b306ea27` | bf16 | 1 025 | 137.0 / 27.8 | 2 891 | 39.0 % | **4.64 ± 0.06** | 7/11/10/9/213 | **+1.20 [+1.02, +1.38]** | 52 / 46 / 3 % | 84.8 % | 0.0 / 0.4 / 99.6 % | 14.0 % | 863 | 1.34 h |
| allam_7b | ALLaM-AI/ALLaM-7B-Instruct-preview @ `a28dd1e6` | bf16 | 707 | 73.7 / 5.8 | 4 665 | 60.0 % | 4.50 ± 0.07 | 11/10/16/18/195 | +1.06 [+0.86, +1.27] | 50 / 42 / 8 % | 91.6 % | 1.6 / 0.4 / 98.0 % | 3.6 % | 392 | 0.29 h |
| native_qwen3_base_dev (reference) | Qwen/Qwen3-4B-Base, our template | bf16 (HF generate) | 1 209 | — / 521.7 | — | — | 3.44 ± 0.10 | 40/44/39/19/108 | — | — | — | 6.8 / 3.2 / 90.0 % | 19.2 % | 735 | — |

*Kept* = the share of the 250 dev answers that survive the P1.6 filters (drops, in rule order: Jais-2 40 length + 19 over 2 400 characters + 7 under 20 characters + 36 Latin; Llama-70B 4 / 0 / 1 loop / 11 / 28; Falcon-H1 2 / 0 / 0 / 6 / 63; Command-R7B 0 / 1 / 0 / 10 / 18; Aya 1 / 3 / 0 / 2 / 32; ALLaM 4 / 0 / 1 loop + 1 degenerate / 7 / 8). The Latin rule is the eval's row flag — any Latin letter —, and nearly all its hits are names and technical terms the system prompt allows (LinkedIn, HTML, a generated password); it is the rule the brief sets for training material, and it decides eligibility for Jais-2 and Falcon-H1. *Projected full wall* = 42 100 prompts × the candidate's mean answer tokens ÷ its bake-off tokens per second + load. Sub-scores (correctness / fluency / instruction following): Aya 4.68 / 4.97 / 4.78, ALLaM 4.59 / 4.93 / 4.66, Command-R7B 4.48 / 4.94 / 4.60, Falcon-H1 4.52 / 4.64 / 4.63, Llama-70B 4.45 / 4.69 / 4.64, Jais-2 4.54 / 4.96 / 4.42, the reference row 3.76 / 4.56 / 3.92. By stratum (short / medium / long): Aya 4.58 / 4.62 / 4.70, ALLaM 4.43 / 4.45 / 4.60, the reference 3.53 / 3.32 / 3.48. Judge flags on Aya: off-topic 0.4 %, truncated 1.2 %, repetition 0 %.

*Loading notes.* Every candidate loaded; Falcon-H1-34B in bf16 at the default `max_num_seqs` ran out of memory in vLLM's CUDA-graph profiling (`torch.OutOfMemoryError … Tried to allocate 73.12 GiB` in `_init_minimal_kv_cache_for_profiling`, on top of 62.99 GiB of weights — a Mamba state per sequence slot), and the brief's fp8 / GPTQ fallbacks would not have fixed an allocation that does not depend on the weights, so a capacity fallback was added first (`engine_kwargs: {max_num_seqs: 32}`, bf16) and it loaded — hence its 458 tok/s. `aya-expanse-32b` was loaded from the local cache of `CohereForAI/aya-expanse-32b` (the repo under its former owner name) at the same commit `b306ea27` the `CohereLabs` name resolves to. vLLM's greedy output is not bit-stable across batch compositions: the determinism column is the exact-match share of the first 200 answers between two processes (39–89 %).

*The choice.* **Aya-Expanse-32B**, by the rule as written: among the four candidates with kept ≥ 80 % (Llama-70B, Command-R7B, Aya, ALLaM) the highest judge mean, 4.64; the runner-up ALLaM-7B (4.50) is 0.14 below — outside one SE (0.06) — so no tie-break applied; projected full wall 1.34 h (< 6 h), throughput rule not applied. **Stop point: proceeded** — every candidate's mean minus one SE is above the reference row's 3.44 (Aya 4.58). Two facts next to the rule: the *paired* Aya − ALLaM difference is +0.14 [−0.00, +0.27], a CI that touches zero, and Aya writes the longest answers of the eligible four (863 characters against the references' 297, ALLaM 392) — the judge's rubric says not to reward length, and Aya's lead holds in every stratum. The dev prompts are markedly easier than the held-out test set: the same untouched base scores 3.44 on them against 2.42 on the test prompts.

**P1.4 — the teacher's ceiling on the test** (rule 2's one permitted use; rule 4's denominator). Aya-Expanse-32B answers the 250 held-out prompts of `configs/contamination/freeform_cidar_heldout_v1.jsonl` exactly as in the bake-off (same system prompt, template, cap 1 025, greedy; 95 s including the load), written as the pseudo-cell `outputs/experiments/qwen_native_vs_araroopat/teacher_aya_expanse_32b_m2_c2400/` (`kind: teacher_ceiling`, `heldout_sha256` `37244059aa92ac48…` in the dump metadata; the raw answers are in that cell's `eval_rows/` and nowhere else) and judged with the main folder's call (`--baseline native_qwen3_base_m2_c2400`; 250 verdicts in 41 s, every other cell reused):

| | teacher ceiling (Aya-Expanse-32B) | native base | native SFT |
|---|---|---|---|
| judge ± SE | **3.65 ± 0.09** | 2.42 ± 0.09 | 2.28 ± 0.09 |
| hist 1/2/3/4/5 | 18/59/36/16/121 | 92/69/29/13/47 | 105/60/31/18/36 |
| paired Δ of the teacher [95 % CI], win / tie / loss | — | +1.24 [+1.06, +1.42], 60 / 34 / 6 % | +1.37 [+1.18, +1.56], 69 / 26 / 6 % |
| sub-scores Δ vs native base (correctness / fluency / instruction following) | +1.18 / +0.39 / +1.15 | | |
| loop / EOS / cap | 0.0 / 100.0 / 0.0 % | 6.8 / 90.8 / 2.4 % | 12.4 / 87.2 / 0.4 % |
| mean chars / ≤ 40-char share | 608 / 2.4 % | 511 / 4.0 % | 207 / 18.8 % |
| judge off-topic / repetition / truncated | 2.0 / 0.0 / 0.0 % | 12.8 / 9.6 / 6.8 % | 14.0 / 15.6 / 4.4 % |
| chrF / BERTScore-F1 | 26.61 / 0.86 | 22.94 / 0.85 | 18.59 / 0.86 |
| judge by tercile (short / medium / long) | 3.81 / 3.39 / 3.76 | 2.58 / 2.23 / 2.43 | 2.30 / 2.36 / 2.18 |

The test prompts are harder than the dev prompts for the teacher too (4.64 on dev, 3.65 on test), as for the native base (3.44 / 2.42).

**P1.5–P1.6 — the full generation and the filter.** `scripts/distill/dump_teacher_prompts.py --slug aya_expanse_32b` → `outputs/data_cache/distill/aya_expanse_32b_v1/prompts.jsonl`: every dev record of the three corpora (cidar 487, bactrian_x_ar 3 375, aya_ar 1 011), every `cidar/train` record (9 227) and, for `bactrian_x_ar/train` and `aya_ar/train`, a seeded subset (sorted ids, `numpy.random.default_rng(42).permutation`, the first N) — 14 000 each at first (42 100 prompts). Zero id overlap and zero normalized-text matches with the held-out prompts (`prompts_manifest.json → heldout_check`). Generation with P1.2's settings (bf16, cap 1 025): 42 078 answered in 48.4 min (2 813 s of generation, 11.51 M tokens, 4 090 tok/s), 22 skipped because the rendered chat exceeded 4 096 − 1 025 tokens, 41 932 ended at EOS and 146 at the cap. **The filter kept too few**: `bactrian_x_ar/train` 9 514 and `aya_ar/train` 9 338 of 14 000 — below the brief's 11 000 (cidar/train keeps 7 394 of 9 227, so its deficit of 1 606 under a 9 000-per-corpus quota spills onto the other two, which then need 9 803 each before any truncation drop). Per the brief both subsets were extended by the next 4 000 ids of the same permutation (`--train-subset bactrian_x_ar=18000 aya_ar=18000`; the first 14 000 are unchanged by construction) and the generation resumed on the 7 995 new prompts (595 s, 3 694–4 223 tok/s; 27 skipped for length in total). Part-1 files are kept next to the final ones (`*_part1.json`, `run_part1.log`). Final prompt file sha256 `0e2977c6208e0899…` (50 100 prompts).

`scripts/distill/filter_teacher_answers.py` (the P1.6 rules in order; dropped prompts not regenerated) → `outputs/data_cache/distill/aya_expanse_32b_v1/teacher_answers_v1.jsonl`, **35 066 records, sha256 `8e4d9cd13e3d1bc8a2cd0782266f570dc771a55f500e0609a1c808955bbcf272`**, and `manifest.json`:

| corpus / split | prompts | skipped (length) | length | char_truncated | loop | degenerate | empty | latin | echo | kept | kept % | preamble (answered) | teacher chars kept, mean / median | reference chars, same records |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cidar / dev | 487 | 0 | 2 | 8 | 0 | 0 | 5 | 73 | 0 | 399 | 81.9 % | 2.5 % | 753 / 635 | 276 / 233 |
| cidar / train | 9 227 | 0 | 22 | 160 | 0 | 0 | 87 | 1 564 | 0 | 7 394 | 80.1 % | 2.4 % | 797 / 693 | 299 / 234 |
| bactrian_x_ar / dev | 3 375 | 2 | 6 | 56 | 2 | 0 | 57 | 982 | 0 | 2 270 | 67.3 % | 1.8 % | 706 / 559 | 389 / 286 |
| bactrian_x_ar / train | 18 000 | 4 | 30 | 247 | 2 | 2 | 310 | 5 126 | 0 | 12 279 | 68.2 % | 2.1 % | 698 / 531 | 388 / 281 |
| aya_ar / dev | 1 011 | 2 | 8 | 6 | 0 | 0 | 18 | 282 | 0 | 695 | 68.7 % | 4.5 % | 581 / 442 | 267 / 129 |
| aya_ar / train | 18 000 | 19 | 110 | 77 | 0 | 0 | 360 | 5 404 | 1 | 12 029 | 66.8 % | 5.2 % | 581 / 385 | 297 / 144 |
| **total** | 50 100 | 27 | 178 | 554 | 4 | 2 | 837 | **13 431** | 1 | **35 066** | 70.0 % | 3.3 % | | |

The Latin rule is 89 % of all drops (13 431 of the 15 034 prompts without a kept answer). Loops are rare in the teacher (4: periods 1, 2, 2, 5). The system prompt asked for no preamble (3.3 % of answers open with one) and no markdown unless asked (on the first 4 000 answers: bold in 46.7 %, a list line in 58.5 %, a markdown header in 6.8 %). The teacher writes 1.8–2.7 × the reference length on the same records.

**P2 — the teacher-answer overlay** (commit `345bf51`). `training.corpus_params.{cidar, bactrian_x_ar, aya_ar}.teacher_answers: <jsonl>` reaches `_load_cidar` / `_load_bactrian_x_ar` / `_load_aya_ar` like `include_datasets` does, and one helper, `finetune_corpora._apply_teacher_answers(records, path, corpus, split)`, replaces each record's `answer` by the teacher's (matched by record id) and **drops every record the file lacks** — the student trains on teacher text only, no reference answer leaks back in. It runs inside each loader after `_partition_dev` and before `load_corpus` applies the exclusions (the order is immaterial: both act on ids, and the teacher file was built from post-exclusion prompts). `id`, `question`, `context`, `source` and `prompt_template` are untouched, so the dev partition (`is_dev_id` on the same ids), the exclusions and the mixture walk (sorted ids, seeded permutation) are unchanged, and the student's prompt is `_format_qa_prompt(record)` — our Alpaca-AR template; the teacher's chat template never appears in training text. The `corpus_params` validator of `TrainingConfig` allows the key only on `free_form` corpora (`CORPUS_CATEGORY`) and requires the file to exist; a file id no record of that corpus / split carries is a warning with the count. *Provenance*: the per-corpus block of the mixture manifest gains `teacher_answers: {path, sha256, matched, dropped_no_answer}` next to `revision` (`sft_mixture.teacher_overlays_for`, read from a per-(corpus, split) record the helper leaves and `load_corpus` clears before every load), in `<cell>/data/sft_mixture_manifest.json`, in `all_metrics.json["training"]["sft"]["data"]["mixture"]`, and — because `_phase_eval_mixture_loaders` loads the dev pools with the same `corpus_params` — in `sft_eval_mixture_manifest.json`. **With the overlay the early-stop signal is the per-token NLL on the dev slices' teacher answers**, comparable across the three arms, and not the NLL on references. `scripts/plan_sft_mixture.py` needed no change: it passes `corpus_params` to `load_mixture_pools`, and its `available` column is the pool after the overlay (checked on a partial answers file before the full generation finished). Tests: `tests/test_finetune_corpora.py::TestTeacherOverlay` (6: replace-by-id and drop, prompt and other fields byte-identical, dev membership unchanged, unknown ids warn with the count, a load without the overlay clears its record, the validator refuses `tydiqa_arabic` and a missing file) and `tests/test_sft_mixture.py::TestTeacherOverlayManifest` (2: the training manifest block with the right counts, and the eval-mixture manifest block built through `_phase_eval_mixture_loaders`).

**P3.1–P3.2 — the three configs and their plans** (commit `02d0dd8`). `configs/experiments/qwen_{araroopat_3phase,bpe16k_3phase,native_sft}_v5_distill.yaml` are their v4 twins with only the header, `experiment.name / description / output_dir / created_at` (v4's `runs` provenance dropped) and `training.corpus_params.{cidar, bactrian_x_ar, aya_ar}.teacher_answers` changed. All three validate; the resolved `training.phases.sft`, `training.corpus_params` and `sweep.tasks` are identical (a direct comparison of the resolved dicts, and the console's `diff_configs`: 40 differing paths AraRooPat vs BPE and 47 AraRooPat vs native, **0 of them under those three blocks** — the rest are the tokenizer, the model and the disabled Phase 1/2 settings). `scripts/plan_sft_mixture.py --config <yaml> --out <json>` needed no change (it reads the overlaid pools; 427 s for AraRooPat over the CAMeL bridge, 77 s BPE, 97 s native):

| corpus | available | planned | AraRooPat: drawn / drop trunc. / drop cut / kept / repeated / loss tokens | BPE-16K | native |
|---|---|---|---|---|---|
| tydiqa_arabic | 13 610 | 10 700 | 10 920 / 205 / 15 / 10 700 / 0 / 219 897 | 10 749 / 49 / 0 / 10 700 / 0 / 105 851 | 10 787 / 85 / 2 / 10 700 / 0 / 152 181 |
| arcd | 550 | 550 | 560 / 10 / 0 / 550 / 9 / 10 376 | 550 / 0 / 0 / 550 / 0 / 6 114 | 550 / 0 / 0 / 550 / 0 / 7 708 |
| arabic_squad_mcq | 45 936 | 6 750 | 6 789 / 38 / 1 / 6 750 / 0 / 27 000 | 6 750 / 0 / 0 / 6 750 / 0 / 13 500 | 6 752 / 2 / 0 / 6 750 / 0 / 13 500 |
| cidar | 7 394 | 7 394 | 7 659 / 0 / 265 / 7 394 / **257** / 2 706 089 | 7 404 / 0 / 10 / 7 394 / 10 / 1 898 673 | 7 395 / 0 / 1 / 7 394 / 1 / 2 296 977 |
| bactrian_x_ar | 12 279 | 9 803 | 10 262 / 60 / 399 / 9 803 / 0 / 3 128 126 | 9 823 / 13 / 7 / 9 803 / 0 / 2 078 179 | 9 855 / 25 / 27 / 9 803 / 0 / 2 598 786 |
| aya_ar | 12 029 | 9 803 | 10 689 / 346 / 540 / 9 803 / 0 / 2 710 480 | 9 933 / 54 / 76 / 9 803 / 0 / 1 774 156 | 10 061 / 122 / 136 / 9 803 / 0 / 2 205 334 |

*Available* = the pool after the overlay and the exclusions (cidar 7 394 of 9 467 train records have a kept teacher answer, bactrian_x_ar 12 279 of 63 642, aya_ar 12 029 of 18 788). cidar's pool is smaller than its 9 000 share, so the water-fill plans 7 394 there and spills 1 606 to the other two (9 803 each). **The rule-1 asymmetry, measured:** free-form draws dropped at 1 024 tokens — AraRooPat 406 prompt-truncation + 1 204 cut-answer of 28 610 draws (5.6 %), BPE-16K 67 + 93 of 27 160 (0.6 %), native 147 + 164 of 27 311 (1.1 %). With `upsample: true` a corpus that runs dry repeats a further permutation rather than spilling: cidar's 265 cut answers under AraRooPat become **257 repeated cidar records** in its training set (BPE 10, native 1). Loss tokens: AraRooPat 8.80 M (free-form 97.1 %), native 7.27 M (97.6 %), BPE 5.88 M (97.9 %) — against v4's 4.50 M: the teacher's answers are 2–3 times longer than the references. Per free-form answer: AraRooPat 366 / 319 / 277 tokens (cidar / bactrian / aya), native 311 / 265 / 225, BPE 257 / 212 / 181. **Overlap of the drawn free-form id sets** (Jaccard): AraRooPat–BPE 0.910 (cidar 0.964, bactrian 0.917, aya 0.865), AraRooPat–native 0.922, BPE–native 0.986 — the arms train on the same records up to the ones AraRooPat's truncation replaces. The eval mixture (dev, 1 000 records at 250 / 150 / 600) plans the same allocation in all three arms (cidar 200 from 399 dev records with a teacher answer, bactrian 200, aya 200).

**P3.3 — memory smoke, native arm** (`total_examples: 400`, 200 micro-steps, no stop, no eval, no checkpoint; scratchpad config, output `outputs/experiments/_smoke/native_v5_smoke/`): peak **54 419 MiB** of 81 559 (`nvidia-smi`, 0.5 s sampling), 0.17 s per micro-step, `Vocab size unchanged (151936), skipping resize` — batch 2 × accumulation 8 fits the 151 936-row embedding at 1 024 tokens; all three arms run it.

**P3.4–P3.5 — the three runs** (`scripts/run_experiment.py --config configs/experiments/<yaml>`, one after the other, AraRooPat → BPE-16K → native, each followed by `diag_heldout_loss.py` and `diag_uncertainty.py`, then one judge pass over the main folder). Every log shows `Vocab size unchanged (17184 / 16000 / 151936), skipping resize` and no embedding re-initialization; the overlay lines (`cidar/train: teacher answers … 7394 of 9467 records matched`, the same for the dev pools) and the eval line every 500 micro-steps are in `run_<arm>.log`.

| arm | run wall | Phase 3 (train + evals) | stop → restored | examples restored | evals | wall / 1 000 examples seen | free-form eval | diagnostics |
|---|---|---|---|---|---|---|---|---|
| AraRooPat | 85.4 min (22:34–23:59) | 64.8 min | 19 500 → **17 000** | 34 000 of 45 000 | 37 | 99.7 s | 8.9 min (520 s timed) | 44 + 31 s |
| BPE-16K | 68.1 min (00:00–01:09) | 60.3 min | 20 000 → **17 500** | 35 000 | 38 | 90.5 s | 5.1 min (296 s) | 12 + 12 s |
| native | 69.0 min (01:09–02:18) | 58.8 min | 18 500 → **16 000** | 32 000 | 35 | 95.4 s | 6.1 min (353 s) | 17 + 17 s |

Judge (`gemma4_31b`, `--baseline native_qwen3_base_m2_c2400`): 3 × 250 verdicts in 37 / 36 / 37 s, 354 s with the cold start. *Stop curves* (per-token NLL on the dev slices' teacher answers for free-form, references for extractive / MCQ; the eval set is 250 / 150 / 600 records per arm, 572–595 of the 600 free-form ids shared):

| micro-step | AraRooPat: total / ext / mcq / ff | BPE-16K | native |
|---|---|---|---|
| 1 500 | 0.9095 / 0.1363 / 0.1472 / 0.9357 | 1.4522 / 0.4273 / 0.3088 / 1.4769 | 1.0599 / 0.2112 / 0.2209 / 1.0825 |
| 5 000 | 0.8605 / 0.1345 / 0.1046 / 0.8852 | 1.3820 / 0.3877 / 0.1668 / 1.4062 | 1.0208 / 0.2071 / 0.1565 / 1.0426 |
| 10 000 | 0.8303 / 0.1223 / 0.0939 / 0.8544 | 1.3450 / 0.3531 / 0.1626 / 1.3691 | 0.9881 / 0.1704 / 0.1605 / 1.0099 |
| 15 000 | 0.8204 / 0.1191 / 0.0815 / 0.8443 | 1.3326 / 0.3329 / 0.1634 / 1.3568 | 0.9780 / 0.1629 / 0.1612 / 0.9997 |
| **best** | **0.8195 / 0.1189 / 0.0795 / 0.8434 @ 17 000** | **1.3310 / 0.3325 / 0.1624 / 1.3552 @ 17 500** | **0.9774 / 0.1643 / 0.1578 / 0.9991 @ 16 000** |
| last | 0.8192 @ 19 500 | 1.3310 @ 20 000 | 0.9770 @ 18 500 |

The per-token numbers are in each arm's own tokens and do not compare across arms. **Per character** (Σ NLL over the free-form dev teacher answers at the restored step ÷ their raw characters): native **0.388**, AraRooPat **0.420**, BPE-16K **0.425** nats/char (181 563 / 123 158 / 152 419 answer tokens for 364 460 / 392 723 / 392 371 characters — AraRooPat's 600 dev answers are shorter because truncation at 1 024 tokens drops more long ones). That is the like-for-like likelihood of the teacher's register: native learns it best, AraRooPat and BPE are level.

**Consolidated free-form table** (250 held-out prompts, rule set of the reference cells: greedy, exact markers, loop stop, 2 400 characters; judge `gemma4_31b`; raw-text loss and answer NLL from `diag_heldout_rawtext_v1*.json` — the answer NLL is the likelihood of the CIDAR *references*, which the distilled arms no longer imitate):

| cell | judge ± SE | hist 1/2/3/4/5 | off-topic / repetition / truncated | chrF / BERTScore-F1 | loop / EOS / cap | mean chars | raw-text nats/char | answer NLL nats/char | best eval-mixture free-form (per token; nats/char) |
|---|---|---|---|---|---|---|---|---|---|
| **AraRooPat v5** | **2.43 ± 0.09** | 92/64/35/12/47 | 12.0 / 8.4 / 3.6 % | 22.84 / 0.85 | **6.8** / 92.8 / 0.4 % | 541 | 0.880 | 0.812 | 0.8434; 0.420 |
| **BPE-16K v5** | **2.13 ± 0.08** | 107/69/32/18/24 | 14.0 / 16.0 / 10.0 % | 22.50 / 0.85 | **15.6** / 84.4 / 0.0 % | 533 | 0.864 | 0.907 | 1.3552; 0.425 |
| **native v5** | **2.52 ± 0.09** | 83/66/38/14/49 | 8.8 / 8.0 / 3.6 % | 23.93 / 0.85 | **8.4** / 90.8 / 0.4 % | 543 | 0.840 | 0.811 | 0.9991; 0.388 |
| teacher ceiling (Aya-Expanse-32B) | 3.65 ± 0.09 | 18/59/36/16/121 | 2.0 / 0.0 / 0.0 % | 26.61 / 0.86 | 0.0 / 100.0 / 0.0 % | 608 | — | — | — |
| native base | 2.42 ± 0.09 | 92/69/29/13/47 | 12.8 / 9.6 / 6.8 % | 22.94 / 0.85 | 6.8 / 90.8 / 2.4 % | 511 | 0.826 | 0.807 | — |
| native SFT | 2.28 ± 0.09 | 105/60/31/18/36 | 14.0 / 15.6 / 4.4 % | 18.59 / 0.86 | 12.4 / 87.2 / 0.4 % | 207 | 0.838 | 0.766 | — |
| AraRooPat v4 | 2.18 ± 0.09 | 108/66/31/12/33 | 16.8 / 11.6 / 8.4 % | 18.61 / 0.86 | 23.2 / 76.4 / 0.4 % | 198 | 0.875 | 0.766 | 1.1623 (references) |
| BPE ffstop | 2.00 ± 0.08 | 117/71/30/10/22 | 13.2 / 17.6 / 15.2 % | 16.60 / 0.85 | 19.6 / 75.6 / 4.8 % | 211 | 0.859 | 0.845 | 2.1177 (references) |
| AraRooPat ffstop | 2.15 ± 0.09 | 112/67/26/11/34 | 18.0 / 16.8 / 7.2 % | 17.70 / 0.86 | 21.2 / 77.6 / 1.2 % | 190 | 0.875 | 0.788 | 1.2297 (references) |

P(EOS) at the reference end / EOS rank-1 share: AraRooPat v5 0.49 / 0.63, BPE v5 0.53 / 0.71, native v5 0.60 / 0.78 (v4 0.62 / 0.83, native SFT 0.64 / 0.91) — the distilled arms expect longer answers than the references give. As expected before the runs, the answer NLL on the CIDAR references *rose* for every distilled arm (AraRooPat 0.766 → 0.812 vs v4, BPE 0.845 → 0.907 vs ffstop, native 0.766 → 0.811 vs SFT): they are trained on a different register. The raw-text loss moved by at most 0.005 nats/char against the previous cell of each arm (0.880 / 0.864 / 0.840).

**Paired bootstraps** (10 000 resamples, seed 0, same 250 prompts; sub-scores correctness / fluency / instruction following):

| pair | Δ | 95 % CI | win / tie / loss | sub-score Δ (C / F / IF) |
|---|---|---|---|---|
| AraRooPat v5 − teacher ceiling | −1.22 | [−1.41, −1.03] | 7 / 34 / 59 % | −1.16 / −0.59 / −1.11 |
| BPE v5 − teacher ceiling | −1.52 | [−1.70, −1.34] | 3 / 27 / 70 % | −1.41 / −0.78 / −1.48 |
| native v5 − teacher ceiling | −1.13 | [−1.32, −0.95] | 8 / 34 / 57 % | −1.13 / −0.28 / −0.84 |
| AraRooPat v5 − native base | +0.02 | [−0.15, +0.18] | 26 / 50 / 25 % | +0.02 / −0.20 / +0.04 |
| BPE v5 − native base | −0.28 | [−0.45, −0.12] | 18 / 51 / 32 % | −0.23 / −0.39 / −0.32 |
| native v5 − native base | +0.10 | [−0.04, +0.25] | 29 / 50 / 20 % | +0.06 / +0.11 / +0.31 |
| AraRooPat v5 − AraRooPat v4 | **+0.25** | **[+0.10, +0.40]** | 32 / 50 / 18 % | +0.19 / +0.09 / +0.38 |
| AraRooPat v5 − AraRooPat ffstop | **+0.28** | **[+0.13, +0.43]** | 32 / 52 / 16 % | +0.21 / +0.12 / +0.56 |
| BPE v5 − BPE ffstop | +0.14 | [−0.01, +0.28] | 28 / 52 / 20 % | +0.12 / +0.02 / +0.24 |
| native v5 − native SFT | **+0.24** | **[+0.09, +0.39]** | 32 / 50 / 18 % | +0.10 / +0.12 / +0.43 |
| AraRooPat v5 − BPE v5 | **+0.30** | **[+0.15, +0.45]** | 33 / 51 / 16 % | +0.25 / +0.19 / +0.36 |
| AraRooPat v5 − native v5 | −0.09 | [−0.24, +0.07] | 22 / 48 / 30 % | −0.04 / −0.31 / −0.27 |
| BPE v5 − native v5 | **−0.39** | **[−0.55, −0.23]** | 16 / 46 / 38 % | −0.28 / −0.50 / −0.63 |

*Uncertainty diagnostic on the distilled cells* (same script as P0; cross-checks exact):

| cell | refs: H/char | H/token | p1 | margin | p1 < 0.5 | NLL/char | gens: H/char | p1 | margin | round-trip exact | before onset H / p1 / margin | control | in-loop p1 | looped rows (with control) | margin lower before onset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AraRooPat v5 | 0.683 | 1.349 | 0.695 | 0.602 | 30.4 % | 0.812 | 0.385 | 0.801 | 0.725 | 231/250 | 1.203 / 0.724 / 0.637 | 0.947 / 0.774 / 0.691 | 0.790 | 14 of 17 (14) | 79 % |
| BPE v5 | 0.749 | 1.884 | 0.622 | 0.529 | 38.9 % | 0.907 | 0.416 | 0.720 | 0.642 | 250/250 | 1.680 / 0.653 / 0.570 | 1.435 / 0.689 / 0.601 | 0.779 | 38 of 39 (38) | 61 % |
| native v5 | 0.685 | 1.561 | 0.641 | 0.542 | 37.5 % | 0.811 | 0.388 | 0.751 | 0.666 | 250/250 | 1.325 / 0.695 / 0.607 | 1.046 / 0.742 / 0.653 | 0.769 | 19 of 21 (19) | 63 % |

At equal reference NLL (0.812 / 0.811) AraRooPat v5 and native v5 have the same reference entropy per character (0.683 / 0.685) and the same generation entropy per character (0.385 / 0.388); the pre-onset margin is again below the control in every arm. Distillation lowered the reference entropy of every arm (AraRooPat 0.744 → 0.683, native 0.764 → 0.685, BPE 0.846 → 0.749).

**Reading (rule 4).** *Retained share* of the teacher's judge score (3.652 on the test prompts): **native 0.69** (2.520), **AraRooPat 0.67** (2.432), **BPE-16K 0.58** (2.132). Against the native distilled arm: AraRooPat −0.09 [−0.24, +0.07] (not separable; 22 prompts where AraRooPat scores ≥ 2 higher, 23 where it scores ≥ 2 lower, 205 within 1), BPE −0.39 [−0.55, −0.23]. Loops against native's 8.4 %: AraRooPat 6.8 % (0.81 ×), BPE 15.6 % (1.86 ×). Distillation moved every arm up by about a quarter point (AraRooPat +0.25 vs v4 and +0.28 vs ffstop, native +0.24 vs SFT, BPE +0.14 vs ffstop — the one interval that touches zero), and it removed the length dependence of AraRooPat's loops (7.1 / 6.0 / 7.2 % by reference tercile against v4's 14.3 / 22.9 / 32.5 %). With the same teacher, data, seed and draw rule, **AraRooPat keeps as much of the teacher as the native vocabulary does within the judge's resolution and more than BPE-16K** (+0.30 [+0.15, +0.45]) — the first AraRooPat-over-BPE interval above zero under identical treatment; the residual cost of AraRooPat is in fluency (−0.31 [−0.44, −0.19] vs native) and instruction following (−0.27), not correctness (−0.04). None of the three is close to the teacher (Δ −1.13 to −1.52). *The P3.6 case:* the DITTO trigger — AraRooPat's distilled loops above 1.5 × native's — is **not met** (0.81 ×); the closing condition — all three arms' loop rates within 5 points — is **not met either**, and only because of BPE-16K (15.6 % against 6.8 / 8.4 %). So neither case as written: for the AraRooPat-vs-native question the vocabulary comparison is itself the finding (1.6 points of loops, a judge gap inside the CI), and the loop gap that remains is BPE's, which a repetition objective queued *for AraRooPat* would not address. §5 item 8 stays open, re-scoped to that residual.

**P4 — loops, lengths, flags.** *Loops by period bucket* (1 / 2–3 / 4–10 / > 10 words; all by the periodic rule): AraRooPat v5 3 / 6 / 5 / 3 (17), BPE v5 9 / 9 / 6 / 15 (39), native v5 4 / 4 / 5 / 8 (21), teacher 0, native base 0 / 4 / 6 / 7 (17), native SFT 4 / 8 / 13 / 6 (31), v4 11 / 15 / 16 / 16 (58). Loop rows among the judge-score-1 rows: AraRooPat v5 10 of 92, BPE v5 23 of 107, native v5 12 of 83, teacher 0 of 18, native base 7 of 92, native SFT 13 of 105, v4 29 of 108.

| loop rate by reference tercile (84 / 83 / 83 prompts) | short | medium | long |
|---|---|---|---|
| AraRooPat v5 | 7.1 % | 6.0 % | 7.2 % |
| BPE-16K v5 | 9.5 % | 15.7 % | 21.7 % |
| native v5 | 7.1 % | 9.6 % | 8.4 % |
| teacher ceiling | 0.0 % | 0.0 % | 0.0 % |
| native base | 4.8 % | 4.8 % | 10.8 % |
| native SFT | 7.1 % | 13.3 % | 16.9 % |
| AraRooPat v4 | 14.3 % | 22.9 % | 32.5 % |

| answer length by tercile (mean / median chars) | short | medium | long | ≤ 40 chars | judge truncated / off-topic | judge by tercile |
|---|---|---|---|---|---|---|
| AraRooPat v5 | 336 / 190 | 534 / 345 | 757 / 619 | 4.8 % | 3.6 / 12.0 % | 2.51 / 2.57 / 2.22 |
| BPE-16K v5 | 347 / 240 | 543 / 397 | 710 / 588 | 3.6 % | 10.0 / 14.0 % | 2.40 / 1.96 / 2.02 |
| native v5 | 321 / 176 | 516 / 329 | 796 / 741 | 4.0 % | 3.6 / 8.8 % | 2.57 / 2.48 / 2.51 |
| teacher ceiling | 398 / 242 | 618 / 451 | 812 / 751 | 2.4 % | 0.0 / 2.0 % | 3.81 / 3.39 / 3.76 |
| references | 79 / 80 | 217 / 219 | 580 / 485 | — | — | — |

The three students write at the teacher's length (the references are 2–4 × shorter in the short and medium terciles); the ≤ 40-character share fell from ~20 % (v4, ffstop, native SFT) to 4 %.

*Examples.* AraRooPat v5 against v4 (33 prompts gained ≥ 2, 14 lost ≥ 2):

| id | instruction | v4 (judge) | v5 (judge) |
|---|---|---|---|
| cidar-3496 | أنشئ جملة تستخدم الأسماء "فهد" و "ظل". | ظل فهد كان متوقعًا ولكنه كان متوقعًا بشكل متوقع . (2) | في ظلام الليل ، ظل فهد الطويل كان متجاهلًا ، كشبح في الأرض . (4) |
| cidar-9340 | المضي في الطريق المهني التقليدي نفسه الذي يمضي فيه من حولك … | the instruction repeated verbatim (1) | a structured discussion of the choice: «يمكن أن يكون خيارا جذابا … وهنا بعض الأسباب» (3) |
| cidar-9363 | ما اعراب شبه الجملة في جملة (في الاتحاد قوة)؟ | «الاتحاد : اسم مجرور بالفتحة» — wrong case sign (1) | «في : حرف جر . الاتحاد : اسم مجرور وعلامة جره الكسرة الظاهرة» (3) |
| cidar-312 | … قيمها على مقياس خمس نجوم، الداء والدواء، منهاج السنة، العقد الفريد | three terse ratings (5) | ratings with invented descriptions (منهاج السنة «كتاب أساسي في علم الفلك») (2) |
| cidar-9111 | اصنع جملة باستخدام الفعل المعطى في الماضي "يهز" | هز الطفل الألعاب الصغيرة في اللعبة . (5) | يهز الأطفال اللعب بالكرة في الملعب . — present, not past (2) |
| cidar-8664 | ماهو وقت شروق الشمس وغروبها في مدينة أبها 23 سبتمبر. | plausible times (3) | «في مدينة أبها ( إسطنبول )» — a wrong city in the answer (1) |

AraRooPat v5 against native v5 (205 prompts within 1 point, 23 where AraRooPat loses ≥ 2, 22 where it gains ≥ 2):

| id | instruction | native v5 (judge) | AraRooPat v5 (judge) |
|---|---|---|---|
| cidar-10142 | حدد الكلمة الصحيحة … الذكرى الخمسون أو الخمسين للنكبة؟ | «الخمسين» + the corrected sentence (4) | the same answer, the same sentence (4) |
| cidar-821 | اذكر قائمة للأحزاب السياسية الهامة في الأردن. | a list with invented party details (2) | a list that loop-stops in a repeated description (1) |
| cidar-9552 | (a long passage) المنصوبات في هذا النص هو (ان) فقط؟ | «لا … مثل (الذي)» — loop-stopped (2) | «لا» + a list of verbs that are not the accusatives asked for (1) |
| cidar-9585 | (a learner's question in Iraqi-accented Arabic) how can I speak like an Arab? | a numbered study plan (5) | «يمكنك أن تكون عربيا باللغة العربية …» — does not answer (2) |
| cidar-9769 | حدد الفاعل للفعل "نَفَى" في الجملة … | «نَفْسٌ أَبِيَّةٌ» (5) | «النوم» — the object, not the subject (1) |
| cidar-7159 | صف المطار الأكبر في الشرق الأوسط. | a fluent description of Dubai International (4) | the same airport, loop-stopped in a repeated clause (2) |

*GPU time.* P0 1.7 min (five cells); bake-off generation 28.9 min (six candidates, main + determinism runs, including Falcon-H1's failed bf16 attempt and a killed fp8 attempt, ~5.5 min); the reference row 9.5 min; bake-off judge 10.5 min; ceiling generation + cell 1.8 min; full generation 48.4 min + extension 9.9 min; ceiling judge 4.7 min; smoke 1.2 min; the three runs 85.4 + 68.1 + 69.0 min; diagnostics 2.2 min; main judge 5.9 min — **≈ 5 h 47 min**. CPU only: the three planners 10 min, the filter twice ≈ 1 min, pseudo-cells ≈ 2 min each on CPU BERTScore, tests ≈ 4 min. Lost wall time (no GPU): ~10 min to a self-deadlocking wait loop (`pgrep -f <script>` inside a heredoc-launched chain matches the launching shell's own command line).

**Artifacts.** `outputs/experiments/teacher_bakeoff/{jais2_8b,llama33_70b_awq,falcon_h1_34b,cmdr7b_arabic,aya_expanse_32b,allam_7b,native_qwen3_base_dev}/` + `bakeoff_table.json` + `comparison_report.txt`; `outputs/data_cache/distill/bakeoff/` (`prompts_dev.jsonl`, `bakeoff_manifest.json` with the caps and the held-out check, per candidate `teacher_raw[_repeat].jsonl`, `generation_meta[_repeat].json`, `run*.log`, Falcon's `run_attempt0_bf16_oom.log`); `outputs/data_cache/distill/aya_expanse_32b_v1/` (`prompts.jsonl`, `prompts_manifest.json`, `teacher_raw.jsonl`, `generation_meta.json` + `_part1` / `_part2`, `teacher_answers_v1.jsonl`, `manifest.json` + `_part1`); `outputs/experiments/qwen_native_vs_araroopat/{teacher_aya_expanse_32b_m2_c2400, araroopat_3phase_v5_distill, bpe_16k_3phase_v5_distill, native_qwen3_sft_v5_distill}/` (`all_metrics.json`, `data/sft_mixture_manifest.json`, `training/data/sft_eval_mixture_manifest.json`, `eval_rows/freeform_cidar.parquet`, `freeform_judge/gemma4_31b.parquet`, `diag_heldout_rawtext_v1.json`, `diag_uncertainty_v1.json`, `training/sft/`); the five P0 `diag_uncertainty_v1[_base].json`; `outputs/experiments/_smoke/native_v5_smoke/`. Configs: `configs/distill/teacher_candidates.yaml`, `configs/experiments/teacher_bakeoff_native_base_dev.yaml`, `configs/experiments/qwen_{araroopat_3phase,bpe16k_3phase,native_sft}_v5_distill.yaml`. Code: `scripts/diag_uncertainty.py`, `src/arabic_eval/distill/`, `scripts/distill/`, the overlay in `finetune_corpora.py` / `sft_mixture.py` / `config.py` / `pipeline/experiment.py`. Commits: `c59c1e7` (P0), `17e790a` (P1), `345bf51` (P2), `02d0dd8` (P3 configs), and the documentation commit carrying this section.

**Verification (2026-09-24, Fable session, independent re-read of every number above against the outputs).** *Reproduced exactly:* the nine cells' judge means and SEs, loop / EOS / cap rates, chrF, mean characters and the off-topic / truncated flag rates; all thirteen paired bootstraps to the third decimal (`paired_compare.py`, seed 0, 10 000 resamples); the teacher file (`8e4d9cd13e3d1bc8a2cd0782266f570dc771a55f500e0609a1c808955bbcf272`, 35 066 records, cap 1 025, snapshot `b306ea27`, loaded from `CohereForAI/aya-expanse-32b`); both held-out checks (0 id hits and 0 normalized-text hits over the 250 bake-off prompts and the 50 100 generation prompts); the bake-off table as `bakeoff_table.py` rebuilds it — means, kept rates, determinism shares, the decision JSON (eligible at ≥ 80 % kept: Llama-3.3-70B-AWQ, Command-R7B-Arabic, Aya-Expanse-32B, ALLaM-7B; winner Aya, runner-up ALLaM, throughput rule not applied, decision `proceed`), Aya − ALLaM +0.14 [−0.00, +0.27], and Aya ahead in every stratum (4.70 / 4.62 / 4.58 against ALLaM's 4.60 / 4.45 / 4.43); the teacher manifest per corpus × split (kept 399 / 7 394 / 2 270 / 12 279 / 695 / 12 029, Latin drops 73 / 1 564 / 982 / 5 126 / 282 / 5 404 = 13 431 of the 15 007 drops, 27 prompts skipped for length, preamble shares, teacher character means); the three training manifests (pools 7 394 / 12 279 / 12 029 after the overlay, planned 7 394 / 9 803 / 9 803, free-form drops 406 + 1 204 of 28 610 draws = 5.6 % for AraRooPat, 67 + 93 of 27 160 = 0.6 % for BPE, 147 + 164 of 27 311 = 1.1 % for native, cidar repeats 257 / 10 / 1, free-form loss-token share 97.1 / 97.9 / 97.6 %, Jaccard of the drawn free-form ids 0.910 / 0.922 / 0.986, the cidar overlay block `matched 7 394, dropped_no_answer 2 073`, `tydiqa_arabic` available 13 610); the resolved `training.phases.sft`, `training.corpus_params` and `sweep.tasks` blocks identical across the three configs (direct comparison of the resolved dicts); the stop curves (stopped at 19 500 / 20 000 / 18 500 micro-steps, `best_eval_step` 17 000 / 17 500 / 16 000 at 0.8195 / 1.3310 / 0.9774, walls 64.8 / 60.3 / 58.8 min); the raw-text loss 0.880 / 0.864 / 0.840 and the reference answer NLL 0.812 / 0.907 / 0.811 nats/char; the P0 block in full — the five cells' reference and generation numbers, cross-checks |Δ| ≤ 1.7e-5, round-trip-exact 250 / 250 / 250 / 241 / 237, the before-onset / control / in-loop triples and the looped-row counts 17, 28 of 31, 47 of 49, 45 of 53, 52 of 58; the P4 counts (period buckets, loop rows among score-1 rows, loop rate and mean length per tercile, ≤ 40-character shares 4.8 / 3.6 / 4.0 % against v4's 19.6 %, the 33 / 14 and 205 / 23 / 22 gain-loss counts); the per-character dev NLL 0.420 / 0.425 / 0.388 by the stated recipe (best free-form eval loss × tokens ÷ Σ len(teacher answer) over the free-form ids of `training/data/sft_eval_mixture_manifest.json`); the four new or touched test files (97 passed) and the six commits `c59c1e7` … `66a899a`. *Read with care, none of it an error:* (a) the restored checkpoint is `best_eval_step` under the `min_delta` rule, not the lowest recorded eval — each arm's minimum came later (AraRooPat 19 500 at 0.81921 against the restored 17 000 at 0.81951; BPE 19 000 at 1.33087 against 1.33102; native 18 000 at 0.97697 against 0.97744), all within 0.0005 nats/token, below `min_delta`, so the 34 000 / 35 000 / 32 000 restored examples stand; (b) P0's prediction (ii) holds on the means in every cell, and per looped row the margin before onset is below the control in 82 / 68 / 68 / 60 / 81 % of rows (base / SFT / BPE / AraRooPat ffstop / v4) — a majority, not every row; (c) the per-character dev NLL is indicative rather than like-for-like: the dev draws differ per arm by the same truncation asymmetry as the training draws (Jaccard 0.911 / 0.926 / 0.983), and AraRooPat's 600 dev records hold 364 460 teacher characters against BPE's 392 723 and native's 392 371, a 7 % shorter draw; (d) `sft_eval_mixture_manifest.json` is written under `<cell>/training/data/`, while the *Phase 3 mixture* section of CLAUDE.md places it under `<cell>/data/` next to the training manifest — the documentation path needs fixing, not the file; (e) the 54 419 MiB smoke peak and the GPU minutes come from `nvidia-smi` sampling and logs that are not stored as artifacts and are taken from the report; (f) the "any Latin letter" rule decided eligibility (Jais-2 28.8 % and Falcon-H1 26.0 % of dev answers carried one) and removed 27 % of the winner's answers — mostly names and technical terms — so a later teacher file could use a ratio rule, at the price of a new file hash; (g) vLLM greedy reproduced itself on 39–89 % of prompts across processes, so the teacher file is one realization of the teacher, not the only one. *Reading.* The three findings stand as written: the flatness explanation of the loop gap is refuted (0.744 against 0.764 nats/char of entropy at equal answer NLL, and loops start at low-margin points in the native base too); under one teacher, one draw rule and one Phase 3 block, AraRooPat keeps 0.67 of the teacher's judge score against native's 0.69 (Δ −0.09 [−0.24, +0.07]) and BPE-16K's 0.58 (Δ +0.30 [+0.15, +0.45]); and the loop rates are 6.8 / 8.4 / 15.6 %. Two cautions for the next reader. The judge is reference-guided and the rubric says not to reward length, but every distilled arm now writes at the teacher's length (533–543 characters against 190–211 before), so the +0.24 to +0.28 gains over the pre-distillation cells and the AraRooPat − BPE interval are the first numbers of this campaign the blind human check (the console's Rate tab) should confirm before they are quoted as findings; the reference-based chrF moved the same way (22.5–23.9 against 16.6–18.6), which is consistent with the judge but is a weak metric on open-ended instructions. And the BPE residual — 15.6 % loops, 21.7 % on the long tercile, the lowest per-token dev NLL gap of the three closed the least — is now the one open generation question; it is a BPE finding, not an AraRooPat one, and §5 item 8 is right to be re-scoped to it.

**Design provenance (2026-09-23, recorded 09-24).** *How the plan got here.* After §3.7 the proposed order was uncertainty diagnostic → sequence-level distillation → a DITTO-style objective, and the first brief for the distillation made the untrained native backbone the teacher, with the BPE and native comparators gated on the AraRooPat arm's result. The user rejected the gate (a tutored AraRooPat against an untutored BPE cannot be attributed to the vocabulary) and set rules 1–4; then rejected the native backbone as teacher — the native arm would have been self-distilled, imitating its own greedy output under its own vocabulary, which is a different exam from re-expressing a foreign model's text through adapted rows — and asked for a neutral Arabic-capable instructor "so that native, BPE and AraRooPat sit the same exam"; then asked whether the bake-off's losers' answers would be used too, which fixed rule 5 (one teacher, whole: a blend re-introduces the mixed-register noise of the three translated corpora, and per-prompt selection by the judge trains the students on the judge's preferences). Distillation across two vocabularies can only be sequence-level — teacher text as the target, Kim & Rush (2016) — because logits over different vocabularies cannot be matched; the teacher's *mode* (greedy) is what the students imitate, which is also why the run tests the loop question at all. *Candidate search (web, 2026-09-23).* The generative Arabic leaderboard AraGen v3 (3C3H, LLM-judged; snapshot of 2026-05-06, 68 models) ranks the open-weight models: Gemma-4-31B-it 72.7 and Gemma-4-26B-A4B 67.4 — **excluded, the judge's family** (the judge is `google/gemma-4-31b-it`; judges favour their own style), which also excludes Fanar-2-27B (a continued-pretrained Gemma-3-27B; Fanar-1-9B is Gemma-2-based, 16.2); Jais-2-70B-Chat 52.4 — **no quantized release, 140 GB bf16 does not fit one H100**; Llama-3.3-70B-Instruct 52.1 — taken as the community AWQ int4 (`casperhansen/llama-3.3-70b-instruct-awq`, ≈ 40 GB); Mistral Large 52.0 — too large; Qwen2.5-72B 50.8 / Qwen3-32B 35.2 — **excluded, they share the native arm's vocabulary** (their text would be "easy" for it in exactly the way rule 1 removes); Falcon-H1-34B-Instruct 45.7; Jais-2-8B-Chat 34.0 (Arabic-first, own 150 K vocabulary, Apache 2.0; vLLM 0.29 has `jais2.py` natively although the model card still asks for a fork). Not on the leaderboard but added: Command-R7B-Arabic (Cohere's MSA-optimised 7B, above Jais-2-8B on Arabic IFEval per the Jais-2 card), Aya-Expanse-32B (Cohere, 23 languages incl. Arabic, CC-BY-NC), and ALLaM-7B-Instruct-preview at the user's suggestion (AraGen-12-24 53.2 per the Jais-2 card, 4 k context, Apache 2.0). Not usable: Falcon-H1-Arabic (3B / 7B / 34B announced 2026-01-05 with OALL 75 % for the 34B) **is not published on the Hub under `tiiuae`** — 137 repos listed on 2026-09-23, none with "Arabic"; the general Falcon-H1-34B-Instruct stood in; AraBERT (an encoder, cannot generate) and AraT5 (a 2021 seq2seq model without instruction following), both named by the user, were set aside for those reasons. Every listed repo's access was verified with the user's token before the brief; the token lives in `~/.cache/huggingface/token` (mode 600) and appears in no config, log, manifest or commit. *Bake-off design and its assumptions.* Selection on 250 cidar-**dev** prompts (training material, never the test), one fixed Arabic system prompt for every candidate (direct MSA answer, no preamble, no markdown unless asked, no other language except names / terms), each model's own chat template with thinking modes off, greedy, cap from `derive_token_cap` with the candidate's own chars-per-token — so the candidates are compared on the register we want the students to learn, not on their default verbosity; the reference row is the untouched backbone on the same 250 dev prompts through the pipeline's own path. Rule as written before any number: highest judge mean among candidates keeping ≥ 80 % of their answers under the training filters, ties within one SE by kept rate → length closer to the references → smaller model; a throughput rule (runner-up within one SE if the winner projects above 6 h of generation) and a stop point (no candidate beating the backbone's reference row by one SE → stop and ask) — neither fired. Assumptions the design carries, stated: a dev-prompt ranking transfers to the test (the dev prompts turned out easier — 3.44 vs 2.42 for the same base model — so the bake-off ranks, it does not predict test scores); the "any Latin letter" filter of rule 3 is the eval's own row flag and was applied unchanged (it decided Jais-2's and Falcon-H1's eligibility and removed 27 % of the winner's answers, mostly names and terms); the teacher's factual errors pass the filters, so every arm inherits them equally and the absolute scores are bounded by this teacher; the students' length follows the teacher's (≈ 540 characters), so the judge's known length preference is the caveat on every distilled number until the human check (§5 item 15).

### 3.9 Tokenization audit on the eval texts, and a defect in the MCQ scorer's continuation window (2026-09-24)

**Why.** Before running the four MCQ benchmarks on the v5 checkpoints (§5 item 16) the user asked whether we had ever checked how the tokenizers encode the actual questions and contexts — whether words reach meaningful tokens or fall to characters. We had not, on the texts that matter: the intrinsic metrics (§0, §3.5) are measured on the tokenizer-training corpus (ArabicText-Large eval split: native Qwen3 fertility 2.30 / compression 2.55, AraRooPat v3 3.22 / 1.82 with `unk_rate` 0.007 and `vocab_coverage` 0.945, BPE-16K 1.51 / 3.90), the admission shares of CLAUDE.md (ROOT+PAT 64.6 % / LIT 16.4 % on 300 Arabic-Exam questions) were measured on the 2026-09-16 tokenizer generation, not on `araroopat_maxpat40k_v3`, the decode-side round trip (§3.5, 91.1 / 98.97) says nothing about the encode side, and native Qwen3's fragmentation of Arabic had never been measured at all. **Audit** (`outputs/experiments/qwen_native_vs_araroopat/_audit/tok_audit_2026-09-24.{py,json}`; CPU; the three campaign tokenizers loaded from their cells; texts = the 250 held-out CIDAR prompts and references, and 300 seeded MCQ prompts per benchmark taken verbatim from the `prompt` column of `native_qwen3_sft`'s eval-row dumps — the exact strings the scorer was handed, 3-shot demos included, 0-shot for ACVA; the native token counts agree with the dumps' `prompt_units` on every row). Arabic-letter word occurrences are classified by encoding each distinct word once (AraRooPat: the family of its first non-clitic token; native / BPE: the word with a leading space, pieces decoded one by one).

| text set (300 prompts / 250 texts) | tokenizer | tokens p50 / p90 | > 1 024 tok | pieces per Arabic word | AraRooPat: ROOT+PAT / closed-class / PROP / LIT / UNK words | native, BPE: whole word / ≥ 4 pieces / has a single-letter piece / has a byte fragment |
|---|---|---|---|---|---|---|
| CIDAR held-out prompts | native | 18 / 56 | 0 | 2.54 | — | 32.0 / 18.5 / 48.4 / 1.3 % |
| | AraRooPat | 23 / 61 | 0 | 2.97 | 68.4 / 21.0 / 0.8 / 6.5 / 0 % | — |
| | BPE-16K | 14 / 52 | 0 | 2.27 | — | 51.9 / 12.4 / 20.4 / 9.6 % |
| CIDAR held-out references | native | 90 / 277 | 0 | 2.61 | — | 31.6 / 20.4 / 45.9 / 1.0 % |
| | AraRooPat | 107 / 328 | 0 | 3.08 | 70.1 / 20.5 / 1.4 / 5.4 / 0 % | — |
| | BPE-16K | 77 / 259 | 0.8 % | 2.33 | — | 52.3 / 14.0 / 19.2 / 11.6 % |
| ACVA (0-shot) | native | 27 / 32 | 0 | 2.50 | — | 30.5 / 28.9 / 44.6 / 0.1 % |
| | AraRooPat | 35 / 45 | 0 | 3.36 | 81.3 / 13.7 / 1.0 / 3.8 / 0 % | — |
| | BPE-16K | 18 / 25 | 0 | 1.80 | — | 54.9 / 5.0 / 6.2 / 4.4 % |
| Alghafa (3-shot) | native | 373 / 1 119 | **18.0 %** | 2.39 | — | 31.1 / 17.7 / 46.2 / 0.2 % |
| | AraRooPat | 413 / 1 546 | **26.3 %** | 3.20 | 70.1 / 18.6 / 1.5 / 9.0 / 1.1 % | — |
| | BPE-16K | 257 / 798 | 1.0 % | 1.71 | — | 54.7 / 4.4 / 9.9 / 2.2 % |
| Arabic-Exam (3-shot) | native | 323 / 443 | 4.7 % | 2.80 | — | 27.1 / 21.5 / 52.0 / 1.2 % |
| | AraRooPat | 417 / 569 | 4.7 % | 3.21 | 67.0 / 15.9 / 1.2 / 14.9 / 0.55 % | — |
| | BPE-16K | 223 / 340 | 4.7 % | 2.69 | — | 47.3 / 13.2 / 28.2 / 12.2 % |
| Culture-Arabic-MMLU (3-shot) | native | 508 / 784 | 4.0 % | 2.18 | — | 34.1 / 12.5 / 42.5 / 0.1 % |
| | AraRooPat | 627 / 1 000 | **8.3 %** | 3.00 | 67.8 / 18.5 / 0.5 / 12.6 / 0.01 % | — |
| | BPE-16K | 354 / 569 | 2.0 % | 1.58 | — | 58.7 / 3.0 / 15.7 / 2.0 % |

Closed-class = PREP + FUNC + clitic-only words. **Reading, AraRooPat.** Two thirds to four fifths of Arabic word occurrences are one `[ROOT_*] [PAT_*]` pair (plus clitic tokens), another 14–21 % are the closed-class single tokens, and the character path (`[LIT_BEGIN] [CHAR_*]… [LIT_END]`) takes 3.8–14.9 %. Most of the character path on the exam benchmarks is **the choice letters**: `أ.` `ب.` `ج.` `د.` and the bare letters of the few-shot answers are 11.1 % of Arabic-Exam's and 8.5 % of Culture-MMLU's word occurrences (`أ.` → `[LIT_BEGIN] [CHAR_أ] [LIT_END] [PUNCT_.]`, four tokens against native's one or two); the rest of the character path is 3.6 % (Arabic-Exam: `الأردنية`, `الجغرافيا`, `الكالسيوم`, `نيوكليوسومات`, `كروماتين`, typos like `المالٌة` / `اآللي`), 4.1 % (Culture-MMLU: `السيناريو`, `دولار`, `هذين`, `هاتين`, `بروتوكول`, `الفيدرالية`) and **8.3 % on Alghafa**, where it is dialect — the `meta_ar_dialects` sub-config (`شنو ديال نتاع بزاف كتير ايش رح زي مو هيدا كتكون`) — and the hamza-less sentiment label `ايجابي` (981 of the sample's 5 341 character-path occurrences; CAMeL has no rooted reading for that spelling, so **the continuation `هو رأي ايجابي` of the word-scored sentiment sub-configs is spelled in six `[CHAR_*]` tokens while `هو رأي سلبي` is `[ROOT_سلب] [PAT_1َ2ْ3ِيٌّ]`**, an asymmetry inside the benchmark that the char-normalised score does not remove). `<unk>` appears in 0.55 % of Arabic-Exam and 1.1 % of Alghafa word occurrences (9 of 69 583 on Culture-MMLU): the character inventory lacks the curly quotes `“ ”` (680 occurrences in the two samples, the review texts of Alghafa), the Persian letters `ی` U+06CC (160) and `ھ` U+06BE (44), the Quranic mark U+06E1 (42) and ornate parentheses `﴾ ﴿`, emoji, and Latin letters glued to Arabic words (a few dozen). One CAMeL clitic tag is missing from `CAMEL_CLITIC_SURFACE` (`ma_interrog`, hit by the word `مم`; a warning per occurrence, the word ends on another path). **Reading, native Qwen3.** Only 27–34 % of Arabic word occurrences are one piece; 42–52 % contain a single-letter piece and 12–29 % are cut into four or more pieces — the "falls into characters" concern applies to the native vocabulary more than to AraRooPat on this measure, though a single-letter piece inside a word is not the same as a whole word spelled out. **BPE-16K** keeps 47–59 % of words whole with 2–12 % of words holding a byte-level fragment (diacritics and rare characters split into UTF-8 bytes: 11.6 % on the CIDAR references, 12.2 % on Arabic-Exam). **Length.** AraRooPat spends 3.0–3.4 tokens per Arabic word against native's 2.2–2.8 and BPE's 1.6–2.7, so at the 1 024-token cap the native cell ran with, 26.3 % of Alghafa prompts and 8.3 % of Culture-MMLU prompts exceed the window under AraRooPat against 18.0 / 4.0 % under native and 1.0 / 2.0 % under BPE — and the native cell's own dumps already show what that does: on `native_qwen3_sft` 19.3 % of Alghafa rows, 6.6 % of Culture-MMLU and 4.3 % of Arabic-Exam are `all_sentinel` (the prompt filled the window, no room for the continuation, every choice got the sentinel and the row's argmax is an artifact). The 4.7 % of Arabic-Exam prompts above 1 024 are the same rows for all three tokenizers (long passages; native max 5 867 tokens, AraRooPat 3 331, BPE 8 042).

**The MCQ scorer's continuation window is wrong for every tokenizer that appends `</s>`.** `_compute_loglikelihood` (`tasks/lighteval/base.py`, unchanged since 2026-05-05) encodes the context and the context + continuation separately and scores the tokens `full[ctx_len] … full[full_len − 1]`, assuming the context encoding is a token-prefix of the full one. Every from-scratch tokenizer appends `</s>` to a standalone encoding, so the context ends in `</s>` where the full encoding has the first continuation token: the window **skips the first continuation token and includes the trailing `</s>`**. Verified with the scorer's exact encode call on `الإجابة:` + ` أ`: native scores `[' أ']` (correct, no specials); AraRooPat scores `[CHAR_أ] [LIT_END] </s>` instead of `[LIT_BEGIN] [CHAR_أ] [LIT_END]` (the letter is in the window, the two shared markers are swapped for the end token — the argmax across choices is mostly intact); **BPE-16K scores `['</s>']` alone** — for a single-piece letter continuation the letter itself is never scored and the choice is decided by P(end-of-sequence | context + letter). For a word continuation (`صح`, `هو رأي ايجابي`) every tokenizer that appends `</s>` loses its first token (`[ROOT_صحح]`, `[FUNC_هو]`, ` هو`) and gains `</s>`. The native wrappers are unaffected (Qwen adds nothing; Llama-3 adds BOS only, a shared prefix), so every native number in this report stands; the training path was never affected (answer-only masking uses the LCP helper of `answer_only_masking.py` precisely because of the appended `</s>`); the free-form task strips the trailing `</s>` explicitly. The tests around the wrapper (`test_pmi_normalization.py`, `test_eval_rows.py`, `test_qwen3_support.py`, `test_alghafa_parser.py`, `test_unk_reports.py`) check accuracies and dumps, never *which* tokens were scored, which is how it survived. **Cells on disk with MCQ results under an EOS-appending tokenizer** (excluding `_superseded/`): `_smoke_qwen3_4b/bpe_8k` (bpe). No Qwen arm other than the native SFT cell has MCQ numbers, so the campaign's Qwen comparison is untouched. The Llama-1B sweeps of May–September (the from-scratch panel, `native_llama_3phase_*`, `all_tokenizers_sweep*`) are no longer under `outputs/experiments/` on this machine (a full-depth scan finds three MCQ cells in total: the two smoke cells and `native_qwen3_sft`), so their from-scratch MCQ accuracies survive only where they were quoted, and every one of them was computed on this window.

**Consequences for the MCQ run (folded into §5 item 16, now items 16 and 18–21).** (1) Fix the scorer before any MCQ eval: strip a trailing `</s>` from both encodings when the tokenizer appended one, take the continuation as the tokens after the longest common prefix (the rule the training path already uses), score exactly those, and pin it with a test on an EOS-appending tiny tokenizer that asserts the scored token ids equal the continuation's — the fix must be a no-op for the native wrappers (re-scoring `native_qwen3_sft` must reproduce its four accuracies). (2) `max_length: 4096` for every arm, so that no row truncates except the same handful of Arabic-Exam passages for everyone; report `all_sentinel` per arm from the dumps and give the accuracy on the intersection of rows untruncated under all three tokenizers as the primary comparison, the full benchmark as secondary. (3) Read Alghafa per sub-config: the four word-scored sub-configs (15 800 rows) score `ايجابي` spelled in characters under AraRooPat against a two-token `سلبي`; the letter-scored sub-configs do not have that asymmetry. (4) The character-inventory gaps and the `ma_interrog` tag are tokenizer to-dos for the next AraRooPat vocabulary, not for this run: the v5 checkpoints fix their vocabularies, and the eval must use the tokenizer they were trained with. (5) Whether to re-score the archived Llama sweeps after the fix is the user's call (item 21).

## 4. What the campaign established

1. The native model's loops were partly measurement (the first loop rule) and partly a greedy attractor of the base model; under the corrected rules the untrained base loops on 6.8 % of prompts and the SFT arm on 12.4 %. SFT on the 30 000-record mixture learns the references' register and length but not their content: judge 2.28 vs 2.42, CI including zero.
2. AraRooPat v1/v2 failed for a reason unrelated to morphology: an arbitrary embedding initialization and a 37 M-token budget left the language model at 1.35 nats/char (native 0.83). Informed initialization plus 131 M tokens brought it to 0.88 and the judge from 1.07 to 2.00; answer NLL now equals the untrained native model's. (Loss figures corrected in §3.6; as first measured, 1.34 → 0.97 against a native 0.82.)
3. At equal adaptation a plain BPE-16K vocabulary ties AraRooPat on the judge and beats it on LM loss (0.86 vs 0.88), loops (20.8 vs 31.2 %) and generation speed (153 vs 111 chars/s), with 1.7× more text seen per token budget; AraRooPat has the lower answer NLL on the held-out references (0.81 vs 0.85 nats/char — the 0.64 first reported for BPE divided by the wrong chars-per-token figure, corrected 2026-09-23, §3.6 *Verification*). **Amended by §3.6:** once both arms train on the free-form-aware stop signal, AraRooPat leads the judge (2.152 vs 1.996, paired Δ +0.156 [+0.000, +0.312]), closes the loop gap (21.2 vs 19.6 %) and widens its answer-NLL lead (0.787 vs 0.845); BPE keeps the LM-loss and speed advantages. The tie was partly an artefact of AraRooPat having trained on a fifth of the mixture. AraRooPat's measurable advantages are decode fidelity to the written word (99.3 % word round trip) and root conservation (0.379 vs 0.048). Both arms sit 0.4 judge points under the untouched native model; that gap is the cost of re-learning any vocabulary in 131 M tokens — and it is **not** explained by language-model quality, which after the §3.6 correction is within 0.03 nats/char of native for both.
4. Phase 2 loss is flat from its first fifth in both arms; more raw text at this learning rate is not the next lever.
5. (§3.6) Two measurement faults were corrected. The raw-text "held-out" loss was read from each cell's *own* pool — training text for the adapted arms, and a different document set per arm; on a properly held-out set the LM gap to native is 0.03 nats/char, not 0.13. And the Phase 3 stop signal carried 12 % of the loss tokens: on one arm it wandered upward at step 1 400 and stopped training at a fifth of the mixture. Neither fault changed a ranking; both changed magnitudes enough to change what the next step should be.
6. (§3.7) Neither a token-level repetition penalty nor more and longer Phase 3 free-form data closes AraRooPat's generation gap. A penalty of 1.2 removes loops in every cell (0–1.2 %) but lowers the judge (AraRooPat −0.30, native base −0.29, both CIs below 0; BPE and native SFT within noise), doubles the off-topic flags and, being token-level, halves AraRooPat's article and و tokens; the prompts on which AraRooPat loops hold no recoverable answer (1.72 once the loop is broken, against 2.08 / 2.32 for BPE / native SFT on the same prompts). v4 — 35 000 examples of a 25/15/60 mixture at 1 024 tokens, three times the distinct free-form records — brought the answer NLL to native SFT's level (0.766 nats/char both) without moving the judge (+0.03, CI spans 0) or the loops (23.2 %, worst in the longest-reference tercile). At equal likelihood of the references native SFT loops on half as many prompts: the gap is in free-running generation, which points to sequence-level remedies (distillation from the native model, a training-time repetition objective), not to Phase 3 data or token-level decoding. In every cell the loop rate rises with the reference length.
7. (§3.8) The flatness explanation of the loop gap does not hold — at equal likelihood the from-scratch vocabularies are as sharp per character as native, and loops start at low-margin points in every model, the untrained native base included. Sequence-level distillation from a third-party teacher (Aya-Expanse-32B, chosen by a bake-off on dev prompts; judge 3.65 on the test prompts) with the same teacher answers, mixture, seed and draw rule for all three arms raised every arm by about a quarter point (AraRooPat +0.25 vs v4, native +0.24 vs SFT, BPE +0.14 vs ffstop) and brought loops to the untrained base's level for AraRooPat and native (6.8 / 8.4 %, flat across reference length) but not for BPE-16K (15.6 %). Under identical treatment AraRooPat retains 0.67 of the teacher's judge score, native 0.69 and BPE-16K 0.58: AraRooPat is not separable from the native vocabulary (−0.09 [−0.24, +0.07]; its residual cost is fluency and instruction following, not correctness) and is above BPE-16K (+0.30 [+0.15, +0.45]) — the first such interval under equal treatment. All three students stay far below the teacher (−1.13 to −1.52). On the teacher's own dev text the native vocabulary has the lowest NLL per character (0.388 vs AraRooPat 0.420, BPE 0.425).

## 5. Open items and next steps (in the order proposed)

1. ~~Free-form dev slice for Phase 3 early stopping~~ — **done, §3.6.** Cells `araroopat_3phase_v3_ffstop` and `bpe_16k_3phase_v3_ffstop`.
2. **Register `configs/contamination/rawtext_heldout_v1.jsonl` in `heldout_sets.yaml` before the next pool is built**, so Stage A drops its 150 documents by content. It was deliberately left unregistered during this campaign because registering it moves the held-out fingerprint, hence the pool fingerprint and `exclusions.json`, which would have confounded the Phase 3 re-runs.
3. ~~Loop-aware decoding ablation on the existing checkpoints~~ — **done, §3.7 (P1).** A token-level repetition penalty (1.2) removes the loops in every cell but lowers the judge and, for AraRooPat, strips clitic tokens; it is neither a fix nor a comparison rule. The greedy rule stays.
4. ~~Phase 3 on more and longer free-form data (AraRooPat v4)~~ — **done, §3.7 (P2).** Answer NLL to native SFT's level, judge and loops unchanged; the gate for the replication failed.
5. **P4 — the v4 Phase 3 block on BPE-16K and native** (`configs/experiments/qwen_bpe16k_3phase_v4.yaml`, `qwen_native_sft_v4.yaml`, validated, Phase 3 block identical field for field): not run, because v4 did not pass the gate. Worth running only if the v4 block becomes the reference Phase 3 — it did lower answer NLL and raise instruction following (+0.18 [+0.03, +0.34] vs ffstop), and the +0.19 [+0.03, +0.35] of v4 over BPE ffstop is not a tokenizer finding until BPE has had the same block.
6. ~~Uncertainty diagnostic on the existing checkpoints~~ — **done, §3.8 (P0).** The from-scratch vocabularies are not flatter than native at equal likelihood; loops start at low-margin points in every cell.
7. ~~Distillation, sequence-level~~ — **done, §3.8**, with a neutral teacher (Aya-Expanse-32B, bake-off on dev prompts) instead of the native backbone, for all three arms under the same data. AraRooPat v5 2.43 / native v5 2.52 / BPE v5 2.13; loops 6.8 / 8.4 / 15.6 %.
8. A training-time repetition objective of the DITTO kind (Xu et al. 2022) — **kept, re-scoped** (§3.8 P3.6): the pre-registered trigger (AraRooPat's distilled loops > 1.5 × native's) did not fire (0.81 ×), and the closing condition (all three within 5 points) failed only on BPE-16K (15.6 %); the residual loop gap is BPE's. Before it, as a cheap measurement, a decoded-text (word-level) no-repeat constraint that cannot hit clitic tokens.
9. Equal-text ablation: AraRooPat at ~227 M tokens (1.73× the token budget) so both arms see the same Arabic.
10. A pre-flight check of `qa_blend.share` against the tokenizer's packed QA block count (two Phase 1 runs, 62 min, were lost to it).
11. The probe's step-0 gate is uninformative under tied embeddings; read step-N only, or untie the head.
12. 13 hamza and 1 ى round-trip mismatches remain (corpus-majority canonicalisations).
13. The free-form eval mixture costs 12 s per pass (33 % overhead at `eval_every_n_steps: 200`). Acceptable, but `eval_every_n_steps: 400` would halve it if a longer phase needs the time. At `max_length: 1024` (v4) a pass is 21.6 s (1 000 records at batch 2); every 500 micro-steps that is 22 % of the phase.
14. ~~`scripts/diag_heldout_loss.py` should emit `nll_per_answer_char` and `reference_chars`~~ — **done, §3.7 (P0)**, with `scripts/diag_backfill_per_char.py` for the older JSONs.

15. **Blind human check on the distilled arms** (console Rate tab, 50 prompts × 3 variants: `teacher_aya_expanse_32b_m2_c2400`, `araroopat_3phase_v5_distill`, `native_qwen3_sft_v5_distill`, then a second set with `bpe_16k_3phase_v5_distill`): the three v5 cells write at the teacher's length, and the judge's length bias is the standing caveat of this eval — the +0.30 AraRooPat − BPE interval and the +0.24 to +0.28 distillation gains should be confirmed by a rater before they are quoted. No GPU.
16. **The four MCQ benchmarks on the three v5 checkpoints** (eval-only re-eval configs, `acva` / `alghafa` / `culture_arabic_mmlu` / `arabic_exam`, ~1–2 h GPU): the campaign's recognition-side eval has not been run on any Qwen arm; with three checkpoints trained identically it is the cheapest next comparison, and it says whether the free-form ranking (native ≈ AraRooPat > BPE) holds under log-likelihood scoring.
17. Documentation fix: `sft_eval_mixture_manifest.json` lives under `<cell>/training/data/`, not `<cell>/data/` as the *Phase 3 mixture* section of CLAUDE.md says (§3.8 *Verification* (d)).

18. **Fix the MCQ scorer's continuation window** (§3.9): strip the appended `</s>` from both encodings and take the continuation as the tokens after the longest common prefix; regression test with an EOS-appending tiny tokenizer asserting the scored ids; `native_qwen3_sft` must reproduce its four accuracies byte for byte. Blocks item 16.
19. Item 16's protocol after §3.9: `max_length: 4096` for every arm, `all_sentinel` shares reported per arm, accuracy on the intersection of untruncated rows as the primary comparison, Alghafa read per sub-config (the word-scored sentiment sub-configs spell `ايجابي` in characters under AraRooPat).
20. Tokenizer to-dos for the next AraRooPat vocabulary (not for the fixed v5 checkpoints): add the curly quotes `“ ”`, the Persian letters `ی` / `ھ`, the Quranic marks, the ornate parentheses `﴾ ﴿` and Latin letters to the character / punctuation inventories (0.55–1.1 % of exam / Alghafa word occurrences carry an `<unk>` today), and `ma_interrog` to `CAMEL_CLITIC_SURFACE`.
21. Every from-scratch MCQ accuracy of the Llama-1B era (no longer on this disk; quoted in earlier documents and CLAUDE.md) was computed on the wrong token window — for a single-piece letter continuation, on `</s>` alone — and is void until re-run under item 18's fix; the native Llama numbers stand. Decide whether any of them is worth re-running.

## 6. GPU time (2026-09-22 onwards)
Four training runs 8 h 52 min (1 h 15 lost attempt + 3 h 17 v3 + 0 h 56 lost attempt + 3 h 25 BPE), probe ~35 min, six diagnostics ~6 min, judge passes ~10 min — ≈ 9 h 43 min. Earlier: v1 ~1 h, v2 ~1.5 h (+ 45 min resume), native runs and re-evals ~2 h.

§3.6 (2026-09-22 evening): AraRooPat ffstop 42 min end to end (Phase 3 26.9 min incl. 33 evals, free-form eval 5 min), BPE ffstop 36 min (Phase 3 25.9 min, 35 evals, eval 7.6 min), ten raw-text diagnostics ~12 min, judge 2 × 250 verdicts ~1 min plus a 7-minute cold start — **≈ 1 h 40 min**. Non-GPU: the held-out build + two verification scans ~13 min, the two mixture dry runs ~35 min (AraRooPat over the CAMeL bridge).

§3.7 (2026-09-23): five eval-only runs 19.6 min, two judge passes 10.4 min (incl. cold starts), memory smoke 1.3 min, AraRooPat v4 1 h 22 min (Phase 3 62.6 min, free-form eval 8.4 min), diagnostic 0.7 min — **≈ 1 h 54 min**. Non-GPU: the v4 mixture dry run 6.6 min, the per-character backfill < 1 min.

§3.8 (2026-09-23/24): P0 1.7 min; bake-off generation 28.9 min (six candidates + determinism runs, incl. Falcon-H1's failed bf16 load); the dev reference row 9.5 min; bake-off judge 10.5 min; ceiling 1.8 min + judge 4.7 min; teacher generation 48.4 + 9.9 min; smoke 1.2 min; the three Phase 3 runs 85.4 + 68.1 + 69.0 min; diagnostics 2.2 min; main judge 5.9 min — **≈ 5 h 47 min**. Non-GPU: three planners 10 min, filters and pseudo-cells a few minutes. Downloads: 143 GB of teacher weights (Aya-Expanse-32B was cached).

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
- Kim, Y. & Rush, A. M. (2016). *Sequence-Level Knowledge Distillation*. EMNLP. (Teacher-generated text as the student's target — the only distillation form across two vocabularies; §3.8.)
- *Rethinking LLM Evaluation with 3C3H: AraGen Benchmark and Leaderboard* (Inception / MBZUAI, HF blog) and the AraGen v3 leaderboard snapshot of 2026-05-06 (benchmarklist.com/benchmarks/aragen_v3) — the generative Arabic ranking the teacher candidates were drawn from (§3.8).
- *Jais 2: A Family of Arabic-Centric Open Large Language Models* (2026). arXiv:2608.13580; model cards `inceptionai/Jais-2-8B-Chat`, `inception42/Jais-2-70B-Chat`.
- *Introducing Falcon-H1-Arabic* (TII, HF blog, 2026-01) and `tiiuae/Falcon-H1-34B-Instruct` (the Arabic variants are not on the Hub).
- *Fanar 2.0: Arabic Generative AI Stack* (2026). arXiv:2603.16397 — `QCRI/Fanar-2-27B-Instruct` is Gemma-3-based, hence excluded as the judge's family.
- Model cards used for the bake-off: `CohereLabs/aya-expanse-32b`, `CohereLabs/c4ai-command-r7b-arabic-02-2025`, `ALLaM-AI/ALLaM-7B-Instruct-preview`, `casperhansen/llama-3.3-70b-instruct-awq`.

## 8. Where the numbers live
`outputs/experiments/qwen_native_vs_araroopat/<cell>/all_metrics.json` (metrics, judge summary, training histories), `eval_rows/freeform_cidar.parquet` (every generation), `freeform_judge/gemma4_31b.parquet` (verdicts), `diag_heldout_loss*.json` (LM loss / answer NLL), `embedding_init_probe.json`, `_p3_decode/roundtrip_v{2,3}_*.json`, `_report_2026-09-22/report_draft.md` (the 2026-09-22 hand-off report), `araroopat_3phase_v2/phase12_run_20260921-115522/` (v2's Phase 1/2 log). Configs: `configs/experiments/qwen_native_{no_presteps_after_LOOP_fix,sft_only,sft_reeval}.yaml`, `qwen_araroopat_3phase{,_v2,_v2_resume_sft,_v3}.yaml`, `qwen_bpe16k_3phase_v3.yaml`. Commits: `6a1d3db` (scheduler, templates, loop stop, SDPA), `6eee412` (periodic loop stop), `a2c671d` (markers + budget), `a879d73`/`f1ee2f5`/`3885dda` (console), `e06910e` (decode), `b204a54` (embedding init), `3ddf26c`…`5c91854` (v3 + comparator), `562da5e` (this record + the v2 configs), `cb68d34` (held-out raw-text set + the corrected loss numbers), `2615c37` (dev slices + `eval_mixture`), `77fded5` (the two ffstop configs + `paired_compare.py`).

§3.6 artifacts: `configs/contamination/rawtext_heldout_v1.jsonl` (+ `.manifest.json`, sha256 `c6522a4f0050702b…`), `<cell>/diag_heldout_rawtext_v1[_warmup][_base].json` (eight checkpoints of the earlier cells, two of the new ones), `<cell>/data/sft_eval_mixture_manifest.json`, `all_metrics.json["training"]["sft"]["eval_history"]`, `scripts/build_rawtext_heldout.py`, `scripts/judge/paired_compare.py`. The pre-2026-09-22 `diag_heldout_loss*.json` files are kept; they record what the old pool-tail rule measured.

§3.7 artifacts: listed under §3.7 *Artifacts*. Commits: `4efedb3` (the §3.6 verification edits), `e10863c` (answer NLL per character + backfill), `ab2e679` (decoding knobs + ablation configs), `a216601` (`paired_compare.py` across two folders), `4e702ce` (v4 + P4 configs), and the documentation commit of §3.7. The 19 backfilled `diag_heldout_*.json` files now carry `answer_nll_total_source: "backfill: …"`; v4's carries `"measured"`.

§3.8 artifacts: listed under §3.8 *Artifacts*. Commits: `c59c1e7` (uncertainty diagnostic), `17e790a` (distill code: bake-off, teacher generation, pseudo-cells, prompts, filter), `345bf51` (teacher-answer overlay + provenance), `02d0dd8` (the three v5 configs), and the documentation commit of §3.8. Teacher file sha256 `8e4d9cd13e3d1bc8a2cd0782266f570dc771a55f500e0609a1c808955bbcf272`.

## 9. Cells of `outputs/experiments/qwen_native_vs_araroopat/`
Current: `native_qwen3_base_m2_c2400`, `native_qwen3_sft_m2_c2400`, `araroopat_3phase_v2`, `araroopat_3phase_v3`, `bpe_16k_3phase_v3`, **`araroopat_3phase_v3_ffstop`**, **`bpe_16k_3phase_v3_ffstop`** (§3.6: the same recipe with the free-form-aware early stop), **`araroopat_3phase_v4`** (§3.7: Phase 3 at 45 000 × 25/15/60, max_length 1 024 — judge 2.18, the highest AraRooPat cell but not separable from ffstop; the current best arms of each vocabulary are v4 / ffstop for AraRooPat and ffstop for BPE). Older rule sets, kept: `native_qwen3` (untrained base, v1 template), `native_qwen3_base`, `native_qwen3_sft`, `araroopat` (v1). Under `_superseded/`: `native_qwen3_base_after_loop_fix`, `native_qwen3_sft_reeval`, `native_qwen3_{base,sft}_m2_c1200`, the two lost v3 attempts.

New in §3.8: **`araroopat_3phase_v5_distill`**, **`bpe_16k_3phase_v5_distill`**, **`native_qwen3_sft_v5_distill`** (v4's Phase 3 on the Aya-Expanse-32B teacher answers; judge 2.43 / 2.13 / 2.52) and the pseudo-cell **`teacher_aya_expanse_32b_m2_c2400`** (the teacher on the test prompts, 3.65; no model). In `outputs/experiments/teacher_bakeoff/`: the six candidate pseudo-cells and `native_qwen3_base_dev` (the untouched base on the same 250 dev prompts).

In `outputs/experiments/qwen_decoding_ablation/` (§3.7 P1, eval only, repetition penalty 1.2): `araroopat_3phase_v3_ffstop_rp12`, `bpe_16k_3phase_v3_ffstop_rp12`, `native_qwen3_sft_rp12`, `native_qwen3_base_rp12`; `_repro/araroopat_3phase_v3_ffstop_greedy` (the greedy reproduction check, identical to its twin in 250 of 250 rows).
