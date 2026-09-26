# Qwen3-4B-Base: native tokenizer vs AraRooPat — campaign record (2026-09-18 → 2026-09-25)

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
| 09-24/25 | scorer window fixed (`2f56441`, no-op on native: 1 200 rows, Δ 0.0); the four MCQ benchmarks at `max_length 4096` on the three v5 arms, the untouched base and the two Phase-2 checkpoints (six cells, 16 h 8 min GPU) | on the letter-scored benchmarks (rows no cell truncates) AraRooPat v5 is **last**: Arabic-Exam 0.549 / 0.555 vs BPE-16K 0.595 / 0.589 vs native v5 0.619 / 0.594 (char / PMI), Culture-MMLU 0.472 / 0.465 vs 0.501 / 0.502 vs 0.520 / 0.504; Phase 3 on teacher text raised recognition (AraRooPat +5–8 points over its Phase 2 checkpoint); ACVA and Alghafa's fixed-label groups are decided by label-prior collapse (PMI picks the character-spelled ` ايجابي` on 97–99 % of rows under AraRooPat), not read (§3.10) |
| 09-25 | few-shot handling diagnostic on the MCQ dumps (CPU; the training MCQ records are single-question, the eval is 3-shot) | no cell copies the demonstrations above chance (last-demo letter 25–31 % vs the gold's 27 / 26 %); the v5 arms lean on the demos least; AraRooPat's deficit is concentrated on gold = أ (0.32 / 0.28 vs 0.49–0.62 on the other letters, أ ranked last on 27 / 24 % of those rows) and paired: −0.17 / −0.12 vs BPE-16K on the أ rows, +0.000 / +0.001 on every other row (and +0.017 / +0.020 vs native v5 there); a post-hoc letter calibration does not move it (≤ 0.007), while it lifts the base and the Phase-2 checkpoints by 3–5 points; letter and first position are confounded → §5 item 26. ACVA under PMI is one prior offset from flipping in every cell (0.47 ↔ 0.67), not a binary-task strength. |
| 09-25 (b) | letter or slot: `label_rotation` / `rows_file` / schema-3 `cont_token_ll` (`c05aa1f`); 20 eval-only cells on seeded 1 000-row subsets of Arabic-Exam and Culture-MMLU (four arms × rotations 0–3 + 0-shot, 3 h 20 min GPU); rot0 replicates the full dumps exactly; the Alghafa MSA scope rule (§3.11) | AraRooPat v5's deficit follows the **letter** أ, not the first slot: AraRooPat − BPE-16K on gold letter أ −0.202 / −0.149 at every slot, on gold slot 1 −0.027 / −0.052 (G_letter +0.230 / +0.140, G_slot −0.003 / +0.010); every trained arm under-selects slot 1 (E_slot +0.08 to +0.20), the arms differ in letter prior (AraRooPat favours د, the others أ); pre-registered rule: *both* on the per-arm effects; `[LIT_END]` ≈ 0.001 nats — the score is `[CHAR_x]`; the gap survives 0-shot (−0.120 / −0.099 on gold أ); under a letter-invariant rotation-averaged decision (a proposal, applied to nothing) AraRooPat − BPE-16K is −0.001 [−0.030, +0.027] on Arabic-Exam and −0.039 [−0.068, −0.011] on Culture-MMLU. Alghafa in scope = MSA choice-text group: AraRooPat − BPE-16K −0.021 [−0.042, −0.002] char; the dialect rows (out of scope) −0.052; 900 of the 5 400 `meta_ar_dialects` rows duplicate `meta_ar_msa` |
| 09-25 (c) | where the letter prior comes from: the Phase 1 (embedding-only) and Phase 2 checkpoints of both from-scratch arms under rotations 0–3 (16 eval-only cells, 2 h 46 min GPU; the P2 rot0 cells replicate §3.10's dumps exactly) | AraRooPat − BPE-16K on gold letter أ is **−0.201 / −0.155 already after Phase 1**, −0.288 / −0.261 after Phase 2, −0.202 / −0.149 at v5: the AraRooPat-specific أ deficit comes with the vocabulary, not with Phase 3. Phase 3 shifts every arm's letter prior the same way (ΔE_أ +0.274 / +0.250 AraRooPat, +0.372 / +0.399 BPE-16K), which turns AraRooPat's weaker أ preference into the absolute aversion and creates its د preference; the pre-registered Phase-3 rule fires. The slot-1 aversion is present from Phase 1 in both from-scratch arms (§3.12) |
| 09-25/26 | tooling: the console's Analysis tab — an LLM answers questions about the cells by running Python on their results (`ae` helpers, sandboxed worker, OpenAI or a local Gemma; `cf9eb11`…`83faafc`) | 5 known-answer questions: local Gemma-4-31B 5/5, gpt-5.6-terra 4/5; the helpers reproduce §3.10's and §3.8's numbers exactly; `accuracy` in `all_metrics.json` = PMI in char+pmi cells (CLAUDE.md corrected) |

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

### 2.9 Tooling (2026-09-25/26): the Analysis tab
The console gained an **Analysis** tab: a question in plain language, a model that answers by running Python on the cells' results in a sandboxed worker (read-only filesystem except the session folder, no network, own PID namespace), every step's code / output / table / chart kept on the page, and the answer's numbers checked against what the steps printed (an answer with unprinted numbers is held once and the model asked to compute them). The helpers it calls (`arabic_eval.analysis`, also usable from a notebook) wrap `scripts/mcq_compare.py` and `scripts/judge/paired_compare.py`, so they reproduce this report exactly — AraRooPat v5 vs native v5 on Arabic-Exam 14 114 rows, 0.551 / 0.621 char, McNemar 1 396 / 2 386 (§3.10); judge AraRooPat v5 − BPE-16K v5 +0.300 [+0.152, +0.452] (§3.8). Measured on 5 known-answer questions (fresh session each; `outputs/analysis/_bench_2026-09-25/`): local `google/gemma-4-31b-it` under vLLM with native tool calls and fp8 weights (31.7 GiB, 64K context) **5 / 5** in 3–9 s — with fenced code blocks it scored 1 / 5 (it writes its own call syntax); `gpt-5.6-terra` with fenced blocks **4 / 5** in 6–15 s (its chat-completions endpoint refuses function tools together with a reasoning effort). The one miss is instructive: asked for the Phase 3 step with the best eval loss, it returned the minimum of `eval_history` (step 19 500, 0.8192) instead of the step early stopping restored (17 000, 0.8195 — the `min_delta` rule). Found on the way: `downstream.<task>.accuracy` follows the *primary* normalization, i.e. PMI in every char+pmi cell (the v5 ones included), not char-norm as CLAUDE.md said (corrected there; read `accuracy_char_norm` / `accuracy_pmi` by name). Commits `cf9eb11` (helpers), `ff6fba9` (worker), `1da325f` (agent, local server, routes), `83faafc` (page); how to use it: `debugger/README.md` → *Analysis tab*.

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

### 3.10 MCQ benchmarks on the v5 checkpoints under the fixed scorer (2026-09-24/25)

**The fix** (commit `2f56441`, §3.9). `_compute_loglikelihood` now strips one trailing `</s>` from the context and the context + continuation encodings, takes the continuation as the tokens after their longest common prefix (`answer_only_masking.continuation_start`, the rule the answer-only training loss already used) and sums exactly those; it returns a `ScoredLogLikelihood` (a `float` carrying `n_tokens` / `truncated`), and the row dump (schema 2) records `cont_tokens` per choice and sets `hit_cap` also when the cap reached a continuation. What is scored now on `الإجابة:` + ` أ`: native Qwen3 `[' أ']` (unchanged), AraRooPat `[LIT_BEGIN] [CHAR_أ] [LIT_END]` (was `[CHAR_أ] [LIT_END] </s>`), BPE-16K `[' أ']` (was `['</s>']`); on ` هو رأي ايجابي`: native `' هو' ' ر' 'أ' 'ي' ' ا' 'يج' 'اب' 'ي'` (unchanged), AraRooPat `[FUNC_هو] [ROOT_ر##] [PAT_1َأْيِ] [LIT_BEGIN] [CHAR_ا] [CHAR_ي] [CHAR_ج] [CHAR_ا] [CHAR_ب] [CHAR_ي] [LIT_END]` (was the same minus `[FUNC_هو]`, plus `</s>`), BPE-16K `' هو' ' رأي' ' اي' 'جابي'` (was minus `' هو'`, plus `</s>`). Tests: `tests/test_lighteval_scorer_window.py` (24). **No-op check** (`_audit/scorer_noop_check_2026-09-24.{py,json}`, run 2): 300 seeded rows per benchmark of `native_qwen3_sft` (its `training/sft` checkpoint, `max_length 1024`, 3-shot / ACVA 0-shot) re-scored and joined to the 2026-09-18 dumps by `row_index` — 1 200 rows, 0 prompt mismatches, argmax agreement 100 % under char and PMI on every task, and every stored value identical once cast to float32 like the dump: max |Δ ll| = 0.0 and max |Δ score_pmi| = 0.0 on 3 369 scored and 409 sentinel choices. (Run 1, comparing unrounded values, read ≤ 4.8e-7 on `ll` and up to 31.98 on the PMI score of sentinel choices — float32's spacing of 64 at −1e9; its JSON is kept as `…_run1.json`.)

**Protocol** (`configs/experiments/mcq_v5/*.yaml`, commit `2441518`; generated through the console's `reeval_config` so each cell's tokenizer block and checkpoint are its own, every other section written explicitly and identical across the six files — resolved `evaluation`, `data`, `training` and `sweep.tasks` compared equal). ACVA 0-shot, Alghafa / Arabic-Exam / Culture-MMLU 3-shot, **`max_length 4096`** for every task and cell (§3.9: at 1 024 up to 26 % of AraRooPat's Alghafa prompts overflow), `score_normalization: char+pmi`, row dumps and intrinsic metrics on, every phase disabled. The primary comparison is on the **intersection** of rows that hit the cap in no cell (`hit_cap` or `sentinel` anywhere drops the row for all) — `native_qwen3_sft`, dumped at 1 024, is included in the tables and in the intersection, flagged `†`, so its truncated rows leave the primary comparison like anyone's; a second pass without it (`mcq_compare_4096`) is the sensitivity check. Pairs: paired bootstrap over rows (10 000 resamples, seed 0 — `freeform_judge.paired_bootstrap`, the convention of `paired_compare.py`) and McNemar discordant counts. Run in the brief's order as one detached chain (`_mcq_logs/run_mcq_v5_chain.sh`): AraRooPat v5 → BPE-16K v5 → native v5 → native base → AraRooPat warmup → BPE-16K warmup, every one exit 0.

| cell (checkpoint) | wall | peak GPU |
|---|---|---|
| `araroopat_3phase_v5_distill_mcq4096` (`araroopat_3phase_v5_distill/training/sft`) | 2 h 54 min (ACVA 13.9 / Alghafa 68.1 / Exam 40.5 / Culture 45.4 min, intrinsic 6 min) | 16.4 GiB |
| `bpe_16k_3phase_v5_distill_mcq4096` (`bpe_16k_3phase_v5_distill/training/sft`) | 2 h 35 min | 12.6 GiB |
| `native_qwen3_sft_v5_distill_mcq4096` (`native_qwen3_sft_v5_distill/training/sft`) | 2 h 41 min | 68.6 GiB |
| `native_qwen3_base_mcq4096` (`Qwen/Qwen3-4B-Base`) | 2 h 40 min | 68.6 GiB |
| `araroopat_3phase_v3_warmup_mcq4096` (`araroopat_3phase_v3/training/warmup`) | 2 h 47 min | 16.4 GiB |
| `bpe_16k_3phase_v3_warmup_mcq4096` (`bpe_16k_3phase_v3/training/warmup`) | 2 h 29 min | 16.4 GiB |

The native cells peak at 68.6 GiB (the 151 936-row output head over a 4 096-token forward, plus the HF loss's float32 copy); it fits the H100, with little room to spare. The first Alghafa pass (AraRooPat) peaked at 12.1 GiB (12 433 MiB) — 4 096 fits for every cell, so no cell fell back to 2 048.

**Diagnostics per cell and task** (all rows; % of rows; `cont_tok` = mean continuation tokens per scored choice — the letter ` أ` is 1 under native and BPE-16K and 3 under AraRooPat, ACVA's ` صح` / ` خطأ` 1 / 2 under native, 1 / 1 under BPE, 2 / 2 under AraRooPat; `native_qwen3_sft` has schema-1 dumps, no `cont_tokens`):

| cell | ACVA hit_cap / all_sent / near_tie % · cont_tok | Alghafa hit_cap / all_sent / near_tie % · cont_tok | Arabic-Exam hit_cap / all_sent / near_tie % · cont_tok | Culture-MMLU hit_cap / all_sent / near_tie % · cont_tok | MEI (ACVA / Alghafa / Exam / Culture) |
|---|---|---|---|---|---|
| AraRooPat v5 | 0.0 / 0.0 / 1.0 · 2.00 | 0.0 / 0.0 / 0.2 · 9.39 | 0.1 / 0.1 / 0.2 · 3.01 | 0.0 / 0.0 / 0.5 · 3.00 | 4.963 / 1.592 / 2.301 / 1.668 |
| BPE-16K v5 | 0.0 / 0.0 / 0.0 · 1.00 | 0.0 / 0.0 / 0.1 · 4.59 | 3.3 / 3.3 / 5.1 · 1.01 | 0.0 / 0.0 / 6.1 · 1.00 | 1.040 / 0.741 / 0.737 / 0.547 |
| native v5 | 0.0 / 0.0 / 0.1 · 1.50 | 0.0 / 0.0 / 0.6 · 7.75 | 2.4 / 2.4 / 5.8 · 1.01 | 0.0 / 0.0 / 4.1 · 1.00 | 1.002 / 0.701 / 0.725 / 0.531 |
| native base | 0.0 / 0.0 / 0.6 · 1.50 | 0.0 / 0.0 / 0.8 · 7.75 | 2.4 / 2.4 / 4.8 · 1.01 | 0.0 / 0.0 / 3.0 · 1.00 | 1.409 / 0.677 / 0.726 / 0.531 |
| AraRooPat warmup | 0.0 / 0.0 / 0.8 · 2.00 | 0.0 / 0.0 / 0.3 · 9.39 | 0.1 / 0.1 / 1.1 · 3.01 | 0.0 / 0.0 / 1.5 · 3.00 | 3.727 / 1.795 / 2.056 / 1.564 |
| BPE-16K warmup | 0.0 / 0.0 / 0.9 · 1.00 | 0.0 / 0.0 / 0.1 · 4.59 | 3.3 / 3.3 / 3.0 · 1.01 | 0.0 / 0.0 / 4.5 · 1.00 | 1.431 / 0.719 / 0.703 / 0.535 |
| native SFT (1 024) † | 0.0 / 0.0 / 0.4 · — | 19.3 / 19.3 / 0.3 · — | 4.3 / 4.3 / 4.2 · — | 6.6 / 6.6 / 4.8 · — | 1.278 / 0.919 / 0.777 / 0.581 |

At 4 096 only Arabic-Exam truncates in the new cells — 17 rows under AraRooPat, 340 under native, 482 under BPE-16K (the same long passages; BPE-16K's byte fragments make them longest, 8 042 tokens in §3.9); `all_sentinel` equals `hit_cap` everywhere. MEI is reported for completeness and not read: it multiplies accuracy by RPS, AraRooPat's RPS (0.379) is mechanically high (`RPS_MECHANICAL_FLAGS`), and the per-task inference times differ by < 10 % across cells.

**Accuracy** (char-norm / PMI; ∩ = the primary intersection, bold):

| cell | ACVA all (char / PMI) | ACVA ∩ (char / PMI) | Alghafa all (char / PMI) | Alghafa ∩ (char / PMI) | Arabic-Exam all (char / PMI) | Arabic-Exam ∩ (char / PMI) | Culture-MMLU all (char / PMI) | Culture-MMLU ∩ (char / PMI) |
|---|---|---|---|---|---|---|---|---|
| AraRooPat v5 | 0.423 / 0.666 | **0.423 / 0.666** | 0.661 / 0.410 | **0.680 / 0.428** | 0.554 / 0.561 | **0.549 / 0.555** | 0.467 / 0.459 | **0.472 / 0.465** |
| BPE-16K v5 | 0.412 / 0.495 | **0.412 / 0.495** | 0.696 / 0.619 | **0.712 / 0.675** | 0.583 / 0.576 | **0.595 / 0.589** | 0.493 / 0.495 | **0.501 / 0.502** |
| native v5 | 0.433 / 0.471 | **0.433 / 0.471** | 0.731 / 0.606 | **0.745 / 0.664** | 0.611 / 0.589 | **0.619 / 0.594** | 0.512 / 0.496 | **0.520 / 0.504** |
| native base | 0.474 / 0.670 | **0.474 / 0.670** | 0.726 / 0.584 | **0.731 / 0.637** | 0.589 / 0.587 | **0.598 / 0.592** | 0.473 / 0.494 | **0.483 / 0.501** |
| AraRooPat warmup | 0.404 / 0.482 | **0.404 / 0.482** | 0.640 / 0.447 | **0.663 / 0.478** | 0.486 / 0.480 | **0.483 / 0.476** | 0.414 / 0.411 | **0.418 / 0.415** |
| BPE-16K warmup | 0.405 / 0.686 | **0.405 / 0.686** | 0.673 / 0.604 | **0.686 / 0.661** | 0.543 / 0.549 | **0.554 / 0.558** | 0.464 / 0.484 | **0.470 / 0.492** |
| native SFT (1 024) † | 0.475 / 0.560 | **0.475 / 0.560** | 0.643 / 0.591 | **0.734 / 0.675** | 0.591 / 0.555 | **0.609 / 0.566** | 0.491 / 0.469 | **0.510 / 0.484** |

∩ sizes: ACVA 9000 of 9000, Alghafa 18542 of 22977, Arabic-Exam 13840 of 14455, Culture-MMLU 13382 of 14327. What the intersection rule removed: ACVA nothing; Alghafa 4 435 rows (19.3 %), all by the 1 024 cell `native_qwen3_sft` — all 80 true/false rows, 3 575 of 5 400 `meta_ar_dialects`, 589 of 900 `meta_ar_msa`, 122 of 155 `soqal`, 69 of 155 `xglue_mlqa`; Arabic-Exam 615 rows (4.3 %) — 482 capped under BPE-16K, 340 under native, 17 under AraRooPat, 133 only by the 1 024 cell; Culture-MMLU 945 rows (6.6 %), all by the 1 024 cell.

**Alghafa per sub-config** (primary intersection, char / PMI; the four fixed-label sub-configs are the true/false and three sentiment ones, whose continuations are the same 2–3 label phrases on every row; the other five score per-row answer text):

| cell | mcq_exams_test_ar (∩ 562) | meta_ar_dialects (∩ 1825) | meta_ar_msa (∩ 311) | multiple_choice_facts_truefalse_balanced_task (∩ 0) | multiple_choice_grounded_statement_soqal_task (∩ 33) | multiple_choice_grounded_statement_xglue_mlqa_task (∩ 86) | multiple_choice_rating_sentiment_no_neutral_task (∩ 8000) | multiple_choice_rating_sentiment_task (∩ 6000) | multiple_choice_sentiment_task (∩ 1725) | fixed-label ∩ | choice-text ∩ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AraRooPat v5 | 0.447 / 0.326 | 0.580 / 0.352 | 0.775 / 0.447 | — / — | 0.758 / 0.394 | 0.849 / 0.442 | 0.854 / 0.527 | 0.590 / 0.339 | 0.341 / 0.387 | 0.697 / 0.440 | 0.585 / 0.361 |
| BPE-16K v5 | 0.456 / 0.338 | 0.620 / 0.412 | 0.794 / 0.543 | — / — | 0.879 / 0.576 | 0.907 / 0.558 | 0.882 / 0.902 | 0.621 / 0.592 | 0.397 / 0.333 | 0.729 / 0.721 | 0.618 / 0.418 |
| native v5 | 0.472 / 0.317 | 0.685 / 0.385 | 0.823 / 0.453 | — / — | 0.909 / 0.485 | 0.907 / 0.477 | 0.902 / 0.860 | 0.668 / 0.615 | 0.413 / 0.383 | 0.759 / 0.714 | 0.667 / 0.383 |
| native base | 0.484 / 0.311 | 0.698 / 0.373 | 0.836 / 0.473 | — / — | 0.879 / 0.424 | 0.884 / 0.465 | 0.876 / 0.832 | 0.657 / 0.586 | 0.401 / 0.332 | 0.741 / 0.683 | 0.678 / 0.375 |
| AraRooPat warmup | 0.443 / 0.324 | 0.538 / 0.333 | 0.765 / 0.424 | — / — | 0.758 / 0.394 | 0.814 / 0.407 | 0.838 / 0.614 | 0.570 / 0.394 | 0.356 / 0.352 | 0.683 / 0.501 | 0.555 / 0.344 |
| BPE-16K warmup | 0.450 / 0.322 | 0.621 / 0.388 | 0.801 / 0.505 | — / — | 0.939 / 0.606 | 0.919 / 0.523 | 0.861 / 0.887 | 0.575 / 0.578 | 0.366 / 0.333 | 0.697 / 0.708 | 0.620 / 0.395 |
| native SFT (1 024) † | 0.472 / 0.329 | 0.662 / 0.376 | 0.830 / 0.463 | — / — | 0.848 / 0.455 | 0.872 / 0.453 | 0.895 / 0.887 | 0.651 / 0.621 | 0.412 / 0.364 | 0.749 / 0.728 | 0.651 / 0.380 |

**Label collapse on the fixed-label groups** — the most-predicted label and its share under char / PMI (4096-only intersection, so the 80 true/false rows are in; ⚠ = ≥ 90 %; gold shares: ACVA صح 0.596 / خطأ 0.404, true/false 0.50 / 0.50, `rating_sentiment_no_neutral` 0.50 / 0.50, the two 3-way ones 0.33 each). Alghafa shuffles the choice order per row, so the dump's position histogram hides this; `mcq_compare.py` reads it by label text (commit `7fb914b`):

| cell | ACVA | multiple_choice_facts_truefalse_balanced_task | multiple_choice_rating_sentiment_no_neutral_task | multiple_choice_rating_sentiment_task | multiple_choice_sentiment_task |
|---|---|---|---|---|---|
| AraRooPat v5 | خطأ 0.98 ⚠ / صح 0.58 | صحيح 0.62 / خطأ 0.55 | هو رأي ايجابي 0.57 / هو رأي ايجابي 0.97 ⚠ | هو رأي ايجابي 0.54 / هو رأي ايجابي 0.99 ⚠ | هي جملة ايجابية 0.96 ⚠ / هي جملة ايجابية 0.58 |
| BPE-16K v5 | خطأ 0.99 ⚠ / خطأ 0.87 | خطأ 0.54 / خطأ 0.55 | هو رأي سلبي 0.56 / هو رأي ايجابي 0.50 | هو رأي سلبي 0.54 / هو رأي سلبي 0.57 | هي جملة ايجابية 0.58 / هي جملة ايجابية 1.00 ⚠ |
| native v5 | خطأ 0.95 ⚠ / خطأ 0.89 | خطأ 0.54 / خطأ 0.57 | هو رأي سلبي 0.54 / هو رأي ايجابي 0.62 | هو رأي سلبي 0.49 / هو رأي ايجابي 0.62 | هي جملة سلبية 0.42 / هي جملة سلبية 0.52 |
| native base | خطأ 0.89 / خطأ 0.58 | خطأ 0.54 / خطأ 0.57 | هو رأي سلبي 0.54 / هو رأي ايجابي 0.64 | هو رأي سلبي 0.47 / هو رأي ايجابي 0.60 | هي جملة سلبية 0.42 / هي جملة سلبية 0.81 |
| AraRooPat warmup | خطأ 1.00 ⚠ / خطأ 0.88 | خطأ 0.54 / خطأ 0.65 | هو رأي سلبي 0.53 / هو رأي ايجابي 0.88 | هو رأي سلبي 0.58 / هو رأي ايجابي 0.91 ⚠ | هي جملة ايجابية 0.80 / هي جملة محايدة 0.90 |
| BPE-16K warmup | خطأ 1.00 ⚠ / صح 0.63 | خطأ 0.54 / خطأ 0.56 | هو رأي سلبي 0.57 / هو رأي ايجابي 0.52 | هو رأي سلبي 0.61 / هو رأي سلبي 0.57 | هي جملة ايجابية 0.72 / هي جملة ايجابية 1.00 ⚠ |

**Pairs** (primary intersection; A − B; McNemar = rows only A / only B got right):

| A − B | task | n ∩ | char Δ [95 % CI] | only A / only B | PMI Δ [95 % CI] | only A / only B |
|---|---|---|---|---|---|---|
| AraRooPat v5 − native base | ACVA | 9000 | -0.051 [-0.058, -0.044] | 271 / 730 | -0.004 [-0.017, +0.008] | 1567 / 1607 |
| BPE-16K v5 − native base | ACVA | 9000 | -0.062 [-0.068, -0.055] | 194 / 748 | -0.175 [-0.187, -0.164] | 739 / 2317 |
| native v5 − native base | ACVA | 9000 | -0.041 [-0.046, -0.036] | 101 / 470 | -0.199 [-0.209, -0.188] | 544 / 2333 |
| AraRooPat v5 − BPE-16K v5 | ACVA | 9000 | +0.011 [+0.007, +0.014] | 183 / 88 | +0.171 [+0.157, +0.185] | 3015 / 1477 |
| AraRooPat v5 − native v5 | ACVA | 9000 | -0.010 [-0.015, -0.005] | 252 / 342 | +0.194 [+0.180, +0.209] | 3251 / 1502 |
| BPE-16K v5 − native v5 | ACVA | 9000 | -0.021 [-0.026, -0.016] | 170 / 355 | +0.023 [+0.016, +0.031] | 737 / 526 |
| native v5 − native SFT (1 024) † | ACVA | 9000 | -0.042 [-0.047, -0.037] | 94 / 475 | -0.089 [-0.096, -0.082] | 167 / 966 |
| AraRooPat v5 − AraRooPat warmup | ACVA | 9000 | +0.019 [+0.016, +0.022] | 184 / 14 | +0.183 [+0.169, +0.197] | 2996 / 1345 |
| BPE-16K v5 − BPE-16K warmup | ACVA | 9000 | +0.007 [+0.005, +0.010] | 80 / 13 | -0.191 [-0.205, -0.177] | 1447 / 3167 |
| AraRooPat v5 − native base | Alghafa | 18542 | -0.051 [-0.057, -0.044] | 1416 / 2361 | -0.209 [-0.216, -0.201] | 1036 / 4909 |
| BPE-16K v5 − native base | Alghafa | 18542 | -0.019 [-0.024, -0.014] | 1044 / 1395 | +0.039 [+0.033, +0.045] | 2025 / 1309 |
| native v5 − native base | Alghafa | 18542 | +0.014 [+0.010, +0.018] | 835 / 575 | +0.027 [+0.023, +0.031] | 1071 / 572 |
| AraRooPat v5 − BPE-16K v5 | Alghafa | 18542 | -0.032 [-0.038, -0.026] | 1491 / 2085 | -0.247 [-0.255, -0.240] | 1118 / 5707 |
| AraRooPat v5 − native v5 | Alghafa | 18542 | -0.065 [-0.071, -0.059] | 1135 / 2340 | -0.236 [-0.243, -0.229] | 684 / 5056 |
| BPE-16K v5 − native v5 | Alghafa | 18542 | -0.033 [-0.038, -0.028] | 738 / 1349 | +0.012 [+0.006, +0.017] | 1493 / 1276 |
| native v5 − native SFT (1 024) † | Alghafa | 18542 | +0.011 [+0.008, +0.015] | 622 / 412 | -0.012 [-0.015, -0.008] | 474 / 691 |
| AraRooPat v5 − AraRooPat warmup | Alghafa | 18542 | +0.017 [+0.012, +0.022] | 1209 / 895 | -0.050 [-0.054, -0.045] | 500 / 1424 |
| BPE-16K v5 − BPE-16K warmup | Alghafa | 18542 | +0.027 [+0.022, +0.031] | 1024 / 531 | +0.015 [+0.012, +0.018] | 503 / 232 |
| AraRooPat v5 − native base | Arabic-Exam | 13840 | -0.049 [-0.059, -0.039] | 1978 / 2656 | -0.037 [-0.046, -0.028] | 1787 / 2298 |
| BPE-16K v5 − native base | Arabic-Exam | 13840 | -0.002 [-0.011, +0.006] | 1632 / 1665 | -0.003 [-0.011, +0.005] | 1602 / 1644 |
| native v5 − native base | Arabic-Exam | 13840 | +0.022 [+0.015, +0.028] | 1216 / 917 | +0.002 [-0.004, +0.007] | 823 / 801 |
| AraRooPat v5 − BPE-16K v5 | Arabic-Exam | 13840 | -0.047 [-0.055, -0.038] | 1425 / 2070 | -0.034 [-0.042, -0.025] | 1540 / 2009 |
| AraRooPat v5 − native v5 | Arabic-Exam | 13840 | -0.071 [-0.079, -0.062] | 1375 / 2352 | -0.038 [-0.048, -0.029] | 1730 / 2263 |
| BPE-16K v5 − native v5 | Arabic-Exam | 13840 | -0.024 [-0.031, -0.017] | 1028 / 1360 | -0.005 [-0.012, +0.003] | 1518 / 1582 |
| native v5 − native SFT (1 024) † | Arabic-Exam | 13840 | +0.010 [+0.003, +0.017] | 1340 / 1197 | +0.027 [+0.021, +0.033] | 1159 / 782 |
| AraRooPat v5 − AraRooPat warmup | Arabic-Exam | 13840 | +0.066 [+0.058, +0.074] | 2082 / 1171 | +0.079 [+0.070, +0.087] | 2446 / 1355 |
| BPE-16K v5 − BPE-16K warmup | Arabic-Exam | 13840 | +0.041 [+0.033, +0.049] | 1836 / 1268 | +0.031 [+0.024, +0.038] | 1436 / 1001 |
| AraRooPat v5 − native base | Culture-MMLU | 13382 | -0.011 [-0.021, -0.001] | 2273 / 2416 | -0.036 [-0.046, -0.026] | 2020 / 2501 |
| BPE-16K v5 − native base | Culture-MMLU | 13382 | +0.018 [+0.009, +0.027] | 2069 / 1827 | +0.001 [-0.007, +0.009] | 1612 / 1594 |
| native v5 − native base | Culture-MMLU | 13382 | +0.038 [+0.030, +0.045] | 1692 / 1189 | +0.003 [-0.004, +0.010] | 1243 / 1202 |
| AraRooPat v5 − BPE-16K v5 | Culture-MMLU | 13382 | -0.029 [-0.036, -0.021] | 1272 / 1657 | -0.037 [-0.046, -0.029] | 1445 / 1944 |
| AraRooPat v5 − native v5 | Culture-MMLU | 13382 | -0.048 [-0.056, -0.040] | 1272 / 1918 | -0.039 [-0.048, -0.030] | 1843 / 2365 |
| BPE-16K v5 − native v5 | Culture-MMLU | 13382 | -0.019 [-0.027, -0.012] | 1094 / 1355 | -0.002 [-0.010, +0.006] | 1421 / 1444 |
| native v5 − native SFT (1 024) † | Culture-MMLU | 13382 | +0.010 [+0.003, +0.018] | 1442 / 1304 | +0.020 [+0.014, +0.026] | 952 / 688 |
| AraRooPat v5 − AraRooPat warmup | Culture-MMLU | 13382 | +0.054 [+0.046, +0.063] | 1985 / 1261 | +0.050 [+0.041, +0.059] | 2277 / 1608 |
| BPE-16K v5 − BPE-16K warmup | Culture-MMLU | 13382 | +0.031 [+0.022, +0.039] | 1831 / 1421 | +0.011 [+0.003, +0.018] | 1337 / 1195 |

Alghafa by group (primary intersection, A − B, 95 % CI):

| A − B (Alghafa ∩) | fixed-label char Δ | fixed-label PMI Δ | choice-text char Δ | choice-text PMI Δ |
|---|---|---|---|---|
| AraRooPat v5 − native base | -0.043 [-0.050, -0.037] | -0.244 [-0.252, -0.235] | -0.093 [-0.111, -0.075] | -0.015 [-0.029, +0.001] |
| BPE-16K v5 − native base | -0.012 [-0.017, -0.006] | +0.038 [+0.031, +0.045] | -0.060 [-0.077, -0.043] | +0.043 [+0.028, +0.058] |
| native v5 − native base | +0.019 [+0.015, +0.023] | +0.030 [+0.026, +0.035] | -0.011 [-0.024, +0.001] | +0.007 [-0.003, +0.018] |
| AraRooPat v5 − BPE-16K v5 | -0.032 [-0.039, -0.025] | -0.282 [-0.290, -0.273] | -0.033 [-0.050, -0.016] | -0.057 [-0.072, -0.042] |
| AraRooPat v5 − native v5 | -0.062 [-0.069, -0.056] | -0.274 [-0.282, -0.266] | -0.082 [-0.098, -0.064] | -0.022 [-0.037, -0.007] |
| BPE-16K v5 − native v5 | -0.030 [-0.035, -0.025] | +0.007 [+0.001, +0.014] | -0.049 [-0.063, -0.034] | +0.035 [+0.021, +0.049] |
| native v5 − native SFT (1 024) † | +0.011 [+0.007, +0.014] | -0.014 [-0.018, -0.010] | +0.016 [+0.004, +0.028] | +0.003 [-0.006, +0.012] |
| AraRooPat v5 − AraRooPat warmup | +0.015 [+0.009, +0.020] | -0.062 [-0.067, -0.057] | +0.030 [+0.018, +0.044] | +0.016 [+0.006, +0.026] |
| BPE-16K v5 − BPE-16K warmup | +0.032 [+0.027, +0.036] | +0.013 [+0.010, +0.016] | -0.001 [-0.014, +0.011] | +0.023 [+0.013, +0.033] |

**Sensitivity — without the 1 024 cell** (`_mcq_compare/mcq_compare_4096.{md,json}`: Alghafa and Culture-MMLU keep all rows, Arabic-Exam 13 973):

| A − B (4096-only ∩) | task | n ∩ | char Δ [95 % CI] | PMI Δ [95 % CI] |
|---|---|---|---|---|
| AraRooPat v5 − native base | ACVA | 9000 | -0.051 [-0.058, -0.044] | -0.004 [-0.017, +0.008] |
| BPE-16K v5 − native base | ACVA | 9000 | -0.062 [-0.068, -0.055] | -0.175 [-0.187, -0.164] |
| native v5 − native base | ACVA | 9000 | -0.041 [-0.046, -0.036] | -0.199 [-0.209, -0.188] |
| AraRooPat v5 − BPE-16K v5 | ACVA | 9000 | +0.011 [+0.007, +0.014] | +0.171 [+0.157, +0.185] |
| AraRooPat v5 − native v5 | ACVA | 9000 | -0.010 [-0.015, -0.005] | +0.194 [+0.180, +0.209] |
| BPE-16K v5 − native v5 | ACVA | 9000 | -0.021 [-0.026, -0.016] | +0.023 [+0.016, +0.031] |
| AraRooPat v5 − AraRooPat warmup | ACVA | 9000 | +0.019 [+0.016, +0.022] | +0.183 [+0.169, +0.197] |
| BPE-16K v5 − BPE-16K warmup | ACVA | 9000 | +0.007 [+0.005, +0.010] | -0.191 [-0.205, -0.177] |
| AraRooPat warmup − BPE-16K warmup | ACVA | 9000 | -0.001 [-0.002, -0.000] | -0.204 [-0.218, -0.189] |
| AraRooPat v5 − native base | Alghafa | 22977 | -0.066 [-0.071, -0.059] | -0.173 [-0.180, -0.167] |
| BPE-16K v5 − native base | Alghafa | 22977 | -0.030 [-0.035, -0.025] | +0.035 [+0.030, +0.040] |
| native v5 − native base | Alghafa | 22977 | +0.005 [+0.002, +0.009] | +0.022 [+0.019, +0.026] |
| AraRooPat v5 − BPE-16K v5 | Alghafa | 22977 | -0.035 [-0.041, -0.029] | -0.208 [-0.215, -0.202] |
| AraRooPat v5 − native v5 | Alghafa | 22977 | -0.071 [-0.076, -0.065] | -0.196 [-0.202, -0.190] |
| BPE-16K v5 − native v5 | Alghafa | 22977 | -0.035 [-0.040, -0.031] | +0.013 [+0.008, +0.018] |
| AraRooPat v5 − AraRooPat warmup | Alghafa | 22977 | +0.021 [+0.016, +0.025] | -0.037 [-0.041, -0.033] |
| BPE-16K v5 − BPE-16K warmup | Alghafa | 22977 | +0.023 [+0.019, +0.027] | +0.015 [+0.012, +0.017] |
| AraRooPat warmup − BPE-16K warmup | Alghafa | 22977 | -0.034 [-0.039, -0.028] | -0.157 [-0.164, -0.150] |
| AraRooPat v5 − native base | Arabic-Exam | 13973 | -0.049 [-0.058, -0.039] | -0.037 [-0.046, -0.028] |
| BPE-16K v5 − native base | Arabic-Exam | 13973 | -0.002 [-0.010, +0.006] | -0.003 [-0.011, +0.005] |
| native v5 − native base | Arabic-Exam | 13973 | +0.022 [+0.015, +0.029] | +0.002 [-0.004, +0.008] |
| AraRooPat v5 − BPE-16K v5 | Arabic-Exam | 13973 | -0.046 [-0.054, -0.038] | -0.034 [-0.042, -0.026] |
| AraRooPat v5 − native v5 | Arabic-Exam | 13973 | -0.071 [-0.079, -0.062] | -0.039 [-0.047, -0.030] |
| BPE-16K v5 − native v5 | Arabic-Exam | 13973 | -0.024 [-0.031, -0.018] | -0.005 [-0.013, +0.003] |
| AraRooPat v5 − AraRooPat warmup | Arabic-Exam | 13973 | +0.067 [+0.059, +0.074] | +0.079 [+0.070, +0.088] |
| BPE-16K v5 − BPE-16K warmup | Arabic-Exam | 13973 | +0.041 [+0.033, +0.049] | +0.031 [+0.025, +0.038] |
| AraRooPat warmup − BPE-16K warmup | Arabic-Exam | 13973 | -0.072 [-0.081, -0.063] | -0.082 [-0.089, -0.074] |
| AraRooPat v5 − native base | Culture-MMLU | 14327 | -0.006 [-0.016, +0.003] | -0.035 [-0.044, -0.025] |
| BPE-16K v5 − native base | Culture-MMLU | 14327 | +0.020 [+0.011, +0.029] | +0.002 [-0.006, +0.009] |
| native v5 − native base | Culture-MMLU | 14327 | +0.038 [+0.031, +0.046] | +0.003 [-0.005, +0.010] |
| AraRooPat v5 − BPE-16K v5 | Culture-MMLU | 14327 | -0.026 [-0.034, -0.019] | -0.036 [-0.044, -0.028] |
| AraRooPat v5 − native v5 | Culture-MMLU | 14327 | -0.045 [-0.053, -0.037] | -0.037 [-0.046, -0.028] |
| BPE-16K v5 − native v5 | Culture-MMLU | 14327 | -0.018 [-0.025, -0.012] | -0.001 [-0.009, +0.006] |
| AraRooPat v5 − AraRooPat warmup | Culture-MMLU | 14327 | +0.053 [+0.045, +0.061] | +0.048 [+0.039, +0.057] |
| BPE-16K v5 − BPE-16K warmup | Culture-MMLU | 14327 | +0.029 [+0.021, +0.037] | +0.011 [+0.004, +0.018] |
| AraRooPat warmup − BPE-16K warmup | Culture-MMLU | 14327 | -0.050 [-0.059, -0.042] | -0.073 [-0.081, -0.066] |

No sign changes against the primary run on the letter-scored tasks; Alghafa's full 22 977 rows widen AraRooPat's char deficit on the choice-text group (−0.114 vs native base on 7 172 rows against −0.093 on 2 817).

**Two scoring artifacts the dumps exposed (neither fixed here — the brief kept the scorer's normalisations unchanged).** (a) *Label-prior collapse.* On a fixed-label group PMI subtracts each label's unconditioned log-likelihood under the bare `الإجابة:`; when one label's prior is several nats lower than the others', PMI hands it that many nats on every row. Under AraRooPat ` هو رأي ايجابي` — `ايجابي` spelled in six `[CHAR_*]` (§3.9) — has an unconditioned ll of −25.94 against −21.72 for ` هو رأي سلبي` (native −24.26 / −21.89): PMI picks it on 97 % / 99 % of the two rating sub-configs (14 000 rows), and AraRooPat v5's Alghafa PMI (0.428 ∩) is mostly that. BPE-16K collapses the same way on ` هي جملة ايجابية` (prior −30.98 against −25.05 / −25.91; 100 % of 1 725 rows). Char-norm collapses too, on ACVA: every cell picks ` خطأ` on 89–100 % of rows, so every ACVA char accuracy (0.404–0.475) sits on the ` خطأ` base rate (0.404). ACVA PMI splits the cells into collapsed (BPE v5 0.87, native v5 0.89, AraRooPat warmup 0.88, native SFT 0.77 on ` خطأ`) and not (AraRooPat v5, native base, BPE warmup: 0.58–0.63 on one label) — and the three uncollapsed cells score 0.666–0.686, the collapsed ones 0.471–0.560. **ACVA and Alghafa's fixed-label accuracies measure how a cell's label priors fall, not recognition; they are not read as tokenizer results below.** (b) *Exact ties from bf16.* The scorer applies `log_softmax` to the model's bf16 logits, so a single-token continuation's log-prob takes ~1 000–1 800 distinct values over ~50 000 scored choices; 1.8–5.3 % of the letter-task rows under native and BPE-16K end in an exact PMI tie, resolved by `np.argmax` to the first choice (accuracy on those rows 0.25–0.35). AraRooPat's letters are 3-token sums and never tie exactly. Measured effect: re-breaking the ties at random changes each cell's accuracy by −0.33 to +0.37 points (all five native / BPE-16K cells, both letter tasks) — below every gap read here, but a float32 `log_softmax` would remove it (and would change the native numbers, which is why it is not in this commit).

**Reading.**
*Ranking under log-likelihood scoring.* The free-form ranking (native ≈ AraRooPat > BPE-16K, §3.8) does **not** carry over. On the two letter-scored benchmarks — the ones neither artifact touches — AraRooPat v5 is last on every comparison: Arabic-Exam 0.549 / 0.555 against BPE-16K 0.595 / 0.589 and native v5 0.619 / 0.594 (AraRooPat − BPE −0.047 [−0.055, −0.038] char, −0.034 [−0.042, −0.025] PMI), Culture-MMLU 0.472 / 0.465 against 0.501 / 0.502 and 0.520 / 0.504 (−0.029 [−0.036, −0.021] / −0.037 [−0.046, −0.029]); BPE-16K and native v5 are not separable under PMI (Exam −0.005 [−0.012, +0.003], Culture −0.002 [−0.010, +0.006]) and native leads by 2 points under char. On Alghafa's choice-text group the order is the same (char 0.585 / 0.618 / 0.667 for AraRooPat / BPE / native v5). So on recognition the order is native ≥ BPE-16K > AraRooPat, by 3–5 points for AraRooPat, where generation had AraRooPat 0.30 judge points above BPE. *Phase 3 on teacher text did not cost recognition — it raised it.* AraRooPat v5 over its Phase 2 checkpoint: Arabic-Exam +0.066 [+0.058, +0.074] / +0.079 [+0.070, +0.087], Culture-MMLU +0.054 / +0.050; BPE-16K v5 over its: +0.041 / +0.031 and +0.031 / +0.011; native v5 over the untouched base: +0.022 / +0.002 and +0.038 / +0.003 (char up, PMI flat); native v5 over the pre-distillation SFT: +0.010 / +0.027 and +0.010 / +0.020. At the Phase 2 checkpoints AraRooPat trailed BPE-16K by 7–8 points on the letter tasks (Exam −0.072 / −0.082, Culture −0.050 / −0.073, 4096-only run); Phase 3 halved that. *Where native's advantage sits.* In the per-row answer text, not in the labels: native v5's lead over BPE-16K is char-norm on the letter tasks (+2 points) and on Alghafa's choice-text sub-configs (+0.049 [+0.034, +0.063]), and it vanishes under PMI on the letter tasks; the fixed-label groups are decided by the label priors of (a). *No near-uniform pathology.* Median decision margins are 0.62–4.82 in every cell and task (Charformer's was 5e-6); the near-tie shares (0–6.1 %) are the exact bf16 ties of (b) on single-token letters, not flat distributions. *What the intersection removed* — the 1 024 cell's truncated rows (19.3 % of Alghafa, 6.6 % of Culture-MMLU) and the long Arabic-Exam passages (4.3 %); the 4096-only run keeps them and moves no conclusion.

**Artifacts.** `outputs/experiments/qwen_native_vs_araroopat/<cell>/` for the six cells (`all_metrics.json`, `intrinsic_metrics.json`, `eval_rows/*.parquet` schema 2), `_mcq_compare/mcq_compare.{md,json}` (primary) and `mcq_compare_4096.{md,json}`, `_mcq_logs/` (chain script, `chain.log`, one console log per cell, `gpu_mem.csv` sampled every 15 s), `_audit/scorer_noop_check_2026-09-24{.py,.json,_run1.json,_<task>.parquet}`. Commits: `ff9f4b5` (§3.9), `2f56441` (scorer fix + schema 2 + tests), `2441518` (configs), `4d405ca` (`mcq_compare.py`), `7fb914b` (label-collapse diagnostic), and the documentation commit of this section.

**Verification (2026-09-25, Fable session, independent re-read of every number above against the dumps and `all_metrics.json`).** *Reproduced exactly:* the fixed scorer's window — with the scorer's own encode calls and `continuation_start`, the scored tokens equal each continuation's own encoding minus specials for native (` أ` → 1 token, ` هو رأي ايجابي` → 8), AraRooPat (3: `[LIT_BEGIN] [CHAR_أ] [LIT_END]`; 11) and BPE-16K (1; 4), so the defect of §3.9 is closed; the no-op check on `native_qwen3_sft` (`_audit/scorer_noop_check_2026-09-24.json`: 300 rows per task, 0 prompt mismatches, argmax agreement 100 % under char and PMI, max |Δ ll| 0.0, max |Δ score_pmi| 0.0 on scored and sentinel choices, scored / sentinel choices 600 / 0, 671 / 242, 982 / 83, 1 116 / 84); the six cells' all-rows accuracies, per-task walls, MEI and intrinsic RPS (BPE-16K 0.0498 v5 against 0.0476 warmup, native 0.0758 against the old cell's 0.0779 — the RPS irreproducibility of §5 item 25 is real); every dump at `schema_version` 2 and `max_length` 4 096 with `hit_cap` / `all_sentinel` / `near_tie` shares as tabulated and mean `cont_tokens` 2.00 / 9.39 / 3.00 / 3.00 for AraRooPat (ACVA / Alghafa / Arabic-Exam / Culture-MMLU), 1.00 / 4.59 / 0.98 / 1.00 for BPE-16K and 1.50 / 7.75 / 0.98 / 1.00 for native (0.98 over all choices, sentinel choices at 0 included; the table's 1.01 averages the scored choices); the six configs' resolved `evaluation`, `data`, `training` and `sweep.tasks` blocks identical, differing only in identity, checkpoint and tokenizer; the intersection sizes 9 000 / 18 542 / 13 840 / 13 382 and every intersection accuracy of every cell including the 1 024-token `native_qwen3_sft`; the pairs I recomputed from the dumps (AraRooPat v5 − BPE-16K v5, − native v5, BPE-16K v5 − native v5, each v5 − its Phase 2 checkpoint, native v5 − native base, on all four tasks): Δ and McNemar counts to the digit, my 2 000-resample intervals within 0.001 of the 10 000-resample ones; the Alghafa sub-config table; the label-collapse shares (mapped per row: AraRooPat's PMI picks ` هو رأي ايجابي` on 97.3 % of the 8 000 no-neutral rows and 99.4 % of the 6 000 rating rows, unconditioned log-probs −25.94 against −21.72; BPE-16K's PMI picks ` هي جملة ايجابية` on 100 % of the 1 725 sentence rows; ACVA's char-norm picks ` خطأ` on 97.8 / 98.9 / 94.5 / 89.5 % of rows for AraRooPat v5 / BPE-16K v5 / native v5 / native base against a 40.4 % gold share); the exact top-two ties (`decision_margin == 0`): 4.4 / 5.3 % of Arabic-Exam / Culture-MMLU rows for BPE-16K v5, 4.9 / 3.1 % native v5, 4.6 / 2.6 % native base, 0.0 % for AraRooPat whose letters are three-token sums; median decision margins 0.62–4.82; the 4096-only run's true/false numbers (0.875 / 0.950 AraRooPat v5, 0.963 / 0.950 BPE-16K v5, 0.963 / 0.925 native v5 on 80 rows, `_mcq_compare/mcq_compare_4096.json`); the seven commits `ff9f4b5` … `76a59f6`; 239 tests in the eight listed files, the per-file counts as reported. The GPU total (16 h 8 min) is consistent with Σ `inference_time_sec` = 15.8 h plus model loads and the intrinsic passes. *One reading error of my own, recorded so nobody repeats it:* mapping `pred_idx_pmi` through the first row's `continuations` order gave a 51 / 49 split on the AraRooPat rating rows; the order is per row and the per-row mapping gives the 97–99 % collapse the report states. *Re-run here from the committed `mcq_compare.py`* (`_mcq_compare/verif_fable_ara_nat_exam.json`): AraRooPat v5 vs native SFT v5 on Arabic-Exam with only those two cells in the intersection — 14 114 rows, 0.551 / 0.621 char and 0.557 / 0.596 PMI, McNemar 1 396 / 2 386 and 1 761 / 2 302 — reproduced. The console session reports that its new Eval-rows compare view reproduces these same numbers; that claim rests on its uncommitted code and is attributed, not re-run. *Reading.* The finding stands as written: with the window fixed and the same rows for every arm, recognition ranks native ≥ BPE-16K > AraRooPat on the letter-scored benchmarks and on Alghafa's choice-text group, the opposite of §3.8's generation order, and the deficit is not a scoring artifact — AraRooPat's letters are now scored as `[LIT_BEGIN] [CHAR_x] [LIT_END]` with the two shared markers contributing equally to every choice, and its Alghafa choice-text continuations are content words, not letters. The deficit was already present at the Phase 2 checkpoints (−7 to −8 points against BPE-16K's) and Phase 3 on teacher text narrowed it to −3 to −5; every arm gained recognition from that Phase 3, so the distillation did not trade recognition for generation. Two consequences for how the campaign's question is answered: the tokenizer's effect is **task-dependent** — the same three checkpoints rank one way when they write and another when they score short answers by log-likelihood — and neither eval alone settles "does AraRooPat help". What I would do next, in order and cheap first: a row-level analysis on the existing dumps of where AraRooPat loses to BPE-16K on the letter tasks (by prompt length in tokens, by the share of character-path and `<unk>` words in the question, by subject), which needs no GPU; the calibrated rule for the fixed-label groups of item 23 (centre each label's score by its benchmark-wide mean before the argmax — computable from the dumps); the float32 `log_softmax` of item 24 with its own no-op-style check; and the human check of item 15, which is still the missing calibration of the generation side.

**Few-shot handling and the first-option deficit (2026-09-25, Fable session, CPU, from the dumps; `_audit/fewshot_copy_diag_2026-09-25.{py,json}`, `_audit/letter_prior_diag_2026-09-25.{py,json}`).** *The question (the user's):* the v5 checkpoints trained on `arabic_squad_mcq` as single-question records — `format_mcq_context_letter_official` + the letter, one record per sequence, 6 750 drawn per arm (identical ids) at a uniform 25.2 / 24.9 / 24.8 / 25.1 % over أ / ب / ج / د, ≈ 4 800–5 250 of them seen before the restored step, 0.19–0.31 % of the Phase 3 loss tokens, distractors = random spans of the passage — while Arabic-Exam and Culture-MMLU are scored 3-shot: three solved questions of the same sub-config, each ending `الإجابة: <letter>`, blank lines, then the question. Nothing in Phase 1–3 had that shape, and the two Phase 2 checkpoints and the base saw no MCQ record at all. *Is the shape mishandled?* No cell shows the signature. On the intersection rows (13 973 / 14 327; three demos parsed on every row) the prediction equals the last demo's letter on 25.4–31.0 % of rows against the gold's own 27.3 / 25.6 % (chance 0.295 / 0.250), and lies among the three demo letters on 58–66 % against the gold's 59.9 / 58.3 %; the three v5 arms sit at the gold's rate (58.3–61.9 %), the native base and the Phase 2 checkpoints slightly above (61.7–66.5 %). On the 4-choice rows, per gold letter, accuracy with that letter among the demos against accuracy without it differs by −4 to +8 points for the v5 arms (AraRooPat on gold-أ rows 0.33 vs 0.30) and, for the base and the Phase 2 checkpoints, by +8 to +16 on the gold-أ rows and −1 to +10 on the other letters (pooled over letters the same effect reads −1.6 to +4.9 for the v5 arms and −0.6 to +8.7 for the base and Phase 2 cells, the AraRooPat warmup on Arabic-Exam at −0.6: the letter mix differs between the two groups, so the per-letter split is the comparable one) — the arms that trained on MCQ records lean on the demonstrations least, and AraRooPat no more than the others. *Where AraRooPat loses instead.* On the 4-choice rows (9 675 / 14 327) its char-norm accuracy by gold letter is أ **0.32 / 0.28** against ب 0.52 / 0.50, ج 0.56 / 0.58, د 0.62 / 0.49 (BPE-16K v5 on gold-أ rows 0.49 / 0.40, native v5 0.63 / 0.54). Paired over rows (`_audit/letter_A_split_2026-09-25.json`; 5 000 resamples, seed 0): AraRooPat v5 − BPE-16K v5 on the gold-أ rows **−0.174 [−0.193, −0.155]** (2 490 rows) / **−0.119 [−0.134, −0.104]** (3 289), on every other row **+0.000 [−0.011, +0.011]** (7 185) / **+0.001 [−0.007, +0.010]** (11 038); against native v5 −0.314 / −0.262 on the أ rows and **+0.017 [+0.006, +0.028]** / **+0.020 [+0.012, +0.029]** on the rest — where the answer is not the first option, AraRooPat equals BPE-16K and is slightly ahead of native v5. Under PMI the AraRooPat − BPE-16K split is the same (−0.231 / −0.144 on the أ rows, +0.023 / −0.004 elsewhere) and PMI costs native v5 its own أ rows (0.635 → 0.397), the over-correction of (a) acting on a letter. When أ is correct it ranks أ *last* on 27 / 24 % of rows (BPE 12 %, native 6 %; median top − أ gap 0.50 / 0.63 nats against 0.12 / 0.38 and 0.00 / 0.00) and its wrong picks spread evenly over ب / ج / د; it picks أ on 4–7 % of the rows where أ is wrong (BPE 10–16 %, native 16–22 %). This is not a constant letter prior: centring each position's score by its benchmark-wide mean before the argmax (the contextual calibration of Zhao et al. 2021, computed post hoc on the dumps) moves the v5 arms by ≤ 0.007 (AraRooPat 0.507 → 0.510 / 0.467 → 0.466, BPE-16K 0.551 → 0.548 / 0.493 → 0.493, native 0.575 → 0.582 / 0.512 → 0.516) while it lifts the native base and the two Phase 2 checkpoints by 3–5 points (their biases: the base picks أ on 53 % of rows, AraRooPat warmup ج on 53–60 %, BPE warmup أ on 48–54 %) — the uniform MCQ records of Phase 3 did calibrate every arm's letter prior, and the AraRooPat gap that remains is row-dependent. The encoding is symmetric: each letter is `[LIT_BEGIN] [CHAR_x] [LIT_END]` as a continuation and inside the prompt (`أ.` → `… [LIT_END] [PUNCT_.]`; `الإجابة: أ` in a demo ends the same way), `[CHAR_أ]` and `[CHAR_ا]` are distinct entries, `cont_tokens` is 3 for all four letters. Letter and position are confounded on these benchmarks (أ is always the first option), so the dumps cannot say whether the deficit belongs to the hamza letter or to the first slot — §5 item 26. *ACVA under PMI* (AraRooPat v5 0.666, native base 0.670, BPE-16K warmup 0.686 against 0.47–0.49 for native v5 / BPE-16K v5 / AraRooPat warmup) is a two-way argmax one prior offset away from flipping in every cell — the three "good" cells predict the majority label صح on 58–63 % of rows, the others خطأ on 87–89 %, the gold majority is 59.6 % — so it is the label-prior artifact of (a), not a binary-task strength of any tokenizer.

### 3.11 Letter or slot — the first-option deficit decomposed (2026-09-25)

**Why.** §3.10 *Few-shot handling* located AraRooPat v5's whole letter-task deficit on the rows whose answer is أ, but on Arabic-Exam and Culture-MMLU أ always labels the first option, so those dumps cannot say whether the deficit belongs to the *letter* (the hamza letter, its three-token spelling `[LIT_BEGIN] [CHAR_أ] [LIT_END]`, the `[LIT_END]` term after it) or to the *slot* (a first-position bias). Shuffling the option texts would not separate them either, since the letters would still sit on their slots. The manipulation is a **rotation of the letters over the slots** in the option lines, the demonstrations and the continuations together, so every letter is observed at every slot. A 0-shot pass answers whether the demonstrations carry it, and per-token log-probabilities show which of the three tokens does.

**The manipulation and the protocol** (commit `c05aa1f`). `label_rotation` k on the two letter-scored tasks: slot `i` is labelled `ARABIC_CHOICE_LETTERS[(i + k) % n]` (k = 1 on four choices: ب ج د أ), and the demonstrations follow because they are rendered through the same two hooks; the gold index stays the slot index. Rotation 0 is byte-identical to the official prompt on real rows (a golden fixture written by the pre-change code: 24 Arabic-Exam rows with 2 / 3 / 4 / 5 choices, with and without a passage, 9 Culture-MMLU rows, and 8 synthetic `arabic_squad_mcq` training records whose `_format_qa_full` text is unchanged: the training template never passes a rotation). `rows_file` scores a listed subset while the task still caches the **full** list, so each row's demonstrations (pool = its sub-config in the full list, seed = `seed + full-list index`) and prompt are the full run's, and the dump's `row_index` is the full-list index. The row dump gains `cont_token_ll` (schema 3): the addends of each choice's `ll`, in order. Subsets (`scripts/select_mcq_subset.py`, commit `c05aa1f`; files commit `0f37bae`): per task, the rows present in the dumps of the four source cells (AraRooPat v5, BPE-16K v5, native v5, native base — all `*_mcq4096`), untruncated in every cell (`mcq_compare.py`'s rule) and with four choices — **9 675** Arabic-Exam and **14 327** Culture-MMLU candidates — then 1 000 drawn with `default_rng(0)`. Gold letter أ / ب / ج / د: Arabic-Exam 259 / 226 / 204 / 311 (25.9 / 22.6 / 20.4 / 31.1 %), Culture-MMLU 232 / 244 / 251 / 273 (23.2 / 24.4 / 25.1 / 27.3 %); `rows_arabic_exam.json` sha256 `fca4a176c6b482e71b81c4cfa5601d97150ecf7124af1f7608d719aa18423026`, `rows_culture_arabic_mmlu.json` `885d5bdbbc23f22caf5cf60edcc4c9c2e61aba5000e00e6692464570613f2e0d`. Twenty cells (`configs/experiments/mcq_letter_slot/<arm>_<cond>.yaml`, outputs `_letter_slot/<cell>/`): the four arms × {rot0, rot1, rot2, rot3 (3-shot, `label_rotation` k), 0shot (rotation 0)}, each mirroring its `mcq_v5` config (tokenizer block, checkpoint), both tasks at `max_length 4096`, char + PMI, intrinsic metrics off; the twenty resolved configs differ only in identity, tokenizer, checkpoint, `label_rotation` and `num_fewshot`.

**Checks.** Before any GPU time, a CPU pre-check rebuilt the full dumps' 3-shot prompts and continuations from the new code under `rows_file` + rotation 0 on 1 000 / 1 000 rows of each task with 0 mismatches, and verified for k = 1..3 that every demonstration's answer letter is the rotated letter of its gold slot (`_letter_slot/_audit/prompt_precheck.{py,json}`). **Replication check** (P1a, `_audit/replication_check.{py,json}`): each rot0 dump joined to its full `*_mcq4096` dump by `row_index`, 8 cell × task pairs — 1 000 rows each, 4 000 scored choices each, 0 sentinel choices; 0 prompt, continuation, `cont_tokens`, gold and argmax (char, PMI) mismatches; max |Δ| **0.0** as float32 on `ll`, `score_char`, `score_pmi` and `uncond_ll`; max |Σ `cont_token_ll` − `ll`| 2.4e-7 (AraRooPat on Culture-MMLU; 0.0 elsewhere). Every one of the 40 cell × task dumps: 1 000 rows, `hit_cap` / `all_sentinel` 0. A rotation leaves the prompt length unchanged on every Culture-MMLU row and on 983 of the 1 000 Arabic-Exam rows; 17 Arabic-Exam rows move by ≤ 2 tokens (≤ 4 under BPE-16K) in every arm, most likely through a 5- or 3-choice demonstration whose rotated answer letter changes its token count (the eval rows are 4-choice, their demonstrations are drawn from the whole sub-config); the longest prompt is 4 050 tokens (BPE-16K), under the cap. The console's Eval-rows browser opens a schema-3 dump (`describe`, `query`, `row`, CSV export; the new column is not projected).

| cell | Arabic-Exam char / PMI | Culture-MMLU char / PMI | wall |
|---|---|---|---|
| `araroopat_v5_rot0` / `rot1` / `rot2` / `rot3` | 0.514 / 0.536 · 0.456 / 0.418 · 0.491 / 0.412 · 0.491 / 0.462 | 0.479 / 0.466 · 0.455 / 0.405 · 0.419 / 0.398 · 0.443 / 0.405 | 10.8 · 11.4 · 11.0 · 10.9 min |
| `bpe16k_v5_rot0..3` | 0.551 / 0.548 · 0.512 / 0.495 · 0.490 / 0.480 · 0.516 / 0.492 | 0.507 / 0.494 · 0.496 / 0.499 · 0.475 / 0.466 · 0.493 / 0.488 | 9.7 · 9.4 · 9.5 · 9.4 min |
| `native_v5_rot0..3` | 0.576 / 0.549 · 0.546 / 0.524 · 0.537 / 0.551 · 0.562 / 0.560 | 0.528 / 0.501 · 0.511 / 0.493 · 0.505 / 0.497 · 0.501 / 0.489 | 10.0 · 9.9 · 9.9 · 9.9 min |
| `native_base_rot0..3` | 0.552 / 0.560 · 0.522 / 0.544 · 0.531 / 0.526 · 0.521 / 0.528 | 0.475 / 0.499 · 0.492 / 0.474 · 0.497 / 0.464 · 0.485 / 0.488 | 9.8 · 9.8 · 10.1 · 9.8 min |
| `<arm>_0shot` (AraRooPat / BPE-16K / native v5 / base) | 0.490 / 0.511 · 0.519 / 0.520 · 0.571 / 0.538 · 0.551 / 0.553 | 0.438 / 0.450 · 0.483 / 0.492 · 0.509 / 0.494 · 0.429 / 0.485 | 9.8 · 9.2 · 9.6 · 9.8 min |

**Analysis** (`scripts/mcq_letter_slot.py`, commit `07c559b`; `_letter_slot/letter_slot.{md,json}`). Rows are the unit and a row's four rotations one cluster; every interval is a cluster bootstrap over rows (5 000 resamples, `default_rng(0)`), one resample matrix per task for every arm and pair, so pair intervals are paired. `E_letter` = acc(gold letter ≠ أ) − acc(gold letter = أ) is a **within-row** contrast (the rotation moves the letter, not the option); `E_slot` = acc(gold slot ≠ 1) − acc(gold slot = 1) is a **between-row** contrast (the rows whose gold is the first option are the same rows at every rotation), so it also carries whatever makes those rows harder. Char-norm unless stated; 4 000 row-evaluations per arm and task.

*1. Letter × slot (per arm; char, PMI in brackets):*

| arm | acc | by gold letter أ / ب / ج / د | by gold slot 1 / 2 / 3 / 4 | E_letter | E_slot | interaction (أ at slot 1) |
|---|---|---|---|---|---|---|
| Arabic-Exam AraRooPat v5 | 0.488 (0.457) | 0.390 / 0.516 / 0.455 / 0.591 | 0.342 / 0.460 / 0.542 / 0.595 | **+0.131 [+0.108, +0.154]** (+0.088 [+0.063, +0.113]) | +0.197 [+0.142, +0.253] (+0.102 [+0.059, +0.145]) | +0.069 [+0.042, +0.095] |
| BPE-16K v5 | 0.517 (0.504) | 0.592 / 0.488 / 0.472 / 0.517 | 0.369 / 0.511 / 0.512 / 0.649 | −0.100 [−0.123, −0.077] (−0.248) | +0.200 [+0.143, +0.257] (+0.204) | +0.035 [+0.007, +0.065] |
| native v5 | 0.555 (0.546) | 0.703 / 0.507 / 0.566 / 0.445 | 0.413 / 0.605 / 0.536 / 0.650 | −0.197 [−0.222, −0.171] (+0.160) | +0.192 [+0.134, +0.247] (+0.134) | +0.049 [+0.015, +0.082] |
| native base | 0.531 (0.539) | 0.766 / 0.415 / 0.529 / 0.416 | 0.477 / 0.512 / 0.466 / 0.634 | −0.313 [−0.341, −0.285] (+0.163) | +0.074 [+0.021, +0.125] (+0.016) | +0.099 [+0.065, +0.135] |
| Culture-MMLU AraRooPat v5 | 0.449 (0.418) | 0.344 / 0.480 / 0.453 / 0.519 | 0.341 / 0.454 / 0.499 / 0.491 | **+0.140 [+0.116, +0.165]** (+0.079 [+0.053, +0.106]) | +0.141 [+0.080, +0.199] (+0.100 [+0.053, +0.145]) | +0.083 [+0.056, +0.111] |
| BPE-16K v5 | 0.493 (0.487) | 0.493 / 0.510 / 0.460 / 0.508 | 0.392 / 0.526 / 0.500 / 0.542 | −0.000 [−0.022, +0.022] (−0.144) | +0.131 [+0.068, +0.194] (+0.147) | +0.034 [+0.006, +0.063] |
| native v5 | 0.511 (0.495) | 0.591 / 0.520 / 0.490 / 0.444 | 0.448 / 0.572 / 0.513 / 0.509 | −0.106 [−0.130, −0.082] (+0.197) | +0.082 [+0.023, +0.142] (+0.031) | +0.071 [+0.038, +0.105] |
| native base | 0.487 (0.481) | 0.700 / 0.425 / 0.382 / 0.442 | 0.518 / 0.473 / 0.448 / 0.509 | −0.284 [−0.310, −0.256] (+0.202) | −0.040 [−0.093, +0.013] (−0.104) | +0.097 [+0.061, +0.132] |

AraRooPat v5's 4 × 4 table (Arabic-Exam; rows gold letter, columns gold slot 1..4; each cell 204–311 evaluations): أ 0.313 / 0.332 / 0.397 / 0.492; ب 0.313 / 0.544 / 0.500 / 0.675; ج 0.251 / 0.367 / 0.588 / 0.601; د 0.490 / 0.597 / 0.681 / 0.611 (Culture-MMLU: أ 0.319 / 0.332 / 0.311 / 0.407; ب 0.345 / 0.525 / 0.486 / 0.549; ج 0.272 / 0.434 / 0.570 / 0.516; د 0.427 / 0.525 / 0.629 / 0.491). All twenty-four 4 × 4 tables are in `letter_slot.md`.

*2. Arm pairs, paired over row × rotation* (char; PMI in brackets):

| A − B | task | Δ | gold letter أ | letter ≠ أ | gold slot 1 | slot ≠ 1 | G_letter | G_slot |
|---|---|---|---|---|---|---|---|---|
| AraRooPat v5 − BPE-16K v5 | Exam | −0.029 [−0.051, −0.008] | **−0.202 [−0.236, −0.169]** | +0.028 [+0.006, +0.051] | −0.027 [−0.065, +0.011] | −0.030 [−0.057, −0.004] | **+0.230 [+0.199, +0.262]** (+0.336) | **−0.003 [−0.049, +0.043]** (−0.102) |
| | Culture | −0.044 [−0.066, −0.022] | **−0.149 [−0.180, −0.117]** | −0.009 [−0.032, +0.014] | −0.052 [−0.094, −0.010] | −0.041 [−0.066, −0.016] | **+0.140 [+0.111, +0.171]** (+0.224) | **+0.010 [−0.038, +0.060]** (−0.047) |
| AraRooPat v5 − native v5 | Exam | −0.067 [−0.088, −0.046] | −0.313 [−0.346, −0.282] | +0.015 [−0.008, +0.037] | −0.071 [−0.107, −0.037] | −0.066 [−0.093, −0.040] | +0.328 [+0.296, +0.360] (−0.072) | +0.006 [−0.038, +0.050] (−0.032) |
| | Culture | −0.062 [−0.083, −0.042] | −0.247 [−0.279, −0.215] | −0.001 [−0.022, +0.021] | −0.108 [−0.149, −0.069] | −0.049 [−0.072, −0.025] | +0.246 [+0.214, +0.278] (−0.118) | +0.059 [+0.013, +0.107] (+0.069) |
| BPE-16K v5 − native v5 | Exam | −0.038 [−0.056, −0.021] | −0.111 [−0.138, −0.084] | −0.014 [−0.033, +0.005] | −0.044 [−0.072, −0.018] | −0.036 [−0.057, −0.014] | +0.097 [+0.069, +0.125] | +0.009 [−0.025, +0.044] |
| | Culture | −0.018 [−0.037, −0.001] | −0.098 [−0.125, −0.071] | +0.008 [−0.011, +0.027] | −0.056 [−0.090, −0.023] | −0.007 [−0.028, +0.014] | +0.106 [+0.078, +0.133] | +0.049 [+0.010, +0.088] |
| AraRooPat v5 − native base | Exam / Culture | −0.043 / −0.038 | −0.376 / −0.356 | +0.067 / +0.068 | −0.135 / −0.178 | −0.011 / +0.004 | +0.443 / +0.424 | +0.124 / +0.182 |
| BPE-16K v5 − native base | Exam / Culture | −0.014 / +0.005 | −0.174 / −0.207 | +0.039 / +0.076 | −0.108 / −0.126 | +0.019 / +0.045 | +0.213 / +0.283 | +0.127 / +0.171 |
| native v5 − native base | Exam / Culture | +0.024 / +0.024 | −0.063 / −0.109 | +0.053 / +0.068 | −0.064 / −0.070 | +0.054 / +0.052 | +0.116 / +0.177 | +0.118 / +0.122 |

`G_letter` = (A − B on letter ≠ أ) − (A − B on letter أ) and `G_slot` likewise for slot 1; in a pair, row content cancels, so these are the contrasts that belong to the arm difference. By gold letter, AraRooPat − BPE-16K is −0.202 / +0.028 / −0.017 / +0.074 on Arabic-Exam (أ ب ج د) and −0.149 / −0.030 / −0.007 / +0.011 on Culture-MMLU; by gold slot −0.027 / −0.051 / +0.029 / −0.054 and −0.052 / −0.072 / −0.001 / −0.051 — the gap moves with أ and is flat over the slots. Under PMI the pairs against native v5 flip on أ (native's PMI over-corrects its own أ prior: native v5 by letter under PMI 0.426 / 0.670 / 0.726 / 0.362), the §3.10 (a) over-correction acting on a letter.

*3. 0-shot against 3-shot rotation 0 (same 1 000 rows):* per arm, 0-shot − 3-shot (char): AraRooPat −0.024 [−0.048, +0.000] / −0.041 [−0.065, −0.017] (Exam / Culture), BPE-16K −0.032 [−0.056, −0.009] / −0.024 [−0.049, +0.002], native v5 −0.005 [−0.026, +0.016] / −0.019 [−0.040, +0.002], native base −0.001 [−0.024, +0.022] / −0.046 [−0.068, −0.023]. The AraRooPat − BPE-16K gap on the gold-أ rows: Exam **−0.120 [−0.173, −0.068]** at 0-shot against −0.166 [−0.228, −0.108] at 3-shot (change +0.046 [−0.008, +0.103]); Culture **−0.099 [−0.157, −0.041]** against −0.108 [−0.169, −0.047] (change +0.009 [−0.059, +0.082]); on the other rows +0.003 / −0.029 at 0-shot. Under PMI: Exam −0.089 [−0.150, −0.027] against −0.181 (change +0.093 [+0.028, +0.160]); Culture −0.043 [−0.108, +0.019] against −0.078 (change +0.034 [−0.037, +0.106]).

*4. Per token.* AraRooPat's two markers carry nothing: mean log P of `[LIT_BEGIN]` −0.0016 to −0.0022 and of `[LIT_END]` −0.0002 to −0.0011 for every letter, gold or not, on both tasks; the whole letter score is `[CHAR_x]` (Arabic-Exam gold / non-gold: أ −1.260 / −2.041, ب −1.111 / −1.957, ج −1.314 / −2.280, د −1.005 / −1.677). Scoring (i) — the sum — reproduces the dump's char-norm decision on every evaluation (0 mismatches of 4 000 per task, from the dump's `ll` and from Σ of the float32 terms); scoring (ii) — `[CHAR_x]` alone — and (iii) — `[CHAR_x]` + `[LIT_END]` — give the same effects: E_letter +0.131 [+0.108, +0.155] / +0.150 [+0.125, +0.174] under (ii), +0.131 / +0.140 under (iii); the AraRooPat − BPE-16K gap on gold أ is −0.197 under (ii) against −0.202 under (i). By slot, every arm puts less mass on whichever letter labels slot 1 and more on slot 4 (AraRooPat's `[CHAR_أ]` over all choices: −2.23 / −1.92 / −1.80 / −1.43 at slots 1–4; BPE-16K's ` أ` −2.18 / −1.93 / −1.85 / −1.41; native v5's −2.20 / −1.86 / −1.94 / −1.49, Arabic-Exam) — at slot 1 the three arms put about the same mass on أ. What differs is **which letter each arm favours**: the native base (strongly), native v5 and BPE-16K favour أ; AraRooPat v5 favours د (`[CHAR_د]` is its most probable letter at slots 1–3: −1.53 / −1.47 / −1.41 against −1.94 to −2.36 for the others at slot 1) and ranks أ below ب and د.

*5. Predictions per rotation* (char; share of rows predicting slot 1 / letter أ / letter د, gold slot-1 share 0.26 Exam, 0.23 Culture). AraRooPat v5, Exam: slot 1 0.13 / 0.15 / 0.10 / 0.30 at rotations 0–3 (at rotation 3 slot 1 carries د), letter أ 0.13 / 0.31 / 0.15 / 0.12, letter د 0.31 / 0.41 / 0.33 / 0.30 — its under-selection stays at slot 1 **unless** slot 1 carries د, and it predicts أ at the gold rate only when أ sits at slot 4. BPE-16K: slot 1 0.23 / 0.17 / 0.11 / 0.19, letter أ 0.23 / 0.44 / 0.30 / 0.28; native v5: slot 1 0.30 / 0.13 / 0.15 / 0.15, letter أ 0.30 / 0.45 / 0.35 / 0.37; native base: letter أ 0.51 / 0.55 / 0.37 / 0.38. Under PMI AraRooPat picks the letter د on 43–63 % of rows at every rotation (Culture 0.43 / 0.63 / 0.62 / 0.54).

*6. Proposal diagnostic — the rotation-averaged decision.* Each option's score averaged over its four letters before the argmax (every letter's prior reaches every option once; char and PMI then coincide exactly, the unconditioned letter prior cancels). Computed from these dumps, **applied to nothing**: Arabic-Exam AraRooPat v5 0.522, BPE-16K 0.523, native v5 0.577, base 0.576 (official rot0 on the same rows: 0.514 / 0.551 / 0.576 / 0.552 char); AraRooPat − BPE-16K **−0.001 [−0.030, +0.027]**, − native v5 −0.055 [−0.086, −0.026], BPE-16K − native v5 −0.054 [−0.080, −0.028]. Culture-MMLU 0.469 / 0.508 / 0.519 / 0.527 (rot0 0.479 / 0.507 / 0.528 / 0.475); AraRooPat − BPE-16K **−0.039 [−0.068, −0.011]**, − native v5 −0.050 [−0.078, −0.022], BPE-16K − native v5 −0.011 [−0.035, +0.013].

**Reading under the pre-registered rules** (applied as written to AraRooPat v5's per-arm effects). *Slot*: does not fire on either task or norm. *Letter*: does not fire. **Both / interaction fires on both tasks under both norms** — E_letter +0.131 / +0.140 and E_slot +0.197 / +0.141 (char), every CI above 0, interaction +0.069 [+0.042, +0.095] / +0.083 [+0.056, +0.111]: أ at slot 1 is *less* bad than the two effects added would predict (0.313 against an additive 0.244 on Arabic-Exam). *Terminal token*: **does not fire** — under scoring (ii) AraRooPat's E_letter is +0.131 / +0.150, outside BPE-16K's E_letter interval [−0.123, −0.077] / [−0.022, +0.022]; the `[LIT_END]` term is ≈ 0.001 nats and the letter's score is `[CHAR_x]`. *0-shot*: under char-norm the gold-أ gap persists within its interval on both tasks and the 0-shot gap excludes 0 — **the demonstrations are not the carrier** (the user's question is closed); under PMI it shrinks on Arabic-Exam (change +0.093, CI above 0, the 0-shot gap still below 0) and is inconclusive on Culture-MMLU (both intervals include 0). *What the per-arm rule cannot separate, the pairs do* (a supplementary reading, not pre-registered): E_slot is **shared by every trained arm** (BPE-16K +0.200 / +0.131, native v5 +0.192 / +0.082) — a property of the rows whose gold is the first option, or of a first-position aversion every Phase-3 model acquired (the untouched base: +0.074 / −0.040) — while E_letter **differs in sign**: AraRooPat disfavours أ, the three others favour it. In the arm pairs, where row content cancels, the AraRooPat-specific part is a letter effect: AraRooPat − BPE-16K G_letter +0.230 [+0.199, +0.262] / +0.140 [+0.111, +0.171] with G_slot −0.003 [−0.049, +0.043] / +0.010 [−0.038, +0.060]; against native v5 G_letter +0.328 / +0.246 with G_slot +0.006 / +0.059. **The mechanism, as far as these dumps show it:** every arm under-selects slot 1; the arms differ in their letter prior; the official prompt always puts أ on slot 1, so an أ-favouring prior (native, BPE-16K — the native base picks أ on 51–54 % of rot0 rows) compensates for the shared slot-1 aversion and a د-favouring one (AraRooPat) compounds it. It is not the three-token spelling, not the terminal token, not the demonstrations. Where AraRooPat v5's د preference comes from is not measured here (§5 item 26).

**What this does to §3.10's headline.** On these rows the official prompt gives AraRooPat v5 − BPE-16K v5 −0.037 (Arabic-Exam) and −0.028 (Culture-MMLU). Over the four rotations the gap is −0.029 [−0.051, −0.008] / −0.044 [−0.066, −0.022], and on the evaluations whose gold letter is not أ it is +0.028 [+0.006, +0.051] / −0.009 [−0.032, +0.014]. Under a letter-invariant decision (diagnostic 6) it is −0.001 [−0.030, +0.027] on Arabic-Exam and −0.039 [−0.068, −0.011] on Culture-MMLU, and AraRooPat trails native v5 on both (−0.055 / −0.050). So the recognition order native ≥ BPE-16K ≥ AraRooPat survives a letter-invariant rule. What does not survive is the Arabic-Exam gap to BPE-16K: there it is the letter-prior interaction with the official prompt. On Culture-MMLU a gap to BPE-16K of about four points remains.

**Proposal (applied to nothing; a rule change applies to every arm as one rule).** Score letter-labelled MCQ by **cyclic-permutation averaging**: for an n-choice row, score it under the n cyclic letter rotations (prompt, demonstrations and continuations rotated together — `label_rotation` does exactly this) and average each option's log-likelihood over the rotations before the argmax. It is letter-invariant by construction, it makes char-norm and PMI coincide, and it costs n forward passes per row (×4 on these benchmarks) — the permutation-debiasing family of Zheng et al. (2024). Whether to adopt it is a separate decision; if adopted, the §3.10 letter-task numbers would be re-read under it for every arm.

**Alghafa: the MSA scope rule (2026-09-25, a scope decision of the user).** The campaign is about MSA: the tokenizer corpus, the pretraining pool (its dialect-marker gate drops dialect documents) and CAMeL's database are MSA by construction, so Alghafa's `meta_ar_dialects` sub-config evaluates text every from-scratch arm was deliberately never trained on, while the native model saw dialect in its own pretraining. **The rule**, identical for every arm: the in-scope Alghafa number is the **MSA choice-text group** — `mcq_exams_test_ar` (562), `meta_ar_msa` (900) and the two grounded-statement sub-configs (155 each), 1 772 rows; `meta_ar_dialects` is out of the headline scope, its numbers kept and reported below; the four fixed-label sub-configs stay flagged as the label-prior artifact of §3.10 (a). In `mcq_compare.py` the groups are `choice_text_msa` and `choice_text_dialect` (`ALGHAFA_CHOICE_TEXT_SCOPE`, commit `07c559b`); `_mcq_compare/mcq_compare_msa.{md,json}` re-runs the six §3.10 cells with the six-cell pair list (every Alghafa row is untruncated at 4 096, so the intersection is all 22 977 rows). **MSA choice-text group** (char / PMI): AraRooPat v5 0.666 / 0.381, BPE-16K v5 0.688 / 0.439, native v5 0.718 / 0.392, native base 0.734 / 0.393, AraRooPat Phase 2 0.643 / 0.366, BPE-16K Phase 2 0.682 / 0.411. Pairs: AraRooPat v5 − BPE-16K v5 **−0.021 [−0.042, −0.002]** char (McNemar 150 / 188, p 0.044), −0.058 [−0.078, −0.037] PMI (132 / 234); AraRooPat v5 − native v5 −0.052 [−0.070, −0.033] char (104 / 196), **−0.011 [−0.030, +0.009]** PMI (147 / 166, p 0.31); BPE-16K v5 − native v5 −0.030 [−0.048, −0.013] char, +0.047 [+0.029, +0.066] PMI; AraRooPat v5 − native base −0.067 / −0.012 [−0.032, +0.008]; each v5 − its Phase 2 checkpoint +0.023 [+0.007, +0.038] (AraRooPat) and +0.006 [−0.009, +0.021] (BPE-16K) char; AraRooPat − BPE-16K at Phase 2 −0.039 [−0.061, −0.018]. **Dialect sub-config (out of scope, 5 400 rows):** AraRooPat v5 0.549 / 0.326, BPE-16K v5 0.599 / 0.374, native v5 0.651 / 0.358, base 0.679 / 0.350; AraRooPat − BPE-16K −0.049 [−0.062, −0.036] / −0.048 [−0.058, −0.038], − native v5 −0.102 [−0.115, −0.089] / −0.032 [−0.041, −0.022]. **Found while measuring it:** 900 of `meta_ar_dialects`' 5 400 rows (a contiguous block, `row_index` 2357–3256) are identical in question, choices and gold to the 900 `meta_ar_msa` rows (Belebele's layout: each item in MSA plus five dialects), so the "dialect" sub-config is 4 500 dialect rows and 900 MSA duplicates (`_letter_slot/alghafa_dialect_split.json`, `_audit/alghafa_dialect_split.py`). On the 4 500 dialect rows proper: AraRooPat v5 0.513 / 0.313, BPE-16K v5 0.566 / 0.356, native v5 0.621 / 0.346, base 0.650 / 0.338; AraRooPat − BPE-16K **−0.052 [−0.067, −0.038]** char, −0.043 [−0.054, −0.032] PMI; on `meta_ar_msa` alone −0.017 [−0.042, +0.009] char. The duplicates are already in scope through `meta_ar_msa`, so the rule is unaffected. **Query paths under AraRooPat v3** (`_letter_slot/alghafa_query_paths.json`; the `question` field of each row — the passage and question, no demonstrations — every Arabic word occurrence encoded alone and classified by its first non-clitic token, the classifier of `_audit/tok_audit_2026-09-24.py`): `meta_ar_msa` (72 146 occurrences) **character path 6.0 %**, `<unk>` 0.0 %, closed class 23.4 % (PREP 12.5, FUNC 10.5, clitic-only 0.4), ROOT+PAT 68.2 %, PROP 2.4 %, 2.89 tokens per word; `meta_ar_dialects` dialect rows proper (349 907) **character path 13.7 %**, `<unk>` 0.0 %, closed class 16.1 %, ROOT+PAT 67.3 %, PROP 2.9 %, 3.36 tokens per word (all 5 400 rows: 12.4 / 0.0 / 17.4 / 67.4 %). **Reading.** The in-scope Alghafa number is the MSA choice-text group. There AraRooPat v5 trails BPE-16K v5 by two points under char-norm; the interval's upper end is −0.002 in this run and −0.001 in the amendment's, so it touches zero and the separation is marginal. Under PMI the gap to BPE-16K is −0.058, clearly separable. Against native v5 the char gap is −0.052, separable; under PMI it is −0.011 [−0.030, +0.009], not separable. The larger Alghafa losses sat on the dialect rows, where AraRooPat trails BPE-16K by five points and native v5 by ten. That is a property of an MSA-only morphological vocabulary: dialect words fall to the character path (13.7 % of question words against 6.0 % on the same passages' MSA versions), and dialect function words (`اللي`, `راح`) find no closed-class token (16.1 % against 23.4 %) while ROOT+PAT coverage barely moves. It is a stated limitation, out of scope, to be revisited only if dialect enters the scope (§5 item 20).

**GPU.** Twenty cells, 199.9 min end to end (9.2–11.4 min each, of which ~1.6 + 2.5 min are the HF dataset loads), peak 26.4 GiB (native cells); the CPU pre-check 15.3 min (dataset loads), the analysis 8 s, `mcq_compare_msa` ~2 min, the query paths 42 s.

**Artifacts.** `outputs/experiments/qwen_native_vs_araroopat/_letter_slot/`: the twenty cells (`all_metrics.json`, `eval_rows/{arabic_exam,culture_arabic_mmlu}.parquet`, schema 3), `letter_slot.{md,json}`, `alghafa_query_paths.json`, `alghafa_dialect_split.json`, `_audit/{prompt_precheck,replication_check,alghafa_query_paths,alghafa_dialect_split}.py` (+ `.json` for the first two), `_logs/` (`make_configs.py`, `run_chain.sh`, `chain.log`, one console log per cell, `gpu_mem.csv`); `_mcq_compare/mcq_compare_msa.{md,json}`. Configs `configs/experiments/mcq_letter_slot/` (twenty + the two rows files). Commits: `73bf45b` (the pending §3.10 edits), `c05aa1f` (`label_rotation`, `rows_file`, schema 3, tests, presets, `select_mcq_subset.py`), `0f37bae` (rows files + configs), `07c559b` (`mcq_letter_slot.py`, the MSA groups in `mcq_compare.py`, tests), `84eae66` (console test), and the documentation commit of this section.

**Verification (2026-09-25, Fable session, independent re-read of every number above against `_letter_slot/`, `_mcq_compare/mcq_compare_msa.json` and the repository).** *Reproduced exactly:* the six commits `73bf45b` … `19eb207`, with the console session's six files still uncommitted and their hunks intact (`CLAUDE.md` two hunks); the two rows files (sha256 `fca4a176…` / `885d5bdb…`, 1 000 sorted unique indices each, candidates 9 675 / 14 327, gold counts 259 / 226 / 204 / 311 and 232 / 244 / 251 / 273); the twenty cells (1 000 rows per task, `hit_cap` = `all_sentinel` = 0, schema 3, `label_rotation` and `num_fewshot` in every dump's metadata) and every accuracy of the cell table; the replication check on all eight rotation-0 pairs (1 000 / 1 000 joined, 0 prompt / continuation / `cont_tokens` / gold mismatches, argmax 100 % under char and PMI, max |Δ| 0.0 in float32 on `ll`, `score_char`, `score_pmi` and `uncond_ll`, 0 sentinel choices, max |Σ `cont_token_ll` − `ll`| 0.0 — the 2.4e-7 reported for AraRooPat on Culture-MMLU is float32 rounding of the same sums); the prompt length under rotation (17 Arabic-Exam rows move by ≤ 2 tokens, ≤ 4 under BPE-16K; none on Culture-MMLU; maximum 4 050); every `E_letter` / `E_slot` / interaction of the letter × slot table under char and PMI, with my own cluster bootstrap (5 000 resamples, seed 0) matching the intervals to the third decimal, the AraRooPat marginals and 4 × 4 table; every entry of the pairs table including `G_letter` / `G_slot`, the overall deltas and the PMI contrasts (+0.336 / +0.224, −0.102 / −0.047); the rotation-averaged decision (AraRooPat − BPE-16K −0.001 [−0.030, +0.027] on Arabic-Exam, −0.039 [−0.068, −0.011] on Culture-MMLU, −0.055 / −0.050 against native v5, the official prompt's −0.037 on the same rows); the 0-shot table, AraRooPat's per-letter 0-shot / 3-shot accuracies, the gold-أ gaps at 0-shot (−0.120 / −0.099 char, −0.089 / −0.043 PMI) and their changes, interval edges within 0.004 (bootstrap noise); the per-token means (`[LIT_BEGIN]` −0.0021, `[LIT_END]` −0.0004 to −0.0010, every `[CHAR_x]` gold / non-gold mean, `[CHAR_أ]` by slot −2.23 / −1.92 / −1.80 / −1.43, د the most probable letter at slots 1–3, the one-token arms' means, أ at slot 1 −2.23 / −2.18 / −2.20); the three diagnostic scorings (0 argmax mismatches under (i); `E_letter` +0.131 / +0.131 / +0.131 and +0.140 / +0.150 / +0.140; `E_slot` +0.197 / +0.158 / +0.197 and +0.141 / +0.114 / +0.141; AraRooPat under (ii) − BPE-16K on the gold-أ rows −0.197 / −0.157); the predicted slot-1 and letter shares per rotation and AraRooPat's PMI د share (0.49–0.63 on Arabic-Exam, 0.43–0.63 on Culture-MMLU); for the Alghafa scope rule every group accuracy, every pair with interval and McNemar counts in `mcq_compare_msa.json` (150 / 188, p 0.044; 132 / 234; 147 / 166, p 0.31), the fixed-label numbers, the 900 duplicates (an independent match of question + choices + gold on the base dump: 900 rows, `row_index` 2357–3256, contiguous), the dialect-proper pairs (−0.052 [−0.067, −0.038] / −0.043; `meta_ar_msa` alone −0.017 [−0.042, +0.009]) and the query-path shares (6.0 / 23.4 / 68.2 / 2.4 %, 2.89 tokens per word; 13.7 / 16.1 / 67.3 / 2.9 %, 3.36; PREP 12.5, FUNC 10.5); the chain (20 cells, Σ wall 199.9 min, 9.2–11.4 min each, `CHAIN START` → `CHAIN DONE P1b` 200.5 min, peak 27 071 MiB); the tests — the ten listed files at the reported per-file counts (36 / 30 / 13 / 4 / 22 / 7 / 12 / 17 / 13 / 42 = 196 passed) and the full suite on CPU (1 506 passed, 2 skipped, 321.6 s); the CLAUDE.md, §1 / §4 / §5 / §6 / §8 / §9 edits and the two memory notes as described. *Read with care, none of it an error:* (a) the pre-registered per-arm rule reads "both" only because `E_slot` is a between-row contrast that every trained arm shares (+0.19 / +0.20 / +0.19 on Arabic-Exam); the AraRooPat-specific part is the within-row letter effect, and the paired contrasts that isolate it (`G_slot` −0.003 / +0.010, intervals through zero) are supplementary, as the report labels them — the reading rests on them together with the sign flip of `E_letter` across arms; (b) the two incidents of the run were recovered — the peer's `CLAUDE.md` hunks are present and unstaged and `docs/report.md` holds only the intended edits — and the new feedback memory (write to a temp file, then move; back up a file holding someone else's uncommitted work) is the right consequence; (c) under PMI native v5's `E_letter` flips sign (+0.160 / +0.197 against −0.197 / −0.106 under char): the unconditioned `الإجابة:` prior removes native's own أ preference and over-corrects it, which is why the PMI أ entries against native read the other way; (d) 0-shot lowers every arm a little (−0.001 to −0.046) and the gold-أ gap to BPE-16K persists under char-norm within its interval, so the demonstrations are not the carrier; under PMI on Arabic-Exam the gap does shrink (+0.093 [+0.027, +0.158]), the one exception; (e) the 1 000-row subsets are 4-choice rows, so the cell accuracies here (AraRooPat 0.514 on Arabic-Exam) are not §3.10's full-benchmark numbers (0.549) and should not be quoted as such. *Reading.* The finding stands as written: AraRooPat's letter-task deficit is a letter effect on أ — the model dislikes its `[CHAR_أ]` term at every slot, and the closing marker carries nothing — on top of a slot-1 aversion every trained arm shares; native's and BPE-16K's أ-favouring priors mask that aversion on the official prompt, AraRooPat's د-favouring prior compounds it. Under a letter-invariant decision the Arabic-Exam gap to BPE-16K is gone and a −0.039 Culture-MMLU gap remains, so the order native ≥ BPE-16K ≥ AraRooPat survives at a smaller size. Under the MSA scope rule Alghafa's in-scope gap to BPE-16K is −0.021 under char-norm with the interval touching zero and −0.011 to native under PMI, not separable; the dialect loss is a property of an MSA-only vocabulary, out of scope. Two things for the next reader: the cyclic-permutation averaging is a fair rule only if adopted for every arm and every letter task as *the* protocol, at four times the cost — the user's call, not a per-arm fix; and the cheaper open question is where AraRooPat's د preference and أ aversion come from (§5 item 26(b): the same four rotations on the two Phase 2 checkpoints, ~45 min GPU), because a letter prior that Phase 3's 6 750 uniform records did not remove says something about how a three-token letter spelling is learned.

### 3.12 Where the letter prior comes from — the Phase 1 and Phase 2 checkpoints under rotation (2026-09-25)

**Why.** §3.11 showed that AraRooPat v5's letter-task deficit to BPE-16K v5 is a *letter* effect on أ. Two facts framed the next question. First, every trained arm under-selects slot 1 (`E_slot` +0.197 / +0.200 / +0.192 on Arabic-Exam for AraRooPat / BPE-16K / native v5), while the untouched native base does so much less (+0.074; −0.040 on Culture-MMLU), and the three v5 arms share one Phase 3 (6 750 synthetic `arabic_squad_mcq` records, gold letter uniform). Second, the letter preference differs per arm and moved during training: at v5 AraRooPat favours د and the others favour أ, but §3.10's rotation-0 dumps of the Phase 2 checkpoints showed AraRooPat predicting ج on 53–60 % of rows and both from-scratch arms weakest on gold ب. At rotation 0 letter and slot coincide, so those dumps could not say which. **The question** (§5 item 26(b)): did AraRooPat acquire its أ aversion and د preference in Phase 3, from uniform-letter MCQ records, or were they already present after Phase 1 (embedding alignment on raw text, body frozen) or Phase 2 (continued pretraining on the 131 M-token mix)? The same question applies to the slot-1 aversion.

**Cells and protocol** (configs commit `a0b7798`, script commit `7b2540d`). Sixteen eval-only cells `<arm>_rot<k>` in `_letter_slot/`: the `embedding_alignment` (P1) and `warmup` (P2) checkpoints of `araroopat_3phase_v3` and `bpe_16k_3phase_v3`, which are the checkpoints the v5 arms' Phase 3 started from (`model.name_or_path` of `qwen_{araroopat,bpe16k}_3phase_v5_distill.yaml`), so P1 → P2 → v5 is one lineage per arm. Each was run at rotations 0–3, 3-shot, with no 0-shot cell, under the §3.11 protocol unchanged: the same two rows files, `max_length` 4096, char + PMI, schema-3 dumps. Each resolved config differs from its `<arm>_v5_rot<k>` twin only in experiment identity and `model.name_or_path`: 0 validation errors, 0 warnings, 0 param warnings (`_audit/validate_p1p2_configs.{py,json}`). The tokenizer blocks equal the v5 arms', and the checkpoints share those tokenizers (vocab 17 184 / 16 000, `Vocab size unchanged … skipping resize` in every log). The chain ran AraRooPat before BPE-16K at each rotation, P1a (the P2 cells) and then P1b (the P1 cells), through `_logs/run_chain_traj.sh` (a copy of §3.11's runner writing `chain_traj.log` / `gpu_mem_traj.csv`). Analysis: `scripts/mcq_letter_slot.py --arms araroopat_p1 araroopat_p2 araroopat_v5 bpe16k_p1 bpe16k_p2 bpe16k_v5 native_base native_v5`, which writes `letter_slot_trajectory.{md,json}` (the committed `letter_slot.{md,json}` are untouched). It uses the same cluster bootstrap over rows (5 000 resamples, seed 0, one resample matrix per task, so every pair is paired). New in the script: `E_x` for every letter and `E_s` for every slot; predicted-letter / slot shares pooled over the rotations; `G_by_letter` / `G_by_slot` on every pair; the trajectory block and the §3.12 rules. Run with the default four arms, it leaves every key of the §3.11 `letter_slot.json` byte-identical (0 changed of the existing keys; only new keys added).

**Checks.** *Replication* (`_audit/replication_check_p2.{py,json}`): both P2 rot0 dumps were joined to the full `*_v3_warmup_mcq4096` dumps of §3.10 by `row_index`, 2 cells × 2 tasks. Every one had 1 000 / 1 000 rows, 4 000 scored choices, 0 sentinel choices, and 0 prompt / continuation / `cont_tokens` / gold / argmax (char, PMI) mismatches. Max |Δ| was **0.0** as float32 on `ll`, `score_char`, `score_pmi` and `uncond_ll`, and max |Σ `cont_token_ll` − `ll`| 0.0. The check was run once after the two rot0 cells and again after P1a; it passed both times, so P1b followed. *Every cell* (`_audit/cell_checks_traj.{py,json}`, 32 cell × task dumps): 1 000 rows, `hit_cap` = `all_sentinel` = `sentinel` = 0, schema 3, `label_rotation` = k, `num_fewshot` 3, `max_length` 4096, 0 choices with |Σ `cont_token_ll` − `ll`| > 1e-5 (max 0.0), longest prompt 4 050. Each dump's prompt, continuations, `cont_tokens` and `prompt_units` equal its v5 twin's at the same rotation on every row (0 mismatches): same text, same tokenizer, only the weights differ. **The P1 cells are embedding-only checkpoints** (Phase 1 trained `embed_tokens` / the tied head on raw text with the Qwen body frozen). Their accuracies are not recognition numbers to set against the others. They are read here only for their letter × slot preferences.

| cell | Arabic-Exam char / PMI (rot0 · rot1 · rot2 · rot3) | Culture-MMLU char / PMI | wall (min) |
|---|---|---|---|
| `araroopat_p2_rot0..3` | 0.435 / 0.448 · 0.434 / 0.433 · 0.406 / 0.401 · 0.430 / 0.430 | 0.425 / 0.419 · 0.395 / 0.380 · 0.402 / 0.382 · 0.359 / 0.370 | 10.8 · 10.8 · 11.2 · 10.8 |
| `bpe16k_p2_rot0..3` | 0.503 / 0.505 · 0.458 / 0.465 · 0.438 / 0.475 · 0.473 / 0.496 | 0.458 / 0.491 · 0.430 / 0.440 · 0.427 / 0.457 · 0.442 / 0.452 | 9.6 · 9.8 · 9.4 · 9.7 |
| `araroopat_p1_rot0..3` | 0.352 / 0.357 · 0.368 / 0.348 · 0.362 / 0.350 · 0.395 / 0.398 | 0.346 / 0.347 · 0.331 / 0.323 · 0.323 / 0.324 · 0.319 / 0.318 | 11.1 · 10.9 · 11.0 · 11.1 |
| `bpe16k_p1_rot0..3` | 0.494 / 0.472 · 0.436 / 0.379 · 0.416 / 0.386 · 0.470 / 0.455 | 0.462 / 0.447 · 0.408 / 0.364 · 0.399 / 0.360 · 0.416 / 0.393 | 10.0 · 10.1 · 9.8 · 10.1 |

*1. Trajectory* (char, over the four rotations; Arabic-Exam / Culture-MMLU; E_x = acc(gold letter ≠ x) − acc(gold letter = x), positive = letter x is disfavoured when it is the answer; favoured letter = largest excess of the predicted share over the gold share, which is exactly 0.25 per letter over a row's rotations):

| arm, stage | acc | E_أ | E_slot1 | favoured letter (share) | most-disfavoured letter, E_x | E_x أ / ب / ج / د | E_s slot 1 / 2 / 3 / 4 |
|---|---|---|---|---|---|---|---|
| AraRooPat P1 | 0.369 / 0.330 | −0.105 [−0.134, −0.077] / −0.058 [−0.084, −0.032] | +0.182 [+0.146, +0.217] / +0.152 [+0.111, +0.191] | ج (0.604 / 0.619) | ب +0.340 [+0.318, +0.363] / ب +0.322 [+0.302, +0.342] | −0.105 / +0.340 / −0.512 / +0.276 · −0.058 / +0.322 / −0.527 / +0.262 | +0.182 / +0.080 / −0.043 / −0.196 · +0.152 / +0.020 / −0.089 / −0.070 |
| AraRooPat P2 | 0.426 / 0.395 | −0.144 [−0.171, −0.117] / −0.110 [−0.137, −0.083] | +0.182 [+0.135, +0.228] / +0.099 [+0.052, +0.148] | ج (0.468 / 0.483) | ب +0.251 [+0.228, +0.275] / د +0.239 [+0.217, +0.262] (ب +0.235) | −0.144 / +0.251 / −0.336 / +0.228 · −0.110 / +0.235 / −0.364 / +0.239 | +0.182 / +0.075 / −0.013 / −0.214 · +0.099 / +0.051 / −0.031 / −0.108 |
| AraRooPat v5 | 0.488 / 0.449 | **+0.131 [+0.108, +0.154] / +0.140 [+0.116, +0.165]** | +0.197 [+0.142, +0.253] / +0.141 [+0.080, +0.199] | د (0.336 / 0.315) | أ (as E_أ) | +0.131 / −0.037 / +0.044 / −0.137 · +0.140 / −0.041 / −0.005 / −0.093 | +0.197 / +0.036 / −0.067 / −0.155 · +0.141 / −0.006 / −0.067 / −0.058 |
| BPE-16K P1 | 0.454 / 0.421 | −0.260 [−0.290, −0.230] / −0.142 [−0.171, −0.113] | +0.148 [+0.103, +0.192] / +0.119 [+0.070, +0.167] | أ (0.412 / 0.345) | ب +0.360 [+0.337, +0.383] / ب +0.331 [+0.308, +0.354] | −0.260 / +0.360 / −0.137 / +0.037 · −0.142 / +0.331 / −0.090 / −0.098 | +0.148 / +0.016 / +0.002 / −0.147 · +0.119 / −0.039 / −0.037 / −0.035 |
| BPE-16K P2 | 0.468 / 0.439 | −0.472 [−0.500, −0.445] / −0.400 [−0.428, −0.371] | +0.144 [+0.102, +0.185] / +0.095 [+0.047, +0.142] | أ (0.566 / 0.526) | ب +0.323 [+0.302, +0.343] / ب +0.312 [+0.291, +0.334] | −0.472 / +0.323 / −0.063 / +0.212 · −0.400 / +0.312 / −0.068 / +0.155 | +0.144 / +0.022 / +0.009 / −0.154 · +0.095 / −0.018 / −0.013 / −0.056 |
| BPE-16K v5 | 0.517 / 0.493 | −0.100 [−0.123, −0.077] / −0.000 [−0.022, +0.022] | +0.200 [+0.143, +0.257] / +0.131 [+0.068, +0.194] | أ (0.313) / ب (0.263) | ج +0.060 [+0.040, +0.080] / ج +0.044 [+0.026, +0.062] | −0.100 / +0.039 / +0.060 / +0.000 · −0.000 / −0.023 / +0.044 / −0.020 | +0.200 / +0.008 / +0.006 / −0.191 · +0.131 / −0.043 / −0.010 / −0.068 |
| native base | 0.531 / 0.487 | −0.313 [−0.341, −0.285] / −0.284 [−0.310, −0.256] | +0.074 [+0.021, +0.125] / −0.040 [−0.093, +0.013] | أ (0.450 / 0.447) | ب +0.155 [+0.133, +0.178] / ج +0.140 [+0.117, +0.164] | −0.313 / +0.155 / +0.003 / +0.154 · −0.284 / +0.083 / +0.140 / +0.060 | +0.074 / +0.025 / +0.083 / −0.149 · −0.040 / +0.018 / +0.052 / −0.030 |
| native v5 | 0.555 / 0.511 | −0.197 [−0.222, −0.171] / −0.106 [−0.130, −0.082] | +0.192 [+0.134, +0.247] / +0.082 [+0.023, +0.142] | أ (0.368 / 0.324) | د +0.147 [+0.126, +0.168] / د +0.090 [+0.070, +0.110] | −0.197 / +0.064 / −0.014 / +0.147 · −0.106 / −0.012 / +0.028 / +0.090 | +0.192 / −0.064 / +0.025 / −0.138 · +0.082 / −0.080 / −0.002 / +0.003 |

Under PMI: AraRooPat E_أ +0.036 [+0.011, +0.063] / +0.069 [+0.046, +0.094] (P1), −0.056 / −0.006 (P2), +0.088 [+0.063, +0.113] / +0.079 [+0.053, +0.106] (v5). BPE-16K +0.272 / +0.328, −0.114 / −0.007, −0.248 / −0.144. Native +0.163 / +0.202 → +0.160 / +0.197. E_slot1 AraRooPat +0.144 / +0.127, +0.157 / +0.076, +0.102 / +0.100; BPE-16K +0.155 / +0.087, +0.152 / +0.069, +0.204 / +0.147. PMI subtracts each checkpoint's own unconditioned letter prior, which itself changes with the stage, so the PMI trajectory of E_أ zig-zags where the char one moves steadily.

*2. AraRooPat's letter × slot table* (char; rows gold letter أ / ب / ج / د, columns gold slot 1..4; 204–311 evaluations per cell on Arabic-Exam, 232–273 on Culture-MMLU). Arabic-Exam **P1**: أ 0.263 / 0.381 / 0.569 / 0.572; ب 0.027 / 0.049 / 0.044 / 0.280; ج 0.537 / 0.712 / 0.892 / 0.871; د 0.112 / 0.088 / 0.108 / 0.293. **P2**: أ 0.390 / 0.469 / 0.559 / 0.685; ب 0.154 / 0.208 / 0.137 / 0.395; ج 0.471 / 0.588 / 0.814 / 0.826; د 0.151 / 0.208 / 0.235 / 0.389. **v5**: أ 0.313 / 0.332 / 0.397 / 0.492; ب 0.313 / 0.544 / 0.500 / 0.675; ج 0.251 / 0.367 / 0.588 / 0.601; د 0.490 / 0.597 / 0.681 / 0.611. Culture-MMLU **P1**: أ 0.233 / 0.320 / 0.542 / 0.385; ب 0.043 / 0.053 / 0.044 / 0.198; ج 0.461 / 0.779 / 0.896 / 0.744; د 0.116 / 0.107 / 0.104 / 0.198. **P2**: أ 0.362 / 0.414 / 0.574 / 0.546; ب 0.211 / 0.217 / 0.108 / 0.330; ج 0.534 / 0.615 / 0.805 / 0.703; د 0.168 / 0.180 / 0.187 / 0.315. **v5**: as §3.11. BPE-16K's six tables, and every arm's, are in `letter_slot_trajectory.md`.

*3. AraRooPat − BPE-16K at each stage* (paired over row × rotation; char; Arabic-Exam / Culture-MMLU):

| stage | Δ | gold letter أ | letter ≠ أ | by gold letter ب / ج / د | gold slot 1 | G_letter | G_slot |
|---|---|---|---|---|---|---|---|
| P1 | −0.085 [−0.104, −0.066] / −0.091 [−0.110, −0.073] | **−0.201 [−0.237, −0.166] / −0.155 [−0.190, −0.120]** | −0.046 / −0.070 | −0.070 / +0.196 / −0.264 · −0.085 / +0.236 / −0.362 | −0.110 / −0.116 | +0.155 [+0.116, +0.193] / +0.085 [+0.047, +0.122] | +0.034 [−0.004, +0.071] / +0.032 [−0.009, +0.074] |
| P2 | −0.042 [−0.060, −0.024] / −0.044 [−0.062, −0.026] | **−0.288 [−0.320, −0.256] / −0.261 [−0.295, −0.227]** | +0.040 / +0.028 | +0.012 / +0.163 / −0.054 · +0.014 / +0.178 / −0.107 | −0.069 / −0.047 | +0.328 [+0.292, +0.365] / +0.289 [+0.253, +0.326] | +0.037 [−0.000, +0.074] / +0.004 [−0.036, +0.045] |
| v5 | −0.029 [−0.051, −0.008] / −0.044 [−0.066, −0.022] | **−0.202 [−0.236, −0.169] / −0.149 [−0.180, −0.117]** | +0.028 / −0.009 | +0.028 / −0.017 / +0.074 · −0.030 / −0.007 / +0.011 | −0.027 / −0.052 | +0.230 [+0.199, +0.262] / +0.140 [+0.111, +0.171] | −0.003 [−0.049, +0.043] / +0.010 [−0.038, +0.060] |

By gold slot (P1 / P2, slots 2–4): −0.134 / −0.049 / −0.051 and −0.083 / −0.025 / +0.000 on Arabic-Exam; −0.136 / −0.053 / −0.066 and −0.096 / −0.031 / −0.006 on Culture-MMLU. Under PMI, G_letter is −0.236 / −0.259 (P1), +0.058 / +0.001 (P2), +0.336 / +0.224 (v5).

*4. Within each arm, between stages* (later − earlier, paired over rows; for such a pair `G_letter` = ΔE_أ and `G_slot` = ΔE_slot1 exactly; char; Arabic-Exam / Culture-MMLU):

| arm, step | Δ | on gold أ | on letter ≠ أ | ΔE_أ | ΔE_slot1 | ΔE_x أ / ب / ج / د |
|---|---|---|---|---|---|---|
| AraRooPat P2 − P1 | +0.057 / +0.066 | +0.086 [+0.057, +0.114] / +0.105 [+0.077, +0.133] | +0.047 / +0.052 | −0.039 [−0.069, −0.008] / −0.053 [−0.084, −0.022] | +0.000 [−0.031, +0.031] / −0.052 [−0.085, −0.021] | −0.039 / −0.089 / +0.176 / −0.048 · −0.053 / −0.087 / +0.163 / −0.023 |
| AraRooPat v5 − P2 | +0.062 / +0.054 | −0.144 [−0.174, −0.114] / −0.134 [−0.165, −0.104] | +0.130 / +0.116 | **+0.274 [+0.242, +0.308] / +0.250 [+0.218, +0.283]** | +0.016 [−0.023, +0.054] / +0.042 [−0.001, +0.082] | +0.274 / −0.288 / +0.380 / −0.366 · +0.250 / −0.276 / +0.358 / −0.332 |
| BPE-16K P2 − P1 | +0.014 / +0.018 | +0.173 [+0.148, +0.198] / +0.211 [+0.183, +0.239] | −0.039 / −0.046 | −0.212 [−0.240, −0.184] / −0.257 [−0.290, −0.226] | −0.003 [−0.028, +0.022] / −0.024 [−0.052, +0.002] | −0.212 / −0.037 / +0.075 / +0.175 · −0.257 / −0.019 / +0.023 / +0.253 |
| BPE-16K v5 − P2 | +0.049 / +0.053 | −0.230 [−0.258, −0.201] / −0.246 [−0.276, −0.217] | +0.142 / +0.153 | **+0.372 [+0.342, +0.403] / +0.399 [+0.367, +0.432]** | +0.056 [+0.018, +0.095] / +0.036 [−0.005, +0.077] | +0.372 / −0.284 / +0.123 / −0.212 · +0.399 / −0.335 / +0.111 / −0.175 |
| native v5 − base | +0.024 / +0.024 | −0.063 / −0.109 | +0.053 / +0.068 | +0.116 [+0.091, +0.141] / +0.177 [+0.148, +0.205] | **+0.118 [+0.089, +0.146] / +0.122 [+0.087, +0.159]** | +0.116 / −0.091 / −0.018 / −0.007 · +0.177 / −0.095 / −0.112 / +0.029 |

*5. Per token.* AraRooPat's `[CHAR_x]` mean log P, gold / non-gold choices pooled over slots (Arabic-Exam; Culture-MMLU in the artifact):

| stage | أ | ب | ج | د | `[LIT_BEGIN]` / `[LIT_END]` (all choices) |
|---|---|---|---|---|---|
| P1 | −1.312 / −1.937 | −1.976 / −2.599 | −0.796 / −1.276 | −1.999 / −2.766 | −0.0800 / −0.0022 |
| P2 | −1.161 / −1.812 | −1.532 / −2.285 | −1.000 / −1.688 | −1.645 / −2.459 | −0.0436 / −0.0046 |
| v5 | −1.260 / −2.041 | −1.111 / −1.957 | −1.314 / −2.280 | −1.005 / −1.677 | −0.0021 / −0.0006 |

(Culture-MMLU `[LIT_BEGIN]` −0.1134 / −0.0373 / −0.0016, `[LIT_END]` −0.0021 / −0.0041 / −0.0004.) `[LIT_BEGIN]` is the first continuation token and is conditioned on the same prompt for all four choices, so its log-prob is identical across a row's choices: the within-row spread is 0.0 in every AraRooPat cell. At P1 its 0.08–0.11 nats cannot move the argmax. Everything that decides is `[CHAR_x]` (plus a `[LIT_END]` ≤ 0.005). Over all choices, أ's margin over the mean of the other three letters is, in nats, P1 → P2 → v5: AraRooPat +0.277 → +0.307 → −0.081 (Culture +0.190 → +0.223 → −0.106), BPE-16K ` أ` +0.929 → +1.296 → +0.273 (+0.678 → +1.030 → +0.119), native +0.771 → +0.399 (+0.727 → +0.276). AraRooPat's best letter token is ج at P1 and P2 and د at v5; BPE-16K's is أ at every stage (ب at v5 on Culture-MMLU, by 0.003). Phase 3 moved AraRooPat's `[CHAR_د]` most (gold −1.645 → −1.005, +0.64 nats) and its `[CHAR_ج]` down (−1.000 → −1.314); `[CHAR_أ]` barely moved (−1.161 → −1.260).

*6. Supplementary, not pre-registered* (`_audit/trajectory_supplement.{py,json}`, same bootstrap). **Letter-prior strength**, measured as the total-variation distance of the predicted-letter shares from uniform (pure preference, since the gold is uniform over a row's rotations), char, P1 → P2 → v5: AraRooPat 0.386 → 0.301 → 0.103 / 0.389 → 0.309 → 0.084; BPE-16K 0.221 → 0.317 → 0.063 / 0.187 → 0.281 → 0.032; native 0.200 → 0.124 / 0.197 → 0.073. Phase 3 flattens both from-scratch arms by about as much: v5 − P2 AraRooPat −0.198 [−0.214, −0.183] / −0.225 [−0.241, −0.210], BPE-16K −0.254 [−0.269, −0.240] / −0.249 [−0.264, −0.235]. AraRooPat's residual at v5 is +0.040 [+0.024, +0.053] / +0.052 [+0.040, +0.064] above BPE-16K's and about native v5's size. Under PMI it rises in Phase 3 (+0.076 / +0.037) — PMI's over-correction onto د. **Rotation-averaged decision** (the §3.11 proposal) per stage: AraRooPat 0.469 → 0.506 → 0.522 / 0.358 → 0.436 → 0.469; BPE-16K 0.525 → 0.552 → 0.523 / 0.466 → 0.503 → 0.508. AraRooPat − BPE-16K under it: −0.056 [−0.088, −0.024] → −0.046 [−0.077, −0.015] → −0.001 [−0.030, +0.027] on Arabic-Exam and −0.108 [−0.140, −0.077] → −0.067 [−0.097, −0.035] → −0.039 [−0.068, −0.011] on Culture-MMLU. Stage changes: AraRooPat P2 − P1 +0.037 [+0.010, +0.064] / +0.078 [+0.050, +0.107], v5 − P2 +0.016 [−0.009, +0.041] / +0.033 [+0.007, +0.059]; BPE-16K +0.027 [+0.005, +0.049] / +0.037 [+0.017, +0.058], then −0.029 [−0.052, −0.005] / +0.005 [−0.017, +0.027].

**Reading under the pre-registered rules** (on AraRooPat's own E_أ, char; the rules as written in the brief, evaluated by `trajectory_rules`):

- *Inherited from the vocabulary or its initialisation* **does not fire** on either task. E_أ at P1 excludes 0, but it is negative (−0.105 / −0.058, أ favoured) and does not overlap v5's.
- *Acquired in Phase 2* **does not fire.** P1's CI excludes 0, and P2 is further from v5 (−0.144 / −0.110).
- *Created or reshaped by Phase 3* **fires on both tasks**, by both of its clauses. E_أ at P2 has its CI below +0.05. The most-disfavoured letter moves from ب at P2 to أ at v5 on Arabic-Exam, and from د (ب +0.235, within 0.004) to أ on Culture-MMLU.
- *Slot*: the slot-1 aversion is **inherited already at P1**, and at P2, for both from-scratch arms on both tasks (AraRooPat +0.182 / +0.152, BPE-16K +0.148 / +0.119 at P1, every CI above 0). *Created by Phase 3* does not fire for either arm: ΔE_slot1 over Phase 3 is +0.016 / +0.042 and +0.056 / +0.036. For native the base → v5 pattern stands: +0.074 / −0.040 → +0.192 / +0.082, ΔE_slot1 +0.118 / +0.122, both CIs above 0.
- *Under PMI* (one line): Phase 3 fires on both tasks (P2 → v5 most-disfavoured ب → ج). Inherited fires on Culture-MMLU (P1 +0.069 [+0.046, +0.094] overlaps v5 +0.079 [+0.053, +0.106]) and misses on Arabic-Exam by 0.0003 (P1 up to +0.0630, v5 from +0.0633). The slot readings are as under char.

**What BPE-16K shows at the same stages (the control), and what that does to the reading.** The Phase-3 rule fires, but the move it detects is not AraRooPat's own. Phase 3 raises E_أ in *every* arm: +0.274 / +0.250 for AraRooPat, **+0.372 / +0.399 for BPE-16K**, +0.116 / +0.177 for native. It removes every arm's ب aversion (ΔE_ب −0.288 / −0.276, −0.284 / −0.335, −0.091 / −0.095) and flattens both from-scratch letter priors by the same amount (TV −0.20 / −0.23 against −0.25 / −0.25). The change in E_x has the same sign letter by letter in both arms (ΔE_أ +, ΔE_ب −, ΔE_ج +, ΔE_د −). So the shift is a property of Phase 3, shared by both vocabularies. Before Phase 3 every arm favours أ as the answer (E_أ < 0 at P1 and P2 in both from-scratch arms, and in the native base). AraRooPat favours it least, BPE-16K most (−0.472 / −0.400 at P2). A shared upward shift therefore flattens BPE-16K to ≈ 0 (−0.100 / −0.000) and pushes AraRooPat past zero into the aversion §3.11 measured.

**The AraRooPat-specific part is older than Phase 3.** In the paired contrast, where row content cancels, AraRooPat − BPE-16K on the gold-أ evaluations is **−0.201 / −0.155 at P1, −0.288 / −0.261 at P2 and −0.202 / −0.149 at v5**. The §3.11 deficit is present at its v5 size after Phase 1, before any body training. It grows in Phase 2, because BPE-16K's أ preference grows (ΔE_أ −0.212 / −0.257) while AraRooPat's barely moves (−0.039 / −0.053). Phase 3 brings it back (G_letter +0.155 / +0.085 → +0.328 / +0.289 → +0.230 / +0.140). The per-token view agrees: at every stage, AraRooPat's `[CHAR_أ]` stands less far above its other letters than BPE-16K's ` أ` does (margin −0.65 / −0.49 nats behind at P1, −0.99 / −0.81 at P2, −0.35 / −0.23 at v5). What *does* change across the stages is the rest of AraRooPat's letter profile. At P1 and P2 it is ج-heavy (ج favoured, 0.60 / 0.62 of the predictions at P1; AraRooPat − BPE-16K on gold ج +0.196 / +0.236), with ب and د starved (gold-د −0.264 / −0.362 against BPE-16K at P1). Phase 3 removes the ج and د distortions relative to BPE-16K (v5: ج −0.017 / −0.007, د +0.074 / +0.011) and leaves the أ gap in place. **The د preference is created by Phase 3.** AraRooPat's E_د goes from +0.276 / +0.262 (P1) and +0.228 / +0.239 (P2) to −0.137 / −0.093 at v5, and `[CHAR_د]` rises 0.64 nats in Phase 3. The AraRooPat − BPE-16K contrast on gold د turns from −0.054 / −0.107 at P2 to +0.074 / +0.011 at v5: Phase 3 moved AraRooPat's د further than BPE-16K's, by 0.13 / 0.12 in accuracy.

**The slot-1 aversion** is shared by the two vocabularies at every stage (G_slot +0.034 / +0.032, +0.037 / +0.004, −0.003 / +0.010, every CI through 0). In the from-scratch arms it is present after Phase 1 at roughly its v5 size. In native it appears in Phase 3. `E_slot` is a between-row contrast, so part of it is row content (the rows whose gold is the first option). The native base's +0.074 / −0.040 is the closest thing to a row-content floor these cells give. Not measured, but two things fit the pattern. In the from-scratch arms, the whole prompt — the option labels and the first option line right after the question — is read through re-learned embeddings from Phase 1 on. And in every Phase 3 record the letter and the slot are confounded (أ always labels the first line; `synthetic_mcq.py` draws the gold position uniformly with `rng.randrange`, so nothing in the builder favours a slot). A correction of an أ-favouring prior, which every arm needed at P2, can then be learned as a letter correction, a slot correction or both. Native, whose base prior was the strongest and which had no earlier stage, shows both.

**What this means for §3.11's headline and for the proposals.** §3.11's finding stands and gains an origin. AraRooPat's letter-task deficit to BPE-16K is a deficit on the letter أ, and it is already there, at the same size, in the embedding-only Phase 1 checkpoint, before the body is trained. It comes with the vocabulary: the three-token literal spelling `[LIT_BEGIN] [CHAR_أ] [LIT_END]`, its `surface_avg` initialisation, or Phase 1's raw-text statistics of single-letter literals. These cells cannot separate the three. Phase 3 does not create it. Phase 3 shifts every arm's letter prior the same way; that turns AraRooPat's weaker أ preference into an absolute aversion and adds a د preference of its own. Under the letter-invariant decision, the gap shrinks with training (−0.056 → −0.046 → −0.001 on Arabic-Exam, −0.108 → −0.067 → −0.039 on Culture-MMLU). So the residual gap on the official prompt is increasingly a letter-prior artefact and less a recognition deficit. The three proposals, **each applied to nothing and each a rule for every arm**:

(1) The *rotation-averaged* (cyclic-permutation) scoring of §3.11, as protocol. It removes every arm's letter prior at evaluation time, whatever its origin, at 4× cost. It is the only one of the three that changes no checkpoint.
(2) *Letter-rotation augmentation of the synthetic MCQ records in Phase 3* (each record rendered under a rotation drawn uniformly, so letters and slots are de-confounded in training). This is the pre-registered consequence of the rule that fired, and the measurement supports it in a narrower form than the rule states. Phase 3 does flatten the letter priors under both spellings (it is not "moving without flattening"). But its records cannot tell a letter correction from a slot correction, and its shared shift is what turns AraRooPat's relative weakness into an absolute aversion.
(3) The *vocabulary-side lever* of §5 item 20: one closed-class token per standalone choice letter, so a letter costs one token under AraRooPat as under BPE-16K and native. The paired contrast supports it, because the AraRooPat-specific part is present after Phase 1. It is not supported by the pre-registered inherited rule, which reads the absolute E_أ and does not fire. Whether the lever would close the gap is not measured; it needs a retrained vocabulary.

**GPU.** Sixteen cells, Σ wall 166.2 min (9.4–11.2 min each; AraRooPat 10.8–11.2, BPE-16K 9.4–10.1). P1a ran 82.1 min end to end (`CHAIN START` 17:04:34 → `CHAIN DONE P1a` 18:26:42), P1b 84.1 min (18:27:10 → 19:51:14). Peak 10 625 MiB (`gpu_mem_traj.csv`). CPU: the analysis 13 s, the supplement and the audits < 1 min each.

**Artifacts.** `outputs/experiments/qwen_native_vs_araroopat/_letter_slot/`: the sixteen cells `{araroopat,bpe16k}_{p1,p2}_rot{0..3}/` (`all_metrics.json`, `eval_rows/{arabic_exam,culture_arabic_mmlu}.parquet`, schema 3); `letter_slot_trajectory.{md,json}` (json sha256 prefix `788460de84dfbafd`); `_audit/{validate_p1p2_configs,replication_check_p2,cell_checks_traj,trajectory_supplement}.py` (+ `.json` each); `_logs/{make_configs_p1p2.py,run_chain_traj.sh,chain_traj.log,chain_traj_P1a.stdout,chain_traj_P1b.stdout,gpu_mem_traj.csv}` and one console log per cell. Configs: `configs/experiments/mcq_letter_slot/{araroopat,bpe16k}_{p1,p2}_rot{0..3}.yaml`. Commits: `b87f9ac` (the §3.11 verification paragraph), `7b2540d` (`mcq_letter_slot.py --arms`, optional 0-shot, the trajectory table, tests), `a0b7798` (the sixteen configs), and the documentation commit of this section.

## 4. What the campaign established

1. The native model's loops were partly measurement (the first loop rule) and partly a greedy attractor of the base model; under the corrected rules the untrained base loops on 6.8 % of prompts and the SFT arm on 12.4 %. SFT on the 30 000-record mixture learns the references' register and length but not their content: judge 2.28 vs 2.42, CI including zero.
2. AraRooPat v1/v2 failed for a reason unrelated to morphology: an arbitrary embedding initialization and a 37 M-token budget left the language model at 1.35 nats/char (native 0.83). Informed initialization plus 131 M tokens brought it to 0.88 and the judge from 1.07 to 2.00; answer NLL now equals the untrained native model's. (Loss figures corrected in §3.6; as first measured, 1.34 → 0.97 against a native 0.82.)
3. At equal adaptation a plain BPE-16K vocabulary ties AraRooPat on the judge and beats it on LM loss (0.86 vs 0.88), loops (20.8 vs 31.2 %) and generation speed (153 vs 111 chars/s), with 1.7× more text seen per token budget; AraRooPat has the lower answer NLL on the held-out references (0.81 vs 0.85 nats/char — the 0.64 first reported for BPE divided by the wrong chars-per-token figure, corrected 2026-09-23, §3.6 *Verification*). **Amended by §3.6:** once both arms train on the free-form-aware stop signal, AraRooPat leads the judge (2.152 vs 1.996, paired Δ +0.156 [+0.000, +0.312]), closes the loop gap (21.2 vs 19.6 %) and widens its answer-NLL lead (0.787 vs 0.845); BPE keeps the LM-loss and speed advantages. The tie was partly an artefact of AraRooPat having trained on a fifth of the mixture. AraRooPat's measurable advantages are decode fidelity to the written word (99.3 % word round trip) and root conservation (0.379 vs 0.048). Both arms sit 0.4 judge points under the untouched native model; that gap is the cost of re-learning any vocabulary in 131 M tokens — and it is **not** explained by language-model quality, which after the §3.6 correction is within 0.03 nats/char of native for both.
4. Phase 2 loss is flat from its first fifth in both arms; more raw text at this learning rate is not the next lever.
5. (§3.6) Two measurement faults were corrected. The raw-text "held-out" loss was read from each cell's *own* pool — training text for the adapted arms, and a different document set per arm; on a properly held-out set the LM gap to native is 0.03 nats/char, not 0.13. And the Phase 3 stop signal carried 12 % of the loss tokens: on one arm it wandered upward at step 1 400 and stopped training at a fifth of the mixture. Neither fault changed a ranking; both changed magnitudes enough to change what the next step should be.
6. (§3.7) Neither a token-level repetition penalty nor more and longer Phase 3 free-form data closes AraRooPat's generation gap. A penalty of 1.2 removes loops in every cell (0–1.2 %) but lowers the judge (AraRooPat −0.30, native base −0.29, both CIs below 0; BPE and native SFT within noise), doubles the off-topic flags and, being token-level, halves AraRooPat's article and و tokens; the prompts on which AraRooPat loops hold no recoverable answer (1.72 once the loop is broken, against 2.08 / 2.32 for BPE / native SFT on the same prompts). v4 — 35 000 examples of a 25/15/60 mixture at 1 024 tokens, three times the distinct free-form records — brought the answer NLL to native SFT's level (0.766 nats/char both) without moving the judge (+0.03, CI spans 0) or the loops (23.2 %, worst in the longest-reference tercile). At equal likelihood of the references native SFT loops on half as many prompts: the gap is in free-running generation, which points to sequence-level remedies (distillation from the native model, a training-time repetition objective), not to Phase 3 data or token-level decoding. In every cell the loop rate rises with the reference length.
7. (§3.8) The flatness explanation of the loop gap does not hold — at equal likelihood the from-scratch vocabularies are as sharp per character as native, and loops start at low-margin points in every model, the untrained native base included. Sequence-level distillation from a third-party teacher (Aya-Expanse-32B, chosen by a bake-off on dev prompts; judge 3.65 on the test prompts) with the same teacher answers, mixture, seed and draw rule for all three arms raised every arm by about a quarter point (AraRooPat +0.25 vs v4, native +0.24 vs SFT, BPE +0.14 vs ffstop) and brought loops to the untrained base's level for AraRooPat and native (6.8 / 8.4 %, flat across reference length) but not for BPE-16K (15.6 %). Under identical treatment AraRooPat retains 0.67 of the teacher's judge score, native 0.69 and BPE-16K 0.58: AraRooPat is not separable from the native vocabulary (−0.09 [−0.24, +0.07]; its residual cost is fluency and instruction following, not correctness) and is above BPE-16K (+0.30 [+0.15, +0.45]) — the first such interval under equal treatment. All three students stay far below the teacher (−1.13 to −1.52). On the teacher's own dev text the native vocabulary has the lowest NLL per character (0.388 vs AraRooPat 0.420, BPE 0.425).

8. (§3.10) On recognition the order is not the generation order. With the MCQ scorer's continuation window fixed (it had skipped the first continuation token and scored `</s>` for every from-scratch tokenizer) and every arm evaluated at 4 096 tokens on the rows no arm truncates, the letter-scored benchmarks rank native v5 ≥ BPE-16K v5 > AraRooPat v5: AraRooPat trails BPE-16K by 3–5 points (Arabic-Exam −0.047 / −0.034, Culture-MMLU −0.029 / −0.037, char / PMI, every CI below 0) and native v5 by 4–7, while BPE-16K and native v5 are within a point under PMI — where the judge had AraRooPat 0.30 above BPE-16K. The distilled Phase 3 did not cost recognition: every arm gained over its predecessor, AraRooPat the most (+5–8 points over its Phase 2 checkpoint, halving its deficit to BPE-16K). ACVA and Alghafa's fixed-label sub-configs cannot rank tokenizers under this protocol: one label takes 89–100 % of the predictions whenever a label's unconditioned prior is several nats off the others' — and a label spelled in characters (AraRooPat's ` ايجابي`) is exactly such a label. *Restated on the MSA scope (§3.11, 2026-09-25):* the two letter-scored benchmarks are MSA and stand as above; the in-scope Alghafa number is the MSA choice-text group (1 772 rows), where AraRooPat v5 trails BPE-16K v5 by −0.021 [−0.042, −0.002] char (−0.058 PMI) and native v5 by −0.052 char (−0.011 [−0.030, +0.009] PMI) — most of the §3.10 Alghafa choice-text loss sat on the dialect rows, out of scope (−0.052 to BPE-16K on the 4 500 dialect rows proper). The letter-task gaps are partly a letter-prior effect of the official prompt: under a letter-invariant (rotation-averaged) decision, on 1 000-row subsets, AraRooPat − BPE-16K is −0.001 [−0.030, +0.027] on Arabic-Exam and −0.039 [−0.068, −0.011] on Culture-MMLU, and AraRooPat still trails native v5 by 0.05 on both — the order native ≥ BPE-16K ≥ AraRooPat survives, the Arabic-Exam gap to BPE-16K does not.
9. (§3.10 *Few-shot handling*) AraRooPat v5's recognition deficit on the two letter-scored benchmarks is confined to the rows whose answer is أ — the first option; letter and position are confounded on these benchmarks: on every other row it equals BPE-16K (Δ +0.000 / +0.001, CIs within ±0.011) and slightly exceeds native v5 (+0.017 / +0.020, CIs above 0). No cell copies the 3-shot demonstrations above chance, the synthetic MCQ training set's gold letter is uniform, and a per-position calibration does not move any v5 arm — **§3.11 identified the cause**: rotating the letters over the slots, AraRooPat v5's deficit to BPE-16K follows the *letter* أ to every slot (−0.202 / −0.149 on gold letter أ, −0.027 / −0.052 on gold slot 1; G_letter +0.230 / +0.140, G_slot −0.003 / +0.010), not the first slot. The pre-registered per-arm rule reads *both* (E_letter +0.131 / +0.140, E_slot +0.197 / +0.141), but the slot effect is shared by every trained arm and only the letter effect differs in sign: every arm under-selects slot 1, and native and BPE-16K favour the letter أ while AraRooPat favours د, so on the official prompt (أ always on slot 1) their prior compensates and AraRooPat's compounds. It is not the three-token spelling (`[LIT_BEGIN]` / `[LIT_END]` ≈ 0.002 / 0.001 nats), not the demonstrations (the gold-أ gap is −0.120 / −0.099 at 0-shot). **§3.12 found its origin:** the AraRooPat − BPE-16K gap on gold أ is already −0.201 / −0.155 in the embedding-only Phase 1 checkpoint (−0.288 / −0.261 after Phase 2, −0.202 / −0.149 at v5), so the AraRooPat-specific deficit comes with the vocabulary — the three-token literal spelling, its initialisation or Phase 1's raw-text statistics, which those cells cannot separate — not with Phase 3. What Phase 3 adds is a letter-prior shift shared by every arm (ΔE_أ +0.274 / +0.250 AraRooPat, +0.372 / +0.399 BPE-16K, +0.116 / +0.177 native) that flattens BPE-16K's أ preference to ≈ 0 and pushes AraRooPat's weaker one into an aversion, plus AraRooPat's د preference; the from-scratch arms' slot-1 aversion is present from Phase 1 in both vocabularies alike. The ACVA "win" is the label prior of §3.10 (a), not recognition.

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
16. ~~The four MCQ benchmarks on the three v5 checkpoints~~ — **done, §3.10** (plus the untouched base and the two Phase-2 checkpoints). **The four MCQ benchmarks on the three v5 checkpoints** (eval-only re-eval configs, `acva` / `alghafa` / `culture_arabic_mmlu` / `arabic_exam`, ~1–2 h GPU): the campaign's recognition-side eval has not been run on any Qwen arm; with three checkpoints trained identically it is the cheapest next comparison, and it says whether the free-form ranking (native ≈ AraRooPat > BPE) holds under log-likelihood scoring.
17. ~~Documentation fix~~ — **done** (`2f56441`). Documentation fix: `sft_eval_mixture_manifest.json` lives under `<cell>/training/data/`, not `<cell>/data/` as the *Phase 3 mixture* section of CLAUDE.md says (§3.8 *Verification* (d)).

18. ~~Fix the MCQ scorer's continuation window~~ — **done, §3.10** (`2f56441`; no-op on `native_qwen3_sft`: 1 200 rows, argmax 100 %, Δ 0.0). **Fix the MCQ scorer's continuation window** (§3.9): strip the appended `</s>` from both encodings and take the continuation as the tokens after the longest common prefix; regression test with an EOS-appending tiny tokenizer asserting the scored ids; `native_qwen3_sft` must reproduce its four accuracies byte for byte. Blocks item 16.
19. ~~Item 16's protocol~~ — **done, §3.10** (4 096 for every arm, intersection primary, 4096-only sensitivity, Alghafa per sub-config and by label). Item 16's protocol after §3.9: `max_length: 4096` for every arm, `all_sentinel` shares reported per arm, accuracy on the intersection of untruncated rows as the primary comparison, Alghafa read per sub-config (the word-scored sentiment sub-configs spell `ايجابي` in characters under AraRooPat).
20. Tokenizer to-dos for the next AraRooPat vocabulary (not for the fixed v5 checkpoints): add the curly quotes `“ ”`, the Persian letters `ی` / `ھ`, the Quranic marks, the ornate parentheses `﴾ ﴿` and Latin letters to the character / punctuation inventories (0.55–1.1 % of exam / Alghafa word occurrences carry an `<unk>` today), and `ma_interrog` to `CAMEL_CLITIC_SURFACE`. Dialect coverage is **out of scope** under the MSA scope rule of §3.11 (Alghafa's dialect rows put 13.7 % of their question words on the character path against 6.0 % for the same passages in MSA, and find no closed-class token for dialect function words); revisit it only if the scope changes. **From §3.12:** one closed-class token per standalone choice letter (the letters `أ ب ج د …` as option labels and as the answer after `الإجابة:`), so a choice letter costs one token under AraRooPat as under BPE-16K and native — AraRooPat's specific deficit on the letter أ is present after Phase 1, before any body training, so it comes with the three-token literal spelling, its `surface_avg` initialisation or Phase 1's raw-text statistics. Untested; it needs a retrained vocabulary and applies to the tokenizer, not to one arm's scoring.
21. Every from-scratch MCQ accuracy of the Llama-1B era (no longer on this disk; quoted in earlier documents and CLAUDE.md) was computed on the wrong token window — for a single-piece letter continuation, on `</s>` alone — and is void until re-run under item 18's fix; the native Llama numbers stand. Decide whether any of them is worth re-running. **Still open (the user's call)** — §3.10 adds that a re-run would also meet the label-prior collapse on ACVA / Alghafa's fixed-label groups and the bf16 ties below.

22. **P1b — the two Phase-2 checkpoints** on the four MCQ benchmarks — **done, §3.10** (`araroopat_3phase_v3_warmup_mcq4096`, `bpe_16k_3phase_v3_warmup_mcq4096`).
23. **Label-prior collapse on fixed-label MCQ groups** (§3.10 (a)): ACVA and Alghafa's true/false and sentiment sub-configs put 89–100 % of predictions on one label under char or PMI depending on each cell's label priors. Options: score these groups by a calibrated rule (e.g. LightEval's PMI with a content-free query per label set, or a per-group prior correction estimated on held-out rows), report them apart, or drop them from tokenizer comparisons. Decide before quoting any ACVA / Alghafa-fixed-label number across tokenizers.
24. **bf16 `log_softmax` in the scorer** (§3.10 (b)): single-token continuations tie exactly on 1.8–5.3 % of letter-task rows (native, BPE-16K); a float32 `log_softmax` removes the ties. Measured effect of re-breaking them at random ≤ 0.37 points; changing it changes native numbers, so it needs its own no-op-style check against the current dumps.
25. Intrinsic RPS is not exactly reproducible for the same tokenizer across runs (BPE-16K 0.0498 vs 0.0476, native 0.0758 vs 0.0779; AraRooPat 0.3788 both) — likely a non-deterministic root extractor path; small, but MEI inherits it.
26. **Done (§3.11, 2026-09-25).** *Outcome:* the deficit follows the letter أ, not the slot (see §4 finding 9); the demonstrations and the terminal token are not the carrier; under a letter-invariant rotation-averaged decision the Arabic-Exam gap to BPE-16K vanishes and a −0.04 Culture-MMLU gap remains. *What follows:* (a) decide whether letter-scored MCQ is scored by cyclic-permutation averaging for every arm (×4 cost; the §3.11 proposal) before quoting letter-task gaps across tokenizers; (b) ~~where AraRooPat v5's preference for the letter د comes from~~ — **done, §3.12** (the Phase 1 and Phase 2 checkpoints of both from-scratch arms under the four rotations, 16 cells, 2 h 46 min GPU): the AraRooPat-specific أ deficit to BPE-16K is present after Phase 1 at its v5 size; Phase 3 created the absolute أ aversion — by a letter-prior shift it applies to every arm, larger for BPE-16K — and the د preference; the slot-1 aversion is present from Phase 1 in both from-scratch arms. What follows, as proposals for the next campaign, applied to nothing and each for every arm: letter-rotation augmentation of the Phase 3 MCQ records (the pre-registered consequence of the rule that fired — the records confound letter and slot) and one token per standalone choice letter (item 20; supported by the paired contrast, not by the pre-registered inherited rule), besides (a); (c) the Culture-MMLU residual under the letter-invariant decision, by subject, from the existing dumps. *The original item:* **The first-option / أ deficit under AraRooPat** (§3.10 *Few-shot handling*): the 3-shot shape is not mishandled by any cell (measured), but AraRooPat v5 scores 0.32 / 0.28 when the answer is أ against 0.49–0.62 on the other letters, and a post-hoc calibration does not move it. Letter and position are confounded on the benchmarks. Two cheap eval-only re-evals on a seeded 1 000-row subset of Arabic-Exam and Culture-MMLU for the three v5 arms would separate them: `num_fewshot: 0` (does the deficit survive without demonstrations) and a shuffled-options pass (gold moved uniformly over the four slots, letters unchanged — position); if it is the slot, look at how the first option line is tokenised after the question under AraRooPat. A third, for the mechanism: a per-token re-score of the three-token letter continuation on the same subset (log P of `[LIT_BEGIN]`, `[CHAR_x]`, `[LIT_END]` per choice — a literal beginning with أ is almost always a longer word in the corpus, so the `[LIT_END]`-after-hamza term is a candidate). Whatever scoring rule comes out of it applies to every arm as one rule, never to AraRooPat alone.

## 6. GPU time (2026-09-22 onwards)
Four training runs 8 h 52 min (1 h 15 lost attempt + 3 h 17 v3 + 0 h 56 lost attempt + 3 h 25 BPE), probe ~35 min, six diagnostics ~6 min, judge passes ~10 min — ≈ 9 h 43 min. Earlier: v1 ~1 h, v2 ~1.5 h (+ 45 min resume), native runs and re-evals ~2 h.

§3.6 (2026-09-22 evening): AraRooPat ffstop 42 min end to end (Phase 3 26.9 min incl. 33 evals, free-form eval 5 min), BPE ffstop 36 min (Phase 3 25.9 min, 35 evals, eval 7.6 min), ten raw-text diagnostics ~12 min, judge 2 × 250 verdicts ~1 min plus a 7-minute cold start — **≈ 1 h 40 min**. Non-GPU: the held-out build + two verification scans ~13 min, the two mixture dry runs ~35 min (AraRooPat over the CAMeL bridge).

§3.7 (2026-09-23): five eval-only runs 19.6 min, two judge passes 10.4 min (incl. cold starts), memory smoke 1.3 min, AraRooPat v4 1 h 22 min (Phase 3 62.6 min, free-form eval 8.4 min), diagnostic 0.7 min — **≈ 1 h 54 min**. Non-GPU: the v4 mixture dry run 6.6 min, the per-character backfill < 1 min.

§3.8 (2026-09-23/24): P0 1.7 min; bake-off generation 28.9 min (six candidates + determinism runs, incl. Falcon-H1's failed bf16 load); the dev reference row 9.5 min; bake-off judge 10.5 min; ceiling 1.8 min + judge 4.7 min; teacher generation 48.4 + 9.9 min; smoke 1.2 min; the three Phase 3 runs 85.4 + 68.1 + 69.0 min; diagnostics 2.2 min; main judge 5.9 min — **≈ 5 h 47 min**. Non-GPU: three planners 10 min, filters and pseudo-cells a few minutes. Downloads: 143 GB of teacher weights (Aya-Expanse-32B was cached).

§3.10 (2026-09-24/25): the scorer no-op check 2 × ~6 min; the six MCQ cells 2 h 54 + 2 h 35 + 2 h 41 + 2 h 40 + 2 h 47 + 2 h 29 min = 16 h 8 min (peak 68.6 GiB on the native cells, 16.4 GiB on the others) — **≈ 16 h 20 min**. Non-GPU: `mcq_compare.py` 2 min per run.

§3.11 (2026-09-25): twenty eval-only cells 199.9 min end to end (9.2–11.4 min each, ~4 min of it the HF dataset loads; peak 26.4 GiB on the native cells) — **≈ 3 h 20 min**. Non-GPU: the prompt pre-check 15.3 min (dataset loads), the analysis 8 s, `mcq_compare_msa` ~2 min, the query paths 42 s, the full test suite 5.4 min (CPU only).

§3.12 (2026-09-25): sixteen eval-only cells, Σ wall 166.2 min (P1a 82.1 + P1b 84.1 min end to end; 9.4–11.2 min each; peak 10.4 GiB) — **≈ 2 h 46 min**. Non-GPU: the analysis 13 s, the supplement and the audits < 1 min each, the config tests 4.1 min, the full test suite 6.5 min (1 578 passed, 2 skipped) (CPU only).

Analysis tab (2026-09-25, tooling): the local Gemma-4-31B server for the protocol and fp8 tests, 20:41–21:08 UTC — **≈ 27 min** (three cold starts of 2.5–4.5 min, 15 benchmark questions, one headless page check).

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
- Zheng, C., Zhou, H., Meng, F., Zhou, J. & Huang, M. (2024). *Large Language Models Are Not Robust Multiple Choice Selectors*. ICLR 2024. (Option-ID / position bias of MCQ selection and permutation-based debiasing — the family of the §3.11 proposal.)
- Model cards used for the bake-off: `CohereLabs/aya-expanse-32b`, `CohereLabs/c4ai-command-r7b-arabic-02-2025`, `ALLaM-AI/ALLaM-7B-Instruct-preview`, `casperhansen/llama-3.3-70b-instruct-awq`.

## 8. Where the numbers live
`outputs/experiments/qwen_native_vs_araroopat/<cell>/all_metrics.json` (metrics, judge summary, training histories), `eval_rows/freeform_cidar.parquet` (every generation), `freeform_judge/gemma4_31b.parquet` (verdicts), `diag_heldout_loss*.json` (LM loss / answer NLL), `embedding_init_probe.json`, `_p3_decode/roundtrip_v{2,3}_*.json`, `_report_2026-09-22/report_draft.md` (the 2026-09-22 hand-off report), `araroopat_3phase_v2/phase12_run_20260921-115522/` (v2's Phase 1/2 log). Configs: `configs/experiments/qwen_native_{no_presteps_after_LOOP_fix,sft_only,sft_reeval}.yaml`, `qwen_araroopat_3phase{,_v2,_v2_resume_sft,_v3}.yaml`, `qwen_bpe16k_3phase_v3.yaml`. Commits: `6a1d3db` (scheduler, templates, loop stop, SDPA), `6eee412` (periodic loop stop), `a2c671d` (markers + budget), `a879d73`/`f1ee2f5`/`3885dda` (console), `e06910e` (decode), `b204a54` (embedding init), `3ddf26c`…`5c91854` (v3 + comparator), `562da5e` (this record + the v2 configs), `cb68d34` (held-out raw-text set + the corrected loss numbers), `2615c37` (dev slices + `eval_mixture`), `77fded5` (the two ffstop configs + `paired_compare.py`).

§3.6 artifacts: `configs/contamination/rawtext_heldout_v1.jsonl` (+ `.manifest.json`, sha256 `c6522a4f0050702b…`), `<cell>/diag_heldout_rawtext_v1[_warmup][_base].json` (eight checkpoints of the earlier cells, two of the new ones), `<cell>/data/sft_eval_mixture_manifest.json`, `all_metrics.json["training"]["sft"]["eval_history"]`, `scripts/build_rawtext_heldout.py`, `scripts/judge/paired_compare.py`. The pre-2026-09-22 `diag_heldout_loss*.json` files are kept; they record what the old pool-tail rule measured.

§3.7 artifacts: listed under §3.7 *Artifacts*. Commits: `4efedb3` (the §3.6 verification edits), `e10863c` (answer NLL per character + backfill), `ab2e679` (decoding knobs + ablation configs), `a216601` (`paired_compare.py` across two folders), `4e702ce` (v4 + P4 configs), and the documentation commit of §3.7. The 19 backfilled `diag_heldout_*.json` files now carry `answer_nll_total_source: "backfill: …"`; v4's carries `"measured"`.

§3.8 artifacts: listed under §3.8 *Artifacts*. Commits: `c59c1e7` (uncertainty diagnostic), `17e790a` (distill code: bake-off, teacher generation, pseudo-cells, prompts, filter), `345bf51` (teacher-answer overlay + provenance), `02d0dd8` (the three v5 configs), and the documentation commit of §3.8. Teacher file sha256 `8e4d9cd13e3d1bc8a2cd0782266f570dc771a55f500e0609a1c808955bbcf272`.

§3.10 artifacts: listed under §3.10 *Artifacts*; configs `configs/experiments/mcq_v5/*.yaml`. Commits: `ff9f4b5` (§3.9), `2f56441` (scorer window fix, row-dump schema 2, `tests/test_lighteval_scorer_window.py`, the CLAUDE.md manifest path), `2441518` (the six configs), `4d405ca` (`scripts/mcq_compare.py` + tests), `7fb914b` (label-collapse diagnostic), and the documentation commit of §3.10.

§3.11 artifacts: listed under §3.11 *Artifacts*; configs `configs/experiments/mcq_letter_slot/` (twenty cells + `rows_arabic_exam.json` sha256 `fca4a176…`, `rows_culture_arabic_mmlu.json` `885d5bdb…`). Commits: `73bf45b` (the pending §3.10 edits), `c05aa1f` (`label_rotation`, `rows_file`, row-dump schema 3, tests, regenerated presets, `scripts/select_mcq_subset.py`), `0f37bae` (rows files + configs), `07c559b` (`scripts/mcq_letter_slot.py` + the MSA scope groups in `scripts/mcq_compare.py`, tests), `84eae66` (console test), and the documentation commit of §3.11.

§3.12 artifacts: listed under §3.12 *Artifacts*; configs `configs/experiments/mcq_letter_slot/{araroopat,bpe16k}_{p1,p2}_rot{0..3}.yaml`. Commits: `b87f9ac` (the §3.11 verification paragraph), `7b2540d` (`scripts/mcq_letter_slot.py --arms`, optional 0-shot, the trajectory table, tests), `a0b7798` (the sixteen configs), and the documentation commit of §3.12.

## 9. Cells of `outputs/experiments/qwen_native_vs_araroopat/`
Current: `native_qwen3_base_m2_c2400`, `native_qwen3_sft_m2_c2400`, `araroopat_3phase_v2`, `araroopat_3phase_v3`, `bpe_16k_3phase_v3`, **`araroopat_3phase_v3_ffstop`**, **`bpe_16k_3phase_v3_ffstop`** (§3.6: the same recipe with the free-form-aware early stop), **`araroopat_3phase_v4`** (§3.7: Phase 3 at 45 000 × 25/15/60, max_length 1 024 — judge 2.18, the highest AraRooPat cell but not separable from ffstop; the current best arms of each vocabulary are v4 / ffstop for AraRooPat and ffstop for BPE). Older rule sets, kept: `native_qwen3` (untrained base, v1 template), `native_qwen3_base`, `native_qwen3_sft`, `araroopat` (v1). Under `_superseded/`: `native_qwen3_base_after_loop_fix`, `native_qwen3_sft_reeval`, `native_qwen3_{base,sft}_m2_c1200`, the two lost v3 attempts.

New in §3.8: **`araroopat_3phase_v5_distill`**, **`bpe_16k_3phase_v5_distill`**, **`native_qwen3_sft_v5_distill`** (v4's Phase 3 on the Aya-Expanse-32B teacher answers; judge 2.43 / 2.13 / 2.52) and the pseudo-cell **`teacher_aya_expanse_32b_m2_c2400`** (the teacher on the test prompts, 3.65; no model). In `outputs/experiments/teacher_bakeoff/`: the six candidate pseudo-cells and `native_qwen3_base_dev` (the untouched base on the same 250 dev prompts).

In `outputs/experiments/qwen_decoding_ablation/` (§3.7 P1, eval only, repetition penalty 1.2): `araroopat_3phase_v3_ffstop_rp12`, `bpe_16k_3phase_v3_ffstop_rp12`, `native_qwen3_sft_rp12`, `native_qwen3_base_rp12`; `_repro/araroopat_3phase_v3_ffstop_greedy` (the greedy reproduction check, identical to its twin in 250 of 250 rows).

New in §3.10 (eval only, the four MCQ benchmarks at `max_length 4096`, schema-2 row dumps): **`araroopat_3phase_v5_distill_mcq4096`**, **`bpe_16k_3phase_v5_distill_mcq4096`**, **`native_qwen3_sft_v5_distill_mcq4096`** (the v5 Phase 3 checkpoints), **`native_qwen3_base_mcq4096`** (the untouched base), **`araroopat_3phase_v3_warmup_mcq4096`** and **`bpe_16k_3phase_v3_warmup_mcq4096`** (the v3 Phase 2 checkpoints). Comparison tables in `_mcq_compare/`, chain logs in `_mcq_logs/`.

New in §3.11 (eval only, schema-3 row dumps, in the sub-folder `_letter_slot/` so the twenty diagnostic cells stay out of the main cell list; `mcq_compare.py --experiment …/_letter_slot` reads them as cells of that folder): `<arm>_<cond>` for the arms `araroopat_v5`, `bpe16k_v5`, `native_v5`, `native_base` (the v5 Phase 3 checkpoints and the untouched base) and the conditions `rot0`–`rot3` (3-shot, letters rotated by k) and `0shot` — Arabic-Exam and Culture-MMLU only, on the 1 000-row subsets. `letter_slot.{md,json}`, `alghafa_query_paths.json`, `alghafa_dialect_split.json` sit in the same folder.

New in §3.12 (eval only, schema-3 row dumps, in `_letter_slot/` too): `araroopat_p1_rot0..3`, `araroopat_p2_rot0..3`, `bpe16k_p1_rot0..3`, `bpe16k_p2_rot0..3` — the Phase 1 (`embedding_alignment`, embedding-only) and Phase 2 (`warmup`) checkpoints of `araroopat_3phase_v3` / `bpe_16k_3phase_v3` (the checkpoints the v5 arms' Phase 3 started from) under rotations 0–3, 3-shot, Arabic-Exam and Culture-MMLU on the 1 000-row subsets. `letter_slot_trajectory.{md,json}` sits in the same folder.
