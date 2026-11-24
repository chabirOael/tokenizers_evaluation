# Where every number of the introduction comes from

Draft of 2026-11-08, extended 2026-11-10 for the v2 introduction. The v1 sections below still hold for the numbers they list (the tokenizer facts, the AraRooPat − BPE-16K, BPE-128K − BPE-16K and multiple-choice numbers); the paragraphs are now: 1 hook, 2 prior work, 3 the gap and the measure, 4 AraRooPat and the protocol, 5 the judge, 6 four findings as bullets, 7 contributions. **The v1 rows that compare against `native` (the native arm trained through the pipeline) are no longer used**: since 2026-11-10 the reference is the model as shipped, and the *v2 additions* section at the end lists those numbers. `D` = `outputs/experiments/_comparative_tables/2026-10-13/` (the CSVs behind `docs/comparative_tables_2026-10.md`, built 2026-10-13; read as CSV · row · column). `report` = `docs/report.md`. `bench` = `outputs/experiments/qwen_native_vs_araroopat/_briefs/opus_prompt_tokenizer_benchmark_2026-11-08.md`. Judge differences are first − second, 95 % paired bootstrap intervals (10 000 resamples, seed 0) on the 250 held-out prompts, judge gpt-oss-120b unless stated (`D/<model>_T2b.csv` source line: `scripts/judge/paired_compare.py` through `arabic_eval.analysis.judge_compare`).

## Paragraph 1 (hook)

| number in the text | source |
|---|---|
| BPE of 16 000 tokens | `D/qwen3_4b_T1.csv` · "vocabulary size (pieces)" · BPE-16K = 16 000 |
| 2.30, 2.26, 2.08 tokens per word (native Qwen3, Llama, Gemma) | `D/qwen3_4b_T1.csv`, `D/llama3b_T1.csv`, `D/gemma3_4b_T1.csv` · "pieces per word ↓ (fertility, ArabicText-Large eval texts)" · native |
| 1.50 (the Arabic BPE-16K) | same row · BPE-16K (identical in the three files) |
| seven LLMs | brief §2 (Alakeel et al. 2026: seven existing LLMs and their tokenizers) |
| the quotation of Alakeel et al. | brief §2, verbatim from their abstract |
| بالقلم → بالق + لم (BPE-16K), بال + ق + لم (Qwen3) | not a number; see *Examples* below (encoded 2026-11-08) |

## Paragraph 3 (AraRooPat)

| number in the text | source |
|---|---|
| 17 184 tokens | `D/qwen3_4b_T1.csv` · "vocabulary size (pieces)" · AraRooPat |
| 99.3 % of 2 000 corpus words | `D/qwen3_4b_T1.csv` · "single words given back exactly (of 2 000 corpus words)" · AraRooPat = 99.3 % (word list `outputs/data_cache/araroopat_roundtrip_words_2000_42.json`, per `D/numbers_sources.md`); report §3.5 (96.40 % → 99.30 %) |
| 250 reference answers | `configs/contamination/freeform_cidar_heldout_v1.jsonl` (250 lines) |
| chrF 99.0 with diacritics removed | `D/qwen3_4b_T1.csv` · "the same with the answers' diacritics removed first" · AraRooPat = 99.0; report §3.5 (98.97) |
| 91.1 with diacritics | `D/qwen3_4b_T1.csv` · "text given back exactly after encode → decode: reference answers, character overlap (chrF, 0–100)" · AraRooPat = 91.1; report §3.5 (91.10) |
| 41.8 % of words where possible | `D/qwen3_4b_T1.csv` · "roots kept inside one piece, where possible ↑ (root conservation, attainable)" · AraRooPat = 0.418 (Llama file 0.417, Gemma 0.418: different morph samples) |
| 5.3 % BPE-16K | same row · BPE-16K = 0.053 (Qwen file) |
| 8.8 % Qwen3 | same row · native = 0.088 (Qwen file) |
| 3.19 tokens per word (AraRooPat), 1.50 (BPE-16K), 2.30 (Qwen3) | `D/qwen3_4b_T1.csv` · "pieces per word ↓ (fertility …)" |
| "by construction" | the T1 note under `D/qwen3_4b_T1.csv` in `docs/comparative_tables_2026-10.md` ("high by construction on the words it writes as root + pattern") |

## Paragraph 4 (the protocol)

| number in the text | source |
|---|---|
| 131 M tokens | `configs/experiments/qwen_bpe128k_3phase_v5_distill.yaml` header: Phases 1 + 2 = 32.8 M + 98.3 M mix tokens (= 131.1 M); report §3.5 ("131 M-token adaptation budget") |
| 45 000 examples, 27 000 free-form | `configs/experiments/qwen_araroopat_3phase_v5_distill.yaml` lines 226–227 (`total_examples: 45000`, shares 0.25 / 0.15 / 0.60 → 11 250 / 6 750 / 27 000); `outputs/experiments/qwen_native_vs_araroopat/araroopat_3phase_v5_distill/all_metrics.json` · `training.sft.data.mixture` · free_form quota = kept = 27 000 |
| five rules | report §3.8 *The five rules* (wording condensed; rule 3's list: no EOS, over 2 400 characters, loop, degenerate, under 20 characters, Latin letter, template / instruction echo) |
| 2 400-character budget | `configs/experiments/qwen_araroopat_3phase_v5_distill.yaml` line 267 (`max_output_chars: 2400`); `CLAUDE.md` *Free-form eval* |
| 250 held-out CIDAR instructions; paraphrase screen | `configs/contamination/freeform_cidar_heldout_v1.jsonl` (250 rows); `CLAUDE.md` *Free-form eval* §1 (E5 near-duplicate gate 0.93) |
| 1 to 5 | rubric `default_v1` (`src/arabic_eval/judge/freeform_judge.py`; `CLAUDE.md` *Free-form eval* §3) |
| κ 0.817 over 24 cells | report §3.14, J5 (line "Over the 24 cells both judged, pooled QWK is **0.817** (6 000 verdicts, exact 0.579)") |
| 95 % bootstrap intervals | `D/*_T2b.csv` source note (10 000 resamples, seed 0) |
| BPEs of 16 000 and 128 000 tokens, same corpus | `D/qwen3_4b_T1.csv` · "vocabulary size (pieces)"; `outputs/tokenizers/{araroopat_maxpat40k_v3,bpe_16k_v3,bpe_128k_v1}/training_provenance.json` · `dataset_name` = `Jr23xd23/ArabicText-Large`, `num_texts` = 668 959 in all three |

## Paragraph 5 (generation results)

| number in the text | source |
|---|---|
| AraRooPat − BPE-16K: +0.212 [+0.088, +0.340] (Qwen3) | `D/qwen3_4b_T2b.csv` · "AraRooPat − BPE-16K" · "gpt-oss-120b — main judge"; `D/all_T7.csv` · Qwen3-4B-Base · G1 |
| +0.132 [+0.004, +0.256] (Llama) | `D/llama3b_T2b.csv` · same row/column; `D/all_T7.csv` · Llama · G1 |
| −0.016 [−0.144, +0.108] (Gemma) | `D/gemma3_4b_T2b.csv` · same; `D/all_T7.csv` · Gemma · G1 |
| "under all four judges, intervals above zero under three" (Qwen3) | `D/all_T7b.csv` · "J6 AraRooPat − BPE-16K": gpt-oss +0.212 [+0.088, +0.340]; Falcon-H1-34B +0.072 [−0.068, +0.216]; Gemma-4-31B +0.300 [+0.152, +0.452]; GPT-5.6 +0.172 [+0.048, +0.300] |
| AraRooPat − native: −0.004 [−0.132, +0.120] (Llama) | `D/llama3b_T2b.csv` · "AraRooPat − native" · gpt-oss; `D/all_T7.csv` · G2 |
| −0.152 [−0.284, −0.016] (Qwen3) | `D/qwen3_4b_T2b.csv` · "AraRooPat − native" · gpt-oss |
| −0.284 [−0.416, −0.152] (Gemma) | `D/gemma3_4b_T2b.csv` · "AraRooPat − native" · gpt-oss |
| BPE-16K − native −0.364, −0.136, −0.268, every interval below zero | `D/{qwen3_4b,llama3b,gemma3_4b}_T2b.csv` · "BPE-16K − native" · gpt-oss: −0.364 [−0.496, −0.232]; −0.136 [−0.268, −0.008]; −0.268 [−0.400, −0.136] |
| native-sized BPE improves on BPE-16K on none (−0.064 to −0.164) | `D/*_T2b.csv` · "BPE-128K − BPE-16K" · gpt-oss: −0.064 [−0.192, +0.064] (Qwen3); −0.100 [−0.220, +0.016] (Llama); −0.164 [−0.288, −0.032] (Gemma); `D/all_T7.csv` · R-128-1 |
| loops 6.8 vs 15.6 % (Qwen3), 12.8 vs 18.4 % (Llama), 5.6 vs 8.4 % (Gemma) | `D/{qwen3_4b,llama3b,gemma3_4b}_T2_continued.csv` · "answers stuck in a loop ↓" · AraRooPat vs "BPE, 16 000 pieces" |
| lowest answer NLL per character among re-learned vocabularies, all three models | same files · "surprise on the reference answers, nats per character ↓": Qwen3 AraRooPat 0.812 / BPE-16K 0.907 / BPE-128K 0.979 / native 0.811; Llama 0.844 / 0.962 / 1.025 / native 0.901; Gemma 0.761 / 0.861 / 0.926 / native 0.760 |
| Qwen3: 0.812 nats; BPE-16K 0.907, native 0.811 | `D/qwen3_4b_T2_continued.csv` · same column |
| "only on Llama can it not be separated from the model's own vocabulary" | G2 above (Llama −0.004 [−0.132, +0.120]; Qwen3 and Gemma intervals below zero) |

## Paragraph 6 (recognition)

| number in the text | source |
|---|---|
| two letter-scored benchmarks; native first, AraRooPat last on all three models (plain scoring) | `D/{model}_T3.csv` · "Arabic school exams (ArabicMMLU): plain score", "translated MMLU (Culture-Arabic-MMLU): plain score": AraRooPat is the lowest of the four trained arms and native the highest in all six model × benchmark cells (Qwen3 0.550 / 0.467 vs native 0.621 / 0.512; Llama 0.398 / 0.317 vs 0.442 / 0.355; Gemma 0.444 / 0.340 vs 0.504 / 0.392); `D/all_T7.csv` · G3 |
| 0.013 to 0.046 behind BPE-16K, every interval below zero | `D/{model}_T3b.csv` · "AraRooPat − BPE-16K", columns exams / MMLU × plain / prior-corrected: Qwen3 −0.046, −0.034, −0.026, −0.036; Llama −0.021, −0.022, −0.024, −0.030; Gemma −0.018, −0.013, −0.029, −0.021 (all intervals below zero). Note: report §4 finding 10 gives the upper end as −0.047 — the §3.10 `scripts/mcq_compare.py` figure — while T3b's rebuilt value is −0.046; the text follows the table. The two are not in conflict (verifying session, 2026-11-08) |
| أ in three tokens vs one | `D/*_T1.csv` · "the letter أ written as a choice answer, in pieces" · AraRooPat 3, others 1 |
| on Qwen3 the deficit follows the letter to every slot | report §3.11 and §4 finding 9 (AraRooPat − BPE-16K on gold letter أ −0.202 / −0.149, on gold slot 1 −0.027 / −0.052) |
| exam gap not separable from zero on any model; Qwen3 −0.001 [−0.030, +0.027] | `D/{model}_T3c.csv` · "Arabic exams: difference to BPE-16K, letter-fair" · AraRooPat: Qwen3 −0.001 [−0.030, +0.027]; Llama +0.010 [−0.024, +0.045]; Gemma −0.025 [−0.059, +0.009]; report §4 finding 10 |
| MMLU gap −0.026 to −0.039, below zero on Qwen3 and Llama | `D/{model}_T3c.csv` · "translated MMLU: difference to BPE-16K, letter-fair" · AraRooPat: Qwen3 −0.039 [−0.068, −0.011]; Llama −0.039 [−0.071, −0.007]; Gemma −0.026 [−0.056, +0.003] |

## Paragraph 7 (the pending benchmark)

| number in the text | source |
|---|---|
| Jais-2: 150 222 entries, 150 272-row table | bench §1 ("150 222 entries (150 000 base + 224 added), the model's embedding table has **150 272** rows"); report §3.18 |
| Fanar-1-9B: 128 256 tokens, 4 773 with an Arabic letter | bench §1; report §3.18 |
| Farasa-BPE-16K: 16 000 tokens, trained as BPE-16K | bench §1 / §5; report §3.18 ("Trained by us on the corpus, preprocessing, trainer settings and special tokens of `bpe_16k_v3`") |
| W1–W5, F1–F3 | bench §9; report §3.18 *Pre-registered readings* |

## Contributions

Every number repeats one above: 17 184, 99.3 %, 41.8 %, 3.19 (paragraph 3); "up to four judges" (paragraph 5, `D/all_T7b.csv`).

## Examples (not numbers), as encoded on 2026-11-08

Tokenizers: `outputs/tokenizers/bpe_16k_v3/tokenizer.json` (HF `tokenizers`), `Qwen/Qwen3-4B-Base` (`AutoTokenizer`, HF cache, offline, `add_special_tokens=False`), `outputs/tokenizers/araroopat_maxpat40k_v3` (the `araroopat` class through the CAMeL bridge; token strings from its `reverse_vocab.json`; BOS/EOS not shown). A word's pieces depend on whether a space precedes it (byte-level BPE), so the text's examples are taken **inside a sentence**:

```
SENTENCE: الطلاب يكتبون الدرس وبالكتاب
  bpe_16k_v3:          ['ال', 'طلاب', ' يكتبون', ' الدر', 'س', ' وبال', 'كتاب']
  Qwen/Qwen3-4B-Base:  ['ال', 'طل', 'اب', ' ي', 'كتب', 'ون', ' ال', 'درس', ' وبال', 'كت', 'اب']
  araroopat:           [CLITICP_ال] [ROOT_طلب] [PAT_1ُ2ّا3ِ] [ROOT_كتب] [PAT_يَ1ْ2ُ3ُونَ] [CLITICP_ال] [ROOT_درس] [PAT_1َ2ْ3ِ]
                       [CLITICP_و] [CLITICP_ب] [CLITICP_ال] [ROOT_كتب] [PAT_1ِ2ا3]   | decode: الطلاب يكتبون الدرس وبالكتاب
SENTENCE: يكتب الطلاب الدرس بالقلم
  bpe_16k_v3:          ['يك', 'تب', ' الطلاب', ' الدر', 'س', ' بالق', 'لم']
  Qwen/Qwen3-4B-Base:  ['ي', 'كتب', ' الطلاب', ' ال', 'درس', ' بال', 'ق', 'لم']
  araroopat:           [ROOT_كتب] [PAT_يُ1ْ2َ3] [CLITICP_ال] [ROOT_طلب] [PAT_1ُ2ّا3ِ] [CLITICP_ال] [ROOT_درس] [PAT_1َ2ْ3ِ]
                       [CLITICP_ب] [CLITICP_ال] [ROOT_قلم] [PAT_1َ2َ3ِ]   | decode: يكتب الطلاب الدرس بالقلم
```

Single words, same AraRooPat tokenizer: `قال` → `[ROOT_ق#ل] [PAT_1ا3َ]` (the weak radical as `#`); `للولد` → `[CLITICP_ل] [CLITICP_ال] [ROOT_#لد] [PAT_وَ2َ3]` (the لِ + الـ contraction undone); each decodes to itself.

How the text renders them: the pattern tokens are written with ف/ع/ل in the slots 1/2/3 (`يَ1ْ2ُ3ُونَ` → يَفْعُلُونَ, `1َ2َ3ِ` → فَعَلِ), and the clitic tokens are written in full, as in the vocabulary (`[CLITICP_ب] [CLITICP_ال]`; follow-up brief §2.1). Two cautions found while checking, neither used in the text: standalone (no preceding space) the cuts differ — BPE-16K writes `يكتبون` alone as يك + تب + ون and Qwen3 writes `كتاب` alone as كت + اب, but inside a sentence BPE-16K keeps ` يكتبون` whole and Qwen3 keeps ` كتاب` whole; and AraRooPat read sentence-initial `يكتب` as the passive يُكْتَب (`[PAT_يُ1ْ2َ3]`, CAMeL's top reading), although it decodes back to يكتب.

## v2 additions (2026-11-10) — the off-the-shelf reference, the judge, the benchmark numbers and the new citations' facts

`B` = `outputs/experiments/_tokenizer_benchmark/`; `rep` = `B/m7/replication.md` (A − B, paired bootstrap, 10 000 resamples, seed 0; the same numbers in `B/paper/tables/P3_preregistered_differences.md`); `base` = the untrained base model with its own vocabulary on the same 250 prompts: cells `qwen_native_vs_araroopat/native_qwen3_base_m2_c2400`, `llama3b_tokenizers/native_base_m2_c2400`, `gemma3_4b_tokenizers/native_base_m2_c2400`. `X − base` pairs: `B/m6/paired/<model>/<judge>/<X>__minus__base.json` (the new arms' files from the campaign; `araroopat`, `bpe16k`, `bpe128k` written 2026-11-10 with the same `paired_compare.compare`). Verified against the files by the Fable session on 2026-11-10 (report §3.18 *Verification*).

| number in the text | source |
|---|---|
| the five-point scale, the sub-scores and the flags (paragraph 5) | `RUBRICS["default_v1"]` in `src/arabic_eval/judge/freeform_judge.py`, quoted in substance; cell score = `summarize_verdicts` (mean of the 250 integer scores; SE = population std ÷ √250); differences = `scripts/judge/paired_compare.py` (`paired_bootstrap`: mean per-prompt difference, 10 000 resamples, seed 0, win / tie / loss) |
| κ 0.817 over 24 cells | `D/qwen3_4b_T2c.csv` (as in v1) |
| base judge 2.376, loops 6.8 % (Qwen3-4B-Base off the shelf) | `base` cell `all_metrics.json` · `downstream.freeform_cidar.judge.gptoss_120b.score_mean` 2.376 (± 0.076), `loop_stop_rate` 0.068 |
| Llama and Gemma base loop on 33 % and 55 % | `base` cells · `loop_stop_rate` 0.332 and 0.552 (EOS 0.54 / 0.068) |
| AraRooPat − base −0.064 [−0.200, +0.068] (Qwen3); Farasa-BPE −0.128; Jais-2 −0.132; BPE-16K −0.276 [−0.428, −0.128]; BPE-128K −0.340 [−0.492, −0.188]; Fanar-1 −0.212 [−0.344, −0.084] | `B/m6/paired/qwen/gptoss_120b/{araroopat,farasa_bpe16k,jais2,bpe16k,bpe128k,fanar1}__minus__base.json` (Farasa [−0.268, +0.012], Jais-2 [−0.276, +0.012]) |
| "under the second judge only BPE-128K does" | `B/m6/paired/qwen/falcon_h1_34b/*__minus__base.json`: AraRooPat +0.000, Farasa +0.056, Jais-2 −0.104 [−0.244, +0.036], Fanar-1 −0.020, BPE-16K −0.072, BPE-128K −0.344 [−0.496, −0.192] |
| "+0.26 to +0.77" (Llama and Gemma, every adapted arm − base, gpt-oss); AraRooPat +0.768 [+0.644, +0.900], +0.412 [+0.276, +0.548] | `B/m6/paired/{llama3b,gemma3_4b}/gptoss_120b/*__minus__base.json`: Llama +0.768 / +0.636 / +0.536 / +0.764 / +0.608 / +0.684, Gemma +0.412 / +0.428 / +0.264 / +0.364 / +0.284 / +0.432 (AraRooPat / BPE-16K / BPE-128K / Farasa-BPE / Jais-2 / Fanar-1) |
| "six tokenizers compared on classification left no single best choice" | Alyafeai et al. 2023, abstract (three new tokenizers + three baselines; "none of the tokenization techniques is the best choice overall"), verified 2026-11-10 on the Springer page |
| MorphBPE "keeps BPE merges inside morpheme boundaries at training time and leaves inference unchanged", "improved reading comprehension for Arabic" | Asgari et al. 2026 (Findings of ACL 2026), abstract: merges restricted during training, inference unchanged; "on the Belebele … MorphBPE yields significant improvements in … Russian and Arabic" |
| AraBERT "pre-segmented its text with Farasa" | Antoun et al. 2020, paper body (the pre-segmented AraBERTv1); **not in the abstract** — cite the section when the paper is typeset |
| "traced to data volume rather than to morphology" | Arnett and Bergen 2025 (COLING 2025), abstract: no evidence for morphological alignment; "Differences in performance can be attributed to disparities in dataset size" |
| "track human preference well" | Zheng et al. 2023 (MT-Bench), abstract: strong LLM judges match human preferences at over 80 % agreement |
| AraGen / 3C3H "the direction Arabic evaluation has taken" | El Filali et al. 2024, Hugging Face blog (no paper exists; arXiv search 2026-11-10 returned nothing) |
| 150 222 entries, 150 272 rows (Jais-2); 128 256 tokens, 4 773 Arabic-bearing (Fanar-1) | `B/m2/released_tokenizers_check.json` · `len`, `table_rows`, `arabic_bearing_pieces_decoded` |
| 65.5 % (variant B), 0.90 (the gate), 96.0 % (variant A) | `B/m3/gate.json` · `measured_variant_B` 0.6548, `threshold` 0.9; `B/m4/intrinsic/farasa_bpe_16k_v1a.json` · `morpheme_integrity_rate` 0.9596 |
| 6.3 µs per word vs 3.6 | `B/m4/speed/speed_v1.json` · cases `farasa_bpe_16k_v1a` encode `us_per_word` 6.323; `farasa_bpe_16k_v1` 3.576 |
| +0.148 [+0.040, +0.264], +0.128 [+0.008, +0.248], −0.064 [−0.192, +0.064] (Farasa-BPE-16K − BPE-16K, gpt-oss) | `rep` W1 farasa_bpe16k − bpe16k, gptoss_120b; Falcon +0.128 / +0.028 / −0.056 ("agreeing in sign on every model") |
| "level everywhere (within ±0.07 under either judge)" | `rep` W2 farasa_bpe16k − araroopat: −0.064 / −0.004 / −0.048 (gpt-oss), +0.056 / +0.000 / +0.032 (Falcon) |
| −0.068, −0.160 [−0.292, −0.024], −0.128 [−0.248, −0.004] (Jais-2 − AraRooPat) | `rep` W2 jais2 − araroopat, gptoss_120b |
| −0.148 [−0.268, −0.028], −0.084, +0.020 (Fanar-1 − AraRooPat) | `rep` W2 fanar1 − araroopat, gptoss_120b (Llama interval [−0.200, +0.036], Gemma [−0.096, +0.136]) |
| "every new arm matches or beats the native-sized BPE-128K" | `rep` W4 rows: all 18 point estimates ≥ +0.020; fanar1 and farasa_bpe16k − bpe128k replicate under Falcon |
| 3.7 % Arabic-bearing (Fanar-1); 2.4–3.2 % (natives) | `B/m12/arabic_bearing_all.json` · fanar1_128k 0.03721; native_qwen3 0.02397, native_llama3b 0.02953, native_gemma3 0.03166 |
| 56 % / 38 % / 3 % never seen in adaptation | `B/m6/adaptation_token_counts.json` · `share_seen_0`: fanar1 0.5614, jais2 0.3790, farasa_bpe16k 0.0282 |
| "cannot write a line break" | `B/paper/threats.md` §11; `B/m4/tokenizer_basics.json` (reference round trip 0.768 for BPE-16K / BPE-128K / Farasa-BPE; answers with a line break 0 for every arm of ours against 175–204 for the shipped and native arms, `docs/report.md` §3.18) |
| "leads … on five of six model comparisons, three of them with intervals that exclude zero" | `rep` W2: Jais-2 − AraRooPat negative on all three models (intervals exclude zero on Llama and Gemma); Fanar-1 − AraRooPat negative on Qwen (excludes zero) and Llama, +0.020 on Gemma |
| 0.812 vs 0.907 nats (answer NLL per character, Qwen3) | as v1 (`D/qwen3_4b_T2_continued.csv`); the native value is no longer quoted |

| "the letter's own tokenization moves multiple-choice accuracy by up to 11 % and reshuffles model rankings" | Sanz-Guerrero et al. 2025 (arXiv 2509.15020; EMNLP 2025 main per the arXiv comment), abstract: "accuracy differences of up to 11% due to this (seemingly irrelevant) tokenization variation as well as reshuffled model rankings" |
| "selecting an option is a different skill from producing an answer" | Balepur et al. 2025 (ACL 2025, pp. 3394–3418), abstract: MCQA "struggles to: 1) test generation/subjectivity; 2) match LLM use cases"; advocates generative formats |
| "log-likelihood scoring misaligns with the model's own generated answers" | Lyu et al. 2024 (arXiv 2402.13887), abstract: "the prevalent probability-based evaluation method inadequately aligns with generation-based prediction" |
| "multiple-choice and open-ended results disagree across models" | Wei et al. 2024 (arXiv 2403.07872), abstract: "the inconsistency between the MCQA evaluation and the generation of open-ended responses" (24 models, 11 benchmarks) |
| "free-form answers matched against a reference measure generative ability where option formats cannot" | Myrzakhan et al. 2024 (arXiv 2406.07545): open-style questions "eliminate selection bias and random guessing"; Chandak et al. 2025 (arXiv 2507.02856): answer matching, "a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers" |
| "A tokenizer's effect on downstream performance is itself stronger on generation than on classification" | Goldman et al. 2024 (Findings of ACL 2024), abstract: the compression–performance correlation is stronger for generation tasks (and smaller models) |
| "the fourteen otherwise identical models of TokSuite" | Altıntaş et al. 2025 (arXiv 2512.20757; ICML 2026 per the arXiv comment), abstract: "fourteen pre-trained models that use different off-the-shelf tokenizers but are otherwise identical" |
| Lotz et al. 2025 in "compression alone does not predict performance" | ACL 2025, pp. 32155–32173, abstract: "tokenizer choice has negligible effects on tasks in English but results in consistent performance differences in multilingual settings"; proposes intrinsic metrics that correlate better than compression |
| EQUATOR (arXiv 2501.00257) — **not cited** | its abstract makes no MCQ-vs-generation correlation claim (verified 2026-11-10); kept in `references_pool.bib` with that note |

Tokenization examples unchanged from v1 (re-encoded 2026-11-08). The judge differences of findings 1 and 3 are the 2026-10 numbers of the v1 tables (`D/<model>_T2b.csv`, `T3b`, `T3c`).
