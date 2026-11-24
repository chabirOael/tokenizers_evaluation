# Where every number of the introduction comes from

Draft of 2026-11-08. `D` = `outputs/experiments/_comparative_tables/2026-10-13/` (the CSVs behind `docs/comparative_tables_2026-10.md`, built 2026-10-13; read as CSV · row · column). `report` = `docs/report.md`. `bench` = `outputs/experiments/qwen_native_vs_araroopat/_briefs/opus_prompt_tokenizer_benchmark_2026-11-08.md`. Judge differences are first − second, 95 % paired bootstrap intervals (10 000 resamples, seed 0) on the 250 held-out prompts, judge gpt-oss-120b unless stated (`D/<model>_T2b.csv` source line: `scripts/judge/paired_compare.py` through `arabic_eval.analysis.judge_compare`).

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
