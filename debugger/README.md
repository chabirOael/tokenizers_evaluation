# AraRooPat Train Explorer

A local web page that replays `AraRooPatTokenizer.train()` on any Arabic text you type
and shows **every intermediate step** — the real code path, the real CAMeL subprocess,
nothing simulated.

## Start it

```bash
# one-time prerequisite (only if you have never used araroopat on this machine)
python -m venv .venv-camel && .venv-camel/bin/pip install -e ".[araroopat-camel]" && .venv-camel/bin/camel_data -i light

# run from the repo root
.venv/bin/python debugger/serve_araroopat_explorer.py --open
```

The CAMeL bridge is warmed at startup (~4–8 s), then the page opens at
<http://127.0.0.1:8765/>. `--port N` changes the port; `--no-warm` defers the CAMeL
spawn to the first request. Stop with Ctrl-C.

## What you see

Type a text (one line = one corpus text), optionally adjust the vocab budget, press
**Run trace**. Sixteen cards appear, in the exact order `train()` executes them, each
naming the source function (`file:line`) and its wall-time:

| # | Step | Function |
|---|---|---|
| 00 | parameters in effect vs. real defaults | `AraRooPatTokenizer.__init__` |
| 01 | Unicode NFKC normalization | `_corpus_prepass` |
| 02 | whitespace split | `str.split` |
| 03 | character classes → Arabic alpha chunks | `_classify_char`, `_extract_alpha_chunks` |
| 04 | dedup → `word_counts` | `Counter` |
| 05 | on-disk cache decision (bypassed here) | `_corpus_prepass` |
| 06 | batched NDJSON round-trip to CAMeL — the literal request/response lines | `CamelBridge.analyze` |
| 07 | per-word validation gates: radicals (`#` kept), ≥3, NTWS, Arabic guard, clitic tag→surface, `normalize_pattern` strip-by-strip incl. the لِ+الـ contraction | `_dict_to_analysis` |
| 08 | `CorpusEntry` records | `CorpusEntry.from_analysis` |
| 09 | root / pattern / proclitic / enclitic frequency tables | `train` |
| 10 | vocab assembly: ID ranges, `add()` order, budget kept/cut | `_build_vocab` |
| 11 | reconstruction table, passes 1–3 with tiers | `_build_reconstruction` |
| 12 | provenance metadata | `_build_metadata` |
| 13 | `encode()` of the input through the vocab just built | `encode` |
| 14 | `decode()` back to Arabic, tier per pair, aligned word diff | `decode` |
| 15 | cross-check against a fresh un-instrumented `train()` | `train` |

Playback: **Play / Prev / Next / Show all**, a clickable step rail, and a reveal-speed
selector. Each word in step 07 expands to show the raw CAMeL candidate dict and the
gate-by-gate decision.

## LLM emission playground (after the trace)

Below the steps, a playground lets you emit **any id sequence the way a language model
would** and run it through the real `decode()` of the tokenizer just trained (the server
keeps that instance; the trace response carries a `trace_id`):

- type ids and/or token strings, whitespace-separated (`1 6 119 128 2` or
  `[CLITICP_و] [ROOT_كتب] [PAT_1ِ2ا3ِ] </s>`), or click tokens in the vocab palette,
  **Prefill from encode** (step 13's stream), or **Random emission**;
- nothing is validated for "grammar" — orphan roots, `[PAT_*]` with no root, an unclosed
  literal, leading enclitics, special tokens mid-stream, ids outside the vocab (silently
  dropped by `decode()`), and `(ROOT, PAT)` pairs unseen in the corpus (→ CAMeL generator,
  tier 2) all go straight in;
- the result shows the decoded text, a token-by-token walkthrough whose **output-so-far**
  column is `decode(ids[:k])` for each prefix (retracted words struck through, new words
  highlighted — a pending root shows nothing until its `[PAT_*]` arrives), a per-pair tier
  table, the generator's NDJSON lines, and counters (orphan roots, PAT-without-root,
  unclosed literal, ignored ids, tier 1/2/3). The annotation column is an interpretation
  from the token family; only the prefix decodes are ground truth.
- `POST /api/decode {"trace_id", "items"}`; 409 if the server no longer holds that trace.
  Cap 512 tokens. Runs are embedded in a saved snapshot (frozen there — no server).

**Save as standalone HTML** downloads a single self-contained file with the results
embedded. It opens from `file://` with no server and no CAMeL, keeps playback, and is
read-only ("snapshot mode").

The second tab embeds the older static *encode* explainer (`docs/araroopat_explainer.html`),
unchanged. The third tab trains on / loads from the real corpus — see below.

## Corpus training (tab 03)

The third tab runs the same sixteen steps on a **real corpus** — `Jr23xd23/ArabicText-Large`,
loaded exactly as the pipeline loads it (`load_arabic_dataset` + `configs/base.yaml`
preprocessing) — or on a tokenizer already saved under `outputs/tokenizers/`. The job runs
in a background thread on **its own CAMeL subprocess**, so tab 01 stays usable; the page
polls `/api/corpus/status` (stage, done/total, ETA, log tail) and re-attaches after a reload.

Two sources, mirroring what is on disk:

| Source | What happens | Cost |
|---|---|---|
| **Train on the corpus** | the real pre-pass rule: `outputs/tokenizers/<cache dir>/corpus_analysis.pkl` is reused when its key matches — every chunk it holds is taken from it and only the chunks it lacks go through CAMeL (decision `hit` / `partial` / `miss`), then the union is written back; then the real `_build_vocab` / `_build_reconstruction` / `_build_metadata` | cache hit: ≈ 8 min (5 s load, ≈ 6.5 min NFKC + chunk + count over 669 K texts, 8 s pickle, ≈ 1 min build) + ≈ 7 min if *verify* is on (an un-instrumented `train()` re-chunks the corpus); fresh pre-pass: hours |
| **Load a saved tokenizer** | `load()` of a `save()` directory | seconds |

`cache policy` (`auto` / `ignore`) and `write cache` are exposed separately (the real
`train()` couples both in `cache_corpus_analysis`); `max rows` caps the train split for a
quick fresh run.

**Same cards, corpus scale.** Every step keeps its id, title, `file:line` and renderer.
Each card starts with a *corpus scale* banner: corpus-wide totals plus what is shown —
a seeded sample (6 texts, 25 words each; `sample size` chunks; top-200 counts; top-150
frequency rows; the top-60 budget candidates plus ±30 rows around the cut; 200 random
reconstruction entries). Where the real run sent nothing to CAMeL (cache hit), the IPC /
gates / peeler cards **re-analyze the sample live** and say so; each replayed word carries
a `= pre-pass entry` / `≠ pre-pass entry` badge against the cached record. The encode →
decode proof runs on the **probe text**, not the corpus. In saved mode the eight pre-pass
cards say they are not reconstructable from artifacts (frequencies come from
`vocab_metadata.json`, kept tokens only); the verify step re-loads the directory.

**Patterns, human-readable.** Every `[PAT_*]` token in the tab gets its **wazn** next to
it (`مَ1ْ2َ3َ` ≙ `مَفْعَلَ`: slots 1/2/3/4 → ف/ع/ل/ل), a hover with an approximate **gloss**
from a closed list (`describe_pattern` in `araroopat_corpus_trace.py`; exact diacritized
wazn first, then the diacritic-stripped skeleton, labelled *loose*) and the first corpus
example **filled** into the template. The *patterns as* control in the play bar switches the
whole tab between CAMeL slots, wazn, or both. Slots missing from a template (hollow /
defective / `#` roots) are pointed out in the search results.

**Search.** `GET /api/corpus/search?q=…&family=…` over the whole vocab: an id; a bracketed
token prefix; root letters (`#` or a weak letter matches a masked radical — `قول` finds
`ق#ل`); a wazn with or without tashkeel; a gloss word (`participle`); and a whole word, which
is also **encoded live** (the exact token sequence, click to send to the playground) and
looked up in the reconstruction table (`reconstructs 'X'` hits). Family chips, paging,
`+ playground` per hit.

**Encode → decode, and why a word went where it went.** *Encode plain text* runs the real
`encode()` on any text (`POST /api/corpus/encode`): the token stream, its `decode()` round-trip,
and every alpha chunk classified by the path it took — the exact all-or-nothing rule of
`_emit_alpha` (`categorize` in `araroopat_corpus_trace.py`): `ROOT+PAT`, `ROOT+PAT (peeled)`,
`PREP`, `LIT: no analysis`, `LIT: root cut`, `LIT: pattern cut`, `LIT: clitic / particle token
missing`. Filter the chunks by path, expand any of them (**process ▾**, `GET
/api/corpus/word_trace?word=`) to replay it live: CAMeL's candidates with every
`_dict_to_analysis` gate (tab 01's cards), the peeler's slicings, the vocab check with the
budget loop's own reason (rank vs `max_patterns`, freq vs `min_pattern_freq`, or never a
candidate), and the tokens `encode()` emits — plus the chunk's corpus record when there is one.
*Corpus words by category* (`GET /api/corpus/words`, train mode only — a saved directory has
no per-word records) lists every unique chunk of the corpus with its path, sorted by
occurrences, with a substring filter, a *peeled only* toggle, paging and the same process
view. **Send to decode** copies any stream into the playground.

**Every record, paged.** The corpus-scale cards keep their seeded samples but no longer stop
there. In the *validate* card every counter (`analyzed`, `rejected → LIT`, `ROOT+PAT`, `PREP`,
`peeled`, `LIT`) is a button that opens a **records browser** inside the card: every unique
chunk on that pre-pass path, by occurrences, with the analysis the pre-pass stored, the
category `encode()` gives it under the built vocab (an analyzed word can still land in LIT
when its root or pattern was cut by the budget), a substring filter, page sizes 25–200,
first / prev / next / last, jump-to-page, and **process ▾** on every row. The *peel* card's
`rescued` / `exhausted` counters open the same browser on the peeled / LIT paths; the
*entries* card is the browser in full-record mode (all eleven `CorpusEntry` columns). The
counters are **exclusive pre-pass paths** (`prepass_path`: lit → peeled → prep → root_pat), so
they sum to the number of unique chunks; a peeled particle counts as *peeled*. Backed by
`WordCategoryIndex`, which now keeps every `CorpusEntry` field for every chunk (shared
strings interned — ≈ 260 MB for the 1.09 M chunks of the full corpus, measured) so the 2.6 GB
of `CorpusEntry` objects can still be dropped after the trace; the last filtered index list is
cached so paging a substring search does not rescan the million rows. `GET
/api/corpus/words?path=root_pat|prep|peeled|lit|analyzed|rejected&full=1&…`. The *freq* card's
`root_freq` and `pat_freq` tables are server-paged browsers over **every candidate**
(`FreqIndex`, `GET /api/corpus/freq?kind=root|pat|prc|enc|prep&q=&kept=&limit=&offset=`): rank,
key with its wazn, frequency bar, **kept / cut** against the vocab (chips), contributing words,
and a search box — roots with the token search's wildcard rule (`قول` finds `ق#ل`), patterns
by CAMeL slots *or* wazn, tashkeel ignored (`مفعول`, `1ا2`). In saved mode the same browsers
read `vocab_metadata.json` (kept tokens only). The process view now also badges the live
analysis against the stored pre-pass record (`= pre-pass entry` / `≠ pre-pass entry`).

**Floating navigation.** A fixed panel at the right edge of the column lists *source & run*,
*progress*, the sixteen step cards and the five tool cards; the entry of the card under the
reading line is highlighted while you scroll, steps playback has not revealed yet are dimmed
(clicking one reveals it), the panel collapses to a glyph strip (button, or automatically under
1650 px) and hides under 1200 px. It only exists inside tab 03.

**Playground and save.** The LLM emission playground is a second instance of tab 01's
(`POST /api/corpus/decode {corpus_id, items}`, same result shape) on the corpus tokenizer;
**Save** writes the instance with the real `save()` to `outputs/tokenizers/<name>` (refuses
to overwrite unless ticked), so it appears in the *saved* list next time. Snapshots are not
supported for this tab (it needs the server).

Routes: `GET /api/corpus/sources`, `POST /api/corpus/train` (202; 409 while a job runs),
`GET /api/corpus/status`, `POST /api/corpus/cancel`, `GET /api/corpus/trace`,
`GET /api/corpus/search`, `POST /api/corpus/encode`, `GET /api/corpus/words` (category / path /
substring / peeled, `full=1` for every CorpusEntry field), `GET /api/corpus/freq`,
`GET /api/corpus/word_trace`, `POST /api/corpus/decode`, `POST /api/corpus/save`.

## Budget defaults

The page defaults `min_root_freq = min_pattern_freq = 1` (the real default is 2) so a
short text still yields a non-empty vocab. Set them to 2 to watch the `break` in the
budget loop cut everything below the threshold (step 10).

## Files

| File | Role |
|---|---|
| `debugger/serve_araroopat_explorer.py` | stdlib `http.server`; `GET /` page, `GET /explainer` old page, `GET /api/health`, `POST /api/trace`, `POST /api/decode` |
| `src/arabic_eval/tokenizers/araroopat_trace.py` | the instrumented replay; calls the real helpers and asserts its explanation equals the real result |
| `src/arabic_eval/tokenizers/araroopat_corpus_trace.py` | tab 03: `CorpusTraceJob` (background, own bridge), wazn / gloss / `describe_pattern`, `TokenIndex.search`, `WordCategoryIndex` (every chunk's record + category), `FreqIndex` (paged, searchable frequency tables), `list_sources`, `save_trained` |
| `docs/araroopat_train_explorer.html` | the page (vanilla JS, no build step); tab 03 is an additive `cx-*` module at the end of the file |
| `tests/test_araroopat_corpus_trace.py` | wazn / gloss / validation, the JS wazn map mirrors the Python one, a full train → cache hit → search → save → load run on an injected mini corpus (live bridge) |

No new Python dependencies. Trace requests are serialized with a lock because the
CAMeL bridge is single-threaded.
