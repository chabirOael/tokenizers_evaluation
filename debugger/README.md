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
`PREP`, `FUNC`, `CLITIC`, `PROP` (a database proper noun between `[PROP_BEGIN]` / `[PROP_END]`),
`LIT: no analysis`, `LIT: root cut`, `LIT: pattern cut`, `LIT: clitic / particle token
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
there. In the *validate* card every counter (`analyzed`, `rejected → LIT`, `ROOT+PAT`, `PREP`, `FUNC`,
`CLITIC`, `PROP`, `peeled`, `LIT`) is a button that opens a **records browser** inside the card: every unique
chunk on that pre-pass path, by occurrences, with the analysis the pre-pass stored, the
category `encode()` gives it under the built vocab (an analyzed word can still land in LIT
when its root or pattern was cut by the budget), a substring filter, page sizes 25–200,
first / prev / next / last, jump-to-page, and **process ▾** on every row. The *peel* card's
`rescued` / `exhausted` counters open the same browser on the peeled / LIT paths; the
*entries* card is the browser in full-record mode (all eleven `CorpusEntry` columns). The
counters are **exclusive pre-pass paths** (`prepass_path`: lit → peeled → prep / func → root_pat), so
they sum to the number of unique chunks; a peeled particle counts as *peeled*. Backed by
`WordCategoryIndex`, which now keeps every `CorpusEntry` field for every chunk (shared
strings interned — ≈ 260 MB for the 1.09 M chunks of the full corpus, measured) so the 2.6 GB
of `CorpusEntry` objects can still be dropped after the trace; the last filtered index list is
cached so paging a substring search does not rescan the million rows. `GET
/api/corpus/words?path=root_pat|prep|func|peeled|lit|analyzed|rejected&full=1&…`. The *freq* card's
`root_freq` and `pat_freq` tables are server-paged browsers over **every candidate**
(`FreqIndex`, `GET /api/corpus/freq?kind=root|pat|prc|enc|prep|func&q=&kept=&limit=&offset=`): rank,
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

---

# Experiment Console

A second local page: configure an experiment YAML in a form, list the existing
ones under `configs/experiments/`, start the run, watch it, cancel it. A started
run is **detached** — closing the tab, the SSH session or the server itself does
not stop it; only *Cancel* (or `kill`) does.

## Start it

```bash
.venv/bin/python debugger/serve_experiment_console.py --open     # http://127.0.0.1:8766/
```

`--port N` changes the port (the AraRooPat explorer uses 8765). The server needs
nothing beyond the main `.venv`; export `HF_TOKEN` *before* starting it — runs
inherit the server's environment, and the header shows whether the token is set.

## Configs tab

| Element | What it does |
|---|---|
| **list** (left) | every `configs/experiments/*.yaml`: model, tokenizer cells (named as `run_sweep` names them — `bpe_32k`, `charformer`, …), eval tasks, enabled phases (`P1 P2 P3`, `–` = disabled), and a results chip (`results` / `7/8 cells`) when `all_metrics.json` already exists under its `output_dir`. Invalid files are listed in red with the Pydantic error. |
| **Form** | the *resolved* config (the file merged over `base.yaml`, exactly what `load_config` produces), rendered from the Pydantic JSON schema. Hand-built widgets for the parts that matter: sweep tokenizer cells (type from the registry, `vocab_sizes`, params prefilled from `configs/tokenizers/<type>.yaml`, reorder / remove), the eval-task checklist with per-task params, the three phase cards (enable toggle in the header, dataset checklist, mix-mode hint showing `steps = mix_tokens / (batch_size × block_size)`), model presets from `configs/models/`. Renaming the experiment renames `output_dir` while it still follows `outputs/experiments/<name>`. Params boxes take flat `key: value` YAML. Every field label carries a `?`: hovering it shows what the field means and a concrete example (the `HINTS` table at the top of the page script, keyed by config path with `*` for phase names and list indexes). |
| **Validate** | `POST /api/config/validate` — the real merge + Pydantic; errors are painted on the offending fields by their `loc` path, the rest listed in the banner. On success the banner names the cells, whether `--sweep` applies, and which cells `run_sweep` would skip because results exist. |
| **YAML** | the file's own text when a config is opened (comments intact); *Form → YAML* renders the form in **full** style (every value explicit, like `all_tokenizers_sweep.yaml`) or **delta** style (only differences from `base.yaml`, like the native_llama files) — both reload to the same config, a test pins that for every file in the repo. *YAML → Form* parses the pane back (invalid YAML still lands in the form, with errors). |
| **Results** | per-cell downstream accuracy (PMI when available), MEI, fertility / compression / RPS read from `all_metrics.json`, plus links to `comparison_report.txt` and `experiment.log`. |
| **Save…** | writes `configs/experiments/<name>.yaml`; refuses to overwrite unless ticked; refuses an invalid config. Rendering from the form drops the comments of a hand-written file, so save under a new name unless you mean to replace it. |
| **▶ Start run…** | validates, then shows what will run (source, cells, tasks, phases, cells that will be skipped) and the CLI overrides: `--sweep` (auto = more than one tokenizer cell, as `run_experiment.py` requires), `--device`, `--seed`. One run at a time by default — two full-FT jobs do not fit the H100; *run concurrently* overrides. An unsaved form is started from a snapshot, no file needed. |

## Runs tab

Every run is a directory `outputs/runs/<YYYYmmdd-HHMMSS>_<config>/`:

| File | Content |
|---|---|
| `run.json` | pid / pgid / process start ticks (defeats PID reuse), argv, timestamps, status, exit code |
| `config.yaml` | the config **as launched** — the run reads this snapshot, so editing the file afterwards never changes a running job |
| `console.log` | stdout + stderr of `run_experiment.py` (the pipeline's own `outputs/logs/<name>/experiment.log` is unchanged and linked too) |
| `exit_code` | written by the launching shell when the process ends, so the status survives a server restart |

The launch is `bash -c 'trap … TERM; python scripts/run_experiment.py … > console.log 2>&1; echo $? > exit_code'`
with `start_new_session=True`: the run is its own session leader. The server
reconciles status from the directory alone (`running` / `cancelling` /
`finished` / `failed` / `cancelled` / `lost` = process gone without an exit
code, e.g. a reboot), so a restarted server lists and controls every run.

The detail panel parses `console.log` into: sweep cells (✓ done, ↷ skipped
because results existed, ● current, ✗ failed), the pipeline stage (`Step N/7`),
the current phase with step / loss / lr / eval_loss and a progress bar, the
current eval task with its tqdm count, the last traceback line on failure, the
results table once `all_metrics.json` exists, and a live tail of the log
(incremental polling every 3 s, `\r` progress lines collapsed). **Cancel** sends
SIGTERM to the process group and SIGKILL 15 s later if it is still alive.

## Eval rows tab

Every scored row of every benchmark, as the model saw it. Reads the Parquet
dumps at `outputs/experiments/<sweep>/<cell>/eval_rows/<task>.parquet` that a
run writes when `evaluation.eval_row_dump` is on (the default in `base.yaml`).

| Element | What it does |
|---|---|
| **experiment / tokenizer / benchmark** | the three pickers, discovered from the filesystem. The tokenizer list shows each cell's registry key and vocab size; the benchmark list shows its row count. |
| **scoring** | which normalization decides the prediction, and therefore which rows count as correct: `primary` (PMI when the run computed it), `pmi`, `char`. The headline accuracy always follows this choice, so it agrees with the outcome filter. |
| **all / correct / wrong** | the outcome filter. |
| **sub-config / flag / sort / search** | sub-config counts follow the current filter. Flags are `truncated` (every choice scored the sentinel), `sentinel`, `hit the cap`, `closest call` (top two within 0.01), `scorings differ`. Sorts: row order, closest call first, most decisive first, and by distance from the gold answer. Search hits the question, or the whole prompt with *prompt too* ticked. |
| **summary strip** | rows selected, accuracy under the chosen scoring, both normalizations side by side, the **prediction histogram** (one tall bar means the model is riding a prior rather than reading the question), and the flag shares — each flag chip is clickable and applies itself as a filter. |
| **table** | one row per eval example: index, sub-config, question, expected, received, margin (`score[pred] − score[gold]`, so 0 whenever the model was right), flags. Click a row to expand it. |
| **row detail** | the **exact prompt** that was fed to the scorer, in a plain-text block with a `¶ spaces` toggle that renders spaces and newlines (the leading space on a continuation is load-bearing), its length in the tokenizer's own unit against `max_length`, and the per-choice table: every continuation with its raw log-likelihood, char-normalised and PMI scores, gold and picked marked. A row whose prompt filled the cap carries an explicit note that the pick is an artifact of truncation, not a decision. |
| **Export CSV** | the rows currently selected, as UTF-8-BOM CSV for a spreadsheet. |

A benchmark still being evaluated shows a banner instead of a table: a Parquet
file has no footer until it closes, so its rows become readable when that
benchmark finishes. The row count comes from the `<task>.progress.json` the
writer keeps beside it.

For an experiment that finished before the dump existed, rebuild one without
retraining:

```bash
.venv/bin/python scripts/dump_eval_rows.py --cell outputs/experiments/<sweep>/<cell> --max-rows 300
.venv/bin/python scripts/dump_eval_rows.py --sweep outputs/experiments/<sweep>          # every cell, full
```

It reloads the cell's tokenizer and phase checkpoint and re-runs only the
scoring. Because scoring is deterministic it also re-checks the archived
accuracy and says whether it reproduces.

Routes: `GET /api/schema`, `GET /api/configs`, `GET /api/configs/get?path=`,
`GET /api/configs/results?path=`, `POST /api/config/validate`, `POST /api/config/render`,
`POST /api/config/parse`, `POST /api/configs/save`, `GET /api/runs`, `GET /api/runs/<id>`,
`GET /api/runs/<id>/log?offset=`, `POST /api/runs/start`, `POST /api/runs/<id>/cancel`,
`GET /api/gpu`, `GET /api/text?path=` (files under `outputs/` only),
`GET /api/eval/tree`, `GET /api/eval/describe?path=`, `GET /api/eval/rows?path=&…`,
`GET /api/eval/row?path=&position=`, `GET /api/eval/export?path=&…`.

## Files

| File | Role |
|---|---|
| `debugger/serve_experiment_console.py` | stdlib `http.server`, thin routing |
| `debugger/experiment_console.html` | the page (vanilla JS, no build step) |
| `src/arabic_eval/tools/experiment_console.py` | schema bundle, config list / read / validate / render / save, `parse_progress`, `RunManager` |
| `src/arabic_eval/tools/eval_rows_browser.py` | eval-row discovery, filtering, paging, CSV export, path guard |
| `src/arabic_eval/evaluation/eval_rows.py` | the dump format: record builder, streaming Parquet writer, reader |
| `scripts/dump_eval_rows.py` | rebuild a finished experiment's dump (eval only, no training) |
| `tests/test_eval_rows.py` | record semantics, write → read invariants (accuracy and prompt fidelity), filters and paging, discovery, export, path guard |
| `tests/test_experiment_console.py` | round-trip of every repo config in both styles, `loc`-tagged errors, save guards, progress parser on real log lines, run lifecycle with a stub command (start, exit code, conflict, cancel with SIGTERM → SIGKILL, rediscovery by a fresh manager, PID-reuse guard) |
