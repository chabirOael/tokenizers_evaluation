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
| **Form** | the *resolved* config (the file merged over `base.yaml`, exactly what `load_config` produces), rendered from the Pydantic JSON schema. Hand-built widgets for the parts that matter: sweep tokenizer cells (type from the registry, `vocab_sizes`, params prefilled from `configs/tokenizers/<type>.yaml`, reorder / remove), the eval-task checklist with a **typed card per task** (see the next row), the three phase cards (enable toggle in the header, dataset checklist, mix-mode hint showing `steps = mix_tokens / (batch_size × block_size)`, a `mixture` sub-card for the ratio-controlled Phase 3 with its own hint `steps = total_examples / batch_size`, and the mixture summarised in the card badge), `corpus_params` (per-corpus loader parameters such as `aya_ar.include_datasets`), model presets from `configs/models/`. The `output_dir` row carries an **infer from name** switch: on, the field is locked to `outputs/experiments/<name>` and follows every rename (also after *YAML → Form* or an assistant edit); off, any path goes. It is re-derived from the config on load, so the 14 conventional configs open with it on and the campaign cells (`outputs/experiments/qwen_native_vs_araroopat/<cell>`) open with it off — renaming those never touches their directory. Params boxes take flat `key: value` YAML. Every field label carries a `?`: hovering it shows what the field means and a concrete example (the `HINTS` table at the top of the page script, keyed by config path with `*` for phase names and list indexes). |
| **task cards** (in *sweep → eval tasks*) | one card per ticked task, one typed row per parameter the task **declares** in its `param_spec()` (`/api/schema` → `task_params`; `TaskConfig.params` itself stays `Dict[str, Any]`): bools as switches, ints / floats as bounded mono inputs, `choices` as selects, `stop_markers` as a one-marker-per-line textarea (`\n` escaped), paths as text; every label's `?` shows the spec's help and default. A row whose key is **absent from the file shows the code default greyed**; only a value that differs from the default is written, typing the default back removes the key, and the **📌 pin** keeps a key explicit at its default (`set` = written because it differs, `pinned` = written by choice). Advanced rows hide behind *more…*; groups (`budget` / `stops` / `scoring` / `misc` on the free-form task) are headings. Keys the task does not declare are listed under *⚠ not declared by this task — ignored at run time* with a delete button and are never dropped silently; **raw** toggles the flat `key: value` textarea over the same object; **defaults** clears the card. The old *preset* button copied `configs/tasks/<type>.yaml` — a file **a run never reads** (params come from `sweep.tasks[].params` only); those files are now generated from the specs by `scripts/render_task_presets.py`. |
| **Validate** | `POST /api/config/validate` — the real merge + Pydantic; errors are painted on the offending fields by their `loc` path, the rest listed in the banner. Warnings (advisory, the run would proceed) join the banner: a task param the task does not declare or of the wrong type / out of range (`param_warnings`, also painted on the row), an `output_dir` another saved config already writes to (the working file itself is excluded through the request's `file`). On success the banner names the cells **that will actually run**, whether `--sweep` applies, and which cells `run_sweep` would skip because results exist. A single `sweep.tokenizers` cell is *not* a sweep: `run_experiment.py` then trains the top-level `tokenizer` block and never reads the sweep cell — the banner and the start dialog warn when the two disagree (a real run trained `native_qwen3` while the page said `1 cell (araroopat)`). |
| **YAML** | the file's own text when a config is opened (comments intact); *Form → YAML* renders the form in **full** style (every value explicit, like `all_tokenizers_sweep.yaml`) or **delta** style (only differences from `base.yaml`, like the native_llama files) — both reload to the same config, a test pins that for every file in the repo. *YAML → Form* parses the pane back (invalid YAML still lands in the form, with errors). |
| **Results** | per-cell downstream accuracy (PMI when available), MEI, fertility / compression / RPS read from `all_metrics.json`, plus links to `comparison_report.txt` and `experiment.log`. |
| **Save…** | writes `configs/experiments/<name>.yaml`; refuses to overwrite unless ticked; refuses an invalid config. Rendering from the form drops the comments of a hand-written file, so save under a new name unless you mean to replace it. |
| **⧉ Clone…** (toolbar, or the `⧉` on a list row) | copies a *saved* file under a new name: new file name, experiment name (follows the file name until edited), `output_dir` with the same *infer from name* switch, description, overwrite tick. The dialog warns when the target file exists, when the `output_dir` is already another config's (two configs writing one directory share results and `run_sweep` skips finished cells), and when the form has unsaved edits (the clone copies the file on disk). `POST /api/configs/clone` rewrites only the `name` / `output_dir` / `description` lines of the source's own text — comments, order and quoting stay (a continued scalar is folded, a missing key is inserted below the block's own); it re-parses the result and, when the layout defeats the rewriter (flow mapping, block scalar) or the copy would not resolve to exactly the intended config, writes a rendered delta instead and says so (`comments_kept: false`). Every file in `configs/experiments/` clones through the rewriter (a test pins it). The clone opens in the editor. |
| **▶ Start run…** | validates, then shows what will run (source, cells, tasks, phases, cells that will be skipped) and the CLI overrides: `--sweep` (auto = more than one tokenizer cell, as `run_experiment.py` requires), `--device`, `--seed`. One run at a time by default — two full-FT jobs do not fit the H100; *run concurrently* overrides. An unsaved form is started from a snapshot, no file needed. |
| **provenance** | every config carries `experiment.created_at` (stamped when the file is first saved from the console — Save… of a new name, Clone…) and `experiment.runs` (one `{started_at, run_id, source}` per start: the console appends to the source file after launching, `scripts/run_experiment.py` appends `source: cli` when its `--config` lives under `configs/experiments/` — a console run passes the `outputs/runs/<id>/config.yaml` snapshot, which is skipped, so nothing is logged twice). Both are edited in the file's own text (`arabic_eval.config_edit`: the block's lines only, comments kept, both PyYAML's and our list-item indentation accepted) and shown read-only: the *provenance* row of the experiment section (run stamps link to the Runs tab) and a `created … · N runs, last …` line on every list row. A clone starts with a fresh `created_at` and no runs. Files from before 2026-09-20 were backfilled once with `scripts/backfill_config_provenance.py` (`created_at` from the commit that added the file, runs from the `run.json` records). |
| **✨ assistant** | opens the assistant drawer (third column): an LLM that fills the form from a prompt, edits it, or answers questions about the fields — see *Config assistant* below. |

## Config assistant (added 2026-09-18)

*✨ assistant* in the editor toolbar opens a chat drawer next to the form. Three things
it does, decided from the message: **create** a config from a description ("BPE 16K and
32K vs native Llama, three phases, PMI scoring, a smoke-sized run"), **modify** the
config in the form ("switch Phase 2 to the pretraining mix with a 7 % QA blend"), and
**answer** questions about the fields ("what must mix_tokens equal?"). It never saves or
starts anything: the result lands in the form and you review, save and start it as usual.

| Element | What it does |
|---|---|
| **picker + status chip** | `configs/assistant/*.yaml`, each an OpenAI-compatible `/chat/completions` endpoint (`model`, `base_url`, `api_key_env`, `temperature`, `max_tokens`, `extra_body`, `history_messages`). The chip is the readiness check: `ready` (the key named by `api_key_env` is exported in the console server's environment), `local · up / down` (a `base_url` on localhost is pinged). `gpt56_terra_api.yaml` (the API model) is the one shipped. A local model works through the same client — start a `vllm serve` by hand and add a YAML with its `base_url` and `api_key_env: null`; measured once on `google/gemma-4-31b-it` from `.venv-judge`, the ~16 K-token prompt needs `--kv-cache-dtype fp8 --max-model-len 24576`, and such a server holds the whole GPU (the *Start run* dialog warns when the GPU is already more than half full with no console run active). |
| **transcript** | your messages and the model's replies (minimal markdown). A reply that changes the config carries the **change list** (`path: old → new`), the validation outcome and, once applied, an **undo** button (restores the form exactly, one level per reply, newest first). An invalid result shows its errors and *apply anyway*. Each reply also shows prompt / completion tokens and the wall time. The transcript survives a reload (localStorage); undo does not. |
| **new config from base.yaml** | tick it to ignore the form and build a new config from the defaults; the result opens as an unsaved config named by the model. It unticks itself after a successful create so the next message edits that config. |
| **quick prompts** | smoke test · no Phase 3 · every tokenizer · why invalid? · explain phases. |
| **show context** | the exact messages the next turn would send, with their size — the debugging view when the model misreads something. |
| **clear** | forgets the conversation (and aborts a reply in flight). |

How a turn works (`src/arabic_eval/tools/config_assistant.py`, route `POST /api/assistant/chat`,
an SSE stream): the page sends the config in the form (or `from_base`), the last
`history_messages` messages and the new one. The server builds the prompt — a
hand-written **primer** of the platform (pipeline, cells, tokenizer families, the
cross-field rules the validator enforces, conventions, the smoke recipe), the **field
reference** rendered from `src/arabic_eval/tools/config_hints.py` (the same table behind
the form's `?` tooltips, served by `/api/schema`), `base.yaml` with its comments
stripped, the registries and tokenizer presets, every `configs/experiments/*.yaml` as a
delta over `base.yaml`, then the **working config as a delta**, its validation state
and the history (≈ 12–14 K tokens; the static part comes first so an endpoint's prompt
cache hits). The model answers in markdown and, when the request changes the config,
ends with one fenced block tagged `edits`: a YAML mapping of **dotted config paths to
new values** (`training.phases.sft.enabled: false`; a value replaces the node it names,
lists and mappings whole, `null` allowed, list indexes allowed) — ~10× cheaper than a
whole config and exact for a modification. The server applies the edits, runs the real
validation (`validate_config`: merge over `base.yaml` + Pydantic) and, on failure,
sends the errors back to the model **once** for a repair (shown in the transcript). The
final event carries the resolved config, the change list and the validation outcome; a
valid one is applied to the form immediately. `POST /api/assistant/preview` returns
the assembled messages; `GET /api/assistant/configs` the picker's list.

Tests: `tests/test_config_assistant.py` — endpoint configs and readiness, the SSE / chat
client on a fake `urlopen`, the context builder on the real repo (every hint path
present, cache invalidation), the reply parser, the dotted-path edits, and whole turns
with a scripted client through the real validation (plain answer, valid edits, the
repair round, giving up after it, a broken block, a create from base); a test also pins
that every hint key names a real schema field.

## Runs tab

Every run is a directory `outputs/runs/<YYYYmmdd-HHMMSS>_<config>/`:

| File | Content |
|---|---|
| `run.json` | pid / pgid / process start ticks (defeats PID reuse), argv, timestamps, status, exit code, the `plan` (enabled phases + steps, tasks, eval flags) |
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

**Layout** (2026-09-18): the detail panel takes the whole width; the run list is a
floating card at the left edge (*☰ runs* in the run header, `×` to hide, remembered
in localStorage) and the step panel one at the right edge, so with both closed the
log has the full window. Under 1100 px both dock above / below the detail instead.
The run list is sortable (2026-09-20): *sort by* `started` (default, newest first) ·
`experiment` · `status` (running → cancelling → failed → lost → finished → cancelled) ·
`duration` (a running run counts up to now) · `kind`, with a `▼/▲` direction toggle;
ties fall back to the start time so the order is stable across the 3-second poll, and the
choice is remembered in localStorage (`xc-runs-sort`).

**Log colouring.** Each line of `console.log` is tokenised by
[Prism](https://prismjs.com)'s `log` grammar (MIT; core + `prism-log`, pinned 1.30.0
from cdnjs with SRI hashes — when the CDN is unreachable the log is plain text, nothing
else changes): dates and times, log levels, logger names, `key=value` numbers, quoted
strings, file paths and URLs, `====` separators. The tokens are mapped onto the page's
colour variables, so the dark theme holds. On top of that the page tints whole lines
for what Prism cannot know: a `WARNING` / `ERROR` head, a Python traceback (from
`Traceback (most recent call last):` down to the un-indented exception line), the
`Step N/7` / `SWEEP cell` / `Experiment … done` headlines, and tqdm bars (dimmed). The
bar above the log has a text filter (case-insensitive substring, live), *hide progress
bars*, a taller-log toggle (⤢, most of the window) and *jump to the end* (⤓); a line is
coloured once when it arrives and cached, so a 3 000-line tail costs nothing per poll.

**Step panel** (added 2026-09-18, *☰ steps* in the run header, open by default,
remembered in localStorage). A floating card at the right edge that shows the
selected run's *whole* plan as an outline and follows it live: for a sweep the
cells (`✓` done, `↷` skipped, `✗` failed, `●` current, `k/n`), then the six pipeline
stages — `load corpus`, `tokenizer` (with its sub-step: *train on N texts* or *load
from …*), `intrinsic eval`, `load + adapt model`, `training` with the three phases
underneath (skipped ones struck through, the running one with step / steps, a bar,
loss, lr, `eval_loss`; the pending ones with their step budget), `downstream eval`
with every task (the running one with its row count and bar, finished ones with their
time) — and a footer (`● running · so far`, `✓ done`, `✗ failed`, `■ cancelled`). The
current stage and sub-step are highlighted; finished ones carry their duration, taken
from the log's own timestamps (the running one against the server clock). Stages the
config turns off (`intrinsic_metrics` / `downstream_metrics: false`) are omitted. A
judge run shows judges → cells → report instead. The *plan* (which phases are
enabled and their `steps`, the tasks, the eval flags) is recorded in `run.json` at
start and derived from the config snapshot for runs that predate it; the *progress*
comes from `parse_progress`, which now also records the first / last timestamp of
every stage, phase and task. On screens narrower than the two-column grid the card
docks under the run detail.

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
`GET /api/eval/row?path=&position=`, `GET /api/eval/export?path=&…`, `GET /api/assistant/configs`, `POST /api/assistant/chat` (SSE), `POST /api/assistant/preview`.

## Files

| File | Role |
|---|---|
| `debugger/serve_experiment_console.py` | stdlib `http.server`, thin routing |
| `debugger/experiment_console.html` | the page (vanilla JS, no build step) |
| `src/arabic_eval/tools/experiment_console.py` | schema bundle, config list / read / validate / render / save, `parse_progress`, `RunManager` |
| `src/arabic_eval/tools/config_hints.py` | the field documentation (`path → (meaning, example)`) behind the form tooltips and the assistant's field reference |
| `src/arabic_eval/tools/config_assistant.py` | the config assistant: endpoint config + readiness, streaming chat client, context builder, `edits` parser, dotted-path apply, the chat turn with its repair round |
| `configs/assistant/*.yaml` | assistant endpoints (the API model) |
| `src/arabic_eval/tools/eval_rows_browser.py` | eval-row discovery, filtering, paging, CSV export, path guard |
| `src/arabic_eval/evaluation/eval_rows.py` | the dump format: record builder, streaming Parquet writer, reader |
| `scripts/dump_eval_rows.py` | rebuild a finished experiment's dump (eval only, no training) |
| `tests/test_eval_rows.py` | record semantics, write → read invariants (accuracy and prompt fidelity), filters and paging, discovery, export, path guard |
| `tests/test_experiment_console.py` | round-trip of every repo config in both styles, `loc`-tagged errors, save guards, progress parser on real log lines, run lifecycle with a stub command (start, exit code, conflict, cancel with SIGTERM → SIGKILL, rediscovery by a fresh manager, PID-reuse guard) |

## Free-form and Rate tabs (added 2026-09-18)

Two tabs for the free-form eval (`freeform_cidar`; see *Free-form eval* in `CLAUDE.md`).

- **Free-form** — every cell with `eval_rows/freeform_cidar.parquet`, joined by `id` with its `freeform_judge/<judge>.parquet` files (`src/arabic_eval/tools/freeform_rows_browser.py`, routes `/api/freeform/tree|rows|row`). Pick experiment → tokenizer → judge; filter on judge score range, judge flag, stop reason (eos / marker / cap), row flag (loop, empty, Latin, hit the cap, char-cut), length stratum and free text; sort by judge score, judge disagreement, chrF or length. The summary strip shows chrF, BERTScore, each judge's mean with a 1–5 histogram, the loop / empty / cap counts and the stop reasons, each clickable as a filter: the active chip is highlighted and a second click clears it, the *N of M rows* chip names every filter in force and an **× all rows** chip (shown only while something is filtered) drops them all at once — the dropdowns above follow. A row opens with instruction, reference, generation (raw too when the stop-marker cut changed it), every judge's verdict and rationale, the per-row metrics, and **the same prompt in every other cell** of the experiment. Cells whose tokenizer cannot generate are listed with the typed reason.
- **Rate** — the blind human check (`src/arabic_eval/tools/freeform_rating.py`, routes `/api/rating/sets|build|items|submit|agreement`). *Build a set* draws N prompts (half random, half where the judges disagree most) × K variants per prompt (baseline + others, round-robin), shuffles the items and keeps the cell only in `<experiment>/freeform_rating/<set>.json`. Type a rater name, press *rate*: one item at a time (instruction, reference, candidate), score 1–5 on the judge's rubric, flags, a note; ratings go to `<set>.ratings/<rater>.json` and resume where you left off. *Agreement* reveals the cells: rater vs each judge (Spearman, exact, quadratic-weighted κ, mean |Δ|), rater vs rater, judge vs judge on the same items, and the per-cell human mean beside the judge means (`<set>.agreement.json`).

**Judge runs from the console** (2026-09-18). The Free-form tab's *judge…* button lists `configs/judges/*.yaml` with a readiness check (`vllm` judges need `.venv-judge` and the extracted Python headers — `scripts/judge/setup_judge_env.sh`; `openai` judges need their `api_key_env` exported in the environment the console server was started from), a baseline cell, an optional per-cell limit, *overwrite* and *run concurrently*. *Start judge run* posts to `/api/judge/start`, which snapshots the judge YAMLs into `outputs/runs/<run_id>/judges/` and launches `scripts/judge/run_judge.sh` (any GPU judge) or `scripts/judge/judge_freeform.py` (API judges only) as a detached run with the same lifecycle as an experiment: `run.json` carries `kind: judge`, the Runs tab parses the judge log into per-judge / per-cell chips, *cancel* works, and a finished run shows the per-cell judge means and links to `freeform_judge_report.json` and the regenerated comparison report. A GPU judge conflicts with an active run like an experiment does; API-only judges start alongside.
