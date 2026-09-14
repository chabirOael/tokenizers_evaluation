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
unchanged.

## Budget defaults

The page defaults `min_root_freq = min_pattern_freq = 1` (the real default is 2) so a
short text still yields a non-empty vocab. Set them to 2 to watch the `break` in the
budget loop cut everything below the threshold (step 10).

## Files

| File | Role |
|---|---|
| `debugger/serve_araroopat_explorer.py` | stdlib `http.server`; `GET /` page, `GET /explainer` old page, `GET /api/health`, `POST /api/trace`, `POST /api/decode` |
| `src/arabic_eval/tokenizers/araroopat_trace.py` | the instrumented replay; calls the real helpers and asserts its explanation equals the real result |
| `docs/araroopat_train_explorer.html` | the page (vanilla JS, no build step) |

No new Python dependencies. Trace requests are serialized with a lock because the
CAMeL bridge is single-threaded.
