"""Per-word UNK occurrence reporting (opt-in CSVs).

Used by two opt-in flags in ``EvaluationConfig``:

  * ``intrinsic_unk_report`` — dumps a CSV listing the source words whose
    per-word encoding produced at least one UNK token in the intrinsic
    eval split. The scalar ``unk_rate`` is unchanged; this adds provenance.

  * ``downstream_unk_report`` — during LightEval MCQ evaluation, scans the
    prompt and every continuation per example and writes a per-task CSV
    of the source words that produced UNK in any of those fields.

Both code paths share the same scan + aggregate helpers and the same
``WordUnkRecord`` shape; the CSV column sets differ (intrinsic carries
``total_token_count`` because each row is a per-word encoding count;
downstream carries ``source_fields`` and ``num_examples_seen_in`` because
each unique word may appear in multiple fields and multiple eval rows).

Design notes:

* Per-word encoding (whitespace split → ``tokenizer.encode(word)``) is the
  same convention the existing ``unk_rate`` loop uses, so the intrinsic
  CSV is exactly the underlying list for the existing scalar.

* Tokenizers without a usable ``unk_token`` id (byte-level Charformer,
  most ByteLevel BPE configurations under Llama-3.2-1B) produce *zero*
  occurrences. Callers should still write a header-only CSV in that case
  so every (tokenizer, task) pair produces a file — easy to diff across
  sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from arabic_eval.tokenizers.base import BaseTokenizer

# Field-name constants (single source of truth). Pipeline + helpers
# reference these so a column rename is a one-line change.
INTRINSIC_UNK_FIELDS: Sequence[str] = (
    "word",
    "unk_token_count",
    "total_token_count",
    "example_context",
)
DOWNSTREAM_UNK_FIELDS: Sequence[str] = (
    "word",
    "unk_token_count",
    "source_fields",
    "num_examples_seen_in",
    "example_context",
)

# Truncation cap for the ``example_context`` column — full eval prompts can
# be thousands of characters; we don't want CSVs that Excel chokes on.
_DEFAULT_CONTEXT_MAX_CHARS = 200


@dataclass
class WordUnkOccurrence:
    """One per-word UNK detection within a single text scan."""
    word: str
    unk_token_count: int
    total_token_count: int
    source_field: str
    example_context: str


@dataclass
class WordUnkRecord:
    """Aggregated per-word record across all occurrences."""
    word: str
    unk_token_count: int = 0
    total_token_count: int = 0
    source_fields: Set[str] = field(default_factory=set)
    num_examples_seen_in: int = 0
    example_context: str = ""


def scan_text(
    tokenizer: BaseTokenizer,
    text: str,
    *,
    source_field: str = "text",
    context_max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
) -> List[WordUnkOccurrence]:
    """Encode each whitespace word of ``text`` individually and yield one
    occurrence per word whose encoding contains the tokenizer's UNK id.

    Returns ``[]`` (rather than raising) when the tokenizer has no usable
    ``unk_token`` entry — callers handle the empty-CSV case via the
    aggregator. Empty / whitespace-only strings also return ``[]``.

    Per-word encoding is the same convention the existing ``unk_rate``
    loop in ``compute_intrinsic_metrics`` uses; that keeps the CSV
    consistent with the scalar metric. Cross-word merges (rare in
    well-trained subword tokenizers, occasional in ByteLevel BPE) would
    require token-to-word alignment via ``aligned_token_offsets`` — out
    of scope for this opt-in debugging report.
    """
    unk_id = tokenizer.special_tokens.get("unk_token")
    if unk_id is None or not text:
        return []
    snippet = text[:context_max_chars]
    out: List[WordUnkOccurrence] = []
    for word in text.split():
        if not word:
            continue
        enc = tokenizer.encode(word)
        unk_count = enc.input_ids.count(unk_id)
        if unk_count <= 0:
            continue
        out.append(
            WordUnkOccurrence(
                word=word,
                unk_token_count=unk_count,
                total_token_count=len(enc.input_ids),
                source_field=source_field,
                example_context=snippet,
            )
        )
    return out


def aggregate_occurrences(
    occurrences_by_example: Iterable[Iterable[WordUnkOccurrence]],
) -> Dict[str, WordUnkRecord]:
    """Bucket occurrences per unique word.

    Each top-level iterable element is one "example":
      * intrinsic mode: one text from the eval split → all its
        per-word occurrences
      * downstream mode: one eval row → all occurrences from its prompt
        plus its continuations

    ``num_examples_seen_in`` is incremented exactly once per example that
    contained the word (deduped within an example). ``source_fields`` is
    the union of all fields the word appeared in. ``example_context``
    keeps the *first* non-empty snippet (deterministic since callers
    iterate examples in order).
    """
    records: Dict[str, WordUnkRecord] = {}
    for ex_occurrences in occurrences_by_example:
        seen_this_example: Set[str] = set()
        for occ in ex_occurrences:
            r = records.setdefault(occ.word, WordUnkRecord(word=occ.word))
            r.unk_token_count += occ.unk_token_count
            r.total_token_count += occ.total_token_count
            r.source_fields.add(occ.source_field)
            if not r.example_context:
                r.example_context = occ.example_context
            seen_this_example.add(occ.word)
        for w in seen_this_example:
            records[w].num_examples_seen_in += 1
    return records


def records_to_rows(
    records: Iterable[WordUnkRecord],
    fieldnames: Sequence[str],
) -> List[Dict[str, Any]]:
    """Format records into ``DictWriter``-ready rows.

    Rows are sorted by descending ``unk_token_count`` (most-fragmented
    words first), with ties broken alphabetically for determinism. Only
    the columns listed in ``fieldnames`` are populated; ``write_failure_csv``
    is generic enough that the same row dict works for either CSV shape.
    """
    ordered = sorted(records, key=lambda r: (-r.unk_token_count, r.word))
    out: List[Dict[str, Any]] = []
    fset = set(fieldnames)
    for r in ordered:
        d: Dict[str, Any] = {"word": r.word}
        if "unk_token_count" in fset:
            d["unk_token_count"] = r.unk_token_count
        if "total_token_count" in fset:
            d["total_token_count"] = r.total_token_count
        if "source_fields" in fset:
            d["source_fields"] = "|".join(sorted(r.source_fields))
        if "num_examples_seen_in" in fset:
            d["num_examples_seen_in"] = r.num_examples_seen_in
        if "example_context" in fset:
            d["example_context"] = r.example_context
        out.append(d)
    return out
