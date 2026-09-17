"""Test-set contamination: held-out sets, hash / n-gram matching, exclusions.

The held-out sets are the official evaluation splits the model must never
train on — TyDiQA-AR ``validation`` (the public dev; TyDi's test is hidden),
ARCD ``validation`` (the paper's test split) — and, once it exists, the
user's free-form eval prompt file. They are declared in
``configs/contamination/heldout_sets.yaml``; a ``file`` set whose ``path``
is null is *pending* and skipped everywhere until the file lands.

Two tests per (training record, held-out record) pair, both on the same
normalization (``normalize_text``: NFKC, diacritics and tatweel stripped,
alef / ة / ى folded, Arabic-Indic digits to ASCII, punctuation removed,
whitespace collapsed):

* **exact** — SHA-1 of the normalized passage, of each paragraph and of
  the question / prompt;
* **n-gram** — word n-grams of the training text looked up in an index of
  the held-out side (8 for passages, 6 for short prompts). Per pair we
  keep the share of the held-out record's n-grams covered (``coverage``)
  and the longest run of consecutive training n-grams hitting that
  record (``run_words``).

Tiers: ``contaminated`` = an exact hash match, or ``coverage`` ≥ 0.5, or a
shared run ≥ 20 words; anything else with a shared n-gram is ``overlap``
(reported, not removed). ``Exclusions`` is the committed list of training
record ids in the contaminated tier; loaders apply it to the ``train`` and
``dev`` splits only, never to an official evaluation split. The pool
builder applies the same index by content at build time
(``HeldoutIndex.scan_text``), because pool document ids only exist after
a build.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger("arabic_eval.data.contamination")

DEFAULT_SETS_FILE = Path("configs/contamination/heldout_sets.yaml")
DEFAULT_EXCLUSIONS_FILE = Path("configs/contamination/exclusions.json")
EXCLUSION_SPLITS = ("train", "dev")          # never an official evaluation split
DEFAULT_NGRAM = 8
COVERAGE_THRESHOLD = 0.5
RUN_WORDS_THRESHOLD = 20
TIER_CONTAMINATED = "contaminated"
TIER_OVERLAP = "overlap"
EXCLUSIONS_SCHEMA = 1

HIT_FIELDS = (
    "corpus", "split", "record_id", "heldout_set", "heldout_id", "exact",
    "shared_ngrams", "heldout_ngrams", "coverage", "run_words", "tier", "snippet",
)

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[ً-ْٰـ]")   # tanwīn … sukūn, superscript alef, tatweel
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})


def normalize_text(text: Optional[str]) -> str:
    """The one normalization every contamination test runs on (both sides)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _DIACRITICS.sub("", text)
    text = text.translate(_FOLD)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).lower()


def text_hash(normalized: str) -> str:
    """SHA-1 (16 hex chars) of a normalized string; ``""`` for an empty one."""
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def split_paragraphs(raw: Optional[str]) -> List[str]:
    return [p for p in (raw or "").replace("\r\n", "\n").split("\n") if p.strip()]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Held-out sets (declared in configs/contamination/heldout_sets.yaml)
# --------------------------------------------------------------------------

@dataclass
class HeldoutSpec:
    name: str
    kind: str                                  # "corpus" | "file"
    ngram: int = DEFAULT_NGRAM
    corpus: Optional[str] = None               # kind == corpus: a finetune_corpora loader name
    split: Optional[str] = None
    path: Optional[Path] = None                # kind == file: JSONL (null → pending)
    fields: Dict[str, str] = field(default_factory=lambda: {
        "id": "id", "prompt": "prompt", "context": "context", "reference": "reference"})

    @property
    def pending(self) -> bool:
        return self.kind == "file" and self.path is None

    def identity(self) -> Dict[str, Any]:
        """What the pool fingerprint and the reports record for this set."""
        if self.kind == "corpus":
            from .finetune_corpora import PINNED_REVISIONS
            return {"kind": "corpus", "corpus": self.corpus, "split": self.split,
                    "revision": PINNED_REVISIONS.get(self.corpus), "ngram": self.ngram}
        if self.pending:
            return {"kind": "file", "status": "pending", "ngram": self.ngram}
        if not self.path.exists():
            raise FileNotFoundError(
                f"held-out set {self.name!r}: {self.path} does not exist (set path: null while the file is pending)")
        return {"kind": "file", "path": str(self.path), "sha256": file_sha256(self.path), "ngram": self.ngram}


@dataclass
class HeldoutThresholds:
    coverage: float = COVERAGE_THRESHOLD
    run_words: int = RUN_WORDS_THRESHOLD


def load_heldout_sets(path: Path | str = DEFAULT_SETS_FILE) -> Tuple[List[HeldoutSpec], HeldoutThresholds]:
    """Parse the held-out declaration file."""
    import yaml
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    default_n = int(raw.get("ngram_default", DEFAULT_NGRAM))
    tiers = raw.get("tiers") or {}
    thresholds = HeldoutThresholds(
        coverage=float(tiers.get("coverage", COVERAGE_THRESHOLD)),
        run_words=int(tiers.get("run_words", RUN_WORDS_THRESHOLD)),
    )
    specs: List[HeldoutSpec] = []
    for name, entry in (raw.get("sets") or {}).items():
        entry = entry or {}
        kind = entry.get("kind")
        if kind not in ("corpus", "file"):
            raise ValueError(f"held-out set {name!r}: kind must be 'corpus' or 'file', got {kind!r}")
        spec = HeldoutSpec(name=name, kind=kind, ngram=int(entry.get("ngram", default_n)))
        if kind == "corpus":
            spec.corpus, spec.split = entry.get("corpus"), entry.get("split")
            if not spec.corpus or not spec.split:
                raise ValueError(f"held-out set {name!r}: a corpus set needs 'corpus' and 'split'")
        else:
            p = entry.get("path")
            spec.path = Path(p) if p else None
            spec.fields.update(entry.get("fields") or {})
        specs.append(spec)
    if not specs:
        raise ValueError(f"{path}: no held-out sets declared")
    return specs, thresholds


def heldout_fingerprint_payload(sets_file: Path | str = DEFAULT_SETS_FILE) -> Dict[str, Any]:
    """Identity of every declared set (pinned revisions, prompt-file hash) —
    what changes the pool fingerprint when a held-out set changes."""
    sets_file = Path(sets_file)
    specs, thresholds = load_heldout_sets(sets_file)
    return {
        "sets_file_sha256": file_sha256(sets_file),
        "thresholds": {"coverage": thresholds.coverage, "run_words": thresholds.run_words},
        "sets": {s.name: s.identity() for s in specs},
    }


@dataclass
class HeldoutRecord:
    set_name: str
    rec_id: str
    question: str      # normalized question / prompt ("" when absent)
    passage: str       # normalized context / passage ("" when absent)
    paragraphs: List[str]
    text: str          # normalized concatenation the n-grams are taken from


def _heldout_record(set_name: str, rec_id: str, question: Optional[str],
                    passage: Optional[str], reference: Optional[str]) -> HeldoutRecord:
    q, p = normalize_text(question), normalize_text(passage)
    parts = [x for x in (q, p, normalize_text(reference)) if x]
    return HeldoutRecord(
        set_name=set_name, rec_id=str(rec_id), question=q, passage=p,
        paragraphs=[normalize_text(x) for x in split_paragraphs(passage)],
        text=" ".join(parts),
    )


def read_prompt_file(path: Path, fields: Mapping[str, str], set_name: str) -> List[HeldoutRecord]:
    """JSONL, one object per line: ``id`` (optional; the line number
    otherwise), ``prompt``, optional ``context`` and ``reference``."""
    out: List[HeldoutRecord] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get(fields.get("prompt", "prompt"))
            if not prompt:
                raise ValueError(f"{path}:{lineno}: no {fields.get('prompt', 'prompt')!r} field")
            out.append(_heldout_record(
                set_name, obj.get(fields.get("id", "id"), lineno), prompt,
                obj.get(fields.get("context", "context")), obj.get(fields.get("reference", "reference")),
            ))
    return out


def heldout_records(spec: HeldoutSpec) -> List[HeldoutRecord]:
    """The records of one live set (``[]`` for a pending one)."""
    if spec.pending:
        return []
    if spec.kind == "file":
        return read_prompt_file(spec.path, spec.fields, spec.name)
    from .finetune_corpora import load_corpus
    recs = load_corpus(spec.corpus, spec.split)     # never with exclusions: this is the reference
    return [_heldout_record(spec.name, r.id, r.question, r.context, r.answer) for r in recs]


# --------------------------------------------------------------------------
# Index + scanner
# --------------------------------------------------------------------------

@dataclass
class Hit:
    heldout_set: str
    heldout_id: str
    exact: str            # "" | "passage" | "paragraph" | "question" | "text"
    shared_ngrams: int
    heldout_ngrams: int
    coverage: float
    run_words: int
    tier: str


class HeldoutIndex:
    """Exact hashes + n-gram postings of the held-out side; ``scan`` a
    training record against it."""

    def __init__(self, records: Sequence[HeldoutRecord], ngram_for_set: Mapping[str, int],
                 thresholds: Optional[HeldoutThresholds] = None,
                 sets: Optional[Sequence[HeldoutSpec]] = None) -> None:
        self.records = list(records)
        self.thresholds = thresholds or HeldoutThresholds()
        self.sets = list(sets or [])
        self.ngram_for_set = dict(ngram_for_set)
        self.ngram_sizes: List[int] = sorted(set(self.ngram_for_set.values())) or [DEFAULT_NGRAM]
        self._exact: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self._grams: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        self._n_grams: List[int] = []
        for idx, r in enumerate(self.records):
            n = self.ngram_for_set.get(r.set_name, DEFAULT_NGRAM)
            for kind, s in (("passage", r.passage), ("question", r.question), ("text", r.text)):
                h = text_hash(s)
                if h and (idx, kind) not in self._exact[h]:
                    self._exact[h].append((idx, kind))
            for p in r.paragraphs:
                h = text_hash(p)
                if h and p != r.passage:
                    self._exact[h].append((idx, "paragraph"))
            words = r.text.split()
            count = max(0, len(words) - n + 1)
            self._n_grams.append(count)
            for pos in range(count):
                self._grams[(n, hash(tuple(words[pos:pos + n])))].append((idx, pos))

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_sets(cls, specs: Sequence[HeldoutSpec], thresholds: Optional[HeldoutThresholds] = None) -> "HeldoutIndex":
        records: List[HeldoutRecord] = []
        for spec in specs:
            recs = heldout_records(spec)
            logger.info("held-out set %s: %s", spec.name, "pending (no file yet)" if spec.pending else f"{len(recs)} records")
            records.extend(recs)
        return cls(records, {s.name: s.ngram for s in specs}, thresholds, specs)

    @classmethod
    def from_sets_file(cls, path: Path | str = DEFAULT_SETS_FILE) -> "HeldoutIndex":
        specs, thresholds = load_heldout_sets(path)
        return cls.from_sets(specs, thresholds)

    def summary(self) -> Dict[str, Any]:
        per_set: Counter = Counter(r.set_name for r in self.records)
        return {
            "records": len(self.records),
            "ngrams": sum(self._n_grams),
            "thresholds": {"coverage": self.thresholds.coverage, "run_words": self.thresholds.run_words},
            "sets": {
                s.name: {"status": "pending" if s.pending else "live", "records": per_set.get(s.name, 0),
                         "ngram": s.ngram, **s.identity()}
                for s in self.sets
            } if self.sets else {name: {"records": n} for name, n in per_set.items()},
        }

    def __len__(self) -> int:
        return len(self.records)

    # -- scanning --------------------------------------------------------------
    def scan(self, parts: Mapping[str, Optional[str]]) -> List[Hit]:
        """Hits of one training record given as named raw parts
        (``question`` / ``context`` / ``answer`` / ``text`` …)."""
        if not self.records:
            return []
        exact: Dict[int, str] = {}
        norm_parts: List[str] = []
        for raw in parts.values():
            if not raw:
                continue
            full = normalize_text(raw)
            norm_parts.append(full)
            candidates = [full] + ([normalize_text(p) for p in split_paragraphs(raw)] if "\n" in raw else [])
            for s in candidates:
                for idx, kind in self._exact.get(text_hash(s), ()):
                    exact.setdefault(idx, kind)
        words = " ".join(norm_parts).split()
        matched: Dict[int, Set[int]] = defaultdict(set)
        run_best: Dict[int, int] = defaultdict(int)
        for n in self.ngram_sizes:
            run_cur: Dict[int, int] = {}
            for i in range(max(0, len(words) - n + 1)):
                postings = self._grams.get((n, hash(tuple(words[i:i + n]))))
                seen_now: Set[int] = set()
                if postings:
                    for idx, pos in postings:
                        if self.ngram_for_set.get(self.records[idx].set_name, DEFAULT_NGRAM) != n:
                            continue
                        matched[idx].add(pos)
                        seen_now.add(idx)
                for idx in seen_now:
                    run_cur[idx] = run_cur.get(idx, 0) + 1
                    if run_cur[idx] > run_best[idx]:
                        run_best[idx] = run_cur[idx]
                for idx in list(run_cur):
                    if idx not in seen_now:
                        del run_cur[idx]
        hits: List[Hit] = []
        for idx in set(matched) | set(exact):
            rec = self.records[idx]
            n = self.ngram_for_set.get(rec.set_name, DEFAULT_NGRAM)
            total = self._n_grams[idx]
            shared = len(matched.get(idx, ()))
            coverage = (shared / total) if total else (1.0 if idx in exact else 0.0)
            run_words = (run_best[idx] + n - 1) if run_best.get(idx) else 0
            contaminated = (idx in exact or coverage >= self.thresholds.coverage
                            or run_words >= self.thresholds.run_words)
            hits.append(Hit(
                heldout_set=rec.set_name, heldout_id=rec.rec_id, exact=exact.get(idx, ""),
                shared_ngrams=shared, heldout_ngrams=total, coverage=round(coverage, 4),
                run_words=run_words, tier=TIER_CONTAMINATED if contaminated else TIER_OVERLAP,
            ))
        hits.sort(key=lambda h: (h.tier != TIER_CONTAMINATED, -h.coverage, h.heldout_set, h.heldout_id))
        return hits

    def scan_text(self, text: str) -> List[Hit]:
        return self.scan({"text": text})

    def is_contaminated(self, text: str) -> Optional[Hit]:
        """The strongest contaminated-tier hit of a plain text, or None."""
        for h in self.scan_text(text):
            if h.tier == TIER_CONTAMINATED:
                return h
        return None


# --------------------------------------------------------------------------
# Scanning the training side
# --------------------------------------------------------------------------

def _row(corpus: str, split: str, rec_id: str, hit: Hit, snippet: str) -> Dict[str, Any]:
    return {
        "corpus": corpus, "split": split, "record_id": str(rec_id), "heldout_set": hit.heldout_set,
        "heldout_id": hit.heldout_id, "exact": hit.exact, "shared_ngrams": hit.shared_ngrams,
        "heldout_ngrams": hit.heldout_ngrams, "coverage": hit.coverage, "run_words": hit.run_words,
        "tier": hit.tier, "snippet": snippet[:160],
    }


def scan_records(index: HeldoutIndex, corpus: str, split: str, records: Iterable[Any]) -> List[Dict[str, Any]]:
    """Every hit of every ``QARecord`` (question + context + answer + choices)."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        parts: Dict[str, Optional[str]] = {"question": rec.question, "context": rec.context, "answer": rec.answer}
        if getattr(rec, "choices", None):
            parts["choices"] = " \n".join(rec.choices)
        for hit in index.scan(parts):
            rows.append(_row(corpus, split, rec.id, hit, rec.question or rec.context or ""))
    return rows


def scan_texts(index: HeldoutIndex, corpus: str, split: str, items: Iterable[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Every hit of every ``(id, text)`` item (pool documents, raw corpora)."""
    rows: List[Dict[str, Any]] = []
    for rec_id, text in items:
        for hit in index.scan_text(text):
            rows.append(_row(corpus, split, rec_id, hit, text))
    return rows


def summarize_hits(rows: Sequence[Mapping[str, Any]], strict: bool = False) -> Dict[str, Any]:
    """Counts per (corpus, held-out set) and per held-out record."""
    per_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    heldout_max: Dict[Tuple[str, str], float] = {}
    excluded: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        key = (r["corpus"], r["heldout_set"])
        d = per_pair.setdefault(key, {"records_overlap": set(), "records_contaminated": set(), "exact": set(),
                                      "heldout_hit": set(), "heldout_contaminated": set()})
        d["records_overlap"].add(r["record_id"])
        d["heldout_hit"].add(r["heldout_id"])
        if r["tier"] == TIER_CONTAMINATED:
            d["records_contaminated"].add(r["record_id"])
            d["heldout_contaminated"].add(r["heldout_id"])
        if r["exact"]:
            d["exact"].add(r["record_id"])
        hk = (r["heldout_set"], r["heldout_id"])
        heldout_max[hk] = max(heldout_max.get(hk, 0.0), float(r["coverage"]))
        if r["tier"] == TIER_CONTAMINATED or strict:
            excluded[r["corpus"]].add(r["record_id"])
    return {
        "pairs": {
            f"{c}|{s}": {k: len(v) for k, v in d.items()} for (c, s), d in sorted(per_pair.items())
        },
        "excluded_per_corpus": {c: len(v) for c, v in sorted(excluded.items())},
        "heldout_records_with_contamination": sorted(
            {f"{s}|{i}" for (s, i), cov in heldout_max.items() if cov >= 0.0}
            & {f"{r['heldout_set']}|{r['heldout_id']}" for r in rows if r["tier"] == TIER_CONTAMINATED}),
    }


# --------------------------------------------------------------------------
# Exclusions (committed list of training record ids to drop)
# --------------------------------------------------------------------------

@dataclass
class Exclusions:
    ids: Dict[str, Set[str]] = field(default_factory=dict)      # corpus → record ids
    meta: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None
    dropped: Dict[str, int] = field(default_factory=dict)       # "corpus/split" → records removed (this process)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], strict: bool = False, **meta: Any) -> "Exclusions":
        ids: Dict[str, Set[str]] = defaultdict(set)
        for r in rows:
            if r["tier"] == TIER_CONTAMINATED or strict:
                ids[r["corpus"]].add(str(r["record_id"]))
        return cls(ids=dict(ids), meta={"strict": strict, **meta})

    @classmethod
    def load(cls, path: Path | str) -> "Exclusions":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"contamination exclusions file {path} not found — run "
                f"scripts/check_contamination.py --write-exclusions, or set "
                f"training.contamination_exclusions: null to train without it")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        ids = {c: set(map(str, v)) for c, v in (raw.get("corpora") or {}).items()}
        meta = {k: v for k, v in raw.items() if k != "corpora"}
        return cls(ids=ids, meta=meta, path=path)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema": EXCLUSIONS_SCHEMA,
            **{k: v for k, v in self.meta.items() if k != "schema"},
            "counts": {c: len(v) for c, v in sorted(self.ids.items())},
            "corpora": {c: sorted(v) for c, v in sorted(self.ids.items())},
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1)
        self.path = path
        return path

    def digest(self) -> str:
        """Content hash (``""`` when nothing is excluded) — goes into cache fingerprints."""
        if not any(self.ids.values()):
            return ""
        payload = json.dumps({c: sorted(v) for c, v in sorted(self.ids.items()) if v}, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def total(self) -> int:
        return sum(len(v) for v in self.ids.values())

    def apply(self, corpus: str, split: str, records: Sequence[Any]) -> List[Any]:
        """Drop the excluded records of ``corpus`` from a ``train`` / ``dev``
        list; an official evaluation split passes through untouched."""
        if split not in EXCLUSION_SPLITS:
            return list(records)
        bad = self.ids.get(corpus)
        if not bad:
            return list(records)
        kept = [r for r in records if str(r.id) not in bad]
        n = len(records) - len(kept)
        self.dropped[f"{corpus}/{split}"] = n
        if n:
            logger.info("contamination: %s/%s: %d of %d records excluded (%s)",
                        corpus, split, n, len(records), self.path or "in-memory")
        return kept


def load_exclusions(path: Optional[Path | str]) -> Optional[Exclusions]:
    """``None`` for a null config value; the committed list otherwise."""
    if path is None or str(path).strip() == "":
        return None
    return Exclusions.load(path)


def new_exclusions_meta(index: HeldoutIndex, **extra: Any) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "heldout": index.summary(),
        **extra,
    }
