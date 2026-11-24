"""A released tokenizer, evaluated as shipped — ``hf_pretrained`` (added 2026-11-08).

Wraps a folder that holds a HuggingFace ``tokenizer.json`` (plus its
``tokenizer_config.json`` / ``special_tokens_map.json``) of a tokenizer released
with **another** model — Jais-2's, Fanar-1's — so it can be put in front of a base
model it was not trained with and go through the same pipeline as every
from-scratch tokenizer: ``train()`` is a no-op, nothing about the vocabulary is
changed, and the model's embedding rows are built by ``model.embedding_init``
exactly as for BPE-16K / AraRooPat (``reuses_model_embeddings`` is False, so the
initialiser runs even when the vocabulary happens to have the base model's row
count — Fanar-1's 128 256 = Llama-3.2-3B's 128 256).

Encode / decode are the raw ``tokenizers.Tokenizer`` loaded from the file — the
exact ids the releasing model was trained with, exact decode, no
``clean_up_tokenization_spaces`` (destructive for BPE; Jais-2's config sets it True,
which is why ``AutoTokenizer`` is not used). The mechanics are inherited from
:class:`BPETokenizer` (same truncation / padding path); only the loading, the
special-token map and the vocabulary-size contract differ.

* ``encode`` adds whatever the file's own post-processor adds (a BOS for both
  Jais-2 and Fanar-1, no EOS). The pipeline is BOS-agnostic (LCP masking,
  ``tokenize_record`` appends the EOS a record lacks, the free-form task strips a
  trailing EOS).
* ``special_tokens``: ``pad_token`` / ``bos_token`` / ``eos_token`` resolved from
  the files' token names. ``unk_token`` is included **only when the file's BPE
  model declares an unk token** (``model.unk_token`` in ``tokenizer.json``):
  Jais-2 is byte-level with no unk in its model — it cannot emit one, so the key is
  omitted (as ``native_qwen3`` does; its config names ``<|endoftext|>`` as "unk",
  which is the pad id and is never produced by encoding). Fanar-1's model declares
  ``<unk>`` (id 3) but has byte fallback, so in practice it is never produced either;
  the key is kept because the object can emit it. The UNK rate reads 0 for both.
* ``vocab_size`` = ``max(vocab_rows, len(tokenizer))`` — the embedding rows the arm
  uses (Jais-2: 150 222 entries in a 150 272-row table; the 50 padding rows are
  never produced).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizers import Tokenizer

from arabic_eval.params_spec import ParamSpec
from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.bpe import BPETokenizer

logger = logging.getLogger("arabic_eval.tokenizers.hf_pretrained")

MARKER = "hf_pretrained_marker.json"
COPIED_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "config.json", "tokenizer.model", "PROVENANCE.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("content")
    return str(value)


@tokenizer_registry.register("hf_pretrained")
class HFPretrainedTokenizer(BPETokenizer):
    """A released ``tokenizer.json``, unchanged, in front of a foreign base model."""

    def __init__(self, path: Optional[str] = None, vocab_rows: Optional[int] = None,
                 label: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._path: Optional[str] = str(path) if path else None
        self._vocab_rows: Optional[int] = int(vocab_rows) if vocab_rows else None
        self._label: Optional[str] = label
        self._unk_declared = False
        self._tokenizer_sha256: Optional[str] = None
        if self._path:
            self._load_files(Path(self._path))

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("path", "str", None, nullable=True,
                      help="Folder holding the released tokenizer.json (+ tokenizer_config.json, special_tokens_map.json); "
                           "loaded unchanged."),
            ParamSpec("vocab_rows", "int", None, nullable=True, min=1,
                      help="Embedding rows the arm uses (vocab_size = max(vocab_rows, len(tokenizer))); null = len(tokenizer). "
                           "Jais-2: 150272 (its model's padded table)."),
            ParamSpec("label", "str", None, nullable=True,
                      help="Display name used in tables and provenance, e.g. 'Jais-2 (released)'."),
        ]

    # ----------------------------------------------------------------- loading
    def _load_files(self, folder: Path) -> None:
        tok_file = folder / "tokenizer.json"
        if not tok_file.exists():
            raise FileNotFoundError(f"hf_pretrained: no tokenizer.json in {folder}")
        self._tokenizer = Tokenizer.from_file(str(tok_file))
        self._tokenizer_sha256 = _sha256(tok_file)
        model = json.loads(tok_file.read_text(encoding="utf-8")).get("model", {})
        self._unk_declared = bool(model.get("unk_token"))
        names: Dict[str, Optional[str]] = {}
        for fname in ("tokenizer_config.json", "special_tokens_map.json"):
            p = folder / fname
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
                    name = _token_name(data.get(key))
                    if name and not names.get(key):
                        names[key] = name
        smap: Dict[str, int] = {}
        for key in ("pad_token", "bos_token", "eos_token"):
            name = names.get(key)
            tid = self._tokenizer.token_to_id(name) if name else None
            if tid is None:
                raise ValueError(f"hf_pretrained: {key} {name!r} of {folder} is not in the vocabulary")
            smap[key] = int(tid)
        if self._unk_declared:
            unk_name = model.get("unk_token")
            tid = self._tokenizer.token_to_id(unk_name)
            if tid is not None:
                smap["unk_token"] = int(tid)
        self._special_token_map = smap
        logger.info("hf_pretrained: loaded %s (%s) — %d entries, vocab_rows=%s, specials %s",
                    folder, self._label or "-", self._tokenizer.get_vocab_size(), self._vocab_rows, smap)

    def train(self, texts: List[str], vocab_size: int, **kwargs: Any) -> None:
        logger.info("hf_pretrained.train() is a no-op — the released tokenizer is used as shipped "
                    "(ignoring vocab_size=%s, %d input texts)", vocab_size, len(texts))

    def save(self, path: Path | str) -> None:
        if self._tokenizer is None or not self._path:
            raise RuntimeError("hf_pretrained: nothing to save (no source path loaded)")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        src = Path(self._path)
        for fname in COPIED_FILES:
            if (src / fname).exists() and (src / fname).resolve() != (path / fname).resolve():
                shutil.copy2(src / fname, path / fname)
        marker = {"path": self._path, "vocab_rows": self._vocab_rows, "label": self._label,
                  "tokenizer_json_sha256": self._tokenizer_sha256}
        (path / MARKER).write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def load(self, path: Path | str) -> None:
        path = Path(path)
        marker_file = path / MARKER
        marker: Dict[str, Any] = {}
        if marker_file.exists():
            marker = json.loads(marker_file.read_text(encoding="utf-8"))
        # Constructor params win; the marker fills what was not given.
        if self._vocab_rows is None and marker.get("vocab_rows"):
            self._vocab_rows = int(marker["vocab_rows"])
        if self._label is None:
            self._label = marker.get("label")
        if self._path is None:
            self._path = marker.get("path") or str(path)
        self._load_files(path)
        want = marker.get("tokenizer_json_sha256")
        if want and want != self._tokenizer_sha256:
            raise ValueError(f"hf_pretrained: {path}/tokenizer.json sha256 {self._tokenizer_sha256} != marker {want}")

    # -------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        n = self._tokenizer.get_vocab_size() if self._tokenizer else 0
        return max(self._vocab_rows or 0, n)

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def reuses_model_embeddings(self) -> bool:
        return False
