"""Streaming raw-text sources for the pretraining mix.

Each source wraps one HF dataset in streaming mode, applies a seeded
shard + buffer shuffle (``IterableDataset.shuffle`` reorders the shard list
too — without it FineWeb-2 yields a 2013 Common-Crawl snapshot), and yields
``SourceDoc`` records. Source-specific prefilters (FineWeb ``language_score``,
Wikipedia list-article titles) live here and are counted in
``BaseSource.skipped`` so the pool manifest can report them.

Adding a source: subclass ``BaseSource``, set ``name`` / ``repo_id`` /
``kind``, implement ``_load`` and ``_to_doc``, and register it in
``SOURCE_REGISTRY``.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Type

from ...config import MixSourceConfig

logger = logging.getLogger(__name__)


@dataclass
class SourceDoc:
    id: str
    text: str
    kind: str                       # "web" | "wikipedia"
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseSource:
    name: str = ""
    repo_id: str = ""
    kind: str = "web"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.skipped: Counter = Counter()   # source-level prefilter counts

    # -- identity -----------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "repo_id": self.repo_id, "kind": self.kind, "params": self.params}

    def resolve_revision(self) -> Optional[str]:
        """Pin the dataset commit for the manifest (best-effort)."""
        try:
            from huggingface_hub import HfApi
            return HfApi().dataset_info(self.repo_id).sha
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: could not resolve revision of %s: %s", self.name, self.repo_id, e)
            return None

    # -- streaming ----------------------------------------------------------
    def _load(self):
        raise NotImplementedError

    def _to_doc(self, row: Dict[str, Any]) -> Optional[SourceDoc]:
        raise NotImplementedError

    def iter_docs(self, seed: int, shuffle_buffer: int) -> Iterator[SourceDoc]:
        ds = self._load()
        if shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        for row in ds:
            doc = self._to_doc(row)
            if doc is not None:
                yield doc

    @staticmethod
    def _text_of(row: Dict[str, Any]) -> str:
        for col in ("text", "content", "document", "passage"):
            v = row.get(col)
            if isinstance(v, str):
                return v
        return ""


class FineWeb2ArabicSource(BaseSource):
    """``HuggingFaceFW/fineweb-2`` — official Arabic (``arb_Arab``) config."""
    name = "fineweb2_arb"
    repo_id = "HuggingFaceFW/fineweb-2"
    kind = "web"

    def _load(self):
        from datasets import load_dataset
        return load_dataset(
            self.repo_id, name=self.params.get("config", "arb_Arab"),
            split="train", streaming=True,
        )

    def _to_doc(self, row):
        min_score = float(self.params.get("min_language_score", 0.90))
        score = row.get("language_score")
        if score is not None and float(score) < min_score:
            self.skipped["language_score"] += 1
            return None
        return SourceDoc(
            id=str(row.get("id", "")), text=self._text_of(row), kind=self.kind,
            meta={"url": row.get("url"), "dump": row.get("dump")},
        )


class WikipediaArabicSource(BaseSource):
    """``wikimedia/wikipedia`` — ``YYYYMMDD.ar`` dump. The title is prepended
    as the first paragraph; list articles (``قائمة …``) are skipped."""
    name = "wikipedia_ar"
    repo_id = "wikimedia/wikipedia"
    kind = "wikipedia"

    def _load(self):
        from datasets import load_dataset
        return load_dataset(
            self.repo_id, self.params.get("dump", "20231101.ar"),
            split="train", streaming=True,
        )

    def _to_doc(self, row):
        title = (row.get("title") or "").strip()
        prefixes = tuple(self.params.get("skip_title_prefixes", ["قائمة"]))
        if prefixes and title.startswith(prefixes):
            self.skipped["list_article"] += 1
            return None
        text = self._text_of(row)
        if title:
            text = f"{title}\n{text}"
        return SourceDoc(
            id=str(row.get("id", "")), text=text, kind=self.kind,
            meta={"url": row.get("url"), "title": title},
        )


class ArabicWeb24Source(BaseSource):
    """``lightonai/ArabicWeb24`` main version (sentence-deduped). Gated on
    the Hub (auto-approve) — needs ``HF_TOKEN`` / ``hf auth login``."""
    name = "arabicweb24"
    repo_id = "lightonai/ArabicWeb24"
    kind = "web"

    def _load(self):
        from datasets import load_dataset
        try:
            return load_dataset(
                self.repo_id,
                data_files=self.params.get("data_files", "ArabicWeb24/**/*.arrow"),
                split="train", streaming=True,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"{self.repo_id} is gated: accept its terms on the Hub and export "
                f"HF_TOKEN (or `hf auth login`) before building the pool ({e})"
            ) from e

    def _to_doc(self, row):
        return SourceDoc(
            id=str(row.get("id") or row.get("url") or ""), text=self._text_of(row),
            kind=self.kind, meta={"url": row.get("url")},
        )


class Arabic101BSource(BaseSource):
    """``ClusterlabAi/101_billion_arabic_words_dataset`` — ungated alternative
    to ArabicWeb24 for the domain-diversity slice."""
    name = "arabic_101b"
    repo_id = "ClusterlabAi/101_billion_arabic_words_dataset"
    kind = "web"

    def _load(self):
        from datasets import load_dataset
        return load_dataset(self.repo_id, split="train", streaming=True)

    def _to_doc(self, row):
        return SourceDoc(
            id=str(row.get("id") or row.get("url") or ""), text=self._text_of(row),
            kind=self.kind, meta={"url": row.get("url")},
        )


SOURCE_REGISTRY: Dict[str, Type[BaseSource]] = {
    cls.name: cls
    for cls in (FineWeb2ArabicSource, WikipediaArabicSource, ArabicWeb24Source, Arabic101BSource)
}


def make_source(cfg: MixSourceConfig) -> BaseSource:
    try:
        cls = SOURCE_REGISTRY[cfg.name]
    except KeyError:
        raise KeyError(
            f"unknown pretraining_mix source {cfg.name!r}; known: {sorted(SOURCE_REGISTRY)}"
        ) from None
    return cls(cfg.params)
