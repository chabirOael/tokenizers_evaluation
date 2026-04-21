"""End-to-end sweep tests — exercises the full pipeline for every tokenizer.

Run with:  pytest tests/test_e2e_sweep.py -v

No internet access or GPU is required:
  - Dataset loading is patched to return synthetic Arabic text / MCQ data.
  - Model loading is patched to return a tiny CPU-based fake LLaMA model.
  - MorphoBPE's Farasa segmenter is patched to return text unchanged.

Each parametrized case drives *one* (tokenizer × task) combination through:
  1. Tokenizer train / encode / decode / save / load
  2. Intrinsic metrics
  3. Collation into training batches
  4. Model adaptation (embedding-layer swap)
  5. Forward pass with loss
  6. Full run_single_experiment call (mirrors the actual sweep loop)

LightEval benchmark tasks (acva, alghafa, culture_arabic_mmlu, arabic_exam) are
covered separately in TestLightEvalBenchmarks and in the parametrized e2e sweep.
"""
from __future__ import annotations

import tempfile
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Synthetic corpus — plain Arabic text
# ---------------------------------------------------------------------------

ARABIC_TEXTS: List[str] = [
    "مرحبا بالعالم هذا نص عربي للاختبار والتجربة اليومية",
    "اللغة العربية لغة جميلة ومعبرة وغنية بالمفردات",
    "الجزائر بلد جميل يقع في شمال أفريقيا على ضفاف البحر",
    "التعلم الآلي علم يدرس كيفية تعلم الحاسوب من البيانات",
    "النص العربي يحتاج إلى معالجة خاصة لأنه يُكتب من اليمين",
    "البرمجة اللغوية للحاسوب تتطلب فهماً عميقاً للرياضيات",
    "تعلم البرمجة يفتح آفاقاً واسعة في عالم التكنولوجيا الحديثة",
    "الذكاء الاصطناعي يغير طريقة تفاعل البشر مع الآلات والبيانات",
    "معالجة اللغات الطبيعية مجال مهم في علوم الحاسوب المتقدمة",
    "النماذج اللغوية الكبيرة قادرة على فهم النصوص وتوليدها بدقة",
    "التحويل الآلي للنصوص العربية يستخدم خوارزميات متطورة جداً",
    "تقنيات التعلم العميق تعتمد على الشبكات العصبية الاصطناعية",
]

# ---------------------------------------------------------------------------
# Synthetic QA corpus
# ---------------------------------------------------------------------------

SAMPLE_QA: Dict[str, List] = {
    "context": [
        "الجزائر عاصمة دولة الجزائر وأكبر مدنها وأهمها",
        "اللغة العربية من أقدم اللغات السامية في العالم",
        "الرياضيات علم يدرس الأعداد والأشكال والعلاقات بينها",
        "الفيزياء علم يدرس قوانين الطبيعة والكون المادي",
    ],
    "question": [
        "ما هي عاصمة الجزائر",
        "ما عائلة اللغة العربية",
        "ما الذي يدرسه علم الرياضيات",
        "ما الذي يدرسه علم الفيزياء",
    ],
    "answers": [
        {"text": ["الجزائر"]},
        {"text": ["السامية"]},
        {"text": ["الأعداد والأشكال"]},
        {"text": ["قوانين الطبيعة"]},
    ],
}

# ---------------------------------------------------------------------------
# Synthetic MCQ corpus — used by all lighteval benchmark mocks.
# Schema: separate A/B/C/D columns + answer letter (handled by _parse_mcq_generic).
# ---------------------------------------------------------------------------

SAMPLE_MCQ_DATA: Dict[str, List] = {
    "question": [
        "ما هي عاصمة الجزائر؟",
        "كم عدد أيام الأسبوع؟",
        "ما هو لون السماء؟",
        "ما هي أكبر قارة في العالم؟",
        "ما هي عاصمة مصر؟",
        "كم عدد أشهر السنة؟",
        "ما هو أطول نهر في العالم؟",
        "من كتب القرآن الكريم؟",
    ],
    "A": ["الجزائر", "خمسة", "أزرق", "آسيا", "القاهرة", "عشرة", "النيل", "الله"],
    "B": ["وهران", "سبعة", "أحمر", "أفريقيا", "الإسكندرية", "اثنا عشر", "الأمازون", "محمد"],
    "C": ["قسنطينة", "ستة", "أخضر", "أوروبا", "أسوان", "ثمانية", "المسيسيبي", "جبريل"],
    "D": ["عنابة", "ثمانية", "أصفر", "أمريكا", "الجيزة", "خمسة عشر", "اليانغتسي", "موسى"],
    "answer": ["A", "B", "A", "A", "A", "B", "A", "A"],
}

# Benchmark dataset names patched by the mock factory.
_BENCHMARK_NAMES = frozenset({
    "OALL/ACVA",
    "OALL/AlGhafa-Native",
    "acmc/arabic_culture_mmlu",
    "arabic_exam",
})


# ---------------------------------------------------------------------------
# Fake LLaMA model
# ---------------------------------------------------------------------------

class _FakeOutput:
    """Mimics the HuggingFace ModelOutput for loss / logits access."""
    def __init__(self, loss: Optional[torch.Tensor], logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class _FakeLayer(nn.Module):
    """Single attention-free transformer layer (just a linear projection)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # Return tuple so _forward_character_cnn can do layer_out[0]
        return (self.proj(hidden_states),)


class _FakeInnerModel(nn.Module):
    """Mimics model.model (LlamaModel) with embed_tokens / layers / norm."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([_FakeLayer(hidden_size) for _ in range(2)])
        self.norm = nn.LayerNorm(hidden_size)


class FakeLlamaForCausalLM(nn.Module):
    """Tiny fake causal LM that satisfies the LlamaAdapter interface.

    Registered attributes mirror those accessed by LlamaAdapter:
      - model.embed_tokens  (replaceable nn.Embedding)
      - model.layers        (nn.ModuleList of decoder layers)
      - model.norm          (LayerNorm)
      - lm_head             (replaceable nn.Linear)
      - config              (SimpleNamespace with hidden_size / vocab_size etc.)
    """

    HIDDEN = 32
    INIT_VOCAB = 200

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeInnerModel(self.INIT_VOCAB, self.HIDDEN)
        self.lm_head = nn.Linear(self.HIDDEN, self.INIT_VOCAB, bias=False)
        self.config = types.SimpleNamespace(
            hidden_size=self.HIDDEN,
            vocab_size=self.INIT_VOCAB,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

    # ---- HuggingFace model interface expected by LlamaAdapter ----

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        old = self.model.embed_tokens
        old_size, hidden = old.weight.shape
        new_embed = nn.Embedding(new_size, hidden, padding_idx=0)
        copy_rows = min(old_size, new_size)
        with torch.no_grad():
            new_embed.weight[:copy_rows] = old.weight[:copy_rows]
        self.model.embed_tokens = new_embed
        self.lm_head = nn.Linear(hidden, new_size, bias=False)
        return new_embed

    def save_pretrained(self, path: Any) -> None:  # no-op in tests
        pass

    # ---- Standard forward (standard + char_jaber embedding types) ----

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> _FakeOutput:
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, attention_mask)[0]
        x = self.model.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return _FakeOutput(loss=loss, logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 5,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.forward(input_ids)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
        return input_ids


# ---------------------------------------------------------------------------
# Helpers: dataset mocks
# ---------------------------------------------------------------------------

def _make_text_dataset(texts: List[str]):
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def _make_qa_dataset():
    from datasets import Dataset
    return Dataset.from_dict(SAMPLE_QA)


def _make_mcq_dataset():
    from datasets import Dataset
    return Dataset.from_dict(SAMPLE_MCQ_DATA)


def _mock_load_dataset_factory():
    """Returns a callable that fakes load_dataset for all four call sites.

    Routing logic:
      - ARCD / qa names  → QA dataset (context/question/answers)
      - Known benchmark names (ACVA, Alghafa, …) → MCQ dataset (A/B/C/D cols)
      - Everything else  → plain Arabic text dataset
    """
    from datasets import DatasetDict

    text_ds = _make_text_dataset(ARABIC_TEXTS)
    text_dd = DatasetDict({"train": text_ds, "test": text_ds})
    qa_ds = _make_qa_dataset()
    qa_dd = DatasetDict({"train": qa_ds, "test": qa_ds})
    mcq_ds = _make_mcq_dataset()
    # Single split only — avoids duplicating questions when concatenate_datasets
    # is called over all splits in LightEvalBenchmarkTask._load_all_examples().
    mcq_dd = DatasetDict({"train": mcq_ds})

    def _mock(name, config=None, cache_dir=None, split=None, **kwargs):
        name_str = str(name)
        is_benchmark = name_str in _BENCHMARK_NAMES
        is_qa = ("arcd" in name_str or "qa" in name_str.lower()) and not is_benchmark

        if is_benchmark:
            return mcq_ds if split is not None else mcq_dd
        if is_qa:
            return qa_ds if split is not None else qa_dd
        return text_ds if split is not None else text_dd

    return _mock


# ---------------------------------------------------------------------------
# Parametrize matrix
# ---------------------------------------------------------------------------

# (tokenizer_type, tokenizer_params, vocab_size_for_training)
_TOK_CASES = [
    pytest.param("bpe",            {"min_frequency": 1}, 100,  id="bpe"),
    pytest.param("wordpiece",      {"min_frequency": 1}, 100,  id="wordpiece"),
    pytest.param("morpho_bpe",     {"min_frequency": 1}, 100,  id="morpho_bpe"),
    pytest.param("character_bert", {"max_char_len": 10, "max_word_vocab": 200}, None, id="character_bert"),
    pytest.param("char_jaber",     {},                   None, id="char_jaber"),
]

# (task_type, task_params)
# train_split_ratio=0.50 for lighteval tasks keeps ≥4 examples in each split
# even with the tiny 8-row synthetic MCQ dataset (16 rows after mock duplication).
_TASK_CASES = [
    pytest.param(
        "text_generation",
        {"max_length": 32, "stride": 16},
        id="text_gen",
    ),
    pytest.param(
        "question_answering",
        {"max_length": 32, "max_new_tokens": 4, "dataset_name": "hsseinmz/arcd"},
        id="qa",
    ),
    pytest.param(
        "acva",
        {"dataset_name": "OALL/ACVA", "train_split_ratio": 0.50, "max_length": 32, "seed": 0},
        id="acva",
    ),
    pytest.param(
        "alghafa",
        {"dataset_name": "OALL/AlGhafa-Native", "train_split_ratio": 0.50, "max_length": 32, "seed": 0},
        id="alghafa",
    ),
    pytest.param(
        "culture_arabic_mmlu",
        {"dataset_name": "acmc/arabic_culture_mmlu", "train_split_ratio": 0.50, "max_length": 32, "seed": 0},
        id="culture_arabic_mmlu",
    ),
    pytest.param(
        "arabic_exam",
        {"dataset_name": "arabic_exam", "train_split_ratio": 0.50, "max_length": 32, "seed": 0},
        id="arabic_exam",
    ),
]


# ---------------------------------------------------------------------------
# Fixture: patched environment shared by all tests in the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched_env():
    """Module-scoped patches active for the entire test module.

    Patches:
      - AutoModelForCausalLM.from_pretrained → FakeLlamaForCausalLM
      - load_dataset in every module that imports it → synthetic data
      - MorphoBPE Farasa segmenter → pass-through (no Java needed)

    All four load_dataset call sites are patched so that any test order is
    safe and no test can trigger a real network call.
    """
    mock_ds = _mock_load_dataset_factory()

    # Farasa mock: segment() returns the text unchanged (no morphological split)
    fake_segmenter = MagicMock()
    fake_segmenter.segment.side_effect = lambda text: text

    with (
        patch(
            "arabic_eval.models.llama_adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=lambda *a, **kw: FakeLlamaForCausalLM(),
        ),
        patch("arabic_eval.data.loader.load_dataset",                        side_effect=mock_ds),
        patch("arabic_eval.tasks.text_generation.load_dataset",              side_effect=mock_ds),
        patch("arabic_eval.tasks.question_answering.load_dataset",           side_effect=mock_ds),
        patch("arabic_eval.tasks.lighteval_benchmarks.load_dataset",         side_effect=mock_ds),
        patch(
            "arabic_eval.tokenizers.morpho_bpe._get_farasa_segmenter",
            return_value=fake_segmenter,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Helper: build and train a tokenizer
# ---------------------------------------------------------------------------

def _build_tokenizer(tok_type: str, tok_params: Dict, vocab_size: Optional[int]):
    """Import registry, build, and train a tokenizer on ARABIC_TEXTS."""
    import arabic_eval.tokenizers  # noqa: F401 — populates registry
    from arabic_eval.registry import tokenizer_registry

    cls = tokenizer_registry.get(tok_type)
    tok = cls(**tok_params)
    tok.train(ARABIC_TEXTS, vocab_size=vocab_size or 0, **tok_params)
    return tok


# ===========================================================================
# 1. Tokenizer unit tests  (train / encode / decode / save-load / intrinsic)
# ===========================================================================

@pytest.mark.usefixtures("patched_env")
class TestTokenizerUnit:

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_train_and_vocab(self, patched_env, tok_type, tok_params, vocab_size):
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        assert tok.vocab_size > 0, "vocab_size must be positive after training"

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_encode_returns_correct_structure(self, patched_env, tok_type, tok_params, vocab_size):
        from arabic_eval.tokenizers.base import EmbeddingType, TokenizerOutput

        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        sample = "مرحبا بالعالم هذا نص عربي"
        out = tok.encode(sample)

        assert isinstance(out, TokenizerOutput)
        assert isinstance(out.input_ids, list) and len(out.input_ids) > 0
        assert isinstance(out.attention_mask, list)
        assert len(out.attention_mask) == len(out.input_ids)

        if tok.embedding_type == EmbeddingType.CHARACTER_CNN:
            assert out.char_ids is not None, "CharacterBERT must populate char_ids"
            assert len(out.char_ids) == len(out.input_ids)
            max_char_len = tok_params.get("max_char_len", 50)
            for row in out.char_ids:
                assert len(row) == max_char_len, (
                    f"char_ids row has wrong length: {len(row)} vs {max_char_len}"
                )
        else:
            assert out.char_ids is None

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_decode_returns_string(self, patched_env, tok_type, tok_params, vocab_size):
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        out = tok.encode("مرحبا بالعالم")
        result = tok.decode(out.input_ids)
        assert isinstance(result, str)
        if tok_type == "char_jaber":
            assert "<unk>" not in result, "decode() must skip UNK tokens, not stringify them"

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_save_load_roundtrip(self, patched_env, tok_type, tok_params, vocab_size):
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        sample = "اللغة العربية جميلة"
        ids_before = tok.encode(sample).input_ids

        with tempfile.TemporaryDirectory() as tmp:
            tok.save(tmp)

            import arabic_eval.tokenizers  # noqa
            from arabic_eval.registry import tokenizer_registry
            fresh = tokenizer_registry.get(tok_type)(**tok_params)
            fresh.load(tmp)

        assert fresh.vocab_size == tok.vocab_size
        ids_after = fresh.encode(sample).input_ids
        assert ids_before == ids_after, "encode output must match after save/load"

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_special_tokens_present(self, patched_env, tok_type, tok_params, vocab_size):
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        st = tok.special_tokens
        for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
            assert key in st, f"special_tokens missing '{key}'"
            assert isinstance(st[key], int)

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_intrinsic_metrics(self, patched_env, tok_type, tok_params, vocab_size):
        from arabic_eval.tokenizers.intrinsic_metrics import compute_intrinsic_metrics

        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        metrics = compute_intrinsic_metrics(tok, ARABIC_TEXTS[:4])

        expected_keys = {"fertility", "compression_ratio", "unk_rate", "vocab_coverage", "vocab_size"}
        assert expected_keys.issubset(metrics.keys())
        assert metrics["vocab_size"] == tok.vocab_size
        assert 0.0 <= metrics["unk_rate"] <= 1.0
        assert 0.0 <= metrics["vocab_coverage"] <= 1.0
        assert metrics["fertility"] > 0


# ===========================================================================
# 2. CharacterBERT-specific bug regression tests
# ===========================================================================

@pytest.mark.usefixtures("patched_env")
class TestCharacterBERTRegression:

    def _make_tok(self):
        from arabic_eval.tokenizers.character_bert import CharacterBERTTokenizer
        tok = CharacterBERTTokenizer(max_char_len=10)
        tok.train(ARABIC_TEXTS, vocab_size=0)
        return tok

    def test_eow_always_present_in_char_ids(self, patched_env):
        """EOW marker must survive even for words that nearly fill max_char_len."""
        from arabic_eval.tokenizers.character_bert import BOW_CHAR, EOW_CHAR, PAD_CHAR

        tok = self._make_tok()
        long_word = "أ" * 15  # 15 chars >> max_char_len=10
        ids = tok._word_to_char_ids(long_word)

        assert len(ids) == tok.max_char_len, "char_ids must always be exactly max_char_len"
        assert ids[0] == BOW_CHAR, "first char must be BOW"
        assert EOW_CHAR in ids, "EOW must always be present even for long words"

    def test_special_tokens_have_distinct_char_ids(self, patched_env):
        """BOS and EOS char ID sequences must be distinct from each other."""
        from arabic_eval.tokenizers.character_bert import (
            BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
        )

        tok = self._make_tok()
        bos_ids = tok._word_to_char_ids(BOS_TOKEN)
        eos_ids = tok._word_to_char_ids(EOS_TOKEN)
        pad_ids = tok._word_to_char_ids(PAD_TOKEN)
        unk_ids = tok._word_to_char_ids(UNK_TOKEN)

        assert bos_ids != eos_ids, "BOS and EOS char sequences must differ"
        assert bos_ids != pad_ids
        assert eos_ids != pad_ids
        assert unk_ids != bos_ids

    def test_load_restores_next_word_id(self, patched_env):
        """After save/load, _next_word_id must reflect the loaded vocabulary."""
        tok = self._make_tok()
        with tempfile.TemporaryDirectory() as tmp:
            tok.save(tmp)
            from arabic_eval.tokenizers.character_bert import CharacterBERTTokenizer
            fresh = CharacterBERTTokenizer(max_char_len=10)
            fresh.load(tmp)
        assert fresh._next_word_id == fresh.vocab_size, (
            "_next_word_id must equal vocab_size after load"
        )


# ===========================================================================
# 3. Collation tests (correct tensor shapes per embedding type)
# ===========================================================================

@pytest.mark.usefixtures("patched_env")
class TestCollation:

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_batch_shapes(self, patched_env, tok_type, tok_params, vocab_size):
        from arabic_eval.data.collation import get_collator

        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        max_len = 32

        samples = []
        for text in ARABIC_TEXTS[:3]:
            enc = tok.encode(text, max_length=max_len, truncation=True)
            entry: Dict[str, Any] = {"input_ids": enc.input_ids}
            if enc.char_ids is not None:
                entry["char_ids"] = enc.char_ids
            samples.append(entry)

        collator = get_collator(tok.embedding_type, pad_token_id=tok.pad_token_id, max_length=max_len)
        batch = collator(samples)

        if tok.embedding_type == "character_cnn":
            assert "char_ids" in batch
            assert batch["char_ids"].dim() == 3        # [B, S, C]
            assert batch["char_ids"].shape[0] == 3
            assert "labels" in batch, "CharacterCNNCollator must produce labels"
            assert batch["labels"].shape == batch["char_ids"].shape[:2]
        else:
            assert "input_ids" in batch
            assert batch["input_ids"].dim() == 2       # [B, S]
            assert batch["input_ids"].shape[0] == 3
            assert "labels" in batch

        assert "attention_mask" in batch


# ===========================================================================
# 4. Model adaptation tests
# ===========================================================================

@pytest.mark.usefixtures("patched_env")
class TestModelAdaptation:

    def _make_adapter(self):
        from arabic_eval.models.llama_adapter import LlamaAdapter
        return LlamaAdapter(model_name_or_path="test", device="cpu", dtype="float32")

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_adapt_does_not_crash(self, patched_env, tok_type, tok_params, vocab_size):
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        adapter = self._make_adapter()
        adapter.adapt_to_tokenizer(tok)   # must not raise

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_embedding_type_after_adapt(self, patched_env, tok_type, tok_params, vocab_size):
        from arabic_eval.models.embeddings.character_cnn import CharacterCNNEmbedding
        from arabic_eval.models.embeddings.char_jaber_embed import CharJaberEmbedding, CharJaberOutputHead

        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        adapter = self._make_adapter()
        adapter.adapt_to_tokenizer(tok)

        inner = adapter.model.model
        if tok_type == "character_bert":
            assert isinstance(inner.embed_tokens, CharacterCNNEmbedding), (
                "CharacterBERT adaptation must install CharacterCNNEmbedding"
            )
        elif tok_type == "char_jaber":
            assert isinstance(inner.embed_tokens, CharJaberEmbedding), (
                "char_jaber adaptation must install CharJaberEmbedding"
            )
            assert isinstance(adapter.model.lm_head, CharJaberOutputHead), (
                "char_jaber adaptation must install CharJaberOutputHead"
            )
        else:
            assert isinstance(inner.embed_tokens, nn.Embedding)
            assert inner.embed_tokens.weight.shape[0] == tok.vocab_size


# ===========================================================================
# 5. Forward-pass tests (one batch, loss is a scalar)
# ===========================================================================

@pytest.mark.usefixtures("patched_env")
class TestForwardPass:

    def _adapted_adapter(self, tok_type, tok_params, vocab_size):
        from arabic_eval.models.llama_adapter import LlamaAdapter
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        adapter = LlamaAdapter(model_name_or_path="test", device="cpu", dtype="float32")
        adapter.adapt_to_tokenizer(tok)
        return adapter, tok

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_forward_yields_loss(self, patched_env, tok_type, tok_params, vocab_size):
        from arabic_eval.data.collation import get_collator

        adapter, tok = self._adapted_adapter(tok_type, tok_params, vocab_size)
        max_len = 32

        samples = []
        for text in ARABIC_TEXTS[:2]:
            enc = tok.encode(text, max_length=max_len, truncation=True)
            entry: Dict[str, Any] = {"input_ids": enc.input_ids}
            if enc.char_ids is not None:
                entry["char_ids"] = enc.char_ids
            samples.append(entry)

        collator = get_collator(tok.embedding_type, pad_token_id=tok.pad_token_id, max_length=max_len)
        batch = collator(samples)
        batch = {k: v.to(adapter.device) for k, v in batch.items()}

        output = adapter.forward(batch)

        assert "loss" in output and "logits" in output
        assert output["loss"] is not None, "loss must not be None (labels must reach the model)"
        assert output["loss"].dim() == 0, "loss must be a scalar tensor"
        assert torch.isfinite(output["loss"]), "loss must be finite"


# ===========================================================================
# 6. LightEval benchmark task tests
# ===========================================================================

@pytest.mark.usefixtures("patched_env")
class TestLightEvalBenchmarks:
    """Unit and integration tests for the four LightEval benchmark tasks.

    Coverage:
      - MCQ parsing with A/B/C/D column schema and choices-list schema
      - 10/90 split correctness (no overlap, correct proportions)
      - get_dataloader produces correctly shaped batches
      - evaluate() returns valid accuracy for standard tokenizers
      - evaluate() returns accuracy=0.0 for CharacterBERT (expected limitation)
      - LightEvalModelWrapper.loglikelihood returns one float per request
    """

    # --- helpers ---

    def _make_task(self, task_type: str = "acva", extra: Optional[Dict] = None):
        import arabic_eval.tasks  # noqa: F401
        from arabic_eval.registry import task_registry
        params = {
            "dataset_name": "OALL/ACVA",
            "train_split_ratio": 0.50,
            "max_length": 32,
            "seed": 0,
        }
        if extra:
            params.update(extra)
        cls = task_registry.get(task_type)
        return cls(params)

    def _make_adapted_adapter(self, tok_type="bpe", tok_params=None, vocab_size=100):
        from arabic_eval.models.llama_adapter import LlamaAdapter
        tok_params = tok_params or {"min_frequency": 1}
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        adapter = LlamaAdapter(model_name_or_path="test", device="cpu", dtype="float32")
        adapter.adapt_to_tokenizer(tok)
        return adapter, tok

    # --- MCQ parsing ---

    @pytest.mark.parametrize("task_type", ["acva", "alghafa", "culture_arabic_mmlu", "arabic_exam"])
    def test_parse_example_abcd_schema(self, patched_env, task_type):
        """_parse_example must handle the A/B/C/D column + answer letter schema."""
        import arabic_eval.tasks  # noqa
        from arabic_eval.registry import task_registry
        task = task_registry.get(task_type)({"dataset_name": "OALL/ACVA"})
        raw = {
            "question": "ما هي عاصمة الجزائر؟",
            "A": "الجزائر", "B": "وهران", "C": "قسنطينة", "D": "عنابة",
            "answer": "A",
        }
        parsed = task._parse_example(raw)
        assert parsed is not None, f"{task_type}: _parse_example returned None for valid ABCD row"
        assert parsed["question"] == "ما هي عاصمة الجزائر؟"
        assert len(parsed["choices"]) == 4
        assert parsed["answer"] == 0  # A → index 0

    def test_parse_example_choices_list_schema(self, patched_env):
        """_parse_example must handle the choices-list + integer answer schema."""
        from arabic_eval.tasks.lighteval_benchmarks import ACVATask
        task = ACVATask({"dataset_name": "OALL/ACVA"})
        raw = {
            "question": "كم عدد الأيام في الأسبوع؟",
            "choices": ["خمسة", "ستة", "سبعة", "ثمانية"],
            "answer": 2,
        }
        parsed = task._parse_example(raw)
        assert parsed is not None
        assert parsed["answer"] == 2
        assert parsed["choices"][2] == "سبعة"

    def test_parse_example_returns_none_for_invalid(self, patched_env):
        """_parse_example must return None for malformed rows (no choices)."""
        from arabic_eval.tasks.lighteval_benchmarks import ACVATask
        task = ACVATask({"dataset_name": "OALL/ACVA"})
        assert task._parse_example({"question": "سؤال بدون خيارات"}) is None
        assert task._parse_example({}) is None

    # --- Data splitting ---

    def test_split_no_overlap(self, patched_env):
        """Fine-tune and eval splits must be disjoint."""
        task = self._make_task()
        fine_tune, eval_examples = task._get_splits()
        assert len(fine_tune) > 0, "fine-tune split must not be empty"
        assert len(eval_examples) > 0, "eval split must not be empty"
        ft_qs = {ex["question"] for ex in fine_tune}
        ev_qs = {ex["question"] for ex in eval_examples}
        assert ft_qs.isdisjoint(ev_qs), "fine-tune and eval splits must not share questions"

    def test_split_is_deterministic(self, patched_env):
        """Two task instances with the same seed must produce identical splits."""
        task_a = self._make_task()
        task_b = self._make_task()
        ft_a, ev_a = task_a._get_splits()
        ft_b, ev_b = task_b._get_splits()
        assert [ex["question"] for ex in ft_a] == [ex["question"] for ex in ft_b]
        assert [ex["question"] for ex in ev_a] == [ex["question"] for ex in ev_b]

    def test_split_different_seeds_differ(self, patched_env):
        """Tasks with different seeds must (very likely) produce different splits."""
        task_a = self._make_task(extra={"seed": 0})
        task_b = self._make_task(extra={"seed": 99})
        ft_a, _ = task_a._get_splits()
        ft_b, _ = task_b._get_splits()
        qs_a = [ex["question"] for ex in ft_a]
        qs_b = [ex["question"] for ex in ft_b]
        assert qs_a != qs_b, "Different seeds should produce different orderings"

    # --- DataLoader ---

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
    def test_get_dataloader_produces_batches(self, patched_env, tok_type, tok_params, vocab_size):
        """get_dataloader must yield at least one batch for every tokenizer type."""
        task = self._make_task()
        tok = _build_tokenizer(tok_type, tok_params, vocab_size)
        dl = task.get_dataloader(tok, split="train", batch_size=2)
        batch = next(iter(dl))

        if tok_type == "character_bert":
            assert "char_ids" in batch
            assert batch["char_ids"].dim() == 3
        else:
            assert "input_ids" in batch
            assert batch["input_ids"].dim() == 2

        assert "attention_mask" in batch
        assert "labels" in batch

    # --- Evaluation ---

    @pytest.mark.parametrize("tok_type,tok_params,vocab_size", [
        pytest.param("bpe",       {"min_frequency": 1}, 100, id="bpe"),
        pytest.param("wordpiece", {"min_frequency": 1}, 100, id="wordpiece"),
        pytest.param("char_jaber", {},                  None, id="char_jaber"),
    ])
    def test_evaluate_returns_valid_accuracy(self, patched_env, tok_type, tok_params, vocab_size):
        """evaluate() must return accuracy in [0, 1] for standard tokenizer types."""
        task = self._make_task()
        adapter, tok = self._make_adapted_adapter(tok_type, tok_params, vocab_size)
        metrics = task.evaluate(adapter, tok, split="test", max_samples=2)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "num_samples" in metrics
        assert metrics["num_samples"] >= 0

    def test_character_bert_evaluate_returns_zero_accuracy(self, patched_env):
        """CharacterBERT must return accuracy=0.0 — log-likelihood not supported."""
        task = self._make_task()
        adapter, tok = self._make_adapted_adapter(
            "character_bert", {"max_char_len": 10, "max_word_vocab": 200}, None
        )
        metrics = task.evaluate(adapter, tok, split="test", max_samples=2)
        assert metrics["accuracy"] == 0.0, (
            "CharacterBERT (character_cnn) must always report accuracy=0.0 "
            "because token-level log-likelihood is not supported"
        )

    # --- LightEvalModelWrapper ---

    def test_loglikelihood_returns_one_float_per_request(self, patched_env):
        """LightEvalModelWrapper.loglikelihood must return exactly one float per pair."""
        from arabic_eval.tasks.lighteval_benchmarks import LightEvalModelWrapper
        adapter, tok = self._make_adapted_adapter()
        wrapper = LightEvalModelWrapper(adapter, tok, max_length=32)
        requests = [
            ("السؤال: ما هي عاصمة الجزائر؟\nالإجابة:", " A"),
            ("السؤال: ما هي عاصمة الجزائر؟\nالإجابة:", " B"),
            ("السؤال: ما هي عاصمة الجزائر؟\nالإجابة:", " C"),
        ]
        scores = wrapper.loglikelihood(requests)
        assert len(scores) == 3, "Must return exactly one score per request"
        assert all(isinstance(s, float) for s in scores), "All scores must be floats"

    def test_loglikelihood_scores_are_finite(self, patched_env):
        """Log-likelihoods must be finite (not NaN or ±inf)."""
        from arabic_eval.tasks.lighteval_benchmarks import LightEvalModelWrapper
        adapter, tok = self._make_adapted_adapter()
        wrapper = LightEvalModelWrapper(adapter, tok, max_length=32)
        scores = wrapper.loglikelihood([
            ("السياق العربي", " A"),
            ("السياق العربي", " B"),
        ])
        import math
        assert all(math.isfinite(s) for s in scores), "All log-likelihood scores must be finite"

    def test_evaluate_mcq_returns_correct_keys(self, patched_env):
        """evaluate_mcq must return both 'accuracy' and 'num_samples'."""
        from arabic_eval.tasks.lighteval_benchmarks import LightEvalModelWrapper
        adapter, tok = self._make_adapted_adapter()
        wrapper = LightEvalModelWrapper(adapter, tok, max_length=32)
        examples = [
            {"question": "ما هي عاصمة الجزائر؟", "choices": ["الجزائر", "وهران", "قسنطينة", "عنابة"], "answer": 0},
            {"question": "كم عدد أيام الأسبوع؟", "choices": ["خمسة", "سبعة", "ستة", "ثمانية"], "answer": 1},
        ]
        result = wrapper.evaluate_mcq(examples)
        assert "accuracy" in result and "num_samples" in result
        assert result["num_samples"] == 2
        assert 0.0 <= result["accuracy"] <= 1.0

    # --- Multi-config dataset fallback ---

    def test_multi_config_fallback_loads_all_sub_configs(self, patched_env):
        """When load_dataset raises 'Config name is missing', _load_all_configs_merged
        must enumerate sub-configs and merge their examples."""
        from arabic_eval.tasks.lighteval_benchmarks import ACVATask

        mcq_ds = _make_mcq_dataset()

        def _load_requiring_config(name, config=None, cache_dir=None, **kwargs):
            if config is None:
                raise ValueError(
                    "Config name is missing. Pick one among: ['cfg_a', 'cfg_b']"
                )
            from datasets import DatasetDict
            return DatasetDict({"train": mcq_ds})

        def _config_names(name):
            return ["cfg_a", "cfg_b"]

        with (
            patch("arabic_eval.tasks.lighteval_benchmarks.load_dataset", side_effect=_load_requiring_config),
            patch("arabic_eval.tasks.lighteval_benchmarks.get_dataset_config_names", side_effect=_config_names),
        ):
            task = ACVATask({"dataset_name": "OALL/ACVA", "train_split_ratio": 0.50})
            examples = task._load_all_examples()

        # 2 configs × 8 rows each = 16 examples total
        assert len(examples) == 2 * len(SAMPLE_MCQ_DATA["question"]), (
            "Multi-config merge must include examples from all sub-configs"
        )

    def test_multi_config_skips_broken_sub_configs(self, patched_env):
        """Broken sub-configs must be skipped with a warning, not raise."""
        from arabic_eval.tasks.lighteval_benchmarks import ACVATask

        mcq_ds = _make_mcq_dataset()

        def _load_requiring_config(name, config=None, cache_dir=None, **kwargs):
            if config is None:
                raise ValueError("Config name is missing.")
            if config == "broken":
                raise RuntimeError("Dataset not found")
            from datasets import DatasetDict
            return DatasetDict({"train": mcq_ds})

        def _config_names(name):
            return ["good", "broken"]

        with (
            patch("arabic_eval.tasks.lighteval_benchmarks.load_dataset", side_effect=_load_requiring_config),
            patch("arabic_eval.tasks.lighteval_benchmarks.get_dataset_config_names", side_effect=_config_names),
        ):
            task = ACVATask({"dataset_name": "OALL/ACVA"})
            examples = task._load_all_examples()

        # Only the "good" config's examples should be present
        assert len(examples) == len(SAMPLE_MCQ_DATA["question"])

    # --- All four benchmark task types via task registry ---

    @pytest.mark.parametrize("task_type,dataset_name", [
        pytest.param("acva",               "OALL/ACVA",               id="acva"),
        pytest.param("alghafa",            "OALL/AlGhafa-Native",     id="alghafa"),
        pytest.param("culture_arabic_mmlu","acmc/arabic_culture_mmlu",id="culture_arabic_mmlu"),
        pytest.param("arabic_exam",        "arabic_exam",             id="arabic_exam"),
    ])
    def test_all_benchmark_tasks_load_examples(self, patched_env, task_type, dataset_name):
        """Every registered benchmark task must load >0 examples from the mock."""
        import arabic_eval.tasks  # noqa
        from arabic_eval.registry import task_registry
        task = task_registry.get(task_type)({
            "dataset_name": dataset_name,
            "train_split_ratio": 0.50,
            "max_length": 32,
            "seed": 0,
        })
        all_ex = task._load_all_examples()
        assert len(all_ex) > 0, f"{task_type}: no examples loaded from mock dataset"
        for ex in all_ex:
            assert "question" in ex and "choices" in ex and "answer" in ex


# ===========================================================================
# 7. Full end-to-end sweep simulation  (run_single_experiment)
# ===========================================================================

def _make_experiment_config(tok_type, tok_params, vocab_size, task_type, task_params, out_dir):
    """Build a minimal ExperimentConfig that can run without GPU or internet."""
    from arabic_eval.config import (
        DataConfig, EvaluationConfig, ExperimentConfig,
        ModelConfig, TaskConfig, TokenizerConfig, TrainingConfig,
    )
    return ExperimentConfig(
        name=f"test_{tok_type}_{task_type}",
        output_dir=str(out_dir),
        seed=0,
        deterministic=False,
        data=DataConfig(
            dataset_name="test-dataset",
            max_train_samples=6,
            max_eval_samples=4,
            preprocessing={"normalize_unicode": True, "remove_diacritics": False, "min_text_length": 5},
        ),
        tokenizer=TokenizerConfig(
            type=tok_type,
            vocab_size=vocab_size,
            params=tok_params,
            save_path=str(out_dir / "tokenizer"),
        ),
        model=ModelConfig(
            type="llama",
            name_or_path="test-llama",
            dtype="float32",
            device="cpu",
        ),
        task=TaskConfig(type=task_type, params=task_params),
        training=TrainingConfig(
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            bf16=False,
            fp16=False,
            save_steps=0,
            eval_steps=0,
            logging_steps=10_000,
            early_stopping_patience=None,
        ),
        evaluation=EvaluationConfig(
            intrinsic_metrics=True,
            downstream_metrics=True,
            num_eval_samples=2,
        ),
    )


@pytest.mark.parametrize("tok_type,tok_params,vocab_size", _TOK_CASES)
@pytest.mark.parametrize("task_type,task_params", _TASK_CASES)
def test_e2e_full_sweep(patched_env, tok_type, tok_params, vocab_size, task_type, task_params):
    """Run run_single_experiment end-to-end for every (tokenizer × task) pair.

    Asserts:
      - No exception is raised.
      - Result contains intrinsic and training sections.
      - Downstream section is present.
      - Losses / metrics are finite numbers.
      - LightEval tasks: accuracy is in [0, 1]; CharacterBERT yields 0.0.
    """
    from pathlib import Path
    from arabic_eval.pipeline.experiment import run_single_experiment

    with tempfile.TemporaryDirectory() as tmp:
        config = _make_experiment_config(
            tok_type, tok_params, vocab_size,
            task_type, task_params,
            Path(tmp),
        )

        result = run_single_experiment(config)

    # Top-level structure
    assert "intrinsic" in result, "Result must contain intrinsic metrics"
    assert "training" in result, "Result must contain training results"
    assert "downstream" in result, "Result must contain downstream metrics"

    # Intrinsic sanity
    intrinsic = result["intrinsic"]
    assert intrinsic.get("vocab_size", 0) > 0
    assert 0.0 <= intrinsic.get("unk_rate", 1.0) <= 1.0

    # Training sanity
    training = result["training"]
    assert "train_loss" in training
    train_loss = training["train_loss"]
    assert isinstance(train_loss, float) and train_loss >= 0, (
        f"train_loss must be a non-negative float, got {train_loss}"
    )

    # Downstream sanity
    downstream = result["downstream"]
    assert task_type in downstream
    task_metrics = downstream[task_type]
    for metric_val in task_metrics.values():
        if isinstance(metric_val, (int, float)):
            assert metric_val >= 0, f"Downstream metric must be >= 0, got {metric_val}"

    # LightEval-specific: accuracy must be in [0, 1]
    is_lighteval = task_type in {"acva", "alghafa", "culture_arabic_mmlu", "arabic_exam"}
    if is_lighteval and "accuracy" in task_metrics:
        assert 0.0 <= task_metrics["accuracy"] <= 1.0, (
            f"LightEval accuracy must be in [0,1], got {task_metrics['accuracy']}"
        )
        if tok_type == "character_bert":
            assert task_metrics["accuracy"] == 0.0, (
                "CharacterBERT must always report accuracy=0.0 for LightEval tasks"
            )
