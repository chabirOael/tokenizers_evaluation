"""End-to-end smoke test for the morphological metric panel.

Trains a small tokenizer per architectural family on a hand-picked Arabic
corpus and runs `compute_morphological_metrics` on the same word sample.
Asserts the documented mechanical extremes — every assertion failure is
loud and points at the responsible row.

Real Farasa, real qalsadi, real tokenizers. Slow (~30–60s on CPU because
of Farasa subprocess startup + per-word segmentation). Run after any
change to `intrinsic_metrics.py` to validate fairness across families.

Usage:  .venv/bin/python scripts/smoke_morph_metrics.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.evaluation.intrinsic_metrics import (
    compute_morphological_metrics,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke_morph")


# 60 short Arabic sentences with a deliberate mix of:
#   - clitics (و, ال, ف, ب, ل, ك, س, ها, هم, ك, ي, نا)
#   - sound triliteral roots (كتب, درس, علم)
#   - weak roots (قال, مشى, جرى)
#   - common nouns + verbs at multiple tenses
CORPUS: List[str] = [
    "كتب الطالب الدرس في المدرسة",
    "يدرس الطلاب اللغة العربية في المعهد",
    "قرأت الكتاب يوم الأحد بعد الظهر",
    "ذهب المعلم إلى الفصل مبكرا اليوم",
    "كتبت الرسالة بالقلم الأزرق الجميل",
    "والكتاب الذي قرأته أعجبني كثيرا",
    "كتابهم مفيد جدا للطلاب الجدد",
    "وكاتب الرواية مشهور في العالم العربي",
    "العلم نور والجهل ظلام كما يقال",
    "في الكتاب آلاف الكلمات النادرة",
    "السماء صافية والشمس مشرقة اليوم",
    "البحر هادئ والرياح خفيفة جدا",
    "الشمس تشرق من الشرق كل صباح",
    "القمر يضيء الليل بنوره الجميل",
    "النجوم تلمع في السماء البعيدة",
    "تطبيق الرياضة مهم للصحة العامة",
    "الطعام الصحي يفيد الجسم والعقل",
    "الماء عنصر أساسي للحياة على الأرض",
    "الأشجار تعطي الأكسجين للهواء",
    "السلام عليكم ورحمة الله وبركاته",
    "وأهلا وسهلا بكم في بيتنا الكبير",
    "اللغة العربية لغة جميلة ومعبرة",
    "الجزائر بلد جميل في شمال أفريقيا",
    "التعلم الآلي علم يدرس البيانات",
    "النص العربي يحتاج إلى معالجة خاصة",
    "البرمجة اللغوية تتطلب فهما عميقا",
    "الذكاء الاصطناعي يغير حياة البشر",
    "النماذج اللغوية الكبيرة قادرة على الفهم",
    "تقنيات التعلم العميق تعتمد على الشبكات",
    "كل عام وأنتم بخير وسعادة",
    "الكتاب خير جليس في الزمان",
    "العلم في الصغر كالنقش على الحجر",
    "من جد وجد ومن زرع حصد",
    "الصبر مفتاح الفرج كما قالوا",
    "التلميذ المجتهد يفوز دائما بالجائزة",
    "كاتبهم لم يحضر الاجتماع أمس",
    "وللطلاب حقوق وواجبات في المدرسة",
    "بكتابك تستطيع تعلم أشياء كثيرة",
    "فالمدرس ساعد الطلاب على الفهم",
    "ولها صديقات كثيرات في الجامعة",
    "كتب الكاتب مقالا جيدا عن السياسة",
    "الكتب التي قرأتها مفيدة جدا",
    "ولدي يحب اللعب بالكرة في الحديقة",
    "أمي تطبخ طعاما لذيذا كل يوم",
    "أبي يعمل في شركة كبيرة بالعاصمة",
    "صديقي وأخوه يدرسان في نفس المدرسة",
    "وذهبنا إلى السوق لشراء الفواكه",
    "الفلاحون يزرعون القمح في الحقول",
    "الطبيب يعالج المرضى في المستشفى",
    "المهندس يصمم المباني العالية",
    "الجنود يحرسون الحدود ليلا ونهارا",
    "والشاعر يكتب قصائد رائعة عن الحب",
    "الفنان يرسم لوحات جميلة بالألوان",
    "الموسيقى تريح النفس وتسعد القلب",
    "الكاتب الجديد نشر روايته الأولى",
    "والمؤلف وقع على نسختي بالأمس",
    "بكتاباتها تستطيع التعبير عن مشاعرها",
    "فبكتابهم الجديد نفهم أفكارهم",
    "ولدراستهم أهمية كبيرة في حياتهم",
    "كأنك تعرف كل شيء عن هذا الموضوع",
] * 4  # 240 sentences total — gives the metric a deeper word pool.


# Fairness bands per tokenizer (each (lo, hi) tuple is INCLUSIVE).
# `None` means "no bound on this side." `EXACT(x)` means must equal x.
class EXACT:
    def __init__(self, value):
        self.value = value


# Each band is a dict: metric -> band specification.
#   - tuple (lo, hi): real number, lo <= v <= hi (None to skip a side)
#   - EXACT(0.0): must be exactly 0.0 (mechanical extreme)
#   - "real": any real number, just not None
#   - "real_in_unit": any real in [0,1]
EXPECTED_BANDS: Dict[str, Dict[str, Any]] = {
    "char_jaber": {
        # char-level: every char boundary is a token boundary, so clitic
        # boundaries trivially align. Each Arabic char is one token, so
        # SFR is large.
        "clitic_separation_accuracy": (0.95, 1.0),
        "morpheme_integrity_rate": (0.95, 1.0),
        "semantic_fragmentation_ratio": (2.0, None),
        "root_conservation_rate": (None, 0.10),
        "root_bearing_token_pct": (None, 1.0),  # tokens are single chars
    },
    "charformer": {
        # byte-level: Arabic chars span 2 bytes. Byte tokens never
        # reconstruct to Arabic-letter offsets, so alignment-dependent
        # metrics (integrity, CSA) are mechanically *not measurable*
        # (return None) — SFR is the discriminator. Bearing-token
        # metrics are mechanical zeros (raw_token_count > 0 but cleaned
        # tokens are empty Arabic-letter strings).
        "clitic_separation_accuracy": EXACT(None),
        "morpheme_integrity_rate": EXACT(None),
        "semantic_fragmentation_ratio": (4.0, None),
        "root_bearing_token_pct": EXACT(0.0),
        "pattern_bearing_token_pct": EXACT(0.0),
    },
    "character_bert": {
        # word-level CharCNN: never splits a word, so internal token
        # boundaries don't exist → integrity/CSA are mechanical zeros.
        # root_conservation is high but not at the architectural ceiling
        # because qalsadi extracts roots for weak/irregular forms that
        # aren't literal subsequences of the word (e.g. defective verbs).
        "clitic_separation_accuracy": (0.0, 0.10),
        "morpheme_integrity_rate": (0.0, 0.10),
        "semantic_fragmentation_ratio": (0.20, 0.80),
        "root_conservation_rate": (0.50, 1.0),
    },
    "morpho_bpe": {
        # Farasa-aware: integrity by design.
        "morpheme_integrity_rate": (0.85, 1.0),
        "clitic_separation_accuracy": (0.85, 1.0),
        "semantic_fragmentation_ratio": (0.7, 1.5),
    },
    "bpe": {
        # plain subword: discriminating range. Just confirm metrics are real.
        "clitic_separation_accuracy": (0.0, 1.0),
        "morpheme_integrity_rate": (0.0, 1.0),
        "semantic_fragmentation_ratio": (0.5, 5.0),
        "root_conservation_rate": (0.0, 1.0),
    },
}


def _build_tokenizer(name: str):
    """Construct + train a tokenizer of the given registry key."""
    import arabic_eval.tokenizers  # noqa: F401  — registers all
    from arabic_eval.registry import tokenizer_registry

    cls = tokenizer_registry.get(name)
    if name == "charformer":
        tok = cls()
        tok.train(CORPUS)
    elif name == "character_bert":
        tok = cls(max_char_len=20)
        tok.train(CORPUS, vocab_size=0)  # vocab grows from corpus
    elif name == "char_jaber":
        tok = cls()
        tok.train(CORPUS)
    elif name == "morpho_bpe":
        tok = cls()
        tok.train(CORPUS, vocab_size=2000)
    elif name in ("bpe", "wordpiece"):
        tok = cls()
        tok.train(CORPUS, vocab_size=2000)
    else:
        raise ValueError(f"unknown tokenizer: {name!r}")
    return tok


def _check_band(name: str, metric: str, value, spec) -> List[str]:
    """Return list of human-readable failure messages for this band."""
    failures: List[str] = []
    if isinstance(spec, EXACT):
        if value != spec.value:
            failures.append(
                f"  [{name}] {metric} = {value!r} but must equal "
                f"{spec.value!r} (mechanical extreme)"
            )
        return failures
    if spec == "real":
        if value is None:
            failures.append(f"  [{name}] {metric} is None — must be a real number")
        return failures
    if spec == "real_in_unit":
        if value is None or not (0.0 <= value <= 1.0):
            failures.append(
                f"  [{name}] {metric} = {value!r} — must be a real in [0,1]"
            )
        return failures
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if value is None:
            failures.append(
                f"  [{name}] {metric} is None — band requires a real number "
                f"in [{lo}, {hi}]"
            )
            return failures
        if lo is not None and value < lo:
            failures.append(
                f"  [{name}] {metric} = {value} below lower bound {lo}"
            )
        if hi is not None and value > hi:
            failures.append(
                f"  [{name}] {metric} = {value} above upper bound {hi}"
            )
    return failures


def _check_invariants(name: str, m: Dict[str, Any]) -> List[str]:
    """Cross-metric invariants that should hold for every tokenizer."""
    failures: List[str] = []
    integ = m.get("morpheme_integrity_rate")
    csa = m.get("clitic_separation_accuracy")
    sfr = m.get("semantic_fragmentation_ratio")

    # Invariant: integrity == 1.0 ⇒ CSA == 1.0 (clitic boundaries ⊆ all
    # morpheme boundaries; if every morpheme boundary is respected, every
    # clitic boundary is too).
    if integ is not None and abs(integ - 1.0) < 1e-6:
        if csa is not None and abs(csa - 1.0) > 1e-6:
            failures.append(
                f"  [{name}] integrity=1.0 but CSA={csa} — invariant violated"
            )

    # SFR must be a real number for any tokenizer where Farasa worked.
    # Detected by integrity OR CSA being non-None.
    if (integ is not None or csa is not None) and sfr is None:
        failures.append(
            f"  [{name}] SFR is None but integrity/CSA computed — SFR is "
            "incorrectly gated on alignment"
        )
    return failures


def main() -> int:
    print(f"Smoke: {len(CORPUS)} sentences, sample_size=120")
    print("=" * 72)

    results: Dict[str, Dict[str, Any]] = {}
    all_failures: List[str] = []

    for name in ["bpe", "morpho_bpe", "character_bert", "char_jaber", "charformer"]:
        print(f"\n[{name}] training + computing metrics...")
        try:
            tok = _build_tokenizer(name)
            m = compute_morphological_metrics(tok, CORPUS, sample_size=120)
        except Exception as e:
            print(f"  FAILED to build/run: {e!r}")
            all_failures.append(f"[{name}] errored: {e!r}")
            continue
        results[name] = m

        print(f"  root_conservation_rate     : {m.get('root_conservation_rate')}")
        print(f"  pattern_conservation_rate  : {m.get('pattern_conservation_rate')}")
        print(f"  morpheme_integrity_rate    : {m.get('morpheme_integrity_rate')}")
        print(f"  clitic_separation_accuracy : {m.get('clitic_separation_accuracy')}")
        print(f"  semantic_fragmentation_ratio: {m.get('semantic_fragmentation_ratio')}")
        print(f"  root_bearing_token_pct     : {m.get('root_bearing_token_pct')}")
        print(f"  pattern_bearing_token_pct  : {m.get('pattern_bearing_token_pct')}")

        bands = EXPECTED_BANDS.get(name, {})
        for metric, spec in bands.items():
            all_failures.extend(_check_band(name, metric, m.get(metric), spec))
        all_failures.extend(_check_invariants(name, m))

    # Save results for downstream diffing.
    out_path = Path("outputs/smoke_morph_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 72)
    if all_failures:
        print(f"FAIL — {len(all_failures)} band/invariant violation(s):")
        for f in all_failures:
            print(f)
        return 1
    print("OK — all bands and invariants pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
