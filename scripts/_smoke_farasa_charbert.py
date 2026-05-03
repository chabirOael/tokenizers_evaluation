"""Smoke test for FarasaCharacterBERTTokenizer.

Confirms:
  1. Training succeeds end-to-end (Farasa segmentation + char/morpheme vocabs).
  2. encode() populates tokens, input_ids, char_ids, and attention_mask correctly.
  3. Morphological metrics compute without None leakage on subword-style outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.tokenizers.farasa_character_bert import FarasaCharacterBERTTokenizer
from arabic_eval.evaluation.intrinsic_metrics import compute_morphological_metrics


SEED_TEXTS = [
    "الكتاب على الطاولة في غرفة الجلوس",
    "ذهب الطالب إلى المدرسة في الصباح الباكر",
    "تكتب المعلمة الدرس على السبورة بعناية",
    "يقرأ الأطفال القصص المصورة في المكتبة",
    "الشمس مشرقة والسماء صافية اليوم",
    "والكتاب الذي قرأته أمس كان رائعا",
    "بالمدرسة معلمون مخلصون يساعدون الطلاب",
    "للطالب المجتهد مكافأة كبيرة من المدير",
    "كتبت الطالبة رسالة لأمها بخط جميل",
    "يدرس المهندسون مشاريع البناء بدقة",
    "السيارة الجديدة سريعة جدا في الطريق",
    "المدينة العربية مزدحمة بالناس والسيارات",
    "وصل القطار إلى المحطة في الموعد المحدد",
    "الحديقة العامة مليئة بالأشجار والورود",
    "يلعب الأولاد كرة القدم في الملعب",
    "الطعام العربي لذيذ ومتنوع جدا",
    "تحدث الرئيس عن الإصلاحات الاقتصادية الجديدة",
    "العلم نور والجهل ظلام في حياة الإنسان",
    "يصلي المسلمون خمس مرات في اليوم",
    "السلام عليكم ورحمة الله وبركاته",
]
TEXTS = SEED_TEXTS * 10  # 200 sentences

print(f"[1/3] Training on {len(TEXTS)} sentences (vocab cap=500)...")
tok = FarasaCharacterBERTTokenizer(max_char_len=20)
tok.train(TEXTS, vocab_size=500, max_word_vocab=500)
print(f"      char_vocab={tok.char_vocab_size}  morpheme_vocab={tok.vocab_size}")

print("[2/3] Encoding a sample sentence...")
sample = "والكتاب الذي قرأته أمس كان رائعا"
enc = tok.encode(sample, max_length=64, padding=True, truncation=True)
print(f"      input_ids len = {len(enc.input_ids)}")
print(f"      attention_mask sum = {sum(enc.attention_mask)}")
print(f"      tokens (first 10) = {enc.tokens[:10]}")
assert enc.char_ids is not None, "char_ids must be populated for character_cnn"
print(f"      char_ids shape = ({len(enc.char_ids)}, {len(enc.char_ids[0])})")
assert len(enc.char_ids[0]) == tok.max_char_len, "char vector length mismatch"
assert len(enc.input_ids) == len(enc.char_ids), "input_ids/char_ids length mismatch"
print("      ✓ shapes consistent")

print("[3/3] Computing morphological metrics (Farasa enabled, sample=40)...")
m = compute_morphological_metrics(tok, TEXTS, sample_size=40, use_farasa=True)
for k, v in m.items():
    print(f"      {k}: {v}")

none_leaks = [k for k, v in m.items() if v is None]
if none_leaks:
    print(f"\n  WARNING: None leakage on {none_leaks}")
else:
    print("\n  ✓ No None leakage")

print("\nDone.")
