"""The free-form eval's text rules applied to teacher answers (main venv).

Two uses:

* ``eval_row`` — a teacher answer as a row of a free-form dump (the bake-off
  pseudo-cells and the ceiling cell): the loop stop with **no stop markers**
  (the markers are our template's, not the teacher's), the 2 400-character
  budget, ``is_degenerate`` on the raw text, the ``latin`` flag and the
  Arabic-letter ratio — exactly the fields ``FreeformCidarTask`` writes.
* ``drop_reason`` — rule 3 of the distillation brief ("don't copy the teacher's
  mistakes"): a training answer is dropped, in this order, when it did not end
  in EOS (``length``), is longer than the budget (``char_truncated`` — dropped,
  not cut), ends in a repetition loop (``loop``), is degenerate by the density
  rules (``degenerate``), is empty or under 20 characters (``empty``), carries
  a Latin letter (``latin``, the eval's row flag), or echoes our template
  (contains one of its section labels) or the instruction (``echo``). A stock
  preamble is counted, never dropped (``has_preamble``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from arabic_eval.data import finetune_corpora as T
from arabic_eval.data.contamination import normalize_text
from arabic_eval.tasks.freeform import metrics as M
from arabic_eval.tasks.freeform.generation import DecodingConfig, text_stop
from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters

MAX_OUTPUT_CHARS = DecodingConfig().max_output_chars       # 2 400
MIN_ANSWER_CHARS = 20
TEMPLATE_LABELS = (T.INSTRUCTION_LABEL, T.INPUT_LABEL, T.CONTEXT_LABEL, T.QUESTION_LABEL, T.ANSWER_LABEL)
PREAMBLES = ("بالتأكيد", "بالطبع", "إليك", "فيما يلي", "حسنًا", "نعم")
_PREAMBLES_NORM = tuple(normalize_text(p) for p in PREAMBLES)
DROP_REASONS = ("length", "char_truncated", "loop", "degenerate", "empty", "latin", "echo")


def eval_row(text: Optional[str], finish_reason: str, max_output_chars: int = MAX_OUTPUT_CHARS) -> Dict[str, Any]:
    """The per-row fields of a free-form dump for one teacher answer.
    ``finish_reason`` is vLLM's: ``"stop"`` = the model's EOS / end of turn,
    ``"length"`` = the token cap. A loop at the tail of the text makes the row
    ``stop_reason="loop"`` with ``generation`` cut right after the first copy,
    as the eval's loop stop does; ``degenerate`` is judged on the raw text."""
    raw = text or ""
    stop = text_stop(raw, (), loop_stop=True)
    gen = stop.text.strip()
    char_truncated = len(gen) > max_output_chars
    if char_truncated:
        gen = gen[:max_output_chars].rstrip()
    if stop.reason == "loop":
        reason = "loop"
    elif finish_reason == "stop":
        reason = "eos"
    else:
        reason = "cap"
    loop = stop.loop if reason == "loop" else None
    return {
        "generation": gen, "generation_raw": raw, "stop_reason": reason,
        "hit_cap": reason == "cap", "hit_loop": reason == "loop",
        "loop_rule": loop.rule if loop is not None else None,
        "loop_period": loop.period if loop is not None else None,
        "char_truncated": char_truncated, "empty": not gen.strip(),
        "degenerate": M.is_degenerate(raw), "latin": contains_latin_letters(gen),
        "arabic_letter_ratio": M.arabic_letter_ratio(gen),
    }


def is_echo(answer: str, instruction: str) -> bool:
    """The answer carries one of our template's section labels, or is the instruction."""
    if any(label in answer for label in TEMPLATE_LABELS):
        return True
    na = normalize_text(answer)
    return bool(na) and na == normalize_text(instruction)


def drop_reason(text: Optional[str], finish_reason: str, instruction: str,
                max_output_chars: int = MAX_OUTPUT_CHARS) -> Optional[str]:
    """The first rule a training answer fails (``DROP_REASONS`` order), or ``None`` to keep it."""
    answer = (text or "").strip()
    if finish_reason != "stop":
        return "length"
    if len(answer) > max_output_chars:
        return "char_truncated"
    if M.detect_loop(answer) is not None:
        return "loop"
    if M.is_degenerate(answer):
        return "degenerate"
    if len(answer) < MIN_ANSWER_CHARS:
        return "empty"
    if contains_latin_letters(answer):
        return "latin"
    if is_echo(answer, instruction):
        return "echo"
    return None


def has_preamble(text: Optional[str]) -> bool:
    """The answer opens with a stock preamble (بالتأكيد / بالطبع / إليك / فيما يلي / حسنًا / نعم),
    compared after the contamination normalization (alef / diacritics / punctuation folded)."""
    head = normalize_text((text or "")[:40])
    return any(head == p or head.startswith(p + " ") for p in _PREAMBLES_NORM)
