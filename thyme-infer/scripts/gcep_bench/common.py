# Modified for anonymous review: release paths, configuration, and documentation.
"""Shared answer-extraction / normalization helpers for gcep benchmark scorers.

Pure stdlib + no torch. All functions operate on the scorer *input view* of the
model response; they never mutate the immutable ``raw_response``.
"""

from __future__ import annotations

import re
from typing import Optional

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAIL_RE = re.compile(r".*</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def strip_think_tags(text: str) -> str:
    """Return a scorer-input view with thinking blocks removed (view only).

    Handles both shapes:
    * paired ``<think>...</think>`` (fully generated thinking block), and
    * a lone ``</think>`` close tag — emitted by thinking chat templates that
      open the block inside the prompt (e.g. Qwen3-VL-*-Thinking), where the
      generated text is ``reasoning...</think>answer``. Everything up to and
      including the LAST ``</think>`` is dropped so the answer survives.

    Non-thinking outputs contain no ``</think>`` and are returned unchanged.
    """
    view = _THINK_RE.sub(" ", text or "")
    if "</think>" in view:
        view = _THINK_TAIL_RE.sub(" ", view)
    return view


def extract_answer_tag(text: str) -> str:
    """Return the content of the LAST <answer>...</answer> block, else ''.

    Never falls back to the full response — callers decide their own fallback.
    """
    if not text:
        return ""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_boxed(text: str) -> str:
    """Return the LAST \\boxed{...} content, else ''."""
    if not text:
        return ""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_model_answer(text: str) -> str:
    """Best-effort final-answer view: <answer> tag -> \\boxed{} -> last non-empty line."""
    ans = extract_answer_tag(text)
    if ans:
        return ans
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(text: str) -> Optional[float]:
    """Extract a single number from text ('1,234.5' -> 1234.5). None if absent."""
    if text is None:
        return None
    m = _NUM_RE.search(str(text).strip())
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def numbers_equal(pred: str, gold: str, *, rel_tol: float = 0.0, abs_tol: float = 0.0) -> Optional[bool]:
    """Numeric equality; None when either side is not numeric.

    rel_tol is applied relative to the GOLD magnitude (ChartQAPro 5% rule).
    Zero gold requires exact zero (ChartQAPro zero-strict rule) unless abs_tol>0.
    """
    p, g = normalize_number(pred), normalize_number(gold)
    if p is None or g is None:
        return None
    if g == 0:
        return p == 0 if abs_tol == 0 else abs(p) <= abs_tol
    if rel_tol > 0:
        return abs(p - g) <= rel_tol * abs(g)
    return p == g


def normalize_text_basic(text: str) -> str:
    """Case-fold + collapse whitespace. No aggressive punctuation stripping."""
    return " ".join(str(text or "").strip().lower().split())


def exact_match_ci(pred: str, gold: str) -> bool:
    return normalize_text_basic(pred) == normalize_text_basic(gold)


# Thyme post_process sentinel: the model exhausted its agent-loop retries
# without emitting an <answer> tag. This is a MODEL format failure (score 0),
# NOT an infrastructure failure — it must never become UNSCORED.
MODEL_NO_ANSWER_SENTINEL = "No answer tag found in the final output."


def is_model_no_answer(prediction: str) -> bool:
    return MODEL_NO_ANSWER_SENTINEL in (prediction or "")


# ---------------------------------------------------------------------------
# MathRuler (vendored interface)
# ---------------------------------------------------------------------------
# The upstream ZoomBench blank-answer track uses MathRuler for math equivalence.
# The real vendored implementation lives in ``mathruler_vendored.py`` (added with
# the ZoomBench adapter, pinned to its upstream commit). ``math_equal`` lazily
# delegates to it so scorers can be written before the vendoring lands.

def math_equal(pred: str, gold: str) -> Optional[bool]:
    """MathRuler-style equivalence; None when not parseable as math.

    Resolution order: pip ``mathruler.grader`` -> local ``mathruler_vendored``
    -> plain numeric equality fallback.
    """
    try:
        from mathruler.grader import grade_answer  # type: ignore
        return bool(grade_answer(gold, pred))
    except ImportError:
        pass
    try:
        from . import mathruler_vendored  # noqa: WPS433 (lazy by design)
        return mathruler_vendored.math_equal(pred, gold)
    except ImportError:
        return numbers_equal(pred, gold)
