"""Math correctness reward function for verl.

Provides a reward_function compatible with verl's reward scoring
interface.  Uses OPD-Lab's answer verifier for boxed-answer extraction
and checking.
"""

from __future__ import annotations

import re
from typing import Any


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from text.

    Handles nested braces correctly.
    """
    # Find all \boxed{ occurrences, take the last one
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()

    # Track brace depth to find matching close
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None

    return text[start : pos - 1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    if answer is None:
        return ""
    s = answer.strip()
    # Remove dollar signs, spaces
    s = s.replace("$", "").replace(" ", "").replace(",", "")
    # Normalize fractions: \frac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # Remove \text{}, \mathrm{}, etc.
    s = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", s)
    # Remove remaining backslash commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Remove braces
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def check_math_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.

    Tries exact string match after normalization, then numeric
    comparison.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        return abs(pred_val - gt_val) < 1e-6
    except (ValueError, OverflowError):
        pass

    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Compute reward score for a single solution.

    This follows verl's reward function signature. Called by verl's
    reward scoring pipeline.

    Parameters
    ----------
    data_source : str
        Name of the dataset (ignored, we always do math grading).
    solution_str : str
        The model-generated response text.
    ground_truth : dict
        Must contain ``ground_truth`` key with the expected answer string.
    extra_info : dict, optional
        Additional info (unused).

    Returns
    -------
    score : float
        1.0 if correct, 0.0 if incorrect.
    """
    # ground_truth is typically {"style": "rule", "ground_truth": "42"}
    expected = ground_truth.get("ground_truth", "")
    if not expected:
        return 0.0

    predicted = extract_boxed_answer(solution_str)
    if predicted is None:
        return 0.0

    return 1.0 if check_math_answer(predicted, expected) else 0.0
