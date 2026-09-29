# Modified for anonymous review: release paths, configuration, and documentation.
"""Minimal ANLS implementation (threshold + max over golds).

Matches the reference `anls` package semantics:
    similarity = 1 - levenshtein(pred, gold) / max(len(pred), len(gold))
    score = similarity if similarity >= threshold else 0
Lowercasing is the CALLER's choice (ChartQAPro lowercases; InfographicVQA
official ANLS also lowercases — no extra normalization either way).
"""

from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def anls_score(*, prediction: str, gold_labels: list[str], threshold: float = 0.5) -> float:
    best = 0.0
    for gold in gold_labels:
        denom = max(len(prediction), len(gold))
        if denom == 0:
            sim = 1.0
        else:
            sim = 1.0 - _levenshtein(prediction, gold) / denom
        if sim >= threshold and sim > best:
            best = sim
    return best
