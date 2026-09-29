# Modified for anonymous review: release paths, configuration, and documentation.
"""Message serializers for multi-image and text-only benchmark inputs.

build_messages(benchmark, sample) -> list[{"type": "image"|"text", "value": str}]

Routing rules:
- PerceptionBench: interleave images at <|image_N|> placeholder positions (1-based,
  official eval.py build_messages semantics); unreferenced images appended in list
  order (upstream asserts none exist, we keep the fallback). All N images enter the
  conversation; the Thyme tool sandbox binds base image 0 (single-image binding,
  explicitly recorded as deviation; tool_traces carry base_image_id).
- HallusionBench: visual_input=0 rows are legal text-only samples — no image item,
  never a blank/fake image.
- MathVerseVO / VisuLogic_GCEP: single image + verbatim question text.

The serializer must never leak GT/hint/metadata into the message: only the
`question` column text and `image_path` column paths are used.
"""

from __future__ import annotations

import re
from pathlib import Path

GCEP4_BENCHMARKS = ("PerceptionBench", "HallusionBench_GCEP", "MathVerseVO", "VisuLogic_GCEP")

_PH_RE = re.compile(r"<\|image_(\d+)\|>")


def _get(sample, key, default=""):
    try:
        v = sample[key]
    except Exception:
        return default
    if v is None:
        return default
    return v


def _image_paths(sample) -> list[str]:
    raw = str(_get(sample, "image_path", "")).strip()
    if not raw or raw.lower() in ("none", "nan", "[]"):
        return []
    return [p for p in raw.split(";") if p and p != "[]"]


def build_messages(benchmark: str, sample) -> list[dict]:
    question = str(_get(sample, "question", ""))
    paths = _image_paths(sample)
    for p in paths:
        assert Path(p).exists(), f"{benchmark}: missing image file {p}"

    if benchmark == "PerceptionBench":
        assert paths, "PerceptionBench row with no images"
        msgs: list[dict] = []
        last = 0
        used: set[int] = set()
        for m in _PH_RE.finditer(question):
            n = int(m.group(1))
            assert 1 <= n <= len(paths), f"placeholder <|image_{n}|> out of range ({len(paths)} images)"
            used.add(n - 1)
            seg = question[last:m.start()]
            if seg:
                msgs.append({"type": "text", "value": seg})
            msgs.append({"type": "image", "value": paths[n - 1]})
            last = m.end()
        if question[last:]:
            msgs.append({"type": "text", "value": question[last:]})
        for k, p in enumerate(paths):  # unreferenced images appended in order
            if k not in used:
                msgs.append({"type": "image", "value": p})
        return msgs

    if benchmark == "HallusionBench_GCEP":
        vi = str(_get(sample, "visual_input", "1"))
        if vi == "0":
            assert not paths, f"text-only HallusionBench_GCEP row carries image_path: {paths}"
            return [{"type": "text", "value": question}]
        assert len(paths) == 1, f"HallusionBench visual row with {len(paths)} images"
        return [{"type": "image", "value": paths[0]}, {"type": "text", "value": question}]

    if benchmark in ("MathVerseVO", "VisuLogic_GCEP"):
        assert len(paths) == 1, f"{benchmark} row with {len(paths)} images"
        return [{"type": "image", "value": paths[0]}, {"type": "text", "value": question}]

    raise KeyError(f"no gcep4 serializer for benchmark {benchmark!r}")
