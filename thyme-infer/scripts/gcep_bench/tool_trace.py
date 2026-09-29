# Modified for anonymous review: release paths, configuration, and documentation.
"""Per-sample tool trace extraction.

Builds a structured per-sample tool trace from ``model.last_conversation_history``
(the same source ``_extract_tool_use_features`` uses), but keeps PER-CALL detail
instead of aggregate counts:

- assistant_messages: every assistant text segment, in order;
- calls: one entry per <code> block with tool type, base image binding and the
  paired <sandbox_output> status / image count;
- used_tool := any successful visual tool call (defined as:
  ``successful visual tool calls > 0``).

Base-image binding: the Thyme sandbox binds exactly one input image per episode
(``model.py:_extract_image_path`` picks the FIRST image of the message and passes
it to ``execute_code_in_sandbox``). For multi-image samples (PerceptionBench) crop
code therefore binds image 0 — this is recorded explicitly as
``base_image_id`` + ``binding: "sandbox_first_input_image"`` rather than parsed
from the code, because the binding is structural, not textual.
"""

from __future__ import annotations

import re
from pathlib import Path

_CODE_BLOCK_RE = re.compile(r"<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>", re.IGNORECASE)
_SANDBOX_OUTPUT_RE = re.compile(r"<sandbox_output>([\s\S]*?)</sandbox_output>", re.IGNORECASE)

_VISUAL_TYPES = {"crop/zoom", "image-enhance"}
_ERROR_MARKERS = ("Traceback", "Error:", "Exception:")


def _classify_tool(code: str) -> str:
    code_lower = code.lower()
    if any(kw in code_lower for kw in ['crop', 'resize', 'zoom', 'paste', 'slice', 'region', '.crop(']):
        return "crop/zoom"
    if any(kw in code_lower for kw in ['enhance', 'filter', 'blur', 'sharpen', 'contrast', 'brightness']):
        return "image-enhance"
    if any(kw in code_lower for kw in ['count', 'sum', 'len(', 'max(', 'min(', 'np.', 'calculate', 'math.']):
        return "compute"
    return "other"


def build_tool_trace(model, *, image_paths: list[str] | None = None) -> dict:
    """Extract the per-call trace for the model's most recent episode."""
    conv = getattr(model, "last_conversation_history", None)
    empty = {"assistant_messages": [], "calls": [], "used_tool": False,
             "n_tool_calls": 0, "n_successful_visual_calls": 0,
             "n_sandbox_outputs": 0, "full_response_chars": 0}
    if conv is None:
        return empty

    assistant_messages: list[str] = []
    for msg in conv:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            assistant_messages.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    if t:
                        assistant_messages.append(t)

    full_response = "".join(assistant_messages)
    if not full_response:
        return empty

    code_blocks = _CODE_BLOCK_RE.findall(full_response)
    sandbox_outputs = _SANDBOX_OUTPUT_RE.findall(full_response)
    base_image_id = ""
    if image_paths:
        base_image_id = Path(image_paths[0]).name

    calls = []
    n_success_visual = 0
    for i, code in enumerate(code_blocks):
        output = sandbox_outputs[i] if i < len(sandbox_outputs) else None
        ok = output is not None and not any(m in output[:300] for m in _ERROR_MARKERS)
        tool_type = _classify_tool(code)
        successful_visual = bool(ok and tool_type in _VISUAL_TYPES)
        n_success_visual += int(successful_visual)
        calls.append({
            "call_idx": i,
            "tool_type": tool_type,
            "code_text": code,
            "base_image_id": base_image_id or None,
            "binding": "sandbox_first_input_image" if base_image_id else "no_input_image",
            "sandbox_status": ("ok" if ok else ("error" if output is not None else "no_output")),
            "output_excerpt": (output or "")[:300],
            # Processed result images are written to a per-episode temp dir by the
            # sandbox and are not assigned stable IDs; we record their count only.
            "n_result_images": None,
        })

    return {
        "assistant_messages": assistant_messages,
        "calls": calls,
        "used_tool": n_success_visual > 0,
        "n_tool_calls": len(code_blocks),
        "n_successful_visual_calls": n_success_visual,
        "n_sandbox_outputs": len(sandbox_outputs),
        "full_response_chars": len(full_response),
    }
