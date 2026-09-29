# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP Judge client。

一个 mixed group = 一次多模态 Judge 请求；temperature=0；json_schema 约束输出；
超时 / 非法 JSON 直接返回 None，由调用方走 clean-teacher OPD fallback。
禁止 per-rollout / per-image 调用，禁止二次 synthesis / repair 调用。

本模块只依赖标准库；m_rlsd 的 RemoteVLMClient 在函数内惰性加载。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

# 兄弟模块 group_types（stdlib-only）：包内相对导入优先；单文件加载时按路径兜底。
try:
    from .group_types import GCEPGroupType
except ImportError:  # pragma: no cover - 仅离线单文件加载路径
    import sys as _sys

    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
    from group_types import GCEPGroupType

# ---------------------------------------------------------------------------
# Judge 服务端点常量（部署约定）
# ---------------------------------------------------------------------------

# Configure the judge endpoint via GCEP_JUDGE_BASE_URL / GCEP_JUDGE_API_KEY.
# Endpoint, authentication and model alias are configured through environment variables.
JUDGE_BASE_URL = os.getenv("GCEP_JUDGE_BASE_URL", "http://localhost:8000/v1")
JUDGE_API_KEY = os.getenv("GCEP_JUDGE_API_KEY", "EMPTY")
JUDGE_MODEL = os.getenv("GCEP_JUDGE_MODEL", "qwen3.5-397b-judge")
JUDGE_TIMEOUT_SEC = 600.0
JUDGE_MAX_TOKENS = 2048

# Limit images per judge request to the configured server budget.
# Correct rollouts receive priority; excess sandbox images are omitted while
# their text labels are retained. The default budget is eight images.
JUDGE_MAX_IMAGES = int(os.getenv("GCEP_JUDGE_MAX_IMAGES", "8"))

# Bound image size to reduce multimodal request payloads. GCEP_JUDGE_MAX_IMAGE_DIM
# sets the longest edge (default 1024); zero preserves the original resolution.
# Resizing may remove fine visual detail, so this setting affects judge inputs.
JUDGE_MAX_IMAGE_DIM = int(os.getenv("GCEP_JUDGE_MAX_IMAGE_DIM", "1024"))


def _downscale_data_url(url: str, max_dim: int) -> str:
    """把 data:image/...;base64 图片降采样到最长边 max_dim，再编码回 data URL。

    仅处理 data: 前缀；非 data URL（http/路径）原样返回。任何失败都返回原 url
    （降采样是优化，绝不允许因此让 judge 失败）。
    """
    if max_dim <= 0 or not isinstance(url, str) or not url.startswith("data:"):
        return url
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image  # 惰性重依赖

        header, _, payload = url.partition(";base64,")
        if not payload:
            return url
        mime = header[len("data:"):] or "image/png"
        raw = _b64.b64decode(payload)
        img = Image.open(_io.BytesIO(raw))
        w, h = img.size
        if max(w, h) <= max_dim:
            return url  # 已经够小，不动
        scale = max_dim / float(max(w, h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return url

# Bypass proxy settings for the configured judge host and loopback endpoints.
_JUDGE_HOST_ENTRY = urlparse(JUDGE_BASE_URL).hostname or ""
_NO_PROXY_ENTRIES = tuple(dict.fromkeys(
    e for e in (_JUDGE_HOST_ENTRY, "127.0.0.1", "localhost") if e
))

# ---------------------------------------------------------------------------
#  Judge system prompts
# ---------------------------------------------------------------------------
# GCEP v2：三种 group type 各自专用 system prompt，共用同一份
# JUDGE_JSON_SCHEMA 与 build_judge_messages。MIXED prompt 与 v1 逐字节一致
# （只改常量名，不改任何文字）；group_type 只是调用侧 metadata，不进 Judge JSON。

MIXED_JUDGE_SYSTEM_PROMPT = """You are a training-time multimodal evidence-chain analyst for a
think-with-image VQA policy.

You receive one task and a group of verifier-labeled CORRECT and
INCORRECT student rollouts.

A rollout contains chronological policy reasoning, free-form Python
code, sandbox text/numeric outputs, zero/one/multiple images returned
by each sandbox interaction, and a final answer.

One Python code block may return MULTIPLE images. They belong to the
SAME sandbox interaction and may jointly form multi-region evidence.

A rollout may have multiple sandbox interactions. Later interactions
may correct earlier visual observations.

The CORRECT/INCORRECT label is defined only by the supplied VQA
correctness label. Trust CORRECT rollouts as correct final solutions.

Your task is to identify the visual evidence chain:

ACQUIRE:
Which actually observed Evidence States already contain sufficient
task-relevant visual evidence?

READ:
What is the minimum atomic visual fact needed by the question?

GROUND:
How does that visual fact map to the answer semantics?

EVIDENCE STATE:
ORIGINAL_STATE contains ORIGINAL_IMAGE.
Every other state is the state immediately after one real sandbox
interaction and lists all images currently available.

A state is SUFFICIENT iff some subset of its available images already
contains enough visual evidence to determine the task-relevant visual
fact without needing a later observation.

For each sufficient state, return the SMALLEST supporting image subset.

CLOSED-SET RULE:
Never invent an image, state, crop, coordinate, code, or synthetic view.
Only use supplied state IDs and image IDs.

READ:
Do not use a predefined taxonomy.
Return:
- query_slot
- observed_value
- one atomic visual_fact

GROUND:
Return one concise evidence-to-answer semantic rule.
Do not state only an answer/option letter.

ROLLOUT DIAGNOSIS:
CORRECT rollout -> stage=CORRECT.

For INCORRECT rollouts return the FIRST broken interface:

ACQUIRE:
No state belonging to this rollout is sufficient.

READ:
A sufficient state exists, but the relevant visual fact is misread.

GROUND:
A sufficient state exists and the fact is read correctly, but final
answer semantics contradict/mis-map that fact.

OTHER:
Failure is outside this visual evidence chain.

Return JSON only according to schema."""

# v1 兼容别名：旧引用（RQ1 harness / 分析脚本）继续用 JUDGE_SYSTEM_PROMPT，
# 与 MIXED_JUDGE_SYSTEM_PROMPT 是同一对象（preflight 用 is 断言等价性）。
JUDGE_SYSTEM_PROMPT = MIXED_JUDGE_SYSTEM_PROMPT

# GCEP v2 ALL_CORRECT prompt（固定模板）。
# 任务语义：successful evidence consolidation；所有 rollout stage=CORRECT。
ALL_CORRECT_JUDGE_SYSTEM_PROMPT = """You are a training-time multimodal evidence-chain analyst for a
think-with-image VQA policy.

You receive one task and a group of verifier-labeled CORRECT student
rollouts.

A rollout contains chronological policy reasoning, free-form Python
code, sandbox text/numeric outputs, zero/one/multiple images returned
by each sandbox interaction, and a final answer.

One Python code block may return MULTIPLE images. They belong to the
SAME sandbox interaction and may jointly form multi-region evidence.

A rollout may have multiple sandbox interactions. Later interactions
may correct or refine earlier visual observations.

The CORRECT label is defined only by the supplied VQA correctness
label. Trust all supplied rollouts as correct final solutions.

Your task is to identify the visual evidence chain shared by these
successful solutions:

ACQUIRE:
Which actually observed Evidence States already contain sufficient
task-relevant visual evidence?

READ:
What is the minimum atomic visual fact needed by the question?

GROUND:
How does that visual fact map to the answer semantics?

EVIDENCE STATE:
ORIGINAL_STATE contains ORIGINAL_IMAGE.
Every other state is the state immediately after one real sandbox
interaction and lists all images currently available.

A state is SUFFICIENT iff some subset of its available images already
contains enough visual evidence to determine the task-relevant visual
fact without needing a later observation.

For each sufficient state, return the SMALLEST supporting image subset.

A rollout may contain redundant later tool calls even after sufficient
evidence has already been obtained. Mark the EARLIEST sufficient state
when possible; do not require later redundant observations.

Multiple images returned by the SAME sandbox interaction may be jointly
necessary and may therefore appear together in support_image_ids.

CLOSED-SET RULE:
Never invent an image, state, crop, coordinate, code, or synthetic view.
Only use supplied state IDs and image IDs.

READ:
Do not use a predefined taxonomy.
Return:
- query_slot
- observed_value
- one atomic visual_fact

GROUND:
Return one concise evidence-to-answer semantic rule.
Do not state only an answer/option letter.

ROLLOUT DIAGNOSIS:
All supplied rollouts are CORRECT.

For every rollout return:
stage=CORRECT
discrepancy=""

Do not search for ACQUIRE, READ, GROUND, or OTHER failures in this
group.

Return JSON only according to schema."""

# GCEP v2 ALL_WRONG prompt（固定模板）。
# 任务语义：对每条错误 rollout 做 Evidence-State-aware 的
# ACQUIRE/READ/GROUND/OTHER 结构化诊断；stage=CORRECT 非法（validator 拒绝）。
ALL_WRONG_JUDGE_SYSTEM_PROMPT = """You are a training-time multimodal evidence-chain analyst for a
think-with-image VQA policy.

You receive one task and a group of verifier-labeled INCORRECT student
rollouts.

A rollout contains chronological policy reasoning, free-form Python
code, sandbox text/numeric outputs, zero/one/multiple images returned
by each sandbox interaction, and a final answer.

One Python code block may return MULTIPLE images. They belong to the
SAME sandbox interaction and may jointly form multi-region evidence.

A rollout may have multiple sandbox interactions. Later interactions
may correct or refine earlier visual observations.

The INCORRECT label is defined only by the supplied VQA correctness
label. All supplied rollouts have incorrect final answers.

Use the supplied Reference Answer together with the task, original
image, sandbox observations, and student trajectory to determine where
each incorrect rollout first breaks the visual evidence chain.

The visual evidence chain is:

ACQUIRE:
Did the rollout obtain sufficient task-relevant visual evidence?

READ:
If sufficient evidence was available, did the rollout correctly read
the minimum atomic visual fact required by the question?

GROUND:
If the visual fact was read correctly, did the rollout map that fact
correctly to the final answer semantics?

EVIDENCE STATE:
ORIGINAL_STATE contains ORIGINAL_IMAGE.
Every other state is the state immediately after one real sandbox
interaction and lists all images currently available.

A state is SUFFICIENT iff some subset of its available images already
contains enough visual evidence to determine the task-relevant visual
fact without needing a later observation.

IMPORTANT:
An INCORRECT rollout may still contain a SUFFICIENT Evidence State.
For example, it may acquire the correct crop but fail at READ or
GROUND.

Therefore inspect every rollout's Evidence States before assigning a
diagnosis.

For each sufficient state that actually exists, return the SMALLEST
supporting image subset.

If no supplied rollout contains any sufficient Evidence State,
sufficient_states may be an empty array.

CLOSED-SET RULE:
Never invent an image, state, crop, coordinate, code, or synthetic view.
Only use supplied state IDs and image IDs.

READ:
Do not use a predefined taxonomy.

Using the supplied task, Reference Answer, original image, and actual
rollout observations, return:
- query_slot
- observed_value
- one atomic visual_fact

These fields are used to support diagnosis of the incorrect rollouts.

GROUND:
Return one concise evidence-to-answer semantic rule.
Do not state only an answer/option letter.

ROLLOUT DIAGNOSIS:

All supplied rollouts are INCORRECT.
Never return stage=CORRECT.

For every rollout return the FIRST broken interface:

ACQUIRE:
No Evidence State belonging to this rollout is sufficient for the
task-relevant visual fact.

READ:
A sufficient Evidence State exists in this rollout, but the rollout
misreads the task-relevant visual fact.

GROUND:
A sufficient Evidence State exists, and the rollout correctly reads
the task-relevant visual fact, but the final answer semantics
contradict or mis-map that fact.

OTHER:
The failure is outside this visual evidence chain or cannot be
reliably attributed to ACQUIRE, READ, or GROUND.

The discrepancy must be a concise diagnosis of THIS rollout's actual
failure.

Do not produce a generic sentence such as:
"The answer is wrong."

Do not invent a corrected Python program or new visual observation.

BINDING RULES:

1. If you assign READ or GROUND to a rollout, you MUST include at least
one sufficient Evidence State for that rollout in sufficient_states.
If the sufficient evidence is the ORIGINAL_IMAGE itself, include
ORIGINAL_STATE — it counts as available evidence for every rollout.

2. The verifier label is authoritative even when a rollout's final
answer appears to match the Reference Answer (normalization/verifier
differences). In that case keep applicable=true and diagnose that
rollout as OTHER with the label conflict as the discrepancy. Reserve
applicable=false for tasks that are not visually grounded at all.

3. You MUST return exactly one diagnosis for every supplied rollout
(all of them). Keep each discrepancy to ONE concise sentence specific
to that rollout.

Return JSON only according to schema."""


def get_judge_system_prompt(group_type: Any = GCEPGroupType.MIXED) -> str:
    """按 group type 取专用 system prompt（三分支，拒绝 if-in-one-giant-prompt）。

    接受 GCEPGroupType 或其 value 字符串（"all_correct"/"mixed"/"all_wrong"）。
    """
    if not isinstance(group_type, GCEPGroupType):
        group_type = GCEPGroupType(str(group_type))
    if group_type == GCEPGroupType.ALL_CORRECT:
        return ALL_CORRECT_JUDGE_SYSTEM_PROMPT
    if group_type == GCEPGroupType.MIXED:
        return MIXED_JUDGE_SYSTEM_PROMPT
    if group_type == GCEPGroupType.ALL_WRONG:
        return ALL_WRONG_JUDGE_SYSTEM_PROMPT
    raise ValueError(f"unknown GCEP group_type: {group_type!r}")

# ---------------------------------------------------------------------------
#  Judge JSON Schema（response_format=json_schema 用）
# ---------------------------------------------------------------------------

JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "name": "gcep_judge_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "applicable": {"type": "boolean"},
            "sufficient_states": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string"},
                        "support_image_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["state_id", "support_image_ids"],
                    "additionalProperties": False,
                },
            },
            "reading": {
                "type": "object",
                "properties": {
                    "query_slot": {"type": "string"},
                    "observed_value": {"type": "string"},
                    "visual_fact": {"type": "string"},
                },
                "required": ["query_slot", "observed_value", "visual_fact"],
                "additionalProperties": False,
            },
            "grounding": {
                "type": "object",
                "properties": {
                    "rule": {"type": "string"},
                },
                "required": ["rule"],
                "additionalProperties": False,
            },
            "rollout_diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rollout_id": {"type": "string"},
                        "stage": {
                            "type": "string",
                            "enum": ["CORRECT", "ACQUIRE", "READ", "GROUND", "OTHER"],
                        },
                        "discrepancy": {"type": "string"},
                    },
                    "required": ["rollout_id", "stage", "discrepancy"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "applicable",
            "sufficient_states",
            "reading",
            "grounding",
            "rollout_diagnoses",
        ],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def ensure_no_proxy(environ: Optional[dict] = None) -> None:
    """把 Judge 内网地址追加进 no_proxy / NO_PROXY（幂等）。

    必须在任何 Judge 请求之前调用，否则配置了 http_proxy 的机器会把
    内网请求错误地发给代理。
    """
    env = os.environ if environ is None else environ
    for var in ("no_proxy", "NO_PROXY"):
        existing = [p.strip() for p in env.get(var, "").split(",") if p.strip()]
        for entry in _NO_PROXY_ENTRIES:
            if entry not in existing:
                existing.append(entry)
        env[var] = ",".join(existing)


def _load_remote_client():
    """惰性加载 m_rlsd.inference.remote_vlm_client（纯标准库模块）。

    优先走已安装的包路径；失败时按 repo 相对路径直接按文件加载，
    保证本模块在不安装 src/ 的环境里也能独立使用。
    """
    try:
        from m_rlsd.inference.remote_vlm_client import (  # type: ignore
            RemoteVLMClient,
            RemoteVLMConfig,
        )

        return RemoteVLMClient, RemoteVLMConfig
    except ImportError:
        pass

    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]
    client_path = repo_root / "src" / "m_rlsd" / "inference" / "remote_vlm_client.py"
    spec = importlib.util.spec_from_file_location("gcep_remote_vlm_client", client_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load remote_vlm_client from {client_path}")
    module = importlib.util.module_from_spec(spec)
    # 先注册进 sys.modules 再 exec：@dataclass 处理 string annotation 时需要
    # 通过 sys.modules[cls.__module__] 反查模块命名空间。
    sys.modules.setdefault("gcep_remote_vlm_client", module)
    spec.loader.exec_module(module)
    return module.RemoteVLMClient, module.RemoteVLMConfig


def get_judge_client(
    base_url: str = JUDGE_BASE_URL,
    api_key: str = JUDGE_API_KEY,
    model: str = JUDGE_MODEL,
    timeout_sec: float = JUDGE_TIMEOUT_SEC,
):
    """构造默认 Judge client。max_retries=0：任何失败都走 fallback，不重试。"""
    ensure_no_proxy()
    RemoteVLMClient, RemoteVLMConfig = _load_remote_client()
    config = RemoteVLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_sec=timeout_sec,
        max_retries=0,
    )
    return RemoteVLMClient(config)


def _rget(obj: Any, key: str, default: Any = None) -> Any:
    """dict / 普通对象双兼容取值。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _rollout_sort_key(rollout_id: Any):
    """R2 < R10 的自然排序。"""
    text = str(rollout_id)
    match = re.match(r"^([A-Za-z_]*)(\d+)$", text)
    if match:
        return (match.group(1), int(match.group(2)))
    return (text, -1)


def _format_options(options: Any) -> str:
    if options is None:
        return "(none)"
    if isinstance(options, str):
        return options
    return "\n".join(str(opt) for opt in options)


def _format_evidence_state(state_meta: Any) -> str:
    """Evidence State metadata 文本；接受预格式化 str 或 dict/对象。"""
    if state_meta is None:
        return "(no evidence state)"
    if isinstance(state_meta, str):
        return state_meta
    fields = (
        "state_id",
        "rollout_id",
        "end_turn",
        "tool_rounds",
        "cumulative_code_tokens",
        "new_image_ids",
        "available_image_ids",
    )
    parts = [f"{k}={_rget(state_meta, k)}" for k in fields]
    return "; ".join(parts)


def _resolve_image_item(image: Any, resolve_image_url) -> Optional[dict[str, Any]]:
    """把 sandbox 图片描述转换成独立 image_url content item。

    image 可以是：
    - str：直接当作 url 或本地路径；
    - dict：{image_id, path|url|image_path|image_url}。
    """
    if isinstance(image, str):
        ref = image
        image_id = ref
    else:
        image_id = _rget(image, "image_id") or _rget(image, "id") or ""
        ref = (
            _rget(image, "url")
            or _rget(image, "image_url")
            or _rget(image, "path")
            or _rget(image, "image_path")
        )
    if not ref:
        return None
    if ref.startswith(("http://", "https://", "data:")):
        url = ref
    else:
        url = resolve_image_url(image_path=ref, image_url=None)
    if not url:
        return None
    # judge 请求里的图片统一降采样（大图 base64 会拖慢 397B prefill）。
    url = _downscale_data_url(url, JUDGE_MAX_IMAGE_DIM)
    return {
        "type": "image_url",
        "image_url": {"url": url},
        "_image_id": image_id,  # 仅本地调试用，发送前剥除
    }


def _strip_internal_keys(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
#  Judge 输入构造
# ---------------------------------------------------------------------------


def build_judge_messages(
    question: str,
    options: Any,
    reference_answer: str,
    original_image_path_or_url: Optional[str],
    rollouts: list[Any],
    group_type: Any = GCEPGroupType.MIXED,
) -> list[dict[str, Any]]:
    """按  顺序构造一次 group-level Judge 请求的消息。

    顺序约定：
    1. Task 只出现一次：Question / Options / Reference Answer / Original Image；
    2. 全部 vqa_norm=1 rollout（按 rollout_id 升序）；
    3. 全部 vqa_norm=0 rollout（按 rollout_id 升序）；
    4. 每条 rollout 内严格 chronological：reasoning -> code -> sandbox
       stdout/stderr -> sandbox images（每张独立 image_url item，禁止
       contact sheet）-> Evidence State metadata -> next turn -> final answer。

    rollout 结构（dict 或对象均可）：
        rollout_id, vqa_norm, final_answer,
        turns: [{turn_id, reasoning, code, sandbox_stdout, sandbox_stderr,
                 sandbox_status, sandbox_images: [...], evidence_state: ...}]

    group_type（GCEP v2）：只决定 system prompt（dispatcher），不改变本函数的
    消息序列化逻辑；默认 MIXED，旧调用方零变化。
    """
    RemoteVLMClient, _ = _load_remote_client()
    resolve_image_url = RemoteVLMClient._resolve_image_url

    content: list[dict[str, Any]] = []

    # 图片预算：judge 服务端单 prompt 最多 JUDGE_MAX_IMAGES 张图。
    # correct rollout 排在最前（_sort_key），其证据图最 informative，优先占预算。
    image_budget = {"left": max(0, int(JUDGE_MAX_IMAGES))}

    # --- Task（仅一次） ---
    task_text = (
        "[Task]\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{_format_options(options)}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        "Original Image (image_id: ORIGINAL_IMAGE):"
    )
    content.append({"type": "text", "text": task_text})
    if original_image_path_or_url:
        item = _resolve_image_item(
            {"image_id": "ORIGINAL_IMAGE", "url": original_image_path_or_url}
            if str(original_image_path_or_url).startswith(("http://", "https://", "data:"))
            else {"image_id": "ORIGINAL_IMAGE", "path": original_image_path_or_url},
            resolve_image_url,
        )
        if item is not None and image_budget["left"] > 0:
            content.append(item)
            image_budget["left"] -= 1

    # --- Rollouts：先 vqa_norm=1 后 vqa_norm=0，各自按 rollout_id 排序 ---
    def _sort_key(rollout: Any):
        vqa = _rget(rollout, "vqa_norm", 0)
        return (0 if vqa == 1 else 1, _rollout_sort_key(_rget(rollout, "rollout_id", "")))

    for rollout in sorted(rollouts, key=_sort_key):
        rollout_id = str(_rget(rollout, "rollout_id", ""))
        vqa_norm = _rget(rollout, "vqa_norm", 0)
        label = "CORRECT" if vqa_norm == 1 else "INCORRECT"
        content.append(
            {
                "type": "text",
                "text": f"[Rollout {rollout_id}] Verifier label: {label}",
            }
        )

        turns = _rget(rollout, "turns", []) or []
        for turn in sorted(turns, key=lambda t: _rget(t, "turn_id", 0)):
            turn_id = _rget(turn, "turn_id", 0)
            reasoning = _rget(turn, "reasoning", "") or ""
            code = _rget(turn, "code", "") or ""
            stdout = _rget(turn, "sandbox_stdout", "") or ""
            stderr = _rget(turn, "sandbox_stderr", "") or ""

            turn_text = f"--- Turn {turn_id} ---\nReasoning:\n{reasoning}"
            if code:
                turn_text += f"\n\nCode:\n{code}"
            if stdout:
                turn_text += f"\n\nSandbox stdout:\n{stdout}"
            if stderr:
                turn_text += f"\n\nSandbox stderr:\n{stderr}"
            content.append({"type": "text", "text": turn_text})

            # 每个 sandbox image 独立成 item，绝不拼 contact sheet。
            images = _rget(turn, "sandbox_images", []) or []
            if not images:
                # 兼容 archive 字段：只有路径列表时自动编号。
                archive_paths = _rget(turn, "sandbox_image_archive_paths", []) or []
                images = [
                    {"image_id": f"{rollout_id}_T{turn_id}_IMG{k}", "path": p}
                    for k, p in enumerate(archive_paths)
                ]
            for k, image in enumerate(images):
                item = _resolve_image_item(image, resolve_image_url)
                if item is None:
                    continue
                image_id = item.pop("_image_id", "") or f"{rollout_id}_T{turn_id}_IMG{k}"
                if image_budget["left"] <= 0:
                    # 预算用尽：跳过图片但保留文本标签，judge 仍知道该图存在过
                    content.append({"type": "text", "text": f"Sandbox image {image_id}: [omitted - image budget exhausted]"})
                    continue
                content.append({"type": "text", "text": f"Sandbox image {image_id}:"})
                content.append(item)
                image_budget["left"] -= 1

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Evidence State after turn {turn_id}:\n"
                        f"{_format_evidence_state(_rget(turn, 'evidence_state'))}"
                    ),
                }
            )

        final_answer = _rget(rollout, "final_answer", "") or ""
        content.append({"type": "text", "text": f"Final answer:\n{final_answer}"})

    user_message = {
        "role": "user",
        "content": [_strip_internal_keys(item) for item in content],
    }
    system_message = {"role": "system", "content": get_judge_system_prompt(group_type)}
    return [system_message, user_message]


# ---------------------------------------------------------------------------
# Judge 调用
# ---------------------------------------------------------------------------


def _judge_group_once(
    question: str,
    options: Any,
    reference_answer: str,
    original_image_path_or_url: Optional[str],
    rollouts: list[Any],
    client: Any,
    max_tokens: int,
    group_type: Any = GCEPGroupType.MIXED,
) -> Optional[dict[str, Any]]:
    """单次 judge 请求；超时 / 传输错误 / 非法 JSON 一律返回 None。"""
    try:
        # 消息构造失败（如 sandbox 图片路径缺失）同样走 fallback，绝不 crash 训练。
        messages = build_judge_messages(
            question, options, reference_answer, original_image_path_or_url, rollouts,
            group_type=group_type,
        )
        text, _raw = client.generate_text(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": JUDGE_JSON_SCHEMA,
                }
            },
        )
    except Exception:
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def judge_group(
    question: str,
    options: Any,
    reference_answer: str,
    original_image_path_or_url: Optional[str],
    rollouts: list[Any],
    *,
    client: Any = None,
    max_tokens: int = JUDGE_MAX_TOKENS,
    max_attempts: int = 2,
    group_type: Any = GCEPGroupType.MIXED,
) -> Optional[dict[str, Any]]:
    """对一个 group 发起 Judge 请求（带有限 retry）。

    返回解析后的 JSON dict；超时 / 传输错误 / 非法 JSON 一律返回 None，
    由调用方按  fallback 到 clean-teacher OPD。绝不发起 repair 调用。

    默认 max_attempts=2——judge 调用失败多为随机网络/解析错误，
    一次同参数 retry 即可捞回（不改变方法语义，只恢复随机失败）。确定性错误
    （如消息构造失败）retry 也会失败，无副作用。

    GCEP v2：group_type 选择专用 system prompt（默认 MIXED，行为与 v1 一致）。
    """
    ensure_no_proxy()
    if client is None:
        client = get_judge_client()
    for _ in range(max(1, int(max_attempts))):
        obj = _judge_group_once(
            question, options, reference_answer,
            original_image_path_or_url, rollouts, client, max_tokens,
            group_type=group_type,
        )
        if obj is not None:
            return obj
    return None


def judge_groups(
    batch: list[dict[str, Any]],
    concurrency: int = 4,
    *,
    client: Any = None,
    max_tokens: int = JUDGE_MAX_TOKENS,
) -> list[Optional[dict[str, Any]]]:
    """并发处理同一 training batch 内的多个 mixed groups（默认并发 4）。

    batch 每项为 judge_group 的位置参数 dict：
        {"question", "options", "reference_answer",
         "original_image_path_or_url"(或 "original_image"), "rollouts",
         "group_type"(可选，GCEP v2；缺省 = MIXED)}
    返回与输入顺序对齐的结果列表；单项失败为 None。
    """
    ensure_no_proxy()
    if client is None:
        client = get_judge_client()

    from concurrent.futures import ThreadPoolExecutor

    def _one(item: dict[str, Any]):
        return judge_group(
            item["question"],
            item.get("options"),
            item.get("reference_answer"),
            item.get("original_image_path_or_url", item.get("original_image")),
            item["rollouts"],
            client=client,
            max_tokens=max_tokens,
            group_type=item.get("group_type", GCEPGroupType.MIXED),
        )

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        return list(executor.map(_one, batch))
