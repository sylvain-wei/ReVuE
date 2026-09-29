# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP rollout-specific privilege certificate 构造。

硬约束：
- ：certificate 绝不包含任何 rollout 的 verbatim Python code，
  也不包含精确 crop coordinates；Acquire privilege 只含
  NO_TOOL/TOOL_ASSISTED、query_slot、anchor 真实 support image(s)、state ID。
- ：多图必须作为独立 multimodal image item 插入，禁止 contact sheet。
- ：self-correction 场景只附 anchor 的 support images（好图），不附坏图。
- ：anchor == ORIGINAL_STATE -> NO_TOOL，且不重复插入原图。

本模块只依赖标准库。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

CERTIFICATE_BEGIN = "<PRIVILEGED_EVIDENCE_CERTIFICATE>"
CERTIFICATE_END = "</PRIVILEGED_EVIDENCE_CERTIFICATE>"

# 过渡句：在 certificate 标签前用一句自然语言向
# teacher 说明这段是什么、该怎么用。要点：①来自同题正/负 rollout 的已验证视觉证据
# ②给出最小充分视觉目标 + 正确读数 ③用它来指导分布 ④但要"当成自己获取的"——
# 防止 teacher 直接抄答案、保持 on-policy 语义（呼应  不含 verbatim code/坐标）。
CERTIFICATE_PREAMBLE = (
    "The following is privileged visual evidence, verified from correct and "
    "incorrect rollouts of this same question. It tells you the minimum sufficient "
    "visual target and the correct reading — use it to inform your answer "
    "distribution, but still solve the task as if you had acquired this evidence "
    "yourself."
)

# support image 在文本中的占位标记：integrator 负责把每个标记替换为
# 一个独立 multimodal image item（顺序与返回的 support_images 列表一致）。
SUPPORT_IMAGE_MARKER_TEMPLATE = '<support_image index="{index}" ref="{ref}"/>'

_NO_TOOL_SUPPORT_LINE = (
    "None (the original task image already contains the sufficient evidence)."
)


def build_privilege_certificate(
    query_slot: str,
    anchor_state_id: str,
    acq_mode: str,
    observed_value: str,
    visual_fact: str,
    grounding_rule: str,
    diagnosis_stage: str,
    discrepancy: str,
    support_images: Optional[Sequence[Any]] = None,
) -> tuple[str, list[Any]]:
    """构造  的 <PRIVILEGED_EVIDENCE_CERTIFICATE> 文本 + 有序 support images。

    参数：
        acq_mode: "NO_TOOL" 或 "TOOL_ASSISTED"（见 anchor.acquisition_mode）。
        support_images: anchor entry 的 support images（有序，元素为调用方约定
            的 image ref / path / id）。self-correction 场景只传好图。

    返回：
        (certificate_text, support_images)。
        certificate_text 中每张 support image 对应一个
        ``<support_image index="k" ref="..."/>`` 占位符，供 multimodal 插入；
        NO_TOOL 时按  强制不附任何额外图片。
    """
    images = list(support_images or [])
    if acq_mode == "NO_TOOL":
        # 原图已在 teacher 原始 task context 中，不重复插入。
        images = []

    if images:
        support_block = "\n".join(
            SUPPORT_IMAGE_MARKER_TEMPLATE.format(index=k, ref=ref)
            for k, ref in enumerate(images)
        )
    else:
        support_block = _NO_TOOL_SUPPORT_LINE

    text = (
        # 过渡句在 certificate 标签之前，帮 teacher 理解这段的用途。
        f"{CERTIFICATE_PREAMBLE}\n\n"
        f"{CERTIFICATE_BEGIN}\n"
        "\n"
        "[ACQUIRE]\n"
        "\n"
        "Target visual information:\n"
        f"{query_slot}\n"
        "\n"
        "Minimum sufficient evidence state:\n"
        f"{anchor_state_id}\n"
        "\n"
        "Acquisition mode:\n"
        f"{acq_mode}\n"
        "\n"
        "Supporting visual evidence:\n"
        f"{support_block}\n"
        "\n"
        "\n"
        "[READ]\n"
        "\n"
        "Queried visual slot:\n"
        f"{query_slot}\n"
        "\n"
        "Observed value:\n"
        f"{observed_value}\n"
        "\n"
        "Atomic visual fact:\n"
        f"{visual_fact}\n"
        "\n"
        "\n"
        "[GROUND]\n"
        "\n"
        "Evidence-to-answer rule:\n"
        f"{grounding_rule}\n"
        "\n"
        "\n"
        "[CURRENT TRAJECTORY]\n"
        "\n"
        "First broken link:\n"
        f"{diagnosis_stage}\n"
        "\n"
        "Discrepancy:\n"
        f"{discrepancy}\n"
        "\n"
        f"{CERTIFICATE_END}"
    )
    return text, images


# ---------------------------------------------------------------------------
# Acquire-only 防泄漏 certificate
# ---------------------------------------------------------------------------
# Acquire-only 只提供目标信息、证据定位和图像，不提供读数及答案映射。
# Full certificate 还提供 Read/Ground 内容，两种条件的区别需在分析中保留。
ACQUIRE_ONLY_PREAMBLE = (
    "The following is privileged guidance about WHERE to look, verified from "
    "correct and incorrect rollouts of this same question. It points you to the "
    "minimum sufficient visual evidence — but does NOT tell you the reading or "
    "the answer. You must still look at the image yourself, use tools if needed, "
    "and reason to your own answer."
)


def build_acquire_only_certificate(
    query_slot: str,
    anchor_state_id: str,
    acq_mode: str,
    support_images: Optional[Sequence[Any]] = None,
) -> tuple[str, list[Any]]:
    """Acquire-only certificate：只给目标视觉信息 + 证据定位 + support 图。

    与 build_privilege_certificate 的差别（防泄漏）：
    - 删掉 [READ]（observed_value / visual_fact）——不给正确读数；
    - 删掉 [GROUND]（grounding_rule）——不给证据→答案映射；
    - 删掉 [CURRENT TRAJECTORY]（diagnosis/discrepancy）——不给错误提示。

    返回 (certificate_text, support_images)，与 full 版同样的 support_image 占位符
    约定；NO_TOOL 时同样不附图。
    """
    images = list(support_images or [])
    if acq_mode == "NO_TOOL":
        images = []

    if images:
        support_block = "\n".join(
            SUPPORT_IMAGE_MARKER_TEMPLATE.format(index=k, ref=ref)
            for k, ref in enumerate(images)
        )
    else:
        support_block = _NO_TOOL_SUPPORT_LINE

    text = (
        f"{ACQUIRE_ONLY_PREAMBLE}\n\n"
        f"{CERTIFICATE_BEGIN}\n"
        "\n"
        "[ACQUIRE]\n"
        "\n"
        "Target visual information:\n"
        f"{query_slot}\n"
        "\n"
        "Minimum sufficient evidence state:\n"
        f"{anchor_state_id}\n"
        "\n"
        "Acquisition mode:\n"
        f"{acq_mode}\n"
        "\n"
        "Supporting visual evidence:\n"
        f"{support_block}\n"
        "\n"
        "Note: the reading and the answer are intentionally withheld — "
        "inspect the evidence and derive them yourself.\n"
        "\n"
        f"{CERTIFICATE_END}"
    )
    return text, images


# ---------------------------------------------------------------------------
# Diagnosis-only privilege（GCEP v2 ALL_WRONG）
# ---------------------------------------------------------------------------
# 与 full / acquire-only certificate 的区别：ALL_WRONG 组没有"正确 trajectory"
# 可锚定，teacher privilege 只携带该 rollout 的 Evidence-State-aware 诊断
# （First broken interface + discrepancy），不含 observed_value / visual_fact /
# grounding.rule / reference answer / support image / gold crop / code。
DIAGNOSIS_ONLY_BEGIN = "<PRIVILEGED_DIAGNOSIS>"
DIAGNOSIS_ONLY_END = "</PRIVILEGED_DIAGNOSIS>"


def build_diagnosis_only_privilege(stage: str, discrepancy: str) -> str:
    """ALL_WRONG 的 diagnosis-only privilege 文本（纯文本、无图）。

    复用 certificate 的插入通道（build_privileged_rollout 把文本追加到第一条
    user 消息尾），support_image_paths 恒为空。
    """
    return (
        f"{DIAGNOSIS_ONLY_BEGIN}\n"
        "\n"
        "This is training-time privileged feedback about the already-sampled\n"
        "student trajectory.\n"
        "\n"
        "First broken evidence-chain interface:\n"
        f"{stage}\n"
        "\n"
        "Diagnosis:\n"
        f"{discrepancy}\n"
        "\n"
        f"{DIAGNOSIS_ONLY_END}"
    )


# ---------------------------------------------------------------------------
# GT-only privilege（vanilla OPD 的 privileged teacher）
# ---------------------------------------------------------------------------
# 与上面 certificate / diagnosis 通道的区别：不含 anchor / 证据状态 / 诊断 /
# support image，只把**已验证的 GT 答案**作为 privileged target 交给 teacher。
# teacher 侧因此在"知道答案"的条件下给出更可学的分布；student 侧 prompt 完全
# 不变（on-policy 语义不变，privilege 只影响 teacher 前向输入）。
# 固定模板；更改措辞会影响实验可比性。
GT_ONLY_TEMPLATE = (
    "\n\nThe verified final answer for this question is provided below.\n"
    "\n"
    "<reference_answer>\n"
    "{gt_answer}\n"
    "</reference_answer>\n"
    "\n"
    "Use this answer as a target when solving the question from the image. "
    "Develop the reasoning yourself and use the available tools when useful. "
    "Show how the visual evidence supports the answer, following the reasoning, "
    "tool-use, and final-answer format specified in the original instructions.\n"
)


def build_gt_only_privilege(gt_answer: str) -> str:
    """GT-only privilege 文本（纯文本、无 support image）。

    与 certificate 相同的插入通道：build_privileged_rollout 把文本追加到第一条
    user 消息尾部（user task 之后、assistant 轨迹之前），support_image_paths 恒为空。
    """
    return GT_ONLY_TEMPLATE.format(gt_answer=str(gt_answer or ""))
