# Modified for this anonymized research release: portable paths and release cleanup.
"""RLSD privileged-teacher 上下文构造模块。

本模块为 GRPO 在线训练（`grpo_trainer.py`）提供 privilege 构造纯函数。

    给定一条 on-policy rollout 的 messages（system / user / 单条 assistant 轨迹）
    以及该样本的 GT 答案（solution）和环境奖励 reward，
    产出一份「加了 privilege 信息」的 messages 副本，供 teacher 前向使用。

关键约束（务必保持）：
    1. privilege 只允许加在 **prompt 段**（system / user 消息），
       绝不能改动 assistant 段（completion）——否则 teacher 与 student 的
       completion token 序列会错位，RLSD 的逐 token log-prob 比值就失去意义。
    2. teacher 与 student 是同一份 policy 权重，唯一差别就是 teacher 的 prompt
       多了 privilege 文本。

支持的 privilege 模式（可切换开关，预留扩展）：
    - "gt_only":                    只把 GT 答案作为提示拼到 user 消息尾部。
    - "rich_final_obs_correct_only": reward>=1 时拼 GT + 最后一个 sandbox
                                     observation；reward<1 时只拼 GT。

新增模式只需写一个 `_build_xxx` 函数并注册进 PRIVILEGE_MODE_REGISTRY 即可。

注意：on-policy Thyme rollout 里，整条 think→code→<sandbox_output>→answer 轨迹
都在 **单条 assistant 消息的 content 字符串**里（由 grpo_trainer 在 assistant content 上追加）。因此「最后一个 observation」是从 assistant
content 里正则解析出来的，而非独立的 tool/user 轮次。
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# privilege 文本模板
# ---------------------------------------------------------------------------
# GT privilege 使用固定文本模板。
_GT_TEMPLATE = "\n\n[Privileged context — the correct answer is: {gt_answer}]\n"

# rich 模式：GT + 最后一个 sandbox 观测证据。
_RICH_TEMPLATE_HEAD = (
    "\n\nThe following privileged information is available only for teacher scoring.\n"
    "It contains the correct final answer and tool evidence from sandbox execution.\n"
    "\n=== Ground-Truth Answer Begin ===\n{gt_answer}\n=== Ground-Truth Answer End ===\n"
)
_RICH_TEMPLATE_OBS = (
    "\n=== Selected Tool Observation Begin ===\n{observation}\n"
    "=== Selected Tool Observation End ===\n"
)

# 解析 assistant content 里的 sandbox 观测块。
_SANDBOX_RE = re.compile(r"<sandbox_output>([\s\S]*?)</sandbox_output>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------
def _append_text_to_first_user(messages: List[dict], text: str) -> None:
    """把 privilege 文本追加到第一条 user 消息尾部（原地修改 messages 副本）。

    content 可能是纯字符串，也可能是 [{"type":"text"|"image", ...}] 列表；
    两种结构都要兼容（多模态样本用列表结构）。
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # 多模态：追加一个 text 段（放在最后，不动 image 段）
            content.append({"type": "text", "text": text})
        else:
            # 纯文本：直接拼字符串
            msg["content"] = str(content) + text
        return  # 只改第一条 user 消息


def _extract_assistant_content(messages: List[dict]) -> str:
    """取出（最后一条）assistant 消息的纯文本 content，用于解析 sandbox 观测。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            c = msg.get("content")
            if isinstance(c, list):
                return "".join(
                    seg.get("text", "") for seg in c if isinstance(seg, dict)
                )
            return str(c)
    return ""


def _last_sandbox_observation(assistant_content: str) -> Optional[str]:
    """从 assistant content 里解析出 **最后一个** <sandbox_output> 观测文本。

    找不到则返回 None（该样本退化为 gt_only 行为）。
    """
    matches = _SANDBOX_RE.findall(assistant_content or "")
    if not matches:
        return None
    return matches[-1].strip()


# ---------------------------------------------------------------------------
# 各 privilege 模式的构造函数
# ---------------------------------------------------------------------------
def _build_gt_only(
    messages: List[dict], solution: str, reward: Optional[float]
) -> List[dict]:
    """gt_only：只把 GT 答案作为提示拼到 user 消息尾部。"""
    conv = copy.deepcopy(messages)
    _append_text_to_first_user(conv, _GT_TEMPLATE.format(gt_answer=solution))
    return conv


def _build_rich_final_obs_correct_only(
    messages: List[dict], solution: str, reward: Optional[float]
) -> List[dict]:
    """rich_final_obs_correct_only：

    - reward >= 1（学生答对）：privilege = GT + 最后一个 sandbox observation；
    - reward <  1（学生答错）：privilege = 仅 GT（不给观测证据）。

    """
    conv = copy.deepcopy(messages)

    # 判定学生是否答对。reward 缺失时按答错处理（更保守，不泄露观测）。
    try:
        student_correct = reward is not None and float(reward) >= 1.0
    except (TypeError, ValueError):
        student_correct = False

    text = _RICH_TEMPLATE_HEAD.format(gt_answer=solution)
    if student_correct:
        obs = _last_sandbox_observation(_extract_assistant_content(messages))
        if obs:
            # 观测文本可能很长，做一个安全上限，避免 prompt 段爆炸。
            obs = obs[:4000]
            text += _RICH_TEMPLATE_OBS.format(observation=obs)

    _append_text_to_first_user(conv, text)
    return conv


# ---------------------------------------------------------------------------
# 模式注册表（扩展点：新增 privilege 只需在此登记一个函数）
# ---------------------------------------------------------------------------
PrivilegeBuilder = Callable[[List[dict], str, Optional[float]], List[dict]]

PRIVILEGE_MODE_REGISTRY: Dict[str, PrivilegeBuilder] = {
    "gt_only": _build_gt_only,
    "rich_final_obs_correct_only": _build_rich_final_obs_correct_only,
}


def build_privilege_messages(
    messages: List[dict],
    solution: str,
    mode: str,
    reward: Optional[float] = None,
) -> List[dict]:
    """构造 privilege-augmented 的 messages 副本（teacher 前向输入）。

    Args:
        messages: 单条 rollout 的完整对话（system/user/assistant）。
        solution: 该样本的 GT 答案（来自 reward_kwargs["solution"]）。
        mode:     privilege 模式，必须在 PRIVILEGE_MODE_REGISTRY 里。
        reward:   该 rollout 的环境奖励，rich_*_correct_only 模式需要它做过滤。

    Returns:
        深拷贝并加了 privilege 文本的 messages（只改 prompt 段）。

    Raises:
        ValueError: mode 未注册。
    """
    if mode not in PRIVILEGE_MODE_REGISTRY:
        raise ValueError(
            f"unknown RLSD privilege mode: {mode!r}; "
            f"available = {sorted(PRIVILEGE_MODE_REGISTRY)}"
        )
    return PRIVILEGE_MODE_REGISTRY[mode](messages, solution, reward)
