# Modified for this anonymized research release: portable paths and release cleanup.
"""轨迹有效性过滤（GCEP_TRAJ_FILTER）。

invalid 表示以下任意情况：
- 未正常完成（没有 </answer>）；
- sandbox 回显超过预算而产生 observation-overflow 标记；
- 存在执行器 RuntimeError 标记；
- 含已有生成侧终止标记，或 metadata 已给出生成侧终止原因。

含 invalid 行的 group：整组 reward 清零、整组不调用 judge；plans=None
使有效行回退 clean OPD。默认情况下 invalid 行不参与 OPD 聚合，独立的
GCEP_OPD_INCLUDE_INVALID 开关可改变聚合行为。

默认关闭，GCEP_TRAJ_FILTER=1 才启用。文本标记也可能出现在模型输出中；
metadata 原因优先，但当前文本判断并不是不可伪造的来源验证。"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

# pipeline 注入标记（grpo_trainer.py O3 注入段 / sandbox 执行器）
_O3_TRUNC_MARK = "...[truncated]</sandbox_output>"
_RTE_MARK = "This instance triggers RuntimeError"

# 生成侧拦截标记（gcep/gen_intercept.py，GEN_INTERCEPT=1 时才可能出现）：
# 复读熔断 / 单回合超长熔断。命中行 = 被主动终止的退化轨迹 → 判 invalid（丢弃不进训练）。
# 注意：这两个标记是"人为追加的终止标记"，与模型自己写的文本无关（模型自己写出的
# 同名文本理论上可能，但概率可忽略；v2 可改为带 nonce 的标记）。
_MARKER_REASONS = (
    ("[rep-terminated]", "gen_repetition"),
    ("[len-terminated]", "gen_too_long"),
    ("[loop-terminated]", "gen_loop"),
)


def traj_filter_enabled() -> bool:
    return os.getenv("GCEP_TRAJ_FILTER", "0") == "1"


def invalid_reason(completion_text: Optional[str]) -> Optional[str]:
    """返回无效原因，或返回 None 表示有效。

    优先检查显式终止和 overflow，再判断 incomplete。因此即使文本包含
    </answer>，只要带有执行器或 observation-overflow 标记仍被判为 invalid。"""
    if not completion_text:
        return "incomplete"
    for marker, reason in _MARKER_REASONS:
        if marker in completion_text:
            return reason
    if _O3_TRUNC_MARK in completion_text or _RTE_MARK in completion_text:
        return "observation_overflow"
    if "</answer>" not in completion_text:
        return "incomplete"
    return None


def detect_invalid(completions: List[str], reasons: Optional[List[Optional[str]]] = None
                   ) -> Tuple[List[bool], dict]:
    """对本地 completions 列表逐条判定。

    Args:
        completions: 逐行 completion 文本。
        reasons: 可选的逐行 metadata 原因（gen_intercept 零暴露路径：生成侧把
            reason 挂在样本上，不再写 in-band 标记）。非空时**优先于文本判定**；
            为 None / 该行为 None 时回落到 invalid_reason 的文本判定。
    Returns:
        (flags, reason_counts)：flags[i]=True 表示该行为 invalid；
        reason_counts 形如 {"incomplete": n1, "observation_overflow": n2}。
    """
    flags: List[bool] = []
    counts = {"incomplete": 0, "observation_overflow": 0,
              "gen_repetition": 0, "gen_too_long": 0, "gen_loop": 0}
    for i, c in enumerate(completions):
        r = None
        if reasons is not None and i < len(reasons) and reasons[i]:
            r = str(reasons[i])
        else:
            r = invalid_reason(c)
        flags.append(r is not None)
        if r is not None:
            counts[r] = counts.get(r, 0) + 1
    return flags, counts
