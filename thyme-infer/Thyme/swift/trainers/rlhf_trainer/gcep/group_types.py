# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP v2 group type 分类（IDEA-1）。

group type 只由 vqa_norm 二值决定（group size 固定 = num_generations）：
    num_correct == ng        -> ALL_CORRECT
    0 < num_correct < ng     -> MIXED
    num_correct == 0         -> ALL_WRONG

绝不使用 total reward / GRPO reward / advantage 决定 group type。

本模块只依赖标准库。
"""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence


class GCEPGroupType(str, Enum):
    ALL_CORRECT = "all_correct"
    MIXED = "mixed"
    ALL_WRONG = "all_wrong"


def classify_group_types(vqa_bin: Sequence[int], num_generations: int) -> List[GCEPGroupType]:
    """逐 group 分类。与 mixed_group_flags 同输入约定（vqa_bin 按 group 连续排列）。"""
    ng = int(num_generations)
    types: List[GCEPGroupType] = []
    for g in range(len(vqa_bin) // ng):
        s = sum(int(v) for v in vqa_bin[g * ng:(g + 1) * ng])
        if s == ng:
            types.append(GCEPGroupType.ALL_CORRECT)
        elif s > 0:
            types.append(GCEPGroupType.MIXED)
        else:
            types.append(GCEPGroupType.ALL_WRONG)
    return types
