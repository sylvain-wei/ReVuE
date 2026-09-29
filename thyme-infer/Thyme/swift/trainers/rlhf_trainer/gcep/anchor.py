# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP anchor 选择。

Judge 不选 gold；group gold 与每条 rollout 的 anchor 都由程序按
确定性 cost tuple 取 lexicographic minimum。

本模块只依赖标准库，state 对象 duck-typed（dict 或带属性的对象均可）。
"""

from __future__ import annotations

from typing import Any, Optional

ORIGINAL_STATE_ID = "ORIGINAL_STATE"
ORIGINAL_IMAGE_ID = "ORIGINAL_IMAGE"


def _sget(state: Any, key: str, default: Any = None) -> Any:
    """dict / 普通对象双兼容取值。"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def state_cost(state: Any, support_count: int) -> tuple:
    """ cost tuple：(tool_rounds, cumulative_code_tokens, support 数, state_id)。

    解释：优先 NO_TOOL（0 轮）-> 更少 sandbox interaction -> 同轮更短代码
    -> 更少 supporting images -> state_id 固定打破平局。
    """
    return (
        int(_sget(state, "tool_rounds", 0) or 0),
        int(_sget(state, "cumulative_code_tokens", 0) or 0),
        int(support_count),
        str(_sget(state, "state_id", "")),
    )


def _entry_cost(entry: dict[str, Any]) -> tuple:
    return state_cost(entry["state"], len(entry["support_image_ids"]))


def build_sufficient_entries(
    judge_output: dict[str, Any],
    evidence_states_by_rollout: Any,
) -> list[dict[str, Any]]:
    """把（已通过 validator 的）judge sufficient_states 解析成 anchor entry 列表。

    每个 entry：{"state", "state_id", "rollout_id", "support_image_ids"}。
    ORIGINAL_STATE 的 rollout_id 为 None。
    """
    # 本地汇总 state，避免依赖 validator 模块，保证单文件可独立加载。
    states: dict[str, Any] = {}
    owner: dict[str, Optional[str]] = {}
    if evidence_states_by_rollout:
        items = (
            evidence_states_by_rollout.items()
            if isinstance(evidence_states_by_rollout, dict)
            else evidence_states_by_rollout
        )
        for rollout_key, state_list in items:
            for state in state_list or []:
                state_id = _sget(state, "state_id")
                if not state_id:
                    continue
                states[str(state_id)] = state
                if str(state_id) == ORIGINAL_STATE_ID:
                    owner[str(state_id)] = None
                else:
                    rollout_id = _sget(state, "rollout_id", None)
                    if rollout_id is None and rollout_key not in (None, ORIGINAL_STATE_ID):
                        rollout_id = rollout_key
                    owner[str(state_id)] = None if rollout_id is None else str(rollout_id)
    if ORIGINAL_STATE_ID not in states:
        states[ORIGINAL_STATE_ID] = {
            "state_id": ORIGINAL_STATE_ID,
            "rollout_id": None,
            "end_turn": -1,
            "tool_rounds": 0,
            "cumulative_code_tokens": 0,
            "new_image_ids": (),
            "available_image_ids": (ORIGINAL_IMAGE_ID,),
        }
        owner[ORIGINAL_STATE_ID] = None

    entries: list[dict[str, Any]] = []
    for item in (judge_output or {}).get("sufficient_states") or []:
        state_id = item.get("state_id")
        state = states.get(state_id)
        if state is None:
            continue
        entries.append(
            {
                "state": state,
                "state_id": state_id,
                "rollout_id": owner.get(state_id),
                "support_image_ids": list(item.get("support_image_ids") or []),
            }
        )
    return entries


def select_group_gold(
    sufficient_entries: list[dict[str, Any]],
    vqa_norms: dict[str, int],
) -> Optional[dict[str, Any]]:
    """：在 ORIGINAL_STATE（若 sufficient）+ 所有 vqa_norm=1 rollout 的
    sufficient states 上取 cost 的 lexicographic minimum。

    无候选（no group gold）返回 None，调用方按  fallback clean OPD。
    """
    candidates = [
        entry
        for entry in sufficient_entries
        if entry["rollout_id"] is None
        or vqa_norms.get(entry["rollout_id"], vqa_norms.get(str(entry["rollout_id"]))) == 1
    ]
    if not candidates:
        return None
    return min(candidates, key=_entry_cost)


def select_rollout_anchor(
    rollout_id: str,
    sufficient_entries: list[dict[str, Any]],
    group_gold: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """：rollout 自己有 sufficient state -> 自己的 cost 最小者；
    否则 -> group gold。
    """
    own = [entry for entry in sufficient_entries if entry["rollout_id"] == str(rollout_id)]
    if own:
        return min(own, key=_entry_cost)
    return group_gold


def acquisition_mode(anchor_entry: dict[str, Any]) -> str:
    """：anchor 是 ORIGINAL_STATE -> NO_TOOL；否则 TOOL_ASSISTED。"""
    if anchor_entry is None:
        return "NO_TOOL"
    if anchor_entry.get("state_id") == ORIGINAL_STATE_ID:
        return "NO_TOOL"
    if int(_sget(anchor_entry.get("state"), "tool_rounds", 0) or 0) == 0:
        return "NO_TOOL"
    return "TOOL_ASSISTED"
