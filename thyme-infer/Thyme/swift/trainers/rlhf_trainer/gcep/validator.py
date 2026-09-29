# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP Judge 输出的 deterministic validation。

Judge 只做 semantic judgment；所有结构合法性由本模块检查。
任一检查失败 -> (False, reason)，调用方按  fallback clean-teacher OPD，
训练绝不能因为 Judge 输出非法而 crash。

本模块只依赖标准库，state 对象 duck-typed（dict 或带属性的对象均可）。
"""

from __future__ import annotations

import os
from typing import Any, Optional

# 兄弟模块 group_types（stdlib-only）：包内相对导入优先；单文件加载时按路径兜底。
try:
    from .group_types import GCEPGroupType
except ImportError:  # pragma: no cover - 仅离线单文件加载路径
    import sys as _sys

    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
    from group_types import GCEPGroupType

VALID_STAGES = ("CORRECT", "ACQUIRE", "READ", "GROUND", "OTHER")
ORIGINAL_STATE_ID = "ORIGINAL_STATE"
ORIGINAL_IMAGE_ID = "ORIGINAL_IMAGE"


def _sget(state: Any, key: str, default: Any = None) -> Any:
    """dict / 普通对象双兼容取值。"""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _default_original_state() -> dict[str, Any]:
    """：ORIGINAL_STATE 永远存在，即使调用方未显式传入。"""
    return {
        "state_id": ORIGINAL_STATE_ID,
        "rollout_id": None,
        "end_turn": -1,
        "tool_rounds": 0,
        "cumulative_code_tokens": 0,
        "new_image_ids": (),
        "available_image_ids": (ORIGINAL_IMAGE_ID,),
    }


def collect_states(evidence_states_by_rollout: Any):
    """汇总 group 内全部 Evidence State。

    参数：dict rollout_id -> [state, ...]；ORIGINAL_STATE 可放在 key
    "ORIGINAL_STATE" 或 None 下，也可以完全不传（自动补默认 ORIGINAL_STATE）。

    返回 (states_by_id, owner_by_id)；owner 为 None 表示 ORIGINAL_STATE。
    """
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
        states[ORIGINAL_STATE_ID] = _default_original_state()
        owner[ORIGINAL_STATE_ID] = None
    return states, owner


def _known_image_ids(states: dict[str, Any]) -> set:
    known = {ORIGINAL_IMAGE_ID}
    for state in states.values():
        for field in ("available_image_ids", "new_image_ids"):
            for image_id in _sget(state, field, ()) or ():
                known.add(str(image_id))
    return known


def reconcile_diagnoses_with_vqa(
    obj: Any,
    evidence_states_by_rollout: Any,
    vqa_norms: dict[str, int],
) -> int:
    """用权威 vqa_norm + EvidenceState 结构约束校正 judge 的 rollout_diagnoses.stage。

    vqa_norm 是规则或判分流程给出的正确性信号，优先于 judge 的 stage
    归因。校正在 validate 之前对齐两者；下游结构校验和 privilege 语义
    保持不变。该校正依赖 vqa_norm 本身准确。

    校正规则（确定性，幂等）——对每条 rollout：
      1. vqa_norm=1 且 stage!=CORRECT        -> CORRECT      （修 check8）
      2. vqa_norm=0 且 stage==CORRECT         -> READ/ACQUIRE （修 check9，
         依据该 rollout 是否 own sufficient state：有证据读错=READ，无证据=ACQUIRE）
      3. vqa_norm=0 且 stage==ACQUIRE 但 own sufficient  -> READ（修 check10）
      4. vqa_norm=0 且 stage==READ/GROUND 但无 own sufficient -> ACQUIRE（修 check11）

    只改 stage 字段；obj 其余部分不动。返回被修正的 rollout 条数。
    对结构不完整的 obj（缺 sufficient_states / diagnoses 非 list）直接返回 0，
    交给后续 validate_gcep_output 正常报错。
    """
    if not isinstance(obj, dict):
        return 0
    diagnoses = obj.get("rollout_diagnoses")
    sufficient_states = obj.get("sufficient_states")
    if not isinstance(diagnoses, list) or not isinstance(sufficient_states, list):
        return 0

    states, owner = collect_states(evidence_states_by_rollout)
    # 每个 rollout 是否 own 至少一个 sufficient state（供 check10/11 一致化）
    sufficient_by_rollout: dict[str, bool] = {}
    for entry in sufficient_states:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("state_id")
        if not isinstance(sid, str) or sid not in states:
            continue
        rid = owner.get(sid)
        if rid is not None:
            sufficient_by_rollout[str(rid)] = True

    n_fixed = 0
    for diag in diagnoses:
        if not isinstance(diag, dict):
            continue
        rid = diag.get("rollout_id")
        stage = diag.get("stage")
        vqa = vqa_norms.get(rid, vqa_norms.get(str(rid)))
        if vqa is None or not isinstance(stage, str):
            continue
        own_suff = bool(sufficient_by_rollout.get(str(rid)))
        new_stage = stage
        if vqa == 1 and stage != "CORRECT":
            new_stage = "CORRECT"
        elif vqa == 0 and stage == "CORRECT":
            new_stage = "READ" if own_suff else "ACQUIRE"
        elif vqa == 0 and stage == "ACQUIRE" and own_suff:
            new_stage = "READ"
        elif vqa == 0 and stage in ("READ", "GROUND") and not own_suff:
            new_stage = "ACQUIRE"
        if new_stage != stage:
            diag["stage"] = new_stage
            n_fixed += 1
    return n_fixed


def _validate_structural(
    obj: Any,
    states: dict[str, Any],
    known_images: set,
    vqa_norms: dict[str, int],
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """group-type 共用的 structural checks（ check 1-7）。

    返回 (ok, reason, ctx)。ok=False 时 ctx=None；applicable=false 返回
    (False, "GCEP_NOT_APPLICABLE", None)（合法 judge 输出，走 fallback）。
    ctx 包含解析后字段 + sufficient_by_rollout 映射，供 per-type semantic 检查复用。
    """
    def fail(check_no: int, detail: str) -> tuple[bool, str, None]:
        return False, f"GCEP_INVALID: check{check_no}: {detail}", None

    # check 1：schema 形状
    if not isinstance(obj, dict):
        return fail(1, "judge output is not a JSON object")
    applicable = obj.get("applicable")
    if not isinstance(applicable, bool):
        return fail(1, "field 'applicable' missing or not boolean")
    if not applicable:
        # applicable=false 是合法 judge 输出，但该 group 不使用 privilege。
        return False, "GCEP_NOT_APPLICABLE", None
    sufficient_states = obj.get("sufficient_states")
    reading = obj.get("reading")
    grounding = obj.get("grounding")
    diagnoses = obj.get("rollout_diagnoses")
    if not isinstance(sufficient_states, list):
        return fail(1, "field 'sufficient_states' missing or not a list")
    if not isinstance(reading, dict):
        return fail(1, "field 'reading' missing or not an object")
    if not isinstance(grounding, dict):
        return fail(1, "field 'grounding' missing or not an object")
    if not isinstance(diagnoses, list):
        return fail(1, "field 'rollout_diagnoses' missing or not a list")
    for entry in sufficient_states:
        if not isinstance(entry, dict) or not isinstance(entry.get("state_id"), str):
            return fail(1, "sufficient_states entry missing string 'state_id'")
        if not isinstance(entry.get("support_image_ids"), list) or not all(
            isinstance(x, str) for x in entry["support_image_ids"]
        ):
            return fail(1, "sufficient_states entry 'support_image_ids' must be list[str]")
    for diag in diagnoses:
        if not isinstance(diag, dict) or not isinstance(diag.get("rollout_id"), str):
            return fail(1, "rollout_diagnoses entry missing string 'rollout_id'")
        if diag.get("stage") not in VALID_STAGES:
            return fail(1, f"invalid diagnosis stage: {diag.get('stage')!r}")
        if not isinstance(diag.get("discrepancy", ""), str):
            return fail(1, "diagnosis 'discrepancy' must be a string")

    # check 2：state ID 必须存在
    for entry in sufficient_states:
        if entry["state_id"] not in states:
            return fail(2, f"unknown state_id: {entry['state_id']}")

    # check 3：state 不重复
    state_ids = [entry["state_id"] for entry in sufficient_states]
    if len(set(state_ids)) != len(state_ids):
        return fail(3, "duplicate state_id in sufficient_states")

    for entry in sufficient_states:
        support = entry["support_image_ids"]
        # check 4：support list 非空
        if not support:
            return fail(4, f"empty support_image_ids for state {entry['state_id']}")
        # check 5：image ID 必须存在
        for image_id in support:
            if image_id not in known_images:
                return fail(5, f"unknown image_id: {image_id}")
        # check 6：support ⊆ state.available_image_ids
        available = {str(x) for x in (_sget(states[entry["state_id"]], "available_image_ids", ()) or ())}
        for image_id in support:
            if image_id not in available:
                return fail(
                    6,
                    f"support image {image_id} not in available_image_ids of "
                    f"state {entry['state_id']}",
                )

    # check 7：diagnoses 恰好覆盖 group 全部 rollout（不多不少不重复）
    expected_rollouts = {str(r) for r in vqa_norms.keys()}
    diag_rollouts = [diag["rollout_id"] for diag in diagnoses]
    if len(set(diag_rollouts)) != len(diag_rollouts):
        return fail(7, "duplicate rollout_id in rollout_diagnoses")
    if set(diag_rollouts) != expected_rollouts:
        return fail(
            7,
            f"diagnoses cover {sorted(set(diag_rollouts))}, "
            f"expected exactly {sorted(expected_rollouts)}",
        )

    ctx = {
        "sufficient_states": sufficient_states,
        "reading": reading,
        "grounding": grounding,
        "diagnoses": diagnoses,
    }
    return True, "OK", ctx


def _sufficient_by_rollout(ctx: dict[str, Any], owner: dict[str, Optional[str]]) -> dict[str, list[str]]:
    """每条 rollout 拥有的 sufficient state 列表（ORIGINAL_STATE 不属于任何 rollout）。"""
    out: dict[str, list[str]] = {}
    for entry in ctx["sufficient_states"]:
        rid = owner.get(entry["state_id"])
        if rid is not None:
            out.setdefault(rid, []).append(entry["state_id"])
    return out


def _check_reading_grounding(ctx: dict[str, Any]) -> tuple[bool, str]:
    """ check 13/14（三种 group type 共用；schema 要求 non-empty）。"""
    for field in ("query_slot", "observed_value", "visual_fact"):
        value = ctx["reading"].get(field)
        if not isinstance(value, str) or not value.strip():
            return False, f"GCEP_INVALID: check13: reading.{field} empty"
    rule = ctx["grounding"].get("rule")
    if not isinstance(rule, str) or not rule.strip():
        return False, "GCEP_INVALID: check14: grounding.rule empty"
    return True, "OK"


def _validate_semantic_mixed(
    ctx: dict[str, Any],
    owner: dict[str, Optional[str]],
    vqa_norms: dict[str, int],
) -> tuple[bool, str]:
    """MIXED group 的 semantic checks =  check 8-14（v1 行为，逐字节保留）。"""
    def fail(check_no: int, detail: str) -> tuple[bool, str]:
        return False, f"GCEP_INVALID: check{check_no}: {detail}"

    sufficient_by_rollout = _sufficient_by_rollout(ctx, owner)
    any_positive_sufficient = False
    for diag in ctx["diagnoses"]:
        rid = diag["rollout_id"]
        stage = diag["stage"]
        vqa = vqa_norms.get(rid, vqa_norms.get(str(rid)))
        own_sufficient = bool(sufficient_by_rollout.get(rid))
        # check 8：vqa_norm=1 -> CORRECT
        if vqa == 1 and stage != "CORRECT":
            return fail(8, f"rollout {rid} has vqa_norm=1 but stage={stage}")
        # check 9：vqa_norm=0 -> != CORRECT
        if vqa == 0 and stage == "CORRECT":
            return fail(9, f"rollout {rid} has vqa_norm=0 but stage=CORRECT")
        # check 10：ACQUIRE -> 无 own sufficient state
        if stage == "ACQUIRE" and own_sufficient:
            return fail(
                10,
                f"rollout {rid} diagnosed ACQUIRE but owns sufficient state(s) "
                f"{sufficient_by_rollout[rid]}",
            )
        # check 11：READ/GROUND -> 必须有 own sufficient state
        if stage in ("READ", "GROUND") and not own_sufficient:
            return fail(
                11,
                f"rollout {rid} diagnosed {stage} but owns no sufficient state",
            )

    # check 12：至少 ORIGINAL_STATE 或某个 positive-rollout state sufficient
    for entry in ctx["sufficient_states"]:
        rid = owner.get(entry["state_id"])
        if rid is None:
            any_positive_sufficient = True
            break
        if vqa_norms.get(rid, vqa_norms.get(str(rid))) == 1:
            any_positive_sufficient = True
            break
    if not any_positive_sufficient:
        return fail(
            12,
            "no sufficient state from ORIGINAL_STATE or any vqa_norm=1 rollout",
        )

    return _check_reading_grounding(ctx)


def _validate_semantic_all_correct(
    ctx: dict[str, Any],
    owner: dict[str, Optional[str]],
    vqa_norms: dict[str, int],
) -> tuple[bool, str]:
    """ALL_CORRECT（GCEP v2）：全部 diagnosis stage=CORRECT 且 discrepancy=""
    （严格，不做 reconcile 自动修）；≥1 个 sufficient state（任意 correct rollout
    或 ORIGINAL_STATE）；reading/grounding 非空。"""
    def fail(check_no: int, detail: str) -> tuple[bool, str]:
        return False, f"GCEP_INVALID: check{check_no}: {detail}"

    for diag in ctx["diagnoses"]:
        rid = diag["rollout_id"]
        stage = diag["stage"]
        vqa = vqa_norms.get(rid, vqa_norms.get(str(rid)))
        # check 8：vqa_norm=1 -> CORRECT（ALL_CORRECT 全部 rollout 适用）
        if vqa == 1 and stage != "CORRECT":
            return fail(8, f"rollout {rid} has vqa_norm=1 but stage={stage}")
        # check 15（v2 新增，ALL_CORRECT 专用）：CORRECT diagnosis 的 discrepancy 必须为空
        if stage == "CORRECT" and str(diag.get("discrepancy", "")).strip():
            return fail(
                15,
                f"rollout {rid} stage=CORRECT but discrepancy non-empty: "
                f"{diag.get('discrepancy')!r}",
            )

    # check 12 变体：至少一个 sufficient state（owner 为 None 或 vqa=1；ALL_CORRECT
    # 中所有 rollout vqa=1，等价于"任意 sufficient state 存在"）。
    any_sufficient = False
    for entry in ctx["sufficient_states"]:
        rid = owner.get(entry["state_id"])
        if rid is None or vqa_norms.get(rid, vqa_norms.get(str(rid))) == 1:
            any_sufficient = True
            break
    if not any_sufficient:
        return fail(12, "no sufficient state from ORIGINAL_STATE or any correct rollout")

    return _check_reading_grounding(ctx)


def _validate_semantic_all_wrong(
    ctx: dict[str, Any],
    owner: dict[str, Optional[str]],
    vqa_norms: dict[str, int],
) -> tuple[bool, str]:
    """ALL_WRONG（GCEP v2）：stage==CORRECT 一律 invalid；ACQUIRE→无 own sufficient；
    READ/GROUND→必须 own sufficient（违反即 invalid，不自动修成 ACQUIRE）；OTHER 不
    强制；sufficient_states 允许整体为空（无 check12）；reading/grounding 非空
    （schema 要求，downstream 不用）。

    结构规则：answer-only rollout 唯一可能的
    sufficient state 是 ORIGINAL_STATE（owner=None），严格的"own sufficient"会让
    诚实的 GROUND 诊断在结构上不可能（judge 不能发明 rollout-owned state，
    closed-set 禁止）。因此 ALL_WRONG 中 **ORIGINAL_STATE ∈ sufficient_states
    视为对所有 rollout 可用的充分证据**（原图本就提供给每条 rollout）：
    READ/GROUND 放行；相应地 ORIGINAL_STATE sufficient 意味着证据从开局就在桌面上，
    ACQUIRE（=没有充分证据）自相矛盾 → check10 同样按此判定。
    """
    def fail(check_no: int, detail: str) -> tuple[bool, str]:
        return False, f"GCEP_INVALID: check{check_no}: {detail}"

    sufficient_by_rollout = _sufficient_by_rollout(ctx, owner)
    original_sufficient = any(
        entry["state_id"] == ORIGINAL_STATE_ID for entry in ctx["sufficient_states"]
    )
    for diag in ctx["diagnoses"]:
        rid = diag["rollout_id"]
        stage = diag["stage"]
        vqa = vqa_norms.get(rid, vqa_norms.get(str(rid)))
        own_sufficient = bool(sufficient_by_rollout.get(rid)) or original_sufficient
        # check 9：vqa_norm=0 -> != CORRECT（ALL_WRONG 中任何 CORRECT 都非法）
        if vqa == 0 and stage == "CORRECT":
            return fail(9, f"rollout {rid} has vqa_norm=0 but stage=CORRECT")
        # check 10：ACQUIRE -> 无 available sufficient evidence
        if stage == "ACQUIRE" and own_sufficient:
            return fail(
                10,
                f"rollout {rid} diagnosed ACQUIRE but sufficient evidence is available "
                f"(own={sufficient_by_rollout.get(rid)}, original={original_sufficient})",
            )
        # check 11：READ/GROUND -> 必须有 available sufficient evidence
        if stage in ("READ", "GROUND") and not own_sufficient:
            return fail(
                11,
                f"rollout {rid} diagnosed {stage} but owns no sufficient state",
            )

    return _check_reading_grounding(ctx)


def validate_judge_output(
    obj: Any,
    group_type: Any,
    evidence_states_by_rollout: Any,
    vqa_norms: dict[str, int],
) -> tuple[bool, str]:
    """GCEP v2 统一入口：structural（check 1-7）+ per-group-type semantic checks。

    group_type 接受 GCEPGroupType 或其 value 字符串。任一检查失败 ->
    (False, reason)，调用方按  fallback clean-teacher OPD，训练绝不 crash。
    """
    if not isinstance(group_type, GCEPGroupType):
        group_type = GCEPGroupType(str(group_type))
    states, owner = collect_states(evidence_states_by_rollout)
    known_images = _known_image_ids(states)
    ok, reason, ctx = _validate_structural(obj, states, known_images, vqa_norms)
    if not ok:
        return ok, reason
    if group_type == GCEPGroupType.MIXED:
        return _validate_semantic_mixed(ctx, owner, vqa_norms)
    if group_type == GCEPGroupType.ALL_CORRECT:
        return _validate_semantic_all_correct(ctx, owner, vqa_norms)
    if group_type == GCEPGroupType.ALL_WRONG:
        return _validate_semantic_all_wrong(ctx, owner, vqa_norms)
    return False, f"GCEP_INVALID: unknown group_type: {group_type!r}"


def validate_gcep_output(
    obj: Any,
    evidence_states_by_rollout: Any,
    vqa_norms: dict[str, int],
) -> tuple[bool, str]:
    """执行  全部 14 项 deterministic 检查（v1 MIXED 行为）。

    返回 (ok, reason)。ok=False 时 reason 以 "GCEP_INVALID:" 开头；
    applicable=false 单独返回 (False, "GCEP_NOT_APPLICABLE")，同样走 fallback。

    GCEP v2：本函数保留为 validate_judge_output(..., MIXED, ...) 的薄 wrapper，
    旧调用方（RQ1 harness / A1）零变化。
    """
    return validate_judge_output(
        obj, GCEPGroupType.MIXED, evidence_states_by_rollout, vqa_norms
    )
