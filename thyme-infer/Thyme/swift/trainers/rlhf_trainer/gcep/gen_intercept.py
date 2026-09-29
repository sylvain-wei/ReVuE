# Modified for this anonymized research release: portable paths and release cleanup.
"""训练生成循环中的轨迹拦截和回显清理。

B1 将疑似 base64 的 case_id 转为稳定短名。
B2 将 sandbox 回显中的过长 base64 字符串替换为占位符。
B3 根据压缩比和重复行占比检测复读。
B4 在累计回合字符数超过预算时终止轨迹。
B5 使用尾部窗口中的循环词占比检测变体复读。

终止时按预算截断并追加 </answer>；invalid 原因通过 metadata 传递，
人工后缀通过 suffix_span 从训练 labels 中排除。traj_filter 仍识别已有
文本终止标记以支持兼容输入。这些启发式规则也可能终止正确轨迹。

默认关闭，仅 GEN_INTERCEPT=1 时启用。评测使用 VLMEvalKit 的独立实现，
不经过本模块。"""
from __future__ import annotations

import hashlib
import os
import re
import zlib
from collections import Counter
from typing import List, Optional, Tuple

# 终止标记（与 traj_filter 约定；改动需同步 traj_filter.MARKER_REASONS）
MARKER_REPETITION = "[rep-terminated]"
MARKER_TOO_LONG = "[len-terminated]"
MARKER_LOOP = "[loop-terminated]"
MARKER_REASONS = {
    MARKER_REPETITION: "gen_repetition",
    MARKER_TOO_LONG: "gen_too_long",
    MARKER_LOOP: "gen_loop",
}

_B64_RUN = re.compile(r"[A-Za-z0-9+/=]{%d,}")
_CASE_ID_B64 = re.compile(r"^[A-Za-z0-9+/=]{16,}$")

# B5：以尾部循环词占比检测措辞变化的复读；这是启发式信号。
LOOP_WORDS = frozenset((
    "final", "thus", "read", "reads", "correct", "correctly", "confirm",
    "confirms", "confirmed", "ensuring", "ensure", "match", "matches", "matchs",
    "point", "points", "during", "yes", "properly", "verify", "verified",
    "again", "below", "conclusion", "assert", "recheck",
))
_WORD_RE = re.compile(r"[A-Za-z]+")


def loop_score(text: str, window: int = 2065) -> float:
    """返回尾部窗口中循环词的占比；窗口不足 50 个词时不判定。"""
    if not text:
        return 0.0
    tail = text[-window:]
    words = _WORD_RE.findall(tail)
    n = len(words)
    if n < 50:  # 窗口太短不判（避免小样本噪声）
        return 0.0
    return sum(1 for w in words if w.lower() in LOOP_WORDS) / n


def enabled() -> bool:
    """总开关：GEN_INTERCEPT=1 启用（默认关 = 既有行为逐字节不变）。"""
    return os.getenv("GEN_INTERCEPT", "0") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def repetition_scores(text: str) -> Tuple[float, float]:
    """返回 (zlib 压缩比, 最高频行占比)。文本太短时占比记 0。"""
    if not text:
        return 1.0, 0.0
    raw = text.encode("utf-8", "ignore")
    ratio = len(zlib.compress(raw)) / max(1, len(raw))
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    top_share = 0.0
    if len(lines) >= 15:
        top_share = Counter(lines).most_common(1)[0][1] / len(lines)
    return ratio, top_share


def is_repetitive(text: str) -> Tuple[bool, str]:
    """B3 判定。返回 (是否命中, 证据字符串)。"""
    min_chars = _env_int("GEN_REP_MIN_CHARS", 2500)  # 短文本不使用复读检测，减少模板枚举误判。
    if len(text) < min_chars:
        return False, ""
    ratio, top = repetition_scores(text)
    if ratio <= _env_float("GEN_REP_ZLIB_MAX", 0.18):
        return True, f"zlib={ratio:.3f}"
    if top >= _env_float("GEN_REP_TOPSHARE", 0.5):
        return True, f"top_line_share={top:.2f}"
    return False, ""


def turn_too_long(accumulated_turn_chars: int) -> bool:
    """累计回合字符数是否超过 GEN_TURN_MAX_CHARS（默认 4200）。

    调用方负责判断回合尚未正常结束。阈值按字符计算，不是精确 token 限额；
    长度截断可能终止本可给出正确答案的轨迹。"""
    return accumulated_turn_chars > _env_int("GEN_TURN_MAX_CHARS", 4200)


def check_turn(response: str, accumulated_chars: int = 0,
               acc_text: str = "") -> Optional[str]:
    """生成侧单次判定入口。

    Args:
        response: 本轮新生成的文本（未追加进 messages 前）。
        accumulated_chars: 本条样本当前回合已累积的 assistant 字符数（不含 response）。
        acc_text: 当前回合已累积的 assistant 原文（可选；用于 B5 尾部判定）。
    Returns:
        None（放行）| "too_long"（B4）| "repetition"（B3）| "loop"（B5）
    """
    if not enabled():
        return None
    acc = accumulated_chars + len(response or "")
    if turn_too_long(acc):
        return "too_long"
    hit, _ev = is_repetitive(response or "")
    if hit:
        return "repetition"
    # B5：变体复读死循环——在"累积尾部 + 本轮"的窗口上判（循环常跨轮累积）
    if loop_score((acc_text or "") + (response or "")) >= _env_float("GEN_LOOP_MIN", 0.10):
        return "loop"
    return None


def terminate(response: str, reason: str, acc_chars: int = 0) -> Tuple[str, str]:
    """把 response 改写为"截断到预算 + 收尾"形式，并返回追加的后缀文本。

    - 截断：命中时本轮 response 已完整生成（最多 ~12K 字符），
      若直接保留会让"界定"落在 cap+12K ≈ 18-20K。这里按预算
      `budget = cap - 已累积字符数` 截断 response，使终止行最终长度 ≈ cap。
    - 收尾：只追加自然的 "</answer>"（短路本轮代码执行与下一轮 process_round）。
    - **零暴露**：不再写入 `[xxx-terminated]` 这类模型从未
      见过的 in-band 标记——标记会被 completion_mask 覆盖、进而被蒸馏（"人工 token
      进 loss"）。invalid 判定改由 metadata（trainer 把 reason 挂在样本上、
      经 traj_filter.detect_invalid(reasons=...) 消费），后缀本身也会在
      `_prepare_batch_inputs` 里从 labels 抹掉。详见 suffix_span。
    Returns:
        (finished_text, appended_suffix)：suffix 供调用方记录以便从 loss mask 中剔除。
    """
    resp = response or ""
    cap = _env_int("GEN_TURN_MAX_CHARS", 4200)
    budget = max(0, cap - max(0, int(acc_chars)))
    if len(resp) > budget:
        resp = resp[:budget]
    suffix = "\n</answer>"
    return resp + suffix, suffix


def suffix_span(input_ids_row, suffix_text: str, tokenizer) -> Optional[Tuple[int, int]]:
    """在一条样本的 token 序列里定位拦截后缀，返回 (start, end) 或 None。

    用途（零暴露）：trainer 在 `_prepare_batch_inputs` 里把该 span 的 labels 置 -100，
    使"人为追加的后缀"既不产生梯度、也不进入任何 teacher 目标。
    实现：把后缀单独 tokenize 后，在整行 token 里**从右往左**找同样长度的子序列；
    失败时退而求其次去掉首个 token（"\n" 可能与截断处末字符合并）再找一次。
    找不到就返回 None（调用方计数并跳过，保持其它行为不变）。
    """
    if not suffix_text or tokenizer is None:
        return None
    try:
        ids = tokenizer(suffix_text, add_special_tokens=False).get("input_ids") or []
    except Exception:
        return None
    if not ids:
        return None
    row = [int(x) for x in input_ids_row]
    for cand in (ids, ids[1:] if len(ids) > 1 else []):
        if not cand:
            continue
        for start in range(len(row) - len(cand), -1, -1):
            if row[start:start + len(cand)] == cand:
                return start, start + len(cand)
    return None


def sanitize_case_id(case_id: str) -> str:
    """B1：base64 形态的 case_id（如 "c29ekge7...SUVORK5CYII="）换成稳定短名。"""
    if not enabled() or not case_id:
        return case_id
    if _CASE_ID_B64.match(case_id):
        digest = hashlib.sha1(case_id.encode("utf-8", "ignore")).hexdigest()[:8]
        return f"img_{digest}"
    return case_id


def sanitize_echo(text: str) -> str:
    """B2：把 sandbox 注入文本里的 base64 长串替换为占位符（保留错误类型等短文本）。"""
    if not enabled() or not text:
        return text
    min_chars = max(64, _env_int("GEN_BLOB_MIN_CHARS", 256))
    pattern = re.compile(r"[A-Za-z0-9+/=]{%d,}" % min_chars)
    return pattern.sub(
        lambda m: f"[base64 blob omitted: {len(m.group(0))} chars]", text
    )


def stats_init() -> dict:
    return {"repetition": 0, "too_long": 0, "loop": 0, "case_id": 0, "blob": 0,
            "suffix_masked": 0, "suffix_miss": 0}
