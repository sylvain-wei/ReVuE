# Modified for this anonymized research release: portable paths and release cleanup.
"""EvidenceImage / EvidenceState 提取。

一条 Thyme rollout 的整条轨迹是 ONE assistant message content 字符串：
    reasoning → <code>...</code> → <sandbox_output>...<image>...</sandbox_output> → ... → <answer>...</answer>

关键语义：
- 一个 <sandbox_output> 块 = 一次 sandbox interaction；
- 一个块内可有多个 <image>（1 interaction + N visual observations，不是 N 次 tool call）。

本模块只做 marker 扫描，不需要 AST semantic parser。
模块顶层不 import swift / transformers / torch，重依赖全部惰性加载。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# marker 定义（与 thyme eval/model.py 的序列化格式一致）
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"<code>([\s\S]*?)</code>")
_SANDBOX_RE = re.compile(r"<sandbox_output>([\s\S]*?)</sandbox_output>")
_IMAGE_RE = re.compile(r"<image>")

ORIGINAL_IMAGE_ID = "ORIGINAL_IMAGE"
ORIGINAL_STATE_ID = "ORIGINAL_STATE"

# ---------------------------------------------------------------------------
# code token 计数：优先用 Thyme-SFT tokenizer，失败时退化为 whitespace 计数
# ---------------------------------------------------------------------------
# 选择开关：
#   "auto"       —— 尝试加载 Thyme tokenizer，失败自动退化 whitespace；
#   "thyme"      —— 强制使用 Thyme tokenizer（加载失败则抛错）；
#   "whitespace" —— 强制 whitespace 计数（不加载任何重依赖）。
TOKEN_COUNT_MODE = "auto"

# 默认 tokenizer 路径（相对 repo root；也可直接设为绝对路径）
THYME_TOKENIZER_PATH = os.path.join("thyme-infer", "checkpoints", "Thyme-SFT")

_TOKENIZER = None          # 惰性缓存
_TOKENIZER_FAILED = False  # 加载失败标记，避免重复尝试


def _repo_root() -> str:
    # 本文件位于 <repo>/thyme-infer/Thyme/swift/trainers/rlhf_trainer/gcep/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 6))


def _get_tokenizer():
    """惰性加载 Thyme-SFT tokenizer；auto 模式下任何失败都返回 None（走 fallback）。"""
    global _TOKENIZER, _TOKENIZER_FAILED
    if TOKEN_COUNT_MODE == "whitespace" or _TOKENIZER_FAILED:
        return None
    if _TOKENIZER is not None:
        return _TOKENIZER
    path = THYME_TOKENIZER_PATH
    if not os.path.isabs(path):
        path = os.path.join(_repo_root(), path)
    try:
        from transformers import AutoTokenizer  # 重依赖，惰性 import

        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        _TOKENIZER_FAILED = True
        if TOKEN_COUNT_MODE == "thyme":
            raise
        return None
    return _TOKENIZER


def count_code_tokens(code: str) -> int:
    """统计一段 code 的 token 数。

    优先使用 Thyme-SFT tokenizer（add_special_tokens=False）；
    tokenizer 不可用（缺 transformers / 缺权重文件）时退化为
    whitespace split 计数 —— 只影响 state_cost 的排序粒度，
    不影响 state / image 的结构正确性。
    """
    if not code:
        return 0
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(code, add_special_tokens=False))
    return len(code.split())


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceImage:
    image_id: str
    rollout_id: str
    turn_id: int
    image_index: int
    path: str


@dataclass(frozen=True)
class EvidenceState:
    state_id: str
    rollout_id: Optional[str]
    end_turn: int
    tool_rounds: int
    cumulative_code_tokens: int
    new_image_ids: tuple
    available_image_ids: tuple


# ---------------------------------------------------------------------------
# 轨迹解析
# ---------------------------------------------------------------------------
def _split_sandbox_text(block: str) -> tuple:
    """把 sandbox 块内文本拆成 (status, stdout, stderr)。

    序列化格式没有显式 status 字段，采用保守启发式：
    文本（去掉 <image> 后）含 Traceback / Error 特征时判为 error，
    整体归入 stderr；否则为 ok / stdout。
    """
    text = _IMAGE_RE.sub("", block).strip()
    if re.search(r"Traceback|\bError\b", text):
        return "error", "", text
    return "ok", text, ""


def parse_turns(assistant_content: str) -> list:
    """把 assistant content 解析成 turn 列表，一次 sandbox interaction = 一个 turn。

    返回 [{turn_id, reasoning, code, sandbox_status, sandbox_stdout,
           sandbox_stderr, image_count}, ...]
    无 sandbox 块（answer-only rollout）时返回 []。
    """
    turns = []
    prev_end = 0
    for turn_id, m in enumerate(_SANDBOX_RE.finditer(assistant_content)):
        # 本 interaction 之前的文本段：reasoning + 至多一个 <code> 块
        seg = assistant_content[prev_end:m.start()]
        code_match = None
        for cm in _CODE_RE.finditer(seg):
            code_match = cm  # 取最后一个 code 块（当前 interaction 触发的那段）
        code = code_match.group(1) if code_match else ""
        reasoning = seg[: code_match.start()] if code_match else seg

        block = m.group(1)
        status, stdout, stderr = _split_sandbox_text(block)
        turns.append(
            {
                "turn_id": turn_id,
                "reasoning": reasoning.strip(),
                "code": code,
                "sandbox_status": status,
                "sandbox_stdout": stdout,
                "sandbox_stderr": stderr,
                "image_count": len(_IMAGE_RE.findall(block)),
            }
        )
        prev_end = m.end()
    return turns


# ---------------------------------------------------------------------------
# Judge 输入包含完整轨迹和模型真实答案。
# ---------------------------------------------------------------------------
# parse_turns 在没有 sandbox 块时返回空列表，供 EvidenceState 构造使用。
# 以下 helper 额外提取尾段推理和模型答案，使 answer-only rollout 也能参与
# judge 消息构造；参考答案不替代模型的实际回答。
_ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>")
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def trailing_reasoning(assistant_content: str) -> str:
    """最后一个 sandbox 块之后的尾段（无 sandbox 块时为全部内容）。

    v1 judge 输入把这段（含 post-tool 最终推理 + 模型真实答案）整体丢掉了。
    """
    ms = list(_SANDBOX_RE.finditer(assistant_content))
    tail = assistant_content[ms[-1].end():] if ms else assistant_content
    return tail.strip()


def extract_final_answer(assistant_content: str) -> str:
    """模型真实最终答案（精简）：最后一个 <answer> 块（块内优先 \\boxed{}，
    块过长时取最后一行）→ 最后一个 \\boxed{} → 最后一行非空文本。

    绝不回退到 solution/reference（那是 v1 伪影的来源）。
    """
    def _concise(block: str) -> str:
        boxed = _BOXED_RE.findall(block)
        if boxed:
            return boxed[-1].strip()
        block = block.strip()
        if len(block) <= 160:
            return block
        for line in reversed(block.splitlines()):
            line = line.strip()
            if line:
                return line
        return block

    matches = _ANSWER_RE.findall(assistant_content)
    if matches:
        return _concise(matches[-1])
    return _concise(assistant_content)


def build_evidence_states(rollout_id, assistant_content: str, image_paths) -> tuple:
    """从一条 rollout 构造 (list[EvidenceImage], list[EvidenceState])。

    image_paths: rollout dict 的 images 列表，images[0] 为原图，
                 其余为各 sandbox 块按 encounter 顺序追加的图。
    EvidenceImage 只登记 sandbox 产出图；原图用常量 id "ORIGINAL_IMAGE" 表示，
    不进入 EvidenceImage 列表（它不属于任何 rollout 的 turn）。
    若 <image> 标签数多于剩余 path 数，path 记为 ""（id 仍按标签数生成，
    保证 state 语义正确）。
    """
    rid = str(rollout_id)
    turns = parse_turns(assistant_content)
    sandbox_paths = list(image_paths[1:]) if image_paths else []

    images: list = []
    states: list = [
        EvidenceState(
            state_id=ORIGINAL_STATE_ID,
            rollout_id=None,
            end_turn=0,
            tool_rounds=0,
            cumulative_code_tokens=0,
            new_image_ids=(),
            available_image_ids=(ORIGINAL_IMAGE_ID,),
        )
    ]

    available = [ORIGINAL_IMAGE_ID]
    cum_code_tokens = 0
    path_cursor = 0  # sandbox_paths 的消费游标
    for turn in turns:
        r = turn["turn_id"]
        cum_code_tokens += count_code_tokens(turn["code"])
        new_ids = []
        for k in range(turn["image_count"]):
            img_id = f"R{rid}_T{r}_IMG{k}"
            path = sandbox_paths[path_cursor] if path_cursor < len(sandbox_paths) else ""
            path_cursor += 1
            images.append(
                EvidenceImage(
                    image_id=img_id,
                    rollout_id=rid,
                    turn_id=r,
                    image_index=k,
                    path=path,
                )
            )
            new_ids.append(img_id)
        available.extend(new_ids)
        states.append(
            EvidenceState(
                state_id=f"R{rid}_AFTER_T{r}",
                rollout_id=rid,
                end_turn=r,
                tool_rounds=r + 1,
                cumulative_code_tokens=cum_code_tokens,
                new_image_ids=tuple(new_ids),
                available_image_ids=tuple(available),
            )
        )
    return images, states


# ---------------------------------------------------------------------------
# Group Gold / Anchor 选择用的代价函数
# ---------------------------------------------------------------------------
def state_cost(state: EvidenceState, num_support_images: int) -> tuple:
    """lexicographic 排序键：先 NO_TOOL/少轮，再短代码，再少 supporting images，
    最后 state_id 固定打破平局。"""
    return (
        state.tool_rounds,
        state.cumulative_code_tokens,
        num_support_images,
        state.state_id,
    )
