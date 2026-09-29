# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP 全量 rollout 归档模块。

设计要点：
- 文本 shard：JSONL 单行/文件，优先 zstandard 压缩（.jsonl.zst），
  zstandard 不可导入时回退 gzip（.jsonl.gz），由 COMPRESSION_BACKEND 标识；
- sandbox 图片：同 filesystem 优先 hardlink（零拷贝），跨设备回退 shutil.copy2，
  多张图全部保存且保持原顺序，缺失文件记录 None 不崩溃；
- 分 rank 写 shard：文件名携带 rank 与单调 seq，禁止多进程 append 同一文件；
- 本模块不 import swift / torch / transformers，可独立导入。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

MASTER_SEED = 42
STORAGE_SAFETY_MARGIN = 1.5  #  存储 preflight 安全余量


def derive_seed(exp_id, global_step, prompt_id, rollout_idx):
    """ 规定的稳定 sub-seed 派生：从 master seed 42 确定性派生。"""
    key = f"{MASTER_SEED}|{exp_id}|{global_step}|{prompt_id}|{rollout_idx}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 1)


# ---------------------------------------------------------------------------
# 压缩后端（zstandard 优先，gzip 回退）
# ---------------------------------------------------------------------------

_BACKEND_CACHE = None


def compression_backend():
    """返回 "zstd" 或 "gzip"；zstandard 为可选依赖，懒检测并缓存。"""
    global _BACKEND_CACHE
    if _BACKEND_CACHE is None:
        try:
            import zstandard  # noqa: F401

            _BACKEND_CACHE = "zstd"
        except Exception:
            _BACKEND_CACHE = "gzip"
    return _BACKEND_CACHE


def shard_suffix():
    return ".jsonl.zst" if compression_backend() == "zstd" else ".jsonl.gz"


def _write_jsonl_line(path, line):
    if compression_backend() == "zstd":
        import zstandard

        cctx = zstandard.ZstdCompressor(level=3)
        with open(path, "wb") as f:
            with cctx.stream_writer(f) as w:
                w.write(line.encode("utf-8"))
    else:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(line)


def read_records(path):
    """按后缀解压并读取 shard 中的全部 JSONL 记录（用于回放/校验）。"""
    path = str(path)
    if path.endswith(".jsonl.zst"):
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with open(path, "rb") as f:
            with dctx.stream_reader(f) as r:
                text = r.read().decode("utf-8")
    elif path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# 轨迹解析：<sandbox_output> 块切分 turns
# ---------------------------------------------------------------------------

_SANDBOX_RE = re.compile(r"<sandbox_output>(.*?)</sandbox_output>", re.DOTALL)
_IMAGE_TAG_RE = re.compile(r"<image[^>]*>.*?</image>|<image[^>]*/?>", re.DOTALL)
_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text):
    m = _CODE_RE.search(text or "")
    return m.group(1).strip() if m else ""


def parse_turns(trajectory):
    """按 <sandbox_output> 块把完整轨迹切为 turns。

    语义（对齐 ）：每个 sandbox 块之前的 policy 文本段为一个 turn；
    最后一个 sandbox 之后若仍有尾部文本，则追加一个无 sandbox 的收尾 turn
    （sandbox_status="none"）。sandbox_stderr 无法从纯文本轨迹恢复，置 ""。
    """
    trajectory = trajectory or ""
    turns = []
    pos = 0

    def _make_turn(reasoning, status, stdout="", n_images=0):
        return {
            "turn_id": len(turns),
            "reasoning": (reasoning or "").strip(),
            "code": _extract_code(reasoning),
            "sandbox_status": status,
            "sandbox_stdout": stdout,
            "sandbox_stderr": "",
            "sandbox_image_archive_paths": [],
            "sandbox_image_count": n_images,  # schema 之外的辅助字段（ 为"至少"）
        }

    for m in _SANDBOX_RE.finditer(trajectory):
        block = m.group(1)
        stdout = _IMAGE_TAG_RE.sub("", block).strip()
        n_images = len(_IMAGE_TAG_RE.findall(block))
        turns.append(_make_turn(trajectory[pos:m.start()], "ok", stdout, n_images))
        pos = m.end()

    tail = trajectory[pos:]
    if tail.strip() or not turns:
        turns.append(_make_turn(tail, "none"))
    return turns


# ---------------------------------------------------------------------------
#  记录构建
# ---------------------------------------------------------------------------


def build_record(
    *,
    experiment_id,
    global_step,
    source_dataset="",
    prompt_id="",
    group_id="",
    rollout_id="",
    rollout_idx=0,
    question="",
    options=None,
    original_image_path="",
    full_raw_trajectory="",
    final_answer="",
    turns=None,
    turn_image_archive_paths=None,
    vqa_norm=0,
    reward_components=None,
    total_reward=0.0,
    token_ids=None,
    policy_token_mask=None,
    interaction_turn_ids=None,
    text_temperature=1.0,
    code_temperature=0.0,
    sampling=None,
    policy_version="",
    checkpoint_source="",
    rollout=None,
    extra_tensors=None,
    gcep_meta=None,
):
    """构建  完整 schema 的 rollout 归档记录。

    - rollout：可选，repo 原生 rollout dict（messages/images/solution/is_truncated），
      用于自动回填 question / full_raw_trajectory / original_image_path / final_answer，
      显式传入的同名参数优先；
    - turns：缺省时从 full_raw_trajectory 用 parse_turns 解析；
    - turn_image_archive_paths：可选 dict {turn_id: [archived_path, ...]}，
      填入对应 turn 的 sandbox_image_archive_paths；
    - extra_tensors：可选 dict，若 repo 已保存 old_logprob / response_mask /
      answer_mask 等，原样并入归档（：不为存档额外做 forward）；
    - 可选张量字段缺省一律为空 list，绝不落 None。
    """
    if rollout is not None:
        messages = rollout.get("messages") or []
        user_msgs = [m for m in messages if m.get("role") == "user"]
        asst_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not question and user_msgs:
            question = user_msgs[0].get("content", "")
        if not full_raw_trajectory and asst_msgs:
            # 整条轨迹位于最后一条 assistant message 的 content 字符串中
            full_raw_trajectory = asst_msgs[-1].get("content", "")
        images = rollout.get("images") or []
        if not original_image_path and images:
            # images[0] 为原始图，其后为按序追加的 sandbox 图
            original_image_path = images[0].get("path", "") or ""
        if not final_answer:
            final_answer = rollout.get("solution", "") or ""

    if turns is None:
        turns = parse_turns(full_raw_trajectory)
    else:
        turns = [dict(t) for t in turns]

    if turn_image_archive_paths:
        for tid, paths in turn_image_archive_paths.items():
            tid = int(tid)
            if 0 <= tid < len(turns):
                turns[tid]["sandbox_image_archive_paths"] = list(paths)

    sampling_dict = {
        "text_temperature": text_temperature,
        "code_temperature": code_temperature,
    }
    if sampling:
        sampling_dict.update(sampling)

    record = {
        "experiment_id": str(experiment_id),
        "global_step": int(global_step),
        "master_seed": MASTER_SEED,
        "derived_rollout_seed": derive_seed(
            experiment_id, global_step, prompt_id, rollout_idx
        ),
        "source_dataset": source_dataset,
        "prompt_id": str(prompt_id),
        "group_id": str(group_id),
        "rollout_id": str(rollout_id),
        "question": question,
        "options": list(options) if options else [],
        "original_image_path": original_image_path,
        "full_raw_trajectory": full_raw_trajectory,
        "final_answer": final_answer,
        "turns": turns,
        "vqa_norm": vqa_norm,
        "reward_components": dict(reward_components) if reward_components else {},
        "total_reward": float(total_reward),
        "token_ids": list(token_ids) if token_ids is not None else [],
        "policy_token_mask": (
            list(policy_token_mask) if policy_token_mask is not None else []
        ),
        "interaction_turn_ids": (
            list(interaction_turn_ids) if interaction_turn_ids is not None else []
        ),
        "sampling": sampling_dict,
        "policy_version": str(policy_version),
        "checkpoint_source": str(checkpoint_source),
    }
    if extra_tensors:
        record.update({k: v for k, v in extra_tensors.items() if v is not None})
    # GCEP v2：可选 per-rollout GCEP 元数据（group_type / privilege_type /
    # diagnosis_stage / discrepancy / judge_valid 等）。旧 reader 忽略未知字段。
    if gcep_meta:
        record["gcep"] = {k: v for k, v in gcep_meta.items() if v is not None}
    return record


# ---------------------------------------------------------------------------
# 归档器
# ---------------------------------------------------------------------------


class RolloutArchiver:
    """RUN_DIR/rollout_archive/ 下的全量 rollout 归档器（ 目录语义）。"""

    def __init__(self, run_dir, experiment_id):
        self.run_dir = Path(run_dir)
        self.experiment_id = str(experiment_id)
        self.archive_root = self.run_dir / "rollout_archive"
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._seq_counters = {}  # (step, rank) -> 下一个 seq，保证 rank 间不共享文件
        self._write_manifest()

    def _write_manifest(self):
        manifest = self.archive_root / "manifest.json"
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    {
                        "experiment_id": self.experiment_id,
                        "master_seed": MASTER_SEED,
                        "compression": compression_backend(),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def archive_rollout(self, record_dict, global_step, rank):
        """把一条 rollout 记录写入 step 目录下 rank 专属 shard，返回文件路径。

        文件名 rank_{rank:02d}_{seq:05d}.jsonl.(zst|gz)：seq 按 (step, rank)
        单调递增，任何两个 rank 绝不写同一个文件。
        """
        step = int(global_step)
        rank = int(rank)
        step_dir = self.archive_root / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        key = (step, rank)
        seq = self._seq_counters.get(key, 0)
        self._seq_counters[key] = seq + 1
        path = step_dir / f"rank_{rank:02d}_{seq:05d}{shard_suffix()}"
        _write_jsonl_line(path, json.dumps(record_dict, ensure_ascii=False) + "\n")
        return path

    def persist_sandbox_images(self, image_paths, step, rollout_id):
        """把 sandbox 返回图持久化到 step 目录的 sandbox_images/ 下。

        同 filesystem 用 os.link hardlink（零拷贝），失败回退 shutil.copy2；
        多张图全部保存并严格保序；缺失文件记录为 None，不抛异常。
        返回与输入等长、按序对齐的 archived path（或 None）列表。
        """
        img_dir = self.archive_root / f"step_{int(step):06d}" / "sandbox_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        archived = []
        for idx, src in enumerate(image_paths or []):
            if not src or not os.path.exists(src):
                archived.append(None)  # 缺失只记录，不崩溃（但绝不静默丢弃位置）
                continue
            suffix = Path(src).suffix or ".png"  # 图片保持原编码/扩展名
            dest = img_dir / f"{rollout_id}_img{idx:02d}{suffix}"
            if dest.exists():
                dest.unlink()
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
            archived.append(str(dest))
        return archived


# ---------------------------------------------------------------------------
#  存储 preflight
# ---------------------------------------------------------------------------


def estimate_storage(num_prompts_measured, total_prompts_planned, dry_run_dir):
    """由给定 dry-run 的平均归档大小外推完整实验磁盘需求（含 1.5x 安全余量）。

    返回 {estimated_bytes, free_bytes, fits, margin}；fits=False 时调用方必须
    在完整实验开始前停止并报告存储空间不足。
    """
    d = Path(dry_run_dir)
    total = 0
    if d.exists():
        for p in d.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    avg = total / max(int(num_prompts_measured), 1)
    estimated = int(avg * int(total_prompts_planned) * STORAGE_SAFETY_MARGIN)
    probe = d if d.exists() else d.parent
    free = int(shutil.disk_usage(str(probe)).free)
    return {
        "estimated_bytes": estimated,
        "free_bytes": free,
        "fits": free >= estimated,
        "margin": STORAGE_SAFETY_MARGIN,
    }
