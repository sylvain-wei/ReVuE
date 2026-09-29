# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP 训练集成层。

本模块承载 grpo_trainer.py 的全部 GCEP 重逻辑；grpo_trainer 里只保留薄 hook。
硬保证：GCEP_TRAINING_MODE 未设 / 为 vanilla_grpo 时，grpo_trainer 行为与现状
逐字节等价（所有 GCEP 路径都不执行）。

训练模式（ 统一 config，避免复制六套 training loop）：
    vanilla_grpo        原生 GRPO（默认，GCEP 全关）
    strong_opd_clean    A0：L = L_OPD^clean，不调用 Judge
    strong_opd_gcep     A1：L = L_OPD，mixed+valid 用 privileged teacher，否则 clean
    strong_opd_gcep_ps  A2：L = L_PS（JSD token 加权）
    opsd_gcep           B1：teacher = 冻结 SFT ckpt，mixed+valid 用 privilege
    grpo_gcep_opd       C1：L = L_GRPO + λ·L_GCEP-OPD（λ=0.01， 分别归一化后相加）
    grpo_gcep_ps_opd    C2：L = L_GRPO + λ·L_PS-GCEP-OPD

显存纪律：teacher 一律 eval()+no_grad+requires_grad_(False)，不挂
optimizer/DeepSpeed；teacher 前向只保留 Top32（PS 模式保留 ≤64 union support 上
的两组 gather logits），全量 logits 立即释放。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# 兄弟模块导入：优先包内相对导入（grpo_trainer 经包路径加载时）；
# 单文件加载（离线测试用 importlib 按文件路径加载）时回退为顶层模块导入。
# ---------------------------------------------------------------------------
try:
    from . import (
        anchor as _anchor,
        evidence_state as _evidence,
        group_types as _gtypes,
        judge_client as _judge,
        opd_losses as _opd,
        privilege as _privilege,
        rollout_archive as _archive,
        validator as _validator,
    )
except ImportError:  # pragma: no cover - 仅离线单文件测试路径
    import sys as _sys

    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
    import anchor as _anchor
    import evidence_state as _evidence
    import group_types as _gtypes
    import judge_client as _judge
    import opd_losses as _opd
    import privilege as _privilege
    import rollout_archive as _archive
    import validator as _validator

GCEPGroupType = _gtypes.GCEPGroupType

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

GCEP_TRAINING_MODES = (
    "vanilla_grpo",
    "strong_opd_clean",
    "strong_opd_gcep",
    "strong_opd_gcep_ps",
    "opsd_gcep",
    "grpo_gcep_opd",
    "grpo_gcep_ps_opd",
    # GCEP v2：
    #   strong_opd_gcep_v2        A1v2（IDEA-1）：ALL_CORRECT/MIXED/ALL_WRONG 三类
    #                             group 全覆盖 privilege；loss 与 A1 逐式相同。
    #   strong_opd_gcep_impact_v2 A2v2（IDEA-2）：mixed-only 覆盖（=A1），
    #                             新 sampled-token |ΔlogP| top20% high/low RKL。
    #   strong_opd_gcep_combined_v2 A3v2：IDEA-1+IDEA-2
    #                             同时开（三类全判 + impact reweight）。
    "strong_opd_gcep_v2",
    "strong_opd_gcep_impact_v2",
    "strong_opd_gcep_combined_v2",
    # MT baselines（matched-teacher
    # controlled adaptations；teacher=冻结 RL-2150，student=Thyme-SFT，几何同 A 系）：
    #   vision_opd_mt  Vision-OPD-MT：teacher 看证据 crop（图像交换），
    #                  JSD(0.5) top-100+tail，is_clip=2.0，token-mean。
    #   vzero_mt       V-Zero-MT：正/负 crop 双 replay，evidence gate（group z），
    #                  gated k1 + PPO vanilla surrogate（精确移植官方 loss）。
    #   vad_mt         VAD-MT：present/degraded 双视图，budgeted_asvc target，
    #                  JSD(0.5) + 非视觉锚 η=0.1，is_clip=2.0，token-mean。
    "vision_opd_mt",
    "vzero_mt",
    "vad_mt",
    # GT-only privilege：
    #   strong_opd_gt_privilege  vanilla OPD（= strong_opd_clean 的 loss/几何）
    #   基础上，teacher 的 user prompt 追加 GT privilege 文本
    #   （gcep/privilege.build_gt_only_privilege，固定模板）。
    #   无 Judge / 无 certificate / 无 support image：privilege 逐 rollout
    #   直接由 solution 构造，loss 与 A0 逐式相同（plain mean Top32 RKL）。
    "strong_opd_gt_privilege",
)

_PRIVILEGE_MODES = frozenset(
    ("strong_opd_gcep", "strong_opd_gcep_ps", "opsd_gcep", "grpo_gcep_opd", "grpo_gcep_ps_opd",
     "strong_opd_gcep_v2", "strong_opd_gcep_impact_v2", "strong_opd_gcep_combined_v2",
     "strong_opd_gt_privilege")
)
_PS_MODES = frozenset(("strong_opd_gcep_ps", "grpo_gcep_ps_opd"))
_GRPO_MODES = frozenset(("grpo_gcep_opd", "grpo_gcep_ps_opd"))
# v2：全 group type 覆盖（A1v2/A3v2）；A2v2 保持 mixed-only 覆盖（与 A1 相同）。
_ALLGROUP_MODES = frozenset(("strong_opd_gcep_v2", "strong_opd_gcep_combined_v2"))
# v2：sampled-token |ΔlogP| high/low 重加权（A2v2/A3v2）。
_IMPACT_MODES = frozenset(("strong_opd_gcep_impact_v2", "strong_opd_gcep_combined_v2"))
# MT baselines：离线证据库视觉条件 + 官方公式移植（无 GCEP privilege/Judge/ARG）。
_MT_MODES = frozenset(("vision_opd_mt", "vzero_mt", "vad_mt"))
# A5：token 分桶消融臂的合法取值（模块级，供 GCEPConfig.from_env 校验）。
_IMPACT_ABLATION_MODES = frozenset(("", "mask_top20pp", "random_select20pp", "mask_low20pp", "mask_random20pp"))
# GT-only privilege：不调 Judge、不建 certificate，privilege 逐
# rollout 由 GT（solution）直接构造（与 RLSD 的 gt_only 语义一致，但走 GCEP
# 的 privileged-teacher 通道）。
_GT_PRIVILEGE_MODES = frozenset(("strong_opd_gt_privilege",))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 6))
# ：strong OPD / GRPO+OPD 的强教师 = 冻结 Thyme-RL ckpt-2150
DEFAULT_STRONG_TEACHER_CKPT = "/path/to/teacher-checkpoint"
# OPSD 的 teacher = 同 SFT ckpt 冻结
DEFAULT_SFT_TEACHER_CKPT = os.path.join(_REPO_ROOT, "thyme-infer", "checkpoints", "Thyme-SFT")

# 前向时需要剔除的非模型入参（与 grpo_trainer._get_per_token_logps 的排除表一致
# 外加 GCEP 自身的挂载键）。
_NON_MODEL_KEYS = frozenset(
    (
        "logits_to_keep",
        "completion_mask",
        "ref_per_token_logps",
        "advantages",
        "old_per_token_logps",
        "truncated_mask",
        "gcep_teacher",
    )
)


# ---------------------------------------------------------------------------
#  统一训练模式 config
# ---------------------------------------------------------------------------
@dataclass
class GCEPConfig:
    """GCEP 训练配置（全部 env 可控，默认全关 = vanilla_grpo）。"""

    training_mode: str = "vanilla_grpo"
    teacher_ckpt: str = ""
    lambda_opd: float = 0.01  # GRPO+OPD 的 OPD 系数
    topk: int = 32  # teacher Top-K support 大小
    judge_concurrency: int = 4  # 同 batch 多 mixed group 的并发
    archive: bool = True  # 全量 rollout 归档（GCEP 实验内默认开）
    experiment_id: str = ""
    teacher_device_map: str = ""  # 空 = 每 rank 单卡 cuda:LOCAL_RANK
    teacher_attn: str = "sdpa"  # teacher 前向 attn 实现（避开老机器 flash_attn GLIBC 问题）
    # --- MT baseline 字段（仅 _MT_MODES 消费，其他模式完全惰性） ---
    mt_evidence_bank: str = ""  # build_bank.py 产物目录（含 manifest.jsonl）
    mt_topk: int = 100  # 官方 distillation_topk=100（vopd/vad student support）
    mt_alpha: float = 0.5  # 官方 alpha=0.5（JSD）
    mt_is_clip: float = 2.0  # 官方 is_clip=2.0
    vad_cap: float = 0.7  # 官方 vc_positive_budget_cap（7B 取 9B 档 0.7）
    vad_anchor_eta: float = 0.1  # 官方 vc_nonvisual_anchor_eta=0.1
    vad_ridge: float = 1e-3  # 官方 vc_projection_ridge
    vad_max_shift: float = 20.0  # 官方 vc_max_logit_shift
    vzero_gamma: float = 1.0  # 官方 REW_OPD_GAMMA
    vzero_alpha: float = 0.5  # 官方 REW_OPD_ALPHA
    vzero_wmin: float = 0.0  # 官方 REW_OPD_W_MIN
    vzero_wmax: float = 2.0  # 官方 REW_OPD_W_MAX
    vzero_eps: float = 1e-6  # 官方 REW_OPD_EPS
    vzero_clip: float = 0.2  # 官方 distillation_loss clip_ratio（low=high=0.2）
    vzero_k1_clamp: float = 10.0  # 官方 loss_max_clamp=10 / log_prob_min_clamp=-10
    # A4：per-rollout impact High 组占比（High=top frac by |Δlogp|，Low=其余，
    # 两组 0.5/0.5 分别加权）。默认 0.2 = A2v2/A3v2 既有行为（top-20%）。
    impact_top_frac: float = 0.2
    # A5：token 分桶的两个消融对照臂。
    #   ""                = 基线（impact top-frac 分桶，逐字节等价）。
    #   "mask_top20pp"    = 去掉 top-frac 高 impact token 的梯度，只训 low 桶：
    #                       loss_i = 0.5·mean(rkl[low])（high 整项删除、保留 0.5 系数）。
    #   "random_select20pp"= privilege 不变，但分桶改为 seed-42 随机 frac/(1-frac)，
    #                       50:50 公式不变（high=perm[:k_top]、low=perm[k_top:]）。
    #   "mask_low20pp"    = 去掉最低 frac 低 impact token 的梯度，只训 top-(1-frac) 桶：
    #                       loss_i = 0.5·mean(rkl[top80%])（low 整项删除、保留 0.5 系数；
    #                       与 mask_top20pp 镜像对称。分桶按 impact 升序 stable sort，
    #                       掩码 = 升序前 ceil(frac·N)）。
    #   "mask_random20pp" = random_select20pp 的掩码版：同 seed-42 随机分桶（perm 逐点一致），
    #                       随机选中的 frac 桶整项删除不反传，只训其余 (1-frac) 桶：
    #                       loss_i = 0.5·mean(rkl[perm[k_top:]])。与 mask_low20pp 对照可
    #                       分离"删掉的是高 impact 桶还是任意 20% 桶"。
    # 默认 "" = 关闭，既有 run/行为/日志完全一致。
    impact_ablation: str = ""
    impact_ablation_seed: int = 42

    @property
    def is_mt_mode(self) -> bool:
        """MT baseline（vision_opd_mt / vzero_mt / vad_mt）：无 privilege/Judge/ARG。"""
        return self.training_mode in _MT_MODES

    @property
    def mt_kind(self) -> str:
        return self.training_mode if self.is_mt_mode else ""

    @classmethod
    def from_env(cls) -> "GCEPConfig":
        mode = os.getenv("GCEP_TRAINING_MODE", "vanilla_grpo").strip() or "vanilla_grpo"
        if mode not in GCEP_TRAINING_MODES:
            raise ValueError(
                f"GCEP_TRAINING_MODE={mode!r} 非法，可选: {list(GCEP_TRAINING_MODES)}"
            )
        if mode in _MT_MODES:
            raise ValueError(
                f"Training mode {mode!r} requires baseline modules and evidence-bank "
                "tools that are not included in this main-method release."
            )
        # A5：消融臂白名单校验（非白名单值立刻报错，不静默退回基线）。
        _abl = os.getenv("GCEP_IMPACT_ABLATION", "").strip()
        if _abl not in _IMPACT_ABLATION_MODES:
            raise ValueError(
                f"GCEP_IMPACT_ABLATION={_abl!r} 非法，可选: {sorted(_IMPACT_ABLATION_MODES)}"
            )
        return cls(
            training_mode=mode,
            teacher_ckpt=os.getenv("GCEP_TEACHER_CKPT", "").strip(),
            lambda_opd=float(os.getenv("GCEP_LAMBDA_OPD", "0.01")),
            topk=int(os.getenv("GCEP_TOPK", "32")),
            judge_concurrency=int(os.getenv("GCEP_JUDGE_CONCURRENCY", "4")),
            archive=bool(int(os.getenv("GCEP_ARCHIVE", "1"))),
            experiment_id=os.getenv("GCEP_EXPERIMENT_ID", os.getenv("RUN_ID", "")).strip(),
            teacher_device_map=os.getenv("GCEP_TEACHER_DEVICE_MAP", "").strip(),
            teacher_attn=os.getenv("GCEP_TEACHER_ATTN", "sdpa").strip(),
            mt_evidence_bank=os.getenv("GCEP_MT_EVIDENCE_BANK", "").strip(),
            mt_topk=int(os.getenv("GCEP_MT_TOPK", "100")),
            mt_alpha=float(os.getenv("GCEP_MT_ALPHA", "0.5")),
            mt_is_clip=float(os.getenv("GCEP_MT_IS_CLIP", "2.0")),
            vad_cap=float(os.getenv("GCEP_VAD_CAP", "0.7")),
            vad_anchor_eta=float(os.getenv("GCEP_VAD_ANCHOR_ETA", "0.1")),
            vad_ridge=float(os.getenv("GCEP_VAD_RIDGE", "0.001")),
            vad_max_shift=float(os.getenv("GCEP_VAD_MAX_SHIFT", "20.0")),
            vzero_gamma=float(os.getenv("GCEP_VZERO_GAMMA", "1.0")),
            vzero_alpha=float(os.getenv("GCEP_VZERO_ALPHA", "0.5")),
            vzero_wmin=float(os.getenv("GCEP_VZERO_WMIN", "0.0")),
            vzero_wmax=float(os.getenv("GCEP_VZERO_WMAX", "2.0")),
            vzero_eps=float(os.getenv("GCEP_VZERO_EPS", "0.000001")),
            vzero_clip=float(os.getenv("GCEP_VZERO_CLIP", "0.2")),
            vzero_k1_clamp=float(os.getenv("GCEP_VZERO_K1_CLAMP", "10.0")),
            impact_top_frac=float(os.getenv("GCEP_IMPACT_TOP_FRAC", "0.2")),
            impact_ablation=_abl,
            impact_ablation_seed=int(os.getenv("GCEP_IMPACT_ABLATION_SEED", "42")),
        )

    # --- 模式语义 ---
    @property
    def enabled(self) -> bool:
        """vanilla_grpo = GCEP 全关（grpo_trainer 走原路径）。"""
        return self.training_mode != "vanilla_grpo"

    @property
    def use_grpo_loss(self) -> bool:
        """C 系：L = L_GRPO + λ·L_OPD；A/B 系：纯 OPD，GRPO 项不进 loss。"""
        return self.training_mode in _GRPO_MODES

    @property
    def use_privilege(self) -> bool:
        """是否需要 GCEP privilege 管线（mixed 判定 + Judge + certificate）。A0 不需要。"""
        return self.training_mode in _PRIVILEGE_MODES

    @property
    def use_ps_weight(self) -> bool:
        """A2/C2：privilege-sensitive JSD token 加权。"""
        return self.training_mode in _PS_MODES

    @property
    def use_all_group_types(self) -> bool:
        """A1v2：ALL_CORRECT/MIXED/ALL_WRONG 三类 group 全部判 Judge 并给 privilege。"""
        return self.training_mode in _ALLGROUP_MODES

    @property
    def use_impact_weight(self) -> bool:
        """A2v2：sampled-token |ΔlogP| top20% high/low 分组 RKL（替代旧 PS-JSD）。"""
        return self.training_mode in _IMPACT_MODES

    @property
    def is_gt_privilege_mode(self) -> bool:
        """GT-only privilege：privilege 逐 rollout 由 GT 构造，不调 Judge/Validator。

        其它一切（privileged teacher 前向、token 对齐校验、失败回退 clean、
        loss 口径）与 A 系完全共用，因此只影响 plan 的**来源**。
        """
        return self.training_mode in _GT_PRIVILEGE_MODES

    def resolve_teacher_ckpt(self) -> str:
        """显式 GCEP_TEACHER_CKPT 优先；否则按模式族取默认。"""
        if self.teacher_ckpt:
            return self.teacher_ckpt
        if self.training_mode == "opsd_gcep":
            return DEFAULT_SFT_TEACHER_CKPT
        return DEFAULT_STRONG_TEACHER_CKPT


# ---------------------------------------------------------------------------
# 冻结 teacher 加载（：eval / no_grad / requires_grad_(False) / 无 optimizer）
# ---------------------------------------------------------------------------
class FrozenTeacherLoader:
    """按需加载冻结 teacher（bf16），每个 rank 持有完整模型副本。

    不做 ZeRO-3 分片，以避免 teacher 前向在不同步的 judge/reward 调用之间
    触发集合通信等待。加载时临时取消 Hugging Face DeepSpeed 配置，使模型
    保留在本 rank 的设备上；student 的训练配置不变。

    完整 teacher 副本会增加每张卡的显存占用，实际峰值取决于模型、序列长度、
    vLLM 配置和训练激活。"""

    def __init__(self, ckpt: str, device_map: str = "", attn: str = "sdpa", accelerator=None):
        self.ckpt = ckpt
        self.device_map = device_map
        self.attn = attn
        self.accelerator = accelerator
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        from transformers import AutoModelForImageTextToText  # 惰性重依赖

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map: Any = self.device_map or {"": f"cuda:{local_rank}"}

        # 暂时禁用 HF deepspeed zero3 全局 config，使 from_pretrained 走普通整模型
        # 加载路径（允许 device_map、不切参数）。加载完立即恢复，不影响 student。
        from transformers.integrations import deepspeed as _hfds

        _saved_ref = getattr(_hfds, "_hf_deepspeed_config_weak_ref", None)
        model = None
        try:
            _hfds._hf_deepspeed_config_weak_ref = None
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.ckpt,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation=self.attn,
                )
            except ValueError:
                # InternVL 教师 ckpt 是 remote-code 配置（InternVLChatConfig），
                # AutoModelForImageTextToText 的内建映射不认 remote 配置类，
                # 且其 config.json 的 auto_map 只含 AutoConfig/AutoModel/
                # AutoModelForCausalLM（无 AutoModelForImageTextToText 键）。
                # 回退 AutoModelForCausalLM：凭 trust_remote_code 走 auto_map 解析到
                # modeling_internvl_chat.InternVLChatModel，拿到 causal-LM 教师。
                # （Qwen 教师走上面的原生路径，不受影响。）
                # 注意：InternVL remote 类只声明 _supports_flash_attn_2（无 sdpa），
                # self.attn="sdpa" 会被 _check_and_enable_sdpa 拒绝，所以这里
                # 按 flash_attention_2 -> eager 顺序重试 attn 实现。
                from transformers import AutoModelForCausalLM
                for _attn in (self.attn, "flash_attention_2", "eager"):
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.ckpt,
                            torch_dtype=torch.bfloat16,
                            device_map=device_map,
                            trust_remote_code=True,
                            attn_implementation=_attn,
                        )
                        break
                    except ValueError:
                        continue
                else:
                    raise
                # InternVL 教师适配（swift 训练范式）：swift 的 internvl3_5 模板在
                # encode 阶段就把视觉 embedding 拼进 inputs_embeds（见
                # template/template/internvl.py:_post_encode），学生 forward 由
                # use_submodel_func 重定向到内层 language_model。教师必须与
                # 学生走同一条前向语义，这里对 InternVL wrapper 同样打补丁——
                # 否则 InternVLChatModel.forward 的 pixel_values 位置参数必填且
                # img_context_token_id=None 会直接崩。
                if "InternVL" in type(model).__name__ and getattr(model, "language_model", None) is not None:
                    from swift.llm.model.utils import use_submodel_func
                    use_submodel_func(model, "language_model")
        finally:
            _hfds._hf_deepspeed_config_weak_ref = _saved_ref

        model.eval()
        model.requires_grad_(False)  # 冻结，绝不创建 optimizer state
        self._model = model
        return model


# ---------------------------------------------------------------------------
# teacher / student 前向工具
# ---------------------------------------------------------------------------
def _filter_model_inputs(encoded_inputs: Dict[str, Any], model=None) -> Dict[str, Any]:
    """剔除训练侧附加键，只保留模型 forward 入参。

    若传入 model，再按其 forward 签名过滤（除非签名含 **kwargs）——
    InternVL 教师经 use_submodel_func 重定向到 language_model 后不再接受
    pixel_values / image_flags（视觉已拼进 inputs_embeds），签名过滤保证
    这类多余键不会打到 forward。Qwen 教师签名自带 pixel_values，不受影响。
    """
    fwd = {k: v for k, v in encoded_inputs.items() if k not in _NON_MODEL_KEYS}
    if model is not None:
        try:
            import inspect
            params = inspect.signature(model.forward).parameters
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                fwd = {k: v for k, v in fwd.items() if k in params}
        except (TypeError, ValueError):
            pass
    return fwd


@torch.no_grad()
def _internvl_teacher_forward(
    teacher_model, encoded_inputs: Dict[str, Any], img_ctx_id: int
):
    """InternVL teacher 的 native 多模态前向（盲 teacher 修复）。

    语义照抄 modeling_internvl_chat.py:generate 的拼接：
    teacher 自己 extract_feature(pixel_values)（vision+projector 都是 teacher 权重），
    teacher 自己 embed_tokens(input_ids)，在 IMG_CONTEXT 位置原位替换为视觉 embedding，
    然后 language_model(inputs_embeds=...)。

    与旧路径（language_model.forward(input_ids)，视觉缺失）的区别：
    这里 teacher 真正看到图像。Qwen 分支不走这里。
    """
    input_ids = encoded_inputs["input_ids"]
    lm = teacher_model.language_model
    emb = lm.get_input_embeddings()(input_ids)
    pixel_values = encoded_inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=emb.device, dtype=emb.dtype)
        vit = teacher_model.extract_feature(pixel_values)
        B, N, C = emb.shape
        emb = emb.reshape(B * N, C)
        selected = input_ids.reshape(B * N).eq(img_ctx_id)
        n_sel = int(selected.sum().item())
        n_vit = int(vit.reshape(-1, C).shape[0])
        if n_sel != n_vit:
            raise RuntimeError(
                f"[GCEP] InternVL teacher 图像 token 数不一致：IMG_CONTEXT={n_sel} vs vit={n_vit}"
            )
        emb[selected] = vit.reshape(-1, C).to(device=emb.device, dtype=emb.dtype)
        emb = emb.reshape(B, N, C)
    kwargs: Dict[str, Any] = {}
    for key in ("attention_mask", "position_ids"):
        if encoded_inputs.get(key) is not None:
            kwargs[key] = encoded_inputs[key]
    return lm(inputs_embeds=emb, return_dict=True, **kwargs)


def _is_internvl_teacher(model) -> bool:
    """是否 InternVL wrapper（InternVLChatModel + use_submodel_func 后的形态）。"""
    return (
        "InternVL" in type(model).__name__
        and getattr(model, "language_model", None) is not None
        and callable(getattr(model, "extract_feature", None))
    )


def _resolve_img_ctx_id(trainer) -> int:
    """从训练模板 processor 解析 <IMG_CONTEXT> 的 token id（InternVL 用）。"""
    proc = trainer.template.processor
    tok = getattr(proc, "tokenizer", proc)
    return int(tok.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])


@torch.no_grad()
def _teacher_completion_logits(
    teacher_model, encoded_inputs: Dict[str, Any], logits_to_keep: int,
    img_ctx_id: Optional[int] = None,
) -> torch.Tensor:
    """teacher no_grad 前向，返回 completion 段全量 logits [B,ltk,V]（调用方负责释放）。"""
    if _is_internvl_teacher(teacher_model):
        assert img_ctx_id is not None, "InternVL teacher 需要 img_ctx_id"
        outputs = _internvl_teacher_forward(teacher_model, encoded_inputs, img_ctx_id)
    else:
        fwd = _filter_model_inputs(encoded_inputs, teacher_model)
        outputs = teacher_model(**fwd)
    logits = outputs.logits
    # 去掉最后一个 logit（对应 next-token pred），对齐 input_ids[:, -ltk:]
    comp = logits[:, -(logits_to_keep + 1):-1, :]
    return comp


@torch.no_grad()
def teacher_topk_logits(
    teacher_model,
    encoded_inputs: Dict[str, Any],
    logits_to_keep: int,
    topk: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 显存纪律：前向 → 只切 completion 段 → 立即 topk reduce → 释放全量 logits。

    Returns:
        (topk_idx [B,ltk,K] long, topk_logits [B,ltk,K] fp32)
    """
    comp = _teacher_completion_logits(teacher_model, encoded_inputs, logits_to_keep)
    k_eff = min(int(topk), comp.shape[-1])
    topk_logits, topk_idx = comp.float().topk(k_eff, dim=-1)
    del comp  # 立即释放 [B,ltk,V] 全量 logits
    return topk_idx, topk_logits


def student_completion_logits(trainer, model, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
    """student 前向取 completion 段全量 logits [B,ltk,V]（保留 grad，供 OPD RKL 反传）。

    与 grpo_trainer._get_per_token_logps 同一条前向路径（含 runaway 样本软跳过
    guard），唯一差别是不做 selective_log_softmax 归约。GRPO 的 per_token_logps
    由 logps_from_student_logits 从同一 logits 派生，避免第二次 student 前向。
    """
    input_ids = inputs["input_ids"]
    logits_to_keep = inputs["logits_to_keep"]
    safe_max_len = int(os.getenv("LOGP_MAX_SEQ_LEN", "100000"))
    if input_ids.shape[1] > safe_max_len:
        # 与 _get_per_token_logps 相同的软跳过语义：该样本贡献零梯度，训练继续。
        return None
    fwd = _filter_model_inputs(inputs)
    with trainer._template_context(trainer.template):
        logits = model(**fwd).logits
    return logits[:, -(logits_to_keep + 1):-1, :]


def logps_from_student_logits(
    trainer, student_logits: torch.Tensor, inputs: Dict[str, Any]
) -> torch.Tensor:
    """从 completion 段全量 logits 派生 per-token logp（与 _get_per_token_logps 同式：
    先除 temperature 再 selective_log_softmax），保证 C 系 GRPO 项与原实现一致。"""
    from trl.trainer.utils import selective_log_softmax

    logits_to_keep = inputs["logits_to_keep"]
    input_ids = inputs["input_ids"][:, -logits_to_keep:]
    return selective_log_softmax(student_logits / trainer.temperature, input_ids)


# ---------------------------------------------------------------------------
# mixed group 判定（：0 < sum(b) < ng；仅用 vqa_norm 二值）
# ---------------------------------------------------------------------------
def mixed_group_flags(vqa_bin: Sequence[int], num_generations: int) -> List[bool]:
    """逐 group 判定 mixed：0 < 答对数 < num_generations。仅使用 vqa_norm 二值。"""
    ng = int(num_generations)
    flags = []
    for g in range(len(vqa_bin) // ng):
        s = sum(int(v) for v in vqa_bin[g * ng:(g + 1) * ng])
        flags.append(0 < s < ng)
    return flags


# ---------------------------------------------------------------------------
# privileged rollout 组装（：z_i 插在 user task 后、assistant 轨迹前）
# ---------------------------------------------------------------------------
def _append_text_to_first_user(messages: List[dict], text: str) -> None:
    """把证书文本追加到第一条 user 消息尾部（兼容 str / 多模态 list content）。"""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content.append({"type": "text", "text": text})
        else:
            msg["content"] = str(content) + text
        return


def build_privileged_rollout(rollout: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """构造 privileged teacher 前向用的 rollout 副本。

    - certificate 文本追加到第一条 user 消息尾部（即 user task 之后、assistant
      轨迹之前）；证书里的 <support_image .../> 占位替换为 <image>；
    - support images（sandbox 图路径）插入 images 列表原图之后，与消息流中
      <image> 占位出现顺序对齐（原图在最前，assistant 里的 sandbox <image> 在后）。
    任何结构性意外都抛异常，由调用方回退 clean。
    """
    import copy as _copy

    priv = _copy.deepcopy(rollout)
    cert_text = plan["certificate_text"]
    support_paths = list(plan.get("support_image_paths") or [])
    for k, ref in enumerate(support_paths):
        marker = _privilege.SUPPORT_IMAGE_MARKER_TEMPLATE.format(index=k, ref=ref)
        cert_text = cert_text.replace(marker, "<image>")
    messages = [dict(m) for m in priv.get("messages") or []]
    # content 为 list 时深拷一层，避免污染原 rollout
    for m in messages:
        if isinstance(m.get("content"), list):
            m["content"] = [dict(seg) for seg in m["content"]]
    priv["messages"] = messages
    # 空 privilege（无证书文本且无 support 图）不插入任何 wrapper（
    # 否则 "\n\n\n" 会让"空 privilege fallback ≡ clean"在 token 级不成立）。
    if cert_text.strip() or support_paths:
        _append_text_to_first_user(priv["messages"], "\n\n" + cert_text + "\n")
    if support_paths:
        images = list(priv.get("images") or [])
        support_items = [{"bytes": None, "path": p} for p in support_paths]
        # 原图保持 images[0]；support 图紧随其后
        priv["images"] = images[:1] + support_items + images[1:]
    return priv


# ---------------------------------------------------------------------------
# GCEP group 处理管线（ mixed →  Judge →  validate → / anchor
# →  certificate；任一环节失败 -> 该组回退 clean，绝不 crash）
# ---------------------------------------------------------------------------
def _assistant_content(rollout: Dict[str, Any]) -> str:
    for msg in reversed(rollout.get("messages") or []):
        if msg.get("role") == "assistant":
            c = msg.get("content")
            if isinstance(c, list):
                return "".join(seg.get("text", "") for seg in c if isinstance(seg, dict))
            return str(c or "")
    return ""


def _question_text(rollout: Dict[str, Any]) -> str:
    for msg in rollout.get("messages") or []:
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, list):
                return "".join(seg.get("text", "") for seg in c if isinstance(seg, dict))
            return str(c or "")
    return ""


def _solution_text(rollout: Dict[str, Any]) -> str:
    sol = rollout.get("solution", "")
    if isinstance(sol, (list, tuple)) and sol:
        sol = sol[0]
    return str(sol or "")


def _image_paths(rollout: Dict[str, Any]) -> List[str]:
    paths = []
    for img in rollout.get("images") or []:
        if isinstance(img, dict):
            paths.append(img.get("path") or "")
        else:
            paths.append(str(img))
    return paths


def _build_judge_rollout(
    rollout_id: str, rollout: Dict[str, Any], vqa: int
) -> Tuple[Dict[str, Any], List[Any], Dict[str, str]]:
    """把一条 rollout 转成 judge_client.build_judge_messages 需要的结构。

    Returns:
        (judge_rollout, states, image_id_to_path)
    """
    content = _assistant_content(rollout)
    paths = _image_paths(rollout)
    ev_images, states = _evidence.build_evidence_states(rollout_id, content, paths)
    image_id_to_path = {im.image_id: im.path for im in ev_images}
    turns = _evidence.parse_turns(content)
    judge_turns = []
    for turn in turns:
        tid = turn["turn_id"]
        sandbox_images = [
            {"image_id": im.image_id, "path": im.path}
            for im in ev_images
            if im.turn_id == tid and im.path
        ]
        # states[0] = ORIGINAL_STATE；states[tid+1] = 第 tid 次 interaction 之后的 state
        state_after = states[tid + 1] if tid + 1 < len(states) else None
        judge_turns.append(
            {
                "turn_id": tid,
                "reasoning": turn["reasoning"],
                "code": turn["code"],
                "sandbox_stdout": turn["sandbox_stdout"],
                "sandbox_stderr": turn["sandbox_stderr"],
                "sandbox_status": turn["sandbox_status"],
                "sandbox_images": sandbox_images,
                "evidence_state": state_after,
            }
        )
    # GCEP v2 保真修复：v1 把最后一个 sandbox 块之后的尾段
    # （post-tool 最终推理 + 模型真实答案）整体丢掉，answer-only rollout 更是
    # 一个 turn 都没有——judge 看不到模型真实推理/答案，GROUND 诊断构造上不可能。
    # 追加一个 reasoning-only 伪 turn（无 code/sandbox/state），不改变消息序列化结构。
    tail = _evidence.trailing_reasoning(content)
    if tail:
        judge_turns.append(
            {
                "turn_id": len(judge_turns),
                "reasoning": tail,
                "code": "",
                "sandbox_stdout": "",
                "sandbox_stderr": "",
                "sandbox_status": "none",
                "sandbox_images": [],
                "evidence_state": None,
            }
        )
    judge_rollout = {
        "rollout_id": rollout_id,
        "vqa_norm": int(vqa),
        # v1 这里填的是 solution（参考答案）→ judge 误以为每条 rollout 答案都等于
        # reference（"label conflict" 伪影）。改为模型真实答案（绝不回退 solution）。
        "final_answer": _evidence.extract_final_answer(content),
        "turns": judge_turns,
    }
    return judge_rollout, states, image_id_to_path


def _build_group_judge_request(
    group_rollouts: List[Dict[str, Any]],
    group_vqa: List[int],
    rollout_ids: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], Dict[str, int]]:
    """为一个 mixed group 构造 judge 请求参数 + 供后处理复用的元数据。

    Returns:
        (request_kwargs, states_by_rollout, image_id_to_path, vqa_norms)
    """
    judge_rollouts = []
    states_by_rollout: Dict[str, Any] = {}
    image_id_to_path: Dict[str, str] = {}
    vqa_norms: Dict[str, int] = {}
    for rid, ro, v in zip(rollout_ids, group_rollouts, group_vqa):
        jr, states, id2path = _build_judge_rollout(rid, ro, v)
        judge_rollouts.append(jr)
        states_by_rollout[rid] = states
        image_id_to_path.update(id2path)
        vqa_norms[rid] = int(v)
    request = {
        "question": _question_text(group_rollouts[0]),
        "options": None,
        "reference_answer": _solution_text(group_rollouts[0]),
        "original_image_path_or_url": (_image_paths(group_rollouts[0]) or [None])[0],
        "rollouts": judge_rollouts,
    }
    return request, states_by_rollout, image_id_to_path, vqa_norms


def _postprocess_group_judge(
    obj: Optional[Dict[str, Any]],
    states_by_rollout: Dict[str, Any],
    image_id_to_path: Dict[str, str],
    vqa_norms: Dict[str, int],
    rollout_ids: List[str],
    n: int,
    group_type: Any = GCEPGroupType.MIXED,
) -> Tuple[List[Optional[Dict[str, Any]]], Dict[str, Any]]:
    """对单个 group 的 judge 结果做 validate → anchor → certificate。

    任一环节失败返回全 None（该组回退 clean）。绝不抛异常。

    GCEP v2：group_type 决定 validator 语义；reconcile 仅对 MIXED 启用
    （ALL_CORRECT 严格 stage=CORRECT+discrepancy=""，自动修只会掩盖 judge 错误）。
    """
    if not isinstance(group_type, GCEPGroupType):
        group_type = GCEPGroupType(str(group_type))
    plans: List[Optional[Dict[str, Any]]] = [None] * n
    stats: Dict[str, Any] = {}
    try:
        stats["judged_groups"] = 1
        if obj is None:
            return plans, stats

        # validate 之前先用权威 vqa_norm 校正 judge 的 stage 归因
        # （check8/9/10/11 语义冲突）。离线归因显示 check9（vqa_norm=0 但 judge
        # 判 CORRECT）是 judge_valid 失败的最大头；vqa_norm 信任级高于 judge，
        # 用它在 validate 前置修正 stage 可显著抬高 judge_valid_ratio，且不改
        # 方法定义（只修判签，不放宽校验）。修正条数计入 stats 便于观察。
        # GCEP v2：reconcile 仅 MIXED 启用（见 docstring）。
        if group_type == GCEPGroupType.MIXED:
            n_reconciled = _validator.reconcile_diagnoses_with_vqa(
                obj, states_by_rollout, vqa_norms
            )
            if n_reconciled:
                stats["stage_reconciled"] = n_reconciled

        ok, reason = _validator.validate_judge_output(
            obj, group_type, states_by_rollout, vqa_norms
        )
        if not ok:
            stats["judge_invalid"] = 0 if reason == "GCEP_NOT_APPLICABLE" else 1
            stats["judge_not_applicable"] = 1 if reason == "GCEP_NOT_APPLICABLE" else 0
            return plans, stats
        stats["judge_valid_groups"] = 1

        diagnoses = {d["rollout_id"]: d for d in obj.get("rollout_diagnoses") or []}
        for d in diagnoses.values():
            stage = str(d.get("stage", "")).lower()
            if stage in ("correct", "acquire", "read", "ground", "other"):
                stats[f"diagnosis_{stage}"] = stats.get(f"diagnosis_{stage}", 0) + 1

        entries = _anchor.build_sufficient_entries(obj, states_by_rollout)
        group_gold = _anchor.select_group_gold(entries, vqa_norms)
        if group_gold is None:
            return plans, stats

        reading = obj["reading"]
        grounding_rule = obj["grounding"]["rule"]
        for idx, rid in enumerate(rollout_ids):
            anchor_entry = _anchor.select_rollout_anchor(rid, entries, group_gold)
            if anchor_entry is None:
                continue
            own = anchor_entry.get("rollout_id") == rid
            stats["own_anchor"] = stats.get("own_anchor", 0) + (1 if own else 0)
            stats["group_gold_fallback"] = stats.get("group_gold_fallback", 0) + (0 if own else 1)
            acq_mode = _anchor.acquisition_mode(anchor_entry)
            support_paths = [
                image_id_to_path[iid]
                for iid in anchor_entry.get("support_image_ids") or []
                if iid in image_id_to_path and image_id_to_path[iid]
            ]
            diag = diagnoses.get(rid, {})
            cert_text, support_refs = _privilege.build_privilege_certificate(
                query_slot=reading["query_slot"],
                anchor_state_id=anchor_entry["state_id"],
                acq_mode=acq_mode,
                observed_value=reading["observed_value"],
                visual_fact=reading["visual_fact"],
                grounding_rule=grounding_rule,
                diagnosis_stage=str(diag.get("stage", "OTHER")),
                discrepancy=str(diag.get("discrepancy", "")),
                support_images=support_paths,
            )
            plans[idx] = {
                "certificate_text": cert_text,
                "support_image_paths": list(support_refs),
                "anchor_state_id": anchor_entry["state_id"],
                "acq_mode": acq_mode,
                "used_own_anchor": bool(own),
                # GCEP v2：归档/统计用 metadata（不影响 teacher 前向逻辑）
                "privilege_type": "full_certificate",
                "group_type": group_type.value,
                "diagnosis_stage": str(diag.get("stage", "OTHER")),
                "discrepancy": str(diag.get("discrepancy", "")),
            }
    except Exception:
        return [None] * n, stats
    return plans, stats


def _postprocess_all_wrong(
    obj: Optional[Dict[str, Any]],
    states_by_rollout: Dict[str, Any],
    vqa_norms: Dict[str, int],
    rollout_ids: List[str],
    n: int,
) -> Tuple[List[Optional[Dict[str, Any]]], Dict[str, Any]]:
    """ALL_WRONG group（GCEP v2 IDEA-1）：validate → 逐 rollout diagnosis-only privilege。

    与 mixed/all-correct 的最大区别：不建 group gold、不选 anchor、不构造 full
    certificate；teacher privilege 只携带该 rollout 的（stage, discrepancy）。
    严格校验：READ/GROUND 无 own sufficient → 整组 invalid → clean fallback
    （不自动修成 ACQUIRE）。绝不抛异常。
    """
    plans: List[Optional[Dict[str, Any]]] = [None] * n
    stats: Dict[str, Any] = {}
    try:
        stats["judged_groups"] = 1
        if obj is None:
            return plans, stats

        ok, reason = _validator.validate_judge_output(
            obj, GCEPGroupType.ALL_WRONG, states_by_rollout, vqa_norms
        )
        if not ok:
            stats["judge_invalid"] = 0 if reason == "GCEP_NOT_APPLICABLE" else 1
            stats["judge_not_applicable"] = 1 if reason == "GCEP_NOT_APPLICABLE" else 0
            return plans, stats
        stats["judge_valid_groups"] = 1

        diagnoses = {d["rollout_id"]: d for d in obj.get("rollout_diagnoses") or []}
        for d in diagnoses.values():
            stage = str(d.get("stage", "")).lower()
            if stage in ("correct", "acquire", "read", "ground", "other"):
                stats[f"diagnosis_{stage}"] = stats.get(f"diagnosis_{stage}", 0) + 1

        for idx, rid in enumerate(rollout_ids):
            diag = diagnoses.get(rid)
            if diag is None:
                continue
            stage = str(diag.get("stage", "OTHER"))
            discrepancy = str(diag.get("discrepancy", ""))
            plans[idx] = {
                "certificate_text": _privilege.build_diagnosis_only_privilege(
                    stage, discrepancy
                ),
                "support_image_paths": [],
                "privilege_type": "diagnosis_only",
                "group_type": GCEPGroupType.ALL_WRONG.value,
                "diagnosis_stage": stage,
                "discrepancy": discrepancy,
            }
    except Exception:
        return [None] * n, stats
    return plans, stats


def _build_gt_privilege_plans(
    all_inputs: Sequence[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    """GT-only privilege：逐 rollout 用 GT（solution）直接构造 teacher plan。

    - 无 Judge / Validator / anchor / certificate（不依赖任何外部服务）；
    - plan 结构与 certificate plan 共用键约定（``certificate_text`` 必填、
      ``support_image_paths`` 恒为空），因此 ``build_privileged_rollout`` 与
      ``gcep_build_teacher_targets`` 零改动即可消费；
    - solution 缺失（空串）的行 plan=None：该行在 teacher 前向时回退 clean
      （ 同款软降级），不影响其他行。

    纯函数、确定性：所有 rank 从各自 gather 到的同一份 all_inputs 计算，
    结果一致，无需 broadcast。
    """
    plans: List[Optional[Dict[str, Any]]] = []
    for rollout in all_inputs:
        solution = _solution_text(rollout)
        if solution.strip():
            plans.append(
                {
                    "certificate_text": _privilege.build_gt_only_privilege(solution),
                    "support_image_paths": [],
                    "privilege_type": "gt_only",
                    "group_type": None,
                }
            )
        else:
            plans.append(None)
    return plans


def compute_group_privilege_plans(
    all_inputs: List[Dict[str, Any]],
    vqa_bin: Sequence[int],
    num_generations: int,
    config: GCEPConfig,
    is_main: bool,
    logger=None,
) -> Tuple[List[Optional[Dict[str, Any]]], Dict[str, float], Dict[str, Any]]:
    """全 batch 的 privilege plan 计算入口（main rank 计算 + broadcast 到各 rank）。

    Returns:
        (plans, stats, group_meta)。
        - plans 与 all_inputs 等长（None = clean-teacher fallback）；
        - stats 为 availability 指标（mixed/all_correct/all_wrong group ratio、
          per-type judge_valid_ratio、full_certificate/diagnosis_only/
          clean_fallback rollout ratio 等，GCEP v2 ）；
        - group_meta = {"judge_records": [...], "rollout_group_types": [...]}，
          供 grpo_trainer 写 group 级 judge sidecar 与 per-rollout 归档字段。

    GCEP v2：use_all_group_types（A1v2）时三类 group 全部判 Judge；
    否则保持 v1 的 mixed-only 路径（A1/A2 行为不变，ALL_CORRECT/ALL_WRONG
    连 judge 都不调用）。
    """
    n = len(all_inputs)
    plans: List[Optional[Dict[str, Any]]] = [None] * n
    ng = int(num_generations)
    n_groups = n // ng if ng > 0 else 0

    # --- group 分类（仅 vqa_norm 二值；绝不用 reward/advantage） ---
    # 所有模式都做全分类（stats/归档口径跨 A1/A1v2/A2v2 一致）；
    # 模式只决定哪些 group 被**判 Judge**：A1v2 三类全判；legacy（A1/A2）仅 MIXED。
    if n_groups > 0:
        gtypes: List[Optional[GCEPGroupType]] = list(
            _gtypes.classify_group_types(vqa_bin, ng)
        )
    else:
        gtypes = []
    if config.use_all_group_types:
        judged_mask = [gt is not None for gt in gtypes]
    else:
        judged_mask = [gt == GCEPGroupType.MIXED for gt in gtypes]

    n_by_type = {
        t: sum(1 for gt in gtypes if gt == t)
        for t in (GCEPGroupType.ALL_CORRECT, GCEPGroupType.MIXED, GCEPGroupType.ALL_WRONG)
    }
    n_judgeable = sum(judged_mask)
    # per-rollout group_type（所有 rank 本地可算，vqa_bin 是全局量）
    rollout_group_types: List[Optional[str]] = []
    for gt in gtypes:
        rollout_group_types.extend([gt.value if gt is not None else None] * ng)
    group_meta: Dict[str, Any] = {
        "judge_records": [],
        "rollout_group_types": rollout_group_types,
    }

    # --- stats 零初始化（所有 rank 同 key，broadcast 后 wandb key 稳定） ---
    stats: Dict[str, float] = {
        "all_correct_group_ratio": (n_by_type[GCEPGroupType.ALL_CORRECT] / n_groups) if n_groups else 0.0,
        "mixed_group_ratio": (n_by_type[GCEPGroupType.MIXED] / n_groups) if n_groups else 0.0,
        "all_wrong_group_ratio": (n_by_type[GCEPGroupType.ALL_WRONG] / n_groups) if n_groups else 0.0,
        "judge_valid_ratio": 0.0,
        "judge_applicable_ratio": 0.0,
        "all_correct_judge_valid_ratio": 0.0,
        "mixed_judge_valid_ratio": 0.0,
        "all_wrong_judge_valid_ratio": 0.0,
        "own_anchor_ratio": 0.0,
        "group_gold_fallback_ratio": 0.0,
        "stage_reconciled_count": 0.0,
        "judge_latency_sec": 0.0,
        "full_certificate_rollout_ratio": 0.0,
        "diagnosis_only_rollout_ratio": 0.0,
    }
    for stage in ("correct", "acquire", "read", "ground", "other"):
        stats[f"diagnosis_{stage}_count"] = 0.0

    need_judge = config.use_privilege and n_judgeable > 0 and not config.is_gt_privilege_mode
    # GT-only privilege：不调 Judge，plan 逐 rollout 由 GT 构造；
    # 其余 stats/group 统计口径不变（group 分类仅遥测，不参与 plan）。
    if config.is_gt_privilege_mode:
        plans = _build_gt_privilege_plans(all_inputs)
        # 静默失败保险：整个 batch 都没有 solution 时 GT privilege 会退化成
        # clean OPD（等价 A0），此时必须留痕（wandb 上表现为
        # gcep/privilege_rollout_ratio=0）。
        if logger is not None and all(p is None for p in plans):
            logger.warning(
                "[GCEP] GT privilege: 本 batch 全部 rollout 缺 solution，"
                "privilege 为空（该步等价 clean OPD）"
            )
    if need_judge:
        if is_main:
            agg: Dict[str, float] = {}
            judge_records: List[Dict[str, Any]] = []
            try:
                # Concurrent judge requests reduce the time other ranks spend
                # waiting for broadcast. Post-processing remains local per group.
                group_meta_list = []  # (group_index, group_type, slice, rollout_ids, states_by, id2path, vqa_norms)
                judge_requests = []
                for g, gt in enumerate(gtypes):
                    if not judged_mask[g]:
                        continue
                    sl = slice(g * ng, (g + 1) * ng)
                    # GCEP_TRAJ_FILTER（gcep/traj_filter.py）：含无效轨迹的 group
                    # 整组不调 judge（plans[sl] 保持 None → 组内有效行自动回退
                    # clean OPD；无效行在 gcep_opd_loss 聚合时被剔除）。默认关。
                    if any(all_inputs[k].get('traj_invalid') for k in range(sl.start, sl.stop)):
                        agg["traj_judge_skipped_groups"] = agg.get("traj_judge_skipped_groups", 0.0) + 1.0
                        continue
                    rids = [f"R{g * ng + i}" for i in range(ng)]
                    request, states_by, id2path, vqa_norms = _build_group_judge_request(
                        all_inputs[sl], [int(v) for v in vqa_bin[sl]], rids
                    )
                    request["group_type"] = gt.value  # GCEP v2：dispatcher 选 prompt
                    judge_requests.append(request)
                    group_meta_list.append((g, gt, sl, rids, states_by, id2path, vqa_norms))

                # 并发 fire 所有 judge 调用（每组一次多模态请求）
                t0 = time.time()
                judge_objs = _judge.judge_groups(
                    judge_requests,
                    concurrency=config.judge_concurrency,
                ) if judge_requests else []
                judge_latency = time.time() - t0

                # 逐组 post-process（本地纯计算；按 group_type 分派）
                for (g, gt, sl, rids, states_by, id2path, vqa_norms), obj in zip(group_meta_list, judge_objs):
                    if gt == GCEPGroupType.ALL_WRONG:
                        group_plans, gstats = _postprocess_all_wrong(
                            obj, states_by, vqa_norms, rids, ng
                        )
                    else:
                        group_plans, gstats = _postprocess_group_judge(
                            obj, states_by, id2path, vqa_norms, rids, ng,
                            group_type=gt,
                        )
                    plans[sl] = group_plans
                    for k, v in gstats.items():
                        agg[k] = agg.get(k, 0.0) + float(v)
                    agg[f"{gt.value}_judged_groups"] = agg.get(f"{gt.value}_judged_groups", 0.0) + 1.0
                    agg[f"{gt.value}_judge_valid_groups"] = (
                        agg.get(f"{gt.value}_judge_valid_groups", 0.0)
                        + float(gstats.get("judge_valid_groups", 0))
                    )
                    judge_records.append(
                        {
                            "group_index": g,
                            "group_type": gt.value,
                            "judge_valid": bool(gstats.get("judge_valid_groups", 0)),
                            "judge_not_applicable": bool(gstats.get("judge_not_applicable", 0)),
                            "judge_invalid": bool(gstats.get("judge_invalid", 0)),
                            "stage_reconciled": int(gstats.get("stage_reconciled", 0)),
                            "num_privileged_rollouts": sum(1 for p in group_plans if p is not None),
                            "judge_output": obj,
                        }
                    )
                if judge_requests:
                    agg["judge_latency_sec"] = agg.get("judge_latency_sec", 0.0) + judge_latency / len(judge_requests)
            except Exception as exc:  # 双保险：任何意外整批回退 clean
                if logger is not None:
                    logger.warning(f"[GCEP] group pipeline failed, fallback all clean: {exc}")
                plans = [None] * n
                agg = {}
                judge_records = []
            # rollout 级 availability 比率（ / v2 ）
            judged = agg.get("judged_groups", 0.0)
            stats["judge_valid_ratio"] = (
                agg.get("judge_valid_groups", 0.0) / judged if judged else 0.0
            )
            stats["judge_applicable_ratio"] = (
                1.0 - agg.get("judge_not_applicable", 0.0) / judged if judged else 0.0
            )
            for t in (GCEPGroupType.ALL_CORRECT, GCEPGroupType.MIXED, GCEPGroupType.ALL_WRONG):
                t_judged = agg.get(f"{t.value}_judged_groups", 0.0)
                stats[f"{t.value}_judge_valid_ratio"] = (
                    agg.get(f"{t.value}_judge_valid_groups", 0.0) / t_judged if t_judged else 0.0
                )
            stats["own_anchor_ratio"] = (
                agg.get("own_anchor", 0.0) / n_judgeable / ng if n_judgeable else 0.0
            )
            stats["group_gold_fallback_ratio"] = (
                agg.get("group_gold_fallback", 0.0) / n_judgeable / ng if n_judgeable else 0.0
            )
            for stage in ("correct", "acquire", "read", "ground", "other"):
                stats[f"diagnosis_{stage}_count"] = agg.get(f"diagnosis_{stage}", 0.0)
            stats["stage_reconciled_count"] = agg.get("stage_reconciled", 0.0)
            if agg.get("judge_latency_sec"):
                stats["judge_latency_sec"] = agg["judge_latency_sec"] / max(judged, 1.0)
            group_meta["judge_records"] = judge_records
        # broadcast plans/stats/group_meta 到全部 rank（单进程 / 未初始化分布式时跳过）
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from accelerate.utils import broadcast_object_list

                payload = [plans, stats, group_meta]
                broadcast_object_list(payload)
                plans, stats, group_meta = payload
        except Exception as exc:
            if logger is not None:
                logger.warning(f"[GCEP] broadcast failed, fallback all clean: {exc}")
            plans, stats = [None] * n, {}
            group_meta = {"judge_records": [], "rollout_group_types": rollout_group_types}

    n_priv = sum(1 for p in plans if p is not None)
    n_full = sum(1 for p in plans if p is not None and p.get("privilege_type") == "full_certificate")
    n_diag = sum(1 for p in plans if p is not None and p.get("privilege_type") == "diagnosis_only")
    stats["clean_fallback_ratio"] = 1.0 - (n_priv / n) if n else 0.0
    stats["privilege_rollout_ratio"] = (n_priv / n) if n else 0.0
    stats["full_certificate_rollout_ratio"] = (n_full / n) if n else 0.0
    stats["diagnosis_only_rollout_ratio"] = (n_diag / n) if n else 0.0
    return plans, stats, group_meta


@torch.no_grad()
def _sampled_token_logprobs(
    logits_row: torch.Tensor, token_ids: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    """[ltk,V] 单行 logits 在采样 token 处的 logprob（A2v2 impact 用）。

    chunked fp32 logsumexp − gathered logit；绝不整张 fp32 拷贝（ 显存纪律）。
    返回 [ltk] fp32 detached。
    """
    T = logits_row.shape[0]
    out = torch.empty(T, dtype=torch.float32, device=logits_row.device)
    for s in range(0, T, int(chunk)):
        sl = logits_row[s:s + chunk].float()
        lse = torch.logsumexp(sl, dim=-1)
        gathered = sl.gather(-1, token_ids[s:s + chunk].unsqueeze(-1)).squeeze(-1)
        out[s:s + chunk] = gathered - lse
        del sl, lse, gathered
    return out.detach()


# ---------------------------------------------------------------------------
# teacher 前向组装：对一个 gas chunk 生成 Top32 目标（+ PS 模式的 union support）
# ---------------------------------------------------------------------------
def gcep_build_teacher_targets(
    trainer,
    teacher_model,
    batch: List[Dict[str, Any]],
    batch_encoded_inputs: Dict[str, Any],
    plans: Sequence[Optional[Dict[str, Any]]],
    config: GCEPConfig,
    traj_invalid: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """对一个 micro-batch 计算 teacher Top32 目标，存入 batch_encoded_inputs['gcep_teacher']。

    - clean 目标：直接用 student 的原始 encoded batch 做 teacher 前向；
    - privileged 行：certificate 插入 user task 后重新 encode 做 teacher 前向，
      completion token 逐位对齐（复用 RLSD 的 assert+软跳过思路），不齐则该行回退 clean；
    - PS 模式：对每行构建 union support（Top32(priv)∪Top32(clean)，≤64），
      两分布各自 gather 到 union 上；无 privilege 的行 priv≡clean（jsd≈0、w≈1）。
    任何异常 -> 受影响行回退 clean，绝不 crash。
    """
    if config.is_mt_mode:
        # MT baseline（vision_opd_mt / vzero_mt / vad_mt）：独立模块，既有路径不动
        try:
            from . import mt_teacher as _mt
        except ImportError:
            import mt_teacher as _mt
        return _mt.mt_build_teacher_targets(
            trainer, teacher_model, batch, batch_encoded_inputs, plans, config,
            traj_invalid=traj_invalid)
    from swift.llm import to_device  # 惰性（训练侧必有 swift）

    t0 = time.time()
    ltk = int(batch_encoded_inputs["logits_to_keep"])
    student_ids = batch_encoded_inputs["input_ids"][:, -ltk:]
    bs = student_ids.shape[0]
    device = student_ids.device
    topk = int(config.topk)

    result: Dict[str, Any] = {
        "used_privilege": [False] * bs,
        "mismatch": 0,
        "teacher_forward_sec": 0.0,
        # A2v2：per-row sampled-token |ΔlogP|（privileged 且 token 对齐的行才有值）
        "impact_rows": [None] * bs,
        "student_token_ids": student_ids,  # 引用，无拷贝（sidecar 归档用）
        # GCEP_TRAJ_FILTER：per-row 无效标记（gcep/traj_filter.py；None=未启用，
        # gcep_opd_loss 聚合时仅消费非 None 的标记，既有路径不受影响）
        "traj_invalid": (list(traj_invalid) if traj_invalid is not None else None),
        "row_meta": [
            {
                "privilege_type": (plans[j].get("privilege_type") if j < len(plans) and plans[j] else None),
                "group_type": (plans[j].get("group_type") if j < len(plans) and plans[j] else None),
            }
            for j in range(bs)
        ],
    }
    try:
        clean_full = _teacher_completion_logits(
            teacher_model, batch_encoded_inputs, ltk,
            img_ctx_id=_resolve_img_ctx_id(trainer))
        clean_top_logits, clean_idx = clean_full.float().topk(min(topk, clean_full.shape[-1]), dim=-1)
    except Exception as exc:
        # teacher clean 前向失败是硬错误之外的唯一不可降级情形：
        # OPD 完全依赖 teacher，此处失败只能抛出让训练停下来检查。
        raise RuntimeError(f"[GCEP] teacher clean forward failed: {exc}") from exc

    target_idx = clean_idx.clone()
    target_logits = clean_top_logits.clone()
    # PS 模式每行都要 union 三元组；无 privilege 行 priv≡clean（自动退化）
    union_idx_rows: List[Optional[torch.Tensor]] = [None] * bs
    union_priv_rows: List[Optional[torch.Tensor]] = [None] * bs
    union_clean_rows: List[Optional[torch.Tensor]] = [None] * bs

    priv_rows = [j for j in range(bs) if j < len(plans) and plans[j] is not None]
    priv_full = None
    priv_encoded = None
    priv_top_logits = priv_idx = None
    if config.use_privilege and priv_rows:
        try:
            priv_rollouts = [build_privileged_rollout(batch[j], plans[j]) for j in priv_rows]
            with trainer._template_context(trainer.template):
                enc = [trainer.template.encode(r) for r in priv_rollouts]
                priv_encoded = to_device(trainer.template.data_collator(enc), trainer.model.device)
            priv_labels = priv_encoded.pop("labels")
            ltk_p = int(
                (priv_labels.shape[-1] - torch.ne(priv_labels, -100).int().argmax(-1)).max().item()
            )
            priv_full = _teacher_completion_logits(
                teacher_model, priv_encoded, ltk_p,
                img_ctx_id=_resolve_img_ctx_id(trainer))
            priv_top_logits, priv_idx = priv_full.float().topk(
                min(topk, priv_full.shape[-1]), dim=-1
            )
            priv_ids = priv_encoded["input_ids"][:, -ltk_p:]
            for k, j in enumerate(priv_rows):
                # token 对齐：teacher(priv) completion 末尾 ltk 个 token 必须与 student
                # 完全一致；否则该行记 mismatch 并回退 clean（RLSD 同款软跳过）。
                if ltk_p != ltk or not torch.equal(priv_ids[k, -ltk:], student_ids[j]):
                    result["mismatch"] += 1
                    continue
                target_idx[j] = priv_idx[k, -ltk:]
                target_logits[j] = priv_top_logits[k, -ltk:]
                result["used_privilege"][j] = True
                # A2v2：sampled-token |ΔlogP| impact（在 del clean_full/priv_full
                # 之前、per-row 切片计算；mismatch/失败行保持 None → A1 plain mean）
                if config.use_impact_weight:
                    try:
                        lp = _sampled_token_logprobs(priv_full[k, -ltk:], student_ids[j])
                        lc = _sampled_token_logprobs(clean_full[j], student_ids[j])
                        result["impact_rows"][j] = (lp - lc).abs()
                        del lp, lc
                    except Exception:
                        result["impact_rows"][j] = None
        except Exception:
            # privilege 前向任何失败：全部回退 clean，不 crash
            result["mismatch"] += len(priv_rows)
            priv_full = None

    if config.use_ps_weight:
        for j in range(bs):
            try:
                if result["used_privilege"][j] and priv_full is not None:
                    k = priv_rows.index(j)
                    p_idx = priv_idx[k, -ltk:]  # [ltk,K]
                    p_full = priv_full[k, -ltk:]  # [ltk,V]
                else:
                    # 无 privilege 行：priv≡clean（jsd≈0、w≈1，自动退化 vanilla OPD）
                    p_idx = clean_idx[j]
                    p_full = clean_full[j]
                c_idx = clean_idx[j]
                c_full = clean_full[j]
                # union = Top32(priv) ∪ Top32(clean)，逐 token 去重（|U|≤64）
                u_idx, u_valid = _dedup_union(p_idx, c_idx)  # [ltk,U]
                neg = torch.finfo(torch.float32).min  # pad 槽位置 -inf（softmax 后为 0）
                union_idx_rows[j] = u_idx
                priv_u = p_full.gather(-1, u_idx).float()
                clean_u = c_full.gather(-1, u_idx).float()
                priv_u = torch.where(u_valid, priv_u, torch.full_like(priv_u, neg))
                clean_u = torch.where(u_valid, clean_u, torch.full_like(clean_u, neg))
                union_priv_rows[j] = priv_u
                union_clean_rows[j] = clean_u
            except Exception:
                # 单行失败：priv≡clean 退化（w≈1）
                union_idx_rows[j] = clean_idx[j]
                union_clean_rows[j] = clean_full[j].gather(-1, clean_idx[j]).float()
                union_priv_rows[j] = union_clean_rows[j]

    del clean_full
    if priv_full is not None:
        del priv_full

    result["target_idx"] = target_idx
    result["target_logits"] = target_logits
    result["union_idx"] = union_idx_rows
    result["union_priv_logits"] = union_priv_rows
    result["union_clean_logits"] = union_clean_rows
    result["teacher_forward_sec"] = time.time() - t0
    return result


def _dedup_union(p_idx: torch.Tensor, c_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top32(priv) ∪ Top32(clean) 的逐 token 去重 union 索引（|U|≤64）。

    p_idx/c_idx: [T,K]。返回 (u_idx [T,U], u_valid [T,U])：U 取各 token union
    大小的最大值，不足处以 index 0 右侧 pad 且 valid=False；调用方 gather 后
    把 pad 槽位置 -inf，使其在 softmax 中概率为 0（不影响归一化正确性）。
    """
    T, K = p_idx.shape
    rows = []
    max_u = 0
    valid_rows = []
    for t in range(T):
        seen = set()
        ids = []
        for v in p_idx[t].tolist():
            ids.append(v)
            seen.add(v)
        valid = [True] * len(ids)
        for v in c_idx[t].tolist():
            if v not in seen:
                ids.append(v)
                seen.add(v)
                valid.append(True)
        max_u = max(max_u, len(ids))
        rows.append(ids)
        valid_rows.append(valid)
    # pad 到等长（pad 槽位 valid=False）
    u_idx = torch.full((T, max_u), 0, dtype=p_idx.dtype, device=p_idx.device)
    u_valid = torch.zeros((T, max_u), dtype=torch.bool, device=p_idx.device)
    for t, (ids, valid) in enumerate(zip(rows, valid_rows)):
        u_idx[t, : len(ids)] = torch.tensor(ids, dtype=p_idx.dtype, device=p_idx.device)
        u_valid[t, : len(valid)] = torch.tensor(valid, dtype=torch.bool, device=p_idx.device)
    return u_idx, u_valid


# ---------------------------------------------------------------------------
# OPD loss 组合（ RKL +  聚合 +  PS 加权 +  分别归一化后相加）
# ---------------------------------------------------------------------------
def _zero_loss_with_graph(*candidates, device) -> torch.Tensor:
    """返回数值为零、尽可能连接输入计算图的标量 loss。

    DeepSpeed backward 需要单元素且带 grad_fn 的 loss。对仍连接计算图的输入
    使用 t.sum() * 0.0，保留梯度路径；若没有这样的输入，则返回普通零张量，
    调用方仍需处理不存在梯度路径的情况。"""
    for t in candidates:
        if isinstance(t, torch.Tensor) and t.numel() > 0 and (
                t.requires_grad or t.grad_fn is not None):
            return t.sum() * 0.0
    return torch.zeros((), device=device)


def _divprobe_enabled() -> bool:
    """GCEP_DIVPROBE=1 时记录加权和门控前的教师与学生分布差异。
    默认关闭，不进行额外探针计算。"""
    return os.getenv("GCEP_DIVPROBE", "0") == "1"


def _opd_include_invalid() -> bool:
    """GCEP_OPD_INCLUDE_INVALID=1 时，invalid 行恢复参与 OPD 聚合。

    该开关只影响 OPD 聚合侧；judge 侧整组跳过与整组 reward 清零保持不变。
    污染组的 plans=None，因此恢复的行使用 clean-teacher 目标。teacher 前向
    已在全批计算，这个开关不额外运行 teacher。vzero_mt 不适用，其教师损失
    仍排除 invalid 行。默认关闭，维持过滤行为。"""
    return os.getenv("GCEP_OPD_INCLUDE_INVALID", "0") == "1"


# A5_random_select20pp 用的惰性全局 RNG（仅消融臂启用时创建）。
# 语义：每行抽 randperm 决定 high/low 桶，而非按 impact 排序。用同一个 generator 的
# 顺序抽取（seed 固定），保证"同一 seed 下逐值可复现"；不同秩/数据顺序下序列自然不同，
# 但对"随机 ≠ impact"这一消融命题无影响。
_gi_abl_rng = None


def _impact_ablation_rng(config: "GCEPConfig"):
    global _gi_abl_rng
    if _gi_abl_rng is None:
        _gi_abl_rng = torch.Generator().manual_seed(int(config.impact_ablation_seed))
    return _gi_abl_rng


def _divp_quantiles(metrics: Dict[str, float], prefix: str, vals: List[torch.Tensor]) -> None:
    """concat 后写 mean/p50/p90/p99 到 metrics（调用方会再冠 gcep/ 前缀）。"""
    if not vals:
        return
    t = torch.cat(vals)
    metrics[f"{prefix}_mean"] = float(t.mean())
    for q, n in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
        metrics[f"{prefix}_{n}"] = float(torch.quantile(t, q))


def gcep_opd_loss(
    student_logits: Optional[torch.Tensor],
    gcep_teacher: Dict[str, Any],
    completion_mask: torch.Tensor,
    config: GCEPConfig,
    per_token_logps: Optional[torch.Tensor] = None,
    old_per_token_logps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算 OPD loss（A/B 系主 loss / C 系辅助 loss）。

    - target = privileged Top32（该行 GCEP valid）或 clean Top32（ fallback）；
    - PS 模式：rkl 乘  的 mean-preserving token 权重；
    - A2v2（use_impact_weight）：privileged 行按 sampled-token |ΔlogP| top20%
      分 High/Low 两组，L_i = 0.5·mean(rkl[High]) + 0.5·mean(rkl[Low])；
      无 impact 行（clean/mismatch）保持 A1 plain mean；
    - 聚合严格按 ：先每条 rollout 在 mask 内取均值，再对 batch 取均值，
      禁止 flatten 全 batch token 求和。

    per_token_logps / old_per_token_logps 仅 MT baseline（is_mt_mode）消费
    （is_clip 与 vzero k1/PPO），既有模式传入与否均无影响。

    Returns:
        (loss scalar, metrics dict)
    """
    if config.is_mt_mode:
        # MT baseline：独立 loss 模块（官方公式移植），既有路径不动
        try:
            from . import mt_teacher as _mt
        except ImportError:
            import mt_teacher as _mt
        return _mt.mt_opd_loss(
            student_logits, gcep_teacher, completion_mask, config,
            per_token_logps=per_token_logps, old_per_token_logps=old_per_token_logps)
    device = completion_mask.device
    metrics: Dict[str, float] = {}
    if student_logits is None or gcep_teacher is None:
        # runaway 样本软跳过 / teacher 目标缺失：零 loss（不影响训练继续）。
        # 修复：零 loss 必须带 grad_fn（deepspeed maybe_loss_for_backward
        # 要求），否则 backward 被 assert 拒绝、整棵树被杀（bs=1 微批下高频触发）。
        return _zero_loss_with_graph(per_token_logps, student_logits, device), metrics

    bs = student_logits.shape[0]
    impact_rows = gcep_teacher.get("impact_rows") or [None] * bs
    # GCEP_TRAJ_FILTER：无效行不进 OPD 聚合（L = Σ e_i L_i / Σ e_i，gcep/traj_filter.py；
    # 标记为 None = 未启用，全部行参与，行为与既有逐字节等价）。
    _traj_inv = gcep_teacher.get("traj_invalid")
    _n_invalid = 0
    _n_excluded = 0
    _inc_inv = _opd_include_invalid()
    per_rollout_losses = []
    jsd_vals = []
    impact_valid_vals = []   # 有效 token 的 impact（跨行 concat，算分位数）
    mean_high_vals = []
    mean_low_vals = []
    rkl_high_vals = []
    rkl_low_vals = []
    sidecar_rows = []        # A2v2 归档（token_ids/impact/high_mask/rkl/mask）
    # GCEP_DIVPROBE（_divprobe_enabled，默认关）：raw 教师-学生差异指标累积
    _DIVP = _divprobe_enabled()
    _dp_rkl, _dp_hit, _dp_mass = [], [], []
    for j in range(bs):
        if _traj_inv is not None and j < len(_traj_inv) and _traj_inv[j]:
            _n_invalid += 1
            if not _inc_inv:
                # 默认语义：invalid 行剔出 OPD 聚合
                _n_excluded += 1
                continue
            # OPD-noFilter（GCEP_OPD_INCLUDE_INVALID=1）：invalid 行照常参与
            # 聚合——它们在污染组 plans=None 下只拿 clean-teacher 目标（抑制方向）
        mask_j = completion_mask[j:j + 1].to(student_logits.dtype)  # [1,ltk]
        t_idx = gcep_teacher["target_idx"][j:j + 1].to(device)  # [1,ltk,K]
        t_log = gcep_teacher["target_logits"][j:j + 1].to(device)  # [1,ltk,K]
        s_sup = student_logits[j:j + 1].gather(-1, t_idx)  # student 可导
        # 两侧都在同一个 Top32 support 上：k=K 时 topk_rkl_per_token 内部的
        # topk 只是置换，support 内 fp32 重归一化即 / 的定义。
        rkl = _opd.topk_rkl_per_token(s_sup, t_log, mask_j, k=t_log.shape[-1])  # [1,ltk]
        if _DIVP:
            # raw 指标（都在 impact/ps 加权之前）：
            #   div_rkl_*           per-token RKL 分布
            #   div_topk_hit_rate   学生采样 token 落入教师 Top32 的比例
            #   div_student_mass_on_topk  学生概率质量集中在教师 Top32 的比例
            _m = completion_mask[j].bool()
            _dp_rkl.append(rkl[0][_m].detach().float())
            _sid = gcep_teacher["student_token_ids"][j].to(device)  # [ltk]
            _hit = (_sid.unsqueeze(-1) == t_idx[0]).any(-1)  # [ltk]
            _dp_hit.append(_hit[_m].detach().float())
            _sl = student_logits[j]  # [ltk,V] bf16
            _lse_full = torch.empty(_sl.shape[0], dtype=torch.float32, device=device)
            for _s0 in range(0, _sl.shape[0], 512):
                _lse_full[_s0:_s0 + 512] = torch.logsumexp(_sl[_s0:_s0 + 512].float(), dim=-1)
            _lse_sup = torch.logsumexp(s_sup[0].float(), dim=-1)  # [ltk]
            _dp_mass.append((_lse_sup - _lse_full).exp()[_m].detach().float())
        impact_row = impact_rows[j] if j < len(impact_rows) else None
        if config.use_impact_weight and impact_row is not None:
            # --- A2v2：per-rollout impact top20% High/Low 分组 ---
            valid = completion_mask[j] > 0  # [ltk] bool
            n_valid = int(valid.sum().item())
            if n_valid <= 1:
                # 单有效 token：按规约只使用 high mean（= plain mean）
                cnt = mask_j.sum().clamp(min=1.0)
                per_rollout_losses.append((rkl * mask_j).sum() / cnt)
            else:
                imp_v = impact_row.to(device).float()[valid]      # [N] detached
                rkl_v = rkl[0][valid]                             # [N] 可导
                # A4：High 组占比由 GCEP_IMPACT_TOP_FRAC 控制（默认 0.2，
                # 与旧式 ceil(N/5) 逐值等价；A4_top10pt=0.10 / A4_top5pt=0.05）。
                k_top = max(1, int(-(-n_valid * config.impact_top_frac // 1)))  # ceil(frac·N)
                if config.impact_ablation == "random_select20pp":
                    # A5_random_select20pp：privilege 不变，
                    # 但分桶改为 seed-42 随机 frac/(1-frac)（high=perm[:k_top]、low=perm[k_top:]），
                    # 50:50 公式不变——消融"是 impact 分配重要，还是仅仅有个 20%/80% 分桶"。
                    _perm = torch.randperm(n_valid, generator=_impact_ablation_rng(config)).to(device)
                    high_idx, low_idx = _perm[:k_top], _perm[k_top:]
                elif config.impact_ablation == "mask_random20pp":
                    # A5_mask_random20pp：与 random_select20pp 同 seed 同
                    # 调用序的随机分桶（perm 逐点一致），随机选中的 frac 桶作为"low"（将被掩码），
                    # 其余 (1-frac) 桶作为"high"（保留 0.5 权重反传）。
                    _perm = torch.randperm(n_valid, generator=_impact_ablation_rng(config)).to(device)
                    high_idx, low_idx = _perm[k_top:], _perm[:k_top]
                elif config.impact_ablation == "mask_low20pp":
                    # A5_mask_low20pp_tokens：掩码最低 frac 桶，
                    # 训 top-(1-frac) 桶。升序 stable sort：low_idx=最低 ceil(frac·N)。
                    order_asc = torch.sort(imp_v, descending=False, stable=True).indices
                    high_idx, low_idx = order_asc[k_top:], order_asc[:k_top]
                else:
                    # stable sort：impact 降序，位置升序打破平局
                    order = torch.sort(imp_v, descending=True, stable=True).indices
                    high_idx, low_idx = order[:k_top], order[k_top:]
                l_high = rkl_v[high_idx].mean()
                l_low = rkl_v[low_idx].mean() if low_idx.numel() > 0 else rkl_v[high_idx].mean()
                if config.impact_ablation == "mask_top20pp":
                    # A5_mask_top20pp：去掉 top-frac 高 impact
                    # token 的梯度，只训 low 桶（high 整项删除、保留 0.5 系数——low token
                    # 梯度与基线逐点一致；该行整体 loss 减半，见消融记录）。
                    per_rollout_losses.append(0.5 * l_low)
                elif config.impact_ablation == "mask_low20pp":
                    # A5_mask_low20pp_tokens：low-20% 桶整项删除（完全不参与反传），
                    # top-80% 桶保留 0.5 系数——高 impact token 梯度与基线逐点一致。
                    per_rollout_losses.append(0.5 * l_high)
                elif config.impact_ablation == "mask_random20pp":
                    # A5_mask_random20pp：随机选中的 20% 桶整项删除（完全不参与反传），
                    # 其余 80% 桶保留 0.5 系数——与 mask_low20pp 只差"删哪一桶"。
                    per_rollout_losses.append(0.5 * l_high)
                else:
                    per_rollout_losses.append(0.5 * l_high + 0.5 * l_low)
                impact_valid_vals.append(imp_v.detach())
                mean_high_vals.append(float(imp_v[high_idx].mean()))
                mean_low_vals.append(float(imp_v[low_idx].mean()) if low_idx.numel() > 0 else 0.0)
                rkl_high_vals.append(float(l_high.detach()))
                rkl_low_vals.append(float(l_low.detach()) if low_idx.numel() > 0 else 0.0)
                high_mask_v = torch.zeros(n_valid, dtype=torch.bool, device=device)
                high_mask_v[high_idx] = True
                full_high_mask = torch.zeros_like(valid)
                full_high_mask[valid] = high_mask_v
                sidecar_rows.append((j, impact_row, full_high_mask, rkl.detach()[0], valid))
        else:
            if config.use_ps_weight:
                u_idx = gcep_teacher["union_idx"][j]
                if u_idx is not None:
                    priv_u = gcep_teacher["union_priv_logits"][j].unsqueeze(0).to(device)
                    clean_u = gcep_teacher["union_clean_logits"][j].unsqueeze(0).to(device)
                    # k>=U 时 ps_jsd_weights 内部 topk 覆盖全部 union 槽位，
                    # 数学上等价于直接在 union support 上重归一化。
                    w, jsd = _opd.ps_jsd_weights(priv_u, clean_u, mask_j, k=priv_u.shape[-1])
                    rkl = rkl * w
                    jsd_vals.append(jsd)
            # 单条 rollout 内 mask 均值
            cnt = mask_j.sum().clamp(min=1.0)
            per_rollout_losses.append((rkl * mask_j).sum() / cnt)

    if _DIVP:
        _divp_quantiles(metrics, "div_rkl", _dp_rkl)
        _divp_quantiles(metrics, "div_student_mass_on_topk", _dp_mass)
        if _dp_hit:
            metrics["div_topk_hit_rate"] = float(torch.cat(_dp_hit).mean())
    if _traj_inv is not None:
        metrics["opd_eligible_ratio"] = (bs - _n_excluded) / max(1, bs)
        metrics["traj_invalid_rows"] = float(_n_invalid)
        if _inc_inv:
            metrics["opd_include_invalid"] = 1.0
    if not per_rollout_losses:
        # 全部行无效（或全部被过滤）：本 micro-batch OPD 项为零，不产生梯度；
        # 训练继续（GRPO 项不受 OPD 项缺失影响；A/B 系纯 OPD 时本 chunk 零 loss）。
        # 零 loss 必须带 grad_fn（见 _zero_loss_with_graph）。
        loss = _zero_loss_with_graph(student_logits, device=device)
        metrics["opd_loss"] = 0.0
        return loss, metrics

    loss = torch.stack(per_rollout_losses).mean()  # batch 内 rollout 均值（过滤后 = Σ e_i L_i / Σ e_i）
    metrics["opd_loss"] = float(loss.detach())
    if jsd_vals:
        jsd_cat = torch.cat(jsd_vals, dim=0)
        metrics["privilege_jsd_mean"] = float(jsd_cat.mean())
        metrics["privilege_jsd_p90"] = float(
            torch.quantile(jsd_cat.flatten().float(), 0.9)
        )
    if impact_valid_vals:
        imp_cat = torch.cat(impact_valid_vals, dim=0).float()
        metrics["impact_mean"] = float(imp_cat.mean())
        for q, name in ((0.5, "p50"), (0.8, "p80"), (0.9, "p90"), (0.95, "p95"), (0.99, "p99")):
            metrics[f"impact_{name}"] = float(torch.quantile(imp_cat, q))
        metrics["mean_high_impact"] = sum(mean_high_vals) / len(mean_high_vals)
        metrics["mean_low_impact"] = sum(mean_low_vals) / len(mean_low_vals)
        metrics["rkl_high"] = sum(rkl_high_vals) / len(rkl_high_vals)
        metrics["rkl_low"] = sum(rkl_low_vals) / len(rkl_low_vals)
    # A5：消融臂自记录（仅启用时写入，默认 ""=不写 → 日志与基线逐字节一致）。
    if config.impact_ablation:
        metrics["impact_ablation"] = float(
            {"mask_top20pp": 1.0, "random_select20pp": 2.0, "mask_low20pp": 3.0,
             "mask_random20pp": 4.0}[config.impact_ablation])
    metrics["teacher_forward_sec"] = float(gcep_teacher.get("teacher_forward_sec", 0.0))
    metrics["teacher_priv_rows"] = float(sum(gcep_teacher.get("used_privilege", [])))
    metrics["teacher_mismatch_rows"] = float(gcep_teacher.get("mismatch", 0))

    # A2v2 per-token sidecar 归档（GCEP_ARCHIVE_IMPACT 门控；失败只告警不 crash）
    if sidecar_rows and config.use_impact_weight and os.getenv("GCEP_ARCHIVE_IMPACT", "1") == "1":
        dump_ctx = gcep_teacher.get("dump_ctx") or {}
        try:
            _dump_impact_sidecar(gcep_teacher, sidecar_rows, dump_ctx)
        except Exception:
            pass
    return loss, metrics


def _dump_impact_sidecar(
    gcep_teacher: Dict[str, Any],
    sidecar_rows: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    dump_ctx: Dict[str, Any],
) -> None:
    """把 privileged 行的 per-token impact/high_mask/rkl 写 npz sidecar。

    路径：<out_dir>/rollout_archive/impact/step_XXXXXXX/rankRR_chunkCCC_rowRRR.npz。
    dump_ctx（grpo_trainer 注入）：step / out_dir / rank / chunk。缺失时放弃归档。
    """
    import numpy as np

    out_dir = dump_ctx.get("out_dir")
    if not out_dir:
        return
    step = int(dump_ctx.get("step", -1))
    rank = int(dump_ctx.get("rank", 0))
    chunk = int(dump_ctx.get("chunk", 0))
    token_ids = gcep_teacher.get("student_token_ids")
    row_meta = gcep_teacher.get("row_meta") or []
    dest = os.path.join(out_dir, "rollout_archive", "impact", f"step_{step:07d}")
    os.makedirs(dest, exist_ok=True)
    for (j, impact_row, high_mask, rkl_row, valid) in sidecar_rows:
        meta = row_meta[j] if j < len(row_meta) else {}
        np.savez_compressed(
            os.path.join(dest, f"rank{rank:02d}_chunk{chunk:03d}_row{j:03d}.npz"),
            token_ids=token_ids[j].detach().cpu().numpy() if token_ids is not None else None,
            impact=impact_row.detach().float().cpu().numpy(),
            high_mask=high_mask.detach().cpu().numpy(),
            rkl=rkl_row.detach().float().cpu().numpy(),
            completion_mask=valid.detach().cpu().numpy(),
            step=step, chunk=chunk, row=j, rank=rank,
            privilege_type=str(meta.get("privilege_type") or ""),
            group_type=str(meta.get("group_type") or ""),
        )


def combine_losses(
    grpo_loss: Optional[torch.Tensor],
    opd_loss: torch.Tensor,
    config: GCEPConfig,
) -> torch.Tensor:
    """：两个 loss 各自独立归一化（grpo_loss 为原 mean loss；opd_loss 为
     的 rollout 均值）之后再相加；A/B 系 grpo_loss=None -> 纯 OPD。"""
    if config.use_grpo_loss and grpo_loss is not None:
        return grpo_loss + config.lambda_opd * opd_loss
    return opd_loss


# ---------------------------------------------------------------------------
# 全量 rollout 归档（分 rank 写 shard；失败只告警不 crash）
# ---------------------------------------------------------------------------
def archive_rollouts(
    trainer,
    inputs: List[Dict[str, Any]],
    total_rewards: torch.Tensor,
    rewards_per_func: torch.Tensor,
    config: GCEPConfig,
    logger=None,
) -> None:
    """把本 rank 的全部 rollout 写入 RUN_DIR/rollout_archive/。

    - inputs 为本 rank 本地切片；total_rewards / rewards_per_func 为 gather 后的
      全局张量，按 process_index 切回本地（与 _prepare_batch_inputs 的 slice 一致）；
    - 文本记录走 rollout_archive.build_record（含 turns 解析、derived seed）；
    - sandbox 图片 persist_sandbox_images hardlink/copy 持久化，
      并按 turn 的 sandbox_image_count 顺序回填 sandbox_image_archive_paths；
    - 任何异常只 warning，绝不影响训练。
    """
    if not config.archive:
        return
    try:
        archiver = getattr(trainer, "_gcep_archiver", None)
        if archiver is None:
            experiment_id = config.experiment_id or os.path.basename(
                str(trainer.args.output_dir).rstrip("/")
            )
            archiver = _archive.RolloutArchiver(trainer.args.output_dir, experiment_id)
            trainer._gcep_archiver = archiver

        rank = int(trainer.accelerator.process_index)
        step = int(trainer.state.global_step)
        offset = rank * len(inputs)
        ng = int(trainer.num_generations)
        local_rewards = total_rewards[offset:offset + len(inputs)]
        rpf_local = rewards_per_func[offset:offset + len(inputs)]
        reward_names = list(getattr(trainer, "reward_func_names", []))

        for i, inp in enumerate(inputs):
            gid = offset + i
            rollout_id = f"r{gid:06d}"
            group_id = f"g{(gid // ng):06d}" if ng > 0 else ""
            # sandbox 图片持久化（images[0] 为原图，其后为按序 sandbox 图）
            raw_paths = _image_paths(inp)
            sandbox_src = [p for p in raw_paths[1:] if p]
            archived = archiver.persist_sandbox_images(sandbox_src, step, rollout_id)
            # 先解析 turns，按每个 turn 的 sandbox_image_count 顺序分配 archived path
            trajectory = _assistant_content(inp)
            turns = _archive.parse_turns(trajectory)
            turn_image_paths: Dict[int, List[str]] = {}
            cursor = 0
            for turn in turns:
                n_img = int(turn.get("sandbox_image_count", 0) or 0)
                if n_img > 0:
                    turn_image_paths[turn["turn_id"]] = [
                        p for p in archived[cursor:cursor + n_img] if p
                    ]
                    cursor += n_img
            record = _archive.build_record(
                experiment_id=archiver.experiment_id,
                global_step=step,
                prompt_id=group_id,
                group_id=group_id,
                rollout_id=rollout_id,
                rollout_idx=gid % ng if ng > 0 else i,
                rollout=inp,
                vqa_norm=int(inp.get("vqa_norm_binary", 0)),
                reward_components={
                    name: float(rpf_local[i, j])
                    for j, name in enumerate(reward_names)
                    if i < rpf_local.shape[0] and j < rpf_local.shape[1]
                },
                total_reward=float(local_rewards[i]) if i < local_rewards.shape[0] else 0.0,
                policy_version=str(step),
                text_temperature=float(getattr(trainer.args, "temperature", 1.0)),
                code_temperature=0.0,
                turn_image_archive_paths=turn_image_paths or None,
                gcep_meta=_gcep_record_meta(inp),
            )
            archiver.archive_rollout(record, step, rank)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"[GCEP] rollout archive failed (training unaffected): {exc}")


def _gcep_record_meta(inp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 inp 提取 per-rollout GCEP 归档元数据；无 GCEP 信息返回 None。

    privilege_type 三态：full_certificate / diagnosis_only / clean_fallback。
    judge_prompt_type 与 group_type 一一对应（dispatcher 单射）。
    """
    plan = inp.get("gcep_privilege_plan")
    group_type = inp.get("gcep_group_type")
    if plan is None and group_type is None:
        return None
    meta = {
        "group_type": group_type,
        "judge_prompt_type": group_type,
        "privilege_type": (
            plan.get("privilege_type") if plan is not None else "clean_fallback"
        ),
        "diagnosis_stage": (plan or {}).get("diagnosis_stage"),
        "discrepancy": (plan or {}).get("discrepancy"),
        "anchor_state_id": (plan or {}).get("anchor_state_id"),
        "acq_mode": (plan or {}).get("acq_mode"),
        "used_own_anchor": (plan or {}).get("used_own_anchor"),
    }
    return meta
