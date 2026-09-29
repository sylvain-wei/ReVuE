# Modified for this anonymized research release: portable paths and release cleanup.
"""GCEP OPD losses：Top32 support 重归一化截断 Reverse-KL + privilege-sensitive JSD 权重 + 聚合。

实现 Top-K 截断 RKL、per-rollout 聚合和 PS-JSD 权重。

dtype 约定：所有 logsumexp / softmax 都在 fp32 下完成，输入可以是 bf16/fp16/fp32。
teacher 侧一律 detach（teacher logits 与 Top-K indices 均 stop-grad， Gradient 约定）。

纯 torch 模块：不 import swift / transformers，可独立加载。
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

TOPK = 32
EPS = 1e-8


def _gather_topk(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按 [B,T,K] 索引 gather logits -> [B,T,K]。"""
    return logits.gather(-1, idx)


def topk_rkl_per_token(
    student_logits: torch.Tensor,
    teacher_target_logits: torch.Tensor,
    policy_mask: Optional[torch.Tensor] = None,
    k: int = TOPK,
) -> torch.Tensor:
    """Top-K support 重归一化的 per-token Reverse KL。

    support = TopK(teacher logits) 索引（teacher 侧整体 stop-grad）。
    student / teacher logits 分别在 support 内做 fp32 重归一化，
    然后逐 token 求 sum_{v in S} p*(logp - logq)。

    注意：不把 student sampled token 强行 union 进 support（ 末段）。

    Args:
        student_logits: [B,T,V]，可带 grad。
        teacher_target_logits: [B,T,V]，函数内部 detach。
        policy_mask: [B,T]，仅用于保持 API 形状校验；返回的是未 mask 的 RKL，
            mask 由调用方在聚合时施加。可为 None。
        k: support 大小（v1 固定 32）。

    Returns:
        rkl: [B,T] fp32，未乘 mask。
    """
    if student_logits.shape != teacher_target_logits.shape:
        raise ValueError(
            f"student/teacher logits 形状不一致: {student_logits.shape} vs {teacher_target_logits.shape}"
        )
    if policy_mask is not None and policy_mask.shape != student_logits.shape[:2]:
        raise ValueError(
            f"policy_mask 形状 {policy_mask.shape} 与 [B,T]={student_logits.shape[:2]} 不一致"
        )

    V = student_logits.shape[-1]
    k_eff = min(k, V)  # 小词表（单测）下退化为全词表

    # teacher 侧整体 stop-grad：logits 与 Top-K 索引都不可导
    teacher_logits = teacher_target_logits.detach()
    topk_idx = teacher_logits.topk(k_eff, dim=-1).indices  # [B,T,K]

    # 只 gather support 上的 [B,T,K] 小切片再转 fp32，避免全词表 fp32 logsumexp
    s_sup = _gather_topk(student_logits, topk_idx).float()
    t_sup = _gather_topk(teacher_logits, topk_idx).float()

    log_p = F.log_softmax(s_sup, dim=-1)  # student 可导
    log_q = F.log_softmax(t_sup, dim=-1)  # teacher 已 detach

    rkl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [B,T]
    return rkl


def masked_rkl(rkl: torch.Tensor, policy_mask: torch.Tensor) -> torch.Tensor:
    """便捷 helper：对未 mask 的 RKL 施加 policy mask。"""
    return rkl * policy_mask.to(rkl.dtype)


def aggregate_opd(
    rkl: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """ 聚合：先对每条 rollout 在 mask 内取均值，再对 batch 内 rollout 取均值。

    禁止 flatten 全 batch token 求和（长 rollout 会支配 loss）。

    Args:
        rkl: [B,T] per-token（已乘或未乘权重均可）损失。
        mask: [B,T] {0,1} policy token mask。

    Returns:
        scalar loss。
    """
    m = mask.to(rkl.dtype)
    tok_sum = (rkl * m).sum(dim=-1)  # [B]
    tok_cnt = m.sum(dim=-1)  # [B]
    # 空 rollout（无 policy token）贡献 0；clamp 仅为除零保护，不改变非空语义
    per_rollout = tok_sum / tok_cnt.clamp(min=1.0)
    # 再对 batch 内 rollout 取均值（分母 = N_rollout）
    return per_rollout.mean()


def _renorm_on_union(
    logits: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """在 union support 上重归一化一个（已 detach 的）teacher 分布。

    idx 允许含重复项（Top32(priv) 与 Top32(clean) 的重叠部分），重复槽位由
    valid=False 标出并置 -inf，等价于去重后的 union 上归一化。

    Returns:
        log_probs: [B,T,U] fp32，无效槽位为 -inf（exp 后为 0）。
    """
    sup = logits.gather(-1, idx).float()  # [B,T,U]
    neg_inf = torch.finfo(sup.dtype).min  # 用 dtype min 而非 -inf，避免 bf16 cast 边角
    sup = torch.where(valid, sup, torch.full_like(sup, neg_inf))
    return F.log_softmax(sup, dim=-1)


def ps_jsd_weights(
    priv_teacher_logits: torch.Tensor,
    clean_teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int = TOPK,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Privilege-Sensitive JSD token 权重。

    union support U = TopK(priv) ∪ TopK(clean)（|U|<=2K），两分布分别在 U 内
    fp32 重归一化；m=0.5*(q+ + q0)，s=0.5*KL(q+||m)+0.5*KL(q0||m)，
    natural log，0<=s<=log2，s 整体 detach。

    a = 1+s；对每条 rollout 在 mask 内做 mean-preserving 归一化：
    w = a / (sum(m*a)/sum(m) + eps)，因此 mean_{t:m=1} w ≈ 1。

    无 GCEP 的 rollout：调用方把 clean 当 priv 传入 -> jsd≈0、w≈1，自动退化
    vanilla OPD。

    Args:
        priv_teacher_logits: [B,T,V] privileged teacher logits（内部 detach）。
        clean_teacher_logits: [B,T,V] clean teacher logits（内部 detach）。
        mask: [B,T] {0,1} policy token mask。
        k: 单边 Top-K 大小（v1 固定 32）。

    Returns:
        (weights [B,T] fp32, jsd [B,T] fp32 detached)。
    """
    if priv_teacher_logits.shape != clean_teacher_logits.shape:
        raise ValueError("priv/clean teacher logits 形状不一致")
    if mask.shape != priv_teacher_logits.shape[:2]:
        raise ValueError("mask 形状与 [B,T] 不一致")

    V = priv_teacher_logits.shape[-1]
    k_eff = min(k, V)

    q_priv = priv_teacher_logits.detach()
    q_clean = clean_teacher_logits.detach()

    idx_p = q_priv.topk(k_eff, dim=-1).indices  # [B,T,K]
    idx_c = q_clean.topk(k_eff, dim=-1).indices  # [B,T,K]

    # clean Top-K 中已出现在 priv Top-K 的槽位标记为重复（union 去重）
    dup_c = (idx_c.unsqueeze(-1) == idx_p.unsqueeze(-2)).any(dim=-1)  # [B,T,K]
    idx_u = torch.cat([idx_p, idx_c], dim=-1)  # [B,T,2K]
    valid_u = torch.cat(
        [torch.ones_like(dup_c), ~dup_c], dim=-1
    )  # priv 侧全有效，clean 侧去重

    log_qp = _renorm_on_union(q_priv, idx_u, valid_u)  # log \bar q^+
    log_q0 = _renorm_on_union(q_clean, idx_u, valid_u)  # log \bar q^0

    qp = log_qp.exp()
    q0 = log_q0.exp()
    m = 0.5 * (qp + q0)  # [B,T,U]
    log_m = m.clamp(min=EPS).log()  # m>0（至多一边为 0），clamp 仅为数值安全

    # KL(q||m) = sum q*(log q - log m)；q=0 的槽位贡献 0，用 where 避免 0*(-inf)=nan
    kl_p = torch.where(qp > 0, qp * (log_qp - log_m), torch.zeros_like(qp)).sum(-1)
    kl_0 = torch.where(q0 > 0, q0 * (log_q0 - log_m), torch.zeros_like(q0)).sum(-1)
    jsd = (0.5 * kl_p + 0.5 * kl_0).clamp(min=0.0).detach()  #  必须 detach

    a = 1.0 + jsd  # [B,T]
    msk = mask.to(a.dtype)
    cnt = msk.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B,1] 防空 rollout 除零
    masked_mean = (msk * a).sum(dim=-1, keepdim=True) / cnt  # [B,1]
    w = a / (masked_mean + eps)
    return w, jsd


def weighted_opd_loss(
    rkl: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """便捷 helper： 的 L_PS = aggregate_opd(rkl * w, mask)。"""
    return aggregate_opd(rkl * weights, mask, eps=eps)
