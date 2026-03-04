import os
import math
import torch
from typing import Iterable, BinaryIO, IO, Union

def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    基于 logits 的数值稳定交叉熵。

    参数：
        logits: 形状为 [..., vocab_size] 的张量，其中最后一个维度是类别维度。
        targets: 形状为 [...] 的 Long 张量，包含 [0, vocab_size) 范围内的类别索引。

    返回：
        标量张量：所有批次元素的平均负对数似然。
    """
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension [..., vocab_size].")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"targets shape {targets.shape} must match logits batch shape {logits.shape[:-1]}.")
    if targets.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise TypeError("targets must be an integer tensor of class indices.")

    # 减去最大值以保持数值稳定性
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m

    # 在类别维度上计算 logsumexp
    lse = torch.logsumexp(shifted, dim=-1)  # shape [...]

    # 收集对应真实类别的 logit
    idx = targets.unsqueeze(-1)  # [..., 1]
    correct = shifted.gather(dim=-1, index=idx).squeeze(-1)  # shape [...]

    # 负对数似然：logsumexp - correct_logit
    nll = lse - correct

    # 返回所有批次元素的平均值
    return nll.mean()

def clip_grad_norm(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6
) -> float:
    """
    原地裁剪梯度，使得全局 L2 范数不超过 max_norm。

    参数：
        params: 参数的可迭代对象，其 .grad 属性将被原地修改。
        max_norm: 允许的最大全局 L2 范数。
        eps: 用于数值稳定性的小常数。

    返回：
        裁剪前的总全局 L2 范数（Python float）。
    """
    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")
    
    # 收集存在的梯度
    grads = []
    for p in params:
        if p is None:
            continue
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            raise RuntimeError("clip_grad_norm_ does not support sparse gradients.")
        grads.append(g)
    
    if len(grads) == 0:
        return 0.0
    
    # 计算全局 L2 范数：sqrt(sum_i ||g_i||_2^2)
    # 使用 float32 累加以保持稳定性和一致性
    total_sq = 0.0
    for g in grads:
        total_sq += float(g.detach().float().pow(2).sum().item())
    total_norm = math.sqrt(total_sq)

    # 计算裁剪系数
    clip_coef = float(max_norm) / (total_norm + float(eps))

    # 如果范数超过阈值，则按相同因子缩放所有梯度（原地）
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

    return float(total_norm)

PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: PathOrFile
) -> None:
    """
    保存包含模型/优化器状态和迭代次数的训练检查点。

    参数：
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: 当前训练迭代（步数）。
        out: 文件路径或二进制类文件对象。
    """
    obj = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(obj, out)    

def load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    加载训练检查点并恢复模型/优化器状态。

    参数：
        src: 文件路径或二进制类文件对象。
        model: 要恢复到的 torch.nn.Module。
        optimizer: 要恢复到的 torch.optim.Optimizer。

    返回：
        检查点中存储的迭代（步数）。
    """
    ckpt = torch.load(src, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")
    
    if "model_state_dict" not in ckpt or "optimizer_state_dict" not in ckpt or "iteration" not in ckpt:
        raise KeyError("Checkpoint dict missing required keys.")

    model.load_state_dict(ckpt["model_state_dict"])    
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["iteration"])