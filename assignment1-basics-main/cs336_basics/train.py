import os
import time
import math
import torch
import numpy as np
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy_from_logits, load_checkpoint, save_checkpoint, clip_grad_norm
from cs336_basics.config import get_default_config
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule_with_warmup

def open_memmap_1d(path: str, np_dtype: str) -> np.memmap:
    """
    打开一个一维的 token 内存映射文件。假设文件是原始二进制数组。
    """
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbytes = os.path.getsize(path)
    if nbytes % itemsize != 0:
        raise ValueError(f"File size is not divisible by dtype size: {path} ({nbytes} bytes, itemsize={itemsize})")
    length = nbytes // itemsize
    return np.memmap(path, mode="r", dtype=dtype, shape=(length,))

def torch_dtype_from_string(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype string: {s}")

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, data: np.memmap, cfg) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.train.eval_batches):
        xb, yb = get_batch(
            dataset=data,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.device
        )
        logits = model(xb)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_from_logits(logits.reshape(B * S, V), yb.reshape(B * S))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))

def main() -> None:
    # 1. 加载配置并设置随机种子
    cfg = get_default_config()

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # 2. 可选的实验跟踪 (weights & biases)
    wandb = None
    if cfg.wandb.enable:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config={
            "data": cfg.data.__dict__,
            "model": cfg.model.__dict__,
            "optim": cfg.optim.__dict__,
            "train": cfg.train.__dict__,
            "wandb": cfg.wandb.__dict__
        })

    # 3. 准备文件系统并加载数据集（内存映射）
    os.makedirs(os.path.dirname(cfg.train.ckpt_path) or ".", exist_ok=True)

    train_mm = open_memmap_1d(cfg.data.train_data_path, cfg.data.np_dtype)
    val_mm = open_memmap_1d(cfg.data.val_data_path, cfg.data.np_dtype)

    # 4. 创建模型并将其移动到目标设备
    device = torch.device(cfg.data.device)
    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)

    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model 

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=d_ff,
        rope_theta=cfg.model.rope_theta,
        max_seq_len=cfg.model.max_seq_len,
        eps=cfg.model.rmsnorm_eps,
        device=device,
        dtype=model_dtype
    ).to(device)

    # 5. 创建优化器并（可选地）从检查点恢复
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr_max,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay
    )

    start_it = 0
    if cfg.train.resume_from is not None and os.path.exists(cfg.train.resume_from):
        start_it = load_checkpoint(cfg.train.resume_from, model, optimizer)

    # 6. 训练循环初始化
    best_val = float("inf")
    last_log_t = time.time()

    # 7. 主训练循环
    for it in range(start_it, cfg.train.max_steps):
        # 7.1 根据调度更新学习率
        lr = lr_cosine_schedule_with_warmup(
            t=it, 
            alpha_max=cfg.optim.lr_max,
            alpha_min=cfg.optim.lr_min,
            T_w=cfg.optim.warmup_iters,
            T_c=cfg.optim.cosine_cycle_iters
        )
        set_optimizer_lr(optimizer, lr)

        # 7.2 采样一批训练数据
        xb, yb = get_batch(
            train_mm,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.device
        )

        # 7.3 前向传播和损失计算
        logits = model(xb)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_from_logits(logits.reshape(B * S, V), yb.reshape(B * S))

        # 7.4 反向传播（梯度计算）
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 7.5 梯度裁剪以保持训练稳定性
        if cfg.optim.grad_clip > 0:
            clip_grad_norm(model.parameters(), cfg.optim.grad_clip, eps=1e-6)
        
        # 7.6 优化器步骤（参数更新）
        optimizer.step()

        # 7.7 定期记录训练指标
        if (it + 1) % cfg.train.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_t, 1e-9)
            tok_s = (cfg.train.batch_size * cfg.data.context_length * cfg.train.log_interval) / dt
            msg = f"it={it+1} loss={loss.item():.4f} lr={lr:.3e} tok/s={tok_s:.1f}"
            print(msg)
            if wandb is not None:
                wandb.log({"train/loss": float(loss.item()), "train/lr": lr, "train/tok_s": tok_s}, step=it + 1)
            last_log_t = now
        
        # 7.8 在验证集上进行定期评估
        if (it + 1) % cfg.train.eval_interval == 0:
            val_loss = estimate_loss(model, val_mm, cfg)
            val_ppl = float(math.exp(val_loss))
            print(f"[eval] it={it+1} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
            if wandb is not None:
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=it + 1)            

            # 保存表现最好的检查点
            if val_loss < best_val:
                best_val = val_loss
                best_path = cfg.train.ckpt_path.replace(".pt", ".best.pt")
                save_checkpoint(model, optimizer, it + 1, best_path)

        # 7.9 定期保存检查点
        if (it + 1) % cfg.train.ckpt_interval == 0:
            save_checkpoint(model, optimizer, it + 1, cfg.train.ckpt_path)

    # 8. 最终检查点和清理
    save_checkpoint(model, optimizer, cfg.train.max_steps, cfg.train.ckpt_path)
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()