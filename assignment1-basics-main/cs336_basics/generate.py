import torch
from typing import Optional

def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    给定 softmax 后的分布 probs，只保留累计概率达到 top_p 所需的 最小集合，
    其余 token 概率置零，然后对保留部分重新归一化

    参数：
        probs: 概率的一维张量（和为 1）。
        top_p: 累积概率阈值，范围在 (0, 1]。
    
    返回：
        过滤后的概率（重新归一化），形状与 probs 相同。
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")    

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    # 保留累积概率 <= top_p 的 token，但至少保留一个 token。
    keep = cum <= top_p
    keep[..., 0] = True

    filtered_sorted_probs = sorted_probs * keep.to(sorted_probs.dtype)
    filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(probs)
    out.scatter_(dim=-1, index=sorted_idx, src=filtered_sorted_probs)
    return out

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    end_token_id: int,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> torch.Tensor:
    
    # 1. 输入参数校验
    if prompt_ids.dim() != 1:
        raise ValueError(f"prompt_ids must be 1D (t,), got shape {tuple(prompt_ids.shape)}")
    if prompt_ids.dtype != torch.long:
        prompt_ids = prompt_ids.to(torch.long)

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    
    # 2. 模型状态切换（评估模型）
    model_was_training = model.training
    model.eval()

    # 3. 初始化生成结果 & 设备对齐
    device = next(model.parameters()).device
    out = prompt_ids.to(device)

    # 4. 读取模型上下文长度（可选）
    context_length: Optional[int] = getattr(model, "context_length", None)

    # 5. 进入生成循环（逐token生成）
    for _ in range(max_new_tokens):
        # 5.1 阶段输入到模型上下文长度
        if context_length is not None and out.numel() > context_length:
            inp = out[-context_length:]
        else:
            inp = out

        # 5.2 模型推理，获取最后一个token的logits
        logits = model(inp.unsqueeze(0))  # (1, S, V)
        next_logits = logits[0, -1, :]    # (V,)

        # 5.3 根据温度选择生成策略
        if temperature == 0.0:
            # 5.3.1 贪心选择（选概率最大的token）
            next_id = int(torch.argmax(next_logits).item())
        else:
            if temperature < 0.0:
                raise ValueError(f"temperature must be >= 0, got {temperature}")
            
            # 5.3.2 温度缩放 + softmax转换为概率
            scaled = next_logits / float(temperature)
            probs = torch.softmax(scaled, dim=-1)

            # 5.3.3 top-p采样
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)
            
            # 5.3.4 从概率分布中随机选择一个token
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        
        # 5.4 拼接新token到生成结果
        out = torch.cat([out, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)

        # 5.5 检查是否遇到结束符，终止循环
        if next_id == int(end_token_id):
            break

    if model_was_training:
        model.train

    return out