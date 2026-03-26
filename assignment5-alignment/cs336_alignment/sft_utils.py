from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Tensor]:
    """
    将 prompt 和 output 分别 tokenize 后拼接，最后构造：
      - input_ids: concat[:-1] # 模型输入序列
      - labels:    concat[1:]  # 模型目标输出序列
      - response_mask: 对应 output token 的 label 位置为 1，其它为 0
    形状： (batch_size, max_len - 1)
    """
    # 前置校验1：确保 prompt 和 output 长度一致
    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have same length, got {len(prompt_strs)} vs {len(output_strs)}"
        )
    
    # 前置校验2：空输入时返回空张量
    batch_size = len(prompt_strs)
    if batch_size == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty}
    
    # 设置 pad_token_id（填充标记ID）
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # eos_token_id: 结束标记ID
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # 逐样本 Tokenize 并记录长度
    prompt_ids_list: List[List[int]] = []
    output_ids_list: List[List[int]] = []
    concat_ids_list: List[List[int]] = []
    prompt_lens: List[int] = []
    output_lens: List[int] = []

    for p, o in zip(prompt_strs, output_strs):
        # add_special_tokens=False：不自动加bos/eos等特殊token（避免重复）
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        o_ids = tokenizer(o, add_special_tokens=False).input_ids
        prompt_ids_list.append(list(p_ids))
        output_ids_list.append(list(o_ids))
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))
        concat_ids_list.append(list(p_ids) + list(o_ids))

    # 填充所有张量至序列最大长度（prompt + output）
    max_len = max(len(x) for x in concat_ids_list)
    full = torch.full((batch_size, max_len), pad_id, dtype=torch.long) # prompt + output 的填充张量

    # 构建 response_mask：对于 output token 的位置为 1，其它为 0
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.long)

    for i, ids in enumerate(concat_ids_list):
        L = len(ids)
        full[i, :L] = torch.tensor(ids, dtype=torch.long)

        P = prompt_lens[i]
        O = output_lens[i]

        # 标记 labels 中属于 output 的位置
        # 原序列：[prompt_token1, prompt_token2, ..., output_token1, output_token2, ...]
        # input_ids：原序列[:-1]，labels：原序列[1:]（错位）
        # 所以output_token在labels中的位置是 [P-1, P+O-2]
        if O > 0:
            start = max(P - 1, 0)
            end = min(start + O, max_len - 1)  # 限制到 mask 长度
            response_mask[i, start:end] = 1

    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 的下一个 token 概率分布熵（在词汇表维度上）。

    Args:
        logits: (batch_size, sequence_length, vocab_size)

    Returns:
        entropies: (batch_size, sequence_length)
    """
    # 维度合法性检查：至少1维（兼容特殊情况，比如单token/单batch）
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dim, got {tuple(logits.shape)}")

    # 1. 计算logsumexp（归一化常数的对数）
    # log_z shape: (batch_size, sequence_length, 1)
    # dim=-1 表示在词汇表维度（vocab_size）上计算
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)   
    
    # 2. 计算归一化的对数概率 log(p(v))
    # logits是原始未归一化的得分，减去log_z后得到 log(p(v))
    # log_probs shape: (batch_size, sequence_length, vocab_size)
    log_probs = logits - log_z                              
    
    # 3. 转换为概率 p(v)（可选，但便于理解熵公式）
    probs = torch.exp(log_probs)                           

    # 4. 计算熵：-sum(p(v) * log(p(v)))
    # sum(dim=-1) 对词汇表维度求和，最终得到每个token的熵
    # entropy shape: (batch_size, sequence_length)
    entropy = -(probs * log_probs).sum(dim=-1)              
    
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    计算因果语言模型每个 token 的条件对数概率。

    Args:
        model: HF 因果 LM，已放在正确设备上。
        input_ids: (B, T)
        labels:    (B, T) token ids
        return_token_entropy: 如果为 True，还返回从 logits 计算的每 token 熵 (B, T)。

    Returns:
        dict 包含：
          - "log_probs": (B, T)
          - "token_entropy": (B, T) 如果请求
    """
    # 输入合法性校验
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"input_ids and labels must be (B, T). Got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have same shape. Got {tuple(input_ids.shape)} vs {tuple(labels.shape)}"
        )

    # 前向计算
    logits = model(input_ids=input_ids).logits  # (B, T, V)

    # 在词表上计算稳定的对数概率
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # 提取目标 label 对应的 token 对数概率
    gathered = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1))  # (B, T, 1)
    token_log_probs = gathered.squeeze(-1)  # (B, T)

    out: Dict[str, torch.Tensor] = {"log_probs": token_log_probs}

    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)  # (B, T)

    return out

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None
) -> torch.Tensor:
    """
    对 tensor 在指定维度上加权求和（mask==1 位置参与），
    然后除以 normalize_constant。
    """
    if tensor.shape != mask.shape:
        raise ValueError(
            f"tensor and mask must have same shape, got {tensor.shape} vs {mask.shape}"
        )

    # 将 mask 转换为与 tensor 相同 dtype 以便乘法
    mask = mask.to(dtype=tensor.dtype)

    # 掩码操作：将 mask 应用到 tensor 上，mask=0 的位置会被置零
    masked_tensor = tensor * mask

    # 根据 dim 参数求和
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    # 归一化：消除不同样本/微批次之间的规模差异，确保 loss 在不同设置下具有可比性
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    执行一次 SFT 微批次训练步骤：
      - 计算掩码后的负对数似然
      - 以 normalize_constant 归一化
      - 按 gradient_accumulation_steps 缩放
      - 调用 backward()

    Args:
        policy_log_probs: (B, T) log p(x_t | x_<t)
        response_mask:    (B, T) 对于响应 token 为 1，否则为 0
        gradient_accumulation_steps: 每次优化器步长的微批次数
        normalize_constant: 在 grad-acc 缩放前的除数

    Returns:
        loss: 标量张量（已根据梯度累积缩放）
        metadata: 有用统计数据的字典（未缩放损失、token 数等）
    """
    # 1. 输入校验
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"policy_log_probs and response_mask must have same shape, "
            f"got {tuple(policy_log_probs.shape)} vs {tuple(response_mask.shape)}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # 2. 计算损失：负对数似然
    nll = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,
    ).mean() # 标量

    # 按梯度累积比例缩放
    loss = nll / float(gradient_accumulation_steps)

    # 3. 反向传播
    # 不更新模型参数，只计算并累积梯度
    # 参数更新由优化器在梯度累积完成后执行）
    loss.backward() 

    # 4. 统计信息收集
    with torch.no_grad(): # 禁用梯度计算，避免额外内存消耗和计算开销
        # 本 microbatch 中响应 token 数量
        resp_tokens = response_mask.to(policy_log_probs.dtype).sum()
        metadata = {
            "nll": nll.detach(),
            "loss_unscaled": nll.detach(),  # alias (sometimes handy)
            "response_tokens": resp_tokens.detach(),
            "mean_log_prob_on_response": (
                (policy_log_probs * response_mask.to(policy_log_probs.dtype)).sum()
                / torch.clamp(resp_tokens, min=1.0)
            ).detach(),
        }

    return loss, metadata


@torch.no_grad() # 禁用梯度计算，避免额外内存消耗和计算开销
def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    num_log: int = 8,
    step: int | None = None,
    stop_str: str | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    生成若干提示的回答并记录：
      - prompt / response / ground_truth
      - reward: format_reward, answer_reward, reward
      - 生成 token 的平均熵
      - 长度统计（平均、正确平均、错误平均）
    
    返回 dict，包含：
      - "samples": list[dict]
      - "stats": dict
    """
    assert len(prompts) == len(ground_truths), "prompts and ground_truths must align"

    # 1. 初始化
    model.eval() # 模型切换为评估/推理模式，禁用 dropout 等训练特定行为
    if device is None:
        device = next(model.parameters()).device

    n = min(num_log, len(prompts))
    prompts = prompts[:n]
    ground_truths = ground_truths[:n]

    # 决定是否采样
    if do_sample is None:
        do_sample = temperature > 0

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 输入预处理（Prompt编码）
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)    

    # 3. 核心推理（模型生成文本）
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequences = gen_out.sequences  # (B, T_total)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    samples: list[dict[str, Any]] = []
    lengths: list[int] = []
    lengths_correct: list[int] = []
    lengths_wrong: list[int] = []
    entropies: list[float] = []

    # 依据 scores 计算每步熵
    avg_ent_per_sample = [0.0 for _ in range(n)]
    if gen_out.scores is not None and len(gen_out.scores) > 0:
        # 累计每个样本每步的熵
        acc = [0.0 for _ in range(n)]
        for step_logits in gen_out.scores:
            for i in range(n):
                acc[i] += float(compute_entropy(step_logits[i]).item())
        denom = float(len(gen_out.scores))
        avg_ent_per_sample = [x / denom for x in acc]

    # 4. 结果后处理（解码 + 奖励计算）
    for i in range(n):
        prompt = prompts[i]
        gt = ground_truths[i]
        pl = int(prompt_lens[i])

        full_ids = sequences[i]
        gen_ids = full_ids[pl:]  # 生成部分

        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if stop_str is not None and stop_str in response_text:
            response_text = response_text.split(stop_str)[0] + stop_str

        reward_dict = reward_fn(response_text, gt)

        gen_len = int(gen_ids.numel())
        avg_ent = float(avg_ent_per_sample[i])

        samples.append(
            {
                "step": step,
                "prompt": prompt,
                "response": response_text,
                "ground_truth": gt,
                "reward": float(reward_dict.get("reward", 0.0)),
                "format_reward": float(reward_dict.get("format_reward", 0.0)),
                "answer_reward": float(reward_dict.get("answer_reward", 0.0)),
                "avg_token_entropy": avg_ent,
                "response_len": gen_len,
            }
        )

        lengths.append(gen_len)
        entropies.append(avg_ent)
        is_correct = float(reward_dict.get("answer_reward", 0.0)) >= 1.0
        if is_correct:
            lengths_correct.append(gen_len)
        else:
            lengths_wrong.append(gen_len)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    stats = {
        "step": step,
        "avg_response_len": _mean([float(x) for x in lengths]),
        "avg_response_len_correct": _mean([float(x) for x in lengths_correct]),
        "avg_response_len_wrong": _mean([float(x) for x in lengths_wrong]),
        "avg_token_entropy": _mean(entropies),
        "avg_reward": _mean([s["reward"] for s in samples]),
        "avg_format_reward": _mean([s["format_reward"] for s in samples]),
        "avg_answer_reward": _mean([s["answer_reward"] for s in samples]),
        "n_logged": len(samples),
    }

    return {"samples": samples, "stats": stats}