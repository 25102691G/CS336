import torch
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一个一维的 token ID numpy 数组中采样语言模型批次。

    参数：
        dataset: 整数 token ID 的一维 numpy 数组（或内存映射文件）。
        batch_size: 要采样的序列数量。
        context_length: 每个输入/目标序列的长度。
        device: PyTorch 设备字符串，例如 "cpu"、"cuda:0"、"mps"。

    返回：
        (inputs, targets): 两个都是形状为 (batch_size, context_length) 的 torch.LongTensor，
        并放置在指定的设备上。
    """

    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    n = int(dataset.shape[0])
    if n < context_length + 1:
        raise ValueError(f"dataset too small: need at least context_length+1 tokens, got n={n}, context_length={context_length}")    
    
    # 创建数据集的 CPU 张量视图
    x = torch.from_numpy(dataset)

    # 在 cpu 上采样起始索引
    max_start = n - context_length - 1
    starts = torch.randint(low=0, high=max_start + 1, size=(batch_size,), device="cpu") # 随机生成采样起始索引

    # 通过广播构建位置 [batch_size, context_length]
    offsets = torch.arange(context_length, device="cpu")
    pos = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, S)

    inputs = x[pos]
    targets = x[pos + 1]
    # 使用 numpy 索引以避免在只读 memmap 上创建 PyTorch 张量时出现警告
    pos_np = pos.numpy()
    inputs = torch.from_numpy(dataset[pos_np])
    targets = torch.from_numpy(dataset[pos_np + 1])

    # 确保数据类型为 int64 作为 token ID，并移动到目标设备
    inputs = inputs.to(dtype=torch.long, device=device)
    targets = targets.to(dtype=torch.long, device=device)
    return inputs, targets