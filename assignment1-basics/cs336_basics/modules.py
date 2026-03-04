import math
import torch
from torch import nn

class Linear(nn.Module):
    """
    一个无偏置的线性层，与 torch.nn.Linear 的接口匹配（除了没有偏置）。
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # 存储 W（不是 W^T）：形状为 (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
        )

        # 初始化：N(0, 2/(d_in+d_out))，截断至 [-3σ, 3σ]
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in) -> (..., d_out)
        # 因为权重形状是 (d_out, d_in)，我们需要计算 x @ weight.T。
        # 使用 einsum 使维度操作更明确。
        return torch.einsum("... i, o i -> ... o", x, self.weight)
    
class Embedding(nn.Module):
    """
    一个可学习的嵌入查找表，等同于 torch.nn.Embedding。

    该模块将整数 token ID 映射为固定维度的连续向量（embedding_dim）。
    嵌入矩阵存储为形状为 (num_embeddings, embedding_dim) 的可学习参数。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        # 权重矩阵形状为 (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype)
        )

        # 初始化：N(0, 1)，截断至 [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...) 整数 -> 输出: (..., d_model)
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    """
    均方根层归一化 (RMSNorm)。

    对于输入向量 a ∈ R^{d_model}：
        RMS(a) = sqrt(mean(a^2) + eps)
        RMSNorm(a) = (a / RMS(a)) * g    
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)

        # 可学习的增益参数 (g)，形状为 (d_model,)
        self.weight = nn.Parameter(torch.ones((self.d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # 在最后一个维度上计算 RMS：sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 归一化并应用增益；在 float32 下进行计算，然后转换回原始类型
        y = (x_fp32 / rms) * self.weight.to(torch.float32)

        return y.to(in_dtype)
    
class SwiGLU(nn.Module):
    """
    使用 SwiGLU 激活函数的位置逐层前馈网络。

    变换公式为：
        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    其中 SiLU(z) = z * sigmoid(z)，⊙ 表示逐元素乘法。

    形状：
        输入：  (..., d_model)
        W1, W3: (d_ff, d_model)   实现为 Linear(d_model -> d_ff)
        W2:     (d_model, d_ff)   实现为 Linear(d_ff -> d_model)
        输出：  (..., d_model)
    """

    @staticmethod
    def round_up_to_multiple(x: int, multiple: int) -> int:
        """将 x 向上舍入到最接近的 `multiple` 的正倍数。"""
        if multiple <= 0:
            raise ValueError("multiple must be a positive integer")
        return int(((x + multiple - 1) // multiple) * multiple)

    @staticmethod
    def default_d_ff(d_model: int, multiple_of: int = 64) -> int:
        """
        计算推荐的 SwiGLU 隐藏层大小。

        我们使用 d_ff ~= (8/3) * d_model，然后向上舍入到硬件友好的倍数（通常为 64）。
        """
        raw = int(math.ceil((8.0 * d_model) / 3.0))
        return SwiGLU.round_up_to_multiple(raw, multiple_of)

    def __init__(
        self, d_model: int, d_ff: int | None = None, *, multiple_of: int = 64,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff) if d_ff is not None else self.default_d_ff(self.d_model, multiple_of)

        # 两个上投影和一个下投影（无偏置）
        # TODO：验证w1和w3是不是升高维
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        gated = self.silu(a) * b
        return self.w2(gated)
    
class RoPE(nn.Module):
    """
    旋转位置嵌入 (RoPE)。

    对输入张量的最后一个维度 (d_k) 应用与位置相关的旋转。
    旋转成对应用于 (x[..., 0], x[..., 1], x[..., 2], x[..., 3]) 等。

    该模块没有可学习参数。它可以预计算并缓存 cos/sin 表。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got d_k={d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        
        self.theta = float(theta) # 旋转频率参数：决定不同维度的角频率
        self.d_k = int(d_k) # 输入向量的维度
        self.max_seq_len = int(max_seq_len) # 支持的最大序列长度：用于预计算 cos/sin 表（需要准备多少个位置的角度值）

        # 预计算偶数索引的逆频率：
        # inv_freq[j] = theta^(-2j/d_k)，其中 j 是对的索引 (0, 1, ..., d_k/2 - 1)。
        pair_idx = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (-pair_idx / self.d_k)

        # 位置 [0, 1, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        # 角度：(max_seq_len, d_k/2)
        angles = positions[:, None] * inv_freq[None, :]

        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)

        # 作为非持久性缓冲区缓存（不保存在 state_dict 中）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 (..., seq_len, d_k) 的张量
            token_positions: 形状为 (..., seq_len) 的整数位置张量

        Return:
            应用 RoPE 后的形状为 (..., seq_len, d_k) 的张量。
        """
        if x.size(-1) != self.d_k:
            raise ValueError(f"Expected x.size(-1)==d_k=={self.d_k}, got {x.size(-1)}")        
        
        # token_positions 用于沿序列维度切片缓存的 cos/sin。
        # 索引后的形状：(..., seq_len, d_k/2)
        pos = token_positions.to(device=x.device)
        cos = self.cos.index_select(0, pos.reshape(-1)).reshape(*pos.shape, -1)
        sin = self.sin.index_select(0, pos.reshape(-1)).reshape(*pos.shape, -1)

        # 提升到 float32 以获得数值稳定性，然后转换回原始类型
        x_fp32 = x.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        # 分离偶数和奇数索引：形状为 (..., seq_len, d_k/2)
        x_even = x_fp32[..., ::2]
        x_odd = x_fp32[..., 1::2]

        # 对每一对应用 2D 旋转。
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 将偶数/奇数交错回 (..., seq_len, d_k)
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)

        return out.to(dtype=x.dtype)
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    在给定维度上进行数值稳定的 softmax。

    Args:
        x (torch.Tensor): 输入张量。
        dim (int): 应用 softmax 的维度。

    Return:
        torch.Tensor: 与 `x` 具有相同形状/数据类型/设备的 softmax 输出。
    """
    # 减去最大值以保证数值稳定性（使exp最大值为1）
    x_max = torch.amax(x, dim=dim, keepdim=True) # （使用 keepdim 以便正确广播）
    z = x - x_max

    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=dim, keepdim=True)

    return exp_z / sum_exp

def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    缩放点积注意力。

    Args:
        query: 形状为 (..., seq_len, d_k) 的张量
        key:   形状为 (..., seq_len, d_k) 的张量
        value: 形状为 (..., seq_len, d_v) 的张量
        mask:  形状为 (seq_len, seq_len) 的可选布尔张量，其中 True 表示
               允许该位置，False 表示被遮掩。

    Return:
        形状为 (..., seq_len, d_v) 的张量
    """
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError("query/key/value must have shape (..., seq_len, d_*)")

    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
            raise ValueError("batch dimensions of query, key, value must match")

    d_k = query.shape[-1]
    if d_k != key.shape[-1]:
        raise ValueError("query and key must have the same d_k")
    
    # 在 float32 中计算注意力 logits 以保证稳定性
    q = query.to(torch.float32)    
    k = key.to(torch.float32)
    v = value.to(torch.float32)

    scale = 1.0 / math.sqrt(d_k)

    # logits: (..., seq_len, seq_len) -> 序列中每个词对其他词的权重
    logits = torch.einsum("... s d, ... t d -> ... s t", q, k) * scale
    
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")

        # 当 mask 中对应位置为 True(1) 时，就使用 logits 中对应位置的值；
        # 当 mask 中对应位置为 False(0) 时，就使用 neg_inf 替换 logits 中对应位置的值，最后将结果重新赋值给 logits
        neg_inf = torch.finfo(torch.float32).min # 使用 float32 的最小值作为 -inf
        logits = torch.where(mask.to(device=logits.device), logits, neg_inf)        
    
    # probs: (..., seq_len, seq_len)
    probs = softmax(logits, dim=-1)

    if mask is not None:
        # 确保遮掩位置精确为零（虽然 softmax(-inf) 应该是 0，
        # 但这使得在极端值下的行为更健壮）。
        probs = probs * mask.to(device=probs.device, dtype=probs.dtype)
    
    # out: (..., seq_len, d_v)
    out = torch.einsum("... s t, ... t d -> ... s d", probs, v)

    # 转换回原始的 value 数据类型
    return out.to(dtype=value.dtype)

class CausalMultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力（无 RoPE）。

    该模块计算：
        Q = W_Q x, K = W_K x, V = W_V x
        heads = SDPA(Q_heads, K_heads, V_heads, causal_mask)
        out = W_O concat(heads)

    形状：
        x:   (..., seq_len, d_model)
        QKV: (..., seq_len, d_model)
        heads 视图: (..., num_heads, seq_len, head_dim)
        输出: (..., seq_len, d_model)
    """
    def __init__(
            self, d_model: int, num_heads: int, 
            device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.head_dim = self.d_model // self.num_heads  # d_k = d_v = d_model / h

        # 优化：合并 QKV 投影以提高计算效率
        self.qkv_proj = Linear(self.d_model, 3 * self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构建一个 (seq_len, seq_len) 的因果掩码，其中 True 表示“允许”
        """
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)) # tril：下三角矩阵（主对角线以上全为0）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arg s
            x: 形状为 (..., seq_len, d_model) 的张量

        Return:
            形状为 (..., seq_len, d_model) 的张量
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.size(-1)}")

        seq_len = x.size(-2)    
        device = x.device

        # 一次性投影并切分
        qkv = self.qkv_proj(x)  # (..., seq_len, 3 * d_model)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)

        # 重塑为多头：(..., seq_len, num_heads, head_dim)
        # 然后将头移动到类似 batch 的维度：(..., num_heads, seq_len, head_dim)
        new_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = q.view(new_shape).transpose(-3, -2)
        k = k.view(new_shape).transpose(-3, -2)
        v = v.view(new_shape).transpose(-3, -2)

        # 在头和 batch 之间共享的因果掩码
        mask = self._causal_mask(seq_len, device=device)

        # SDPA: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 合并多头: (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        # 输出投影: (..., seq_len, d_model)
        return self.output_proj(out)

class CausalMultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    对 Q 和 K（而非 V）应用 RoPE 的因果多头自注意力。

    此版本使用融合的 QKV 投影：
        qkv = W_qkv x
        q, k, v = split(qkv)
    """
    def __init__(
            self, d_model: int, num_heads: int, theta: float, max_seq_len: int,
            device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.head_dim = self.d_model // self.num_heads

        # 独立的投影（匹配参考 state_dict 的键）。
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        # RoPE 在每个头的维度上操作
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """构建一个 (seq_len, seq_len) 的因果掩码，其中 True 表示“允许”。"""
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 (..., seq_len, d_model) 的张量
            token_positions: 形状为 (..., seq_len) 的张量

        Return:
            形状为 (..., seq_len, d_model) 的张量
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.size(-1)}")

        seq_len = x.size(-2)    
        device = x.device

        # 投影到 Q, K, V: (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头：(..., seq_len, num_heads, head_dim)
        # 然后转置为 (..., num_heads, seq_len, head_dim)
        new_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = q.view(new_shape).transpose(-3, -2)
        k = k.view(new_shape).transpose(-3, -2)
        v = v.view(new_shape).transpose(-3, -2)

        # 对每个头的 Q 和 K 应用 RoPE（头被视为类似 batch 的维度）
        q = self.rope(q, token_positions.unsqueeze(-2))
        k = self.rope(k, token_positions.unsqueeze(-2))

        # 在头和 batch 之间共享的因果掩码
        mask = self._causal_mask(seq_len, device=device)

        # 注意力: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 合并多头回: (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        return self.output_proj(out)
    
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer 块。

    结构 (pre-norm)：
        y = x + Attn(RMSNorm(x))
        z = y + FFN(RMSNorm(y))
    
    此块使用带有 RoPE 的因果多头自注意力。
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, *,
        max_seq_len: int, theta: float, eps: float = 1e-5,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.ln1 = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttentionWithRoPE(
            d_model=self.d_model,
            num_heads=self.num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
    
        self.ln2 = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 (batch, seq_len, d_model) 的张量
            token_positions: 形状为 (batch, seq_len) 或可广播到该形状的张量

        Return:
            形状为 (batch, seq_len, d_model) 的张量
        """
        # Pre-norm 注意力 + 残差
        h = self.ln1(x)
        x = x + self.attn(h, token_positions)

        # Pre-norm FFN + 残差
        h = self.ln2(x)
        x = x + self.ffn(h)

        return x