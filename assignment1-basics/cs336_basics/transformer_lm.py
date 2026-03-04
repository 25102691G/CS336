import torch
from torch import nn
from cs336_basics.modules import Embedding, Linear, RMSNorm, TransformerBlock

class TransformerLM(nn.Module):
    """
    一个Transformer语言模型，由以下部分组成：
      词元嵌入 -> N个预归一化Transformer块 -> 最终RMSNorm -> 语言模型头。

    此实现在每个TransformerBlock的注意力模块内部使用RoPE。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        *,
        rope_theta: float,
        max_seq_len: int | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)    
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len if max_seq_len is not None else context_length)

        # 词元嵌入表：(vocab_size, d_model)
        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device=device, dtype=dtype)

        # 预归一化Transformer块堆叠
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=self.max_seq_len,
                    theta=rope_theta,
                    eps=eps,
                    device=device,
                    dtype=dtype
                )
                for _ in range(self.num_layers)
            ]
        )

        # 在语言模型头之前的最终归一化
        self.ln_final = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)

        # 输出投影到词汇表logits（输出每个词的概率）：权重形状 (vocab_size, d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        参数:
            in_indices: 形状为 (batch, seq_len) 的LongTensor

        返回:
            logits: 形状为 (batch, seq_len, vocab_size) 的张量
        """
        if in_indices.dim() != 2:
            raise ValueError(f"in_indices必须具有形状(batch, seq_len)，但得到{tuple(in_indices.shape)}")

        batch, seq_len = in_indices.shape
        if seq_len > self.context_length:
            raise ValueError(f"seq_len={seq_len}超过了context_length={self.context_length}")

        # RoPE的词元位置：(batch, seq_len)
        token_positions = torch.arange(seq_len, device=in_indices.device, dtype=torch.long).view(1, seq_len)
        token_positions = token_positions.expand(batch, seq_len)

        # 嵌入词元：(batch, seq_len, d_model)
        x = self.token_embeddings(in_indices)

        # 应用Transformer块
        for block in self.layers:
            x = block(x, token_positions)
        
        # 最终归一化和词汇表投影
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits