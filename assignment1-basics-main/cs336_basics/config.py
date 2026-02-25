from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    # 内存映射的 token 文件（一维二进制文件）
    train_data_path: str = "workspace/bin/int32/tinystories_train.int32.bin"
    val_data_path: str = "workspace/bin/int32/tinystories_valid.int32.bin"
    # 创建 token 文件时使用的 Numpy 数据类型
    np_dtype: str = "int32"

    context_length: int = 256

    # get_batch() 使用的设备字符串
    # device: str = "cuda:0"    # Windows/Linux GPU
    device: str = "mps"         # macOs

@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # 如果为 None，将在模型构建时默认为 4 * d_model
    d_ff: Optional[int] = None

    rope_theta: float = 10_000.0
    # 如果为 None，模型将使用 context_length
    max_seq_len: Optional[int] = None

    rmsnorm_eps: float = 1e-5

    # 用于模型参数的 torch 数据类型字符串
    torch_dtype: str = "float32"

@dataclass
class OptimizerConfig:
    lr_max: float = 3e-4
    lr_min: float = 3e-5

    warmup_iters: int = 200
    cosine_cycle_iters: int = 10_000
    
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1

    grad_clip: float = 1.0

@dataclass
class TrainingConfig:
    max_steps: int = 10_000
    batch_size: int = 64
    
    log_interval: int = 50
    eval_interval: int = 500
    eval_batches: int = 20

    ckpt_interval: int = 1000
    ckpt_path: str = "checkpoints/ckpt.pt"
    resume_from: Optional[str] = "checkpoints/ckpt.pt" # 如果存在，则从此检查点恢复，否则从头开始训练(设置为0)

    seed: int = 0

@dataclass
class WandbConfig:
    enable: bool = False
    project: str = "cs336-a1"
    run_name: str = "train"

@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

def get_default_config() -> TrainConfig:
    """
    返回默认的训练配置。
    """
    cfg = TrainConfig()

    # 默认保持模型/数据的 context_length 一致
    cfg.model.context_length = cfg.data.context_length

    return cfg