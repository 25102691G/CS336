import gc
import time
import torch
from typing import Sequence
import cs336_basics.config as config_mod
import cs336_basics.train as train_mod

def run_one(lr_max: float, *, lr_min_ratio: float = 0.1, tag: str = "") -> None:
    """
    通过修补 train.get_default_config() 来运行一个训练任务，使其返回一个修改后的配置。
    """
    cfg = config_mod.get_default_config()

    # ---- 覆盖扫描所需的配置 ----
    cfg.optim.lr_max = float(lr_max)
    cfg.optim.lr_min = float(lr_max * lr_min_ratio)  # 重要：显式设置（不要依赖数据类的默认值）
    lr_tag = "lr" + f"{lr_max:.4g}".replace(".", "p").replace("-", "m")
    cfg.run.run_name = f"{lr_tag}{('_' + tag) if tag else ''}"

    # ---- 修补 train_mod.get_default_config ----
    def _patched_get_default_config():
        return cfg

    train_mod.get_default_config = _patched_get_default_config

    print("\n" + "=" * 80)
    print(f"[SWEEP] start run: lr_max={cfg.optim.lr_max} lr_min={cfg.optim.lr_min} run_name={cfg.run.run_name}")
    print("=" * 80)

    # 运行训练
    train_mod.main()

    # 清理以减少跨次运行的 GPU 内存碎片
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(lrs: Sequence[float]) -> None:
    t0 = time.time()
    for lr in lrs:
        run_one(lr, lr_min_ratio=0.1)
    dt = time.time() - t0
    print(f"\n[SWEEP] all done. total wall time: {dt/60:.1f} min")

if __name__ == "__main__":
    # 粗略搜索
    # lrs = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2]
    # 精细搜索
    # lrs = [5e-4, 8e-4, 1.0e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2.0e-3, 2.2e-3]
    # 发散
    lrs = [1e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1e-1]

    main(lrs)