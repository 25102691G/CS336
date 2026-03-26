import argparse
import json
import time
from typing import Dict, Any
import os
import random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_baseline import evaluate_vllm

from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, log_generations

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    '''初始化 vLLM 实例，设置随机种子，并应用必要的补丁以避免分布式环境和 profiling 相关的错误。'''
    vllm_set_random_seed(seed)
    world_size_path = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_path, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            # dtype=torch.bfloat16,
            dtype=torch.float16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm: LLM):
    '''将当前训练的 policy 模型权重加载到 vLLM 实例中，以便在评估阶段使用最新的模型进行生成和奖励计算。'''
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


class SFTDataset(Dataset):
    def __init__(self, path: str, limit: int = 0, seed: int = 0):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        if limit and limit > 0:
            rnd = random.Random(seed)
            rnd.shuffle(self.data)
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["prompt"], ex["response"], ex
    

def collate_fn(batch, tokenizer):
    '''批处理函数：将原始的 prompt-response 对转换为模型输入格式，包括编码、构建 attention mask、生成 labels 和 response_mask 等。输入是一个包含多个 (prompt, response) 的列表，输出是一个字典，包含 input_ids、labels、response_mask 等张量，准备好用于模型训练。'''
    prompts = [x[0] for x in batch]
    outputs = [x[1] for x in batch]

    toks = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    return toks


def build_math_val_prompts_and_gts(val_path: str, prompt_file: str, max_examples: int = 0):
    '''构建 MATH 验证集的提示词和对应的 ground truth 答案列表，使用指定的 prompt 模板格式化每个问题。'''
    prompt_template = Path(prompt_file).read_text(encoding="utf-8")

    val = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val.append(json.loads(line))

    if max_examples and max_examples > 0:
        val = val[:max_examples]

    prompts, gts = [], []
    for ex in val:
        q = ex.get("problem") or ex.get("question") or ex.get("prompt")
        gt = ex.get("answer") or ex.get("ground_truth") or ex.get("target")
        if q is None or gt is None:
            raise KeyError(f"Validation example missing question/answer fields: keys={list(ex.keys())}")
        prompts.append(prompt_template.format(question=q))
        gts.append(gt)

    return prompts, gts    


def filter_correct_sft_samples(data_path: str, out_path: str):
    """
    过滤：筛选出 SFT 数据集中的正确答案样本，基于 r1_zero_reward_fn 计算奖励分数，并仅保留那些奖励分数达到正确答案阈值（如 1.0）的样本。最终将过滤后的样本写入新的 JSONL 文件，并返回过滤统计信息。
    """
    kept = []
    total = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            total += 1
            gt = ex.get("answer") or ex.get("ground_truth")
            if gt is None:
                raise RuntimeError(
                    "sft.jsonl does not contain ground-truth fields (answer/ground_truth). "
                )
            resp = ex["response"]
            scores = r1_zero_reward_fn(resp, gt)
            if float(scores.get("answer_reward", 0.0)) >= 1.0:
                kept.append(ex)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in kept:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    return {"filtered/kept": len(kept), "filtered/total": total}


def main():
    # 1. 解析命令行参数（用户可自定义训练配置）
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="cs336_alignment/data/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--sft_path", default="cs336_alignment/data/MATH/sft.jsonl")
    ap.add_argument("--val_path", default="cs336_alignment/data/MATH/validation.jsonl")
    ap.add_argument("--prompt_file", default="cs336_alignment/prompts/r1_zero.prompt")

    ap.add_argument("--train_device", default="cuda:0")
    ap.add_argument("--vllm_device", default="cuda:1")

    ap.add_argument("--train_samples", type=int, default=0, help="0 means full dataset")
    ap.add_argument("--filter_correct", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_acc_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/sft_experiment")
    ap.add_argument("--eval_max_examples", type=int, default=500)
    args = ap.parse_args()

    # 2. 初始化日志系统（所有训练/评估事件都会记录到 log.jsonl）
    run_dir = Path(args.out_dir) / f"samples{args.train_samples or 'full'}_{'filtered' if args.filter_correct else 'all'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    opt_step = 0  # counts optimizer updates
    step = 0
    micro_idx = 0    

    def log_event(event: Dict[str, Any], *, also_print: bool = True):
        """
        将训练 / 评估事件（loss、指标、步骤）写入 JSONL 文件，可选打印到终端
        """
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "time": time.time(),
            "step": step,
            "micro_idx": micro_idx,
            "opt_step": opt_step,
            **event,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()

        if also_print:
            # keep terminal readable
            if "msg" in event:
                print(event["msg"])
            else:
                print(payload)    

    # 3. 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 4. 自动检测设备配置
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("未检测到 GPU，此脚本需要 GPU 支持")
    
    if args.train_device is None:
        args.train_device = "cuda:0"
    
    if args.vllm_device is None:
        if gpu_count >= 2:
            args.vllm_device = "cuda:1"
        else:
            args.vllm_device = "cuda:0"
            print(f"警告：只有 1 块 GPU，训练和 vLLM 推理将共享同一设备 ({args.vllm_device})")
    
    # 5. 验证设备是否可用
    for device in [args.train_device, args.vllm_device]:
        device_idx = int(device.split(":")[1]) if ":" in device else 0
        if device_idx >= gpu_count:
            raise RuntimeError(f"设备 {device} 不存在，系统只有 {gpu_count} 块 GPU")

    # 6. 加载预训练模型作为初始 policy，准备进行 SFT 微调
    model_path = Path(args.model_id).resolve() if os.path.exists(args.model_id) else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    
    # 5. 加载预训练模型作为初始 policy，准备进行 SFT 微调
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # torch_dtype=torch.bfloat16, # 显存占用大
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2", # 需要算力大于8.0的GPU
    ).to(args.train_device)
    policy.train()

    # 6. 初始化 vLLM 实例，用于后续评估阶段的高效文本生成和奖励计算
    llm = init_vllm(args.model_id, device=args.vllm_device, seed=args.seed)

    # 7. 加载验证集（用于评估模型效果）
    eval_prompts, eval_gts = build_math_val_prompts_and_gts(
        args.val_path, args.prompt_file, max_examples=args.eval_max_examples
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 8. （可选）过滤训练数据：仅保留模型回答正确的样本
    data_path = args.sft_path
    if args.filter_correct:
        filtered_path = str(Path(args.out_dir) / "filtered_sft.jsonl")
        stats = filter_correct_sft_samples(args.sft_path, filtered_path)
        log_event({"type": "filter_stats", "stats": stats, "msg": f"Filter stats: {stats}"})
        data_path = filtered_path

    # 9. 加载训练数据集
    dataset = SFTDataset(data_path, limit=args.train_samples, seed=args.seed)

    # 10. 构建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    # 11. 初始化优化器
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # training loop
    opt.zero_grad(set_to_none=True)

    # 12. 训练循环
    for epoch in range(10_000_000):
        for batch in loader:
            # 更新全局步骤计数器
            step += 1
            micro_idx += 1

            # 数据移动到训练设备
            input_ids = batch["input_ids"].to(args.train_device)
            labels = batch["labels"].to(args.train_device)
            response_mask = batch["response_mask"].to(args.train_device)

            # 计算模型输出（回答部分的对数概率）
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = out["log_probs"]            

            # 微批次训练（计算损失 + 反向传播)
            loss, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_acc_steps,
                normalize_constant=1.0,
            )

            # 优化器周期性更新
            if micro_idx % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    log_event({"type": "train_loss", "loss": float(loss.detach())}, also_print=False)

            # 周期性评估
            if step % args.eval_interval == 0:
                policy.eval()
                with torch.no_grad():
                    load_policy_into_vllm_instance(policy, llm)
                    rows = evaluate_vllm(
                        vllm_model=llm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=eval_prompts,
                        ground_truths=eval_gts,
                        eval_sampling_params=eval_sampling_params,
                        request_batch_size=64,
                    )

                # generation log records
                # gen_log = log_generations(
                #     model=policy,
                #     tokenizer=tokenizer,
                #     prompts=eval_prompts[:8],
                #     ground_truths=eval_gts[:8],
                #     reward_fn=r1_zero_reward_fn,
                #     num_log=8,
                #     step=step,
                #     stop_str="</answer>",
                #     max_new_tokens=512,
                #     temperature=0.0,
                # )

                # log_event({"type": "gen_stats", "gen_stats": gen_log["stats"], "msg": f"gen stats: {gen_log['stats']}"})

                n = len(rows)
                eval_acc = sum(r.answer_reward for r in rows) / n if n else 0.0
                eval_format = sum(r.format_reward for r in rows) / n if n else 0.0
                eval_reward = sum(r.reward for r in rows) / n if n else 0.0
                metrics = {
                    "eval/accuracy": eval_acc,
                    "eval/format_rate": eval_format,
                    "eval/avg_reward": eval_reward,
                    "eval/n": n,                    
                }
                log_event({"type": "eval_metrics", "loss": float(loss.detach()), "metrics": metrics,
                        "msg": f"[step={step}] loss={float(loss.detach()):.4f} {metrics}"})
                policy.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # save
    policy.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    log_event({"type": "save", "out_dir": str(run_dir), "msg": f"Saved: {run_dir}"})


if __name__ == "__main__":
    main()
