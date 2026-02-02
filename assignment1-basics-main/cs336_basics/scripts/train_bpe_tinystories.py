import os
import time
import pickle
import psutil

from cs336_basics.train_bpe import train_bpe

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    proc = psutil.Process(os.getpid()) # 获取当前进程

    print(f"Starting BPE training on TinyStories with vocab size {vocab_size}...")
    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=8
    )
    t1 = time.perf_counter()

    rss_gb = proc.memory_info().rss / (1024 ** 3) # 监控内存消耗（以 GB 为单位）

    # 保存到磁盘
    vocab_path = os.path.join(output_dir, "tinystories_bpe_vocab_10000.pkl")
    merges_path = os.path.join(output_dir, "tinystories_bpe_merges_10000.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # 词表中字节长度最长的 token
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    elapsed_s = t1 - t0
    elapsed_min = elapsed_s / 60.0
    elapsed_hr  = elapsed_s / 3600.0

    print(f"Saved vocab -> {vocab_path}")
    print(f"Saved merges -> {merges_path}")
    print(f"Elapsed: {elapsed_s:.2f}s ({elapsed_min:.2f} min, {elapsed_hr:.4f} hour)")
    print(f"RSS (approx): {rss_gb:.2f} GB")
    print(f"Longest token id={longest_id}, bytes_len={len(longest_bytes)}")
    print(f"Longest token (decoded): {repr(longest_str)}")

if __name__ == "__main__":
    main()