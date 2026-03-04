import os
import time
import random
import numpy as np
from typing import Iterator, TextIO

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"

def iter_docs_by_eot(f: TextIO, eot: str) -> Iterator[str]:
    buf = ""
    for chunk in f:
        buf += chunk
        while True:
            idx = buf.find(eot)
            if idx < 0:
                break
            doc = buf[:idx]
            buf = buf[idx + len(eot) :]
            if doc.strip():
                yield doc
    # 将剩余部分作为最后一个文档发送（如果不为空）。
    if buf.strip():
        yield buf


def reservoir_sample_docs(path: str, eot: str, k: int, seed: int) -> list[str]:
    '''
    从文本文件中流式采样 k 个文档，使用 EOT 标记分隔文档。
    '''
    rnd = random.Random(seed)
    sample: list[str] = []
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for doc in iter_docs_by_eot(f, eot):
            n += 1
            if len(sample) < k:
                sample.append(doc)
            else:
                j = rnd.randrange(n)
                if j < k:
                    sample[j] = doc
    return sample

def bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    '''
    计算给定文档列表的平均每 token 字节数(总字节数/总token数)。
    '''
    total_bytes = 0
    total_tokens = 0
    for d in docs:
        b = len(d.encode("utf-8"))
        ids = tokenizer.encode(d)
        total_bytes += b
        total_tokens += len(ids)
    bpt = total_bytes / max(1, total_tokens)
    return bpt, total_bytes, total_tokens

def iter_first_n_bytes_as_lines(path: str, nbytes: int) -> Iterator[str]:
    seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                break
            seen += len(line.encode("utf-8"))
            yield line
            if seen >= nbytes:
                break

def measure_throughput_bytes_per_sec(tokenizer: Tokenizer, it: Iterator[str], repeats: int = 1) -> float:
    lines = list(it)
    total_bytes = sum(len(x.encode("utf-8")) for x in lines)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for _id in tokenizer.encode_iterable(lines):
            pass
    t1 = time.perf_counter()

    secs = (t1 - t0) / max(1, repeats)
    return total_bytes / max(1e-9, secs)

def encode_to_uint16_bin(tokenizer: Tokenizer, input_path: str, output_path: str, *, chunk_tokens: int = 1_000_000) -> None:
    """
    流式编码大型文本文件为 uint16 token ID，并写入 .bin 文件

    这避免了将整个 token 序列存储在内存中，通过缓冲固定数量的 token ID 并定期刷新到磁盘来实现
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 使用追加二进制模式，以便我们可以多次刷新数据块
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, open(output_path, "wb") as fout:
        buf = np.empty(chunk_tokens, dtype=np.uint16)
        n = 0

        for tid in tokenizer.encode_iterable(fin):
            # 安全检查：确保 tid 在 uint16 范围内
            if tid < 0 or tid > 65535:
                raise ValueError(f"token id {tid} out of uint16 range")
            
            buf[n] = tid
            n += 1

            if n == chunk_tokens:
                fout.write(buf.tobytes())
                n = 0
        
        # 刷新尾部数据
        if n:
            fout.write(buf[:n].tobytes())

def encode_to_int32_bin(tokenizer: Tokenizer, input_path: str, output_path: str, *, chunk_tokens: int = 1_000_000) -> None:
    """
    流式编码大型文本文件为 int32 token ID，并写入 .bin 文件

    这避免了将整个 token 序列存储在内存中，通过缓冲固定数量的 token ID 并定期刷新到磁盘来实现
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 使用追加二进制模式，以便我们可以多次刷新数据块
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, open(output_path, "wb") as fout:
        buf = np.empty(chunk_tokens, dtype=np.int32)
        n = 0

        for tid in tokenizer.encode_iterable(fin):
            # 安全检查：确保 tid 在 int32 范围内
            if tid < -2147483648 or tid > 2147483647:
                raise ValueError(f"token id {tid} out of int32 range")
            
            buf[n] = tid
            n += 1

            if n == chunk_tokens:
                fout.write(buf.tobytes())
                n = 0
        
        # 刷新尾部数据
        if n:
            fout.write(buf[:n].tobytes())

def main():
    # 1) 训练好的分词器路径
    tin_vocab_path = "workspace/tinystories_bpe_vocab_10000.pkl"
    tin_merges_path = "workspace/tinystories_bpe_merges_10000.pkl"

    owt_vocab_path = "workspace/owt_bpe_vocab_32000.pkl"
    owt_merges_path = "workspace/owt_bpe_merges_32000.pkl"

    # 2) 数据集路径
    tinystories_path = "data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "data/owt_train.txt"

    # 3) 加载分词器
    tin_tok = Tokenizer.from_files(tin_vocab_path, tin_merges_path, special_tokens=[EOT])
    owt_tok = Tokenizer.from_files(owt_vocab_path, owt_merges_path, special_tokens=[EOT])

    # 4) 从每个语料库中采样 10 个文档，而无需加载整个文件
    seed = 42
    print("Sampling TinyStories docs (streaming)...")
    tin_docs = reservoir_sample_docs(tinystories_path, EOT, k=10, seed=seed)
    print("Sampling OWT docs (streaming)...")
    owt_docs = reservoir_sample_docs(owt_path, EOT, k=10, seed=seed)

    # (a) 领域内压缩效率
    tin_on_tin, tin_bytes, tin_tokens = bytes_per_token(tin_tok, tin_docs)
    owt_on_owt, owt_bytes, owt_tokens = bytes_per_token(owt_tok, owt_docs)

    # (b) 跨领域压缩效率
    tin_on_owt, x_bytes, x_tokens = bytes_per_token(tin_tok, owt_docs)

    print("\n=== (a) In-domain bytes/token ===")
    print(f"TinyStories tokenizer on TinyStories: {tin_on_tin:.4f} bytes/token "
          f"(bytes={tin_bytes}, tokens={tin_tokens})")
    print(f"OWT tokenizer on OWT:               {owt_on_owt:.4f} bytes/token "
          f"(bytes={owt_bytes}, tokens={owt_tokens})")

    print("\n=== (b) Cross-domain bytes/token ===")
    print(f"TinyStories tokenizer on OWT:       {tin_on_owt:.4f} bytes/token "
          f"(bytes={x_bytes}, tokens={x_tokens})")

    # (c) 吞吐量估算
    # 使用语料库的一小部分前缀，以避免过长的测量时间

    sample_for_speed = iter_first_n_bytes_as_lines(owt_path, nbytes=5_000_000)  # 5MB
    thr = measure_throughput_bytes_per_sec(owt_tok, sample_for_speed, repeats=1)
    total_bytes_82gb = 82 * (1024 ** 3)
    est_secs = total_bytes_82gb / thr
    est_hours = est_secs / 3600

    print("\n=== (c) Throughput estimate ===")
    print(f"Measured throughput (OWT tokenizer): {thr:.2f} bytes/s")
    print(f"Estimated time for 82GB: {est_hours:.2f} hours")

    # (d) 序列化 token ID 用于语言模型训练
    # print("\n=== (d) Encoding datasets to uint16 ===")
    
    # encode_to_uint16_bin(tin_tok, "data/TinyStoriesV2-GPT4-train.txt", "workspace/bin/uint16/tinystories_train.uint16.bin")
    # encode_to_uint16_bin(tin_tok, "data/TinyStoriesV2-GPT4-valid.txt", "workspace/bin/uint16/tinystories_valid.uint16.bin")

    # encode_to_uint16_bin(owt_tok, "data/owt_train.txt", "workspace/bin/uint16/owt_train.uint16.bin")
    # encode_to_uint16_bin(owt_tok, "data/owt_valid.txt", "workspace/bin/uint16/owt_valid.uint16.bin")

    # print("Done. Saved uint16 .bin files under workspace/.")

    print("\n=== (d) Encoding datasets to int32 ===")
    
    encode_to_int32_bin(tin_tok, "data/TinyStoriesV2-GPT4-train.txt", "workspace/bin/int32/tinystories_train.int32.bin")
    encode_to_int32_bin(tin_tok, "data/TinyStoriesV2-GPT4-valid.txt", "workspace/bin/int32/tinystories_valid.int32.bin")

    encode_to_int32_bin(owt_tok, "data/owt_train.txt", "workspace/bin/int32/owt_train.int32.bin")
    encode_to_int32_bin(owt_tok, "data/owt_valid.txt", "workspace/bin/int32/owt_valid.int32.bin")

    print("Done. Saved int32 .bin files under workspace/.")

if __name__ == "__main__":
    main()