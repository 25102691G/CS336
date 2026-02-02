import os
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

# GPT-2 风格的正则表达式预分词器（需要第三方 `regex` 库）
'''
1. 缩写形式：'(?:[sdmt]|ll|ve|re)
    匹配常见的英文缩写，如 's, 't, 're, 've 等
2. 字母序列：?\p{L}+
    匹配一个或多个连续的字母（Unicode字母）
3. 数字序列：?\p{N}+
    匹配一个或多个连续的数字
4. 特殊字符：?[^\s\p{L}\p{N}]+
    匹配非空格、非字母、非数字的标点符号等
5. 空白字符：\s+(?!\S)|\s+
    匹配行尾的空白字符或普通空白字符序列
'''
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 每个工作进程都会编译自己的正则表达式实例
RX = None

def init_worker():
    '''
    初始化正则表达式实例。
    '''
    global RX
    RX = re.compile(GPT2_PRETOKENIZE_PATTERN)

def count_word_freq_from_text(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    对文本块执行预分词：
      1) 按特殊标记拆分（特殊标记不参与训练统计）
      2) 应用 GPT-2 正则表达式预分词
      3) 将每个部分编码为 UTF-8 字节并拆分为单字节 token
      4) 统计 token 序列频率

    Return:
        dict[tuple[bytes, ...], int]: { (H, e, l, l, o): 5, ... }
    """
    if not text:
        return {}

    # 按每个特殊标记拆分；从训练统计中剔除特殊标记
    spans = [text]
    for s_tok in special_tokens:
        new_spans: list[str] = []
        for sp in spans:
            if sp:
                new_spans.extend(sp.split(s_tok))
        spans = new_spans
    
    # 对每个 token 进行编码并统计频率
    word_freq: dict[tuple[bytes, ...], int] = {}
    for sp in spans:
        if not sp:
            continue
        for m in RX.finditer(sp):
            piece = m.group(0)
            if not piece:
                continue
            bts = piece.encode("utf-8")
            key = tuple(bytes([b]) for b in bts)
            word_freq[key] = word_freq.get(key, 0) + 1
    return word_freq

def process_chunk(args) -> dict[tuple[bytes, ...], int]:
    """
    处理单个文件块的工作进程入口点。
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # 解码时忽略错误，以避免 UTF-8 边界问题
    text = chunk.decode("utf-8", errors="ignore")
    return count_word_freq_from_text(text, special_tokens)

def build_word_freq_serial(input_path : str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    init_worker()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return count_word_freq_from_text(text, special_tokens)

def build_word_freq_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
    *,
    num_chunks: int | None = None
) -> dict[tuple[bytes, ...], int]:
    """
    使用多进程构建词频统计。
    块边界与特殊标记边界对齐。
    """
    if num_processes <= 1 or not special_tokens:
        return build_word_freq_serial(input_path, special_tokens)
    
    if num_chunks is None:
        num_chunks = max(num_processes * 32, num_processes)

    split_special_token = special_tokens[0].encode("utf-8")  # 例如 b"<|endoftext|>"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)
    
    tasks = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    merged = Counter()
    with Pool(processes=num_processes, initializer=init_worker, maxtasksperchild=8) as pool:
        for partial in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            merged.update(partial)
    
    return dict(merged)

def pairs_in_word(word: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    """
    统计单个单词序列中相邻对的出现次数。
    """
    counts: dict[tuple[bytes, bytes], int] = {}
    if len(word) < 2:
        return counts
    prev = word[0]
    for cur in word[1:]:
        p = (prev, cur)
        counts[p] = counts.get(p, 0) + 1
        prev = cur
    return counts

def apply_merge(word: tuple[bytes, ...], a: bytes, b: bytes, new_token: bytes) -> tuple[bytes, ...]:
    """
    将出现的 (a,b) 替换为 new_token。
    """
    if len(word) < 2:
        return word
    merged: list[bytes] = []
    i = 0
    L = len(word)
    while i < L:
        if i < L - 1 and word[i] == a and word[i + 1] ==b:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)

def build_pair_stats(
    word_freq: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    构建：
      - pair_counts: 每个相邻对的全局加权计数
      - pair_to_words: 倒排索引（相邻对 -> 包含该对的单词集合）
    """
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word, freq in word_freq.items():
        if len(word) < 2:
            continue

        # 统计该单词中每个相邻对的出现次数
        local = pairs_in_word(word)
        for p, occ in local.items():
            pair_counts[p] = pair_counts.get(p, 0) + occ * freq
            s = pair_to_words.get(p)
            if s is None:
                pair_to_words[p] = {word}
            else:
                s.add(word)
    
    return pair_counts, pair_to_words

def remove_word_contrib(
    word: tuple[bytes, ...],
    freq: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """
    从 pair_counts 和 pair_to_words 中移除一个单词的贡献。
    """
    local = pairs_in_word(word)
    for p, occ in local.items():
        s = pair_to_words.get(p)
        if s is not None:
            s.discard(word)
            if not s:
                del pair_to_words[p]
        
        new_c = pair_counts.get(p, 0) - occ * freq
        if new_c <= 0:
            pair_counts.pop(p, None)
        else:
            pair_counts[p] = new_c

def add_word_contrib(
    word: tuple[bytes, ...],
    add_freq: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    *,
    word_is_new: bool,
) -> None:
    """
    向 pair_counts 和 pair_to_words 中添加一个单词的贡献。
    """
    if len(word) < 2:
        return
    local = pairs_in_word(word)
    for p, occ in local.items():
        pair_counts[p] = pair_counts.get(p, 0) + occ * add_freq
        if word_is_new:
            s = pair_to_words.get(p)
            if s is None:
                pair_to_words[p] = {word}
            else:
                s.add(word)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *, # 后续参数必须使用关键字参数传递
    num_processes: int | None = None # 默认值为None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个字节级 BPE 分词器。

    Args:
        input_path: 输入文件路径
        vocab_size: 词表大小
        special_tokens: 特殊标记列表
        num_processes: 用于预分词和计数的进程数

    Returns:
        vocab: 词表，映射 token ID 到字节
        merges: 合并规则列表
    """
    # ---- 参数验证 ----
    if vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size too small: must be >= 256 + len(special_tokens)")

    # ---- 词表初始化：256 个单字节 token + 特殊标记 ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    # ---- 预分词 + 计数 ----
    if num_processes is None:
        num_processes = min(8, os.cpu_count() or 1)
    
    file_size = os.path.getsize(input_path)

    # 对于小文件，多进程开销占主导地位；使用串行处理
    if num_processes <= 1 or file_size < 1_000_000:  # ~1MB
        word_freq = build_word_freq_serial(input_path, special_tokens)
    else:
        word_freq = build_word_freq_parallel(input_path, special_tokens, num_processes, num_chunks=num_processes * 32)
    
    if not word_freq:
        return vocab, []

    # ---- BPE 合并 ----
    pair_counts, pair_to_words = build_pair_stats(word_freq)
    merges: list[tuple[bytes, bytes]] = []
    while next_id < vocab_size:
        if not pair_counts:
            break
    
        # 选择频率最高的；如果频率相同，则选择字典序最大的
        (a, b), best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break
        
        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        affected = pair_to_words.get((a, b))
        if not affected:
            pair_counts.pop((a, b), None)
            continue

        # 替换每个单词中出现的 (a,b)为 new_token
        add_back: dict[tuple[bytes, ...], int] = {}
        # 1. 从词表中删除单词(a,b)
        for word in list(affected): # 使用list()以避免在迭代时修改集合
            freq = word_freq.get(word)
            if freq is None:
                continue

            remove_word_contrib(word, freq, pair_counts, pair_to_words)
            del word_freq[word]

            new_word = apply_merge(word, a, b, new_token)
            add_back[new_word] = add_back.get(new_word, 0) + freq
        # 2. 添加单词new_token
        for new_word, add_freq in add_back.items():
            existed = new_word in word_freq
            word_freq[new_word] = word_freq.get(new_word, 0) + add_freq
            add_word_contrib(new_word, add_freq, pair_counts, pair_to_words, word_is_new=not existed)

    return vocab, merges