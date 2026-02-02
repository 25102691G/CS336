import os
import pickle
import regex as re
from typing import Any, Iterable, Iterator

GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    字节级 BPE 分词器，兼容 GPT-2 风格的预分词。
    词表（Vocab）将 token_id 映射到字节。合并规则（Merges）是按创建顺序排列的 list[(bytes, bytes)]。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        
        # 反向词表：字节 -> id
        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # 合并排名：排名越低，优先级越高
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: idx for idx, pair in enumerate(self.merges)
        }

        # 用于 GPT-2 预分词的正则表达式
        self.rx = re.compile(GPT2_PRETOKENIZE_PATTERN)

        # 特殊标记
        self.special_tokens: list[str] = special_tokens or []
        self.special_bytes: list[bytes] = []
        self.special_id: dict[str, int] = {}

        if self.special_tokens:
            # 将缺失的特殊标记添加到词表中
            for s in self.special_tokens:
                b = s.encode("utf-8")
                if b not in self.byte_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = b
                    self.byte_to_id[b] = new_id
                self.special_id[s] = self.byte_to_id[b]
                self.special_bytes.append(b)
            
            # 构建一个“最长优先”的特殊标记匹配器
            # 我们将它们保留为字符串，以便在 encode() 中进行边界安全的匹配
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("|".join(re.escape(s) for s in sorted_special)) # 创建一个正则表达式，用于匹配特殊标记
            self._max_special_len = max(len(s) for s in self.special_tokens)
        else:
            self._special_re = None
            self._max_special_len = 0

        # 用于预分词字节序列的 BPE 缓存
        self._bpe_cache: dict[bytes, list[bytes]] = {}

    @classmethod # 类方法：可以直接通过类名调用，不需要实例化对象
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """
        从文件加载词表和合并规则。
        """
        # 匹配之前使用的训练输出（pickle 转储）
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # ---------------------------
    # 公共 API
    # ---------------------------
    def encode(self, text: str) -> list[int]:
        if not text:
            return []    
        
        # 如果没有特殊标记，直接编码整个文本
        if self._special_re is None:
            return self._encode_plain(text)
        
        # 按特殊标记拆分文本，同时将它们保留为独立部分
        ids: list[int] = []
        last = 0
        for m in self._special_re.finditer(text):
            # 1. 如果当前匹配到的特殊标记之前有普通文本
            if m.start() > last:
                ids.extend(self._encode_plain(text[last : m.start()]))

            # 2. 处理当前匹配到的特殊标记
            s = m.group(0)
            ids.append(self.special_id[s]) # 在init方法中已经构建好了 special_id 映射

            # 3. 更新 last 指针
            last = m.end()

        # 处理特殊标记之后的剩余普通文本
        if last < len(text):
            ids.extend(self._encode_plain(text[last:]))
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        内存高效的流式编码，与 Tokenizer.encode(full_text) 的结果一致
        """
        buf = ""

        for chunk in iterable:
            if not chunk:
                continue
            buf += chunk

            while True:
                matches = list(self.rx.finditer(buf))
                if len(matches) <= 1:
                    break

                # 保留最后一个匹配项不处理；发送它之前的所有内容。
                cut = matches[-1].start()
                if cut <= 0:
                    break

                process_part = buf[:cut]
                buf = buf[cut:]

                for _id in self.encode(process_part):
                    yield _id
        
        # 刷新剩余部分
        if buf:
            for _id in self.encode(buf):
                yield _id

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""            
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # ---------------------------
    # 内部辅助函数
    # ---------------------------
    def _encode_plain(self, text: str) -> list[int]:
        """
        对不含特殊标记的一段文本进行编码。
        """
        out: list[int] = []

        for m in self.rx.finditer(text):
            piece = m.group(0)
            if not piece:
                continue
            piece_bytes = piece.encode("utf-8")
            for tok_bytes in self._bpe(piece_bytes):
                out.append(self.byte_to_id[tok_bytes])
        return out
    
    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        对单个预分词字节序列应用 BPE 合并（按排名）。
        返回词表字节 token 列表。
        """
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached
        
        word: list[bytes] = [bytes([b]) for b in token_bytes]
        # 单字节 token，直接返回
        if len(word) <= 1:
            self._bpe_cache[token_bytes] = word
            return word
        
        while True:
            best_pair = None
            best_rank = None

            # 在相邻对中查找排名最高的对
            prev = word[0]
            for cur in word[1:]:
                p = (prev, cur)
                r = self.merge_ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = p
                prev = cur
            
            if best_pair is None:
                break
        
            a, b = best_pair
            new_token = a + b

            # 合并所有出现的 (a, b)
            merged: list[bytes] = []
            i = 0
            L = len(word)
            while i < L:
                if i < L - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
            if len(word) <= 1:
                break
        
        self._bpe_cache[token_bytes] = word
        return word