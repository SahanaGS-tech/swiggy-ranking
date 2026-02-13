from __future__ import annotations
from typing import List, Dict, Tuple
import re
import os

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_text(text: str) -> str:
    """Basic normalization: lowercase, normalize dash, keep letters/numbers/spaces."""
    text = str(text).lower()
    text = text.replace("—", " ")  # normalize em dash
    # Replace any non-alphanumeric character with space
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer after cleaning."""
    text = clean_text(text)
    if not text:
        return []
    return text.split(" ")


def build_vocab(
    texts: List[str],
    min_freq: int = 2,
    max_size: int = 20000,
) -> Dict[str, int]:
    """
    Build a vocab mapping token -> id.
    Reserve 0 for <pad>, 1 for <unk>.
    """
    from collections import Counter

    counter = Counter()
    for t in texts:
        tokens = tokenize(t)
        counter.update(tokens)

    # Start with special tokens
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    # Most common tokens above min_freq
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """Map tokens to ids using vocab, fallback to <unk>."""
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(tok, unk_id) for tok in tokens]


def pad_or_truncate(ids: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    """Pad with pad_id or truncate to fixed length."""
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


class RankingDataset(Dataset):
    """
    PyTorch Dataset for (query, dish, label) pairs.

    Expects a CSV with columns: query, dish, label.
    """

    def __init__(
        self,
        csv_path: str,
        vocab: Dict[str, int],
        max_len_query: int = 32,
        max_len_dish: int = 64,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # Basic sanity
        for col in ["query", "dish", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"Expected column '{col}' in {csv_path}")
        self.vocab = vocab
        self.max_len_query = max_len_query
        self.max_len_dish = max_len_dish
        self.pad_id = self.vocab[PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.df)

    def encode_text(self, text: str, max_len: int) -> torch.LongTensor:
        tokens = tokenize(text)
        ids = encode_tokens(tokens, self.vocab)
        ids = pad_or_truncate(ids, max_len=max_len, pad_id=self.pad_id)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        q_text = row["query"]
        d_text = row["dish"]
        label = float(row["label"])

        q_ids = self.encode_text(q_text, self.max_len_query)
        d_ids = self.encode_text(d_text, self.max_len_dish)

        return {
            "query_ids": q_ids,
            "dish_ids": d_ids,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def build_vocab_from_pairs_csv(csv_path: str) -> Dict[str, int]:
    """
    Convenience helper: read train_pairs.csv and build a vocab from both query and dish text.
    """
    df = pd.read_csv(csv_path)
    if not {"query", "dish"} <= set(df.columns):
        raise ValueError("train_pairs.csv must contain 'query' and 'dish' columns")

    texts: List[str] = []
    texts.extend(df["query"].astype(str).tolist())
    texts.extend(df["dish"].astype(str).tolist())

    vocab = build_vocab(texts, min_freq=2, max_size=20000)
    return vocab
