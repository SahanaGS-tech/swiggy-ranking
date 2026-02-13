# src/model.py

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingModel(nn.Module):
    """
    Simple query–dish relevance model:

    - Shared embedding layer for query and dish tokens.
    - Mean-pool to get query_vec, dish_vec.
    - Build z = [q, d, |q - d|].
    - MLP: FC1 + ReLU + FC2 -> logit.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # 3D because z = [q, d, |q-d|], each of size D
        self.fc1 = nn.Linear(3 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.padding_idx = padding_idx

    def encode(self, ids: torch.LongTensor) -> torch.Tensor:
        """
        Encode a batch of token-id sequences into a single vector per sequence
        using mean pooling (ignoring padding positions).

        ids: (B, L)
        returns: (B, D)
        """
        # ids: (B, L)
        emb = self.embedding(ids)  # (B, L, D)

        # mask: 1 for real tokens, 0 for padding
        mask = (ids != self.padding_idx).unsqueeze(-1)  # (B, L, 1)
        emb_masked = emb * mask  # zero out PAD embeddings

        # sum over L
        sum_emb = emb_masked.sum(dim=1)  # (B, D)
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1) avoid divide by zero

        mean_emb = sum_emb / lengths  # (B, D)
        return mean_emb

    def forward(
        self,
        query_ids: torch.LongTensor,
        dish_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        query_ids: (B, Lq)
        dish_ids:  (B, Ld)

        Returns logits: (B,)
        """
        q_vec = self.encode(query_ids)   # (B, D)
        d_vec = self.encode(dish_ids)   # (B, D)

        diff = torch.abs(q_vec - d_vec)  # (B, D)

        z = torch.cat([q_vec, d_vec, diff], dim=-1)  # (B, 3D)

        h = self.fc1(z)          # (B, H)
        h = self.relu(h)         # (B, H)
        logits = self.fc2(h)     # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        return logits
