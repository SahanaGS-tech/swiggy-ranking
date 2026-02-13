# src/train.py

from __future__ import annotations
import os
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from src.dataset import build_vocab_from_pairs_csv, RankingDataset
from src.model import RankingModel


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(
    csv_path: str,
    max_len_query: int = 32,
    max_len_dish: int = 64,
    batch_size: int = 64,
    val_ratio: float = 0.1,
) -> Tuple[Dict, Dict, Dict]:
    """
    Build vocab, dataset, and return dataloaders + vocab + dataset sizes.

    Returns:
        loaders: {"train": DataLoader, "val": DataLoader}
        vocab: token -> id mapping
        info: {"n_train": int, "n_val": int}
    """
    # 1. Build vocab from the full CSV (queries + dishes)
    vocab = build_vocab_from_pairs_csv(csv_path)

    # 2. Create full dataset
    full_ds = RankingDataset(
        csv_path=csv_path,
        vocab=vocab,
        max_len_query=max_len_query,
        max_len_dish=max_len_dish,
    )

    # 3. Train/val split
    n_total = len(full_ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # 4. DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    loaders = {"train": train_loader, "val": val_loader}
    info = {"n_train": n_train, "n_val": n_val}
    return loaders, vocab, info


def train_one_epoch(
    model: RankingModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        query_ids = batch["query_ids"].to(device)   # (B, Lq)
        dish_ids = batch["dish_ids"].to(device)     # (B, Ld)
        labels = batch["label"].to(device)          # (B,)

        optimizer.zero_grad()

        logits = model(query_ids, dish_ids)         # (B,)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: RankingModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on validation loader.

    Returns:
        avg_loss, avg_accuracy
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    n_batches = 0

    for batch in loader:
        query_ids = batch["query_ids"].to(device)
        dish_ids = batch["dish_ids"].to(device)
        labels = batch["label"].to(device)          # (B,)

        logits = model(query_ids, dish_ids)         # (B,)
        loss = criterion(logits, labels)

        # Predictions: sigmoid + threshold 0.5
        probs = torch.sigmoid(logits)               # (B,)
        preds = (probs >= 0.5).float()              # (B,)

        correct = (preds == labels).sum().item()
        total = labels.numel()

        running_loss += loss.item()
        running_correct += correct
        running_total += total
        n_batches += 1

    avg_loss = running_loss / max(1, n_batches)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc


def main():
    # ---------- Config ----------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "processed", "train_pairs.csv")

    embed_dim = 64
    hidden_dim = 128
    max_len_query = 32
    max_len_dish = 64
    batch_size = 64
    val_ratio = 0.1
    num_epochs = 5
    learning_rate = 1e-3
    weight_decay = 0.0

    device = get_device()
    print(f"Using device: {device}")

    # ---------- Data ----------
    loaders, vocab, info = create_dataloaders(
        csv_path=csv_path,
        max_len_query=max_len_query,
        max_len_dish=max_len_dish,
        batch_size=batch_size,
        val_ratio=val_ratio,
    )

    print(f"Train examples: {info['n_train']}, Val examples: {info['n_val']}")
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # ---------- Model, loss, optimizer ----------
    model = RankingModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        padding_idx=vocab["<pad>"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ---------- Training loop ----------
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

        # Basic "best model" tracking (you can later save to disk)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model checkpoint
            ckpt_path = os.path.join(project_root, "model_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "embed_dim": embed_dim,
                        "hidden_dim": hidden_dim,
                        "max_len_query": max_len_query,
                        "max_len_dish": max_len_dish,
                    },
                },
                ckpt_path,
            )
            print(f"  ↳ New best model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
