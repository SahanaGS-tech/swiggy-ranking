# src/inference.py

from __future__ import annotations
import os
import time
from typing import List, Tuple

import torch

from src.model import RankingModel
from src.dataset import (
    tokenize,
    encode_tokens,
    pad_or_truncate,
)
from src.data_prep import load_raw_dataset, build_dish_text  # reuse training logic


# ---------- Checkpoint loading & encoding helpers ----------

def load_checkpoint(project_root: str):
    ckpt_path = os.path.join(project_root, "model_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    config = ckpt["config"]

    model = RankingModel(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        padding_idx=vocab["<pad>"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, vocab, config


def encode_text_for_inference(
    text: str,
    vocab,
    max_len: int,
) -> torch.LongTensor:
    tokens = tokenize(text)
    ids = encode_tokens(tokens, vocab)
    ids = pad_or_truncate(ids, max_len=max_len, pad_id=vocab["<pad>"])
    return torch.tensor(ids, dtype=torch.long)


# ---------- Batched ranking ----------

@torch.no_grad()
def rank_dishes_for_query_batched(
    model: RankingModel,
    vocab,
    config,
    dishes: List[Tuple[str, str]],  # (name, dish_text)
    query: str,
    top_k: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Batched scoring: encode all dishes into a single tensor and do ONE forward pass.

    dishes: list of (name, dish_text)
    returns: list of (name, dish_text, score) sorted by score desc
    """
    device = torch.device("cpu")
    model.to(device)

    max_len_query = config["max_len_query"]
    max_len_dish = config["max_len_dish"]

    # 1. Encode query once: shape (1, Lq)
    q_ids = encode_text_for_inference(query, vocab, max_len_query).unsqueeze(0).to(device)

    # 2. Encode ALL dishes to a single tensor (N, Ld)
    dish_ids_list = []
    names = []
    dish_texts = []

    for name, dish_text in dishes:
        d_ids = encode_text_for_inference(dish_text, vocab, max_len_dish)
        dish_ids_list.append(d_ids)
        names.append(name)
        dish_texts.append(dish_text)

    dish_ids_batch = torch.stack(dish_ids_list, dim=0).to(device)  # (N, Ld)

    # 3. Repeat query ids N times to match dish batch
    N = dish_ids_batch.size(0)
    q_ids_batch = q_ids.expand(N, -1)  # (N, Lq)

    # 4. Single forward pass over all N dishes
    logits = model(q_ids_batch, dish_ids_batch)     # (N,)
    probs = torch.sigmoid(logits)                   # (N,)

    scores = []
    for name, dish_text, p in zip(names, dish_texts, probs.tolist()):
        scores.append((name, dish_text, p))

    # 5. Sort and return top_k
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


# ---------- Main demo + latency benchmark ----------

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load model + vocab + config
    model, vocab, config = load_checkpoint(project_root)
    print("Loaded model from checkpoint.")

    # 2. Load raw dishes (same as training)
    raw_path = os.path.join(project_root, "data", "raw", "indian_food.csv")
    df_raw = load_raw_dataset(raw_path)

    dishes: List[Tuple[str, str]] = []
    for _, row in df_raw.iterrows():
        name = str(row["name"])
        dish_text = build_dish_text(row)
        dishes.append((name, dish_text))

    print(f"Total dishes available: {len(dishes)}")

    # 3. Qualitative examples using batched ranking
    queries = [
        "dessert with carrots",              # use plural form to match training
        "quick dessert under 15 minutes",
        "punjab dessert",                    # use 'punjab' (state) rather than 'punjabi'
        "spicy south indian main course",
    ]

    # Lets try with the unknown query to see how the model generalizes
    # queries = [
    #     "low sugar bengali dessert",
    #     "dessert with beetroot",
    #     "non veg starter from west bengal",
    #     "healthy south indian breakfast",
    #     "no onion no garlic north indian main course",
    #     "high protein vegetarian recipe",
    # ]

    for q in queries:
        print("\n==============================")
        print(f"Query: {q}")
        top_results = rank_dishes_for_query_batched(model, vocab, config, dishes, q, top_k=5)
        for name, dish_text, score in top_results:
            print(f"  {score:.3f}  {name}")

        # 4. Latency benchmark for the actual catalog (255 dishes)
    print("\n==============================")
    print("Latency benchmark (batched scoring, real catalog size)")

    dishes_for_benchmark = dishes  # no replication; use all real dishes
    print(f"Benchmarking with N={len(dishes_for_benchmark)} dishes")

    test_query = "spicy south indian main course"

    # warmup run (cold start)
    start_warm = time.perf_counter()
    _ = rank_dishes_for_query_batched(
        model, vocab, config, dishes_for_benchmark, test_query, top_k=5
    )
    warm_elapsed_ms = (time.perf_counter() - start_warm) * 1000.0

    # timed run (steady-state)
    start = time.perf_counter()
    _ = rank_dishes_for_query_batched(
        model, vocab, config, dishes_for_benchmark, test_query, top_k=5
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(
        f"Warm-up (cold start) batched scoring of {len(dishes_for_benchmark)} "
        f"items took {warm_elapsed_ms:.2f} ms on CPU"
    )
    print(
        f"Steady-state batched scoring of {len(dishes_for_benchmark)} "
        f"items took {elapsed_ms:.2f} ms on CPU"
    )



if __name__ == "__main__":
    main()
