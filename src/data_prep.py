# src/data_prep.py

import os
import random
from typing import List, Dict

import pandas as pd


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load the indian_food.csv and add some helper columns."""
    df = pd.read_csv(path)

    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    required = [
        "name",
        "ingredients",
        "diet",
        "prep_time",
        "cook_time",
        "flavor_profile",
        "course",
        "state",
        "region",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Total time feature
    df["total_time"] = df["prep_time"] + df["cook_time"]

    # Clean up state/region: treat -1 as missing
    df["state"] = df["state"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df.loc[df["state"] == "-1", "state"] = ""
    df.loc[df["region"] == "-1", "region"] = ""

    # Build a text field for the dish
    df["dish_text"] = df.apply(build_dish_text, axis=1)

    return df


def build_dish_text(row: pd.Series) -> str:
    """Construct a human-readable description of a dish."""
    parts = [row["name"]]

    ing = str(row["ingredients"]).strip()
    if ing:
        parts.append(f"— {ing}")

    meta_bits = []

    diet = str(row["diet"]).strip()
    if diet:
        meta_bits.append(diet)

    flavor = str(row["flavor_profile"]).strip()
    if flavor:
        meta_bits.append(flavor)

    course = str(row["course"]).strip()
    if course:
        meta_bits.append(course)

    state = str(row["state"]).strip()
    region = str(row["region"]).strip()

    loc_bits = []
    if state:
        loc_bits.append(state)
    if region:
        loc_bits.append(f"{region} India")

    meta_str = " ".join(meta_bits)
    loc_str = ", ".join(loc_bits)

    tail = []
    if meta_str:
        tail.append(meta_str)
    if loc_str:
        tail.append(f"from {loc_str}")

    if tail:
        parts.append(". " + " ".join(tail) + ".")

    return " ".join(parts).strip()


def get_ingredients(row: pd.Series, max_ings: int = 3) -> List[str]:
    """Parse ingredients string into a small list of lowercased tokens."""
    raw = str(row["ingredients"])
    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    return tokens[:max_ings]


def generate_name_queries(
    df: pd.DataFrame,
    negatives_per_positive: int = 3,
    rng: random.Random = None,
) -> List[Dict]:
    """Generate (query, dish, label) using dish names."""
    if rng is None:
        rng = random.Random(42)

    examples = []
    indices = list(df.index)

    for i, row in df.iterrows():
        name = str(row["name"]).lower().strip()
        if not name:
            continue

        # A few variants of the same intent
        queries = [
            name,
            f"how to make {name}",
            f"{name} recipe",
        ]

        # All other dishes are candidate negatives
        other_indices = [j for j in indices if j != i]

        for q in queries:
            # Positive pair
            examples.append(
                {
                    "query": q,
                    "dish": row["dish_text"],
                    "label": 1,
                }
            )

            # Negatives
            neg_indices = rng.sample(
                other_indices, min(negatives_per_positive, len(other_indices))
            )
            for j in neg_indices:
                neg_row = df.loc[j]
                examples.append(
                    {
                        "query": q,
                        "dish": neg_row["dish_text"],
                        "label": 0,
                    }
                )

    return examples


def generate_ingredient_queries(
    df: pd.DataFrame,
    negatives_per_positive: int = 3,
    rng: random.Random = None,
) -> List[Dict]:
    """Generate queries like 'dessert with carrot' using ingredients."""
    if rng is None:
        rng = random.Random(123)

    examples = []
    indices = list(df.index)

    # Precompute lowercased ingredients string per dish for membership tests
    df["ingredients_lc"] = df["ingredients"].astype(str).str.lower()

    for i, row in df.iterrows():
        ings = get_ingredients(row, max_ings=2)
        if not ings:
            continue

        course = str(row["course"]).lower().strip() or "dish"

        for ing in ings:
            q1 = f"{course} with {ing}"
            q2 = f"{ing} {course}"

            for q in [q1, q2]:
                # Positive: current dish only
                examples.append(
                    {
                        "query": q,
                        "dish": row["dish_text"],
                        "label": 1,
                    }
                )

                # Negatives: dishes that do NOT contain this ingredient
                candidate_negs = [
                    j
                    for j in indices
                    if j != i and ing not in df.loc[j, "ingredients_lc"]
                ]
                if not candidate_negs:
                    continue

                neg_indices = rng.sample(
                    candidate_negs,
                    min(negatives_per_positive, len(candidate_negs)),
                )
                for j in neg_indices:
                    neg_row = df.loc[j]
                    examples.append(
                        {
                            "query": q,
                            "dish": neg_row["dish_text"],
                            "label": 0,
                        }
                    )

    return examples


def generate_region_queries(
    df: pd.DataFrame,
    negatives_per_positive: int = 3,
    rng: random.Random = None,
) -> List[Dict]:
    """Generate queries using state/region info."""
    if rng is None:
        rng = random.Random(999)

    examples = []
    indices = list(df.index)

    for i, row in df.iterrows():
        state = str(row["state"]).strip().lower()
        region = str(row["region"]).strip().lower()
        course = str(row["course"]).strip().lower() or "dish"

        queries = []

        if state:
            queries.append(f"{state} {course}")
            queries.append(f"{state} {course} recipe")
        if region:
            queries.append(f"{region} indian {course}")

        if not queries:
            continue

        # Positives: all dishes sharing same state/region
        def is_positive(j: int) -> bool:
            r = df.loc[j]
            s2 = str(r["state"]).strip().lower()
            reg2 = str(r["region"]).strip().lower()
            return (state and s2 == state) or (region and reg2 == region)

        positive_indices = [j for j in indices if is_positive(j)]
        negative_indices = [j for j in indices if j not in positive_indices]

        if not positive_indices or not negative_indices:
            continue

        for q in queries:
            for j in positive_indices:
                pos_row = df.loc[j]
                examples.append(
                    {
                        "query": q,
                        "dish": pos_row["dish_text"],
                        "label": 1,
                    }
                )

            neg_sample = rng.sample(
                negative_indices,
                min(negatives_per_positive * len(positive_indices), len(negative_indices)),
            )
            for j in neg_sample:
                neg_row = df.loc[j]
                examples.append(
                    {
                        "query": q,
                        "dish": neg_row["dish_text"],
                        "label": 0,
                    }
                )

    return examples


def generate_time_queries(
    df: pd.DataFrame,
    negatives_per_positive: int = 3,
    rng: random.Random = None,
) -> List[Dict]:
    """Generate queries like 'quick dessert under 20 minutes' or 'slow cooked dessert'."""
    if rng is None:
        rng = random.Random(2025)

    examples = []
    indices = list(df.index)

    # Define time buckets
    def quick_15(t): return t <= 15
    def quick_30(t): return t <= 30
    def slow_60(t): return t >= 60

    buckets = [
        ("quick {course} under 15 minutes", quick_15),
        ("quick {course} under 30 minutes", quick_30),
        ("slow cooked {course}", slow_60),
    ]

    for template, predicate in buckets:
        for i, row in df.iterrows():
            total = row["total_time"]
            course = str(row["course"]).strip().lower() or "dish"
            q = template.format(course=course)

            if predicate(total):
                # Positive
                examples.append(
                    {
                        "query": q,
                        "dish": row["dish_text"],
                        "label": 1,
                    }
                )

        # For each query, sample negatives from dishes not matching predicate
        pos_indices = [i for i, r in df.iterrows() if predicate(r["total_time"])]
        neg_indices = [i for i, r in df.iterrows() if not predicate(r["total_time"])]

        if not pos_indices or not neg_indices:
            continue

        for i in pos_indices:
            row = df.loc[i]
            course = str(row["course"]).strip().lower() or "dish"
            q = template.format(course=course)
            neg_sample = rng.sample(
                neg_indices,
                min(negatives_per_positive, len(neg_indices)),
            )
            for j in neg_sample:
                neg_row = df.loc[j]
                examples.append(
                    {
                        "query": q,
                        "dish": neg_row["dish_text"],
                        "label": 0,
                    }
                )

    return examples


def build_training_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Run all generators and combine into a single (query, dish, label) DataFrame."""
    rng = random.Random(42)

    all_examples: List[Dict] = []

    all_examples += generate_name_queries(df, rng=rng)
    all_examples += generate_ingredient_queries(df, rng=rng)
    all_examples += generate_region_queries(df, rng=rng)
    all_examples += generate_time_queries(df, rng=rng)

    train_df = pd.DataFrame(all_examples)
    # Optional: drop duplicates
    train_df = train_df.drop_duplicates()

    return train_df


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "raw", "indian_food.csv")
    out_path = os.path.join(project_root, "data", "processed", "train_pairs.csv")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_raw = load_raw_dataset(raw_path)
    train_pairs = build_training_pairs(df_raw)
    train_pairs.to_csv(out_path, index=False)

    print(f"Generated {len(train_pairs)} training pairs → {out_path}")


if __name__ == "__main__":
    main()
