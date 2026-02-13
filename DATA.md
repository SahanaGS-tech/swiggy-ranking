# DATA.md – Training Data Summary

This file explains, briefly, how `data/processed/train_pairs.csv` is created for the
Swiggy query → dish ranking assignment.

---

## Pipeline Flow

```mermaid
flowchart TD
    A["data/raw/indian_food.csv(Indian Food 255 dishes)"] --> B["Preprocessing(src/data_prep.py)"]
    B --> B1["Clean state & regionMap -1 → unknown"]
    B1 --> B2["Compute total_time= prep_time + cook_time"]
    B2 --> B3["Build dish_textname — ingredients. diet course from region."]
    B3 --> C["Synthetic Query Generation"]
    C --> C1["Name-based queries"]
    C --> C2["Ingredient-based queries"]
    C --> C3["Region-based queries"]
    C --> C4["Time-based queries"]
    C --> C5["Diet / Flavor queries"]
    C1 & C2 & C3 & C4 & C5 --> D["Pair with dishes"]
    D --> D1["Positive pairs (label=1)query ↔ matching dish"]
    D --> D2["Negative pairs (label=0)query ↔ contradicting dish"]
    D1 & D2 --> E["Deduplicate"]
    E --> F["data/processed/train_pairs.csv~31k pairs → 28.5k train / 3.1k val"]
```

---

## 1. Source dataset

- File: `data/raw/indian_food.csv`
- Dataset: **Indian Food 101** (Kaggle)
- Columns used:
  - `name`
  - `ingredients`
  - `diet`
  - `prep_time`, `cook_time`
  - `flavor_profile`
  - `course`
  - `state`, `region`

---

## 2. Preprocessing

All logic lives in `src/data_prep.py`:

- Load the CSV with pandas
- Clean / normalize `state` and `region`
- Map missing or `-1` values to `"unknown"`
- Compute:

  ```text
  total_time = prep_time + cook_time
  ```

- Build a single `dish_text` string per row, for example:

  ```text
  Gajar ka halwa — Carrots, milk, sugar, ghee, cashews, raisins.
  vegetarian sweet dessert from Punjab, North India.
  ```

This `dish_text` is used as the `dish` field for training and inference.

---

## 3. Synthetic queries and labels

For each dish, the script generates several synthetic query types, such as:

- Name‑based: `"{name}"`, `"{name} recipe"`
- Ingredient‑based: `"dessert with {ingredient}"`, `"{course} with {ingredient}"`
- Region‑based: `"{state} dessert"`, `"south indian main course"`
- Time‑based: `"quick {course} under 15 minutes"` for dishes with small `total_time`
- Diet / flavor: `"vegetarian {course}"`, `"spicy {course}"`, etc.

Each final row in `train_pairs.csv` is:

```text
(query: str, dish: str, label: int)
```

- **Positive (label = 1)**  
  Query is generated from the **same** dish and matches its attributes
  (course, region, diet, time, ingredients).

- **Negative (label = 0)**  
  Same query paired with **other** dishes that contradict some of those attributes
  (different region, course, diet, or time). Negatives are sampled, not exhaustive.

After generation:

- All examples are concatenated
- Exact duplicates are dropped
- The result is written to `data/processed/train_pairs.csv`

For the run used in the assignment, there are roughly:

- ~31k total pairs
- ~28.5k training examples
- ~3.1k validation examples

---

## 4. Data limitations (brief)

- Synthetic, template-generated queries only
- Only 255 dishes in catalog
- No restaurant / price / availability info
- No nutrition or macro fields
- Mostly vegetarian, very few non-veg
