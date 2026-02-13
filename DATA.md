# DATA.md – Training Data Generation

This document explains how `train_pairs.csv` is generated from the raw Indian Food 101 dataset, including data sources, preprocessing steps, query generation templates, and labeling logic.

---

## 📋 Table of Contents

1. [Data Source](#data-source)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Dish Text Construction](#dish-text-construction)
4. [Query Generation Templates](#query-generation-templates)
5. [Label Generation Strategy](#label-generation-strategy)
6. [Final Dataset Statistics](#final-dataset-statistics)
7. [Data Limitations](#data-limitations)
8. [Example Training Pairs](#example-training-pairs)

---

## Data Source

### Indian Food 101 Dataset

- **Source**: [Kaggle – Indian Food 101](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101)
- **License**: CC0: Public Domain
- **File**: `indian_food.csv`
- **Size**: 255 dishes

### Columns Used

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `name` | string | Dish name | "Gajar ka halwa" |
| `ingredients` | string | Comma-separated ingredients | "Carrots, milk, sugar, ghee, cashews, raisins" |
| `diet` | string | Dietary category | "vegetarian", "non vegetarian" |
| `prep_time` | int | Preparation time (minutes) | 15 |
| `cook_time` | int | Cooking time (minutes) | 30 |
| `flavor_profile` | string | Dominant flavor | "sweet", "spicy", "savory" |
| `course` | string | Meal course | "dessert", "main course", "starter", "snack" |
| `state` | string | Indian state of origin | "Punjab", "West Bengal" |
| `region` | string | Indian region | "North India", "South India", "East India" |

---

## Preprocessing Pipeline

Implemented in [src/data_prep.py](src/data_prep.py):`load_raw_dataset()`

### Step 1: Load and Normalize

```python
# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Handle missing values
df.loc[df["state"] == "-1", "state"] = ""
df.loc[df["region"] == "-1", "region"] = ""
```

### Step 2: Feature Engineering

```python
# Compute total cooking time
df["total_time"] = df["prep_time"] + df["cook_time"]

# Clean state/region strings
df["state"] = df["state"].astype(str).str.strip()
df["region"] = df["region"].astype(str).str.strip()
```

### Step 3: Build Dish Text

For each dish, construct a single text string combining all relevant fields (see next section).

---

## Dish Text Construction

Function: `build_dish_text(row)` in [src/data_prep.py](src/data_prep.py)

**Format:**
```
{name} — {ingredients}. {diet} {flavor} {course} from {state}, {region}.
```

**Example:**

| Input Fields | Value |
|--------------|-------|
| name | Gajar ka halwa |
| ingredients | Carrots, milk, sugar, ghee, cashews, raisins |
| diet | vegetarian |
| flavor_profile | sweet |
| course | dessert |
| state | Punjab |
| region | North India |

**Output:**
```
Gajar ka halwa — Carrots, milk, sugar, ghee, cashews, raisins. vegetarian sweet dessert from Punjab, North India.
```

This `dish_text` is used for:
- Training (as the "dish" column in `train_pairs.csv`)
- Inference (encoded and scored against user queries)

---

## Query Generation Templates

We generate 4 types of synthetic queries using template-based methods. Each query type targets different user intents.

### 1. Name-Based Queries

**Intent**: User searches for a specific dish by name.

**Function**: `generate_name_queries()`

**Templates:**
- `{name}`
- `how to make {name}`
- `{name} recipe`

**Example:**
```
Query: "gajar ka halwa"
Positive match: Gajar ka halwa (label=1)
Negative match: Rasgulla (label=0)
```

**Negatives:** Randomly sampled dishes (3 per positive).

---

### 2. Ingredient-Based Queries

**Intent**: User wants dishes containing a specific ingredient.

**Function**: `generate_ingredient_queries()`

**Templates:**
- `{course} with {ingredient}`
- `{ingredient} {course}`

**Example:**
```
Query: "dessert with carrots"
Positive match: Gajar ka halwa (label=1)
Negative match: Rasgulla (doesn't contain carrots) (label=0)
```

**Negatives:** Dishes that do NOT contain the queried ingredient.

**Ingredient extraction:**
- Parse ingredients string (comma-separated)
- Take first 2-3 ingredients per dish
- Lowercase for matching

---

### 3. Region/State-Based Queries

**Intent**: User searches for dishes from a specific region or state.

**Function**: `generate_region_queries()`

**Templates:**
- `{state} {course}`
- `{state} {course} recipe`
- `{region} indian {course}`

**Example:**
```
Query: "punjab dessert"
Positive match: Gajar ka halwa (from Punjab) (label=1)
Positive match: Pinni (from Punjab) (label=1)
Negative match: Rasgulla (from West Bengal) (label=0)
```

**Positives:** All dishes from the same state OR region.

**Negatives:** Dishes from different states/regions.

---

### 4. Time-Based Queries

**Intent**: User wants quick or slow-cooked dishes.

**Function**: `generate_time_queries()`

**Templates:**
- `quick {course} under 15 minutes`
- `quick {course} under 30 minutes`
- `slow cooked {course}`

**Predicates:**
- **Quick 15**: `total_time <= 15`
- **Quick 30**: `total_time <= 30`
- **Slow cooked**: `total_time >= 60`

**Example:**
```
Query: "quick dessert under 15 minutes"
Positive match: Phirni (total_time=10) (label=1)
Negative match: Gajar ka halwa (total_time=45) (label=0)
```

**Negatives:** Dishes not matching the time constraint.

---

## Label Generation Strategy

### Positive Pairs (label = 1)

A query is a **positive match** for a dish if:
- The query was generated FROM that dish's own fields (name, ingredients, region, time, etc.)
- The dish satisfies the query's constraints

**Example:**
```
Query: "punjab dessert"
Dish: "Gajar ka halwa — Carrots, milk, sugar... vegetarian sweet dessert from Punjab, North India."
Label: 1 (positive)
```

### Negative Pairs (label = 0)

A query is a **negative match** for a dish if:
- The dish contradicts one or more query constraints
- Different region/state
- Different course
- Doesn't contain the queried ingredient
- Violates time constraints

**Example:**
```
Query: "punjab dessert"
Dish: "Rasgulla — Milk, sugar, cheese... vegetarian sweet dessert from West Bengal, East India."
Label: 0 (negative – different state/region)
```

### Sampling Strategy

- **Positives**: All dishes matching the query
- **Negatives**: Randomly sample 3 negatives per positive (configurable)
- **Purpose**: Balance dataset and prevent overfitting to positives

---

## Final Dataset Statistics

### File: `data/processed/train_pairs.csv`

**Columns:**
- `query` (string): Synthetic query text
- `dish` (string): Dish text (from `build_dish_text()`)
- `label` (int): 0 or 1

**Example rows:**

| query | dish | label |
|-------|------|-------|
| "gajar ka halwa" | "Gajar ka halwa — Carrots, milk..." | 1 |
| "dessert with carrots" | "Gajar ka halwa — Carrots, milk..." | 1 |
| "dessert with carrots" | "Rasgulla — Milk, sugar, cheese..." | 0 |
| "punjab dessert" | "Gajar ka halwa — Carrots, milk..." | 1 |
| "punjab dessert" | "Rasgulla — Milk, sugar, cheese..." | 0 |

### Dataset Size

From a typical run:

```
Total examples: ~31,000
├── Train: ~28,500 (90%)
└── Validation: ~3,100 (10%)
```

### Class Distribution

- **Positive pairs (label=1)**: ~25%
- **Negative pairs (label=0)**: ~75%

(Exact ratio depends on `negatives_per_positive` parameter)

### Vocabulary

After tokenization and filtering (min_freq=2):
- **Vocab size**: ~662 tokens
- **Special tokens**: `<pad>`, `<unk>`

---

## Data Limitations

### 1. Synthetic Queries
- ❌ **Not real user searches** – Templates may not capture actual Swiggy query patterns
- ❌ **Limited diversity** – Only 4 query types (name, ingredient, region, time)
- ❌ **No typos or variations** – Real users misspell, use Hinglish, abbreviate

### 2. Small Catalog
- ❌ **Only 255 dishes** – Swiggy has thousands of dishes
- ❌ **Mostly vegetarian** – Limited non-vegetarian examples
- ❌ **Regional bias** – Overrepresents North Indian cuisine

### 3. Binary Labels
- ❌ **No graded relevance** – Can't distinguish "perfect match" from "good match"
- ❌ **Label noise** – Synthetic negatives may not reflect true irrelevance

### 4. Missing Features
- ❌ **No nutrition info** – Can't handle "low calorie", "high protein" queries
- ❌ **No restaurant data** – Can't factor in ratings, price, availability
- ❌ **No health tags** – Missing "gluten-free", "vegan", "diabetic-friendly"

### 5. Language
- ❌ **English only** – No support for Hindi, Hinglish, or regional languages
- ❌ **Formal text** – Doesn't match casual user language

### 6. Temporal & Contextual
- ❌ **No time of day** – Can't infer "breakfast dishes" without explicit course field
- ❌ **No location** – Can't personalize by city or region

---

## Example Training Pairs

### Example 1: Name-Based Query

```csv
query,dish,label
"gajar ka halwa","Gajar ka halwa — Carrots, milk, sugar, ghee, cashews, raisins. vegetarian sweet dessert from Punjab, North India.",1
"gajar ka halwa","Rasgulla — Milk, sugar, cheese. vegetarian sweet dessert from West Bengal, East India.",0
```

**Explanation:**
- **Positive**: Exact name match
- **Negative**: Different dish (random sample)

---

### Example 2: Ingredient-Based Query

```csv
query,dish,label
"dessert with carrots","Gajar ka halwa — Carrots, milk, sugar, ghee, cashews, raisins. vegetarian sweet dessert from Punjab, North India.",1
"dessert with carrots","Rasgulla — Milk, sugar, cheese. vegetarian sweet dessert from West Bengal, East India.",0
```

**Explanation:**
- **Positive**: Gajar ka halwa contains "carrots"
- **Negative**: Rasgulla does NOT contain "carrots"

---

### Example 3: Region-Based Query

```csv
query,dish,label
"punjab dessert","Gajar ka halwa — Carrots, milk, sugar, ghee, cashews, raisins. vegetarian sweet dessert from Punjab, North India.",1
"punjab dessert","Pinni — Wheat flour, ghee, jaggery, dry fruits. vegetarian sweet dessert from Punjab, North India.",1
"punjab dessert","Rasgulla — Milk, sugar, cheese. vegetarian sweet dessert from West Bengal, East India.",0
```

**Explanation:**
- **Positives**: Both Gajar ka halwa and Pinni are from Punjab
- **Negative**: Rasgulla is from West Bengal

---

### Example 4: Time-Based Query

```csv
query,dish,label
"quick dessert under 15 minutes","Phirni — Rice, milk, sugar, cardamom. vegetarian sweet dessert from Punjab, North India.",1
"quick dessert under 15 minutes","Gajar ka halwa — Carrots, milk, sugar, ghee, cashews, raisins. vegetarian sweet dessert from Punjab, North India.",0
```

**Explanation:**
- **Positive**: Phirni has `total_time=10` (≤15)
- **Negative**: Gajar ka halwa has `total_time=45` (>15)

---

## Code Reference

All data generation logic is implemented in:

- **[src/data_prep.py](src/data_prep.py)** – Main data preparation script
  - `load_raw_dataset()` – Load and preprocess CSV
  - `build_dish_text()` – Construct dish text
  - `generate_name_queries()` – Name-based query pairs
  - `generate_ingredient_queries()` – Ingredient-based query pairs
  - `generate_region_queries()` – Region-based query pairs
  - `generate_time_queries()` – Time-based query pairs
  - `build_training_pairs()` – Combine all query types
  - `main()` – Entry point

**Run data generation:**
```bash
python -m src.data_prep
```

---

## Future Data Improvements

### Short-term
- **Add diet-based queries** – "vegan dessert", "non-veg starter"
- **Add flavor-based queries** – "spicy main course", "sweet snack"
- **Combine multiple constraints** – "quick spicy south indian main course"

### Medium-term
- **Hard negatives** – Sample similar but wrong dishes (e.g., other desserts for "dessert with carrots")
- **Graded labels** – Use 0-4 scale instead of binary (perfect/good/fair/poor/irrelevant)
- **Query paraphrasing** – Use synonyms, rephrasings ("quick" → "fast", "under 15 minutes" → "less than 15 min")

### Long-term
- **Real user logs** – Train on actual Swiggy search data
- **Multi-lingual** – Add Hindi, Hinglish, Tamil, Bengali queries
- **Restaurant features** – Incorporate ratings, price, availability, delivery time
- **Nutrition queries** – "low calorie dessert", "high protein snack"
- **Contextual queries** – "breakfast", "late night snack", "office lunch"

---

## Summary

This data generation pipeline creates ~31,000 synthetic query-dish pairs from 255 dishes using template-based methods. While limited compared to real-world Swiggy data, it demonstrates:

✅ **Query diversity** – 4 query types covering different user intents
✅ **Label strategy** – Clear positive/negative logic based on attribute matching
✅ **Reproducibility** – Fixed random seeds, documented templates
✅ **Modularity** – Easy to extend with new query types

For production use, the next step would be to train on real user query logs with click/conversion labels.

---

**For training and model details, see [README.md](README.md).**
