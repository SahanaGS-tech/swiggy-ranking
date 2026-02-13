# Architecture Diagrams

The Mermaid diagrams in [README.md](../README.md#architecture-diagrams) will **automatically render** on GitHub, VS Code, and most modern markdown viewers. No need for separate image files!

## Diagrams

### 1. Data Pipeline

Shows how training pairs are generated from the raw Indian Food 101 dataset:
- Raw CSV → Preprocessing → Feature engineering
- Query generation (name, ingredient, region, time-based)
- Label assignment (positives + negatives)
- Final `train_pairs.csv` output

**View in:** [README.md - Data Pipeline](../README.md#data-pipeline-flow)

---

### 2. Model + Training Flow

Shows the neural network architecture and training process:
- Tokenization → Vocabulary building
- Shared embedding layer
- Mean pooling → Interaction features
- MLP scorer → Loss → Optimizer
- Checkpoint saving

**View in:** [README.md - Model Training](../README.md#model--training-flow)

---

## Note

✅ Mermaid diagrams render automatically on GitHub - no PNG files needed!

If you need static images for presentations, you can:
1. Screenshot the rendered diagrams on GitHub
2. Use [Mermaid Live Editor](https://mermaid.live/) to export as PNG/SVG
