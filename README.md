# Fact-Checking Encoders

This repository contains datasets and tools for training and evaluating evidence relevance classification models.

## Repository Structure

-   **`factcheck-embeddings/`**: Core project directory containing model training and evaluation logic. This is where the primary code resides.
-   **`claim_evidence_pairs_jan_2026_*.json`**: Training and test datasets.
-   **`data/`**: (Ignored) Local data processing artifacts.

## Documentation

For detailed usage instructions, performance metrics, and reproduction steps, please see the:

### ➡️ [FactCheck Embeddings Documentation](factcheck-embeddings/README.md)

## Quick Start

```bash
cd factcheck-embeddings
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Dataset Statistics

### Training Set
- **Total samples**: 1,250 claims
- **Total evidence snippets**: 53,492

### Test Set
- **Total samples**: 313 claims
- **Total evidence snippets**: 13,311
