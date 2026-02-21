# Veracity Prediction Dataset

## Overview

This directory contains datasets for training and evaluating veracity prediction models. The data consists of claim-evidence pairs with stance labels (how the evidence relates to the claim) and relevance labels (how relevant the evidence is to the claim).

## Files

- **`claim_evidence_pairs_jan_2026_train.json`**: Training dataset
- **`claim_evidence_pairs_jan_2026_test.json`**: Test dataset
- **`factcheck-embeddings/`**: Core project directory containing model training and evaluation logic.

## Getting Started

To train or evaluate models on this dataset, please refer to the [FactCheck Embeddings Usage Guide](factcheck-embeddings/README.md).

### Reproducing Results
If you have been provided with a model checkpoint, place it in `factcheck-embeddings/runs/factcheck_relevance_cpu/` and follow the reproduction steps in the [project README](factcheck-embeddings/README.md#reproducibility--sharing).

### Quick Setup

```bash
cd factcheck-embeddings
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Dataset Statistics

### Training Set
- **Total samples**: 1,250 claims
- **Average evidence per claim**: 42.79
- **Total evidence snippets**: 53,492

### Test Set
- **Total samples**: 313 claims
- **Average evidence per claim**: 42.53
- **Total evidence snippets**: 13,311

### Train/Test Split
- **Split ratio**: 80% train / 20% test
- **Splitting method**: Random shuffle before split

## Schema

Each dataset is a JSON array where each element represents a claim with associated evidence and labels.

### Top-Level Structure

```json
[
  {
    "claim": "string",
    "stance_label": "string",
    "evidence": [...]
  }
]
```

### Field Descriptions

#### Claim Object

| Field | Type | Description |
|-------|------|-------------|
| `claim` | string | The claim text to be verified |
| `stance_label` | string | Overall stance of evidence relative to the claim |
| `evidence` | array | List of evidence snippets retrieved for this claim |

#### Stance Labels

The `stance_label` field indicates how the evidence relates to the claim:

| Label | Count (Train) | Count (Test) | Description |
|-------|---------------|--------------|-------------|
| `REFUTES` | 1,047 (83.8%) | 262 (83.7%) | Evidence contradicts the claim |
| `MIXED` | 134 (10.7%) | 33 (10.5%) | Evidence both supports and refutes aspects of the claim |
| `SUPPORTS` | 41 (3.3%) | 11 (3.5%) | Evidence confirms the claim |
| `NOT_ENOUGH_INFO` | 28 (2.2%) | 7 (2.2%) | Insufficient evidence to verify the claim |

#### Evidence Object

Each evidence item in the `evidence` array has the following structure:

```json
{
  "snippet": "string",
  "relevance_label": "string",
  "relevance_explanation": "string",
  "query": "string",
  "cosine_similarity": float
}
```

| Field | Type | Description |
|-------|------|-------------|
| `snippet` | string | The text snippet retrieved as evidence |
| `relevance_label` | string | Classification of how relevant this snippet is to the claim |
| `relevance_explanation` | string | Human-readable explanation of the relevance judgment |
| `query` | string | The search query used to retrieve this evidence (defaults to claim text if no rewritten query available) |
| `cosine_similarity` | float | Similarity score between claim and evidence (range: 0.0-1.0) |

#### Relevance Labels

The `relevance_label` field indicates how relevant each evidence snippet is to verifying the claim:

| Label | Count (Train) | Count (Test) | Description |
|-------|---------------|--------------|-------------|
| `NOT_RELEVANT` | 44,825 (83.8%) | 11,423 (85.8%) | Evidence is not relevant to the claim |
| `RELEVANT` | 8,279 (15.5%) | 1,805 (13.6%) | Evidence is directly relevant to the claim |
| `ERROR` | 374 (0.7%) | 81 (0.6%) | Error occurred during relevance labeling |
| `PARTIALLY_RELEVANT` | 12 (<0.1%) | 2 (<0.1%) | Evidence is somewhat relevant but not directly addressing the claim |
| `UNKNOWN` | 2 (<0.1%) | 0 (0%) | Relevance could not be determined |

## Example Record

```json
{
  "claim": "As novas mudanças no PIX e o rastreio dos pagamentos",
  "stance_label": "REFUTES",
  "evidence": [
    {
      "snippet": "Em 2025, o Pix passará por alterações para aumentar a segurança e proteger milhões de pessoas que utilizam o sistema diariamente...",
      "relevance_label": "RELEVANT",
      "relevance_explanation": "The evidence discusses new PIX rule changes and mechanisms for tracking and blocking payment flows, directly matching the claim about changes in PIX and payment tracing.",
      "query": "As novas mudanças no PIX e o rastreio dos pagamentos",
      "cosine_similarity": 0.7048648076774
    }
  ]
}
```

## Data Source

The data was extracted from `test_jan_2026_full_relevance_annotated.jsonl` using the `extract_retrieval_labels.py` script. The source data includes:

- Claims from the Factiverse fact-checking system
- Evidence retrieved via the Factiverse search API
- Manual relevance annotations for each evidence snippet
- Stance labels for overall claim verification

## Use Cases

This dataset is designed for:

1. **Stance Detection**: Training models to classify the relationship between claims and evidence
2. **Evidence Relevance Classification**: Training models to identify relevant evidence for fact-checking
3. **Multi-document Claim Verification**: Evaluating systems that aggregate evidence from multiple sources
4. **Retrieval Quality Assessment**: Analyzing the quality of evidence retrieval systems

## Notes

- The dataset is highly imbalanced towards `REFUTES` claims and `NOT_RELEVANT` evidence
- Multiple languages are present in the dataset (Portuguese, Spanish, English, etc.)
- Each claim has been paired with all retrieved evidence, regardless of relevance
- The `cosine_similarity` scores can be used for re-ranking or filtering evidence
- Some evidence snippets may be truncated or incomplete
