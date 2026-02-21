# FactCheck Embeddings

This repository contains automated codegen to train a Tevatron relevance bi-encoder for fact-checking.

## Goal
Train a dense retriever to distinguish between `RELEVANT`/`PARTIALLY_RELEVANT` and `NOT_RELEVANT` evidence snippets for a given claim.

## Setup
1. **Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   pip install accelerate aiohttp==3.10.11
   ```
2. **Data**:
   Place your raw JSON files in `data/raw/`.

## Pipeline
Run the full pipeline on CPU:
```bash
bash scripts/run_all_cpu.sh
```

Individual steps:
1. **Build Data**: `bash scripts/run_build.sh`
2. **Train**: `bash scripts/run_train_cpu.sh`
3. **Encode & Retrieve**: `bash scripts/run_encode.sh` and `bash scripts/run_retrieve.sh`
4. **Evaluate**: `bash scripts/run_eval.sh`

## Implementation Details

### Training Details
- **Architecture**: Bi-encoder using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Framework**: Tevatron (dense retrieval framework).
- **CPU Defaults**: Optimized for CPU iteration (batch size 4, max lengths 64/192).
- **Compatibility**: Includes a monkeypatch in `src/factcheck_relevance/train.py` to support `transformers` 4.47+ signatures.

### Negative Sampling
We use **Hard-Negative Sampling** based on the `cosine_similarity` provided in the raw data:
1. For each positive, we draw a fixed number of negatives (`k_neg`).
2. We build a "hard pool" of the top 20 negatives by similarity.
3. We sample `k_hard` (default 67%) from this hard pool.
4. We sample the remaining `k_rand` from the rest of the negatives.

### Loss Function
The model uses **SimpleContrastiveLoss** (InfoNCE). This is the standard loss used by Tevatron for dense retrieval, which pushes positive pairs closer and negative pairs further apart in the embedding space.

### Data Balancing
- **Query Level**: Claims with zero positive evidence snippets are skipped during training to avoid noise.
- **Instance Level**: For every positive snippet, we generate one training instance with a fixed ratio of negatives (1:3 by default). This ensures the model sees a balanced number of contrastive examples for every relevant snippet.

## Configs
- `configs/data_build.yaml`: Data processing settings (split ratios, sampling params).
- `configs/cpu_train.yaml`: Training hyperparameters (learning rate, epochs).
- `configs/inference.yaml`: Retrieval settings (top-k, batch sizes).


## Reproducibility
To reproduce the findings on a different machine:
1. **Clone the repository** (logic and configs only).
2. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   pip install accelerate aiohttp==3.10.11
   ```
3. **Training Details**:
   - The model was trained for **1 epoch** (sufficient for this MiniLM backbone and dataset size).
   - Parameters are defined in [cpu_train.yaml](configs/cpu_train.yaml).
   - Detailed training metrics and internal state are recorded in `runs/factcheck_relevance_cpu/trainer_state.json`.
4. **Provide Raw Data**:
   Place the original `claim_evidence_pairs_jan_2026_train.json` and `claim_evidence_pairs_jan_2026_test.json` files in `data/raw/`.
5. **Run Reproduction Script**:
   ```bash
   # For global collection evaluation
   bash scripts/run_global.sh
   
   # For test-set only evaluation
   bash scripts/run_test.sh
   ```

*Note: Use the provided `.gitignore` to manage local artifacts and environments.*


## Methodology

### Training Data Creation
The training dataset is derived from raw claim-evidence pairs using the following logic:
1.  **Instance Generation**: For every manually labeled positive snippet, a distinct training instance is created.
2.  **Contrastive Sampling**: Each instance consists of one positive and a fixed ratio of negatives (default 1:3).
3.  **Hard-Negative Selection**: We use a **Teacher-Guided** approach. Negatives are sorted by the `cosine_similarity` scores present in the raw data.
    -   **Hard Pool**: The top 20 most similar negatives for a claim form the "hard" candidate pool.
    -   **Sampling Ratio**: 67% of negatives are sampled from this hard pool, and 33% are sampled randomly.
    -   **Goal**: This forces the model to learn subtle boundaries between true relevance and high-level semantic similarity.

### Evaluation Assumptions
We report metrics under two distinct collection assumptions:
-   **Local (Official Test Set)**: Retrieval against only the ~42 candidate snippets per claim.
-   **Global (Master Collection)**: Retrieval against all ~53,000 unique snippets in the entire dataset.

### Limitations & Caveats
-   **Teacher Dependency**: Quality of hard negatives depends on the scores in the raw data.
-   **Model Size**: Optimized for CPU using MiniLM; may have lower semantic resolution than 7B+ parameter models.

## Reproducibility & Sharing

### Model Checkpoints
The trained model checkpoints should be stored in:
`factcheck-embeddings/runs/factcheck_relevance_cpu/`

### Sharing with Vinay
To share the project state for reproduction:
1.  **Setup Environment**:
    ```bash
    cd factcheck-embeddings
    python -m venv venv
    source venv/bin/activate
    pip install -e .
    ```
2.  **Reproduce**:
    -   Place shared model files in `runs/factcheck_relevance_cpu/`.
    -   Place raw JSON data in the root or `data/raw/`.
    -   Run evaluation: `bash scripts/run_test.sh`

## Performance Results

### Final Model Comparison (Official Test Set)
| Model / Strategy | MRR@10 | nDCG@10 | Recall@5 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **MiniLM-L12 (InfoNCE/Tevatron)** | **0.7905** | **0.6541** | **0.5070** | **0.8421** |
| MiniLM-L12 (SetFit) | 0.5798 | 0.4304 | 0.3166 | 0.6119 |
| Untuned Gemma (300M) | 0.0343 | 0.0317 | 0.0195 | 0.1457 |

> [!TIP]
> InfoNCE training (10 epochs) significantly outperforms SetFit (3 epochs) and zero-shot Gemma.

### Detailed Split Metrics
| Dataset | MRR@10 | nDCG@10 | Recall@5 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **Dev Set** (Local) | 0.7047 | 0.4930 | 0.3660 | 0.7097 |
| **Master Collection** (Global) | 0.7315 | 0.5880 | 0.4350 | 0.7959 |
