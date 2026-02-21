# FactCheck Embeddings

This repository training recipes for a bi-encoder for fact-checking.

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


## Reproducibility & Sharing

### Model Checkpoints
The trained model checkpoints are stored in:
`factcheck-embeddings/runs/factcheck_relevance_cpu/`

This directory contains the `pytorch_model.bin` (or `model.safetensors`), `config.json`, and the tokenizer files required for inference.

### Sharing with Vinay


1.  **Setup**:
    ```bash
    cd factcheck-embeddings
    python -m venv venv
    source venv/bin/activate
    pip install -e .
    ```
2.  **Reproduce**:
    -   Place the shared model files back into `runs/factcheck_relevance_cpu/`.
    -   Place the raw data in the root or `data/raw/`.
    -   Run the evaluation script:
        ```bash
        bash scripts/run_test.sh
        ```

## Performance Results

## Methodology

### Training Data Generation
The training data is constructed using a **contrastive learning** approach derived from raw annotations:
- **Positive Alignment**: Every manually verified relevant snippet is treated as a positive target.
- **Hard-Negative Mining**: We sample "hard" negatives—snippets that look semantically similar to the claim but are non-relevant. These are selected from a pool of the top 20 most similar negatives per claim.
- **Deduplication**: Claims and evidence are globally deduplicated to avoid data leakage.

### SetFit Training Details
For the SetFit baseline:
- **Framework**: [SetFit](https://github.com/huggingface/setfit) (Sentence Transformer Fine-tuning).
- **Backbone**: `paraphrase-multilingual-MiniLM-L12-v2`.
- **Hyperparameters**: Trained for **3 epochs** with default contrastive settings.
- **Architecture**: A two-stage process involved fine-tuning the SBERT backbone on sentence pairs, followed by training a classification head.

### Evaluation Metrics & Assumptions
Our evaluation results are reported based on two scenarios:
1. **Reranking (Local)**: Searching only against the candidate pool (~42 snippets) retrieved for a specific claim.
2. **Retrieval (Global)**: Searching against the **Global Master Collection** of all unique evidence snippets (~53k).

### Limitations
- **Teacher Dependency**: Hard negatives rely on the initial retrieval scores; biases in the source model may propagate.
- **Model Capacity**: The 12-layer MiniLM is optimized for CPU; larger models may provide higher semantic resolution.

## Performance Results


### Final Model Comparison (Official Test Set)

| Model / Strategy | MRR@10 | nDCG@10 | Recall@5 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **MiniLM-L12 (InfoNCE/Tevatron)** | **0.7905** | **0.6541** | **0.5070** | **0.8421** |
| MiniLM-L12 (SetFit) | 0.5798 | 0.4304 | 0.3166 | 0.6119 |
| Untuned Gemma (300M) | 0.0343 | 0.0317 | 0.0195 | 0.1457 |




> [!TIP]
> The Tevatron-based InfoNCE approach (10 epochs) significantly outperforms both the SetFit fine-tuning (3 epochs) and the zero-shot baseline.

### Detailed Split Metrics
| Dataset | MRR@10 | nDCG@10 | Recall@5 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **Dev Set** (Local) | 0.7047 | 0.4930 | 0.3660 | 0.7097 |
| **Official Test Set** (Local) | 0.7905 | 0.6541 | 0.5070 | 0.8421 |
| **Master Collection** (Global) | 0.7315 | 0.5880 | 0.4350 | 0.7959 |
