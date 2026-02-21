#!/bin/bash
set -e

echo "🧪 Starting Test Set Evaluation Pipeline"

# 1. Build test data
echo "Building Tevatron test datasets..."
python -m factcheck_relevance.build_data --config configs/test_data_build.yaml

# Create output dir for test runs
mkdir -p runs/factcheck_relevance_test

# 2. Encode corpus
echo "Encoding test corpus..."
python -m factcheck_relevance.encode --config configs/test_inference.yaml

# 3. Encode test queries
echo "Encoding test queries..."
python -m factcheck_relevance.encode --config configs/test_inference.yaml --is_query

# 4. Search
echo "Running FAISS retrieval on test set..."
python -m factcheck_relevance.retrieve --config configs/test_inference.yaml

# 5. Evaluate
echo "Evaluating test set performance..."
python -m factcheck_relevance.eval --config configs/test_inference.yaml

echo "✅ Test set evaluation completed successfully!"
