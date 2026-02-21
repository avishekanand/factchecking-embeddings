#!/bin/bash
set -e

echo "🌍 Starting Global Retrieval Pipeline (Master Collection)"

# 1. Build global data
echo "Building unified global corpus..."
python -m factcheck_relevance.build_global --config configs/global_data_build.yaml

# Create output dir
mkdir -p runs/factcheck_relevance_global

# 2. Encode global corpus
echo "Encoding global corpus..."
python -m factcheck_relevance.encode --config configs/global_inference.yaml

# 3. Encode all queries
echo "Encoding all queries..."
python -m factcheck_relevance.encode --config configs/global_inference.yaml --is_query

# 4. Search
echo "Running FAISS retrieval on global collection..."
python -m factcheck_relevance.retrieve --config configs/global_inference.yaml

# 5. Evaluate
echo "Evaluating global retrieval performance..."
python -m factcheck_relevance.eval --config configs/global_inference.yaml

echo "✅ Global eval completed successfully!"
