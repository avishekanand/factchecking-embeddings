#!/bin/bash
set -e

echo "Running FAISS retrieval..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m factcheck_relevance.retrieve --config configs/inference.yaml
