#!/bin/bash
set -e

echo "Encoding corpus..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUDA_VISIBLE_DEVICES=""
python -m factcheck_relevance.encode --config configs/inference.yaml

echo "Encoding queries..."
python -m factcheck_relevance.encode --config configs/inference.yaml --is_query
