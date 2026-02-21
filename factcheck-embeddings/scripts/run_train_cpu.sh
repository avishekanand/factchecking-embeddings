#!/bin/bash
set -e

echo "Starting Tevatron training on CPU..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUDA_VISIBLE_DEVICES=""
python -m factcheck_relevance.train --config configs/cpu_train.yaml
