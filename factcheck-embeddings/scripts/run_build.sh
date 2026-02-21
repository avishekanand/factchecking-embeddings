#!/bin/bash
set -e

echo "Building Tevatron datasets..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m factcheck_relevance.build_data --config configs/data_build.yaml
