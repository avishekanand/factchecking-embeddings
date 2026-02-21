#!/bin/bash
set -e

echo "Evaluating retrieval performance..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m factcheck_relevance.eval \
  --qrels data/tevatron/dev_qrels.tsv \
  --run runs/factcheck_relevance_cpu/dev.run
