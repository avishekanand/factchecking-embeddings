#!/bin/bash
set -e

CONFIG=${1:-configs/baseline_gemma.yaml}

echo "🚀 Starting Baseline Evaluation: $CONFIG"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 1. Encode corpus with prefixing and cache
echo "--- Encoding Corpus ---"
python -m factcheck_relevance.baseline_encode --config $CONFIG

# 2. Encode queries with prefixing and cache
echo "--- Encoding Queries ---"
python -m factcheck_relevance.baseline_encode --config $CONFIG --is_query

# 3. Retrieve
echo "--- Running Retrieval ---"
python -m factcheck_relevance.retrieve --config $CONFIG

# 4. Evaluate
echo "--- Evaluating Results ---"
python -m factcheck_relevance.eval \
  --qrels $(grep "qrels_path" $CONFIG | awk '{print $2}' | tr -d '"') \
  --run $(grep "run_path" $CONFIG | awk '{print $2}' | tr -d '"')

echo "✅ Baseline evaluation completed!"
