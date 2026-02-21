#!/bin/bash
set -e

# Change to project root if script is run from scripts/
cd "$(dirname "$0")/.."

echo "🚀 Starting Full FactCheck Tevatron Pipeline on CPU"

bash scripts/run_build.sh
bash scripts/run_train_cpu.sh
bash scripts/run_encode.sh
bash scripts/run_retrieve.sh
bash scripts/run_eval.sh

echo "✅ Pipeline completed successfully!"
