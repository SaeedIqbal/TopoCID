#!/bin/bash
# Compute NDE, NIE, and Minimal Sufficient Interventions on all 6 datasets

DATASETS=("GOOD-HIV" "GOOD-CMNIST" "DrugOOD-IC50" "MUTAG" "PROTEINS" "NCI1")
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=========================================="
echo "Starting Causal Verification Pipeline"
echo "=========================================="

for ds in "${DATASETS[@]}"; do
    echo ">>> Processing Dataset: $ds"
    python "$SCRIPT_DIR/run_experiments.py" --mode causal --dataset "$ds" --root "/home/phd/datasets/"
    if [ $? -ne 0 ]; then
        echo "Error encountered while processing $ds. Exiting."
        exit 1
    fi
done

echo "=========================================="
echo "Causal Verification Completed Successfully."
echo "=========================================="