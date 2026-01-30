#!/bin/bash

set -euo pipefail

# Configuration
MODELS=("clip") # ("clip" "dino")
CATEGORIES=("car") # ("car" "hat" "mug" "shirt") # Include "all" to train on all.
SAES=("TopK" "ReLU" "SpaDE" "SzpaDeLDiag" "SzpaDeLDiagLocal" "SzpaDeL" "SzpaDeLRank1" "SzpaDeLMultiHead") # No SzpaDeLLocal since it OOMs.
ITERATIONS=2000
EXPANSION=4
K=6 # param from color_img.py

# Create a logs directory if it doesn't exist
mkdir -p experiment_logs

# Run experiments for each model and category
for model in "${MODELS[@]}"; do
    for cat in "${CATEGORIES[@]}"; do
        echo "------------------------------------------------"
        echo "Processing Category: $cat | Model: $model"
        echo "------------------------------------------------"

        for sae in "${SAES[@]}"; do
            echo "Training $sae..."

            # Run the training script
            # --no_log is used here to avoid requiring W&B login in all environments,
            # remove it if you have W&B configured.
            uv run python3 SpaDE/main.py \
                --SAE "$sae" \
                --model_type "$model" \
                --category "$cat" \
                --iterations "$ITERATIONS" \
                --expansion_factor "$EXPANSION" \
                --no_log \
                --save_model \
                --epochs 10 \
                --batch_size 32 \
                --data_subdir output_k$K \
                > "experiment_logs/${sae}_${model}_${cat}.log" 2>&1

            uv run python3 SpaDE/visualize_pca.py \
                --data_path "data/activations/activations_${model}_${cat}.pt" \
                --model_path "results/models/${sae}_${model}_${cat}.pth" \
                --sae_type "$sae" \
                --dims 3
        done
    done
done

echo "All experiments completed."
