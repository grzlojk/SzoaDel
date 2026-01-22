#!/bin/bash

# 1. Train all models (Existing logic)
# Ensure run_experiments.sh is executed first or uncomment lines below:
# bash run_experiments.sh

MODELS=("clip")
CATEGORIES=("car") 
SAES=("TopK" "ReLU" "SpaDE" "SzpaDeLDiag" "SzpaDeL" "SzpaDeLRank1" "SzpaDeLMultiHead")
EXPANSION=4
K=6

echo "=========================================="
echo "      Generating Concept Visualizations"
echo "=========================================="

for model_type in "${MODELS[@]}"; do
    for cat in "${CATEGORIES[@]}"; do
        for sae in "${SAES[@]}"; do
            echo "Visualizing concepts for $sae..."
            
            uv run python3 SpaDE/find_concepts.py \
                --data_path "data/activations/activations_${model_type}_${cat}.pt" \
                --model_path "results/models/${sae}_${model_type}_${cat}.pth" \
                --sae_type "$sae" \
                --expansion_factor "$EXPANSION" \
                --k "$K" \
                --output_dir "results/concepts/${sae}_${cat}"
        done
    done
done

echo "=========================================="
echo "      Running Benchmark (MSE vs L0)"
echo "=========================================="

uv run python3 SpaDE/benchmark_plot.py \
    --metrics_dir "results/metrics" \
    --output_file "results/benchmark_plot.png"

echo "Done! Check results/concepts/ and results/benchmark_plot.png"
