#!/bin/bash
set -euo pipefail

# Ensure we have data
if [ ! -d "data/real_data/mixed/images" ]; then
    echo "Downloading Real Data..."
    uv run python3 download_real_data.py
fi

# Configuration
MODEL_TYPE="clip"
CATEGORY="mixed" # Matches folder name in download script
DATA_DIR="data/real_data"
DATA_SUBDIR="images"

# List of SAEs to benchmark
SAES=("TopK" "ReLU" "SpaDE" "SzpaDeLDiag" "SzpaDeL" "SzpaDeLRank1" "SzpaDeLMultiHead")

# 1 epoch ~ 16 iterations
# 250 * 16 = 4000
ITERATIONS=4000 # 25000 # all models converge far below 250 epochs (most around 100-150)
EXPANSION=4 # d_model (768) * 4 = 3072 features
K=32

mkdir -p experiment_logs
mkdir -p results/concepts

echo "========================================================"
echo " PHASE 1: Extract Activations (Happens once)"
echo "========================================================"
# Note: We rely on main.py to call prepare_data internally.

echo "Activations will be extracted during the first model run."

echo "========================================================"
echo " PHASE 2: Train SAEs"
echo "========================================================"

for sae in "${SAES[@]}"; do
    echo "------------------------------------------------"
    echo "Training SAE: $sae on ImageNet-subset"
    echo "------------------------------------------------"

    uv run python3 SpaDE/main.py \
        --SAE "$sae" \
        --model_type "$MODEL_TYPE" \
        --category "$CATEGORY" \
        --data_dir "$DATA_DIR" \
        --data_subdir "$DATA_SUBDIR" \
        --iterations "$ITERATIONS" \
        --expansion_factor "$EXPANSION" \
        --k "$K" \
        --batch_size 512 \
        --lr 1e-3 \
        --no_log \
        --save_model \
        > "experiment_logs/${sae}_real.log" 2>&1
        
    # Note: I increased batch_size to 512 for stability 
    # and lowered LR slightly to 1e-3 for safety.
done

echo "========================================================"
echo " PHASE 3: Visualize Learned Concepts"
echo "========================================================"

for sae in "${SAES[@]}"; do
    echo "Generating images for $sae..."
    uv run python3 SpaDE/find_concepts.py \
        --data_path "data/activations/activations_${MODEL_TYPE}_${CATEGORY}.pt" \
        --model_path "results/models/${sae}_${MODEL_TYPE}_${CATEGORY}.pth" \
        --sae_type "$sae" \
        --expansion_factor "$EXPANSION" \
        --k "$K" \
        --output_dir "results/concepts/${sae}_real" \
        --num_features 10
done

echo "========================================================"
echo " PHASE 4: Benchmark Plot"
echo "========================================================"

uv run python3 SpaDE/benchmark_plot.py \
    --metrics_dir "results/metrics" \
    --output_file "results/real_data_benchmark.png"

echo "DONE. Results at:"
echo "1. Benchmark: results/real_data_benchmark.png"
echo "2. Concepts: results/concepts/"
