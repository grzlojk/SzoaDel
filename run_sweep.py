import subprocess
import os
import shutil
import sys

# --- CONFIGURATION ---
PYTHON_EXEC = "python3"
SCRIPT_PATH = "SpaDE/main.py"
CONCEPTS_SCRIPT = "SpaDE/find_concepts.py"
DATA_DIR = "data/real_data"
CATEGORY = "mixed"
MODEL_TYPE = "clip"
# 1 epoch ~ 16 iterations
# 300 * 16 = 4800
# ITERATIONS = (
#     4800  # 25000 # all models converge far below 250 epochs (most around 100-150)
# )
ITERATIONS = 12000
EXPANSION = 4
BATCH_SIZE = 2048

os.makedirs("experiment_logs", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/models_sweep", exist_ok=True)
os.makedirs("results/concepts", exist_ok=True)

ACTIVATIONS_PATH = f"data/activations/activations_{MODEL_TYPE}_{CATEGORY}.pt"

print("Phase 0: Checking Data Cache")
if not os.path.exists(ACTIVATIONS_PATH):
    print("Cache not found. Running extraction once...")
    subprocess.run(
        [
            "uv",
            "run",
            PYTHON_EXEC,
            SCRIPT_PATH,
            "--SAE",
            "TopK",  # Dummy model just to trigger data load
            "--model_type",
            MODEL_TYPE,
            "--category",
            CATEGORY,
            "--data_dir",
            DATA_DIR,
            "--data_subdir",
            "images",
            "--iterations",
            "1",  # Exit immediately
            "--no_log",
        ]
    )
else:
    print(f"Cache found at {ACTIVATIONS_PATH}. Skipping extraction.")


# --- THE SWEEP LIST ---
experiments = []

# 1. TopK: Vary K
for k in [16, 32, 64, 128]:
    experiments.append(
        {
            "SAE": "TopK",
            "k": k,
            "lr": 1e-3,
            "lambda_val": 0,
            "fix": False,
            "tag": f"k{k}",
        }
    )

# 2. ReLU: Vary Lambda
for lam in [1e-4, 1e-3]:
    experiments.append(
        {
            "SAE": "ReLU",
            "k": 32,
            "lr": 1e-3,
            "lambda_val": lam,
            "fix": False,
            "tag": f"lam{lam}",
        }
    )

# 3. SpaDE & SzpaDeL Variants
# We test aggressive lambda (0.05) to force sparsity vs moderate (0.01)
spade_models = [
    "SpaDE",
    "SzpaDeLDiag",
    "SzpaDeLDiagLocal",
    "SzpaDeL",
    "SzpaDeLRank1",
    "SzpaDeLMultiHead",
]

for sae in spade_models:
    for lam in [0.01, 0.05]:
        experiments.append(
            {
                "SAE": sae,
                "k": 32,
                "lr": 5e-4,
                "lambda_val": lam,
                "fix": True,
                "tag": f"lam{lam}",
            }
        )

    experiments.append(
        {
            "SAE": sae,
            "k": 32,
            "lr": 5e-4,
            "lambda_val": 0,  # 0 -> Triggers "None" in main.py (Default Paper Init)
            "fix": False,  # False -> Learnable Parameter (Original Paper behavior)
            "tag": "lamSpaDE",
        }
    )

# --- EXECUTION LOOP ---
total = len(experiments)
print(f"Starting Smart Sweep with {total} experiments...")

for i, exp in enumerate(experiments):
    sae = exp["SAE"]
    tag = exp["tag"]
    lr = exp["lr"]
    lam = exp["lambda_val"]
    k = exp["k"]
    fix = exp["fix"]

    print(f"\n[{i + 1}/{total}] Processing {sae} | {tag} ...")

    # --- 1. TRAIN MODEL ---

    default_csv = f"metrics_{sae}_{MODEL_TYPE}_{CATEGORY}.csv"
    default_pth = f"{sae}_{MODEL_TYPE}_{CATEGORY}.pth"

    cmd = [
        "uv",
        "run",
        PYTHON_EXEC,
        SCRIPT_PATH,
        "--SAE",
        sae,
        "--model_type",
        MODEL_TYPE,
        "--category",
        CATEGORY,
        "--data_dir",
        DATA_DIR,
        "--data_subdir",
        "images",
        "--iterations",
        str(ITERATIONS),
        "--expansion_factor",
        str(EXPANSION),
        "--batch_size",
        str(BATCH_SIZE),
        "--lr",
        str(lr),
        "--k",
        str(k),
        "--no_log",
        "--save_model",
        "--models_subdir",
        "results/models_sweep",
        "--metrics_subdir",
        "results/metrics",
    ]
    if lam > 0:
        cmd.extend(["--lambda_val", str(lam)])
    if fix:
        cmd.append("--fix_lambda")

    log_file = f"experiment_logs/sweep_{sae}_{tag}.log"

    with open(log_file, "w") as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)
        except subprocess.CalledProcessError:
            print(f"!!! Training Failed for {sae} {tag}. Check {log_file}")
            continue

    # --- 2. RENAME ARTIFACTS ---

    src_csv = os.path.join("results/metrics", default_csv)
    dst_csv = os.path.join(
        "results/metrics", f"metrics_{sae}_{MODEL_TYPE}_{CATEGORY}_{tag}.csv"
    )
    if os.path.exists(src_csv):
        shutil.move(src_csv, dst_csv)

    src_pth = os.path.join("results/models_sweep", default_pth)
    dst_pth = os.path.join(
        "results/models_sweep", f"{sae}_{MODEL_TYPE}_{CATEGORY}_{tag}.pth"
    )

    if os.path.exists(src_pth):
        shutil.move(src_pth, dst_pth)
        print(f"   -> Model saved: {os.path.basename(dst_pth)}")
    else:
        print("   !!! Model file not found. Skipping visualization.")
        continue

    # --- 3. GENERATE CONCEPTS ---

    print(f"   -> Generating concepts...")
    vis_cmd = [
        "uv",
        "run",
        PYTHON_EXEC,
        CONCEPTS_SCRIPT,
        "--data_path",
        ACTIVATIONS_PATH,
        "--model_path",
        dst_pth,
        "--sae_type",
        sae,
        "--expansion_factor",
        str(EXPANSION),
        "--k",
        str(k),
        "--output_dir",
        f"results/concepts/{sae}_{tag}",
        "--num_features",
        "10",  # Visualize top 10 features
    ]

    with open(log_file, "a") as f:
        f.write("\n\n--- VISUALIZATION LOG ---\n")
        subprocess.run(vis_cmd, stdout=f, stderr=f)

print("\n========================================")
print("Sweep Completed. Running Plotter...")
print("========================================")

subprocess.run(
    [
        "uv",
        "run",
        "python3",
        "SpaDE/benchmark_plot_detailed.py",
        "--metrics_dir",
        "results/metrics",
        "--output_file",
        "results/benchmark_sweep_detailed.png",
    ]
)

print("Done!")
print("1. Plot: results/benchmark_sweep_detailed.png")
print("2. Concepts: results/concepts/")
