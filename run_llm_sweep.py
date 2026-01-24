import subprocess
import os
import shutil
import sys

# --- CONFIGURATION ---
PYTHON_EXEC = "python3"
SCRIPT_PATH = "SpaDE/main.py"
# Update to use the LLM viz script
CONCEPTS_SCRIPT = "SpaDE/find_concepts_llm.py"
MODEL_TYPE = "qwen"
CATEGORY = "fineweb"
ITERATIONS = 100  # 4800
EXPANSION = 4
BATCH_SIZE = 512

os.makedirs("experiment_logs", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/models_llm", exist_ok=True)
os.makedirs("results/concepts_llm", exist_ok=True)

ACTIVATIONS_PATH = "data/activations/activations_qwen_fineweb.pt"

if not os.path.exists(ACTIVATIONS_PATH):
    print("Activations not found. Running extraction...")
    subprocess.run(["uv", "run", PYTHON_EXEC, "extract_llm_data.py"], check=True)

experiments = []

# 1. TopK (Baseline)
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


# --- LOOP ---
total = len(experiments)
print(f"Starting Smart Sweep with {total} experiments...")
for i, exp in enumerate(experiments):
    sae = exp["SAE"]
    tag = exp["tag"]
    lr = exp["lr"]
    lam = exp["lambda_val"]
    k = exp["k"]
    fix = exp["fix"]

    print(f"\n[{i + 1}/{total}] Running {sae} | {tag} ...")

    # Filenames
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
        "results/models_llm",
        "--metrics_subdir",
        "results/metrics",
    ]
    if lam > 0:
        cmd.extend(["--lambda_val", str(lam)])
    if fix:
        cmd.append("--fix_lambda")

    # Run Training
    log_file = f"experiment_logs/llm_{sae}_{tag}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=f)

    # Rename CSV
    src_csv = os.path.join("results/metrics", default_csv)
    dst_csv = os.path.join(
        "results/metrics", f"metrics_{sae}_{MODEL_TYPE}_{CATEGORY}_{tag}.csv"
    )
    if os.path.exists(src_csv):
        shutil.move(src_csv, dst_csv)

    # Rename Model
    src_pth = os.path.join("results/models_llm", default_pth)
    dst_pth = os.path.join(
        "results/models_llm", f"{sae}_{MODEL_TYPE}_{CATEGORY}_{tag}.pth"
    )
    if os.path.exists(src_pth):
        shutil.move(src_pth, dst_pth)

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
        f"results/concepts_llm/{sae}_{tag}",
    ]
    subprocess.run(vis_cmd)


subprocess.run(
    [
        "uv",
        "run",
        "python3",
        "SpaDE/benchmark_plot_detailed.py",
        "--metrics_dir",
        "results/metrics",
        "--output_file",
        "results/benchmark_llm_sweep.png",
    ]
)

print("Done! Check results/metrics/ and results/concepts_llm/")
