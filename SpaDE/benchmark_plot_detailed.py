import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse
import seaborn as sns
from adjustText import adjust_text


def setup_publication_style():
    """
    Sets up Matplotlib parameters for a clean, publication-quality look.
    """
    sns.set_theme(style="ticks", context="talk", font_scale=1.1)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.dpi": 300,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="results/metrics")
    parser.add_argument(
        "--output_file", type=str, default="results/benchmark_sweep_detailed.png"
    )
    args = parser.parse_args()

    # --- 1. Data Loading ---
    csv_files = glob.glob(os.path.join(args.metrics_dir, "*.csv"))
    if not csv_files:
        print("No metrics CSV found!")
        return

    data = []
    print(f"Found {len(csv_files)} files. Parsing...")

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            last_row = df.iloc[-1]
            mse = last_row.get("val/MSE", last_row.get("train/MSE", 0))
            l0 = last_row.get("val/L0", last_row.get("train/L0", 0))

            filename = os.path.basename(f)
            name_no_ext = filename.replace("metrics_", "").replace(".csv", "")
            parts = name_no_ext.split("_")
            sae_name = parts[0]

            if len(parts) >= 4:
                raw_tag = parts[3]
                # Format tag nicely (e.g., lam0.05 -> λ=0.05)
                tag = raw_tag.replace("lam", "λ=").replace("k", "k=")
            else:
                tag = ""

            data.append(
                {
                    "SAE": sae_name,
                    "MSE": mse,
                    "L0": l0,
                    "Tag": tag,
                }
            )
        except Exception as e:
            print(f"Skipping {f}: {e}")

    df_res = pd.DataFrame(data)
    if df_res.empty:
        return

    # --- 2. Plotting ---
    setup_publication_style()
    plt.figure(figsize=(10, 7))

    unique_models = df_res["SAE"].unique()
    palette = sns.color_palette("bright", n_colors=len(unique_models))

    # X = L0 (Sparsity), Y = MSE (Reconstruction)
    sns.scatterplot(
        data=df_res,
        x="L0",
        y="MSE",
        hue="SAE",
        style="SAE",
        s=150,
        alpha=0.9,
        palette=palette,
        edgecolor="black",
        linewidth=1,
    )

    # --- 3. Annotations with Fix ---
    texts = []
    for i, row in df_res.iterrows():
        # Only label if tag exists to avoid clutter
        if row["Tag"]:
            texts.append(
                plt.text(
                    row["L0"],
                    row["MSE"],
                    row["Tag"],
                    fontsize=10,
                    color="#333333",
                    weight="semibold",
                )
            )

    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            alpha=0.6,
            lw=1,
            shrinkA=5,
            shrinkB=5,
        ),
        force_points=0.3,
        force_text=0.3,
    )

    # --- 4. Labels ---
    plt.title(
        "SAE Benchmark: Sparsity vs Reconstruction", fontsize=18, pad=20, weight="bold"
    )
    plt.xlabel(r"Active Neurons ($L_0$)", fontsize=14, weight="bold")
    plt.ylabel(r"Reconstruction Loss (MSE)", fontsize=14, weight="bold")

    plt.legend(title="Architecture", title_fontsize=12, fontsize=11, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output_file}")


if __name__ == "__main__":
    main()
