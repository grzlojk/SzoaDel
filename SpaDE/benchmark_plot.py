import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="results/metrics")
    parser.add_argument(
        "--output_file", type=str, default="results/benchmark_pareto.png"
    )
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(args.metrics_dir, "*.csv"))
    if not csv_files:
        print("No metrics CSV found!")
        return

    data = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                last_row = df.iloc[-1]

                filename = os.path.basename(f)
                parts = filename.replace("metrics_", "").replace(".csv", "").split("_")
                sae_name = parts[0]

                entry = {
                    "SAE": sae_name,
                    "MSE": last_row.get("val/MSE", last_row.get("train/MSE", 0)),
                    "L0": last_row.get("val/L0", last_row.get("train/L0", 0)),
                    "File": filename,
                }
                data.append(entry)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df_res = pd.DataFrame(data)
    print("Benchmark Results:")
    print(df_res)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    sns.scatterplot(
        data=df_res, x="L0", y="MSE", hue="SAE", style="SAE", s=150, palette="tab10"
    )

    for i, row in df_res.iterrows():
        plt.text(row["L0"], row["MSE"] + 0.005, row["SAE"], fontsize=9)

    plt.title("SAE Benchmark: Sparsity vs Reconstruction (Lower is Better)")
    plt.xlabel("L0 (Number of Active Neurons)")
    plt.ylabel("MSE (Reconstruction Loss)")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    plt.savefig(args.output_file, dpi=300)
    print(f"Plot saved to {args.output_file}")


if __name__ == "__main__":
    main()
