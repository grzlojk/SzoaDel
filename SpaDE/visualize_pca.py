import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import re
import sys

sys.path.append(os.path.dirname(__file__))
from models import SpaDE, TopKSAE, ReLUSAE, SzpaDeLDiag, SzpaDeLDiagLocal, SzpaDeL, SzpaDeLLocal

def parse_color_from_path(path):
    """
    Extracts RGB values from filename like 'car_255_0_0.jpg'
    Returns normalized [R, G, B] in range [0, 1].
    """
    filename = os.path.basename(path)
    # Match pattern _R_G_B.jpg
    match = re.search(r'_(\d+)_(\d+)_(\d+)\.jpg$', filename)
    if match:
        r, g, b = map(int, match.groups())
        return [r / 255.0, g / 255.0, b / 255.0]
    else:
        raise ValueError("Invalid image name.")

def load_sae_model(sae_type, model_path, input_dim, expansion_factor, k=32):
    latent_dim = input_dim * expansion_factor
    models_map = {
        "SpaDE": SpaDE,
        "TopK": lambda i, l: TopKSAE(i, l, k=k),
        "ReLU": ReLUSAE,
        "SzpaDeLDiag": SzpaDeLDiag,
        "SzpaDeLDiagLocal": SzpaDeLDiagLocal,
        "SzpaDeL": SzpaDeL,
        "SzpaDeLLocal": SzpaDeLLocal,
    }

    if sae_type not in models_map:
        raise ValueError(f"Unknown SAE type: {sae_type}")

    if model_path is None:
        raise ValueError("--model_path is required when --sae_type is not Raw")

    model = models_map[sae_type](input_dim, latent_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Visualize color distribution in latent space using PCA")
    parser.add_argument("--data_path", type=str, required=True, help="Path to activations .pt file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained SAE .pth model (omit for Raw)")
    parser.add_argument(
        "--sae_type",
        type=str,
        choices=["Raw", "ReLU", "TopK", "SpaDE", "SzpaDeLDiag", "SzpaDeLDiagLocal", "SzpaDeL", "SzpaDeLLocal"],
        default="Raw",
    )
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--k", type=int, default=32, help="k for TopK SAE")
    parser.add_argument("--dims", type=int, default=3, choices=[2, 3], help="PCA dimensions")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")

    activations = data["activations"].to(torch.float32)
    metadata = data["metadata"]
    cat_tag = data["category"]

    raw_features = activations.cpu().numpy()

    if args.sae_type == "Raw":
        sae_features = raw_features
        feature_name1 = "Raw activations"
        model_tag = "Raw"
    else:
        model = load_sae_model(args.sae_type, args.model_path, activations.shape[-1], args.expansion_factor, args.k)
        print(f"Computing latent activations (z) using {args.sae_type} SAE...")
        with torch.no_grad():
            _, latents = model(activations)
        sae_features = latents.cpu().numpy()
        feature_name1 = f"Latent (z) - {args.sae_type}"
        model_tag = args.sae_type

    feature_name2 = "Raw activations"

    num_acts = raw_features.shape[0]
    num_meta = len(metadata)

    colors = []
    if num_acts > num_meta:
        patches_per_img = num_acts // num_meta
        print(f"Mapping {patches_per_img} patches per image to colors.")
        for path in metadata:
            c = parse_color_from_path(path)
            colors.extend([c] * patches_per_img)
    else:
        colors = [parse_color_from_path(path) for path in metadata]
    colors = np.array(colors[:num_acts])

    print(f"Running PCA ({args.dims}D) for SAE features...")
    pca1 = PCA(n_components=args.dims)
    pca_results1 = pca1.fit_transform(sae_features)

    print(f"Running PCA ({args.dims}D) for raw features...")
    pca2 = PCA(n_components=args.dims)
    pca_results2 = pca2.fit_transform(raw_features)

    # Create plot
    fig = plt.figure(figsize=(28, 11))
    if args.dims == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    axes = [ax1, ax2]
    for i, (pca_results, feature_name, ax) in enumerate([(pca_results1, feature_name1, axes[0]), (pca_results2, feature_name2, axes[1])]):
        if args.dims == 3:
            scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2],
                                 c=colors, alpha=0.5, s=5)
            ax.set_zlabel("PCA 3")
            ax.view_init(elev=20, azim=45)
        else:
            scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1],
                                 c=colors, alpha=0.5, s=5)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA of {feature_name} ({cat_tag})")
        ax.grid(True, linestyle='--', alpha=0.3)

    # Generate filename
    prefix = f"pca_{model_tag}_vs_raw_{cat_tag}_{args.dims}d".lower()
    save_path = os.path.join(args.output_dir, f"{prefix}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
