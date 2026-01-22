import os
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import (
    SpaDE,
    TopKSAE,
    ReLUSAE,
    SzpaDeLDiag,
    SzpaDeLDiagLocal,
    SzpaDeL,
    SzpaDeLLocal,
    SzpaDeLRank1,
    SzpaDeLMultiHead,
)


def load_sae_model(sae_type, model_path, input_dim, expansion_factor, k=32):
    latent_dim = input_dim * expansion_factor
    if sae_type == "SpaDE":
        model = SpaDE(input_dim, latent_dim)
    elif sae_type == "TopK":
        model = TopKSAE(input_dim, latent_dim, k=k)
    elif sae_type == "ReLU":
        model = ReLUSAE(input_dim, latent_dim)
    elif sae_type == "SzpaDeLDiag":
        model = SzpaDeLDiag(input_dim, latent_dim)
    elif sae_type == "SzpaDeLDiagLocal":
        model = SzpaDeLDiagLocal(input_dim, latent_dim)
    elif sae_type == "SzpaDeL":
        model = SzpaDeL(input_dim, latent_dim)
    elif sae_type == "SzpaDeLRank1":
        model = SzpaDeLRank1(input_dim, latent_dim)
    elif sae_type == "SzpaDeLMultiHead":
        model = SzpaDeLMultiHead(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown SAE: {sae_type}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to activations .pt"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth")
    parser.add_argument("--sae_type", type=str, required=True)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--top_n_images", type=int, default=9, help="Images per feature"
    )
    parser.add_argument(
        "--num_features", type=int, default=20, help="How many features to visualize"
    )
    parser.add_argument("--output_dir", type=str, default="results/concepts")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data: {args.data_path}")
    data = torch.load(args.data_path, map_location="cpu")
    activations = data["activations"].to(dtype=torch.float32)  # [N, D]
    metadata = data["metadata"]  # List of file paths

    if activations.shape[0] != len(metadata):
        print(
            f"Warning: {activations.shape[0]} vectors vs {len(metadata)} images. Truncating to match."
        )
        min_len = min(activations.shape[0], len(metadata))
        activations = activations[:min_len]
        metadata = metadata[:min_len]

    input_dim = activations.shape[-1]

    print(f"Loading Model: {args.sae_type}")
    model = load_sae_model(
        args.sae_type, args.model_path, input_dim, args.expansion_factor, args.k
    )
    model.to(device)

    print("Computing Latents...")
    batch_size = 1024
    all_latents = []

    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            _, latents = model(batch)
            all_latents.append(latents.cpu())

    # [N_images, N_features]
    all_latents = torch.cat(all_latents, dim=0)

    # Identify "Alive" features (highest max activation)
    # We want to look at features that actually fire.
    max_activations, _ = all_latents.max(dim=0)
    sorted_vals, sorted_feature_indices = torch.sort(max_activations, descending=True)

    print(f"Visualizing top {args.num_features} most active features...")

    for i in range(args.num_features):
        feature_idx = sorted_feature_indices[i].item()

        # Get activations for this specific feature across all images
        feat_acts = all_latents[:, feature_idx]

        # Get top N images for this feature
        top_vals, top_indices = torch.topk(feat_acts, k=args.top_n_images)

        grid_size = int(math.ceil(math.sqrt(args.top_n_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.suptitle(f"Feature {feature_idx} (Max Act: {top_vals[0]:.2f})", fontsize=16)

        axes = axes.flatten()

        for j, ax in enumerate(axes):
            if j < len(top_indices):
                img_idx = top_indices[j].item()
                val = top_vals[j].item()
                img_path = metadata[img_idx]

                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    fname = os.path.basename(img_path)
                    ax.set_title(f"{fname}\nAct: {val:.2f}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error loading", ha="center")

            ax.axis("off")

        out_name = f"{args.sae_type}_feat_{feature_idx}.png"
        save_path = os.path.join(args.output_dir, out_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
