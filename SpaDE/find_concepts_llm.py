import os
import torch
import argparse
from transformers import AutoTokenizer
import sys

# Import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import (
    SpaDE,
    TopKSAE,
    ReLUSAE,
    SzpaDeLDiag,
    SzpaDeLDiagLocal,
    SzpaDeL,
    SzpaDeLRank1,
    SzpaDeLMultiHead,
    HybridSAE,
)


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    # Background colors
    BG_RED = "\033[41m"


def load_sae_model(sae_type, model_path, input_dim, expansion_factor, k=32):
    latent_dim = input_dim * expansion_factor
    if sae_type == "Hybrid":
        model = HybridSAE(input_dim, latent_dim, k=k)
    elif sae_type == "TopK":
        model = TopKSAE(input_dim, latent_dim, k=k)
    elif sae_type == "SpaDE":
        model = SpaDE(input_dim, latent_dim)
    elif sae_type == "SzpaDeLRank1":
        model = SzpaDeLRank1(input_dim, latent_dim)
    elif sae_type == "SzpaDeLDiag":
        model = SzpaDeLDiag(input_dim, latent_dim)
    elif sae_type == "SzpaDeLDiagLocal":
        model = SzpaDeLDiagLocal(input_dim, latent_dim)
    elif sae_type == "SzpaDeL":
        model = SzpaDeL(input_dim, latent_dim)
    elif sae_type == "ReLU":
        model = ReLUSAE(input_dim, latent_dim)
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sae_type", type=str, required=True)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results/concepts_llm")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    data = torch.load(args.data_path, map_location="cpu")
    activations = data["activations"]
    token_ids = data["token_ids"]

    # Load Tokenizer for decoding
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    input_dim = activations.shape[-1]
    # print(f"Input Dim: {input_dim}")

    model = load_sae_model(
        args.sae_type, args.model_path, input_dim, args.expansion_factor, args.k
    )
    model.to(device)

    print("Computing Latents...")
    # Run in batches
    batch_size = 512
    all_latents = []
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            if (
                args.sae_type == "Hybrid"
                or args.sae_type == "SpaDE"
                or "SzpaDeL" in args.sae_type
            ):
                _, latents = model(batch)
            else:
                _, latents = model(batch)
            all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents, dim=0)

    # Find Top Features
    max_acts, _ = all_latents.max(dim=0)
    sorted_vals, sorted_indices = torch.sort(max_acts, descending=True)

    print(f"\n--- Visualizing Top Concepts for {args.sae_type} ---\n")

    output_text = []
    output_text.append(f"Model: {args.sae_type}")

    for i in range(10):  # Show top 10 features
        feat_idx = sorted_indices[i].item()

        # Get top 10 examples for this feature
        feat_acts = all_latents[:, feat_idx]
        top_vals, top_inds = torch.topk(feat_acts, k=10)

        header = f"\nFeature {feat_idx} (Max Act: {top_vals[0]:.2f})"
        print(Colors.HEADER + header + Colors.ENDC)
        output_text.append(header)

        for j, idx in enumerate(top_inds):
            idx = idx.item()
            val = top_vals[j].item()

            # Get context (e.g. 10 tokens before, 5 after)
            start = max(0, idx - 10)
            end = min(len(token_ids), idx + 5)

            context_tokens = token_ids[start:end]

            # Identify the specific token that fired
            # The token at 'idx' in the big flattened list corresponds to the activation
            # But we sliced [start:end], so relative index is:
            rel_idx = idx - start

            decoded_parts = []
            for t_i, t in enumerate(context_tokens):
                word = tokenizer.decode([t])
                if t_i == rel_idx:
                    decoded_parts.append(f"[{word}]({val:.1f})")  # Mark activation
                    print(Colors.BG_RED + word + Colors.ENDC, end="")
                else:
                    decoded_parts.append(word)
                    print(word, end="")
            print()  # Newline

            output_text.append("".join(decoded_parts))

    # Save text report
    with open(os.path.join(args.output_dir, f"{args.sae_type}_concepts.txt"), "w") as f:
        f.write("\n".join(output_text))


if __name__ == "__main__":
    main()
