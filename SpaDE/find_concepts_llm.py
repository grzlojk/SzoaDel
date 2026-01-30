import os
import torch
import argparse
from transformers import AutoTokenizer
import sys
import html
from tqdm import tqdm
import heapq

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


def load_sae_model(sae_type, model_path, input_dim, expansion_factor, k=32):
    """Loads the SAE model based on the specified type."""
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
    elif sae_type == "HybridSAE":
        model = HybridSAE(input_dim, latent_dim)
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

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    input_dim = activations.shape[-1]
    latent_dim = input_dim * args.expansion_factor
    num_samples = activations.shape[0]

    model = load_sae_model(
        args.sae_type, args.model_path, input_dim, args.expansion_factor, args.k
    )
    model.to(device)

    # --- Pass 1: Compute aggregate statistics in a memory-efficient way ---
    print("Analyzing features (Pass 1/2): Calculating aggregate statistics...")
    max_acts = torch.full((latent_dim,), -torch.inf, device="cpu")
    feature_non_zero_counts = torch.zeros(latent_dim, dtype=torch.int64, device="cpu")

    batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Pass 1"):
            batch = activations[i : i + batch_size].to(device)
            _, latents = model(batch)

            # Update max activations
            batch_max_acts = latents.max(dim=0).values.cpu()
            max_acts = torch.max(max_acts, batch_max_acts)

            # Update non-zero counts
            feature_non_zero_counts += (latents > 1e-6).sum(dim=0).cpu()

    feature_sparsity = feature_non_zero_counts.float() / num_samples
    sorted_vals, sorted_indices = torch.sort(max_acts, descending=True)

    # --- Pass 2: Find top activating examples for the top features ---
    print("Analyzing features (Pass 2/2): Finding top examples...")
    num_features_to_report = 20
    num_examples_per_feature = 10

    features_to_analyze = sorted_indices[:num_features_to_report]

    # Use a dictionary of min-heaps to track top-k examples for each feature
    top_examples = {feat_idx.item(): [] for feat_idx in features_to_analyze}

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Pass 2"):
            batch = activations[i : i + batch_size].to(device)
            _, latents = model(batch)

            for feat_idx in features_to_analyze:
                feat_idx_item = feat_idx.item()
                activations_for_feature = latents[:, feat_idx]

                for local_idx, act_val in enumerate(activations_for_feature):
                    act_val_item = act_val.item()
                    global_idx = i + local_idx

                    # Add to heap if it's not full, or if the current item is larger than the smallest in the heap
                    if len(top_examples[feat_idx_item]) < num_examples_per_feature:
                        heapq.heappush(
                            top_examples[feat_idx_item], (act_val_item, global_idx)
                        )
                    elif act_val_item > top_examples[feat_idx_item][0][0]:
                        heapq.heapreplace(
                            top_examples[feat_idx_item], (act_val_item, global_idx)
                        )

    # Sort the results for display
    for feat_idx in top_examples:
        top_examples[feat_idx].sort(key=lambda x: x[0], reverse=True)

    # --- HTML Report Generation ---
    print("Generating HTML report...")
    html_output = f"""
    <html>
    <head>
        <title>SAE Concepts for {args.sae_type}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; color: #212529; }}
            h1, h2, h3 {{ color: #343a40; }}
            .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .feature {{ border: 1px solid #dee2e6; padding: 15px; margin-bottom: 25px; border-radius: 5px; }}
            .feature h2 {{ color: #007bff; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            .peak-highlight {{ background-color: #dc3545; color: white; padding: 2px 4px; border-radius: 3px; }}
            .context {{ display: block; margin-bottom: 8px; padding: 5px; border-radius: 3px; font-family: monospace; font-size: 1.1em; }}
            .metadata {{ font-size: 0.9em; color: #6c757d; margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visualizing Top Concepts for: {args.sae_type}</h1>
    """

    for feat_idx in features_to_analyze:
        feat_idx = feat_idx.item()
        if max_acts[feat_idx] <= 1e-6:
            continue

        sparsity = feature_sparsity[feat_idx].item()
        max_act_for_feature = max_acts[feat_idx].item()

        html_output += f'<div class="feature"><h2>Feature {feat_idx}</h2>'
        html_output += f'<div class="metadata"><strong>Max Activation:</strong> {max_act_for_feature:.4f} | <strong>Sparsity (L0):</strong> {1 - sparsity:.4%}</div>'
        html_output += "<h3>Top Activating Examples</h3>"

        # --- Display Top Activating Examples ---
        with torch.no_grad():
            for val, idx in top_examples[feat_idx]:
                start = max(0, idx - 10)
                end = min(len(token_ids), idx + 10)
                rel_idx = idx - start

                # Re-compute latents for just this small context window
                context_activations_input = activations[start:end].to(device)
                _, context_latents = model(context_activations_input)
                context_activations = context_latents[:, feat_idx].cpu()

                context_tokens = token_ids[start:end]
                decoded_parts = []
                for t_i, (t, act_val_tensor) in enumerate(
                    zip(context_tokens, context_activations)
                ):
                    act_val = act_val_tensor.item()
                    word = tokenizer.decode([t])
                    word_html = html.escape(word)

                    style_str = ""
                    if t_i == rel_idx:
                        style_str = 'class="peak-highlight"'
                    elif act_val > 0.01:
                        intensity = min(act_val / max_act_for_feature, 1.0)
                        style_str = f'style="background-color: rgba(0, 123, 255, {intensity:.2f}); color: {"white" if intensity > 0.5 else "black"}; padding: 2px 4px; border-radius: 3px;"'

                    decoded_parts.append(f"<span {style_str}>{word_html}</span>")

                html_output += f'<div class="context">{"".join(decoded_parts)} (act: {val:.2f})</div>'
        html_output += "</div>"

    html_output += "</div></body></html>"

    output_path = os.path.join(args.output_dir, f"{args.sae_type}_concepts.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Rich report saved to {output_path}")


if __name__ == "__main__":
    main()
