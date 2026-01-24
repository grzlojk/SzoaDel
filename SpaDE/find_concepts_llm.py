import os
import torch
import argparse
from transformers import AutoTokenizer
import sys
import html

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

    # print("Loading data...")
    data = torch.load(args.data_path, map_location="cpu")
    activations = data["activations"]
    token_ids = data["token_ids"]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    input_dim = activations.shape[-1]

    model = load_sae_model(
        args.sae_type, args.model_path, input_dim, args.expansion_factor, args.k
    )
    model.to(device)

    # print("Computing Latents...")
    batch_size = 512
    all_latents = []
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            # This logic can be simplified if all models have the same output signature
            _, latents = model(batch)
            all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents, dim=0)

    # --- Pre-compute statistics ---
    # print("Analyzing features...")
    max_acts, _ = all_latents.max(dim=0)
    sorted_vals, sorted_indices = torch.sort(max_acts, descending=True)

    # Calculate feature sparsity (L0 norm)
    feature_sparsity = (all_latents > 1e-6).float().mean(dim=0)

    # --- HTML Report Generation ---
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

    for i in range(20):  # Show top 20 features
        feat_idx = sorted_indices[i].item()

        # Skip features that never activate
        if max_acts[feat_idx] == 0:
            continue

        feat_acts = all_latents[:, feat_idx]
        top_vals, top_inds = torch.topk(feat_acts, k=10)

        sparsity = feature_sparsity[feat_idx].item()

        header = f"Feature {feat_idx}"
        html_output += f'<div class="feature"><h2>{header}</h2>'
        html_output += f'<div class="metadata"><strong>Max Activation:</strong> {top_vals[0]:.4f} | <strong>Sparsity (L0):</strong> {sparsity:.4%}</div>'

        # --- Display Top Activating Examples ---
        html_output += "<h3>Top Activating Examples</h3>"
        for j, idx in enumerate(top_inds):
            idx = idx.item()
            val = top_vals[j].item()

            # Use the max activation for this specific feature as the ceiling for color scaling
            max_act_for_feature = top_vals[0].item()

            start = max(0, idx - 10)
            end = min(len(token_ids), idx + 10)

            context_tokens = token_ids[start:end]
            context_activations = all_latents[start:end, feat_idx]
            rel_idx = idx - start

            decoded_parts = []
            for t_i, (t, act_val_tensor) in enumerate(
                zip(context_tokens, context_activations)
            ):
                act_val = act_val_tensor.item()
                word = tokenizer.decode([t])
                word_html = html.escape(word)  # Sanitize for HTML

                style_str = ""
                # Main highlight for the single most activating token in this example
                if t_i == rel_idx:
                    style_str = 'class="peak-highlight"'
                # Secondary heatmap for any activating token
                elif act_val > 0.01:  # Threshold to avoid coloring everything
                    # Scale intensity from 0 to 1 based on this feature's max activation
                    intensity = min(act_val / max_act_for_feature, 1.0)
                    # Use a blue color (rgba for transparency)
                    style_str = f'style="background-color: rgba(0, 123, 255, {intensity:.2f}); color: {"white" if intensity > 0.5 else "black"}; padding: 2px 4px; border-radius: 3px;"'

                decoded_parts.append(f"<span {style_str}>{word_html}</span>")

            html_output += (
                f'<div class="context">{"".join(decoded_parts)} (act: {val:.2f})</div>'
            )
        html_output += "</div>"

    html_output += "</div></body></html>"

    # Save HTML report
    output_path = os.path.join(args.output_dir, f"{args.sae_type}_concepts.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Rich report saved to {output_path}")


if __name__ == "__main__":
    main()
