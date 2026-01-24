import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
from tqdm import tqdm
import pandas as pd
import json

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
    HybridSAE,
)
from data_utils import prepare_data


def get_args():
    parser = argparse.ArgumentParser(description="Train SAE on ViT activations")

    # Data
    parser.add_argument("--data_dir", type=str, default="white obejcts")
    parser.add_argument("--data_subdir", type=str, default="output_k3")
    parser.add_argument(
        "--category", type=str, default=None, help="Category subfolder (e.g. car, hat)"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["clip", "dino", "qwen"], default="clip"
    )
    parser.add_argument("--activations_subdir", type=str, default="data/activations")
    parser.add_argument("--val_split", type=float, default=0.1)

    # Model Architecture
    parser.add_argument(
        "--SAE",
        type=str,
        choices=[
            "SpaDE",
            "TopK",
            "ReLU",
            "SzpaDeLDiag",
            "SzpaDeLDiagLocal",
            "SzpaDeL",
            "SzpaDeLLocal",
            "SzpaDeLRank1",
            "SzpaDeLMultiHead",
            "HybridSAE",
        ],
        required=True,
    )
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--expansion_factor", type=int, default=4)

    # Hiperparametry
    parser.add_argument("--k", type=int, default=32, help="k for TopK SAE")
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=None,
        help="Initial Scale for SpaDE family (if None, uses paper default), L1 coeff for ReLU",
    )
    parser.add_argument("--fix_lambda", action="store_true", help="Fix scale parameter")

    # Training
    parser.add_argument(
        "--iterations", type=int, default=100, help="Total training iterations"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Explicit epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Output
    parser.add_argument("--metrics_subdir", type=str, default="results/metrics")
    parser.add_argument(
        "--save_model", action="store_true", help="Explicitly save the trained model"
    )
    parser.add_argument("--models_subdir", type=str, default="results/models")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="sae-experiments")
    parser.add_argument("--no_log", action="store_true")

    args = parser.parse_args()

    # Default lambda logic
    if args.lambda_val is None:
        if args.SAE == "ReLU":
            args.lambda_val = 3e-4
        # Dla SpaDE zostawiamy None, bo klasa modelu sama wyliczy 1/(2*d)

    return args


def calculate_metrics(x, x_hat, z):
    mse = nn.functional.mse_loss(x_hat, x).item()
    l0 = (z > 0).float().sum(dim=-1).mean().item()
    return mse, l0


def evaluate(model, dataloader, device, args):
    model.eval()
    val_loss_sum = 0
    val_mse_sum = 0
    val_l0_sum = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            x_hat, z = model(x)

            mse_loss = nn.functional.mse_loss(x_hat, x)
            loss = mse_loss
            if args.SAE == "ReLU":
                loss += args.lambda_val * z.abs().sum(dim=-1).mean()

            mse_val, l0_val = calculate_metrics(x, x_hat, z)
            val_loss_sum += loss.item()
            val_mse_sum += mse_val
            val_l0_sum += l0_val
            num_batches += 1

    return {
        "val/loss": val_loss_sum / num_batches,
        "val/MSE": val_mse_sum / num_batches,
        "val/L0": val_l0_sum / num_batches,
    }


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    # Directories setup
    os.makedirs(args.activations_subdir, exist_ok=True)
    os.makedirs(args.metrics_subdir, exist_ok=True)
    if args.save_model:
        os.makedirs(args.models_subdir, exist_ok=True)

    cat_tag = args.category if args.category else "all"
    data_path = os.path.join(
        args.activations_subdir, f"activations_{args.model_type}_{cat_tag}.pt"
    )

    if not args.no_log:
        run_name = f"{args.SAE}_{args.model_type}_{cat_tag}"
        wandb.init(project=args.wandb_project, config=vars(args), name=run_name)

    print(f"Loading/Extracting data: {data_path}")
    activations = prepare_data(
        data_dir=args.data_dir,
        data_subdir=args.data_subdir,
        category=args.category,
        model_type=args.model_type,
        save_path=data_path,
    )

    activations = activations.to(torch.float32)
    if activations.shape[-1] != args.input_dim:
        print(f"Updating input_dim to {activations.shape[-1]}")
        args.input_dim = activations.shape[-1]

    full_dataset = TensorDataset(activations)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Obliczanie Epok na podstawie Iteracji ---
    batches_per_epoch = len(train_loader)
    if args.epochs is None:
        args.epochs = math.ceil(args.iterations / batches_per_epoch)
        print(
            f"Set epochs to {args.epochs} to satisfy {args.iterations} iterations (batches/epoch: {batches_per_epoch})"
        )

    total_steps = args.epochs * batches_per_epoch
    print(f"Total training steps: {total_steps}")

    total_steps = args.epochs * batches_per_epoch
    latent_dim = args.input_dim * args.expansion_factor
    print(f"Initializing {args.SAE} | Latent: {latent_dim}")

    if args.SAE == "SpaDE":
        model = SpaDE(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "SzpaDeLDiag":
        model = SzpaDeLDiag(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "SzpaDeLDiagLocal":
        model = SzpaDeLDiagLocal(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "SzpaDeL":
        model = SzpaDeL(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "SzpaDeLLocal":
        model = SzpaDeLLocal(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "SzpaDeLRank1":
        model = SzpaDeLRank1(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "SzpaDeLMultiHead":
        model = SzpaDeLMultiHead(
            args.input_dim,
            latent_dim,
            initial_scale=args.lambda_val,
            fix_scale=args.fix_lambda,
        ).to(device)
    elif args.SAE == "HybridSAE":
        model = HybridSAE(
            args.input_dim,
            latent_dim,
        ).to(device)
    elif args.SAE == "TopK":
        model = TopKSAE(args.input_dim, latent_dim, k=args.k).to(device)
    elif args.SAE == "ReLU":
        model = ReLUSAE(args.input_dim, latent_dim).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Scheduler: Cosine Decay od LR do min_LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    print(f"Starting training on {device}...")
    global_step = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            x = batch[0].to(device)
            x_hat, z = model(x)

            mse_loss = nn.functional.mse_loss(x_hat, x)
            loss = mse_loss

            if args.SAE == "ReLU":
                loss += args.lambda_val * z.abs().sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()

            # --- Gradient Clipping ---
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if global_step % 20 == 0:
                mse_val, l0_val = calculate_metrics(x, x_hat, z)
                logs = {
                    "step": global_step,
                    "train/loss": loss.item(),
                    "train/MSE": mse_val,
                    "train/L0": l0_val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                if hasattr(model, "scale_param") and not args.fix_lambda:
                    logs[f"train/scale"] = torch.nn.functional.softplus(
                        model.scale_param
                    ).item()

                if not args.no_log:
                    wandb.log(logs)
                history.append(logs)

            global_step += 1
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.1e}",
                }
            )

            if global_step >= args.iterations:
                print("Reached target iterations.")
                break

        # Walidacja
        val_metrics = evaluate(model, val_loader, device, args)
        val_metrics["epoch"] = epoch
        val_metrics["step"] = global_step
        if not args.no_log:
            wandb.log(val_metrics)
        history.append(val_metrics)
        print(
            f"  Val MSE: {val_metrics['val/MSE']:.6f} | L0: {val_metrics['val/L0']:.2f}"
        )

        if global_step >= args.iterations:
            break

    # Final name tag
    model_tag = f"{args.SAE}_{args.model_type}_{cat_tag}"

    # Save metrics
    metrics_path = os.path.join(args.metrics_subdir, f"metrics_{model_tag}.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Optional model save
    if args.save_model:
        model_path = os.path.join(args.models_subdir, f"{model_tag}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if not args.no_log:
        wandb.finish()


if __name__ == "__main__":
    main()
