import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
from tqdm import tqdm

from models import SpaDE, TopKSAE, ReLUSAE
from data_utils import prepare_data

def get_args():
    parser = argparse.ArgumentParser(description="Train SAE on ViT activations")
    
    # Dane
    parser.add_argument("--img_examples", type=int, default=10000, help="Number of images")
    parser.add_argument("--data_path", type=str, default="activations.pt")
    parser.add_argument("--val_split", type=float, default=0.1) # Mniejszy val split, żeby więcej poszło na trening
    
    # Model Architecture
    parser.add_argument("--SAE", type=str, choices=["SpaDE", "TopK", "ReLU"], required=True)
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--expansion_factor", type=int, default=4)
    
    # Hiperparametry
    parser.add_argument("--k", type=int, default=32, help="k for TopK SAE")
    parser.add_argument("--lambda_val", type=float, default=None, 
                        help="Initial Scale for SpaDE (if None, uses paper default), L1 coeff for ReLU")
    parser.add_argument("--fix_lambda", action="store_true", help="Fix SpaDE scale")
    
    # Trening (Zgodnie z Paperem)
    # Jeśli podasz --iterations, --epochs zostanie zignorowane/przeliczone
    parser.add_argument("--iterations", type=int, default=8000, help="Total training iterations (overrides epochs)")
    parser.add_argument("--epochs", type=int, default=None, help="Explicit epochs (if iterations is not set)")
    parser.add_argument("--batch_size", type=int, default=512, help="Paper uses 512")
    parser.add_argument("--lr", type=float, default=1e-2, help="Paper uses 1e-2")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Cosine decay target")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    
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
        "val/L0": val_l0_sum / num_batches
    }

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not args.no_log:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    print("Loading data...")
    activations = prepare_data(args.img_examples, save_path=args.data_path)
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
        print(f"Set epochs to {args.epochs} to satisfy {args.iterations} iterations (batches/epoch: {batches_per_epoch})")
    
    total_steps = args.epochs * batches_per_epoch
    print(f"Total training steps: {total_steps}")

    # --- Model Init ---
    latent_dim = args.input_dim * args.expansion_factor
    print(f"Initializing {args.SAE} | Latent: {latent_dim}")
    
    if args.SAE == "SpaDE":
        model = SpaDE(args.input_dim, latent_dim, 
                      initial_scale=args.lambda_val, 
                      fix_scale=args.fix_lambda).to(device)
    elif args.SAE == "TopK":
        model = TopKSAE(args.input_dim, latent_dim, k=args.k).to(device)
    elif args.SAE == "ReLU":
        model = ReLUSAE(args.input_dim, latent_dim).to(device)
    
    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Scheduler: Cosine Decay od LR do min_LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )
    
    print(f"Starting training on {device}...")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
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
            scheduler.step() # Step scheduler every iteration
            
            if not args.no_log and global_step % 20 == 0:
                mse_val, l0_val = calculate_metrics(x, x_hat, z)
                logs = {
                    "train/loss": loss.item(),
                    "train/MSE": mse_val,
                    "train/L0": l0_val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch
                }
                if args.SAE == "SpaDE" and not args.fix_lambda:
                    # Logujemy faktyczną wartość skali po softplus
                    logs["train/spade_scale"] = torch.nn.functional.softplus(model.scale_param).item()
                
                wandb.log(logs)
            
            global_step += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.1e}"})
            
            if global_step >= args.iterations:
                print("Reached target iterations.")
                break
        
        # Walidacja
        val_metrics = evaluate(model, val_loader, device, args)
        if not args.no_log:
            val_metrics["epoch"] = epoch
            wandb.log(val_metrics)
        print(f"  Val MSE: {val_metrics['val/MSE']:.6f} | L0: {val_metrics['val/L0']:.2f}")

        if global_step >= args.iterations:
            break

    torch.save(model.state_dict(), f"{args.SAE}_sae.pth")
    if not args.no_log:
        wandb.finish()

if __name__ == "__main__":
    main()