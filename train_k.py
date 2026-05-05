import os
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchcfm.models.basic_transformer.transformer import VisionTransformerCFMWrapper
from torchcfm.conditional_flow_matching import *
from torch.utils.tensorboard import SummaryWriter

from mod.mod_inference import load_mod_feature_model, mod_loss_per_sample

import traj_dataset
from model import TrajectoryCFMModel, TrajectoryCFMModel_v2
import train_utils
from general_utils import denorm_xy_torch
from eval_utils import sample_future_cfm_euler_k


# For exp version and logging
# for running: python train.py v1 > runs/v1.txt 2>&1

# =========================
# Config
# =========================
# version = sys.argv[1]
param_setup_v = sys.argv[1]
pred_len = int(sys.argv[2])
sigma = float(sys.argv[3])
if_mod = bool(int(sys.argv[4]))
K = int(sys.argv[5])
lambda_mod = float(sys.argv[6])

# version = f"V{param_setup_v}-pred{pred_len}-sigma{sigma}-ut{if_ut}-mod{if_mod}-div{if_div}-smooth{if_smooth}"
version = f"V{param_setup_v}-pred{pred_len}-sigma{sigma}-mod{if_mod}-K{K}-lambdaMod{lambda_mod}"
checkpoint_dir = f"checkpoints/{version}"
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = f"runs/{version}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

debug = False

obs_len = 4
stride = pred_len // 2
coord_dim = 2
n_epochs = 100

##### for generating k samples #####
# K = 10
# lambda_mod = 1e-3

lr = 1e-3
lr_min = 1e-6  # cosine annealing floor (like MoFlow)

lambda_gt = 1.0
lambda_smooth = 1e-2

train_utils.set_random_seed(42)
device = torch.device('cuda')

### print all config for logging
print("Experiment version:", version)
print("Config:")
print(f"  pred_len: {pred_len}")
print(f"  sigma: {sigma}")
print(f"  if_mod: {if_mod}")
print(f"  K: {K}")
print(f"  lambda_gt: {lambda_gt}")
print(f"  lambda_mod: {lambda_mod}")
print(f"  lambda_smooth: {lambda_smooth}")

# =========================
# Dataset
# =========================
if debug:
    df_train = pd.read_parquet(f"dataset/atc/debug_{pred_len}/atc1_train_split.parquet")
    df_val   = pd.read_parquet(f"dataset/atc/debug_{pred_len}/atc1_val_split.parquet")
    df_test  = pd.read_parquet(f"dataset/atc/debug_{pred_len}/atc1_test_split.parquet")
else:
    df_train = pd.read_parquet(f"dataset/atc/full_{pred_len}/atc1_train_split.parquet")
    df_val   = pd.read_parquet(f"dataset/atc/full_{pred_len}/atc1_val_split.parquet")
    df_test  = pd.read_parquet(f"dataset/atc/full_{pred_len}/atc1_test_split.parquet")

min_bound, max_bound = traj_dataset.load_normalization_stats("dataset/atc/atc1_normalization_stats.npz")
data_min = torch.as_tensor(min_bound, device=device, dtype=torch.float32)
data_max = torch.as_tensor(max_bound, device=device, dtype=torch.float32)
train_ds = traj_dataset.TrajectoryDataset.process_df(df_train, obs_len, pred_len, stride, min_bound, max_bound)
val_ds = traj_dataset.TrajectoryDataset.process_df(df_val,  obs_len, pred_len, stride, min_bound, max_bound)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=16, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=16)

# =========================
# Train
# =========================
D = 2 * pred_len # flattened future dimension
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
# model = TrajectoryCFMModel(obs_len, pred_len).to(device)
model = TrajectoryCFMModel_v2(obs_len, pred_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

# --- Cosine annealing LR with linear warmup (like MoFlow) ---
total_steps = n_epochs * len(train_loader)
warmup_steps = max(1, int(total_steps * 0.05))  # 5% warmup
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: max(lr_min / lr, step / warmup_steps)
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=lr_min
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
)

mod_model = load_mod_feature_model("mod/feature_best.pt", device=device)

best_minADE = float("inf")
best_epoch = -1
val_n_steps = 10  # ODE integration steps for validation sampling
global_step = 0

for epoch in range(1, n_epochs+1):
    model.train()
    train_loss_acc = torch.zeros((), device=device)
    
    for X_obs, Y_fut, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        X_obs = X_obs.to(device)      # (B, obs_len, F)
        Y_fut = Y_fut.to(device)      # (B, pred_len, 2)

        B = X_obs.size(0)
        D = pred_len * 2

        x1 = Y_fut.reshape(B, -1)     # (B, D)

        eps = torch.randn(B, K, pred_len, 2, device=device)
        eps = torch.cumsum(eps, dim=2)                # makes it a random walk (smooth-ish)
        eps = eps / eps.abs().amax(dim=(2,3), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
        x0 = eps.view(B, K, D)

        # Flatten to (B*K, D) for FM + model
        x0_flat = x0.reshape(B*K, D)
        x1_flat = x1[:, None].expand(B, K, D).reshape(B*K, D)
        X_obs_flat = X_obs[:, None].expand(B, K, *X_obs.shape[1:]).reshape(B*K, *X_obs.shape[1:])

        # Sample (t, x_t, u_t) for all B*K, K has same t
        # t, x_t, u_t = FM.sample_location_and_conditional_flow(x0_flat, x1_flat)
        # t_sample = torch.rand(B, device=device) * 0.99 # (B,)
        t_sample = train_utils.sample_t_logit_normal(B, device=device, mu_t=-0.5, sigma_t=1.5)
        t = t_sample[:, None].expand(B, K).reshape(B*K)             # (B*K,) — same t for all K
        
        ######## Use sigma ################################
        # if if_ut:
        mu_t = x0_flat + t[:, None] * (x1_flat - x0_flat)
        x_t = mu_t + sigma * torch.randn_like(mu_t)
        # print(f"ut is true, using sigma {sigma} for sampling x_t")
        ######## Or without sigma, just straight interpolation ########
        # else:
        #     x_t = x0_flat + t[:, None] * (x1_flat - x0_flat)
        #########################################################
        
        u_t = x1_flat - x0_flat
        
        t, x_t, u_t = t.to(device), x_t.to(device), u_t.to(device)

        # Predict x1
        x1_pred_flat = model(x_t, t, X_obs_flat)        # (B*K, D)
        x1_pred = x1_pred_flat.view(B, K, D)            # (B, K, D)
        
        # Get best-of-k loss
        gt_mse = (x1_pred - x1[:, None]).pow(2).mean(dim=-1)  # (B, K)
        k_star = gt_mse.argmin(dim=1)     # (B,)
        x1_star = x1_pred[torch.arange(B, device=device), k_star] # (B, D)

        raw_mse = (x1_star - x1).pow(2).mean(dim=-1)           # (B,)
        # L_gt = (raw_mse / (1 - t_sample).clamp(min=1e-2).pow(2)).mean()  # scalar
        L_gt = raw_mse.mean() # change from v14.
        
        # MoD loss
        y_pred = x1_pred.view(B, K, pred_len, 2)          # (B,K,T,2)
        
        last_xy = X_obs[:, -1, :2]                    # (B,2) normalized absolute
        L_anchor = (y_pred[:, :, 0, :] - last_xy[:, None, :]).pow(2).mean()
        
        L_mod_bk = mod_loss_per_sample(
            Y_pred_norm=y_pred,
            X_obs_norm=X_obs,
            mod_model=mod_model,
            min_np=data_min,
            max_np=data_max,
            dt=1.0,
        )  # (B,K)

        # L_mod = L_mod_bk.mean()
        # E = L_mod_bk
        # E0 = E.detach().median()      # or a fixed constant / running statistic
        # L_mod = torch.relu(E - E0).mean()
        L_mod = L_mod_bk.mean() # mean over batch and K.

        # y_pred: (B,K,T,2)
        # y_centered = y_pred - y_pred[:, :, :1, :]          # (B,K,T,2) starts at 0
        # y_flat = y_centered.reshape(B, K, -1)  # (B,K,2T)

        # # pairwise distances
        # diff = y_flat[:, :, None, :] - y_flat[:, None, :, :]      # (B,K,K,2T)
        # dist2 = (diff**2).mean(dim=-1)                            # (B,K,K)
        # mask = ~torch.eye(K, device=device, dtype=torch.bool)          # (K,K)
        # mask = mask.unsqueeze(0).expand(B, K, K)                       # (B,K,K)
        # dist2_off = dist2[mask].view(B, K*(K-1))            # off-diagonal only
        # L_div = torch.exp(-dist2_off / 0.1).mean()

        #smooth loss       
        y_all = x1_pred.view(B, K, pred_len, 2)         # (B,K,T,2)
        vel = y_all[:, :, 1:] - y_all[:, :, :-1]        # (B,K,T-1,2)
        acc = vel[:, :, 1:] - vel[:, :, :-1]            # (B,K,T-2,2)
        L_smooth = acc.pow(2).mean()

        # loss = lambda_gt * L_gt + lambda_smooth * L_smooth + lambda_mod * L_mod + lambda_div * L_div + L_anchor
        # loss = lambda_gt * L_gt + (lambda_smooth * L_smooth if if_smooth else 0) + (lambda_mod * L_mod if if_mod else 0) + (lambda_div * L_div if if_div else 0)
        loss = lambda_gt * L_gt + lambda_smooth * L_smooth + (lambda_mod * L_mod if if_mod else 0)
        # print("loss components includes: if_mod ", if_mod, ", if_div ", if_div, ", if_smooth ", if_smooth)
        # print("train loss values are: L_gt ", L_gt.item(), ", L_mod ", L_mod.item(), ", L_div ", L_div.item(), ", L_smooth ", L_smooth.item(), ", L_anchor ", L_anchor.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        # train_loss += loss.item() * B
        train_loss_acc += loss.detach() * B

        if global_step % 100 == 0:
            writer.add_scalar("Loss/train_step", loss.detach().item(), global_step)  # occasional sync
        global_step += 1

    train_loss = (train_loss_acc / len(train_loader.dataset)).item()
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f}")
        
    # =========================================================================
    # Validation: full ODE inference, not training loss.
    # Instead of sampling a random t and computing the denoising loss,
    # we integrate the full ODE from noise to clean trajectories (100 Euler
    # steps) and compute minADE/minFDE in real-world meters.
    # =========================================================================
    model.eval()
    all_min_ade = []
    all_min_fde = []
    with torch.no_grad():
        for X_obs, Y_fut, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val inference]"):
            X_obs = X_obs.to(device)      # (B, obs_len, F)
            Y_fut = Y_fut.to(device)      # (B, pred_len, 2)

            # Full ODE integration: noise -> clean trajectory
            Yk_pred_norm = sample_future_cfm_euler_k(
                model, X_obs, pred_len=pred_len, K=K,
                n_steps=val_n_steps, device=device,
            )  # (B, K, pred_len, 2)

            # Compute minADE/minFDE in real units
            Yk_pred_real = denorm_xy_torch(Yk_pred_norm, min_bound, max_bound)          # (B,K,T,2)
            Y_true_real  = denorm_xy_torch(Y_fut, min_bound, max_bound)[:, None]       # (B,1,T,2)
            dist = torch.linalg.norm(Yk_pred_real - Y_true_real, dim=-1)   # (B,K,T)
            ade_k = dist.mean(dim=2)   # (B,K)
            fde_k = dist[:, :, -1]     # (B,K)

            min_ade, _ = ade_k.min(dim=1)   # (B,)
            min_fde, _ = fde_k.min(dim=1)   # (B,)
            all_min_ade.append(min_ade.cpu())
            all_min_fde.append(min_fde.cpu())

    val_minADE = torch.cat(all_min_ade).mean().item()
    val_minFDE = torch.cat(all_min_fde).mean().item()

    writer.add_scalar("Loss/train_epoch", train_loss, epoch)
    writer.add_scalar("Val/minADE", val_minADE, epoch)
    writer.add_scalar("Val/minFDE", val_minFDE, epoch)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.3f}, Val minADE@{K} = {val_minADE:.4f}, Val minFDE@{K} = {val_minFDE:.4f}. Best minADE = {best_minADE:.4f} (epoch {best_epoch})")
    
    if val_minADE < best_minADE:
        best_minADE = val_minADE
        best_epoch = epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_minADE": val_minADE,
                "val_minFDE": val_minFDE,
            },
            f"checkpoints/{version}/best_model.pth"
        )
        
    if epoch % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_minADE": val_minADE,
                "val_minFDE": val_minFDE,
            },
            f"checkpoints/{version}/epoch_{epoch}.pth"
        )
        
writer.close()
print(f"Training finished. Best epoch = {best_epoch} (val minADE={best_minADE:.6f})")