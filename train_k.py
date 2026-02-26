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


# For exp version and logging
# for running: python train.py v1 > runs/v1.txt 2>&1

# =========================
# Config
# =========================
# version = sys.argv[1]
param_setup_v = sys.argv[1]
pred_len = int(sys.argv[2])
sigma = float(sys.argv[3])
if_ut = bool(int(sys.argv[4]))
if_mod = bool(int(sys.argv[5]))
if_div = bool(int(sys.argv[6]))
if_smooth = bool(int(sys.argv[7]))

version = f"V{param_setup_v}-predlen{pred_len}-sigma{sigma}-ut{if_ut}-mod{if_mod}-div{if_div}-smooth{if_smooth}"

# version = "v8-20-k-mod"
# version = "full-v8-20-k-gt-sigma0.2"
# version = "full-v8-20-k-mod"
# version = "full-v8-20-k-mod-sigma0.2"
# version = "full-v8-20-k-gt"
# version = "v7-20-k-gt-smooth"
checkpoint_dir = f"checkpoints/{version}"
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = f"runs/{version}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# debug = True
debug = False

obs_len = 4
# pred_len = 20
stride = pred_len // 2
coord_dim = 2
n_epochs = 200
# sigma = 0.2
# sigma = 0.0

##### for generating k samples #####
K = 5
lambda_gt = 1.0
lambda_mod = 1e-3
lambda_smooth = 1e-3
lambda_div = 1e-3


train_utils.set_random_seed(42)
device = torch.device('cuda')


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

min, max = traj_dataset.load_normalization_stats("dataset/atc/atc1_normalization_stats.npz")
data_min = torch.as_tensor(min, device=device, dtype=torch.float32)
data_max = torch.as_tensor(max, device=device, dtype=torch.float32)
train_ds = traj_dataset.TrajectoryDataset.process_df(df_train, obs_len, pred_len, stride, min, max)
val_ds = traj_dataset.TrajectoryDataset.process_df(df_val,  obs_len, pred_len, stride, min, max)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=16, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=16)

# =========================
# Train
# =========================
D = 2 * pred_len # flattened future dimension
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
# model = TrajectoryCFMModel(obs_len, pred_len).to(device)
model = TrajectoryCFMModel_v2(obs_len, pred_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

mod_model = load_mod_feature_model("mod/feature_best.pt", device=device)

best_val_loss = float("inf")
best_epoch = -1
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

        # Sample (t, x_t, u_t) for all B*K
        # t, x_t, u_t = FM.sample_location_and_conditional_flow(x0_flat, x1_flat)
        # t = torch.rand(B*K, device=device) * 0.999
        t = torch.rand(B*K, device=device)
        
        ######## Use sigma ################################
        if if_ut:
            mu_t = x0_flat + t[:, None] * (x1_flat - x0_flat)
            x_t = mu_t + sigma * torch.randn_like(mu_t)
        ######## Or without sigma, just straight interpolation ########
        else:
            x_t = x0_flat + t[:, None] * (x1_flat - x0_flat)
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
        L_gt = (x1_star - x1).pow(2).mean()
        
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
        y_centered = y_pred - y_pred[:, :, :1, :]          # (B,K,T,2) starts at 0
        y_flat = y_centered.reshape(B, K, -1)  # (B,K,2T)

        # pairwise distances
        diff = y_flat[:, :, None, :] - y_flat[:, None, :, :]      # (B,K,K,2T)
        dist2 = (diff**2).mean(dim=-1)                            # (B,K,K)
        mask = ~torch.eye(K, device=device, dtype=torch.bool)          # (K,K)
        mask = mask.unsqueeze(0).expand(B, K, K)                       # (B,K,K)
        dist2_off = dist2[mask].view(B, K*(K-1))            # off-diagonal only
        L_div = torch.exp(-dist2_off / 0.1).mean()

        #smooth loss # OPTIONS: can be change to all K samples instead of just the best one
        # y_star = x1_star.view(B, pred_len, 2)
        # vel = y_star[:, 1:] - y_star[:, :-1]
        # acc = vel[:, 1:] - vel[:, :-1]
        # L_smooth = acc.pow(2).mean()
        
        y_all = x1_pred.view(B, K, pred_len, 2)         # (B,K,T,2)
        vel = y_all[:, :, 1:] - y_all[:, :, :-1]        # (B,K,T-1,2)
        acc = vel[:, :, 1:] - vel[:, :, :-1]            # (B,K,T-2,2)
        L_smooth = acc.pow(2).mean()

        # loss = lambda_gt * L_gt + lambda_smooth * L_smooth + lambda_mod * L_mod + lambda_div * L_div + L_anchor
        loss = lambda_gt * L_gt + (lambda_smooth * L_smooth if if_smooth else 0) + (lambda_mod * L_mod if if_mod else 0) + (lambda_div * L_div if if_div else 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # train_loss += loss.item() * B
        train_loss_acc += loss.detach() * B

        if global_step % 100 == 0:
            writer.add_scalar("Loss/train_step", loss.detach().item(), global_step)  # occasional sync
        global_step += 1

    train_loss = (train_loss_acc / len(train_loader.dataset)).item()
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f}")
        
    # Validation
    model.eval()
    val_loss_acc = torch.zeros((), device=device)
    with torch.no_grad():
        for X_obs, Y_fut, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
            X_obs = X_obs.to(device)
            Y_fut = Y_fut.to(device)

            B = X_obs.size(0)
            x1 = Y_fut.reshape(B, -1)
            
            eps = torch.randn(B, K, pred_len, 2, device=device)
            eps = torch.cumsum(eps, dim=2)                # makes it a random walk (smooth-ish)
            eps = eps / eps.abs().amax(dim=(2,3), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
            x0 = eps.view(B, K, D)

            # Flatten to (B*K, D) for FM + model
            x0_flat = x0.reshape(B*K, D)
            x1_flat = x1[:, None].expand(B, K, D).reshape(B*K, D)
            X_obs_flat = X_obs[:, None].expand(B, K, *X_obs.shape[1:]).reshape(B*K, *X_obs.shape[1:])

            # Sample (t, x_t, u_t) for all B*K
            # t, x_t, u_t = FM.sample_location_and_conditional_flow(x0_flat, x1_flat)
            # t, x_t, u_t = t.to(device), x_t.to(device), u_t.to(device)
            t = torch.rand(B*K, device=device)
            
            ######## Use sigma ################################
            if if_ut:
                mu_t = x0_flat + t[:, None] * (x1_flat - x0_flat)
                x_t = mu_t + sigma * torch.randn_like(mu_t)
            ######## Or without sigma, just straight interpolation ########
            else:
                x_t = x0_flat + t[:, None] * (x1_flat - x0_flat)
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
            L_gt = (x1_star - x1).pow(2).mean()
            
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
            y_centered = y_pred - y_pred[:, :, :1, :]          # (B,K,T,2) starts at 0
            y_flat = y_centered.reshape(B, K, -1)  # (B,K,2T)

            # pairwise distances
            diff = y_flat[:, :, None, :] - y_flat[:, None, :, :]      # (B,K,K,2T)
            dist2 = (diff**2).mean(dim=-1)                            # (B,K,K)
            mask = ~torch.eye(K, device=device, dtype=torch.bool)          # (K,K)
            mask = mask.unsqueeze(0).expand(B, K, K)                       # (B,K,K)
            dist2_off = dist2[mask].view(B, K*(K-1))            # off-diagonal only
            L_div = torch.exp(-dist2_off / 0.1).mean()                              # minimize => push apart

            #smooth loss # OPTIONS: can be change to all K samples instead of just the best one
            # y_star = x1_star.view(B, pred_len, 2)
            # vel = y_star[:, 1:] - y_star[:, :-1]
            # acc = vel[:, 1:] - vel[:, :-1]
            # L_smooth = acc.pow(2).mean()
            
            y_all = x1_pred.view(B, K, pred_len, 2)         # (B,K,T,2)
            vel = y_all[:, :, 1:] - y_all[:, :, :-1]        # (B,K,T-1,2)
            acc = vel[:, :, 1:] - vel[:, :, :-1]            # (B,K,T-2,2)
            L_smooth = acc.pow(2).mean()

            # loss = lambda_gt * L_gt + lambda_smooth * L_smooth + lambda_mod * L_mod + lambda_div * L_div + L_anchor
            loss = lambda_gt * L_gt + (lambda_smooth * L_smooth if if_smooth else 0) + (lambda_mod * L_mod if if_mod else 0) + (lambda_div * L_div if if_div else 0)

            val_loss_acc += loss * X_obs.size(0)
    
    val_loss = (val_loss_acc / len(val_loader.dataset)).item()
    writer.add_scalar("Loss/train_epoch", train_loss, epoch)
    writer.add_scalar("Loss/val_epoch", val_loss, epoch)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}. Best val = {best_val_loss:.3f} (epoch {best_epoch})")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f"checkpoints/{version}/best_model.pth"
        )
        
    if epoch % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f"checkpoints/{version}/epoch_{epoch}.pth"
        )
        
writer.close()
print(f"Training finished. Best epoch = {best_epoch} (val={best_val_loss:.6f})")

