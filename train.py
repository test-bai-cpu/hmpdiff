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

import traj_dataset
from model import TrajectoryCFMModel, TrajectoryCFMModel_v2
import train_utils


# For exp version and logging
# for running: python train.py v1 > runs/v1.txt 2>&1


# =========================
# Config
# =========================
# version = sys.argv[1]
version = "v6-20"
checkpoint_dir = f"checkpoints/{version}"
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = f"runs/{version}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

debug = False

obs_len = 4
pred_len = 20
stride = 10
coord_dim = 2 
n_epochs = 100
sigma = 0.0

train_utils.set_random_seed(42)
device = torch.device('cuda')


# =========================
# Dataset
# =========================
if debug:
    df_train = pd.read_parquet("dataset/atc/debug/atc1_train_split.parquet")
    df_val   = pd.read_parquet("dataset/atc/debug/atc1_val_split.parquet")
    df_test  = pd.read_parquet("dataset/atc/debug/atc1_test_split.parquet")
else:
    df_train = pd.read_parquet("dataset/atc/full/atc1_train_split.parquet")
    df_val   = pd.read_parquet("dataset/atc/full/atc1_val_split.parquet")
    df_test  = pd.read_parquet("dataset/atc/full/atc1_test_split.parquet")

min, max = traj_dataset.load_normalization_stats("dataset/atc/atc1_normalization_stats.npz")
train_ds = traj_dataset.TrajectoryDataset.process_df(df_train, obs_len, pred_len, stride, min, max)
val_ds = traj_dataset.TrajectoryDataset.process_df(df_val,  obs_len, pred_len, stride, min, max)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)


# =========================
# Train
# =========================
D = 2 * pred_len # flattened future dimension
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
# model = TrajectoryCFMModel(obs_len, pred_len).to(device)
model = TrajectoryCFMModel_v2(obs_len, pred_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

best_val_loss = float("inf")
best_epoch = -1
global_step = 0

for epoch in range(1, n_epochs+1):
    model.train()
    train_loss = 0.0
    
    for X_obs, Y_fut, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        X_obs = X_obs.to(device)      # (B, obs_len, F)
        Y_fut = Y_fut.to(device)      # (B, pred_len, 2)

        B = X_obs.size(0)
        D = pred_len * 2

        x1 = Y_fut.reshape(B, -1)     # (B, D)
        # x0 = torch.randn_like(x1)     # (B, D)
        # x0 = (torch.rand_like(x1) * 2.0) - 1.0   # U[-1,1]
        eps = torch.randn(B, pred_len, 2, device=device)
        eps = torch.cumsum(eps, dim=1)                # makes it a random walk (smooth-ish)
        eps = eps / eps.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
        x0 = eps.view(B, -1)

        t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)  # each (B, D) except t (B,)
        t, x_t, u_t = t.to(device), x_t.to(device), u_t.to(device)
        
        x1_pred = model(x_t, t, X_obs)
        # w = 1.0 / (1 - t).clamp(min=1e-3)
        # w = w.clamp(max=100.0)
        # loss = (w[:,None] * (x1_pred - x1).pow(2)).mean()
        loss = (x1_pred - x1).pow(2).mean()
        
        # v_pred = model(x_t, t, X_obs)
        # x1_pred = x_t + (1-t[:, None]) * v_pred
        # # loss = ((x1_pred - x1) ** 2).mean()
        # w = 1.0 / (1 - t).clamp(min=1e-3)
        # loss = (w[:,None] * (x1_pred - x1).pow(2)).mean()
        
        y = x1_pred.view(B, pred_len, 2)
        vel = y[:, 1:] - y[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        loss = loss + 1e-3 * acc.pow(2).mean()

        # loss = ((v_pred - u_t) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * B

        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        global_step += 1
        
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f}")
        
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_obs, Y_fut, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
            X_obs = X_obs.to(device)
            Y_fut = Y_fut.to(device)

            B = X_obs.size(0)
            x1 = Y_fut.reshape(B, -1)
            # x0 = (torch.rand_like(x1) * 2.0) - 1.0
            
            eps = torch.randn(B, pred_len, 2, device=device)
            eps = torch.cumsum(eps, dim=1)                # makes it a random walk (smooth-ish)
            eps = eps / eps.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
            x0 = eps.view(B, -1)

            t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
            t, x_t, u_t = t.to(device), x_t.to(device), u_t.to(device)
            
            
            x1_pred = model(x_t, t, X_obs)
            w = 1.0 / (1 - t).clamp(min=1e-3)
            w = w.clamp(max=100.0)
            loss = (w[:,None] * (x1_pred - x1).pow(2)).mean()
            
            # v_pred = model(x_t, t, X_obs)
            # x1_pred = x_t + v_pred * (1-t[:, None])
            # w = 1.0 / (1 - t).clamp(min=1e-3)
            # loss = (w[:,None] * (x1_pred - x1).pow(2)).mean()

            y = x1_pred.view(B, pred_len, 2)
            vel = y[:, 1:] - y[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            loss = loss + 1e-3 * acc.pow(2).mean()

            # loss = ((v_pred - u_t) ** 2).mean()
            val_loss += loss.item() * B
    
    val_loss /= len(val_loader.dataset)
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

