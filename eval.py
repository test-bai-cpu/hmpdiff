import os
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from mod.mod_inference import load_mod_model, load_mod_feature_model
from torchcfm.models.basic_transformer.transformer import VisionTransformerCFMWrapper
from torchcfm.conditional_flow_matching import *

import traj_dataset
from model import TrajectoryCFMModel, TrajectoryCFMModel_v2
from eval_utils import evaluate_cfm, make_person_loader
import train_utils

# version = "full-v8-20-k-mod-sigma0.2"
version = "full-v8-20-k-gt-sigma0.2"
debug = False

obs_len = 4
pred_len = 20
stride = 10
K = 8   # number of samples to generate for each input

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
test_ds = traj_dataset.TrajectoryDataset.process_df(df_test,  obs_len, pred_len, stride, min, max)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=8)

device = torch.device('cuda')

# model = TrajectoryCFMModel(obs_len=4, pred_len=pred_len)
model = TrajectoryCFMModel_v2(obs_len=4, pred_len=pred_len)
ckpt = torch.load(f"checkpoints/{version}/best_model.pth", map_location=device)
# ckpt = torch.load(f"checkpoints/{version}/epoch_10.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)

train_utils.set_random_seed(42)
# mod_model = load_mod_model("mod/best.pt", device=device)
mod_model = load_mod_feature_model("mod/feature_best.pt", device=device)

ade, fde = evaluate_cfm(
    model,
    test_loader,
    K=K,
    pred_len=pred_len,
    min=min,    # to get ADE/FDE in real units
    max=max,
    device=device,
    mod_model=mod_model,
)

# chosen_person_id = 10319384800

# person_loader = make_person_loader(
#     test_loader,
#     chosen_person_id,
#     meta_csv="traj_dataset_metadata_v1.csv",
#     batch_size=64,
#     shuffle=False,
# )

# evaluate_cfm(model, person_loader, K=8, pred_len=20, min=min, max=max, device="cuda")