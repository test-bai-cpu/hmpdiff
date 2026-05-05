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
from eval_utils import plot_person_from_df
import train_utils

from mod.mod_inference import load_mod_model, load_mod_feature_model

# version = "V2-fix-t-pred20-sigma0.2-utTrue-modFalse-divTrue-smoothTrue"
# version = "V2-fix-t-pred20-sigma0.2-utTrue-modTrue-divTrue-smoothTrue"

pred_len = 20
version = f"V9-pred{pred_len}-sigma0.1-modTrue-K5-lambdaMod0.001"
print("Evaluating version:", version)

# =========================
# Dataset
# =========================

##### Process the data from atc1_all_processed.parquet ######
##### 1. Split to 60 seconds segments ######
##### 2. Normalize ######

# df = pd.read_parquet("dataset/atc/atc1_all_processed.parquet")
# df = pd.read_parquet("dataset/atc/full/atc1_train_split.parquet")
# df = pd.read_parquet("dataset/atc/full/atc1_val_split.parquet")
df = pd.read_parquet("dataset/atc/full/atc1_test_split.parquet")
# df = pd.read_parquet(f"dataset/atc/full_frame_{pred_len}/atc1_test_split.parquet")
df["orientation"] = df["orientation"] % (2 * np.pi)
min, max = traj_dataset.load_normalization_stats("dataset/atc/atc1_normalization_stats.npz")
device = torch.device('cuda')

obs_len = 4
random_seed = 42
train_utils.set_random_seed(random_seed)

# model = TrajectoryCFMModel(obs_len=obs_len, pred_len=pred_len)
model = TrajectoryCFMModel_v2(obs_len=obs_len, pred_len=pred_len)
ckpt = torch.load(f"checkpoints/{version}/best_model.pth", map_location=device)
# ckpt = torch.load(f"checkpoints/{version}/epoch_10.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# person_id_list = pd.read_csv("traj_dataset_metadata.csv")["person_id"].tolist()

# person_id = 102419543700
# person_id_list = [10249210800]
# person_id_list = [10319384800]
# person_id_list = [103114004300, 103114004700, 103114005200, 103114005901, 103114010000, 103114010101, 103114010800, 103114011600, 103114012401, 103114024600]

# person_id_list = [112519582000, 111410250900, 111815102704]
# person_id_list = [103115183500, 110419493100]
# person_id_list = [102413552400,102813341601]


person_id_list = [111410250900] ## pred=20
# person_id_list = [103115183500] ## pred=40


## get person_id from the file traj_dataset_metadata

mod_model = load_mod_feature_model("mod/feature_best.pt", device=device)

for person_id in person_id_list:
    plot_person_from_df(
        df=df,
        person_id=person_id,
        model=model,
        min=min,
        max=max,
        obs_len=obs_len,
        pred_len=pred_len,
        device=device,
        mod_model=mod_model,
        random_seed=random_seed,
        version=version,
    )
    
    
