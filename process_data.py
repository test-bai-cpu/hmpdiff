import os
import pandas as pd
import numpy as np
import traj_dataset


# =========================
# Dataset
# =========================

##### Process the data from atc1_all_processed.parquet ######
##### 1. Split to 60 seconds segments ######
##### 2. Normalize ######
df = pd.read_parquet("dataset/atc/atc1_all_processed.parquet")
df["orientation"] = df["orientation"] % (2 * np.pi)

###### Compute normalization stats ################
# min = df[["x", "y"]].min().values  # (2,)
# max = df[["x", "y"]].max().values  # (2,)
# np.savez("dataset/atc/atc1_normalization_stats.npz", min=min, max=max)
###############################################################

# ---------- Save processed splits ----------
obs_len = 4
pred_len = 60
debug = False
df_train, df_val, df_test = traj_dataset.train_val_test_split_by_person(df, val_ratio=0.1, test_ratio=0.1, obs_len=obs_len, pred_len=pred_len, seed=42, debug=debug)

if debug:
    folder = f"debug_{pred_len}"
    os.makedirs(f"dataset/atc/{folder}", exist_ok=True)
    df_train.to_parquet(f"dataset/atc/{folder}/atc1_train_split.parquet")
    df_val.to_parquet(f"dataset/atc/{folder}/atc1_val_split.parquet")
    df_test.to_parquet(f"dataset/atc/{folder}/atc1_test_split.parquet")
else:
    folder = f"full_{pred_len}"
    os.makedirs(f"dataset/atc/{folder}", exist_ok=True)
    df_train.to_parquet(f"dataset/atc/{folder}/atc1_train_split.parquet")
    df_val.to_parquet(f"dataset/atc/{folder}/atc1_val_split.parquet")
    df_test.to_parquet(f"dataset/atc/{folder}/atc1_test_split.parquet")