import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import List, Tuple


class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, epoch_time: np.ndarray):
        """
        X: (N, obs_len, F)
        Y: (N, pred_len, 2)
        epoch_time: (N, obs_len + pred_len)
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.epoch_time = torch.from_numpy(epoch_time).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.epoch_time[idx]


    @classmethod
    def process_df(
        cls,
        df: pd.DataFrame,
        obs_len: int,
        pred_len: int,
        stride: int,
        min: np.ndarray,
        max: np.ndarray,
    ):
        """
        Build dataset from df using given normalization stats.
        """
        df_norm = normalize_df_min_max(df, min, max)
        X, Y, epoch_time = build_sequences_from_df(df_norm, obs_len=obs_len, pred_len=pred_len, stride=stride)
        ds = cls(X, Y, epoch_time)
        
        return ds

def train_val_test_split_by_person(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    obs_len: int = 4,
    pred_len: int = 60,
    seed: int = 0,
    debug: bool = False,
):
    """
    Split df into train/val/test by person_id (no leakage).
    """
    
    df = df.copy()
    min_traj_len = obs_len + pred_len
    counts = df.groupby("person_id").size()
    valid_ids = counts[counts >= min_traj_len].index.values
    df = df[df["person_id"].isin(valid_ids)]
    
    ##### keep the person ids with continous moving traj, removing the ones with idling behavior #####
    keep_ids = []
    for pid, g in df.groupby("person_id"):
        # Sort by time to ensure correct temporal order
        g_sorted = g.sort_values("epoch_time")
        if has_continuous_motion(g_sorted):
            keep_ids.append(pid)
    df = df[df["person_id"].isin(keep_ids)]

    print(f"[split] Keeping {len(keep_ids)} persons with >= {min_traj_len} steps and continuous motion.")

    rng = np.random.RandomState(seed)
    persons = df["person_id"].unique()
    rng.shuffle(persons)
    n_total = len(persons)

    # ---- DEBUG mode here ----
    if debug == True:
        k = max(1, int(n_total * 0.3))   # use only 30% of persons
        persons = persons[:k]
        print(f"[Debug mode] Using only fraction={0.3:.2f} -> {k} persons.")
    # --------------------------------

    n_total = len(persons)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_ids = persons[:n_test]
    val_ids = persons[n_test:n_test + n_val]
    train_ids = persons[n_test + n_val:]

    df_train = df[df["person_id"].isin(train_ids)].copy()
    df_val   = df[df["person_id"].isin(val_ids)].copy()
    df_test  = df[df["person_id"].isin(test_ids)].copy()

    return df_train, df_val, df_test


def build_sequences_from_df(
    df: pd.DataFrame,
    obs_len: int = 4,
    pred_len: int = 60,
    stride: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    df: parquet-loaded DataFrame with columns:
        ['epoch_time', 'person_id', 'x', 'y', 'speed', 'orientation']
    Returns:
        X: (N, obs_len, F)
        Y: (N, pred_len, 2)   # predict future (x, y)
    """
    
    feature_cols = ["x", "y", "speed", "orientation"]

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    time_list: List[np.ndarray] = []

    pid_list = []
    seq_idx_list = []

    # Ensure sorted per person
    df = df.sort_values(["person_id", "epoch_time"]).copy()
    seq_len = obs_len + pred_len
    
    for pid, g in df.groupby("person_id"):
        g = g.sort_values("epoch_time")
        arr = g[feature_cols].to_numpy(dtype=np.float32)

        times = g["epoch_time"].to_numpy()   # same length as arr
        idxs = g.index.to_numpy()            # original df indices

        T = len(arr)
        
        if T < seq_len:
            continue  # too short

        # Sliding windows
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            seq = arr[start:end]  # (seq_len, F)

            obs = seq[:obs_len]              # (obs_len, F)
            fut = seq[obs_len:, :2]          # (pred_len, 2) -> (x, y) only

            X_list.append(obs)
            Y_list.append(fut)
            time_list.append(times[start:end])
            
            # ---- meta information for THIS sequence ----
            seq_idx = len(X_list) - 1       # <--- the sequence index
            seq_idx_list.append(seq_idx)
            pid_list.append(pid)

    X = np.stack(X_list, axis=0)  # (N, obs_len, F)
    Y = np.stack(Y_list, axis=0)  # (N, pred_len, 2)
    TIME = np.stack(time_list, axis=0)  # (N, seq_len)
    
    meta = pd.DataFrame({
        "seq_idx": seq_idx_list,
        "person_id": pid_list,
    })
    meta["person_id"] = meta["person_id"].astype(float).astype("int64")
    meta.to_csv("traj_dataset_metadata.csv", index=False)

    return X, Y, TIME


def compute_normalization_stats(X_train):
    """
    X_train: (N, obs_len, F)
    Compute mean/std over the entire TRAIN SET ONLY.
    """
    # flatten all time steps and samples -> shape (N * obs_len, 2)
    XY = X_train[:, :, :2].reshape(-1, 2)   # first two features are x, y
    mean_xy = XY.mean(axis=0)
    std_xy = XY.std(axis=0) + 1e-8
    
    return mean_xy, std_xy


def normalize_df_min_max(
    df: pd.DataFrame,
    min: np.ndarray,
    max: np.ndarray,
) -> pd.DataFrame:
    """
    Normalize the 'x' and 'y' columns of df to [-1, 1] using min/max.
    """
    df_norm = df.copy()
    df_norm["x"] = 2 * (df["x"] - min[0]) / (max[0] - min[0]) - 1
    df_norm["y"] = 2 * (df["y"] - min[1]) / (max[1] - min[1]) - 1
    return df_norm


def to_curr_pos_relative(
    X: np.ndarray,
    Y: np.ndarray,
    obs_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert absolute positions to current-position-relative coordinates.

    Anchor = last observed position (at time obs_len).

    X: (N, obs_len, F)   with features [x, y, speed, orientation]
    Y: (N, pred_len, 2)  with [x, y]

    Returns:
        X_rel: same shape as X, but x,y are relative to last observed position
        Y_rel: same shape as Y, but x,y are relative to last observed position
    """
    X_rel = X.copy()
    Y_rel = Y.copy()    

    # Last observed position per sample: (N, 2)
    last_pos = X[:, obs_len - 1, :2]          # (x_last, y_last)

    # Make X relative for x,y
    X_rel[:, :, :2] = X[:, :, :2] - last_pos[:, None, :]

    # Make Y relative for x,y
    Y_rel = Y - last_pos[:, None, :]          # broadcast over pred_len

    # speed (X[:,:,2]) and orientation (X[:,:,3]) stay unchanged
    return X_rel, Y_rel


### to absolute coordinates
# Y_pred_rel = Y_pred_rel_norm * std + mean
def to_absolute(
    Y_rel: np.ndarray,
    X_obs_abs: np.ndarray,
    obs_len: int
) -> np.ndarray:
    """
    Convert relative predicted positions Y_rel back to absolute positions.

    Y_rel: (N, pred_len, 2)
    X_obs_abs: (N, obs_len, F)
    obs_len: length of observation window

    Returns:
        Y_abs: (N, pred_len, 2)
    """
    # anchor: last observed absolute pos
    last_pos = X_obs_abs[:, obs_len - 1, :2]          # (N, 2)
    Y_abs = Y_rel + last_pos[:, None, :]              # (N, pred_len, 2)
    return Y_abs


def load_normalization_stats(file_path: str):
    """
    Load normalization stats from file.
    """
    stats = np.load(file_path)

    min = stats["min"]   # array shape (2,)
    max  = stats["max"]    # array shape (2,)

    print("Loaded min:", min)
    print("Loaded max:", max)
    
    return min, max


def has_continuous_motion(
    g: pd.DataFrame,
    idle_threshold: float = 1.0,
    idle_check_interval: int = 5,
) -> bool:
    """
    Return True if this person's trajectory is 'continuous' (no idling),
    False if we detect idling.

    g: DataFrame for a single person_id, sorted by time.
    idle_threshold: distance (in x,y units) below which we consider it 'idle'
                    over idle_check_interval steps.
    idle_check_interval: number of timesteps between the positions we compare.
    """
    # use only x, y
    xy = g[["x", "y"]].to_numpy()   # (T, 2)
    T = xy.shape[0]

    if T <= idle_check_interval:
        # Too short to check properly; you can decide True/False.
        # Since you already filter by min_traj_len, this rarely matters.
        return False

    # distance between positions separated by idle_check_interval
    # xy[t + k] - xy[t]
    diff = xy[idle_check_interval:, :] - xy[:-idle_check_interval, :]   # (T - k, 2)
    dist = np.linalg.norm(diff, axis=1)                                 # (T - k,)

    # If ANY such distance is smaller than idle_threshold -> idling detected
    if np.any(dist < idle_threshold):
        return False    # has idle, reject this person
    return True         # no idle detected, keep this person