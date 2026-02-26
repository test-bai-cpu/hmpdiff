import os, time
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from mod.mod_utils import normalize_coords_siren, normalize_coords_space
from mod.models import MoDGMMSirenHybridModel, MoDGMMFeatureModel
from mod.loss_funcs import wrapped_GMM_nll
from general_utils import denormalize_positions_torch
from tqdm import tqdm

import sys


# model_file = "mod/best.pt"
# mod_model = MoDGMMSirenHybridModel(input_size=3, num_components=3)

# state = torch.load(model_file, map_location="cpu", weights_only=True)
# mod_model.load_state_dict(state)
# mod_model = mod_model.to(device)
# mod_model.eval()
# for p in mod_model.parameters():
#     p.requires_grad_(False)


def load_mod_model(model_file: str, device: torch.device):
    mod_model = MoDGMMSirenHybridModel(input_size=3, num_components=3)
    state = torch.load(model_file, map_location="cpu", weights_only=True)
    mod_model.load_state_dict(state)

    mod_model = mod_model.to(device)
    mod_model.eval()
    for p in mod_model.parameters():
        p.requires_grad_(False)

    return mod_model


def load_mod_feature_model(model_file: str, device: torch.device):
    mod_model = MoDGMMFeatureModel(input_size=2, num_components=3, grid_size=(32,32))
    state = torch.load(model_file, map_location="cpu", weights_only=True)
    mod_model.load_state_dict(state)

    mod_model = mod_model.to(device)
    mod_model.eval()
    for p in mod_model.parameters():
        p.requires_grad_(False)

    return mod_model


def energy_mod(x_energy, X_obs, query_time, mod_model, dt=1.0, min=None, max=None):
    """
    x_flat: (B, D) flattened future trajectory
    X_obs:  (B, obs_len, F) observed trajectory
    """
    
    ## check the dim for x_flat and X_obs ##
    # print(x_energy.shape)
    # print(X_obs.shape)
    
    B = x_energy.size(0)
    D = x_energy.size(1)
    pred_len = D // 2
    obs_len = X_obs.size(1)
    # print("pred_len:", pred_len)
    
    # Denormalize trajectory and convert to absolute coordinates
    x_pred = x_energy.view(B, pred_len, 2)    # (B, pred_len, 2)
    x_pred_real = denormalize_positions_torch(x_pred, min, max)    # (B, pred_len, 2)
    observed_real = denormalize_positions_torch(X_obs[:, :, 0:2], min, max)    # (pred_len, 2)
    last_obs_abs_xy = observed_real[:, -1, 0:2]              # (2,)

    # print("check all shapes here:")
    # print("x_pred_real:", x_pred_real.shape)
    # print("observed_real:", observed_real.shape)
    # print("last_obs_abs_xy:", last_obs_abs_xy.shape)
    # print("query_time:", query_time.shape, query_time.dim())
    
    pred_vel = (x_pred_real[:, 1:] - x_pred_real[:, :-1]) / dt                 # (B, pred_len-1, 2)
    pred_vel_full = torch.cat([pred_vel[:, 0:1], pred_vel], dim=1)   # (B, pred_len, 2)
    pred_speed = torch.sqrt((pred_vel_full ** 2).sum(dim=-1))  # (B, pred_len)
    pred_angle = torch.atan2(pred_vel_full[..., 1], pred_vel_full[..., 0])  # (B, pred_len)
    
    # angle to be in [0, 2pi]
    pred_angle = pred_angle % (2 * np.pi)
    
    pred_speed_flat = pred_speed.reshape(-1)    # (N,)
    pred_angle_flat = pred_angle.reshape(-1)    # (N,)
    
    N = B * pred_len
    pred_pos_flat = x_pred_real.reshape(N, 2)
    time_flat = query_time.repeat_interleave(pred_len)
    time_flat = time_flat.to(pred_pos_flat.device, dtype=pred_pos_flat.dtype)
    pred_inputs = torch.cat(
        [pred_pos_flat, time_flat.unsqueeze(-1)], dim=-1
    )

    ### TODO: can use the obs_inputs to decide how much we scale the computed NLL guidance value ###
    norm_pred_inputs = normalize_coords_space(pred_pos_flat)
    GMM_params, _ = mod_model(norm_pred_inputs)                   # (N,K,6

    # norm_pred_inputs = normalize_coords_siren(pred_inputs)
    # GMM_params, _ = mod_model(norm_pred_inputs)                   # (N,K,6)
    # print("GMM_params:", GMM_params.shape)
    # print("GMM content example:", GMM_params[0, :, :])
    
    yb = torch.stack([pred_speed_flat, pred_angle_flat], dim=-1)  # (N, 2)
    nll_flat = wrapped_GMM_nll(GMM_params, yb, reduction="none")
    E = nll_flat.view(B, pred_len).mean(dim=1).sum()
    # nll = gmm_nll_speed_angle(speed_flat, angle_flat, GMM_params, assume_logits=assume_logits)  # (N,)
    # E = nll.view(B, pred_len).mean(dim=1).sum()
    
    # sys.exit(0)
    
    return E
    
    
    
# with torch.no_grad():  # Disable gradient computation for inference
#     # means, vars, corr_coef = model(example_locations)
#     GMM_params, _ = mod_model(norm_inputs)
#     # GMM_params, densities, _ = model(example_locations)
    
# data = []

# num_components = GMM_params.shape[1]

# for j in range(example_locations.size(0)):
#     # density = densities[j]
#     for i in range(num_components):
#         weight = GMM_params[j, i, 0]       # Shape: (batch_size, num_components)
#         speed_mean = GMM_params[j, i, 1]   # Shape: (batch_size, num_components)
#         angle_mean = GMM_params[j, i, 2]   # Shape: (batch_size, num_components)
#         speed_var = GMM_params[j, i, 3]    # Shape: (batch_size, num_components)
#         angle_var = GMM_params[j, i, 4]    # Shape: (batch_size, num_components)
#         corr_coef = GMM_params[j, i, 5]    # Shape: (batch_size, num_components)
        
#         # row = [float(example_locations[j, 0]), float(example_locations[j, 1]), float(speed_mean), float(angle_mean), float(speed_var), float(angle_var), float(corr_coef), float(weight), float(density)]
#         row = [float(example_locations[j, 0]), float(example_locations[j, 1]), float(speed_mean), float(angle_mean), float(speed_var), float(angle_var), float(corr_coef), float(weight)]
#         data.append(row)
        
# df = pd.DataFrame(data, columns=["x", "y", "mean_speed", "mean_motion_angle", "var_speed", "var_motion_angle", "coef", "weight"])


def mod_loss_per_sample(
    Y_pred_norm: torch.Tensor,   # (B, K, T, 2)  normalized in your traj space
    X_obs_norm: torch.Tensor,    # (B, obs_len, F) normalized; pos in [:,:,0:2]
    mod_model,
    min_np,
    max_np,
    dt: float = 1.0,
):
    """
    Returns:
        L_mod_bk: (B, K)  average NLL over timesteps (lower = more MoD-consistent)
    """
    device = Y_pred_norm.device
    dtype  = Y_pred_norm.dtype
    B, K, T, _ = Y_pred_norm.shape

    # ---- denormalize predicted future and observed past to real coordinates
    # Assume your denormalize_positions_torch supports torch tensors.
    # If min/max are numpy arrays, that's fine (your function likely handles it).
    Y_pred_real = denormalize_positions_torch(
        Y_pred_norm.view(B*K, T, 2), min_np, max_np
    ).view(B, K, T, 2)  # (B,K,T,2)

    # X_obs_real = denormalize_positions_torch(
    #     X_obs_norm[:, :, 0:2], min_np, max_np
    # )  # (B, obs_len, 2)

    # (Optional) if your future is relative, you might need to shift by last obs.
    # In your original code you computed last_obs_abs_xy but didn't use it,
    # so I'll keep behavior identical: do nothing.

    # ---- compute speed + angle from velocities
    vel = (Y_pred_real[:, :, 1:] - Y_pred_real[:, :, :-1]) / dt          # (B,K,T-1,2)
    vel_full = torch.cat([vel[:, :, 0:1], vel], dim=2)                   # (B,K,T,2)

    speed = torch.sqrt((vel_full ** 2).sum(dim=-1) + 1e-12)              # (B,K,T)
    angle = torch.atan2(vel_full[..., 1], vel_full[..., 0])              # (B,K,T)

    # wrap angle to [0, 2pi)
    angle = angle.remainder(2.0 * torch.pi)

    # ---- flatten for MoD model evaluation
    N = B * K * T
    pos_flat = Y_pred_real.reshape(N, 2)                                 # (N,2)
    speed_flat = speed.reshape(N)                                        # (N,)
    angle_flat = angle.reshape(N)                                        # (N,)

    # ---- mod_model input (your current mod_model uses only normalized xy)
    norm_pos = normalize_coords_space(pos_flat)                          # (N,2)
    GMM_params, _ = mod_model(norm_pos)                                  # (N, M, 6) or similar

    yb = torch.stack([speed_flat, angle_flat], dim=-1)                   # (N,2)

    # wrapped_GMM_nll should return (N,) when reduction="none"
    nll_flat = wrapped_GMM_nll(GMM_params, yb, reduction="none")         # (N,)

    nll = nll_flat.view(B, K, T)                                         # (B,K,T)
    L_mod_bk = nll.sum(dim=2)                                            # (B,K) sum over time, i.e., total NLL for each trajectory.
    return L_mod_bk


def get_mod_score(
    Yk_pred_real: torch.Tensor,  # (B,K,T,2) in REAL coords
    mod_model: nn.Module,
    dt_real: float = 1.0,
) -> torch.Tensor:
    """
    Compute MoD alignment score for each predicted trajectory.

    Returns:
        mod_score: (B,K) lower is better
    """
    B, K, T, _ = Yk_pred_real.shape
    device = Yk_pred_real.device
    dtype = Yk_pred_real.dtype
    BK = B * K
    pos = Yk_pred_real.reshape(BK, T, 2)  # (BK, T, 2)
    
    # Velocity -> speed/angle
    vel = (pos[:, 1:] - pos[:, :-1]) / dt_real              # (BK, T-1, 2)
    vel_full = torch.cat([vel[:, 0:1], vel], dim=1)               # (BK, T, 2)

    speed = torch.sqrt((vel_full ** 2).sum(dim=-1))  # (BK, T)
    angle = torch.atan2(vel_full[..., 1], vel_full[..., 0])           # (BK, T)
    angle = torch.remainder(angle, 2.0 * math.pi)                     # [0, 2pi)

    N = BK * T
    pos_flat = pos.reshape(N, 2)
    speed_flat = speed.reshape(N)
    angle_flat = angle.reshape(N)
    yb = torch.stack([speed_flat, angle_flat], dim=-1)                # (N, 2)
    
    with torch.no_grad():
        norm_pred_inputs = normalize_coords_space(pos_flat)                # (N,2)
        GMM_params, _ = mod_model(norm_pred_inputs)  # (N, num_modes=3, 6)
        nll_flat = wrapped_GMM_nll(GMM_params, yb, reduction="none")  # (N,)

        # Per-trajectory NLL: mean over time
        score_k = nll_flat.view(B, K, T).mean(dim=-1)                   # (B, K)
    
    return score_k