import os
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from mod.mod_inference import energy_mod, get_mod_score
from general_utils import denormalize_positions, denorm_xy_torch


# @torch.no_grad()
def  evaluate_cfm(
    model: nn.Module,
    test_loader,
    K: int,
    pred_len: int,
    min: np.ndarray = None,
    max: np.ndarray = None,
    device: str = "cuda",
    mod_model: nn.Module = None,
):
    """
    Evaluate model on test dataset by sampling futures and computing ADE/FDE.

    Assumes test_loader yields:
        X_obs: (B, obs_len, F)  - normalized
        Y_true_norm: (B, pred_len, 2) - normalized

    Returns:
        ade_mean, fde_mean
    """
    model.eval()

    all_min_ade = []
    all_min_fde = []
    all_avg_ade = []
    all_avg_fde = []
    all_min_ade_mod = []
    all_min_fde_mod = []
    all_avg_mod_score = []
    
    all_masd = []
    all_avar = []
    all_fvar = []

    # all_ade = []
    # all_fde = []

    for X_obs, Y_true_norm, epoch_time in test_loader:
        X_obs = X_obs.to(device)          # (B, obs_len, F)
        Y_true_norm = Y_true_norm.to(device)  # (B, pred_len, 2)

        # # Euler sampling
        # Y_pred_norm = sample_future_cfm_euler(
        #     model,
        #     X_obs,
        #     pred_len=pred_len,
        #     n_steps=100,
        #     device=device,
        # )                                 # (B, pred_len, 2)

        # # # 1) Sample predicted future in normalized relative coords
        # Y_pred_norm = sample_future_cfm_ode(
        #     model,
        #     X_obs,
        #     pred_len=pred_len,
        #     n_steps=100,
        #     device=device,
        # )                                 # (B, pred_len, 2)

        # Y_pred_norm = sample_future_cfm_ode_with_guidance(
        #     model,
        #     X_obs,
        #     epoch_time,
        #     pred_len=pred_len,
        #     n_steps=100,
        #     device=device,
        #     min=min,
        #     max=max,
        #     mod_model=mod_model,
        # )                                 # (B, pred_len, 2)

        # Y_pred_norm = sample_future_cfm_euler_with_guidance(
        #     model,
        #     X_obs,
        #     epoch_time,
        #     pred_len=pred_len,
        #     n_steps=100,
        #     device=device,
        #     min=min,
        #     max=max,
        #     mod_model=mod_model,
        # )                                 # (B, pred_len, 2)

        Yk_pred_norm = sample_future_cfm_euler_k(
        # Yk_pred_norm = sample_future_cfm_euler_k_transformer(
            model,
            X_obs,
            pred_len=pred_len,
            K=K,
            n_steps=100,
            device=device,
        )  # (B, K, pred_len, 2)

        # 2) Compute difference for K
        # Y_true_expand = Y_true_norm[:, None]     # (B,1,T,2)
        # diff_norm = Yk_pred_norm - Y_true_expand   # (B,K,T,2)
        # diff_x = diff_norm[..., 0] * (max[0] - min[0]) * 0.5
        # diff_y = diff_norm[..., 1] * (max[1] - min[1]) * 0.5
        # dist = torch.sqrt(diff_x**2 + diff_y**2)   # (B,K,T)
        # ade_k = dist.mean(dim=2)     # (B,K)
        # fde_k = dist[:, :, -1]       # (B,K)
        
        Yk_pred_real = denorm_xy_torch(Yk_pred_norm, min, max)              # (B,K,T,2)
        Y_true_real  = denorm_xy_torch(Y_true_norm,  min, max)[:, None]     # (B,1,T,2)
        diff = Yk_pred_real - Y_true_real                                   # (B,K,T,2)
        dist = torch.linalg.norm(diff, dim=-1)                              # (B,K,T)
        ade_k = dist.mean(dim=2)   # (B,K)
        fde_k = dist[:, :, -1]     # (B,K)
        
        mod_score_k  = get_mod_score(Yk_pred_real, mod_model) # (B,K) # the lower the better
        best_k_min_mod = mod_score_k.argmin(dim=1)   # (B,)
        # ADE/FDE for the trajectory with MIN mod_score
        ade_min_mod = ade_k.gather(1, best_k_min_mod[:, None]).squeeze(1)  # (B,)
        fde_min_mod = fde_k.gather(1, best_k_min_mod[:, None]).squeeze(1)  # (B,)
        
        min_ade, _ = ade_k.min(dim=1)   # (B,)
        min_fde, _ = fde_k.min(dim=1)   # (B,)
        avg_ade = ade_k.mean(dim=1)     # (B,)
        avg_fde = fde_k.mean(dim=1)     # (B,)
        avg_mod_score = mod_score_k.mean(dim=1)  # (B,)
        
        masd = get_masd(Yk_pred_real)  # (B,)
        avar, fvar = get_trajectory_var(Yk_pred_real) # (B,), (B,)
        
        all_min_ade.append(min_ade.cpu())
        all_min_fde.append(min_fde.cpu())
        all_avg_ade.append(avg_ade.cpu())
        all_avg_fde.append(avg_fde.cpu())
        all_min_ade_mod.append(ade_min_mod.cpu())
        all_min_fde_mod.append(fde_min_mod.cpu())
        all_avg_mod_score.append(avg_mod_score.cpu())
        
        all_masd.append(masd.cpu())
        all_avar.append(avar.cpu())
        all_fvar.append(fvar.cpu())

        # 2) Compute difference for 1 output traj
        # diff_norm = Y_pred_norm - Y_true_norm    # (B, pred_len, 2)
        # diff_x = diff_norm[..., 0] * (max[0] - min[0]) * 0.5
        # diff_y = diff_norm[..., 1] * (max[1] - min[1]) * 0.5
        # dist = torch.sqrt(diff_x**2 + diff_y**2)   # (B, pred_len)
 
        # ade = dist.mean(dim=1)      # (B,)
        # fde = dist[:, -1]           # (B,)

        # all_ade.append(ade.cpu())
        # all_fde.append(fde.cpu())

    # all_ade = torch.cat(all_ade).mean().item()
    # all_fde = torch.cat(all_fde).mean().item()

    # unit_str = " (original units)" if min is not None else " (normalized units)"
    # print(f"Test ADE{unit_str}: {all_ade:.4f}")
    # print(f"Test FDE{unit_str}: {all_fde:.4f}")

    minADE = torch.cat(all_min_ade).mean().item()
    minFDE = torch.cat(all_min_fde).mean().item()
    avgADE = torch.cat(all_avg_ade).mean().item()
    avgFDE = torch.cat(all_avg_fde).mean().item()
    modADE = torch.cat(all_min_ade_mod).mean().item()
    modFDE = torch.cat(all_min_fde_mod).mean().item()
    mod_score = torch.cat(all_avg_mod_score).mean().item()
    
    masd_value = torch.cat(all_masd).mean().item()
    avar_value = torch.cat(all_avar).mean().item()
    fvar_value = torch.cat(all_fvar).mean().item()
    
    
    all_ade = torch.cat(all_min_ade).cpu().numpy()
    all_fde = torch.cat(all_min_fde).cpu().numpy()

    print(f"minADE@{K}: {minADE:.4f}")
    print(f"minFDE@{K}: {minFDE:.4f}")
    print(f"avgADE@{K}: {avgADE:.4f}")
    print(f"avgFDE@{K}: {avgFDE:.4f}")
    print(f"ADE_mod@{K}: {modADE:.4f}")
    print(f"FDE_mod@{K}: {modFDE:.4f}")
    print(f"avg_mod_score@{K}: {mod_score:.4f}")
    
    print(f"MASD@{K}: {masd_value:.4f}")
    print(f"Average variance across trajectory (avar)@{K}: {avar_value:.4f}")
    print(f"Final variance (fvar)@{K}: {fvar_value:.4f}")
    
    print("number of persons in test set:", all_ade.shape)

    return all_ade, all_fde

@torch.no_grad()
def sample_future_cfm_euler_k_transformer(
    model: nn.Module,
    X_obs: torch.Tensor,
    pred_len: int,
    K: int = 8,
    n_steps: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate K future trajectories from the trained CFM model (parallel).

    Returns:
        Y_pred_norm: (B, K, pred_len, 2)
    """
    model.eval()
    X_obs = X_obs.to(device)
    B = X_obs.size(0)
    D = pred_len * 2

    # sample x0 for all hypotheses
    eps = torch.randn(B, K, pred_len, 2, device=device)
    eps = torch.cumsum(eps, dim=2)
    eps = eps / eps.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    x_t = eps.view(B, K, D)  # (B, K, D)

    ts = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)
    dt = ts[1] - ts[0]

    for k in range(n_steps):
        t_k = ts[k].item()
        t_batch = torch.full((B,), t_k, device=device, dtype=x_t.dtype)  # (B,) not (B*K,)

        x1_pred = model(x_t, t_batch, X_obs)  # (B, K, D)
        denom = max(1.0 - t_k, 1e-3)
        v = (x1_pred - x_t) / denom
        x_t = x_t + dt * v

    Y_pred_norm = x_t.view(B, K, pred_len, 2)
    return Y_pred_norm

def get_masd(Yk_pred_real):
    """
    Compute MASD (map-aware self-distance) for a set of predicted trajectories.

    Yk_pred_real: (B, K, T, 2)

    Returns:
        masd: (B,) average pairwise distance between the K trajectories
    """
    predictions = Yk_pred_real.permute(0, 2, 1, 3)  # (B, T, K, 2)
    pairwise_distances = torch.cdist(predictions, predictions, p=2)  # (B, T, K, K)
    max_dist = pairwise_distances.max(dim=-1)[0].max(dim=-1)[0]  # (B, T)
    masd = max_dist.mean(dim=-1)  # (B,)
    return masd

def get_trajectory_var(Yk_pred_real):
    """
    Compute variance of predicted trajectories across K samples.

    Yk_pred_real: (B, K, T, 2)

    Returns:
        avar: (B,) average variance across all time steps and dimensions
        fvar: (B,) variance at the final time step
    """
    a_var = Yk_pred_real.var(dim=(1, 3)).mean(dim=1)  # (B,)
    f_var = Yk_pred_real[:, :, -1, :].var(dim=(1, 2))  # (B,)
    return a_var, f_var



@torch.no_grad()
def sample_future_cfm_euler(
    model: nn.Module,
    X_obs: torch.Tensor,
    pred_len: int,
    n_steps: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate future trajectories from the trained CFM model.

    Args:
        model: TrajectoryCFMModel
        X_obs: (B, obs_len, F)  - normalized, current-pos-relative past
        pred_len: e.g. 60
        n_steps: ODE integration steps from t=0 to t=1
        device: 'cuda' or 'cpu'

    Returns:
        Y_pred_norm: (B, pred_len, 2)
        in the SAME space as Y in the dataset:
        - current-pos-relative
        - normalized (x,y)
    """
    model.eval()
    X_obs = X_obs.to(device)
    B = X_obs.size(0)
    D = pred_len * 2

    # x_t = (torch.rand(B, D, device=device) * 2.0) - 1.0   # U[-1,1]

    eps = torch.randn(B, pred_len, 2, device=device)
    eps = torch.cumsum(eps, dim=1)                # makes it a random walk (smooth-ish)
    eps = eps / eps.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
    x_t = eps.view(B, -1)
    # check the dim of x_t
    # print("x_t shape:", x_t.shape)

    t0, t1 = 0.0, 1.0
    ts = torch.linspace(t0, t1, steps=n_steps + 1, device=device)
    dt = ts[1] - ts[0]

    for k in range(n_steps):
        t_k = ts[k]              # scalar
        t_batch = t_k.expand(B)  # (B,)
        x1_pred = model(x_t, t_batch, X_obs)   # (B, D)
        denom = (1.0 - t_batch).clamp(min=1e-3).unsqueeze(-1)
        v = (x1_pred - x_t) / denom           # (B, D)  <- velocity field
        x_t = x_t + dt * v

    Y_pred_norm = x_t.view(B, pred_len, 2)
    return Y_pred_norm


@torch.no_grad()
def sample_future_cfm_euler_k(
    model: nn.Module,
    X_obs: torch.Tensor,
    pred_len: int,
    K: int = 8,
    n_steps: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate K future trajectories from the trained CFM model (parallel).

    Returns:
        Y_pred_norm: (B, K, pred_len, 2)
    """
    model.eval()
    X_obs = X_obs.to(device)
    B = X_obs.size(0)
    D = pred_len * 2

    # sample x0 for all hypotheses
    eps = torch.randn(B, K, pred_len, 2, device=device)
    eps = torch.cumsum(eps, dim=2)
    eps = eps / eps.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    x_t = eps.view(B * K, D)  # (B*K, D)

    # repeat conditioning
    X_obs_rep = X_obs[:, None].expand(B, K, *X_obs.shape[1:]).reshape(B * K, *X_obs.shape[1:])

    ts = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)
    dt = ts[1] - ts[0]

    for k in range(n_steps):
        t_k = ts[k].item()
        t_batch = torch.full((B * K,), t_k, device=device, dtype=x_t.dtype)

        x1_pred = model(x_t, t_batch, X_obs_rep)  # (B*K, D)
        denom = max(1.0 - t_k, 1e-3)
        v = (x1_pred - x_t) / denom
        x_t = x_t + dt * v

    Y_pred_norm = x_t.view(B, K, pred_len, 2)
    return Y_pred_norm


def sample_future_cfm_euler_with_guidance(
    model: nn.Module,
    X_obs: torch.Tensor,
    epoch_time: torch.Tensor,
    pred_len: int,
    n_steps: int = 32,
    device: str = "cuda",
    min: np.ndarray = None,
    max: np.ndarray = None,
    mod_model: nn.Module = None,
    dt_real: float = 1.0,
    guide_until: float = 1.0,   # only apply guidance for t <= guide_until (speed + stability)
    grad_clip: float = 0.0,     # set e.g. 5.0 to clip, or 0 to disable
) -> torch.Tensor:
    """
    Generate future trajectories from the trained CFM model.

    Args:
        model: TrajectoryCFMModel
        X_obs: (B, obs_len, F)  - normalized, current-pos-relative past
        pred_len: e.g. 60
        n_steps: ODE integration steps from t=0 to t=1
        device: 'cuda' or 'cpu'

    Returns:
        Y_pred_norm: (B, pred_len, 2)
        in the SAME space as Y in the dataset:
        - current-pos-relative
        - normalized (x,y)
    """
    model.eval()
    X_obs = X_obs.to(device)
    B = X_obs.size(0)
    D = pred_len * 2
    obs_len = X_obs.size(1)

    # x_t = (torch.rand(B, D, device=device) * 2.0) - 1.0   # U[-1,1]

    eps = torch.randn(B, pred_len, 2, device=device)
    eps = torch.cumsum(eps, dim=1)                # makes it a random walk (smooth-ish)
    eps = eps / eps.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
    x_t = eps.view(B, -1)
    # check the dim of x_t
    # print("x_t shape:", x_t.shape)

    t0, t1 = 0.0, 1.0
    ts = torch.linspace(t0, t1, steps=n_steps + 1, device=device)
    dt = ts[1] - ts[0]

    epoch_time = epoch_time.to(device)
    query_time = epoch_time[:, obs_len]  # (B,)

    for k in range(n_steps):
        t_k = ts[k]              # scalar
        t_batch = t_k.expand(B)  # (B,)
        with torch.no_grad():
            x1_pred = model(x_t, t_batch, X_obs)   # (B, D)
        denom = (1.0 - t_batch).clamp(min=1e-3).unsqueeze(-1)
        v = (x1_pred - x_t) / denom           # (B, D)  <- velocity field
        
        # ---- MoD guidance grad_E ----
        if (mod_model is None) or (t_k.item() > guide_until):
            v_guided = v
        else:
            # enable grad ONLY for energy term
            with torch.enable_grad():
                x_energy = x_t.detach().requires_grad_(True)  # (B, D)
                E = energy_mod(
                    x_energy,
                    X_obs,
                    query_time,
                    mod_model,
                    dt=dt_real,
                    min=min,
                    max=max,
                )
                grad_E = torch.autograd.grad(E, x_energy, create_graph=False)[0]  # (B, D)

            # optional: clip / sanitize
            grad_E = torch.nan_to_num(grad_E, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip and grad_clip > 0:
                grad_E = grad_E.clamp(-grad_clip, grad_clip)

            lam = guidance_schedule(t_batch)  # (B,)
            v_guided = v - lam[:, None] * grad_E
        
        x_t = x_t + dt * v_guided

    Y_pred_norm = x_t.view(B, pred_len, 2)
    return Y_pred_norm


from torchdiffeq import odeint

@torch.no_grad()
def sample_future_cfm_ode(
    model: nn.Module,
    X_obs: torch.Tensor,
    pred_len: int,
    n_steps: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate future trajectories from the trained CFM model.

    Args:
        model: TrajectoryCFMModel
        X_obs: (B, obs_len, F)  - normalized, current-pos-relative past
        pred_len: e.g. 60
        n_steps: ODE integration steps from t=0 to t=1
        device: 'cuda' or 'cpu'

    Returns:
        Y_pred_norm: (B, pred_len, 2)
        in the SAME space as Y in the dataset:
        - current-pos-relative
        - normalized (x,y)
    """
    model.eval()
    X_obs = X_obs.to(device)
    B = X_obs.size(0)
    D = pred_len * 2

    # x0 = (torch.rand(B, D, device=device) * 2.0) - 1.0   # U[-1,1]
    eps = torch.randn(B, pred_len, 2, device=device)
    eps = torch.cumsum(eps, dim=1)                # makes it a random walk (smooth-ish)
    eps = eps / eps.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
    x0 = eps.view(B, -1)
    t_span = torch.linspace(1e-3, 1.0 - 1e-3, n_steps + 1, device=device)

    def ode_rhs(t_scalar, x_state):
        """
        t_scalar: 0-dim tensor (scalar)
        x_state:  (B*D,) flattened
        returns:  dx/dt flattened (B*D,)
        """
        x_flat = x_state.view(B, D)         # (B, D)
        t_batch = t_scalar.expand(B)        # (B,)
           
        # v = model(x_flat, t_batch, X_obs)  # (B, D)
        
        x1_pred = model(x_flat, t_batch, X_obs)
        denom = (1.0 - t_batch).clamp(min=1e-3).unsqueeze(-1)
        v = (x1_pred - x_flat) / denom
        
        return v.view(-1)                   # flatten back

    traj = odeint(
        ode_rhs,
        x0.view(-1),         # flattened initial state
        t_span,
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
    )
    x1 = traj[-1].view(B, D)                # final x at t=1
    Y_pred_norm = x1.view(B, pred_len, 2)
    return Y_pred_norm


# @torch.no_grad()
def sample_future_cfm_ode_with_guidance(
    model: nn.Module,
    X_obs: torch.Tensor,
    epoch_time: torch.Tensor,
    pred_len: int,
    n_steps: int = 32,
    device: str = "cuda",
    min: np.ndarray = None,
    max: np.ndarray = None,
    mod_model: nn.Module = None,
    dt_real: float = 1.0,
) -> torch.Tensor:
    """
    Generate future trajectories from the trained CFM model.

    Args:
        model: TrajectoryCFMModel
        X_obs: (B, obs_len, F)  - normalized, current-pos-relative past
        epoch_time: (B, seq_len) - original epoch times for each sample
        pred_len: e.g. 60
        n_steps: ODE integration steps from t=0 to t=1
        device: 'cuda' or 'cpu'

    Returns:
        Y_pred_norm: (B, pred_len, 2)
        in the SAME space as Y in the dataset:
        - current-pos-relative
        - normalized (x,y)
    """
    model.eval()
    X_obs = X_obs.to(device)
    B = X_obs.size(0)
    D = pred_len * 2
    obs_len = X_obs.size(1)

    # x0 = (torch.rand(B, D, device=device) * 2.0) - 1.0   # U[-1,1]
    eps = torch.randn(B, pred_len, 2, device=device)
    eps = torch.cumsum(eps, dim=1)                # makes it a random walk (smooth-ish)
    eps = eps / eps.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-6)   # scale to [-1,1]
    x0 = eps.view(B, -1)
    t_span = torch.linspace(1e-3, 1.0 - 1e-3, n_steps + 1, device=device)
    
    epoch_time = epoch_time.to(device)
    query_time = epoch_time[:, obs_len] # (B,)
    # query_time = query_time.float()
    
    def ode_rhs(t_scalar, x_state):
        """
        t_scalar: 0-dim tensor (scalar)
        x_state:  (B*D,) flattened
        returns:  dx/dt flattened (B*D,)
        """
        x_flat = x_state.view(B, D)         # (B, D)
        t_batch = t_scalar.expand(B)        # (B,)
        
        # v = model(x_flat, t_batch, X_obs)  # (B, D)
        with torch.no_grad():
            x1_pred = model(x_flat, t_batch, X_obs)
        denom = (1.0 - t_batch).clamp(min=1e-3).unsqueeze(-1)
        v = (x1_pred - x_flat) / denom
        
        x_energy = x_flat.detach().requires_grad_(True)
        E = energy_mod(
            x_energy, 
            X_obs, 
            query_time, 
            mod_model,
            dt=dt_real,
            min=min, 
            max=max)
        grad_E = torch.autograd.grad(E, x_energy, create_graph=False)[0]

        lam = guidance_schedule(t_batch)
        v_guided = v - lam[:, None] * grad_E

        return v_guided.view(-1)                   # flatten back

    traj = odeint(
        ode_rhs,
        x0.view(-1),         # flattened initial state
        t_span,
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
    )
    x1 = traj[-1].view(B, D)                # final x at t=1
    Y_pred_norm = x1.view(B, pred_len, 2)
    return Y_pred_norm


def guidance_schedule(t, base=10.0, power=2):
    # stronger early, weaker near 1
    return base * (1.0 - t).clamp(min=1e-3) ** power

# def guidance_schedule(t, base=1.0, power=0.0):
#     return base * torch.ones_like(t)


def plot_person_from_df(
    df: pd.DataFrame,
    person_id: int,
    model: nn.Module,
    min: np.ndarray,
    max: np.ndarray,
    obs_len: int = 4,
    pred_len: int = 60,
    device: str = "cuda",
    mod_model: nn.Module = None,
    random_seed: int = 1,
    version: str = "tmp",
):
    """
    df: raw ATC DataFrame with columns ['epoch_time','person_id','x','y','speed','orientation']
    person_id: the person you want to inspect
    model: trained TrajectoryCFMModel
    min, max: normalization stats for x,y (shape (2,))
    which: which sliding window of this person's trajectory to use (0 = first window)
    """
    K = 5   # number of outputs
    
    
    feature_cols = ["x", "y", "speed", "orientation"]
    seq_len = obs_len + pred_len
    
    #if frame column is in df, we rename it to epoch_time for consistency
    if "frame" in df.columns:
        df = df.rename(columns={"frame": "epoch_time"})
        
    # 1) Extract this person's full trajectory from df (raw, unnormalized)
    g = df[df["person_id"] == person_id].sort_values("epoch_time").reset_index(drop=True)
    if len(g) < seq_len:
        print(f"Person {person_id} too short: len={len(g)}, need >= {seq_len}")
        return
    # only keep first seq_len points for simplicity
    g = g.iloc[:seq_len]
    # print(g)
    g["x"] = 2 * (g["x"] - min[0]) / (max[0] - min[0]) - 1
    g["y"] = 2 * (g["y"] - min[1]) / (max[1] - min[1]) - 1
    
    # get epoch_time for whole sequence (shape (seq_len,))
    epoch_time = torch.from_numpy(g["epoch_time"].to_numpy(dtype=np.float32)).to(device)
    epoch_time = epoch_time.unsqueeze(0)
    
    arr = g[feature_cols].to_numpy(dtype=np.float32)  # (T, 4)
    
    seq = arr[:seq_len]        # (seq_len, 4)
    obs_abs = seq[:obs_len]     # (obs_len, 4)
    fut_abs = seq[obs_len:, :2] # (pred_len, 2)  (x,y)

    X_batch = torch.from_numpy(obs_abs[None, ...]).float().to(device)   # (1, obs_len, 4)
    
    # Y_pred_norm = sample_future_cfm_ode(
    #     model, X_batch, pred_len=pred_len, n_steps=100, device=device
    # )   # (1, pred_len, 2)
    
    # Y_pred_norm = sample_future_cfm_euler(
    #     model,
    #     X_batch,
    #     pred_len=pred_len,
    #     n_steps=100,
    #     device=device,
    # )                                 # (1, pred_len, 2)

    # Y_pred_norm = sample_future_cfm_euler_with_guidance(
    #     model,
    #     X_batch,
    #     epoch_time,
    #     pred_len=pred_len,
    #     n_steps=100,
    #     device=device,
    #     min=min,
    #     max=max,
    #     mod_model=mod_model,
    # )                                 # (1, pred_len, 2)

    Yk_pred_norm = sample_future_cfm_euler_k(
        model,
        X_batch,
        pred_len=pred_len,
        K=K,
        n_steps=100,
        device=device,
    )  # (1, K, pred_len, 2)
    Yk_pred_norm = Yk_pred_norm[0].detach().cpu().numpy()  # (K, pred_len, 2)

    # 5) Denormalize prediction and convert to absolute coordinates
    observed_real = denormalize_positions(obs_abs[:, 0:2], min, max)  # (obs_len, 2)
    future_real = denormalize_positions(fut_abs, min, max)            # (pred_len, 2)
    last_obs_abs_xy = observed_real[-1, 0:2]              # (2,)
    
    ##### for multiple output #####
    Yk_pred_real = []
    ades = []
    fdes = []

    for k in range(K):
        y_real = denormalize_positions(Yk_pred_norm[k], min, max)   # (pred_len,2)
        Yk_pred_real.append(y_real)

        dist = np.linalg.norm(y_real - future_real, axis=-1)
        ades.append(dist.mean())
        fdes.append(dist[-1])

    best_k = int(np.argmin(ades))
    print(f"[Person {person_id}] minADE@{K}={ades[best_k]:.4f}, minFDE@{K}={fdes[best_k]:.4f} (best_k={best_k})")
    ##############################
    
    
    # ##### for multiple output, get the mod score ######
    # mod_score_k  = get_mod_score(Yk_pred_real, mod_model)
    # best_k_mod = int(np.argmin(mod_score_k))
    # print(f"[Person {person_id}] best_k_mod={best_k_mod}, mod_score={mod_score_k[best_k_mod]:.4f}, ADE={ades[best_k_mod]:.4f}, FDE={fdes[best_k_mod]:.4f}")
    # ###################################################
    
    
    
    ##### for single output #####
    # Y_pred_real = denormalize_positions(Y_pred_norm[0].cpu().numpy(), min, max)    # (pred_len, 2)
    # diff = Y_pred_real - future_real                            # (pred_len, 2)
    # dist = np.linalg.norm(diff, axis=-1)                      # (pred_len,)

    # ade = dist.mean()
    # fde = dist[-1]
    # print(f"[Person {person_id}] ADE={ade:.4f}, FDE={fde:.4f}")
    ##############################

    plt.clf()
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.set_facecolor('white')
    img = plt.imread("map/atc/localization_grid_white.jpg")
    ax.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])
    ax.plot(observed_real[:, 0], observed_real[:, 1], "o-g", markersize=4, label="Observed (past)")
    last_obs = observed_real[-1:, :2]  # shape (1, 2)
    future_connected = np.concatenate([last_obs, future_real[:, :2]], axis=0)
    ax.plot(future_connected[:, 0], future_connected[:, 1], "o-r", markersize=4, label="GT future")

    
    # plt.plot(Y_pred_real[:, 0],   Y_pred_real[:, 1],   "o-b", label="Pred future")
    
    for k in range(K):
        y = Yk_pred_real[k]
        y_connected = np.concatenate([last_obs, y[:, :2]], axis=0)
        ax.plot(y_connected[:, 0], y_connected[:, 1], "o-b", alpha=0.35, markersize=3,
                label="Pred future" if k == 0 else None)
        
    
    # # Plotting the normalized version
    # X_obs_norm = obs_abs
    # Y_pred_norm = Y_pred_norm[0].cpu().numpy()
    
    # plt.plot(obs_abs[:, 0], obs_abs[:, 1], "o-g", label="Observed (past)")
    # plt.plot(fut_abs[:, 0],   fut_abs[:, 1],   "o-r", label="GT future")
    # plt.plot(Y_pred_norm[:, 0],   Y_pred_norm[:, 1],   "o-b", label="Pred future")

    # plt.scatter(
    #     last_obs_abs_xy[0],
    #     last_obs_abs_xy[1],
    #     c="k",
    #     marker="x",
    #     s=80,
    #     label="Anchor (last obs)",
    # )

    all_pts = np.concatenate([observed_real[:, :2], future_real[:, :2]]
                             + [Yk_pred_real[k][:, :2] for k in range(K)], axis=0)
    pad = 5  # meters of padding around trajectories
    # ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    # ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    
    # # for pred 40
    # ax.set_xlim(-5, 50)
    # ax.set_ylim(-27, 10)
    
    # for pred 20
    ax.set_xlim(-33, 7)
    ax.set_ylim(-2.5, 17.5)

    # ax.set_title(f"Person {person_id}")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)
    os.makedirs(f"traj_pred_quacompare/{version}", exist_ok=True)
    # fig.savefig(f"traj_pred_quacompare/{version}/person_{person_id}_seed_{random_seed}.png", bbox_inches='tight')
    fig.savefig(f"traj_pred_quacompare/{version}/person_{person_id}_seed_{random_seed}.pdf", bbox_inches='tight')
    plt.close(fig)

    # plt.title(f"Person {person_id}")
    # plt.legend()
    # plt.axis("equal")
    # plt.grid(True)
    # # plt.show()
    # os.makedirs(f"traj_pred_final/{version}", exist_ok=True)
    # plt.savefig(f"traj_pred_final/{version}/person_{person_id}_seed_{random_seed}.png")
    
    
    
from torch.utils.data import DataLoader, Subset

def make_person_loader(
    base_loader: DataLoader,
    person_ids,
    meta_csv: str = "traj_dataset_metadata.csv",
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int | None = None,
):
    """
    Build a DataLoader that only contains samples belonging to given person_id(s).

    Args:
        base_loader: your existing DataLoader (train/val/test). We reuse its dataset.
        person_ids: int or list[int] of person_id to keep
        meta_csv: path to traj_dataset_metadata.csv generated by build_sequences_from_df
        batch_size: if None, reuse base_loader.batch_size
        shuffle: usually False for evaluation
        num_workers: if None, reuse base_loader.num_workers

    Returns:
        person_loader: DataLoader over Subset(dataset, selected_indices)
    """
    if isinstance(person_ids, (int,)):
        person_ids = [person_ids]
    person_ids = set(person_ids)

    meta = pd.read_csv(meta_csv)
    meta["person_id"] = meta["person_id"].astype(float).astype("int64")

    # indices in dataset corresponding to chosen people
    sel = meta[meta["person_id"].isin(person_ids)]["seq_idx"].to_numpy()
    sel = sel.astype(int)

    if len(sel) == 0:
        raise ValueError(f"No sequences found for person_ids={person_ids}. Check meta_csv and IDs.")

    subset_ds = Subset(base_loader.dataset, sel)

    bs = batch_size if batch_size is not None else base_loader.batch_size
    nw = num_workers if num_workers is not None else base_loader.num_workers

    person_loader = DataLoader(
        subset_ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        drop_last=False,
        pin_memory=getattr(base_loader, "pin_memory", False),
    )
    return person_loader