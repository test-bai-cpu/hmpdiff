import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torchdiffeq
import sys


from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.basic_transformer.transformer import VisionTransformerCFMWrapper


class TrajectoryCFMMLP(nn.Module):
    def __init__(self, history_dim: int, future_dim: int, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = future_dim + history_dim + 1  # +1 for time t
        last_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, future_dim))
        self.net = nn.Sequential(*layers)
        self.future_dim = future_dim
        self.history_dim = history_dim

    def forward(self, t, x: torch.Tensor, cond_hist: torch.Tensor) -> torch.Tensor:
        # t: scalar or [B], x: [B, D], cond_hist: [B, H]
        b = x.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if t.dim() == 0:
            t = t.repeat(b)
        t_feat = t.view(b, 1)
        x_flat = x.view(b, -1)
        h_flat = cond_hist.view(b, -1)
        inp = torch.cat([x_flat, h_flat, t_feat], dim=1)
        out = self.net(inp)
        return out.view(b, -1)
    
class VectorFieldNet(nn.Module):
    def __init__(self, dim: int, cond_dim: int, time_dim: int = 16, hidden_dim: int = 256):
        """
        dim:       D = 2 * pred_len  (flattened future)
        cond_dim:  output dim of PastEncoder
        time_dim:  embedding size for t
        hidden_dim: MLP hidden size
        """
        super().__init__()
        # small time embedding for scalar t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        in_dim = dim + cond_dim + time_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, h_cond: torch.Tensor) -> torch.Tensor:
        """
        x_t:   (B, D)
        t:     (B,)
        h_cond:(B, cond_dim)
        returns v_pred: (B, D)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)     # (B, 1)

        t_emb = self.time_mlp(t)    # (B, time_dim)

        inp = torch.cat([x_t, t_emb, h_cond], dim=-1)  # (B, D + time_dim + cond_dim)
        v_pred = self.mlp(inp)      # (B, D)
        return v_pred
    
    
class PastEncoder(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 64, out_dim: int = 128):
        """
        Encodes past trajectory X_obs: (B, obs_len, in_dim) -> (B, out_dim)
        """
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        """
        x_obs: (B, obs_len, in_dim)
        returns: (B, out_dim)
        """
        _, h_last = self.gru(x_obs)      # h_last: (1, B, hidden_dim)
        h_last = h_last.squeeze(0)       # (B, hidden_dim)
        h_cond = self.proj(h_last)       # (B, out_dim)
        return h_cond
    
    
class TrajectoryCFMModel(nn.Module):
    """
    Full model for conditional flow matching for trajectory prediction.
    
    Inputs:
        x_t:    (B, D)   - point on ODE path at time t
        t:      (B,)     - time
        X_obs:  (B, obs_len, 4)  - observed past
    
    Output:
        v_pred: (B, D)   - vector field prediction
    """
    def __init__(
        self,
        obs_len=4,
        pred_len=60,
        past_hidden_dim=64,
        past_out_dim=128,
        vf_hidden_dim=256,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.D = pred_len * 2  # flattened future dimension

        self.encoder = PastEncoder(
            in_dim=4, hidden_dim=past_hidden_dim, out_dim=past_out_dim
        )
        self.vector_field = VectorFieldNet(
            dim=self.D, cond_dim=past_out_dim, hidden_dim=vf_hidden_dim
        )

    def forward(self, x_t, t, X_obs):
        """
        Compute v_theta(x_t, t | X_obs).
        """
        h_cond = self.encoder(X_obs)           # (B, cond_dim)
        v_pred = self.vector_field(x_t, t, h_cond)
        return v_pred
    

class TrajectoryCFMModel_v2(nn.Module):
    """
    Full model for conditional flow matching for trajectory prediction.
    Output the trajectory, instead of the velocity field.
    Inputs:
        x_t:    (B, D)   - point on ODE path at time t
        t:      (B,)     - time
        X_obs:  (B, obs_len, 4)  - observed past
    
    Output:
        v_pred: (B, D)   - vector field prediction
    """
    def __init__(
        self,
        obs_len=4,
        pred_len=60,
        past_hidden_dim=64,
        past_out_dim=128,
        hidden_dim=256,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.D = pred_len * 2

        self.encoder = PastEncoder(in_dim=4, hidden_dim=past_hidden_dim, out_dim=past_out_dim)

        # outputs x1_pred directly
        self.data_head = nn.Sequential(
            nn.Linear(self.D + past_out_dim + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.D),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
        )

    def forward(self, x_t, t, X_obs):
        h = self.encoder(X_obs)                 # (B, past_out_dim)
        if t.dim() == 1:
            t = t[:, None]                      # (B,1)
        t_emb = self.time_mlp(t)                # (B,16)
        inp = torch.cat([x_t, h, t_emb], dim=-1)
        x1_pred = self.data_head(inp)           # (B,D)  <- predicted clean trajectory
        return x1_pred