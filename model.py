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

import math

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for scalar time, following MoFlow."""
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.theta) * torch.arange(half, device=device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
    

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
    

class TrajectoryCFMModel_v3(nn.Module):
    """
    Transformer-based conditional flow matching model for trajectory prediction.
    Adapted from MoFlow's ETHMotionTransformer for single-agent setting.
 
    Inputs (K-aware):
        x_t:    (B, K, D)           noisy trajectories at flow time τ
        t:      (B,)                flow time (shared across K)
        X_obs:  (B, obs_len, 4)     observed past trajectory
 
    Output:
        x1_pred: (B, K, D)          predicted clean trajectories
    """
 
    def __init__(
        self,
        obs_len: int = 4,
        pred_len: int = 60,
        K: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        past_hidden_dim: int = 64,
        past_out_dim: int = 128,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.K = K
        self.D = pred_len * 2
        self.d_model = d_model
 
        # ---- Past encoder (GRU, same as v2) ----
        self.past_encoder = PastEncoder(
            in_dim=4, hidden_dim=past_hidden_dim, out_dim=past_out_dim
        )
        # project encoder output to d_model
        self.past_proj = nn.Linear(past_out_dim, d_model)
 
        # ---- Time embedding (sinusoidal, like MoFlow) ----
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model, theta=10000),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
 
        # ---- Noisy trajectory embedding ----
        # Each timestep (x, y) -> d_model token
        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
 
        # ---- Temporal positional encoding (learned, for pred_len positions) ----
        self.temporal_pe = nn.Embedding(pred_len, d_model)
 
        # ---- K-sample query embedding (like motion_query_embedding in MoFlow) ----
        self.k_query_emb = nn.Embedding(K, d_model)
 
        # ---- Cross-K self-attention (like noisy_y_attn_k in MoFlow) ----
        self.attn_across_k = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
 
        # ---- Cross-time self-attention (replaces agent-attention in MoFlow) ----
        self.attn_across_t = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
 
        # ---- Fusion MLP: concat(encoder_out, y_emb, t_emb) -> d_model ----
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model + d_model + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
 
        # ---- Transformer decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
 
        # ---- Readout: token -> (x, y) per timestep ----
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
 
        self._init_weights()
 
    def _init_weights(self):
        """Light Xavier init for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        X_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:   (B, K, D)  or  (B, D)  — noisy trajectories
            t:     (B,)                    — flow time
            X_obs: (B, obs_len, 4)         — observed past
 
        Returns:
            x1_pred: same shape as x_t     — predicted clean trajectories
        """
        # Handle flat (B, D) input for backward compatibility
        squeezed = False
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)   # (B, 1, D)
            squeezed = True
 
        B, K, D = x_t.shape
        T = self.pred_len
        device = x_t.device
 
        # --- 1. Encode past ---
        h_past = self.past_encoder(X_obs)           # (B, past_out_dim)
        h_past = self.past_proj(h_past)             # (B, d_model)
 
        # --- 2. Time embedding ---
        # MoFlow scales flow time by 1000 for sinusoidal embedding
        t_emb = self.time_mlp(t * 1000.0)           # (B, d_model)
 
        # --- 3. Noisy trajectory -> per-timestep tokens ---
        y = x_t.view(B, K, T, 2)                    # (B, K, T, 2)
        y_emb = self.noisy_y_mlp(y)                 # (B, K, T, d_model)
 
        # Add temporal positional encoding
        t_pos = self.temporal_pe(torch.arange(T, device=device))  # (T, d_model)
        y_emb = y_emb + t_pos[None, None, :, :]     # broadcast over B, K
 
        # Add K-query embedding
        k_pos = self.k_query_emb(torch.arange(K, device=device))  # (K, d_model)
        y_emb = y_emb + k_pos[None, :, None, :]     # broadcast over B, T
 
        # --- 4. Attention across K (like MoFlow's noisy_y_attn_k) ---
        # Reshape: treat each (B, T) position independently, attend over K
        y_k = y_emb.permute(0, 2, 1, 3).reshape(B * T, K, self.d_model)  # (B*T, K, d)
        y_k = self.attn_across_k(y_k)
        y_emb = y_k.reshape(B, T, K, self.d_model).permute(0, 2, 1, 3)   # (B, K, T, d)
 
        # --- 5. Attention across time (like MoFlow's noisy_y_attn_a, but over T) ---
        y_t = y_emb.reshape(B * K, T, self.d_model)  # (B*K, T, d)
        y_t = self.attn_across_t(y_t)
        y_emb = y_t.reshape(B, K, T, self.d_model)   # (B, K, T, d)
 
        # --- 6. Fuse context: encoder + noisy_y + time ---
        # Expand h_past and t_emb to (B, K, T, d_model)
        h_past_exp = h_past[:, None, None, :].expand(B, K, T, self.d_model)
        t_emb_exp = t_emb[:, None, None, :].expand(B, K, T, self.d_model)
 
        fused = self.fusion_mlp(
            torch.cat([h_past_exp, y_emb, t_emb_exp], dim=-1)
        )  # (B, K, T, d_model)
 
        # Re-add positional encodings (like MoFlow's post_pe_cat_mlp)
        fused = fused + t_pos[None, None, :, :] + k_pos[None, :, None, :]
 
        # --- 7. Transformer decoder ---
        # Query: fused tokens (B*K, T, d)
        # Memory: past context as a single token (B*K, 1, d)
        query = fused.reshape(B * K, T, self.d_model)
        memory = h_past[:, None, :].expand(B, K, self.d_model).reshape(B * K, 1, self.d_model)
        # Add time info to memory
        t_mem = t_emb[:, None, :].expand(B, K, self.d_model).reshape(B * K, 1, self.d_model)
        memory = memory + t_mem
 
        decoded = self.decoder(query, memory)         # (B*K, T, d_model)
 
        # --- 8. Readout ---
        out = self.readout_mlp(decoded)               # (B*K, T, 2)
        x1_pred = out.reshape(B, K, T, 2).reshape(B, K, D)  # (B, K, D)
 
        if squeezed:
            x1_pred = x1_pred.squeeze(1)              # (B, D)
 
        return x1_pred
 