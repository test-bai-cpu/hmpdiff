"""A basic Vision Transformer encoder for CFM tasks (MID-style baseline).

Design goals:
- Minimal, readable, dependency-light.
- Patch embedding -> TransformerEncoder -> head that maps back to input shape.
- Compatible wrapper exposing forward(t, x, y=None) like UNetModelWrapper.

Usage:
    model = VisionTransformerCFMWrapper(dim=(C,H,W), embed_dim=256, depth=8, num_heads=8)
    vt = model(t, x)  # x is Tensor [B,C,H,W]
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if timesteps.dim() > 1:
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DecoderLayerCFM(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_val = dropout

        # Query self-attention projections
        self.sa_qcontent_proj = nn.Linear(d_model, d_model, bias=True)
        self.sa_qpos_proj = nn.Linear(d_model, d_model, bias=True)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model, bias=True)
        self.sa_kpos_proj = nn.Linear(d_model, d_model, bias=True)
        self.sa_v_proj = nn.Linear(d_model, d_model, bias=True)
        self.sa_o_proj = nn.Linear(d_model, d_model, bias=True)

        # Cross-attention projections (query to context)
        self.ca_qcontent_proj = nn.Linear(d_model, d_model, bias=True)
        self.ca_qpos_proj = nn.Linear(d_model, d_model, bias=True)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model, bias=True)
        self.ca_kpos_proj = nn.Linear(d_model, d_model, bias=True)
        self.ca_v_proj = nn.Linear(d_model, d_model, bias=True)
        self.ca_o_proj = nn.Linear(d_model, d_model, bias=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        query: torch.Tensor,                      # [B, N, D]
        context: torch.Tensor,                    # [B, M, D]
        query_sa_pos_embeddings: torch.Tensor,    # [B, N, D]
        query_ca_pos_embeddings: torch.Tensor,    # [B, N, D]
        context_ca_pos_embeddings: torch.Tensor,  # [B, M, D]
    ) -> torch.Tensor:
        b, n, d = query.shape
        m = context.shape[1]
        h = self.nhead
        dh = d // h

        # Self-Attention on queries
        sa_q = self.sa_qcontent_proj(query) + self.sa_qpos_proj(query_sa_pos_embeddings)  # [B,N,D]
        sa_k = self.sa_kcontent_proj(query) + self.sa_kpos_proj(query_sa_pos_embeddings)  # [B,N,D]
        sa_v = self.sa_v_proj(query)                                                     # [B,N,D]

        sa_q = sa_q.view(b, n, h, dh).transpose(1, 2)  # [B,H,N,dh]
        sa_k = sa_k.view(b, n, h, dh).transpose(1, 2)
        sa_v = sa_v.view(b, n, h, dh).transpose(1, 2)
        sa_out = F.scaled_dot_product_attention(sa_q, sa_k, sa_v, dropout_p=self.dropout_val if self.training else 0.0)
        sa_out = sa_out.transpose(1, 2).contiguous().view(b, n, d)
        sa_out = self.sa_o_proj(sa_out)
        query = query + self.dropout1(sa_out)
        query = self.norm1(query)

        # Cross-Attention: queries attend to context
        ca_q = self.ca_qcontent_proj(query) + self.ca_qpos_proj(query_ca_pos_embeddings)  # [B,N,D]
        ca_k = self.ca_kcontent_proj(context) + self.ca_kpos_proj(context_ca_pos_embeddings)  # [B,M,D]
        ca_v = self.ca_v_proj(context)  # [B,M,D]

        ca_q = ca_q.view(b, n, h, dh).transpose(1, 2)  # [B,H,N,dh]
        ca_k = ca_k.view(b, m, h, dh).transpose(1, 2)  # [B,H,M,dh]
        ca_v = ca_v.view(b, m, h, dh).transpose(1, 2)  # [B,H,M,dh]
        ca_out = F.scaled_dot_product_attention(ca_q, ca_k, ca_v, dropout_p=self.dropout_val if self.training else 0.0)
        ca_out = ca_out.transpose(1, 2).contiguous().view(b, n, d)
        ca_out = self.ca_o_proj(ca_out)
        query = query + self.dropout2(ca_out)
        query = self.norm2(query)

        # FFN
        ffn = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = self.norm3(query + ffn)
        return query

class VisionTransformerCFM(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        use_time_embedding: bool = True,
        num_classes: Optional[int] = None,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_time_embedding = use_time_embedding
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        h, w = image_size

        # Sequence specs
        self.coord_dim = w
        self.future_len = h
        self.hist_len = (cond_dim // self.coord_dim) if cond_dim is not None else 3
        assert self.coord_dim == 2, "Expected coord_dim=2 for (x,y)"

        # History (context) embedding and positional encodings
        self.hist_proj = nn.Linear(self.coord_dim, embed_dim)
        # Noisy state (xt) per-step embedding for flow field conditioning
        self.xt_proj = nn.Linear(self.coord_dim, embed_dim)
        ctx_enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(ctx_enc_layer, num_layers=2)

        # Precompute sinusoidal position encodings for history and future queries (registered as buffers)
        with torch.no_grad():
            hist_pos = sinusoidal_embedding(torch.arange(self.hist_len), embed_dim).unsqueeze(0)  # [1, Hh, D]
            fut_pos = sinusoidal_embedding(torch.arange(self.future_len), embed_dim).unsqueeze(0)  # [1, Hf, D]
        self.register_buffer("hist_pos_embed", hist_pos)
        self.register_buffer("future_pos_embed", fut_pos)

        self.norm = nn.LayerNorm(embed_dim)
        # Per-step regression head: D -> 2
        self.traj_head = nn.Linear(embed_dim, self.coord_dim)
        self.in_channels = in_channels

        if use_time_embedding:
            self.time_proj = nn.Linear(embed_dim, embed_dim)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, embed_dim)

        if cond_dim is not None:
            # keep for backward compatibility (unused in the new path)
            self.cond_proj = nn.Linear(cond_dim, embed_dim)

        # Query initialization: learned content for each future timestep
        self.query_content = nn.Parameter(torch.zeros(1, self.future_len, embed_dim))
        nn.init.trunc_normal_(self.query_content, std=0.02)
        # Optional shallow encoder on queries
        q_pre_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.query_pre_encoder = nn.TransformerEncoder(q_pre_layer, num_layers=2)

        # DMT-style decoder layers: query self-attention + cross-attention to context + FFN
        self.num_heads = num_heads
        self.decoder_layers = nn.ModuleList([
            DecoderLayerCFM(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=0.1)
            for _ in range(depth)
        ])

    def forward(
        self,
        t: Optional[torch.Tensor],
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Inputs: x kept for API compatibility (expects [B,1,60,2]); condition must be [B,3,2] or [B,6]
        b = x.shape[0]

        # 1) Build context tokens from history condition: condition is [B, 3*2] or [B,3,2]
        assert condition is not None, "condition (history) must be provided as [B,3,2] or [B,6]"
        if condition.dim() == 2:
            hist = condition.view(b, self.hist_len, self.coord_dim)
        else:
            hist = condition
        ctx = self.hist_proj(hist)  # [B,3,D]
        ctx_pos = self.hist_pos_embed.to(hist.device).expand(b, -1, -1)  # [B,3,D]
        context = self.context_encoder(ctx + ctx_pos)    # [B,3,D]

        # 2) Initialize per-step future queries with content + positional enc + time embedding
        q = self.query_content.expand(b, -1, -1)  # [B,60,D]
        q_pos = self.future_pos_embed.to(self.query_content.device).expand(b, -1, -1)  # [B,60,D]
        q = q + q_pos
        # Add per-step noisy state embedding (xt)
        assert x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == self.future_len and x.shape[3] == self.coord_dim, \
            "x must be [B,1,future_len,coord_dim]"
        xt_steps = x[:, 0, :, :]               # [B,60,2]
        xt_emb = self.xt_proj(xt_steps)        # [B,60,D]
        q = q + xt_emb
        if self.use_time_embedding and t is not None:
            t_emb = sinusoidal_embedding(t.view(-1), self.embed_dim)
            t_emb = self.time_proj(t_emb)  # [B,D]
            q = q + t_emb[:, None, :]
        q = self.query_pre_encoder(q)  # [B,60,D]

        # 3) Decoder stack: SA on queries + CA to context
        for layer in self.decoder_layers:
            q = layer(
                query=q,
                context=context,
                query_sa_pos_embeddings=q_pos,
                query_ca_pos_embeddings=q_pos,
                context_ca_pos_embeddings=ctx_pos,
            )

        q = self.norm(q)
        vt = self.traj_head(q)         # [B,60,2] predicted velocity field per step
        out = vt.view(b, 1, self.future_len, self.coord_dim)  # [B,1,60,2]
        return out


class VisionTransformerCFMWrapper(nn.Module):
    def __init__(
        self,
        dim: Tuple[int, int, int],
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        use_time_embedding: bool = True,
        num_classes: Optional[int] = None,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        c, h, w = dim
        self.model = VisionTransformerCFM(
            image_size=(h, w),
            in_channels=c,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            use_time_embedding=use_time_embedding,
            num_classes=num_classes,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(t, x, y, condition=condition)


