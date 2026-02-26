"""Robot-Centric Transformer models.

This module provides a minimal, self-contained Transformer that consumes
task-centric and observation-centric tokens, then predicts the next state
conditioned on time.

Token categories:

- Ego State (required): current ego state x_t, shape [B, 1, D_ego]
- Task (required): position/direction to goal, shape [B, 1, D_task]
- Obstacle (optional): aggregated obstacle info, shape [B, M_obs, D_obs] or [B, 1, D_obs]
- Other Agents (optional): shape [B, N_agents, D_agent]
- Other Goals (optional): per-agent goals, shape [B, N_agents, D_goal]
- Reference Path (optional): ego reference path over horizon, [B, H, D_path]
- Other Reference Paths (optional): per-agent reference paths, [B, N_agents, H, D_path]
  (No previous-action conditioning; decoding uses only an ego query.)

Ego State and Task are required. Time `t` is embedded and added as a global
conditioning signal. The wrapper exposes forward(t, x, y=None, observation=None)
in UNet-style, where `x` is a Tensor treated as the ego_state, and any extra
inputs are provided via `observation` as a dictionary.
"""

from typing import Optional, Dict, Tuple, List, Union

import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal embeddings like diffusion models.

    timesteps: shape [B] or [B, 1]
    returns: [B, dim]
    """
    if timesteps.dim() > 1:
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = 2 * math.pi * timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TokenTypeEmbedding(nn.Module):
    """Learned token type embeddings for categories.

    Types used:
        0=ego_state, 1=task, 2=obstacle,
        3=other_agent, 4=other_goal, 5=ref_path,
        6=other_ref_paths
    """

    def __init__(self, num_types: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(type_ids)


class PositionalEmbedding(nn.Module):
    """Learned positional embedding up to a maximum length."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.emb(positions)


class RobotCentricTransformerModel(nn.Module):
    """Robot-Centric Transformer with dedicated encoders and an action decoder.

    Builds a token sequence per batch by concatenating:
        [ego_state, task, obstacle(s), other agent(s), other goal(s), ref path, other ref paths]
    Runs a Transformer encoder to produce context memory, then conditions a
    decoder that predicts the instantaneous flow from an ego query.
    """

    def __init__(
        self,
        task_dim: int,
        ego_dim: int,
        obstacle_dim: Optional[int],
        other_agent_dim: Optional[int],
        other_goal_dim: Optional[int],
        ref_path_dim: Optional[int],
        other_ref_path_dim: Optional[int],
        state_dim: int,
        embed_dim: int = 256,
        ff_mult: int = 4,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        max_agents: int = 512,
        max_horizon: int = 512,
        use_time_embedding: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.use_time_embedding = use_time_embedding

        # Dedicated encoders
        self.task_encoder = MLP(task_dim, embed_dim, hidden_dim=embed_dim)
        self.ego_encoder = MLP(ego_dim, embed_dim, hidden_dim=embed_dim)
        self.obstacle_encoder = MLP(obstacle_dim, embed_dim, hidden_dim=embed_dim) if obstacle_dim is not None else None
        self.other_agent_encoder = MLP(other_agent_dim, embed_dim, hidden_dim=embed_dim) if other_agent_dim is not None else None
        self.other_goal_encoder = MLP(other_goal_dim, embed_dim, hidden_dim=embed_dim) if other_goal_dim is not None else None
        self.ref_path_encoder = MLP(ref_path_dim, embed_dim, hidden_dim=embed_dim) if ref_path_dim is not None else None
        self.other_ref_path_encoder = MLP(other_ref_path_dim, embed_dim, hidden_dim=embed_dim) if other_ref_path_dim is not None else None

        # Flow head (vector field output)
        self.flow_head = nn.Linear(embed_dim, state_dim)

        # Token type and positional embeddings
        # types: 0=ego_state, 1=task, 2=obstacle, 3=other_agent, 4=other_goal, 5=ref_path, 6=other_ref_paths
        self.type_emb = TokenTypeEmbedding(num_types=7, dim=embed_dim)
        self.agent_pos_emb = PositionalEmbedding(max_agents, embed_dim)
        self.horizon_pos_emb = PositionalEmbedding(max_horizon, embed_dim)

        # Optional time conditioning projected to the model dimension (GoalFlow-style MLP)
        self.time_proj = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if use_time_embedding
            else None
        )
        # Concatenate time embedding to every token, then project back to embed_dim
        self.time_concat_proj = (
            nn.Linear(embed_dim * 2, embed_dim)
            if use_time_embedding
            else None
        )

        # Transformer encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * embed_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * embed_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.final_ln = nn.LayerNorm(embed_dim)
        # Post-norm + dropout for encoder memory before feeding into the decoder
        self.memory_post_ln = nn.LayerNorm(embed_dim)
        self.memory_post_dropout = nn.Dropout(0.1)

    def _build_context_tokens(
        self,
        ego_state: torch.Tensor,
        task: torch.Tensor,
        obstacle: Optional[torch.Tensor],
        other_agents: Optional[torch.Tensor],
        other_goals: Optional[torch.Tensor],
        ref_path: Optional[torch.Tensor],
        other_ref_paths: Optional[torch.Tensor],
        obstacle_mask: Optional[torch.Tensor],
        other_agents_mask: Optional[torch.Tensor],
        other_goals_mask: Optional[torch.Tensor],
        ref_path_mask: Optional[torch.Tensor],
        other_ref_paths_mask: Optional[torch.Tensor],
        t_embed: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and concatenate tokens.

        Returns:
            tokens: [B, S, D]
            padding_mask: [B, S] (True for padding positions)
        """
        device = task.device
        bsz = task.shape[0]

        tokens = []
        mask_list: List[torch.Tensor] = []

        # Ego state token [B, 1, D] (required)
        e_tok = self.ego_encoder(ego_state)
        e_types = torch.full((bsz, e_tok.shape[1]), 0, dtype=torch.long, device=device)
        e_tok = e_tok + self.type_emb(e_types)
        tokens.append(e_tok)
        mask_list.append(torch.zeros((bsz, e_tok.shape[1]), dtype=torch.bool, device=device))

        # Task token [B, 1, D]
        t_tok = self.task_encoder(task)
        t_types = torch.full((bsz, t_tok.shape[1]), 1, dtype=torch.long, device=device)
        t_tok = t_tok + self.type_emb(t_types)
        tokens.append(t_tok)
        mask_list.append(torch.zeros((bsz, t_tok.shape[1]), dtype=torch.bool, device=device))

        # Obstacle tokens (optional) [B, M_obs, D]
        if obstacle is not None and self.obstacle_encoder is not None:
            o_tok = self.obstacle_encoder(obstacle)
            o_types = torch.full((bsz, o_tok.shape[1]), 2, dtype=torch.long, device=device)
            o_tok = o_tok + self.type_emb(o_types)
            tokens.append(o_tok)
            if obstacle_mask is None:
                mask_list.append(torch.zeros((bsz, o_tok.shape[1]), dtype=torch.bool, device=device))
            else:
                mask_list.append(obstacle_mask.to(device=device, dtype=torch.bool))

        # Other agents (optional) [B, N_agents, D]
        if other_agents is not None and self.other_agent_encoder is not None:
            a_tok = self.other_agent_encoder(other_agents)
            # Validate positional embedding capacity
            max_agents_supported = self.agent_pos_emb.emb.num_embeddings
            if a_tok.shape[1] > max_agents_supported:
                raise ValueError(
                    f"N_agents={a_tok.shape[1]} exceeds max_agents={max_agents_supported}; increase max_agents."
                )
            a_idx = torch.arange(a_tok.shape[1], device=device).view(1, -1).expand(bsz, -1)
            a_types = torch.full_like(a_idx, 3)
            a_tok = a_tok + self.type_emb(a_types) + self.agent_pos_emb(a_idx)
            tokens.append(a_tok)
            if other_agents_mask is None:
                mask_list.append(torch.zeros((bsz, a_tok.shape[1]), dtype=torch.bool, device=device))
            else:
                mask_list.append(other_agents_mask.to(device=device, dtype=torch.bool))

        # Other goals (optional) [B, N_agents, D]
        if other_goals is not None and self.other_goal_encoder is not None:
            og_tok = self.other_goal_encoder(other_goals)
            # Validate positional embedding capacity
            max_agents_supported = self.agent_pos_emb.emb.num_embeddings
            if og_tok.shape[1] > max_agents_supported:
                raise ValueError(
                    f"N_agents (goals)={og_tok.shape[1]} exceeds max_agents={max_agents_supported}; increase max_agents."
                )
            og_idx = torch.arange(og_tok.shape[1], device=device).view(1, -1).expand(bsz, -1)
            og_types = torch.full_like(og_idx, 4)
            og_tok = og_tok + self.type_emb(og_types) + self.agent_pos_emb(og_idx)
            tokens.append(og_tok)
            if other_goals_mask is None:
                mask_list.append(torch.zeros((bsz, og_tok.shape[1]), dtype=torch.bool, device=device))
            else:
                mask_list.append(other_goals_mask.to(device=device, dtype=torch.bool))

        # Reference path (optional) [B, H, D]
        if ref_path is not None and self.ref_path_encoder is not None:
            rp_tok = self.ref_path_encoder(ref_path)
            # Validate positional embedding capacity
            max_horizon_supported = self.horizon_pos_emb.emb.num_embeddings
            if rp_tok.shape[1] > max_horizon_supported:
                raise ValueError(
                    f"H (ref_path)={rp_tok.shape[1]} exceeds max_horizon={max_horizon_supported}; increase max_horizon."
                )
            rp_idx = torch.arange(rp_tok.shape[1], device=device).view(1, -1).expand(bsz, -1)
            rp_types = torch.full_like(rp_idx, 5)
            rp_tok = rp_tok + self.type_emb(rp_types) + self.horizon_pos_emb(rp_idx)
            tokens.append(rp_tok)
            if ref_path_mask is None:
                mask_list.append(torch.zeros((bsz, rp_tok.shape[1]), dtype=torch.bool, device=device))
            else:
                mask_list.append(ref_path_mask.to(device=device, dtype=torch.bool))

        # Other reference paths (optional) [B, N_agents, H, D]
        if other_ref_paths is not None and self.other_ref_path_encoder is not None:
            orp_tok = self.other_ref_path_encoder(other_ref_paths)  # [B, N, H, D]
            n_agents = orp_tok.shape[1]
            horizon = orp_tok.shape[2]
            # Validate positional embedding capacity
            max_agents_supported = self.agent_pos_emb.emb.num_embeddings
            max_horizon_supported = self.horizon_pos_emb.emb.num_embeddings
            if n_agents > max_agents_supported:
                raise ValueError(
                    f"N_agents (other_ref_paths)={n_agents} exceeds max_agents={max_agents_supported}; increase max_agents."
                )
            if horizon > max_horizon_supported:
                raise ValueError(
                    f"H (other_ref_paths)={horizon} exceeds max_horizon={max_horizon_supported}; increase max_horizon."
                )
            # Add agent + horizon positional embeddings with broadcasting
            agent_idx = torch.arange(n_agents, device=device).view(1, n_agents)
            hor_idx = torch.arange(horizon, device=device).view(1, horizon)
            agent_pos = self.agent_pos_emb(agent_idx).unsqueeze(2)  # [1, N, 1, D]
            hor_pos = self.horizon_pos_emb(hor_idx).unsqueeze(1)    # [1, 1, H, D]
            orp_tok = orp_tok + agent_pos + hor_pos
            # Type embedding
            orp_tok = orp_tok.reshape(bsz, n_agents * horizon, -1)
            orp_types = torch.full((bsz, orp_tok.shape[1]), 6, dtype=torch.long, device=device)
            orp_tok = orp_tok + self.type_emb(orp_types)
            tokens.append(orp_tok)
            if other_ref_paths_mask is None:
                mask_list.append(torch.zeros((bsz, n_agents * horizon), dtype=torch.bool, device=device))
            else:
                mask_list.append(other_ref_paths_mask.reshape(bsz, n_agents * horizon).to(device=device, dtype=torch.bool))

        x = torch.cat(tokens, dim=1)  # [B, S, D]
        padding_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 0 else torch.zeros((bsz, 0), dtype=torch.bool, device=device)

        if t_embed is not None and self.time_concat_proj is not None:
            # GoalFlow-style: concatenate time condition per token, then project
            t_embed = t_embed.to(dtype=x.dtype)
            t_rep = t_embed[:, None, :].expand(x.shape[0], x.shape[1], -1)
            x = torch.cat([x, t_rep], dim=-1)
            x = self.time_concat_proj(x)
        return x, padding_mask

    # No action tokens are used; decoding is driven by an ego query only.

    def forward(
        self,
        ego_state: torch.Tensor,
        task: torch.Tensor,
        obstacle: Optional[torch.Tensor],
        other_agents: Optional[torch.Tensor],
        other_goals: Optional[torch.Tensor],
        ref_path: Optional[torch.Tensor],
        other_ref_paths: Optional[torch.Tensor],
        obstacle_mask: Optional[torch.Tensor] = None,
        other_agents_mask: Optional[torch.Tensor] = None,
        other_goals_mask: Optional[torch.Tensor] = None,
        ref_path_mask: Optional[torch.Tensor] = None,
        other_ref_paths_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict instantaneous vector field (flow) v_theta(x_t, t, obs).

        - ego_state: [B, 1, D_ego] (required)
        - task: [B, 1, D_task] (required)
        - obstacle: [B, M_obs, D_obs] or None
        - other_agents: [B, N_agents, D_agent] or None
        - other_goals: [B, N_agents, D_goal] or None
        - ref_path: [B, H, D_path] or None
        - other_ref_paths: [B, N_agents, H, D_path] or None
        - t: [B] or scalar; used for time conditioning
        Returns: flow [B, state_dim]
        """
        device = task.device
        bsz = task.shape[0]

        # Time conditioning
        if self.use_time_embedding and t is not None:
            # Ensure t on same device/dtype as task and within [0,1]
            if t.dim() == 0:
                t = t.expand(task.shape[0])
            t = t.to(device=device, dtype=torch.float32)
            t = torch.clamp(t, 0.0, 1.0)
            t_emb = sinusoidal_embedding(t.view(-1), self.embed_dim)
            t_emb = self.time_proj(t_emb)
        else:
            t_emb = None

        # Build memory/context tokens and padding mask
        context_tokens, padding_mask = self._build_context_tokens(
            ego_state=ego_state,
            task=task,
            obstacle=obstacle,
            other_agents=other_agents,
            other_goals=other_goals,
            ref_path=ref_path,
            other_ref_paths=other_ref_paths,
            obstacle_mask=obstacle_mask,
            other_agents_mask=other_agents_mask,
            other_goals_mask=other_goals_mask,
            ref_path_mask=ref_path_mask,
            other_ref_paths_mask=other_ref_paths_mask,
            t_embed=t_emb,
        )
        memory = self.encoder(context_tokens, src_key_padding_mask=padding_mask)  # [B, S, D]
        # Apply post-norm and dropout to memory before decoder cross-attention
        memory = self.memory_post_ln(memory)
        memory = self.memory_post_dropout(memory)

        # Ego latent: first token is ego_state (since we prepend it)
        ego_latent = memory[:, 0, :]  # [B, D]

        # Build a query sequence using only ego token
        ego_as_seq = ego_latent.unsqueeze(1)  # [B, 1, D]
        if t_emb is not None and self.time_concat_proj is not None:
            # Ensure dtype consistency (e.g., under AMP)
            t_emb_cat = t_emb.to(dtype=ego_as_seq.dtype)
            ego_as_seq = torch.cat([ego_as_seq, t_emb_cat[:, None, :]], dim=-1)
            ego_as_seq = self.time_concat_proj(ego_as_seq)
        decoded = self.decoder(
            tgt=ego_as_seq,
            memory=memory,
            tgt_mask=None,
            memory_key_padding_mask=padding_mask,
        )

        # Predict flow from ego-decoded representation
        h = decoded[:, -1, :]  # [B, D]
        h = self.final_ln(h)
        flow = self.flow_head(h)  # [B, state_dim]
        return flow


class RobotCentricTransformerWrapper(nn.Module):
    """Wrapper exposing forward(t, x, y=None, observation=None) in UNet-style.

    - x: Tensor, treated as ego_state; will be flattened to [B, D_ego].
    - observation: dict carrying any extra optional inputs and masks, with keys:
        'task' (required), and optional 'obstacle', 'other_agents', 'other_goals',
        'ref_path', 'other_ref_paths', and corresponding masks '*_mask'.
    - y: kept for signature compatibility with existing pipeline; ignored.
    """

    def __init__(
        self,
        task_dim: int,
        ego_dim: int,
        obstacle_dim: Optional[int],
        other_agent_dim: Optional[int],
        other_goal_dim: Optional[int],
        ref_path_dim: Optional[int],
        other_ref_path_dim: Optional[int],
        state_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        ff_mult: int = 4,
        max_agents: int = 512,
        max_horizon: int = 512,
        use_time_embedding: bool = True,
    ):
        super().__init__()
        # Persist expected dims for shape assertions
        self._ego_dim = ego_dim
        self._task_dim = task_dim
        self._obstacle_dim = obstacle_dim
        self._other_agent_dim = other_agent_dim
        self._other_goal_dim = other_goal_dim
        self._ref_path_dim = ref_path_dim
        self._other_ref_path_dim = other_ref_path_dim
        self.model = RobotCentricTransformerModel(
            task_dim=task_dim,
            ego_dim=ego_dim,
            obstacle_dim=obstacle_dim,
            other_agent_dim=other_agent_dim,
            other_goal_dim=other_goal_dim,
            ref_path_dim=ref_path_dim,
            other_ref_path_dim=other_ref_path_dim,
            state_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            ff_mult=ff_mult,
            max_agents=max_agents,
            max_horizon=max_horizon,
            use_time_embedding=use_time_embedding,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        observation: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # UNet-style: x is Tensor (ego_state), extras are provided via `observation` dict
        if not isinstance(x, torch.Tensor):
            raise TypeError("RobotCentricTransformerWrapper expects x to be a Tensor.")
        extras = observation or {}
        orig_shape = x.shape
        bsz = x.shape[0]
        flat = x.reshape(bsz, -1)
        if flat.shape[1] != self._ego_dim:
            raise ValueError(
                f"Flattened ego_state dim must be {self._ego_dim}, got {flat.shape[1]}"
            )
        ego_state = flat.unsqueeze(1)

        def to_3d(name: str, tns: torch.Tensor, last: Optional[int]) -> torch.Tensor:
            if tns.dim() == 2:
                out = tns.unsqueeze(1)
            elif tns.dim() == 3:
                out = tns
            else:
                raise ValueError(f"{name} must be 2D or 3D, got {tuple(tns.shape)}")
            if last is not None and out.shape[-1] != last:
                raise ValueError(f"{name} last dim must be {last}, got {out.shape[-1]}")
            return out

        if "task" not in extras or extras["task"] is None:
            # Default to a zero task token if not provided
            task = torch.zeros(bsz, 1, self._task_dim, device=x.device, dtype=flat.dtype)
        else:
            task = to_3d("task", extras["task"], self._task_dim)

        obstacle = None
        if extras.get("obstacle", None) is not None:
            obstacle = to_3d("obstacle", extras["obstacle"], self._obstacle_dim if self._obstacle_dim is not None else None)

        other_agents = None
        if extras.get("other_agents", None) is not None:
            other_agents = to_3d("other_agents", extras["other_agents"], self._other_agent_dim if self._other_agent_dim is not None else None)

        other_goals = None
        if extras.get("other_goals", None) is not None:
            other_goals = to_3d("other_goals", extras["other_goals"], self._other_goal_dim if self._other_goal_dim is not None else None)

        ref_path = None
        if extras.get("ref_path", None) is not None:
            ref_path = to_3d("ref_path", extras["ref_path"], self._ref_path_dim if self._ref_path_dim is not None else None)

        other_ref_paths = extras.get("other_ref_paths", None)
        if other_ref_paths is not None:
            if other_ref_paths.dim() != 4:
                raise ValueError(
                    f"other_ref_paths must be 4D [B,N_agents,H,D], got {tuple(other_ref_paths.shape)}"
                )
            if self._other_ref_path_dim is not None and other_ref_paths.shape[-1] != self._other_ref_path_dim:
                raise ValueError(
                    f"other_ref_paths last dim must be {self._other_ref_path_dim}, got {other_ref_paths.shape[-1]}"
                )

        obstacle_mask = extras.get("obstacle_mask", None)
        other_agents_mask = extras.get("other_agents_mask", None)
        other_goals_mask = extras.get("other_goals_mask", None)
        ref_path_mask = extras.get("ref_path_mask", None)
        other_ref_paths_mask = extras.get("other_ref_paths_mask", None)

        flow = self.model(
            ego_state=ego_state,
            task=task,
            obstacle=obstacle,
            other_agents=other_agents,
            other_goals=other_goals,
            ref_path=ref_path,
            other_ref_paths=other_ref_paths,
            obstacle_mask=obstacle_mask,
            other_agents_mask=other_agents_mask,
            other_goals_mask=other_goals_mask,
            ref_path_mask=ref_path_mask,
            other_ref_paths_mask=other_ref_paths_mask,
            t=t,
        )

        # Reshape to original x shape
        out = flow.reshape(*orig_shape)
        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)
        return out


