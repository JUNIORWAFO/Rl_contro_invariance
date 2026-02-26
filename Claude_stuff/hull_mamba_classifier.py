"""
hull_mamba_classifier.py
=========================
Mamba (Selective State Space Model) hull classifier.

Why Mamba over Transformer for this problem:
  - Your env IS a linear SSM: x_{t+1} = Ax + Bu. Mamba's inductive bias
    directly matches this generative structure.
  - O(T) inference vs O(T²) for Transformers — critical for long episodes.
  - Selective scan: the model learns WHICH past states to remember,
    naturally implementing a learned "hull memory".
  - No causal mask needed — SSMs are inherently causal by construction.

Design:
  - Input:   states only  (xDim,) per step
  - Model:   Stack of MambaBlocks (selective SSM + gated MLP)
  - Output:  Per-step soft probability P(state_t ∈ hull | s_0, ..., s_t)
  - Uncertainty: Bayesian SSM via stochastic hidden state perturbation

Architecture per MambaBlock:
  Residual(
    LayerNorm →
    Linear expand →
    Split: [z_gate | x_ssm]  →
    SSM selective scan (A, B, C, Δ all input-dependent) →
    Gated output (x_ssm ⊙ σ(z_gate)) →
    Linear project back
  )

Interface: identical to SequentialHullClassifier — drop-in replacement.
    monitor = MambaHullClassifier(xDim, action_bounds=...)
    result  = monitor.process(record)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Import shared types ───────────────────────────────────────────────────────
try:
    from hull_monitors import (
        EpisodeRecord, HullResult, PostEpisodeHullMonitor, _scan_episode,
    )
    from hull_sequence_classifier import (
        SequentialHullResult, TrajectoryLabelBuffer, HullClassifierTrainer,
        LabeledTrajectory,
    )
except ImportError:
    # ── Standalone fallbacks ──────────────────────────────────────────────────
    @dataclass
    class EpisodeRecord:
        states:  np.ndarray
        actions: np.ndarray
        rewards: np.ndarray
        ep_id:   int = 0

        @property
        def T(self): return len(self.actions)

    @dataclass
    class HullResult:
        in_hull_mask:    np.ndarray
        first_hull_step: Optional[int]
        is_valid_target: bool
        source:          str = "unknown"

        @property
        def hull_reached(self): return self.first_hull_step is not None

        def relabel_rewards(self, rewards, hull_reward):
            if not self.hull_reached: return rewards.copy()
            out = rewards.copy(); out[self.in_hull_mask] = hull_reward; return out

    @dataclass
    class SequentialHullResult(HullResult):
        probs:          np.ndarray = field(default_factory=lambda: np.array([]))
        uncertainty:    np.ndarray = field(default_factory=lambda: np.array([]))
        confident_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

        def relabel_rewards_confident(self, rewards, hull_reward):
            if not self.hull_reached: return rewards.copy()
            out = rewards.copy()
            out[self.in_hull_mask & self.confident_mask] = hull_reward
            return out

    @dataclass
    class LabeledTrajectory:
        states:  np.ndarray
        actions: np.ndarray
        labels:  np.ndarray

    class TrajectoryLabelBuffer:
        def __init__(self, max_episodes=1000, max_ep_len=512, pos_neg_ratio=0.3):
            self.max_ep_len    = max_ep_len
            self.pos_neg_ratio = pos_neg_ratio
            self._buffer: deque = deque(maxlen=max_episodes)

        def push(self, record, labels):
            T = min(record.T, self.max_ep_len)
            self._buffer.append(LabeledTrajectory(
                states=record.states[:T+1].copy(),
                actions=record.actions[:T].copy(),
                labels=labels[:T].astype(np.float32),
            ))

        def __len__(self): return len(self._buffer)

        def positive_fraction(self):
            if not self._buffer: return 0.0
            return float(np.concatenate([e.labels for e in self._buffer]).mean())

        def sample_batch(self, batch_size, device):
            if len(self._buffer) < batch_size: return None
            pos = [e for e in self._buffer if e.labels.max() > 0.5]
            neg = [e for e in self._buffer if e.labels.max() <= 0.5]
            n_pos = max(1, int(batch_size * self.pos_neg_ratio))
            chosen = []
            if pos:
                idx = np.random.choice(len(pos), min(n_pos, len(pos)), replace=False)
                chosen += [pos[i] for i in idx]
            if neg and len(chosen) < batch_size:
                idx = np.random.choice(len(neg), min(batch_size-len(chosen), len(neg)), replace=False)
                chosen += [neg[i] for i in idx]
            if not chosen: return None
            T_max = max(e.labels.shape[0] for e in chosen)
            xDim  = chosen[0].states.shape[-1]
            B     = len(chosen)
            s = np.zeros((B, T_max+1, xDim), dtype=np.float32)
            l = np.zeros((B, T_max),          dtype=np.float32)
            m = np.zeros((B, T_max),          dtype=bool)
            for i, ep in enumerate(chosen):
                Ti = ep.labels.shape[0]
                s[i, :Ti+1] = ep.states
                l[i, :Ti]   = ep.labels
                m[i, :Ti]   = True
            return (torch.tensor(s, device=device),
                    None,   # no actions — states only
                    torch.tensor(l, device=device),
                    torch.tensor(m, device=device))

    class PostEpisodeHullMonitor:
        """Stub — replace with real implementation."""
        def __init__(self, **kw):
            self.is_valid_target = False
            self.target = None
        def process(self, record):
            mask = np.zeros(record.T, dtype=bool)
            return HullResult(mask, None, False, "StubOracle")
        def full_reset(self): pass


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTIVE SCAN CORE  (pure PyTorch — no CUDA kernel dependency)
# ══════════════════════════════════════════════════════════════════════════════

def selective_scan(
    u:     torch.Tensor,   # (B, T, d_inner)
    delta: torch.Tensor,   # (B, T, d_inner)
    A:     torch.Tensor,   # (d_inner, d_state)  — log-space, fixed
    B:     torch.Tensor,   # (B, T, d_state)     — input-dependent
    C:     torch.Tensor,   # (B, T, d_state)     — input-dependent
    D:     torch.Tensor,   # (d_inner,)           — skip connection
) -> torch.Tensor:
    """
    Selective SSM scan: discretize continuous SSM then scan over time.

    Continuous SSM:  h'(t) = A h(t) + B(t) u(t)
                     y(t)  = C(t) h(t) + D u(t)

    Discretized (ZOH):
        Ā = exp(Δ · A)
        B̄ = (Ā - I) · A⁻¹ · B  ≈ Δ · B  (first-order approx, stable)
        h_t = Ā h_{t-1} + B̄_t u_t
        y_t = C_t h_t + D u_t

    The selectivity comes from Δ, B, C all being functions of the input u,
    allowing the model to decide per-step what to store and what to forget.

    Returns: y  (B, T, d_inner)
    """
    B_sz, T, d_inner = u.shape
    d_state          = A.shape[1]

    # Discretize: Δ is (B, T, d_inner), A is (d_inner, d_state)
    # delta_A: (B, T, d_inner, d_state)
    delta_A = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )

    # delta_B_u: (B, T, d_inner, d_state)
    delta_B_u = (
        delta.unsqueeze(-1)              # (B, T, d_inner, 1)
        * B.unsqueeze(2)                 # (B, T, 1, d_state)
        * u.unsqueeze(-1)                # (B, T, d_inner, 1)
    )

    # Sequential scan over time
    h = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(T):
        # h: (B, d_inner, d_state)
        h = delta_A[:, t] * h + delta_B_u[:, t]        # selective state update
        # y_t = C_t · h_t  summed over d_state → (B, d_inner)
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)   # (B, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)          # (B, T, d_inner)
    return y + u * D.unsqueeze(0).unsqueeze(0)          # skip connection


# ══════════════════════════════════════════════════════════════════════════════
#  MAMBA BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    One Mamba block with residual connection:

        x_norm = LayerNorm(x)
        [z, x_ssm] = Linear_expand(x_norm).split(2)     ← gating
        x_ssm = Conv1d(x_ssm)                            ← local context
        x_ssm = SiLU(x_ssm)
        Δ, B, C = Linear_ssm_params(x_ssm).split(...)   ← input-dependent SSM params
        y = SelectiveScan(x_ssm, Δ, A, B, C, D)
        out = y ⊙ SiLU(z)                               ← gated output
        out = Linear_out(out)
        return x + out                                   ← residual

    The A matrix is initialized as -diag(1..d_state) in log-space and kept
    negative to ensure stable (contracting) dynamics — analogous to your
    system needing stable A for the env to stay bounded.
    """

    def __init__(
        self,
        d_model:   int,
        d_state:   int   = 16,     # SSM state dimension (N in Mamba paper)
        d_conv:    int   = 4,      # local conv kernel size
        expand:    int   = 2,      # expansion factor for inner dim
        dt_rank:   int   = None,   # rank of Δ projection (None → d_model//16)
        dt_min:    float = 0.001,
        dt_max:    float = 0.1,
        dt_init:   str   = "random",
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.d_inner  = d_model * expand
        self.dt_rank  = dt_rank or math.ceil(d_model / 16)

        # Input projection: expand and split into [z_gate, x_ssm]
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local depthwise conv for short-range context
        self.conv1d   = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.d_inner,
            bias         = True,
        )

        # SSM parameter projections (all input-dependent → "selective")
        self.x_proj   = nn.Linear(self.d_inner,
                                   self.dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize Δ (dt) in log-space between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_proj.bias = nn.Parameter(
            dt + torch.log(-torch.expm1(-dt))   # inverse softplus
        )

        # A: fixed negative matrix in log-space  → stable dynamics
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: skip connection scalar per channel
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm for pre-norm residual
        self.norm     = nn.LayerNorm(d_model)

        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, T, d_model)
        returns: (B, T, d_model)
        """
        residual = x
        x        = self.norm(x)
        B, T, _  = x.shape

        # Expand and gate
        xz      = self.in_proj(x)                              # (B, T, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                         # each (B, T, d_inner)

        # Local conv (causal: trim right padding)
        x_ssm_t = x_ssm.transpose(1, 2)                        # (B, d_inner, T)
        x_ssm_t = self.conv1d(x_ssm_t)[:, :, :T]              # causal trim
        x_ssm   = x_ssm_t.transpose(1, 2)                      # (B, T, d_inner)
        x_ssm   = F.silu(x_ssm)

        # Project to SSM parameters
        x_dbl   = self.x_proj(x_ssm)                           # (B, T, dt_rank + 2*d_state)
        dt, B_mat, C_mat = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Δ: softplus to keep positive
        dt      = F.softplus(self.dt_proj(dt))                 # (B, T, d_inner)

        # A: negative definite
        A       = -torch.exp(self.A_log)                       # (d_inner, d_state)

        # Selective scan
        y       = selective_scan(x_ssm, dt, A, B_mat, C_mat, self.D)  # (B, T, d_inner)

        # Gate with z
        y       = y * F.silu(z)

        # Project back
        out     = self.out_proj(y)                             # (B, T, d_model)
        return residual + self.dropout(out)


# ══════════════════════════════════════════════════════════════════════════════
#  BAYESIAN SSM BLOCK  (uncertainty via stochastic hidden state)
# ══════════════════════════════════════════════════════════════════════════════

class BayesianMambaBlock(MambaBlock):
    """
    MambaBlock extended with stochastic hidden state perturbation
    for epistemic uncertainty estimation.

    During uncertainty inference, Gaussian noise is injected into the
    SSM hidden state at each step. Multiple forward passes with different
    noise samples give a distribution over outputs — analogous to
    MC Dropout but grounded in the SSM state dynamics.

    noise_std controls the perturbation scale. Set to 0 for deterministic mode.
    """

    def __init__(self, *args, noise_std: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std    = noise_std
        self._noisy_mode  = False

    def set_noisy(self, flag: bool):
        self._noisy_mode = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._noisy_mode or self.noise_std == 0.0:
            return super().forward(x)

        # Inject noise into the SSM scan
        residual = x
        x        = self.norm(x)
        B, T, _  = x.shape

        xz            = self.in_proj(x)
        x_ssm, z      = xz.chunk(2, dim=-1)
        x_ssm_t       = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_ssm         = F.silu(x_ssm_t)

        x_dbl         = self.x_proj(x_ssm)
        dt, B_mat, C_mat = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt            = F.softplus(self.dt_proj(dt))
        A             = -torch.exp(self.A_log)

        # ── Noisy selective scan ──────────────────────────────────────────────
        B_sz, T, d_inner = x_ssm.shape
        d_state          = A.shape[1]
        delta_A          = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B_u        = (dt.unsqueeze(-1) * B_mat.unsqueeze(2) * x_ssm.unsqueeze(-1))

        h   = torch.zeros(B_sz, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys  = []
        for t in range(T):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            # Inject noise into hidden state
            h = h + torch.randn_like(h) * self.noise_std
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        y   = torch.stack(ys, dim=1) + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y   = y * F.silu(z)
        out = self.out_proj(y)
        return residual + self.dropout(out)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE INPUT ENCODER  (states only)
# ══════════════════════════════════════════════════════════════════════════════

class StateEncoder(nn.Module):
    """
    Encodes raw state vectors into d_model tokens.

    States-only design — the SSM's recurrent memory implicitly captures
    trajectory history, so we don't need explicit delta or action inputs.

    Two projection paths are fused:
      - Direct:    state → d_model
      - Normalized: LayerNorm(state) → d_model  (stabilizes large state ranges)
    The fusion lets the model learn how much normalization to apply.
    """

    def __init__(self, xDim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm         = nn.LayerNorm(xDim)
        self.proj_raw     = nn.Linear(xDim, d_model // 2, bias=True)
        self.proj_normed  = nn.Linear(xDim, d_model // 2, bias=True)
        self.fusion       = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.drop         = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        states:  (B, T, xDim)   — state sequence (NOT including s_T+1)
        returns: (B, T, d_model)
        """
        raw    = self.proj_raw(states)
        normed = self.proj_normed(self.norm(states))
        fused  = self.fusion(torch.cat([raw, normed], dim=-1))
        return self.drop(fused)


# ══════════════════════════════════════════════════════════════════════════════
#  HULL PROBABILITY HEAD  (soft output)
# ══════════════════════════════════════════════════════════════════════════════

class HullProbHead(nn.Module):
    """
    Predicts P(state_t ∈ hull | s_0..s_t) as a soft probability in [0,1].

    Uses a two-layer MLP with residual to prevent vanishing gradients
    in deep SSM stacks.

    Soft labels from the oracle allow the model to express uncertainty
    about borderline hull membership (states near the hull boundary).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h:       (B, T, d_model)
        returns: (B, T)   probabilities in [0, 1]
        """
        return torch.sigmoid(self.net(h).squeeze(-1))


# ══════════════════════════════════════════════════════════════════════════════
#  FULL MAMBA HULL NET
# ══════════════════════════════════════════════════════════════════════════════

class MambaHullNet(nn.Module):
    """
    Full model: StateEncoder → N × BayesianMambaBlock → HullProbHead

    No positional encoding needed — SSMs are inherently order-aware
    through their recurrent state dynamics.

    Uncertainty estimation via stochastic hidden state injection
    (BayesianMambaBlock.set_noisy(True) + multiple forward passes).
    """

    def __init__(
        self,
        xDim:       int,
        d_model:    int   = 64,
        d_state:    int   = 16,
        d_conv:     int   = 4,
        expand:     int   = 2,
        num_layers: int   = 4,
        dropout:    float = 0.1,
        noise_std:  float = 0.05,
    ):
        super().__init__()
        self.encoder = StateEncoder(xDim, d_model, dropout)

        self.blocks  = nn.ModuleList([
            BayesianMambaBlock(
                d_model  = d_model,
                d_state  = d_state,
                d_conv   = d_conv,
                expand   = expand,
                dropout  = dropout,
                noise_std = noise_std,
            )
            for _ in range(num_layers)
        ])

        self.head = HullProbHead(d_model, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        states:  (B, T, xDim)  — state sequence s_0..s_{T-1}
        returns: (B, T)        — P(s_t ∈ hull | s_0..s_t)
        """
        x = self.encoder(states)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def set_noisy(self, flag: bool):
        for block in self.blocks:
            block.set_noisy(flag)

    def predict_with_uncertainty(
        self,
        states:    torch.Tensor,
        n_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic hidden state uncertainty estimation.

        Injects Gaussian noise into SSM hidden states across n_samples
        forward passes. The variance of outputs reflects how sensitive
        hull membership is to trajectory perturbations — a natural
        measure of proximity to the hull boundary.

        Returns:
            mean: (B, T) — mean probability
            std:  (B, T) — epistemic uncertainty
        """
        self.set_noisy(True)
        self.train()   # keeps dropout active too

        with torch.no_grad():
            samples = torch.stack(
                [self(states) for _ in range(n_samples)], dim=0
            )   # (n_samples, B, T)

        self.set_noisy(False)
        self.eval()

        return samples.mean(dim=0), samples.std(dim=0)

    @torch.no_grad()
    def predict_deterministic(self, states: torch.Tensor) -> torch.Tensor:
        self.set_noisy(False)
        self.eval()
        return self(states)

    def recurrent_step(
        self,
        state_t: torch.Tensor,   # (B, xDim) — single step
        h_prev:  List[torch.Tensor],   # list of (B, d_inner, d_state) per layer
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        True O(1) recurrent inference — process one step at a time
        without recomputing from scratch. Useful for online deployment.

        Returns:
            prob:   (B,)  — P(s_t ∈ hull)
            h_next: list of updated hidden states per layer
        """
        x      = self.encoder(state_t.unsqueeze(1)).squeeze(1)  # (B, d_model)
        h_next = []

        for i, block in enumerate(self.blocks):
            # Manual single-step SSM update (bypass batch scan)
            h_i    = h_prev[i]                                  # (B, d_inner, d_state)
            x_exp  = x.unsqueeze(1)                             # (B, 1, d_model)
            xz     = block.in_proj(block.norm(x_exp))
            x_ssm, z = xz.chunk(2, dim=-1)

            # Conv: single step — just use the last d_conv values
            x_ssm  = F.silu(x_ssm.squeeze(1))                  # (B, d_inner)

            x_dbl  = block.x_proj(x_ssm)
            dt, B_mat, C_mat = x_dbl.split(
                [block.dt_rank, block.d_state, block.d_state], dim=-1)
            dt     = F.softplus(block.dt_proj(dt))              # (B, d_inner)
            A      = -torch.exp(block.A_log)                    # (d_inner, d_state)

            # Single SSM step
            dA     = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))    # (B, d_inner, d_state)
            dBu    = dt.unsqueeze(-1) * B_mat.unsqueeze(1) * x_ssm.unsqueeze(-1)
            h_new  = dA * h_i + dBu                            # (B, d_inner, d_state)
            y      = (h_new * C_mat.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            y      = y * F.silu(z.squeeze(1))
            x      = x + block.out_proj(y)                     # residual
            h_next.append(h_new)

        prob = torch.sigmoid(self.head.net(x.unsqueeze(1)).squeeze())
        return prob, h_next

    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize zero hidden states for all layers."""
        return [
            torch.zeros(batch_size, block.d_inner, block.d_state, device=device)
            for block in self.blocks
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL BUFFER (states-only version)
# ══════════════════════════════════════════════════════════════════════════════

class StateLabelBuffer:
    """
    Label buffer that stores states-only trajectories.
    Lighter than TrajectoryLabelBuffer (no action storage).
    Supports soft labels in [0, 1].
    """

    def __init__(
        self,
        max_episodes:  int   = 1000,
        max_ep_len:    int   = 512,
        pos_neg_ratio: float = 0.3,
    ):
        self.max_ep_len    = max_ep_len
        self.pos_neg_ratio = pos_neg_ratio
        self._buffer: deque = deque(maxlen=max_episodes)

    def push(self, states: np.ndarray, labels: np.ndarray):
        """
        states: (T, xDim)   — state sequence (NOT T+1)
        labels: (T,)        — soft labels in [0, 1]
        """
        T = min(len(labels), self.max_ep_len)
        self._buffer.append({
            "states": states[:T].astype(np.float32),
            "labels": labels[:T].astype(np.float32),
        })

    def __len__(self): return len(self._buffer)

    def positive_fraction(self) -> float:
        if not self._buffer: return 0.0
        return float(np.concatenate([e["labels"] for e in self._buffer]).mean())

    def sample_batch(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns: states (B, T_max, xDim), labels (B, T_max), mask (B, T_max)
        """
        if len(self._buffer) < max(2, batch_size // 4):
            return None

        pos = [e for e in self._buffer if e["labels"].max() > 0.5]
        neg = [e for e in self._buffer if e["labels"].max() <= 0.5]

        n_pos   = max(1, int(batch_size * self.pos_neg_ratio))
        chosen  = []
        if pos:
            idx = np.random.choice(len(pos), min(n_pos, len(pos)), replace=False)
            chosen += [pos[i] for i in idx]
        if neg and len(chosen) < batch_size:
            n_neg = batch_size - len(chosen)
            idx   = np.random.choice(len(neg), min(n_neg, len(neg)), replace=False)
            chosen += [neg[i] for i in idx]
        if not chosen:
            return None

        T_max = max(e["labels"].shape[0] for e in chosen)
        xDim  = chosen[0]["states"].shape[-1]
        B     = len(chosen)

        states_arr = np.zeros((B, T_max, xDim), dtype=np.float32)
        labels_arr = np.zeros((B, T_max),       dtype=np.float32)
        mask_arr   = np.zeros((B, T_max),       dtype=bool)

        for i, ep in enumerate(chosen):
            Ti = ep["labels"].shape[0]
            states_arr[i, :Ti] = ep["states"]
            labels_arr[i, :Ti] = ep["labels"]
            mask_arr[i,   :Ti] = True

        return (
            torch.tensor(states_arr, device=device),
            torch.tensor(labels_arr, device=device),
            torch.tensor(mask_arr,   device=device),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAMBA TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class MambaHullTrainer:
    """
    Training loop for MambaHullNet.

    Loss: soft BCE (supports oracle soft labels) with positive upweighting.
    LR schedule: warmup + cosine annealing.
    Gradient clipping to keep SSM A eigenvalues stable.
    """

    def __init__(
        self,
        net:          MambaHullNet,
        lr:           float = 3e-4,
        weight_decay: float = 1e-4,
        pos_weight:   float = 5.0,
        warmup_steps: int   = 300,
        device:       torch.device = torch.device("cpu"),
    ):
        self.net    = net
        self.device = device

        # Separate param groups: SSM matrices get lower LR for stability
        ssm_params  = [p for n, p in net.named_parameters()
                       if "A_log" in n or "D" in n]
        other_params = [p for n, p in net.named_parameters()
                        if "A_log" not in n and "D" not in n]

        self.optimizer = optim.AdamW([
            {"params": ssm_params,   "lr": lr * 0.1},   # SSM matrices: slow
            {"params": other_params, "lr": lr},
        ], weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=lr * 0.01)

        self._pos_weight = pos_weight
        self._step       = 0
        self._warmup     = warmup_steps
        self.loss_history: List[float] = []

    def train_epoch(
        self,
        buffer:    StateLabelBuffer,
        batch_size: int = 32,
        n_batches:  int = 10,
    ) -> float:
        self.net.train()
        self.net.set_noisy(False)
        total = 0.0
        count = 0

        for _ in range(n_batches):
            batch = buffer.sample_batch(batch_size, self.device)
            if batch is None:
                break
            states, labels, mask = batch

            preds = self.net(states)                             # (B, T)

            # Soft BCE with positive upweighting
            weights             = torch.ones_like(labels)
            weights[labels > 0.5] = self._pos_weight

            loss_raw = F.binary_cross_entropy(preds, labels, reduction="none")
            loss     = (loss_raw * weights * mask.float()).sum() \
                     / (mask.float().sum() + 1e-8)

            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients — important for SSM stability
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            # Ensure A stays negative (stable SSM)
            with torch.no_grad():
                for block in self.net.blocks:
                    block.A_log.data.clamp_(min=0.0)  # log|A| >= 0 → |A| >= 1, clamp

            self.optimizer.step()

            # Warmup
            if self._step < self._warmup:
                scale = (self._step + 1) / self._warmup
                for pg in self.optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * scale if "initial_lr" in pg \
                               else pg["lr"] * scale
            else:
                self.scheduler.step()

            self._step += 1
            total      += loss.item()
            count      += 1

        mean = total / max(count, 1)
        self.loss_history.append(mean)
        return mean

    def compute_metrics(
        self,
        buffer:    StateLabelBuffer,
        batch_size: int   = 64,
        threshold:  float = 0.5,
    ) -> dict:
        self.net.eval()
        self.net.set_noisy(False)
        batch = buffer.sample_batch(batch_size, self.device)
        if batch is None:
            return {}
        states, labels, mask = batch

        with torch.no_grad():
            preds = self.net(states)

        pred_bin = (preds > threshold).float()
        tp = ((pred_bin == 1) & (labels > 0.5) & mask).float().sum()
        fp = ((pred_bin == 1) & (labels <= 0.5) & mask).float().sum()
        fn = ((pred_bin == 0) & (labels > 0.5) & mask).float().sum()
        tn = ((pred_bin == 0) & (labels <= 0.5) & mask).float().sum()

        acc  = (tp + tn) / (mask.float().sum() + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        return {
            "accuracy":   acc.item(),
            "precision":  prec.item(),
            "recall":     rec.item(),
            "f1":         f1.item(),
            "pos_frac":   buffer.positive_fraction(),
            "train_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MAMBA HULL CLASSIFIER MONITOR  (drop-in for SequentialHullClassifier)
# ══════════════════════════════════════════════════════════════════════════════

class MambaHullClassifier:
    """
    Drop-in replacement for SequentialHullClassifier using a Mamba SSM.

    Identical external interface:
        monitor = MambaHullClassifier(xDim, action_bounds=action_bounds)
        result  = monitor.process(record)   # SequentialHullResult

    Key differences from Transformer version:
      - Input: states only (no actions, no deltas)
      - Uncertainty: stochastic hidden state (not MC Dropout)
      - Inference: O(T) instead of O(T²)
      - Supports true O(1) online recurrent inference via .step()
    """

    def __init__(
        self,
        xDim:                    int,
        action_bounds,
        target                   = None,
        # Oracle
        oracle_method:           str   = "linprog",
        hull_tolerance:          float = 1e-6,
        label_every:             int   = 10,
        train_every:             int   = 5,
        # Mamba architecture
        d_model:                 int   = 64,
        d_state:                 int   = 16,
        d_conv:                  int   = 4,
        expand:                  int   = 2,
        num_layers:              int   = 4,
        dropout:                 float = 0.1,
        noise_std:               float = 0.05,
        # Training
        lr:                      float = 3e-4,
        weight_decay:            float = 1e-4,
        pos_weight:              float = 5.0,
        batch_size:              int   = 32,
        n_batches_per_train:     int   = 20,
        buffer_size:             int   = 1000,
        max_ep_len:              int   = 512,
        min_buffer_before_train: int   = 10,
        # Inference
        n_samples:               int   = 30,
        threshold:               float = 0.5,
        uncertainty_threshold:   float = 0.15,
        # Device
        device:                  str   = "cpu",
    ):
        self.xDim              = xDim
        self.action_bounds     = action_bounds
        self.label_every       = label_every
        self.train_every       = train_every
        self.batch_size        = batch_size
        self.n_batches         = n_batches_per_train
        self.min_buffer        = min_buffer_before_train
        self.n_samples         = n_samples
        self.threshold         = threshold
        self.unc_threshold     = uncertainty_threshold
        self.max_ep_len        = max_ep_len
        self.device            = torch.device(device)

        # Oracle
        uDim = 1   # dummy — oracle doesn't use actions in states-only mode
        try:
            self._oracle = PostEpisodeHullMonitor(
                xDim=xDim, uDim=uDim, action_bounds=action_bounds,
                target=target, method=oracle_method, hull_tolerance=hull_tolerance,
            )
        except Exception:
            self._oracle = None

        self.target          = (None if target is None
                                else np.asarray(target, dtype=np.float32))
        self.is_valid_target = self.target is not None

        # Mamba model
        self.net = MambaHullNet(
            xDim=xDim, d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, num_layers=num_layers,
            dropout=dropout, noise_std=noise_std,
        ).to(self.device)

        # Buffer and trainer (states only)
        self.buffer  = StateLabelBuffer(
            max_episodes=buffer_size, max_ep_len=max_ep_len)
        self.trainer = MambaHullTrainer(
            self.net, lr=lr, weight_decay=weight_decay,
            pos_weight=pos_weight, device=self.device)

        # Counters
        self._ep_count    = 0
        self._label_count = 0
        self._trained     = False
        self._last_metrics: dict = {}

    # ── Main entry point ─────────────────────────────────────────────────────

    def process(self, record: EpisodeRecord) -> SequentialHullResult:
        self._ep_count += 1
        run_oracle = (self._ep_count % self.label_every == 0) or not self._trained

        if run_oracle:
            return self._process_oracle(record)
        return self._process_mamba(record)

    # ── Oracle path ───────────────────────────────────────────────────────────

    def _process_oracle(self, record: EpisodeRecord) -> SequentialHullResult:
        oracle_result = self._oracle.process(record) if self._oracle else None

        if oracle_result is not None:
            labels = oracle_result.in_hull_mask.astype(np.float32)

            # Use states s_0..s_{T-1} as inputs (current states, not next)
            states_input = record.states[:record.T]
            self.buffer.push(states_input, labels)
            self._label_count += 1

            # Sync target
            if (self._oracle and self._oracle.is_valid_target
                    and not self.is_valid_target):
                self.target          = self._oracle.target
                self.is_valid_target = True

            # Retrain
            if (self._label_count % self.train_every == 0
                    and len(self.buffer) >= self.min_buffer):
                loss = self.trainer.train_epoch(
                    self.buffer, self.batch_size, self.n_batches)
                self._last_metrics = self.trainer.compute_metrics(self.buffer)
                self._last_metrics["train_loss"] = loss
                self._trained = True
                print(f"[MambaHullClassifier] Trained | "
                      f"loss={loss:.4f} | "
                      f"f1={self._last_metrics.get('f1', 0):.3f} | "
                      f"buffer={len(self.buffer)}")

            return SequentialHullResult(
                in_hull_mask    = oracle_result.in_hull_mask,
                first_hull_step = oracle_result.first_hull_step,
                is_valid_target = self.is_valid_target,
                source          = "Oracle",
                probs           = labels,
                uncertainty     = np.zeros(record.T, dtype=np.float32),
                confident_mask  = np.ones(record.T, dtype=bool),
            )

        return self._process_mamba(record)

    # ── Mamba inference path ──────────────────────────────────────────────────

    def _process_mamba(self, record: EpisodeRecord) -> SequentialHullResult:
        T       = record.T
        T_eff   = min(T, self.max_ep_len)

        states  = torch.tensor(
            record.states[:T_eff][np.newaxis],
            dtype=torch.float32, device=self.device,
        )   # (1, T_eff, xDim)

        if self._trained:
            mean, std = self.net.predict_with_uncertainty(states, self.n_samples)
            probs = mean.squeeze(0).cpu().numpy()   # (T_eff,)
            unc   = std.squeeze(0).cpu().numpy()    # (T_eff,)
        else:
            probs = np.full(T_eff, 0.5, dtype=np.float32)
            unc   = np.full(T_eff, 1.0, dtype=np.float32)

        # Pad if truncated
        if T_eff < T:
            pad   = T - T_eff
            probs = np.concatenate([probs, np.full(pad, 0.5)])
            unc   = np.concatenate([unc,   np.full(pad, 1.0)])

        in_hull        = probs >= self.threshold
        confident_mask = unc   <= self.unc_threshold
        first_hit      = int(np.argmax(in_hull)) if in_hull.any() else None

        return SequentialHullResult(
            in_hull_mask    = in_hull,
            first_hull_step = first_hit,
            is_valid_target = self.is_valid_target,
            source          = "Mamba",
            probs           = probs,
            uncertainty     = unc,
            confident_mask  = confident_mask,
        )

    # ── Online recurrent inference (O(1) per step) ────────────────────────────

    def begin_episode(self, batch_size: int = 1):
        """
        Initialize recurrent state for online step-by-step inference.
        Call at episode start instead of collecting the full trajectory.
        """
        self._h = self.net.init_hidden(batch_size, self.device)
        self._online_probs: List[float] = []

    def step_online(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Process a single state step online (O(1) per step).
        Returns (prob, uncertainty) for this step.

        Use this for real-time hull checking without waiting for episode end.
        """
        s_t    = torch.tensor(state[np.newaxis], dtype=torch.float32,
                              device=self.device)   # (1, xDim)
        prob, self._h = self.net.recurrent_step(s_t, self._h)
        p = float(prob.item())
        self._online_probs.append(p)
        return p, 0.0   # uncertainty requires multiple passes; 0 for online mode

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_model_summary(self) -> dict:
        n_params = sum(p.numel() for p in self.net.parameters())
        return {
            "parameters":      n_params,
            "trained":         self._trained,
            "buffer_episodes": len(self.buffer),
            "pos_fraction":    self.buffer.positive_fraction(),
            "oracle_calls":    self._label_count,
            "total_episodes":  self._ep_count,
        }

    def get_training_metrics(self) -> dict:
        return self._last_metrics

    def save(self, path: str):
        torch.save({
            "net_state":  self.net.state_dict(),
            "target":     self.target,
            "is_valid":   self.is_valid_target,
            "ep_count":   self._ep_count,
            "metrics":    self._last_metrics,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.target          = ckpt.get("target")
        self.is_valid_target = ckpt.get("is_valid", False)
        self._ep_count       = ckpt.get("ep_count", 0)
        self._last_metrics   = ckpt.get("metrics", {})
        self._trained        = True
        self.net.eval()

    def full_reset(self):
        self.target = None
        self.is_valid_target = False
        self._trained = False
        self._ep_count = self._label_count = 0
        self.buffer = StateLabelBuffer()
        if self._oracle:
            self._oracle.full_reset()
        for m in self.net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    xDim          = 4
    uDim          = 2
    action_bounds = ([-1.0] * uDim, [1.0] * uDim)
    T             = 80

    def make_record(ep_id=0):
        return EpisodeRecord(
            states  = np.random.randn(T + 1, xDim).astype(np.float32),
            actions = np.random.randn(T,     uDim).astype(np.float32),
            rewards = np.random.randn(T).astype(np.float32),
            ep_id   = ep_id,
        )

    print("Building MambaHullNet...")
    net = MambaHullNet(xDim=xDim, d_model=64, d_state=16, num_layers=3)
    n_p = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_p:,}")

    print("\nBuilding MambaHullClassifier...")
    monitor = MambaHullClassifier(
        xDim          = xDim,
        action_bounds = action_bounds,
        label_every   = 3,
        train_every   = 2,
        d_model       = 64,
        d_state       = 16,
        num_layers     = 3,
        n_samples     = 10,
        min_buffer_before_train = 2,
    )

    print("\nRunning 20 episodes (batch mode)...")
    for i in range(20):
        t0  = time.time()
        rec = make_record(ep_id=i)
        r   = monitor.process(rec)
        dt  = (time.time() - t0) * 1000
        print(f"  ep={i:>3} | {r.source:<10} | "
              f"hit={str(r.hull_reached):<5} | "
              f"p̄={r.probs.mean():.3f} | "
              f"σ̄={r.uncertainty.mean():.3f} | "
              f"conf={r.confident_mask.mean():.2f} | "
              f"{dt:.1f}ms")

    print("\nModel summary:")
    for k, v in monitor.get_model_summary().items():
        print(f"  {k}: {v}")

    print("\nOnline recurrent inference (O(1) per step)...")
    rec = make_record(ep_id=99)
    monitor.begin_episode()
    for t in range(10):
        p, u = monitor.step_online(rec.states[t])
        print(f"  t={t} | p={p:.4f} | u={u:.4f}")
