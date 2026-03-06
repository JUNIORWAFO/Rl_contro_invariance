"""
actors_spring.py — Actor / Critic networks for the Spring-Mass-Damper system.

Extracted from spring_mass_dampener.py and aligned with the pipeline's
rl_algorithms.py conventions.

Networks
--------
  SpringMLPActor    — MLP that outputs a deterministic action (μ)
                      Wraps DQN backbone pattern from spring_mass_dampener.py
  SpringGaussianActor — Stochastic actor: outputs (μ, log σ) for PPO / SAC
  SpringRNNActor    — GRU actor ported from RNNActor in spring_mass_dampener.py
  SpringCritic      — MLP value head V(s) for PPO
  SpringQCritic     — MLP Q-function Q(s,a) for DDPG / SAC / TD3

All action-bound clamping uses registered buffers so tensors move
automatically with .to(device).

Factory
-------
  make_spring_actor(kind, obs_dim, act_dim, hidden, act_bound)
  make_spring_critic(kind, obs_dim, act_dim, hidden)
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: next power-of-two hidden size (from DQN in spring_mass_dampener.py)
# ══════════════════════════════════════════════════════════════════════════════

def _p2(h: int) -> int:
    """Round h up to the next power of 2, minimum 8."""
    return int(2 ** max(int(np.floor(np.log2(max(h, 8)))), 3))


# ══════════════════════════════════════════════════════════════════════════════
#  1. DETERMINISTIC MLP ACTOR
# ══════════════════════════════════════════════════════════════════════════════

class SpringMLPActor(nn.Module):
    """
    Deterministic MLP actor.  Direct port of the DQN backbone in
    spring_mass_dampener.py, repurposed as a continuous-action actor.

    Forward: state (B, obs_dim) → action (B, act_dim), clamped to act_bound.
    """

    def __init__(
        self,
        obs_dim:   int,
        act_dim:   int,
        hidden:    int   = 64,
        act_bound: float = 5.0,
    ):
        super().__init__()
        h = _p2(hidden)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, h // 4),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 4, h // 8),   nn.ReLU(),
            nn.Linear(h // 8, act_dim),
        )
        self.register_buffer("act_low",  torch.full((act_dim,), -act_bound))
        self.register_buffer("act_high", torch.full((act_dim,),  act_bound))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(state), self.act_low, self.act_high)


# ══════════════════════════════════════════════════════════════════════════════
#  2. STOCHASTIC GAUSSIAN ACTOR  (PPO / SAC)
# ══════════════════════════════════════════════════════════════════════════════

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class SpringGaussianActor(nn.Module):
    """
    Stochastic actor: outputs μ and log σ for a diagonal Gaussian policy.

    Used by PPO (via log_prob) and SAC (via reparameterisation).

    Forward returns:
        mu        (B, act_dim)
        log_std   (B, act_dim)  — clamped to [LOG_STD_MIN, LOG_STD_MAX]
    """

    def __init__(
        self,
        obs_dim:   int,
        act_dim:   int,
        hidden:    int   = 64,
        act_bound: float = 5.0,
    ):
        super().__init__()
        h = _p2(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h),     nn.ReLU(),
            nn.Linear(h, h // 2),     nn.ReLU(),
            nn.Linear(h // 2, h // 4), nn.ReLU(),
        )
        self.mu_head      = nn.Linear(h // 4, act_dim)
        self.log_std_head = nn.Linear(h // 4, act_dim)

        self.register_buffer("act_low",  torch.full((act_dim,), -act_bound))
        self.register_buffer("act_high", torch.full((act_dim,),  act_bound))
        self.act_bound = act_bound

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat    = self.trunk(state)
        mu      = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action, compute log_prob and entropy.

        Returns
        -------
        action   : (B, act_dim) — squashed & scaled
        log_prob : (B,)
        entropy  : (B,)
        """
        mu, log_std = self(state)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        raw  = dist.rsample()                                   # reparameterised

        # tanh squash then scale to act_bound
        squashed  = torch.tanh(raw)
        action    = squashed * self.act_bound
        action    = action.clamp(self.act_low, self.act_high)

        # log_prob with change-of-variables correction
        log_prob  = dist.log_prob(raw)
        log_prob -= torch.log(self.act_bound * (1 - squashed.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# ══════════════════════════════════════════════════════════════════════════════
#  3. GRU ACTOR  (direct port of RNNActor from spring_mass_dampener.py)
# ══════════════════════════════════════════════════════════════════════════════

class SpringRNNActor(nn.Module):
    """
    GRU-based actor — port of RNNActor from spring_mass_dampener.py.

    Interface
    ---------
    forward(state, hidden) → (action, new_hidden)
        state  : (1, B, obs_dim)
        hidden : (num_layers, B, hidden_size)
    init_hidden(batch_size) → zeros on same device as parameters
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_size: int,
        act_dim:    int,
        num_layers: int   = 1,
        act_bound:  float = 5.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.register_buffer("act_low",  torch.full((act_dim,), -act_bound))
        self.register_buffer("act_high", torch.full((act_dim,),  act_bound))

        self.rnn = nn.GRU(obs_dim, hidden_size, num_layers, batch_first=False)
        self.fc0 = nn.Linear(hidden_size * num_layers, hidden_size)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_size,     hidden_size),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size,     2 * act_dim),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2 * act_dim,     act_dim),
        )

    def forward(
        self, state: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.rnn(state, hidden)
        try:
            out = self.fc(out.permute(1, 0, 2).flatten(start_dim=1))
        except RuntimeError:
            out = self.fc(out[-1, :])
        return torch.clamp(out, self.act_low, self.act_high), hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=self.act_low.device,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  4. VALUE CRITIC V(s)  — used by PPO
# ══════════════════════════════════════════════════════════════════════════════

class SpringCritic(nn.Module):
    """
    MLP value function V(s) for PPO.

    Forward: state (B, obs_dim) → value (B, 1)
    """

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        h = _p2(hidden)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h),      nn.ReLU(),
            nn.Linear(h, h // 2),       nn.ReLU(),
            nn.Linear(h // 2, h // 4),  nn.ReLU(),
            nn.Linear(h // 4, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Q-CRITIC Q(s,a)  — used by DDPG / SAC / TD3
#     (port of Critic from spring_mass_dampener.py, simplified)
# ══════════════════════════════════════════════════════════════════════════════

class SpringQCritic(nn.Module):
    """
    MLP Q-function Q(s, a) for off-policy algorithms.

    Port of the Critic class in spring_mass_dampener.py (hidden_size=0 path,
    no actor hidden ingestion — keeps it self-contained).

    Forward: state (B, obs_dim), action (B, act_dim) → Q-value (B, 1)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        h = _p2(hidden)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h),  nn.ReLU(),
            nn.Linear(h, h // 2),              nn.ReLU(),
            nn.Linear(h // 2, h // 4),         nn.ReLU(),
            nn.Linear(h // 4, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        if action.ndim == 3 and action.shape[1] == 1:
            action = action.squeeze(1)
        return self.net(torch.cat([state, action], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def make_spring_actor(
    kind:      Literal["mlp", "gaussian", "rnn"] = "gaussian",
    obs_dim:   int   = 2,
    act_dim:   int   = 1,
    hidden:    int   = 64,
    act_bound: float = 5.0,
) -> nn.Module:
    """
    Factory for spring actor networks.

    kind='mlp'      → SpringMLPActor        (deterministic, for DDPG/TD3)
    kind='gaussian' → SpringGaussianActor   (stochastic,   for PPO/SAC)
    kind='rnn'      → SpringRNNActor        (recurrent,    for partial obs)
    """
    if kind == "mlp":
        return SpringMLPActor(obs_dim, act_dim, hidden, act_bound)
    elif kind == "gaussian":
        return SpringGaussianActor(obs_dim, act_dim, hidden, act_bound)
    elif kind == "rnn":
        return SpringRNNActor(obs_dim, hidden, act_dim, act_bound=act_bound)
    else:
        raise ValueError(f"Unknown actor kind '{kind}'. Choose mlp | gaussian | rnn.")


def make_spring_critic(
    kind:    Literal["value", "q"] = "value",
    obs_dim: int  = 2,
    act_dim: int  = 1,
    hidden:  int  = 64,
) -> nn.Module:
    """
    Factory for spring critic networks.

    kind='value' → SpringCritic   V(s)    (for PPO)
    kind='q'     → SpringQCritic  Q(s,a)  (for DDPG / SAC / TD3)
    """
    if kind == "value":
        return SpringCritic(obs_dim, hidden)
    elif kind == "q":
        return SpringQCritic(obs_dim, act_dim, hidden)
    else:
        raise ValueError(f"Unknown critic kind '{kind}'. Choose value | q.")


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, obs_dim, act_dim = 4, 2, 1

    s = torch.randn(B, obs_dim)
    a = torch.randn(B, act_dim)

    print("── MLP actor ──")
    mlp = make_spring_actor("mlp", obs_dim, act_dim)
    print(f"  out shape: {mlp(s).shape}")

    print("── Gaussian actor ──")
    gauss = make_spring_actor("gaussian", obs_dim, act_dim)
    act, lp, ent = gauss.get_action(s)
    print(f"  action={act.shape}  log_prob={lp.shape}  entropy={ent.shape}")

    print("── RNN actor ──")
    rnn = make_spring_actor("rnn", obs_dim, act_dim)
    h   = rnn.init_hidden(B)
    out, h2 = rnn(s.unsqueeze(0), h)
    print(f"  out={out.shape}  hidden={h2.shape}")

    print("── Value critic ──")
    vc = make_spring_critic("value", obs_dim)
    print(f"  V(s)={vc(s).shape}")

    print("── Q critic ──")
    qc = make_spring_critic("q", obs_dim, act_dim)
    print(f"  Q(s,a)={qc(s, a).shape}")
