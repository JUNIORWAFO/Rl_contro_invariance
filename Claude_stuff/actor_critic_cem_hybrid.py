"""
actor_critic_cem_hybrid.py
===========================
Sample-efficient actor-critic suite for LinearUpdateEnv.

Algorithms:
  - DDPG  — deterministic baseline, simple actor-critic
  - TD3   — twin critics + delayed policy updates + target smoothing
  - SAC   — max-entropy stochastic, automatic temperature
  - CEM   — linear gain search (u = Kx), compact and interpretable
  - SACxCEM — hybrid: SAC actor seeds CEM population every N episodes,
              CEM distills best linear gains back into SAC replay buffer

All agents share:
  - Same ReplayBuffer with HullMonitor relabeling support
  - Identical .select_action() / .push() / .update() interface
  - Compatible with unified train() loop

The key sample-efficiency ideas per agent:
  DDPG  → deterministic policy, every sample fully used
  TD3   → clipped double-Q + noise smoothing, fewer overestimation artifacts
  SAC   → max entropy drives broad exploration, auto-α avoids manual tuning
  CEM   → only xDim*uDim+uDim params, converges in far fewer episodes
  Hybrid→ SAC warm-starts CEM (no cold-start), CEM gives SAC dense reward signal
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# ── Shared types ──────────────────────────────────────────────────────────────
try:
    from hull_monitors import EpisodeRecord, HullResult, PostEpisodeHullMonitor
except ImportError:
    @dataclass
    class EpisodeRecord:
        states: np.ndarray; actions: np.ndarray
        rewards: np.ndarray; ep_id: int = 0
        @property
        def T(self): return len(self.actions)

    @dataclass
    class HullResult:
        in_hull_mask: np.ndarray; first_hull_step: Optional[int]
        is_valid_target: bool; source: str = "unknown"
        @property
        def hull_reached(self): return self.first_hull_step is not None
        def relabel_rewards(self, r, hr):
            out = r.copy(); out[self.in_hull_mask] = hr; return out


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


class ReplayBuffer:
    """
    Unified replay buffer for all actor-critic agents.

    Features:
      - O(1) push/sample
      - Episode boundary tracking for HullMonitor relabeling
      - Priority flag per transition (CEM-generated transitions marked high-priority)
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: torch.device):
        self.capacity  = capacity
        self.device    = device
        self.ptr = self.size = 0

        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts      = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews      = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones     = np.zeros((capacity, 1),        dtype=np.float32)
        self.priority  = np.zeros((capacity, 1),        dtype=np.float32)

        self._ep_start    = 0
        self._ep_ends:    List[int] = []

    def push(self, obs, act, rew, next_obs, done, priority: float = 1.0):
        self.obs[self.ptr]      = obs
        self.acts[self.ptr]     = act
        self.rews[self.ptr]     = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = done
        self.priority[self.ptr] = priority

        if done:
            self._ep_ends.append(self.ptr)
            self._ep_start = (self.ptr + 1) % self.capacity
            if len(self._ep_ends) > 10_000:
                self._ep_ends.pop(0)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def relabel_last_episode(self, result: HullResult, hull_reward: float):
        """Overwrite rewards for last episode using HullResult mask."""
        if not self._ep_ends:
            return
        ep_end   = self._ep_ends[-1]
        ep_start = self._ep_start
        ep_len   = (ep_end - ep_start) % self.capacity + 1
        mask     = result.in_hull_mask[:ep_len]
        for i, hit in enumerate(mask):
            if hit:
                idx = (ep_start + i) % self.capacity
                self.rews[idx] = hull_reward

    def sample(self, batch_size: int, prioritized: bool = False):
        """Sample a batch. If prioritized, oversample high-priority transitions."""
        if prioritized and self.size > batch_size:
            p     = self.priority[:self.size].flatten()
            p     = p / p.sum()
            idx   = np.random.choice(self.size, size=batch_size,
                                     replace=False, p=p)
        else:
            idx   = np.random.randint(0, self.size, size=batch_size)

        def t(arr): return torch.tensor(arr[idx], device=self.device)
        return t(self.obs), t(self.acts), t(self.rews), \
               t(self.next_obs), t(self.dones)

    def __len__(self): return self.size


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256,
                 n_layers: int = 2, output_activation=None):
        super().__init__()
        dims   = [in_dim] + [hidden] * n_layers + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DDPG — Deterministic Policy Gradient
# ══════════════════════════════════════════════════════════════════════════════

class DDPGActor(nn.Module):
    """Deterministic actor: μ(s) → action, squashed to action bounds."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 act_scale: float, act_bias: torch.Tensor):
        super().__init__()
        self.net       = MLP(obs_dim, act_dim, hidden,
                             output_activation=nn.Tanh())
        self.act_scale = act_scale
        self.register_buffer("act_bias", act_bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.act_scale + self.act_bias


class DDPGCritic(nn.Module):
    """Q(s, a) → scalar."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class OUNoise:
    """
    Ornstein-Uhlenbeck noise for temporally correlated exploration.
    Better than Gaussian for continuous control — smoother action sequences.
    """

    def __init__(self, act_dim: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2):
        self.mu    = mu * np.ones(act_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = (self.theta * (self.mu - self.state)
              + self.sigma * np.random.randn(*self.state.shape))
        self.state += dx
        return self.state.copy()


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient.

    Architecture: deterministic actor μ(s), single Q-critic.
    Exploration: OU noise added to actions during training.
    Target networks: soft-updated (Polyak averaging).

    Sample efficiency notes:
      - Off-policy replay → reuses every transition
      - Deterministic policy → no variance from stochastic sampling
      - Weakness: single critic → Q overestimation, corrected in TD3
    """

    def __init__(
        self,
        obs_space,
        act_space,
        hidden:       int   = 256,
        lr_actor:     float = 1e-4,
        lr_critic:    float = 1e-3,
        gamma:        float = 0.99,
        tau:          float = 0.005,
        buffer_size:  int   = 1_000_000,
        batch_size:   int   = 256,
        warmup_steps: int   = 1000,
        noise_sigma:  float = 0.2,
        noise_decay:  float = 0.9995,
        noise_min:    float = 0.01,
        device:       str   = "cpu",
    ):
        self.device      = torch.device(device)
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.warmup      = warmup_steps

        obs_dim   = obs_space.shape[0]
        act_dim   = act_space.shape[0]
        act_scale = float((act_space.high - act_space.low).max() / 2.0)
        act_bias  = torch.tensor(
            (act_space.high + act_space.low) / 2.0,
            dtype=torch.float32, device=self.device)

        self.actor          = DDPGActor(obs_dim, act_dim, hidden,
                                        act_scale, act_bias).to(self.device)
        self.actor_target   = copy.deepcopy(self.actor)
        self.critic         = DDPGCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target  = copy.deepcopy(self.critic)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer   = ReplayBuffer(buffer_size, obs_dim, act_dim, self.device)
        self.noise    = OUNoise(act_dim, sigma=noise_sigma)
        self._noise_sigma  = noise_sigma
        self._noise_decay  = noise_decay
        self._noise_min    = noise_min
        self.act_low  = act_space.low
        self.act_high = act_space.high

        self._steps   = 0
        self.metrics: Dict[str, float] = {}
        self.name     = "DDPG"

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            x   = torch.tensor(obs, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
            act = self.actor(x).squeeze(0).cpu().numpy()
        if explore and self._steps < self.warmup:
            return np.random.uniform(self.act_low, self.act_high)
        if explore:
            act = act + self._noise_sigma * self.noise.sample()
        return np.clip(act, self.act_low, self.act_high)

    def push(self, obs, act, rew, next_obs, done):
        self.buffer.push(obs, act, rew, next_obs, done)
        self._steps += 1
        if done:
            self.noise.reset()
            self._noise_sigma = max(
                self._noise_sigma * self._noise_decay, self._noise_min)

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size or self._steps < self.warmup:
            return {}

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_acts    = self.actor_target(next_obs)
            target_q     = rews + (1 - dones) * self.gamma * \
                           self.critic_target(next_obs, next_acts)

        critic_loss = F.mse_loss(self.critic(obs, acts), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        soft_update(self.actor_target,  self.actor,  self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        self.metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "noise_sigma": self._noise_sigma,
        }
        return self.metrics


# ══════════════════════════════════════════════════════════════════════════════
#  2. TD3 — Twin Delayed Deep Deterministic Policy Gradient
# ══════════════════════════════════════════════════════════════════════════════

class TD3Critic(nn.Module):
    """Twin Q-networks — take min to suppress overestimation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, act):
        return self.q1(torch.cat([obs, act], dim=-1))


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient.

    Three improvements over DDPG for sample efficiency:
      1. Clipped double-Q: min(Q1, Q2) eliminates overestimation → more
         accurate value estimates → fewer wasted samples on bad trajectories
      2. Delayed actor update: actor updated every d critic steps →
         critic is accurate before actor moves, preventing compounding errors
      3. Target policy smoothing: noise added to target actions → critic
         learns smoother Q landscape → more stable gradient signal

    These combine to make TD3 significantly more sample-efficient than DDPG.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        hidden:              int   = 256,
        lr:                  float = 3e-4,
        gamma:               float = 0.99,
        tau:                 float = 0.005,
        buffer_size:         int   = 1_000_000,
        batch_size:          int   = 256,
        warmup_steps:        int   = 1000,
        exploration_noise:   float = 0.1,
        target_noise:        float = 0.2,   # smoothing noise on target
        target_noise_clip:   float = 0.5,
        policy_delay:        int   = 2,     # actor update every d critic steps
        device:              str   = "cpu",
    ):
        self.device     = torch.device(device)
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.warmup     = warmup_steps
        self.d          = policy_delay

        obs_dim   = obs_space.shape[0]
        act_dim   = act_space.shape[0]
        act_scale = float((act_space.high - act_space.low).max() / 2.0)
        act_bias  = torch.tensor(
            (act_space.high + act_space.low) / 2.0,
            dtype=torch.float32, device=self.device)

        self.actor         = DDPGActor(obs_dim, act_dim, hidden,
                                       act_scale, act_bias).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = TD3Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer   = ReplayBuffer(buffer_size, obs_dim, act_dim, self.device)
        self.act_low  = torch.tensor(act_space.low,  device=self.device)
        self.act_high = torch.tensor(act_space.high, device=self.device)

        self._expl_noise    = exploration_noise * act_scale
        self._target_noise  = target_noise * act_scale
        self._noise_clip    = target_noise_clip * act_scale

        self._steps    = 0
        self._updates  = 0
        self.metrics: Dict[str, float] = {}
        self.name     = "TD3"

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if explore and self._steps < self.warmup:
            return np.random.uniform(
                self.act_low.cpu().numpy(),
                self.act_high.cpu().numpy())
        with torch.no_grad():
            x   = torch.tensor(obs, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
            act = self.actor(x).squeeze(0)
        if explore:
            noise = torch.randn_like(act) * self._expl_noise
            act   = (act + noise).clamp(self.act_low, self.act_high)
        return act.cpu().numpy()

    def push(self, obs, act, rew, next_obs, done):
        self.buffer.push(obs, act, rew, next_obs, done)
        self._steps += 1

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size or self._steps < self.warmup:
            return {}

        self._updates += 1
        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)

        # ── Critic update (every step) ────────────────────────────────────────
        with torch.no_grad():
            noise       = (torch.randn_like(acts) * self._target_noise
                           ).clamp(-self._noise_clip, self._noise_clip)
            next_acts   = (self.actor_target(next_obs) + noise
                           ).clamp(self.act_low, self.act_high)
            q1_t, q2_t  = self.critic_target(next_obs, next_acts)
            target_q    = rews + (1 - dones) * self.gamma * torch.min(q1_t, q2_t)

        q1, q2      = self.critic(obs, acts)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Delayed actor update ──────────────────────────────────────────────
        actor_loss = torch.tensor(0.0)
        if self._updates % self.d == 0:
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            soft_update(self.actor_target,  self.actor,  self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        self.metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }
        return self.metrics


# ══════════════════════════════════════════════════════════════════════════════
#  3. SAC — Soft Actor-Critic
# ══════════════════════════════════════════════════════════════════════════════

class SACActor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.shared  = MLP(obs_dim, hidden, hidden, activate_last=True) \
            if hasattr(MLP, 'activate_last') else \
            nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden),  nn.ReLU())
        self.mu_head = nn.Linear(hidden, act_dim)
        self.ls_head = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h      = self.shared(obs)
        mu     = self.mu_head(h)
        log_std = self.ls_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std         = log_std.exp()
        u           = mu + std * torch.randn_like(mu)
        action      = torch.tanh(u)
        log_prob    = Normal(mu, std).log_prob(u).sum(-1, keepdim=True)
        log_prob   -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, obs, act):
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)


class SACAgent:
    """
    Soft Actor-Critic.

    Sample efficiency advantages:
      - Off-policy replay: every transition is reused many times
      - Maximum entropy: explicitly encourages diverse exploration,
        preventing collapse to a narrow region of the hull space
      - Auto-α: entropy temperature self-tunes, no manual sweep needed
      - Twin critics + reparameterization: low-variance, stable gradient

    Extra: extract_linear_gain() linearizes the actor around a state for
    CEM warm-starting — the key coupling point in the hybrid.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        hidden:        int   = 256,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        tau:           float = 0.005,
        alpha_init:    float = 0.2,
        auto_alpha:    bool  = True,
        buffer_size:   int   = 1_000_000,
        batch_size:    int   = 256,
        warmup_steps:  int   = 1000,
        device:        str   = "cpu",
    ):
        self.device     = torch.device(device)
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.warmup     = warmup_steps
        self.auto_alpha = auto_alpha

        obs_dim  = obs_space.shape[0]
        act_dim  = act_space.shape[0]

        self.act_scale = float((act_space.high - act_space.low).max() / 2.0)
        self.act_bias  = torch.tensor(
            (act_space.high + act_space.low) / 2.0,
            dtype=torch.float32, device=self.device)
        self.act_low   = act_space.low
        self.act_high  = act_space.high

        self.actor          = SACActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic         = SACCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target  = copy.deepcopy(self.critic)
        hard_update(self.critic_target, self.critic)

        self.actor_opt  = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        if auto_alpha:
            self.target_entropy = float(-act_dim)
            self.log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32,
                device=self.device, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        self.buffer  = ReplayBuffer(buffer_size, obs_dim, act_dim, self.device)
        self._steps  = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.metrics: Dict[str, float] = {}
        self.name    = "SAC"

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if self._steps < self.warmup:
            return np.random.uniform(self.act_low, self.act_high)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            if explore:
                act, _ = self.actor.sample(x)
            else:
                mu, _  = self.actor(x)
                act    = torch.tanh(mu)
        return np.clip(
            (act * self.act_scale + self.act_bias).squeeze().cpu().numpy(),
            self.act_low, self.act_high)

    def push(self, obs, act, rew, next_obs, done, priority: float = 1.0):
        self.buffer.push(obs, act, rew, next_obs, done, priority)
        self._steps += 1

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size or self._steps < self.warmup:
            return {}

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)

        # ── Critic ────────────────────────────────────────────────────────────
        with torch.no_grad():
            na, nlp  = self.actor.sample(next_obs)
            nq       = self.critic_target.min_q(next_obs, na)
            tq       = rews + (1 - dones) * self.gamma * (nq - self.alpha * nlp)

        q1, q2      = self.critic(obs, acts)
        critic_loss = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor ─────────────────────────────────────────────────────────────
        new_act, log_pi = self.actor.sample(obs)
        actor_loss = (self.alpha * log_pi
                      - self.critic.min_q(obs, new_act)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Alpha ─────────────────────────────────────────────────────────────
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        soft_update(self.critic_target, self.critic, self.tau)

        self.metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
        }
        return self.metrics

    # ── CEM coupling ──────────────────────────────────────────────────────────

    def extract_linear_gain(
        self,
        anchor_states: Optional[np.ndarray] = None,
        n_points:      int = 64,
    ) -> np.ndarray:
        """
        Linearize the SAC actor to extract a linear gain K such that u ≈ Kx + b.

        Method: fit least-squares K, b to (state → mean action) pairs sampled
        from the actor around anchor_states (or random if None).

        Returns flat parameter vector [K.flatten(), b] for CEM warm-starting.
        """
        self.actor.eval()
        with torch.no_grad():
            if anchor_states is not None:
                n = min(n_points, len(anchor_states))
                idx = np.random.choice(len(anchor_states), n, replace=False)
                s   = torch.tensor(anchor_states[idx], dtype=torch.float32,
                                   device=self.device)
            else:
                s = torch.randn(n_points, self.obs_dim, device=self.device)

            mu, _ = self.actor(s)
            u     = (torch.tanh(mu) * self.act_scale + self.act_bias)

        S = s.cpu().numpy()    # (N, obs_dim)
        U = u.cpu().numpy()    # (N, act_dim)

        # Least-squares: [S | 1] @ [K^T; b^T] = U
        S_aug = np.hstack([S, np.ones((len(S), 1))])
        params, _, _, _ = np.linalg.lstsq(S_aug, U, rcond=None)
        K = params[:-1].T    # (act_dim, obs_dim)
        b = params[-1]       # (act_dim,)

        return np.concatenate([K.flatten(), b]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  4. CEM with Linear Gain — compact, interpretable, seeded by SAC
# ══════════════════════════════════════════════════════════════════════════════

class LinearCEM:
    """
    Cross-Entropy Method searching over linear gain K in u = Kx + b.

    Parameter space: xDim * uDim + uDim  (much smaller than neural net weights)
    This makes CEM converge in far fewer episodes than searching over network weights.

    The Gaussian distribution over [K.flatten(), b] is updated each generation
    by fitting to the elite trajectories.

    Key design for sample efficiency:
      - Population evaluated in independent env copies (parallelizable)
      - Warm-start from SAC actor via extract_linear_gain() → no cold start
      - All CEM trajectories pushed into SAC replay buffer with high priority →
        SAC learns from the best CEM trajectories immediately
    """

    def __init__(
        self,
        obs_space,
        act_space,
        population_size:  int   = 50,
        elite_frac:       float = 0.2,
        extra_std:        float = 0.05,   # extra noise to prevent collapse
        std_min:          float = 0.01,
        std_decay:        float = 0.98,
        n_eval_steps:     int   = 200,
        gamma:            float = 0.99,
    ):
        self.obs_dim   = obs_space.shape[0]
        self.act_dim   = act_space.shape[0]
        self.act_low   = act_space.low
        self.act_high  = act_space.high

        self.N          = population_size
        self.n_elite    = max(2, int(population_size * elite_frac))
        self.extra_std  = extra_std
        self.std_min    = std_min
        self.std_decay  = std_decay
        self.n_eval     = n_eval_steps
        self.gamma      = gamma

        self.param_dim  = self.obs_dim * self.act_dim + self.act_dim
        self.theta_mean = np.zeros(self.param_dim, dtype=np.float32)
        self.theta_std  = np.ones(self.param_dim,  dtype=np.float32) * 0.5

        self._best_theta = self.theta_mean.copy()
        self._best_score = -np.inf
        self.generation  = 0
        self.metrics: Dict[str, float] = {}
        self.name        = "CEM"

    def _unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K = theta[:self.obs_dim * self.act_dim].reshape(self.act_dim, self.obs_dim)
        b = theta[self.obs_dim * self.act_dim:]
        return K, b

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        K, b = self._unpack(self._best_theta)
        return np.clip(K @ obs + b, self.act_low, self.act_high)

    def warm_start(self, theta_init: np.ndarray, std: float = 0.1):
        """
        Seed the CEM distribution from an external policy (e.g. SAC actor).
        Sets the mean and resets std to allow refinement around the seed.
        """
        self.theta_mean = theta_init.copy()
        self.theta_std  = np.ones_like(theta_init) * std
        print(f"[CEM] Warm-started from external policy "
              f"(param_norm={np.linalg.norm(theta_init):.3f})")

    def evolve(
        self,
        env_factory,
        monitor         = None,
        hull_reward:    float = 100.0,
        sac_buffer:     Optional[ReplayBuffer] = None,
        cem_priority:   float = 2.0,    # priority boost for CEM transitions in SAC buffer
    ) -> Tuple[dict, List[EpisodeRecord]]:
        """
        One CEM generation.

        If sac_buffer is provided, all evaluated transitions are pushed
        into it with high priority so SAC learns from CEM's exploration.
        """
        population = (self.theta_mean
                      + np.random.randn(self.N, self.param_dim).astype(np.float32)
                      * self.theta_std)

        scores     = np.zeros(self.N, dtype=np.float32)
        records    = []

        for i, theta in enumerate(population):
            K, b    = self._unpack(theta)
            env     = env_factory()
            obs, _  = env.reset()
            ep_states, ep_acts, ep_rews = [obs.copy()], [], []

            for _ in range(self.n_eval):
                act = np.clip(K @ obs + b, self.act_low, self.act_high)
                next_obs, rew, term, trunc, _ = env.step(act)
                ep_states.append(next_obs.copy())
                ep_acts.append(act.copy())
                ep_rews.append(rew)
                if term or trunc:
                    break
                obs = next_obs

            rec = EpisodeRecord(
                states  = np.array(ep_states, dtype=np.float32),
                actions = np.array(ep_acts,   dtype=np.float32),
                rewards = np.array(ep_rews,   dtype=np.float32),
                ep_id   = self.generation * self.N + i,
            )

            # Hull relabeling
            if monitor is not None:
                result   = monitor.process(rec)
                ep_rews_r = result.relabel_rewards(
                    np.array(ep_rews), hull_reward)
            else:
                ep_rews_r = np.array(ep_rews)

            # Push into SAC buffer with high priority
            if sac_buffer is not None:
                for t in range(len(ep_acts)):
                    sac_buffer.push(
                        ep_states[t], ep_acts[t], ep_rews_r[t],
                        ep_states[t + 1],
                        float(t == len(ep_acts) - 1),
                        priority=cem_priority,
                    )

            # Discounted return
            G = float(sum(r * self.gamma**t
                          for t, r in enumerate(ep_rews_r)))
            scores[i] = G
            records.append(rec)

        # ── Elite update ──────────────────────────────────────────────────────
        elite_idx = np.argsort(scores)[-self.n_elite:]
        elites    = population[elite_idx]

        if scores[elite_idx[-1]] > self._best_score:
            self._best_score = scores[elite_idx[-1]]
            self._best_theta = population[elite_idx[-1]].copy()

        self.theta_mean = elites.mean(axis=0)
        self.theta_std  = np.maximum(
            elites.std(axis=0) + self.extra_std, self.std_min)
        self.theta_std  = np.maximum(self.theta_std * self.std_decay, self.std_min)

        self.generation += 1
        self.metrics = {
            "best_score":  float(self._best_score),
            "mean_score":  float(scores.mean()),
            "elite_score": float(scores[elite_idx].mean()),
            "std_mean":    float(self.theta_std.mean()),
            "generation":  self.generation,
        }
        return self.metrics, records

    def get_best_gain(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the best (K, b) found so far."""
        return self._unpack(self._best_theta)


# ══════════════════════════════════════════════════════════════════════════════
#  5. SAC × CEM HYBRID
#     SAC seeds CEM → CEM fills SAC buffer → tight feedback loop
# ══════════════════════════════════════════════════════════════════════════════

class SACxCEM:
    """
    SAC × CEM hybrid for maximum sample efficiency.

    Interaction protocol:
      Phase A — SAC explores (n_sac_steps steps):
        • Standard SAC training collecting diverse transitions
        • Builds up a Q landscape that spans the hull space

      Phase B — CEM exploits (every cem_interval episodes):
        • SAC actor is linearized → K, b via extract_linear_gain()
        • CEM is warm-started at this K, b with small std
        • CEM runs n_cem_gens generations, pushing transitions into SAC buffer
          with high priority (cem_priority > 1.0)
        • Best CEM gain K* is distilled back into SAC replay

      This loop means:
        • SAC never gets stuck (CEM injects diverse elite trajectories)
        • CEM never starts cold (SAC provides a good initialization)
        • Hull reward propagates in both directions simultaneously

    Single .select_action() always uses SAC (stochastic for exploration).
    .select_action_exploit() uses best CEM gain (deterministic, best known).
    """

    def __init__(
        self,
        obs_space,
        act_space,
        # SAC params
        sac_hidden:          int   = 256,
        sac_lr:              float = 3e-4,
        sac_gamma:           float = 0.99,
        sac_tau:             float = 0.005,
        sac_buffer_size:     int   = 1_000_000,
        sac_batch_size:      int   = 256,
        sac_warmup:          int   = 1000,
        auto_alpha:          bool  = True,
        # CEM params
        cem_population:      int   = 50,
        cem_elite_frac:      float = 0.2,
        cem_n_eval_steps:    int   = 200,
        cem_priority:        float = 2.0,
        # Hybrid coupling
        cem_interval:        int   = 20,     # run CEM every N SAC episodes
        n_cem_gens:          int   = 5,      # CEM generations per coupling
        warm_start_std:      float = 0.1,    # std when seeding CEM from SAC
        n_anchor_states:     int   = 128,    # states used to linearize SAC
        device:              str   = "cpu",
    ):
        self.obs_space    = obs_space
        self.act_space    = act_space
        self.device       = torch.device(device)
        self.cem_interval = cem_interval
        self.n_cem_gens   = n_cem_gens
        self.warm_start_std = warm_start_std
        self.n_anchor     = n_anchor_states
        self.cem_priority = cem_priority

        self.sac = SACAgent(
            obs_space, act_space,
            hidden=sac_hidden, lr=sac_lr, gamma=sac_gamma, tau=sac_tau,
            buffer_size=sac_buffer_size, batch_size=sac_batch_size,
            warmup_steps=sac_warmup, auto_alpha=auto_alpha, device=device,
        )
        self.cem = LinearCEM(
            obs_space, act_space,
            population_size=cem_population, elite_frac=cem_elite_frac,
            n_eval_steps=cem_n_eval_steps,
        )

        self._ep_count      = 0
        self._anchor_states: List[np.ndarray] = []
        self.metrics: Dict[str, float] = {}
        self.name = "SACxCEM"

    # ── Standard interface ────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Stochastic SAC action — used during rollout collection."""
        return self.sac.select_action(obs, explore)

    def select_action_exploit(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic best CEM gain — used for evaluation."""
        return self.cem.select_action(obs)

    def push(self, obs, act, rew, next_obs, done):
        self.sac.push(obs, act, rew, next_obs, done)
        self._anchor_states.append(obs.copy())
        if len(self._anchor_states) > self.n_anchor * 4:
            self._anchor_states = self._anchor_states[-self.n_anchor * 4:]
        if done:
            self._ep_count += 1

    def update(self) -> dict:
        """SAC gradient update — called every step."""
        return self.sac.update()

    def relabel_last_episode(self, result: HullResult, hull_reward: float):
        """Hull relabeling delegated to SAC buffer."""
        self.sac.buffer.relabel_last_episode(result, hull_reward)

    # ── Hybrid coupling ───────────────────────────────────────────────────────

    def maybe_run_cem(
        self,
        env_factory,
        monitor         = None,
        hull_reward:    float = 100.0,
    ) -> bool:
        """
        Called after each SAC episode. Triggers CEM coupling every cem_interval episodes.
        Returns True if CEM was run this call.
        """
        if (self._ep_count == 0
                or self._ep_count % self.cem_interval != 0
                or self.sac._steps < self.sac.warmup):
            return False

        print(f"\n[SACxCEM] Episode {self._ep_count} — "
              f"coupling SAC → CEM (gen {self.cem.generation})")

        # Step 1: linearize SAC actor → warm-start CEM
        anchor = (np.array(self._anchor_states[-self.n_anchor:])
                  if self._anchor_states else None)
        theta_init = self.sac.extract_linear_gain(anchor, self.n_anchor)
        self.cem.warm_start(theta_init, std=self.warm_start_std)

        # Step 2: run CEM, push transitions into SAC buffer
        for g in range(self.n_cem_gens):
            cem_metrics, _ = self.cem.evolve(
                env_factory     = env_factory,
                monitor         = monitor,
                hull_reward     = hull_reward,
                sac_buffer      = self.sac.buffer,
                cem_priority    = self.cem_priority,
            )

        # Step 3: run extra SAC updates to absorb CEM data
        n_extra = self.cem.N * self.n_cem_gens // 4
        for _ in range(n_extra):
            self.sac.update()

        K, b = self.cem.get_best_gain()
        self.metrics = {
            **cem_metrics,
            "sac_alpha": self.sac.alpha,
            "K_norm":    float(np.linalg.norm(K)),
        }
        print(f"[SACxCEM] CEM done | best={cem_metrics['best_score']:.2f} | "
              f"K_norm={np.linalg.norm(K):.3f}")
        return True

    @property
    def buffer(self):
        return self.sac.buffer

    def get_linear_gain(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current best linear gain (K, b) from CEM."""
        return self.cem.get_best_gain()

    def get_sac_metrics(self) -> dict:
        return self.sac.metrics

    def get_cem_metrics(self) -> dict:
        return self.cem.metrics


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(
    env,
    agent,                                # DDPG | TD3 | SAC | SACxCEM
    monitor             = None,
    total_steps:  int   = 200_000,
    hull_reward:  float = 100.0,
    log_interval: int   = 10,
    eval_every:   int   = 50,
    env_factory         = None,           # required for SACxCEM CEM coupling
    device:       str   = "cpu",
) -> dict:
    """
    Unified loop for all actor-critic agents.
    Returns training history dict.
    """
    obs, _    = env.reset()
    ep_states = [obs.copy()]
    ep_acts   = []
    ep_rews   = []
    ep_count  = 0
    step      = 0
    history   = {"returns": [], "hull_hits": [], "steps": []}

    print(f"\n{'━'*65}")
    print(f"  Training: {agent.name}  |  total_steps={total_steps:,}")
    print(f"{'━'*65}")
    print(f"  {'Step':>10} | {'Ep':>6} | {'Return':>10} | "
          f"{'Hull':>6} | {'Alpha/Noise':>12}")
    print(f"  {'─'*60}")

    while step < total_steps:
        act  = agent.select_action(obs, explore=True)
        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc

        ep_states.append(next_obs.copy())
        ep_acts.append(act.copy())
        ep_rews.append(rew)

        agent.push(obs, act, rew, next_obs, float(done))
        agent.update()

        obs   = next_obs
        step += 1

        if done:
            ep_count += 1
            record    = EpisodeRecord(
                states  = np.array(ep_states, dtype=np.float32),
                actions = np.array(ep_acts,   dtype=np.float32),
                rewards = np.array(ep_rews,   dtype=np.float32),
                ep_id   = ep_count,
            )

            hull_hit = False
            if monitor is not None:
                result   = monitor.process(record)
                hull_hit = result.hull_reached
                # Relabel buffer
                if hasattr(agent, "relabel_last_episode"):
                    agent.relabel_last_episode(result, hull_reward)
                elif hasattr(agent, "buffer"):
                    agent.buffer.relabel_last_episode(result, hull_reward)

            # SACxCEM coupling trigger
            if isinstance(agent, SACxCEM) and env_factory is not None:
                agent.maybe_run_cem(env_factory, monitor, hull_reward)

            ep_ret = float(np.sum(ep_rews))
            history["returns"].append(ep_ret)
            history["hull_hits"].append(hull_hit)
            history["steps"].append(step)

            if ep_count % log_interval == 0:
                avg_ret  = float(np.mean(history["returns"][-log_interval:]))
                hit_rate = float(np.mean(history["hull_hits"][-log_interval:]))
                extra    = (f"α={agent.metrics.get('alpha', 0):.3f}"
                            if "alpha" in agent.metrics
                            else f"σ={agent.metrics.get('noise_sigma', 0):.3f}"
                            if "noise_sigma" in agent.metrics
                            else "")
                print(f"  {step:>10} | {ep_count:>6} | {avg_ret:>10.2f} | "
                      f"{'✓' if hull_hit else '✗':>6} | {extra:>12} | "
                      f"hit_rate={hit_rate:.2f}")

            obs, _    = env.reset()
            ep_states = [obs.copy()]
            ep_acts   = []
            ep_rews   = []

    print(f"\n  Training complete — {ep_count} episodes, "
          f"hull hit rate: {np.mean(history['hull_hits']):.3f}")
    return history


def evaluate(env, agent, n_episodes: int = 20,
             hull_reward: float = 100.0, monitor=None) -> dict:
    """Deterministic evaluation of a trained agent."""
    returns, lengths, hull_hits = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_states, ep_acts, ep_rews = [obs.copy()], [], []
        done = False

        while not done:
            act  = agent.select_action(obs, explore=False)
            obs, rew, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_states.append(obs.copy()); ep_acts.append(act); ep_rews.append(rew)

        hit = False
        if monitor is not None:
            rec = EpisodeRecord(
                states  = np.array(ep_states, dtype=np.float32),
                actions = np.array(ep_acts,   dtype=np.float32),
                rewards = np.array(ep_rews,   dtype=np.float32),
            )
            hit = monitor.process(rec).hull_reached

        returns.append(sum(ep_rews))
        lengths.append(len(ep_rews))
        hull_hits.append(hit)

    return {
        "mean_return":   float(np.mean(returns)),
        "std_return":    float(np.std(returns)),
        "mean_length":   float(np.mean(lengths)),
        "hull_hit_rate": float(np.mean(hull_hits)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    class DummySpace:
        def __init__(self, low, high):
            self.low   = np.array(low,  dtype=np.float32)
            self.high  = np.array(high, dtype=np.float32)
            self.shape = self.low.shape

    obs_s = DummySpace([-5., -5.], [5., 5.])
    act_s = DummySpace([-1., -1.], [1., 1.])

    print("Constructing all agents...")
    agents = {
        "DDPG":    DDPGAgent(obs_s, act_s),
        "TD3":     TD3Agent(obs_s,  act_s),
        "SAC":     SACAgent(obs_s,  act_s),
        "CEM":     LinearCEM(obs_s, act_s),
        "SACxCEM": SACxCEM(obs_s,   act_s),
    }

    obs = np.random.randn(2).astype(np.float32)
    print(f"\nAction shapes:")
    for name, agent in agents.items():
        act = agent.select_action(obs)
        print(f"  {name:<10}: {act}")

    print(f"\nParameter / buffer counts:")
    for name, agent in agents.items():
        if hasattr(agent, "sac"):
            n = sum(p.numel() for p in agent.sac.actor.parameters())
            print(f"  {name:<10}: SAC actor params={n:,}, "
                  f"CEM params={agent.cem.param_dim}")
        elif hasattr(agent, "actor"):
            n = sum(p.numel() for p in agent.actor.parameters())
            print(f"  {name:<10}: actor params={n:,}")
        elif hasattr(agent, "param_dim"):
            print(f"  {name:<10}: linear params={agent.param_dim}")

    # Test SAC linearization
    print("\nTesting SAC → CEM warm-start...")
    sac = agents["SAC"]
    cem = agents["CEM"]
    theta = sac.extract_linear_gain(n_points=32)
    cem.warm_start(theta, std=0.1)
    K, b = cem.get_best_gain()
    print(f"  K shape: {K.shape}, b shape: {b.shape}")
    print(f"  K norm: {np.linalg.norm(K):.4f}")
    print("\nAll agents ready.")
