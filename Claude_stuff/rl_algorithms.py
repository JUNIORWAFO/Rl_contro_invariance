"""
rl_algorithms.py
=================
Four RL algorithms tailored for LinearUpdateEnv.

Each exploits a different aspect of the problem structure:

  1. ModelBasedRL   — Uses known A, B to do gradient-based policy optimization
                      directly through the dynamics. Zero model learning needed.
                      Most sample-efficient possible for this env.

  2. SAC            — Soft Actor-Critic with automatic entropy tuning.
                      Off-policy → reuses every transition, great for sparse rewards.
                      Pairs naturally with HER buffer for hull relabeling.

  3. CEM / ES       — Cross-Entropy Method + Evolution Strategies.
                      Gradient-free → robust to reward discontinuities (hull boundary).
                      Parallelizable, no neural net required.

  4. MPC-RL Hybrid  — Model Predictive Control uses A, B to plan H steps ahead.
                      RL value function corrects for finite horizon bias.
                      Best of both: planning precision + learned long-horizon value.

All algorithms share:
  - Same LinearUpdateEnv interface (step returns 5 values)
  - Compatible with any HullMonitor for reward relabeling
  - EpisodeRecord output for post-episode hull processing

Usage:
    env     = LinearUpdateEnv(A, B, ...)
    monitor = PostEpisodeHullMonitor(...)

    # Pick any algorithm:
    agent = ModelBasedRL(A, B, env.observation_space, env.action_space)
    agent = SACAgent(env.observation_space, env.action_space)
    agent = CEMAgent(env.observation_space, env.action_space)
    agent = MPCRLAgent(A, B, env.observation_space, env.action_space)

    # Unified training loop:
    train(env, agent, monitor, total_steps=500_000)
"""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

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
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging: target ← τ·source + (1-τ)·target"""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1.0 - tau) * t.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


class ReplayBuffer:
    """
    Off-policy replay buffer supporting:
      - Standard (s, a, r, s', done) transitions
      - HER relabeling via .relabel_last_episode()
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.ptr = self.size = 0

        self.obs     = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),         dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),         dtype=np.float32)

        # Track episode boundaries for HER
        self._ep_start = 0
        self._ep_boundaries: List[Tuple[int, int]] = []

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = done
        if done:
            self._ep_boundaries.append((self._ep_start, self.ptr))
            self._ep_start = (self.ptr + 1) % self.capacity
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def relabel_last_episode(self, result: HullResult, hull_reward: float):
        """Retroactively overwrite rewards for the most recent episode."""
        if not self._ep_boundaries:
            return
        start, end = self._ep_boundaries[-1]
        ep_len = (end - start) % self.capacity + 1
        if len(result.in_hull_mask) < ep_len:
            return
        for i, flag in enumerate(result.in_hull_mask[:ep_len]):
            if flag:
                idx = (start + i) % self.capacity
                self.rewards[idx] = hull_reward

    def sample(self, batch_size: int):
        idx   = np.random.randint(0, self.size, size=batch_size)
        def t(x): return torch.tensor(x[idx], device=self.device)
        return t(self.obs), t(self.actions), t(self.rewards), \
               t(self.next_obs), t(self.dones)

    def __len__(self): return self.size


# ══════════════════════════════════════════════════════════════════════════════
#  1.  MODEL-BASED RL
#  Exploit known A, B — gradient directly through dynamics
# ══════════════════════════════════════════════════════════════════════════════

class LinearPolicy(nn.Module):
    """
    Deterministic policy: u = K·x  (linear feedback gain).
    This is the optimal structure for LQR-like problems.
    Extended with a nonlinear correction for hull-seeking behavior.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 64, nonlinear: bool = True):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim, bias=False)  # K matrix

        self.nonlinear_head = None
        if nonlinear:
            self.nonlinear_head = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, act_dim),
            )
            nn.init.zeros_(self.nonlinear_head[-1].weight)
            nn.init.zeros_(self.nonlinear_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.linear(x)
        if self.nonlinear_head is not None:
            u = u + self.nonlinear_head(x)
        return u


class ModelBasedRL:
    """
    Model-Based Policy Optimization using known linear dynamics.

    Key idea: since A and B are known, we can unroll the dynamics for H steps
    and backpropagate the reward gradient directly through them into the policy.
    No value function needed. No replay buffer needed.

    Objective (differentiable through dynamics):
        max_π  Σ_{t=0}^{H} γ^t · r(x_t, u_t)
        s.t.   x_{t+1} = A·x_t + B·π(x_t)

    This is essentially shooting-based optimal control with a learned policy
    as the parametric decision variable.

    Sample efficiency: extremely high — each env step generates a full
    H-step gradient signal through the differentiable model rollout.
    """

    def __init__(
        self,
        A:            np.ndarray,
        B:            np.ndarray,
        obs_space,
        act_space,
        horizon:      int   = 20,        # differentiable rollout length
        gamma:        float = 0.99,
        lr:           float = 3e-4,
        hidden:       int   = 64,
        hull_reward:  float = 100.0,
        boundary_pen: float = -100.0,
        device:       str   = "cpu",
    ):
        self.device = torch.device(device)
        self.H      = horizon
        self.gamma  = gamma

        self.A = torch.tensor(A, dtype=torch.float32, device=self.device)
        self.B = torch.tensor(B, dtype=torch.float32, device=self.device)

        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        self.act_low  = torch.tensor(act_space.low,  device=self.device)
        self.act_high = torch.tensor(act_space.high, device=self.device)
        self.obs_low  = torch.tensor(obs_space.low,  device=self.device)
        self.obs_high = torch.tensor(obs_space.high, device=self.device)

        self.policy    = LinearPolicy(obs_dim, act_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.hull_reward  = hull_reward
        self.boundary_pen = boundary_pen

        # Hull center tracked from monitor (updated externally)
        self.hull_center: Optional[torch.Tensor] = None

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x   = torch.tensor(obs, dtype=torch.float32, device=self.device)
            u   = self.policy(x)
            u   = torch.clamp(u, self.act_low, self.act_high)
        return u.cpu().numpy()

    def update(self, obs_batch: np.ndarray) -> dict:
        """
        Differentiable model rollout from a batch of starting states.
        Backprop reward gradient through H-step unrolled dynamics.
        """
        B_sz = obs_batch.shape[0]
        x    = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

        total_reward = torch.zeros(B_sz, device=self.device)
        discount     = 1.0

        for _ in range(self.H):
            u     = self.policy(x)
            u     = torch.clamp(u, self.act_low, self.act_high)

            x_next = (self.A @ x.unsqueeze(-1) + self.B @ u.unsqueeze(-1)).squeeze(-1)

            # Differentiable reward signal
            reward = self._diff_reward(x, x_next)
            total_reward = total_reward + discount * reward
            discount    *= self.gamma
            x            = x_next

        loss = -total_reward.mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item(), "mean_reward": total_reward.mean().item()}

    def _diff_reward(self, x: torch.Tensor,
                     x_next: torch.Tensor) -> torch.Tensor:
        """
        Differentiable reward proxy:
          - Boundary penalty: soft sigmoid penalizes states near/outside bounds
          - Hull reward: if hull center known, reward proximity to it
        """
        # Soft boundary penalty (differentiable approximation)
        margin   = torch.min(
            x_next - self.obs_low, self.obs_high - x_next
        ).min(dim=-1).values
        boundary = torch.sigmoid(margin * 5.0) * 2.0 - 1.0   # in (-1, 1)

        if self.hull_center is not None:
            dist   = torch.norm(x_next - self.hull_center, dim=-1)
            reward = -0.1 * dist + boundary * 10.0
        else:
            # Before hull is known: encourage exploration (maximize state spread)
            reward = boundary * 0.1

        return reward

    def set_hull_center(self, center: np.ndarray):
        self.hull_center = torch.tensor(
            center, dtype=torch.float32, device=self.device)


# ══════════════════════════════════════════════════════════════════════════════
#  2.  SAC  — Soft Actor-Critic
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, n_layers=2, activate_last=False):
        super().__init__()
        dims   = [in_dim] + [hidden] * n_layers + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or activate_last:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class SACActor(nn.Module):
    """
    Stochastic actor: outputs (μ, log_σ) → reparameterized Normal.
    Actions squashed through tanh to respect bounds.
    """
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, hidden=256, act_scale=1.0):
        super().__init__()
        self.net     = MLP(obs_dim, hidden, hidden, activate_last=True)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.ls_head = nn.Linear(hidden, act_dim)
        self.act_scale = act_scale

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.net(x)
        mu     = self.mu_head(h)
        log_std = self.ls_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) with reparameterization + tanh squashing."""
        mu, log_std = self(x)
        std         = log_std.exp()
        eps         = torch.randn_like(mu)
        u           = mu + eps * std                          # pre-squash

        # Tanh squash
        action      = torch.tanh(u) * self.act_scale

        # Log prob with tanh correction
        log_prob    = Normal(mu, std).log_prob(u).sum(-1, keepdim=True)
        log_prob   -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    """Twin Q-networks for clipped double-Q learning."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x  = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, obs, act):
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)


class SACAgent:
    """
    Soft Actor-Critic for continuous action spaces.

    Why SAC for your problem:
      - Off-policy → high sample efficiency via replay buffer
      - Maximum entropy → natural exploration, resists premature convergence
        to suboptimal hull (important when hull reward is non-stationary)
      - Automatic entropy tuning → no manual temperature tuning
      - Pairs perfectly with HER relabeling (buffer is reusable)

    Hull reward is injected via buffer.relabel_last_episode() after
    each episode, so SAC never sees the raw environment rewards for
    hull-hit steps — only the relabeled ones.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        hidden:        int   = 256,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        tau:           float = 0.005,
        alpha:         float = 0.2,      # initial entropy coefficient
        auto_alpha:    bool  = True,     # automatic entropy tuning
        buffer_size:   int   = 1_000_000,
        batch_size:    int   = 256,
        warmup_steps:  int   = 1000,     # random actions before training
        update_every:  int   = 1,
        device:        str   = "cpu",
    ):
        self.device       = torch.device(device)
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        self.update_every = update_every

        obs_dim  = obs_space.shape[0]
        act_dim  = act_space.shape[0]

        # Action scaling (tanh outputs [-1,1], scale to actual bounds)
        act_scale  = float((act_space.high - act_space.low).max() / 2.0)
        self.act_bias  = torch.tensor(
            (act_space.high + act_space.low) / 2.0,
            dtype=torch.float32, device=self.device)
        self.act_scale = act_scale

        # Networks
        self.actor        = SACActor(obs_dim, act_dim, hidden, act_scale).to(self.device)
        self.critic       = SACCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_opt   = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt  = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.auto_alpha  = auto_alpha
        if auto_alpha:
            self.target_entropy = -act_dim          # heuristic: -|A|
            self.log_alpha      = torch.tensor(
                np.log(alpha), requires_grad=True,
                device=self.device, dtype=torch.float32)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha     = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.buffer   = ReplayBuffer(buffer_size, obs_dim, act_dim, self.device)
        self._steps   = 0
        self.metrics: Dict[str, float] = {}

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic or self._steps < self.warmup_steps:
                if self._steps < self.warmup_steps:
                    # Random warmup
                    return np.random.uniform(-self.act_scale, self.act_scale,
                                             size=(self.actor.mu_head.out_features,))
                mu, _ = self.actor(x)
                return (torch.tanh(mu) * self.act_scale + self.act_bias).squeeze().cpu().numpy()
            action, _ = self.actor.sample(x)
            return (action + self.act_bias).squeeze().cpu().numpy()

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self._steps += 1

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size or self._steps < self.warmup_steps:
            return {}
        if self._steps % self.update_every != 0:
            return {}

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_act, next_lp = self.actor.sample(next_obs)
            next_q  = self.critic_target.min_q(next_obs, next_act)
            target_q = rews + (1 - dones) * self.gamma * (next_q - self.alpha * next_lp)

        q1, q2    = self.critic(obs, acts)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        new_act, log_pi = self.actor.sample(obs)
        q_pi            = self.critic.min_q(obs, new_act)
        actor_loss      = (self.alpha * log_pi - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Alpha update ──────────────────────────────────────────────────────
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # Polyak update target
        soft_update(self.critic_target, self.critic, self.tau)

        self.metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss.item(),
        }
        return self.metrics


# ══════════════════════════════════════════════════════════════════════════════
#  3.  CEM + NATURAL EVOLUTION STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

class CEMAgent:
    """
    Cross-Entropy Method for policy optimization.

    Treats the policy as a multivariate Gaussian over parameters θ.
    Each generation: sample N policies, evaluate, keep top-k elite,
    fit new Gaussian to elites.

    Why CEM for this problem:
      - Gradient-free: robust to discontinuous hull-boundary rewards
      - Naturally parallelizable: N policy evaluations are independent
      - No hyperparameter sensitivity (no learning rate instability)
      - Works well when reward landscape has multiple modes (multiple valid hulls)

    The policy is a linear+bias map: u = Kx + b
    This keeps the parameter space small and well-conditioned.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        population_size: int   = 64,
        elite_frac:      float = 0.2,
        noise_std_init:  float = 0.5,
        noise_std_min:   float = 0.01,
        noise_decay:     float = 0.995,
        n_eval_steps:    int   = 200,     # steps per policy evaluation
        gamma:           float = 0.99,
        extra_noise:     float = 0.1,     # extra noise injected to avoid collapse
    ):
        self.obs_dim    = obs_space.shape[0]
        self.act_dim    = act_space.shape[0]
        self.act_low    = act_space.low
        self.act_high   = act_space.high

        self.N          = population_size
        self.n_elite    = max(1, int(population_size * elite_frac))
        self.noise_std  = noise_std_init
        self.noise_min  = noise_std_min
        self.noise_decay = noise_decay
        self.n_eval     = n_eval_steps
        self.gamma      = gamma
        self.extra_noise = extra_noise

        # Policy: u = W·x + b  (linear policy)
        self.param_dim  = self.obs_dim * self.act_dim + self.act_dim
        self.theta_mean = np.zeros(self.param_dim, dtype=np.float32)
        self.theta_std  = np.ones(self.param_dim, dtype=np.float32) * noise_std_init

        self._best_theta = self.theta_mean.copy()
        self._best_score = -np.inf
        self.generation  = 0
        self.metrics: Dict[str, float] = {}

    def _theta_to_policy(self, theta: np.ndarray):
        """Unpack flat parameter vector into (W, b)."""
        W = theta[:self.obs_dim * self.act_dim].reshape(self.act_dim, self.obs_dim)
        b = theta[self.obs_dim * self.act_dim:]
        return W, b

    def _policy_action(self, theta: np.ndarray, obs: np.ndarray) -> np.ndarray:
        W, b = self._theta_to_policy(theta)
        u    = W @ obs + b
        return np.clip(u, self.act_low, self.act_high)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Use current best policy (mean of distribution)."""
        return self._policy_action(self._best_theta, obs)

    def evolve(self, env_factory, hull_reward: float = 100.0,
               monitor=None) -> dict:
        """
        Run one generation of CEM:
          1. Sample N parameter vectors from current distribution
          2. Evaluate each in a fresh env copy
          3. Keep top-k elites, update distribution
        """
        # Sample population
        population = self.theta_mean + \
            np.random.randn(self.N, self.param_dim).astype(np.float32) * self.theta_std

        scores    = np.zeros(self.N)
        ep_records = []

        for i, theta in enumerate(population):
            env = env_factory()
            obs, _ = env.reset()
            states  = [obs.copy()]
            actions = []
            rewards = []

            for _ in range(self.n_eval):
                act = self._policy_action(theta, obs)
                obs, r, term, trunc, _ = env.step(act)
                states.append(obs.copy()); actions.append(act.copy())
                rewards.append(r)
                if term or trunc: break

            # Post-episode hull check
            record = EpisodeRecord(
                states  = np.array(states,  dtype=np.float32),
                actions = np.array(actions, dtype=np.float32),
                rewards = np.array(rewards, dtype=np.float32),
                ep_id   = self.generation * self.N + i,
            )
            if monitor is not None:
                result  = monitor.process(record)
                rewards = result.relabel_rewards(np.array(rewards), hull_reward)

            # Discounted return
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
            scores[i]  = G
            ep_records.append(record)

        # Select elites
        elite_idx  = np.argsort(scores)[-self.n_elite:]
        elites     = population[elite_idx]
        best_idx   = elite_idx[-1]

        if scores[best_idx] > self._best_score:
            self._best_score = scores[best_idx]
            self._best_theta = population[best_idx].copy()

        # Update distribution — fit Gaussian to elites + extra noise
        self.theta_mean = elites.mean(axis=0)
        self.theta_std  = elites.std(axis=0) + self.extra_noise
        self.theta_std  = np.maximum(self.theta_std, self.noise_min)
        self.noise_std  = max(self.noise_std * self.noise_decay, self.noise_min)

        self.generation += 1
        self.metrics = {
            "best_score":  float(self._best_score),
            "mean_score":  float(scores.mean()),
            "elite_mean":  float(scores[elite_idx].mean()),
            "noise_std":   float(self.noise_std),
            "generation":  self.generation,
        }
        return self.metrics, ep_records


class NESAgent:
    """
    Natural Evolution Strategy — gradient-based but gradient-free wrt env.

    Unlike CEM (which fits a distribution to elites), NES estimates the
    gradient of E[F(θ)] using the log-derivative trick:
        ∇_μ E[F] = E[(F(θ) - b) · ∇_μ log p(θ|μ)]
                 = (1/N·σ) Σ_i F_i · ε_i    (antithetic sampling)

    where b is a baseline (mean fitness), σ is the noise scale.
    This gives a natural policy gradient without differentiating through env.

    Antithetic sampling: evaluate θ+ε and θ-ε in pairs to reduce variance.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        population_size: int   = 50,
        sigma:           float = 0.1,
        lr:              float = 0.01,
        n_eval_steps:    int   = 200,
        gamma:           float = 0.99,
        lr_decay:        float = 0.999,
    ):
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.act_low = act_space.low; self.act_high = act_space.high

        assert population_size % 2 == 0, "NES needs even population for antithetic pairs"
        self.N        = population_size
        self.sigma    = sigma
        self.lr       = lr
        self.n_eval   = n_eval_steps
        self.gamma    = gamma
        self.lr_decay = lr_decay

        self.param_dim   = self.obs_dim * self.act_dim + self.act_dim
        self.theta       = np.zeros(self.param_dim, dtype=np.float32)
        self.generation  = 0
        self.metrics: Dict[str, float] = {}

    def _policy_action(self, theta, obs):
        W = theta[:self.obs_dim * self.act_dim].reshape(self.act_dim, self.obs_dim)
        b = theta[self.obs_dim * self.act_dim:]
        return np.clip(W @ obs + b, self.act_low, self.act_high)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._policy_action(self.theta, obs)

    def _evaluate(self, theta, env_factory, monitor=None,
                  hull_reward=100.0) -> Tuple[float, EpisodeRecord]:
        env   = env_factory()
        obs, _= env.reset()
        states, actions, rewards = [obs.copy()], [], []

        for _ in range(self.n_eval):
            act = self._policy_action(theta, obs)
            obs, r, term, trunc, _ = env.step(act)
            states.append(obs.copy()); actions.append(act.copy()); rewards.append(r)
            if term or trunc: break

        record = EpisodeRecord(
            states  = np.array(states,  dtype=np.float32),
            actions = np.array(actions, dtype=np.float32),
            rewards = np.array(rewards, dtype=np.float32),
        )
        if monitor is not None:
            result  = monitor.process(record)
            rewards = list(result.relabel_rewards(np.array(rewards), hull_reward))

        G = sum(r * self.gamma**t for t, r in enumerate(rewards))
        return float(G), record

    def evolve(self, env_factory, hull_reward=100.0, monitor=None) -> dict:
        """
        One NES generation using antithetic sampling.
        """
        half  = self.N // 2
        noise = np.random.randn(half, self.param_dim).astype(np.float32)

        scores     = np.zeros(self.N, dtype=np.float32)
        ep_records = []

        for i in range(half):
            f_pos, r_pos = self._evaluate(
                self.theta + self.sigma * noise[i], env_factory, monitor, hull_reward)
            f_neg, r_neg = self._evaluate(
                self.theta - self.sigma * noise[i], env_factory, monitor, hull_reward)
            scores[i]        = f_pos
            scores[half + i] = f_neg
            ep_records.extend([r_pos, r_neg])

        # Rank-normalize fitness for stable gradient
        ranks         = np.argsort(np.argsort(scores)).astype(np.float32)
        ranks         = (ranks / (self.N - 1)) - 0.5   # in [-0.5, 0.5]

        # NES gradient: weighted sum of noise directions
        gradient = np.zeros_like(self.theta)
        for i in range(half):
            gradient += (ranks[i] - ranks[half + i]) * noise[i]
        gradient /= (self.N * self.sigma)

        # Gradient ascent
        self.theta  += self.lr * gradient
        self.lr     *= self.lr_decay
        self.generation += 1

        self.metrics = {
            "best_score":  float(scores.max()),
            "mean_score":  float(scores.mean()),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "lr":          self.lr,
            "generation":  self.generation,
        }
        return self.metrics, ep_records


# ══════════════════════════════════════════════════════════════════════════════
#  4.  MPC-RL HYBRID
#  Model Predictive Control + learned value function tail correction
# ══════════════════════════════════════════════════════════════════════════════

class ValueNetwork(nn.Module):
    """Learned value function V(s) to correct finite-horizon MPC bias."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x): return self.net(x).squeeze(-1)


class MPCRLAgent:
    """
    Model Predictive Control with learned value function tail.

    Action selection at each step t:
      1. MPC: plan H-step sequence {u_0..u_{H-1}} to maximize
              Σ_{k=0}^{H-1} γ^k r_k + γ^H V(x_H)
              using the known dynamics x_{k+1} = Ax_k + Bu_k
      2. Execute only u_0 (receding horizon)
      3. RL: learn V(s) from observed returns (TD or MC)

    Why this combination is powerful:
      - MPC handles short-horizon precision (exact dynamics + planning)
      - V(s) handles long-horizon credit assignment (what MPC can't see)
      - Hull reward shapes V(s) naturally: V(s) ≈ hull_reward for s near hull
      - No policy network needed — actions come directly from optimization

    Optimization: Random Shooting (sample M random sequences, pick best)
    Can be upgraded to CEM-MPC for higher quality at more compute.
    """

    def __init__(
        self,
        A:              np.ndarray,
        B:              np.ndarray,
        obs_space,
        act_space,
        horizon:        int   = 15,
        n_samples:      int   = 512,      # random shooting candidates
        gamma:          float = 0.99,
        value_lr:       float = 3e-4,
        value_hidden:   int   = 128,
        value_update_every: int = 10,
        buffer_size:    int   = 100_000,
        batch_size:     int   = 256,
        warmup_steps:   int   = 500,
        device:         str   = "cpu",
        # MPC-CEM refinement (optional)
        cem_iters:      int   = 3,        # CEM refinement iterations (0 = pure random shooting)
        cem_elite_frac: float = 0.1,
    ):
        self.device     = torch.device(device)
        self.H          = horizon
        self.M          = n_samples
        self.gamma      = gamma
        self.cem_iters  = cem_iters
        self.cem_elite  = max(1, int(n_samples * cem_elite_frac))

        self.A = torch.tensor(A, dtype=torch.float32, device=self.device)
        self.B = torch.tensor(B, dtype=torch.float32, device=self.device)

        obs_dim  = obs_space.shape[0]
        act_dim  = act_space.shape[0]
        self.act_low  = torch.tensor(act_space.low,  dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(act_space.high, dtype=torch.float32, device=self.device)
        self.obs_low  = torch.tensor(obs_space.low,  dtype=torch.float32, device=self.device)
        self.obs_high = torch.tensor(obs_space.high, dtype=torch.float32, device=self.device)
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim

        # Value function
        self.value      = ValueNetwork(obs_dim, value_hidden).to(self.device)
        self.value_target = copy.deepcopy(self.value)
        self.value_opt  = optim.Adam(self.value.parameters(), lr=value_lr)

        # Transition replay buffer for value learning
        self.buffer     = ReplayBuffer(buffer_size, obs_dim, act_dim, self.device)
        self.batch_size = batch_size
        self.warmup     = warmup_steps
        self.update_every = value_update_every

        self._steps     = 0
        self.metrics: Dict[str, float] = {}

        # Hull reward shaping target (set externally from monitor)
        self.hull_center: Optional[torch.Tensor] = None
        self.hull_target: Optional[torch.Tensor] = None

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """MPC action selection via random shooting + CEM refinement."""
        with torch.no_grad():
            x0 = torch.tensor(obs, dtype=torch.float32, device=self.device)
            u0 = self._mpc_plan(x0)
        return u0.cpu().numpy()

    def _mpc_plan(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Plan H-step action sequence using random shooting.
        Optionally refine with CEM iterations.
        Returns the first action of the best sequence.
        """
        # Initial random candidates: (M, H, act_dim)
        u_mean = torch.zeros(self.H, self.act_dim, device=self.device)
        u_std  = torch.ones( self.H, self.act_dim, device=self.device)

        for cem_it in range(max(1, self.cem_iters)):
            # Sample action sequences
            eps    = torch.randn(self.M, self.H, self.act_dim, device=self.device)
            U      = (u_mean.unsqueeze(0) + u_std.unsqueeze(0) * eps)
            U      = U.clamp(self.act_low, self.act_high)        # (M, H, act_dim)

            # Roll out dynamics for all M sequences in parallel
            returns = self._rollout_returns(x0, U)               # (M,)

            if self.cem_iters > 0:
                # CEM: update distribution from elites
                elite_idx = torch.argsort(returns)[-self.cem_elite:]
                elites    = U[elite_idx]                          # (n_elite, H, act_dim)
                u_mean    = elites.mean(dim=0)
                u_std     = elites.std(dim=0).clamp(min=0.05)
            else:
                break

        # Return first action of best sequence
        best_idx = torch.argmax(returns)
        return U[best_idx, 0]                                    # (act_dim,)

    def _rollout_returns(self, x0: torch.Tensor,
                          U: torch.Tensor) -> torch.Tensor:
        """
        Parallel rollout of M action sequences from x0.
        Returns discounted returns including V(x_H) tail correction.

        x0: (obs_dim,)
        U:  (M, H, act_dim)
        Returns: (M,)
        """
        M      = U.shape[0]
        x      = x0.unsqueeze(0).expand(M, -1)    # (M, obs_dim)
        G      = torch.zeros(M, device=self.device)
        disc   = 1.0

        for h in range(self.H):
            u      = U[:, h, :]                    # (M, act_dim)
            x_next = (
                (self.A @ x.unsqueeze(-1)).squeeze(-1)
                + (self.B @ u.unsqueeze(-1)).squeeze(-1)
            )                                       # (M, obs_dim)

            reward = self._reward_proxy(x, x_next)  # (M,)
            G     += disc * reward
            disc  *= self.gamma
            x      = x_next

        # Tail correction: V(x_H)
        if self._steps >= self.warmup:
            V_tail = self.value(x)                  # (M,)
            G     += disc * V_tail

        return G

    def _reward_proxy(self, x: torch.Tensor,
                       x_next: torch.Tensor) -> torch.Tensor:
        """
        Differentiable reward proxy used inside MPC rollout.
        - Boundary: penalize states outside obs space
        - Hull proximity: reward closeness to hull center/target
        """
        # Boundary check (soft)
        margin  = torch.min(
            x_next - self.obs_low,
            self.obs_high - x_next
        ).min(dim=-1).values
        in_bounds = torch.sigmoid(margin * 10.0)  # ≈1 inside, ≈0 outside

        if self.hull_center is not None:
            dist   = torch.norm(x_next - self.hull_center, dim=-1)
            reward = -0.1 * dist * in_bounds + (in_bounds - 0.5) * 10.0
        else:
            # Encourage spreading (maximizing hull volume) before hull known
            spread = torch.norm(x_next - x, dim=-1)
            reward = spread * in_bounds * 0.1

        return reward

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self._steps += 1

    def update(self) -> dict:
        """Learn value function from replay buffer via TD(0)."""
        if len(self.buffer) < self.batch_size or self._steps < self.warmup:
            return {}
        if self._steps % self.update_every != 0:
            return {}

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            v_next  = self.value_target(next_obs).unsqueeze(-1)
            v_target = rews + (1 - dones) * self.gamma * v_next

        v_pred  = self.value(obs).unsqueeze(-1)
        v_loss  = F.mse_loss(v_pred, v_target)

        self.value_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_opt.step()
        soft_update(self.value_target, self.value, tau=0.005)

        self.metrics = {"value_loss": v_loss.item()}
        return self.metrics

    def set_hull_center(self, center: np.ndarray):
        self.hull_center = torch.tensor(
            center, dtype=torch.float32, device=self.device)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED TRAINING LOOP
#  Works with all four agents via duck typing
# ══════════════════════════════════════════════════════════════════════════════

def train(
    env,
    agent,
    monitor               = None,
    total_steps:    int   = 200_000,
    hull_reward:    float = 100.0,
    log_interval:   int   = 10,
    eval_interval:  int   = 50,
    device:         str   = "cpu",
):
    """
    Unified training loop compatible with:
      ModelBasedRL, SACAgent, MPCRLAgent  → step-by-step interaction
      CEMAgent, NESAgent                  → handled separately (population-based)
    """
    is_population = isinstance(agent, (CEMAgent, NESAgent))
    if is_population:
        return _train_population(env, agent, monitor, total_steps,
                                  hull_reward, log_interval)

    obs, _    = env.reset()
    ep_states  = [obs.copy()]
    ep_actions = []
    ep_rewards = []
    ep_count   = 0
    step       = 0

    print(f"\nTraining {type(agent).__name__} | total_steps={total_steps}")
    print(f"{'Step':>10} | {'Episode':>8} | {'Return':>10} | {'Hull':>6} | Info")
    print("─" * 65)

    while step < total_steps:
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        ep_states.append(next_obs.copy())
        ep_actions.append(action.copy())
        ep_rewards.append(reward)

        # Push transition (SAC and MPC-RL)
        if hasattr(agent, "push"):
            agent.push(obs, action, reward, next_obs, float(done))

        obs   = next_obs
        step += 1

        # Step-level updates (SAC, MPC-RL)
        if hasattr(agent, "update"):
            metrics = agent.update()

        if done:
            ep_count += 1
            record = EpisodeRecord(
                states  = np.array(ep_states,  dtype=np.float32),
                actions = np.array(ep_actions, dtype=np.float32),
                rewards = np.array(ep_rewards, dtype=np.float32),
                ep_id   = ep_count,
            )

            # Post-episode hull processing
            hull_hit = False
            if monitor is not None:
                result = monitor.process(record)
                hull_hit = result.hull_reached

                # SAC: relabel buffer
                if hasattr(agent, "buffer") and hasattr(agent.buffer, "relabel_last_episode"):
                    agent.buffer.relabel_last_episode(result, hull_reward)

                # Model-based / MPC: update hull center
                if result.hull_reached and monitor.target is not None:
                    center = monitor.target.mean(axis=0)
                    if hasattr(agent, "set_hull_center"):
                        agent.set_hull_center(center)

            # Model-based: update from episode starting states
            if isinstance(agent, ModelBasedRL) and len(ep_states) > 1:
                start_states = np.array(ep_states[:-1], dtype=np.float32)
                if len(start_states) > 0:
                    batch = start_states[
                        np.random.choice(len(start_states),
                                         min(32, len(start_states)), replace=False)]
                    agent.update(batch)

            ep_ret = sum(ep_rewards)
            if ep_count % log_interval == 0:
                m_str = " | ".join(f"{k}={v:.3f}"
                                   for k, v in (agent.metrics.items()
                                                if hasattr(agent, "metrics") else {}).items()
                                   if not k.endswith("_loss"))
                print(f"{step:>10} | {ep_count:>8} | {ep_ret:>10.2f} | "
                      f"{'✓' if hull_hit else '✗':>6} | {m_str}")

            # Reset
            obs, _     = env.reset()
            ep_states  = [obs.copy()]
            ep_actions = []
            ep_rewards = []

    return agent


def _train_population(env, agent, monitor, total_steps,
                       hull_reward, log_interval):
    """Training loop for population-based methods (CEM, NES)."""
    import functools
    env_factory = lambda: type(env)(
        env.A, env.B,
        (env.observation_space.low, env.observation_space.high),
        (env.action_space.low,      env.action_space.high),
    )
    total_evals = 0
    gen         = 0

    print(f"\nTraining {type(agent).__name__} | total_steps={total_steps}")
    print(f"{'Evals':>8} | {'Gen':>5} | {'Best':>10} | {'Mean':>10} | Info")
    print("─" * 55)

    while total_evals < total_steps:
        metrics, ep_records = agent.evolve(
            env_factory, hull_reward=hull_reward, monitor=monitor)
        total_evals += agent.N * agent.n_eval
        gen         += 1

        if gen % log_interval == 0:
            m_str = " | ".join(f"{k}={v:.3f}"
                                for k, v in metrics.items()
                                if k in ("best_score", "mean_score"))
            print(f"{total_evals:>8} | {gen:>5} | {m_str}")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM COMPARISON RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def compare_algorithms(env_factory, monitor_factory,
                        total_steps=100_000, seed=42, device="cpu"):
    """
    Run all four algorithms on the same env and print a comparison table.
    """
    import time
    np.random.seed(seed)
    torch.manual_seed(seed)

    env   = env_factory()
    obs_s = env.observation_space
    act_s = env.action_space
    A, B  = env.A, env.B

    agents = {
        "ModelBased": ModelBasedRL(A, B, obs_s, act_s, device=device),
        "SAC":        SACAgent(obs_s, act_s, device=device),
        "MPC-RL":     MPCRLAgent(A, B, obs_s, act_s, device=device),
        "CEM":        CEMAgent(obs_s, act_s),
    }

    results = {}
    for name, agent in agents.items():
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        t0      = time.time()
        monitor = monitor_factory()
        trained = train(env_factory(), agent, monitor,
                         total_steps=total_steps, log_interval=20)
        dt      = time.time() - t0
        results[name] = {
            "wall_time":   dt,
            "hull_valid":  monitor.is_valid_target,
            "hull_points": len(monitor.target) if monitor.target is not None else 0,
        }

    print(f"\n{'='*60}")
    print(f"  {'Algorithm':<12} | {'Time (s)':>9} | {'Hull Valid':>10} | {'Hull Points':>11}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        print(f"  {name:<12} | {r['wall_time']:>9.1f} | "
              f"{str(r['hull_valid']):>10} | {r['hull_points']:>11}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    A = np.array([[0.9, 0.1], [0.0, 0.9]], dtype=np.float32)
    B = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    state_bounds  = ([-5., -5.], [5., 5.])
    action_bounds = ([-1., -1.], [1., 1.])

    print("Testing individual algorithm construction...")

    class DummySpace:
        def __init__(self, low, high):
            self.low   = np.array(low,  dtype=np.float32)
            self.high  = np.array(high, dtype=np.float32)
            self.shape = self.low.shape

    obs_s = DummySpace(state_bounds[0],  state_bounds[1])
    act_s = DummySpace(action_bounds[0], action_bounds[1])

    mb  = ModelBasedRL(A, B, obs_s, act_s)
    sac = SACAgent(obs_s, act_s)
    cem = CEMAgent(obs_s, act_s)
    nes = NESAgent(obs_s, act_s)
    mpc = MPCRLAgent(A, B, obs_s, act_s)

    # Quick action selection test
    obs = np.random.randn(2).astype(np.float32)
    print(f"  ModelBased action: {mb.select_action(obs)}")
    print(f"  SAC action:        {sac.select_action(obs)}")
    print(f"  CEM action:        {cem.select_action(obs)}")
    print(f"  NES action:        {nes.select_action(obs)}")
    print(f"  MPC-RL action:     {mpc.select_action(obs)}")

    n_mb  = sum(p.numel() for p in mb.policy.parameters())
    n_sac = sum(p.numel() for p in list(sac.actor.parameters())
                             + list(sac.critic.parameters()))
    n_mpc = sum(p.numel() for p in mpc.value.parameters())

    print(f"\nParameter counts:")
    print(f"  ModelBased policy:  {n_mb:,}")
    print(f"  SAC actor+critic:   {n_sac:,}")
    print(f"  CEM policy params:  {cem.param_dim}")
    print(f"  NES policy params:  {nes.param_dim}")
    print(f"  MPC value net:      {n_mpc:,}")
    print("\nAll agents constructed successfully.")
