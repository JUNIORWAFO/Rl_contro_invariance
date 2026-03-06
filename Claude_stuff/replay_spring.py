"""
replay_spring.py — Replay memory for the Spring-Mass-Damper RL pipeline.

Ported from Section 4 of spring_mass_dampener.py and aligned with the
ReplayBuffer in rl_algorithms.py.

Classes
-------
  SpringReplayBuffer          — standard numpy ring buffer (fast, off-policy)
  SpringSuccessReplayBuffer   — two-stage buffer; commits only on success
                                (port of SuccessReplayMemory)
  SpringPPORolloutBuffer      — on-policy rollout buffer for PPO
                                stores (obs, act, logp, rew, val, done) per step

All buffers expose:
    .push(...)   — add a transition
    .sample(B)   — uniform random batch  (not applicable for rollout buffer)
    .ready       — True when enough samples are available
    .__len__     — current fill level
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
#  1. STANDARD RING-BUFFER  (off-policy, matches ReplayBuffer in rl_algorithms.py)
# ══════════════════════════════════════════════════════════════════════════════

class SpringReplayBuffer:
    """
    Uniform experience replay backed by pre-allocated numpy arrays.

    Matches the ReplayBuffer interface in rl_algorithms.py but stores the
    extra `center` field used by the  springreward shaping (optional).

    push(obs, action, reward, next_obs, done)
    sample(batch_size) → dict of torch.Tensors
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_dim:  int = 2,
        act_dim:  int = 1,
        device:   str = "cpu",
    ):
        self.cap     = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device  = device

        self.s   = np.zeros((capacity, obs_dim), np.float32)
        self.a   = np.zeros((capacity, act_dim), np.float32)
        self.r   = np.zeros(capacity,            np.float32)
        self.ns  = np.zeros((capacity, obs_dim), np.float32)
        self.d   = np.zeros(capacity,            np.float32)

        self.ptr  = 0
        self.size = 0

    # ── write ────────────────────────────────────────────────────────────────

    def push(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        self.s[self.ptr]  = obs
        self.a[self.ptr]  = action
        self.r[self.ptr]  = reward
        self.ns[self.ptr] = next_obs
        self.d[self.ptr]  = float(done)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    # ── read ─────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.tensor(self.s[idx],  device=self.device),
            "action":   torch.tensor(self.a[idx],  device=self.device),
            "reward":   torch.tensor(self.r[idx],  device=self.device).unsqueeze(-1),
            "next_obs": torch.tensor(self.ns[idx], device=self.device),
            "done":     torch.tensor(self.d[idx],  device=self.device).unsqueeze(-1),
        }

    @property
    def ready(self) -> bool:
        return self.size >= 256          # minimum batch threshold

    def __len__(self) -> int:
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
#  2. SUCCESS REPLAY BUFFER
#     Port of SuccessReplayMemory from spring_mass_dampener.py
# ══════════════════════════════════════════════════════════════════════════════

class SpringSuccessReplayBuffer:
    """
    Two-stage replay buffer.  Experiences accumulate in a temporary deque
    and are only committed to the main buffer when push_success() is called
    (i.e. at the end of a successful episode).

    Directly ported from SuccessReplayMemory in spring_mass_dampener.py with
    the numpy-backed main buffer from SpringReplayBuffer.

    API
    ---
    push(...)           — write to temporary buffer
    push_success()      — commit temp → main
    flush()             — discard temp (failed episode)
    sample(B)           — uniform sample from main buffer
    """

    def __init__(
        self,
        capacity:      int = 100_000,
        temp_capacity: int = 10_000,
        obs_dim:       int = 2,
        act_dim:       int = 1,
        device:        str = "cpu",
    ):
        self.main = SpringReplayBuffer(capacity, obs_dim, act_dim, device)
        self.temp: deque = deque(maxlen=temp_capacity)

    # ── temp ─────────────────────────────────────────────────────────────────

    def push(self, obs, action, reward, next_obs, done):
        self.temp.append((obs, action, reward, next_obs, done))

    def flush(self):
        self.temp.clear()

    def push_success(self):
        """Commit all temporary experiences to the main ring buffer."""
        for transition in self.temp:
            self.main.push(*transition)
        self.flush()

    # ── main ─────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.main.sample(batch_size)

    @property
    def ready(self) -> bool:
        return self.main.ready

    def __len__(self) -> int:
        return len(self.main)


# ══════════════════════════════════════════════════════════════════════════════
#  3. ON-POLICY ROLLOUT BUFFER  — for PPO with vectorised envs
# ══════════════════════════════════════════════════════════════════════════════

class PPORolloutBuffer:
    """
    On-policy rollout buffer for PPO with make_vec_env (SyncVectorEnv).

    Stores n_steps × n_envs transitions, then exposes them for PPO mini-batch
    updates.  Supports GAE-λ advantage estimation.

    Usage
    -----
    buf = PPORolloutBuffer(n_steps=128, n_envs=4, obs_dim=2, act_dim=1)
    buf.reset()
    for step in range(n_steps):
        buf.add(obs, action, reward, done, log_prob, value)
    buf.compute_returns(last_value, gamma, gae_lambda)
    for batch in buf.iterate(mini_batch_size):
        ...  # train on batch
    """

    def __init__(
        self,
        n_steps:    int   = 128,
        n_envs:     int   = 4,
        obs_dim:    int   = 2,
        act_dim:    int   = 1,
        device:     str   = "cpu",
    ):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device  = device

        # pre-allocate storage
        self.obs      = np.zeros((n_steps, n_envs, obs_dim), np.float32)
        self.actions  = np.zeros((n_steps, n_envs, act_dim), np.float32)
        self.rewards  = np.zeros((n_steps, n_envs),           np.float32)
        self.dones    = np.zeros((n_steps, n_envs),           np.float32)
        self.log_probs= np.zeros((n_steps, n_envs),           np.float32)
        self.values   = np.zeros((n_steps, n_envs),           np.float32)
        self.advantages = np.zeros((n_steps, n_envs),         np.float32)
        self.returns    = np.zeros((n_steps, n_envs),         np.float32)

        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(
        self,
        obs:      np.ndarray,    # (n_envs, obs_dim)
        action:   np.ndarray,    # (n_envs, act_dim)
        reward:   np.ndarray,    # (n_envs,)
        done:     np.ndarray,    # (n_envs,)  bool/float
        log_prob: np.ndarray,    # (n_envs,)
        value:    np.ndarray,    # (n_envs,)
    ):
        assert self.ptr < self.n_steps, "Buffer full — call reset() first."
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.n_steps

    # ── GAE-λ return computation ─────────────────────────────────────────────

    def compute_returns(
        self,
        last_value: np.ndarray,    # (n_envs,) bootstrap V(s_T)
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE-λ advantages and discounted returns in-place.

        Aₜ = δₜ + (γλ) δₜ₊₁ + (γλ)² δₜ₊₂ + …
        δₜ = rₜ + γ V(sₜ₊₁)(1−dₜ) − V(sₜ)
        """
        last_gae = np.zeros(self.n_envs, np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta    = (self.rewards[t]
                        + gamma * next_value * next_non_terminal
                        - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ── mini-batch iteration ─────────────────────────────────────────────────

    def iterate(
        self,
        mini_batch_size: int = 64,
        normalize_adv: bool  = True,
    ):
        """
        Yield shuffled mini-batches of flat (n_steps × n_envs) transitions.

        Each yielded dict contains torch.Tensors on self.device.
        """
        N = self.n_steps * self.n_envs
        idx = np.random.permutation(N)

        # flatten (n_steps, n_envs, ...) → (N, ...)
        obs      = self.obs.reshape(N, self.obs_dim)
        actions  = self.actions.reshape(N, self.act_dim)
        log_probs= self.log_probs.reshape(N)
        values   = self.values.reshape(N)
        advs     = self.advantages.reshape(N)
        rets     = self.returns.reshape(N)

        if normalize_adv:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for start in range(0, N, mini_batch_size):
            mb_idx = idx[start: start + mini_batch_size]
            yield {
                "obs":       torch.tensor(obs[mb_idx],       device=self.device),
                "action":    torch.tensor(actions[mb_idx],   device=self.device),
                "log_prob":  torch.tensor(log_probs[mb_idx], device=self.device),
                "value":     torch.tensor(values[mb_idx],    device=self.device),
                "advantage": torch.tensor(advs[mb_idx],      device=self.device),
                "return":    torch.tensor(rets[mb_idx],      device=self.device),
            }


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── SpringReplayBuffer ──")
    buf = SpringReplayBuffer(capacity=1000, obs_dim=2, act_dim=1)
    for _ in range(300):
        buf.push(np.random.randn(2), np.random.randn(1), float(np.random.randn()),
                 np.random.randn(2), bool(np.random.rand() > 0.95))
    batch = buf.sample(32)
    print(f"  obs: {batch['obs'].shape}  reward: {batch['reward'].shape}")

    print("── SpringSuccessReplayBuffer ──")
    sbuf = SpringSuccessReplayBuffer(obs_dim=2, act_dim=1)
    for _ in range(10):
        sbuf.push(np.zeros(2), np.zeros(1), 1.0, np.zeros(2), False)
    sbuf.push_success()
    print(f"  main size={len(sbuf)}")

    print("── PPORolloutBuffer ──")
    rbuf = PPORolloutBuffer(n_steps=16, n_envs=4, obs_dim=2, act_dim=1)
    for _ in range(16):
        rbuf.add(
            obs=np.random.randn(4, 2).astype(np.float32),
            action=np.random.randn(4, 1).astype(np.float32),
            reward=np.random.randn(4).astype(np.float32),
            done=np.zeros(4, np.float32),
            log_prob=np.random.randn(4).astype(np.float32),
            value=np.random.randn(4).astype(np.float32),
        )
    rbuf.compute_returns(last_value=np.zeros(4, np.float32))
    batches = list(rbuf.iterate(mini_batch_size=8))
    print(f"  n_batches={len(batches)}  batch_obs={batches[0]['obs'].shape}")
