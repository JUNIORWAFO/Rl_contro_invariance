"""
rl_algorithms.py
=================
RL agents for LinearUpdateEnv, all sharing one interface:

    agent.select_action(obs, explore=True) -> np.ndarray
    agent.push(obs, act, rew, next_obs, done)
    agent.update() -> dict   (training metrics, empty dict if not ready)

Agents
------
  RandomAgent        — uniform random baseline
  CEMAgent           — Cross-Entropy Method (model-free, gradient-free)
  DDPGAgent          — Deep Deterministic Policy Gradient
  TD3Agent           — Twin Delayed DDPG
  SACAgent           — Soft Actor-Critic (entropy-regularised)
  ModelBasedAgent    — learned linear model + LQR/MPC planning
  SACxCEMAgent       — CEM explores, SAC exploits (hybrid)

All gradient-based agents optionally accept a HullDistanceModule for
potential-based reward shaping (plug in after hull is identified).

Shared utilities
----------------
  ReplayBuffer       — uniform experience replay
  make_agent(name, env) — factory used by benchmark.py
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, xDim: int = 4, uDim: int = 2):
        self.cap = capacity
        self.xDim = xDim; self.uDim = uDim
        self.s  = np.zeros((capacity, xDim), np.float32)
        self.a  = np.zeros((capacity, uDim), np.float32)
        self.r  = np.zeros(capacity,          np.float32)
        self.ns = np.zeros((capacity, xDim), np.float32)
        self.d  = np.zeros(capacity,          np.float32)
        self.ptr = self.size = 0

    def push(self, s, a, r, ns, done):
        i = self.ptr
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.ns[i]=ns; self.d[i]=float(done)
        self.ptr  = (i+1) % self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, batch: int) -> Tuple:
        idx = np.random.randint(0, self.size, batch)
        return (self.s[idx], self.a[idx], self.r[idx],
                self.ns[idx], self.d[idx])

    def __len__(self): return self.size


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM AGENT  (baseline)
# ══════════════════════════════════════════════════════════════════════════════

class RandomAgent:
    name = "random"
    def __init__(self, env): self.env = env
    def select_action(self, obs, explore=True): return self.env.action_space.sample()
    def push(self, *args): pass
    def update(self): return {}
    def set_hull_module(self, m): pass


# ══════════════════════════════════════════════════════════════════════════════
#  CEM AGENT  (gradient-free, good for short horizons)
# ══════════════════════════════════════════════════════════════════════════════

class CEMAgent:
    """
    Cross-Entropy Method with linear policy  u = K x + b.

    Maintains a Gaussian distribution over (K, b) parameters.
    Each update: sample N policies, evaluate top-k, refit Gaussian.
    Elite fraction determines selection pressure.
    """
    name = "cem"

    def __init__(self, env, population: int = 50, elite_frac: float = 0.2,
                 n_eval_steps: int = 200, noise_init: float = 0.5,
                 noise_floor: float = 0.01):
        xDim = env.xDim; uDim = env.uDim
        self.env    = env
        self.pop    = population
        self.elite  = max(2, int(population * elite_frac))
        self.T      = n_eval_steps
        self.dim    = uDim * (xDim + 1)   # K + b

        self.mu  = np.zeros(self.dim,  np.float32)
        self.std = np.full(self.dim, noise_init, np.float32)
        self.floor = noise_floor
        self._best_K = np.zeros((uDim, xDim), np.float32)
        self._best_b = np.zeros(uDim,         np.float32)
        self._xDim = xDim; self._uDim = uDim
        self._act_low  = env.act_low
        self._act_high = env.act_high
        self._hull_module = None

    def set_hull_module(self, m): self._hull_module = m

    def _unpack(self, theta):
        K = theta[:self._uDim*self._xDim].reshape(self._uDim, self._xDim)
        b = theta[self._uDim*self._xDim:]
        return K, b

    def _eval(self, theta):
        K, b = self._unpack(theta)
        obs, _ = self.env.reset()
        total  = 0.
        for _ in range(self.T):
            u   = np.clip(K @ obs + b, self._act_low, self._act_high)
            obs, r, done, _, _ = self.env.step(u)
            total += r
            if done: break
        return total

    def select_action(self, obs, explore=True):
        u = self._best_K @ obs + self._best_b
        if explore:
            u = u + np.random.randn(self._uDim).astype(np.float32) * 0.1
        return np.clip(u, self._act_low, self._act_high)

    def push(self, *args): pass   # CEM doesn't use a replay buffer

    def update(self) -> dict:
        thetas  = (self.mu + np.random.randn(self.pop, self.dim).astype(np.float32)
                   * self.std)
        returns = np.array([self._eval(t) for t in thetas])
        elite_i = np.argsort(returns)[-self.elite:]
        elites  = thetas[elite_i]
        self.mu  = elites.mean(0)
        self.std = np.maximum(elites.std(0), self.floor)
        self._best_K, self._best_b = self._unpack(self.mu)
        return {"cem_return": float(returns[elite_i].mean()),
                "cem_std":    float(self.std.mean())}


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    def _mlp(dims: List[int], activation=nn.ReLU, output_activation=None) -> nn.Sequential:
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            act = activation if i < len(dims)-2 else output_activation
            if act: layers.append(act())
        return nn.Sequential(*layers)

    class _Critic(nn.Module):
        def __init__(self, xDim, uDim, hidden=256):
            super().__init__()
            self.net = _mlp([xDim+uDim, hidden, hidden, 1])
        def forward(self, s, a):
            return self.net(torch.cat([s, a], -1)).squeeze(-1)

    class _TwinCritic(nn.Module):
        def __init__(self, xDim, uDim, hidden=256):
            super().__init__()
            self.q1 = _Critic(xDim, uDim, hidden)
            self.q2 = _Critic(xDim, uDim, hidden)
        def both(self, s, a): return self.q1(s,a), self.q2(s,a)
        def min(self, s, a):  return torch.min(self.q1(s,a), self.q2(s,a))

    class _DetActor(nn.Module):
        """Deterministic actor for DDPG/TD3."""
        def __init__(self, xDim, uDim, act_scale, act_bias, hidden=256):
            super().__init__()
            self.net  = _mlp([xDim, hidden, hidden, uDim], output_activation=nn.Tanh)
            self.register_buffer("scale", torch.tensor(act_scale, dtype=torch.float32))
            self.register_buffer("bias",  torch.tensor(act_bias,  dtype=torch.float32))
        def forward(self, s): return self.net(s) * self.scale + self.bias

    class _SACActorNet(nn.Module):
        """Stochastic actor for SAC — outputs (mu, log_std)."""
        LOG_STD_MIN = -5.; LOG_STD_MAX = 2.
        def __init__(self, xDim, uDim, act_scale, act_bias, hidden=256):
            super().__init__()
            self.net     = _mlp([xDim, hidden, hidden])
            self.mu_head = nn.Linear(hidden, uDim)
            self.ls_head = nn.Linear(hidden, uDim)
            self.register_buffer("scale", torch.tensor(act_scale, dtype=torch.float32))
            self.register_buffer("bias",  torch.tensor(act_bias,  dtype=torch.float32))

        def forward(self, s):
            h   = F.relu(self.net(s))
            mu  = self.mu_head(h)
            ls  = self.ls_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            return mu, ls

        def sample(self, s):
            mu, ls = self(s)
            std    = ls.exp()
            u      = mu + std * torch.randn_like(std)
            # tanh squash + log-prob correction
            log_pi = (-0.5*((u-mu)/std).pow(2) - ls
                      - 0.5*np.log(2*np.pi)).sum(-1)
            a      = torch.tanh(u)
            log_pi = log_pi - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
            return a * self.scale + self.bias, log_pi

        @torch.no_grad()
        def act(self, s_np, explore=True):
            s = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0)
            if explore:
                a, _ = self.sample(s)
            else:
                mu, _ = self(s); a = torch.tanh(mu) * self.scale + self.bias
            return a.squeeze(0).numpy()

    # ── shared training helpers ───────────────────────────────────────────────

    def _to_tensors(*arrays, device="cpu"):
        return [torch.tensor(a, dtype=torch.float32).to(device) for a in arrays]


# ══════════════════════════════════════════════════════════════════════════════
#  DDPG
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class DDPGAgent:
        name = "ddpg"

        def __init__(self, env, hidden=256, lr=3e-4, gamma=0.99, tau=0.005,
                     batch=256, buffer_cap=100_000, expl_noise=0.1, device="cpu"):
            xDim = env.xDim; uDim = env.uDim
            scale = (env.act_high - env.act_low) / 2.
            bias  = (env.act_high + env.act_low) / 2.
            self._device = torch.device(device)
            self._act_low  = env.act_low; self._act_high = env.act_high
            self._gamma = gamma; self._tau = tau; self._batch = batch
            self._noise = expl_noise
            self._buf   = ReplayBuffer(buffer_cap, xDim, uDim)
            self._hull_module = None

            self._actor        = _DetActor(xDim, uDim, scale, bias, hidden).to(self._device)
            self._actor_target = copy.deepcopy(self._actor)
            self._critic       = _Critic(xDim, uDim, hidden).to(self._device)
            self._critic_target= copy.deepcopy(self._critic)

            self._opt_a = optim.Adam(self._actor.parameters(),  lr=lr)
            self._opt_c = optim.Adam(self._critic.parameters(), lr=lr)

        def set_hull_module(self, m): self._hull_module = m

        def select_action(self, obs, explore=True):
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                a = self._actor(s).squeeze(0).cpu().numpy()
            if explore:
                a = a + np.random.randn(*a.shape).astype(np.float32) * self._noise
            return np.clip(a, self._act_low, self._act_high)

        def push(self, s, a, r, ns, done): self._buf.push(s, a, r, ns, done)

        def update(self) -> dict:
            if len(self._buf) < self._batch: return {}
            s,a,r,ns,d = self._buf.sample(self._batch)
            s,a,r_t,ns,d = _to_tensors(s,a,r,ns,d, device=self._device.type)

            # Hull shaping
            if self._hull_module:
                r_t = r_t + self._hull_module.reward_shaping(s, scale=0.5)

            with torch.no_grad():
                na  = self._actor_target(ns)
                qt  = r_t + self._gamma*(1-d)*self._critic_target(ns, na)
            qv    = self._critic(s, a)
            lc    = F.mse_loss(qv, qt)
            self._opt_c.zero_grad(); lc.backward(); self._opt_c.step()

            la    = -self._critic(s, self._actor(s)).mean()
            self._opt_a.zero_grad(); la.backward(); self._opt_a.step()

            for tp, p in zip(self._actor_target.parameters(), self._actor.parameters()):
                tp.data.mul_(1-self._tau).add_(self._tau*p.data)
            for tp, p in zip(self._critic_target.parameters(), self._critic.parameters()):
                tp.data.mul_(1-self._tau).add_(self._tau*p.data)

            return {"critic_loss": lc.item(), "actor_loss": la.item()}


# ══════════════════════════════════════════════════════════════════════════════
#  TD3
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class TD3Agent:
        name = "td3"

        def __init__(self, env, hidden=256, lr=3e-4, gamma=0.99, tau=0.005,
                     batch=256, buffer_cap=100_000, expl_noise=0.1,
                     policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                     device="cpu"):
            xDim = env.xDim; uDim = env.uDim
            scale = (env.act_high - env.act_low) / 2.
            bias  = (env.act_high + env.act_low) / 2.
            self._device = torch.device(device)
            self._act_low = env.act_low; self._act_high = env.act_high
            self._gamma   = gamma; self._tau = tau; self._batch = batch
            self._expl    = expl_noise; self._pnoise = policy_noise
            self._nclip   = noise_clip; self._delay  = policy_delay
            self._step    = 0
            self._buf     = ReplayBuffer(buffer_cap, xDim, uDim)
            self._hull_module = None

            self._actor        = _DetActor(xDim, uDim, scale, bias, hidden).to(self._device)
            self._actor_target = copy.deepcopy(self._actor)
            self._critic       = _TwinCritic(xDim, uDim, hidden).to(self._device)
            self._critic_target= copy.deepcopy(self._critic)

            self._opt_a = optim.Adam(self._actor.parameters(),  lr=lr)
            self._opt_c = optim.Adam(self._critic.parameters(), lr=lr)

        def set_hull_module(self, m): self._hull_module = m

        def select_action(self, obs, explore=True):
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                a = self._actor(s).squeeze(0).cpu().numpy()
            if explore:
                a = a + np.random.randn(*a.shape).astype(np.float32)*self._expl
            return np.clip(a, self._act_low, self._act_high)

        def push(self, s, a, r, ns, done): self._buf.push(s, a, r, ns, done)

        def update(self) -> dict:
            if len(self._buf) < self._batch: return {}
            self._step += 1
            s,a,r,ns,d = self._buf.sample(self._batch)
            s,a,r_t,ns,d = _to_tensors(s,a,r,ns,d, device=self._device.type)

            if self._hull_module:
                r_t = r_t + self._hull_module.reward_shaping(s, scale=0.5)

            with torch.no_grad():
                noise = (torch.randn_like(a)*self._pnoise).clamp(-self._nclip, self._nclip)
                na    = (self._actor_target(ns)+noise).clamp(
                         torch.tensor(self._act_low, device=self._device),
                         torch.tensor(self._act_high,device=self._device))
                qt = r_t + self._gamma*(1-d)*self._critic_target.min(ns, na)

            q1, q2 = self._critic.both(s, a)
            lc = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
            self._opt_c.zero_grad(); lc.backward(); self._opt_c.step()

            la = None
            if self._step % self._delay == 0:
                la = -self._critic.q1(s, self._actor(s)).mean()
                self._opt_a.zero_grad(); la.backward(); self._opt_a.step()
                for tp,p in zip(self._actor_target.parameters(), self._actor.parameters()):
                    tp.data.mul_(1-self._tau).add_(self._tau*p.data)
                for tp,p in zip(self._critic_target.parameters(), self._critic.parameters()):
                    tp.data.mul_(1-self._tau).add_(self._tau*p.data)

            return {"critic_loss": lc.item(),
                    "actor_loss":  la.item() if la else float("nan")}


# ══════════════════════════════════════════════════════════════════════════════
#  SAC
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class SACAgent:
        name = "sac"

        def __init__(self, env, hidden=256, lr=3e-4, gamma=0.99, tau=0.005,
                     alpha=0.2, auto_alpha=True, batch=256,
                     buffer_cap=100_000, device="cpu"):
            xDim = env.xDim; uDim = env.uDim
            scale = (env.act_high - env.act_low) / 2.
            bias  = (env.act_high + env.act_low) / 2.
            self._device = torch.device(device)
            self._act_low = env.act_low; self._act_high = env.act_high
            self._gamma = gamma; self._tau = tau; self._batch = batch
            self._buf   = ReplayBuffer(buffer_cap, xDim, uDim)
            self._hull_module = None

            self._actor  = _SACActorNet(xDim, uDim, scale, bias, hidden).to(self._device)
            self._critic = _TwinCritic(xDim, uDim, hidden).to(self._device)
            self._critic_target = copy.deepcopy(self._critic)
            self._opt_a  = optim.Adam(self._actor.parameters(),  lr=lr)
            self._opt_c  = optim.Adam(self._critic.parameters(), lr=lr)

            self._auto_alpha = auto_alpha
            if auto_alpha:
                self._log_alpha   = torch.zeros(1, requires_grad=True,
                                                 device=self._device)
                self._target_ent  = -float(uDim)
                self._opt_alpha   = optim.Adam([self._log_alpha], lr=lr)
                self._alpha       = self._log_alpha.exp().item()
            else:
                self._alpha = alpha

        def set_hull_module(self, m): self._hull_module = m

        @property
        def alpha(self): return self._alpha

        def select_action(self, obs, explore=True):
            return self._actor.act(obs.astype(np.float32), explore)

        def push(self, s, a, r, ns, done): self._buf.push(s, a, r, ns, done)

        def update(self) -> dict:
            if len(self._buf) < self._batch: return {}
            s,a,r,ns,d = self._buf.sample(self._batch)
            s,a,r_t,ns,d = _to_tensors(s,a,r,ns,d, device=self._device.type)

            if self._hull_module:
                r_t = r_t + self._hull_module.reward_shaping(s, scale=0.5)

            # Critic
            with torch.no_grad():
                na, lp = self._actor.sample(ns)
                qt     = r_t + self._gamma*(1-d)*(
                             self._critic_target.min(ns,na) - self._alpha*lp)
            q1,q2 = self._critic.both(s,a)
            lc    = F.mse_loss(q1,qt)+F.mse_loss(q2,qt)
            self._opt_c.zero_grad(); lc.backward(); self._opt_c.step()

            # Actor
            a_new, lp_new = self._actor.sample(s)
            qmin          = self._critic.min(s, a_new)
            la            = (self._alpha*lp_new - qmin).mean()
            self._opt_a.zero_grad(); la.backward(); self._opt_a.step()

            # Alpha
            l_alpha = None
            if self._auto_alpha:
                l_alpha = -(self._log_alpha*(lp_new+self._target_ent).detach()).mean()
                self._opt_alpha.zero_grad(); l_alpha.backward()
                self._opt_alpha.step()
                self._alpha = self._log_alpha.exp().item()

            for tp,p in zip(self._critic_target.parameters(), self._critic.parameters()):
                tp.data.mul_(1-self._tau).add_(self._tau*p.data)

            return {"critic_loss": lc.item(), "actor_loss": la.item(),
                    "alpha": self._alpha,
                    "alpha_loss": l_alpha.item() if l_alpha else float("nan")}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-BASED AGENT  (linear model identification + MPC)
# ══════════════════════════════════════════════════════════════════════════════

class ModelBasedAgent:
    """
    Identifies the linear model x_{t+1} = Â x_t + B̂ u_t online via least-squares,
    then plans with finite-horizon MPC minimising Σ‖x‖² subject to action bounds.

    No neural network — pure numpy. Fast and sample-efficient for truly linear systems.
    Works poorly when the env has significant noise or nonlinearity.
    """
    name = "model_based"

    def __init__(self, env, horizon: int = 20, n_rand_steps: int = 200,
                 reg: float = 1e-4):
        self.env       = env
        self.xDim      = env.xDim; self.uDim = env.uDim
        self.horizon   = horizon
        self.reg       = reg
        self._act_low  = env.act_low; self._act_high = env.act_high
        self._hull_module = None

        # Data for system identification
        self._X:  List[np.ndarray] = []   # states x_t
        self._U:  List[np.ndarray] = []   # actions u_t
        self._Xn: List[np.ndarray] = []   # next states x_{t+1}
        self._n_rand = n_rand_steps
        self._steps  = 0

        # Model estimate
        self._A_hat = np.eye(self.xDim, dtype=np.float32)
        self._B_hat = np.zeros((self.xDim, self.uDim), np.float32)
        self._model_ready = False

    def set_hull_module(self, m): self._hull_module = m

    def _fit_model(self):
        """Least-squares identification: [A|B] from (X, U) → Xn."""
        if len(self._X) < self.xDim + self.uDim + 2:
            return
        X  = np.array(self._X,  np.float32)   # (N, xDim)
        U  = np.array(self._U,  np.float32)   # (N, uDim)
        Xn = np.array(self._Xn, np.float32)   # (N, xDim)
        Z  = np.concatenate([X, U], 1)         # (N, xDim+uDim)
        # min ‖Z θ - Xn‖²_F  →  θ = (Z^T Z + λI)^{-1} Z^T Xn
        reg = self.reg * np.eye(Z.shape[1], dtype=np.float32)
        try:
            theta = np.linalg.solve(Z.T@Z + reg, Z.T@Xn)   # (xDim+uDim, xDim)
        except np.linalg.LinAlgError:
            return
        self._A_hat = theta[:self.xDim].T
        self._B_hat = theta[self.xDim:].T
        self._model_ready = True

    def _mpc(self, obs: np.ndarray) -> np.ndarray:
        """Finite-horizon MPC: minimise Σ‖x_t‖² via gradient descent on actions."""
        A, B = self._A_hat, self._B_hat
        T    = self.horizon
        # Initialise action sequence at zero
        Us   = np.zeros((T, self.uDim), np.float32)
        lr   = 0.05
        for _ in range(30):
            # Forward rollout
            xs = [obs.copy()]
            for t in range(T):
                xs.append(A @ xs[-1] + B @ Us[t])
            xs = np.array(xs)   # (T+1, xDim)
            # Gradient dJ/dU_t = B^T (2 Σ_{s≥t} dJ/dx_s)
            dJ_dx = 2. * xs[1:]   # (T, xDim)
            for t in range(T-1, 0, -1):
                dJ_dx[t-1] += A.T @ dJ_dx[t]
            dJ_dU = dJ_dx @ B    # (T, uDim)
            Us    = np.clip(Us - lr*dJ_dU, self._act_low, self._act_high)
        return Us[0]

    def select_action(self, obs, explore=True):
        if not self._model_ready or self._steps < self._n_rand:
            u = np.random.uniform(self._act_low, self._act_high).astype(np.float32)
        else:
            u = self._mpc(obs)
        if explore:
            u = u + np.random.randn(self.uDim).astype(np.float32)*0.05
        return np.clip(u, self._act_low, self._act_high)

    def push(self, s, a, r, ns, done):
        self._X.append(s); self._U.append(a); self._Xn.append(ns)
        self._steps += 1
        if self._steps % 50 == 0:
            self._fit_model()

    def update(self) -> dict:
        if self._model_ready:
            A_err = float(np.linalg.norm(self._A_hat))
            return {"A_hat_norm": A_err, "model_ready": 1.}
        return {"model_ready": 0.}


# ══════════════════════════════════════════════════════════════════════════════
#  SAC × CEM  (CEM explores, SAC exploits)
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class SACxCEMAgent:
        """
        Two-phase hybrid:
          • When hull not yet identified: CEM explores (gradient-free, diverse)
          • When hull identified:        SAC exploits (gradient-based, fine-grained)

        Transition triggered by is_hull_valid flag (set by the training loop
        after the monitor confirms a valid target).

        Both share a replay buffer so SAC can learn from CEM's exploration data.
        """
        name = "sac_x_cem"

        def __init__(self, env, **kwargs):
            self._sac = SACAgent(env, **{k:v for k,v in kwargs.items()
                                          if k not in ("population","elite_frac",
                                                        "n_eval_steps","noise_init")})
            self._cem = CEMAgent(env, **{k:v for k,v in kwargs.items()
                                          if k in ("population","elite_frac",
                                                    "n_eval_steps","noise_init")})
            self._use_sac = False

        def set_hull_valid(self, valid: bool): self._use_sac = valid
        def set_hull_module(self, m):
            self._sac.set_hull_module(m)
            self._cem.set_hull_module(m)

        def select_action(self, obs, explore=True):
            if self._use_sac:
                return self._sac.select_action(obs, explore)
            return self._cem.select_action(obs, explore)

        def push(self, s, a, r, ns, done):
            self._sac.push(s, a, r, ns, done)   # SAC buffer always fed

        def update(self) -> dict:
            metrics = {}
            if self._use_sac:
                metrics.update(self._sac.update())
            else:
                metrics.update(self._cem.update())
            return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_agent(name: str, env, **kwargs):
    """Used by benchmark.py. Returns an initialised agent."""
    registry = {"random": RandomAgent, "cem": CEMAgent,
                "model_based": ModelBasedAgent}
    if _TORCH:
        registry.update({"ddpg": DDPGAgent, "td3": TD3Agent,
                         "sac": SACAgent, "sac_x_cem": SACxCEMAgent})
    if name not in registry:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(registry)}")
    return registry[name](env, **kwargs)
