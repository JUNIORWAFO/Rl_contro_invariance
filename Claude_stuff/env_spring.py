"""
env_spring.py — Spring-Mass-Damper environment, gymnasium-compatible.

System dynamics (continuous, Euler-discretised):
    m ẍ + c ẋ + k x = u

State:  [x, ẋ]  (position, velocity)
Action: [u]     (force input)

Fits the LinearUpdateEnv interface in env.py:
    x_{t+1} = A x_t + B u_t

Factory
-------
make_spring_env(m, c, k, dt)  → LinearUpdateEnv (from env.py)
make_vec_spring_env(n_envs)   → SyncVectorEnv   (for PPO / vectorised training)

Hull-aware reward shaping (dense) is wired in via reward_fn so the env
stays compatible with every monitor in hull_monitors.py and every agent
in rl_algorithms.py.
"""
from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

# Import the shared LinearUpdateEnv from the pipeline
from env import LinearUpdateEnv, make_continuous_env


# ══════════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def spring_reward_fn(
    state:  np.ndarray,
    action: np.ndarray,
    *,
    pos_weight:  float = 1.0,
    vel_weight:  float = 0.1,
    act_weight:  float = 0.01,
) -> float:
    """
    Dense quadratic reward for the spring system.

        r = -(pos_weight·x² + vel_weight·ẋ² + act_weight·u²)

    Goal: drive state to origin with minimal force.
    Compatible with LinearUpdateEnv.reward_fn signature: fn(state, action) → float.
    """
    x, xdot = float(state[0]), float(state[1])
    u = float(action[0])
    return -(pos_weight * x**2 + vel_weight * xdot**2 + act_weight * u**2)


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def spring_matrices(
    m: float = 1.0,
    c: float = 0.3,
    k: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time matrices for:  m ẍ + c ẋ + k x = u

        Ac = [[0,      1  ],
              [-k/m,  -c/m]]

        Bc = [[0  ],
              [1/m]]
    """
    Ac = np.array([[0.0,    1.0  ],
                   [-k/m,  -c/m ]], dtype=np.float32)
    Bc = np.array([[0.0 ],
                   [1/m ]], dtype=np.float32)
    return Ac, Bc


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY — single env
# ══════════════════════════════════════════════════════════════════════════════

def make_spring_env(
    m: float = 1.0,
    c: float = 0.3,
    k: float = 1.0,
    dt: float = 0.05,
    max_steps: int = 200,
    x0_scale: float = 2.0,
    act_bound: float = 5.0,
    noise_std: float = 0.0,
    reward_fn: Optional[Callable] = None,
    **kwargs,
) -> LinearUpdateEnv:
    """
    Build a Spring-Mass-Damper LinearUpdateEnv.

    Parameters
    ----------
    m, c, k     : mass, damping, stiffness
    dt          : Euler integration step
    max_steps   : episode horizon
    x0_scale    : initial state ‖x₀‖ scale (gaussian)
    act_bound   : symmetric action bound  [-act_bound, +act_bound]
    noise_std   : process noise  σ (0 = deterministic)
    reward_fn   : override default spring_reward_fn

    Returns
    -------
    LinearUpdateEnv  (from env.py) — drop-in compatible with the full pipeline.
    """
    Ac, Bc = spring_matrices(m, c, k)
    fn = reward_fn if reward_fn is not None else spring_reward_fn

    act_low  = np.array([-act_bound], dtype=np.float32)
    act_high = np.array([ act_bound], dtype=np.float32)

    return make_continuous_env(
        Ac, Bc,
        dt=dt,
        max_steps=max_steps,
        x0_dist="gaussian",
        x0_scale=x0_scale,
        noise_std=noise_std,
        act_low=act_low,
        act_high=act_high,
        reward_fn=fn,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY — vectorised env  (make_vec_env)
# ══════════════════════════════════════════════════════════════════════════════

def make_vec_spring_env(
    n_envs: int = 4,
    m: float = 1.0,
    c: float = 0.3,
    k: float = 1.0,
    dt: float = 0.05,
    max_steps: int = 200,
    x0_scale: float = 2.0,
    act_bound: float = 5.0,
    noise_std: float = 0.0,
    reward_fn: Optional[Callable] = None,
) -> SyncVectorEnv:
    """
    Build a SyncVectorEnv of n_envs spring environments.

    SyncVectorEnv runs all envs in a single process; use AsyncVectorEnv for
    CPU-bound workloads where n_envs > os.cpu_count().

    Each sub-env gets a different seed so trajectories are independent.

    Returns
    -------
    gymnasium.vector.SyncVectorEnv
        .observation_space  : Box(n_envs, 2)
        .action_space       : Box(n_envs, 1)
        .reset()            → obs (n_envs, 2), infos
        .step(actions)      → obs, rewards, terminateds, truncateds, infos
    """
    def _make(seed: int):
        def _init():
            env = make_spring_env(
                m=m, c=c, k=k, dt=dt,
                max_steps=max_steps,
                x0_scale=x0_scale,
                act_bound=act_bound,
                noise_std=noise_std,
                reward_fn=reward_fn,
            )
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [_make(seed=i) for i in range(n_envs)]
    return SyncVectorEnv(env_fns)


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Single env ──")
    env = make_spring_env()
    obs, _ = env.reset(seed=0)
    print(f"  {env}")
    print(f"  obs₀ = {obs}")
    for t in range(4):
        obs, r, done, trunc, _ = env.step(env.action_space.sample())
        print(f"  t={t+1}  obs={obs}  r={r:.4f}")

    print("\n── Vec env (4 parallel) ──")
    venv = make_vec_spring_env(n_envs=4)
    obs, _ = venv.reset()
    print(f"  obs shape = {obs.shape}")
    actions = venv.action_space.sample()
    obs, rews, terms, truncs, _ = venv.step(actions)
    print(f"  step obs={obs.shape}  rewards={rews.round(3)}")
    venv.close()
