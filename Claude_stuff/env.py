"""
env.py — LinearUpdateEnv, gymnasium-compatible.

x_{t+1} = A x_t + B u_t + w_t   (discrete)
dx/dt   = Ac x + Bc u            (continuous, Euler-discretised)

Factory functions
-----------------
make_env(A, B)                  user-supplied matrices
make_random_stable_env(xDim, uDim, rho)
make_double_integrator(dt)
make_continuous_env(Ac, Bc, dt)
make_benchmark_envs()           dict of named envs for benchmark.py
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np

try:
    from gymnasium import spaces
    _GYM = True
except ImportError:
    _GYM = False


# ── Minimal Box shim when gymnasium is absent ──────────────────────────────

class _Box:
    def __init__(self, low, high):
        self.low   = np.asarray(low,  np.float32)
        self.high  = np.asarray(high, np.float32)
        self.shape = self.low.shape
    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════

class LinearUpdateEnv:
    """
    Discrete-time linear system with optional continuous-time origin.

    Parameters
    ----------
    A, B        : system matrices (xDim×xDim, xDim×uDim)
    x0_dist     : 'zero' | 'gaussian' | 'uniform'
    x0_scale    : scale of initial-state noise
    noise_std   : process noise σ  (0 = deterministic)
    act_low/high: action bounds
    max_steps   : episode horizon
    target      : (M, xDim) hull target points for HullMonitor, or None
    continuous  : treat A as Ac and apply Euler integration with step dt
    dt          : Euler step size
    reward_fn   : callable(state, action) → float, or None  (default: -‖x‖²)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        A:           np.ndarray,
        B:           np.ndarray,
        x0_dist:     str   = "gaussian",
        x0_scale:    float = 1.0,
        noise_std:   float = 0.0,
        act_low:     Optional[np.ndarray] = None,
        act_high:    Optional[np.ndarray] = None,
        max_steps:   int   = 200,
        target:      Optional[np.ndarray] = None,
        continuous:  bool  = False,
        dt:          float = 0.05,
        reward_fn:   Optional[Callable] = None,
    ):
        self.A_raw    = np.asarray(A, np.float32)
        self.B_raw    = np.asarray(B, np.float32)
        self.xDim     = self.A_raw.shape[0]
        self.uDim     = self.B_raw.shape[1]
        self.x0_dist  = x0_dist
        self.x0_scale = x0_scale
        self.noise_std= noise_std
        self.max_steps= max_steps
        self.target   = (np.asarray(target, np.float32) if target is not None else None)
        self.continuous = continuous
        self.dt       = dt
        self.reward_fn= reward_fn

        # Effective discrete matrices
        if continuous:
            I         = np.eye(self.xDim, dtype=np.float32)
            self.A    = I + dt * self.A_raw
            self.B    = dt * self.B_raw
        else:
            self.A    = self.A_raw
            self.B    = self.B_raw

        # Action bounds
        self.act_low  = (act_low  if act_low  is not None
                         else np.full(self.uDim, -1., np.float32))
        self.act_high = (act_high if act_high is not None
                         else np.full(self.uDim,  1., np.float32))
        self.act_low  = np.asarray(self.act_low,  np.float32)
        self.act_high = np.asarray(self.act_high, np.float32)

        # Gym-compatible spaces
        inf = np.full(self.xDim, np.inf, np.float32)
        if _GYM:
            self.observation_space = spaces.Box(-inf, inf, dtype=np.float32)
            self.action_space      = spaces.Box(self.act_low, self.act_high,
                                                dtype=np.float32)
        else:
            self.observation_space = _Box(-inf, inf)
            self.action_space      = _Box(self.act_low, self.act_high)

        self._state: np.ndarray = np.zeros(self.xDim, np.float32)
        self._steps: int  = 0
        self._done:  bool = True

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
               options: Optional[dict]  = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        if self.x0_dist == "zero":
            x0 = np.zeros(self.xDim, np.float32)
        elif self.x0_dist == "uniform":
            x0 = np.random.uniform(-self.x0_scale, self.x0_scale,
                                    self.xDim).astype(np.float32)
        else:
            x0 = (np.random.randn(self.xDim) * self.x0_scale).astype(np.float32)
        self._state = x0
        self._steps = 0
        self._done  = False
        return self._state.copy(), {}

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Call reset() before step().")
        u   = np.clip(action, self.act_low, self.act_high).astype(np.float32)
        w   = (np.random.randn(self.xDim).astype(np.float32) * self.noise_std
               if self.noise_std > 0 else 0.)
        xn  = self.A @ self._state + self.B @ u + w
        r   = (float(self.reward_fn(self._state, u)) if self.reward_fn
               else -float(np.dot(self._state, self._state)))
        self._state = xn.astype(np.float32)
        self._steps += 1
        done = self._steps >= self.max_steps
        self._done = done
        return self._state.copy(), r, done, False, {}

    def get_state(self) -> np.ndarray: return self._state.copy()
    def set_state(self, x: np.ndarray): self._state = np.asarray(x, np.float32)

    def __repr__(self):
        mode = "continuous" if self.continuous else "discrete"
        rho  = float(np.abs(np.linalg.eigvals(self.A)).max())
        return (f"LinearUpdateEnv(xDim={self.xDim}, uDim={self.uDim}, "
                f"mode={mode}, ρ={rho:.3f}, T={self.max_steps})")


# ── Factory functions ─────────────────────────────────────────────────────────

def make_env(A: np.ndarray, B: np.ndarray, **kw) -> LinearUpdateEnv:
    return LinearUpdateEnv(A, B, **kw)


def make_random_stable_env(
    xDim: int = 4, uDim: int = 2,
    spectral_radius: float = 0.90,
    seed: Optional[int] = None, **kw,
) -> LinearUpdateEnv:
    """Random stable discrete-time system with ρ(A) = spectral_radius."""
    if seed is not None: np.random.seed(seed)
    A = np.random.randn(xDim, xDim).astype(np.float32)
    r = np.abs(np.linalg.eigvals(A)).max()
    if r > 1e-8: A *= spectral_radius / r
    B = np.random.randn(xDim, uDim).astype(np.float32)
    B /= np.linalg.norm(B, axis=0, keepdims=True).clip(1e-8)
    return LinearUpdateEnv(A, B, **kw)


def make_double_integrator(dt: float = 0.1, **kw) -> LinearUpdateEnv:
    """2D double integrator: state=[pos, vel], control=[accel]."""
    Ac = np.array([[0., 1.], [0., 0.]], np.float32)
    Bc = np.array([[0.], [1.]], np.float32)
    return LinearUpdateEnv(Ac, Bc, continuous=True, dt=dt, **kw)


def make_continuous_env(Ac: np.ndarray, Bc: np.ndarray,
                         dt: float = 0.05, **kw) -> LinearUpdateEnv:
    """Euler-discretised continuous system dx/dt = Ac x + Bc u."""
    return LinearUpdateEnv(Ac, Bc, continuous=True, dt=dt, **kw)


def make_benchmark_envs() -> Dict[str, LinearUpdateEnv]:
    """Named env suite used by benchmark.py."""
    return {
        "stable_2d":        make_random_stable_env(2, 1, 0.85, seed=0,
                                max_steps=200, x0_scale=2.0),
        "stable_4d":        make_random_stable_env(4, 2, 0.90, seed=1,
                                max_steps=200, x0_scale=2.0),
        "near_unstable_4d": make_random_stable_env(4, 2, 0.98, seed=2,
                                max_steps=200, x0_scale=2.0),
        "double_integrator":make_double_integrator(dt=0.1, max_steps=200,
                                x0_scale=2.0,
                                act_low=np.array([-2.], np.float32),
                                act_high=np.array([2.], np.float32)),
        "stable_6d":        make_random_stable_env(6, 3, 0.88, seed=3,
                                max_steps=200, x0_scale=2.0),
    }


if __name__ == "__main__":
    for name, env in make_benchmark_envs().items():
        obs, _ = env.reset(seed=0)
        print(f"  {name:<22} {env}")
