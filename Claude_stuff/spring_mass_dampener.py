# =============================================================================
# Spring-Mass-Dampener Q-Learning
# Refactored: bugs fixed, duplicates removed, FW as primary hull method
# =============================================================================

# ---- Colab setup (comment out if running locally) ---------------------------
# from google.colab import drive
# drive.mount('/content/drive')
# import os, shutil
# out_dir = './drive/MyDrive/Q_learning/Spring_mass_dampener'
# if os.path.exists(out_dir): shutil.rmtree(out_dir)
# os.makedirs(out_dir)
# !pip install control cvxpy

# ---- Imports ----------------------------------------------------------------
import os
import random
import warnings
from collections import deque

import control as ct                        # noqa: F401 (kept for caller use)
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.optimize import linprog
from tqdm import tqdm                       # noqa: F401

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import register

import matplotlib.pyplot as plt             # noqa: F401
from IPython.display import clear_output    # noqa: F401


# =============================================================================
# Section 1 – Convex-Hull Utilities
# Primary method: Frank-Wolfe (fw)
# Secondary:      cvxpy (distance-based QP)
# Tertiary:       linprog (scipy feasibility LP)
# =============================================================================

# ---------------------------------------------------------------------------
# 1a. Bounding-box fast reject
# ---------------------------------------------------------------------------

def bbox_reject(V: np.ndarray, x: np.ndarray, eps: float = 0.0) -> bool:
    """Return True when x is *outside* the axis-aligned bounding box of V."""
    V, x = np.asarray(V), np.asarray(x)
    return not np.all((x >= V.min(axis=0) - eps) & (x <= V.max(axis=0) + eps))


# ---------------------------------------------------------------------------
# 1b. Frank-Wolfe membership (randomised linear oracle)
# ---------------------------------------------------------------------------

def fw_membership(
    V: np.ndarray,
    x: np.ndarray,
    eps: float = 1e-6,
    max_iter: int = 3000,
    sample_size: int = 512,
    prune_tol: float = 1e-14,
    stall_iter: int = 200,
    verbose: bool = False,
    seed=None,
):
    """
    Minimise ||y - x||_2 over y in conv(V) via Frank-Wolfe with a randomised
    linear oracle.

    Returns
    -------
    inside : bool
    info   : dict  (residual, active_indices, weights, y)
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(V, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    m, n = V.shape
    assert x.shape == (n,), f"x must have shape ({n},), got {x.shape}"

    # initialise at the closest vertex
    i0 = int(np.argmin(np.sum((V - x) ** 2, axis=1)))
    y = V[i0].copy()
    idx = [i0]
    w = np.array([1.0])

    r = y - x
    res = float(np.linalg.norm(r))
    if res <= eps:
        return True, {"residual": res, "active_indices": idx, "weights": w, "y": y}

    best_res, no_improve = res, 0

    for t in range(1, max_iter + 1):
        cand = np.arange(m) if sample_size >= m else rng.integers(0, m, size=sample_size)
        j = int(cand[np.argmin(V[cand] @ r)])
        d = V[j] - y
        denom = float(np.dot(d, d))
        if denom <= 1e-30:
            break

        gamma = float(np.clip(-np.dot(r, d) / denom, 0.0, 1.0))
        y = y + gamma * d
        r = y - x
        res = float(np.linalg.norm(r))

        w *= (1.0 - gamma)
        if j in idx:
            w[idx.index(j)] += gamma
        else:
            idx.append(j)
            w = np.append(w, gamma)

        keep = w > prune_tol
        if not np.all(keep):
            w = w[keep]
            idx = [ii for ii, k in zip(idx, keep) if k]
            w /= w.sum()

        if verbose and (t % 100 == 0 or res <= eps):
            print(f"[FW] iter={t:4d}  res={res:.3e}  active={len(idx)}")

        if res <= eps:
            return True, {"residual": res, "active_indices": idx, "weights": w, "y": y}

        if res < best_res - 1e-12 * max(1.0, best_res):
            best_res, no_improve = res, 0
        else:
            no_improve += 1
            if no_improve >= stall_iter:
                break

    return False, {"residual": res, "active_indices": idx, "weights": w, "y": y}


def _certify_qp(
    V: np.ndarray,
    x: np.ndarray,
    active_indices,
    eps: float = 1e-6,
    solver: str = "OSQP",
):
    """
    Exact projection of x onto conv(V[active_indices]) via QP (certification
    step after FW stalls near the boundary).
    """
    V = np.asarray(V, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    VA = V[np.asarray(active_indices, dtype=int)]
    k = VA.shape[0]

    lam = cp.Variable(k, nonneg=True)
    obj = cp.Minimize(cp.sum_squares(VA.T @ lam - x))
    prob = cp.Problem(obj, [cp.sum(lam) == 1])
    prob.solve(solver=solver, verbose=False)

    if lam.value is None:
        return False, {"residual": np.inf, "lambda_active": None, "y": None}

    lam_val = np.asarray(lam.value).reshape(-1)
    y_val = VA.T @ lam_val
    res = float(np.linalg.norm(y_val - x))
    return (res <= eps), {"residual": res, "lambda_active": lam_val, "y": y_val}


def point_in_hull_fw(
    V: np.ndarray,
    x: np.ndarray,
    eps: float = 1e-6,
    bbox_eps: float = None,
    fw_max_iter: int = 3000,
    fw_sample_size: int = 512,
    certify: bool = True,
    certify_solver: str = "OSQP",
    uncertain_margin: float = None,
    verbose: bool = False,
    seed=None,
):
    """
    Scalable convex-hull membership test (Frank-Wolfe + optional QP certification).

    Pipeline
    --------
    1. Bounding-box fast reject.
    2. Randomised Frank-Wolfe approximate solve.
    3. If uncertain and ``certify=True``: exact QP on the FW active set.

    Returns
    -------
    inside : bool
    info   : dict
    """
    V = np.asarray(V)
    x = np.asarray(x)
    if bbox_eps is None:
        bbox_eps = eps
    if uncertain_margin is None:
        uncertain_margin = 5.0

    if bbox_reject(V, x, eps=bbox_eps):
        return False, {"stage": "bbox_reject", "residual": None,
                       "active_indices": [], "weights_active": None}

    inside_fw, fw_info = fw_membership(
        V, x, eps=eps, max_iter=fw_max_iter, sample_size=fw_sample_size,
        verbose=verbose, seed=seed,
    )

    if inside_fw:
        return True, {"stage": "fw", "residual": fw_info["residual"],
                      "active_indices": fw_info["active_indices"],
                      "weights_active": fw_info["weights"]}

    if not certify or fw_info["residual"] > uncertain_margin * eps:
        stage = "fw_no_certify" if not certify else "fw_confident_outside"
        return False, {"stage": stage, "residual": fw_info["residual"],
                       "active_indices": fw_info["active_indices"],
                       "weights_active": fw_info["weights"]}

    inside_qp, qp_info = _certify_qp(
        V, x, fw_info["active_indices"], eps=eps, solver=certify_solver,
    )
    return inside_qp, {"stage": "certify_qp", "residual": qp_info["residual"],
                       "active_indices": fw_info["active_indices"],
                       "weights_active": qp_info["lambda_active"]}


# ---------------------------------------------------------------------------
# 1c. CVXPY distance-based membership (secondary method)
# ---------------------------------------------------------------------------

def _hull_distance_cvxpy(
    V: np.ndarray,
    x: np.ndarray,
    solver=cp.SCS,
) -> float:
    """Euclidean distance from x to conv(V). Returns np.inf on solver failure."""
    V = np.asarray(V, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n_v = V.shape[0]
    lam = cp.Variable(n_v, nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(V.T @ lam - x)),
        [cp.sum(lam) == 1],
    )
    try:
        prob.solve(solver=solver, verbose=False)
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float(np.sqrt(max(prob.value, 0.0)))
    except cp.error.SolverError:
        pass
    return np.inf


def point_in_hull_cvxpy(
    V: np.ndarray,
    x: np.ndarray,
    tolerance: float = 1e-6,
    solver=cp.SCS,
) -> bool:
    """Check membership via CVXPY distance projection (with bbox pre-filter)."""
    V = np.asarray(V)
    x = np.asarray(x)
    if V.size == 0:
        return False
    lo, hi = V.min(axis=0), V.max(axis=0)
    if not np.all((x >= lo - tolerance) & (x <= hi + tolerance)):
        return False
    return _hull_distance_cvxpy(V, x, solver=solver) <= tolerance


# ---------------------------------------------------------------------------
# 1d. linprog feasibility membership (tertiary method)
# ---------------------------------------------------------------------------

def point_in_hull_linprog(
    V: np.ndarray,
    x: np.ndarray,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check membership via scipy linprog feasibility.
    Uses a zero objective (pure feasibility); returns True iff feasible.
    """
    V = np.asarray(V, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if V.size == 0:
        return False
    lo, hi = V.min(axis=0), V.max(axis=0)
    if not np.all((x >= lo - tolerance) & (x <= hi + tolerance)):
        return False
    n_v = V.shape[0]
    c = np.zeros(n_v)                               # FIX: was random
    A = np.r_[V.T, np.ones((1, n_v))]
    b = np.r_[x, 1.0]
    result = linprog(c, A_eq=A, b_eq=b,
                     bounds=[(0, 1)] * n_v, method="highs")
    return bool(result.success and result.status == 0)


# ---------------------------------------------------------------------------
# 1e. Unified dispatch
# ---------------------------------------------------------------------------

def point_in_hull(
    V: np.ndarray,
    x: np.ndarray,
    method: str = "fw",
    tolerance: float = 1e-6,
    fw_sample_size: int = 512,
    fw_max_iter: int = 3000,
    certify: bool = True,
    certify_solver: str = "OSQP",
    cvxpy_solver=cp.SCS,
    seed=None,
) -> bool:
    """
    Unified hull membership dispatch.

    Parameters
    ----------
    method : 'fw' | 'cvxpy' | 'linprog'
    """
    if method == "fw":
        inside, _ = point_in_hull_fw(
            V, x, eps=tolerance,
            fw_max_iter=fw_max_iter, fw_sample_size=fw_sample_size,
            certify=certify, certify_solver=certify_solver,
            uncertain_margin=np.sqrt(max(V.shape[1], 1)),
            seed=seed,
        )
        return inside
    elif method == "cvxpy":
        return point_in_hull_cvxpy(V, x, tolerance=tolerance, solver=cvxpy_solver)
    elif method == "linprog":
        return point_in_hull_linprog(V, x, tolerance=tolerance)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fw', 'cvxpy', or 'linprog'.")


def batch_point_in_hull(
    V: np.ndarray,
    points: np.ndarray,
    method: str = "fw",
    tolerance: float = 1e-6,
    **kwargs,
) -> np.ndarray:
    """Vectorised wrapper around ``point_in_hull`` for a batch of points."""
    return np.array(
        [point_in_hull(V, p, method=method, tolerance=tolerance, **kwargs)
         for p in points],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# 1f. Full-dimensionality check
# ---------------------------------------------------------------------------

def is_hull_full_dimensional(V: np.ndarray, tol: float = 1e-7) -> bool:
    """
    True iff conv(V) is full-dimensional (rank test on vertex differences).

    FIX: original code transposed the diff matrix incorrectly.
    """
    V = np.asarray(V)
    if V.ndim != 2 or V.shape[0] == 0:
        return False
    n_v, dim = V.shape
    if n_v < dim + 1:
        return False
    Diff = V[1:, :] - V[0, :]              # FIX: was V.T[1:,:] – V.T[0,:]
    return int(np.linalg.matrix_rank(Diff, tol=tol)) == dim


def _fw_direction_check(V, point, v, n_v, tol):
    """Single-direction feasibility sub-problem (used for full-dim check)."""
    lam = cp.Variable(n_v, nonneg=True)
    r = cp.Variable(1)
    prob = cp.Problem(
        cp.Minimize(-r),
        [cp.sum(lam) == 1, V.T @ lam == point + r * v],
    )
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except cp.error.SolverError:
        return False
    return prob.status == cp.OPTIMAL and float(r.value) > tol


def is_hull_full_dimensional_cvxpy(
    V: np.ndarray,
    tol: float = 1e-7,
    n_jobs: int = 1,
) -> bool:
    """
    Full-dimensionality check via CVXPY sub-problems (2·dim directions).
    Set n_jobs=-1 for parallel execution via joblib.
    """
    V = np.asarray(V)
    if V.ndim != 2 or V.shape[0] < V.shape[1] + 1:
        return False
    n_v, dim = V.shape
    point = V.mean(axis=0)

    directions = []
    for i in range(2 * dim):
        v = np.zeros(dim)
        v[i % dim] = (-1) ** (i // dim)
        directions.append(v)

    if n_jobs == 1:
        results = [_fw_direction_check(V, point, v, n_v, tol) for v in directions]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fw_direction_check)(V, point, v, n_v, tol)
            for v in directions
        )
    return all(results)


# ---------------------------------------------------------------------------
# 1g. Target validity check
# ---------------------------------------------------------------------------

def is_target_valid(
    target: np.ndarray,
    u: np.ndarray,
    u_min,
    u_max,
    tol: float = 1e-7,
    solver: str = "SCS",
) -> bool:
    """
    Check if a trajectory (target) is a valid target set given input sequence u
    and per-channel input bounds u_min / u_max (arrays or scalars).

    FIX: eps is now a scalar variable; u_min/u_max support per-channel arrays;
         is_valid no longer unconditionally reset to False.
    """
    target = np.asarray(target)         # shape (n, T)
    u = np.asarray(u)                   # shape (m, T-1)

    X_ = target[:, :-1]                 # (n, T-1)
    X_1 = target[:, 1:]                 # (n, T-1)
    X_diff = X_ - target[:, -1].reshape(-1, 1)
    n, T = target.shape
    m = u.shape[0]
    n_x = X_diff.shape[1]

    # broadcast scalar bounds to arrays
    u_min = np.full(m, -np.inf) if np.isscalar(u_min) and u_min == -np.inf \
        else np.broadcast_to(np.asarray(u_min, dtype=float), (m,)).copy()
    u_max = np.full(m, np.inf) if np.isscalar(u_max) and u_max == np.inf \
        else np.broadcast_to(np.asarray(u_max, dtype=float), (m,)).copy()

    lambda_1 = cp.Variable(n_x, nonneg=True)
    lambda_2 = cp.Variable(n_x)
    eps = cp.Variable(nonneg=True)          # FIX: scalar, not vector
    u_1 = cp.Variable(m)

    holder = np.zeros((n,))
    z = cp.Variable(m + n)

    constraints = [
        z == np.vstack([X_, u]) @ lambda_2,
        X_diff @ lambda_1 - X_1 @ lambda_2 == 0,
        z[:n] == holder,
        z[n:] == u_1,
        cp.sum(lambda_1) <= 1,
        lambda_1 - eps >= 0,
        lambda_2 >= 0,
    ]
    for i in range(m):
        if not np.isinf(u_min[i]):
            constraints.append(u_min[i] <= u_1[i] + u[i, -1])
        if not np.isinf(u_max[i]):
            constraints.append(u_1[i] + u[i, -1] <= u_max[i])

    prob = cp.Problem(cp.Minimize(-eps), constraints)
    try:
        prob.solve(solver=solver, verbose=False)
    except cp.error.SolverError:
        return False

    return prob.status == cp.OPTIMAL and float(eps.value) > tol


# ---------------------------------------------------------------------------
# 1h. Polytope helper class
# ---------------------------------------------------------------------------

class Polytope:
    """
    Convex polytope defined by its vertex set.

    Methods
    -------
    is_full_dimensional()
    is_point_inside(point, method='fw', ...)
    batch_is_inside(points, method='fw', ...)
    find_barycentric_coordinates(point)
    """

    def __init__(self, vertices: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if self.vertices.ndim != 2:
            raise ValueError("vertices must be a 2-D array (n_vertices, dim).")
        self.n_vertices, self.dim = self.vertices.shape

    def is_full_dimensional(self, tol: float = 1e-15) -> bool:
        return is_hull_full_dimensional(self.vertices, tol=tol)

    def is_point_inside(
        self,
        point: np.ndarray,
        method: str = "fw",
        tolerance: float = 1e-6,
        fw_sample_size: int = 64,
        fw_max_iter: int = 10_000,
        cvxpy_solver=cp.SCS,
    ) -> bool:
        point = np.asarray(point)
        if point.shape != (self.dim,):
            raise ValueError(f"point must have shape ({self.dim},).")
        return point_in_hull(
            self.vertices, point, method=method, tolerance=tolerance,
            fw_sample_size=fw_sample_size, fw_max_iter=fw_max_iter,
            uncertain_margin=np.sqrt(self.dim),
            cvxpy_solver=cvxpy_solver,
        )

    def batch_is_inside(
        self,
        points: np.ndarray,
        method: str = "fw",
        tolerance: float = 1e-6,
        fw_sample_size: int = 64,
        fw_max_iter: int = 10_000,
        cvxpy_solver=cp.SCS,
    ) -> np.ndarray:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(f"points must have shape (N, {self.dim}).")
        return batch_point_in_hull(
            self.vertices, points, method=method, tolerance=tolerance,
            fw_sample_size=fw_sample_size, fw_max_iter=fw_max_iter,
            cvxpy_solver=cvxpy_solver,
        )

    def find_barycentric_coordinates(self, point: np.ndarray):
        """
        Return barycentric coordinates of point w.r.t. vertices, or None if
        infeasible.  Uses a zero-cost feasibility LP (not random objective).
        """
        point = np.asarray(point, dtype=np.float64)
        n_v = self.n_vertices
        c = np.zeros(n_v)                   # FIX: was random
        A = np.r_[self.vertices.T, np.ones((1, n_v))]
        b = np.r_[point, 1.0]
        result = linprog(c, A_eq=A, b_eq=b,
                         bounds=[(0, 1)] * n_v, method="highs")
        return result.x if (result.success and result.status == 0) else None


# =============================================================================
# Section 2 – Gymnasium Environment
# =============================================================================

class LinearUpdateEnv(gym.Env):
    """
    Linear dynamical system environment for Q-learning / RL.

        x_{t+1} = A x_t + B u_t

    The agent is rewarded for exploring new regions (expanding the convex hull
    of visited states) and penalised for leaving the state bounds.

    Parameters
    ----------
    A, B            : system matrices
    state_bounds    : (low, high) arrays
    action_bounds   : (low, high) arrays
    alpha           : EMA coefficient for the running centre
    max_steps_per_episode : episode horizon
    rewards         : (step_penalty, boundary_penalty, goal_reward_scale, goal_reward)
    target          : optional pre-specified target polytope (n_points, xDim)
    hull_method     : 'fw' | 'cvxpy' | 'linprog'
    hull_tolerance  : membership tolerance
    fw_sample_size  : FW oracle sample size
    fw_max_iter     : FW iteration cap
    rewards_type    : 'dense' | 'sparse'
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        A,
        B,
        state_bounds,
        action_bounds,
        alpha: float = 0.9,
        max_steps_per_episode: int = 100,
        rewards: tuple = (-0.01, 0.01, -1.0, 1.0),
        target=None,
        hull_method: str = "fw",
        cvxpy_solver=None,
        fw_eps: float = 1e-6,
        fw_sample_size: int = 64,
        fw_max_iter: int = 10_000,
        hull_tolerance: float = 1e-6,
        rewards_type: str = "dense",
    ):
        super().__init__()

        self.A = np.asarray(A, dtype=np.float32)
        self.B = np.asarray(B, dtype=np.float32)
        self.xDim = self.A.shape[0]
        self.uDim = self.B.shape[1]

        self.alpha = alpha
        self.max_steps = max_steps_per_episode
        self.rewards = rewards
        self.rewards_type = rewards_type

        # Spaces
        self.observation_space = spaces.Box(
            low=np.asarray(state_bounds[0], dtype=np.float32),
            high=np.asarray(state_bounds[1], dtype=np.float32),
            shape=(self.xDim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(action_bounds[0], dtype=np.float32),
            high=np.asarray(action_bounds[1], dtype=np.float32),
            shape=(self.uDim,), dtype=np.float32,
        )
        self.action_bounds = action_bounds

        # Hull membership settings
        self.hull_method = hull_method
        self.cvxpy_solver = cvxpy_solver if cvxpy_solver is not None else cp.SCS
        self.fw_eps = fw_eps
        self.fw_sample_size = fw_sample_size
        self.fw_max_iter = fw_max_iter
        self.hull_tolerance = hull_tolerance

        # Target set
        self._init_target = None if target is None else np.asarray(target)
        self._has_preset_target = target is not None

        # Runtime state (initialised in reset)
        self.state = None
        self.steps = 0
        self.auto = 0
        self.history = None
        self.Input = None
        self.Hist_Input = None
        self.center = None
        self._state_min = None
        self._state_max = None
        self.target = None
        self.is_valid_target = False
        self._target_stats_ready = False

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.clip(
            self.np_random.normal(0, 1, self.observation_space.shape),
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

        self.history = np.zeros((self.max_steps + 1, self.xDim), dtype=np.float32)
        self.history[0] = self.state
        self.Input = np.zeros((self.max_steps, self.uDim), dtype=np.float32)
        self.Hist_Input = None

        self.center = self.state.copy()
        self._state_min = self.state.copy()
        self._state_max = self.state.copy()

        self.steps = 0
        self.auto = 0

        # restore preset target (if any) and reset derived stats
        self.target = None if self._init_target is None else self._init_target.copy()
        self.is_valid_target = self._has_preset_target
        self._target_stats_ready = False

        return self.state.copy(), self._make_info()

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.Input[self.steps] = action

        next_state = (
            self.A @ self.state.reshape(-1, 1) + self.B @ action.reshape(-1, 1)
        ).flatten()

        in_bounds = self.observation_space.contains(next_state)

        if self.is_valid_target:
            reward = self._reward_with_target(next_state, in_bounds)
        else:
            reward = self._reward_explore(next_state, in_bounds)

        self.state = next_state.astype(np.float32)
        self.steps += 1

        # guard against index overflow (FIX: off-by-one)
        if self.steps <= self.max_steps:
            self.history[self.steps] = self.state

        terminated = self.auto != 0
        truncated = (self.steps >= self.max_steps) and not terminated

        return self.state.copy(), float(reward), terminated, truncated, self._make_info()

    def sample_action(self):
        return np.clip(
            self.np_random.normal(0, 1, self.action_space.shape),
            self.action_bounds[0], self.action_bounds[1],
        )

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_info(self):
        return {
            "NMRDP": self.is_valid_target,
            "auto": self.auto,
            "center": self.center.copy(),
        }

    def _in_hull(self, points_hull: np.ndarray, query: np.ndarray) -> bool:
        """Route hull membership query through the configured method."""
        return point_in_hull(
            points_hull, query,
            method=self.hull_method,
            tolerance=self.hull_tolerance,
            fw_sample_size=self.fw_sample_size,
            fw_max_iter=self.fw_max_iter,
            cvxpy_solver=self.cvxpy_solver,
        )

    def _append_hist_input(self):
        current = self.Input[:self.steps].copy()
        self.Hist_Input = current if self.Hist_Input is None \
            else np.vstack([self.Hist_Input, current])

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_explore(self, next_state: np.ndarray, in_bounds: bool) -> float:
        """
        Reward used when no valid target is set yet (exploration phase).
        FIX: is_valid_target is set correctly and never immediately overwritten.
        """
        # Update running statistics
        self.center = (1 - self.alpha) * self.center + self.alpha * self.state
        self._state_min = np.minimum(self._state_min, self.state)
        self._state_max = np.maximum(self._state_max, self.state)

        if not in_bounds:
            self.auto = 2
            return float(self.rewards[2])

        visited = self.history[: self.steps + 1]
        in_goal = self._in_hull(visited, next_state)

        if in_goal:
            self.auto = 1
            self._append_hist_input()

            # Build candidate target trajectory
            cand = self.history.copy()
            # FIX: only write if within bounds
            if self.steps + 1 <= self.max_steps:
                cand[self.steps + 1] = next_state
            cand = cand[: self.steps + 2]

            new_valid = False
            if self.steps >= self.xDim + 1:
                new_valid = is_target_valid(
                    cand.T,
                    self.Input[: self.steps + 1].T,
                    self.action_bounds[0],
                    self.action_bounds[1],
                )
            # FIX: assign result; don't unconditionally set False
            self.is_valid_target = new_valid

            if self.target is None:
                self.target = cand.copy()
            else:
                self.target = np.vstack([self.target, self.history[: self.steps + 1]])

            return float(self.rewards[3])

        distance = np.linalg.norm(next_state - self.center)
        if self.rewards_type == "dense":
            return float(self.rewards[0]) * distance / np.sqrt(self.xDim)
        return float(self.rewards[0])

    def _reward_with_target(self, next_state: np.ndarray, in_bounds: bool) -> float:
        """Reward used when a valid target set is available."""
        # Lazily compute target statistics once per episode
        if not self._target_stats_ready:
            self.center = np.mean(self.target, axis=0)
            self._state_min = np.min(self.target, axis=0)
            self._state_max = np.max(self.target, axis=0)
            self._target_stats_ready = True

        distance = np.linalg.norm(next_state - self.center)

        if not in_bounds:
            self.auto = 2
            # FIX: was using self.rewards[1] instead of penalty index
            return float(self.rewards[1]) / (1 + np.exp(-0.5 * distance ** 2))

        in_goal = self._in_hull(self.target, next_state)

        if in_goal:
            self.auto = 1
            self._append_hist_input()
            self.target = np.vstack(
                [self.target, self.history[: self.steps + 1]]
            )
            return float(self.rewards[2]) * (1 + np.exp(-distance ** 2))

        return float(self.rewards[0]) / (1 + np.exp(-distance ** 2))


# =============================================================================
# Section 3 – Neural Network Architectures
# =============================================================================

# ---------------------------------------------------------------------------
# 3a. MLP (DQN backbone)
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """
    Simple MLP with geometrically shrinking hidden layers.
    hidden_size is rounded up to the next power of 2 (≥ 8).
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        h = int(2 ** (max(int(np.floor(np.log2(hidden_size))), 3) + 1))
        self.model = nn.Sequential(
            nn.Linear(input_size, h),      nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2),          nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, h // 4),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 4, h // 8),     nn.ReLU(),
            nn.Linear(h // 8, output_size),
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# 3b. GRU Actor
# ---------------------------------------------------------------------------

class RNNActor(nn.Module):
    """
    GRU-based actor.  Action bounds are stored as registered buffers so they
    move with the model to the correct device.

    FIX: bare ``except:`` replaced with ``except RuntimeError``.
    """

    def __init__(
        self, input_size, hidden_size, action_size,
        num_layers: int = 1,
        action_low=None, action_high=None,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        _low = action_low if action_low is not None \
            else [-float("inf")] * action_size
        _high = action_high if action_high is not None \
            else [float("inf")] * action_size

        self.register_buffer("action_low",
                             torch.tensor(_low, dtype=torch.float32).to(device))
        self.register_buffer("action_high",
                             torch.tensor(_high, dtype=torch.float32).to(device))

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        self.fc0 = nn.Linear(hidden_size * num_layers, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size, 2 * action_size), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2 * action_size, action_size),
        )

    def forward(self, state, hidden):
        out, hidden = self.rnn(state, hidden)
        try:
            out = self.fc(out.permute(1, 0, 2).flatten(start_dim=1))
        except RuntimeError:                    # FIX: no bare except
            out = self.fc(out[-1, :])
        return torch.clamp(out, self.action_low, self.action_high), hidden

    def init_hidden(self, batch_size: int):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# ---------------------------------------------------------------------------
# 3c. Transformer Actors
# ---------------------------------------------------------------------------

class _TransformerActorBase(nn.Module):
    """Shared embedding + encoder scaffold for Transformer actors."""

    def __init__(self, input_size, hidden_size, action_size,
                 num_layers=1, nhead=4, dim_feedforward=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, action_size)

    def init_hidden(self, batch_size: int):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class TransformerActor(_TransformerActorBase):
    """
    Transformer actor that carries a single recurrent context token (first
    hidden-layer token) across time steps.
    """

    def forward(self, state, hidden):
        # state: (1, batch, input_size)  hidden: (num_layers, batch, hidden_size)
        batch_size = state.shape[1]
        emb = self.state_embedding(state.squeeze(0)).unsqueeze(1)   # (B,1,H)
        ctx = hidden[0].unsqueeze(1)                                  # (B,1,H)
        out = self.transformer_encoder(torch.cat([ctx, emb], dim=1)) # (B,2,H)
        q = self.fc(out[:, 0, :])
        new_hidden = out[:, 0, :].unsqueeze(0).expand(self.num_layers, -1, -1)
        return q, new_hidden


class TransformerActorAllHidden(_TransformerActorBase):
    """
    Transformer actor that propagates all num_layers hidden tokens.
    """

    def forward(self, state, hidden):
        batch_size = state.shape[1]
        emb = self.state_embedding(state.squeeze(0)).unsqueeze(1)    # (B,1,H)
        ctx = hidden.permute(1, 0, 2)                                 # (B,L,H)
        out = self.transformer_encoder(torch.cat([emb, ctx], dim=1)) # (B,1+L,H)
        q = self.fc(out[:, 0, :])
        new_hidden = out[:, 1:, :].permute(1, 0, 2)                  # (L,B,H)
        return q, new_hidden


class TransformerActorConfigurableHidden(_TransformerActorBase):
    """
    Transformer actor with configurable number of propagated hidden layers.
    output_hidden_layers_to_propagate=0  →  no recurrence (feedforward)
    output_hidden_layers_to_propagate=k  →  propagate last k layers
    """

    def __init__(self, input_size, hidden_size, action_size,
                 num_layers=1, nhead=4, dim_feedforward=256,
                 output_hidden_layers_to_propagate=None):
        super().__init__(input_size, hidden_size, action_size,
                         num_layers, nhead, dim_feedforward)
        k = num_layers if output_hidden_layers_to_propagate is None \
            else output_hidden_layers_to_propagate
        if not (0 <= k <= num_layers):
            raise ValueError(
                f"output_hidden_layers_to_propagate must be in [0, {num_layers}]."
            )
        self.k = k

    def forward(self, state, hidden):
        emb = self.state_embedding(state.squeeze(0)).unsqueeze(1)    # (B,1,H)
        ctx = hidden.permute(1, 0, 2)                                 # (B,L,H)
        out = self.transformer_encoder(torch.cat([emb, ctx], dim=1)) # (B,1+L,H)
        q = self.fc(out[:, 0, :])

        new_hidden = torch.zeros_like(hidden)
        if self.k > 0:
            updated = out[:, 1: 1 + self.k, :].permute(1, 0, 2)     # (k,B,H)
            new_hidden[self.num_layers - self.k:] = updated
        return q, new_hidden


# ---------------------------------------------------------------------------
# 3d. Critic networks
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    MLP critic that optionally ingests the actor's hidden state.

    FIX: fc0 is always created when hidden_size > 0, regardless of actor_layers,
    avoiding AttributeError when actor_layers == 0.
    """

    def __init__(self, state_size, hidden_size, action_size, actor_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.actor_layers = actor_layers

        input_dim = state_size + action_size
        if hidden_size > 0:
            # FIX: always define fc0 when hidden_size > 0
            fc0_in = hidden_size * max(actor_layers, 1)
            self.fc0 = nn.Linear(fc0_in, hidden_size)
            input_dim += hidden_size

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, state, action, hidden=None):
        if state.ndim == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim == 3 and action.shape[1] == 1:
            action = action.squeeze(1)

        if self.hidden_size > 0:
            if hidden is None:
                raise ValueError("hidden required when hidden_size > 0.")
            if hidden.ndim == 3:
                h = F.relu(self.fc0(hidden.permute(1, 0, 2).flatten(start_dim=1)))
            elif hidden.ndim == 2:
                h = F.relu(self.fc0(hidden))
            else:
                raise ValueError(f"Unexpected hidden shape: {hidden.shape}")
            x = torch.cat([state, action, h], dim=1)
        else:
            x = torch.cat([state, action], dim=1)

        return self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))


class RNNCritic(nn.Module):
    """GRU-based critic that processes (state, action, actor_hidden) jointly."""

    def __init__(self, state_size, action_size, critic_hidden_size,
                 actor_hidden_size, num_critic_layers: int = 1):
        super().__init__()
        self.critic_hidden_size = critic_hidden_size
        self.num_critic_layers = num_critic_layers

        rnn_in = state_size + action_size + actor_hidden_size
        self.rnn = nn.GRU(rnn_in, critic_hidden_size, num_critic_layers, batch_first=True)
        self.fc1 = nn.Linear(critic_hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state, action, actor_hidden, critic_hidden):
        if state.ndim == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        if action.ndim == 3 and action.shape[1] == 1:
            action = action.squeeze(1)

        if actor_hidden.ndim == 3:
            ah = actor_hidden.permute(1, 0, 2).mean(dim=1)
        else:
            ah = actor_hidden

        rnn_in = torch.cat([state, action, ah], dim=1).unsqueeze(1)
        rnn_out, new_hidden = self.rnn(rnn_in, critic_hidden)
        x = rnn_out.squeeze(1)
        return self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))), new_hidden

    def init_hidden(self, batch_size: int, device: str = "cpu"):
        return torch.zeros(self.num_critic_layers, batch_size,
                           self.critic_hidden_size, device=device)


# =============================================================================
# Section 4 – Replay Memory
# =============================================================================

class ReplayMemory:
    """Standard experience replay buffer."""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, auto, action, next_state, next_auto,
             reward, terminated, truncated):
        self.memory.append((state, auto, action, next_state, next_auto,
                            reward, terminated, truncated))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryWithCenter(ReplayMemory):
    """Replay buffer that also stores the running centre of states."""

    def push(self, state, auto, center, action, next_state, next_auto,
             next_center, reward, terminated, truncated):
        self.memory.append((state, center, auto, action, next_state,
                            next_auto, next_center, reward, terminated, truncated))


class ReplayMemoryWithHidden(ReplayMemory):
    """Replay buffer that also stores actor hidden states."""

    def push(self, state, hidden, auto, action, next_state, next_hidden,
             next_auto, reward, terminated, truncated):
        self.memory.append((state, hidden, auto, action, next_state,
                            next_hidden, next_auto, reward, terminated, truncated))


class SuccessReplayMemory:
    """
    Two-stage replay buffer: experiences accumulate in a temporary buffer and
    are only committed to the main buffer when ``push_success()`` is called
    (i.e. on successful episode completion).
    """

    def __init__(self, capacity: int, temp_capacity: int):
        self.memory = deque(maxlen=capacity)
        self.temp = deque(maxlen=temp_capacity)

    # ---- temporary buffer ---------------------------------------------------
    def push(self, *args):
        self.temp.append(args)

    def flush(self):
        self.temp.clear()

    def push_success(self):
        """Commit all temporary experiences to the main buffer."""
        self.memory.extend(self.temp)
        self.flush()

    # ---- main buffer --------------------------------------------------------
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SuccessReplayMemoryWithCenter(SuccessReplayMemory):
    pass   # same interface, different tuple layout — push(*args) handles it


class SuccessReplayMemoryWithHidden(SuccessReplayMemory):
    pass


# =============================================================================
# Section 5 – Utilities
# =============================================================================

def create_directory(path: str):
    """Create directory, silently skip if it already exists."""
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")


# =============================================================================
# Section 6 – Environment Registration & Smoke Test
# =============================================================================

try:
    register(
        id="LinearUpdateEnv-v0",
        entry_point="__main__:LinearUpdateEnv",
    )
    print("LinearUpdateEnv-v0 registered.")
except Exception as e:
    print(f"Registration skipped (already exists or error): {e}")


if __name__ == "__main__":
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.05], [0.1]])
    state_bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
    action_bounds = (np.array([-1.0]), np.array([1.0]))

    for method in ("fw", "cvxpy", "linprog"):
        print(f"\n── Smoke test: hull_method='{method}' ──")
        env = LinearUpdateEnv(
            A=A, B=B,
            state_bounds=state_bounds,
            action_bounds=action_bounds,
            hull_method=method,
            hull_tolerance=1e-5,
        )
        obs, info = env.reset()
        print(f"  reset → {obs},  info={info}")
        for i in range(5):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            print(f"  step {i+1}: r={r:.4f}  term={term}  trunc={trunc}")
            if term or trunc:
                print("  episode ended early.")
                break
