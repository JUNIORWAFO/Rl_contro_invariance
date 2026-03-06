"""
optimization.py
================
Frank-Wolfe projector onto conv(hull_pts).

Why Frank-Wolfe instead of scipy SLSQP for projection?
  The linear oracle over a convex hull is trivial:
    s* = argmin_{p ∈ hull_pts}  ⟨p, g⟩  =  hull_pts[argmin_i p_i · g]
  One dot-product scan — O(M) — no matrix factorisation.
  SLSQP solves an O(M²) QP each call. For M ≥ 30 FW is faster;
  for M ≥ 100 it's an order of magnitude faster.
  FW converges at O(1/t) for ½‖·‖² objectives (strongly convex),
  and the line search is exact (closed form for quadratic objectives).

Public API
----------
  FrankWolfeProjector
    .solve(x, hull_pts) → OptResult          single projection
    .batch_project(xs, hull_pts) → (N, D)   batch projections
    .distance(x, hull_pts) → float

  OptResult
    .x          projected point (D,)
    .distance   ‖x_query − x_proj‖
    .iters      number of FW iterations
    .converged  bool
    .gap_hist   list of duality gaps per iteration (convergence certificate)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class OptResult:
    x:         np.ndarray    # projected point
    distance:  float         # ‖query − projection‖
    iters:     int
    converged: bool
    gap_hist:  List[float] = field(default_factory=list)


class FrankWolfeProjector:
    """
    Project query point x onto conv(hull_pts) using Frank-Wolfe.

    Algorithm (with exact line search for ½‖y−x‖²):
      y₀ = mean(hull_pts)
      for t = 0, 1, ...:
          g    = y_t − x                         # gradient of ½‖y−x‖²
          s    = hull_pts[argmin_i ⟨pᵢ, g⟩]     # linear oracle  O(M)
          gap  = ⟨g, y_t − s⟩                   # FW duality gap
          if gap < tol: converged
          γ*   = ⟨g, y_t−s⟩ / ‖y_t−s‖²         # exact line search
          y_{t+1} = y_t + γ* (s − y_t)

    The duality gap is a convergence certificate:
      gap ≥ ½‖y_t − x*‖²   (objective suboptimality)
    so gap < tol guarantees we are within tol of the true projection.

    Parameters
    ----------
    max_iter : int     maximum FW iterations
    tol      : float   duality gap threshold for early stopping
    """

    def __init__(self, max_iter: int = 300, tol: float = 1e-7):
        self.max_iter = max_iter
        self.tol      = tol

    def solve(self, x: np.ndarray, hull_pts: np.ndarray) -> OptResult:
        x  = np.asarray(x,        dtype=np.float64)
        P  = np.asarray(hull_pts, dtype=np.float64)   # (M, D)

        # Initialise at centroid
        y        = P.mean(0).copy()
        gap_hist = []

        for t in range(self.max_iter):
            g    = y - x                        # gradient
            dots = P @ g                         # (M,)  inner products
            s    = P[dots.argmin()]              # linear oracle

            gap  = float(g @ (y - s))           # FW duality gap ≥ 0
            gap_hist.append(gap)

            if gap < self.tol:
                return OptResult(y.astype(np.float32),
                                 float(np.linalg.norm(y-x)),
                                 t+1, True, gap_hist)

            # Exact line search for ½‖y−x‖²:  γ* = ⟨g, y−s⟩ / ‖y−s‖²
            d     = y - s
            denom = float(np.dot(d, d))
            gamma = float(np.clip(np.dot(g, d) / denom, 0., 1.)) \
                    if denom > 1e-14 else 1.

            y = y + gamma * (s - y)

        return OptResult(y.astype(np.float32),
                         float(np.linalg.norm(y-x)),
                         self.max_iter, False, gap_hist)

    def distance(self, x: np.ndarray, hull_pts: np.ndarray) -> float:
        return self.solve(x, hull_pts).distance

    def batch_project(self, xs: np.ndarray,
                       hull_pts: np.ndarray) -> np.ndarray:
        """xs: (N, D) → projected points (N, D)"""
        return np.vstack([self.solve(xi, hull_pts).x for xi in xs])

    def batch_distance(self, xs: np.ndarray,
                        hull_pts: np.ndarray) -> np.ndarray:
        """xs: (N, D) → distances (N,)"""
        return np.array([self.distance(xi, hull_pts) for xi in xs],
                        dtype=np.float32)


if __name__ == "__main__":
    import time
    np.random.seed(0)
    xDim, M = 4, 50
    pts = np.random.randn(M, xDim).astype(np.float32)

    # Outside point
    x_out = np.array([5., 5., 5., 5.], np.float32)

    fw  = FrankWolfeProjector(max_iter=300, tol=1e-7)
    t0  = time.perf_counter()
    res = fw.solve(x_out, pts)
    ms  = (time.perf_counter() - t0) * 1e3

    print(f"FW projection:  dist={res.distance:.6f}  iters={res.iters}"
          f"  converged={res.converged}  ({ms:.2f} ms)")
    print(f"Final gap: {res.gap_hist[-1]:.2e}")

    # Compare with scipy SLSQP
    try:
        from distance_to_hull import ExactLPDistance
        exact = ExactLPDistance()
        t0    = time.perf_counter()
        d_ex  = exact(x_out, pts)
        ms_ex = (time.perf_counter()-t0)*1e3
        print(f"SLSQP exact:    dist={d_ex:.6f}  ({ms_ex:.2f} ms)")
        print(f"Error vs SLSQP: {abs(res.distance-d_ex):.2e}")
    except ImportError:
        pass

    # Batch test
    xs  = np.random.randn(10, xDim).astype(np.float32) * 3.
    dists = fw.batch_distance(xs, pts)
    print(f"Batch distances (10 points): {dists.round(4)}")
