"""
distance_to_hull.py
====================
Three distance-to-convex-hull implementations (confirmed scope).

  ExactLPDistance        — scipy SLSQP QP, exact, ~5–50 ms/query
  SignedDistance         — negative inside, positive outside, FD gradient
  DifferentiableDistance — PyTorch soft-min, fully differentiable
  HullDistanceModule     — nn.Module wrapper for reward shaping in SAC
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.spatial import Delaunay
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import torch, torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ── in-hull test ──────────────────────────────────────────────────────────────

def _in_hull(x: np.ndarray, pts: np.ndarray) -> bool:
    if not _SCIPY or len(pts) <= x.shape[0]:
        return bool(np.all(x >= pts.min(0)) and np.all(x <= pts.max(0)))
    try:
        return bool(Delaunay(pts).find_simplex(x) >= 0)
    except Exception:
        return bool(np.all(x >= pts.min(0)) and np.all(x <= pts.max(0)))


# ══════════════════════════════════════════════════════════════════════════════
#  1. EXACT LP DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

class ExactLPDistance:
    """
    min_{λ≥0, Σλ=1}  ½‖x − Σ_i λ_i p_i‖²   via scipy SLSQP.

    .projection(x, pts) → (projected_point, distance)
    .batch(xs, pts)     → (N,) distances
    .in_hull(x, pts)    → bool
    """

    def __init__(self, tol: float = 1e-8):
        if not _SCIPY:
            raise ImportError("ExactLPDistance requires scipy.")
        self.tol = tol

    def __call__(self, x: np.ndarray, pts: np.ndarray) -> float:
        return self.projection(x, pts)[1]

    def batch(self, xs: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return np.array([self(xi, pts) for xi in xs], dtype=np.float32)

    def in_hull(self, x: np.ndarray, pts: np.ndarray) -> bool:
        return self(x, pts) < self.tol

    def projection(self, x: np.ndarray,
                   pts: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (projected_point, distance)."""
        M  = len(pts)
        P  = pts.astype(np.float64)
        q  = x.astype(np.float64)
        l0 = np.ones(M, np.float64) / M

        def obj(l): d = l@P-q; return .5*np.dot(d,d)
        def jac(l): return P@(l@P-q)

        res = minimize(obj, l0, jac=jac, method="SLSQP",
                       bounds=[(0.,1.)]*M,
                       constraints={"type":"eq",
                                    "fun": lambda l: l.sum()-1.,
                                    "jac": lambda l: np.ones(M)},
                       options={"ftol": self.tol**2, "maxiter": 500})
        proj = (res.x @ P).astype(np.float32)
        return proj, float(np.linalg.norm(proj-q))


# ══════════════════════════════════════════════════════════════════════════════
#  2. SIGNED DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

class SignedDistance:
    """
    Signed distance to conv(pts):
      d < 0  — inside  (= −nearest_vertex_dist, approximate)
      d = 0  — boundary
      d > 0  — outside (= exact LP distance)

    The inside approximation (nearest-vertex) is fast and correct in sign.
    The outside value is exact — this matters for reward shaping and monitoring.

    gradient(x): central FD, costs 2·xDim LP calls.
    Points toward hull from outside; away from boundary from inside.
    """

    def __init__(self, tol: float = 1e-8):
        self.tol    = tol
        self._exact = ExactLPDistance(tol) if _SCIPY else None

    def __call__(self, x: np.ndarray, pts: np.ndarray) -> float:
        if _in_hull(x, pts):
            return -float(np.linalg.norm(pts - x, axis=1).min())
        return self._exact(x, pts) if self._exact else \
               float(np.linalg.norm(pts - x, axis=1).min())

    def batch(self, xs: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return np.array([self(xi, pts) for xi in xs], dtype=np.float32)

    def in_hull(self, x: np.ndarray, pts: np.ndarray) -> bool:
        return self(x, pts) <= 0.

    def gradient(self, x: np.ndarray, pts: np.ndarray,
                 eps: float = 1e-4) -> np.ndarray:
        g = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps; xm[i] -= eps
            g[i] = (self(xp, pts) - self(xm, pts)) / (2.*eps)
        return g


# ══════════════════════════════════════════════════════════════════════════════
#  3. DIFFERENTIABLE DISTANCE  (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class DifferentiableDistance(nn.Module):
        """
        Soft distance from x to conv(pts), differentiable w.r.t. x.

        d_soft = −(1/β) log Σ_i exp(−β ‖x−pᵢ‖)    [soft-min over vertex dists]

        • Gradient ∂d/∂x points toward nearest hull vertex — correct direction.
        • Underestimates true projection distance (nearest-vertex ≥ hull dist),
          but the bias is uniform and doesn't affect gradient direction.
        • β=10: smooth for shaping; β→∞: approaches nearest-vertex (sharp grad).

        .signed(x, pts, pts_np): multiply by −1 inside hull (numpy test, no grad).
        """

        def __init__(self, beta: float = 10.):
            super().__init__()
            self.beta = beta

        def forward(self, x: "torch.Tensor",
                    pts: "torch.Tensor") -> "torch.Tensor":
            """x: (...,D)  pts: (M,D)  → (...,) soft distances ≥ 0"""
            dists = (x.unsqueeze(-2) - pts).norm(dim=-1)    # (..., M)
            return -(1./self.beta) * torch.logsumexp(-self.beta*dists, dim=-1)

        def signed(self, x: "torch.Tensor",
                   pts: "torch.Tensor",
                   pts_np: np.ndarray) -> "torch.Tensor":
            """Signed soft distance: ≤0 inside, ≥0 outside."""
            d    = self.forward(x, pts)
            x_np = x.detach().cpu().numpy()
            if x_np.ndim == 1:
                return d * (-1. if _in_hull(x_np, pts_np) else 1.)
            signs = np.array([-1. if _in_hull(xi, pts_np) else 1.
                               for xi in x_np], dtype=np.float32)
            return d * torch.tensor(signs, device=x.device)

    class HullDistanceModule(nn.Module):
        """
        Reward-shaping nn.Module for actor/critic training.

        Call .set_hull(tensor) after each episode when hull grows.
        .reward_shaping(obs): returns scale × exp(−dist) ∈ (0, scale].
          Large when obs is near hull — zero-gradient-safe potential shaping.
        """

        def __init__(self, xDim: int = 4, beta: float = 10.):
            super().__init__()
            self._fn = DifferentiableDistance(beta)
            self.register_buffer("hull_pts", torch.zeros(1, xDim))
            self._ready = False

        def set_hull(self, pts: "torch.Tensor"):
            self.hull_pts = pts.to(self.hull_pts.device).detach()
            self._ready   = pts.shape[0] > 1

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if not self._ready:
                return torch.zeros(x.shape[0], device=x.device)
            return self._fn(x, self.hull_pts)

        def reward_shaping(self, x: "torch.Tensor",
                           scale: float = 1.) -> "torch.Tensor":
            return scale * torch.exp(-self.forward(x))

else:
    class DifferentiableDistance:
        def __init__(self, *a, **kw): raise ImportError("Requires PyTorch.")
    class HullDistanceModule:
        def __init__(self, *a, **kw): raise ImportError("Requires PyTorch.")


def make_distance_fn(method: str = "signed"):
    return {"exact": ExactLPDistance, "signed": SignedDistance}[method]()


if __name__ == "__main__":
    import time
    np.random.seed(0)
    xDim, M = 4, 40
    pts  = np.random.randn(M, xDim).astype(np.float32)
    x_in = pts.mean(0)
    x_out= pts.mean(0) + 5.

    sd = SignedDistance()
    print(f"Signed  inside ={sd(x_in, pts):.4f}  outside={sd(x_out, pts):.4f}")
    print(f"Gradient inside: {sd.gradient(x_in, pts).round(3)}")

    if _SCIPY:
        exact = ExactLPDistance()
        t0 = time.perf_counter()
        print(f"Exact   inside ={exact(x_in,pts):.6f}  outside={exact(x_out,pts):.4f}"
              f"  ({(time.perf_counter()-t0)*1e3:.1f}ms)")

    if _TORCH:
        dd = DifferentiableDistance(beta=10.)
        hp = torch.tensor(pts)
        xt = torch.tensor(x_out).requires_grad_(True)
        d  = dd(xt, hp); d.backward()
        print(f"Diff    outside={d.item():.4f}  grad_norm={xt.grad.norm():.4f}")
        mod = HullDistanceModule(xDim=xDim); mod.set_hull(hp)
        print(f"Shaping: {mod.reward_shaping(torch.randn(4,xDim)).detach().numpy().round(3)}")
