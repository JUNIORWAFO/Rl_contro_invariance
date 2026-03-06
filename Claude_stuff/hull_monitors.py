"""
hull_monitors.py
=================
Six hull monitor architectures, all sharing the same interface:

    monitor.process(record: EpisodeRecord) -> HullResult
    monitor.full_reset()

Architectures
-------------
  OracleMonitor          — exact LP feasibility every episode (ground truth)
  PostEpisodeHullMonitor — LP scan, caches target hull across episodes
  HERMonitor             — Hindsight Experience Replay style relabeling
  AsyncMonitor           — oracle runs every N episodes, interpolates between
  CommitteeMonitor       — majority vote across multiple sub-monitors
  LearnedMonitor         — thin wrapper; delegates to a HullClassifier

Protocol  (for type-checking in benchmark.py)
---------
  class MonitorProtocol(Protocol):
      def process(self, record: EpisodeRecord) -> HullResult: ...
      def full_reset(self): ...
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from shared_types import EpisodeRecord, HullResult

try:
    from scipy.optimize import linprog
    from scipy.spatial import Delaunay
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── shared hull-membership primitive ─────────────────────────────────────────

def _in_hull_lp(x: np.ndarray, pts: np.ndarray, tol: float = 1e-8) -> bool:
    """Exact LP in-hull test.  Returns True iff x ∈ conv(pts)."""
    if not _SCIPY or len(pts) == 0:
        return False
    M = len(pts)
    A_eq = np.vstack([pts.T, np.ones((1, M))])
    b_eq = np.append(x, 1.)
    res  = linprog(np.zeros(M), A_eq=A_eq, b_eq=b_eq,
                   bounds=[(0, None)]*M, method="highs",
                   options={"disp": False})
    return res.status == 0


def _scan_episode(states: np.ndarray, target: np.ndarray,
                   tol: float = 1e-8) -> np.ndarray:
    """Return boolean mask (T,): True at steps where state ∈ conv(target)."""
    T = len(states)
    if len(target) == 0:
        return np.zeros(T, bool)
    mask = np.array([_in_hull_lp(s, target, tol) for s in states])
    return mask


def _first_true(mask: np.ndarray) -> Optional[int]:
    idxs = np.where(mask)[0]
    return int(idxs[0]) if len(idxs) else None


# ══════════════════════════════════════════════════════════════════════════════
#  1. ORACLE MONITOR  (ground truth, expensive)
# ══════════════════════════════════════════════════════════════════════════════

class OracleMonitor:
    """Runs exact LP feasibility test every step, every episode. Ground truth."""

    name = "oracle"

    def __init__(self, target: Optional[np.ndarray] = None, tol: float = 1e-8):
        self.target          = (np.asarray(target, np.float32)
                                if target is not None else None)
        self.tol             = tol
        self.is_valid_target = target is not None

    def process(self, record: EpisodeRecord) -> HullResult:
        if self.target is None or not self.is_valid_target:
            return HullResult(np.zeros(record.T, bool), None, False, self.name)
        mask  = _scan_episode(record.states[:-1], self.target, self.tol)
        first = _first_true(mask)
        return HullResult(mask, first, True, self.name)

    def full_reset(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  2. POST-EPISODE HULL MONITOR  (caches target, LP scan)
# ══════════════════════════════════════════════════════════════════════════════

class PostEpisodeHullMonitor:
    """
    Discovers and caches the hull target incrementally.

    On each episode, checks whether any visited state could extend the known
    hull. Once a valid target is established, runs LP scan each episode.
    Cheaper than OracleMonitor because it reuses the cached hull.
    """

    name = "post_episode"

    def __init__(self, target: Optional[np.ndarray] = None,
                 tol: float = 1e-8,
                 min_hull_pts: int = 4):
        self.target          = (np.asarray(target, np.float32)
                                if target is not None else None)
        self.tol             = tol
        self.min_hull_pts    = min_hull_pts
        self.is_valid_target = target is not None
        self._ep_count       = 0
        self._hull_buffer: List[np.ndarray] = (
            list(target) if target is not None else [])

    def process(self, record: EpisodeRecord) -> HullResult:
        self._ep_count += 1

        # Accumulate visited states into hull buffer
        for s in record.states:
            self._hull_buffer.append(s.copy())
        # Keep at most 500 points (most recent)
        if len(self._hull_buffer) > 500:
            self._hull_buffer = self._hull_buffer[-500:]

        if (not self.is_valid_target
                and len(self._hull_buffer) >= self.min_hull_pts):
            self.target          = np.array(self._hull_buffer, np.float32)
            self.is_valid_target = True

        if not self.is_valid_target:
            return HullResult(np.zeros(record.T, bool), None, False, self.name)

        mask  = _scan_episode(record.states[:-1], self.target, self.tol)
        first = _first_true(mask)
        return HullResult(mask, first, self.is_valid_target, self.name)

    def full_reset(self):
        self.target          = None
        self.is_valid_target = False
        self._hull_buffer    = []
        self._ep_count       = 0


# ══════════════════════════════════════════════════════════════════════════════
#  3. HER MONITOR  (Hindsight Experience Replay style)
# ══════════════════════════════════════════════════════════════════════════════

class HERMonitor:
    """
    HER-style monitor: treats the final state of each episode as a
    hindsight goal and relabels the episode retrospectively.

    For each episode, the 'hull' is the convex hull of all episode-final states
    seen so far. This creates an expanding target that any episode can hit.

    Strategy: 'final' — use final state as hindsight goal (like HER)
              'future' — use a random future state in the same episode
    """

    name = "her"

    def __init__(self, strategy: str = "final",
                 min_goals: int = 3, tol: float = 1e-8):
        self.strategy        = strategy
        self.min_goals       = min_goals
        self.tol             = tol
        self._goals: List[np.ndarray] = []
        self.target          = None
        self.is_valid_target = False

    def process(self, record: EpisodeRecord) -> HullResult:
        # Hindsight goal for this episode
        if self.strategy == "final":
            goal = record.states[-1].copy()
        else:  # future
            t    = np.random.randint(0, record.T)
            goal = record.states[t].copy()

        self._goals.append(goal)
        if len(self._goals) > 200:
            self._goals = self._goals[-200:]

        if len(self._goals) >= self.min_goals:
            self.target          = np.array(self._goals, np.float32)
            self.is_valid_target = True

        if not self.is_valid_target:
            return HullResult(np.zeros(record.T, bool), None, False, self.name)

        mask  = _scan_episode(record.states[:-1], self.target, self.tol)
        first = _first_true(mask)
        return HullResult(mask, first, True, self.name)

    def full_reset(self):
        self._goals          = []
        self.target          = None
        self.is_valid_target = False


# ══════════════════════════════════════════════════════════════════════════════
#  4. ASYNC MONITOR  (oracle every N episodes, interpolate between)
# ══════════════════════════════════════════════════════════════════════════════

class AsyncMonitor:
    """
    Runs the expensive LP oracle only every oracle_every episodes.
    Between oracle calls, reuses the last known HullResult mask pattern
    (with a recency-weighted heuristic based on state distance to target).

    Cheaper than PostEpisodeHullMonitor for long runs.
    """

    name = "async"

    def __init__(self, target: Optional[np.ndarray] = None,
                 oracle_every: int = 10, tol: float = 1e-8):
        self._oracle         = PostEpisodeHullMonitor(target, tol)
        self.oracle_every    = oracle_every
        self.tol             = tol
        self._ep_count       = 0
        self._last_result:   Optional[HullResult] = None
        self.is_valid_target = self._oracle.is_valid_target
        self.target          = self._oracle.target

    def process(self, record: EpisodeRecord) -> HullResult:
        self._ep_count += 1
        run_oracle = (self._ep_count % self.oracle_every == 0
                      or self._last_result is None)

        if run_oracle:
            result              = self._oracle.process(record)
            self._last_result   = result
            self.is_valid_target= self._oracle.is_valid_target
            self.target         = self._oracle.target
            return result

        # Between oracle calls: cheap approximation via nearest-vertex distance
        if self.target is None:
            return HullResult(np.zeros(record.T, bool), None, False, self.name)

        dists = np.linalg.norm(
            record.states[:-1, None] - self.target[None], axis=-1).min(1)
        thresh = np.percentile(dists, 25)          # bottom 25% = near hull
        mask   = dists <= thresh
        first  = _first_true(mask)
        return HullResult(mask, first, self.is_valid_target, self.name + "_approx")

    def full_reset(self):
        self._oracle.full_reset()
        self._ep_count       = 0
        self._last_result    = None
        self.is_valid_target = False
        self.target          = None


# ══════════════════════════════════════════════════════════════════════════════
#  5. COMMITTEE MONITOR  (majority vote)
# ══════════════════════════════════════════════════════════════════════════════

class CommitteeMonitor:
    """
    Majority vote across multiple sub-monitors.

    Useful when individual monitors are uncertain (e.g. learned classifiers).
    A step is labelled in-hull iff ≥ threshold fraction of monitors agree.

    The oracle member (if any) acts as a veto: if oracle says 'not in hull'
    the result is always False regardless of other votes.
    """

    name = "committee"

    def __init__(self, monitors: List, threshold: float = 0.5,
                 oracle_veto: bool = False):
        self.monitors    = monitors
        self.threshold   = threshold
        self.oracle_veto = oracle_veto

    @property
    def is_valid_target(self) -> bool:
        return any(m.is_valid_target for m in self.monitors)

    @property
    def target(self):
        for m in self.monitors:
            if hasattr(m, "target") and m.target is not None:
                return m.target
        return None

    def process(self, record: EpisodeRecord) -> HullResult:
        results = [m.process(record) for m in self.monitors]
        T       = record.T

        votes  = np.zeros(T, np.float32)
        for r in results:
            votes += r.in_hull_mask.astype(np.float32)
        mask   = votes / len(self.monitors) >= self.threshold

        # Oracle veto
        if self.oracle_veto:
            oracle_r = results[0]                   # assume first is oracle
            mask     = mask & oracle_r.in_hull_mask

        first    = _first_true(mask)
        is_valid = any(r.is_valid_target for r in results)
        return HullResult(mask, first, is_valid, self.name)

    def full_reset(self):
        for m in self.monitors: m.full_reset()


# ══════════════════════════════════════════════════════════════════════════════
#  6. LEARNED MONITOR  (thin wrapper around a HullClassifier)
# ══════════════════════════════════════════════════════════════════════════════

class LearnedMonitor:
    """
    Delegates process() to any object implementing HullClassifier.process().
    Acts as the standard monitor interface for the benchmark.

    The wrapped classifier handles its own oracle schedule internally.
    """

    name = "learned"

    def __init__(self, classifier, monitor_name: str = "learned"):
        self._clf            = classifier
        self.name            = monitor_name

    @property
    def is_valid_target(self) -> bool:
        return getattr(self._clf, "is_valid_target", False)

    @property
    def target(self):
        return getattr(self._clf, "target", None)

    def process(self, record: EpisodeRecord) -> HullResult:
        return self._clf.process(record)

    def full_reset(self):
        self._clf.full_reset()


# ── factory ───────────────────────────────────────────────────────────────────

def make_monitor(name: str, target=None, **kw):
    """Convenience factory used by benchmark.py."""
    return {
        "oracle":       lambda: OracleMonitor(target, **kw),
        "post_episode": lambda: PostEpisodeHullMonitor(target, **kw),
        "her":          lambda: HERMonitor(**kw),
        "async":        lambda: AsyncMonitor(target, **kw),
    }[name]()
