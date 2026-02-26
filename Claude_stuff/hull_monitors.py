"""
hull_monitors.py
================
Six architectural approaches for hull-based reward labeling,
all decoupled from LinearUpdateEnv.

Structures:
  1. HERHullMonitor        — Hindsight Experience Replay style relabeling
  2. AsyncHullMonitor      — Background process with Queue-based pipeline
  3. LearnedHullClassifier — Neural net approximator trained from exact labels
  4. PhaseHullMonitor      — Phased: explore → fit → fine-tune (RLHF-style)
  5. ReachabilityGraph     — Graph/topological reachable set
  6. CommitteeHullMonitor  — Multi-monitor voting ensemble

Each monitor exposes a consistent interface:
  .process(record: EpisodeRecord) -> HullResult
  .reset()
  .full_reset()
"""

from __future__ import annotations

import time
import threading
import multiprocessing as mp
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeRecord:
    """Immutable snapshot of one episode. Produced by training loop."""
    states:  np.ndarray   # (T+1, xDim)
    actions: np.ndarray   # (T,   uDim)
    rewards: np.ndarray   # (T,)
    ep_id:   int = 0

    @property
    def T(self):
        return len(self.actions)

    def __repr__(self):
        return f"EpisodeRecord(ep={self.ep_id}, T={self.T})"


@dataclass
class HullResult:
    """Output of any HullMonitor.process() call."""
    in_hull_mask:    np.ndarray        # bool (T,) — per-step label
    first_hull_step: Optional[int]     # first step inside hull, or None
    is_valid_target: bool
    source:          str = "unknown"   # which monitor produced this

    @property
    def hull_reached(self) -> bool:
        return self.first_hull_step is not None

    def relabel_rewards(self, rewards: np.ndarray, hull_reward: float) -> np.ndarray:
        """Return reward array with hull_reward injected at hull-hit steps."""
        if not self.hull_reached:
            return rewards.copy()
        out = rewards.copy()
        out[self.in_hull_mask] = hull_reward
        return out

    def __repr__(self):
        return (f"HullResult(source={self.source!r}, hit={self.hull_reached}, "
                f"step={self.first_hull_step}, valid={self.is_valid_target})")


# ── Hull check helpers (stubs — replace with your actual implementations) ────

def _point_in_hull(points, query, p_min, p_max,
                   method="linprog", tolerance=1e-6,
                   cvxpy_solver=None, fw_eps=1e-6,
                   fw_sample_size=64, fw_max_iter=10000) -> bool:
    """Dispatch to the appropriate hull-membership routine."""
    if method == "linprog":
        return is_point_inside_hull_2(points, query, p_min, p_max, tolerance=tolerance)
    elif method == "cvxpy":
        return is_point_inside_hull_cvxpy(points, query, p_min, p_max,
                                          tolerance=tolerance, solver=cvxpy_solver)
    elif method == "fw":
        result, _ = point_in_hull_scaled(points, query, eps=fw_eps,
                                         fw_sample_size=fw_sample_size,
                                         fw_max_iter=fw_max_iter)
        return result
    raise ValueError(f"Unknown method: {method!r}")


def _scan_episode(record: EpisodeRecord, target, is_valid_target,
                  t_min, t_max, method, **kw) -> Tuple[np.ndarray, Optional[int]]:
    """
    Core scan loop shared by all monitors.
    Returns (in_hull_mask, first_hit_step).
    """
    T        = record.T
    in_hull  = np.zeros(T, dtype=bool)
    first_hit = None

    for t in range(1, T):
        next_state = record.states[t + 1] if (t + 1) <= T else record.states[t]

        if not is_valid_target:
            traj  = record.states[:t + 1]
            if len(traj) < 2:
                continue
            mn, mx = traj.min(axis=0), traj.max(axis=0)
            hit = _point_in_hull(traj, next_state, mn, mx, method=method, **kw)
        else:
            hit = _point_in_hull(target, next_state, t_min, t_max, method=method, **kw)

        if hit:
            in_hull[t] = True
            if first_hit is None:
                first_hit = t

    return in_hull, first_hit


# ══════════════════════════════════════════════════════════════════════════════
#  1. HER-STYLE HULL MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class HERHullMonitor:
    """
    Hindsight Experience Replay style relabeling.

    Every visited state is treated as a potential goal. After the episode,
    we scan backward and ask: "if the goal had been state[t], would any
    earlier step have reached it?"  This massively multiplies the number
    of useful training signals even when hull hits are rare.

    Strategy options:
      "future"  — for each step t, use a random future state as the goal
      "final"   — use the last state of the episode as the goal
      "episode" — use every state in the episode as a goal (expensive)
    """

    def __init__(
        self,
        xDim:           int,
        uDim:           int,
        action_bounds,
        target          = None,
        method:         str   = "linprog",
        hull_tolerance: float = 1e-6,
        strategy:       str   = "future",   # "future" | "final" | "episode"
        k:              int   = 4,           # number of HER relabelings per step
        min_steps:      int   = None,
    ):
        self.xDim          = xDim
        self.uDim          = uDim
        self.action_bounds = action_bounds
        self.method        = method
        self.tolerance     = hull_tolerance
        self.strategy      = strategy
        self.k             = k
        self.min_steps     = min_steps or (xDim + 1)

        self.target          = None if target is None else np.asarray(target, dtype=np.float32)
        self.is_valid_target = self.target is not None
        self.hull            = self.target.copy() if self.target is not None else None
        self.Hist_Input      = None

        self._center = self._min = self._max = None
        self._dirty  = True

    def process(self, record: EpisodeRecord) -> HullResult:
        T = record.T

        # ── Standard hull scan ────────────────────────────────────────────────
        self._ensure_stats()
        in_hull, first_hit = _scan_episode(
            record, self.target, self.is_valid_target,
            self._min, self._max, self.method, tolerance=self.tolerance)

        # ── HER relabeling ────────────────────────────────────────────────────
        her_mask = np.zeros(T, dtype=bool)

        if self.strategy == "final":
            goal_indices = [T]   # last state
        elif self.strategy == "future":
            goal_indices = self._sample_future_goals(T)
        else:  # "episode"
            goal_indices = list(range(1, T + 1))

        for goal_idx in goal_indices:
            goal_state = record.states[min(goal_idx, T)]
            # Build a tiny "hull" from states up to goal_idx
            goal_hull = record.states[:goal_idx + 1]
            if len(goal_hull) < 2:
                continue
            mn, mx = goal_hull.min(axis=0), goal_hull.max(axis=0)

            for t in range(goal_idx):
                if not her_mask[t]:
                    hit = _point_in_hull(goal_hull, record.states[t],
                                         mn, mx, method=self.method,
                                         tolerance=self.tolerance)
                    if hit:
                        her_mask[t] = True

        # Merge real hull hits with HER relabeled hits
        combined_mask = in_hull | her_mask
        first_combined = int(np.argmax(combined_mask)) if combined_mask.any() else None

        # Update internal state
        if first_hit is not None:
            self._update(record, first_hit)

        return HullResult(
            in_hull_mask    = combined_mask,
            first_hull_step = first_combined,
            is_valid_target = self.is_valid_target,
            source          = "HER",
        )

    def _sample_future_goals(self, T: int) -> List[int]:
        """Sample k future time indices for HER."""
        indices = []
        for _ in range(self.k):
            t = np.random.randint(1, T + 1)
            indices.append(t)
        return indices

    def _update(self, record, first_hit):
        traj    = record.states[:first_hit + 1]
        actions = record.actions[:first_hit]
        self.Hist_Input = (actions.copy() if self.Hist_Input is None
                           else np.vstack([self.Hist_Input, actions]))
        if not self.is_valid_target:
            self.target  = traj.copy() if self.target is None else np.vstack([self.target, traj])
            self._dirty  = True
        else:
            self.hull = np.vstack([self.hull, traj]) if self.hull is not None else traj.copy()

    def _ensure_stats(self):
        if self._dirty and self.target is not None:
            self._center = np.mean(self.target, axis=0)
            self._min    = np.min(self.target,  axis=0)
            self._max    = np.max(self.target,  axis=0)
            self._dirty  = False

    def full_reset(self):
        self.target = self.hull = self.Hist_Input = None
        self.is_valid_target = False
        self._dirty = True


# ══════════════════════════════════════════════════════════════════════════════
#  2. ASYNC HULL MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class AsyncHullMonitor:
    """
    Background-process hull checker.

    Architecture:
      Training loop  →  puts EpisodeRecord into self.in_queue
      Worker process →  consumes records, runs hull check, puts HullResult
                        into self.out_queue
      Training loop  →  calls .collect_results() before each PPO update
                        to retrieve completed HullResults keyed by ep_id

    The training loop never blocks on hull computation.
    """

    def __init__(
        self,
        xDim:          int,
        uDim:          int,
        action_bounds,
        target         = None,
        method:        str   = "linprog",
        hull_tolerance: float = 1e-6,
        worker_timeout: float = 0.01,    # seconds to wait on queue.get
        max_queue_size: int   = 128,
    ):
        self.xDim          = xDim
        self.uDim          = uDim
        self.action_bounds = action_bounds
        self.method        = method
        self.tolerance     = hull_tolerance
        self.timeout       = worker_timeout

        # Shared state (protected by lock)
        self._manager   = mp.Manager()
        self._shared    = self._manager.dict({
            "target":          None,
            "is_valid_target": target is not None,
            "t_min":           None,
            "t_max":           None,
        })
        if target is not None:
            t = np.asarray(target, dtype=np.float32)
            self._shared["target"]          = t
            self._shared["t_min"]           = t.min(axis=0)
            self._shared["t_max"]           = t.max(axis=0)

        self.in_queue   = mp.Queue(maxsize=max_queue_size)
        self.out_queue  = mp.Queue()

        self._results: dict[int, HullResult] = {}
        self._worker   = mp.Process(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, record: EpisodeRecord):
        """Non-blocking. Drop record into the work queue."""
        try:
            self.in_queue.put_nowait(record)
        except mp.queues.Full:
            pass   # drop if backlogged — training must not block

    def collect_results(self) -> dict[int, HullResult]:
        """
        Drain the output queue and return a dict {ep_id: HullResult}.
        Call this before each PPO update to get all completed labels.
        """
        while not self.out_queue.empty():
            try:
                ep_id, result = self.out_queue.get_nowait()
                self._results[ep_id] = result
            except Exception:
                break
        out = dict(self._results)
        self._results.clear()
        return out

    def process(self, record: EpisodeRecord) -> Optional[HullResult]:
        """
        Synchronous fallback: submit and block until result arrives.
        Use for testing or when you don't mind waiting.
        """
        self.submit(record)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            results = self.collect_results()
            if record.ep_id in results:
                return results[record.ep_id]
            time.sleep(0.005)
        return None

    def set_target(self, target: np.ndarray):
        t = np.asarray(target, dtype=np.float32)
        self._shared["target"]          = t
        self._shared["is_valid_target"] = True
        self._shared["t_min"]           = t.min(axis=0)
        self._shared["t_max"]           = t.max(axis=0)

    def shutdown(self):
        self._worker.terminate()
        self._worker.join()
        self._manager.shutdown()

    # ── Worker (runs in separate process) ─────────────────────────────────────

    def _worker_loop(self):
        """Runs in background process. Never touches the env."""
        while True:
            try:
                record = self.in_queue.get(timeout=self.timeout)
            except Exception:
                continue

            target          = self._shared["target"]
            is_valid_target = self._shared["is_valid_target"]
            t_min           = self._shared["t_min"]
            t_max           = self._shared["t_max"]

            in_hull, first_hit = _scan_episode(
                record, target, is_valid_target,
                t_min, t_max, self.method, tolerance=self.tolerance)

            # Update shared target state if hull was hit and no target yet
            if first_hit is not None and not is_valid_target:
                traj = record.states[:first_hit + 1]
                self._shared["target"]          = traj
                self._shared["is_valid_target"] = False   # let validation happen externally
                self._shared["t_min"]           = traj.min(axis=0)
                self._shared["t_max"]           = traj.max(axis=0)

            result = HullResult(
                in_hull_mask    = in_hull,
                first_hull_step = first_hit,
                is_valid_target = is_valid_target,
                source          = "Async",
            )
            self.out_queue.put((record.ep_id, result))

    def full_reset(self):
        self._shared["target"]          = None
        self._shared["is_valid_target"] = False
        self._shared["t_min"]           = None
        self._shared["t_max"]           = None


# ══════════════════════════════════════════════════════════════════════════════
#  3. LEARNED HULL CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class HullClassifierNet(nn.Module):
    def __init__(self, xDim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class LearnedHullClassifier:
    """
    Neural network approximator for hull membership.

    Pipeline:
      - Exact oracle (any hull method) runs every `label_every` episodes
        to generate ground-truth labels
      - Labels are stored in a replay buffer
      - Classifier is retrained periodically
      - At inference time, uses cheap classifier forward pass

    The exact oracle is the bottleneck; the classifier replaces it online.
    """

    def __init__(
        self,
        xDim:           int,
        uDim:           int,
        action_bounds,
        target          = None,
        method:         str   = "linprog",
        hull_tolerance: float = 1e-6,
        hidden:         int   = 64,
        lr:             float = 1e-3,
        label_every:    int   = 10,     # run exact oracle every N episodes
        train_every:    int   = 5,      # retrain classifier every N labelings
        buffer_size:    int   = 50_000,
        batch_size:     int   = 256,
        train_epochs:   int   = 20,
        threshold:      float = 0.5,    # P(in_hull) threshold
        device:         str   = "cpu",
    ):
        self.xDim          = xDim
        self.method        = method
        self.tolerance     = hull_tolerance
        self.label_every   = label_every
        self.train_every   = train_every
        self.batch_size    = batch_size
        self.train_epochs  = train_epochs
        self.threshold     = threshold
        self.device        = torch.device(device)

        # Oracle state
        self.target          = None if target is None else np.asarray(target, dtype=np.float32)
        self.is_valid_target = self.target is not None
        self._min = self._max = None
        self._dirty = True

        # Classifier
        self.net       = HullClassifierNet(xDim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self._trained  = False

        # Label replay buffer: list of (state, label) tuples
        self._buf_states = deque(maxlen=buffer_size)
        self._buf_labels = deque(maxlen=buffer_size)

        # Counters
        self._ep_count     = 0
        self._label_count  = 0

    def process(self, record: EpisodeRecord) -> HullResult:
        self._ep_count += 1

        # ── Run exact oracle every label_every episodes ───────────────────────
        run_oracle = (self._ep_count % self.label_every == 0) or not self._trained

        if run_oracle:
            self._ensure_stats()
            in_hull, first_hit = _scan_episode(
                record, self.target, self.is_valid_target,
                self._min, self._max, self.method, tolerance=self.tolerance)

            # Store labels in replay buffer
            for t in range(record.T):
                self._buf_states.append(record.states[t].copy())
                self._buf_labels.append(float(in_hull[t]))

            self._label_count += 1

            # Retrain classifier
            if self._label_count % self.train_every == 0 and len(self._buf_states) > self.batch_size:
                self._train_classifier()

            # Update oracle state
            if first_hit is not None:
                self._update(record, first_hit)

            return HullResult(in_hull_mask=in_hull, first_hull_step=first_hit,
                              is_valid_target=self.is_valid_target, source="Oracle")

        else:
            # ── Use classifier for cheap inference ────────────────────────────
            states_t = torch.tensor(record.states[:record.T], dtype=torch.float32,
                                    device=self.device)
            with torch.no_grad():
                probs = self.net(states_t).cpu().numpy()

            in_hull   = probs >= self.threshold
            first_hit = int(np.argmax(in_hull)) if in_hull.any() else None

            return HullResult(in_hull_mask=in_hull, first_hull_step=first_hit,
                              is_valid_target=self.is_valid_target, source="Classifier")

    def _train_classifier(self):
        states = np.array(self._buf_states, dtype=np.float32)
        labels = np.array(self._buf_labels, dtype=np.float32)

        self.net.train()
        for _ in range(self.train_epochs):
            idx   = np.random.permutation(len(states))
            for s in range(0, len(states), self.batch_size):
                mb    = idx[s:s + self.batch_size]
                x     = torch.tensor(states[mb], device=self.device)
                y     = torch.tensor(labels[mb], device=self.device)
                loss  = self.criterion(self.net(x), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.net.eval()
        self._trained = True

    def _update(self, record, first_hit):
        traj = record.states[:first_hit + 1]
        if not self.is_valid_target:
            self.target = traj.copy() if self.target is None else np.vstack([self.target, traj])
            self._dirty = True

    def _ensure_stats(self):
        if self._dirty and self.target is not None:
            self._min   = self.target.min(axis=0)
            self._max   = self.target.max(axis=0)
            self._dirty = False

    def full_reset(self):
        self.target = None
        self.is_valid_target = False
        self._trained = False
        self._buf_states.clear()
        self._buf_labels.clear()
        self._dirty = True

    def save_classifier(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load_classifier(self, path: str):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        self._trained = True


# ══════════════════════════════════════════════════════════════════════════════
#  4. PHASE HULL MONITOR  (RLHF-style)
# ══════════════════════════════════════════════════════════════════════════════

class Phase:
    EXPLORE   = "explore"    # collect trajectories, no hull reward
    FIT       = "fit"        # run hull monitor on all collected data
    FINETUNE  = "finetune"   # freeze hull, inject hull reward


class PhaseHullMonitor:
    """
    Three-phase training pipeline:

      Phase 1 — EXPLORE:
        Train with boundary reward only. Collect trajectory data.
        Advance when enough episodes have been collected.

      Phase 2 — FIT:
        Offline: run HullMonitor on all stored EpisodeRecords.
        Find best hull, validate target, set is_valid_target.
        Immediately advance to FINETUNE.

      Phase 3 — FINETUNE:
        Freeze hull. Inject hull reward retroactively.
        Optionally cycle back to EXPLORE for curriculum learning.
    """

    def __init__(
        self,
        xDim:                  int,
        uDim:                  int,
        action_bounds,
        target                 = None,
        method:                str   = "linprog",
        hull_tolerance:        float = 1e-6,
        explore_episodes:      int   = 100,   # episodes before fitting
        max_stored_records:    int   = 500,
        cycle:                 bool  = True,  # loop back to explore after finetune
        finetune_episodes:     int   = 200,   # finetune episodes before cycling
        min_steps:             int   = None,
    ):
        self.xDim              = xDim
        self.method            = method
        self.tolerance         = hull_tolerance
        self.explore_episodes  = explore_episodes
        self.finetune_episodes = finetune_episodes
        self.cycle             = cycle
        self.min_steps         = min_steps or (xDim + 1)

        self.target            = None if target is None else np.asarray(target, dtype=np.float32)
        self.is_valid_target   = self.target is not None
        self.hull              = self.target.copy() if self.target is not None else None
        self.Hist_Input        = None

        self._min = self._max = None
        self._dirty = True

        # Phase state
        self.phase             = Phase.FINETUNE if self.is_valid_target else Phase.EXPLORE
        self._phase_ep_count   = 0
        self._stored: List[EpisodeRecord] = []
        self._max_stored       = max_stored_records

    @property
    def current_phase(self) -> str:
        return self.phase

    def process(self, record: EpisodeRecord) -> HullResult:
        self._phase_ep_count += 1

        if self.phase == Phase.EXPLORE:
            return self._process_explore(record)
        elif self.phase == Phase.FIT:
            return self._process_fit()
        else:
            return self._process_finetune(record)

    def _process_explore(self, record: EpisodeRecord) -> HullResult:
        # Store record for later fitting
        if len(self._stored) < self._max_stored:
            self._stored.append(record)

        # Advance to fit phase when ready
        if self._phase_ep_count >= self.explore_episodes:
            print(f"[PhaseMonitor] Advancing EXPLORE → FIT "
                  f"({len(self._stored)} records stored)")
            self.phase = Phase.FIT
            self._phase_ep_count = 0

        # No hull reward during exploration
        return HullResult(
            in_hull_mask    = np.zeros(record.T, dtype=bool),
            first_hull_step = None,
            is_valid_target = False,
            source          = "Phase:EXPLORE",
        )

    def _process_fit(self) -> HullResult:
        """
        Offline fitting over all stored records.
        Find the best trajectory (most steps before hull hit).
        """
        best_record    = None
        best_first_hit = None
        best_mask      = None

        for rec in self._stored:
            in_hull, first_hit = _scan_episode(
                rec, self.target, self.is_valid_target,
                self._min, self._max, self.method, tolerance=self.tolerance)
            if first_hit is not None:
                if best_first_hit is None or first_hit > best_first_hit:
                    best_record    = rec
                    best_first_hit = first_hit
                    best_mask      = in_hull

        if best_record is not None:
            traj = best_record.states[:best_first_hit + 1]
            self.target = traj.copy() if self.target is None else np.vstack([self.target, traj])
            self._dirty = True
            self._ensure_stats()
            print(f"[PhaseMonitor] FIT complete: hull from {len(traj)} states. "
                  f"Advancing FIT → FINETUNE")

        self.phase = Phase.FINETUNE
        self._phase_ep_count = 0
        self._stored.clear()

        # Return empty result — fit phase is instantaneous
        return HullResult(
            in_hull_mask    = np.zeros(1, dtype=bool),
            first_hull_step = None,
            is_valid_target = self.is_valid_target,
            source          = "Phase:FIT",
        )

    def _process_finetune(self, record: EpisodeRecord) -> HullResult:
        self._ensure_stats()
        in_hull, first_hit = _scan_episode(
            record, self.target, self.is_valid_target,
            self._min, self._max, self.method, tolerance=self.tolerance)

        if first_hit is not None:
            traj = record.states[:first_hit + 1]
            self.hull = np.vstack([self.hull, traj]) if self.hull is not None else traj.copy()

        # Optionally cycle back
        if self.cycle and self._phase_ep_count >= self.finetune_episodes:
            print(f"[PhaseMonitor] Cycling FINETUNE → EXPLORE")
            self.phase = Phase.EXPLORE
            self._phase_ep_count = 0

        return HullResult(
            in_hull_mask    = in_hull,
            first_hull_step = first_hit,
            is_valid_target = self.is_valid_target,
            source          = "Phase:FINETUNE",
        )

    def _ensure_stats(self):
        if self._dirty and self.target is not None:
            self._min   = self.target.min(axis=0)
            self._max   = self.target.max(axis=0)
            self._dirty = False

    def full_reset(self):
        self.target = self.hull = self.Hist_Input = None
        self.is_valid_target = False
        self.phase = Phase.EXPLORE
        self._phase_ep_count = 0
        self._stored.clear()
        self._dirty = True


# ══════════════════════════════════════════════════════════════════════════════
#  5. REACHABILITY GRAPH MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ReachabilityGraph:
    """
    Graph-based reachable set representation.

    Nodes: quantized/clustered states (grid cells or k-means centroids)
    Edges: observed transitions between nodes

    Hull membership → graph reachability query (BFS/DFS).
    Works for non-convex and disconnected reachable sets.

    Two quantization modes:
      "grid"    — axis-aligned grid cells (fast, fixed resolution)
      "online"  — grow node list as new states arrive (adaptive)
    """

    def __init__(
        self,
        xDim:           int,
        uDim:           int,
        action_bounds,
        state_bounds,
        grid_resolution: float = 0.5,    # cell size for grid mode
        mode:            str   = "grid", # "grid" | "online"
        online_radius:   float = 0.3,    # merge radius for online mode
        min_visits:      int   = 2,      # node must be visited N times to be "confirmed"
        target_nodes:    Optional[set] = None,
    ):
        self.xDim           = xDim
        self.state_bounds   = state_bounds
        self.resolution     = grid_resolution
        self.mode           = mode
        self.online_radius  = online_radius
        self.min_visits     = min_visits

        # Graph: adjacency list  {node_id: set of neighbor node_ids}
        self._graph:   dict[int, set] = defaultdict(set)
        self._visits:  dict[int, int] = defaultdict(int)
        self._centroids: list         = []  # for online mode
        self._node_states: dict[int, np.ndarray] = {}

        self.target_nodes: set = target_nodes or set()
        self.is_valid_target   = len(self.target_nodes) > 0

        self.Hist_Input = None

    def process(self, record: EpisodeRecord) -> HullResult:
        T = record.T
        in_hull   = np.zeros(T, dtype=bool)
        first_hit = None

        # Map trajectory to node sequence
        node_seq = [self._state_to_node(record.states[t]) for t in range(T + 1)]

        # Build graph edges from this episode
        for t in range(T):
            n_from = node_seq[t]
            n_to   = node_seq[t + 1]
            self._graph[n_from].add(n_to)
            self._visits[n_from] += 1

        # If no target nodes yet, use visited nodes as target seed
        if not self.target_nodes and len(self._graph) > self.min_visits * 2:
            # Seed target: nodes with high out-degree (frequently transitioned through)
            confirmed = {n for n, v in self._visits.items() if v >= self.min_visits}
            if confirmed:
                self.target_nodes    = confirmed
                self.is_valid_target = True

        # Check reachability for each step
        if self.target_nodes:
            for t in range(1, T):
                node = node_seq[t]
                # Is this node reachable FROM the target set?
                # i.e. is there a path from any target node to this node?
                if self._is_reachable(node, self.target_nodes, max_depth=10):
                    in_hull[t] = True
                    if first_hit is None:
                        first_hit = t

        # Accumulate inputs
        if first_hit is not None:
            actions = record.actions[:first_hit]
            self.Hist_Input = (actions.copy() if self.Hist_Input is None
                               else np.vstack([self.Hist_Input, actions]))

        return HullResult(
            in_hull_mask    = in_hull,
            first_hull_step = first_hit,
            is_valid_target = self.is_valid_target,
            source          = "Graph",
        )

    def _state_to_node(self, state: np.ndarray) -> int:
        if self.mode == "grid":
            return self._grid_hash(state)
        else:
            return self._online_node(state)

    def _grid_hash(self, state: np.ndarray) -> int:
        """Map state to integer grid cell index."""
        low  = np.asarray(self.state_bounds[0])
        cell = np.floor((state - low) / self.resolution).astype(int)
        # Cantor-style hash for arbitrary dimensions
        h = 0
        for i, c in enumerate(cell):
            h += c * (10000 ** i)
        return int(h)

    def _online_node(self, state: np.ndarray) -> int:
        """Find or create a node for this state."""
        for i, c in enumerate(self._centroids):
            if np.linalg.norm(state - c) < self.online_radius:
                self._centroids[i] = 0.9 * c + 0.1 * state   # running mean
                return i
        self._centroids.append(state.copy())
        node_id = len(self._centroids) - 1
        self._node_states[node_id] = state.copy()
        return node_id

    def _is_reachable(self, query_node: int, target_set: set,
                      max_depth: int = 10) -> bool:
        """BFS from query node; check if any target node is reachable."""
        visited = {query_node}
        frontier = {query_node}
        for _ in range(max_depth):
            if frontier & target_set:
                return True
            next_frontier = set()
            for n in frontier:
                next_frontier |= self._graph.get(n, set())
            frontier = next_frontier - visited
            visited |= frontier
            if not frontier:
                break
        return bool(frontier & target_set)

    def get_graph_stats(self) -> dict:
        return {
            "num_nodes":   len(self._graph),
            "num_edges":   sum(len(v) for v in self._graph.values()),
            "target_nodes": len(self.target_nodes),
        }

    def full_reset(self):
        self._graph.clear()
        self._visits.clear()
        self._centroids.clear()
        self._node_states.clear()
        self.target_nodes    = set()
        self.is_valid_target = False
        self.Hist_Input      = None


# ══════════════════════════════════════════════════════════════════════════════
#  6. COMMITTEE HULL MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class VotingStrategy:
    UNANIMOUS    = "unanimous"    # all must agree → most conservative
    MAJORITY     = "majority"     # >50% agree
    ANY          = "any"          # at least one agrees → most optimistic
    WEIGHTED     = "weighted"     # weighted by monitor confidence scores


class CommitteeHullMonitor:
    """
    Ensemble of hull monitors with configurable voting strategy.

    Each member is any object with a .process(EpisodeRecord) -> HullResult
    interface. This means you can mix different methods, tolerances,
    or even entirely different architectures in one committee.

    Optionally tracks per-monitor accuracy (when ground truth is available)
    and adjusts weights accordingly.

    Usage:
        monitors = [
            PostEpisodeHullMonitor(..., method="linprog"),
            PostEpisodeHullMonitor(..., method="fw"),
            ReachabilityGraph(...),
        ]
        committee = CommitteeHullMonitor(monitors, strategy="majority")
        result = committee.process(record)
        # result.uncertainty: fraction of monitors that disagreed
    """

    def __init__(
        self,
        monitors:           list,
        strategy:           str   = VotingStrategy.MAJORITY,
        weights:            Optional[List[float]] = None,
        track_disagreement: bool  = True,
    ):
        assert len(monitors) >= 2, "Committee needs at least 2 monitors"
        self.monitors           = monitors
        self.strategy           = strategy
        self.weights            = np.array(weights or [1.0] * len(monitors), dtype=float)
        self.weights           /= self.weights.sum()
        self.track_disagreement = track_disagreement

        self._n                 = len(monitors)
        self._disagreement_log: List[float] = []

        # For weighted strategy: track per-monitor hit rates
        self._monitor_hits      = np.zeros(self._n)
        self._monitor_calls     = np.zeros(self._n)

    @property
    def is_valid_target(self) -> bool:
        return any(m.is_valid_target for m in self.monitors)

    def process(self, record: EpisodeRecord) -> "CommitteeHullResult":
        # Run all monitors
        results = [m.process(record) for m in self.monitors]
        masks   = np.stack([r.in_hull_mask for r in results], axis=0)  # (N, T)

        # Aggregate per strategy
        if self.strategy == VotingStrategy.UNANIMOUS:
            final_mask = masks.all(axis=0)
        elif self.strategy == VotingStrategy.MAJORITY:
            final_mask = masks.mean(axis=0) > 0.5
        elif self.strategy == VotingStrategy.ANY:
            final_mask = masks.any(axis=0)
        elif self.strategy == VotingStrategy.WEIGHTED:
            weighted   = (masks * self.weights[:, None]).sum(axis=0)
            final_mask = weighted > 0.5
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

        # Compute disagreement metric (fraction of steps where monitors disagree)
        if self.track_disagreement:
            step_agreement = masks.mean(axis=0)             # per-step agreement rate
            disagreement   = float(np.mean(step_agreement * (1 - step_agreement)) * 4)
            self._disagreement_log.append(disagreement)
        else:
            disagreement = 0.0

        # Per-monitor contribution tracking
        for i, r in enumerate(results):
            self._monitor_calls[i] += 1
            if r.hull_reached:
                self._monitor_hits[i] += 1

        first_hit     = int(np.argmax(final_mask)) if final_mask.any() else None
        valid_targets = [r.is_valid_target for r in results]

        return CommitteeHullResult(
            in_hull_mask        = final_mask,
            first_hull_step     = first_hit,
            is_valid_target     = any(valid_targets),
            source              = f"Committee:{self.strategy}",
            individual_results  = results,
            disagreement        = disagreement,
            vote_fractions      = masks.mean(axis=0),
        )

    def update_weights_from_performance(self):
        """
        Reweight monitors by their historical hit rate.
        Call periodically to shift trust toward better-performing monitors.
        """
        rates = np.where(self._monitor_calls > 0,
                         self._monitor_hits / self._monitor_calls,
                         0.0)
        if rates.sum() > 0:
            self.weights = rates / rates.sum()
            print(f"[Committee] Updated weights: "
                  + ", ".join(f"M{i}={w:.3f}" for i, w in enumerate(self.weights)))

    def get_disagreement_history(self) -> np.ndarray:
        return np.array(self._disagreement_log)

    def get_monitor_stats(self) -> dict:
        return {
            f"monitor_{i}": {
                "hits":      int(self._monitor_hits[i]),
                "calls":     int(self._monitor_calls[i]),
                "hit_rate":  float(self._monitor_hits[i] / max(1, self._monitor_calls[i])),
                "weight":    float(self.weights[i]),
            }
            for i in range(self._n)
        }

    def full_reset(self):
        for m in self.monitors:
            m.full_reset()
        self._disagreement_log.clear()
        self._monitor_hits[:]  = 0
        self._monitor_calls[:] = 0


@dataclass
class CommitteeHullResult(HullResult):
    """Extended HullResult with committee-specific fields."""
    individual_results: List[HullResult] = field(default_factory=list)
    disagreement:       float            = 0.0
    vote_fractions:     np.ndarray       = field(default_factory=lambda: np.array([]))

    @property
    def high_confidence(self) -> np.ndarray:
        """Steps where all monitors unanimously agreed."""
        if not self.individual_results:
            return self.in_hull_mask
        masks = np.stack([r.in_hull_mask for r in self.individual_results])
        return masks.all(axis=0)

    @property
    def uncertain_steps(self) -> np.ndarray:
        """Steps where monitors disagreed."""
        return self.vote_fractions > 0.0 & (self.vote_fractions < 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  BASE POST-EPISODE MONITOR  (the simple version from previous design)
# ══════════════════════════════════════════════════════════════════════════════

class PostEpisodeHullMonitor:
    """
    The baseline: scan episode offline after it ends.
    Use this as a member of CommitteeHullMonitor or standalone.
    """

    def __init__(
        self,
        xDim,
        uDim,
        action_bounds,
        target               = None,
        method               = "linprog",
        hull_tolerance       = 1e-6,
        min_steps            = None,
    ):
        self.xDim          = xDim
        self.method        = method
        self.tolerance     = hull_tolerance
        self.action_bounds = action_bounds
        self.min_steps     = min_steps or (xDim + 1)

        self.target          = None if target is None else np.asarray(target, dtype=np.float32)
        self.is_valid_target = self.target is not None
        self.hull            = self.target.copy() if self.target is not None else None
        self.Hist_Input      = None

        self._min = self._max = None
        self._dirty = True

    def process(self, record: EpisodeRecord) -> HullResult:
        self._ensure_stats()
        in_hull, first_hit = _scan_episode(
            record, self.target, self.is_valid_target,
            self._min, self._max, self.method, tolerance=self.tolerance)

        if first_hit is not None:
            traj    = record.states[:first_hit + 1]
            actions = record.actions[:first_hit]
            self.Hist_Input = (actions.copy() if self.Hist_Input is None
                               else np.vstack([self.Hist_Input, actions]))
            if not self.is_valid_target:
                self.target = (traj.copy() if self.target is None
                               else np.vstack([self.target, traj]))
                self._dirty = True
            else:
                self.hull = (np.vstack([self.hull, traj]) if self.hull is not None
                             else traj.copy())

        return HullResult(
            in_hull_mask    = in_hull,
            first_hull_step = first_hit,
            is_valid_target = self.is_valid_target,
            source          = f"PostEpisode:{self.method}",
        )

    def _ensure_stats(self):
        if self._dirty and self.target is not None:
            self._min   = self.target.min(axis=0)
            self._max   = self.target.max(axis=0)
            self._dirty = False

    def full_reset(self):
        self.target = self.hull = self.Hist_Input = None
        self.is_valid_target = False
        self._dirty = True


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_monitor(kind: str, xDim: int, uDim: int, action_bounds,
                 state_bounds=None, target=None, **kwargs):
    """
    Factory function for convenient monitor creation.

    kind: "post_episode" | "her" | "async" | "learned" | "phase" | "graph" | "committee"

    For "committee", pass monitors=[list of already-constructed monitors].
    """
    shared = dict(xDim=xDim, uDim=uDim, action_bounds=action_bounds, target=target)

    if kind == "post_episode":
        return PostEpisodeHullMonitor(**shared, **kwargs)
    elif kind == "her":
        return HERHullMonitor(**shared, **kwargs)
    elif kind == "async":
        return AsyncHullMonitor(**shared, **kwargs)
    elif kind == "learned":
        return LearnedHullClassifier(**shared, **kwargs)
    elif kind == "phase":
        return PhaseHullMonitor(**shared, **kwargs)
    elif kind == "graph":
        assert state_bounds is not None, "graph monitor requires state_bounds"
        return ReachabilityGraph(xDim=xDim, uDim=uDim, action_bounds=action_bounds,
                                 state_bounds=state_bounds, **kwargs)
    elif kind == "committee":
        monitors = kwargs.pop("monitors")
        return CommitteeHullMonitor(monitors=monitors, **kwargs)
    else:
        raise ValueError(f"Unknown monitor kind: {kind!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Dummy system
    xDim          = 2
    uDim          = 2
    action_bounds = ([-1, -1], [1, 1])
    state_bounds  = ([-5, -5], [5, 5])

    def make_dummy_record(ep_id=0, T=50):
        states  = np.random.randn(T + 1, xDim).astype(np.float32)
        actions = np.random.randn(T, uDim).astype(np.float32)
        rewards = np.random.randn(T).astype(np.float32)
        return EpisodeRecord(states=states, actions=actions,
                             rewards=rewards, ep_id=ep_id)

    print("=" * 60)
    print("1. HER Monitor")
    her = make_monitor("her", xDim, uDim, action_bounds, strategy="future", k=4)
    r   = her.process(make_dummy_record())
    print(r)

    print("\n2. Phase Monitor")
    phase = make_monitor("phase", xDim, uDim, action_bounds, explore_episodes=3)
    for i in range(5):
        r = phase.process(make_dummy_record(ep_id=i))
        print(f"  ep={i} phase={phase.current_phase} {r}")

    print("\n3. Learned Classifier")
    clf = make_monitor("learned", xDim, uDim, action_bounds,
                       label_every=2, train_every=2, buffer_size=1000)
    for i in range(6):
        r = clf.process(make_dummy_record(ep_id=i))
        print(f"  ep={i} source={r.source}")

    print("\n4. Reachability Graph")
    graph = make_monitor("graph", xDim, uDim, action_bounds,
                         state_bounds=state_bounds, grid_resolution=0.5)
    for i in range(5):
        r = graph.process(make_dummy_record(ep_id=i))
        print(f"  ep={i} {r} | graph={graph.get_graph_stats()}")

    print("\n5. Committee Monitor")
    m1 = make_monitor("post_episode", xDim, uDim, action_bounds, method="linprog")
    m2 = make_monitor("her",          xDim, uDim, action_bounds, strategy="final")
    m3 = make_monitor("graph",        xDim, uDim, action_bounds,
                      state_bounds=state_bounds)
    committee = make_monitor("committee", xDim, uDim, action_bounds,
                             monitors=[m1, m2, m3], strategy="majority")
    r = committee.process(make_dummy_record())
    print(r)
    print("  Monitor stats:", committee.get_monitor_stats())

    print("\n6. Async Monitor (submit + collect)")
    async_mon = make_monitor("async", xDim, uDim, action_bounds)
    for i in range(4):
        async_mon.submit(make_dummy_record(ep_id=i))
    time.sleep(0.5)
    results = async_mon.collect_results()
    for ep_id, r in results.items():
        print(f"  ep={ep_id} {r}")
    async_mon.shutdown()

    print("\nDone.")
