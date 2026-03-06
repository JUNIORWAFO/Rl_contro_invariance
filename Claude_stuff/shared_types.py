"""
types.py — shared dataclasses used across the entire project.
Import from here only; never redefine these elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EpisodeRecord:
    """Complete record of one episode, consumed by every HullMonitor."""
    states:  np.ndarray   # (T+1, xDim)
    actions: np.ndarray   # (T,   uDim)
    rewards: np.ndarray   # (T,)
    ep_id:   int = 0

    @property
    def T(self) -> int:       return len(self.actions)
    @property
    def xDim(self) -> int:    return self.states.shape[1]
    @property
    def uDim(self) -> int:    return self.actions.shape[1]


@dataclass
class HullResult:
    """Output of any Monitor.process(record) call."""
    in_hull_mask:    np.ndarray      # (T,) bool
    first_hull_step: Optional[int]   # index of first in-hull step, or None
    is_valid_target: bool
    source:          str = "unknown"

    @property
    def hull_reached(self) -> bool:  return self.first_hull_step is not None
    @property
    def hull_fraction(self) -> float:
        return float(self.in_hull_mask.mean()) if len(self.in_hull_mask) else 0.0

    def relabel_rewards(self, rewards: np.ndarray, hull_reward: float) -> np.ndarray:
        out = rewards.copy()
        out[self.in_hull_mask] = hull_reward
        return out


@dataclass
class BenchmarkResult:
    """One row in the benchmark comparison table."""
    env_name:            str
    monitor_name:        str
    agent_name:          str
    total_steps:         int
    hull_hit_rate:       float          # fraction of episodes with hull_reached
    steps_to_first_hit:  Optional[int]  # env steps before first success
    final_return:        float          # mean return, last 10 episodes
    wall_time_total:     float          # seconds
    wall_time_per_ep:    float
    monitor_time_per_ep: float
    agent_update_ms:     float          # ms per update call
    peak_memory_mb:      float = 0.0
