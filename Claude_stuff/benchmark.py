"""
benchmark.py
=============
Unified comparison of all (env × monitor × agent) combinations.

Measures four metrics per run:
  1. Sample efficiency   — hull_hit_rate and steps_to_first_hit vs episode
  2. Wall-clock time     — wall_time_per_ep, monitor_time_per_ep
  3. Computation/update  — agent_update_ms (ms per .update() call)
  4. Peak memory         — tracemalloc peak in MB

Usage
-----
  python benchmark.py                          # full suite, saves results.csv
  python benchmark.py --envs stable_4d        # single env
  python benchmark.py --agents sac ddpg       # subset of agents
  python benchmark.py --episodes 200          # shorter runs

Output
------
  results.csv          — one row per (env, monitor, agent, seed)
  results_summary.txt  — mean ± std table across seeds
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from types       import EpisodeRecord, HullResult, BenchmarkResult
from env         import make_benchmark_envs, LinearUpdateEnv
from hull_monitors import (OracleMonitor, PostEpisodeHullMonitor,
                            HERMonitor, AsyncMonitor)
from rl_algorithms import make_agent

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_AGENTS   = ["random", "cem", "model_based", "sac", "td3", "ddpg"]
DEFAULT_MONITORS = ["oracle", "post_episode", "her", "async"]
DEFAULT_EPISODES = 300
DEFAULT_SEEDS    = [0, 1, 2]


def _make_monitor(name: str, env: LinearUpdateEnv):
    """Construct a fresh monitor for a given env."""
    # Seed the oracle with a small set of reachable states
    # so 'oracle' and 'post_episode' have a fair starting target.
    seed_pts = _collect_seed_points(env, n_steps=500)
    if name == "oracle":
        return OracleMonitor(seed_pts)
    if name == "post_episode":
        return PostEpisodeHullMonitor(seed_pts)
    if name == "her":
        return HERMonitor(strategy="final", min_goals=5)
    if name == "async":
        return AsyncMonitor(seed_pts, oracle_every=10)
    raise ValueError(f"Unknown monitor '{name}'")


def _collect_seed_points(env: LinearUpdateEnv, n_steps: int = 500) -> np.ndarray:
    """
    Roll out a random policy to seed the hull monitor with reachable states.
    This gives all monitors a common ground truth to start from.
    """
    pts = []
    obs, _ = env.reset(seed=999)
    for _ in range(n_steps):
        a = env.action_space.sample()
        obs, _, done, _, _ = env.step(a)
        pts.append(obs.copy())
        if done:
            obs, _ = env.reset()
    return np.array(pts, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def _memory_tracker():
    tracemalloc.start()
    yield
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak  # bytes — caller reads via closure


class _MemTracker:
    """Context manager that stores peak MB."""
    def __enter__(self):
        tracemalloc.start()
        return self
    def __exit__(self, *_):
        _, self.peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    @property
    def peak_mb(self): return self.peak_bytes / 1e6


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_single(
    env_name:      str,
    monitor_name:  str,
    agent_name:    str,
    n_episodes:    int = DEFAULT_EPISODES,
    seed:          int = 0,
    hull_reward:   float = 10.,
    update_every:  int = 1,        # update agent every N steps
    verbose:       bool = False,
) -> BenchmarkResult:
    """
    Run one (env, monitor, agent) combination for n_episodes.
    Returns a BenchmarkResult with all four metric categories.
    """
    np.random.seed(seed)
    if _TORCH: torch.manual_seed(seed)

    # Build env fresh (so A,B are not shared across seeds for random envs)
    envs    = make_benchmark_envs()
    env     = envs[env_name]
    monitor = _make_monitor(monitor_name, env)
    agent   = make_agent(agent_name, env)

    # Metrics accumulators
    ep_returns:       List[float]  = []
    hull_hits:        List[bool]   = []
    wall_times:       List[float]  = []
    monitor_times:    List[float]  = []
    update_times:     List[float]  = []
    total_steps:      int = 0
    first_hit_step:   Optional[int] = None

    mem_tracker = _MemTracker()

    with mem_tracker:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_states  = [obs.copy()]
            ep_actions: List[np.ndarray] = []
            ep_rewards: List[float]      = []
            ep_start = time.perf_counter()

            done = False
            while not done:
                a = agent.select_action(obs, explore=True)
                obs_new, r, done, _, _ = env.step(a)

                ep_states.append(obs_new.copy())
                ep_actions.append(a.copy())
                ep_rewards.append(r)
                total_steps += 1

                # Push raw transition
                agent.push(obs, a, r, obs_new, done)
                obs = obs_new

                # Agent update
                if total_steps % update_every == 0:
                    t_upd = time.perf_counter()
                    metrics = agent.update()
                    update_times.append((time.perf_counter()-t_upd)*1e3)

            ep_wall = time.perf_counter() - ep_start

            # Build episode record
            record = EpisodeRecord(
                states  = np.array(ep_states,  dtype=np.float32),
                actions = np.array(ep_actions, dtype=np.float32),
                rewards = np.array(ep_rewards, dtype=np.float32),
                ep_id   = ep,
            )

            # Monitor
            t_mon = time.perf_counter()
            result: HullResult = monitor.process(record)
            monitor_times.append(time.perf_counter()-t_mon)

            # Relabel rewards if hull reached
            if result.hull_reached:
                relabeled = result.relabel_rewards(record.rewards, hull_reward)
                # Re-push relabeled transitions into replay buffer
                for t in range(record.T):
                    agent.push(record.states[t], record.actions[t],
                               relabeled[t], record.states[t+1],
                               t == record.T-1)

            # Track hull hit
            hit = result.hull_reached
            hull_hits.append(hit)
            if hit and first_hit_step is None:
                first_hit_step = total_steps

            # Track SACxCEM phase transition
            if hasattr(agent, "set_hull_valid"):
                agent.set_hull_valid(result.is_valid_target)

            ep_returns.append(float(record.rewards.sum()))
            wall_times.append(ep_wall)

            if verbose and ep % 50 == 0:
                hit_rate = np.mean(hull_hits[-50:]) if ep >= 50 else np.mean(hull_hits)
                print(f"  ep={ep:4d}  ret={ep_returns[-1]:7.1f}  "
                      f"hit_rate={hit_rate:.3f}  steps={total_steps}")

    # Aggregate
    final_returns    = ep_returns[-10:] if len(ep_returns) >= 10 else ep_returns
    agent_update_ms  = float(np.mean(update_times)) if update_times else 0.

    return BenchmarkResult(
        env_name             = env_name,
        monitor_name         = monitor_name,
        agent_name           = agent_name,
        total_steps          = total_steps,
        hull_hit_rate        = float(np.mean(hull_hits)),
        steps_to_first_hit   = first_hit_step,
        final_return         = float(np.mean(final_returns)),
        wall_time_total      = float(np.sum(wall_times)),
        wall_time_per_ep     = float(np.mean(wall_times)),
        monitor_time_per_ep  = float(np.mean(monitor_times)),
        agent_update_ms      = agent_update_ms,
        peak_memory_mb       = mem_tracker.peak_mb,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FULL BENCHMARK SUITE
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    env_names:    List[str] = None,
    agent_names:  List[str] = None,
    monitor_names:List[str] = None,
    n_episodes:   int       = DEFAULT_EPISODES,
    seeds:        List[int] = None,
    output_csv:   str       = "results.csv",
    verbose:      bool      = False,
) -> List[BenchmarkResult]:

    env_names     = env_names     or list(make_benchmark_envs().keys())
    agent_names   = agent_names   or DEFAULT_AGENTS
    monitor_names = monitor_names or DEFAULT_MONITORS
    seeds         = seeds         or DEFAULT_SEEDS

    # Filter agents to available ones
    try:
        from rl_algorithms import make_agent as _ma
        avail = []
        for ag in agent_names:
            try: _ma(ag, list(make_benchmark_envs().values())[0]); avail.append(ag)
            except (ValueError, ImportError): pass
        agent_names = avail
    except ImportError:
        agent_names = ["random"]

    total_runs = len(env_names)*len(monitor_names)*len(agent_names)*len(seeds)
    print(f"\n{'━'*66}")
    print(f"  Benchmark: {len(env_names)} envs × {len(monitor_names)} monitors × "
          f"{len(agent_names)} agents × {len(seeds)} seeds = {total_runs} runs")
    print(f"  Episodes per run: {n_episodes}")
    print(f"{'━'*66}")

    results: List[BenchmarkResult] = []
    run_i = 0

    for env_name in env_names:
        for monitor_name in monitor_names:
            for agent_name in agent_names:
                for seed in seeds:
                    run_i += 1
                    tag = f"{env_name}/{monitor_name}/{agent_name}/seed={seed}"
                    print(f"  [{run_i:3d}/{total_runs}] {tag}", flush=True)
                    t0  = time.perf_counter()
                    try:
                        r = run_single(env_name, monitor_name, agent_name,
                                        n_episodes, seed, verbose=verbose)
                        results.append(r)
                        print(f"          hit={r.hull_hit_rate:.3f}  "
                              f"ret={r.final_return:7.1f}  "
                              f"mon={r.monitor_time_per_ep*1e3:.1f}ms/ep  "
                              f"upd={r.agent_update_ms:.2f}ms  "
                              f"mem={r.peak_memory_mb:.1f}MB  "
                              f"({time.perf_counter()-t0:.1f}s)")
                    except Exception as e:
                        print(f"          ERROR: {e}")

    if results:
        _save_csv(results, output_csv)
        _print_summary(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(results: List[BenchmarkResult], path: str):
    fields = list(BenchmarkResult.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"\n  Results saved → {path}")


def _print_summary(results: List[BenchmarkResult]):
    """Print a grouped mean±std table to stdout."""
    from collections import defaultdict
    groups: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        key = f"{r.env_name}/{r.monitor_name}/{r.agent_name}"
        groups[key].append(r)

    col_w = 36
    print(f"\n{'━'*90}")
    print(f"  {'Config':<{col_w}} {'hit_rate':>10} {'return':>10} "
          f"{'mon_ms':>9} {'upd_ms':>9} {'mem_MB':>9}")
    print(f"  {'─'*88}")

    prev_env = None
    for key in sorted(groups):
        rs     = groups[key]
        env    = rs[0].env_name
        if env != prev_env:
            print(f"\n  ── {env} ──")
            prev_env = env

        hit  = np.array([r.hull_hit_rate        for r in rs])
        ret  = np.array([r.final_return          for r in rs])
        mon  = np.array([r.monitor_time_per_ep   for r in rs])*1e3
        upd  = np.array([r.agent_update_ms       for r in rs])
        mem  = np.array([r.peak_memory_mb        for r in rs])

        short = key.split("/",1)[1]   # drop env prefix
        print(f"  {short:<{col_w}} "
              f"{hit.mean():>6.3f}±{hit.std():>5.3f}  "
              f"{ret.mean():>7.1f}±{ret.std():>5.1f}  "
              f"{mon.mean():>7.2f}  "
              f"{upd.mean():>7.3f}  "
              f"{mem.mean():>7.1f}")

    print(f"{'━'*90}\n")


def save_summary_txt(results: List[BenchmarkResult], path: str = "results_summary.txt"):
    buf = io.StringIO()
    import sys; old = sys.stdout; sys.stdout = buf
    _print_summary(results)
    sys.stdout = old
    Path(path).write_text(buf.getvalue())
    print(f"  Summary saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CURVE HELPER  (for plotting hull_hit_rate vs steps)
# ══════════════════════════════════════════════════════════════════════════════

def run_with_curves(
    env_name:     str,
    monitor_name: str,
    agent_name:   str,
    n_episodes:   int = DEFAULT_EPISODES,
    seed:         int = 0,
) -> Dict:
    """
    Like run_single but also returns per-episode curves for plotting.
    Returns dict with keys: returns, hit_mask, steps, plus BenchmarkResult.
    """
    np.random.seed(seed)
    if _TORCH: torch.manual_seed(seed)

    envs    = make_benchmark_envs()
    env     = envs[env_name]
    monitor = _make_monitor(monitor_name, env)
    agent   = make_agent(agent_name, env)

    ep_returns, ep_hits, ep_steps = [], [], []
    total_steps = 0; first_hit = None

    with _MemTracker() as mem:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            states=[obs.copy()]; actions=[]; rewards=[]
            done=False
            while not done:
                a=agent.select_action(obs, explore=True)
                obs,r,done,_,_=env.step(a)
                states.append(obs.copy()); actions.append(a); rewards.append(r)
                agent.push(states[-2],a,r,obs,done)
                agent.update()
                total_steps+=1

            rec=EpisodeRecord(np.array(states,np.float32),
                               np.array(actions,np.float32),
                               np.array(rewards,np.float32), ep)
            result=monitor.process(rec)
            hit=result.hull_reached
            if hit and first_hit is None: first_hit=total_steps

            if hasattr(agent,"set_hull_valid"):
                agent.set_hull_valid(result.is_valid_target)

            ep_returns.append(float(np.array(rewards).sum()))
            ep_hits.append(hit)
            ep_steps.append(total_steps)

    bres = BenchmarkResult(
        env_name=env_name, monitor_name=monitor_name, agent_name=agent_name,
        total_steps=total_steps,
        hull_hit_rate=float(np.mean(ep_hits)),
        steps_to_first_hit=first_hit,
        final_return=float(np.mean(ep_returns[-10:])),
        wall_time_total=0., wall_time_per_ep=0.,
        monitor_time_per_ep=0., agent_update_ms=0.,
        peak_memory_mb=mem.peak_mb,
    )
    return {"result": bres, "returns": ep_returns,
            "hits": ep_hits, "steps": ep_steps}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hull RL benchmark")
    p.add_argument("--envs",     nargs="+", default=None)
    p.add_argument("--agents",   nargs="+", default=None)
    p.add_argument("--monitors", nargs="+", default=None)
    p.add_argument("--episodes", type=int,  default=DEFAULT_EPISODES)
    p.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--output",   default="results.csv")
    p.add_argument("--verbose",  action="store_true")
    args = p.parse_args()

    results = run_benchmark(
        env_names     = args.envs,
        agent_names   = args.agents,
        monitor_names = args.monitors,
        n_episodes    = args.episodes,
        seeds         = args.seeds,
        output_csv    = args.output,
        verbose       = args.verbose,
    )
    save_summary_txt(results)
