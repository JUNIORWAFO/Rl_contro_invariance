"""
ppo_spring.py — PPO training loop for the Spring-Mass-Damper system.

Uses:
  env_spring.py     → make_vec_spring_env (SyncVectorEnv)
  actors_spring.py  → SpringGaussianActor, SpringCritic
  replay_spring.py  → PPORolloutBuffer

Algorithm: Proximal Policy Optimisation (Schulman et al., 2017)
  - Clipped surrogate objective
  - GAE-λ advantage estimation
  - Separate value-function loss (MSE)
  - Entropy bonus for exploration
  - Vectorised environment collection (n_envs parallel workers)

Quick start
-----------
  python ppo_spring.py                       # train with defaults
  python ppo_spring.py --n-envs 8 --steps 200000
  python ppo_spring.py --eval               # eval only (loads checkpoint)

Checkpoints are saved to  ./checkpoints/ppo_spring_<step>.pt
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_spring    import make_vec_spring_env, make_spring_env
from actors_spring import SpringGaussianActor, SpringCritic
from replay_spring import PPORolloutBuffer


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_HP: Dict = dict(
    # env
    n_envs       = 4,
    m            = 1.0,
    c            = 0.3,
    k            = 1.0,
    dt           = 0.05,
    max_steps    = 200,
    x0_scale     = 2.0,
    act_bound    = 5.0,
    noise_std    = 0.0,
    # rollout
    n_steps      = 128,       # steps per env per rollout
    # PPO
    total_steps  = 500_000,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    clip_eps     = 0.2,
    value_coef   = 0.5,
    entropy_coef = 0.01,
    max_grad_norm= 0.5,
    # optimiser
    lr           = 3e-4,
    n_epochs     = 10,        # gradient epochs per rollout
    mini_batch   = 64,
    # network
    hidden       = 64,
    # misc
    seed         = 0,
    device       = "cpu",
    ckpt_dir     = "./checkpoints",
    log_interval = 10,        # episodes between console logs
)


# ══════════════════════════════════════════════════════════════════════════════
#  PPO AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    PPO agent for continuous-action environments.

    Wraps SpringGaussianActor + SpringCritic with a clipped PPO update.

    Parameters
    ----------
    obs_dim, act_dim, act_bound : env dimensions
    hp : hyperparameter dict (see DEFAULT_HP)
    """

    def __init__(
        self,
        obs_dim:   int,
        act_dim:   int,
        act_bound: float,
        hp:        Dict,
    ):
        self.hp     = hp
        self.device = torch.device(hp["device"])

        self.actor  = SpringGaussianActor(
            obs_dim, act_dim, hp["hidden"], act_bound
        ).to(self.device)

        self.critic = SpringCritic(
            obs_dim, hp["hidden"]
        ).to(self.device)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=hp["lr"])
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=hp["lr"])

        self.clip_eps     = hp["clip_eps"]
        self.value_coef   = hp["value_coef"]
        self.entropy_coef = hp["entropy_coef"]
        self.max_grad_norm= hp["max_grad_norm"]

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,        # (n_envs, obs_dim)
    ):
        """
        Sample actions for a batch of observations.

        Returns
        -------
        actions   : np.ndarray  (n_envs, act_dim)
        log_probs : np.ndarray  (n_envs,)
        values    : np.ndarray  (n_envs,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions, log_probs, _ = self.actor.get_action(obs_t)
        values = self.critic(obs_t).squeeze(-1)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.critic(obs_t).squeeze(-1).cpu().numpy()

    # ── PPO update ───────────────────────────────────────────────────────────

    def update(self, rollout: PPORolloutBuffer) -> Dict[str, float]:
        """
        Run n_epochs of mini-batch PPO updates on the filled rollout buffer.

        Returns
        -------
        dict with mean losses over all mini-batches.
        """
        hp = self.hp
        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []

        for _ in range(hp["n_epochs"]):
            for batch in rollout.iterate(
                mini_batch_size=hp["mini_batch"],
                normalize_adv=True,
            ):
                obs      = batch["obs"].to(self.device)
                action   = batch["action"].to(self.device)
                old_lp   = batch["log_prob"].to(self.device)
                adv      = batch["advantage"].to(self.device)
                ret      = batch["return"].to(self.device)

                # ── actor loss (clipped surrogate) ────────────────────────────
                _, new_lp, entropy = self.actor.get_action(obs)
                # re-evaluate log_prob at the stored actions
                mu, log_std = self.actor(obs)
                std  = log_std.exp()
                dist = torch.distributions.Normal(mu, std)
                # un-squash stored actions for log_prob re-evaluation
                raw_a = action / self.actor.act_bound           # (B, act_dim)
                raw_a = raw_a.clamp(-0.9999, 0.9999)
                raw_a = torch.atanh(raw_a)
                new_lp = dist.log_prob(raw_a)
                new_lp -= torch.log(
                    self.actor.act_bound * (1 - torch.tanh(raw_a).pow(2)) + 1e-6
                )
                new_lp = new_lp.sum(dim=-1)
                entropy_b = dist.entropy().sum(dim=-1)

                ratio     = (new_lp - old_lp).exp()
                surr1     = ratio * adv
                surr2     = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── critic loss (MSE) ─────────────────────────────────────────
                val_pred   = self.critic(obs).squeeze(-1)
                critic_loss = nn.functional.mse_loss(val_pred, ret)

                # ── entropy bonus ─────────────────────────────────────────────
                entropy_loss = -entropy_b.mean()

                # ── combined loss ─────────────────────────────────────────────
                loss = (actor_loss
                        + self.value_coef   * critic_loss
                        + self.entropy_coef * entropy_loss)

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

        return {
            "actor_loss":   float(np.mean(actor_losses)),
            "critic_loss":  float(np.mean(critic_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "total_loss":   float(np.mean(total_losses)),
        }

    # ── checkpoint ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "hp":     self.hp,
        }, path)
        print(f"  [PPO] checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"  [PPO] loaded checkpoint ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(hp: Dict) -> PPOAgent:
    """
    Main PPO training loop with vectorised environments (SyncVectorEnv).

    Procedure
    ---------
    1. Build n_envs parallel Spring-Mass-Damper envs via make_vec_spring_env.
    2. Collect n_steps per env → PPORolloutBuffer.
    3. Compute GAE-λ returns.
    4. Run n_epochs of mini-batch PPO updates.
    5. Log metrics every log_interval completed episodes.
    6. Save checkpoint every 50 rollouts.
    """
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])

    # ── build vectorised env ─────────────────────────────────────────────────
    venv = make_vec_spring_env(
        n_envs    = hp["n_envs"],
        m         = hp["m"],
        c         = hp["c"],
        k         = hp["k"],
        dt        = hp["dt"],
        max_steps = hp["max_steps"],
        x0_scale  = hp["x0_scale"],
        act_bound = hp["act_bound"],
        noise_std = hp["noise_std"],
    )

    obs_dim = venv.single_observation_space.shape[0]   # 2  (pos, vel)
    act_dim = venv.single_action_space.shape[0]        # 1  (force)

    # ── agent & rollout buffer ────────────────────────────────────────────────
    agent = PPOAgent(obs_dim, act_dim, hp["act_bound"], hp)
    rollout = PPORolloutBuffer(
        n_steps  = hp["n_steps"],
        n_envs   = hp["n_envs"],
        obs_dim  = obs_dim,
        act_dim  = act_dim,
        device   = hp["device"],
    )

    total_env_steps = hp["total_steps"]
    steps_per_rollout = hp["n_steps"] * hp["n_envs"]
    n_rollouts = total_env_steps // steps_per_rollout

    print(f"\n{'='*60}")
    print(f"  PPO — Spring-Mass-Damper")
    print(f"  n_envs={hp['n_envs']}  n_steps={hp['n_steps']}  "
          f"total_steps={total_env_steps:,}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  device={hp['device']}")
    print(f"{'='*60}\n")

    obs, _ = venv.reset()
    ep_returns = np.zeros(hp["n_envs"], np.float32)
    ep_lengths = np.zeros(hp["n_envs"], int)
    completed_returns: List[float] = []
    global_step = 0
    t0 = time.time()

    for rollout_i in range(n_rollouts):

        # ── collect rollout ───────────────────────────────────────────────────
        rollout.reset()
        for _ in range(hp["n_steps"]):
            actions, log_probs, values = agent.get_action(obs)

            next_obs, rewards, terminateds, truncateds, infos = venv.step(actions)
            dones = terminateds | truncateds

            rollout.add(
                obs      = obs,
                action   = actions,
                reward   = rewards,
                done     = dones.astype(np.float32),
                log_prob = log_probs,
                value    = values,
            )

            ep_returns += rewards
            ep_lengths += 1

            for env_idx in range(hp["n_envs"]):
                if dones[env_idx]:
                    completed_returns.append(float(ep_returns[env_idx]))
                    ep_returns[env_idx] = 0.0
                    ep_lengths[env_idx] = 0

            obs = next_obs
            global_step += hp["n_envs"]

        # ── bootstrap last value ──────────────────────────────────────────────
        last_value = agent.get_value(obs)
        rollout.compute_returns(
            last_value  = last_value,
            gamma       = hp["gamma"],
            gae_lambda  = hp["gae_lambda"],
        )

        # ── PPO update ────────────────────────────────────────────────────────
        metrics = agent.update(rollout)

        # ── logging ───────────────────────────────────────────────────────────
        n_ep = len(completed_returns)
        if n_ep > 0 and (rollout_i + 1) % hp["log_interval"] == 0:
            recent = completed_returns[-50:]
            elapsed = time.time() - t0
            sps = global_step / elapsed
            print(
                f"  step={global_step:>8,}  ep={n_ep:>5}  "
                f"ret={np.mean(recent):>8.2f}±{np.std(recent):.2f}  "
                f"actor_loss={metrics['actor_loss']:>7.4f}  "
                f"critic_loss={metrics['critic_loss']:>7.4f}  "
                f"entropy={-metrics['entropy_loss']:>5.3f}  "
                f"sps={sps:>6.0f}"
            )

        # ── checkpoint ────────────────────────────────────────────────────────
        if (rollout_i + 1) % 50 == 0:
            ckpt_path = os.path.join(
                hp["ckpt_dir"], f"ppo_spring_{global_step}.pt"
            )
            agent.save(ckpt_path)

    venv.close()
    print(f"\n  Training complete.  Total steps: {global_step:,}")
    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    agent:    PPOAgent,
    hp:       Dict,
    n_eval:   int = 10,
    render:   bool = False,
) -> Dict[str, float]:
    """
    Evaluate the trained agent on a single (non-vectorised) env.

    Returns mean return and episode length over n_eval episodes.
    """
    env = make_spring_env(
        m=hp["m"], c=hp["c"], k=hp["k"],
        dt=hp["dt"], max_steps=hp["max_steps"],
        x0_scale=hp["x0_scale"], act_bound=hp["act_bound"],
    )
    agent.actor.eval()

    returns, lengths = [], []
    for ep in range(n_eval):
        obs, _ = env.reset(seed=ep)
        ep_ret = 0.0
        ep_len = 0
        done = False
        while not done:
            obs_t = torch.tensor(obs[None], dtype=torch.float32,
                                  device=agent.device)
            mu, log_std = agent.actor(obs_t)
            action = mu.squeeze(0).cpu().numpy()          # greedy (mean)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"  eval ep {ep+1}/{n_eval}  return={ep_ret:.2f}  len={ep_len}")

    agent.actor.train()
    stats = {
        "mean_return": float(np.mean(returns)),
        "std_return":  float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }
    print(f"\n  Eval summary: return={stats['mean_return']:.2f}±"
          f"{stats['std_return']:.2f}  len={stats['mean_length']:.1f}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO for Spring-Mass-Damper")
    p.add_argument("--n-envs",        type=int,   default=DEFAULT_HP["n_envs"])
    p.add_argument("--n-steps",       type=int,   default=DEFAULT_HP["n_steps"])
    p.add_argument("--steps",         type=int,   default=DEFAULT_HP["total_steps"],
                   dest="total_steps")
    p.add_argument("--lr",            type=float, default=DEFAULT_HP["lr"])
    p.add_argument("--clip-eps",      type=float, default=DEFAULT_HP["clip_eps"])
    p.add_argument("--entropy-coef",  type=float, default=DEFAULT_HP["entropy_coef"])
    p.add_argument("--hidden",        type=int,   default=DEFAULT_HP["hidden"])
    p.add_argument("--n-epochs",      type=int,   default=DEFAULT_HP["n_epochs"])
    p.add_argument("--mini-batch",    type=int,   default=DEFAULT_HP["mini_batch"])
    p.add_argument("--device",        type=str,   default=DEFAULT_HP["device"])
    p.add_argument("--seed",          type=int,   default=DEFAULT_HP["seed"])
    p.add_argument("--ckpt-dir",      type=str,   default=DEFAULT_HP["ckpt_dir"])
    p.add_argument("--log-interval",  type=int,   default=DEFAULT_HP["log_interval"])
    p.add_argument("--eval",          action="store_true",
                   help="Run evaluation only (requires --ckpt)")
    p.add_argument("--ckpt",          type=str,   default=None,
                   help="Checkpoint path for --eval")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hp = {**DEFAULT_HP, **vars(args)}

    if hp["eval"]:
        # ── eval only ────────────────────────────────────────────────────────
        if hp["ckpt"] is None:
            raise ValueError("Pass --ckpt <path> for evaluation mode.")
        agent = PPOAgent(
            obs_dim=2, act_dim=1, act_bound=hp["act_bound"], hp=hp
        )
        agent.load(hp["ckpt"])
        evaluate(agent, hp, n_eval=10)
    else:
        # ── train + eval ──────────────────────────────────────────────────────
        agent = train(hp)
        evaluate(agent, hp, n_eval=10)
