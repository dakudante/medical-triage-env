from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from models import ResetRequest
from patients import DEPARTMENTS
from server.triage_environment import MedicalTriageEnvironment

from rl.feature_extractor import ObservationFeatureExtractor
from rl.policy_net import RESOURCE_KEYS, TriagePolicyNet


@dataclass
class Transition:
    state: torch.Tensor
    esi_action: int
    dept_action: int
    routing_action: int
    resource_action: torch.Tensor
    log_prob: float
    reward: float
    done: bool
    value: float


ROUTING_TO_IDX = {"admit": 0, "wait": 1, "reroute": 2}


def discount_cumsum(values: List[float], gamma: float) -> List[float]:
    out = [0.0] * len(values)
    running = 0.0
    for i in reversed(range(len(values))):
        running = values[i] + gamma * running
        out[i] = running
    return out


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], gamma: float, gae_lambda: float) -> tuple[List[float], List[float]]:
    advantages = [0.0] * len(rewards)
    last_adv = 0.0
    extended_values = values + [0.0]
    for t in reversed(range(len(rewards))):
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * extended_values[t + 1] * next_non_terminal - extended_values[t]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[t] = last_adv
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def collect_episode(env: MedicalTriageEnvironment, policy: TriagePolicyNet, extractor: ObservationFeatureExtractor, task_id: str, seed: int | None) -> tuple[List[Transition], float]:
    request = ResetRequest(task_id=task_id, use_procedural=(task_id != "hard"), seed=seed)
    obs = env.reset(request)
    done = False
    transitions: List[Transition] = []
    total_reward = 0.0

    while not done:
        state = extractor.transform(obs)
        out = policy.sample_action(state, deterministic=False)
        next_obs, reward, done, _info = env.step(out.action)
        resource_tensor = torch.tensor([float(getattr(out.action.resource_request, k)) for k in RESOURCE_KEYS], dtype=torch.float32)
        transitions.append(Transition(
            state=state,
            esi_action=out.action.esi_level - 1,
            dept_action=policy.departments.index(out.action.department),
            routing_action=ROUTING_TO_IDX[out.action.routing_decision or "admit"],
            resource_action=resource_tensor,
            log_prob=float(out.log_prob.item()),
            reward=float(reward.value),
            done=done,
            value=float(out.value.item()),
        ))
        total_reward += float(reward.value)
        obs = next_obs

    return transitions, total_reward


def build_batch(episodes: List[List[Transition]], gamma: float, gae_lambda: float) -> Dict[str, torch.Tensor]:
    states = []
    esi_actions = []
    dept_actions = []
    routing_actions = []
    resource_actions = []
    old_log_probs = []
    returns = []
    advantages = []

    for episode in episodes:
        rewards = [t.reward for t in episode]
        values = [t.value for t in episode]
        dones = [t.done for t in episode]
        adv, ret = compute_gae(rewards, values, dones, gamma, gae_lambda)
        for t, a, r in zip(episode, adv, ret):
            states.append(t.state)
            esi_actions.append(t.esi_action)
            dept_actions.append(t.dept_action)
            routing_actions.append(t.routing_action)
            resource_actions.append(t.resource_action)
            old_log_probs.append(t.log_prob)
            advantages.append(a)
            returns.append(r)

    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)

    return {
        "states": torch.stack(states),
        "esi_actions": torch.tensor(esi_actions, dtype=torch.long),
        "dept_actions": torch.tensor(dept_actions, dtype=torch.long),
        "routing_actions": torch.tensor(routing_actions, dtype=torch.long),
        "resource_actions": torch.stack(resource_actions),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
        "advantages": adv_tensor,
    }


def evaluate_policy(policy: TriagePolicyNet, extractor: ObservationFeatureExtractor, episodes: int = 6, seed: int = 123) -> Dict[str, float]:
    rewards: List[float] = []
    for i in range(episodes):
        env = MedicalTriageEnvironment()
        task_id = ["easy", "medium", "hard"][i % 3]
        obs = env.reset(ResetRequest(task_id=task_id, use_procedural=(task_id != "hard"), seed=seed + i))
        done = False
        total = 0.0
        while not done:
            state = extractor.transform(obs)
            out = policy.sample_action(state, deterministic=True)
            obs, reward, done, _ = env.step(out.action)
            total += float(reward.value)
        rewards.append(total)
    return {
        "episodes": episodes,
        "mean_episode_reward": round(sum(rewards) / len(rewards), 4),
        "min_episode_reward": round(min(rewards), 4),
        "max_episode_reward": round(max(rewards), 4),
    }


def save_checkpoint(output_dir: Path, policy: TriagePolicyNet, extractor: ObservationFeatureExtractor, optimizer: torch.optim.Optimizer, metrics: Dict[str, float]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ppo_lite_checkpoint.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "departments": policy.departments,
        "feature_dim": extractor.feature_dim,
        "metrics": metrics,
        "version": "7.1.0",
    }, path)
    return path


def train(args: argparse.Namespace) -> Dict[str, float]:
    torch.manual_seed(args.seed)
    extractor = ObservationFeatureExtractor()
    policy = TriagePolicyNet(input_dim=extractor.feature_dim, departments=DEPARTMENTS, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    all_episode_rewards: List[float] = []
    last_metrics: Dict[str, float] = {}

    for update in range(1, args.updates + 1):
        episodes: List[List[Transition]] = []
        update_rewards: List[float] = []
        for episode_idx in range(args.episodes_per_update):
            env = MedicalTriageEnvironment()
            task_id = ["easy", "medium", "hard"][episode_idx % 3]
            trajectory, total_reward = collect_episode(env, policy, extractor, task_id, args.seed + update * 100 + episode_idx)
            episodes.append(trajectory)
            update_rewards.append(total_reward)
            all_episode_rewards.append(total_reward)

        batch = build_batch(episodes, gamma=args.gamma, gae_lambda=args.gae_lambda)

        for _epoch in range(args.ppo_epochs):
            log_prob, entropy, values = policy.evaluate_actions(
                batch["states"],
                batch["esi_actions"],
                batch["dept_actions"],
                batch["routing_actions"],
                batch["resource_actions"],
            )
            ratio = torch.exp(log_prob - batch["old_log_probs"])
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
            policy_loss = -torch.min(ratio * batch["advantages"], clipped_ratio * batch["advantages"]).mean()
            value_loss = nn.functional.mse_loss(values, batch["returns"])
            entropy_bonus = entropy.mean()

            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

        eval_metrics = evaluate_policy(policy, extractor, episodes=6, seed=args.seed + update)
        last_metrics = {
            "update": update,
            "train_mean_episode_reward": round(sum(update_rewards) / len(update_rewards), 4),
            **eval_metrics,
        }
        print(json.dumps(last_metrics))

    checkpoint_path = save_checkpoint(Path(args.output_dir), policy, extractor, optimizer, last_metrics)
    metrics_path = Path(args.output_dir) / "train_metrics.json"
    metrics_path.write_text(json.dumps({
        "final": last_metrics,
        "all_episode_rewards": all_episode_rewards,
    }, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path}")
    return last_metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO-lite agent for Medical Triage OpenEnv.")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--episodes-per-update", type=int, default=9)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
