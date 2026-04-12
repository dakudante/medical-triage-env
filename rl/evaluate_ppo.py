from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import ResetRequest
from server.triage_environment import MedicalTriageEnvironment

from rl.feature_extractor import ObservationFeatureExtractor
from rl.policy_net import TriagePolicyNet


def evaluate(checkpoint_path: Path, episodes: int, seed: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    extractor = ObservationFeatureExtractor()
    policy = TriagePolicyNet(
        input_dim=checkpoint["feature_dim"],
        departments=checkpoint["departments"],
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    totals = []
    details = []
    for i in range(episodes):
        task_id = ["easy", "medium", "hard"][i % 3]
        env = MedicalTriageEnvironment()
        obs = env.reset(ResetRequest(task_id=task_id, use_procedural=(task_id != "hard"), seed=seed + i))
        done = False
        total = 0.0
        steps = 0
        while not done:
            state = extractor.transform(obs)
            out = policy.sample_action(state, deterministic=True)
            obs, reward, done, info = env.step(out.action)
            total += float(reward.value)
            steps += 1
        totals.append(total)
        details.append({"task_id": task_id, "reward": round(total, 4), "steps": steps})

    report = {
        "episodes": episodes,
        "mean_episode_reward": round(sum(totals) / len(totals), 4),
        "min_episode_reward": round(min(totals), 4),
        "max_episode_reward": round(max(totals), 4),
        "details": details,
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO-lite checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=101)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    report = evaluate(Path(args.checkpoint), args.episodes, args.seed)
    print(json.dumps(report, indent=2))
