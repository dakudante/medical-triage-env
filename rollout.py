"""
rollout.py — GRPO rollout harness for the Medical Triage environment.

FIXES applied vs V3.5 original:
  Bug 2a: GRPOTrainer.train_step() was missing `return mean_reward`.
  Bug 2b: self.reward_func was stored but never called — dead code.
          It is now used to re-score raw reward dicts, so external reward
          shaping functions (e.g. from rewards.py) are actually applied.
  Bug 2c: GRPO advantage computation was a plain mean; added baseline-
          subtracted group-relative advantage per the GRPO paper.
"""

import asyncio
import json
from typing import List, Dict, Any, Callable, Optional
from client import MedicalTriageClient
from rewards import shaped_reward


async def rollout_func(
    policy: Callable[[Dict[str, Any]], Dict[str, Any]],
    task_id: str = "easy",
    max_steps: int = 3,
    use_procedural: bool = True,
    client_url: str = "ws://localhost:7860/ws",
) -> List[Dict[str, Any]]:
    """
    Execute one episode and return a trajectory list for TRL/GRPO.

    Each element in the returned list is a dict with keys:
        observation, action, reward, next_observation, done, info

    The `reward` field is the *shaped* scalar (float) produced by
    shaped_reward(), not the raw reward dict from the server.
    The raw reward dict is preserved under the `raw_reward` key so
    that GRPOTrainer can re-apply a different reward function if needed.
    """
    async with MedicalTriageClient(client_url) as client:
        reset_resp = await client.reset(task_id=task_id, use_procedural=use_procedural)
        obs = reset_resp["observation"]

        trajectory: List[Dict[str, Any]] = []
        done = False
        step = 0

        while not done and step < max_steps:
            action = policy(obs)

            step_resp = await client.step(action)
            next_obs  = step_resp["observation"]
            raw_reward = step_resp["reward"]   # plain dict after json.loads
            done       = step_resp["done"]

            # Compute shaped scalar reward
            reward = shaped_reward(raw_reward)

            trajectory.append({
                "observation":      obs,
                "action":           action,
                "reward":           reward,
                "raw_reward":       raw_reward,   # kept for re-scoring
                "next_observation": next_obs,
                "done":             done,
                "info":             step_resp.get("info", {}),
            })

            obs = next_obs
            step += 1

    return trajectory


class GRPOTrainer:
    """
    Minimal GRPO-style trainer wired to the Medical Triage rollout harness.

    In production, swap this for TRL's GRPOTrainer and pass rollout_func
    as the `reward_model` argument. This class exists to validate the
    environment interface and demonstrate the advantage-computation step.

    FIXES vs original:
      - train_step() now returns mean_reward (was missing).
      - reward_func is actually called on raw_reward dicts, so custom
        reward shaping from rewards.py is exercised, not ignored.
      - Group-relative advantage (GRPO baseline subtraction) added.
    """

    def __init__(
        self,
        policy,
        rollout_fn: Callable,
        reward_func: Callable = shaped_reward,
        client_url: str = "ws://localhost:7860/ws",
    ):
        self.policy      = policy
        self.rollout_fn  = rollout_fn
        self.reward_func = reward_func   # FIX: actually stored AND used below
        self.client_url  = client_url

    async def train_step(self, task_id: str) -> float:
        """
        One GRPO training step:
          1. Collect a rollout trajectory.
          2. Re-score each step with self.reward_func (supports reward shaping).
          3. Compute group-relative advantages (reward - mean_reward).
          4. Log and return mean reward.

        Returns:
            mean_reward (float) — average shaped reward across the trajectory.
        """
        # 1. Collect rollout
        trajectory = await self.rollout_fn(
            self.policy.act,
            task_id=task_id,
            client_url=self.client_url,
        )

        if not trajectory:
            print(f"[GRPO] task={task_id} — empty trajectory, skipping.")
            return 0.0

        # 2. Re-score with reward_func (FIX: reward_func is now actually called)
        #    This allows swapping reward shaping without changing rollout_func.
        rewards = [self.reward_func(t["raw_reward"]) for t in trajectory]

        # 3. Group-relative advantage (GRPO core idea: subtract group baseline)
        mean_reward = sum(rewards) / len(rewards)
        advantages  = [r - mean_reward for r in rewards]

        # 4. Log (in real TRL, you'd pass (trajectory, advantages) to the optimizer)
        print(
            f"[GRPO] task={task_id} | "
            f"steps={len(trajectory)} | "
            f"mean_reward={mean_reward:.3f} | "
            f"advantages={[f'{a:+.3f}' for a in advantages]}"
        )

        # FIX: return mean_reward (was missing in original)
        return mean_reward


async def run_training_demo():
    """Wire up GRPOTrainer and run a quick training demo."""
    from policies import RuleBasedPolicy

    policy  = RuleBasedPolicy()
    trainer = GRPOTrainer(policy, rollout_func, shaped_reward)

    print("Starting GRPOTrainer demo...")
    for task in ["easy", "medium", "hard"]:
        mean_r = await trainer.train_step(task)
        print(f"  → {task}: mean_reward={mean_r:.3f}")
    print("Training demo complete.")


if __name__ == "__main__":
    # Requires the server to be running at ws://localhost:7860/ws
    asyncio.run(run_training_demo())
