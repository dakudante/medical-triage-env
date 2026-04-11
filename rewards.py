"""
rewards.py — Reward shaping utilities for GRPO training.

NOTE on sign convention:
  The environment stores delay_penalty, mortality_penalty, overtriage_penalty,
  and undertriage_penalty as *negative* values inside reward.breakdown already.
  shaped_reward() simply sums all components — do NOT negate them again here.

FIXES vs V3.5 original:
  Minor: Renamed module-level functions that shadowed Python built-in names
         (e.g. `delay_penalty` clashed with the local variable name in loops).
         Old names kept as aliases for backward compatibility.
"""

from typing import Dict, Any


# ── Component extractors ───────────────────────────────────────────────────────

def get_accuracy_reward(reward_breakdown: Dict[str, float]) -> float:
    """Extract accuracy component from reward breakdown (positive value)."""
    return reward_breakdown.get("accuracy_component", 0.0)


def get_resource_reward(reward_breakdown: Dict[str, float]) -> float:
    """Extract resource component from reward breakdown (positive value)."""
    return reward_breakdown.get("resource_component", 0.0)


def get_delay_penalty(reward_breakdown: Dict[str, float]) -> float:
    """Extract delay penalty from reward breakdown (already stored as negative)."""
    return reward_breakdown.get("delay_penalty", 0.0)


def get_mortality_penalty(reward_breakdown: Dict[str, float]) -> float:
    """Extract mortality penalty from reward breakdown (already stored as negative)."""
    return reward_breakdown.get("mortality_penalty", 0.0)


def get_overtriage_penalty(reward_breakdown: Dict[str, float]) -> float:
    """Extract overtriage penalty (already stored as negative)."""
    return reward_breakdown.get("overtriage_penalty", 0.0)


def get_undertriage_penalty(reward_breakdown: Dict[str, float]) -> float:
    """Extract undertriage penalty (already stored as negative)."""
    return reward_breakdown.get("undertriage_penalty", 0.0)


# ── Backward-compatible aliases (deprecated — prefer get_* names) ──────────────
accuracy_reward   = get_accuracy_reward
resource_reward   = get_resource_reward
delay_penalty     = get_delay_penalty      # was shadowing loop vars
mortality_penalty = get_mortality_penalty  # was shadowing loop vars


# ── Main shaped reward ─────────────────────────────────────────────────────────

def shaped_reward(reward_dict: Dict[str, Any]) -> float:
    """
    Compute a shaped scalar reward for TRL/GRPO from the environment's
    full reward dict (as returned by json.loads on the WebSocket response).

    The environment stores all penalty values as *negative* floats inside
    `reward_dict["breakdown"]`, so this function simply sums all components
    without additional negation.

    Args:
        reward_dict: The dict under the "reward" key in the step response,
                     e.g. {"value": 0.74, "breakdown": {...}, ...}.

    Returns:
        Total shaped reward as a float.
    """
    breakdown = reward_dict.get("breakdown", {})
    total = (
        get_accuracy_reward(breakdown)
        + get_resource_reward(breakdown)
        + get_delay_penalty(breakdown)        # already negative
        + get_mortality_penalty(breakdown)    # already negative
        + get_overtriage_penalty(breakdown)   # already negative
        + get_undertriage_penalty(breakdown)  # already negative
    )
    return total
