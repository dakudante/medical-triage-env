import random
from typing import Dict, Any

class RandomPolicy:
    """A policy that makes random triage decisions."""
    def __init__(self, departments: list):
        self.departments = departments

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "esi_level": random.randint(1, 5),
            "department": random.choice(self.departments),
            "reasoning": "Randomly assigned for baseline comparison.",
            "routing_decision": random.choice(["admit", "wait"]),
            "resource_request": {
                "icu_bed": random.random() > 0.8,
                "er_bed": random.random() > 0.5,
                "ventilator": random.random() > 0.9,
                "ct_scanner": random.random() > 0.7,
                "cardiac_monitor": random.random() > 0.6,
                "or_room": random.random() > 0.9,
                "cath_lab": random.random() > 0.95
            }
        }

class RuleBasedPolicy:
    """A policy that uses simple clinical rules for triage decisions."""
    def __init__(self):
        pass

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        vitals = observation.get("vitals", {})
        hr = vitals.get("hr", 80)
        o2 = vitals.get("o2_sat", 98)
        
        # Simple rule-based logic
        if o2 < 90 or hr > 130:
            esi = 1
            dept = "Resuscitation"
            resources = {
                "icu_bed": True, "er_bed": False, "ventilator": o2 < 85,
                "ct_scanner": False, "cardiac_monitor": True, "or_room": False, "cath_lab": False
            }
        elif o2 < 94 or hr > 110:
            esi = 2
            dept = "Emergency"
            resources = {
                "icu_bed": False, "er_bed": True, "ventilator": False,
                "ct_scanner": True, "cardiac_monitor": True, "or_room": False, "cath_lab": False
            }
        else:
            esi = 3
            dept = "Emergency"
            resources = {
                "icu_bed": False, "er_bed": True, "ventilator": False,
                "ct_scanner": False, "cardiac_monitor": False, "or_room": False, "cath_lab": False
            }

        return {
            "esi_level": esi,
            "department": dept,
            "reasoning": f"Rule-based decision: HR={hr}, O2={o2}.",
            "routing_decision": "admit" if esi <= 3 else "wait",
            "resource_request": resources
        }


class CurriculumPolicy:
    """
    Pillar 4.2: Auto-curriculum policy that selects task difficulty based
    on recent rolling average score.

    Thresholds (configurable):
      score < 0.65  -> easy
      0.65-0.78     -> medium
      > 0.78        -> hard

    Wraps any base policy (LLM agent, rule-based, random) and handles
    difficulty selection automatically. Pass to rollout_func as the
    task_id selector.

    Usage:
        curriculum = CurriculumPolicy(base_policy=RuleBasedPolicy(), window=10)
        task_id = curriculum.select_task()          # auto-selects difficulty
        curriculum.record_score(0.82)               # update rolling window
    """

    def __init__(
        self,
        base_policy=None,
        window: int = 10,
        thresholds: list = None,
    ):
        self.base_policy = base_policy
        self.window = window
        self.thresholds = thresholds or [0.65, 0.78]
        self._score_history: list[float] = []
        self._task_history: list[str] = []

    def select_task(self) -> str:
        """Select next task difficulty based on rolling average score."""
        if len(self._score_history) < 3:
            return "easy"  # bootstrap on easy until we have enough data
        avg = sum(self._score_history[-self.window:]) / min(len(self._score_history), self.window)
        if avg < self.thresholds[0]:
            return "easy"
        elif avg < self.thresholds[1]:
            return "medium"
        else:
            return "hard"

    def record_score(self, score: float) -> None:
        """Record episode score to update rolling window."""
        self._score_history.append(score)

    def rolling_average(self) -> float:
        """Return current rolling average over the last `window` episodes."""
        if not self._score_history:
            return 0.0
        recent = self._score_history[-self.window:]
        return round(sum(recent) / len(recent), 3)

    def act(self, observation: dict) -> dict:
        """Delegate action to base_policy if provided."""
        if self.base_policy:
            return self.base_policy.act(observation)
        raise ValueError("CurriculumPolicy.act() requires a base_policy. "
                         "Set base_policy=RuleBasedPolicy() or your LLM agent.")

    def summary(self) -> dict:
        """Return curriculum progress summary."""
        return {
            "episodes_played":  len(self._score_history),
            "rolling_average":  self.rolling_average(),
            "current_level":    self.select_task(),
            "thresholds":       self.thresholds,
            "recent_scores":    self._score_history[-self.window:],
        }
