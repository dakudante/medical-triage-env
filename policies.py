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
