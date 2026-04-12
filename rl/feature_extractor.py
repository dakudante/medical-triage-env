from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

TEXT_KEYWORDS: List[str] = [
    "chest pain", "shortness of breath", "overdose", "poison", "burn", "fracture",
    "seizure", "bleeding", "pregnant", "pregnancy", "fever", "infection",
    "trauma", "headache", "abdominal pain", "stroke", "weakness", "palpitations",
    "diabetic", "anaphylaxis", "allergy", "collapse", "unconscious", "vomiting",
]


def _parse_bp(bp_value: str | None) -> tuple[float, float]:
    if not bp_value or "/" not in str(bp_value):
        return 120.0, 80.0
    try:
        sys_bp, dia_bp = str(bp_value).split("/", 1)
        return float(sys_bp), float(dia_bp)
    except Exception:
        return 120.0, 80.0


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo)


@dataclass
class ObservationFeatureExtractor:
    keyword_vocab: List[str] = None

    def __post_init__(self) -> None:
        if self.keyword_vocab is None:
            self.keyword_vocab = list(TEXT_KEYWORDS)

    @property
    def feature_dim(self) -> int:
        # numerics + one-hot like keyword flags
        return 21 + len(self.keyword_vocab)

    def text_source(self, observation) -> str:
        parts = [
            getattr(observation, "presentation", "") or "",
            getattr(observation, "nurse_summary", "") or "",
            getattr(observation, "message", "") or "",
        ]
        return " ".join(parts).lower()

    def transform(self, observation) -> torch.Tensor:
        vitals: Dict[str, float] = dict(getattr(observation, "vitals", {}) or {})
        bp_sys, bp_dia = _parse_bp(vitals.get("bp"))
        hospital = getattr(observation, "hospital_state", None)

        features = [
            _norm(float(vitals.get("hr", 80)), 30, 180),
            _norm(float(vitals.get("o2_sat", 98)), 50, 100),
            _norm(float(vitals.get("rr", 18)), 5, 45),
            _norm(float(vitals.get("temp", 37.0)), 34, 42),
            _norm(bp_sys, 60, 220),
            _norm(bp_dia, 30, 140),
            _norm(float(getattr(observation, "step", 0)), 0, max(1, float(getattr(observation, "max_steps", 6)))),
            _norm(float(getattr(observation, "max_steps", 6)), 1, 10),
            _norm(float(getattr(observation, "timesteps_waited", 0)), 0, 10),
            float(bool(getattr(observation, "has_deteriorated", False))),
            float(bool(getattr(observation, "partial_obs_applied", False))),
            float(bool(getattr(observation, "handoff_mode", False))),
            _norm(float(getattr(observation, "current_esi", 3)), 1, 5),
            _norm(float(getattr(observation, "consciousness_score", 1.0)), 0, 1),
            _norm(float(getattr(observation, "best_score_so_far", 0.0)), 0, 1),
        ]

        if hospital is not None:
            features.extend([
                _norm(float(hospital.icu_beds_available), 0, max(1, float(hospital.icu_beds_total))),
                _norm(float(hospital.er_beds_available), 0, max(1, float(hospital.er_beds_total))),
                _norm(float(hospital.resus_bays_available), 0, max(1, float(hospital.resus_bays_total))),
                _norm(float(hospital.ventilators_available), 0, max(1, float(hospital.ventilators_total))),
                _norm(float(hospital.doctors_available), 0, 10),
                _norm(float(hospital.nurses_available), 0, 20),
            ])
        else:
            features.extend([0.5] * 6)

        text = self.text_source(observation)
        features.extend(1.0 if kw in text else 0.0 for kw in self.keyword_vocab)

        return torch.tensor(features, dtype=torch.float32)
