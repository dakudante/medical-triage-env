"""
Medical Triage OpenEnv — Core Environment (V3)
===============================================
New systems over V2:
  1. HospitalResourceManager  — tracks bed / device / staff availability
  2. PatientProgressionEngine — degrades vitals per timestep
  3. XAIEngine                — generates structured decision explanations
  4. ProceduralPatientGenerator — samples patients from condition templates
  5. Multi-objective reward   — accuracy + resource + delay + mortality + over/undertriage
"""

from __future__ import annotations

import copy
import random
import time
import uuid
from typing import Optional

from models import (
    TriageAction, TriageObservation, TriageReward, TaskConfig,
    HospitalState, PatientProgressionState, XAIExplanation,
    DifferentialDiagnosis, ResetRequest,
)
from server.patients import (
    PATIENTS, PATIENT_MAP, EASY_CASES, MEDIUM_CASES, HARD_CASES,
    PAEDIATRIC_CASES,
    DEPARTMENTS, ESI_DESCRIPTIONS,
    CONDITION_TEMPLATES, TEMPLATE_MAP, generate_patient_from_template,
)


# ─────────────────────────────────────────────────────────────────────────────
# REWARD WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

REWARD_WEIGHTS = {
    "accuracy": 0.70,        # ESI + department composite
    "resource": 0.30,        # resource allocation efficiency
    "delay": 0.15,           # subtracted for wait time
    "mortality": 0.15,       # subtracted for deterioration / death
}

OVERTRIAGE_PENALTY = 0.15    # sending ESI-4/5 to ICU/Resus
UNDERTRIAGE_PENALTY_MILD = 0.15   # ESI-1 assigned ESI-3
UNDERTRIAGE_PENALTY_SEVERE = 0.35  # ESI-1/2 assigned ESI-4/5

TASK_CONFIGS = {
    "easy": {
        "name": "Textbook Triage",
        "description": "Clear, unambiguous presentations with obvious ESI level and department.",
        "cases": EASY_CASES,
    },
    "medium": {
        "name": "Overlapping Presentations",
        "description": "Symptoms that could indicate multiple conditions. Clinical reasoning required.",
        "cases": MEDIUM_CASES,
    },
    "hard": {
        "name": "Subtle & Ambiguous Cases",
        "description": "Complex patients with subtle findings, rare presentations, or misleading combinations.",
        "cases": HARD_CASES,
    },
    # Pillar 1.3: Paediatric-specific task
    "paediatric": {
        "name": "Paediatric Triage",
        "description": (
            "Children aged 0-16 with age-appropriate vital ranges. "
            "Paediatric normal values differ from adults — HR up to 140 is normal in infants. "
            "Cases include febrile seizure, epiglottitis, and appendicitis in adolescents."
        ),
        "cases": PAEDIATRIC_CASES,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: HOSPITAL RESOURCE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class HospitalResourceManager:
    """Manages hospital capacity and resource allocation across an episode."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.state = HospitalState(
            icu_beds_total=cfg.get("icu_beds", 4),
            icu_beds_available=cfg.get("icu_beds", 4),
            er_beds_total=cfg.get("er_beds", 10),
            er_beds_available=cfg.get("er_beds", 10),
            resus_bays_total=cfg.get("resus_bays", 2),
            resus_bays_available=cfg.get("resus_bays", 2),
            ventilators_total=cfg.get("ventilators", 3),
            ventilators_available=cfg.get("ventilators", 3),
            ct_scanners_total=cfg.get("ct_scanners", 2),
            ct_scanners_available=cfg.get("ct_scanners", 2),
            mri_scanners_total=cfg.get("mri_scanners", 1),
            mri_scanners_available=cfg.get("mri_scanners", 1),
            or_rooms_total=cfg.get("or_rooms", 2),
            or_rooms_available=cfg.get("or_rooms", 2),
            cardiac_monitors_total=cfg.get("cardiac_monitors", 6),
            cardiac_monitors_available=cfg.get("cardiac_monitors", 6),
            cath_labs_total=cfg.get("cath_labs", 1),
            cath_labs_available=cfg.get("cath_labs", 1),
            doctors_available=cfg.get("doctors", 3),
            nurses_available=cfg.get("nurses", 6),
        )
        # Track which resources were allocated per patient for later release
        self._allocations: dict[str, dict] = {}

    def check(self, patient_id: str, resources_required: dict) -> tuple[bool, list[str]]:
        return self.state.can_fulfil(resources_required)

    def admit(self, patient_id: str, resources_required: dict) -> None:
        self.state.allocate(resources_required)
        self._allocations[patient_id] = resources_required
        self.state.patients_waiting = max(0, self.state.patients_waiting - 1)

    def discharge(self, patient_id: str) -> None:
        if patient_id in self._allocations:
            self.state.release(self._allocations.pop(patient_id))

    def score_resource_decision(
        self, action: TriageAction, patient: dict
    ) -> tuple[float, list[str]]:
        """
        Score the agent's resource allocation decision.
        Returns (score 0-1, feedback).
        """
        feedback = []
        required = patient.get("resources_required", {})
        request = action.resource_request

        if request is None:
            # No resource request — infer from ESI level
            feedback.append("No explicit resource_request provided — inferred from ESI level.")
            return 0.7, feedback   # partial credit for implicit allocation

        # Convert request to comparable dict
        req_dict = request.model_dump()

        # Count matches vs required
        true_positives = sum(
            1 for k in ["icu_bed", "er_bed", "ventilator", "ct_scanner",
                        "or_room", "cardiac_monitor", "cath_lab"]
            if req_dict.get(k, False) == bool(required.get(k, False))
        )
        total_fields = 7
        accuracy = true_positives / total_fields

        # Penalise requesting scarce resources unnecessarily (overtriage of resources)
        wasted = []
        for k in ["icu_bed", "ventilator", "or_room", "cath_lab"]:
            if req_dict.get(k, False) and not required.get(k, False):
                wasted.append(k.replace("_", " "))

        waste_penalty = len(wasted) * 0.1
        if wasted:
            feedback.append(
                f"Unnecessary high-cost resource request: {', '.join(wasted)}. "
                f"These are now unavailable for other patients."
            )

        # Penalise missing critical resources
        missed = []
        for k in ["icu_bed", "ventilator", "cardiac_monitor"]:
            if not req_dict.get(k, False) and required.get(k, False):
                missed.append(k.replace("_", " "))
        if missed:
            feedback.append(
                f"Missing critical resources in request: {', '.join(missed)}."
            )

        score = max(0.0, min(1.0, accuracy - waste_penalty))

        if score >= 0.9:
            feedback.append("Resource allocation: optimal.")
        elif score >= 0.7:
            feedback.append("Resource allocation: acceptable.")
        else:
            feedback.append("Resource allocation: needs review.")

        return round(score, 3), feedback

    def snapshot(self) -> HospitalState:
        return self.state.model_copy(deep=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: PATIENT PROGRESSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PatientProgressionEngine:
    """Manages per-patient vital decay over time."""

    def __init__(self):
        self._states: dict[str, PatientProgressionState] = {}

    def register(self, patient: dict) -> PatientProgressionState:
        vitals = dict(patient["vitals"])
        state = PatientProgressionState(
            patient_id=patient["id"],
            current_vitals=copy.deepcopy(vitals),
            initial_vitals=copy.deepcopy(vitals),
            current_esi=patient["correct_esi"],
            initial_esi=patient["correct_esi"],
        )
        self._states[patient["id"]] = state
        return state

    def tick(self, patient_id: str, patient: dict) -> list[str]:
        """
        Advance patient's condition by one timestep.
        Returns a list of deterioration event strings.
        """
        state = self._states.get(patient_id)
        if state is None or state.is_deceased:
            return []
        progression = patient.get("progression", {})
        return state.apply_timestep(progression)

    def get_state(self, patient_id: str) -> Optional[PatientProgressionState]:
        return self._states.get(patient_id)

    # Pillar 2.1: acuity-proportional delay rates (per step)
    _ESI_DELAY_RATES = {1: 0.12, 2: 0.08, 3: 0.05, 4: 0.02, 5: 0.01}

    def compute_delay_penalty(self, patient_id: str, base_penalty: float = 0.05) -> float:
        """
        Pillar 2.1: acuity-proportional delay penalty.
        ESI-1: -0.12/step, ESI-2: -0.08/step, ESI-3: -0.05/step, ESI-4/5: minimal.
        Capped at 0.40 to avoid runaway penalties on 6-step episodes.
        """
        state = self._states.get(patient_id)
        if not state:
            return 0.0
        esi = state.current_esi
        rate = self._ESI_DELAY_RATES.get(esi, base_penalty)
        return min(0.40, round(state.timesteps_waited * rate, 3))

    def compute_mortality_penalty(self, patient_id: str) -> float:
        """Penalty proportional to accumulated mortality risk."""
        state = self._states.get(patient_id)
        if not state:
            return 0.0
        if state.is_deceased:
            return 0.50
        return round(state.cumulative_mortality_risk * 0.4, 3)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: XAI ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class XAIEngine:
    """Generates structured, human-readable explanations for triage decisions."""

    def generate(
        self,
        patient: dict,
        prog_state: Optional[PatientProgressionState] = None,
    ) -> XAIExplanation:
        """Build an XAIExplanation from a patient's xai_metadata."""
        meta = patient.get("xai_metadata", {})
        vitals = (prog_state.current_vitals if prog_state else patient["vitals"])

        # Sort symptoms by weight
        sym_weights = meta.get("symptom_weights", {})
        top_symptoms = sorted(
            [{"symptom": k, "weight": v} for k, v in sym_weights.items()],
            key=lambda x: -x["weight"],
        )

        differentials = [
            DifferentialDiagnosis(
                diagnosis=d["diagnosis"], probability=d["probability"]
            )
            for d in meta.get("differential_diagnoses", [])
        ]

        # Build summary
        primary = meta.get("primary_diagnosis", "Unknown")
        confidence = meta.get("confidence", 0.8)
        top_sym_str = ", ".join(s["symptom"] for s in top_symptoms[:3])
        vital_flags = meta.get("vital_flags", [])
        vital_str = "; ".join(vital_flags[:2]) if vital_flags else "vitals concerning"

        summary = (
            f"This patient's presentation most likely represents {primary} "
            f"(confidence {confidence:.0%}). Key clinical features: {top_sym_str}. "
            f"Abnormal findings: {vital_str}. "
            f"{meta.get('key_reasoning_points', [''])[0] if meta.get('key_reasoning_points') else ''}"
        )

        return XAIExplanation(
            primary_diagnosis=primary,
            confidence=confidence,
            differential_diagnoses=differentials,
            top_symptoms=top_symptoms[:5],
            vital_flags=vital_flags,
            key_reasoning_points=meta.get("key_reasoning_points", []),
            summary=summary,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: PROCEDURAL PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ProceduralPatientGenerator:
    """Generates statistically-realistic patients from condition templates."""

    def __init__(self):
        self.templates = CONDITION_TEMPLATES
        self._easy = [t for t in self.templates if t["difficulty"] == "easy"]
        self._medium = [t for t in self.templates if t["difficulty"] == "medium"]
        self._hard = [t for t in self.templates if t["difficulty"] == "hard"]

    def generate(self, difficulty: str = "random") -> dict:
        if difficulty == "easy":
            pool = self._easy
        elif difficulty == "medium":
            pool = self._medium
        elif difficulty == "hard":
            pool = self._hard
        else:
            pool = self.templates

        template = random.choice(pool)
        return generate_patient_from_template(template)

    def generate_batch(self, n: int, difficulty: str = "random") -> list[dict]:
        return [self.generate(difficulty) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_esi_score(patient: dict, submitted_esi: int) -> tuple[float, float, list[str]]:
    correct_esi = patient["correct_esi"]
    partial_credit = patient["esi_partial_credit"]
    feedback = []

    base_score = partial_credit.get(submitted_esi, 0.0)

    undertriage_penalty = 0.0
    if correct_esi <= 2 and submitted_esi >= 4:
        undertriage_penalty = UNDERTRIAGE_PENALTY_SEVERE
        feedback.append(
            f"DANGEROUS UNDERTRIAGE: Patient needed ESI-{correct_esi} but assigned "
            f"ESI-{submitted_esi}. This patient could deteriorate fatally in the waiting room."
        )
    elif correct_esi == 1 and submitted_esi >= 3:
        undertriage_penalty = UNDERTRIAGE_PENALTY_MILD
        feedback.append(
            f"Undertriage risk: ESI-1 patient assigned ESI-{submitted_esi}. "
            f"Immediate intervention is required."
        )
    elif submitted_esi == correct_esi:
        feedback.append(f"Correct ESI level: ESI-{correct_esi}.")
    elif abs(submitted_esi - correct_esi) == 1:
        feedback.append(
            f"ESI level close: assigned ESI-{submitted_esi}, correct is ESI-{correct_esi}."
        )
    else:
        feedback.append(
            f"ESI level incorrect: assigned ESI-{submitted_esi}, correct is ESI-{correct_esi}."
        )

    final_score = max(0.01, base_score - undertriage_penalty)
    return round(final_score, 3), round(undertriage_penalty, 3), feedback


def compute_department_score(patient: dict, submitted_dept: str) -> tuple[float, list[str]]:
    dept_scores = patient["department_scores"]
    correct_dept = patient["correct_department"]
    feedback = []

    normalized = {k.lower(): v for k, v in dept_scores.items()}
    score = normalized.get(submitted_dept.lower(), 0.0)

    if submitted_dept.lower() == correct_dept.lower():
        feedback.append(f"Correct department: {correct_dept}.")
    elif score > 0.5:
        feedback.append(
            f"Department acceptable: {submitted_dept} is reasonable, but {correct_dept} is optimal."
        )
    elif score > 0:
        feedback.append(f"Department suboptimal: consider {correct_dept}.")
    else:
        if submitted_dept not in DEPARTMENTS:
            feedback.append(
                f"Unknown department '{submitted_dept}'. Valid: {', '.join(DEPARTMENTS)}."
            )
        else:
            feedback.append(f"Incorrect department: {submitted_dept}. Optimal: {correct_dept}.")

    return round(score, 3), feedback


def compute_overtriage_penalty(patient: dict, action: TriageAction) -> tuple[float, list[str]]:
    """Penalise sending a low-acuity patient to high-acuity resources."""
    correct_esi = patient["correct_esi"]
    feedback = []
    penalty = 0.0

    if correct_esi >= 4:
        if action.department in ("Resuscitation",):
            penalty = OVERTRIAGE_PENALTY
            feedback.append(
                f"OVERTRIAGE: ESI-{correct_esi} patient sent to Resuscitation. "
                f"This wastes critical capacity."
            )
        elif action.department in ("Emergency", "Cardiology") and correct_esi == 5:
            penalty = OVERTRIAGE_PENALTY * 0.5
            feedback.append(
                f"Minor overtriage: ESI-5 patient sent to {action.department}. "
                f"General practice or General department is more appropriate."
            )
        req = action.resource_request
        if req and (req.icu_bed or req.ventilator):
            penalty += OVERTRIAGE_PENALTY
            feedback.append(
                f"Resource overtriage: ICU bed / ventilator requested for ESI-{correct_esi} patient."
            )

    return round(penalty, 3), feedback


def compute_calibration_bonus(action: TriageAction, esi_score: float) -> float:
    """
    Pillar 3.2: Confidence-calibration bonus.
    Correct decisions with clear reasoning get +0.02.
    Wrong decisions with overconfident language get -0.05.
    Overconfidence signals: short reasoning (<20 chars) or words like 'obvious', 'clearly', 'definitely'.
    """
    reasoning = (action.reasoning or "").strip()
    is_correct = esi_score >= 0.90
    is_confident = (
        len(reasoning) < 20 or
        any(w in reasoning.lower() for w in ["obvious", "clearly", "definitely", "certainly", "without doubt"])
    )
    has_reasoning = len(reasoning) >= 30

    if is_correct and has_reasoning:
        return 0.02   # rewarded for explaining correct decision
    elif not is_correct and is_confident:
        return -0.05  # penalised for overconfident wrong decision
    return 0.0


def compute_safety_score(patient: dict, action: TriageAction, undertriage_penalty: float) -> float:
    """
    Pillar 5.2: Safety tier score — only meaningful for ESI-1/2 patients.
    For non-critical patients always returns 1.0 (no safety concern).
    For ESI-1/2 patients: 1.0 if correctly triaged, degrades proportionally to undertriage severity.
    """
    correct_esi = patient.get("correct_esi", 3)
    if correct_esi >= 3:
        return 1.0  # not a critical patient — safety score not applicable
    if undertriage_penalty >= 0.35:
        return 0.01  # dangerous undertriage of critical patient
    elif undertriage_penalty >= 0.15:
        return 0.40  # moderate undertriage
    elif action.esi_level <= 2:
        return 1.0   # correctly identified as critical
    return 0.70      # partial credit


def score_triage_v3(
    patient: dict,
    action: TriageAction,
    resource_manager: HospitalResourceManager,
    progression_engine: PatientProgressionEngine,
) -> tuple[float, TriageReward, list[str]]:
    """
    Full multi-objective scoring function for V4 (reward formula upgrade).
    """
    # 1. Clinical accuracy (ESI + department)
    esi_score, undertriage_penalty, esi_feedback = compute_esi_score(patient, action.esi_level)
    dept_score, dept_feedback = compute_department_score(patient, action.department)
    accuracy_score = round(0.6 * esi_score + 0.4 * dept_score, 3)

    # 2. Resource allocation
    resource_score, resource_feedback = resource_manager.score_resource_decision(action, patient)

    # 3. Overtriage penalty
    overtriage_penalty, overtriage_feedback = compute_overtriage_penalty(patient, action)

    # 4. Delay and mortality penalties (from progression engine)
    patient_id = patient["id"]
    delay_penalty = progression_engine.compute_delay_penalty(patient_id)
    mortality_penalty = progression_engine.compute_mortality_penalty(patient_id)

    # 5. Pillar 3.2: Confidence-calibration bonus
    calibration_bonus = compute_calibration_bonus(action, esi_score)

    # 6. Pillar 5.2: Safety tier score
    safety_score = compute_safety_score(patient, action, undertriage_penalty)

    # 7. Compose reward (V4 formula)
    total = (
        REWARD_WEIGHTS["accuracy"] * accuracy_score
        + REWARD_WEIGHTS["resource"] * resource_score
        - REWARD_WEIGHTS["delay"] * delay_penalty
        - REWARD_WEIGHTS["mortality"] * mortality_penalty
        - overtriage_penalty
        - undertriage_penalty
        + calibration_bonus
    )
    total = max(0.01, min(0.99, round(total, 3)))

    reward = TriageReward(
        value=total,
        esi_score=esi_score,
        department_score=dept_score,
        resource_score=resource_score,
        delay_penalty=delay_penalty,
        mortality_penalty=mortality_penalty,
        overtriage_penalty=overtriage_penalty,
        undertriage_penalty=undertriage_penalty,
        calibration_bonus=calibration_bonus,
        safety_score=safety_score,
        breakdown={
            "accuracy_component": round(REWARD_WEIGHTS["accuracy"] * accuracy_score, 3),
            "resource_component": round(REWARD_WEIGHTS["resource"] * resource_score, 3),
            "delay_penalty": -round(REWARD_WEIGHTS["delay"] * delay_penalty, 3),
            "mortality_penalty": -round(REWARD_WEIGHTS["mortality"] * mortality_penalty, 3),
            "overtriage_penalty": -overtriage_penalty,
            "undertriage_penalty": -undertriage_penalty,
            "calibration_bonus": calibration_bonus,
            "safety_score": safety_score,
        },
    )

    all_feedback = (
        esi_feedback + dept_feedback + resource_feedback
        + overtriage_feedback
    )

    # Progression events
    prog_state = progression_engine.get_state(patient_id)
    if prog_state and prog_state.deterioration_events:
        all_feedback.extend(prog_state.deterioration_events[-3:])

    if action.reasoning:
        all_feedback.append(f'Your reasoning: "{action.reasoning[:150]}"')

    return total, reward, all_feedback


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

try:
    from openenv.core import Environment as OpenEnvBase
except ImportError:
    OpenEnvBase = object


class MedicalTriageEnvironment(OpenEnvBase):
    """
    Medical Triage Environment — OpenEnv compliant.
    Inherits from openenv.core.Environment (with graceful fallback).
    Integrates ResourceManager, ProgressionEngine, XAIEngine, and ProceduralGenerator.
    """

    def __init__(self):
        self.episode_id: Optional[str] = None
        self.current_patient: Optional[dict] = None
        self.current_task_id: Optional[str] = None
        self.step_count: int = 0
        self.max_steps: int = 6          # Pillar 2.1: extended to 6 steps
        self.done: bool = False
        self.best_score: float = 0.0
        self.history: list[dict] = []
        self.start_time: float = time.time()

        # Pillar 2.2 / 2.3: partial observability and nurse handoff state
        self.partial_obs: bool = False
        self.missing_vitals: list[str] = []
        self.seed: Optional[int] = None

        # Pillar 3.3 / 5: outcome tracking and audit log
        self.final_outcome: Optional[str] = None
        self._audit_log: list[dict] = []   # persists across episodes

        # V3 modules
        self.resource_manager: Optional[HospitalResourceManager] = None
        self.progression_engine: Optional[PatientProgressionEngine] = None
        self.xai_engine = XAIEngine()
        self.procedural_gen = ProceduralPatientGenerator()

    def reset(self, request: Optional[ResetRequest] = None) -> TriageObservation:
        task_id = None
        hospital_config = None
        use_procedural = False

        if request:
            task_id = request.task_id
            hospital_config = request.hospital_config
            use_procedural = request.use_procedural

            # Pillar 4.4: seed-based deterministic evaluation
            self.seed = request.seed
            if self.seed is not None:
                random.seed(self.seed)

            # Pillar 2.2: partial observability mode
            self.partial_obs = getattr(request, "partial_obs", False)
        else:
            self.seed = None
            self.partial_obs = False

        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.done = False
        self.best_score = 0.0
        self.history = []
        self.missing_vitals = []
        self.final_outcome = None
        self.start_time = time.time()

        # Pillar 2.1: extended episode horizon
        # mass_casualty keeps 3 steps (5 patients); all others get 6 steps
        self.max_steps = 3 if task_id == "mass_casualty" else 6

        # Resolve task
        if task_id and task_id in TASK_CONFIGS:
            self.current_task_id = task_id
        else:
            self.current_task_id = random.choice(["easy", "medium", "hard"])

        # Select patient — hard tasks always use curated cases (V5 fix preserved)
        if use_procedural and self.current_task_id != "hard":
            self.current_patient = self.procedural_gen.generate(self.current_task_id)
        else:
            cases = TASK_CONFIGS[self.current_task_id]["cases"]
            self.current_patient = copy.deepcopy(random.choice(cases))

        # Pillar 2.2: apply partial observability — hide 1-2 vitals randomly
        if self.partial_obs:
            hideable = ["rr", "temp", "o2_sat", "hr"]
            n_hide = random.randint(1, 2)
            self.missing_vitals = random.sample(hideable, n_hide)
        else:
            self.missing_vitals = []

        # Initialise modules
        self.resource_manager = HospitalResourceManager(hospital_config)
        self.progression_engine = PatientProgressionEngine()
        self.progression_engine.register(self.current_patient)

        return self._build_observation(
            message="New episode started. Assess the patient and assign triage level.",
            last_scores=None,
            feedback=[],
        )

    def step(self, action: TriageAction) -> tuple[TriageObservation, TriageReward, bool, dict]:
        if self.done:
            obs = self._build_observation("Episode already done. Call /reset.", None, [])
            return obs, TriageReward(value=0.01, esi_score=0.01, department_score=0.01, breakdown={}), True, {"error": "episode_done"}

        self.step_count += 1
        patient = self.current_patient

        # 1. Advance patient progression before scoring (they waited another step)
        if self.step_count > 1:
            decay_events = self.progression_engine.tick(patient["id"], patient)
        else:
            decay_events = []

        # 2. Score the action
        total_score, reward, feedback = score_triage_v3(
            patient, action, self.resource_manager, self.progression_engine
        )

        self.best_score = max(self.best_score, total_score)

        # 3. Handle routing decision
        if action.routing_decision == "admit" or action.routing_decision is None:
            can_admit, gaps = self.resource_manager.check(
                patient["id"], patient.get("resources_required", {})
            )
            if can_admit:
                self.resource_manager.admit(patient["id"], patient.get("resources_required", {}))
                if gaps:
                    feedback.append("Resources allocated successfully.")
            else:
                feedback.append(
                    f"Resource constraint: cannot admit immediately — "
                    f"unavailable: {', '.join(gaps)}. Patient queued."
                )
                self.resource_manager.state.patients_waiting += 1

        # 4. Episode completion logic
        if total_score >= 0.75:
            feedback.append("Excellent triage decision!")
            self.done = True
        elif total_score >= 0.7:
            feedback.append("Good assessment. Review remaining issues.")
        elif total_score < 0.3:
            feedback.append("Significant revision needed. Review ESI scale, resource allocation, and vitals.")

        prog_state = self.progression_engine.get_state(patient["id"])
        if prog_state and prog_state.is_deceased:
            feedback.append(
                "OUTCOME: Patient deteriorated fatally due to delayed or incorrect triage."
            )
            self.done = True

        if self.step_count >= self.max_steps:
            self.done = True
            feedback.append(
                f"Episode ended after {self.max_steps} attempts. "
                f"Correct: ESI-{patient['correct_esi']} → {patient['correct_department']}. "
                f"Reasoning: {patient['reasoning']}"
            )

        self.history.append({
            "step": self.step_count,
            "esi_level": action.esi_level,
            "department": action.department,
            "routing_decision": action.routing_decision,
            "esi_score": reward.esi_score,
            "dept_score": reward.department_score,
            "resource_score": reward.resource_score,
            "delay_penalty": reward.delay_penalty,
            "mortality_penalty": reward.mortality_penalty,
            "total_score": total_score,
        })

        obs = self._build_observation(
            message="Triage decision evaluated.",
            last_scores={
                "esi": reward.esi_score,
                "dept": reward.department_score,
                "resource": reward.resource_score,
                "total": total_score,
            },
            feedback=feedback,
        )

        # Pillar 5.1: Failure mode classifier
        failure_mode = self._classify_failure(patient, action, reward, total_score)

        # Pillar 3.3: Episode-level outcome label (set when episode ends)
        if self.done:
            prog_state = self.progression_engine.get_state(patient["id"])
            is_fatal = prog_state and prog_state.is_deceased
            if is_fatal:
                self.final_outcome = "FATAL"
            elif reward.undertriage_penalty >= 0.35:
                self.final_outcome = "HARMFUL"
            elif self.best_score >= 0.75 and self.step_count == 1:
                self.final_outcome = "OPTIMAL"
            elif self.best_score >= 0.75 and self.step_count <= 3:
                self.final_outcome = "ACCEPTABLE"
            elif self.best_score >= 0.75:
                self.final_outcome = "DELAYED"
            else:
                self.final_outcome = "HARMFUL"

            # Propagate final_outcome into reward
            reward.final_outcome = self.final_outcome

        # Pillar 5.3: Audit log entry
        audit_entry = {
            "episode_id": self.episode_id,
            "task_id": self.current_task_id,
            "patient_id": patient["id"],
            "difficulty": patient["difficulty"],
            "step": self.step_count,
            "action": {
                "esi_level": action.esi_level,
                "department": action.department,
                "routing_decision": action.routing_decision,
                "reasoning": (action.reasoning or "")[:200],
            },
            "scores": {
                "total": total_score,
                "esi": reward.esi_score,
                "dept": reward.department_score,
                "resource": reward.resource_score,
                "safety": reward.safety_score,
                "calibration_bonus": reward.calibration_bonus,
            },
            "failure_mode": failure_mode,
            "final_outcome": self.final_outcome,
            "correct_esi": patient["correct_esi"],
            "correct_dept": patient["correct_department"],
            "done": self.done,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._audit_log.append(audit_entry)
        if len(self._audit_log) > 200:   # keep last 200 entries in memory
            self._audit_log = self._audit_log[-200:]

        info = {
            "step": self.step_count,
            "best_score": self.best_score,
            "task_id": self.current_task_id,
            "patient_id": patient["id"],
            "correct_esi": patient["correct_esi"],
            "correct_dept": patient["correct_department"],
            "hospital_state": self.resource_manager.snapshot().model_dump(),
            "failure_mode": failure_mode,
            "final_outcome": self.final_outcome,
            "safety_score": reward.safety_score,
        }

        return obs, reward, self.done, info

    def _classify_failure(
        self, patient: dict, action: TriageAction, reward: "TriageReward", total_score: float
    ) -> str:
        """Pillar 5.1: Classify the primary failure mode for this step."""
        correct_esi = patient.get("correct_esi", 3)
        correct_dept = patient.get("correct_department", "")

        if reward.undertriage_penalty >= 0.35:
            return "UNDERTRIAGE_CRITICAL"
        elif reward.undertriage_penalty >= 0.15:
            return "UNDERTRIAGE_MODERATE"
        elif reward.overtriage_penalty >= 0.15:
            return "OVERTRIAGE"
        elif action.department.lower() != correct_dept.lower() and reward.esi_score >= 0.90:
            return "DEPARTMENT_MISMATCH"
        elif reward.resource_score < 0.50:
            return "RESOURCE_WASTE"
        elif reward.delay_penalty >= 0.20:
            return "DELAY_EXCESSIVE"
        elif total_score >= 0.75:
            return "CORRECT"
        else:
            return "PARTIAL"

    def get_tasks(self) -> list[TaskConfig]:
        return [
            TaskConfig(
                task_id=tid,
                name=cfg["name"],
                difficulty=tid,
                description=cfg["description"],
                action_schema={
                    "esi_level": {"type": "integer", "minimum": 1, "maximum": 5},
                    "department": {"type": "string", "enum": DEPARTMENTS},
                    "reasoning": {"type": "string"},
                    "resource_request": {
                        "type": "object",
                        "description": "Explicit resource allocation request",
                    },
                    "routing_decision": {
                        "type": "string",
                        "enum": ["admit", "wait", "reroute"],
                    },
                },
            )
            for tid, cfg in TASK_CONFIGS.items()
        ]

    def state(self) -> dict:
        prog_state = (
            self.progression_engine.get_state(self.current_patient["id"])
            if self.progression_engine and self.current_patient
            else None
        )
        return {
            "episode_id": self.episode_id,
            "task_id": self.current_task_id,
            "patient_id": self.current_patient["id"] if self.current_patient else None,
            "difficulty": self.current_patient["difficulty"] if self.current_patient else None,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "best_score": self.best_score,
            "history": self.history,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "hospital_state": self.resource_manager.snapshot().model_dump() if self.resource_manager else None,
            "progression": prog_state.model_dump() if prog_state else None,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        message: str,
        last_scores: Optional[dict],
        feedback: list[str],
    ) -> TriageObservation:
        patient = self.current_patient
        prog_state = (
            self.progression_engine.get_state(patient["id"])
            if self.progression_engine else None
        )
        xai = self.xai_engine.generate(patient, prog_state)

        # Resource gap check
        hospital_state = self.resource_manager.snapshot() if self.resource_manager else None
        _, gaps = (
            self.resource_manager.check(patient["id"], patient.get("resources_required", {}))
            if self.resource_manager else (True, [])
        )

        hint = None
        if self.step_count > 1 and (not last_scores or last_scores.get("total", 0) < 0.9):
            hint = patient.get("reasoning", "")

        current_vitals = prog_state.current_vitals if prog_state else patient["vitals"]
        current_esi = prog_state.current_esi if prog_state else patient["correct_esi"]

        # Pillar 2.2: apply partial observability — mask hidden vitals
        display_vitals = dict(current_vitals)
        if self.partial_obs and self.missing_vitals:
            for v in self.missing_vitals:
                if v in display_vitals:
                    display_vitals[v] = "not yet measured"

        # Pillar 2.3: nurse handoff format on step 2+ when partial_obs is active
        handoff_mode = False
        nurse_summary = None
        if self.partial_obs and self.step_count >= 2:
            handoff_mode = True
            measured = {k: v for k, v in display_vitals.items() if v != "not yet measured"}
            vital_str = ", ".join(f"{k.upper()} {v}" for k, v in measured.items())
            nurse_summary = (
                f"Charge nurse reports: {patient['presentation'][:120]}... "
                f"Measured vitals: {vital_str}. "
                f"Note: {', '.join(self.missing_vitals)} not yet obtained — patient arrived without prior workup."
            )

        return TriageObservation(
            episode_id=self.episode_id,
            task_id=self.current_task_id,
            difficulty=patient["difficulty"],
            step=self.step_count,
            max_steps=self.max_steps,
            done=self.done,
            message=message,
            patient_id=patient["id"],
            presentation=patient["presentation"],
            vitals=display_vitals,
            initial_vitals=prog_state.initial_vitals if prog_state else patient["vitals"],
            timesteps_waited=prog_state.timesteps_waited if prog_state else 0,
            consciousness_score=prog_state.consciousness_score if prog_state else 1.0,
            current_esi=current_esi,
            has_deteriorated=prog_state.deteriorated if prog_state else False,
            deterioration_events=prog_state.deterioration_events if prog_state else [],
            esi_scale={str(k): v for k, v in ESI_DESCRIPTIONS.items()},
            valid_departments=DEPARTMENTS,
            hospital_state=hospital_state,
            xai_explanation=xai,
            resources_required=patient.get("resources_required"),
            resource_gaps=gaps,
            last_esi_score=last_scores.get("esi") if last_scores else None,
            last_dept_score=last_scores.get("dept") if last_scores else None,
            last_resource_score=last_scores.get("resource") if last_scores else None,
            last_total_score=last_scores.get("total") if last_scores else None,
            best_score_so_far=self.best_score,
            feedback=feedback,
            hint=hint,
            partial_obs_applied=self.partial_obs,
            missing_vitals=self.missing_vitals,
            handoff_mode=handoff_mode,
            nurse_summary=nurse_summary,
        )
score_triage = score_triage_v3