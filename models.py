"""
Medical Triage OpenEnv — Expanded Pydantic Models (V3)
=======================================================
New additions over V2:
  - HospitalState          : tracks resource availability
  - PatientProgressionState: per-patient vital trajectory
  - XAIExplanation         : structured explainability output
  - TriageAction (expanded): resource_request + routing_decision
  - TriageObservation (expanded): hospital_state + xai_explanation + current vitals
  - TriageReward (expanded): multi-objective decomposition
"""

from __future__ import annotations
from enum import IntEnum
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict

try:
    from openenv.core import Environment as OpenEnvBase
except ImportError:
    # Graceful fallback if openenv-core is not installed in the validator context
    OpenEnvBase = object


# ─────────────────────────────────────────────────────────────────────────────
# ESI LEVEL ENUM  (Fix: explicit type-safe enum for triage levels)
# ─────────────────────────────────────────────────────────────────────────────

class ESILevel(IntEnum):
    """Emergency Severity Index levels used worldwide in ED triage."""
    IMMEDIATE   = 1  # Life-threatening — requires instant intervention
    EMERGENT    = 2  # High risk — should not wait
    URGENT      = 3  # Stable but needs multiple resources
    LESS_URGENT = 4  # Needs one resource only
    NON_URGENT  = 5  # No resources needed


# ─────────────────────────────────────────────────────────────────────────────
# HOSPITAL STATE
# ─────────────────────────────────────────────────────────────────────────────

class HospitalState(BaseModel):
    """
    Live snapshot of hospital resource availability.
    The Resource Manager updates this after every action.
    """
    # Bed capacity
    icu_beds_total: int = Field(4, description="Total ICU beds in this simulation.")
    icu_beds_available: int = Field(4, description="ICU beds currently free.")
    er_beds_total: int = Field(10, description="Total ER beds.")
    er_beds_available: int = Field(10, description="ER beds currently free.")
    resus_bays_total: int = Field(2, description="Total resuscitation bays.")
    resus_bays_available: int = Field(2, description="Resus bays currently free.")

    # Equipment
    ventilators_total: int = Field(3, description="Total ventilators.")
    ventilators_available: int = Field(3, description="Ventilators currently free.")
    ct_scanners_total: int = Field(2, description="Total CT scanners.")
    ct_scanners_available: int = Field(2, description="CT scanners currently free.")
    mri_scanners_total: int = Field(1, description="Total MRI scanners.")
    mri_scanners_available: int = Field(1, description="MRI scanners currently free.")
    or_rooms_total: int = Field(2, description="Total operating rooms.")
    or_rooms_available: int = Field(2, description="OR rooms currently free.")
    cardiac_monitors_total: int = Field(6, description="Total cardiac monitors.")
    cardiac_monitors_available: int = Field(6, description="Monitors currently free.")
    cath_labs_total: int = Field(1, description="Total cath labs.")
    cath_labs_available: int = Field(1, description="Cath labs currently free.")

    # Staff
    doctors_available: int = Field(3, description="Doctors currently free.")
    nurses_available: int = Field(6, description="Nurses currently free.")

    # Queue
    patients_waiting: int = Field(0, description="Patients in waiting room.")
    patients_in_treatment: int = Field(0, description="Patients currently receiving care.")

    def can_fulfil(self, resources_required: dict) -> tuple[bool, list[str]]:
        """
        Check whether the hospital can currently fulfil a patient's resource needs.
        Returns (can_fulfil, list_of_unavailable_resources).
        """
        gaps = []
        if resources_required.get("icu_bed") and self.icu_beds_available < 1:
            gaps.append("ICU bed")
        if resources_required.get("er_bed") and self.er_beds_available < 1:
            gaps.append("ER bed")
        if resources_required.get("ventilator") and self.ventilators_available < 1:
            gaps.append("ventilator")
        if resources_required.get("ct_scanner") and self.ct_scanners_available < 1:
            gaps.append("CT scanner")
        if resources_required.get("or_room") and self.or_rooms_available < 1:
            gaps.append("OR room")
        if resources_required.get("cardiac_monitor") and self.cardiac_monitors_available < 1:
            gaps.append("cardiac monitor")
        if resources_required.get("cath_lab") and self.cath_labs_available < 1:
            gaps.append("cath lab")

        nr = resources_required.get("nurse_ratio", 0)
        if nr > 0 and self.nurses_available < nr:
            gaps.append(f"{nr} nurse(s)")
        if self.doctors_available < 1:
            gaps.append("doctor")

        return (len(gaps) == 0), gaps

    def allocate(self, resources_required: dict) -> None:
        """Consume resources for an admitted patient."""
        if resources_required.get("icu_bed"):
            self.icu_beds_available = max(0, self.icu_beds_available - 1)
        if resources_required.get("er_bed"):
            self.er_beds_available = max(0, self.er_beds_available - 1)
        if resources_required.get("ventilator"):
            self.ventilators_available = max(0, self.ventilators_available - 1)
        if resources_required.get("ct_scanner"):
            self.ct_scanners_available = max(0, self.ct_scanners_available - 1)
        if resources_required.get("or_room"):
            self.or_rooms_available = max(0, self.or_rooms_available - 1)
        if resources_required.get("cardiac_monitor"):
            self.cardiac_monitors_available = max(0, self.cardiac_monitors_available - 1)
        if resources_required.get("cath_lab"):
            self.cath_labs_available = max(0, self.cath_labs_available - 1)
        nr = resources_required.get("nurse_ratio", 0)
        if nr > 0:
            self.nurses_available = max(0, self.nurses_available - nr)
        self.doctors_available = max(0, self.doctors_available - 1)
        self.patients_in_treatment += 1

    def release(self, resources_required: dict) -> None:
        """Free resources when a patient is discharged / transferred."""
        if resources_required.get("icu_bed"):
            self.icu_beds_available = min(self.icu_beds_total, self.icu_beds_available + 1)
        if resources_required.get("er_bed"):
            self.er_beds_available = min(self.er_beds_total, self.er_beds_available + 1)
        if resources_required.get("ventilator"):
            self.ventilators_available = min(self.ventilators_total, self.ventilators_available + 1)
        if resources_required.get("ct_scanner"):
            self.ct_scanners_available = min(self.ct_scanners_total, self.ct_scanners_available + 1)
        if resources_required.get("or_room"):
            self.or_rooms_available = min(self.or_rooms_total, self.or_rooms_available + 1)
        if resources_required.get("cardiac_monitor"):
            self.cardiac_monitors_available = min(self.cardiac_monitors_total, self.cardiac_monitors_available + 1)
        if resources_required.get("cath_lab"):
            self.cath_labs_available = min(self.cath_labs_total, self.cath_labs_available + 1)
        nr = resources_required.get("nurse_ratio", 0)
        if nr > 0:
            self.nurses_available = min(self.nurses_available + nr, 99)
        self.doctors_available = min(self.doctors_available + 1, 99)
        self.patients_in_treatment = max(0, self.patients_in_treatment - 1)


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT PROGRESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

class PatientProgressionState(BaseModel):
    """
    Tracks a single patient's real-time vital trajectory.
    Updated each timestep the patient waits without treatment.
    """
    patient_id: str
    timesteps_waited: int = Field(0, description="Number of environment steps this patient has waited.")
    current_vitals: dict = Field(default_factory=dict, description="Current (possibly degraded) vitals.")
    initial_vitals: dict = Field(default_factory=dict, description="Vitals at presentation.")
    current_esi: int = Field(..., description="Current ESI level — may upgrade as condition worsens.")
    initial_esi: int = Field(..., description="ESI level at presentation.")
    consciousness_score: float = Field(1.0, ge=0.0, le=1.0, description="1.0=alert, 0.0=unresponsive.")
    cumulative_mortality_risk: float = Field(0.0, ge=0.0, le=1.0)
    deteriorated: bool = Field(False, description="True if patient has crossed a critical threshold.")
    deterioration_events: list[str] = Field(default_factory=list)
    is_deceased: bool = Field(False)

    def apply_timestep(self, progression_profile: dict) -> list[str]:
        """
        Apply one timestep of vital decay. Returns a list of deterioration event strings.
        """
        events = []
        if self.is_deceased:
            return events

        pt = progression_profile.get("per_timestep", {})
        self.timesteps_waited += 1

        # Apply deltas
        vitals = dict(self.current_vitals)

        # o2_sat
        o2 = vitals.get("o2_sat", 100)
        o2 = max(0, round(o2 + pt.get("o2_sat_delta", 0)))
        vitals["o2_sat"] = o2

        # hr
        hr = vitals.get("hr", 80)
        hr = max(0, round(hr + pt.get("hr_delta", 0)))
        vitals["hr"] = hr

        # bp_systolic (extract from "120/80" format if needed)
        bp_str = vitals.get("bp", "120/80")
        try:
            bp_sys, bp_dia = [int(x) for x in bp_str.split("/")]
        except Exception:
            bp_sys, bp_dia = 120, 80
        bp_sys = max(0, round(bp_sys + pt.get("bp_systolic_delta", 0)))
        vitals["bp"] = f"{bp_sys}/{bp_dia}"

        # consciousness
        self.consciousness_score = max(
            0.0, round(self.consciousness_score + pt.get("consciousness_delta", 0.0), 2)
        )

        self.current_vitals = vitals

        # Check critical thresholds
        thresholds = progression_profile.get("critical_thresholds", {})
        if thresholds.get("o2_sat") and o2 <= thresholds["o2_sat"]:
            events.append(f"CRITICAL: SpO₂ dropped to {o2}% (threshold {thresholds['o2_sat']}%)")
            self._escalate_esi()
        if thresholds.get("hr") and hr >= thresholds["hr"]:
            events.append(f"CRITICAL: HR reached {hr} bpm (threshold {thresholds['hr']})")
            self._escalate_esi()
        if thresholds.get("bp_systolic") and bp_sys <= thresholds["bp_systolic"]:
            events.append(f"CRITICAL: BP dropped to {bp_sys} mmHg (threshold {thresholds['bp_systolic']})")
            self._escalate_esi()
        if thresholds.get("consciousness") and self.consciousness_score <= thresholds["consciousness"]:
            events.append(f"CRITICAL: Consciousness score {self.consciousness_score:.2f} (threshold {thresholds['consciousness']})")
            self._escalate_esi()

        # Mortality accumulation
        mortality_step = progression_profile.get("mortality_risk_per_step", 0.0)
        self.cumulative_mortality_risk = min(1.0, round(
            self.cumulative_mortality_risk + mortality_step, 3
        ))
        if self.cumulative_mortality_risk >= 1.0:
            self.is_deceased = True
            events.append("PATIENT DECEASED: delayed treatment led to fatal deterioration.")

        if events:
            self.deteriorated = True
            self.deterioration_events.extend(events)

        return events


    def _escalate_esi(self) -> None:
        if self.current_esi > 1:
            self.current_esi -= 1  # lower number = more urgent


# ─────────────────────────────────────────────────────────────────────────────
# XAI EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

class DifferentialDiagnosis(BaseModel):
    diagnosis: str
    probability: float = Field(..., ge=0.0, le=1.0)


class XAIExplanation(BaseModel):
    """
    Structured explainability output shown to the agent alongside each observation.
    Generated by the XAI engine from the patient's xai_metadata.
    """
    primary_diagnosis: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in primary diagnosis.")
    differential_diagnoses: list[DifferentialDiagnosis] = Field(default_factory=list)

    # Top contributing symptoms (sorted by weight)
    top_symptoms: list[dict] = Field(
        default_factory=list,
        description="[{symptom: str, weight: float}] sorted descending by weight.",
    )

    # Abnormal vital flags
    vital_flags: list[str] = Field(default_factory=list)

    # Key clinical reasoning bullets
    key_reasoning_points: list[str] = Field(default_factory=list)

    # Human-readable summary
    summary: str = Field("", description="One-paragraph plain-English explanation of the triage recommendation.")

    def render_text(self) -> str:
        """Render a human-readable explanation string."""
        lines = [
            f"Primary diagnosis: {self.primary_diagnosis} (confidence {self.confidence:.0%})",
            "",
            "Key symptoms driving this assessment:",
        ]
        for sym in self.top_symptoms[:4]:
            lines.append(f"  • {sym['symptom']} (weight {sym['weight']:.2f})")

        if self.vital_flags:
            lines.append("\nAbnormal vitals:")
            for flag in self.vital_flags:
                lines.append(f"  ⚠ {flag}")

        if self.key_reasoning_points:
            lines.append("\nClinical reasoning:")
            for point in self.key_reasoning_points:
                lines.append(f"  → {point}")

        lines.append(f"\nDifferential diagnoses:")
        for dd in sorted(self.differential_diagnoses, key=lambda x: -x.probability)[:4]:
            lines.append(f"  {dd.diagnosis}: {dd.probability:.0%}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE REQUEST (part of TriageAction)
# ─────────────────────────────────────────────────────────────────────────────

class ResourceRequest(BaseModel):
    """
    The agent's explicit resource allocation request for this patient.
    The Resource Manager will check availability and either fulfil or queue.
    """
    icu_bed: bool = Field(False)
    er_bed: bool = Field(False)
    resus_bay: bool = Field(False)
    ventilator: bool = Field(False)
    ct_scanner: bool = Field(False)
    or_room: bool = Field(False)
    cardiac_monitor: bool = Field(False)
    cath_lab: bool = Field(False)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "icu_bed": True,
                "er_bed": False,
                "resus_bay": False,
                "ventilator": False,
                "ct_scanner": True,
                "or_room": False,
                "cardiac_monitor": True,
                "cath_lab": False,
            }
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRIAGE ACTION (expanded)
# ─────────────────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    Expanded action: ESI level + department + resource request + routing decision.
    All fields except esi_level and department are optional for backward compatibility.
    """
    esi_level: int = Field(
        ..., ge=1, le=5,
        description="ESI triage level: 1=Immediate, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent",
    )
    department: str = Field(
        ...,
        description=(
            "Recommended department. One of: Resuscitation, Emergency, Cardiology, "
            "Neurology, Trauma, Pediatrics, Orthopedics, General, "
            "Psychiatry, Obstetrics, Gastroenterology, Pulmonology"
        ),
    )
    reasoning: Optional[str] = Field(
        None,
        description="Clinical reasoning for the triage decision — used to generate XAI output.",
        max_length=800,
    )

    # ── V3 additions ──────────────────────────────────────────────────────────
    resource_request: Optional[ResourceRequest] = Field(
        None,
        description="Explicit resource allocation request. If omitted, the system infers from ESI + department.",
    )
    routing_decision: Optional[str] = Field(
        None,
        description=(
            "How to handle this patient given current capacity: "
            "'admit' (allocate resources now), "
            "'wait' (queue until resources free), "
            "'reroute' (transfer to another facility). "
            "If omitted, the system defaults to 'admit'."
        ),
    )
    priority_rank: Optional[int] = Field(
        None, ge=1,
        description=(
            "When multiple patients are queued simultaneously, the agent may "
            "optionally rank them (1 = highest priority). Not graded individually "
            "but contributes to wait-time rewards."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "esi_level": 2,
                "department": "Emergency",
                "reasoning": "Chest pain with diaphoresis — suspected ACS. Needs ECG and troponin immediately.",
                "resource_request": {
                    "icu_bed": False,
                    "er_bed": True,
                    "cardiac_monitor": True,
                    "ct_scanner": False,
                },
                "routing_decision": "admit",
                "priority_rank": 1,
            }
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRIAGE OBSERVATION (expanded)
# ─────────────────────────────────────────────────────────────────────────────

class PatientVitals(BaseModel):
    bp: str
    hr: int
    o2_sat: int
    rr: int
    temp: float


class TriageObservation(BaseModel):
    """
    Expanded observation: patient presentation + live vitals + hospital state + XAI.
    """
    # Episode metadata
    episode_id: str
    task_id: str
    difficulty: str = Field(..., description="easy / medium / hard")
    step: int
    max_steps: int
    done: bool
    message: str

    # Patient information
    patient_id: str
    presentation: str = Field(..., description="Full patient presentation text.")

    # Live vitals (may have changed from initial due to progression)
    vitals: dict = Field(..., description="Current vitals after any progression.")
    initial_vitals: dict = Field(default_factory=dict, description="Vitals at presentation (for comparison).")
    timesteps_waited: int = Field(0, description="Steps the patient has waited without treatment.")
    consciousness_score: float = Field(1.0, description="1.0=alert, 0.0=unresponsive.")
    current_esi: int = Field(..., description="Current ESI (may have escalated due to deterioration).")
    has_deteriorated: bool = Field(False, description="True if patient has crossed a critical threshold.")
    deterioration_events: list[str] = Field(default_factory=list)

    # ESI reference
    esi_scale: dict = Field(
        default_factory=lambda: {
            "1": "Immediate — life-threatening, requires instant intervention",
            "2": "Emergent — high risk, should not wait",
            "3": "Urgent — stable but needs multiple resources",
            "4": "Less Urgent — needs one resource only",
            "5": "Non-Urgent — no resources needed",
        }
    )
    valid_departments: list[str] = Field(default_factory=list)

    # ── V3 additions ──────────────────────────────────────────────────────────
    hospital_state: Optional[HospitalState] = Field(
        None,
        description="Current hospital resource availability snapshot.",
    )
    xai_explanation: Optional[XAIExplanation] = Field(
        None,
        description="Structured explainability output for this patient.",
    )
    resources_required: Optional[dict] = Field(
        None,
        description="Resources this patient is known to need (from template/case database).",
    )
    resource_gaps: list[str] = Field(
        default_factory=list,
        description="Resources the patient needs that are currently unavailable.",
    )

    # Feedback from previous step
    last_esi_score: Optional[float] = Field(None)
    last_dept_score: Optional[float] = Field(None)
    last_resource_score: Optional[float] = Field(None)
    last_total_score: Optional[float] = Field(None)
    best_score_so_far: float = Field(0.0)
    feedback: list[str] = Field(default_factory=list)
    hint: Optional[str] = Field(None, description="Clinical hint shown after first failure.")

    # Pillar 2.2 / 2.3: partial observability and nurse handoff fields
    partial_obs_applied: bool = Field(
        False,
        description="True if partial observability mode is active for this episode.",
    )
    missing_vitals: list[str] = Field(
        default_factory=list,
        description="Vitals marked as not yet measured in partial_obs mode.",
    )
    handoff_mode: bool = Field(
        False,
        description="True when observation is delivered as a nurse handoff (step 2+).",
    )
    nurse_summary: Optional[str] = Field(
        None,
        description="Nurse handoff summary replacing direct patient presentation on step 2+.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRIAGE REWARD (expanded — multi-objective)
# ─────────────────────────────────────────────────────────────────────────────

class TriageReward(BaseModel):
    """
    Multi-objective reward with full decomposition.

    Formula:
        total = w_accuracy  * accuracy_score
              + w_resource  * resource_score
              - delay_penalty
              - mortality_penalty
              - overtriage_penalty
              - undertriage_penalty

    Default weights: accuracy=0.45, resource=0.25, delay=0.15, mortality=0.15
    """
    value: float = Field(..., ge=0.0, le=1.0, description="Overall reward (0.0–1.0).")

    # Component scores
    esi_score: float = Field(..., description="ESI level accuracy (0–1).")
    department_score: float = Field(..., description="Department correctness (0–1).")
    resource_score: float = Field(1.0, description="Resource allocation efficiency (0–1).")
    delay_penalty: float = Field(0.0, description="Penalty for excessive patient wait time.")
    mortality_penalty: float = Field(0.0, description="Penalty for patient deterioration / death.")
    overtriage_penalty: float = Field(0.0, description="Penalty for sending non-critical patients to ICU/Resus.")
    undertriage_penalty: float = Field(0.0, description="Penalty for sending critical patients to low-acuity areas.")

    breakdown: dict = Field(default_factory=dict, description="Full weighted breakdown of all components.")

    # Pillar 3.2: Confidence-calibration bonus
    calibration_bonus: float = Field(
        0.0,
        description=(
            "Bonus/penalty for reasoning calibration. "
            "Correct decision + confident reasoning: +0.02. "
            "Wrong decision + overconfident reasoning: -0.05. "
            "Computed from action.reasoning length and keyword confidence signals."
        ),
    )

    # Pillar 5.2: Safety tier score — ESI-1/2 handling only
    safety_score: float = Field(
        1.0,
        ge=0.0, le=1.0,
        description=(
            "Safety tier score (0-1) measuring ESI-1/2 critical case handling. "
            "Independent of overall accuracy — an agent can score 0.85 overall "
            "but 0.2 on safety_score if it consistently undertriages critical patients. "
            "1.0 = no critical undertriage. 0.0 = dangerous undertriage of ESI-1/2 patient."
        ),
    )

    # Pillar 3.3: Episode-level outcome label
    final_outcome: Optional[str] = Field(
        None,
        description=(
            "Episode outcome label: OPTIMAL (correct step 1), ACCEPTABLE (correct within 3 steps), "
            "DELAYED (correct but 4-6 steps), HARMFUL (undertriage), FATAL (patient died). "
            "Set when episode ends (done=True)."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": 0.74,
                "esi_score": 0.99,
                "department_score": 0.99,
                "resource_score": 0.80,
                "delay_penalty": 0.05,
                "mortality_penalty": 0.0,
                "overtriage_penalty": 0.0,
                "undertriage_penalty": 0.0,
                "breakdown": {
                    "accuracy_component": 0.44,
                    "resource_component": 0.20,
                    "delay_penalty": -0.05,
                    "mortality_penalty": -0.0,
                    "overtriage_penalty": -0.0,
                    "undertriage_penalty": -0.0,
                },
            }
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# API REQUEST / RESPONSE MODELS (unchanged from V2 for API compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class TaskConfig(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    action_schema: dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="easy / medium / hard / mass_casualty / random.")
    num_patients: int = Field(1, ge=1, le=5, description="Number of patients (1-5).")
    use_procedural: bool = Field(False, description="If True, use procedural generation instead of static bank.")
    hospital_config: Optional[dict] = Field(None, description="Override default hospital resource counts.")

    # Pillar 4.4: seed-based deterministic evaluation
    seed: Optional[int] = Field(
        None,
        description=(
            "Optional random seed for fully deterministic episode. "
            "When set, patient selection and procedural generation use random.seed(seed). "
            "Eliminates score variance - essential for fair model comparison."
        ),
    )

    # Pillar 2.2: partial observability mode
    partial_obs: bool = Field(
        False,
        description=(
            "If True, enables partial observability: 1-2 vitals randomly marked "
            "not yet measured, presentation may omit one symptom. "
            "Trains agents to reason under real ED uncertainty."
        ),
    )


class StepRequest(BaseModel):
    action: TriageAction


class StepResponse(BaseModel):
    observation: TriageObservation
    reward: TriageReward
    done: bool
    info: dict


class ResetResponse(BaseModel):
    observation: TriageObservation


class StateResponse(BaseModel):
    state: dict


class GraderRequest(BaseModel):
    patient_id: str
    esi_level: int = Field(..., ge=1, le=5)
    department: str
    resource_request: Optional[dict] = None
    routing_decision: Optional[str] = None


class GraderResponse(BaseModel):
    patient_id: str
    esi_score: float
    department_score: float
    resource_score: float
    total_score: float
    feedback: list[str]
    correct_esi: int
    correct_department: str
    xai_explanation: Optional[XAIExplanation] = None
