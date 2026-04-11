"""
Medical Triage OpenEnv — Expanded Patient Database (V3)
========================================================
Changes from V2
---------------
1. CONDITION TEMPLATES (procedural generation)
   Each entry now has a `template` block from which an infinite supply of
   statistically-realistic patients can be sampled at runtime.

2. PROGRESSION PROFILES
   Each patient carries a `progression` block that drives per-timestep vital
   decay when the patient is delayed or mis-triaged.

3. RESOURCE REQUIREMENTS
   Each patient declares which hospital resources they need so the Resource
   Manager can check availability before allocating.

4. XAI METADATA
   Every case ships with ranked symptom weights, likely differential
   diagnoses, and a confidence model so the XAI engine can produce
   transparent decision explanations.

ESI Levels:
  1 – Immediate       : Life-threatening, requires instant intervention
  2 – Emergent        : High risk / severe distress, should not wait
  3 – Urgent          : Stable but needs multiple resources
  4 – Less Urgent     : One resource only
  5 – Non-Urgent      : No resources needed

Departments:
  Resuscitation, Emergency, Cardiology, Neurology, Trauma, Pediatrics,
  Orthopedics, General, Psychiatry, Obstetrics, Gastroenterology,
  Pulmonology
"""

from __future__ import annotations
import random
import copy
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate a patient instance from a condition template
# ─────────────────────────────────────────────────────────────────────────────

def _rand(lo: float, hi: float, decimals: int = 0) -> float:
    val = random.uniform(lo, hi)
    return round(val, decimals) if decimals else int(round(val))


def generate_patient_from_template(template: dict, instance_suffix: str = "") -> dict:
    """
    Randomly instantiate a patient from a condition template.
    Returns a fully-formed patient dict compatible with the environment.
    """
    t = template
    tid = t["condition_id"]
    suffix = instance_suffix or f"_{random.randint(1000, 9999)}"
    patient_id = f"proc_{tid}{suffix}"

    age = _rand(*t["age_range"])
    hr = _rand(*t["vitals_ranges"]["hr"])
    bp_sys = _rand(*t["vitals_ranges"]["bp_systolic"])
    bp_dia = _rand(*t["vitals_ranges"]["bp_diastolic"])
    o2 = _rand(*t["vitals_ranges"]["o2_sat"])
    rr = _rand(*t["vitals_ranges"]["rr"])
    temp = _rand(*t["vitals_ranges"]["temp"], decimals=1)

    # Probabilistic symptoms
    symptoms = [s for s, p in t["symptom_probabilities"].items() if random.random() < p]
    if not symptoms:
        symptoms = list(t["symptom_probabilities"].keys())[:2]  # fallback — at least 2

    sex = random.choice(["male", "female"])
    age_str = f"{age}-year-old {sex}"
    sym_str = ", ".join(symptoms)

    presentation = (
        f"{age_str}. {t['condition_name']}. Presenting with: {sym_str}. "
        f"BP {bp_sys}/{bp_dia}, HR {hr}, O2 sat {o2}%, RR {rr}, Temp {temp}°C."
    )

    return {
        "id": patient_id,
        "difficulty": t["difficulty"],
        "is_procedural": True,
        "source_template": tid,
        "presentation": presentation,
        "vitals": {
            "bp": f"{bp_sys}/{bp_dia}",
            "hr": hr,
            "o2_sat": o2,
            "rr": rr,
            "temp": temp,
        },
        "correct_esi": t["correct_esi"],
        "correct_department": t["correct_department"],
        "reasoning": t["reasoning_template"].format(
            age=age, sex=sex, hr=hr, o2=o2, bp_sys=bp_sys
        ),
        "esi_partial_credit": t["esi_partial_credit"],
        "department_options": t["department_options"],
        "department_scores": t["department_scores"],

        # ── V3 additions ──────────────────────────────────────────────────
        "progression": copy.deepcopy(t["progression"]),
        "resources_required": copy.deepcopy(t["resources_required"]),
        "xai_metadata": _build_xai_metadata(t, hr, o2, bp_sys, symptoms),
    }


def _build_xai_metadata(
    t: dict, hr: int, o2: int, bp_sys: int, symptoms: list[str]
) -> dict:
    """Build per-instance XAI metadata from the template's base weights."""
    base = t.get("xai_base", {})
    symptom_weights: dict[str, float] = {}

    for sym in symptoms:
        weight = base.get("symptom_weights", {}).get(sym, 0.3)
        # Add slight instance noise
        symptom_weights[sym] = round(min(1.0, weight + random.uniform(-0.05, 0.05)), 2)

    # Vital abnormality flags
    vital_flags = []
    if hr > 100:
        vital_flags.append(f"HR {hr} (tachycardia)")
    if hr < 60:
        vital_flags.append(f"HR {hr} (bradycardia)")
    if o2 < 94:
        vital_flags.append(f"SpO₂ {o2}% (hypoxia)")
    if bp_sys > 160:
        vital_flags.append(f"BP {bp_sys} (hypertension)")
    if bp_sys < 90:
        vital_flags.append(f"BP {bp_sys} (hypotension)")

    return {
        "symptom_weights": symptom_weights,
        "vital_flags": vital_flags,
        "differential_diagnoses": base.get("differentials", []),
        "primary_diagnosis": base.get("primary_diagnosis", t["condition_name"]),
        "confidence": base.get("base_confidence", 0.85),
        "key_reasoning_points": base.get("key_reasoning_points", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION TEMPLATES
# Each template is the generative schema for a medical condition.
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_TEMPLATES: list[dict] = [

    # ── Template 1: STEMI / Acute Coronary Syndrome ──────────────────────────
    {
        "condition_id": "stemi",
        "condition_name": "Acute coronary syndrome — suspected STEMI",
        "difficulty": "easy",
        "age_range": (45, 80),
        "vitals_ranges": {
            "hr": (95, 120),
            "bp_systolic": (160, 210),
            "bp_diastolic": (90, 120),
            "o2_sat": (90, 96),
            "rr": (18, 24),
            "temp": (36.4, 37.2),
        },
        "symptom_probabilities": {
            "crushing chest pain": 0.95,
            "radiation to left arm": 0.70,
            "radiation to jaw": 0.50,
            "diaphoresis": 0.80,
            "nausea": 0.65,
            "pallor": 0.70,
            "shortness of breath": 0.60,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "Classic ACS presentation in a {age}-year-old {sex}. "
            "Chest pain with diaphoresis, HR {hr}, O2 {o2}% — rule out STEMI immediately. ESI-2."
        ),
        "esi_partial_credit": {1: 0.7, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
        "department_options": ["Emergency", "Cardiology", "Resuscitation"],
        "department_scores": {"Emergency": 0.99, "Cardiology": 0.8, "Resuscitation": 0.9},

        # ── Progression: vitals deteriorate on delay ──────────────────────
        "progression": {
            "description": "STEMI — every 30-min delay worsens outcome. Risk of cardiogenic shock.",
            "per_timestep": {
                "o2_sat_delta": -1,        # drops 1% per timestep
                "hr_delta": +3,            # tachycardia worsens
                "bp_systolic_delta": -4,   # cardiogenic shock progression
                "consciousness_delta": 0,
            },
            "mortality_risk_per_step": 0.08,   # 8% mortality increase per missed step
            "critical_thresholds": {
                "o2_sat": 88,              # triggers ESI upgrade to 1
                "bp_systolic": 80,         # cardiogenic shock
                "hr": 140,
            },
        },

        # ── Resource requirements ─────────────────────────────────────────
        "resources_required": {
            "er_bed": True,
            "icu_bed": False,          # may escalate post-PCI
            "cardiac_monitor": True,
            "ecg_machine": True,
            "cath_lab": True,          # for primary PCI
            "ventilator": False,
            "ct_scanner": False,
            "or_room": False,
            "mri_scanner": False,
            "nurse_ratio": 1,          # 1 dedicated nurse
            "doctor_specialist": "Cardiology",
        },

        # ── XAI metadata ─────────────────────────────────────────────────
        "xai_base": {
            "primary_diagnosis": "STEMI",
            "differentials": [
                {"diagnosis": "STEMI", "probability": 0.72},
                {"diagnosis": "NSTEMI / ACS", "probability": 0.18},
                {"diagnosis": "Aortic dissection", "probability": 0.06},
                {"diagnosis": "PE", "probability": 0.04},
            ],
            "symptom_weights": {
                "crushing chest pain": 0.90,
                "diaphoresis": 0.80,
                "radiation to left arm": 0.75,
                "radiation to jaw": 0.70,
                "pallor": 0.55,
            },
            "vital_emphasis": ["hr", "bp_systolic", "o2_sat"],
            "key_reasoning_points": [
                "Diaphoresis + chest pain = immediate cardiac rule-out",
                "Elevated BP suggests active ischemia, not yet shock",
                "ECG within 10 minutes is the priority intervention",
            ],
            "base_confidence": 0.88,
        },
    },

    # ── Template 2: Pulmonary Embolism ────────────────────────────────────────
    {
        "condition_id": "pe",
        "condition_name": "Pulmonary embolism — suspected",
        "difficulty": "hard",
        "age_range": (30, 80),
        "vitals_ranges": {
            "hr": (100, 150),
            "bp_systolic": (90, 130),
            "bp_diastolic": (60, 85),
            "o2_sat": (75, 93),
            "rr": (20, 30),
            "temp": (36.5, 37.5),
        },
        "symptom_probabilities": {
            "dyspnea": 0.95,
            "chest pain (pleuritic)": 0.65,
            "calf swelling / DVT": 0.50,
            "syncope": 0.20,
            "haemoptysis": 0.15,
            "recent immobility or long-haul flight": 0.60,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "SpO₂ {o2}%, HR {hr} with dyspnea — Wells score likely >4. "
            "Urgent CT-PA needed in this {age}-year-old {sex}. ESI-2."
        ),
        "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
        "department_options": ["Emergency", "Pulmonology", "General"],
        "department_scores": {"Emergency": 0.99, "Pulmonology": 0.7, "General": 0.1},

        "progression": {
            "description": "PE — hypoxia and haemodynamic collapse can be rapid.",
            "per_timestep": {
                "o2_sat_delta": -2,
                "hr_delta": +4,
                "bp_systolic_delta": -3,
                "consciousness_delta": -0.1,
            },
            "mortality_risk_per_step": 0.10,
            "critical_thresholds": {
                "o2_sat": 85,
                "bp_systolic": 85,
                "hr": 150,
            },
        },

        "resources_required": {
            "er_bed": True,
            "icu_bed": False,
            "cardiac_monitor": True,
            "ecg_machine": True,
            "cath_lab": False,
            "ventilator": False,
            "ct_scanner": True,     # CT-PA mandatory
            "or_room": False,
            "mri_scanner": False,
            "nurse_ratio": 1,
            "doctor_specialist": "Pulmonology",
        },

        "xai_base": {
            "primary_diagnosis": "Pulmonary embolism",
            "differentials": [
                {"diagnosis": "Pulmonary embolism", "probability": 0.65},
                {"diagnosis": "Pneumonia", "probability": 0.15},
                {"diagnosis": "Pneumothorax", "probability": 0.10},
                {"diagnosis": "STEMI (right heart)", "probability": 0.10},
            ],
            "symptom_weights": {
                "dyspnea": 0.85,
                "calf swelling / DVT": 0.80,
                "chest pain (pleuritic)": 0.70,
                "recent immobility or long-haul flight": 0.75,
                "syncope": 0.60,
            },
            "vital_emphasis": ["o2_sat", "hr", "rr"],
            "key_reasoning_points": [
                "Wells score: DVT symptoms + immobility + tachycardia = high probability",
                "SpO₂ below 94% at rest is the clinical alarm trigger",
                "CT-PA must not be delayed for lab confirmation",
            ],
            "base_confidence": 0.78,
        },
    },

    # ── Template 3: Sepsis ────────────────────────────────────────────────────
    {
        "condition_id": "sepsis",
        "condition_name": "Sepsis — suspected",
        "difficulty": "medium",
        "age_range": (18, 90),
        "vitals_ranges": {
            "hr": (100, 140),
            "bp_systolic": (80, 110),
            "bp_diastolic": (45, 70),
            "o2_sat": (90, 97),
            "rr": (20, 28),
            "temp": (38.0, 40.5),
        },
        "symptom_probabilities": {
            "fever": 0.85,
            "chills / rigors": 0.60,
            "altered mental status": 0.50,
            "productive cough": 0.40,
            "dysuria (UTI source)": 0.35,
            "abdominal pain": 0.30,
            "skin flushing": 0.55,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "Fever {temp}°C, HR {hr}, BP {bp_sys} — meets ≥2 SIRS criteria. "
            "Sepsis bundle within 1 hour in this {age}-year-old {sex}. ESI-2."
        ),
        "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.4, 4: 0.01, 5: 0.01},
        "department_options": ["Emergency", "General", "Resuscitation"],
        "department_scores": {"Emergency": 0.99, "General": 0.4, "Resuscitation": 0.7},

        "progression": {
            "description": "Sepsis → septic shock. Each hour of delay significantly raises mortality.",
            "per_timestep": {
                "o2_sat_delta": -1,
                "hr_delta": +5,
                "bp_systolic_delta": -5,
                "consciousness_delta": -0.15,
            },
            "mortality_risk_per_step": 0.07,
            "critical_thresholds": {
                "bp_systolic": 80,
                "hr": 140,
                "o2_sat": 90,
            },
        },

        "resources_required": {
            "er_bed": True,
            "icu_bed": True,           # often escalates to ICU
            "cardiac_monitor": True,
            "ecg_machine": False,
            "cath_lab": False,
            "ventilator": False,       # may be needed later
            "ct_scanner": False,
            "or_room": False,
            "mri_scanner": False,
            "nurse_ratio": 1,
            "doctor_specialist": "Emergency",
        },

        "xai_base": {
            "primary_diagnosis": "Sepsis",
            "differentials": [
                {"diagnosis": "Sepsis", "probability": 0.70},
                {"diagnosis": "Severe UTI (non-septic)", "probability": 0.15},
                {"diagnosis": "Pneumonia without sepsis", "probability": 0.10},
                {"diagnosis": "Adrenal crisis", "probability": 0.05},
            ],
            "symptom_weights": {
                "fever": 0.75,
                "altered mental status": 0.80,
                "chills / rigors": 0.65,
                "skin flushing": 0.55,
            },
            "vital_emphasis": ["hr", "bp_systolic", "temp"],
            "key_reasoning_points": [
                "SIRS criteria met: tachycardia + temperature abnormality",
                "Suspected infection source required to diagnose sepsis",
                "Hour-1 sepsis bundle: cultures, IV fluids, antibiotics",
            ],
            "base_confidence": 0.82,
        },
    },

    # ── Template 4: Aortic Dissection ─────────────────────────────────────────
    {
        "condition_id": "aortic_dissection",
        "condition_name": "Aortic dissection — suspected",
        "difficulty": "hard",
        "age_range": (40, 75),
        "vitals_ranges": {
            "hr": (80, 110),
            "bp_systolic": (160, 220),
            "bp_diastolic": (90, 130),
            "o2_sat": (93, 98),
            "rr": (16, 22),
            "temp": (36.5, 37.2),
        },
        "symptom_probabilities": {
            "severe tearing back or chest pain": 0.90,
            "BP differential between arms >20 mmHg": 0.65,
            "syncope": 0.20,
            "focal neurological deficit": 0.15,
            "pulse deficit": 0.30,
        },
        "correct_esi": 1,
        "correct_department": "Resuscitation",
        "reasoning_template": (
            "BP differential + tearing pain in a hypertensive {age}-year-old {sex}. "
            "Aortic dissection until proven otherwise. ESI-1, CT-aortogram stat."
        ),
        "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.01, 4: 0.01, 5: 0.01},
        "department_options": ["Resuscitation", "Emergency", "Cardiology"],
        "department_scores": {"Resuscitation": 0.99, "Emergency": 0.6, "Cardiology": 0.4},

        "progression": {
            "description": "Dissection propagation is rapid; aortic rupture is fatal.",
            "per_timestep": {
                "o2_sat_delta": 0,
                "hr_delta": +2,
                "bp_systolic_delta": -6,
                "consciousness_delta": -0.2,
            },
            "mortality_risk_per_step": 0.12,
            "critical_thresholds": {
                "bp_systolic": 75,
                "consciousness": 0.4,
            },
        },

        "resources_required": {
            "er_bed": False,
            "icu_bed": True,
            "cardiac_monitor": True,
            "ecg_machine": True,
            "cath_lab": False,
            "ventilator": False,
            "ct_scanner": True,
            "or_room": True,           # cardiothoracic surgery standby
            "mri_scanner": False,
            "nurse_ratio": 2,          # high-dependency
            "doctor_specialist": "Cardiothoracic surgery",
        },

        "xai_base": {
            "primary_diagnosis": "Aortic dissection (Type A)",
            "differentials": [
                {"diagnosis": "Aortic dissection", "probability": 0.75},
                {"diagnosis": "STEMI", "probability": 0.12},
                {"diagnosis": "PE", "probability": 0.08},
                {"diagnosis": "Musculoskeletal pain", "probability": 0.05},
            ],
            "symptom_weights": {
                "severe tearing back or chest pain": 0.90,
                "BP differential between arms >20 mmHg": 0.95,
                "pulse deficit": 0.85,
                "syncope": 0.60,
            },
            "vital_emphasis": ["bp_systolic"],
            "key_reasoning_points": [
                "BP arm differential is the single most specific sign",
                "Tearing quality of pain distinguishes from ACS",
                "Do NOT give anticoagulation before dissection excluded",
            ],
            "base_confidence": 0.80,
        },
    },

    # ── Template 5: Preeclampsia / Eclampsia ──────────────────────────────────
    {
        "condition_id": "preeclampsia",
        "condition_name": "Severe preeclampsia",
        "difficulty": "medium",
        "age_range": (16, 40),
        "vitals_ranges": {
            "hr": (80, 105),
            "bp_systolic": (150, 175),
            "bp_diastolic": (98, 115),
            "o2_sat": (95, 99),
            "rr": (16, 22),
            "temp": (36.6, 37.4),
        },
        "symptom_probabilities": {
            "severe headache": 0.85,
            "visual disturbances": 0.70,
            "facial / hand oedema": 0.75,
            "epigastric pain": 0.40,
            "nausea / vomiting": 0.45,
            "proteinuria": 0.90,
        },
        "correct_esi": 2,
        "correct_department": "Obstetrics",
        "reasoning_template": (
            "BP {bp_sys}/high + headache + visual changes in a {age}-year-old pregnant {sex}. "
            "Severe preeclampsia — seizure risk. Magnesium + antihypertensives now. ESI-2."
        ),
        "esi_partial_credit": {1: 0.6, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
        "department_options": ["Obstetrics", "Emergency", "General"],
        "department_scores": {"Obstetrics": 0.99, "Emergency": 0.7, "General": 0.1},

        "progression": {
            "description": "Preeclampsia can progress to eclamptic seizure without warning.",
            "per_timestep": {
                "o2_sat_delta": 0,
                "hr_delta": +2,
                "bp_systolic_delta": +3,
                "consciousness_delta": -0.1,
            },
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {
                "bp_systolic": 180,
                "consciousness": 0.5,
            },
        },

        "resources_required": {
            "er_bed": False,
            "icu_bed": False,
            "cardiac_monitor": True,
            "ecg_machine": False,
            "cath_lab": False,
            "ventilator": False,
            "ct_scanner": False,
            "or_room": True,           # C-section may be needed
            "mri_scanner": False,
            "nurse_ratio": 1,
            "doctor_specialist": "Obstetrics",
        },

        "xai_base": {
            "primary_diagnosis": "Severe preeclampsia",
            "differentials": [
                {"diagnosis": "Severe preeclampsia", "probability": 0.80},
                {"diagnosis": "Chronic hypertension with superimposed PE", "probability": 0.12},
                {"diagnosis": "HELLP syndrome", "probability": 0.08},
            ],
            "symptom_weights": {
                "severe headache": 0.80,
                "visual disturbances": 0.85,
                "facial / hand oedema": 0.70,
                "proteinuria": 0.75,
            },
            "vital_emphasis": ["bp_systolic", "bp_diastolic"],
            "key_reasoning_points": [
                "Severe BP criteria: systolic ≥160 or diastolic ≥110",
                "Visual changes indicate cerebral involvement",
                "Magnesium sulphate for seizure prophylaxis is first priority",
            ],
            "base_confidence": 0.87,
        },
    },

    # ── Template 6: Renal Colic ───────────────────────────────────────────────
    {
        "condition_id": "renal_colic",
        "condition_name": "Renal colic — ureteric stone",
        "difficulty": "medium",
        "age_range": (20, 55),
        "vitals_ranges": {
            "hr": (95, 115),
            "bp_systolic": (130, 155),
            "bp_diastolic": (80, 100),
            "o2_sat": (97, 99),
            "rr": (16, 22),
            "temp": (36.5, 37.4),
        },
        "symptom_probabilities": {
            "flank pain radiating to groin": 0.95,
            "colicky pain pattern": 0.85,
            "nausea / vomiting": 0.70,
            "haematuria": 0.80,
            "restlessness": 0.65,
        },
        "correct_esi": 3,
        "correct_department": "Emergency",
        "reasoning_template": (
            "Classic renal colic in a {age}-year-old {sex}. "
            "Afebrile — no infected stone. Needs IV analgesia + CT KUB. ESI-3."
        ),
        "esi_partial_credit": {1: 0.1, 2: 0.6, 3: 0.99, 4: 0.2, 5: 0.01},
        "department_options": ["Emergency", "General", "Gastroenterology"],
        "department_scores": {"Emergency": 0.99, "General": 0.5, "Gastroenterology": 0.3},

        "progression": {
            "description": "Afebrile renal colic is painful but rarely immediately life-threatening.",
            "per_timestep": {
                "o2_sat_delta": 0,
                "hr_delta": +1,
                "bp_systolic_delta": +2,
                "consciousness_delta": 0,
            },
            "mortality_risk_per_step": 0.005,
            "critical_thresholds": {
                "temp": 38.5,           # fever = infected stone → ESI-2 upgrade
            },
        },

        "resources_required": {
            "er_bed": True,
            "icu_bed": False,
            "cardiac_monitor": False,
            "ecg_machine": False,
            "cath_lab": False,
            "ventilator": False,
            "ct_scanner": True,         # CT KUB
            "or_room": False,
            "mri_scanner": False,
            "nurse_ratio": 0,
            "doctor_specialist": None,
        },

        "xai_base": {
            "primary_diagnosis": "Ureteric colic",
            "differentials": [
                {"diagnosis": "Ureteric stone", "probability": 0.80},
                {"diagnosis": "Appendicitis", "probability": 0.10},
                {"diagnosis": "Ovarian pathology", "probability": 0.07},
                {"diagnosis": "AAA (in older patients)", "probability": 0.03},
            ],
            "symptom_weights": {
                "flank pain radiating to groin": 0.90,
                "colicky pain pattern": 0.80,
                "haematuria": 0.75,
                "restlessness": 0.50,
            },
            "vital_emphasis": ["temp", "hr"],
            "key_reasoning_points": [
                "Colicky unilateral flank-to-groin pattern is highly specific",
                "Fever would indicate infected hydronephrosis — urgent decompression",
                "CT KUB preferred over USS for stone detection",
            ],
            "base_confidence": 0.88,
        },
    },

    # ── Template 7: Serotonin Syndrome / Stimulant Toxidrome ─────────────────
    {
        "condition_id": "serotonin_syndrome",
        "condition_name": "Serotonin syndrome / stimulant toxidrome",
        "difficulty": "hard",
        "age_range": (18, 45),
        "vitals_ranges": {
            "hr": (120, 160),
            "bp_systolic": (155, 195),
            "bp_diastolic": (90, 120),
            "o2_sat": (94, 97),
            "rr": (20, 26),
            "temp": (38.5, 41.0),
        },
        "symptom_probabilities": {
            "hyperthermia": 0.90,
            "muscle rigidity / clonus": 0.80,
            "agitation / altered consciousness": 0.85,
            "dilated pupils": 0.70,
            "diaphoresis": 0.75,
            "possible substance ingestion": 0.80,
        },
        "correct_esi": 1,
        "correct_department": "Resuscitation",
        "reasoning_template": (
            "Temp {temp}°C, HR {hr}, rigidity + altered consciousness — "
            "serotonin syndrome in this {age}-year-old {sex}. Risk of rhabdo + arrest. ESI-1."
        ),
        "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.1, 4: 0.01, 5: 0.01},
        "department_options": ["Resuscitation", "Emergency", "Psychiatry"],
        "department_scores": {"Resuscitation": 0.99, "Emergency": 0.6, "Psychiatry": 0.01},

        "progression": {
            "description": "Hyperthermia + rigidity → rhabdomyolysis → renal failure → cardiac arrest.",
            "per_timestep": {
                "o2_sat_delta": -1,
                "hr_delta": +5,
                "bp_systolic_delta": +4,
                "consciousness_delta": -0.25,
            },
            "mortality_risk_per_step": 0.15,
            "critical_thresholds": {
                "temp": 41.0,
                "hr": 165,
                "consciousness": 0.3,
            },
        },

        "resources_required": {
            "er_bed": False,
            "icu_bed": True,
            "cardiac_monitor": True,
            "ecg_machine": True,
            "cath_lab": False,
            "ventilator": True,
            "ct_scanner": False,
            "or_room": False,
            "mri_scanner": False,
            "nurse_ratio": 2,
            "doctor_specialist": "Toxicology",
        },

        "xai_base": {
            "primary_diagnosis": "Serotonin syndrome",
            "differentials": [
                {"diagnosis": "Serotonin syndrome", "probability": 0.55},
                {"diagnosis": "Stimulant toxidrome (cocaine / MDMA)", "probability": 0.25},
                {"diagnosis": "Neuroleptic malignant syndrome", "probability": 0.15},
                {"diagnosis": "Heat stroke", "probability": 0.05},
            ],
            "symptom_weights": {
                "hyperthermia": 0.85,
                "muscle rigidity / clonus": 0.90,
                "agitation / altered consciousness": 0.80,
                "dilated pupils": 0.65,
            },
            "vital_emphasis": ["temp", "hr", "o2_sat"],
            "key_reasoning_points": [
                "Hunter criteria: clonus + agitation + hyperthermia = serotonin syndrome",
                "Immediate cooling + benzodiazepines before any other intervention",
                "NMS distinguishable by lead-pipe rigidity and slower onset",
            ],
            "base_confidence": 0.72,
        },
    },

    # ── Template 8: Minor Laceration (ESI-5) ─────────────────────────────────
    {
        "condition_id": "minor_laceration",
        "condition_name": "Minor laceration — no neurovascular compromise",
        "difficulty": "easy",
        "age_range": (5, 80),
        "vitals_ranges": {
            "hr": (65, 90),
            "bp_systolic": (110, 135),
            "bp_diastolic": (65, 85),
            "o2_sat": (98, 100),
            "rr": (14, 18),
            "temp": (36.4, 37.1),
        },
        "symptom_probabilities": {
            "superficial skin laceration": 1.0,
            "bleeding controlled with pressure": 0.90,
            "no numbness": 0.95,
            "full range of motion": 0.90,
            "tetanus up to date": 0.70,
        },
        "correct_esi": 5,
        "correct_department": "General",
        "reasoning_template": (
            "Minor wound in a {age}-year-old {sex}, bleeding controlled, neurovascularly intact. "
            "ESI-5 — no emergency resources needed."
        ),
        "esi_partial_credit": {1: 0.01, 2: 0.01, 3: 0.1, 4: 0.6, 5: 0.99},
        "department_options": ["General", "Emergency", "Orthopedics"],
        "department_scores": {"General": 0.99, "Emergency": 0.4, "Orthopedics": 0.3},

        "progression": {
            "description": "Minor laceration — clinical condition essentially static.",
            "per_timestep": {
                "o2_sat_delta": 0,
                "hr_delta": 0,
                "bp_systolic_delta": 0,
                "consciousness_delta": 0,
            },
            "mortality_risk_per_step": 0.0,
            "critical_thresholds": {},
        },

        "resources_required": {
            "er_bed": False,
            "icu_bed": False,
            "cardiac_monitor": False,
            "ecg_machine": False,
            "cath_lab": False,
            "ventilator": False,
            "ct_scanner": False,
            "or_room": False,
            "mri_scanner": False,
            "nurse_ratio": 0,
            "doctor_specialist": None,
        },

        "xai_base": {
            "primary_diagnosis": "Superficial laceration",
            "differentials": [
                {"diagnosis": "Superficial laceration", "probability": 0.97},
                {"diagnosis": "Deep laceration requiring exploration", "probability": 0.03},
            ],
            "symptom_weights": {
                "superficial skin laceration": 0.95,
                "bleeding controlled with pressure": 0.70,
            },
            "vital_emphasis": [],
            "key_reasoning_points": [
                "Neurovascular exam normal — no tendon or vessel involvement",
                "Tetanus status should always be confirmed",
                "No imaging or IV access required",
            ],
            "base_confidence": 0.97,
        },
    },
]

TEMPLATE_MAP: dict[str, dict] = {t["condition_id"]: t for t in CONDITION_TEMPLATES}


# ─────────────────────────────────────────────────────────────────────────────
# STATIC PATIENT BANK (original 18 cases, now enriched with V3 fields)
# ─────────────────────────────────────────────────────────────────────────────

def _static(base: dict, progression: dict, resources: dict, xai: dict) -> dict:
    """Attach V3 fields to a static patient case."""
    return {
        **base,
        "is_procedural": False,
        "progression": progression,
        "resources_required": resources,
        "xai_metadata": {
            "symptom_weights": xai.get("symptom_weights", {}),
            "vital_flags": xai.get("vital_flags", []),
            "differential_diagnoses": xai.get("differentials", []),
            "primary_diagnosis": xai.get("primary_diagnosis", ""),
            "confidence": xai.get("confidence", 0.85),
            "key_reasoning_points": xai.get("key_reasoning_points", []),
        },
    }


PATIENTS = [

    # ─────────────────────────────────────────────────────────────────────────
    # EASY
    # ─────────────────────────────────────────────────────────────────────────

    _static(
        base={
            "id": "easy_001",
            "difficulty": "easy",
            "presentation": (
                "67-year-old male. Crushing chest pain radiating to left arm and jaw for 30 minutes. "
                "Diaphoretic, pale, nauseous. BP 185/110, HR 102, O2 sat 94%, RR 20. "
                "History of hypertension and type 2 diabetes."
            ),
            "vitals": {"bp": "185/110", "hr": 102, "o2_sat": 94, "rr": 20, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": (
                "Classic STEMI presentation. ESI-2: must be seen immediately, "
                "does not yet require immediate airway intervention."
            ),
            "esi_partial_credit": {1: 0.7, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "Cardiology", "Resuscitation"],
            "department_scores": {"Emergency": 0.99, "Cardiology": 0.8, "Resuscitation": 0.9},
        },
        progression={
            "description": "STEMI — every delayed minute risks larger infarct.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": 3, "bp_systolic_delta": -4, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.08,
            "critical_thresholds": {"o2_sat": 88, "bp_systolic": 80},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": True, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Cardiology",
        },
        xai={
            "primary_diagnosis": "STEMI",
            "differentials": [
                {"diagnosis": "STEMI", "probability": 0.72},
                {"diagnosis": "NSTEMI", "probability": 0.18},
                {"diagnosis": "Aortic dissection", "probability": 0.06},
                {"diagnosis": "PE", "probability": 0.04},
            ],
            "symptom_weights": {
                "crushing chest pain": 0.90, "diaphoresis": 0.80,
                "radiation to left arm": 0.75, "radiation to jaw": 0.70,
            },
            "vital_flags": ["HR 102 (tachycardia)", "SpO₂ 94% (borderline hypoxia)", "BP 185 (hypertension)"],
            "confidence": 0.88,
            "key_reasoning_points": [
                "Diaphoresis + chest pain = immediate cardiac rule-out",
                "ECG within 10 minutes is mandatory",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_002",
            "difficulty": "easy",
            "presentation": (
                "8-year-old girl. Fell off bicycle, obvious deformity of right forearm, "
                "moderate pain (6/10). Skin intact, good distal pulse. "
                "BP 105/65, HR 95, O2 sat 99%, afebrile."
            ),
            "vitals": {"bp": "105/65", "hr": 95, "o2_sat": 99, "rr": 18, "temp": 36.6},
            "correct_esi": 2,
            "correct_department": "Pediatrics",
            "reasoning": "Pediatric fracture with deformity. Needs X-ray, IV sedation. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.6, 4: 0.1, 5: 0.01},
            "department_options": ["Pediatrics", "Orthopedics", "Emergency"],
            "department_scores": {"Pediatrics": 0.99, "Orthopedics": 0.7, "Emergency": 0.8},
        },
        progression={
            "description": "Paediatric fracture — swelling may compromise circulation over time.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 1, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.001,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Orthopedics",
        },
        xai={
            "primary_diagnosis": "Distal radius fracture",
            "differentials": [
                {"diagnosis": "Distal radius fracture", "probability": 0.85},
                {"diagnosis": "Greenstick fracture", "probability": 0.10},
                {"diagnosis": "Soft tissue injury", "probability": 0.05},
            ],
            "symptom_weights": {"obvious deformity": 0.95, "moderate pain": 0.70},
            "vital_flags": [],
            "confidence": 0.92,
            "key_reasoning_points": [
                "Visible deformity requires imaging regardless of pulse",
                "Paediatric patients need weight-based sedation protocol",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_003",
            "difficulty": "easy",
            "presentation": (
                "22-year-old male. Unresponsive. Pinpoint pupils, RR 6, cyanotic lips. "
                "Empty opioid bottle nearby. BP 88/54, HR 58, O2 sat 82%."
            ),
            "vitals": {"bp": "88/54", "hr": 58, "o2_sat": 82, "rr": 6, "temp": 35.9},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Opioid OD, O2 82%, RR 6 — immediate naloxone + airway. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.4, 3: 0.01, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "General"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.5, "General": 0.01},
        },
        progression={
            "description": "Opioid OD — anoxic brain injury begins rapidly without airway management.",
            "per_timestep": {"o2_sat_delta": -3, "hr_delta": -2, "bp_systolic_delta": -4, "consciousness_delta": -0.3},
            "mortality_risk_per_step": 0.18,
            "critical_thresholds": {"o2_sat": 70, "hr": 40, "bp_systolic": 60},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": True,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Emergency",
        },
        xai={
            "primary_diagnosis": "Opioid overdose",
            "differentials": [
                {"diagnosis": "Opioid overdose", "probability": 0.90},
                {"diagnosis": "Mixed drug overdose", "probability": 0.08},
                {"diagnosis": "Pontine stroke", "probability": 0.02},
            ],
            "symptom_weights": {"pinpoint pupils": 0.95, "respiratory depression": 0.90},
            "vital_flags": ["SpO₂ 82% (critical hypoxia)", "HR 58 (bradycardia)", "RR 6 (respiratory failure)"],
            "confidence": 0.93,
            "key_reasoning_points": [
                "Toxidrome triad: pinpoint pupils + respiratory depression + altered consciousness",
                "Naloxone 0.4–2 mg IV/IM — may need repeated doses for fentanyl",
                "Airway takes absolute priority",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_004",
            "difficulty": "easy",
            "presentation": (
                "35-year-old female. Paper cut on right index finger. "
                "Bleeding stopped. No numbness, full ROM, tetanus current. Vitals normal."
            ),
            "vitals": {"bp": "118/76", "hr": 72, "o2_sat": 99, "rr": 16, "temp": 36.7},
            "correct_esi": 5,
            "correct_department": "General",
            "reasoning": "Minor laceration, bleeding controlled, no NV compromise. ESI-5.",
            "esi_partial_credit": {1: 0.01, 2: 0.01, 3: 0.1, 4: 0.6, 5: 0.99},
            "department_options": ["General", "Emergency", "Orthopedics"],
            "department_scores": {"General": 0.99, "Emergency": 0.4, "Orthopedics": 0.3},
        },
        progression={
            "description": "Minor wound — no progression expected.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.0,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": None,
        },
        xai={
            "primary_diagnosis": "Superficial finger laceration",
            "differentials": [{"diagnosis": "Superficial laceration", "probability": 0.99}],
            "symptom_weights": {"superficial laceration": 0.95},
            "vital_flags": [],
            "confidence": 0.98,
            "key_reasoning_points": ["Neurovascular exam normal", "No imaging required"],
        },
    ),

    _static(
        base={
            "id": "easy_005",
            "difficulty": "easy",
            "presentation": (
                "55-year-old male. Thunderclap headache, onset during weightlifting. "
                "Neck stiffness, photophobia. BP 210/120, HR 88, O2 sat 97%, GCS 14."
            ),
            "vitals": {"bp": "210/120", "hr": 88, "o2_sat": 97, "rr": 18, "temp": 37.1},
            "correct_esi": 2,
            "correct_department": "Neurology",
            "reasoning": "Thunderclap headache + neck stiffness = SAH until proven otherwise. ESI-2.",
            "esi_partial_credit": {1: 0.7, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
            "department_options": ["Neurology", "Emergency", "Resuscitation"],
            "department_scores": {"Neurology": 0.99, "Emergency": 0.9, "Resuscitation": 0.6},
        },
        progression={
            "description": "SAH — re-bleed risk is highest in first hours.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 2, "bp_systolic_delta": +3, "consciousness_delta": -0.15},
            "mortality_risk_per_step": 0.09,
            "critical_thresholds": {"consciousness": 0.5, "bp_systolic": 230},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Neurology",
        },
        xai={
            "primary_diagnosis": "Subarachnoid haemorrhage",
            "differentials": [
                {"diagnosis": "SAH", "probability": 0.70},
                {"diagnosis": "Hypertensive emergency", "probability": 0.20},
                {"diagnosis": "Meningitis", "probability": 0.10},
            ],
            "symptom_weights": {"thunderclap headache": 0.95, "neck stiffness": 0.85},
            "vital_flags": ["BP 210/120 (hypertensive crisis)"],
            "confidence": 0.86,
            "key_reasoning_points": [
                "Sentinel bleed classically precedes catastrophic haemorrhage",
                "Non-contrast CT must be done within 6 hours of onset",
                "LP if CT negative and suspicion remains",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_006",
            "difficulty": "easy",
            "presentation": (
                "29-year-old female. Sore throat, mild fever 37.8°C, runny nose for 2 days. "
                "No dysphagia, no drooling. O2 sat 99%, HR 80."
            ),
            "vitals": {"bp": "115/72", "hr": 80, "o2_sat": 99, "rr": 16, "temp": 37.8},
            "correct_esi": 5,
            "correct_department": "General",
            "reasoning": "Mild URI, stable vitals, no airway compromise. ESI-5.",
            "esi_partial_credit": {1: 0.01, 2: 0.01, 3: 0.2, 4: 0.7, 5: 0.99},
            "department_options": ["General", "Emergency", "Pulmonology"],
            "department_scores": {"General": 0.99, "Emergency": 0.3, "Pulmonology": 0.2},
        },
        progression={
            "description": "URI — stable, no expected deterioration.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.0,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": None,
        },
        xai={
            "primary_diagnosis": "Viral upper respiratory infection",
            "differentials": [
                {"diagnosis": "Viral URI", "probability": 0.80},
                {"diagnosis": "Streptococcal pharyngitis", "probability": 0.18},
                {"diagnosis": "Peritonsillar abscess", "probability": 0.02},
            ],
            "symptom_weights": {"sore throat": 0.70, "mild fever": 0.55, "runny nose": 0.60},
            "vital_flags": [],
            "confidence": 0.90,
            "key_reasoning_points": [
                "No airway compromise signs (drooling, stridor, muffled voice)",
                "Appropriate for GP, not emergency department",
            ],
        },
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # MEDIUM (6 cases)
    # ─────────────────────────────────────────────────────────────────────────

    _static(
        base={
            "id": "medium_001",
            "difficulty": "medium",
            "presentation": (
                "45-year-old female. Chest tightness and dyspnea 1 hour. History of panic disorder. "
                "Hyperventilating, tingling in fingers. BP 138/88, HR 112, O2 sat 96%, RR 24."
            ),
            "vitals": {"bp": "138/88", "hr": 112, "o2_sat": 96, "rr": 24, "temp": 36.9},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Cannot exclude ACS without ECG+troponin in a 45-year-old. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.5, 4: 0.1, 5: 0.01},
            "department_options": ["Emergency", "Cardiology", "Psychiatry"],
            "department_scores": {"Emergency": 0.99, "Cardiology": 0.8, "Psychiatry": 0.1},
        },
        progression={
            "description": "If cardiac — deterioration possible. If panic — stable.",
            "per_timestep": {"o2_sat_delta": -0.5, "hr_delta": 2, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.03,
            "critical_thresholds": {"o2_sat": 90},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Emergency",
        },
        xai={
            "primary_diagnosis": "ACS vs panic disorder (rule-out required)",
            "differentials": [
                {"diagnosis": "Panic disorder", "probability": 0.55},
                {"diagnosis": "NSTEMI / ACS", "probability": 0.35},
                {"diagnosis": "PE", "probability": 0.10},
            ],
            "symptom_weights": {"chest tightness": 0.75, "tachycardia": 0.70, "hyperventilation": 0.65},
            "vital_flags": ["HR 112 (tachycardia)", "RR 24 (tachypnea)"],
            "confidence": 0.70,
            "key_reasoning_points": [
                "Panic history does not exclude ACS — protocols require cardiac rule-out",
                "ECG and troponin are mandatory in any 45-year-old with chest symptoms",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_002",
            "difficulty": "medium",
            "presentation": (
                "72-year-old male. Confusion and fever (38.2°C) for 1 day. Cloudy urine. "
                "BP 128/82, HR 96, O2 sat 95%, RR 20."
            ),
            "vitals": {"bp": "128/82", "hr": 96, "o2_sat": 95, "rr": 20, "temp": 38.2},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Elderly confusion + fever = sepsis workup. O2 95% borderline. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.4, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "General", "Neurology"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Neurology": 0.4},
        },
        progression={
            "description": "UTI-sourced sepsis in elderly — can deteriorate to septic shock rapidly.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": 4, "bp_systolic_delta": -4, "consciousness_delta": -0.12},
            "mortality_risk_per_step": 0.07,
            "critical_thresholds": {"bp_systolic": 85, "o2_sat": 90},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Emergency",
        },
        xai={
            "primary_diagnosis": "Urosepsis",
            "differentials": [
                {"diagnosis": "Urosepsis", "probability": 0.65},
                {"diagnosis": "Pneumonia-associated sepsis", "probability": 0.20},
                {"diagnosis": "Delirium NOS", "probability": 0.15},
            ],
            "symptom_weights": {"acute confusion": 0.80, "fever": 0.75, "cloudy urine": 0.70},
            "vital_flags": ["HR 96 (tachycardia)", "SpO₂ 95% (borderline)", "Temp 38.2°C"],
            "confidence": 0.78,
            "key_reasoning_points": [
                "Confusion in elderly is the presenting sign of sepsis — not just 'delirium'",
                "SIRS criteria met: fever + tachycardia",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_003",
            "difficulty": "medium",
            "presentation": (
                "28-year-old male. Right flank pain to groin, 8/10, colicky. Nausea, one vomit. "
                "Haematuria on dipstick. BP 142/90, HR 108, O2 sat 98%, afebrile."
            ),
            "vitals": {"bp": "142/90", "hr": 108, "o2_sat": 98, "rr": 18, "temp": 36.8},
            "correct_esi": 3,
            "correct_department": "Emergency",
            "reasoning": "Renal colic, afebrile. IV analgesia + CT KUB. ESI-3.",
            "esi_partial_credit": {1: 0.1, 2: 0.6, 3: 0.99, 4: 0.2, 5: 0.01},
            "department_options": ["Emergency", "General", "Gastroenterology"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Gastroenterology": 0.3},
        },
        progression={
            "description": "Afebrile stone — pain escalates but not immediately life-threatening.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 1, "bp_systolic_delta": 2, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.004,
            "critical_thresholds": {"temp": 38.5},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": None,
        },
        xai={
            "primary_diagnosis": "Ureteric colic",
            "differentials": [
                {"diagnosis": "Ureteric stone", "probability": 0.82},
                {"diagnosis": "Appendicitis", "probability": 0.10},
                {"diagnosis": "AAA (less likely, young)", "probability": 0.03},
            ],
            "symptom_weights": {"flank pain radiating to groin": 0.90, "haematuria": 0.75, "colicky pattern": 0.80},
            "vital_flags": ["HR 108 (tachycardia — pain response)"],
            "confidence": 0.87,
            "key_reasoning_points": [
                "Afebrile excludes infected hydronephrosis (which would be ESI-2)",
                "CT KUB is gold standard",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_004",
            "difficulty": "medium",
            "presentation": (
                "19-year-old female. 32 weeks pregnant. Severe headache, visual disturbances, "
                "hand/face swelling. BP 158/102, HR 90, O2 sat 98%, RR 18."
            ),
            "vitals": {"bp": "158/102", "hr": 90, "o2_sat": 98, "rr": 18, "temp": 36.9},
            "correct_esi": 2,
            "correct_department": "Obstetrics",
            "reasoning": "Severe preeclampsia — eclamptic seizure risk. Magnesium now. ESI-2.",
            "esi_partial_credit": {1: 0.6, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
            "department_options": ["Obstetrics", "Emergency", "General"],
            "department_scores": {"Obstetrics": 0.99, "Emergency": 0.7, "General": 0.1},
        },
        progression={
            "description": "Preeclampsia can evolve to eclampsia within minutes.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 2, "bp_systolic_delta": 3, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"bp_systolic": 180},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Obstetrics",
        },
        xai={
            "primary_diagnosis": "Severe preeclampsia",
            "differentials": [
                {"diagnosis": "Severe preeclampsia", "probability": 0.80},
                {"diagnosis": "HELLP syndrome", "probability": 0.12},
                {"diagnosis": "Gestational hypertension", "probability": 0.08},
            ],
            "symptom_weights": {"severe headache": 0.80, "visual disturbances": 0.85, "oedema": 0.70},
            "vital_flags": ["BP 158/102 (severe range preeclampsia criteria)"],
            "confidence": 0.87,
            "key_reasoning_points": [
                "BP ≥160/110 = severe criteria — treat before full workup",
                "Visual changes indicate cerebral involvement",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_005",
            "difficulty": "medium",
            "presentation": (
                "50-year-old male. Epigastric pain after large meal, burning, "
                "partially relieved by antacids. History of GERD. "
                "BP 125/80, HR 78, O2 sat 99%, afebrile."
            ),
            "vitals": {"bp": "125/80", "hr": 78, "o2_sat": 99, "rr": 16, "temp": 36.7},
            "correct_esi": 3,
            "correct_department": "Emergency",
            "reasoning": "50M with chest-radiating pain — must exclude ACS. ECG + troponin. ESI-3.",
            "esi_partial_credit": {1: 0.2, 2: 0.6, 3: 0.99, 4: 0.3, 5: 0.01},
            "department_options": ["Emergency", "Gastroenterology", "Cardiology"],
            "department_scores": {"Emergency": 0.99, "Gastroenterology": 0.6, "Cardiology": 0.7},
        },
        progression={
            "description": "Stable GERD presentation — unlikely rapid deterioration.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.01,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Emergency",
        },
        xai={
            "primary_diagnosis": "GERD (ACS excluded)",
            "differentials": [
                {"diagnosis": "GERD / peptic ulcer", "probability": 0.60},
                {"diagnosis": "NSTEMI / ACS", "probability": 0.30},
                {"diagnosis": "Oesophageal spasm", "probability": 0.10},
            ],
            "symptom_weights": {"epigastric burning": 0.70, "relief with antacids": 0.65},
            "vital_flags": [],
            "confidence": 0.65,
            "key_reasoning_points": [
                "In men >45, GERD-like symptoms require cardiac rule-out",
                "Relief with antacids does not exclude ACS",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_006",
            "difficulty": "medium",
            "presentation": (
                "40-year-old female. Palpitations and dizziness 3 hours. WPW history, no meds. "
                "HR 178 irregular, BP 100/68, O2 sat 97%, RR 18."
            ),
            "vitals": {"bp": "100/68", "hr": 178, "o2_sat": 97, "rr": 18, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Cardiology",
            "reasoning": "WPW + AF = ventricular fibrillation risk. No AV nodal blockers. ESI-2.",
            "esi_partial_credit": {1: 0.6, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Cardiology", "Emergency", "Resuscitation"],
            "department_scores": {"Cardiology": 0.99, "Emergency": 0.8, "Resuscitation": 0.6},
        },
        progression={
            "description": "WPW + AF — degeneration to VF is sudden and unpredictable.",
            "per_timestep": {"o2_sat_delta": -0.5, "hr_delta": 5, "bp_systolic_delta": -3, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.08,
            "critical_thresholds": {"hr": 200, "bp_systolic": 80},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Cardiology",
        },
        xai={
            "primary_diagnosis": "AF with WPW (pre-excited AF)",
            "differentials": [
                {"diagnosis": "Pre-excited AF (WPW)", "probability": 0.75},
                {"diagnosis": "SVT", "probability": 0.15},
                {"diagnosis": "Ventricular tachycardia", "probability": 0.10},
            ],
            "symptom_weights": {"palpitations": 0.80, "dizziness": 0.70},
            "vital_flags": ["HR 178 (extreme tachycardia)", "BP 100/68 (borderline hypotension)"],
            "confidence": 0.82,
            "key_reasoning_points": [
                "AV nodal blockers (digoxin, verapamil, adenosine) are contraindicated",
                "Procainamide or DC cardioversion are appropriate treatments",
            ],
        },
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # HARD (6 cases)
    # ─────────────────────────────────────────────────────────────────────────

    _static(
        base={
            "id": "hard_001",
            "difficulty": "hard",
            "presentation": (
                "34-year-old female. Fatigue 3 weeks, orthostatic dizziness, fever 37.9°C, "
                "joint pain. SLE on hydroxychloroquine. BP 108/70 → 92/60 on standing, HR 94, O2 98%."
            ),
            "vitals": {"bp": "108/70", "hr": 94, "o2_sat": 98, "rr": 16, "temp": 37.9},
            "correct_esi": 3,
            "correct_department": "Emergency",
            "reasoning": "Lupus flare vs infection vs drug toxicity. Orthostatic hypotension + fever in immunocompromised. ESI-3.",
            "esi_partial_credit": {1: 0.2, 2: 0.6, 3: 0.99, 4: 0.3, 5: 0.01},
            "department_options": ["Emergency", "General", "Neurology"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Neurology": 0.2},
        },
        progression={
            "description": "Could deteriorate if infection or adrenal crisis missed.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 2, "bp_systolic_delta": -3, "consciousness_delta": -0.05},
            "mortality_risk_per_step": 0.03,
            "critical_thresholds": {"bp_systolic": 80},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Rheumatology",
        },
        xai={
            "primary_diagnosis": "Lupus flare vs infection",
            "differentials": [
                {"diagnosis": "SLE flare", "probability": 0.45},
                {"diagnosis": "Infection in immunocompromised", "probability": 0.35},
                {"diagnosis": "Drug toxicity (hydroxychloroquine)", "probability": 0.15},
                {"diagnosis": "Adrenal insufficiency", "probability": 0.05},
            ],
            "symptom_weights": {"joint pain": 0.70, "orthostatic hypotension": 0.75, "fever": 0.65},
            "vital_flags": ["Orthostatic BP drop >20 mmHg (autonomic / volume depletion)"],
            "confidence": 0.60,
            "key_reasoning_points": [
                "Immunosuppressed patients can be infected with normal-range WBC",
                "Orthostatic drop suggests volume depletion or autonomic dysfunction",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_002",
            "difficulty": "hard",
            "presentation": (
                "58-year-old male. Vague abdominal discomfort 6 hours. AF on warfarin. "
                "Pain diffuse 3/10. BP 132/84, HR 88 (irregular), O2 sat 97%, afebrile."
            ),
            "vitals": {"bp": "132/84", "hr": 88, "o2_sat": 97, "rr": 17, "temp": 36.9},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "AF + warfarin + abdominal pain = mesenteric ischaemia until excluded. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "Gastroenterology", "General"],
            "department_scores": {"Emergency": 0.99, "Gastroenterology": 0.5, "General": 0.1},
        },
        progression={
            "description": "Mesenteric ischaemia has catastrophic mortality if missed.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 3, "bp_systolic_delta": -5, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.12,
            "critical_thresholds": {"bp_systolic": 80},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Vascular surgery",
        },
        xai={
            "primary_diagnosis": "Mesenteric ischaemia (rule-out)",
            "differentials": [
                {"diagnosis": "Mesenteric ischaemia", "probability": 0.50},
                {"diagnosis": "Bowel obstruction", "probability": 0.25},
                {"diagnosis": "GI bleed", "probability": 0.15},
                {"diagnosis": "Non-specific abdominal pain", "probability": 0.10},
            ],
            "symptom_weights": {"abdominal pain with AF": 0.85, "pain out of proportion": 0.90},
            "vital_flags": ["HR 88 irregular (atrial fibrillation)"],
            "confidence": 0.58,
            "key_reasoning_points": [
                "AF is a major risk factor for arterial embolism including mesenteric",
                "Pain severity often underestimates extent of ischaemia",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_003",
            "difficulty": "hard",
            "presentation": (
                "26-year-old male. Police-assisted. Agitated, paranoid, incoherent. "
                "Possible substance ingestion. Temp 39.8°C, HR 138, BP 168/96, O2 96%, pupils dilated. Muscle rigidity."
            ),
            "vitals": {"bp": "168/96", "hr": 138, "o2_sat": 96, "rr": 22, "temp": 39.8},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Hyperthermia + rigidity + tachycardia = serotonin syndrome. Rhabdo + arrest risk. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.1, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "Psychiatry"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.6, "Psychiatry": 0.01},
        },
        progression={
            "description": "Serotonin syndrome — hyperthermia rapidly causes multi-organ failure.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": 5, "bp_systolic_delta": 4, "consciousness_delta": -0.25},
            "mortality_risk_per_step": 0.15,
            "critical_thresholds": {"temp": 41.0, "hr": 165},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": True,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Toxicology",
        },
        xai={
            "primary_diagnosis": "Serotonin syndrome",
            "differentials": [
                {"diagnosis": "Serotonin syndrome", "probability": 0.55},
                {"diagnosis": "Stimulant toxidrome", "probability": 0.25},
                {"diagnosis": "Neuroleptic malignant syndrome", "probability": 0.15},
                {"diagnosis": "Heat stroke", "probability": 0.05},
            ],
            "symptom_weights": {"hyperthermia": 0.85, "muscle rigidity": 0.90, "agitation": 0.80},
            "vital_flags": ["Temp 39.8°C (hyperpyrexia)", "HR 138 (extreme tachycardia)"],
            "confidence": 0.72,
            "key_reasoning_points": [
                "Hunter criteria: clonus + agitation + hyperthermia",
                "Immediate cooling + benzodiazepines before toxicology",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_004",
            "difficulty": "hard",
            "presentation": (
                "62-year-old female. Dyspnea on exertion 2 weeks, now at rest. "
                "Post 14-hour flight 5 days ago. Left calf swelling. "
                "BP 118/76, HR 102, O2 sat 93%, RR 22. BMI 34."
            ),
            "vitals": {"bp": "118/76", "hr": 102, "o2_sat": 93, "rr": 22, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Long-haul + DVT + dyspnea + O2 93% = high Wells score PE. Urgent CT-PA. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "Pulmonology", "General"],
            "department_scores": {"Emergency": 0.99, "Pulmonology": 0.7, "General": 0.1},
        },
        progression={
            "description": "PE — hypoxia and haemodynamic instability progress without anticoagulation.",
            "per_timestep": {"o2_sat_delta": -2, "hr_delta": 4, "bp_systolic_delta": -3, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.10,
            "critical_thresholds": {"o2_sat": 85, "bp_systolic": 85},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Pulmonology",
        },
        xai={
            "primary_diagnosis": "Pulmonary embolism",
            "differentials": [
                {"diagnosis": "Pulmonary embolism", "probability": 0.70},
                {"diagnosis": "Pneumonia", "probability": 0.15},
                {"diagnosis": "Acute HF exacerbation", "probability": 0.10},
                {"diagnosis": "COPD exacerbation", "probability": 0.05},
            ],
            "symptom_weights": {"DVT signs": 0.80, "post-immobility": 0.75, "dyspnea": 0.85},
            "vital_flags": ["SpO₂ 93% (hypoxia at rest)", "HR 102 (tachycardia)", "RR 22 (tachypnea)"],
            "confidence": 0.78,
            "key_reasoning_points": [
                "Wells score >4: DVT + tachycardia + immobility",
                "SpO₂ 93% at rest must be treated as a clinical alarm",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_005",
            "difficulty": "hard",
            "presentation": (
                "44-year-old male. Severe mid-back pain at rest, sudden onset. "
                "Poorly controlled hypertension. BP 202/114 right arm, 168/98 left arm. "
                "HR 92, O2 sat 96%, RR 20."
            ),
            "vitals": {"bp": "202/114", "hr": 92, "o2_sat": 96, "rr": 20, "temp": 36.9},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "34 mmHg BP differential + tearing pain = aortic dissection. CT-aortogram stat. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.01, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "Cardiology"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.6, "Cardiology": 0.4},
        },
        progression={
            "description": "Aortic dissection — propagation to coronaries or rupture is rapidly fatal.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 2, "bp_systolic_delta": -6, "consciousness_delta": -0.2},
            "mortality_risk_per_step": 0.12,
            "critical_thresholds": {"bp_systolic": 75, "consciousness": 0.4},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Cardiothoracic surgery",
        },
        xai={
            "primary_diagnosis": "Aortic dissection",
            "differentials": [
                {"diagnosis": "Aortic dissection (Type A)", "probability": 0.75},
                {"diagnosis": "STEMI (involving right coronary)", "probability": 0.12},
                {"diagnosis": "PE", "probability": 0.08},
                {"diagnosis": "Musculoskeletal pain", "probability": 0.05},
            ],
            "symptom_weights": {"BP arm differential": 0.95, "tearing back pain": 0.90},
            "vital_flags": ["BP differential 34 mmHg (highly specific for dissection)"],
            "confidence": 0.80,
            "key_reasoning_points": [
                "BP differential >20 mmHg between arms = aortic dissection until proven otherwise",
                "Do NOT anticoagulate until dissection excluded",
                "Target BP <120 mmHg systolic with esmolol",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_006",
            "difficulty": "hard",
            "presentation": (
                "17-year-old male. Headache + confusion 2 hours. Head injury playing football 3 days ago. "
                "GCS 13, pupils equal. BP 148/88, HR 58, O2 sat 99%, RR 14."
            ),
            "vitals": {"bp": "148/88", "hr": 58, "o2_sat": 99, "rr": 14, "temp": 36.7},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Lucid interval + Cushing's triad = epidural haematoma. Neurosurgical emergency. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.4, 3: 0.1, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Neurology", "Emergency"],
            "department_scores": {"Resuscitation": 0.99, "Neurology": 0.7, "Emergency": 0.6},
        },
        progression={
            "description": "Epidural haematoma — herniation and death can occur within 30 minutes.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": -3, "bp_systolic_delta": +5, "consciousness_delta": -0.3},
            "mortality_risk_per_step": 0.16,
            "critical_thresholds": {"consciousness": 0.3, "hr": 45},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": True,
            "ct_scanner": True, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Neurosurgery",
        },
        xai={
            "primary_diagnosis": "Epidural haematoma",
            "differentials": [
                {"diagnosis": "Epidural haematoma", "probability": 0.80},
                {"diagnosis": "Subdural haematoma", "probability": 0.15},
                {"diagnosis": "Cerebral contusion", "probability": 0.05},
            ],
            "symptom_weights": {"lucid interval": 0.90, "post-trauma confusion": 0.85},
            "vital_flags": ["HR 58 (bradycardia — Cushing reflex)", "BP 148/88 (Cushing reflex)", "RR 14 (slowing)"],
            "confidence": 0.88,
            "key_reasoning_points": [
                "Lucid interval is pathognomonic for epidural haematoma",
                "Cushing's triad: hypertension + bradycardia + respiratory slowing",
                "Immediate CT head + neurosurgery",
            ],
        },
    ),
]

PATIENT_MAP = {p["id"]: p for p in PATIENTS}
EASY_CASES = [p for p in PATIENTS if p["difficulty"] == "easy"]
MEDIUM_CASES = [p for p in PATIENTS if p["difficulty"] == "medium"]
HARD_CASES = [p for p in PATIENTS if p["difficulty"] == "hard"]

DEPARTMENTS = [
    "Resuscitation", "Emergency", "Cardiology", "Neurology",
    "Trauma", "Pediatrics", "Orthopedics", "General",
    "Psychiatry", "Obstetrics", "Gastroenterology", "Pulmonology",
]

ESI_DESCRIPTIONS = {
    1: "Immediate — life-threatening, requires instant intervention",
    2: "Emergent — high risk, should not wait",
    3: "Urgent — stable but needs multiple resources",
    4: "Less Urgent — needs one resource only",
    5: "Non-Urgent — no resources needed",
}
