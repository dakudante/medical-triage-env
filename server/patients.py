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
            age=age, sex=sex, hr=hr, o2=o2,
            bp_sys=bp_sys, bp_dia=bp_dia,
            rr=rr, temp=temp,
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

    # ── Template 9: Anaphylaxis ───────────────────────────────────────────────
    {
        "condition_id": "anaphylaxis",
        "condition_name": "Anaphylaxis — severe allergic reaction",
        "difficulty": "easy",
        "age_range": (18, 65),
        "vitals_ranges": {
            "hr": (110, 140), "bp_systolic": (70, 95), "bp_diastolic": (40, 60),
            "o2_sat": (88, 94), "rr": (22, 28), "temp": (36.5, 37.2),
        },
        "symptom_probabilities": {
            "urticaria": 0.90, "facial swelling": 0.80, "stridor": 0.70,
            "dyspnoea": 0.85, "known allergen exposure": 0.75, "nausea": 0.60,
        },
        "correct_esi": 1,
        "correct_department": "Resuscitation",
        "reasoning_template": (
            "Anaphylaxis in a {age}-year-old {sex}. Stridor + BP {bp_sys}/{bp_dia} "
            "= airway compromise + distributive shock. ESI-1 — IM epinephrine immediately."
        ),
        "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.05, 4: 0.01, 5: 0.01},
        "department_options": ["Resuscitation", "Emergency", "General"],
        "department_scores": {"Resuscitation": 0.99, "Emergency": 0.7, "General": 0.01},
        "progression": {
            "description": "Anaphylaxis — airway swelling progresses rapidly; epinephrine is time-critical.",
            "per_timestep": {"o2_sat_delta": -3, "hr_delta": +6, "bp_systolic_delta": -8, "consciousness_delta": -0.15},
            "mortality_risk_per_step": 0.15,
            "critical_thresholds": {"o2_sat": 85, "bp_systolic": 65},
        },
        "resources_required": {
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": True,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Emergency medicine",
        },
        "xai_base": {
            "primary_diagnosis": "Anaphylaxis",
            "differentials": [
                {"diagnosis": "Anaphylaxis", "probability": 0.90},
                {"diagnosis": "Angioedema (non-anaphylactic)", "probability": 0.06},
                {"diagnosis": "Vocal cord dysfunction", "probability": 0.04},
            ],
            "symptom_weights": {
                "stridor": 0.95, "urticaria": 0.85, "hypotension": 0.90, "allergen exposure": 0.88,
            },
            "vital_emphasis": ["bp_systolic", "o2_sat", "hr"],
            "key_reasoning_points": [
                "Stridor = upper airway compromise — IM epinephrine 0.5mg immediately",
                "Hypotension + urticaria after allergen = anaphylaxis by WHO criteria",
                "IV access, supine position, high-flow O2",
            ],
            "base_confidence": 0.92,
        },
    },

    # ── Template 10: DKA ─────────────────────────────────────────────────────
    {
        "condition_id": "dka",
        "condition_name": "Diabetic ketoacidosis (DKA)",
        "difficulty": "medium",
        "age_range": (14, 45),
        "vitals_ranges": {
            "hr": (105, 130), "bp_systolic": (95, 115), "bp_diastolic": (60, 75),
            "o2_sat": (96, 99), "rr": (22, 30), "temp": (36.8, 37.5),
        },
        "symptom_probabilities": {
            "nausea": 0.85, "vomiting": 0.80, "abdominal pain": 0.70,
            "polyuria": 0.75, "polydipsia": 0.70, "known diabetes": 0.90,
            "fruity breath": 0.55,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "DKA in a {age}-year-old {sex} with diabetes. "
            "HR {hr}, RR {rr} (Kussmaul), glucose likely markedly elevated. ESI-2."
        ),
        "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
        "department_options": ["Emergency", "General", "Resuscitation"],
        "department_scores": {"Emergency": 0.99, "General": 0.5, "Resuscitation": 0.6},
        "progression": {
            "description": "DKA — cerebral oedema and hypokalaemia can cause cardiac arrest.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +3, "bp_systolic_delta": -4, "consciousness_delta": -0.08},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"bp_systolic": 85, "consciousness": 0.4},
        },
        "resources_required": {
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Endocrinology",
        },
        "xai_base": {
            "primary_diagnosis": "Diabetic ketoacidosis",
            "differentials": [
                {"diagnosis": "DKA", "probability": 0.82},
                {"diagnosis": "Hyperglycaemic hyperosmolar state", "probability": 0.10},
                {"diagnosis": "Starvation ketosis", "probability": 0.05},
                {"diagnosis": "Sepsis-triggered DKA", "probability": 0.03},
            ],
            "symptom_weights": {
                "Kussmaul breathing": 0.90, "vomiting + diabetes": 0.85,
                "abdominal pain": 0.65, "fruity breath": 0.70,
            },
            "vital_emphasis": ["hr", "rr", "bp_systolic"],
            "key_reasoning_points": [
                "Tachypnoea in a diabetic with vomiting = Kussmaul = metabolic acidosis",
                "Check ketones, VBG, K+ immediately — hypokalaemia is the killer",
                "Fixed-rate insulin 0.1 units/kg/hr; do not start until K+ >3.5",
            ],
            "base_confidence": 0.85,
        },
    },

    # ── Template 11: Heat Stroke ──────────────────────────────────────────────
    {
        "condition_id": "heat_stroke",
        "condition_name": "Exertional heat stroke",
        "difficulty": "hard",
        "age_range": (18, 50),
        "vitals_ranges": {
            "hr": (125, 150), "bp_systolic": (85, 105), "bp_diastolic": (55, 70),
            "o2_sat": (92, 96), "rr": (24, 32), "temp": (40.2, 42.0),
        },
        "symptom_probabilities": {
            "confusion": 0.90, "anhidrosis": 0.80, "exertional context": 0.95,
            "hot dry skin": 0.75, "nausea": 0.60, "seizure": 0.20,
        },
        "correct_esi": 1,
        "correct_department": "Resuscitation",
        "reasoning_template": (
            "Heat stroke in a {age}-year-old {sex}. Temp {temp}°C, confusion, anhidrosis. "
            "Core cooling is the intervention — ESI-1."
        ),
        "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.05, 4: 0.01, 5: 0.01},
        "department_options": ["Resuscitation", "Emergency", "Trauma"],
        "department_scores": {"Resuscitation": 0.99, "Emergency": 0.65, "Trauma": 0.3},
        "progression": {
            "description": "Heat stroke — multi-organ failure develops rapidly without cooling.",
            "per_timestep": {"o2_sat_delta": -2, "hr_delta": +5, "bp_systolic_delta": -6, "consciousness_delta": -0.2},
            "mortality_risk_per_step": 0.13,
            "critical_thresholds": {"consciousness": 0.3, "bp_systolic": 75},
        },
        "resources_required": {
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Emergency medicine",
        },
        "xai_base": {
            "primary_diagnosis": "Exertional heat stroke",
            "differentials": [
                {"diagnosis": "Exertional heat stroke", "probability": 0.78},
                {"diagnosis": "Serotonin syndrome", "probability": 0.10},
                {"diagnosis": "Sepsis with hyperthermia", "probability": 0.08},
                {"diagnosis": "Malignant hyperthermia", "probability": 0.04},
            ],
            "symptom_weights": {
                "temp >40°C": 0.95, "anhidrosis": 0.88, "exertional context": 0.90, "confusion": 0.85,
            },
            "vital_emphasis": ["temp", "hr", "bp_systolic"],
            "key_reasoning_points": [
                "Heat stroke = temp >40°C + CNS dysfunction; anhidrosis distinguishes from heat exhaustion",
                "Cold water immersion is most effective cooling method",
                "Target core temp <39°C within 30 minutes — every minute matters",
            ],
            "base_confidence": 0.82,
        },
    },

    # ── Template 12: Lithium Toxicity ─────────────────────────────────────────
    {
        "condition_id": "lithium_toxicity",
        "condition_name": "Lithium toxicity",
        "difficulty": "hard",
        "age_range": (35, 75),
        "vitals_ranges": {
            "hr": (48, 62), "bp_systolic": (95, 115), "bp_diastolic": (58, 72),
            "o2_sat": (95, 98), "rr": (12, 16), "temp": (36.5, 37.2),
        },
        "symptom_probabilities": {
            "coarse tremor": 0.88, "confusion": 0.82, "known lithium use": 0.99,
            "recent dehydration/illness": 0.70, "ataxia": 0.60, "vomiting": 0.55,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "Lithium toxicity in a {age}-year-old {sex} on lithium. "
            "Coarse tremor, confusion, bradycardia HR {hr} — urgent lithium level and renal function. ESI-2."
        ),
        "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
        "department_options": ["Emergency", "Psychiatry", "General", "Neurology"],
        "department_scores": {"Emergency": 0.99, "Psychiatry": 0.3, "General": 0.4, "Neurology": 0.5},
        "progression": {
            "description": "Lithium toxicity — seizures and arrhythmias worsen without intervention.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": -2, "bp_systolic_delta": -3, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.06,
            "critical_thresholds": {"hr": 44, "bp_systolic": 80},
        },
        "resources_required": {
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Toxicology",
        },
        "xai_base": {
            "primary_diagnosis": "Lithium toxicity",
            "differentials": [
                {"diagnosis": "Lithium toxicity", "probability": 0.65},
                {"diagnosis": "Hyponatraemia (SIADH)", "probability": 0.15},
                {"diagnosis": "Encephalopathy from unrelated cause", "probability": 0.12},
                {"diagnosis": "Bipolar relapse", "probability": 0.08},
            ],
            "symptom_weights": {
                "coarse tremor": 0.88, "lithium use": 0.99, "bradycardia": 0.72, "confusion": 0.80,
            },
            "vital_emphasis": ["hr", "bp_systolic"],
            "key_reasoning_points": [
                "Dehydration (illness, NSAID use, low salt diet) precipitates lithium toxicity",
                "Coarse tremor — not fine — is the hallmark of toxicity vs therapeutic side effect",
                "Lithium level + U&E + ECG; haemodialysis if level >3.5 mmol/L or severe symptoms",
            ],
            "base_confidence": 0.65,
        },
    },

    # ── Template 13: Post-Op Complication ─────────────────────────────────────
    {
        "condition_id": "post_op_complication",
        "condition_name": "Post-operative complication — surgical site infection / collection",
        "difficulty": "medium",
        "age_range": (25, 75),
        "vitals_ranges": {
            "hr": (95, 115), "bp_systolic": (105, 130), "bp_diastolic": (65, 85),
            "o2_sat": (96, 99), "rr": (16, 22), "temp": (38.4, 39.2),
        },
        "symptom_probabilities": {
            "wound erythema": 0.80, "wound swelling": 0.75, "post-op fever": 0.90,
            "increasing pain": 0.85, "purulent discharge": 0.50, "recent surgery": 0.99,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "Post-operative complication in a {age}-year-old {sex}. "
            "Fever {temp}°C, wound infection — SIRS criteria met. ESI-2."
        ),
        "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
        "department_options": ["Emergency", "General", "Gastroenterology"],
        "department_scores": {"Emergency": 0.99, "General": 0.7, "Gastroenterology": 0.3},
        "progression": {
            "description": "Post-op sepsis — intra-abdominal collection can progress to septic shock.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +4, "bp_systolic_delta": -4, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"bp_systolic": 88},
        },
        "resources_required": {
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "General surgery",
        },
        "xai_base": {
            "primary_diagnosis": "Post-operative surgical site infection",
            "differentials": [
                {"diagnosis": "Wound infection", "probability": 0.50},
                {"diagnosis": "Deep collection / abscess", "probability": 0.28},
                {"diagnosis": "Anastomotic leak", "probability": 0.14},
                {"diagnosis": "DVT / PE", "probability": 0.08},
            ],
            "symptom_weights": {
                "post-op fever": 0.82, "wound erythema": 0.75, "increasing pain": 0.85, "recent surgery": 0.99,
            },
            "vital_emphasis": ["hr", "temp", "bp_systolic"],
            "key_reasoning_points": [
                "Post-op fever day 1-3 = wind (atelectasis); day 3-5 = wound; day 5-7 = deep collection",
                "SIRS criteria met — treat as sepsis until proven otherwise",
                "CT abdomen with IV contrast to exclude intra-abdominal collection",
            ],
            "base_confidence": 0.70,
        },
    },

    # ── Template 14: Hypoglycaemia ────────────────────────────────────────────
    {
        "condition_id": "hypoglycaemia",
        "condition_name": "Severe hypoglycaemia — diabetes medication induced",
        "difficulty": "hard",
        "age_range": (55, 80),
        "vitals_ranges": {
            "hr": (88, 108), "bp_systolic": (128, 148), "bp_diastolic": (78, 92),
            "o2_sat": (96, 99), "rr": (14, 18), "temp": (36.5, 37.0),
        },
        "symptom_probabilities": {
            "confusion": 0.85, "diaphoresis": 0.82, "tremor": 0.78,
            "known diabetes": 0.99, "sulphonylurea or insulin use": 0.90,
            "pallor": 0.65, "aggression": 0.40,
        },
        "correct_esi": 2,
        "correct_department": "Emergency",
        "reasoning_template": (
            "Severe hypoglycaemia in a {age}-year-old {sex} with diabetes. "
            "Confusion + diaphoresis — immediate IV glucose required. ESI-2."
        ),
        "esi_partial_credit": {1: 0.6, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
        "department_options": ["Emergency", "General", "Neurology"],
        "department_scores": {"Emergency": 0.99, "General": 0.5, "Neurology": 0.3},
        "progression": {
            "description": "Hypoglycaemia — prolonged neuroglycopenia causes irreversible brain injury.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +3, "bp_systolic_delta": +3, "consciousness_delta": -0.15},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"consciousness": 0.3},
        },
        "resources_required": {
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Endocrinology",
        },
        "xai_base": {
            "primary_diagnosis": "Severe sulphonylurea-induced hypoglycaemia",
            "differentials": [
                {"diagnosis": "Sulphonylurea hypoglycaemia", "probability": 0.68},
                {"diagnosis": "Insulin overdose", "probability": 0.18},
                {"diagnosis": "Stroke mimicking hypoglycaemia", "probability": 0.08},
                {"diagnosis": "Sepsis with hypoglycaemia", "probability": 0.06},
            ],
            "symptom_weights": {
                "diaphoresis": 0.82, "confusion": 0.85, "diabetes + sulphonylurea": 0.95, "tremor": 0.78,
            },
            "vital_emphasis": ["consciousness_score"],
            "key_reasoning_points": [
                "Always check BGL before attributing confusion to dementia or stroke",
                "Sulphonylurea hypoglycaemia is prolonged — monitor for 24h after correction",
                "IV dextrose 10% preferred; avoid 50% bolus (vascular injury)",
            ],
            "base_confidence": 0.88,
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

    # ════════════════════════════════════════════════════════════════════════════
    # EASY CASES 007–018  (12 new easy patients)
    # ════════════════════════════════════════════════════════════════════════════

    _static(
        base={
            "id": "easy_007",
            "difficulty": "easy",
            "presentation": (
                "45-year-old male. Severe allergic reaction 15 minutes after eating shellfish. "
                "Urticaria, facial swelling, stridor. BP 88/56, HR 118, O2 sat 94%, RR 24."
            ),
            "vitals": {"bp": "88/56", "hr": 118, "o2_sat": 94, "rr": 24, "temp": 37.0},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Anaphylaxis with stridor and hypotension. Airway compromise imminent. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.05, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "General"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.7, "General": 0.01},
        },
        progression={
            "description": "Anaphylaxis — airway can close within minutes without epinephrine.",
            "per_timestep": {"o2_sat_delta": -3, "hr_delta": +5, "bp_systolic_delta": -8, "consciousness_delta": -0.15},
            "mortality_risk_per_step": 0.15,
            "critical_thresholds": {"o2_sat": 88, "bp_systolic": 70},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": True,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Emergency medicine",
        },
        xai={
            "primary_diagnosis": "Anaphylaxis",
            "differentials": [
                {"diagnosis": "Anaphylaxis", "probability": 0.92},
                {"diagnosis": "Angioedema (non-anaphylactic)", "probability": 0.05},
                {"diagnosis": "Vocal cord dysfunction", "probability": 0.03},
            ],
            "symptom_weights": {"stridor": 0.95, "urticaria": 0.85, "hypotension": 0.90, "facial swelling": 0.80},
            "vital_flags": ["BP 88/56 (hypotension — distributive shock)", "HR 118 (compensatory tachycardia)", "O2 94% (airway compromise)"],
            "confidence": 0.95,
            "key_reasoning_points": [
                "Stridor = upper airway involvement — IM epinephrine immediately",
                "Hypotension + urticaria after allergen exposure = anaphylaxis by definition",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_008",
            "difficulty": "easy",
            "presentation": (
                "28-year-old male. Witnessed seizure lasting 3 minutes, now post-ictal. "
                "No known epilepsy. BP 138/86, HR 102, O2 sat 96%, RR 16, temp 37.1°C."
            ),
            "vitals": {"bp": "138/86", "hr": 102, "o2_sat": 96, "rr": 16, "temp": 37.1},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "First unprovoked seizure, post-ictal, needs urgent workup. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "Neurology", "Resuscitation"],
            "department_scores": {"Emergency": 0.99, "Neurology": 0.7, "Resuscitation": 0.5},
        },
        progression={
            "description": "Post-ictal — risk of recurrent seizure or status epilepticus.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": -2, "bp_systolic_delta": -2, "consciousness_delta": 0.1},
            "mortality_risk_per_step": 0.02,
            "critical_thresholds": {"o2_sat": 90},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Neurology",
        },
        xai={
            "primary_diagnosis": "First unprovoked seizure",
            "differentials": [
                {"diagnosis": "Idiopathic epilepsy", "probability": 0.45},
                {"diagnosis": "Structural lesion (tumour/AVM)", "probability": 0.25},
                {"diagnosis": "Metabolic (hypoglycaemia, hyponatraemia)", "probability": 0.20},
                {"diagnosis": "Alcohol withdrawal", "probability": 0.10},
            ],
            "symptom_weights": {"witnessed tonic-clonic": 0.90, "post-ictal confusion": 0.85},
            "vital_flags": ["HR 102 (post-ictal tachycardia)"],
            "confidence": 0.80,
            "key_reasoning_points": [
                "First seizure always needs CT head to exclude structural cause",
                "Check glucose immediately — hypoglycaemia is treatable and common",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_009",
            "difficulty": "easy",
            "presentation": (
                "72-year-old female. Found on floor by family, unable to get up. "
                "Right hip pain, leg externally rotated and shortened. BP 142/88, HR 88, O2 sat 97%."
            ),
            "vitals": {"bp": "142/88", "hr": 88, "o2_sat": 97, "rr": 16, "temp": 36.6},
            "correct_esi": 2,
            "correct_department": "Orthopedics",
            "reasoning": "Classic hip fracture in elderly. Needs urgent reduction and surgery. ESI-2.",
            "esi_partial_credit": {1: 0.2, 2: 0.99, 3: 0.4, 4: 0.1, 5: 0.01},
            "department_options": ["Orthopedics", "Emergency", "Trauma"],
            "department_scores": {"Orthopedics": 0.99, "Emergency": 0.7, "Trauma": 0.8},
        },
        progression={
            "description": "Hip fracture — prolonged floor time risks pressure sores, DVT, and deconditioning.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 1, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.02,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Orthopedics",
        },
        xai={
            "primary_diagnosis": "Hip fracture (neck of femur)",
            "differentials": [
                {"diagnosis": "Neck of femur fracture", "probability": 0.88},
                {"diagnosis": "Intertrochanteric fracture", "probability": 0.10},
                {"diagnosis": "Hip dislocation", "probability": 0.02},
            ],
            "symptom_weights": {"external rotation": 0.90, "limb shortening": 0.88, "inability to weight-bear": 0.85},
            "vital_flags": [],
            "confidence": 0.90,
            "key_reasoning_points": [
                "External rotation + shortening = fracture until proven otherwise",
                "X-ray pelvis and hip; if negative and high suspicion → MRI",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_010",
            "difficulty": "easy",
            "presentation": (
                "5-year-old male. Ingested unknown quantity of paracetamol tablets (~20 tablets, 500mg each) "
                "approximately 2 hours ago. Currently asymptomatic. BP 96/60, HR 110, O2 sat 99%."
            ),
            "vitals": {"bp": "96/60", "hr": 110, "o2_sat": 99, "rr": 20, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Pediatrics",
            "reasoning": "Paediatric paracetamol overdose. Currently asymptomatic but at high toxicity risk. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Pediatrics", "Emergency", "Resuscitation"],
            "department_scores": {"Pediatrics": 0.99, "Emergency": 0.8, "Resuscitation": 0.5},
        },
        progression={
            "description": "Paracetamol overdose — liver failure develops 48-72 hrs later if untreated.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.04,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Pediatrics",
        },
        xai={
            "primary_diagnosis": "Paracetamol overdose",
            "differentials": [
                {"diagnosis": "Paracetamol toxicity", "probability": 0.95},
                {"diagnosis": "Mixed ingestion", "probability": 0.05},
            ],
            "symptom_weights": {"known ingestion": 0.99, "asymptomatic": 0.60},
            "vital_flags": ["HR 110 (mild tachycardia — anxiety/pain)"],
            "confidence": 0.95,
            "key_reasoning_points": [
                "Asymptomatic at 2h does NOT mean safe — hepatotoxicity peaks at 72h",
                "Urgent paracetamol level and Rumack-Matthew nomogram assessment",
                "N-acetylcysteine if level above treatment line",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_011",
            "difficulty": "easy",
            "presentation": (
                "19-year-old female. 8 weeks pregnant, sudden onset right lower quadrant pain, "
                "shoulder tip pain. BP 94/60, HR 124, O2 sat 98%, pale, diaphoretic."
            ),
            "vitals": {"bp": "94/60", "hr": 124, "o2_sat": 98, "rr": 22, "temp": 36.5},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Ruptured ectopic pregnancy with haemodynamic instability. Surgical emergency. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.05, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Obstetrics", "Emergency"],
            "department_scores": {"Resuscitation": 0.99, "Obstetrics": 0.7, "Emergency": 0.6},
        },
        progression={
            "description": "Ruptured ectopic — haemoperitoneum expanding, haemorrhagic shock imminent.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": +8, "bp_systolic_delta": -10, "consciousness_delta": -0.2},
            "mortality_risk_per_step": 0.14,
            "critical_thresholds": {"bp_systolic": 75, "hr": 150},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Obstetrics",
        },
        xai={
            "primary_diagnosis": "Ruptured ectopic pregnancy",
            "differentials": [
                {"diagnosis": "Ruptured ectopic pregnancy", "probability": 0.85},
                {"diagnosis": "Ovarian cyst rupture", "probability": 0.10},
                {"diagnosis": "Appendicitis with perforation", "probability": 0.05},
            ],
            "symptom_weights": {"shoulder tip pain": 0.88, "haemodynamic instability": 0.95, "pregnancy + RLQ pain": 0.90},
            "vital_flags": ["BP 94/60 (haemorrhagic shock)", "HR 124 (severe tachycardia)"],
            "confidence": 0.90,
            "key_reasoning_points": [
                "Shoulder tip pain = diaphragmatic irritation from haemoperitoneum",
                "Any pregnant woman with shock + abdominal pain = ectopic until proven otherwise",
                "Immediate theatre — do NOT delay for USS if haemodynamically unstable",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_012",
            "difficulty": "easy",
            "presentation": (
                "55-year-old male. Sudden onset worst headache of his life while at rest. "
                "Neck stiffness, photophobia. BP 168/98, HR 82, O2 sat 98%, temp 37.2°C."
            ),
            "vitals": {"bp": "168/98", "hr": 82, "o2_sat": 98, "rr": 16, "temp": 37.2},
            "correct_esi": 2,
            "correct_department": "Neurology",
            "reasoning": "Thunderclap headache + meningism. Subarachnoid haemorrhage until proven otherwise. ESI-2.",
            "esi_partial_credit": {1: 0.6, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
            "department_options": ["Neurology", "Resuscitation", "Emergency"],
            "department_scores": {"Neurology": 0.99, "Resuscitation": 0.8, "Emergency": 0.7},
        },
        progression={
            "description": "SAH — rebleeding risk highest in first 24h, can be fatal.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 2, "bp_systolic_delta": +4, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.07,
            "critical_thresholds": {"consciousness": 0.5},
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
                {"diagnosis": "Subarachnoid haemorrhage", "probability": 0.70},
                {"diagnosis": "Migraine with aura", "probability": 0.15},
                {"diagnosis": "Meningitis", "probability": 0.10},
                {"diagnosis": "Hypertensive emergency", "probability": 0.05},
            ],
            "symptom_weights": {"thunderclap onset": 0.92, "neck stiffness": 0.80, "worst headache ever": 0.88},
            "vital_flags": ["BP 168/98 (hypertension — Cushing response)"],
            "confidence": 0.82,
            "key_reasoning_points": [
                "Thunderclap headache = SAH until CT + LP negative",
                "CT head first — if negative at 6h, LP for xanthochromia",
                "Do NOT give analgesia before assessment if conscious level uncertain",
            ],
        },
    ),

    # ════════════════════════════════════════════════════════════════════════════
    # MEDIUM CASES 007–018  (12 new medium patients)
    # ════════════════════════════════════════════════════════════════════════════

    _static(
        base={
            "id": "medium_007",
            "difficulty": "medium",
            "presentation": (
                "62-year-old male with known COPD. Increasing breathlessness over 3 days, "
                "productive cough with green sputum, fever 38.4°C. BP 138/86, HR 96, O2 sat 88% on air, RR 26."
            ),
            "vitals": {"bp": "138/86", "hr": 96, "o2_sat": 88, "rr": 26, "temp": 38.4},
            "correct_esi": 2,
            "correct_department": "Pulmonology",
            "reasoning": "COPD exacerbation with pneumonia. O2 sat 88% and RR 26 — emergent. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Pulmonology", "Emergency", "General"],
            "department_scores": {"Pulmonology": 0.99, "Emergency": 0.8, "General": 0.2},
        },
        progression={
            "description": "COPD exacerbation — hypercapnic respiratory failure can develop rapidly.",
            "per_timestep": {"o2_sat_delta": -2, "hr_delta": +3, "bp_systolic_delta": 0, "consciousness_delta": -0.05},
            "mortality_risk_per_step": 0.06,
            "critical_thresholds": {"o2_sat": 82, "rr": 30},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Pulmonology",
        },
        xai={
            "primary_diagnosis": "COPD exacerbation with community-acquired pneumonia",
            "differentials": [
                {"diagnosis": "COPD exacerbation + pneumonia", "probability": 0.60},
                {"diagnosis": "COPD exacerbation alone", "probability": 0.25},
                {"diagnosis": "Pulmonary embolism", "probability": 0.10},
                {"diagnosis": "Cardiac pulmonary oedema", "probability": 0.05},
            ],
            "symptom_weights": {"productive cough": 0.75, "fever": 0.70, "O2 sat 88%": 0.90},
            "vital_flags": ["O2 88% (severe hypoxia)", "RR 26 (tachypnoea)", "Fever 38.4°C"],
            "confidence": 0.72,
            "key_reasoning_points": [
                "COPD + infection: CXR, ABG, and controlled oxygen (target 88-92%) are critical",
                "Watch for hypercapnia — high-flow O2 can suppress respiratory drive",
                "Early NIV if pH <7.35 and PaCO2 elevated",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_008",
            "difficulty": "medium",
            "presentation": (
                "38-year-old female. 2 days post appendectomy. Fever 38.9°C, wound redness and swelling, "
                "increasing abdominal pain. BP 118/74, HR 104, O2 sat 98%, RR 18."
            ),
            "vitals": {"bp": "118/74", "hr": 104, "o2_sat": 98, "rr": 18, "temp": 38.9},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Post-op infection — wound dehiscence vs intra-abdominal collection. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "General", "Gastroenterology"],
            "department_scores": {"Emergency": 0.99, "General": 0.7, "Gastroenterology": 0.3},
        },
        progression={
            "description": "Post-op sepsis — can progress to septic shock if collection missed.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +4, "bp_systolic_delta": -4, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"bp_systolic": 90},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "General surgery",
        },
        xai={
            "primary_diagnosis": "Post-operative wound infection / intra-abdominal collection",
            "differentials": [
                {"diagnosis": "Wound infection", "probability": 0.50},
                {"diagnosis": "Intra-abdominal abscess", "probability": 0.30},
                {"diagnosis": "Anastomotic leak", "probability": 0.15},
                {"diagnosis": "DVT / PE", "probability": 0.05},
            ],
            "symptom_weights": {"post-op fever": 0.80, "wound erythema": 0.75, "worsening pain": 0.85},
            "vital_flags": ["HR 104 (tachycardia — SIRS criterion)", "Fever 38.9°C"],
            "confidence": 0.70,
            "key_reasoning_points": [
                "Post-op fever day 2: wound and collection are top differential",
                "CT abdomen/pelvis with contrast to exclude collection",
                "SIRS criteria met — treat as early sepsis",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_009",
            "difficulty": "medium",
            "presentation": (
                "26-year-old female. Known type 1 diabetes. Nausea, vomiting, abdominal pain for 12 hours. "
                "Blood glucose 28 mmol/L at home. BP 104/68, HR 116, O2 sat 98%, RR 24, temp 37.0°C."
            ),
            "vitals": {"bp": "104/68", "hr": 116, "o2_sat": 98, "rr": 24, "temp": 37.0},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "DKA — tachycardia, hyperglycaemia, Kussmaul breathing pattern. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "General", "Resuscitation"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Resuscitation": 0.6},
        },
        progression={
            "description": "DKA — cerebral oedema and hypokalaemia can cause cardiac arrest.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +3, "bp_systolic_delta": -4, "consciousness_delta": -0.08},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"bp_systolic": 85, "consciousness": 0.5},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Endocrinology",
        },
        xai={
            "primary_diagnosis": "Diabetic ketoacidosis (DKA)",
            "differentials": [
                {"diagnosis": "DKA", "probability": 0.85},
                {"diagnosis": "Hyperglycaemic hyperosmolar state", "probability": 0.08},
                {"diagnosis": "Gastroenteritis with incidental hyperglycaemia", "probability": 0.04},
                {"diagnosis": "Sepsis-triggered DKA", "probability": 0.03},
            ],
            "symptom_weights": {"glucose 28 mmol/L": 0.90, "vomiting": 0.70, "abdominal pain": 0.65, "tachypnoea": 0.80},
            "vital_flags": ["HR 116 (tachycardia — dehydration)", "RR 24 (Kussmaul breathing — metabolic acidosis)"],
            "confidence": 0.88,
            "key_reasoning_points": [
                "RR 24 in a diabetic with vomiting = Kussmaul respiration = DKA",
                "Check VBG, ketones, electrolytes — potassium is critical",
                "Fixed-rate insulin infusion + cautious IV fluids",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_010",
            "difficulty": "medium",
            "presentation": (
                "14-year-old male. Acute-onset scrotal pain and swelling for 3 hours. "
                "Nausea, vomiting. BP 118/72, HR 98, O2 sat 99%, temp 37.3°C. Testicle high-riding."
            ),
            "vitals": {"bp": "118/72", "hr": 98, "o2_sat": 99, "rr": 16, "temp": 37.3},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Testicular torsion — 6-hour window for salvage. Surgical emergency. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "Orthopedics", "General"],
            "department_scores": {"Emergency": 0.99, "Orthopedics": 0.1, "General": 0.4},
        },
        progression={
            "description": "Testicular torsion — testicular infarction after 6 hours is irreversible.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.01,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Urology",
        },
        xai={
            "primary_diagnosis": "Testicular torsion",
            "differentials": [
                {"diagnosis": "Testicular torsion", "probability": 0.72},
                {"diagnosis": "Epididymo-orchitis", "probability": 0.18},
                {"diagnosis": "Torsion of appendix testis", "probability": 0.08},
                {"diagnosis": "Inguinal hernia", "probability": 0.02},
            ],
            "symptom_weights": {"acute scrotal pain": 0.85, "high-riding testicle": 0.92, "nausea/vomiting": 0.70},
            "vital_flags": [],
            "confidence": 0.80,
            "key_reasoning_points": [
                "High-riding testicle + absent cremasteric reflex = torsion until proven otherwise",
                "Do NOT delay theatre for USS if clinical suspicion is high",
                "6-hour window: >90% salvage rate; 24h: <10%",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_011",
            "difficulty": "medium",
            "presentation": (
                "48-year-old female. Severe right upper quadrant pain radiating to right shoulder for 4 hours. "
                "Nausea, fever 38.6°C. BP 132/82, HR 102, O2 sat 98%, RR 18. Murphy's sign positive."
            ),
            "vitals": {"bp": "132/82", "hr": 102, "o2_sat": 98, "rr": 18, "temp": 38.6},
            "correct_esi": 2,
            "correct_department": "Gastroenterology",
            "reasoning": "Acute cholecystitis — fever + Murphy's sign + RUQ pain. ESI-2.",
            "esi_partial_credit": {1: 0.2, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
            "department_options": ["Gastroenterology", "Emergency", "General"],
            "department_scores": {"Gastroenterology": 0.99, "Emergency": 0.75, "General": 0.6},
        },
        progression={
            "description": "Cholecystitis — perforation or ascending cholangitis can develop.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +3, "bp_systolic_delta": -2, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.03,
            "critical_thresholds": {"bp_systolic": 90},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Gastroenterology",
        },
        xai={
            "primary_diagnosis": "Acute cholecystitis",
            "differentials": [
                {"diagnosis": "Acute cholecystitis", "probability": 0.65},
                {"diagnosis": "Biliary colic", "probability": 0.15},
                {"diagnosis": "Ascending cholangitis", "probability": 0.12},
                {"diagnosis": "Hepatitis", "probability": 0.05},
                {"diagnosis": "Right-sided pneumonia", "probability": 0.03},
            ],
            "symptom_weights": {"Murphy's sign": 0.88, "RUQ pain": 0.75, "fever": 0.70, "shoulder radiation": 0.65},
            "vital_flags": ["HR 102 (tachycardia)", "Fever 38.6°C"],
            "confidence": 0.75,
            "key_reasoning_points": [
                "Murphy's sign + fever + RUQ = cholecystitis by Tokyo criteria",
                "USS is first line — CT if USS inconclusive",
                "Watch for Charcot's triad (RUQ pain, fever, jaundice) = ascending cholangitis",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_012",
            "difficulty": "medium",
            "presentation": (
                "33-year-old male. Known schizophrenia, off medication 2 weeks. Brought by police, "
                "agitated, threatening harm to others, visual hallucinations. BP 144/90, HR 106, O2 sat 98%."
            ),
            "vitals": {"bp": "144/90", "hr": 106, "o2_sat": 98, "rr": 18, "temp": 37.0},
            "correct_esi": 2,
            "correct_department": "Psychiatry",
            "reasoning": "Acute psychosis with risk to others. Needs urgent psychiatric assessment and safe environment. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
            "department_options": ["Psychiatry", "Emergency", "General"],
            "department_scores": {"Psychiatry": 0.99, "Emergency": 0.75, "General": 0.1},
        },
        progression={
            "description": "Acute psychosis — escalating agitation risks harm to staff and patient.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +2, "bp_systolic_delta": +2, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.01,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Psychiatry",
        },
        xai={
            "primary_diagnosis": "Acute schizophrenic relapse",
            "differentials": [
                {"diagnosis": "Schizophrenic relapse (medication non-compliance)", "probability": 0.65},
                {"diagnosis": "Substance-induced psychosis", "probability": 0.20},
                {"diagnosis": "Organic psychosis (encephalitis, metabolic)", "probability": 0.10},
                {"diagnosis": "Bipolar disorder with psychotic features", "probability": 0.05},
            ],
            "symptom_weights": {"visual hallucinations": 0.80, "known schizophrenia": 0.85, "medication non-compliance": 0.90},
            "vital_flags": ["HR 106 (agitation-related tachycardia)"],
            "confidence": 0.72,
            "key_reasoning_points": [
                "Always exclude organic cause — ECG and bloods before antipsychotics",
                "Visual hallucinations in a known schizophrenic suggest relapse, but screen for delirium",
                "Safe environment first — risk to staff if untreated",
            ],
        },
    ),

    # ════════════════════════════════════════════════════════════════════════════
    # HARD CASES 007–018  (12 new hard patients)
    # ════════════════════════════════════════════════════════════════════════════

    _static(
        base={
            "id": "hard_007",
            "difficulty": "hard",
            "presentation": (
                "41-year-old female. 6 days post laparoscopic cholecystectomy. Mild jaundice noticed today. "
                "RUQ discomfort, temp 37.6°C, dark urine. BP 124/78, HR 88, O2 sat 99%."
            ),
            "vitals": {"bp": "124/78", "hr": 88, "o2_sat": 99, "rr": 15, "temp": 37.6},
            "correct_esi": 3,
            "correct_department": "Gastroenterology",
            "reasoning": "Post-cholecystectomy jaundice — bile duct injury or retained stone. Needs urgent ERCP/imaging. ESI-3.",
            "esi_partial_credit": {1: 0.1, 2: 0.5, 3: 0.99, 4: 0.3, 5: 0.01},
            "department_options": ["Gastroenterology", "Emergency", "General"],
            "department_scores": {"Gastroenterology": 0.99, "Emergency": 0.6, "General": 0.5},
        },
        progression={
            "description": "Bile duct injury — sepsis from biloma can develop over hours.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +2, "bp_systolic_delta": -1, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.02,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Gastroenterology",
        },
        xai={
            "primary_diagnosis": "Post-cholecystectomy bile duct injury or retained stone",
            "differentials": [
                {"diagnosis": "Retained common bile duct stone", "probability": 0.45},
                {"diagnosis": "Bile duct injury (clip or division)", "probability": 0.30},
                {"diagnosis": "Biloma / bile leak", "probability": 0.18},
                {"diagnosis": "Post-op hepatitis", "probability": 0.07},
            ],
            "symptom_weights": {"post-op jaundice": 0.88, "dark urine": 0.75, "RUQ pain": 0.65},
            "vital_flags": ["Low-grade fever 37.6°C (early biliary sepsis)"],
            "confidence": 0.60,
            "key_reasoning_points": [
                "Jaundice within 1 week of cholecystectomy = bile duct complication until proven otherwise",
                "MRCP or CT with contrast to identify leak or obstruction",
                "Early hepatobiliary surgery referral",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_008",
            "difficulty": "hard",
            "presentation": (
                "52-year-old male. Intermittent chest tightness for 3 weeks, now at rest. "
                "ECG shows ST depression in V4-V6. BP 158/94, HR 78, O2 sat 97%, no diaphoresis."
            ),
            "vitals": {"bp": "158/94", "hr": 78, "o2_sat": 97, "rr": 16, "temp": 36.9},
            "correct_esi": 2,
            "correct_department": "Cardiology",
            "reasoning": "NSTEMI — ST depression at rest with chest tightness. High-risk ACS. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Cardiology", "Emergency", "Resuscitation"],
            "department_scores": {"Cardiology": 0.99, "Emergency": 0.8, "Resuscitation": 0.6},
        },
        progression={
            "description": "NSTEMI — risk of progression to STEMI or VF without antiplatelet therapy.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": +4, "bp_systolic_delta": -3, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.06,
            "critical_thresholds": {"o2_sat": 90, "bp_systolic": 90},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Cardiology",
        },
        xai={
            "primary_diagnosis": "NSTEMI (non-ST elevation MI)",
            "differentials": [
                {"diagnosis": "NSTEMI", "probability": 0.65},
                {"diagnosis": "Unstable angina", "probability": 0.20},
                {"diagnosis": "Vasospastic angina", "probability": 0.08},
                {"diagnosis": "PE", "probability": 0.05},
                {"diagnosis": "GERD", "probability": 0.02},
            ],
            "symptom_weights": {"ST depression V4-V6": 0.88, "rest pain": 0.85, "3-week history": 0.70},
            "vital_flags": ["BP 158/94 (hypertension — increases cardiac work)"],
            "confidence": 0.72,
            "key_reasoning_points": [
                "ST depression at rest = high-risk ACS — admit for troponin serial measurement",
                "No diaphoresis or haemodynamic instability does NOT exclude NSTEMI",
                "GRACE score to stratify risk and timing of angiography",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_009",
            "difficulty": "hard",
            "presentation": (
                "67-year-old male. Type 2 diabetes on metformin and gliclazide. Found confused at home "
                "by wife. Sweating, trembling. BP 138/84, HR 92, O2 sat 98%, blood glucose 2.1 mmol/L."
            ),
            "vitals": {"bp": "138/84", "hr": 92, "o2_sat": 98, "rr": 16, "temp": 36.7},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Severe hypoglycaemia with altered consciousness. Needs IV glucose immediately. ESI-2.",
            "esi_partial_credit": {1: 0.6, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "General", "Neurology"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Neurology": 0.3},
        },
        progression={
            "description": "Severe hypoglycaemia — prolonged neuroglycopenia causes irreversible brain damage.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +3, "bp_systolic_delta": +3, "consciousness_delta": -0.15},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"consciousness": 0.3},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Endocrinology",
        },
        xai={
            "primary_diagnosis": "Severe hypoglycaemia",
            "differentials": [
                {"diagnosis": "Sulphonylurea-induced hypoglycaemia", "probability": 0.70},
                {"diagnosis": "Insulin overdose", "probability": 0.15},
                {"diagnosis": "Stroke with incidental low glucose", "probability": 0.10},
                {"diagnosis": "Sepsis with hypoglycaemia", "probability": 0.05},
            ],
            "symptom_weights": {"glucose 2.1 mmol/L": 0.99, "confusion": 0.85, "diaphoresis": 0.80, "tremor": 0.75},
            "vital_flags": ["BGL 2.1 mmol/L (severe hypoglycaemia)"],
            "confidence": 0.90,
            "key_reasoning_points": [
                "Sulphonylurea hypoglycaemia can be prolonged — monitor for 24h after correction",
                "IV dextrose 10% preferred over 50% bolus to avoid rebound and vascular injury",
                "Do not assume confusion is from dementia — always check BGL first",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_010",
            "difficulty": "hard",
            "presentation": (
                "29-year-old female. 34 weeks pregnant. Called ambulance for decreased fetal movements "
                "for 2 days. Mild lower abdominal cramping. BP 118/72, HR 86, O2 sat 99%, no bleeding."
            ),
            "vitals": {"bp": "118/72", "hr": 86, "o2_sat": 99, "rr": 16, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Obstetrics",
            "reasoning": "Decreased fetal movements at 34 weeks — fetal distress must be excluded urgently. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.4, 4: 0.1, 5: 0.01},
            "department_options": ["Obstetrics", "Emergency", "General"],
            "department_scores": {"Obstetrics": 0.99, "Emergency": 0.6, "General": 0.2},
        },
        progression={
            "description": "Fetal distress — placental abruption or cord compromise can lead to stillbirth.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.04,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Obstetrics",
        },
        xai={
            "primary_diagnosis": "Decreased fetal movements — fetal distress",
            "differentials": [
                {"diagnosis": "Fetal compromise (placental insufficiency)", "probability": 0.40},
                {"diagnosis": "Normal fetal movement variation", "probability": 0.30},
                {"diagnosis": "Placental abruption (early)", "probability": 0.20},
                {"diagnosis": "Cord entanglement", "probability": 0.10},
            ],
            "symptom_weights": {"decreased FM 2 days": 0.85, "mild cramps": 0.55},
            "vital_flags": [],
            "confidence": 0.55,
            "key_reasoning_points": [
                "Decreased fetal movements >12h must be investigated — CTG immediately",
                "Normal maternal vitals do NOT exclude fetal compromise",
                "USS biophysical profile if CTG non-reassuring",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_011",
            "difficulty": "hard",
            "presentation": (
                "71-year-old male. Gradual onset confusion and drowsiness over 3 days. "
                "On lithium for bipolar disorder. BP 108/66, HR 58, O2 sat 97%, dry mucous membranes, coarse tremor."
            ),
            "vitals": {"bp": "108/66", "hr": 58, "o2_sat": 97, "rr": 14, "temp": 37.0},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Lithium toxicity — classic triad: confusion, coarse tremor, bradycardia with dehydration. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "Psychiatry", "General", "Neurology"],
            "department_scores": {"Emergency": 0.99, "Psychiatry": 0.3, "General": 0.4, "Neurology": 0.5},
        },
        progression={
            "description": "Lithium toxicity — seizures, arrhythmias, and permanent cerebellar damage if untreated.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": -2, "bp_systolic_delta": -3, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.06,
            "critical_thresholds": {"hr": 45, "bp_systolic": 85},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Toxicology",
        },
        xai={
            "primary_diagnosis": "Lithium toxicity",
            "differentials": [
                {"diagnosis": "Lithium toxicity", "probability": 0.65},
                {"diagnosis": "Hyponatraemia (lithium-induced SIADH)", "probability": 0.15},
                {"diagnosis": "Delirium from unrelated cause", "probability": 0.12},
                {"diagnosis": "Bipolar relapse", "probability": 0.08},
            ],
            "symptom_weights": {"coarse tremor": 0.88, "lithium use": 0.95, "confusion": 0.80, "bradycardia": 0.70},
            "vital_flags": ["BP 108/66 (hypotension)", "HR 58 (bradycardia — lithium toxicity)", "Dry mucous membranes (dehydration elevates lithium levels)"],
            "confidence": 0.65,
            "key_reasoning_points": [
                "Dehydration raises lithium levels — any illness causing fluid loss is a trigger",
                "Coarse tremor (not fine) differentiates lithium toxicity from therapeutic side effect",
                "Urgent lithium level, U&E, ECG — consider haemodialysis if severe",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_012",
            "difficulty": "hard",
            "presentation": (
                "23-year-old male. Marathon runner, collapsed 500m from finish line. Confused, hot, "
                "not sweating. Temp 41.2°C, BP 98/62, HR 138, O2 sat 94%, RR 28."
            ),
            "vitals": {"bp": "98/62", "hr": 138, "o2_sat": 94, "rr": 28, "temp": 41.2},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Exertional heat stroke — temp >40°C + CNS dysfunction. Core cooling is a time-critical intervention. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.05, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "Trauma"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.65, "Trauma": 0.3},
        },
        progression={
            "description": "Heat stroke — MODS, rhabdomyolysis, DIC develop within minutes without cooling.",
            "per_timestep": {"o2_sat_delta": -2, "hr_delta": +5, "bp_systolic_delta": -6, "consciousness_delta": -0.2},
            "mortality_risk_per_step": 0.13,
            "critical_thresholds": {"consciousness": 0.3, "bp_systolic": 80},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Emergency medicine",
        },
        xai={
            "primary_diagnosis": "Exertional heat stroke",
            "differentials": [
                {"diagnosis": "Exertional heat stroke", "probability": 0.80},
                {"diagnosis": "Serotonin syndrome", "probability": 0.08},
                {"diagnosis": "Sepsis with hyperthermia", "probability": 0.07},
                {"diagnosis": "Exercise-associated collapse (benign)", "probability": 0.05},
            ],
            "symptom_weights": {"temp 41.2°C": 0.95, "anhidrosis": 0.88, "confusion": 0.85, "exertion": 0.90},
            "vital_flags": ["Temp 41.2°C (severe hyperthermia)", "HR 138 (shock)", "O2 94% (early respiratory compromise)"],
            "confidence": 0.85,
            "key_reasoning_points": [
                "Heat stroke = temp >40°C + CNS dysfunction — anhidrosis distinguishes from heat exhaustion",
                "Cold water immersion is most effective — target <39°C within 30 minutes",
                "Watch for rhabdomyolysis, AKI, coagulopathy — aggressive IV fluids",
            ],
        },
    ),

    # ════════════════════════════════════════════════════════════════════════════
    # EASY CASES 013–018
    # ════════════════════════════════════════════════════════════════════════════

    _static(
        base={
            "id": "easy_013",
            "difficulty": "easy",
            "presentation": (
                "3-year-old female. Swallowed a small button battery 1 hour ago, confirmed by parents. "
                "Currently crying but alert. BP 90/58, HR 128, O2 sat 99%, RR 24."
            ),
            "vitals": {"bp": "90/58", "hr": 128, "o2_sat": 99, "rr": 24, "temp": 36.9},
            "correct_esi": 2,
            "correct_department": "Pediatrics",
            "reasoning": "Button battery ingestion — oesophageal lodgement causes liquefactive necrosis within 2 hours. Urgent endoscopy. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.2, 4: 0.01, 5: 0.01},
            "department_options": ["Pediatrics", "Emergency", "Gastroenterology"],
            "department_scores": {"Pediatrics": 0.99, "Emergency": 0.8, "Gastroenterology": 0.6},
        },
        progression={
            "description": "Button battery — tissue necrosis progresses rapidly, perforation within hours.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.06,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Pediatrics",
        },
        xai={
            "primary_diagnosis": "Button battery ingestion",
            "differentials": [
                {"diagnosis": "Button battery ingestion", "probability": 0.95},
                {"diagnosis": "Coin ingestion", "probability": 0.05},
            ],
            "symptom_weights": {"witnessed ingestion": 0.99, "age <5": 0.80},
            "vital_flags": [],
            "confidence": 0.98,
            "key_reasoning_points": [
                "Button battery in oesophagus = endoscopic emergency within 2 hours",
                "Do NOT induce vomiting — honey can be given en route to hospital (age >1yr)",
                "CXR to locate — double halo sign confirms button battery vs coin",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_014",
            "difficulty": "easy",
            "presentation": (
                "60-year-old male. Sudden painless loss of vision in right eye, like 'curtain coming down'. "
                "Onset 2 hours ago. BP 152/94, HR 76, O2 sat 98%, no headache."
            ),
            "vitals": {"bp": "152/94", "hr": 76, "o2_sat": 98, "rr": 15, "temp": 36.7},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Retinal detachment or central retinal artery occlusion — sudden vision loss is ophthalmic emergency. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "Neurology", "General"],
            "department_scores": {"Emergency": 0.99, "Neurology": 0.6, "General": 0.2},
        },
        progression={
            "description": "CRAO — retinal ischaemia is irreversible beyond 90 minutes.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.01,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Ophthalmology",
        },
        xai={
            "primary_diagnosis": "Central retinal artery occlusion or retinal detachment",
            "differentials": [
                {"diagnosis": "Central retinal artery occlusion", "probability": 0.50},
                {"diagnosis": "Retinal detachment", "probability": 0.35},
                {"diagnosis": "Vitreous haemorrhage", "probability": 0.10},
                {"diagnosis": "Transient ischaemic attack (amaurosis fugax)", "probability": 0.05},
            ],
            "symptom_weights": {"painless sudden vision loss": 0.92, "curtain-like onset": 0.85},
            "vital_flags": ["BP 152/94 (hypertension — risk factor for CRAO)"],
            "confidence": 0.75,
            "key_reasoning_points": [
                "Painless sudden monocular vision loss = ophthalmic emergency",
                "CRAO: 90-minute treatment window — ocular massage and carbogen therapy",
                "Retinal detachment: urgent ophthalmology review, no pressure on eye",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_015",
            "difficulty": "easy",
            "presentation": (
                "50-year-old male. Brought by ambulance after being found unresponsive. "
                "Empty vodka bottles nearby. GCS 8, vomit in airway, snoring respirations. "
                "BP 102/64, HR 58, O2 sat 88%, RR 8, temp 35.2°C."
            ),
            "vitals": {"bp": "102/64", "hr": 58, "o2_sat": 88, "rr": 8, "temp": 35.2},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Acute alcohol intoxication with airway compromise, O2 88%, RR 8. Immediate airway management. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.05, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "General"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.6, "General": 0.01},
        },
        progression={
            "description": "Airway obstruction — aspiration and hypoxic arrest imminent without intervention.",
            "per_timestep": {"o2_sat_delta": -4, "hr_delta": -4, "bp_systolic_delta": -6, "consciousness_delta": -0.2},
            "mortality_risk_per_step": 0.18,
            "critical_thresholds": {"o2_sat": 82, "hr": 40, "rr": 5},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": True,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Emergency medicine",
        },
        xai={
            "primary_diagnosis": "Acute alcohol poisoning with airway compromise",
            "differentials": [
                {"diagnosis": "Acute alcohol toxicity", "probability": 0.70},
                {"diagnosis": "Mixed drug and alcohol ingestion", "probability": 0.15},
                {"diagnosis": "Subdural haematoma (fall)", "probability": 0.10},
                {"diagnosis": "Hypoglycaemia", "probability": 0.05},
            ],
            "symptom_weights": {"GCS 8": 0.90, "RR 8": 0.95, "O2 88%": 0.92, "alcohol context": 0.80},
            "vital_flags": ["O2 88% (critical hypoxia)", "RR 8 (respiratory depression)", "Temp 35.2°C (hypothermia)"],
            "confidence": 0.78,
            "key_reasoning_points": [
                "GCS <8 = cannot protect airway — intubate",
                "Always check glucose — hypoglycaemia mimics and complicates alcohol intoxication",
                "CT head to exclude subdural if any history of fall",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_016",
            "difficulty": "easy",
            "presentation": (
                "42-year-old female. Bee sting 10 minutes ago. Local swelling at sting site only. "
                "No urticaria, no throat tightness, no dyspnoea. BP 122/78, HR 82, O2 sat 99%."
            ),
            "vitals": {"bp": "122/78", "hr": 82, "o2_sat": 99, "rr": 15, "temp": 36.8},
            "correct_esi": 4,
            "correct_department": "General",
            "reasoning": "Local reaction to bee sting only — no systemic allergic features. ESI-4, one resource (antihistamine).",
            "esi_partial_credit": {1: 0.01, 2: 0.01, 3: 0.3, 4: 0.99, 5: 0.5},
            "department_options": ["General", "Emergency", "Resuscitation"],
            "department_scores": {"General": 0.99, "Emergency": 0.5, "Resuscitation": 0.01},
        },
        progression={
            "description": "Local reaction — monitor for 30 minutes for delayed systemic response.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.0,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "General",
        },
        xai={
            "primary_diagnosis": "Local allergic reaction to bee sting",
            "differentials": [
                {"diagnosis": "Local allergic reaction", "probability": 0.90},
                {"diagnosis": "Early anaphylaxis (not yet systemic)", "probability": 0.10},
            ],
            "symptom_weights": {"local swelling only": 0.90, "normal vitals": 0.85, "no systemic symptoms": 0.92},
            "vital_flags": [],
            "confidence": 0.90,
            "key_reasoning_points": [
                "Local reaction confined to sting site = ESI-4",
                "Observe 30 minutes for delayed systemic response",
                "Oral antihistamine + topical steroid sufficient",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_017",
            "difficulty": "easy",
            "presentation": (
                "78-year-old male. Fell from standing height in bathroom. Right wrist pain and deformity. "
                "On warfarin for AF. BP 146/88, HR 74, O2 sat 97%, GCS 15, no head strike."
            ),
            "vitals": {"bp": "146/88", "hr": 74, "o2_sat": 97, "rr": 16, "temp": 36.6},
            "correct_esi": 3,
            "correct_department": "Orthopedics",
            "reasoning": "Distal radius fracture, anticoagulated — needs X-ray, INR, fracture management. ESI-3.",
            "esi_partial_credit": {1: 0.1, 2: 0.4, 3: 0.99, 4: 0.4, 5: 0.01},
            "department_options": ["Orthopedics", "Emergency", "General"],
            "department_scores": {"Orthopedics": 0.99, "Emergency": 0.7, "General": 0.3},
        },
        progression={
            "description": "Stable fracture — anticoagulation warrants INR check before manipulation.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.0,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Orthopedics",
        },
        xai={
            "primary_diagnosis": "Distal radius fracture (Colles' fracture)",
            "differentials": [
                {"diagnosis": "Distal radius fracture", "probability": 0.88},
                {"diagnosis": "Scaphoid fracture", "probability": 0.08},
                {"diagnosis": "Soft tissue injury", "probability": 0.04},
            ],
            "symptom_weights": {"wrist deformity": 0.90, "fall on outstretched hand": 0.85},
            "vital_flags": [],
            "confidence": 0.90,
            "key_reasoning_points": [
                "Anticoagulated patient: check INR before manipulation or theatre",
                "X-ray wrist AP and lateral — check for scaphoid tenderness too",
                "Colles' fracture in elderly: consider underlying osteoporosis",
            ],
        },
    ),

    _static(
        base={
            "id": "easy_018",
            "difficulty": "easy",
            "presentation": (
                "25-year-old male. Laceration to palm of right hand from broken glass, 4cm wound, "
                "bleeding controlled with direct pressure. Moves all fingers, sensation intact. "
                "BP 118/74, HR 78, O2 sat 99%."
            ),
            "vitals": {"bp": "118/74", "hr": 78, "o2_sat": 99, "rr": 15, "temp": 36.7},
            "correct_esi": 4,
            "correct_department": "General",
            "reasoning": "Simple laceration, neurovascular intact, bleeding controlled. ESI-4 — needs wound closure only.",
            "esi_partial_credit": {1: 0.01, 2: 0.01, 3: 0.4, 4: 0.99, 5: 0.3},
            "department_options": ["General", "Emergency", "Orthopedics"],
            "department_scores": {"General": 0.99, "Emergency": 0.6, "Orthopedics": 0.3},
        },
        progression={
            "description": "Stable laceration — no deterioration expected.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.0,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": False, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "General",
        },
        xai={
            "primary_diagnosis": "Simple hand laceration",
            "differentials": [
                {"diagnosis": "Simple laceration", "probability": 0.92},
                {"diagnosis": "Tendon injury (occult)", "probability": 0.05},
                {"diagnosis": "Foreign body retention", "probability": 0.03},
            ],
            "symptom_weights": {"normal sensation": 0.85, "full finger movement": 0.88, "bleeding controlled": 0.80},
            "vital_flags": [],
            "confidence": 0.92,
            "key_reasoning_points": [
                "Full neurovascular examination mandatory before closure",
                "Glass injuries: X-ray to exclude retained foreign body",
                "Tetanus status check; irrigate wound thoroughly before suturing",
            ],
        },
    ),

    # ════════════════════════════════════════════════════════════════════════════
    # MEDIUM CASES 013–018
    # ════════════════════════════════════════════════════════════════════════════

    _static(
        base={
            "id": "medium_013",
            "difficulty": "medium",
            "presentation": (
                "55-year-old male. Sudden onset severe tearing chest pain radiating to the back. "
                "History of Marfan syndrome. BP right arm 168/96, left arm 132/80. HR 98, O2 sat 96%."
            ),
            "vitals": {"bp": "168/96", "hr": 98, "o2_sat": 96, "rr": 18, "temp": 36.8},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Aortic dissection — BP differential 36mmHg between arms + Marfan syndrome + tearing pain. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.1, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Cardiology", "Emergency"],
            "department_scores": {"Resuscitation": 0.99, "Cardiology": 0.6, "Emergency": 0.7},
        },
        progression={
            "description": "Aortic dissection — rupture or coronary involvement can cause rapid death.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": +4, "bp_systolic_delta": -6, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.12,
            "critical_thresholds": {"bp_systolic": 80, "consciousness": 0.5},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Cardiothoracic surgery",
        },
        xai={
            "primary_diagnosis": "Aortic dissection (Type A)",
            "differentials": [
                {"diagnosis": "Aortic dissection", "probability": 0.85},
                {"diagnosis": "STEMI", "probability": 0.08},
                {"diagnosis": "PE", "probability": 0.05},
                {"diagnosis": "Musculoskeletal pain", "probability": 0.02},
            ],
            "symptom_weights": {"BP arm differential": 0.95, "tearing character": 0.88, "Marfan syndrome": 0.90},
            "vital_flags": ["BP differential 36 mmHg (>20 = highly suspicious for dissection)"],
            "confidence": 0.88,
            "key_reasoning_points": [
                "BP differential >20mmHg between arms = aortic dissection until CT proves otherwise",
                "Do NOT anticoagulate before excluding dissection",
                "Target systolic BP <120 with IV labetalol/esmolol",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_014",
            "difficulty": "medium",
            "presentation": (
                "68-year-old female. 3-day history of confusion, reduced urine output. "
                "On NSAIDs for knee arthritis. BP 158/98, HR 88, O2 sat 97%, peripheral oedema, creatinine 580 μmol/L (baseline 90)."
            ),
            "vitals": {"bp": "158/98", "hr": 88, "o2_sat": 97, "rr": 18, "temp": 36.9},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Acute kidney injury — creatinine 580 vs baseline 90, NSAID-induced. Risk of hyperkalaemia and pulmonary oedema. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "General", "Cardiology"],
            "department_scores": {"Emergency": 0.99, "General": 0.6, "Cardiology": 0.3},
        },
        progression={
            "description": "AKI — hyperkalaemia causing fatal arrhythmia is primary risk.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": +2, "bp_systolic_delta": +3, "consciousness_delta": -0.05},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"o2_sat": 92},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Nephrology",
        },
        xai={
            "primary_diagnosis": "NSAID-induced acute kidney injury",
            "differentials": [
                {"diagnosis": "NSAID-induced AKI", "probability": 0.60},
                {"diagnosis": "Pre-renal AKI (dehydration)", "probability": 0.20},
                {"diagnosis": "Intrinsic renal disease", "probability": 0.12},
                {"diagnosis": "Obstructive AKI", "probability": 0.08},
            ],
            "symptom_weights": {"creatinine x6 of baseline": 0.95, "NSAID use": 0.85, "oedema": 0.70},
            "vital_flags": ["BP 158/98 (hypertension — fluid overload)"],
            "confidence": 0.75,
            "key_reasoning_points": [
                "Stop NSAIDs immediately — they reduce renal blood flow",
                "Urgent K+ level — hyperkalaemia is the acute killer in AKI",
                "ECG for peaked T waves; if K+ >6.0 treat immediately",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_015",
            "difficulty": "medium",
            "presentation": (
                "31-year-old female. Known Crohn's disease on azathioprine. 2-day fever 39.1°C, "
                "perianal pain and swelling. BP 118/72, HR 104, O2 sat 98%, RR 18."
            ),
            "vitals": {"bp": "118/72", "hr": 104, "o2_sat": 98, "rr": 18, "temp": 39.1},
            "correct_esi": 2,
            "correct_department": "Gastroenterology",
            "reasoning": "Perianal abscess in immunocompromised Crohn's patient — high risk of sepsis and fistula. ESI-2.",
            "esi_partial_credit": {1: 0.2, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
            "department_options": ["Gastroenterology", "Emergency", "General"],
            "department_scores": {"Gastroenterology": 0.99, "Emergency": 0.75, "General": 0.5},
        },
        progression={
            "description": "Perianal sepsis in immunocompromised patient — Fournier's gangrene is a risk.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +3, "bp_systolic_delta": -3, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.04,
            "critical_thresholds": {"bp_systolic": 88},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Gastroenterology",
        },
        xai={
            "primary_diagnosis": "Perianal abscess — Crohn's-related",
            "differentials": [
                {"diagnosis": "Perianal abscess (Crohn's)", "probability": 0.65},
                {"diagnosis": "Idiopathic perianal abscess", "probability": 0.20},
                {"diagnosis": "Perianal fistula with abscess", "probability": 0.12},
                {"diagnosis": "Fournier's gangrene (early)", "probability": 0.03},
            ],
            "symptom_weights": {"perianal swelling + fever": 0.85, "Crohn's + immunosuppression": 0.88},
            "vital_flags": ["HR 104 (tachycardia — SIRS)", "Temp 39.1°C"],
            "confidence": 0.75,
            "key_reasoning_points": [
                "Immunosuppressed patients can develop severe sepsis with subtler signs",
                "MRI pelvis is gold standard for complex perianal disease",
                "Surgical drainage is definitive — antibiotics alone insufficient for abscess",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_016",
            "difficulty": "medium",
            "presentation": (
                "44-year-old male. Returned from 3-week trip to sub-Saharan Africa 5 days ago. "
                "Fever 39.8°C, rigors, headache, myalgia. BP 108/68, HR 114, O2 sat 97%, RR 22."
            ),
            "vitals": {"bp": "108/68", "hr": 114, "o2_sat": 97, "rr": 22, "temp": 39.8},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Returning traveller with fever — malaria must be excluded within hours. Haemodynamic compromise. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "General", "Resuscitation"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Resuscitation": 0.6},
        },
        progression={
            "description": "Malaria — cerebral involvement and haemolytic anaemia develop rapidly in P. falciparum.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": +4, "bp_systolic_delta": -4, "consciousness_delta": -0.08},
            "mortality_risk_per_step": 0.07,
            "critical_thresholds": {"consciousness": 0.6, "bp_systolic": 88},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Infectious disease",
        },
        xai={
            "primary_diagnosis": "Malaria (P. falciparum)",
            "differentials": [
                {"diagnosis": "Malaria", "probability": 0.60},
                {"diagnosis": "Typhoid fever", "probability": 0.15},
                {"diagnosis": "Dengue fever", "probability": 0.12},
                {"diagnosis": "Viral haemorrhagic fever", "probability": 0.05},
                {"diagnosis": "Bacterial sepsis", "probability": 0.08},
            ],
            "symptom_weights": {"Africa travel history": 0.90, "rigors": 0.85, "high fever": 0.80},
            "vital_flags": ["BP 108/68 (hypotension — early shock)", "HR 114", "Temp 39.8°C"],
            "confidence": 0.65,
            "key_reasoning_points": [
                "Returning traveller with fever = malaria until thick and thin films negative x3",
                "P. falciparum can kill within 24 hours — do not wait for result to treat if high suspicion",
                "Isolate until VHF excluded if from high-risk region",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_017",
            "difficulty": "medium",
            "presentation": (
                "16-year-old female. Taken 30 paracetamol tablets (500mg) 6 hours ago after argument with boyfriend. "
                "Now expressing regret. BP 108/70, HR 92, O2 sat 99%, mild nausea."
            ),
            "vitals": {"bp": "108/70", "hr": 92, "o2_sat": 99, "rr": 16, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Significant paracetamol overdose 6 hours ago — within treatment window. Urgent level + NAC. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "Psychiatry", "Pediatrics"],
            "department_scores": {"Emergency": 0.99, "Psychiatry": 0.4, "Pediatrics": 0.7},
        },
        progression={
            "description": "Paracetamol overdose — hepatic failure peaks at 72 hours if untreated.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.03,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Toxicology",
        },
        xai={
            "primary_diagnosis": "Deliberate paracetamol overdose",
            "differentials": [
                {"diagnosis": "Paracetamol toxicity", "probability": 0.95},
                {"diagnosis": "Mixed ingestion", "probability": 0.05},
            ],
            "symptom_weights": {"stated dose and time": 0.99, "nausea at 6h": 0.65},
            "vital_flags": [],
            "confidence": 0.95,
            "key_reasoning_points": [
                "4-hour paracetamol level on Rumack-Matthew nomogram — 6h still within window",
                "NAC if level above treatment line or if timing uncertain",
                "Psychiatric assessment mandatory — this is a deliberate overdose in a minor",
            ],
        },
    ),

    _static(
        base={
            "id": "medium_018",
            "difficulty": "medium",
            "presentation": (
                "58-year-old male. Progressively worsening low back pain for 6 weeks, now with new urinary "
                "incontinence and bilateral leg weakness since this morning. BP 136/84, HR 82, O2 sat 98%."
            ),
            "vitals": {"bp": "136/84", "hr": 82, "o2_sat": 98, "rr": 16, "temp": 36.7},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Cauda equina syndrome — new urinary incontinence + bilateral leg weakness + back pain. Neurosurgical emergency within 6 hours. ESI-2.",
            "esi_partial_credit": {1: 0.3, 2: 0.99, 3: 0.3, 4: 0.01, 5: 0.01},
            "department_options": ["Emergency", "Neurology", "Orthopedics"],
            "department_scores": {"Emergency": 0.99, "Neurology": 0.7, "Orthopedics": 0.5},
        },
        progression={
            "description": "Cauda equina — irreversible sphincter damage occurs with delayed decompression.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.01,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": True, "mri_scanner": True,
            "nurse_ratio": 0, "doctor_specialist": "Neurosurgery",
        },
        xai={
            "primary_diagnosis": "Cauda equina syndrome",
            "differentials": [
                {"diagnosis": "Cauda equina syndrome", "probability": 0.75},
                {"diagnosis": "Cord compression (upper motor neuron)", "probability": 0.15},
                {"diagnosis": "Bilateral lumbar radiculopathy", "probability": 0.08},
                {"diagnosis": "Spinal metastasis", "probability": 0.02},
            ],
            "symptom_weights": {"urinary incontinence": 0.92, "bilateral weakness": 0.88, "back pain 6 weeks": 0.70},
            "vital_flags": [],
            "confidence": 0.80,
            "key_reasoning_points": [
                "Cauda equina triad: back pain + urinary dysfunction + bilateral leg weakness",
                "MRI lumbar spine within 4 hours — delay causes permanent incontinence",
                "Urgent neurosurgical referral regardless of time of day",
            ],
        },
    ),

    # ════════════════════════════════════════════════════════════════════════════
    # HARD CASES 013–018
    # ════════════════════════════════════════════════════════════════════════════

    _static(
        base={
            "id": "hard_013",
            "difficulty": "hard",
            "presentation": (
                "49-year-old female. 2 weeks of fatigue, weight loss 4kg, night sweats. "
                "Now acute confusion. No fever. Na+ 118 mmol/L on paramedic bloods. "
                "BP 112/70, HR 96, O2 sat 97%."
            ),
            "vitals": {"bp": "112/70", "hr": 96, "o2_sat": 97, "rr": 17, "temp": 36.8},
            "correct_esi": 2,
            "correct_department": "Emergency",
            "reasoning": "Severe hyponatraemia (Na 118) with acute confusion — rapid correction risks osmotic demyelination. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Emergency", "General", "Neurology"],
            "department_scores": {"Emergency": 0.99, "General": 0.5, "Neurology": 0.5},
        },
        progression={
            "description": "Hyponatraemia — seizures and brainstem herniation if Na drops further.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +2, "bp_systolic_delta": -2, "consciousness_delta": -0.1},
            "mortality_risk_per_step": 0.06,
            "critical_thresholds": {"consciousness": 0.4},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Endocrinology",
        },
        xai={
            "primary_diagnosis": "Severe hyponatraemia — SIADH vs adrenal insufficiency",
            "differentials": [
                {"diagnosis": "SIADH (occult malignancy)", "probability": 0.45},
                {"diagnosis": "Adrenal insufficiency", "probability": 0.25},
                {"diagnosis": "Hypothyroidism", "probability": 0.15},
                {"diagnosis": "Thiazide-induced hyponatraemia", "probability": 0.15},
            ],
            "symptom_weights": {"Na 118": 0.99, "weight loss + night sweats": 0.80, "confusion": 0.85},
            "vital_flags": ["Na+ 118 mmol/L (severe — seizure threshold)"],
            "confidence": 0.55,
            "key_reasoning_points": [
                "B symptoms (weight loss, night sweats) with SIADH = paraneoplastic until proven otherwise",
                "Correct Na no faster than 8-10 mmol/L per 24h — osmotic demyelination syndrome if too fast",
                "CT chest/abdomen to exclude malignancy",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_014",
            "difficulty": "hard",
            "presentation": (
                "37-year-old male. 3 days of progressive bilateral leg weakness, started in feet. "
                "Now unable to walk. No trauma. BP 128/80, HR 78, O2 sat 97%, RR 18, areflexia bilaterally."
            ),
            "vitals": {"bp": "128/80", "hr": 78, "o2_sat": 97, "rr": 18, "temp": 37.0},
            "correct_esi": 2,
            "correct_department": "Neurology",
            "reasoning": "Guillain-Barré syndrome — ascending paralysis, areflexia, risk of respiratory failure. ESI-2.",
            "esi_partial_credit": {1: 0.5, 2: 0.99, 3: 0.3, 4: 0.05, 5: 0.01},
            "department_options": ["Neurology", "Emergency", "Resuscitation"],
            "department_scores": {"Neurology": 0.99, "Emergency": 0.7, "Resuscitation": 0.5},
        },
        progression={
            "description": "GBS — respiratory muscles involved in 30%; intubation may be needed within hours.",
            "per_timestep": {"o2_sat_delta": -2, "hr_delta": +2, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.05,
            "critical_thresholds": {"o2_sat": 91, "rr": 30},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": True,
            "nurse_ratio": 1, "doctor_specialist": "Neurology",
        },
        xai={
            "primary_diagnosis": "Guillain-Barré syndrome",
            "differentials": [
                {"diagnosis": "Guillain-Barré syndrome", "probability": 0.70},
                {"diagnosis": "Transverse myelitis", "probability": 0.15},
                {"diagnosis": "Spinal cord compression", "probability": 0.10},
                {"diagnosis": "Botulism", "probability": 0.05},
            ],
            "symptom_weights": {"ascending weakness": 0.88, "areflexia": 0.92, "no trauma": 0.70},
            "vital_flags": ["RR 18 (monitor closely — respiratory failure is the key risk)"],
            "confidence": 0.72,
            "key_reasoning_points": [
                "GBS: ascending + areflexia — monitor FVC serially; intubate if FVC <15 mL/kg",
                "MRI spine to exclude cord compression first",
                "IVIg or plasmapheresis if confirmed GBS",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_015",
            "difficulty": "hard",
            "presentation": (
                "63-year-old male. On warfarin for DVT. Presented with sudden onset headache 2 hours ago, "
                "now right arm weakness and slurred speech. INR 3.8. BP 188/106, HR 84, O2 sat 97%."
            ),
            "vitals": {"bp": "188/106", "hr": 84, "o2_sat": 97, "rr": 16, "temp": 36.9},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Anticoagulated stroke — supratherapeutic INR 3.8 with haemorrhagic stroke likely. Immediate reversal and neurosurgery. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.1, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Neurology", "Emergency"],
            "department_scores": {"Resuscitation": 0.99, "Neurology": 0.7, "Emergency": 0.6},
        },
        progression={
            "description": "Anticoagulated intracerebral haemorrhage — haematoma expansion within minutes.",
            "per_timestep": {"o2_sat_delta": -1, "hr_delta": +3, "bp_systolic_delta": +6, "consciousness_delta": -0.2},
            "mortality_risk_per_step": 0.14,
            "critical_thresholds": {"consciousness": 0.4, "bp_systolic": 220},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": True, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": True, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Neurosurgery",
        },
        xai={
            "primary_diagnosis": "Anticoagulated intracerebral haemorrhage",
            "differentials": [
                {"diagnosis": "Intracerebral haemorrhage (warfarin)", "probability": 0.70},
                {"diagnosis": "Ischaemic stroke", "probability": 0.20},
                {"diagnosis": "Hypertensive emergency with focal deficit", "probability": 0.07},
                {"diagnosis": "Subdural haematoma", "probability": 0.03},
            ],
            "symptom_weights": {"INR 3.8": 0.90, "focal deficits": 0.85, "severe headache": 0.80, "BP 188": 0.75},
            "vital_flags": ["BP 188/106 (hypertensive — worsening haematoma expansion)", "INR 3.8 (supratherapeutic)"],
            "confidence": 0.75,
            "key_reasoning_points": [
                "Anticoagulated + stroke = haemorrhagic until CT proves otherwise — do NOT give tPA",
                "Immediate reversal: 4-factor PCC + Vitamin K for warfarin",
                "Target systolic BP <140 after haemorrhage confirmed",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_016",
            "difficulty": "hard",
            "presentation": (
                "28-year-old female. 12 weeks pregnant. Vomiting 10+ times/day for 3 weeks, "
                "unable to keep any food or fluid down. Weight loss 5kg. "
                "BP 96/60, HR 118, O2 sat 99%, dry mouth, skin tenting."
            ),
            "vitals": {"bp": "96/60", "hr": 118, "o2_sat": 99, "rr": 16, "temp": 37.1},
            "correct_esi": 2,
            "correct_department": "Obstetrics",
            "reasoning": "Hyperemesis gravidarum with significant dehydration and haemodynamic compromise. ESI-2.",
            "esi_partial_credit": {1: 0.4, 2: 0.99, 3: 0.4, 4: 0.05, 5: 0.01},
            "department_options": ["Obstetrics", "Emergency", "General"],
            "department_scores": {"Obstetrics": 0.99, "Emergency": 0.7, "General": 0.3},
        },
        progression={
            "description": "Hyperemesis — Wernicke's encephalopathy from thiamine deficiency if prolonged.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": +2, "bp_systolic_delta": -3, "consciousness_delta": -0.03},
            "mortality_risk_per_step": 0.02,
            "critical_thresholds": {"bp_systolic": 80},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 1, "doctor_specialist": "Obstetrics",
        },
        xai={
            "primary_diagnosis": "Hyperemesis gravidarum",
            "differentials": [
                {"diagnosis": "Hyperemesis gravidarum", "probability": 0.70},
                {"diagnosis": "Molar pregnancy", "probability": 0.10},
                {"diagnosis": "Gastroenteritis", "probability": 0.10},
                {"diagnosis": "Appendicitis in pregnancy", "probability": 0.07},
                {"diagnosis": "Pancreatitis", "probability": 0.03},
            ],
            "symptom_weights": {"pregnancy + intractable vomiting": 0.85, "5kg weight loss": 0.80, "dehydration": 0.88},
            "vital_flags": ["BP 96/60 (significant dehydration)", "HR 118 (severe tachycardia)"],
            "confidence": 0.72,
            "key_reasoning_points": [
                "Haemodynamic compromise = hospitalisation and IV fluid resuscitation",
                "Always give IV thiamine before glucose to prevent Wernicke's encephalopathy",
                "USS to confirm viable intrauterine pregnancy and exclude molar pregnancy",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_017",
            "difficulty": "hard",
            "presentation": (
                "55-year-old male. Referred from GP with 6-week history of dysphagia to solids, "
                "now liquids. Weight loss 7kg. No pain. BP 128/78, HR 76, O2 sat 98%."
            ),
            "vitals": {"bp": "128/78", "hr": 76, "o2_sat": 98, "rr": 15, "temp": 36.8},
            "correct_esi": 3,
            "correct_department": "Gastroenterology",
            "reasoning": "Progressive dysphagia with weight loss — oesophageal malignancy until proven otherwise. Urgent workup. ESI-3.",
            "esi_partial_credit": {1: 0.05, 2: 0.3, 3: 0.99, 4: 0.4, 5: 0.01},
            "department_options": ["Gastroenterology", "Emergency", "General"],
            "department_scores": {"Gastroenterology": 0.99, "Emergency": 0.4, "General": 0.5},
        },
        progression={
            "description": "Oesophageal obstruction — aspiration risk if unable to manage secretions.",
            "per_timestep": {"o2_sat_delta": 0, "hr_delta": 0, "bp_systolic_delta": 0, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.01,
            "critical_thresholds": {},
        },
        resources={
            "er_bed": True, "icu_bed": False, "cardiac_monitor": False,
            "ecg_machine": False, "cath_lab": False, "ventilator": False,
            "ct_scanner": True, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 0, "doctor_specialist": "Gastroenterology",
        },
        xai={
            "primary_diagnosis": "Oesophageal carcinoma",
            "differentials": [
                {"diagnosis": "Oesophageal carcinoma", "probability": 0.60},
                {"diagnosis": "Achalasia", "probability": 0.18},
                {"diagnosis": "Benign stricture (GERD)", "probability": 0.12},
                {"diagnosis": "Extrinsic compression (lymphoma)", "probability": 0.10},
            ],
            "symptom_weights": {"progressive dysphagia solids→liquids": 0.90, "7kg weight loss": 0.85},
            "vital_flags": [],
            "confidence": 0.65,
            "key_reasoning_points": [
                "Solid→liquid progression = mechanical obstruction (malignancy) not motility disorder",
                "2-week wait upper GI endoscopy — CT chest/abdomen for staging",
                "Nutritional support — NG feeding if cannot manage liquids",
            ],
        },
    ),

    _static(
        base={
            "id": "hard_018",
            "difficulty": "hard",
            "presentation": (
                "32-year-old male. 4 days of progressive jaw stiffness, now unable to open mouth. "
                "Wound on foot from rusty nail 10 days ago, not cleaned. "
                "BP 138/86, HR 104, O2 sat 97%, generalised muscle spasms."
            ),
            "vitals": {"bp": "138/86", "hr": 104, "o2_sat": 97, "rr": 18, "temp": 38.2},
            "correct_esi": 1,
            "correct_department": "Resuscitation",
            "reasoning": "Tetanus — trismus + generalised spasms + unvaccinated wound. Laryngeal spasm can cause asphyxiation. ESI-1.",
            "esi_partial_credit": {1: 0.99, 2: 0.5, 3: 0.1, 4: 0.01, 5: 0.01},
            "department_options": ["Resuscitation", "Emergency", "Neurology"],
            "department_scores": {"Resuscitation": 0.99, "Emergency": 0.6, "Neurology": 0.4},
        },
        progression={
            "description": "Tetanus — laryngospasm and respiratory arrest can occur suddenly.",
            "per_timestep": {"o2_sat_delta": -2, "hr_delta": +4, "bp_systolic_delta": +4, "consciousness_delta": 0},
            "mortality_risk_per_step": 0.10,
            "critical_thresholds": {"o2_sat": 90},
        },
        resources={
            "er_bed": False, "icu_bed": True, "cardiac_monitor": True,
            "ecg_machine": False, "cath_lab": False, "ventilator": True,
            "ct_scanner": False, "or_room": False, "mri_scanner": False,
            "nurse_ratio": 2, "doctor_specialist": "Infectious disease",
        },
        xai={
            "primary_diagnosis": "Tetanus",
            "differentials": [
                {"diagnosis": "Tetanus", "probability": 0.80},
                {"diagnosis": "Strychnine poisoning", "probability": 0.08},
                {"diagnosis": "Meningitis with neck stiffness", "probability": 0.07},
                {"diagnosis": "Dystonic reaction (antipsychotics)", "probability": 0.05},
            ],
            "symptom_weights": {"trismus": 0.92, "generalised spasms": 0.90, "contaminated wound": 0.88},
            "vital_flags": ["HR 104 (autonomic instability)", "Temp 38.2°C"],
            "confidence": 0.85,
            "key_reasoning_points": [
                "Trismus + wound + spasms = tetanus — rare but immediately life-threatening",
                "Human tetanus immunoglobulin + tetanus toxoid + metronidazole",
                "ICU admission — laryngospasm risk; early intubation if spasms worsening",
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
