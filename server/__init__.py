# Medical Triage OpenEnv — Server Package V6
# environment.py replaces triage_environment.py (OpenEnv spec compliance — Fix 4)
from .environment import MedicalTriageEnvironment, score_triage
from .patients import PATIENTS, PATIENT_MAP, DEPARTMENTS, ESI_DESCRIPTIONS
