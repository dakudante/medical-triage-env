"""Compatibility shim exposing the original 18 curated cases for legacy tests/tools."""
from server.patients import *  # type: ignore
from server import patients as _p

EASY_CASES = list(_p.EASY_CASES[:6])
MEDIUM_CASES = list(_p.MEDIUM_CASES[:6])
HARD_CASES = list(_p.HARD_CASES[:6])
PATIENTS = EASY_CASES + MEDIUM_CASES + HARD_CASES
PATIENT_MAP = {p["id"]: p for p in PATIENTS}
