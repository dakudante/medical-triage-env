"""
FastAPI server for the Medical Triage OpenEnv environment — V6.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, POST /grader, GET /baseline

V6 changes:
  - environment.py replaces triage_environment.py (OpenEnv spec compliance)
  - All HTTP endpoints are now async (Fix 7)
  - /state returns richer metadata for score variance tracking (Fix 13)
  - Mass casualty task wired through ResetRequest hospital_config (Fix 8)
  - Per-connection env isolation retained from V5 (race-condition fix)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

from models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse,
    GraderRequest, GraderResponse,
    TriageAction,
)
from .patients import PATIENT_MAP, DEPARTMENTS
from .environment import MedicalTriageEnvironment, score_triage

app = FastAPI(
    title="Medical Triage — OpenEnv V6",
    description=(
        "An OpenEnv-compliant RL environment where AI agents learn to triage patients "
        "using the Emergency Severity Index (ESI). Agents assess patient presentations, "
        "assign ESI levels (1–5), recommend departments, and allocate hospital resources. "
        "Graders use a multi-objective V3 reward formula with asymmetric undertriage penalties."
    ),
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP endpoints share one env instance (single-user HTTP sessions).
# WebSocket connections each get their own isolated instance (see /ws below).
_http_env = MedicalTriageEnvironment()

# Mass casualty task config — wired through hospital_config override
MASS_CASUALTY_CONFIG = {
    "task_id": "mass_casualty",
    "num_patients": 5,
    "use_procedural": True,
    "hospital_config": {
        "icu_beds": 2,
        "er_beds": 4,
        "resus_bays": 1,
        "doctors_available": 2,
        "nurses_available": 3,
    },
}


@app.get("/")
async def root():
    return {
        "name": "Medical Triage OpenEnv",
        "version": "6.0.0",
        "description": "RL environment for ESI-based medical triage.",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
        "esi_scale": {
            1: "Immediate", 2: "Emergent", 3: "Urgent",
            4: "Less Urgent", 5: "Non-Urgent",
        },
        "valid_departments": DEPARTMENTS,
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = None):
    """
    Start a new episode. Pass task_id: easy | medium | hard | mass_casualty.

    V6: mass_casualty task spawns 5 patients with reduced hospital capacity
    (2 ICU beds, 4 ER beds, 2 doctors). Tests prioritisation under resource pressure.
    use_procedural defaults to False. Hard tasks always use curated cases.
    """
    if request is None:
        request = ResetRequest(use_procedural=False)

    # Wire mass_casualty through hospital_config
    if request.task_id == "mass_casualty":
        request = ResetRequest(
            task_id="hard",
            num_patients=MASS_CASUALTY_CONFIG["num_patients"],
            use_procedural=MASS_CASUALTY_CONFIG["use_procedural"],
            hospital_config=MASS_CASUALTY_CONFIG["hospital_config"],
        )

    obs = _http_env.reset(request=request)
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Submit a triage decision (ESI level + department) and receive reward."""
    obs, reward, done, info = _http_env.step(request.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
async def state():
    """
    Return current episode metadata.
    Includes episode_id, task_id, patient_id, step, done, best_score.
    Use this to track score variance across runs.
    """
    return StateResponse(state=_http_env.state())


@app.get("/tasks")
async def get_tasks():
    """List all tasks with action schema. Includes mass_casualty task (V6)."""
    base_tasks = [t.model_dump() for t in _http_env.get_tasks()]

    # Append mass_casualty task definition
    mass_casualty_task = {
        "task_id": "mass_casualty",
        "name": "Mass Casualty Incident",
        "difficulty": "hard",
        "description": (
            "5 simultaneous patients arrive with mixed acuity. "
            "Hospital running at reduced capacity: 2 ICU beds, 4 ER beds, 2 doctors available. "
            "Agent must triage and route all patients under resource pressure."
        ),
        "expected_score_range": [0.50, 0.75],
        "pass_threshold": 0.55,
        "num_patients": 5,
        "action_schema": {
            "esi_level": {"type": "integer", "minimum": 1, "maximum": 5},
            "department": {"type": "string", "enum": DEPARTMENTS},
            "routing_decision": {"type": "string", "enum": ["admit", "wait", "reroute"]},
            "resource_request": {"type": "object"},
            "reasoning": {"type": "string"},
        },
    }
    base_tasks.append(mass_casualty_task)
    return {"tasks": base_tasks}


@app.post("/grader", response_model=GraderResponse)
async def grader(request: GraderRequest):
    """
    Standalone grader. Score a triage decision for a specific patient
    without affecting the running episode.
    """
    if request.patient_id not in PATIENT_MAP:
        raise HTTPException(status_code=404, detail=f"Patient '{request.patient_id}' not found.")

    patient = PATIENT_MAP[request.patient_id]
    action = TriageAction(esi_level=request.esi_level, department=request.department)
    total, reward, feedback = score_triage(patient, action)

    return GraderResponse(
        patient_id=request.patient_id,
        esi_score=reward.esi_score,
        department_score=reward.department_score,
        resource_score=getattr(reward, "resource_score", 1.0),
        total_score=total,
        feedback=feedback,
        correct_esi=patient["correct_esi"],
        correct_department=patient["correct_department"],
    )


@app.get("/baseline")
async def baseline():
    """
    Rule-based baseline on all 3 standard difficulty tasks.
    Returns reproducible scores for hackathon submission.
    Documented variance: ±0.05 due to random patient selection per reset.
    """
    baseline_cases = {
        "easy":   ("easy_001",   2, "Emergency"),
        "medium": ("medium_001", 2, "Emergency"),
        "hard":   ("hard_001",   3, "Emergency"),
    }

    results = {}
    total = 0.0

    for difficulty, (patient_id, esi, dept) in baseline_cases.items():
        patient = PATIENT_MAP[patient_id]
        action  = TriageAction(esi_level=esi, department=dept)
        score, reward, feedback = score_triage(patient, action)
        results[difficulty] = {
            "patient_id":       patient_id,
            "submitted_esi":    esi,
            "submitted_dept":   dept,
            "correct_esi":      patient["correct_esi"],
            "correct_dept":     patient["correct_department"],
            "esi_score":        reward.esi_score,
            "department_score": reward.department_score,
            "total_score":      score,
            "feedback":         feedback,
        }
        total += score

    return {
        "agent":         "rule_based_baseline",
        "version":       "V6",
        "average_score": round(total / len(results), 3),
        "task_scores":   results,
        "note": (
            "Scores reflect fixed patient IDs for reproducibility. "
            "±0.05 variance expected in episode mode due to random patient selection per reset."
        ),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket endpoint. Each connection gets its own
    isolated MedicalTriageEnvironment instance — no shared state.
    Supports: reset, step messages.
    """
    await websocket.accept()
    env = MedicalTriageEnvironment()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "reset":
                task_id = message.get("task_id", "easy")
                use_procedural = message.get("use_procedural", False)
                hospital_config = None

                if task_id == "mass_casualty":
                    task_id = "hard"
                    use_procedural = True
                    hospital_config = MASS_CASUALTY_CONFIG["hospital_config"]

                request = ResetRequest(
                    task_id=task_id,
                    use_procedural=use_procedural,
                    num_patients=message.get("num_patients", 1),
                    hospital_config=hospital_config,
                )
                obs = env.reset(request=request)
                await websocket.send_text(json.dumps({"observation": obs.model_dump()}))

            elif msg_type == "step":
                action_data = message.get("action", {})
                action = TriageAction(**action_data)
                obs, reward, done, info = env.step(action)
                await websocket.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "reward":      reward.model_dump(),
                    "done":        done,
                    "info":        info,
                }))

            elif msg_type == "state":
                await websocket.send_text(json.dumps({"state": env.state()}))

            else:
                await websocket.send_text(json.dumps({"error": "Unknown message type. Use: reset, step, state"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except Exception:
            pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
