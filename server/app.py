"""
FastAPI server for the Medical Triage OpenEnv environment.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, POST /grader, GET /baseline

FIX (Bug 1): env is now instantiated per-WebSocket connection, not as a module-level
singleton. The previous global `env` caused a race condition: concurrent WebSocket
clients shared the same episode state and corrupted each other's resets and steps.
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
from .triage_environment import MedicalTriageEnvironment, score_triage

app = FastAPI(
    title="Medical Triage — OpenEnv",
    description=(
        "An OpenEnv-compliant RL environment where AI agents learn to triage patients "
        "using the Emergency Severity Index (ESI). Agents assess patient presentations, "
        "assign ESI levels (1–5), and recommend departments. Graders use clinical "
        "protocols with asymmetric undertriage penalties."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── FIX (Bug 1): Removed module-level `env = MedicalTriageEnvironment()`.
# A single global env was shared across all WebSocket connections, causing
# race conditions when multiple clients reset or stepped concurrently.
# Each WebSocket connection now gets its own isolated env instance (see below).

# The HTTP endpoints (/reset, /step, /state) still need a shared env for
# stateful HTTP sessions. If you need concurrent HTTP sessions too, switch to
# a session-keyed dict. For now, one env for HTTP is acceptable for single-user use.
_http_env = MedicalTriageEnvironment()


@app.get("/")
def root():
    return {
        "name": "Medical Triage OpenEnv",
        "version": "1.0.0",
        "description": "RL environment for ESI-based medical triage.",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
        "esi_scale": {
            1: "Immediate", 2: "Emergent", 3: "Urgent",
            4: "Less Urgent", 5: "Non-Urgent",
        },
        "valid_departments": DEPARTMENTS,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    """Start a new episode with a random patient from the specified difficulty pool.

    V5: use_procedural defaults to False. Pass use_procedural=true to enable
    procedural generation for easy/medium tasks. Hard tasks always use curated
    cases regardless of this flag (preserves difficulty variance).
    """
    if request is None:
        request = ResetRequest(use_procedural=False)
    # V5 fix: removed forced override of use_procedural=True
    obs = _http_env.reset(request=request)
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Submit a triage decision (ESI level + department) and receive reward."""
    obs, reward, done, info = _http_env.step(request.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket endpoint for the environment.

    FIX (Bug 1): Each connection gets its own MedicalTriageEnvironment instance.
    This prevents concurrent clients from trampling each other's episode state.
    """
    await websocket.accept()

    # ── Per-connection environment ─────────────────────────────────────────────
    env = MedicalTriageEnvironment()  # isolated per connection

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "reset":
                task_id = message.get("task_id", "easy")
                use_procedural = message.get("use_procedural", False)  # V5: default False
                request = ResetRequest(
                    task_id=task_id,
                    use_procedural=use_procedural,
                    num_patients=1,
                )
                obs = env.reset(request=request)
                await websocket.send_text(json.dumps({"observation": obs.model_dump()}))

            elif msg_type == "step":
                action_data = message.get("action", {})
                action = TriageAction(**action_data)
                obs, reward, done, info = env.step(action)
                await websocket.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "reward": reward.model_dump(),
                    "done": done,
                    "info": info,
                }))

            else:
                await websocket.send_text(json.dumps({"error": "Unknown message type"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except Exception:
            pass  # connection may already be gone


@app.get("/state", response_model=StateResponse)
def state():
    """Return current episode metadata."""
    return StateResponse(state=_http_env.state())


@app.get("/tasks")
def get_tasks():
    """List all tasks with action schema."""
    return {"tasks": [t.model_dump() for t in _http_env.get_tasks()]}


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest):
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
        total_score=total,
        feedback=feedback,
        correct_esi=patient["correct_esi"],
        correct_department=patient["correct_department"],
    )


@app.get("/baseline")
def baseline():
    """
    Rule-based baseline agent on all 3 difficulty tasks.
    Uses one representative patient per difficulty level.
    Returns reproducible scores for hackathon submission.
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
            "patient_id":        patient_id,
            "submitted_esi":     esi,
            "submitted_dept":    dept,
            "correct_esi":       patient["correct_esi"],
            "correct_dept":      patient["correct_department"],
            "esi_score":         reward.esi_score,
            "department_score":  reward.department_score,
            "total_score":       score,
            "feedback":          feedback,
        }
        total += score

    return {
        "agent":         "rule_based_baseline",
        "average_score": round(total / len(results), 3),
        "task_scores":   results,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
