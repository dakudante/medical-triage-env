---
title: Medical Triage OpenEnv V6
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - triage
  - medical
  - agent
license: mit
---

# Medical Triage — OpenEnv Environment V6

First medical triage RL environment in OpenEnv. Agents learn to assign ESI
triage levels and recommend departments for emergency patients.

**Endpoints**: `/reset` · `/step` · `/state` · `/tasks` · `/grader` · `/baseline`

[![openenv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/DakuDante/medical-triage-env)
[![Domain](https://img.shields.io/badge/Domain-Healthcare-red)](https://huggingface.co/spaces/DakuDante/medical-triage-env)
[![Version](https://img.shields.io/badge/Version-6.0.0-green)](https://huggingface.co/spaces/DakuDante/medical-triage-env)

> First medical triage RL environment in the OpenEnv ecosystem.

---

## Overview

**Medical Triage** is an OpenEnv environment where AI agents learn to triage emergency patients using the **Emergency Severity Index (ESI)** — the standard triage protocol used in hospitals worldwide.

Given a patient presentation (symptoms, history, vitals), the agent must:
1. Assign an ESI priority level (1–5)
2. Recommend the correct department
3. Request appropriate hospital resources
4. Specify a routing decision (admit / wait / reroute)

Medication errors and triage mistakes cost thousands of lives annually. This environment trains agents to make fast, accurate triage decisions across a diverse pool of 18 synthetic patient cases across 4 tasks including a mass casualty scenario.

---

## Quick Start

```python
# HTTP client (simple)
import requests

BASE = "https://DakuDante-medical-triage-env.hf.space"

obs = requests.post(f"{BASE}/reset", json={"task_id": "easy"}).json()["observation"]
print(obs["presentation"])

result = requests.post(f"{BASE}/step", json={
    "action": {
        "esi_level": 2,
        "department": "Emergency",
        "reasoning": "Tachycardia + chest pain suggests ACS.",
        "resource_request": {"er_bed": True, "cardiac_monitor": True}
    }
}).json()
print(result["reward"]["value"])
```

```python
# WebSocket client (recommended for RL training)
import asyncio
from client import MedicalTriageClient

async def run():
    async with MedicalTriageClient("wss://DakuDante-medical-triage-env.hf.space/ws") as client:
        obs = await client.reset("easy")
        result = await client.step({
            "esi_level": 2,
            "department": "Emergency",
            "resource_request": {"er_bed": True, "cardiac_monitor": True}
        })
        print(f"Score: {result['reward']['value']}")

asyncio.run(run())
```

---

## RL Formulation

This environment follows the standard RL interface:

| Component | Definition |
|---|---|
| **State space** | Patient presentation (text) + current vitals + initial vitals + hospital resource state + step count + deterioration events |
| **Action space** | ESI level ∈ {1,2,3,4,5} × Department ∈ {12 options} × ResourceRequest (7 binary flags) × RoutingDecision ∈ {admit, wait, reroute} |
| **Reward** | Dense, multi-objective, shaped per step — not sparse end-of-episode |
| **Episode** | Max 3 steps; early termination when score ≥ 0.75 |
| **Stochasticity** | Random patient from difficulty pool on each reset; vital progression adds per-step noise |
| **Partial observability** | Hospital resource availability changes across steps; vitals degrade dynamically |

---

## ESI Scale

| Level | Name | Description |
|---|---|---|
| 1 | Immediate | Life-threatening — requires intervention NOW |
| 2 | Emergent | High risk — should not wait |
| 3 | Urgent | Stable but needs multiple resources |
| 4 | Less Urgent | Needs one resource only |
| 5 | Non-Urgent | No resources needed |

---

## Environment Details

| Property | Value |
|---|---|
| Domain | Emergency medicine triage |
| Action | ESI level (1–5) + department + optional reasoning + resource request + routing |
| Observation | Presentation + vitals + trend + hospital state + ESI scale + feedback + hints |
| Reward | V3 multi-objective (70% accuracy + 30% resource − penalties), capped [0.01, 0.99] |
| Max steps | 3 per episode |
| Patient pool | 18 curated cases (6 easy / 6 medium / 6 hard) |
| Episode randomization | Random patient from difficulty pool each reset |
| ESI type | `ESILevel` IntEnum — IMMEDIATE=1 … NON_URGENT=5 |

---

## Tasks

| Task ID | Name | Difficulty | Patients | Expected Score | Pass Threshold |
|---|---|---|---|---|---|
| `easy` | Textbook Triage | Easy | 6 curated | 0.85–1.00 | 0.75 |
| `medium` | Overlapping Presentations | Medium | 6 curated | 0.70–0.90 | 0.70 |
| `hard` | Subtle & Ambiguous | Hard | 6 curated | 0.55–0.80 | 0.60 |
| `mass_casualty` | Mass Casualty Incident | Hard | 5 simultaneous | 0.50–0.75 | 0.55 |

### Easy — Textbook Presentations
Classic unambiguous cases: obvious MI, opioid OD, pediatric fracture, minor wound.

### Medium — Overlapping Symptoms
Panic attack vs cardiac, elderly confusion, renal colic, preeclampsia, GERD vs ACS.

### Hard — Subtle & Ambiguous
Aortic dissection (BP differential), epidural hematoma (lucid interval), serotonin syndrome, lupus flare, pulmonary embolism.

### Mass Casualty Incident (V6 new)
5 simultaneous patients with mixed acuity. Hospital running at reduced capacity:
2 ICU beds, 4 ER beds, 1 resus bay, 2 doctors. Tests prioritisation and resource-aware routing under pressure.

```python
# Trigger mass casualty task
obs = requests.post(f"{BASE}/reset", json={"task_id": "mass_casualty"}).json()
```

---

## Action Space

```json
{
  "esi_level": 2,
  "department": "Emergency",
  "reasoning": "Tachycardia + chest pain suggests ACS, needs immediate workup.",
  "routing_decision": "admit",
  "resource_request": {
    "icu_bed": false,
    "er_bed": true,
    "ventilator": false,
    "ct_scanner": false,
    "cardiac_monitor": true,
    "or_room": false,
    "cath_lab": false
  }
}
```

**Required**: `esi_level`, `department`
**Optional but scored**: `resource_request` (worth 30% of reward), `routing_decision`, `reasoning`

---

## Observation Space

```json
{
  "presentation": "58-year-old male. Chest pain radiating to left arm, diaphoresis...",
  "vitals": {"bp": "90/60", "hr": 118, "o2_sat": 94, "rr": 22, "temp": 37.2},
  "initial_vitals": {"bp": "95/65", "hr": 110, "o2_sat": 96, "rr": 20, "temp": 37.2},
  "timesteps_waited": 1,
  "consciousness_score": 0.95,
  "has_deteriorated": false,
  "deterioration_events": [],
  "hospital_state": {
    "icu_beds_available": 3,
    "er_beds_available": 8,
    "doctors_available": 2
  },
  "xai_explanation": {
    "primary_diagnosis": "Acute Coronary Syndrome",
    "confidence": 0.88,
    "key_reasoning_points": ["ST changes", "Troponin rise expected"]
  },
  "feedback": ["ESI level was correct. Department score: 0.60 — consider Cardiology."],
  "hint": "Chest pain + diaphoresis + radiation to arm = ACS until proven otherwise.",
  "last_total_score": 0.61,
  "esi_scale": {"1": "Immediate", "2": "Emergent", "3": "Urgent", "4": "Less Urgent", "5": "Non-Urgent"}
}
```

---

## Reward Design

```
total = accuracy_component + resource_component
        − delay_penalty − mortality_penalty
        − overtriage_penalty − undertriage_penalty
total = clamp(total, 0.01, 0.99)
```

| Component | Weight | Notes |
|---|---|---|
| `accuracy_component` | 0.70 | ESI score (60%) + department score (40%) |
| `resource_component` | 0.30 | Full credit with correct `resource_request`; 0.21 partial if omitted |
| `delay_penalty` | −0.15/step | Applied each step beyond step 1 |
| `mortality_penalty` | −0.15 | Applied if patient deteriorates while waiting |
| `undertriage_penalty_severe` | −0.35 | ESI-1/2 patient assigned ESI-4/5 |
| `undertriage_penalty_mild` | −0.15 | ESI-1 patient assigned ESI-3 |
| `overtriage_penalty` | −0.15 | ESI-4/5 patient sent to Resuscitation/ICU |

**Signal density**: Reward is shaped at every step — not sparse. Feedback and clinical hints are provided after each subthreshold step. The episode terminates early when `score >= 0.75`.

**Asymmetric penalties**: Undertriage (−0.35) is penalised more severely than overtriage (−0.15), reflecting real clinical risk asymmetry — sending a critical patient to the wrong area can be fatal.

---

## Valid Departments

`Resuscitation` · `Emergency` · `Cardiology` · `Neurology` · `Trauma` · `Pediatrics` · `Orthopedics` · `General` · `Psychiatry` · `Obstetrics` · `Gastroenterology` · `Pulmonology`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode (pass `task_id`: easy/medium/hard/mass_casualty) |
| POST | `/step` | Submit triage decision |
| GET | `/state` | Current episode metadata (episode_id, step, done, best_score) |
| GET | `/tasks` | All tasks with action schema |
| POST | `/grader` | Score any patient without affecting episode |
| GET | `/baseline` | Rule-based baseline on all tasks |

---

## Setup

```bash
# Local
pip install -r requirements.txt
uvicorn server.app:app --port 7860 --reload

# Docker
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env

# Validate
python validate.py --url http://localhost:7860
```

---

## RL Training

```python
# GRPO rollout
from rollout import rollout_func
from policies import RuleBasedPolicy
import asyncio

policy = RuleBasedPolicy()
trajectory = asyncio.run(rollout_func(policy.act, task_id="medium"))
print(f"Episode reward: {sum(t['reward'] for t in trajectory):.3f}")
```

```python
# Full GRPO training demo
from rollout import run_training_demo
import asyncio
asyncio.run(run_training_demo())
```

---

## Proxy & Evaluation Architecture

Two distinct communication channels:

1. **LLM calls** — `OpenAI(base_url=API_BASE_URL)` sends HTTP requests to the LiteLLM proxy. Visible in proxy logs.
2. **Environment transport** — WebSocket connections to `ENV_WS_URL` go directly to the HF Space. Not intercepted by the LiteLLM proxy.

Evaluators checking proxy call counts will see LLM calls but zero environment transport calls — this is correct. Use HTTP endpoints (`/reset`, `/step`) if proxy interception of environment traffic is required.

---

## Benchmark Results

Scores from Qwen/Qwen2.5-72B-Instruct via HuggingFace Router under the V3 reward formula.

| Task | Difficulty | Score | Expected Range |
|---|---|---|---|
| easy | Easy | 0.92 | 0.85–1.00 |
| medium | Medium | 0.81 | 0.70–0.90 |
| hard | Hard | 0.69 | 0.55–0.80 |
| **Average** | | **0.81** | |

> **Score variance note**: ±0.05 variance expected in episode mode due to random patient selection per reset. Scores above reflect averages over 3 runs. Use fixed patient IDs via `/grader` for fully deterministic evaluation. Scores of 1.00 are not achievable under the V3 formula — any result showing 1.00 indicates a lookup-table bypass (removed in V5).

---

## Limitations

- **Patient pool**: 18 curated cases. Agents trained here should be tested on a wider case pool before strong generalisation claims. Procedural generation is available via `use_procedural=true` for easy/medium tasks for unlimited variety.
- **Clinical validation**: The reward function encodes ESI heuristics but has not been formally validated against real ED outcome data (mortality, length of stay).
- **Observation cleanliness**: Current observations are well-structured. Real ED data is noisier and more incomplete — a known gap addressed in the V7 roadmap.

---

## V6 Changes

- Mass casualty task (4th task, 5 patients, reduced hospital capacity)
- `ESILevel` IntEnum for type-safe action specification
- `MedicalTriageEnvironment` inherits from `openenv.core.Environment`
- All HTTP endpoints are now `async`
- `environment.py` replaces `triage_environment.py` (OpenEnv spec compliance)
- Per-task `pass_threshold` added to openenv.yaml
- Score variance documentation added
- RL Formulation section added to README
- EnvClient quick-start snippet added

See [CHANGELOG.md](CHANGELOG.md) for full history.


## PPO-lite training (V7.1)

This version adds a small but real RL pipeline built on PyTorch. It trains a multi-head policy over ESI, department, routing, and resource requests using PPO-style clipped updates.

### Quick start

```bash
python train_ppo.py --updates 4 --episodes-per-update 6 --output-dir artifacts
python evaluate_ppo.py --checkpoint artifacts/ppo_lite_checkpoint.pt --episodes 9
```

Artifacts saved:
- `artifacts/ppo_lite_checkpoint.pt`
- `artifacts/train_metrics.json`

The trainer uses the local `MedicalTriageEnvironment` directly, so it does not require the websocket server to be running.
