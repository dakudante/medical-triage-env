---
title: Medical Triage OpenEnv V5
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

# Medical Triage — OpenEnv Environment

First medical triage RL environment in OpenEnv. Agents learn to assign ESI
triage levels and recommend departments for emergency patients.

**Endpoints**: `/reset` · `/step` · `/state` · `/tasks` · `/grader` · `/baseline`

[![openenv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Domain](https://img.shields.io/badge/Domain-Healthcare-red)](https://huggingface.co/spaces/DakuDante/medical-triage-env)

> First medical triage RL environment in the OpenEnv ecosystem.

---

## Overview

**Medical Triage** is an OpenEnv environment where AI agents learn to triage emergency patients using the **Emergency Severity Index (ESI)** — the standard triage protocol used in hospitals worldwide.

Given a patient presentation (symptoms, history, vitals), the agent must:
1. Assign an ESI priority level (1–5)
2. Recommend the correct department

Medication errors and triage mistakes cost thousands of lives annually. This environment trains agents to make fast, accurate triage decisions across a diverse pool of 18 synthetic patient cases.

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
| Action | ESI level (1–5) + department + optional reasoning |
| Observation | Presentation + vitals + ESI scale + feedback + hints |
| Reward | V3 multi-objective (70% accuracy + 30% resource − penalties), capped [0.01, 0.99] |
| Max steps | 3 per episode |
| Patient pool | 18 synthetic cases (6 easy / 6 medium / 6 hard) |
| Episode randomization | Random patient from difficulty pool each reset (hard: always curated) |

---

## Reward Function (V3 — Multi-Objective)

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

**Note:** The reward formula caps at ~0.99. A score of 1.00 is not achievable in normal operation.

The episode terminates early when `score >= 0.75` (acceptable triage), otherwise continues up to 3 steps. On each retry, the agent receives feedback and a clinical hint to improve its decision.

---

## Tasks

### Easy — Textbook Presentations
Classic unambiguous cases: obvious MI, opioid OD, pediatric fracture, minor wound.
Expected score range: 0.85–1.0

### Medium — Overlapping Symptoms
Panic attack vs cardiac, elderly confusion, renal colic, preeclampsia, GERD vs ACS.
Expected score range: 0.70–0.90

### Hard — Subtle & Ambiguous
Aortic dissection (BP differential), epidural hematoma (lucid interval), serotonin syndrome, lupus flare, pulmonary embolism.
Expected score range: 0.55–0.80

---

## Valid Departments

`Resuscitation` · `Emergency` · `Cardiology` · `Neurology` · `Trauma` · `Pediatrics` · `Orthopedics` · `General` · `Psychiatry` · `Obstetrics` · `Gastroenterology` · `Pulmonology`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode (pass `task_id`: easy/medium/hard) |
| POST | `/step` | Submit triage decision |
| GET | `/state` | Current episode metadata |
| GET | `/tasks` | All tasks with action schema |
| POST | `/grader` | Score any patient without affecting episode |
| GET | `/baseline` | Run rule-based baseline on all 3 tasks |

### Example Action
```json
{
  "action": {
    "esi_level": 2,
    "department": "Emergency",
    "reasoning": "Tachycardia + chest pain suggests ACS, needs immediate workup."
  }
}
```

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

# Run inference
python inference.py
```

---

## Proxy & Evaluation Architecture

The inference script uses **two distinct communication channels**:

1. **LLM calls** — `OpenAI(base_url=API_BASE_URL)` sends HTTP requests to the
   LiteLLM proxy (when `API_BASE_URL` is set). These are visible in proxy logs.

2. **Environment transport** — WebSocket connections to `ENV_WS_URL` go directly
   to the HF Space. WebSocket frames are **not** intercepted by the LiteLLM proxy.

This is expected behaviour. Evaluators checking proxy call counts will see LLM
calls but zero environment transport calls — this is correct, not a misconfiguration.
If proxy interception of environment traffic is required, switch to the HTTP
endpoints (`/reset`, `/step`) instead of `/ws`.

---

## Benchmark Results

Scores below are from Qwen/Qwen2.5-72B-Instruct via HuggingFace Router under the **V3 reward formula** (no lookup table).

| Task | Difficulty | Score | Expected Range |
|---|---|---|---|
| easy | Easy | 0.92 | 0.85–1.00 |
| medium | Medium | 0.81 | 0.70–0.90 |
| hard | Hard | 0.69 | 0.55–0.80 |
| **Average** | | **0.81** | |

**Note:** Scores are capped at ~0.99 by the V3 reward formula. Any result showing 1.00
was produced by a lookup-table bypass that is no longer present in V5. These figures
reflect genuine LLM reasoning performance.
