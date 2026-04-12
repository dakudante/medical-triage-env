---
title: Medical Triage OpenEnv
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

[![openenv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/DakuDante/medical-triage-env)
[![Domain](https://img.shields.io/badge/Domain-Healthcare-red)](https://huggingface.co/spaces/DakuDante/medical-triage-env)
[![Version](https://img.shields.io/badge/Version-1.0.0-green)](https://huggingface.co/spaces/DakuDante/medical-triage-env)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> The first medical triage RL environment in the OpenEnv ecosystem. Train agents to make fast, accurate emergency triage decisions across 57 curated patient cases and 5 task types.

**Endpoints**: `/reset` · `/step` · `/state` · `/tasks` · `/grader` · `/baseline`

---

## Overview

**Medical Triage** is an OpenEnv-compliant environment where AI agents learn to triage emergency patients using the **Emergency Severity Index (ESI)** — the standard 5-level triage protocol used in hospitals worldwide.

Given a patient presentation (symptoms, history, vitals), the agent must:

1. Assign an **ESI priority level** (1 = Immediate → 5 = Non-Urgent)
2. Recommend the correct **department** (e.g. Resuscitation, Cardiology, Trauma)
3. Request appropriate **hospital resources** (ICU bed, ventilator, CT scanner, etc.)
4. Specify a **routing decision** (admit / wait / reroute)

The reward is dense and multi-objective — shaped at every step with clinical feedback and hints — making this environment well-suited for GRPO, PPO, and other online RL algorithms.

---

## Quick Start

```python
# HTTP client
import requests

BASE = "https://DakuDante-medical-triage-env.hf.space"

obs = requests.post(f"{BASE}/reset", json={"task_id": "easy"}).json()["observation"]
print(obs["presentation"])

result = requests.post(f"{BASE}/step", json={
    "action": {
        "esi_level": 2,
        "department": "Emergency",
        "reasoning": "Tachycardia + chest pain — likely ACS.",
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
        obs = await client.reset("medium")
        result = await client.step({
            "esi_level": 2,
            "department": "Emergency",
            "resource_request": {"er_bed": True, "cardiac_monitor": True}
        })
        print(f"Score: {result['reward']['value']:.3f}")

asyncio.run(run())
```

---

## Tasks

| Task ID | Name | Difficulty | Pool | Pass Threshold | Expected Score |
|---|---|---|---|---|---|
| `easy` | Textbook Triage | Easy | 19 cases | 0.75 | 0.85–1.00 |
| `medium` | Overlapping Presentations | Medium | 19 cases | 0.70 | 0.70–0.90 |
| `hard` | Subtle & Ambiguous | Hard | 19 cases | 0.60 | 0.55–0.80 |
| `mass_casualty` | Mass Casualty Incident | Hard | 5 simultaneous | 0.55 | 0.50–0.75 |
| `paediatric` | Paediatric Triage | Medium | 8 cases | 0.65 | 0.65–0.90 |

### Easy — Textbook Presentations
Classic unambiguous cases: obvious MI, opioid OD, paediatric fracture, anaphylaxis, minor wound.

### Medium — Overlapping Symptoms
Panic attack vs cardiac event, elderly confusion, renal colic, preeclampsia, GERD vs ACS.

### Hard — Subtle & Ambiguous
Aortic dissection (BP differential), epidural hematoma (lucid interval), serotonin syndrome, lupus flare, pulmonary embolism.

### Mass Casualty Incident
5 simultaneous patients with mixed acuity. Hospital at reduced capacity: 2 ICU beds, 4 ER beds, 1 resus bay, 2 doctors. Tests prioritisation and resource-aware routing under pressure.

### Paediatric Triage
Age-appropriate presentations with paediatric vital ranges: febrile seizure, croup, bronchiolitis, intussusception, and paediatric trauma. Agents must apply child-specific ESI thresholds and appropriate department routing.

```python
# Select any task
obs = requests.post(f"{BASE}/reset", json={"task_id": "mass_casualty"}).json()
```

---

## RL Formulation

| Component | Definition |
|---|---|
| **State** | Patient presentation (text) + current vitals + initial vitals + hospital resource state + step count + deterioration events + XAI explanation |
| **Action** | ESI level ∈ {1–5} × Department ∈ {12 options} × ResourceRequest (7 binary flags) × RoutingDecision ∈ {admit, wait, reroute} |
| **Reward** | Dense, multi-objective, shaped per step — not sparse end-of-episode |
| **Episode** | Max 6 steps; early termination when score ≥ 0.75 |
| **Stochasticity** | Random patient from difficulty pool on each reset; vital progression adds per-step noise |
| **Partial observability** | Hospital resource availability changes across steps; vitals degrade dynamically; `partial_obs=True` hides 1–2 vitals |

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
**Optional but scored**: `resource_request` (worth 30% of total reward), `routing_decision`, `reasoning`

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
  "feedback": ["ESI level correct. Department score 0.60 — consider Cardiology."],
  "hint": "Chest pain + diaphoresis + arm radiation = ACS until proven otherwise.",
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
| `delay_penalty` | −0.12 to −0.01/step | Acuity-proportional; applied each step beyond step 1 |
| `mortality_penalty` | −0.15 | Applied if patient deteriorates while waiting |
| `undertriage_severe` | −0.35 | ESI-1/2 patient assigned ESI-4/5 |
| `undertriage_mild` | −0.15 | ESI-1 patient assigned ESI-3 |
| `overtriage_penalty` | −0.15 | ESI-4/5 patient sent to Resuscitation/ICU |

**Asymmetric penalties**: Undertriage (−0.35) is penalised far more severely than overtriage (−0.15), reflecting real clinical risk — sending a critical patient to the wrong area can be fatal.

**Signal density**: Reward is shaped at every step. Feedback and clinical hints are returned after each subthreshold action. Episodes terminate early when `score ≥ 0.75`.

---

## Valid Departments

`Resuscitation` · `Emergency` · `Cardiology` · `Neurology` · `Trauma` · `Pediatrics` · `Orthopedics` · `General` · `Psychiatry` · `Obstetrics` · `Gastroenterology` · `Pulmonology`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode (`task_id`: easy / medium / hard / mass_casualty / paediatric) |
| `POST` | `/step` | Submit triage decision |
| `GET` | `/state` | Episode metadata (episode_id, step, done, best_score) |
| `GET` | `/tasks` | All tasks with full action schema |
| `POST` | `/grader` | Score any patient without affecting active episode |
| `GET` | `/baseline` | Rule-based baseline scores across all tasks |
| `POST` | `/rollout_batch` | Parallel batch resets (up to 16) for GRPO training |
| `GET` | `/leaderboard` | Top 10 submitted agent scores |
| `GET` | `/audit` | Last 50 episode audit entries with failure-mode summaries |

---

## Setup

```bash
# Local development
pip install -r requirements.txt
uvicorn server.app:app --port 7860 --reload

# Docker
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env

# Validate endpoints
python validate.py --url http://localhost:7860
```

---

## RL Training

### GRPO / Rollout

```python
from rollout import rollout_func
from policies import RuleBasedPolicy
import asyncio

policy = RuleBasedPolicy()
trajectory = asyncio.run(rollout_func(policy.act, task_id="medium"))
print(f"Episode reward: {sum(t['reward'] for t in trajectory):.3f}")
```

### PPO-lite (local, no server required)

```bash
# Install RL dependencies first
pip install -r requirements-rl.txt

# Train
python train_ppo.py --updates 4 --episodes-per-update 6 --output-dir artifacts

# Evaluate checkpoint
python evaluate_ppo.py --checkpoint artifacts/ppo_lite_checkpoint.pt --episodes 9
```

Artifacts saved to `artifacts/`:
- `ppo_lite_checkpoint.pt` — policy weights
- `train_metrics.json` — per-update loss, reward, and entropy curves

### Curriculum Learning

```python
from policies import CurriculumPolicy, RuleBasedPolicy

policy = CurriculumPolicy(base_policy=RuleBasedPolicy())
# Auto-selects task difficulty based on rolling average score:
# score < 0.65 → easy  |  0.65–0.78 → medium  |  > 0.78 → hard
task = policy.select_task()
```

---

## Deterministic Evaluation

For reproducible benchmarking, use fixed seeds or the `/grader` endpoint:

```python
# Seed-based — eliminates ±0.05 episode variance
obs = requests.post(f"{BASE}/reset", json={"task_id": "hard", "seed": 42}).json()

# Grader — score a specific patient without starting an episode
result = requests.post(f"{BASE}/grader", json={
    "patient_id": "hard_001",
    "action": {"esi_level": 1, "department": "Resuscitation"}
}).json()
```

---

## Proxy & Evaluation Architecture

Two distinct communication channels:

1. **LLM calls** — `OpenAI(base_url=API_BASE_URL)` sends HTTP to the LiteLLM proxy. Visible in proxy logs.
2. **Environment transport** — WebSocket connections to `ENV_WS_URL` go directly to the HF Space. Not intercepted by the proxy.

Evaluators checking proxy call counts will see LLM calls but zero environment transport calls — this is correct behaviour. Use HTTP endpoints if proxy interception of environment traffic is required.

---

## Benchmark Results

Scores from `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router (7 runs, random patient selection).

| Task | Mean Score | Min | Max | Pass Threshold |
|---|---|---|---|---|
| easy | 0.887 | 0.829 | 0.986 | 0.75 ✅ |
| medium | 0.915 | 0.884 | 0.970 | 0.70 ✅ |
| hard | 0.923 | 0.829 | 0.990 | 0.60 ✅ |
| **Average** | **0.908** | **0.880** | **0.934** | |

> Score variance of ±0.05 is expected in episode mode due to random patient sampling. Use fixed seeds via `/reset` or fixed patient IDs via `/grader` for fully deterministic evaluation.

---

## Limitations

- **Clinical validation**: The reward function encodes ESI heuristics but has not been formally validated against real ED outcome data (mortality, length of stay). This is a research and training environment.
- **Patient pool**: 57 curated cases + procedural generation for easy/medium tasks. Agents should be tested on a wider case distribution before strong generalisation claims.
- **Observation cleanliness**: Observations are well-structured synthetic data. Real ED data is noisier, more incomplete, and contains transcription errors — a known gap for future work.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.
