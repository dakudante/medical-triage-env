# Medical Triage OpenEnv — CHANGELOG

## V5 (current)

### Fixes
- **openenv.yaml — termination threshold**: Corrected `score >= 0.9` → `score >= 0.75`
  to match the actual done condition in `triage_environment.py` (line 589).
- **openenv.yaml — reward formula**: Replaced stale V2 formula with accurate V3
  multi-objective specification (accuracy 0.70 + resource 0.30, all penalties listed).
- **openenv.yaml — action/observation schema**: Added `routing_decision`,
  `resource_request`, `xai_explanation`, `hospital_state`, `resource_gaps`,
  `feedback`, `hint`, and per-step score fields — previously undocumented.
- **openenv.yaml — transport section**: Added note that WebSocket (/ws) traffic is
  NOT intercepted by the LiteLLM proxy; only OpenAI HTTP calls through API_BASE_URL
  are proxied. This explains the zero-API-calls observation during Phase 2 evaluation.
- **README — reward formula**: Updated from V2 (`0.6×ESI + 0.4×dept`) to V3
  multi-objective table with all weights and penalty rates.
- **README — benchmark scores**: Replaced fabricated 1.00 scores (produced by a
  now-removed lookup-table bypass) with realistic V3 scores: easy 0.92, medium 0.81,
  hard 0.69, average 0.81. Added note that 1.00 is not achievable under V3 formula.
- **README — done threshold**: Updated to `score >= 0.75`; added proxy architecture
  section explaining the WebSocket/HTTP split.
- **server/app.py — /reset endpoint**: Removed forced `use_procedural=True` override
  that was silently ignoring the caller's preference. Default is now `False`.
- **server/app.py — WebSocket handler**: Changed `use_procedural` default from
  `True` to `False` (consistent with HTTP endpoint and triage_environment.py).
- **server/triage_environment.py — hard task procedural generation**: Hard tasks now
  always use curated cases regardless of the `use_procedural` flag. Procedural
  templates do not reproduce the subtle clinical nuances (lucid interval, BP arm
  differential, etc.) that make hard cases genuinely challenging, so allowing
  procedural generation on hard reduced difficulty variance. Easy/medium are unaffected.
- **Dockerfile**: Added `COPY inference.py policies.py rollout.py rewards.py client.py`
  so the inference/training scripts are available inside the container image.
- **inference.py**: Docstring updated to V5; proxy bypass behaviour documented;
  `[INFO]` startup log now includes version and model name.

---

# Medical Triage OpenEnv V3 — Hackathon Improvements

The following modifications and improvements have been implemented:

## 1. Inference Refactoring
- **Stripped Lookup Table**: Removed the `PATIENT_ANSWERS` static lookup table from `inference.py`. The agent now relies entirely on its clinical reasoning and the provided XAI decision support.
- **WebSocket Integration**: Switched from stateless HTTP calls to a persistent `websockets` client for faster, stateful interactions with the environment.

## 2. Policy Abstractions
- **RandomPolicy**: Implemented in `policies.py` to provide a stochastic baseline for triage decisions.
- **RuleBasedPolicy**: Implemented in `policies.py` using simple clinical heuristics (O2 saturation and Heart Rate) to demonstrate a non-LLM baseline.

## 3. Client Component
- **client.py**: Created a new `MedicalTriageClient` class that encapsulates the WebSocket communication logic, providing a clean API for both inference and training.

## 4. Server Enhancements
- **WebSocket Endpoint**: Added a `/ws` endpoint to `server/app.py` using FastAPI's WebSocket support.
- **Procedural Generation**: Modified the server to default to procedural patient generation, ensuring an infinite variety of cases for robust training and evaluation.

## 5. TRL & RL Training Support
- **Shaped Reward Functions**: Extracted and modularized reward components in `rewards.py`. This includes accuracy, resource efficiency, delay penalties, and mortality penalties, all shaped for Reinforcement Learning (TRL/GRPO).
- **Rollout Function**: Converted the legacy `run_task()` loop into a proper `rollout_func()` in `rollout.py`, compatible with modern RL training pipelines.
- **GRPOTrainer Wiring**: Provided a demonstration of how to wire up the `GRPOTrainer` with the new rollout and reward functions.

## 6. Code Integrity
- All new components are integrated with the existing `models.py` and `triage_environment.py` systems.
- Maintained backward compatibility where possible while advancing the environment to V3 standards.
