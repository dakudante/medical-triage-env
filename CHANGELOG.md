## V7 (current)

### Pillar 1.3 — Paediatric sub-pool (NEW)
- `PAEDIATRIC_CASES` list added to `patients.py` — 3 curated cases: febrile seizure
  (easy, ESI-2), epiglottitis (medium, ESI-1), appendicitis in adolescent (hard, ESI-2)
- `age_group` field added to all patients via `_static()` — auto-detected from
  presentation text; values: `paediatric` (0-16), `adult` (17-64), `elderly` (65+)
- `paediatric` task added to `TASK_CONFIGS` and exposed via `GET /tasks`
- Paediatric vital context in presentations — HR up to 140 normal in infants,
  weight-based drug dosing noted, paediatric-specific differentials in XAI

### Pillar 2 — Partial observability & multi-turn realism (COMPLETE)
- 2.1: max_steps 3 → 6, acuity-proportional delay rates
- 2.2: partial_obs mode — 1-2 vitals hidden as "not yet measured"
- 2.3: nurse handoff format on step 2+ when partial_obs is active

### Pillar 3 — Reward function V4 (COMPLETE)
- 3.1: Acuity-proportional delay rates wired via `_ESI_DELAY_RATES` dict
- 3.2: `compute_calibration_bonus()` — +0.02 for correct + reasoned decisions,
  -0.05 for overconfident wrong decisions; added to `TriageReward.calibration_bonus`
- 3.3: `final_outcome` label set when `done=True`:
  OPTIMAL / ACCEPTABLE / DELAYED / HARMFUL / FATAL

### Pillar 4 — Training infrastructure (COMPLETE)
- 4.1: POST /rollout_batch — parallel asyncio.gather resets
- 4.2: CurriculumPolicy in policies.py
- 4.3: GET /leaderboard + POST /leaderboard/submit
- 4.4: seed: Optional[int] in ResetRequest for deterministic episodes

### Pillar 5 — Safety analysis & explainability (COMPLETE)
- 5.1: `_classify_failure()` in environment — tags each step:
  UNDERTRIAGE_CRITICAL / UNDERTRIAGE_MODERATE / OVERTRIAGE / DEPARTMENT_MISMATCH /
  RESOURCE_WASTE / DELAY_EXCESSIVE / CORRECT / PARTIAL
  Returned in `info["failure_mode"]` on every step
- 5.2: `compute_safety_score()` — separate 0-1 metric for ESI-1/2 handling only.
  Added to `TriageReward.safety_score`. An agent scoring 0.85 overall can score
  0.01 on safety_score if it dangerously undertriages a critical patient
- 5.3: Per-episode audit log — every step writes a structured JSON entry to
  `_audit_log` (max 200 in memory). `GET /audit` returns last 50 entries with
  failure_mode_summary and outcome_summary aggregates

### Bug Fixes (from V7 initial)
- Fixed `generate_patient_from_template()` KeyError on `{temp}`, `{rr}`, `{bp_dia}`
  in 6 new condition template reasoning strings

## V7 (current)

### New Features — Pillar 2: Partial observability & multi-turn realism

**2.1 Extended episode horizon (3 → 6 steps)**
- `MedicalTriageEnvironment.max_steps` raised from 3 to 6 for all tasks
  (mass_casualty keeps 3 steps — 5 simultaneous patients is already demanding)
- Delay penalty is now acuity-proportional:
  ESI-1: −0.12/step · ESI-2: −0.08 · ESI-3: −0.05 · ESI-4: −0.02 · ESI-5: −0.01
  Cap raised from 0.30 to 0.40 to accommodate longer episodes
- Forces agents to commit early while rewarding fast correct decisions

**2.2 Partial observability mode**
- `ResetRequest`: new `partial_obs: bool = False` field
- When enabled: 1–2 vitals randomly set to `"not yet measured"` in the observation
- `TriageObservation`: new `partial_obs_applied: bool` and `missing_vitals: list[str]` fields
- Trains agents to reason under real ED uncertainty (patients often arrive without full workup)

**2.3 Nurse handoff observation format**
- On step 2+ when `partial_obs=True`, observation shifts to nurse handoff perspective
- `TriageObservation`: new `handoff_mode: bool` and `nurse_summary: str` fields
- Nurse summary: "Charge nurse reports: [presentation excerpt]... Measured vitals: [available]. Note: [missing] not yet obtained."
- Tests whether agents can integrate second-hand clinical information

### New Features — Pillar 4: Training infrastructure & RL pipeline

**4.1 Parallel batch rollout endpoint**
- New `POST /rollout_batch` endpoint — runs up to 16 simultaneous fresh resets via `asyncio.gather`
- Body: `{task_id, n_episodes, use_procedural, partial_obs, seed}`
- Returns list of episode observations ready for GRPO training batch collection
- Eliminates need to run N sequential WebSocket connections for batch training

**4.2 Curriculum learning policy**
- `CurriculumPolicy` class added to `policies.py`
- Auto-selects task difficulty from rolling average score (window=10, configurable)
- Thresholds: score < 0.65 → easy · 0.65–0.78 → medium · > 0.78 → hard
- Methods: `select_task()`, `record_score(score)`, `rolling_average()`, `summary()`
- Wraps any base policy (RuleBasedPolicy, LLM agent, RandomPolicy)

**4.3 In-memory leaderboard**
- `GET /leaderboard` — returns top 10 runs sorted by average score
- `POST /leaderboard/submit` — accepts `{agent_name, easy_score, medium_score, hard_score}`, returns rank
- Enables community model benchmarking without external infrastructure
- Resets on server restart (note documented in endpoint response)

**4.4 Seed-based deterministic evaluation**
- `ResetRequest`: new `seed: Optional[int] = None` field
- When set: `random.seed(seed)` applied before patient selection and procedural generation
- Fully reproducible episodes — eliminates ±0.05 score variance
- Supported in both HTTP (`/reset`) and WebSocket (`{"type":"reset","seed":42}`) transports
- Per-episode seeds via `/rollout_batch`: `base_seed + episode_index`

### Bug Fixes
- **CRITICAL — procedural generation crash** (`generate_patient_from_template`):
  6 new V6.5 templates used `{temp}`, `{rr}`, `{bp_dia}` in `reasoning_template`
  but `.format()` only received `age, sex, hr, o2, bp_sys`. Fixed by passing all
  8 vitals variables. Affected: `sepsis`, `serotonin_syndrome`, `anaphylaxis`,
  `dka`, `heat_stroke`, `post_op_complication`.

### Version
- `pyproject.toml`: 6.0.0 → 7.0.0
- `inference.py`: version log updated to V7

# Medical Triage OpenEnv — CHANGELOG

## V7 (current)

### Bug Fixes
- **CRITICAL — procedural generation crash**: `generate_patient_from_template()` in
  `server/patients.py` called `reasoning_template.format(age, sex, hr, o2, bp_sys)`
  but 6 new V6.5 templates used additional placeholders `{temp}`, `{rr}`, and
  `{bp_dia}` in their reasoning strings. This caused a `KeyError` whenever the
  server picked a procedural patient from those templates, crashing the WebSocket
  reset and returning `{"error": "'rr'"}` (or `'temp'`, `'bp_dia'`) to the client.
  Fix: `.format()` now passes all 8 vitals variables: `age, sex, hr, o2, bp_sys,
  bp_dia, rr, temp`. All 14 templates verified clean.
  Affected templates: `sepsis`, `serotonin_syndrome`, `anaphylaxis`, `dka`,
  `heat_stroke`, `post_op_complication`.

### Patient Pool (Pillar 1 — V7 Roadmap)
- **54 curated cases confirmed**: 18 easy / 18 medium / 18 hard — 3× the original
  V5 pool of 18. All cases verified to have complete vitals, xai_metadata,
  progression profiles, and resource requirements.
- **14 condition templates**: Up from 8 in V6. New: `anaphylaxis`, `dka`,
  `heat_stroke`, `lithium_toxicity`, `post_op_complication`, `hypoglycaemia`.
- **Version**: inference.py updated to V7.

# Medical Triage OpenEnv — CHANGELOG

## V6 (current)

### New Features
- **Mass Casualty task**: 4th task (`mass_casualty`) — 5 simultaneous patients, reduced hospital capacity (2 ICU beds, 4 ER beds, 2 doctors). Tests prioritisation under resource pressure. Wired through `ResetRequest.hospital_config`.
- **ESILevel Enum**: `ESILevel(IntEnum)` in `models.py` — IMMEDIATE=1, EMERGENT=2, URGENT=3, LESS_URGENT=4, NON_URGENT=5. Type-safe action specification.
- **OpenEnv base class**: `MedicalTriageEnvironment` now inherits from `openenv.core.Environment` (with graceful fallback if openenv-core not installed).

### Fixes
- **environment.py**: Renamed from `triage_environment.py` to `environment.py` for OpenEnv spec compliance. All imports updated.
- **Async HTTP endpoints**: All HTTP endpoints (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`) are now `async def` (OpenEnv spec compliance).
- **State endpoint enriched**: `/state` now returns `episode_id`, `task_id`, `patient_id`, `step`, `done`, `best_score` — richer metadata for score variance tracking.
- **Per-task pass thresholds**: `openenv.yaml` now includes `pass_threshold` per task (easy: 0.75, medium: 0.70, hard: 0.60, mass_casualty: 0.55).
- **Score variance documented**: README and `/baseline` response now document ±0.05 expected variance and instructions for deterministic evaluation via `/grader`.
- **WebSocket state message**: `/ws` now handles `{"type": "state"}` messages in addition to reset and step.
- **Version bump**: `pyproject.toml` and `app.py` updated to 6.0.0.

### Documentation
- **EnvClient quick-start**: README now includes both HTTP and WebSocket quick-start code snippets.
- **RL Formulation section**: Explicit state space, action space, reward, episode, stochasticity, and partial observability definitions added to README.
- **Limitations section**: Patient pool size (18 cases), clinical validation gap, and observation cleanliness documented proactively.
- **openenv.yaml enriched**: Full action schema with `ESILevel` enum reference, per-task pass thresholds, score variance note, limitations block.

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
