# Medical Triage OpenEnv — Changelog

## 1.0.0 — Final Submission

### Infrastructure
- `rl/` package now included in Docker image so evaluators can inspect training scripts without a separate install
- `torch` moved from `requirements.txt` to `requirements-rl.txt` — keeps the production image slim (~2 GB savings); install with `pip install -r requirements-rl.txt` only when running PPO training locally
- `pyproject.toml`: version bumped to `1.0.0`; `rl*` added to `setuptools.packages.find`; torch moved to `[rl]` optional dependency group

### Patient Pool
- Pool expanded to **57 curated cases**: 19 easy / 19 medium / 19 hard
- **8 paediatric cases** added covering febrile seizure, croup, bronchiolitis, intussusception, and paediatric trauma with age-appropriate vital ranges

### New Task — Paediatric Triage
- `paediatric` task added as the 5th task type alongside easy / medium / hard / mass_casualty
- Documented in `openenv.yaml` with `patient_pool_size: 8`, `pass_threshold: 0.65`, `expected_score_range: [0.65, 0.90]`
- Exposed via `GET /tasks` — evaluators can select with `task_id=paediatric`

### Test Suite
- `TestPatientBank` assertions corrected: `len(PATIENTS) == 57`, `len(EASY_CASES) == 19`, etc.

### Code Hygiene
- All stale version strings removed from `server/app.py`, `server/__init__.py`, `server/environment.py`, `rl/__init__.py`, `rl/train_ppo.py`, `inference.py`
- FastAPI app title updated to `"Medical Triage — OpenEnv"`, version `"1.0.0"`
- Reward formula description updated to V4 in app description and `openenv.yaml`

---

## 0.7.1 — PPO-lite Training Pipeline

- Added `rl/train_ppo.py` — multi-head PPO policy over ESI, department, routing, and resource requests
- Added `rl/evaluate_ppo.py`, `rl/policy_net.py`, `rl/feature_extractor.py`
- Trainer uses `MedicalTriageEnvironment` directly (no server required for local training)
- Artifacts: `artifacts/ppo_lite_checkpoint.pt`, `artifacts/train_metrics.json`

---

## 0.7.0 — Partial Observability & Training Infrastructure

### Partial Observability
- `max_steps` raised 3 → 6; acuity-proportional delay rates (`ESI-1: −0.12/step` → `ESI-5: −0.01/step`)
- `partial_obs` mode: 1–2 vitals randomly hidden as `"not yet measured"`
- Nurse handoff format activates on step 2+ when `partial_obs=True`

### Training Infrastructure
- `POST /rollout_batch` — up to 16 parallel resets via `asyncio.gather`
- `CurriculumPolicy` in `policies.py` — auto-escalates difficulty from rolling average score
- `GET /leaderboard` + `POST /leaderboard/submit` — in-memory community benchmarking
- `seed: Optional[int]` on `ResetRequest` for fully deterministic episodes

### Reward V4
- Calibration bonus: `+0.02` for correct + reasoned, `−0.05` for overconfident wrong
- `final_outcome` label on episode end: `OPTIMAL / ACCEPTABLE / DELAYED / HARMFUL / FATAL`
- `compute_safety_score()` — separate 0–1 metric for ESI-1/2 handling only

### Safety & Explainability
- `_classify_failure()` tags every step with failure mode
- Per-episode audit log via `GET /audit`

### Bug Fixes
- Fixed `generate_patient_from_template()` `KeyError` on `{temp}`, `{rr}`, `{bp_dia}` in 6 condition templates

---

## 0.6.0 — Mass Casualty & OpenEnv Compliance

- `mass_casualty` task: 5 simultaneous patients, reduced hospital capacity
- `ESILevel` IntEnum — type-safe action specification
- `MedicalTriageEnvironment` inherits from `openenv.core.Environment`
- All HTTP endpoints converted to `async def`
- `environment.py` replaces `triage_environment.py`
- Per-task `pass_threshold` added to `openenv.yaml`

---

## 0.5.0 — Reward V3 & Spec Compliance

- Reward formula updated to V3: `0.70 × accuracy + 0.30 × resource − penalties`
- Asymmetric undertriage penalty: `−0.35` for ESI-1/2 → ESI-4/5
- `openenv.yaml` corrected: done threshold, full action/observation schema, proxy architecture note
- Lookup-table bypass removed; benchmark scores recalibrated

---

## 0.3.0 — Initial RL Infrastructure

- `client.py`, `policies.py`, `rollout.py`, `rewards.py` added
- WebSocket endpoint added to server
- `rollout_func()` compatible with GRPO training pipelines
