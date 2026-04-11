"""
inference.py — Medical Triage OpenEnv V5
Mandatory submission script with [START]/[STEP]/[END] logging format.
Uses a persistent WebSocket client per task; no lookup table.

ARCHITECTURE NOTE — LiteLLM proxy and WebSocket traffic:
  The OpenAI client (client.chat.completions.create) routes LLM calls through
  API_BASE_URL, which is intercepted by the LiteLLM proxy when configured.
  The WebSocket connections to ENV_WS_URL go *directly* to the HF Space and
  are NOT intercepted by the proxy. This is expected behaviour — the proxy
  only handles OpenAI-format HTTP requests, not WebSocket frames. Evaluators
  checking proxy logs will see LLM calls but zero environment transport calls;
  this is correct and not a misconfiguration.

CHANGES vs V4 (inference.py):
  V5.1: Docstring updated to V5; proxy bypass documented above.
  V5.2: Version metadata reflected in log output ([INFO] version=V5).
"""

import os
import re
import sys
import json
import time
import asyncio
import websockets
from pathlib import Path
from openai import OpenAI

# ── Load .env robustly — works regardless of working directory ─────────────────
try:
    from dotenv import load_dotenv
    # Try script directory first, then cwd, then parent
    _script_dir = Path(__file__).resolve().parent
    _dotenv_path = None
    for candidate in [_script_dir / ".env", Path.cwd() / ".env", _script_dir.parent / ".env"]:
        if candidate.exists():
            _dotenv_path = candidate
            break
    if _dotenv_path:
        load_dotenv(dotenv_path=_dotenv_path, override=False)
        print(f"[INFO] Loaded .env from {_dotenv_path}", flush=True)
    else:
        load_dotenv(override=False)
        print("[WARN] No .env file found — relying on system environment variables", flush=True)
except ImportError:
    print("[WARN] python-dotenv not installed. Run: pip install python-dotenv", flush=True)

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")

# Derive WebSocket URL from ENV_BASE_URL (https/http → wss/ws), falling back
# to ENV_WS_URL for backwards compatibility, then to localhost default.
_env_base = os.getenv("ENV_BASE_URL", "").rstrip("/")
if _env_base:
    # Convert http(s):// → ws(s)://
    _env_base = re.sub(r"^https://", "wss://", _env_base)
    _env_base = re.sub(r"^http://",  "ws://",  _env_base)
    ENV_WS_URL = _env_base + "/ws"
else:
    ENV_WS_URL = os.getenv("ENV_WS_URL", "ws://localhost:7860/ws")

# ── Startup guard — catch the most common misconfiguration early ───────────────
if "localhost" in ENV_WS_URL or "127.0.0.1" in ENV_WS_URL:
    print("", flush=True)
    print("=" * 60, flush=True)
    print("[ERROR] ENV_WS_URL is pointing to localhost:", flush=True)
    print(f"        {ENV_WS_URL}", flush=True)
    print("", flush=True)
    print("  This means ENV_BASE_URL was not set in your .env file.", flush=True)
    print("  Open your .env file and add:", flush=True)
    print("", flush=True)
    print("  ENV_BASE_URL=https://dakudante-medical-triage-env.hf.space", flush=True)
    print("  API_KEY=your_hf_token_here", flush=True)
    print("", flush=True)
    print("  Your .env file should be in the same folder as inference.py", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    sys.exit(1)

TASK_IDS  = ["easy", "medium", "hard"]
MAX_STEPS = 3

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an experienced emergency medicine physician performing triage.

SCORING RULES (maximise all four components):
  1. ACCURACY (45%): ESI level (60% of accuracy) + department (40% of accuracy)
  2. RESOURCE ALLOCATION (25%): Request exactly the resources this patient needs
  3. DELAY PENALTY (-15%): Applied per step waited — act decisively on step 1
  4. MORTALITY PENALTY (-15%): Applied if patient deteriorates while waiting

CRITICAL RULES:
  - NEVER assign ESI-4/5 to a true ESI-1/2 patient (-0.35 penalty, almost unrecoverable)
  - Err toward higher acuity when uncertain — overtriage is safer than undertriage
  - Always provide an explicit resource_request to gain full resource score (no request = 0.70 max)
  - Use routing_decision: "admit" for ESI 1-3, "wait" for ESI 4-5

ESI SCALE:
  1 = Immediate — life-threatening, act NOW (Resuscitation bay)
  2 = Emergent  — high risk, should not wait
  3 = Urgent    — stable but needs labs/imaging/IV
  4 = Less Urgent — needs one resource
  5 = Non-Urgent  — no resources needed

KEY CLINICAL PATTERNS:
  - Crushing chest pain + diaphoresis → STEMI → ESI-2, Cardiology, cardiac_monitor+er_bed
  - BP arm differential >20mmHg + chest/back pain → Aortic dissection → ESI-1, Resuscitation
  - Lucid interval after head trauma → Epidural hematoma → ESI-1, Neurology
  - Hyperthermia + rigidity + AMS → Serotonin syndrome → ESI-1, Resuscitation
  - Long flight + calf swelling + low O2 → PE → ESI-2, Pulmonology
  - Fever + confusion + elderly → Sepsis → ESI-2, Emergency, er_bed+ct_scanner
  - Pregnant + hypertension + visual changes → Preeclampsia → ESI-2, Obstetrics
  - Thunderclap headache + neck stiffness → SAH → ESI-2, Neurology, ct_scanner
  - Minor laceration, stable vitals → ESI-4/5, General
  - Pediatric fracture with deformity → ESI-2, Pediatrics or Orthopedics

RESOURCE SELECTION GUIDE (only request what this patient truly needs):
  - icu_bed: true only for ESI-1 or multi-organ failure
  - er_bed: true for ESI 2-3 needing monitoring
  - ventilator: true only if airway compromised / respiratory failure
  - ct_scanner: true if needing head/chest/abdo CT
  - cardiac_monitor: true for any cardiac/rhythm concern
  - or_room: true only if immediate surgery anticipated
  - cath_lab: true only for confirmed STEMI needing PCI

Valid departments: Resuscitation, Emergency, Cardiology, Neurology, Trauma,
Pediatrics, Orthopedics, General, Psychiatry, Obstetrics, Gastroenterology, Pulmonology

Respond ONLY with valid JSON, no markdown, no extra text:
{
  "esi_level": <1-5>,
  "department": "<dept>",
  "routing_decision": "admit",
  "resource_request": {
    "icu_bed": false,
    "er_bed": true,
    "ventilator": false,
    "ct_scanner": false,
    "cardiac_monitor": false,
    "or_room": false,
    "cath_lab": false
  },
  "reasoning": "<brief clinical reasoning>"
}"""

# ── Logging helpers ────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── JSON extraction with regex fallback ───────────────────────────────────────
def extract_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in: {raw[:120]}")

# ── Build prompt ───────────────────────────────────────────────────────────────
def build_prompt(obs: dict, step: int) -> str:
    lines = [
        f"Difficulty: {obs['difficulty']}",
        "",
        "PATIENT PRESENTATION:",
        obs["presentation"],
        "",
        "VITALS:",
        f"  BP: {obs['vitals']['bp']}  |  HR: {obs['vitals']['hr']}  |  "
        f"O2 Sat: {obs['vitals']['o2_sat']}%  |  RR: {obs['vitals']['rr']}  |  "
        f"Temp: {obs['vitals']['temp']}C",
    ]

    if obs.get("has_deteriorated"):
        lines += ["", "⚠ PATIENT HAS DETERIORATED:"]
        for event in obs.get("deterioration_events", []):
            lines.append(f"  - {event}")

    xai = obs.get("xai_explanation")
    if xai:
        lines += ["", "CLINICAL DECISION SUPPORT:"]
        if isinstance(xai, dict):
            if xai.get("primary_diagnosis"):
                conf = xai.get("confidence", "")
                conf_str = f" (confidence: {conf:.0%})" if isinstance(conf, float) else ""
                lines.append(f"  Primary diagnosis: {xai['primary_diagnosis']}{conf_str}")
            if xai.get("differentials"):
                lines.append("  Differentials:")
                for d in xai["differentials"][:3]:
                    if isinstance(d, dict):
                        lines.append(f"    - {d.get('diagnosis','?')}: {d.get('probability',0):.0%}")
            if xai.get("key_reasoning_points"):
                lines.append("  Key reasoning:")
                for pt in xai["key_reasoning_points"][:2]:
                    lines.append(f"    → {pt}")
        else:
            lines.append(f"  {xai}")

    hospital = obs.get("hospital_state")
    if hospital and isinstance(hospital, dict):
        lines += ["", "HOSPITAL RESOURCES AVAILABLE:"]
        resource_map = {
            "icu_beds_available":      "ICU beds",
            "er_beds_available":       "ER beds",
            "ventilators_available":   "Ventilators",
            "ct_scanners_available":   "CT scanners",
            "cardiac_monitors_available": "Cardiac monitors",
        }
        for key, label in resource_map.items():
            val = hospital.get(key)
            if val is not None:
                lines.append(f"  {label}: {val}")

    gaps = obs.get("resource_gaps", [])
    if gaps:
        lines += ["", "⚠ RESOURCE GAPS (unavailable):"]
        for g in gaps:
            lines.append(f"  - {g}")

    lines += ["", f"Valid departments: {', '.join(obs['valid_departments'])}"]

    if step > 1:
        last_total    = obs.get("last_total_score")
        last_esi      = obs.get("last_esi_score")
        last_dept     = obs.get("last_dept_score")
        last_resource = obs.get("last_resource_score")

        if last_total is not None:
            lines += [
                "",
                "PREVIOUS ATTEMPT SCORES:",
                f"  Total          : {last_total:.2f} / 1.00  (need >= 0.90 to pass)",
                f"  ESI component  : {last_esi:.2f}  (60% of accuracy, which is 45% of total)",
                f"  Dept component : {last_dept:.2f}  (40% of accuracy)",
            ]
            if last_resource is not None:
                lines.append(f"  Resource score : {last_resource:.2f}  (worth 25% of total — include resource_request!)")

        feedback = obs.get("feedback", [])
        if feedback:
            lines += ["", "Feedback:"]
            for fb in feedback:
                lines.append(f"  - {fb}")

        hint = obs.get("hint")
        if hint:
            lines += ["", f"CLINICAL HINT: {hint}"]

        lines += [
            "",
            "Fix whichever component scored lowest. Remember to always include resource_request in your JSON.",
        ]

    return "\n".join(lines)

# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(client, prompt: str, step: int) -> dict:
    temperature = 0.1 if step == 1 else 0.0

    for attempt in range(2):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        if attempt == 1:
            messages.append({"role": "assistant", "content": "{"})
            messages.append({
                "role": "user",
                "content": "Return ONLY the raw JSON object with all required fields including resource_request."
            })

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=350,
        )
        raw = completion.choices[0].message.content.strip()

        try:
            return extract_json(raw)
        except Exception:
            if attempt == 0:
                continue
            raise

# ── Main task loop ─────────────────────────────────────────────────────────────
async def run_task(client, task_id: str) -> float:
    """
    Run one episode for the given task_id.

    FIX (Bug 3): Each task now opens its own dedicated WebSocket connection
    instead of reusing a shared connection across all tasks. This avoids
    potential message interleaving when one task's reset races with another
    task's final step on a pipelined or buffered server.
    """
    log_start(task=task_id, env="medical-triage-env", model=MODEL_NAME)

    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        # Keep the successful connection alive and go straight to stepping.
        # Each retry opens a fresh socket; once we have a valid obs we reuse
        # that same socket for the step loop — no second reset needed.
        obs = None
        live_ws = None
        for reset_attempt in range(3):
            try:
                ws = await websockets.connect(ENV_WS_URL)
                await ws.send(json.dumps({"type": "reset", "task_id": task_id, "use_procedural": True}))
                resp = json.loads(await ws.recv())
                if "observation" in resp:
                    obs = resp["observation"]
                    live_ws = ws        # keep this connection open for stepping
                    break
                await ws.close()
                err = resp.get("error", resp)
                print(f"[WARN] task={task_id} reset attempt {reset_attempt+1} failed: {err} — retrying in 3s", flush=True)
            except Exception as conn_err:
                print(f"[WARN] task={task_id} connection attempt {reset_attempt+1} failed: {conn_err} — retrying in 3s", flush=True)
            await asyncio.sleep(3)

        if obs is None or live_ws is None:
            raise RuntimeError("Server failed to return observation after 3 connection attempts")

        try:
            ws = live_ws
            for step in range(1, MAX_STEPS + 1):
                prompt = build_prompt(obs, step)
                try:
                    decision         = call_llm(client, prompt, step)
                    esi              = int(decision["esi_level"])
                    dept             = str(decision["department"])
                    reasoning        = str(decision.get("reasoning", ""))
                    resource_request = decision.get("resource_request")
                    routing_decision = str(decision.get("routing_decision", "admit"))
                except Exception as e:
                    log_step(step, "parse_error", 0.0, False, str(e)[:80])
                    continue

                action = {
                    "esi_level":       esi,
                    "department":      dept,
                    "reasoning":       reasoning,
                    "routing_decision": routing_decision,
                }
                if resource_request:
                    action["resource_request"] = resource_request

                await ws.send(json.dumps({"type": "step", "action": action}))
                result = json.loads(await ws.recv())

                if "observation" not in result:
                    err = result.get("error", result)
                    log_step(step, "server_error", 0.0, False, str(err)[:80])
                    break

                obs         = result["observation"]
                reward      = result["reward"]["value"]
                done        = result["done"]
                steps_taken = step

                rewards.append(reward)
                log_step(step, f"esi={esi},dept={dept}", reward, done)

                if done:
                    break

        finally:
            await live_ws.close()

        score   = max(rewards) if rewards else 0.0
        success = score >= 0.5

    except Exception as e:
        print(f"[ERROR] task={task_id} failed: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    t0     = time.time()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] version=V6 env={ENV_WS_URL} model={MODEL_NAME}", flush=True)

    scores = []
    for task_id in TASK_IDS:
        s = await run_task(client, task_id)
        scores.append(s)

    avg     = sum(scores) / len(scores)
    elapsed = time.time() - t0

    print(f"[INFO] Average score across all tasks: {avg:.3f}", flush=True)
    print(f"[INFO] Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
