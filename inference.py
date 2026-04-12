"""
inference.py — Medical Triage OpenEnv
Mandatory submission script with [START]/[STEP]/[END] logging format.

Uses plain HTTP requests to the environment (no WebSocket dependency).
LLM calls go through API_BASE_URL (intercepted by the LiteLLM proxy).
"""

import os
import re
import sys
import json
import time

# Optional dotenv — never crash if missing
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import requests
from openai import OpenAI

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://dakudante-medical-triage-env.hf.space").rstrip("/")

TASK_IDS  = ["easy", "medium", "hard"]
MAX_STEPS = 3

SYSTEM_PROMPT = """You are an experienced emergency medicine physician performing triage.

SCORING RULES:
  1. ACCURACY (70%): ESI level (60%) + department (40%)
  2. RESOURCE ALLOCATION (30%): Request exactly the resources this patient needs
  3. DELAY PENALTY (-15%): Applied per step waited
  4. MORTALITY PENALTY (-15%): Applied if patient deteriorates

CRITICAL: NEVER assign ESI-4/5 to a true ESI-1/2 patient (-0.35 penalty).
Always include resource_request — omitting it caps your resource score at 0.70.
Use routing_decision: "admit" for ESI 1-3, "wait" for ESI 4-5.

ESI SCALE:
  1 = Immediate, 2 = Emergent, 3 = Urgent, 4 = Less Urgent, 5 = Non-Urgent

KEY PATTERNS:
  - Crushing chest pain + diaphoresis → ESI-2, Emergency, cardiac_monitor+er_bed
  - BP arm differential >20mmHg + chest/back pain → ESI-1, Resuscitation
  - Lucid interval after head trauma → ESI-1, Resuscitation
  - Hyperthermia + rigidity + AMS → ESI-1, Resuscitation
  - Long flight + calf swelling + low O2 → ESI-2, Pulmonology
  - Fever + confusion + elderly → ESI-2, Emergency
  - Pregnant + hypertension + visual changes → ESI-2, Obstetrics
  - Thunderclap headache → ESI-2, Neurology
  - Opioid OD + pinpoint pupils + RR<10 → ESI-1, Resuscitation
  - Paediatric fracture with deformity → ESI-2, Pediatrics

Valid departments: Resuscitation, Emergency, Cardiology, Neurology, Trauma,
Pediatrics, Orthopedics, General, Psychiatry, Obstetrics, Gastroenterology, Pulmonology

Respond ONLY with valid JSON:
{
  "esi_level": <1-5>,
  "department": "<dept>",
  "routing_decision": "admit",
  "resource_request": {
    "icu_bed": false, "er_bed": true, "ventilator": false,
    "ct_scanner": false, "cardiac_monitor": false, "or_room": false, "cath_lab": false
  },
  "reasoning": "<brief clinical reasoning>"
}"""


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def env_reset(task_id):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()["observation"]

def env_step(esi, dept, reasoning, resource_request, routing):
    action = {"esi_level": esi, "department": dept, "reasoning": reasoning, "routing_decision": routing}
    if resource_request:
        action["resource_request"] = resource_request
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_json(raw):
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
    raise ValueError(f"No valid JSON found: {raw[:120]}")


def build_prompt(obs, step):
    lines = [
        f"Difficulty: {obs.get('difficulty', 'unknown')}",
        "",
        "PATIENT PRESENTATION:",
        obs.get("presentation", ""),
        "",
        "VITALS:",
        f"  BP: {obs['vitals']['bp']}  HR: {obs['vitals']['hr']}  "
        f"O2: {obs['vitals']['o2_sat']}%  RR: {obs['vitals']['rr']}  Temp: {obs['vitals']['temp']}C",
    ]

    if obs.get("has_deteriorated"):
        lines += ["", "PATIENT HAS DETERIORATED:"]
        for event in obs.get("deterioration_events", []):
            lines.append(f"  - {event}")

    xai = obs.get("xai_explanation")
    if xai and isinstance(xai, dict):
        lines += ["", "CLINICAL DECISION SUPPORT:"]
        if xai.get("primary_diagnosis"):
            conf = xai.get("confidence", "")
            cs = f" ({conf:.0%})" if isinstance(conf, float) else ""
            lines.append(f"  Primary: {xai['primary_diagnosis']}{cs}")
        for d in (xai.get("differentials") or [])[:3]:
            if isinstance(d, dict):
                lines.append(f"  - {d.get('diagnosis','?')}: {d.get('probability',0):.0%}")
        for pt in (xai.get("key_reasoning_points") or [])[:2]:
            lines.append(f"  -> {pt}")

    hospital = obs.get("hospital_state")
    if hospital and isinstance(hospital, dict):
        lines += ["", "HOSPITAL RESOURCES:"]
        for key, label in [
            ("icu_beds_available", "ICU beds"),
            ("er_beds_available", "ER beds"),
            ("ventilators_available", "Ventilators"),
            ("ct_scanners_available", "CT scanners"),
            ("cardiac_monitors_available", "Cardiac monitors"),
        ]:
            val = hospital.get(key)
            if val is not None:
                lines.append(f"  {label}: {val}")

    lines += ["", f"Valid departments: {', '.join(obs.get('valid_departments', []))}"]

    if step > 1:
        last_total = obs.get("last_total_score")
        if last_total is not None:
            lines += [
                "",
                "PREVIOUS ATTEMPT SCORES:",
                f"  Total: {last_total:.2f}  ESI: {obs.get('last_esi_score',0):.2f}  "
                f"Dept: {obs.get('last_dept_score',0):.2f}  Resource: {obs.get('last_resource_score',0):.2f}",
            ]
        for fb in obs.get("feedback", []):
            lines.append(f"  * {fb}")
        hint = obs.get("hint")
        if hint:
            lines += ["", f"HINT: {hint}"]
        lines += ["", "Correct your answer based on the scores and feedback above."]

    return "\n".join(lines)


def call_llm(client, prompt, step):
    temperature = 0.1 if step == 1 else 0.0
    for attempt in range(2):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        if attempt == 1:
            messages.append({"role": "assistant", "content": "{"})
            messages.append({"role": "user", "content": "Return ONLY the raw JSON."})
        completion = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=temperature, max_tokens=350,
        )
        raw = completion.choices[0].message.content.strip()
        try:
            return extract_json(raw)
        except Exception:
            if attempt == 0:
                continue
            raise


def run_task(client, task_id):
    log_start(task=task_id, env="medical-triage-env", model=MODEL_NAME)
    rewards, steps_taken, score, success = [], 0, 0.0, False

    try:
        obs = env_reset(task_id)

        for step in range(1, MAX_STEPS + 1):
            prompt = build_prompt(obs, step)

            try:
                decision         = call_llm(client, prompt, step)
                esi              = int(decision["esi_level"])
                dept             = str(decision["department"])
                reasoning        = str(decision.get("reasoning", ""))
                resource_request = decision.get("resource_request")
                routing          = str(decision.get("routing_decision", "admit"))
            except Exception as e:
                log_step(step, "parse_error", 0.0, False, str(e)[:80])
                continue

            try:
                result = env_step(esi, dept, reasoning, resource_request, routing)
            except Exception as e:
                log_step(step, "env_error", 0.0, False, str(e)[:80])
                break

            obs         = result["observation"]
            reward      = result["reward"]["value"]
            done        = result["done"]
            steps_taken = step
            rewards.append(reward)
            log_step(step, f"esi={esi},dept={dept}", reward, done)

            if done:
                break

        score   = max(rewards) if rewards else 0.0
        success = score >= 0.5

    except Exception as e:
        print(f"[ERROR] task={task_id}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    t0     = time.time()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Connecting to environment at {ENV_BASE_URL}/", flush=True)
    try:
        requests.get(f"{ENV_BASE_URL}/", timeout=15).raise_for_status()
    except Exception as e:
        print(f"[WARN] Health check: {e} — continuing", flush=True)

    scores = []
    for task_id in TASK_IDS:
        try:
            s = run_task(client, task_id)
        except Exception as e:
            print(f"[ERROR] task={task_id}: {e}", flush=True)
            s = 0.0
        scores.append(s)

    avg     = sum(scores) / len(scores) if scores else 0.0
    elapsed = time.time() - t0
    print(f"[INFO] Average score across all tasks: {avg:.3f}", flush=True)
    print(f"[INFO] Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
