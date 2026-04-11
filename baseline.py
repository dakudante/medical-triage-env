"""
Baseline inference script for Medical Triage OpenEnv.
Uses the OpenAI API client to run an LLM agent against the environment.

Usage:
    OPENAI_API_KEY=<key> python baseline.py [--env-url <url>] [--model <model>]
"""

import os
import sys
import json
import argparse
import requests
from openai import OpenAI

DEFAULT_ENV_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
DEFAULT_MODEL   = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
TASK_IDS        = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an experienced emergency medicine physician performing triage.

You will receive a patient presentation with vitals. You must:
1. Assign an ESI (Emergency Severity Index) level from 1 to 5
2. Recommend the most appropriate department

ESI Scale:
  1 = Immediate — life-threatening, requires instant intervention NOW
  2 = Emergent  — high risk / severe distress, should not wait
  3 = Urgent    — stable but needs multiple resources (labs, imaging, IV)
  4 = Less Urgent — needs one resource only
  5 = Non-Urgent  — no resources needed, could see GP

Key clinical rules:
- When cardiac cause cannot be excluded, treat as ESI-2
- Undertriage (assigning too low a level) is more dangerous than overtriage
- Elderly + confusion + fever = sepsis workup (ESI-2)
- Pediatric patients with fractures = ESI-2 regardless of stability
- Bilateral BP difference >20mmHg + chest/back pain = aortic dissection (ESI-1)
- Lucid interval after head trauma + new confusion = epidural hematoma (ESI-1)

You MUST respond ONLY with valid JSON in this exact format:
{
  "esi_level": <integer 1-5>,
  "department": "<one of the valid departments>",
  "reasoning": "<brief clinical reasoning>"
}

No other text. No markdown. Just the JSON object."""


def get_client(base_url=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set.")
        sys.exit(1)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    elif os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    return OpenAI(**kwargs)


def env_reset(env_url, task_id):
    r = requests.post(f"{env_url}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()["observation"]


def env_step(env_url, esi_level, department, reasoning=""):
    r = requests.post(f"{env_url}/step", json={
        "action": {"esi_level": esi_level, "department": department, "reasoning": reasoning}
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def build_prompt(obs):
    lines = [
        f"Difficulty: {obs['difficulty']}",
        "",
        "PATIENT PRESENTATION:",
        obs["presentation"],
        "",
        "VITALS:",
        f"  BP: {obs['vitals']['bp']}  |  HR: {obs['vitals']['hr']}  |  "
        f"O2 Sat: {obs['vitals']['o2_sat']}%  |  RR: {obs['vitals']['rr']}  |  "
        f"Temp: {obs['vitals']['temp']}°C",
        "",
        f"Valid departments: {', '.join(obs['valid_departments'])}",
    ]
    if obs.get("feedback"):
        lines += ["", "Previous feedback:", *[f"  - {f}" for f in obs["feedback"]]]
    if obs.get("hint"):
        lines += ["", f"Clinical hint: {obs['hint']}"]
    return "\n".join(lines)


def run_task(client, env_url, task_id, model, verbose=True):
    if verbose:
        print(f"\n{'='*60}\nTask: {task_id.upper()}\n{'='*60}")

    obs = env_reset(env_url, task_id)
    best_score = 0.0
    best_action = {}

    for step in range(obs["max_steps"]):
        prompt = build_prompt(obs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        try:
            decision = json.loads(raw)
            esi   = int(decision["esi_level"])
            dept  = str(decision["department"])
            reasoning = str(decision.get("reasoning", ""))
        except Exception as e:
            if verbose:
                print(f"  Step {step+1}: Failed to parse response: {raw[:80]}")
            continue

        result = env_step(env_url, esi, dept, reasoning)
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        if verbose:
            print(f"\n  Step {step+1}: ESI-{esi} → {dept} | score={reward['value']:.3f}")
            print(f"    ESI score: {reward['esi_score']:.2f} | Dept score: {reward['department_score']:.2f}")
            if obs.get("feedback"):
                for fb in obs["feedback"][:3]:
                    print(f"    [{fb[:90]}]")

        if reward["value"] > best_score:
            best_score  = reward["value"]
            best_action = {"esi_level": esi, "department": dept}

        if done:
            break

    if verbose:
        print(f"\n  Best score: {best_score:.3f}")

    return {
        "task_id":    task_id,
        "best_score": best_score,
        "best_action": best_action,
    }


def main():
    parser = argparse.ArgumentParser(description="Medical Triage OpenEnv Baseline")
    parser.add_argument("--env-url",  default=DEFAULT_ENV_URL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    client = get_client(args.base_url)

    print(f"\nMedical Triage — Baseline Inference")
    print(f"  Environment : {args.env_url}")
    print(f"  Model       : {args.model}")

    try:
        requests.get(f"{args.env_url}/", timeout=10).raise_for_status()
    except Exception as e:
        print(f"\n[ERROR] Cannot reach environment: {e}")
        sys.exit(1)

    results = [run_task(client, args.env_url, tid, args.model, not args.quiet) for tid in TASK_IDS]

    avg = sum(r["best_score"] for r in results) / len(results)
    print(f"\n{'='*60}\nBASELINE SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['task_id']:<10} score={r['best_score']:.3f}  action={r['best_action']}")
    print(f"  {'AVERAGE':<10} score={avg:.3f}")
    print("="*60)
    print("\nJSON output:")
    print(json.dumps({"model": args.model, "average_score": round(avg,3), "results": results}, indent=2))


if __name__ == "__main__":
    main()
