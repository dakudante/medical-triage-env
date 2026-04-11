#!/usr/bin/env python3
"""
Pre-submission validator for Medical Triage OpenEnv.
Run before submitting to catch disqualification issues.

Usage:
    python validate.py --url http://localhost:7860
    python validate.py --url https://YOUR-HF-SPACE.hf.space
"""

import sys
import argparse
import requests

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"{tag} {label}")
    if detail:
        print(f"         {detail}")
    return condition


def validate(base_url):
    base_url = base_url.rstrip("/")
    results = []
    print(f"\nValidating: {base_url}\n{'='*56}")

    # 1. Root
    print("\n[1] Server reachability")
    try:
        r = requests.get(f"{base_url}/", timeout=15)
        results.append(check("GET / returns 200", r.status_code == 200))
        data = r.json()
        results.append(check("  has valid_departments", "valid_departments" in data))
        results.append(check("  has esi_scale", "esi_scale" in data))
    except Exception as e:
        results.append(check("GET /", False, str(e)))
        print("  Cannot reach server — aborting.")
        return False

    # 2. Tasks
    print("\n[2] /tasks")
    try:
        r = requests.get(f"{base_url}/tasks", timeout=15)
        results.append(check("/tasks returns 200", r.status_code == 200))
        tasks = r.json().get("tasks", [])
        results.append(check("  3 tasks present", len(tasks) == 3, f"found {len(tasks)}"))
        for diff in ["easy", "medium", "hard"]:
            results.append(check(f"  task '{diff}' exists", any(t["task_id"] == diff for t in tasks)))
        for t in tasks:
            schema = t.get("action_schema", {})
            results.append(check(f"  '{t['task_id']}' has esi_level in schema", "esi_level" in schema))
            results.append(check(f"  '{t['task_id']}' has department in schema", "department" in schema))
    except Exception as e:
        results.append(check("/tasks", False, str(e)))

    # 3. Reset
    print("\n[3] /reset")
    obs = None
    for difficulty in ["easy", "medium", "hard"]:
        try:
            r = requests.post(f"{base_url}/reset", json={"task_id": difficulty}, timeout=15)
            results.append(check(f"  reset difficulty={difficulty}", r.status_code == 200))
            obs = r.json().get("observation", {})
            for field in ["patient_id", "presentation", "vitals", "esi_scale",
                          "valid_departments", "step", "max_steps", "done"]:
                results.append(check(f"    observation.{field} present", field in obs))
            results.append(check("    done=False on reset", obs.get("done") is False))
            results.append(check("    step=0 on reset", obs.get("step") == 0))
        except Exception as e:
            results.append(check(f"  reset difficulty={difficulty}", False, str(e)))

    # 4. Step
    print("\n[4] /step")
    try:
        r = requests.post(f"{base_url}/reset", json={"task_id": "easy"}, timeout=15)
        r = requests.post(f"{base_url}/step",
                          json={"action": {"esi_level": 2, "department": "Emergency"}},
                          timeout=15)
        results.append(check("POST /step returns 200", r.status_code == 200))
        data = r.json()
        for field in ["observation", "reward", "done", "info"]:
            results.append(check(f"  has '{field}'", field in data))
        reward = data.get("reward", {})
        for field in ["value", "esi_score", "department_score"]:
            results.append(check(f"  reward.{field} present", field in reward))
        val = reward.get("value", -1)
        results.append(check("  reward.value in [0.0, 1.0]", 0.0 <= val <= 1.0, f"value={val}"))
    except Exception as e:
        results.append(check("POST /step", False, str(e)))

    # 5. State
    print("\n[5] /state")
    try:
        r = requests.get(f"{base_url}/state", timeout=15)
        results.append(check("GET /state returns 200", r.status_code == 200))
        state = r.json().get("state", {})
        for field in ["episode_id", "task_id", "patient_id", "step", "done", "best_score"]:
            results.append(check(f"  state.{field} present", field in state))
    except Exception as e:
        results.append(check("GET /state", False, str(e)))

    # 6. Grader — varying scores
    print("\n[6] /grader")
    try:
        # Correct answer
        rg = requests.post(f"{base_url}/grader",
                           json={"patient_id": "easy_001", "esi_level": 2, "department": "Emergency"},
                           timeout=15)
        results.append(check("  /grader returns 200", rg.status_code == 200))
        good_score = rg.json().get("total_score", -1)
        results.append(check("  correct answer scores high", good_score >= 0.8, f"score={good_score}"))

        # Wrong answer
        rb = requests.post(f"{base_url}/grader",
                           json={"patient_id": "easy_001", "esi_level": 5, "department": "General"},
                           timeout=15)
        bad_score = rb.json().get("total_score", 1.0)
        results.append(check("  wrong answer scores low", bad_score < 0.4, f"score={bad_score}"))
        results.append(check("  grader scores vary (not constant)", good_score != bad_score))

        # All 3 tasks have gradeable patients
        for pid in ["easy_001", "medium_001", "hard_001"]:
            r = requests.post(f"{base_url}/grader",
                              json={"patient_id": pid, "esi_level": 2, "department": "Emergency"},
                              timeout=15)
            results.append(check(f"  /grader works for {pid}", r.status_code == 200))
    except Exception as e:
        results.append(check("/grader", False, str(e)))

    # 7. Baseline
    print("\n[7] /baseline")
    try:
        r = requests.get(f"{base_url}/baseline", timeout=60)
        results.append(check("GET /baseline returns 200", r.status_code == 200))
        data = r.json()
        results.append(check("  has average_score", "average_score" in data))
        results.append(check("  has task_scores", "task_scores" in data))
        avg = data.get("average_score", -1)
        results.append(check("  average_score in [0.0, 1.0]", 0.0 <= avg <= 1.0, f"avg={avg}"))
        results.append(check("  covers 3 difficulties", len(data.get("task_scores", {})) >= 3))
    except Exception as e:
        results.append(check("GET /baseline", False, str(e)))

    # 8. Undertriage penalty check
    print("\n[8] Undertriage penalty (critical safety check)")
    try:
        # ESI-1 patient assigned ESI-5 should score very low
        r = requests.post(f"{base_url}/grader",
                          json={"patient_id": "easy_003", "esi_level": 5, "department": "General"},
                          timeout=15)
        score = r.json().get("total_score", 1.0)
        results.append(check(
            "  ESI-1 patient assigned ESI-5 scores < 0.2",
            score < 0.2,
            f"score={score} — undertriage penalty applied correctly"
        ))
    except Exception as e:
        results.append(check("Undertriage penalty check", False, str(e)))

    # Summary
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*56}")
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed — safe to submit!")
    else:
        print(f"{total - passed} check(s) FAILED — fix before submitting.")
    print("="*56)
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()
    success = validate(args.url)
    sys.exit(0 if success else 1)
