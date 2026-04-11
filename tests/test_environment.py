"""
Test suite for Medical Triage OpenEnv environment.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import TriageAction, TriageReward, TriageObservation
from patients import PATIENTS, PATIENT_MAP, EASY_CASES, MEDIUM_CASES, HARD_CASES, DEPARTMENTS
from server.triage_environment import (
    MedicalTriageEnvironment, score_triage,
    compute_esi_score, compute_department_score,
)


@pytest.fixture
def env():
    return MedicalTriageEnvironment()

@pytest.fixture
def easy_patient():
    return PATIENT_MAP["easy_001"]   # Classic MI

@pytest.fixture
def hard_patient():
    return PATIENT_MAP["hard_005"]   # Aortic dissection


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModels:
    def test_action_valid(self):
        a = TriageAction(esi_level=2, department="Emergency")
        assert a.esi_level == 2

    def test_action_esi_out_of_range(self):
        with pytest.raises(Exception):
            TriageAction(esi_level=6, department="Emergency")

    def test_action_esi_zero_invalid(self):
        with pytest.raises(Exception):
            TriageAction(esi_level=0, department="Emergency")

    def test_reward_range(self):
        r = TriageReward(value=0.8, esi_score=0.9, department_score=0.6, breakdown={})
        assert 0.0 <= r.value <= 1.0


# ── Scoring tests ─────────────────────────────────────────────────────────────

class TestESIScoring:

    def test_perfect_esi_score(self, easy_patient):
        # easy_001 correct ESI = 2
        score, penalty, fb = compute_esi_score(easy_patient, 2)
        assert score == 1.0
        assert penalty == 0.0

    def test_one_level_off_partial_credit(self, easy_patient):
        score, penalty, fb = compute_esi_score(easy_patient, 3)
        assert 0.0 < score < 1.0

    def test_dangerous_undertriage_penalty(self, easy_patient):
        # ESI-1/2 patient assigned ESI-4 = dangerous
        score, penalty, fb = compute_esi_score(easy_patient, 4)
        assert penalty >= 0.3
        assert any("DANGEROUS" in f for f in fb)

    def test_esi1_patient_undertriage(self):
        patient = PATIENT_MAP["easy_003"]   # opioid OD, ESI-1
        score, penalty, fb = compute_esi_score(patient, 3)
        assert penalty >= 0.15

    def test_correct_esi1(self):
        patient = PATIENT_MAP["easy_003"]
        score, penalty, fb = compute_esi_score(patient, 1)
        assert score == 1.0
        assert penalty == 0.0

    def test_scoring_deterministic(self, easy_patient):
        scores = [compute_esi_score(easy_patient, 2)[0] for _ in range(5)]
        assert len(set(scores)) == 1

    def test_esi5_patient_no_undertriage_penalty(self):
        patient = PATIENT_MAP["easy_004"]   # paper cut, ESI-5
        score, penalty, fb = compute_esi_score(patient, 4)
        assert penalty == 0.0   # ESI-5 patient assigned ESI-4 is overtriage not undertriage


class TestDepartmentScoring:

    def test_correct_department_full_score(self, easy_patient):
        score, fb = compute_department_score(easy_patient, "Emergency")
        assert score == 1.0

    def test_acceptable_department_partial(self, easy_patient):
        score, fb = compute_department_score(easy_patient, "Cardiology")
        assert 0.5 < score < 1.0

    def test_wrong_department_zero(self, easy_patient):
        score, fb = compute_department_score(easy_patient, "Orthopedics")
        assert score == 0.0

    def test_unknown_department_zero(self, easy_patient):
        score, fb = compute_department_score(easy_patient, "MadeUpDept")
        assert score == 0.0
        assert any("Unknown" in f or "Valid" in f for f in fb)

    def test_case_insensitive_matching(self, easy_patient):
        score1, _ = compute_department_score(easy_patient, "Emergency")
        score2, _ = compute_department_score(easy_patient, "emergency")
        assert score1 == score2


class TestCompositeScoring:

    def test_perfect_score(self, easy_patient):
        action = TriageAction(esi_level=2, department="Emergency")
        total, reward, fb = score_triage(easy_patient, action)
        assert total == 1.0
        assert reward.value == 1.0

    def test_partial_score_composition(self, easy_patient):
        # ESI correct (1.0), dept suboptimal
        action = TriageAction(esi_level=2, department="Orthopedics")
        total, reward, fb = score_triage(easy_patient, action)
        assert reward.esi_score == 1.0
        assert reward.department_score == 0.0
        assert abs(total - 0.6) < 0.01   # 60% ESI + 40% dept = 0.6

    def test_undertriage_reduces_total(self):
        patient = PATIENT_MAP["easy_001"]   # ESI-2
        action = TriageAction(esi_level=4, department="Emergency")
        total, reward, fb = score_triage(patient, action)
        assert reward.undertriage_penalty > 0
        assert total < 0.3


# ── Environment lifecycle tests ───────────────────────────────────────────────

class TestLifecycle:

    def test_reset_returns_observation(self, env):
        obs = env.reset("easy")
        assert isinstance(obs, TriageObservation)
        assert obs.difficulty == "easy"
        assert obs.step == 0
        assert obs.done is False

    def test_reset_randomizes_patient(self, env):
        # Run 10 resets and check we don't always get the same patient
        patients = set()
        for _ in range(10):
            obs = env.reset("easy")
            patients.add(obs.patient_id)
        assert len(patients) > 1, "Reset should randomize patient selection"

    def test_step_increments_counter(self, env):
        env.reset("easy")
        for i in range(1, 4):
            action = TriageAction(esi_level=3, department="General")
            obs, _, _, _ = env.step(action)
            assert obs.step == i

    def test_episode_ends_at_max_steps(self, env):
        env.reset("easy")
        done = False
        for _ in range(4):
            action = TriageAction(esi_level=3, department="General")
            _, _, done, _ = env.step(action)
        assert done is True

    def test_episode_ends_on_high_score(self, env):
        env.reset("easy")
        env.current_patient = PATIENT_MAP["easy_001"]
        action = TriageAction(esi_level=2, department="Emergency")
        _, reward, done, _ = env.step(action)
        if reward.value >= 0.9:
            assert done is True

    def test_step_after_done_error(self, env):
        env.reset("easy")
        env.done = True
        action = TriageAction(esi_level=2, department="Emergency")
        _, _, done, info = env.step(action)
        assert done is True
        assert "error" in info

    def test_state_fields(self, env):
        env.reset("easy")
        s = env.state()
        for field in ["episode_id", "task_id", "patient_id", "step", "done", "best_score"]:
            assert field in s

    def test_reset_clears_state(self, env):
        env.reset("easy")
        env.step(TriageAction(esi_level=2, department="Emergency"))
        env.reset("hard")
        s = env.state()
        assert s["step"] == 0
        assert s["best_score"] == 0.0
        assert s["history"] == []

    def test_hint_appears_after_step(self, env):
        env.reset("hard")
        obs0 = env.reset("hard")
        assert obs0.hint is None
        action = TriageAction(esi_level=5, department="General")
        env.step(action)
        obs2, _, _, _ = env.step(action)
        assert obs2.hint is not None

    def test_three_tasks_available(self, env):
        tasks = env.get_tasks()
        assert len(tasks) == 3
        ids = {t.task_id for t in tasks}
        assert ids == {"easy", "medium", "hard"}


# ── Patient bank tests ────────────────────────────────────────────────────────

class TestPatientBank:

    def test_18_patients_total(self):
        assert len(PATIENTS) == 18

    def test_6_per_difficulty(self):
        assert len(EASY_CASES) == 6
        assert len(MEDIUM_CASES) == 6
        assert len(HARD_CASES) == 6

    def test_all_patients_have_required_fields(self):
        required = ["id", "difficulty", "presentation", "vitals",
                    "correct_esi", "correct_department",
                    "esi_partial_credit", "department_scores", "reasoning"]
        for p in PATIENTS:
            for f in required:
                assert f in p, f"Patient {p['id']} missing field: {f}"

    def test_all_esi_levels_in_credit_tables(self):
        for p in PATIENTS:
            for lvl in [1, 2, 3, 4, 5]:
                assert lvl in p["esi_partial_credit"], \
                    f"Patient {p['id']} missing ESI-{lvl} in partial credit table"

    def test_correct_esi_always_scores_highest(self):
        for p in PATIENTS:
            correct = p["correct_esi"]
            correct_score = p["esi_partial_credit"][correct]
            for lvl, score in p["esi_partial_credit"].items():
                assert score <= correct_score, \
                    f"Patient {p['id']}: ESI-{lvl} scores {score} > correct ESI-{correct} score {correct_score}"

    def test_all_departments_valid(self):
        for p in PATIENTS:
            assert p["correct_department"] in DEPARTMENTS, \
                f"Patient {p['id']} correct dept '{p['correct_department']}' not in DEPARTMENTS"

    def test_scoring_deterministic_all_patients(self):
        for p in PATIENTS:
            action = TriageAction(esi_level=p["correct_esi"], department=p["correct_department"])
            scores = [score_triage(p, action)[0] for _ in range(3)]
            assert len(set(scores)) == 1, f"Non-deterministic scoring for {p['id']}"

    def test_correct_answer_always_highest_score(self):
        for p in PATIENTS:
            correct_action = TriageAction(esi_level=p["correct_esi"], department=p["correct_department"])
            correct_score, _, _ = score_triage(p, correct_action)
            for esi in [1, 2, 3, 4, 5]:
                for dept in [p["correct_department"], "General"]:
                    action = TriageAction(esi_level=esi, department=dept)
                    score, _, _ = score_triage(p, action)
                    assert score <= correct_score + 0.01, \
                        f"Patient {p['id']}: ESI-{esi}/{dept} scored {score} > correct {correct_score}"
