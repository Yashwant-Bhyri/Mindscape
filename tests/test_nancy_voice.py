import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend_api.nancy_voice import _build_system_prompt, _build_voice_session_plan, _persist_voice_session
from clinic_state import get_async_alerts, get_nancy_interactions
from product_data import get_doctor, get_patient


class NancyVoiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        runtime_file = Path(self.temp_dir.name) / "clinic_state.json"
        os.environ["MINDSCAPE_RUNTIME_FILE"] = str(runtime_file)
        self.doctor_id = "amira-khan"
        self.patient_id = "nadia-s"
        self.doctor = get_doctor(self.doctor_id)
        self.patient = get_patient(self.doctor_id, self.patient_id)

    def tearDown(self):
        os.environ.pop("MINDSCAPE_RUNTIME_FILE", None)
        self.temp_dir.cleanup()

    def test_build_voice_session_plan_includes_tool_handoff_notes(self):
        plan = _build_voice_session_plan(
            doctor=self.doctor,
            patient=self.patient,
            user_turns=["I've been sleeping four hours and feeling panicky all week."],
            tool_events=[
                {"submitted_checkin": True, "created_alert": True, "logged_statement": True},
            ],
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertIn("structured daily check-in", plan["clinician_summary"])
        self.assertIn("doctor alert", plan["clinician_summary"].lower())
        self.assertEqual(plan["conversation_goal"], "live voice daily check-in")
        self.assertEqual(plan["recommended_follow_up"], "Review the voice-session alert and transcript alongside the structured handoff.")
        self.assertEqual(plan["safety_note"], "Voice session triggered live escalation.")
        self.assertIn(plan["escalation_level"], {"watch", "urgent"})

    def test_persist_voice_session_creates_single_touchpoint_without_duplicate_alert(self):
        _persist_voice_session(
            doctor=self.doctor,
            patient=self.patient,
            user_turns=["My anxiety has been worse at night, but I'm safe right now."],
            tool_events=[{"created_alert": True, "submitted_checkin": False, "logged_statement": False}],
        )

        interactions = get_nancy_interactions(self.doctor_id, self.patient_id)
        alerts = get_async_alerts(self.doctor_id)

        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]["mode"], "voice session")
        self.assertEqual(len(alerts), 0)

    def test_voice_prompts_separate_daily_from_general_support(self):
        daily = _build_system_prompt(self.patient, self.doctor, "Patient context", mode="daily")
        support = _build_system_prompt(self.patient, self.doctor, "Patient context", mode="support")

        self.assertIn("VOICE MODE: DAILY CONVERSATION", daily)
        self.assertIn("call submit_daily_checkin exactly once", daily)
        self.assertIn("not a survey", daily)
        self.assertIn("VOICE MODE: GENERAL SUPPORT", support)
        self.assertIn("Do not run the daily questionnaire here", support)
        self.assertIn("Speak only in English", daily)
        self.assertIn("Speak only in English", support)


if __name__ == "__main__":
    unittest.main()
