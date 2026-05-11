import os
import tempfile
import unittest

from async_care_service import (
    build_async_care_summary,
    build_daily_nancy_status,
    get_doctor_alerts,
    process_daily_checkin,
    process_patient_message,
    update_alert_status,
)
from clinic_state import get_async_alerts, get_daily_checkins, get_nancy_interactions, get_patient_messages


DOCTOR = {
    "id": "doctor-test",
    "name": "Dr. Test",
    "hospital_name": "Test Hospital",
    "department_name": "Mood Unit",
}

PATIENT = {
    "id": "patient-test",
    "name": "Patient Test",
    "diagnosis": "PTSD with panic symptoms",
    "risk": "Medium",
    "care_plan": "Weekly review",
    "nancy_tasks": [{"title": "Grounding practice", "instructions": "Use grounding audio nightly."}],
    "daily_checkins": [],
    "nancy_interactions": [],
    "patient_messages": [],
    "async_alerts": [],
}


class AsyncCareServiceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["MINDSCAPE_RUNTIME_FILE"] = os.path.join(self.tempdir.name, "clinic_state.json")
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("DEEPSEEK_API_KEY", None)

    def tearDown(self):
        os.environ.pop("MINDSCAPE_RUNTIME_FILE", None)
        self.tempdir.cleanup()

    def test_daily_checkin_creates_review_alert_when_signals_are_high(self):
        result = process_daily_checkin(
            DOCTOR,
            PATIENT,
            {
                "mood_score": "2",
                "anxiety_score": "9",
                "sleep_hours": "3",
                "cognition_score": "4",
                "memory_score": "4",
                "functioning_score": "3",
                "medication_adherence": "70",
                "daily_update": "I had a panic attack and feel overwhelmed.",
                "significant_events": "Flashback after commute.",
                "clinical_summary": "Poor concentration and increasing avoidance.",
            },
        )
        self.assertEqual(result["plan"]["escalation_level"], "watch")
        self.assertIsNotNone(result["alert"])
        alerts = get_async_alerts(DOCTOR["id"], PATIENT["id"])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["status"], "new")

    def test_patient_message_to_doctor_can_open_urgent_alert(self):
        result = process_patient_message(
            DOCTOR,
            PATIENT,
            "I do not feel safe and I think about killing myself.",
            "doctor",
        )
        self.assertIsNotNone(result["alert"])
        self.assertEqual(result["alert"]["severity"], "urgent")
        self.assertIsNotNone(result["interaction"])

        messages = get_patient_messages(DOCTOR["id"], PATIENT["id"])
        self.assertTrue(any(message["sender_role"] == "nancy" and message["recipient"] == "patient" for message in messages))
        self.assertTrue(any(message["sender_role"] == "nancy" and message["recipient"] == "doctor" for message in messages))
        self.assertGreaterEqual(len(get_nancy_interactions(DOCTOR["id"], PATIENT["id"])), 1)

    def test_daily_status_is_due_until_checkin_completed_today(self):
        self.assertTrue(build_daily_nancy_status(PATIENT)["due"])
        process_daily_checkin(
            DOCTOR,
            PATIENT,
            {
                "mood_score": "5",
                "anxiety_score": "4",
                "sleep_hours": "7",
                "cognition_score": "6",
                "memory_score": "6",
                "functioning_score": "6",
                "medication_adherence": "100",
                "daily_update": "A steady day.",
            },
        )
        patient_with_checkins = {
            **PATIENT,
            "daily_checkins": get_daily_checkins(DOCTOR["id"], PATIENT["id"]),
        }
        status = build_daily_nancy_status(patient_with_checkins)
        self.assertFalse(status["due"])
        self.assertTrue(status["completed_today"])

    def test_alert_status_can_be_acknowledged_and_resolved(self):
        result = process_patient_message(
            DOCTOR,
            PATIENT,
            "I do not feel safe and I think about killing myself.",
            "doctor",
        )
        alert_id = result["alert"]["id"]
        updated = update_alert_status(DOCTOR["id"], alert_id, "acknowledged")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "acknowledged")
        resolved = update_alert_status(DOCTOR["id"], alert_id, "resolved")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.status, "resolved")

    def test_summary_counts_open_and_urgent_alerts(self):
        process_patient_message(
            DOCTOR,
            PATIENT,
            "I do not feel safe and I think about killing myself.",
            "doctor",
        )
        summary = build_async_care_summary(
            {
                **PATIENT,
                "patient_messages": [{"sender_role": "patient", "body": "I do not feel safe and I think about killing myself."}],
                "async_alerts": get_async_alerts(DOCTOR["id"], PATIENT["id"]),
            },
            DOCTOR,
        )
        self.assertEqual(summary.open_alerts, 1)
        self.assertEqual(summary.urgent_alerts, 1)
        self.assertEqual(len(get_doctor_alerts(DOCTOR["id"])), 1)


if __name__ == "__main__":
    unittest.main()
