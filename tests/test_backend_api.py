import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class BackendApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        runtime_file = Path(self.temp_dir.name) / "clinic_state.json"
        os.environ["MINDSCAPE_RUNTIME_FILE"] = str(runtime_file)

        from backend_api.main import app

        self.client = TestClient(app)
        self.doctor_id = "amira-khan"
        self.patient_id = "nadia-s"

    def tearDown(self):
        os.environ.pop("MINDSCAPE_RUNTIME_FILE", None)
        self.temp_dir.cleanup()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_doctor_gateway_payload(self):
        response = self.client.get("/api/doctors")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("doctors", payload)
        self.assertGreaterEqual(len(payload["doctors"]), 1)
        self.assertIn("patient_count", payload["doctors"][0])

    def test_patient_gateway_payload(self):
        response = self.client.get("/api/patient-access")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("doctors", payload)
        self.assertGreaterEqual(len(payload["doctors"]), 1)
        self.assertGreaterEqual(len(payload["doctors"][0]["patients"]), 1)
        self.assertIn("daily_nancy_due", payload["doctors"][0]["patients"][0])

    def test_patient_payload(self):
        response = self.client.get(f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["patient"]["id"], self.patient_id)
        self.assertIn("async_summary", payload)
        self.assertIn("timeline", payload)

    def test_daily_checkin_creates_snapshot(self):
        response = self.client.post(
            f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/checkins",
            json={
                "mood_score": "3",
                "anxiety_score": "8",
                "sleep_hours": "4",
                "daily_update": "Felt overwhelmed and isolated today.",
                "clinical_summary": "Lower mood, ruminative thinking, reduced concentration.",
                "safety_concerns": "",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("result", payload)
        self.assertIn("snapshot", payload)
        self.assertGreaterEqual(payload["snapshot"]["async_summary"]["message_count"], 1)

    def test_nancy_route_payload(self):
        response = self.client.get(f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/nancy")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("settings_preview", payload)
        self.assertIn("daily_nancy_status", payload)
        self.assertEqual(payload["patient"]["id"], self.patient_id)

    def test_patient_message_to_doctor_triggers_nancy_reply_and_handoff(self):
        response = self.client.post(
            f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/messages",
            json={
                "body": "I slept badly and I want Dr. Khan to know I feel more anxious today.",
                "recipient": "doctor",
                "sender_role": "patient",
            },
        )
        self.assertEqual(response.status_code, 200)
        snapshot = response.json()["snapshot"]["patient"]
        messages = snapshot["patient_messages"]
        self.assertTrue(any(item["sender_role"] == "nancy" and item["recipient"] == "patient" for item in messages))
        self.assertTrue(any(item["sender_role"] == "nancy" and item["recipient"] == "doctor" for item in messages))
        self.assertGreaterEqual(len(snapshot["nancy_interactions"]), 1)

    def test_touchpoint_relay_creates_patient_visible_message(self):
        response = self.client.post(
            f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/nancy/touchpoints",
            json={
                "mode": "doctor directive",
                "patient_report": "Check in about sleep and panic anticipation.",
                "patient_message": "Hi, this is Nancy. Dr. Khan asked me to check in about your sleep and panic anticipation.",
                "clinician_summary": "Doctor requested a patient-facing Nancy relay about sleep and panic anticipation.",
                "relay_to_patient": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNotNone(payload["relayed_message"])

        companion = self.client.get(f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/companion")
        self.assertEqual(companion.status_code, 200)
        companion_payload = companion.json()
        self.assertGreaterEqual(len(companion_payload["patient_inbox"]), 1)
        self.assertIn("sleep and panic anticipation", companion_payload["patient_inbox"][0]["body"])
        self.assertGreaterEqual(len(companion_payload["patient_nancy_updates"]), 1)
        self.assertTrue(companion_payload["patient_nancy_updates"][0]["relayed_to_patient"])

    def test_patient_companion_hides_clinician_only_touchpoint(self):
        response = self.client.post(
            f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/nancy/touchpoints",
            json={
                "mode": "doctor note",
                "patient_report": "",
                "clinician_summary": "Clinician-only note about adherence concern.",
                "relay_to_patient": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        companion = self.client.get(f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/companion")
        self.assertEqual(companion.status_code, 200)
        companion_payload = companion.json()
        nancy_items = [item for item in companion_payload["timeline"] if item["kind"] == "nancy"]
        self.assertEqual(nancy_items, [])

    @patch("backend_api.main._nancy_llm_call")
    def test_clinician_nancy_chat_creates_structured_actions(self, mock_llm):
        mock_llm.return_value = """
        {
          "response": "I saved that as a Nancy directive and a structured doctor handoff.",
          "actions": [
            {
              "type": "add_nancy_task",
              "data": {
                "title": "Nightly grounding check",
                "instructions": "Ask about grounding practice each evening and encourage completion without pressure.",
                "category": "Recovery",
                "due_label": "Review next visit"
              }
            },
            {
              "type": "add_nancy_touchpoint",
              "data": {
                "mode": "doctor directive",
                "conversation_goal": "Monitor sleep and panic anticipation",
                "patient_report": "Doctor reported worsening sleep and panic anticipation.",
                "patient_message": "Hi, this is Nancy. Dr. Khan asked me to check in about your sleep and panic anticipation.",
                "clinician_summary": "Doctor wants Nancy to monitor worsening sleep and panic anticipation.",
                "observed_mood": "monitoring",
                "functioning_note": "Sleep disruption may be affecting mornings.",
                "cognition_note": "No new cognition update.",
                "medication_note": "No medication change requested.",
                "safety_note": "",
                "recommended_follow_up": "Nancy should ask about sleep, panic anticipation, and grounding completion.",
                "escalation_level": "watch",
                "relay_to_patient": true
              }
            }
          ]
        }
        """

        response = self.client.post(
            f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/nancy/clinician-chat",
            json={
                "message": "Please have Nancy check sleep nightly and keep an eye on panic anticipation.",
                "history": [],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["actions_taken"]), 2)
        self.assertEqual(payload["actions_taken"][0]["type"], "add_nancy_task")
        self.assertEqual(payload["actions_taken"][1]["type"], "add_nancy_touchpoint")
        self.assertIsNotNone(payload["actions_taken"][1]["message_id"])

        refreshed = self.client.get(f"/api/doctors/{self.doctor_id}/patients/{self.patient_id}/nancy")
        self.assertEqual(refreshed.status_code, 200)
        snapshot = refreshed.json()["patient"]
        self.assertEqual(snapshot["nancy_tasks"][0]["title"], "Nightly grounding check")
        self.assertIn("panic anticipation", snapshot["nancy_interactions"][0]["clinician_summary"])
        self.assertTrue(snapshot["nancy_interactions"][0]["relayed_to_patient"])

    def test_forum_save_toggle_updates_thread_flags(self):
        research = self.client.get(f"/api/doctors/{self.doctor_id}/research").json()
        threads = research["forum_threads"]
        self.assertGreaterEqual(len(threads), 1)
        tid = threads[0]["id"]

        r1 = self.client.post(f"/api/doctors/{self.doctor_id}/research/threads/{tid}/save")
        self.assertEqual(r1.status_code, 200)
        self.assertTrue(r1.json()["saved"])
        t1 = next(x for x in r1.json()["research"]["forum_threads"] if x["id"] == tid)
        self.assertTrue(t1.get("saved"))

        r2 = self.client.post(f"/api/doctors/{self.doctor_id}/research/threads/{tid}/save")
        self.assertEqual(r2.status_code, 200)
        self.assertFalse(r2.json()["saved"])
        t2 = next(x for x in r2.json()["research"]["forum_threads"] if x["id"] == tid)
        self.assertFalse(t2.get("saved"))

    @patch("backend_api.main._nancy_llm_call")
    def test_doctor_insights_chat_returns_plain_response(self, mock_llm):
        mock_llm.return_value = "You have two open alerts; start with the urgent sleep-related callback."

        response = self.client.post(
            f"/api/doctors/{self.doctor_id}/doctor-insights/chat",
            json={"message": "What should I prioritize today?", "history": []},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("urgent", payload["response"].lower())
        mock_llm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
