import os
import tempfile
import unittest

from session_service import (
    build_patient_timeline,
    build_session_prep,
    result_to_view_model,
    save_clinician_session_note,
)


DOCTOR = {
    "id": "doctor-session",
    "name": "Dr. Session",
}

PATIENT = {
    "id": "patient-session",
    "name": "Patient Session",
    "diagnosis": "PTSD",
    "care_plan": "Trauma-focused CBT",
    "next_appointment": "May 12, 2026 at 09:30",
    "history": "Three-month recovery arc.",
    "last_update": "Improving sleep.",
    "last_daily_checkin_summary": "Mood lower after trigger-heavy commute.",
    "last_session_summary": "Reduced avoidance but persistent hypervigilance.",
    "last_nancy_summary": "Nancy flagged worsening sleep and panic anticipation.",
    "alerts": ["Monitor sensory-trigger spikes."],
    "async_alerts": [
        {"id": "a1", "severity": "urgent", "status": "new", "created_at": "2026-05-10T10:00:00", "summary": "Urgent concern"}
    ],
    "session_notes": [
        {"id": "note-1", "created_at": "2026-05-10T11:00:00", "title": "Review note", "note": "Clinician interpretation"}
    ],
}


class SessionServiceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["MINDSCAPE_RUNTIME_FILE"] = os.path.join(self.tempdir.name, "clinic_state.json")

    def tearDown(self):
        os.environ.pop("MINDSCAPE_RUNTIME_FILE", None)
        self.tempdir.cleanup()

    def test_build_session_prep_includes_async_alert_counts(self):
        prep = build_session_prep(DOCTOR, PATIENT)
        self.assertIsNotNone(prep)
        self.assertEqual(prep.open_async_alerts, 1)
        self.assertEqual(prep.urgent_async_alerts, 1)
        self.assertEqual(prep.patient_name, "Patient Session")

    def test_result_to_view_model_formats_session_payload(self):
        result = {
            "bsv": {"valence": -0.2, "arousal": 0.7, "dominance": 0.4},
            "hypothesis": {"name": "PTSD", "confidence": "Moderate", "evidence": ["flashbacks", "hypervigilance"]},
            "reasoning": "Clinical summary",
            "retrieved_evidence": ["[DSM] PTSD criteria"],
            "follow_up": ["Ask about nightmares"],
            "treatment_plan": "Trauma-focused CBT",
            "reference_cases": [{"title": "Case A"}],
            "traumatic_markers": ["bus commute disclosure"],
            "emotion_trajectory": [{"phase": "Middle", "dominant_emotion": "Fear"}],
            "safety_gate": "PASS",
            "visual_bsv": {"face_detected": True, "blink_rate_per_min": 22, "active_aus": ["AU4"]},
        }
        view = result_to_view_model("hello", "OpenAI Whisper", result)
        self.assertEqual(view.hypothesis_name, "PTSD")
        self.assertIn("Grounded Medical Evidence", view.hypothesis_evidence)
        self.assertTrue(view.visual_bsv_face_detected)

    def test_save_clinician_session_note_persists(self):
        note = save_clinician_session_note(
            DOCTOR,
            {"id": "patient-session", "name": "Patient Session"},
            "Post-session note",
            "Patient tolerated the session with guarded affect.",
            "Maintain weekly therapy cadence.",
            "Continue current plan",
        )
        self.assertIsNotNone(note)
        self.assertEqual(note["title"], "Post-session note")

    def test_session_timeline_includes_session_note_entries(self):
        timeline = build_patient_timeline(PATIENT)
        kinds = [entry.kind for entry in timeline]
        self.assertIn("session_note", kinds)


if __name__ == "__main__":
    unittest.main()
