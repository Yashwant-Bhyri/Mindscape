import os
from dataclasses import asdict
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from clinic_state import add_session_note, add_session_record, get_session_notes
from session_models import SessionAnalysisView, SessionPrepBrief, TimelineEntry


def build_session_prep(doctor: dict, patient: dict | None) -> SessionPrepBrief | None:
    if not patient:
        return None

    async_alerts = patient.get("async_alerts", []) or []
    open_alerts = [alert for alert in async_alerts if alert.get("status") != "resolved"]
    urgent_alerts = [alert for alert in open_alerts if alert.get("severity") == "urgent"]

    return SessionPrepBrief(
        patient_name=patient.get("name", "Unknown"),
        diagnosis=patient.get("diagnosis", "Unknown"),
        care_plan=patient.get("care_plan", "None"),
        next_appointment=patient.get("next_appointment", "Unknown"),
        history_summary=patient.get("history", ""),
        recent_medical_summary=patient.get("last_update", ""),
        latest_questionnaire_insight=patient.get("last_daily_checkin_summary", "No questionnaire yet."),
        last_consultation_insight=patient.get("last_session_summary", "No prior recorded session."),
        latest_nancy_handoff=patient.get("last_nancy_summary", "No Nancy update yet."),
        open_async_alerts=len(open_alerts),
        urgent_async_alerts=len(urgent_alerts),
        highlighted_alerts=(patient.get("alerts", []) or [])[:4],
    )


def result_to_view_model(transcript: str, stt_provider: str, result: dict) -> SessionAnalysisView:
    bsv = result.get("bsv", {})
    hypothesis = result.get("hypothesis", {})
    evidence = hypothesis.get("evidence", [])
    grounded_evidence = result.get("retrieved_evidence", [])

    if isinstance(evidence, list):
        final_evidence_text = "\n- " + "\n- ".join(evidence) if evidence else ""
    else:
        final_evidence_text = str(evidence)
    if grounded_evidence:
        final_evidence_text += "\n\n**Grounded Medical Evidence:**\n- " + "\n- ".join(grounded_evidence)

    visual_bsv = result.get("visual_bsv", {})

    return SessionAnalysisView(
        transcript=transcript,
        stt_provider=stt_provider,
        bsv_valence=float(bsv.get("valence", 0.0)),
        bsv_arousal=float(bsv.get("arousal", 0.0)),
        bsv_dominance=float(bsv.get("dominance", 0.0)),
        hypothesis_name=hypothesis.get("name", "Unknown"),
        hypothesis_confidence=hypothesis.get("confidence", ""),
        hypothesis_reasoning=result.get("reasoning", ""),
        hypothesis_evidence=final_evidence_text,
        retrieved_evidence=result.get("retrieved_evidence", []),
        follow_up_questions=result.get("follow_up", []),
        treatment_plan=result.get("treatment_plan", ""),
        reference_cases=result.get("reference_cases", []),
        traumatic_markers=result.get("traumatic_markers", []),
        emotion_trajectory=result.get("emotion_trajectory", []),
        safety_gate=result.get("safety_gate", "FAIL"),
        visual_bsv_facial_valence=float(visual_bsv.get("facial_valence", 0.0)),
        visual_bsv_facial_arousal=float(visual_bsv.get("facial_arousal", 0.0)),
        visual_bsv_blink_rate=float(visual_bsv.get("blink_rate_per_min", 0.0)),
        visual_bsv_flat_affect=float(visual_bsv.get("flat_affect_score", 0.0)),
        visual_bsv_gaze_stability=float(visual_bsv.get("gaze_stability", 1.0)),
        visual_bsv_twitch_zones=visual_bsv.get("twitch_zones", []),
        visual_bsv_face_detected=bool(visual_bsv.get("face_detected", False)),
        visual_bsv_active_aus=visual_bsv.get("active_aus", []),
    )


def persist_session_result(doctor: dict, patient: dict | None, transcript: str, stt_provider: str, result: dict) -> dict | None:
    if not patient:
        return None
    return add_session_record(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        transcript=transcript,
        transcription_provider=stt_provider,
        result=result,
    )


def save_clinician_session_note(
    doctor: dict,
    patient: dict | None,
    title: str,
    note: str,
    plan_update: str = "",
    disposition: str = "",
) -> dict | None:
    if not patient or not note.strip():
        return None
    return add_session_note(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        title=title,
        note=note,
        plan_update=plan_update,
        disposition=disposition,
    )


def load_session_notes(doctor_id: str, patient_id: str | None) -> list[dict]:
    if not patient_id:
        return []
    return get_session_notes(doctor_id, patient_id)


def build_patient_timeline(patient: dict) -> list[TimelineEntry]:
    timeline: list[TimelineEntry] = []
    for checkin in patient.get("daily_checkins", []):
        timeline.append(TimelineEntry(kind="checkin", created_at=checkin.get("created_at", ""), payload=checkin))
    for interaction in patient.get("nancy_interactions", []):
        timeline.append(TimelineEntry(kind="nancy", created_at=interaction.get("created_at", ""), payload=interaction))
    for message in patient.get("patient_messages", []):
        timeline.append(TimelineEntry(kind="message", created_at=message.get("created_at", ""), payload=message))
    for session_record in patient.get("session_records", []):
        timeline.append(TimelineEntry(kind="session", created_at=session_record.get("created_at", ""), payload=session_record))
    for session_note in patient.get("session_notes", []):
        timeline.append(TimelineEntry(kind="session_note", created_at=session_note.get("created_at", ""), payload=session_note))
    for event in patient.get("sos_events", []):
        timeline.append(TimelineEntry(kind="sos", created_at=event.get("created_at", ""), payload=event))
    for outreach in patient.get("outreach_logs", []):
        timeline.append(TimelineEntry(kind="outreach", created_at=outreach.get("created_at", ""), payload=outreach))
    for alert in patient.get("async_alerts", []):
        timeline.append(TimelineEntry(kind="alert", created_at=alert.get("created_at", ""), payload=alert))
    return sorted(timeline, key=lambda item: item.created_at, reverse=True)


def summary_to_dict(summary) -> dict:
    return asdict(summary)


def _session_full_diagnosis_enabled() -> bool:
    """Heavy MindScape engine (STT + retrieval + LLM stack). Off by default to avoid OOM on Session."""
    return os.getenv("MINDSCAPE_SESSION_FULL_DIAGNOSIS", "").lower() in ("1", "true", "yes")


def _stub_session_analysis_result(filepath: str, patient_context: str | None = None) -> tuple[str, str, dict]:
    label = os.path.basename(filepath) or "session audio"
    transcript = (
        f"[Lightweight session mode] Received «{label}». "
        "Full audio diagnosis (SenseVoice, retrieval, multimodal stack) is disabled unless you set "
        "MINDSCAPE_SESSION_FULL_DIAGNOSIS=1 on the API server."
    )
    if patient_context and patient_context.strip():
        transcript += " (Patient context was supplied but not processed in lightweight mode.)"
    result: dict = {
        "bsv": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        "hypothesis": {
            "name": "Analysis deferred",
            "confidence": "N/A",
            "evidence": ["Enable MINDSCAPE_SESSION_FULL_DIAGNOSIS=1 to run the full pipeline."],
        },
        "reasoning": transcript,
        "retrieved_evidence": [],
        "follow_up": [],
        "treatment_plan": "Continue clinical assessment in-chart; re-run analysis with full diagnosis enabled if needed.",
        "reference_cases": [],
        "traumatic_markers": [],
        "emotion_trajectory": [],
        "safety_gate": "PASS",
        "visual_bsv": {
            "facial_valence": 0.0,
            "facial_arousal": 0.0,
            "blink_rate_per_min": 0.0,
            "flat_affect_score": 0.0,
            "gaze_stability": 1.0,
            "twitch_zones": [],
            "face_detected": False,
            "active_aus": [],
        },
    }
    return transcript, "stub", result


def analyze_audio_file(filepath: str, patient_context: str | None = None):
    if not _session_full_diagnosis_enabled():
        transcript, provider, result = _stub_session_analysis_result(filepath, patient_context=patient_context)
        yield {"phase": "transcribed", "transcript": transcript, "provider": provider}
        yield {
            "phase": "completed",
            "transcript": transcript,
            "provider": provider,
            "result": result,
            "view_model": result_to_view_model(transcript, provider, result).to_dict(),
        }
        return

    import mindscape_engine  # noqa: PLC0415 — defer heavy stack until explicitly enabled

    transcript, provider = mindscape_engine.transcribe_audio(filepath)
    yield {"phase": "transcribed", "transcript": transcript, "provider": provider}

    result = {}
    for update in mindscape_engine.get_diagnosis(transcript, filepath, patient_context=patient_context):
        if "result" in update:
            result = update["result"]
        yield update

    yield {
        "phase": "completed",
        "transcript": transcript,
        "provider": provider,
        "result": result,
        "view_model": result_to_view_model(transcript, provider, result).to_dict(),
    }


def analyze_uploaded_file(file_obj: BinaryIO, filename: str, patient_context: str | None = None):
    suffix = os.path.splitext(filename or "")[1] or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix, prefix="mindscape-upload-") as temp_file:
        temp_file.write(file_obj.read())
        temp_path = temp_file.name

    try:
        for update in analyze_audio_file(temp_path, patient_context=patient_context):
            yield update
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def analyze_live_capture(duration: int = 5, patient_context: str | None = None):
    import mindscape_engine  # noqa: PLC0415

    filepath = mindscape_engine.record_audio(duration=duration)
    try:
        for update in analyze_audio_file(filepath, patient_context=patient_context):
            yield update
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass
