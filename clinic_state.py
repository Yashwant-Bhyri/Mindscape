import json
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import re
import uuid


BASE_RUNTIME_DIR = Path(__file__).resolve().parent / "data" / "runtime"


def _runtime_file() -> Path:
    override = os.getenv("MINDSCAPE_RUNTIME_FILE", "").strip()
    if override:
        return Path(override)
    return BASE_RUNTIME_DIR / "clinic_state.json"


def _runtime_dir() -> Path:
    return _runtime_file().parent


def _default_state():
    return {
        "patients_by_doctor": {},
        "outreach_logs_by_doctor": {},
        "patient_checkins_by_doctor": {},
        "patient_sessions_by_doctor": {},
        "patient_session_notes_by_doctor": {},
        "nancy_tasks_by_doctor": {},
        "nancy_interactions_by_doctor": {},
        "patient_messages_by_doctor": {},
        "patient_sos_events_by_doctor": {},
        "async_alerts_by_doctor": {},
        "forum_threads_by_doctor": {},
        "forum_extra_replies_by_doctor": {},
        "forum_votes_by_doctor": {},
        "forum_saved_by_doctor": {},
    }


def _ensure_runtime_dir():
    _runtime_dir().mkdir(parents=True, exist_ok=True)


def load_runtime_state():
    _ensure_runtime_dir()
    runtime_file = _runtime_file()
    if not runtime_file.exists():
        return _default_state()

    try:
        return json.loads(runtime_file.read_text())
    except (json.JSONDecodeError, OSError):
        return _default_state()


def save_runtime_state(state: dict):
    _ensure_runtime_dir()
    _runtime_file().write_text(json.dumps(state, indent=2))


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "patient"


def _timestamp_label() -> str:
    return datetime.now().strftime("%b %d, %Y at %H:%M")


def get_runtime_patients(doctor_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("patients_by_doctor", {}).get(doctor_id, []))


def add_patient_record(doctor_id: str, name: str, concern: str, context: str) -> dict:
    state = load_runtime_state()
    doctor_patients = state.setdefault("patients_by_doctor", {}).setdefault(doctor_id, [])

    now_label = _timestamp_label()
    patient_id = f"{_slugify(name)}-{uuid.uuid4().hex[:6]}"
    patient_record = {
        "id": patient_id,
        "name": name.strip(),
        "age": "New intake",
        "status": "New intake",
        "risk": "Medium",
        "next_appointment": "To be scheduled",
        "diagnosis": concern.strip(),
        "history": context.strip() or "Initial context captured during intake.",
        "last_update": f"Intake created {now_label}.",
        "care_plan": "Initial review pending live diagnostic session and physician validation.",
        "health_history": [context.strip() or "Initial intake context pending enrichment."],
        "diagnosis_reports": [
            {
                "title": "Initial intake packet",
                "date": datetime.now().strftime("%B %d, %Y"),
                "summary": concern.strip() or "Presenting concern to be clarified during first clinician session.",
            }
        ],
        "personal_logs": [
            {
                "date": datetime.now().strftime("%b %d"),
                "entry": "Patient intake created. Awaiting first direct session record.",
            }
        ],
        "care_team_notes": [
            {
                "source": "Intake desk",
                "note": context.strip() or "Initial context collected; detailed history still pending.",
            }
        ],
        "goals": [
            "Complete first diagnostic session.",
            "Validate clinical history with physician review.",
        ],
        "alerts": [
            "New patient record requires physician validation before major care decisions.",
        ],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "is_runtime_record": True,
    }
    doctor_patients.insert(0, patient_record)
    save_runtime_state(state)
    return deepcopy(patient_record)


def get_outreach_logs(doctor_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("outreach_logs_by_doctor", {}).get(doctor_id, []))


def add_outreach_log(doctor_id: str, target: str, message: str, patient_id: str | None = None) -> dict:
    state = load_runtime_state()
    doctor_logs = state.setdefault("outreach_logs_by_doctor", {}).setdefault(doctor_id, [])
    log_entry = {
        "patient": target.strip() or "Unassigned outreach",
        "patient_id": patient_id,
        "channel": "Queued follow-up",
        "summary": message.strip(),
        "time": "Just now",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "is_runtime_record": True,
    }
    doctor_logs.insert(0, log_entry)
    save_runtime_state(state)
    return deepcopy(log_entry)


def _patient_bucket(state: dict, key: str, doctor_id: str, patient_id: str) -> list[dict]:
    return state.setdefault(key, {}).setdefault(doctor_id, {}).setdefault(patient_id, [])


def get_daily_checkins(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("patient_checkins_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_daily_checkin(doctor_id: str, patient_id: str, patient_name: str, payload: dict) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "patient_checkins_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()
    checkin_record = {
        "id": f"checkin-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "mood_score": payload.get("mood_score", ""),
        "anxiety_score": payload.get("anxiety_score", ""),
        "sleep_hours": payload.get("sleep_hours", ""),
        "energy_score": payload.get("energy_score", ""),
        "stress_score": payload.get("stress_score", ""),
        "cognition_score": payload.get("cognition_score", ""),
        "memory_score": payload.get("memory_score", ""),
        "functioning_score": payload.get("functioning_score", ""),
        "medication_adherence": payload.get("medication_adherence", ""),
        "side_effects": payload.get("side_effects", ""),
        "daily_update": payload.get("daily_update", ""),
        "safety_concerns": payload.get("safety_concerns", ""),
        "significant_events": payload.get("significant_events", ""),
        "clinical_summary": payload.get("clinical_summary", ""),
    }
    bucket.insert(0, checkin_record)
    save_runtime_state(state)
    return deepcopy(checkin_record)


def get_session_records(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("patient_sessions_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_session_record(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    transcript: str,
    transcription_provider: str,
    result: dict,
) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "patient_sessions_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()
    bsv = result.get("bsv", {})
    visual_bsv = result.get("visual_bsv", {})
    hypothesis = result.get("hypothesis", {})
    emotion_trajectory = result.get("emotion_trajectory", [])
    traumatic_markers = result.get("traumatic_markers", [])

    transcript_excerpt = " ".join(transcript.split())[:700]
    if len(transcript.split()) > 100:
        transcript_excerpt += "..."

    event_markers = []
    for marker in traumatic_markers[:3]:
        event_markers.append(
            {
                "label": "Trauma / disclosure peak",
                "detail": marker,
                "response": "Monitor whether arousal intensifies or affect collapses after the disclosure.",
            }
        )
    for phase in emotion_trajectory[:3]:
        event_markers.append(
            {
                "label": f"{phase.get('phase', 'Session')} emotional phase",
                "detail": phase.get("trigger", phase.get("dominant_emotion", "")),
                "response": phase.get("dominant_emotion", "Unknown"),
            }
        )

    session_record = {
        "id": f"session-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "transcription_provider": transcription_provider,
        "hypothesis_name": hypothesis.get("name", "Unknown"),
        "hypothesis_confidence": hypothesis.get("confidence", ""),
        "reasoning": result.get("reasoning", ""),
        "treatment_plan": result.get("treatment_plan", ""),
        "follow_up": result.get("follow_up", []),
        "retrieved_evidence": result.get("retrieved_evidence", []),
        "safety_gate": result.get("safety_gate", "FAIL"),
        "bsv": {
            "valence": float(bsv.get("valence", 0.0)),
            "arousal": float(bsv.get("arousal", 0.0)),
            "dominance": float(bsv.get("dominance", 0.0)),
        },
        "visual_bsv": {
            "facial_valence": float(visual_bsv.get("facial_valence", 0.0)),
            "facial_arousal": float(visual_bsv.get("facial_arousal", 0.0)),
            "gaze_stability": float(visual_bsv.get("gaze_stability", 0.0)),
            "blink_rate_per_min": float(visual_bsv.get("blink_rate_per_min", 0.0)),
            "flat_affect_score": float(visual_bsv.get("flat_affect_score", 0.0)),
            "twitch_zones": visual_bsv.get("twitch_zones", []),
        },
        "emotion_trajectory": emotion_trajectory,
        "traumatic_markers": traumatic_markers,
        "event_markers": event_markers,
        "transcript_excerpt": transcript_excerpt,
    }
    bucket.insert(0, session_record)
    save_runtime_state(state)
    return deepcopy(session_record)


def get_session_notes(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("patient_session_notes_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_session_note(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    title: str,
    note: str,
    plan_update: str = "",
    disposition: str = "",
) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "patient_session_notes_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()
    note_record = {
        "id": f"session-note-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "title": title.strip() or "Clinician review note",
        "note": note.strip(),
        "plan_update": plan_update.strip(),
        "disposition": disposition.strip() or "Continue review",
        "source": "Clinician",
    }
    bucket.insert(0, note_record)
    save_runtime_state(state)
    return deepcopy(note_record)


def get_nancy_tasks(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("nancy_tasks_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_nancy_task(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    title: str,
    instructions: str,
    category: str = "Recovery",
    due_label: str = "",
) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "nancy_tasks_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()
    task_record = {
        "id": f"nancy-task-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "title": title.strip(),
        "instructions": instructions.strip(),
        "category": category.strip() or "Recovery",
        "due_label": due_label.strip(),
        "status": "Active",
        "source": "Doctor directive",
    }
    bucket.insert(0, task_record)
    save_runtime_state(state)
    return deepcopy(task_record)


def get_nancy_interactions(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("nancy_interactions_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_nancy_interaction(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    payload: dict,
) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "nancy_interactions_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()

    escalation_level = (payload.get("escalation_level") or "routine").strip().lower()
    safety_note = (payload.get("safety_note") or "").strip()
    requires_doctor_review = escalation_level in {"watch", "urgent"} or bool(safety_note)

    interaction_record = {
        "id": f"nancy-log-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "mode": (payload.get("mode") or "voice").strip(),
        "conversation_goal": (payload.get("conversation_goal") or "").strip(),
        "patient_report": (payload.get("patient_report") or "").strip(),
        "clinician_summary": (payload.get("clinician_summary") or "").strip(),
        "observed_mood": (payload.get("observed_mood") or "").strip(),
        "functioning_note": (payload.get("functioning_note") or "").strip(),
        "cognition_note": (payload.get("cognition_note") or "").strip(),
        "medication_note": (payload.get("medication_note") or "").strip(),
        "safety_note": safety_note,
        "recommended_follow_up": (payload.get("recommended_follow_up") or "").strip(),
        "escalation_level": escalation_level,
        "requires_doctor_review": requires_doctor_review,
        "audience": (payload.get("audience") or "clinician").strip(),
        "patient_visible": bool(payload.get("patient_visible", False)),
        "patient_message": (payload.get("patient_message") or "").strip(),
        "relayed_to_patient": bool(payload.get("relayed_to_patient", False)),
        "source": "Nancy companion",
    }
    bucket.insert(0, interaction_record)
    save_runtime_state(state)
    return deepcopy(interaction_record)


def get_patient_messages(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("patient_messages_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_patient_message(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    sender_role: str,
    recipient: str,
    body: str,
    channel: str = "in-app chat",
) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "patient_messages_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()
    message_record = {
        "id": f"message-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "sender_role": sender_role.strip() or "patient",
        "recipient": recipient.strip() or "doctor",
        "body": body.strip(),
        "channel": channel.strip() or "in-app chat",
    }
    bucket.insert(0, message_record)
    save_runtime_state(state)
    return deepcopy(message_record)


def get_sos_events(doctor_id: str, patient_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("patient_sos_events_by_doctor", {}).get(doctor_id, {}).get(patient_id, []))


def add_sos_event(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    payload: dict,
) -> dict:
    state = load_runtime_state()
    bucket = _patient_bucket(state, "patient_sos_events_by_doctor", doctor_id, patient_id)
    created_at = datetime.now()
    sos_record = {
        "id": f"sos-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "region": payload.get("region", "Shenzhen, Guangdong, China"),
        "district": payload.get("district", ""),
        "severity": payload.get("severity", "urgent"),
        "reason": payload.get("reason", ""),
        "rule": payload.get("rule", "shenzhen-psychiatric-escalation-v1"),
        "doctor_hospital": payload.get("doctor_hospital", ""),
        "doctor_department": payload.get("doctor_department", ""),
        "recommended_hospital": payload.get("recommended_hospital", ""),
        "recommended_department": payload.get("recommended_department", ""),
        "emergency_number": payload.get("emergency_number", "120"),
        "safety_number": payload.get("safety_number", "110"),
        "notes": payload.get("notes", ""),
        "status": payload.get("status", "Escalated"),
    }
    bucket.insert(0, sos_record)
    save_runtime_state(state)
    return deepcopy(sos_record)


def get_async_alerts(
    doctor_id: str,
    patient_id: str | None = None,
    statuses: set[str] | None = None,
) -> list[dict]:
    state = load_runtime_state()
    alerts = deepcopy(state.get("async_alerts_by_doctor", {}).get(doctor_id, []))
    if patient_id:
        alerts = [alert for alert in alerts if alert.get("patient_id") == patient_id]
    if statuses:
        alerts = [alert for alert in alerts if alert.get("status") in statuses]
    return alerts


def add_async_alert(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    severity: str,
    source: str,
    title: str,
    summary: str,
    recommended_follow_up: str = "",
    route: str = "/patient",
    metadata: dict | None = None,
) -> dict:
    state = load_runtime_state()
    alerts = state.setdefault("async_alerts_by_doctor", {}).setdefault(doctor_id, [])
    created_at = datetime.now()
    alert_record = {
        "id": f"alert-{uuid.uuid4().hex[:8]}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "severity": (severity or "watch").strip().lower(),
        "source": source.strip() or "async care",
        "title": title.strip() or "Async care review needed",
        "summary": summary.strip(),
        "recommended_follow_up": recommended_follow_up.strip(),
        "route": route.strip() or "/patient",
        "status": "new",
        "created_at": created_at.isoformat(timespec="seconds"),
        "updated_at": created_at.isoformat(timespec="seconds"),
        "date_label": created_at.strftime("%b %d, %Y"),
        "time_label": created_at.strftime("%H:%M"),
        "metadata": metadata or {},
    }
    alerts.insert(0, alert_record)
    save_runtime_state(state)
    return deepcopy(alert_record)


def update_async_alert_status(
    doctor_id: str,
    alert_id: str,
    status: str,
    actor: str = "doctor",
    note: str = "",
) -> dict | None:
    state = load_runtime_state()
    alerts = state.setdefault("async_alerts_by_doctor", {}).setdefault(doctor_id, [])
    updated_at = datetime.now().isoformat(timespec="seconds")
    status = (status or "").strip().lower()

    for alert in alerts:
        if alert.get("id") != alert_id:
            continue
        alert["status"] = status or alert.get("status", "new")
        alert["updated_at"] = updated_at
        alert["last_action_by"] = actor
        if note.strip():
            alert["last_action_note"] = note.strip()
        save_runtime_state(state)
        return deepcopy(alert)
    return None


# ---------------------------------------------------------------------------
# Doctor research forum (runtime threads + replies to seed threads)
# ---------------------------------------------------------------------------


def get_runtime_forum_threads(doctor_id: str) -> list[dict]:
    state = load_runtime_state()
    return deepcopy(state.get("forum_threads_by_doctor", {}).get(doctor_id, []))


def get_forum_extra_replies(doctor_id: str) -> dict[str, list[dict]]:
    state = load_runtime_state()
    return deepcopy(state.get("forum_extra_replies_by_doctor", {}).get(doctor_id, {}))


def add_forum_thread(
    doctor_id: str,
    *,
    title: str,
    body: str,
    flair: str,
    kind: str,
    author: str,
    link: str = "",
    link_title: str = "",
    community: str = "Doctor's Corner",
) -> dict:
    state = load_runtime_state()
    bucket = state.setdefault("forum_threads_by_doctor", {}).setdefault(doctor_id, [])
    created = datetime.now()
    thread = {
        "id": f"rt-{uuid.uuid4().hex[:12]}",
        "title": title.strip(),
        "body": body.strip(),
        "flair": flair.strip() or "Discussion",
        "kind": (kind or "discussion").strip().lower(),
        "community": community.strip() or "Doctor's Corner",
        "author": author.strip() or "Clinician",
        "created": created.isoformat(timespec="seconds"),
        "date_label": created.strftime("%b %d, %Y at %H:%M"),
        "link": link.strip(),
        "link_title": link_title.strip(),
        "replies": [],
        "source": "runtime",
    }
    bucket.insert(0, thread)
    save_runtime_state(state)
    return deepcopy(thread)


def add_forum_reply(doctor_id: str, thread_id: str, *, body: str, author: str) -> dict | None:
    if not body.strip():
        return None
    state = load_runtime_state()
    created = datetime.now()
    reply = {
        "id": f"r-{uuid.uuid4().hex[:10]}",
        "author": author.strip() or "Clinician",
        "body": body.strip(),
        "created": created.isoformat(timespec="seconds"),
        "date_label": created.strftime("%b %d, %Y at %H:%M"),
    }

    for thread in state.setdefault("forum_threads_by_doctor", {}).setdefault(doctor_id, []):
        if thread.get("id") == thread_id:
            thread.setdefault("replies", []).append(reply)
            save_runtime_state(state)
            return deepcopy(reply)

    extras = state.setdefault("forum_extra_replies_by_doctor", {}).setdefault(doctor_id, {})
    bucket = extras.setdefault(thread_id, [])
    bucket.append(reply)
    save_runtime_state(state)
    return deepcopy(reply)


def get_forum_votes(doctor_id: str) -> dict[str, int]:
    state = load_runtime_state()
    raw = state.get("forum_votes_by_doctor", {}).get(doctor_id, {})
    return {str(k): int(v) for k, v in raw.items()}


def increment_forum_vote(doctor_id: str, thread_id: str) -> int:
    state = load_runtime_state()
    doc = state.setdefault("forum_votes_by_doctor", {}).setdefault(doctor_id, {})
    n = int(doc.get(thread_id, 0)) + 1
    doc[thread_id] = n
    save_runtime_state(state)
    return n


_FORUM_SAVED_CAP = 250


def get_forum_saved_thread_ids(doctor_id: str) -> list[str]:
    state = load_runtime_state()
    raw = state.get("forum_saved_by_doctor", {}).get(doctor_id, [])
    return list(raw) if isinstance(raw, list) else []


def toggle_forum_saved_thread(doctor_id: str, thread_id: str) -> bool:
    """Return True if thread is saved after toggle, False if removed from saved."""
    if not thread_id.strip():
        return False
    state = load_runtime_state()
    bucket = state.setdefault("forum_saved_by_doctor", {}).setdefault(doctor_id, [])
    if thread_id in bucket:
        bucket[:] = [x for x in bucket if x != thread_id]
        save_runtime_state(state)
        return False
    bucket.insert(0, thread_id)
    del bucket[_FORUM_SAVED_CAP:]
    save_runtime_state(state)
    return True
