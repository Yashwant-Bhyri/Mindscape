from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from async_care_service import (
    build_async_care_summary,
    build_daily_nancy_questions,
    build_daily_nancy_status,
    build_patient_timeline,
    get_doctor_alerts,
)
import hashlib

from clinic_state import (
    get_daily_checkins,
    get_forum_extra_replies,
    get_nancy_interactions,
    get_nancy_tasks,
    get_outreach_logs,
    get_forum_saved_thread_ids,
    get_forum_votes,
    get_patient_messages,
    get_runtime_forum_threads,
    get_session_notes,
    get_session_records,
    get_sos_events,
)
from product_data import (
    DEFAULT_DOCTOR_ID,
    DOCTORS,
    ORGANIZATIONS,
    get_doctor,
    get_doctors_for_organization,
    get_organization,
    get_patient,
    get_patients_for_doctor,
)
from medical_news_feed import get_live_medical_headlines_as_threads, get_news_brief_lines
from pubmed_research_brief import get_pubmed_psychiatry_brief_lines
from session_service import build_session_prep
from shenzhen_directory import load_shenzhen_hospital_directory, load_shenzhen_psychiatric_directory

# Import new robust seed data
import json
import os
from pathlib import Path
_SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"

def _load_seed(name: str) -> list[dict]:
    path = _SEED_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    return []


def doctor_exists(doctor_id: str | None) -> bool:
    return bool(doctor_id) and doctor_id in DOCTORS


def get_doctor_or_default(doctor_id: str | None) -> dict:
    return deepcopy(get_doctor(doctor_id or DEFAULT_DOCTOR_ID))


def get_patient_or_none(doctor_id: str, patient_id: str | None) -> dict | None:
    patient = get_patient(doctor_id, patient_id) if patient_id else None
    return merge_patient_runtime_data(patient, doctor_id) if patient else None


def get_patient_merged_for_session(doctor_id: str, patient_id: str | None) -> dict | None:
    """Like get_patient_or_none but drops heavy timeline blobs after summaries are computed (Session GET only)."""
    patient = get_patient(doctor_id, patient_id) if patient_id else None
    return merge_patient_runtime_data(patient, doctor_id, session_response=True) if patient else None


def merge_patient_runtime_data(
    patient: dict | None, doctor_id: str, *, session_response: bool = False
) -> dict | None:
    if not patient:
        return None

    merged = dict(patient)
    patient_id = patient.get("id")
    if not patient_id:
        return merged

    checkins = get_daily_checkins(doctor_id, patient_id)
    sessions = get_session_records(doctor_id, patient_id)
    session_notes = get_session_notes(doctor_id, patient_id)
    nancy_tasks = get_nancy_tasks(doctor_id, patient_id)
    nancy_interactions = get_nancy_interactions(doctor_id, patient_id)
    messages = get_patient_messages(doctor_id, patient_id)
    sos_events = get_sos_events(doctor_id, patient_id)
    async_alerts = [alert.to_dict() for alert in get_doctor_alerts(doctor_id) if alert.patient_id == patient_id]
    outreach = [
        entry
        for entry in get_outreach_logs(doctor_id)
        if entry.get("patient_id") == patient_id or entry.get("patient") == patient.get("name")
    ]

    merged["daily_checkins"] = checkins
    merged["session_records"] = sessions
    merged["session_notes"] = session_notes
    merged["nancy_tasks"] = nancy_tasks
    merged["nancy_interactions"] = nancy_interactions
    merged["patient_messages"] = messages
    merged["sos_events"] = sos_events
    merged["async_alerts"] = async_alerts
    merged["outreach_logs"] = outreach

    latest_checkin = checkins[0] if checkins else None
    latest_session = sessions[0] if sessions else None
    latest_session_note = session_notes[0] if session_notes else None
    latest_nancy = nancy_interactions[0] if nancy_interactions else None
    latest_message = messages[0] if messages else None
    latest_sos = sos_events[0] if sos_events else None
    latest_alert = async_alerts[0] if async_alerts else None
    latest_outreach = outreach[0] if outreach else None

    if latest_checkin:
        merged["last_daily_checkin_summary"] = latest_checkin.get("clinical_summary") or latest_checkin.get("daily_update", "")
        merged.setdefault("personal_logs", [])
        merged["personal_logs"] = [
            {
                "date": latest_checkin.get("date_label", ""),
                "entry": latest_checkin.get("daily_update", "") or latest_checkin.get("clinical_summary", ""),
            }
        ] + merged["personal_logs"]
    if latest_session:
        merged["last_session_summary"] = latest_session.get("reasoning", "")
        merged["last_update"] = (
            f"Session on {latest_session.get('date_label', '')} at {latest_session.get('time_label', '')}: "
            f"{latest_session.get('hypothesis_name', '')} ({latest_session.get('hypothesis_confidence', '')})."
        )
        merged.setdefault("diagnosis_reports", [])
        merged["diagnosis_reports"] = [
            {
                "title": latest_session.get("hypothesis_name", "Session analysis"),
                "date": latest_session.get("date_label", ""),
                "summary": latest_session.get("reasoning", ""),
            }
        ] + merged["diagnosis_reports"]
    if latest_session_note:
        merged["last_session_note_summary"] = latest_session_note.get("note", "")
        merged.setdefault("care_team_notes", [])
        merged["care_team_notes"] = [
            {
                "source": "Clinician note",
                "note": latest_session_note.get("note", ""),
            }
        ] + merged["care_team_notes"]
    if latest_nancy:
        merged["last_nancy_summary"] = latest_nancy.get("clinician_summary") or latest_nancy.get("patient_report", "")
        merged.setdefault("personal_logs", [])
        merged["personal_logs"] = [
            {
                "date": latest_nancy.get("date_label", ""),
                "entry": latest_nancy.get("patient_report", "") or latest_nancy.get("clinician_summary", ""),
            }
        ] + merged["personal_logs"]
        merged.setdefault("care_team_notes", [])
        merged["care_team_notes"] = [
            {
                "source": "Nancy companion",
                "note": latest_nancy.get("clinician_summary") or latest_nancy.get("recommended_follow_up", ""),
            }
        ] + merged["care_team_notes"]
    if latest_message:
        merged["last_message_summary"] = latest_message.get("body", "")
    if latest_sos:
        merged["last_sos_summary"] = (
            f"{latest_sos.get('severity', 'urgent').title()} SOS routed via {latest_sos.get('emergency_number', '120')} "
            f"to {latest_sos.get('recommended_hospital', latest_sos.get('doctor_hospital', 'hospital'))}."
        )
        merged.setdefault("alerts", [])
        merged["alerts"] = [merged["last_sos_summary"]] + merged["alerts"]
    if latest_alert:
        merged["last_async_alert_summary"] = latest_alert.get("summary", "")
        merged.setdefault("alerts", [])
        merged["alerts"] = [latest_alert.get("title", "Async care review needed")] + merged["alerts"]
    if latest_outreach:
        merged["last_outreach_summary"] = latest_outreach.get("summary", "")
        merged.setdefault("care_team_notes", [])
        merged["care_team_notes"] = [
            {
                "source": latest_outreach.get("channel", "Queued follow-up"),
                "note": latest_outreach.get("summary", ""),
            }
        ] + merged["care_team_notes"]

    if session_response:
        # Session prep only needs summary fields + async_alerts for counts + alerts for highlights.
        # Full timelines were loaded to compute latest_* above; drop them to avoid huge RAM / GC churn.
        merged.pop("daily_checkins", None)
        merged.pop("session_records", None)
        merged.pop("nancy_tasks", None)
        merged.pop("nancy_interactions", None)
        merged.pop("patient_messages", None)
        merged.pop("sos_events", None)
        merged.pop("outreach_logs", None)
        notes = merged.get("session_notes") or []
        merged["session_notes"] = list(notes)[:80]

    return merged


def build_patient_context(patient: dict | None, doctor_id: str | None = None) -> str | None:
    if not patient:
        return None

    resolved_doctor = get_doctor(doctor_id or DEFAULT_DOCTOR_ID)
    health_history = patient.get("health_history", [])
    goals = patient.get("goals", [])
    alerts = patient.get("alerts", [])
    report_summaries = [report.get("summary", "") for report in patient.get("diagnosis_reports", [])[:2]]
    latest_checkin = (patient.get("daily_checkins") or [None])[0]
    latest_session = (patient.get("session_records") or [None])[0]

    context_lines = [
        f"Patient: {patient.get('name', 'Unknown')}",
        f"Diagnosis: {patient.get('diagnosis', 'Unknown')}",
        f"Risk: {patient.get('risk', 'Unknown')}",
        f"Status: {patient.get('status', 'Unknown')}",
        f"Care plan: {patient.get('care_plan', 'None')}",
        f"Next appointment: {patient.get('next_appointment', 'Unknown')}",
        f"Supervising hospital: {resolved_doctor.get('hospital_name', 'Unknown')}",
        f"Supervising department: {resolved_doctor.get('department_name', 'Unknown')}",
    ]
    if health_history:
        context_lines.append("Recent health history: " + " | ".join(health_history[:3]))
    if report_summaries:
        context_lines.append("Recent reports: " + " | ".join(report_summaries))
    if goals:
        context_lines.append("Goals: " + " | ".join(goals[:3]))
    if alerts:
        context_lines.append("Alerts: " + " | ".join(alerts[:3]))
    if latest_checkin:
        context_lines.append(
            "Latest daily check-in: "
            f"mood {latest_checkin.get('mood_score', '')}/10, anxiety {latest_checkin.get('anxiety_score', '')}/10, "
            f"sleep {latest_checkin.get('sleep_hours', '')}h, cognition {latest_checkin.get('cognition_score', '')}/10. "
            f"Summary: {latest_checkin.get('clinical_summary') or latest_checkin.get('daily_update', '')}"
        )
    if latest_session:
        context_lines.append(
            "Latest session insight: "
            f"{latest_session.get('hypothesis_name', '')} ({latest_session.get('hypothesis_confidence', '')}). "
            f"BSV V={latest_session.get('bsv', {}).get('valence', 0.0)}, "
            f"A={latest_session.get('bsv', {}).get('arousal', 0.0)}, "
            f"D={latest_session.get('bsv', {}).get('dominance', 0.0)}."
        )
    latest_nancy = (patient.get("nancy_interactions") or [None])[0]
    if latest_nancy:
        context_lines.append(
            "Latest Nancy handoff: "
            f"{latest_nancy.get('clinician_summary') or latest_nancy.get('patient_report', '')}"
        )
    latest_message = (patient.get("patient_messages") or [None])[0]
    if latest_message:
        context_lines.append(
            "Latest patient message: "
            f"{latest_message.get('sender_role', 'patient')} -> {latest_message.get('recipient', 'doctor')}: "
            f"{latest_message.get('body', '')}"
        )
    latest_sos = (patient.get("sos_events") or [None])[0]
    if latest_sos:
        context_lines.append(
            "Latest SOS event: "
            f"{latest_sos.get('severity', 'urgent')} escalation to {latest_sos.get('recommended_hospital', '')} "
            f"via {latest_sos.get('emergency_number', '120')}."
        )
    latest_alert = (patient.get("async_alerts") or [None])[0]
    if latest_alert:
        context_lines.append(
            "Latest async alert: "
            f"{latest_alert.get('severity', 'watch')} / {latest_alert.get('status', 'new')} - "
            f"{latest_alert.get('summary', '')}"
        )
    nancy_tasks = patient.get("nancy_tasks") or []
    if nancy_tasks:
        context_lines.append(
            "Active between-session tasks: "
            + " | ".join(
                f"{task.get('title', '')}: {task.get('instructions', '')}"
                for task in nancy_tasks[:3]
            )
        )
    return "\n".join(context_lines)


def build_nancy_support_text(patient: dict, doctor: dict | None = None) -> str:
    doctor_name = (doctor or get_doctor_or_default(None)).get("name", "your doctor")
    doctor_display_name = doctor_name if doctor_name.lower().startswith("dr. ") else f"Dr. {doctor_name}"

    latest_checkin = (patient.get("daily_checkins") or [None])[0]
    mood = latest_checkin.get("mood_score", "") if latest_checkin else ""
    anxiety = latest_checkin.get("anxiety_score", "") if latest_checkin else ""
    sleep = latest_checkin.get("sleep_hours", "") if latest_checkin else ""

    task_prompt = ""
    if patient.get("nancy_tasks"):
        first_task = patient["nancy_tasks"][0]
        task_prompt = f" Also, how is {first_task.get('title', 'the plan from your doctor')} going?"

    return (
        f"Hi {patient.get('name', 'there')}, this is Nancy checking in for {doctor_display_name}. "
        f"I saw your recent update"
        + (f" with mood {mood}/10" if mood else "")
        + (f", anxiety {anxiety}/10" if anxiety else "")
        + (f", and sleep {sleep} hours" if sleep else "")
        + ". I wanted to check how you're doing right now and whether you'd like to share anything important before the next session."
        + task_prompt
    )


def support_recommended(patient: dict) -> bool:
    latest_checkin = (patient.get("daily_checkins") or [None])[0]
    if not latest_checkin:
        return False
    try:
        mood_value = float(latest_checkin.get("mood_score", 5) or 5)
        anxiety_value = float(latest_checkin.get("anxiety_score", 5) or 5)
        return mood_value <= 3 or anxiety_value >= 8 or bool(latest_checkin.get("safety_concerns", "").strip())
    except (TypeError, ValueError):
        return False


def serialize_timeline(patient: dict, audience: str = "doctor") -> list[dict]:
    return [entry.__dict__ for entry in build_patient_timeline(patient, audience=audience)]


def patient_visible_nancy_updates(patient: dict) -> list[dict]:
    return [
        entry
        for entry in patient.get("nancy_interactions", [])
        if entry.get("patient_visible") or entry.get("patient_message")
    ]


def patient_inbox_messages(patient: dict) -> list[dict]:
    visible_messages = []
    for message in patient.get("patient_messages", []):
        sender_role = message.get("sender_role", "")
        recipient = message.get("recipient", "")
        if sender_role == "system":
            continue
        if sender_role == "nancy" and recipient == "doctor":
            continue
        visible_messages.append(message)
    return visible_messages


def get_platform_payload() -> dict:
    return {
        "default_doctor_id": DEFAULT_DOCTOR_ID,
        "organizations": deepcopy(ORGANIZATIONS),
        "doctor_count": len(DOCTORS),
        "organization_count": len(ORGANIZATIONS),
    }


def get_doctors_payload() -> dict:
    doctors = [doctor_summary(doctor_id) for doctor_id in DOCTORS]
    return {
        "default_doctor_id": DEFAULT_DOCTOR_ID,
        "doctors": doctors,
    }


def get_patient_access_payload() -> dict:
    doctors = []
    for doctor_id in DOCTORS:
        doctor = deepcopy(get_doctor(doctor_id))
        patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
        doctors.append(
            {
                "id": doctor["id"],
                "name": doctor["name"],
                "specialty": doctor.get("specialty", ""),
                "hospital_name": doctor.get("hospital_name", ""),
                "patients": [
                    {
                        "id": patient["id"],
                        "name": patient["name"],
                        "diagnosis": patient.get("diagnosis", ""),
                        "care_plan": patient.get("care_plan", ""),
                        "risk": patient.get("risk", ""),
                        "next_appointment": patient.get("next_appointment", ""),
                        "daily_nancy_due": build_daily_nancy_status(patient)["due"],
                    }
                    for patient in patients
                ],
            }
        )
    return {"doctors": doctors}


def get_organization_payload(organization_id: str) -> dict:
    organization = deepcopy(get_organization(organization_id))
    doctors = [doctor_summary(doctor["id"]) for doctor in get_doctors_for_organization(organization_id)]
    organization["doctors"] = doctors
    return organization


def doctor_summary(doctor_id: str) -> dict:
    doctor = deepcopy(get_doctor(doctor_id))
    patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    doctor["patient_count"] = len(patients)
    doctor["patients_preview"] = [
        {
            "id": patient["id"],
            "name": patient["name"],
            "diagnosis": patient["diagnosis"],
            "risk": patient["risk"],
            "next_appointment": patient["next_appointment"],
            "last_update": patient["last_update"],
        }
        for patient in patients[:3]
    ]
    return doctor


def get_workspace_payload(doctor_id: str) -> dict:
    doctor = deepcopy(get_doctor(doctor_id))
    patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    alerts = [alert.to_dict() for alert in get_doctor_alerts(doctor_id)]
    nancy_watchtower = []
    for patient in patients:
        for interaction in patient.get("nancy_interactions", [])[:2]:
            nancy_watchtower.append({**interaction, "patient_name": patient.get("name", "Unknown")})
    nancy_watchtower.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {
        "doctor": doctor,
        "patients": patients,
        "alerts": alerts,
        "open_alerts": [alert for alert in alerts if alert.get("status") != "resolved"],
        "urgent_alerts": [alert for alert in alerts if alert.get("status") != "resolved" and alert.get("severity") == "urgent"],
        "outreach_logs": get_outreach_logs(doctor_id),
        "nancy_watchtower": nancy_watchtower[:10],
    }


def get_doctor_profile_payload(doctor_id: str) -> dict:
    doctor = deepcopy(get_doctor(doctor_id))
    organization = deepcopy(get_organization(doctor["organization_id"]))
    patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    return {
        "doctor": doctor,
        "organization": organization,
        "patient_count": len(patients),
        "patients_preview": [
            {
                "id": patient["id"],
                "name": patient["name"],
                "diagnosis": patient["diagnosis"],
                "risk": patient["risk"],
                "last_update": patient["last_update"],
            }
            for patient in patients[:4]
        ],
    }


def _stable_thread_id(prefix: str, title: str) -> str:
    digest = hashlib.sha256(f"{prefix}:{title}".encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _load_forum_threads_seed(doctor_id: str) -> list[dict]:
    path = _SEED_DIR / "doctor_forum_threads.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    threads = raw.get(doctor_id) or raw.get("_default", [])
    return deepcopy(threads) if isinstance(threads, list) else []


def _legacy_forum_posts_to_threads(doctor_id: str, posts: list[dict]) -> list[dict]:
    """Migrate old forum_posts card shape into thread records."""
    migrated: list[dict] = []
    for i, p in enumerate(posts):
        title = (p.get("title") or "Discussion").strip()
        migrated.append(
            {
                "id": _stable_thread_id(f"legacy-{doctor_id}", title),
                "title": title,
                "flair": "Discussion",
                "kind": "discussion",
                "community": (p.get("community") or "Doctor's Corner").strip(),
                "author": "Community",
                "created": "2026-01-01T12:00:00",
                "body": (p.get("summary") or "").strip(),
                "link": "",
                "link_title": "",
                "replies": [],
                "source": "legacy",
            }
        )
    return migrated


def build_merged_forum_threads(doctor_id: str, doctor: dict) -> list[dict]:
    seed = _load_forum_threads_seed(doctor_id)
    if not seed:
        seed = _legacy_forum_posts_to_threads(doctor_id, doctor.get("forum_posts") or [])
    extras_map = get_forum_extra_replies(doctor_id)
    merged_seed: list[dict] = []
    for t in seed:
        tid = t.get("id") or _stable_thread_id("seed", t.get("title", ""))
        base = dict(t)
        base["id"] = tid
        extra = list(extras_map.pop(tid, []))
        base["replies"] = list(base.get("replies") or []) + extra
        base.setdefault("source", "seed")
        merged_seed.append(base)

    runtime = get_runtime_forum_threads(doctor_id)
    merged_runtime: list[dict] = []
    for t in runtime:
        tid = t.get("id") or ""
        extra = list(extras_map.pop(tid, []))
        rt = dict(t)
        rt["replies"] = list(rt.get("replies") or []) + extra
        merged_runtime.append(rt)

    # Generic RSS/NewsAPI headlines are OFF by default — they read like Twitter, not research-grade forum posts.
    live_news: list[dict] = []
    if os.getenv("MEDICAL_NEWS_FORUM_THREADS", "").lower() in ("1", "true", "yes"):
        news_cap = int(os.getenv("MEDICAL_NEWS_MAX", "12"))
        try:
            live_news = get_live_medical_headlines_as_threads(limit=news_cap)
        except Exception:
            live_news = []

    combined = live_news + merged_seed + merged_runtime
    combined.sort(key=lambda x: x.get("created") or "", reverse=True)

    votes = get_forum_votes(doctor_id)
    saved_ids = set(get_forum_saved_thread_ids(doctor_id))
    for row in combined:
        replies = row.get("replies") or []
        row["reply_count"] = len(replies)
        activity = row.get("activity")
        if not activity:
            n = len(replies)
            row["activity"] = f"{n} repl{'ies' if n != 1 else 'y'}"
        tid = row.get("id") or ""
        seed_boost = int(row.get("seed_score") or 0)
        row["vote_score"] = seed_boost + int(votes.get(tid, 0))
        row["saved"] = tid in saved_ids
    return combined


def get_research_payload(doctor_id: str) -> dict:
    doctor = deepcopy(get_doctor(doctor_id))
    forum_threads = build_merged_forum_threads(doctor_id, doctor)
    forum_posts = [
        {
            "title": t.get("title"),
            "community": t.get("community"),
            "activity": t.get("activity"),
            "summary": (t.get("body") or "")[:280],
        }
        for t in forum_threads
    ]
    base_brief = list(doctor.get("weekly_brief") or [])

    pubmed_lines: list[str] = []
    if os.getenv("WEEKLY_BRIEF_PUBMED", "").lower() in ("1", "true", "yes"):
        try:
            pubmed_lines = get_pubmed_psychiatry_brief_lines(
                max_lines=int(os.getenv("WEEKLY_BRIEF_PUBMED_LINES", "4")),
            )
        except Exception:
            pubmed_lines = []

    rss_lines: list[str] = []
    if os.getenv("MEDICAL_NEWS_BRIEF", "").lower() in ("1", "true", "yes"):
        try:
            rss_lines = get_news_brief_lines(max_lines=int(os.getenv("MEDICAL_NEWS_BRIEF_LINES", "4")))
        except Exception:
            rss_lines = []

    weekly_brief = (
        base_brief
        + ([f"PubMed · {line}" for line in pubmed_lines] if pubmed_lines else [])
        + ([f"Headline digest · {line}" for line in rss_lines] if rss_lines else [])
    )

    return {
        "doctor": doctor,
        "weekly_brief": weekly_brief,
        "performance": doctor.get("performance", []),
        "forum_threads": forum_threads,
        "forum_posts": forum_posts,
        "live_headlines_count": len([t for t in forum_threads if t.get("live_headline") or t.get("source") == "medical_news"]),
    }


def get_patient_payload(doctor_id: str, patient_id: str) -> dict:
    doctor = deepcopy(get_doctor(doctor_id))
    patient = get_patient_or_none(doctor_id, patient_id)
    if not patient:
        raise KeyError(patient_id)
    async_summary = build_async_care_summary(patient, doctor)
    return {
        "doctor": doctor,
        "patient": patient,
        "async_summary": asdict(async_summary),
        "timeline": serialize_timeline(patient, audience="doctor"),
        "support_recommended": support_recommended(patient),
    }


def get_companion_payload(doctor_id: str, patient_id: str) -> dict:
    payload = get_patient_payload(doctor_id, patient_id)
    payload["latest_checkin"] = payload["async_summary"].get("latest_checkin")
    patient = payload["patient"]
    payload["timeline"] = serialize_timeline(patient, audience="patient")
    payload["patient_inbox"] = patient_inbox_messages(patient)
    payload["patient_nancy_updates"] = patient_visible_nancy_updates(patient)
    payload["daily_nancy_status"] = build_daily_nancy_status(patient)
    return payload


def get_nancy_payload(doctor_id: str, patient_id: str | None = None) -> dict:
    doctor = deepcopy(get_doctor(doctor_id))
    patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    patient = get_patient_or_none(doctor_id, patient_id) if patient_id else None
    return {
        "doctor": doctor,
        "panel_patients": patients,
        "patient": patient,
        "daily_nancy_status": build_daily_nancy_status(patient) if patient else None,
        "daily_nancy_questions": build_daily_nancy_questions(patient) if patient else [],
        "patient_nancy_updates": patient_visible_nancy_updates(patient) if patient else [],
        "patient_inbox": patient_inbox_messages(patient) if patient else [],
        "hospital_directory": load_shenzhen_hospital_directory(limit=120),
        "psychiatric_directory": load_shenzhen_psychiatric_directory(),
    }


def _slim_doctor_for_session(doctor: dict) -> dict:
    """Avoid shipping the entire doctor seed blob on the Session route."""
    return {
        "id": doctor.get("id"),
        "name": doctor.get("name"),
        "specialty": doctor.get("specialty"),
        "hospital_name": doctor.get("hospital_name"),
    }


def _slim_patient_for_session(patient: dict) -> dict:
    """Session UI only needs identity + prep card fields (prep is built separately from full merge)."""
    return {
        "id": patient.get("id"),
        "name": patient.get("name"),
        "diagnosis": patient.get("diagnosis"),
    }


def get_session_payload(doctor_id: str, patient_id: str | None = None) -> dict:
    doctor = get_doctor(doctor_id)
    patient = get_patient_merged_for_session(doctor_id, patient_id) if patient_id else None
    session_records = get_session_records(doctor_id, patient_id) if patient_id else []
    latest_session = session_records[0] if session_records else None
    prep = build_session_prep(doctor, patient)
    notes = list(patient.get("session_notes", []) if patient else [])
    notes = notes[:80]
    return {
        "doctor": _slim_doctor_for_session(doctor),
        "patient": _slim_patient_for_session(patient) if patient else None,
        "session_prep": prep.to_dict() if prep else None,
        "latest_session": latest_session,
        "session_notes": notes,
    }


# ============================================================
# Robust MVP Seed Data Accessors (New Expanded Scale)
# ============================================================

def get_expanded_hospitals() -> list[dict]:
    return _load_seed("hospitals.json")

def get_expanded_doctors() -> list[dict]:
    return _load_seed("doctors_expanded.json") or _load_seed("doctors.json")

def get_expanded_patients() -> list[dict]:
    return _load_seed("patients_base.json")

def get_patient_longitudinal(patient_id: str) -> dict | None:
    path = _SEED_DIR / "patient_longitudinal" / f"{patient_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

def get_doctor_with_expanded_panel(doctor_id: str) -> dict | None:
    """Return doctor with real seeded patients from expanded data."""
    doctors = get_expanded_doctors()
    for d in doctors:
        if d["id"] == doctor_id:
            doc = deepcopy(d)
            all_patients = get_expanded_patients()
            doc["patients"] = [p for p in all_patients if p.get("doctor_id") == doctor_id]
            return doc
    return None
