from dataclasses import asdict
from datetime import date, datetime

from async_care import build_async_care_plan, evaluate_async_risk
from async_care_models import AsyncAlertRecord, AsyncCareSummary, CheckinPayload, TimelineEntry
from clinic_state import (
    add_async_alert,
    add_daily_checkin,
    add_nancy_interaction,
    add_patient_message,
    get_async_alerts,
    update_async_alert_status,
)


DAILY_CHECKIN_QUESTIONS = [
    {
        "key": "daily_update",
        "label": "Today",
        "prompt": "Before we do numbers, what has today actually felt like for you?",
    },
    {
        "key": "mood_score",
        "label": "Mood",
        "prompt": "If mood is 0 to 10, where would you put yourself today?",
    },
    {
        "key": "anxiety_score",
        "label": "Anxiety",
        "prompt": "And anxiety on that same 0 to 10 scale?",
    },
    {
        "key": "sleep_hours",
        "label": "Sleep",
        "prompt": "How many hours did you sleep, and did it feel restful or broken?",
    },
    {
        "key": "energy_score",
        "label": "Energy",
        "prompt": "How has your energy been today, from 0 to 10?",
    },
    {
        "key": "stress_score",
        "label": "Stress",
        "prompt": "How much stress are you carrying today, from 0 to 10?",
    },
    {
        "key": "functioning_score",
        "label": "Functioning",
        "prompt": "How much were you able to do the basics today, like eating, hygiene, work, study, or connecting with someone?",
    },
    {
        "key": "cognition_score",
        "label": "Focus",
        "prompt": "How has focus or clear thinking been today, from 0 to 10?",
    },
    {
        "key": "memory_score",
        "label": "Memory",
        "prompt": "Any memory or recall issues today, and what number would you give it from 0 to 10?",
    },
    {
        "key": "medication_adherence",
        "label": "Medication",
        "prompt": "Did you take medication as planned today? A rough percentage is fine.",
    },
    {
        "key": "side_effects",
        "label": "Side effects",
        "prompt": "Any side effects, discomfort, or medication questions you want your doctor to know about?",
    },
    {
        "key": "significant_events",
        "label": "Events",
        "prompt": "Did anything significant happen today, good or hard, that may matter clinically?",
    },
    {
        "key": "safety_concerns",
        "label": "Safety",
        "prompt": "Last safety check: any thoughts of harming yourself, feeling unsafe, or being unable to stay safe?",
    },
]


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def daily_checkin_completed_today(patient: dict, today: date | None = None) -> bool:
    today = today or datetime.now().date()
    for checkin in patient.get("daily_checkins", []) or []:
        if _parse_date(str(checkin.get("created_at", ""))) == today:
            return True
    return False


def build_daily_nancy_questions(patient: dict) -> list[dict]:
    questions = [dict(question) for question in DAILY_CHECKIN_QUESTIONS]
    for task in (patient.get("nancy_tasks") or [])[:3]:
        title = (task.get("title") or "the plan from your doctor").strip()
        instructions = (task.get("instructions") or "").strip()
        questions.insert(
            -1,
            {
                "key": "clinical_summary",
                "label": title,
                "prompt": (
                    f"Your doctor asked me to check on {title}. "
                    + (f"{instructions} " if instructions else "")
                    + "How has that been going?"
                ),
            },
        )
    return questions


def build_daily_nancy_status(patient: dict) -> dict:
    completed_today = daily_checkin_completed_today(patient)
    latest_checkin = (patient.get("daily_checkins") or [None])[0]
    return {
        "due": not completed_today,
        "completed_today": completed_today,
        "latest_checkin": latest_checkin,
        "question_count": len(build_daily_nancy_questions(patient)),
        "questions": build_daily_nancy_questions(patient),
    }


def summarize_checkin_payload(payload: CheckinPayload) -> str:
    summary_bits = []
    for label, value, suffix in [
        ("mood", payload.mood_score, "/10"),
        ("anxiety", payload.anxiety_score, "/10"),
        ("sleep", payload.sleep_hours, "h"),
        ("cognition", payload.cognition_score, "/10"),
        ("memory", payload.memory_score, "/10"),
    ]:
        if value.strip():
            summary_bits.append(f"{label} {value.strip()}{suffix}")

    for narrative in [
        payload.daily_update,
        payload.clinical_summary,
        payload.significant_events,
        payload.safety_concerns,
    ]:
        narrative = narrative.strip()
        if narrative:
            summary_bits.append(narrative)

    return "; ".join(summary_bits) if summary_bits else "Patient completed an async daily report."


def create_nancy_touchpoint(doctor: dict, patient: dict, plan: dict, patient_report: str, mode: str) -> dict:
    return add_nancy_interaction(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        payload={
            "mode": mode,
            "conversation_goal": plan.get("conversation_goal", "between-session support"),
            "patient_report": patient_report,
            "clinician_summary": plan.get("clinician_summary", ""),
            "observed_mood": plan.get("observed_mood", ""),
            "functioning_note": plan.get("functioning_note", ""),
            "cognition_note": plan.get("cognition_note", ""),
            "medication_note": plan.get("medication_note", ""),
            "safety_note": plan.get("safety_note", ""),
            "recommended_follow_up": plan.get("recommended_follow_up", ""),
            "escalation_level": plan.get("escalation_level", "routine"),
        },
    )


def create_async_alert_if_needed(
    doctor: dict,
    patient: dict,
    plan: dict,
    source: str,
    route: str = "/patient",
) -> dict | None:
    if not plan.get("requires_doctor_review"):
        return None

    risk_reasons = plan.get("risk_reasons", [])
    title = {
        "urgent": "Urgent async care escalation",
        "watch": "Async care review needed",
    }.get(plan.get("escalation_level", "watch"), "Async care review needed")
    summary_parts = [plan.get("clinician_summary", "").strip()]
    if risk_reasons:
        summary_parts.append("Signals: " + " | ".join(risk_reasons[:3]))

    alert = add_async_alert(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        severity=plan.get("escalation_level", "watch"),
        source=source,
        title=title,
        summary=" ".join(part for part in summary_parts if part),
        recommended_follow_up=plan.get("recommended_follow_up", ""),
        route=route,
        metadata={
            "doctor_name": doctor.get("name", ""),
            "hospital_name": doctor.get("hospital_name", ""),
            "department_name": doctor.get("department_name", ""),
        },
    )
    add_patient_message(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        sender_role="system",
        recipient="doctor",
        body=(
            f"{title} for {patient['name']}. "
            f"{plan.get('recommended_follow_up', '').strip()}"
        ).strip(),
        channel="async care alert",
    )
    return alert


def process_daily_checkin(doctor: dict, patient: dict, payload_dict: dict) -> dict:
    payload = CheckinPayload.from_dict(payload_dict)
    checkin_record = add_daily_checkin(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        payload=payload.to_dict(),
    )
    plan = build_async_care_plan(
        doctor=doctor,
        patient=patient,
        checkin_payload=payload.to_dict(),
        patient_message="",
    )

    if plan.get("patient_response"):
        add_patient_message(
            doctor_id=doctor["id"],
            patient_id=patient["id"],
            patient_name=patient["name"],
            sender_role="nancy",
            recipient="patient",
            body=plan["patient_response"],
            channel="nancy questionnaire follow-up",
        )

    interaction = create_nancy_touchpoint(
        doctor=doctor,
        patient=patient,
        plan=plan,
        patient_report=summarize_checkin_payload(payload),
        mode="daily questionnaire",
    )
    alert = create_async_alert_if_needed(doctor, patient, plan, source="daily questionnaire", route="/companion")

    return {
        "checkin": checkin_record,
        "plan": plan,
        "interaction": interaction,
        "alert": alert,
    }


def process_patient_message(
    doctor: dict,
    patient: dict,
    body: str,
    recipient: str,
    nancy_response_override: str | None = None,
) -> dict:
    message_record = add_patient_message(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        sender_role="patient",
        recipient=recipient,
        body=body,
        channel="patient companion chat",
    )

    plan = build_async_care_plan(
        doctor=doctor,
        patient=patient,
        patient_message=body,
    )
    if nancy_response_override and nancy_response_override.strip():
        plan["patient_response"] = nancy_response_override.strip()
    if plan.get("patient_response"):
        add_patient_message(
            doctor_id=doctor["id"],
            patient_id=patient["id"],
            patient_name=patient["name"],
            sender_role="nancy",
            recipient="patient",
            body=plan["patient_response"],
            channel="nancy async reply",
        )
    if plan.get("clinician_summary"):
        add_patient_message(
            doctor_id=doctor["id"],
            patient_id=patient["id"],
            patient_name=patient["name"],
            sender_role="nancy",
            recipient="doctor",
            body=plan["clinician_summary"],
            channel="nancy doctor handoff",
        )
    interaction = create_nancy_touchpoint(
        doctor=doctor,
        patient=patient,
        plan=plan,
        patient_report=body.strip(),
        mode="patient messaging",
    )
    alert = create_async_alert_if_needed(doctor, patient, plan, source="patient message", route="/patient")
    return {
        "message": message_record,
        "plan": plan,
        "interaction": interaction,
        "alert": alert,
    }


def process_nancy_proactive_ping(doctor: dict, patient: dict, support_message: str) -> dict:
    add_patient_message(
        doctor_id=doctor["id"],
        patient_id=patient["id"],
        patient_name=patient["name"],
        sender_role="nancy",
        recipient="patient",
        body=support_message,
        channel="nancy proactive outreach",
    )
    interaction = create_nancy_touchpoint(
        doctor=doctor,
        patient=patient,
        plan={
            "conversation_goal": "proactive support outreach",
            "clinician_summary": f"Nancy proactively reached out to {patient['name']} based on recent async state.",
            "observed_mood": "monitoring",
            "functioning_note": "Proactive outreach triggered from longitudinal check-in signals.",
            "cognition_note": "No new cognition data captured during outbound ping.",
            "medication_note": "No new medication note.",
            "safety_note": "",
            "recommended_follow_up": "Review whether the patient responds to proactive outreach before the next session.",
            "escalation_level": "watch" if patient.get("risk") == "High" else "routine",
        },
        patient_report=support_message,
        mode="proactive text",
    )
    return {"interaction": interaction}


def build_async_care_summary(patient: dict, doctor: dict) -> AsyncCareSummary:
    latest_checkin = (patient.get("daily_checkins") or [None])[0]
    patient_messages = [entry for entry in patient.get("patient_messages", []) if entry.get("sender_role") == "patient"]
    latest_patient_message = patient_messages[0].get("body", "") if patient_messages else ""
    risk = evaluate_async_risk(patient, checkin_payload=latest_checkin or {}, patient_message=latest_patient_message)
    latest_nancy = (patient.get("nancy_interactions") or [None])[0]
    alerts = patient.get("async_alerts", []) or []
    open_alerts = [alert for alert in alerts if alert.get("status") != "resolved"]
    urgent_alerts = [alert for alert in open_alerts if alert.get("severity") == "urgent"]
    review_needed = sum(1 for entry in patient.get("nancy_interactions", []) if entry.get("requires_doctor_review"))
    return AsyncCareSummary(
        risk_level=latest_nancy.get("escalation_level", risk["level"]) if latest_nancy else risk["level"],
        risk_reasons=risk["reasons"][:4],
        last_handoff=patient.get("last_nancy_summary", "No Nancy handoff logged yet."),
        task_count=len(patient.get("nancy_tasks", [])),
        message_count=len(patient.get("patient_messages", [])),
        review_needed=review_needed,
        open_alerts=len(open_alerts),
        urgent_alerts=len(urgent_alerts),
        doctor_name=doctor.get("name", "Doctor"),
        latest_checkin=latest_checkin,
    )


def build_patient_timeline(patient: dict, audience: str = "doctor") -> list[TimelineEntry]:
    timeline: list[TimelineEntry] = []
    for checkin in patient.get("daily_checkins", []):
        timeline.append(TimelineEntry(kind="checkin", created_at=checkin.get("created_at", ""), payload=checkin))
    for interaction in patient.get("nancy_interactions", []):
        if audience == "patient":
            if not interaction.get("patient_visible") and not interaction.get("patient_message"):
                continue
        timeline.append(TimelineEntry(kind="nancy", created_at=interaction.get("created_at", ""), payload=interaction))
    for message in patient.get("patient_messages", []):
        if audience == "patient":
            sender_role = message.get("sender_role", "")
            recipient = message.get("recipient", "")
            if sender_role in {"system"}:
                continue
            if sender_role == "nancy" and recipient == "doctor":
                continue
        timeline.append(TimelineEntry(kind="message", created_at=message.get("created_at", ""), payload=message))
    for session_record in patient.get("session_records", []):
        timeline.append(TimelineEntry(kind="session", created_at=session_record.get("created_at", ""), payload=session_record))
    for event in patient.get("sos_events", []):
        timeline.append(TimelineEntry(kind="sos", created_at=event.get("created_at", ""), payload=event))
    for outreach in patient.get("outreach_logs", []):
        timeline.append(TimelineEntry(kind="outreach", created_at=outreach.get("created_at", ""), payload=outreach))
    if audience != "patient":
        for alert in patient.get("async_alerts", []):
            timeline.append(TimelineEntry(kind="alert", created_at=alert.get("created_at", ""), payload=alert))
    return sorted(timeline, key=lambda item: item.created_at, reverse=True)


def get_doctor_alerts(doctor_id: str, statuses: set[str] | None = None) -> list[AsyncAlertRecord]:
    return [AsyncAlertRecord.from_dict(alert) for alert in get_async_alerts(doctor_id, statuses=statuses)]


def update_alert_status(doctor_id: str, alert_id: str, status: str, actor: str = "doctor", note: str = "") -> AsyncAlertRecord | None:
    updated = update_async_alert_status(doctor_id, alert_id, status, actor=actor, note=note)
    return AsyncAlertRecord.from_dict(updated) if updated else None


def summary_to_dict(summary: AsyncCareSummary) -> dict:
    return asdict(summary)
