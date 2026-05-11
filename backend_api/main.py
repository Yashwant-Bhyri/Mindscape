from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # load .env before any os.getenv calls

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from async_care_service import (
    process_daily_checkin,
    process_nancy_proactive_ping,
    process_patient_message,
    update_alert_status,
)
from clinic_state import (
    add_async_alert,
    add_forum_reply,
    add_forum_thread,
    increment_forum_vote,
    toggle_forum_saved_thread,
    add_nancy_interaction,
    add_nancy_task,
    add_outreach_log,
    add_patient_message,
    add_patient_record,
    add_session_note,
    add_sos_event,
)
from nancy_agent import build_nancy_settings_json
from product_data import ORGANIZATIONS
from session_service import analyze_uploaded_file, save_clinician_session_note
from shenzhen_directory import choose_emergency_hospital

from .doctor_insights_context import DOCTOR_INSIGHTS_SYSTEM, build_doctor_insights_context
from .doctor_insights_voice import run_doctor_insights_voice_session
from .nancy_voice import run_nancy_voice_session
from .facade import (
    build_nancy_support_text,
    build_patient_context,
    doctor_exists,
    get_companion_payload,
    get_doctors_payload,
    get_doctor_profile_payload,
    get_nancy_payload,
    get_organization_payload,
    get_patient_or_none,
    get_patient_access_payload,
    get_patient_payload,
    get_platform_payload,
    get_research_payload,
    get_session_payload,
    get_workspace_payload,
    get_doctor_or_default,
)


class IntakeRequest(BaseModel):
    name: str
    concern: str
    context: str = ""


class OutreachRequest(BaseModel):
    target: str = ""
    message: str
    patient_id: str | None = None


class DailyCheckinRequest(BaseModel):
    mood_score: str = "5"
    anxiety_score: str = "5"
    sleep_hours: str = "7"
    energy_score: str = "5"
    stress_score: str = "5"
    cognition_score: str = "5"
    memory_score: str = "5"
    functioning_score: str = "5"
    medication_adherence: str = "100"
    side_effects: str = ""
    daily_update: str = ""
    safety_concerns: str = ""
    significant_events: str = ""
    clinical_summary: str = ""


class MessageRequest(BaseModel):
    body: str
    recipient: str = Field(pattern="^(nancy|doctor|both)$")
    sender_role: str = "patient"


class NancyDirectiveRequest(BaseModel):
    title: str
    instructions: str
    category: str = "Recovery"
    due_label: str = ""


class NancyTouchpointRequest(BaseModel):
    mode: str = "voice"
    conversation_goal: str = ""
    patient_report: str = ""
    patient_message: str = ""
    clinician_summary: str = ""
    observed_mood: str = ""
    functioning_note: str = ""
    cognition_note: str = ""
    medication_note: str = ""
    safety_note: str = ""
    recommended_follow_up: str = ""
    escalation_level: str = "routine"
    relay_to_patient: bool = False


class SosRequest(BaseModel):
    reason: str
    severity: str = "urgent"
    district: str = "福田区"
    notes: str = ""


class AlertStatusRequest(BaseModel):
    status: str = Field(pattern="^(new|acknowledged|resolved)$")
    actor: str = "doctor"
    note: str = ""


class SessionNoteRequest(BaseModel):
    title: str = ""
    note: str
    plan_update: str = ""
    disposition: str = "Continue current plan"


class ForumThreadCreate(BaseModel):
    title: str
    body: str
    flair: str = "Discussion"
    kind: str = "discussion"
    author: str = "You"
    link: str = ""
    link_title: str = ""
    community: str = "Doctor's Corner"


class ForumReplyCreate(BaseModel):
    body: str
    author: str = "You"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="MindScape API",
    version="0.1.0",
    summary="FastAPI backend for the MindScape clinical MVP",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _doctor_or_404(doctor_id: str) -> dict:
    if not doctor_exists(doctor_id):
        raise HTTPException(status_code=404, detail="Doctor not found")
    return get_doctor_or_default(doctor_id)


def _patient_or_404(doctor_id: str, patient_id: str) -> dict:
    patient = get_patient_or_none(doctor_id, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def _finalize_upload_analysis(file_obj, filename: str, patient_context: str | None) -> dict[str, Any]:
    completed: dict[str, Any] | None = None
    updates: list[dict[str, Any]] = []
    for update in analyze_uploaded_file(file_obj, filename, patient_context=patient_context):
        updates.append(update)
        if update.get("phase") == "completed":
            completed = update
    if not completed:
        raise HTTPException(status_code=500, detail="Session analysis did not complete")
    completed["updates"] = updates
    return completed


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/platform")
def platform():
    return get_platform_payload()


@app.get("/api/doctors")
def doctors():
    return get_doctors_payload()


@app.get("/api/patient-access")
def patient_access():
    return get_patient_access_payload()


@app.get("/api/organizations")
def organizations():
    return {"organizations": ORGANIZATIONS}


@app.get("/api/organizations/{organization_id}")
def organization_detail(organization_id: str):
    return get_organization_payload(organization_id)


@app.get("/api/doctors/{doctor_id}")
def doctor_profile(doctor_id: str):
    _doctor_or_404(doctor_id)
    return get_doctor_profile_payload(doctor_id)


@app.get("/api/doctors/{doctor_id}/workspace")
def doctor_workspace(doctor_id: str):
    return get_workspace_payload(doctor_id)


@app.post("/api/doctors/{doctor_id}/intake")
def create_intake(doctor_id: str, payload: IntakeRequest):
    _doctor_or_404(doctor_id)
    if not payload.name.strip() or not payload.concern.strip():
        raise HTTPException(status_code=400, detail="Patient name and concern are required")
    patient = add_patient_record(doctor_id, payload.name, payload.concern, payload.context)
    return {"patient": patient}


@app.post("/api/doctors/{doctor_id}/outreach")
def create_outreach(doctor_id: str, payload: OutreachRequest):
    doctor = _doctor_or_404(doctor_id)
    target = payload.target.strip()
    patient_id = payload.patient_id
    if patient_id:
        patient = _patient_or_404(doctor_id, patient_id)
        target = target or patient["name"]
    if not target or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Target and message are required")
    log = add_outreach_log(doctor["id"], target, payload.message, patient_id=patient_id)
    return {"outreach": log}


@app.get("/api/doctors/{doctor_id}/research")
def doctor_research(doctor_id: str):
    return get_research_payload(doctor_id)


@app.post("/api/doctors/{doctor_id}/research/threads")
def create_forum_thread(doctor_id: str, payload: ForumThreadCreate):
    _doctor_or_404(doctor_id)
    title = payload.title.strip()
    body = payload.body.strip()
    if not title or not body:
        raise HTTPException(status_code=400, detail="Title and body are required")
    thread = add_forum_thread(
        doctor_id,
        title=title,
        body=body,
        flair=payload.flair.strip() or "Discussion",
        kind=(payload.kind or "discussion").strip().lower(),
        author=payload.author.strip() or "You",
        link=payload.link.strip(),
        link_title=payload.link_title.strip(),
        community=payload.community.strip() or "Doctor's Corner",
    )
    return {"thread": thread, "research": get_research_payload(doctor_id)}


@app.post("/api/doctors/{doctor_id}/research/threads/{thread_id}/replies")
def create_forum_reply(doctor_id: str, thread_id: str, payload: ForumReplyCreate):
    _doctor_or_404(doctor_id)
    body = payload.body.strip()
    if not body:
        raise HTTPException(status_code=400, detail="Reply body is required")
    reply = add_forum_reply(doctor_id, thread_id, body=body, author=payload.author.strip() or "You")
    if reply is None:
        raise HTTPException(status_code=400, detail="Could not save reply")
    return {"reply": reply, "research": get_research_payload(doctor_id)}


@app.post("/api/doctors/{doctor_id}/research/threads/{thread_id}/vote")
def vote_forum_thread(doctor_id: str, thread_id: str):
    _doctor_or_404(doctor_id)
    count = increment_forum_vote(doctor_id, thread_id)
    return {"vote_count": count, "research": get_research_payload(doctor_id)}


@app.post("/api/doctors/{doctor_id}/research/threads/{thread_id}/save")
def toggle_save_forum_thread(doctor_id: str, thread_id: str):
    _doctor_or_404(doctor_id)
    saved = toggle_forum_saved_thread(doctor_id, thread_id)
    return {"saved": saved, "research": get_research_payload(doctor_id)}


@app.get("/api/doctors/{doctor_id}/patients/{patient_id}")
def patient_profile(doctor_id: str, patient_id: str):
    return get_patient_payload(doctor_id, patient_id)


@app.get("/api/doctors/{doctor_id}/patients/{patient_id}/companion")
def patient_companion(doctor_id: str, patient_id: str):
    return get_companion_payload(doctor_id, patient_id)


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/checkins")
def create_checkin(doctor_id: str, patient_id: str, payload: DailyCheckinRequest):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    result = process_daily_checkin(doctor=doctor, patient=patient, payload_dict=payload.model_dump())
    refreshed = get_companion_payload(doctor_id, patient_id)
    return {"result": result, "snapshot": refreshed}


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/messages")
def create_message(doctor_id: str, patient_id: str, payload: MessageRequest):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    body = payload.body.strip()
    if not body:
        raise HTTPException(status_code=400, detail="Message body is required")

    if payload.sender_role.strip().lower() != "patient":
        message = add_patient_message(
            doctor_id=doctor_id,
            patient_id=patient_id,
            patient_name=patient["name"],
            sender_role=payload.sender_role.strip().lower(),
            recipient=payload.recipient,
            body=body,
        )
        refreshed = get_patient_payload(doctor_id, patient_id)
        return {"message": message, "snapshot": refreshed}

    result = process_patient_message(doctor=doctor, patient=patient, body=body, recipient=payload.recipient)
    refreshed = get_patient_payload(doctor_id, patient_id)
    return {"result": result, "snapshot": refreshed}


@app.get("/api/doctors/{doctor_id}/patients/{patient_id}/nancy")
def patient_nancy(doctor_id: str, patient_id: str):
    payload = get_nancy_payload(doctor_id, patient_id)
    patient = payload["patient"]
    doctor = payload["doctor"]
    payload["settings_preview"] = build_nancy_settings_json(
        patient_context=build_patient_context(patient, doctor_id) or "",
        doctor_context=(
            f"Doctor: {doctor.get('name', '')}\n"
            f"Specialty: {doctor.get('specialty', '')}\n"
            f"Hospital: {doctor.get('hospital_name', '')}\n"
            f"Department: {doctor.get('department_name', '')}"
        ),
        reminders=[task.get("title", "") for task in patient.get("nancy_tasks", [])[:5]],
    )
    return payload


@app.get("/api/doctors/{doctor_id}/nancy")
def doctor_nancy(doctor_id: str):
    return get_nancy_payload(doctor_id)


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/nancy/directives")
def create_nancy_directive(doctor_id: str, patient_id: str, payload: NancyDirectiveRequest):
    patient = _patient_or_404(doctor_id, patient_id)
    if not payload.title.strip() or not payload.instructions.strip():
        raise HTTPException(status_code=400, detail="Directive title and instructions are required")
    task = add_nancy_task(
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_name=patient["name"],
        title=payload.title,
        instructions=payload.instructions,
        category=payload.category,
        due_label=payload.due_label,
    )
    return {"task": task, "snapshot": get_nancy_payload(doctor_id, patient_id)}


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/nancy/touchpoints")
def create_nancy_touchpoint(doctor_id: str, patient_id: str, payload: NancyTouchpointRequest):
    patient = _patient_or_404(doctor_id, patient_id)
    if not payload.patient_report.strip() and not payload.clinician_summary.strip() and not payload.patient_message.strip():
        raise HTTPException(status_code=400, detail="Patient report or clinician summary is required")
    result = _create_nancy_touchpoint_record(
        doctor_id=doctor_id,
        patient=patient,
        payload_dict=payload.model_dump(),
        default_channel="nancy clinician relay",
    )
    return {
        "interaction": result["interaction"],
        "relayed_message": result["relayed_message"],
        "snapshot": get_nancy_payload(doctor_id, patient_id),
    }


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/nancy/support-ping")
def create_nancy_support_ping(doctor_id: str, patient_id: str):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    support_message = build_nancy_support_text(patient, doctor)
    result = process_nancy_proactive_ping(doctor, patient, support_message)
    return {"result": result, "support_message": support_message, "snapshot": get_nancy_payload(doctor_id, patient_id)}


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/sos")
def create_sos(doctor_id: str, patient_id: str, payload: SosRequest):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    if not payload.reason.strip():
        raise HTTPException(status_code=400, detail="SOS reason is required")

    district = payload.district.strip() or doctor.get("hospital_district", "")
    recommended = choose_emergency_hospital(district=district, preferred_hospital=doctor.get("hospital_name", ""))
    event = add_sos_event(
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_name=patient["name"],
        payload={
            "region": "Shenzhen, Guangdong, China",
            "district": district,
            "severity": payload.severity,
            "reason": payload.reason,
            "rule": "shenzhen-psychiatric-escalation-v1",
            "doctor_hospital": doctor.get("hospital_name", ""),
            "doctor_department": doctor.get("department_name", ""),
            "recommended_hospital": recommended.get("name", doctor.get("hospital_name", "")),
            "recommended_department": doctor.get("department_name", "Emergency / Psychiatry"),
            "emergency_number": "120",
            "safety_number": "110",
            "notes": payload.notes,
            "status": "Escalated to hospital + doctor",
        },
    )
    add_async_alert(
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_name=patient["name"],
        severity="urgent",
        source="sos escalation",
        title="Urgent SOS escalation opened",
        summary=(
            f"SOS escalation for {patient['name']}: {payload.reason.strip()}. "
            f"Route: {event['recommended_hospital']} / {event['recommended_department']}."
        ),
        recommended_follow_up="Immediate doctor review and emergency coordination required.",
        route="/patient",
        metadata={
            "district": district,
            "hospital": event["recommended_hospital"],
            "department": event["recommended_department"],
        },
    )
    add_patient_message(
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_name=patient["name"],
        sender_role="system",
        recipient="doctor",
        body=(
            f"SOS triggered for {patient['name']}. Severity: {event['severity']}. "
            f"Route: 120, {event['recommended_hospital']}, {event['recommended_department']}."
        ),
        channel="sos escalation",
    )
    return {"event": event, "snapshot": get_patient_payload(doctor_id, patient_id)}


@app.post("/api/doctors/{doctor_id}/alerts/{alert_id}")
def mutate_alert(doctor_id: str, alert_id: str, payload: AlertStatusRequest):
    updated = update_alert_status(doctor_id, alert_id, payload.status, actor=payload.actor, note=payload.note)
    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"alert": updated.to_dict(), "workspace": get_workspace_payload(doctor_id)}


@app.get("/api/doctors/{doctor_id}/patients/{patient_id}/session")
def patient_session(doctor_id: str, patient_id: str):
    return get_session_payload(doctor_id, patient_id)


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/session/notes")
def create_session_note(doctor_id: str, patient_id: str, payload: SessionNoteRequest):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    note = save_clinician_session_note(
        doctor=doctor,
        patient=patient,
        title=payload.title,
        note=payload.note,
        plan_update=payload.plan_update,
        disposition=payload.disposition,
    )
    if not note:
        raise HTTPException(status_code=400, detail="Session note could not be saved")
    return {"note": note, "snapshot": get_session_payload(doctor_id, patient_id)}


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/session/analyze-upload")
def analyze_session_upload(doctor_id: str, patient_id: str, file: UploadFile = File(...)):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    completed = _finalize_upload_analysis(
        file.file,
        file.filename or "upload.wav",
        patient_context=build_patient_context(patient, doctor_id),
    )
    from session_service import persist_session_result

    saved = persist_session_result(
        doctor=doctor,
        patient=patient,
        transcript=completed["transcript"],
        stt_provider=completed["provider"],
        result=completed["result"],
    )
    return {"analysis": completed, "saved_record": saved, "snapshot": get_session_payload(doctor_id, patient_id)}


# ---------------------------------------------------------------------------
# Nancy Voice Agent — realtime Nancy WebSocket proxy
# ---------------------------------------------------------------------------

@app.websocket("/ws/nancy/{doctor_id}/{patient_id}")
async def nancy_voice_ws(ws: WebSocket, doctor_id: str, patient_id: str):
    await run_nancy_voice_session(ws, doctor_id, patient_id)


@app.websocket("/ws/doctor-insights/{doctor_id}")
async def doctor_insights_voice_ws(ws: WebSocket, doctor_id: str):
    await run_doctor_insights_voice_session(ws, doctor_id)


# ---------------------------------------------------------------------------
# Nancy text assistants
# ---------------------------------------------------------------------------

_NANCY_PATIENT_SYSTEM = """You are Nancy, a clinically supervised AI companion working on behalf of a psychiatrist.
You are in the "Speak With Nancy" surface, not the once-a-day daily questionnaire.
Your role is to have a warm, supportive conversation with the patient for questions, issues, complaints,
symptom changes, logistics, help requests, and anything they want routed to the care team.

CLINICAL CONSTRAINTS:
- Never diagnose, prescribe, or provide therapy
- Always route safety concerns to the doctor immediately (create_alert)
- Stay within the doctor-approved directives listed below
- Be warm, calm, and non-judgmental
- This is the patient-side conversation surface, not the clinician control console
- Do not speak like an admin tool, chart bot, or doctor workflow copilot
- Do not run the full daily questionnaire here. If the patient asks for the daily check-in, tell them to use
  the Daily Conversation section.

WHAT YOU CAN DO:
1. Respond supportively and conversationally
2. Log important patient updates with add_patient_message
3. Escalate safety concerns with create_alert
4. Encourage the patient to complete the separate Daily Conversation if they have not done it today

OUTPUT FORMAT — always respond with valid JSON:
{
  "response": "What you say to the patient (warm, spoken, 1-3 sentences)",
  "actions": [
    {
      "type": "add_patient_message",
      "data": { "body": "Patient mentioned..." }
    },
    {
      "type": "create_alert",
      "data": { "severity": "urgent|watch|routine", "summary": "..." }
    }
  ]
}

Only include actions when you have enough information to act. Never invent data.
If actions is empty, return an empty array [].
"""

_NANCY_CLINICIAN_SYSTEM = """You are Nancy's clinician copilot for a psychiatrist using MindScape.
You are speaking to the doctor or clinical operator, not to the patient.

YOUR JOB:
- actively listen to the doctor's updates, concerns, and instructions
- help convert doctor intent into Nancy directives, support touchpoints, escalations, and concise follow-up plans
- summarize relevant patient records that are already present in context
- be operationally helpful, clinically aware, and concise

BOUNDARIES:
- Do not roleplay as if you are speaking to the patient unless explicitly asked to draft a patient-facing message
- Do not give diagnosis, medication changes, or legal/forensic advice
- Do not fabricate chart facts that are not present in context
- If the doctor gives ambiguous instructions, make the safest reasonable interpretation and say what you assumed
- Keep replies useful for a busy clinician: concise, direct, and action-oriented

AVAILABLE ACTIONS:
1. add_nancy_task
   Use when the doctor wants Nancy to follow up on a task, reminder, symptom, coping exercise, or plan.
2. add_nancy_touchpoint
   Use when the doctor gives a clinically useful update or wants a structured Nancy handoff saved.
3. create_alert
   Use when the doctor wants a review flag explicitly created.
4. queue_support_ping
   Use when the doctor wants Nancy to proactively reach out to the patient.
5. relay_to_patient
   Use when the doctor wants a patient-visible Nancy message sent without exposing clinician-only notes.

OUTPUT FORMAT — always respond with valid JSON:
{
  "response": "Short reply to the doctor explaining what you understood, what you did, or the record summary requested.",
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
        "conversation_goal": "Track worsening sleep and panic anticipation",
        "patient_report": "Doctor reported increased panic anticipation and fragmented sleep.",
        "patient_message": "Hi, this is Nancy. Dr. Khan asked me to check in about your sleep and whether panic anticipation has felt stronger lately.",
        "clinician_summary": "Doctor update logged for Nancy follow-up.",
        "observed_mood": "monitoring",
        "functioning_note": "Sleep disruption may be affecting daytime functioning.",
        "cognition_note": "No new cognition update.",
        "medication_note": "No medication change requested.",
        "safety_note": "",
        "recommended_follow_up": "Nancy should check sleep, panic anticipation, and coping completion.",
        "escalation_level": "watch",
        "relay_to_patient": true
      }
    },
    {
      "type": "create_alert",
      "data": { "severity": "urgent|watch|routine", "summary": "..." }
    },
    {
      "type": "queue_support_ping",
      "data": { "message": "Optional override message to send to the patient" }
    },
    {
      "type": "relay_to_patient",
      "data": { "message": "Short patient-facing Nancy message to deliver." }
    }
  ]
}

Rules:
- Only include actions that are actually supported by the doctor's message.
- If the doctor is only asking for a summary or recommendation, actions can be [].
- When drafting a directive, prefer concrete, non-judgmental language Nancy can reuse.
- When drafting a touchpoint, keep it clinically structured and chart-friendly.
- If the doctor wants the patient to actually receive a Nancy message, use relay_to_patient or set relay_to_patient=true on add_nancy_touchpoint.
"""


def _make_llm_client():
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Nancy LLM is not configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
    configured_base = os.getenv("OPENAI_BASE_URL", "").strip()
    uses_openrouter = bool(os.getenv("OPENROUTER_API_KEY")) or "openrouter.ai" in configured_base or key.startswith("sk-or-v1-")
    base = configured_base or ("https://openrouter.ai/api/v1" if uses_openrouter else None)
    return OpenAI(api_key=key, base_url=base)


def _nancy_llm_call(system: str, history: list[dict], *, max_tokens: int = 600) -> str:
    client = _make_llm_client()
    model = os.getenv("NANCY_OPENAI_MODEL") or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek/deepseek-v4-flash"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + history,
        max_tokens=max_tokens,
        temperature=0.7,
        timeout=30,
    )
    return resp.choices[0].message.content or ""


def _relay_nancy_message_to_patient(
    doctor_id: str,
    patient_id: str,
    patient_name: str,
    body: str,
    *,
    channel: str,
) -> dict | None:
    cleaned = body.strip()
    if not cleaned:
        return None
    return add_patient_message(
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_name=patient_name,
        sender_role="nancy",
        recipient="patient",
        body=cleaned,
        channel=channel,
    )


def _create_nancy_touchpoint_record(
    doctor_id: str,
    patient: dict,
    payload_dict: dict,
    *,
    default_channel: str,
) -> dict:
    relay_to_patient = bool(payload_dict.get("relay_to_patient", False))
    patient_message = str(payload_dict.get("patient_message", "")).strip()
    patient_report = str(payload_dict.get("patient_report", "")).strip()
    relay_body = (patient_message or patient_report) if relay_to_patient else ""

    interaction_payload = {
        **payload_dict,
        "patient_message": relay_body,
        "patient_visible": bool(relay_body),
        "relayed_to_patient": bool(relay_body),
        "audience": "shared" if relay_body else "clinician",
    }
    interaction = add_nancy_interaction(
        doctor_id=doctor_id,
        patient_id=patient["id"],
        patient_name=patient["name"],
        payload=interaction_payload,
    )
    relayed_message = _relay_nancy_message_to_patient(
        doctor_id,
        patient["id"],
        patient["name"],
        relay_body,
        channel=default_channel,
    )
    return {"interaction": interaction, "relayed_message": relayed_message}


class NancyChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class NancyChatRequest(BaseModel):
    message: str
    history: list[NancyChatMessage] = []


def _nancy_directive_text(patient: dict) -> str:
    directives = patient.get("nancy_tasks", [])
    return "\n".join(
        f"- {t.get('title','')}: {t.get('instructions','')}" for t in directives[:5]
    ) or "No specific directives set."


def _build_clinician_nancy_context(doctor: dict, patient: dict, doctor_id: str) -> str:
    patient_ctx = build_patient_context(patient, doctor_id) or ""
    recent_checkins = patient.get("daily_checkins", [])[:3]
    recent_interactions = patient.get("nancy_interactions", [])[:3]
    recent_messages = patient.get("patient_messages", [])[:4]
    open_alerts = [alert for alert in patient.get("async_alerts", []) if alert.get("status") != "resolved"][:4]

    lines = [
        f"SUPERVISING DOCTOR: {doctor.get('name','')}, {doctor.get('specialty','')}",
        f"PATIENT CONTEXT:\n{patient_ctx}",
        f"DOCTOR-APPROVED DIRECTIVES:\n{_nancy_directive_text(patient)}",
    ]
    if recent_checkins:
        lines.append(
            "RECENT DAILY CHECK-INS:\n"
            + "\n".join(
                f"- {item.get('date_label','')}: mood {item.get('mood_score','')}/10, anxiety {item.get('anxiety_score','')}/10, sleep {item.get('sleep_hours','')}h. "
                f"{item.get('clinical_summary') or item.get('daily_update','')}"
                for item in recent_checkins
            )
        )
    if recent_interactions:
        lines.append(
            "RECENT NANCY HANDOFFS:\n"
            + "\n".join(
                f"- {item.get('date_label','')}: {item.get('clinician_summary') or item.get('patient_report','')}"
                for item in recent_interactions
            )
        )
    if recent_messages:
        lines.append(
            "RECENT PATIENT / NANCY MESSAGES:\n"
            + "\n".join(
                f"- {item.get('sender_role','unknown')} -> {item.get('recipient','unknown')}: {item.get('body','')}"
                for item in recent_messages
            )
        )
    if open_alerts:
        lines.append(
            "OPEN ALERTS:\n"
            + "\n".join(
                f"- {item.get('severity','watch')}: {item.get('summary','')}"
                for item in open_alerts
            )
        )
    return "\n\n".join(lines)


def _parse_nancy_json(raw: str) -> dict:
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return {"response": raw.strip(), "actions": []}


def _execute_patient_nancy_actions(
    doctor_id: str,
    patient_id: str,
    doctor: dict,
    patient: dict,
    actions: list[dict],
) -> list[dict]:
    actions_taken: list[dict] = []
    for action in actions:
        atype = action.get("type", "")
        data = action.get("data", {})
        try:
            if atype == "submit_daily_checkin":
                result = process_daily_checkin(doctor=doctor, patient=patient, payload_dict=data)
                actions_taken.append({"type": atype, "status": "ok", "result": result})
            elif atype == "add_patient_message":
                msg = add_patient_message(
                    doctor_id=doctor_id,
                    patient_id=patient_id,
                    patient_name=patient["name"],
                    sender_role="patient",
                    recipient="doctor",
                    body=data.get("body", ""),
                    channel="nancy patient chat",
                )
                actions_taken.append({"type": atype, "status": "ok", "message_id": msg.get("id")})
            elif atype == "create_alert":
                alert = add_async_alert(
                    doctor_id=doctor_id,
                    patient_id=patient_id,
                    patient_name=patient["name"],
                    severity=data.get("severity", "watch"),
                    source="nancy patient chat",
                    title=f"Nancy flagged: {data.get('summary','')[:60]}",
                    summary=data.get("summary", ""),
                    recommended_follow_up="Review patient-side Nancy conversation.",
                    route="/patient",
                )
                actions_taken.append({"type": atype, "status": "ok", "alert_id": alert.get("id")})
        except Exception as exc:
            actions_taken.append({"type": atype, "status": "error", "error": str(exc)})
    return actions_taken


def _execute_clinician_nancy_actions(
    doctor_id: str,
    patient_id: str,
    doctor: dict,
    patient: dict,
    actions: list[dict],
) -> list[dict]:
    actions_taken: list[dict] = []
    for action in actions:
        atype = action.get("type", "")
        data = action.get("data", {})
        try:
            if atype == "add_nancy_task":
                task = add_nancy_task(
                    doctor_id=doctor_id,
                    patient_id=patient_id,
                    patient_name=patient["name"],
                    title=data.get("title", ""),
                    instructions=data.get("instructions", ""),
                    category=data.get("category", "Recovery"),
                    due_label=data.get("due_label", ""),
                )
                actions_taken.append({"type": atype, "status": "ok", "task_id": task.get("id")})
            elif atype == "add_nancy_touchpoint":
                result = _create_nancy_touchpoint_record(
                    doctor_id=doctor_id,
                    patient=patient,
                    payload_dict=data,
                    default_channel="nancy clinician relay",
                )
                actions_taken.append(
                    {
                        "type": atype,
                        "status": "ok",
                        "interaction_id": result["interaction"].get("id"),
                        "message_id": result["relayed_message"].get("id") if result["relayed_message"] else None,
                    }
                )
            elif atype == "create_alert":
                alert = add_async_alert(
                    doctor_id=doctor_id,
                    patient_id=patient_id,
                    patient_name=patient["name"],
                    severity=data.get("severity", "watch"),
                    source="nancy clinician assistant",
                    title=f"Doctor requested Nancy review: {data.get('summary','')[:50]}",
                    summary=data.get("summary", ""),
                    recommended_follow_up="Review the clinician-requested Nancy follow-up.",
                    route="/nancy",
                )
                actions_taken.append({"type": atype, "status": "ok", "alert_id": alert.get("id")})
            elif atype == "queue_support_ping":
                override = str(data.get("message", "")).strip()
                support_message = override or build_nancy_support_text(patient, doctor)
                result = process_nancy_proactive_ping(doctor, patient, support_message)
                actions_taken.append({"type": atype, "status": "ok", "interaction_id": result["interaction"].get("id")})
            elif atype == "relay_to_patient":
                relayed = _relay_nancy_message_to_patient(
                    doctor_id,
                    patient_id,
                    patient["name"],
                    str(data.get("message", "")).strip(),
                    channel="nancy clinician relay",
                )
                actions_taken.append(
                    {
                        "type": atype,
                        "status": "ok" if relayed else "error",
                        "message_id": relayed.get("id") if relayed else None,
                    }
                )
        except Exception as exc:
            actions_taken.append({"type": atype, "status": "error", "error": str(exc)})
    return actions_taken


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/nancy/chat")
def nancy_chat(doctor_id: str, patient_id: str, payload: NancyChatRequest):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)
    patient_ctx = build_patient_context(patient, doctor_id) or ""
    system = (
        _NANCY_PATIENT_SYSTEM
        + f"\n\nPATIENT CONTEXT:\n{patient_ctx}"
        + f"\n\nDOCTOR-APPROVED DIRECTIVES:\n{_nancy_directive_text(patient)}"
        + f"\n\nSUPERVISING DOCTOR: {doctor.get('name','')}, {doctor.get('specialty','')}"
    )

    history_dicts = [{"role": m.role, "content": m.content} for m in payload.history]
    history_dicts.append({"role": "user", "content": payload.message})

    try:
        raw = _nancy_llm_call(system, history_dicts)
        parsed = _parse_nancy_json(raw)
        response_text: str = parsed.get("response", raw.strip())
        actions: list[dict] = parsed.get("actions", [])
    except Exception:
        fallback = process_patient_message(
            doctor=doctor,
            patient=patient,
            body=payload.message,
            recipient="nancy",
        )
        plan = fallback.get("plan") or {}
        return {
            "response": plan.get("patient_response", "I logged that for your care team."),
            "actions_taken": [{"type": "patient_message", "status": "ok"}],
            "raw_llm": "",
        }

    actions_taken = _execute_patient_nancy_actions(doctor_id, patient_id, doctor, patient, actions)
    message_result = process_patient_message(
        doctor=doctor,
        patient=patient,
        body=payload.message,
        recipient="nancy",
        nancy_response_override=response_text,
    )
    actions_taken.append(
        {
            "type": "patient_message",
            "status": "ok",
            "message_id": message_result["message"].get("id"),
        }
    )

    return {
        "response": response_text,
        "actions_taken": actions_taken,
        "raw_llm": raw,
    }


@app.post("/api/doctors/{doctor_id}/patients/{patient_id}/nancy/clinician-chat")
def nancy_clinician_chat(doctor_id: str, patient_id: str, payload: NancyChatRequest):
    doctor = _doctor_or_404(doctor_id)
    patient = _patient_or_404(doctor_id, patient_id)

    system = _NANCY_CLINICIAN_SYSTEM + "\n\n" + _build_clinician_nancy_context(doctor, patient, doctor_id)

    history_dicts = [{"role": m.role, "content": m.content} for m in payload.history]
    history_dicts.append({"role": "user", "content": payload.message})

    raw = _nancy_llm_call(system, history_dicts)
    parsed = _parse_nancy_json(raw)

    response_text: str = parsed.get("response", raw.strip())
    actions: list[dict] = parsed.get("actions", [])
    actions_taken = _execute_clinician_nancy_actions(doctor_id, patient_id, doctor, patient, actions)

    return {
        "response": response_text,
        "actions_taken": actions_taken,
        "raw_llm": raw,
    }


@app.post("/api/doctors/{doctor_id}/doctor-insights/chat")
def doctor_insights_chat(doctor_id: str, payload: NancyChatRequest):
    """Doctor workspace + forum voice/text insights (no patient-scoped Nancy actions)."""
    _doctor_or_404(doctor_id)
    system = DOCTOR_INSIGHTS_SYSTEM + "\n\n" + build_doctor_insights_context(doctor_id)
    history_dicts = [{"role": m.role, "content": m.content} for m in payload.history]
    history_dicts.append({"role": "user", "content": payload.message})
    raw = _nancy_llm_call(system, history_dicts, max_tokens=900)
    return {"response": raw.strip()}
