"""
Nancy Voice Agent — OpenAI Realtime API (single WebSocket, voice-to-voice).

Browser ──PCM 24 kHz──► FastAPI WS ──[base64 PCM]──► OpenAI Realtime
Browser ◄─PCM 24 kHz── FastAPI WS ◄─[base64 PCM]── OpenAI Realtime
                             │
           response.output_item.done (function_call) → execute → item.create
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any
from urllib.parse import quote

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from async_care import build_async_care_plan
from async_care_service import (
    create_async_alert_if_needed,
    create_nancy_touchpoint,
    process_daily_checkin,
)
from clinic_state import add_async_alert, add_patient_message

REALTIME_SAMPLE_RATE = 24000
NANCY_REALTIME_MODEL = os.getenv("NANCY_REALTIME_MODEL", "gpt-realtime")
NANCY_VOICE = "coral"  # warm female voice

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI Realtime format — flat, not wrapped in "function")
# ---------------------------------------------------------------------------
NANCY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "submit_daily_checkin",
        "description": (
            "Submit the patient's daily health questionnaire data extracted from the conversation. "
            "Call this once you have gathered enough scores through natural dialogue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mood_score": {"type": "string", "description": "Mood 0-10"},
                "anxiety_score": {"type": "string", "description": "Anxiety 0-10"},
                "sleep_hours": {"type": "string", "description": "Hours slept last night"},
                "energy_score": {"type": "string", "description": "Energy 0-10"},
                "stress_score": {"type": "string", "description": "Stress 0-10"},
                "cognition_score": {"type": "string", "description": "Mental clarity 0-10"},
                "memory_score": {"type": "string", "description": "Memory 0-10"},
                "functioning_score": {"type": "string", "description": "Daily functioning 0-10"},
                "medication_adherence": {"type": "string", "description": "Medication adherence 0-100%"},
                "side_effects": {"type": "string", "description": "Side effects mentioned"},
                "daily_update": {"type": "string", "description": "Summary of what patient shared"},
                "safety_concerns": {"type": "string", "description": "Any safety concerns raised"},
                "significant_events": {"type": "string", "description": "Significant events mentioned"},
            },
        },
    },
    {
        "type": "function",
        "name": "create_doctor_alert",
        "description": (
            "Create an alert for the supervising doctor when the patient discloses something "
            "clinically significant, a safety concern, or something that needs immediate attention."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["urgent", "watch", "routine"],
                    "description": "urgent = safety risk, watch = concerning, routine = informational",
                },
                "summary": {"type": "string", "description": "What the doctor needs to know"},
            },
            "required": ["severity", "summary"],
        },
    },
    {
        "type": "function",
        "name": "log_patient_statement",
        "description": "Log an important disclosure or statement from the patient to the medical record.",
        "parameters": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "The patient's statement to record verbatim"},
            },
            "required": ["body"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(patient: dict, doctor: dict, patient_context: str, mode: str = "support") -> str:
    directives = patient.get("nancy_tasks", [])
    directive_text = (
        "\n".join(f"- {t.get('title','')}: {t.get('instructions','')}" for t in directives[:5])
        or "No specific directives assigned yet."
    )
    shared = (
        f"You are Nancy, a warm and professional AI clinical companion representing "
        f"{doctor.get('name', 'the supervising doctor')} at {doctor.get('hospital_name', 'the hospital')}.\n\n"
        f"PATIENT: {patient.get('name', 'the patient')}\n"
        f"DIAGNOSIS: {patient.get('diagnosis', 'Unknown')}\n"
        f"CARE PLAN: {patient.get('care_plan', 'None on file')}\n"
        f"RISK LEVEL: {patient.get('risk', 'Unknown')}\n\n"
        f"FULL CLINICAL CONTEXT:\n{patient_context}\n\n"
        f"DOCTOR-APPROVED DIRECTIVES:\n{directive_text}\n\n"
        f"- You are speaking with {patient.get('name', 'the patient')} right now. Be human and warm."
        "\n- Speak only in English unless the patient explicitly asks to switch languages."
        "\n- If you ever begin replying in another language by mistake, immediately continue in English."
    )
    if mode == "daily":
        return (
            shared
            + "\n\nVOICE MODE: DAILY CONVERSATION\n"
            "This is the patient's once-a-day Nancy conversation. It must feel like warm care, not a survey.\n"
            "Your job is to gently gather the daily clinical picture through natural dialogue, then call submit_daily_checkin exactly once when you have enough information.\n\n"
            "CONVERSATION STYLE:\n"
            "- Start with a calm greeting and ask how today has actually felt.\n"
            "- Do not announce a checklist or questionnaire.\n"
            "- Ask one gentle question at a time, and respond to what the patient says before moving on.\n"
            "- Weave in mood, anxiety, sleep, energy, stress, functioning, focus, memory, medication adherence, side effects, significant events, doctor directives, and safety.\n"
            "- If the patient gives a rich narrative, summarize and ask only the missing essentials.\n"
            "- Use soft transitions: 'Can I ask about sleep for a moment?', 'I also want to make sure your doctor understands...'\n"
            "- Keep most spoken turns to 1-3 sentences.\n"
            "- Be caretaker-kind: steady, warm, nonjudgmental, never robotic.\n\n"
            "SAFETY AND BOUNDARIES:\n"
            "- If safety risk appears, pause the normal flow, give brief urgent guidance, and call create_doctor_alert.\n"
            "- Never diagnose, prescribe, or replace therapy.\n"
            "- Never imply secrecy from the doctor.\n\n"
            "SUBMISSION:\n"
            "- Call submit_daily_checkin only after you have enough daily signal.\n"
            "- If a precise score is missing, ask naturally. Do not invent numbers.\n"
            "- After submitting, tell the patient you sent a clear summary to the doctor and invite anything else they want to add."
        )
    return (
        shared
        + "\n\nVOICE MODE: GENERAL SUPPORT\n"
        "This is not the once-a-day daily check-in. Do not run the daily questionnaire here.\n"
        "Use this conversation for questions, concerns, symptoms, complaints, logistics, help requests, or anything the patient wants routed to the care team.\n\n"
        "YOUR ROLE:\n"
        "- Respond warmly and supportively.\n"
        "- If the patient shares clinically important information, call log_patient_statement.\n"
        "- If the patient raises a safety concern, call create_doctor_alert immediately.\n"
        "- If they ask for the daily check-in, guide them to the Daily Conversation section.\n"
        "- Never diagnose, prescribe, or replace therapy.\n"
        "- Keep responses short and spoken-friendly."
    )


async def _execute_function(
    fn_name: str,
    fn_input: dict,
    doctor_id: str,
    patient_id: str,
    doctor: dict,
    patient: dict,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": fn_name,
        "output": "Done.",
        "created_alert": False,
        "created_interaction": False,
        "logged_statement": False,
        "submitted_checkin": False,
    }
    try:
        if fn_name == "submit_daily_checkin":
            workflow = process_daily_checkin(doctor=doctor, patient=patient, payload_dict=fn_input)
            plan = workflow.get("plan", {})
            result["output"] = plan.get("clinician_summary") or "Daily check-in submitted successfully."
            result["created_alert"] = workflow.get("alert") is not None
            result["created_interaction"] = workflow.get("interaction") is not None
            result["submitted_checkin"] = True
        elif fn_name == "create_doctor_alert":
            add_async_alert(
                doctor_id=doctor_id,
                patient_id=patient_id,
                patient_name=patient["name"],
                severity=fn_input.get("severity", "watch"),
                source="nancy voice agent",
                title=f"Nancy flagged: {fn_input.get('summary', '')[:60]}",
                summary=fn_input.get("summary", ""),
                recommended_follow_up="Review Nancy voice session.",
                route="/patient",
            )
            result["output"] = "Alert created for the doctor."
            result["created_alert"] = True
        elif fn_name == "log_patient_statement":
            add_patient_message(
                doctor_id=doctor_id,
                patient_id=patient_id,
                patient_name=patient["name"],
                sender_role="patient",
                recipient="doctor",
                body=fn_input.get("body", ""),
                channel="nancy voice",
            )
            result["output"] = "I've noted that for your doctor."
            result["logged_statement"] = True
    except Exception as exc:
        result["output"] = f"Could not complete: {exc}"
    return result


def _build_realtime_url(model: str) -> str:
    resolved_model = (model or NANCY_REALTIME_MODEL).strip() or NANCY_REALTIME_MODEL
    return f"wss://api.openai.com/v1/realtime?model={quote(resolved_model)}"


def _compact_turns(turns: list[str], *, limit: int = 6, max_chars: int = 1800) -> str:
    cleaned = [" ".join(turn.split()) for turn in turns if turn and turn.strip()]
    if not cleaned:
        return ""
    joined = " | ".join(cleaned[-limit:])
    return joined[:max_chars].rstrip()


def _build_voice_session_plan(
    doctor: dict,
    patient: dict,
    user_turns: list[str],
    tool_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    patient_report = _compact_turns(user_turns)
    if not patient_report and not tool_events:
        return None

    plan = build_async_care_plan(
        doctor=doctor,
        patient=patient,
        patient_message=patient_report,
    )

    tool_notes: list[str] = []
    if any(event.get("submitted_checkin") for event in tool_events):
        tool_notes.append("Nancy captured and submitted a structured daily check-in during the call.")
    if any(event.get("created_alert") for event in tool_events):
        tool_notes.append("A doctor alert was already raised during the live voice session.")
    if any(event.get("logged_statement") for event in tool_events):
        tool_notes.append("Important patient statements were logged into the chart during the call.")

    clinician_summary = plan.get("clinician_summary", "").strip() or "Voice session completed."
    if tool_notes:
        clinician_summary = f"{clinician_summary} {' '.join(tool_notes)}".strip()

    recommended_follow_up = plan.get("recommended_follow_up", "").strip()
    if any(event.get("created_alert") for event in tool_events):
        recommended_follow_up = (
            "Review the voice-session alert and transcript alongside the structured handoff."
        )

    safety_note = plan.get("safety_note", "").strip()
    if any(event.get("created_alert") for event in tool_events):
        safety_note = "Voice session triggered live escalation."

    escalation_level = str(plan.get("escalation_level", "routine")).strip().lower() or "routine"
    if any(event.get("created_alert") for event in tool_events) and escalation_level == "routine":
        escalation_level = "watch"

    return {
        **plan,
        "conversation_goal": (
            "live voice daily check-in"
            if any(event.get("submitted_checkin") for event in tool_events)
            else "live Nancy voice support conversation"
        ),
        "clinician_summary": clinician_summary,
        "recommended_follow_up": recommended_follow_up,
        "safety_note": safety_note,
        "escalation_level": escalation_level,
        "requires_doctor_review": escalation_level in {"watch", "urgent"},
        "patient_report": patient_report or "Voice session completed without a transcript excerpt.",
    }


def _persist_voice_session(
    doctor: dict,
    patient: dict,
    user_turns: list[str],
    tool_events: list[dict[str, Any]],
) -> None:
    plan = _build_voice_session_plan(doctor, patient, user_turns, tool_events)
    if not plan:
        return

    create_nancy_touchpoint(
        doctor=doctor,
        patient=patient,
        plan=plan,
        patient_report=plan["patient_report"],
        mode="voice session",
    )

    if any(event.get("created_alert") for event in tool_events):
        return

    create_async_alert_if_needed(
        doctor=doctor,
        patient=patient,
        plan=plan,
        source="nancy voice session",
        route="/nancy",
    )


async def _connect_realtime(openai_key: str, model: str):
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    url = _build_realtime_url(model)
    connect_kwargs = {
        "max_size": 10 * 1024 * 1024,
    }

    try:
        return await websockets.connect(
            url,
            additional_headers=headers,
            **connect_kwargs,
        )
    except TypeError:
        return await websockets.connect(
            url,
            extra_headers=headers,
            **connect_kwargs,
        )


# ---------------------------------------------------------------------------
# Main session handler
# ---------------------------------------------------------------------------

async def run_nancy_voice_session(
    websocket: WebSocket,
    doctor_id: str,
    patient_id: str,
) -> None:
    from .facade import build_patient_context, get_doctor_or_default, get_patient_or_none

    await websocket.accept()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "OPENAI_API_KEY not configured on server."})
        )
        await websocket.close()
        return

    doctor = get_doctor_or_default(doctor_id)
    patient = get_patient_or_none(doctor_id, patient_id)
    if not patient:
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Patient {patient_id} not found."})
        )
        await websocket.close()
        return

    raw_mode = str(websocket.query_params.get("mode", "support")).strip().lower()
    mode = "daily" if raw_mode == "daily" else "support"
    patient_ctx = build_patient_context(patient, doctor_id) or ""
    instructions = _build_system_prompt(patient, doctor, patient_ctx, mode=mode)
    user_turns: list[str] = []
    tool_events: list[dict[str, Any]] = []
    greeting_sent = False

    oai_ws = None
    try:
        oai_ws = await _connect_realtime(openai_key, NANCY_REALTIME_MODEL)
        # Consume the initial session.created event
        await oai_ws.recv()

        # Configure Nancy's session
        await oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": instructions,
                "voice": NANCY_VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600,
                    "create_response": True,
                    "interrupt_response": True,
                },
                "tools": NANCY_TOOLS if mode == "daily" else [tool for tool in NANCY_TOOLS if tool.get("name") != "submit_daily_checkin"],
                "tool_choice": "auto",
            },
        }))

        # Signal browser that the session is live
        await websocket.send_text(json.dumps({"type": "ready"}))

        stop = asyncio.Event()

        async def browser_to_oai() -> None:
            """Forward browser PCM audio → OpenAI as base64."""
            nonlocal greeting_sent
            try:
                while not stop.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        audio_b64 = base64.b64encode(msg["bytes"]).decode()
                        await oai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }))
                    elif "text" in msg and msg["text"]:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "close":
                            break
                        if ctrl.get("type") == "client_ready" and not greeting_sent:
                            greeting_sent = True
                            await oai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {
                                    "instructions": (
                                        (
                                            f"Greet {patient.get('name', 'the patient')} warmly in English, say this is today's daily conversation, "
                                            "and ask what today has actually felt like. Do not mention forms or questionnaires."
                                        )
                                        if mode == "daily"
                                        else (
                                            f"Greet {patient.get('name', 'the patient')} warmly in English in one short sentence, "
                                            "then invite them to share what they need help with."
                                        )
                                    ),
                                },
                            }))
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                stop.set()
                try:
                    await oai_ws.close()
                except Exception:
                    pass

        async def oai_to_browser() -> None:
            """Forward OpenAI events → browser; handle function calls inline."""
            try:
                async for raw in oai_ws:
                    if stop.is_set():
                        break
                    if isinstance(raw, bytes):
                        continue

                    event: dict = json.loads(raw)
                    etype = event.get("type", "")

                    if etype == "error":
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": event.get("error", {}).get("message", "Realtime session error."),
                            }))
                        except Exception:
                            pass

                    if etype == "input_audio_buffer.speech_started":
                        try:
                            await websocket.send_text(
                                json.dumps({"type": "UserStartedSpeaking"})
                            )
                        except Exception:
                            pass

                    elif etype == "response.created":
                        try:
                            await websocket.send_text(
                                json.dumps({"type": "AgentStartedSpeaking"})
                            )
                        except Exception:
                            pass

                    elif etype in {"response.audio.delta", "response.output_audio.delta"}:
                        audio_bytes = base64.b64decode(event.get("delta", ""))
                        if audio_bytes:
                            try:
                                await websocket.send_bytes(audio_bytes)
                            except Exception:
                                pass

                    elif etype in {"response.audio.done", "response.output_audio.done"}:
                        try:
                            await websocket.send_text(
                                json.dumps({"type": "AgentAudioDone"})
                            )
                        except Exception:
                            pass

                    elif etype in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
                        transcript = event.get("transcript", "")
                        if transcript:
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "ConversationText",
                                    "role": "assistant",
                                    "content": transcript,
                                }))
                            except Exception:
                                pass

                    elif etype == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            user_turns.append(transcript)
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "ConversationText",
                                    "role": "user",
                                    "content": transcript,
                                }))
                            except Exception:
                                pass

                    elif etype == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id", "")
                            fn_name = item.get("name", "")
                            fn_args = json.loads(item.get("arguments", "{}"))

                            tool_result = await _execute_function(
                                fn_name, fn_args, doctor_id, patient_id, doctor, patient
                            )
                            tool_events.append(tool_result)

                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "action",
                                    "function": fn_name,
                                    "result": tool_result["output"],
                                }))
                            except Exception:
                                pass

                            # Return result to OpenAI and trigger follow-up response
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": tool_result["output"],
                                },
                            }))
                            await oai_ws.send(json.dumps({"type": "response.create"}))

            except Exception:
                pass
            finally:
                stop.set()

        await asyncio.gather(browser_to_oai(), oai_to_browser())

    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        if oai_ws is not None:
            try:
                await oai_ws.close()
            except Exception:
                pass
        try:
            _persist_voice_session(
                doctor=doctor,
                patient=patient,
                user_turns=user_turns,
                tool_events=tool_events,
            )
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
