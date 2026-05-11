"""
Doctor Insights Voice — OpenAI Realtime API (same bridge pattern as Nancy voice).

Browser ──PCM 24 kHz──► FastAPI WS ──► OpenAI Realtime (audio + tools).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from .doctor_insights_context import DOCTOR_INSIGHTS_SYSTEM, build_doctor_insights_context
from .facade import get_doctor_or_default
from .nancy_voice import NANCY_REALTIME_MODEL, NANCY_VOICE, _connect_realtime

DOCTOR_INSIGHTS_REALTIME_MODEL = os.getenv(
    "DOCTOR_INSIGHTS_REALTIME_MODEL",
    NANCY_REALTIME_MODEL,
)

_DOCTOR_REALTIME_INSTRUCTIONS = (
    DOCTOR_INSIGHTS_SYSTEM
    + "\n\nREALTIME VOICE SESSION:\n"
    "- You are in a live spoken conversation with the supervising psychiatrist.\n"
    "- Keep replies concise for speech unless they ask for depth.\n"
    "- Use refresh_mindscape_snapshot when they want the freshest panel, alerts, forum, or brief "
    "(for example after they took an action elsewhere in MindScape).\n"
)

DOCTOR_INSIGHTS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "refresh_mindscape_snapshot",
        "description": (
            "Fetch the latest MindScape snapshot: panel, open alerts, Nancy handoffs, weekly brief lines, "
            "performance signals, and top forum threads. Call when the doctor asks for updated priorities "
            "or explicitly wants fresh data."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]

_TOOL_OUTPUT_MAX = 14000


def _execute_doctor_insights_tool(fn_name: str, doctor_id: str) -> str:
    if fn_name == "refresh_mindscape_snapshot":
        text = build_doctor_insights_context(doctor_id)
        if len(text) > _TOOL_OUTPUT_MAX:
            return text[:_TOOL_OUTPUT_MAX] + "\n…(truncated)"
        return text or "(empty snapshot)"
    return "Unknown tool."


async def run_doctor_insights_voice_session(websocket: WebSocket, doctor_id: str) -> None:
    await websocket.accept()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "OPENAI_API_KEY not configured on server."})
        )
        await websocket.close()
        return

    doctor = get_doctor_or_default(doctor_id)
    snapshot_text = build_doctor_insights_context(doctor_id)
    instructions = (
        _DOCTOR_REALTIME_INSTRUCTIONS
        + "\n\nCURRENT SNAPSHOT AT SESSION START:\n"
        + snapshot_text
    )
    greeting_sent = False

    oai_ws = None
    try:
        oai_ws = await _connect_realtime(openai_key, DOCTOR_INSIGHTS_REALTIME_MODEL)
        await oai_ws.recv()

        await oai_ws.send(
            json.dumps(
                {
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
                        "tools": DOCTOR_INSIGHTS_TOOLS,
                        "tool_choice": "auto",
                    },
                }
            )
        )

        await websocket.send_text(json.dumps({"type": "ready"}))

        stop = asyncio.Event()

        async def browser_to_oai() -> None:
            nonlocal greeting_sent
            try:
                while not stop.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        audio_b64 = base64.b64encode(msg["bytes"]).decode()
                        await oai_ws.send(
                            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
                        )
                    elif "text" in msg and msg["text"]:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "close":
                            break
                        if ctrl.get("type") == "client_ready" and not greeting_sent:
                            greeting_sent = True
                            doc_name = (doctor.get("name") or "doctor").strip()
                            await oai_ws.send(
                                json.dumps(
                                    {
                                        "type": "response.create",
                                        "response": {
                                            "instructions": (
                                                f"Greet Dr. {doc_name} in one short warm sentence. "
                                                "Say you're MindScape Doctor Insights on live voice, "
                                                "and ask what they want first — panel priorities, alerts, forum signals, or weekly brief."
                                            ),
                                        },
                                    }
                                )
                            )
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                stop.set()
                try:
                    await oai_ws.close()
                except Exception:
                    pass

        async def oai_to_browser() -> None:
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
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": event.get("error", {}).get("message", "Realtime session error."),
                                    }
                                )
                            )
                        except Exception:
                            pass

                    if etype == "input_audio_buffer.speech_started":
                        try:
                            await websocket.send_text(json.dumps({"type": "UserStartedSpeaking"}))
                        except Exception:
                            pass

                    elif etype == "response.created":
                        try:
                            await websocket.send_text(json.dumps({"type": "AgentStartedSpeaking"}))
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
                            await websocket.send_text(json.dumps({"type": "AgentAudioDone"}))
                        except Exception:
                            pass

                    elif etype in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
                        transcript = event.get("transcript", "")
                        if transcript:
                            try:
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "ConversationText",
                                            "role": "assistant",
                                            "content": transcript,
                                        }
                                    )
                                )
                            except Exception:
                                pass

                    elif etype == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            try:
                                await websocket.send_text(
                                    json.dumps(
                                        {"type": "ConversationText", "role": "user", "content": transcript}
                                    )
                                )
                            except Exception:
                                pass

                    elif etype == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id", "")
                            fn_name = item.get("name", "")

                            output = _execute_doctor_insights_tool(fn_name, doctor_id)

                            try:
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "action",
                                            "function": fn_name,
                                            "result": output[:500] + ("…" if len(output) > 500 else ""),
                                        }
                                    )
                                )
                            except Exception:
                                pass

                            await oai_ws.send(
                                json.dumps(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": output,
                                        },
                                    }
                                )
                            )
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
            await websocket.close()
        except Exception:
            pass
