"""Shared text context for doctor insights (REST chat + Realtime voice)."""

from __future__ import annotations

from .facade import get_research_payload, get_workspace_payload

DOCTOR_INSIGHTS_SYSTEM = """You are MindScape Doctor Insights — a concise assistant for a psychiatrist using MindScape.

You receive CONTEXT assembled from this doctor's workspace (panel, alerts, Nancy handoffs) plus research forum snapshot and weekly brief. Answer only from that CONTEXT plus safe general clinical reasoning. If something is not in CONTEXT, say you do not see it in MindScape and suggest where they might check (e.g. full chart, forum thread).

Voice-first: prefer short spoken-length answers unless the doctor asks for depth. No JSON. No bullet walls unless they ask for a list.

Safety: you are not rendering patient care decisions; remind them to verify in the chart for medication or safety-critical actions."""


def build_doctor_insights_context(doctor_id: str) -> str:
    workspace = get_workspace_payload(doctor_id)
    research = get_research_payload(doctor_id)
    doctor = workspace.get("doctor") or {}
    patients = workspace.get("patients") or []
    open_alerts = workspace.get("open_alerts") or []
    watchtower = workspace.get("nancy_watchtower") or []
    threads = list(research.get("forum_threads") or [])
    threads.sort(key=lambda t: int(t.get("vote_score") or 0), reverse=True)
    threads = threads[:15]

    lines = [
        f"DOCTOR: {doctor.get('name', '')} · {doctor.get('specialty', '')}",
        f"PANEL ({len(patients)} patients):",
    ]
    for p in patients[:24]:
        lines.append(
            f"- {p.get('name', '?')} | {p.get('diagnosis', '')} | risk {p.get('risk', '')} | last: {(p.get('last_update') or '')[:120]}"
        )
    if open_alerts:
        lines.append("OPEN ALERTS:")
        for a in open_alerts[:16]:
            lines.append(
                f"- [{a.get('severity', '')}] {(a.get('title') or '')[:80]} — {(a.get('summary') or '')[:160]}"
            )
    else:
        lines.append("OPEN ALERTS: none in snapshot.")

    if watchtower:
        lines.append("RECENT NANCY HANDOFFS (watchtower):")
        for w in watchtower[:10]:
            summary = (w.get("clinician_summary") or w.get("patient_report") or "")[:200]
            lines.append(f"- {w.get('patient_name', '?')}: {summary}")

    wb = research.get("weekly_brief") or []
    if wb:
        lines.append("WEEKLY BRIEF (agent digest lines):")
        for line in wb[:18]:
            lines.append(f"- {line}")

    perf = research.get("performance") or []
    if perf:
        lines.append("PERFORMANCE SIGNALS:")
        for item in perf[:12]:
            lines.append(f"- {item.get('label', '')}: {item.get('value', '')}")

    if threads:
        lines.append("FORUM THREADS (top by engagement in snapshot):")
        for t in threads:
            body = (t.get("body") or "")[:160].replace("\n", " ")
            lines.append(
                f"- [{t.get('kind') or t.get('flair') or 'post'}] {t.get('title', '')} · {body}"
            )

    return "\n".join(lines)
