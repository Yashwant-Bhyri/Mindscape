import os

# Apply before any AI libraries load to prevent Mesop thread contention.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["USE_TF"] = "0"

import re
from contextlib import contextmanager
from dataclasses import field

import mesop as me

from async_care import build_async_care_plan, evaluate_async_risk
from async_care_service import (
    build_async_care_summary as service_build_async_care_summary,
    build_patient_timeline as service_build_patient_timeline,
    process_daily_checkin,
    process_nancy_proactive_ping,
    process_patient_message,
    summary_to_dict,
    update_alert_status,
)
from clinic_state import (
    add_async_alert,
    add_daily_checkin,
    add_nancy_interaction,
    add_nancy_task,
    add_outreach_log,
    add_patient_record,
    add_patient_message,
    add_session_record,
    add_session_note,
    add_sos_event,
    get_async_alerts,
    get_daily_checkins,
    get_nancy_interactions,
    get_nancy_tasks,
    get_outreach_logs,
    get_patient_messages,
    get_session_notes,
    get_session_records,
    get_sos_events,
    update_async_alert_status,
)
from nancy_agent import build_nancy_settings_json
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
from shenzhen_directory import (
    choose_emergency_hospital,
    load_shenzhen_hospital_directory,
    load_shenzhen_psychiatric_directory,
)
from session_service import (
    analyze_live_capture,
    analyze_uploaded_file,
    build_session_prep,
    load_session_notes,
    persist_session_result,
    result_to_view_model,
    save_clinician_session_note,
)

APP_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap"
]

DOCTOR_SCOPED_ROUTES = {
    "/doctor",
    "/workspace",
    "/patient",
    "/companion",
    "/nancy",
    "/research",
    "/session",
}

PATIENT_CONTEXT_ROUTES = {
    "/patient",
    "/companion",
    "/nancy",
    "/session",
    "/research",
    "/workspace",
    "/doctor",
}

PALETTE = {
    "bg": "linear-gradient(180deg, #f7f1e7 0%, #edf3f4 100%)",
    "ink": "#10212b",
    "muted": "#5f6b75",
    "line": "rgba(16, 33, 43, 0.12)",
    "surface": "rgba(255, 253, 249, 0.88)",
    "surface_strong": "#fffdfa",
    "teal": "#0f766e",
    "teal_soft": "rgba(15, 118, 110, 0.10)",
    "navy": "#16384d",
    "navy_soft": "rgba(22, 56, 77, 0.10)",
    "amber": "#c66a1a",
    "amber_soft": "rgba(198, 106, 26, 0.10)",
    "rose": "#b5475c",
    "rose_soft": "rgba(181, 71, 92, 0.10)",
    "success": "#2d7b4b",
    "success_soft": "rgba(45, 123, 75, 0.10)",
    "shadow": "0 18px 50px rgba(27, 44, 60, 0.10)",
}

LANDING_PILLARS = [
    {
        "title": "Clinical Intelligence That Stays Grounded",
        "body": "Real-time sessions, retrieval-backed diagnostic support, and longitudinal evidence trails stay attached to the actual patient journey.",
    },
    {
        "title": "A Doctor Operating System, Not Just A Demo",
        "body": "Patient dashboards, appointment orchestration, check-ins, research ingestion, and collaboration all live in one physician-grade workflow.",
    },
    {
        "title": "Research Moves At Care Speed",
        "body": "Agent-curated weekly updates surface changing patterns, new evidence, and cross-clinic discussion threads before knowledge goes stale.",
    },
]

PLATFORM_STEPS = [
    "Onboard the organization and map doctor portfolios, research interests, and care priorities.",
    "Enter the doctor workspace to triage patients, review history, and open active communication loops.",
    "Launch a live session to run the current MindScape diagnostic engine inside a broader clinical context.",
    "Feed outcomes back into research, peer discussion, and automated weekly evidence briefs.",
]

@me.stateclass
class ApplicationState:
    transcript: str = ""
    is_recording: bool = False
    is_processing: bool = False
    status_message: str = "Ready for physician review"

    bsv_valence: float = 0.0
    bsv_arousal: float = 0.0
    bsv_dominance: float = 0.0

    hypothesis_name: str = "Awaiting session analysis"
    hypothesis_confidence: str = ""
    hypothesis_reasoning: str = ""
    hypothesis_evidence: str = ""
    retrieved_evidence: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    treatment_plan: str = ""
    reference_cases: list[dict] = field(default_factory=list)
    traumatic_markers: list[str] = field(default_factory=list)
    emotion_trajectory: list[dict] = field(default_factory=list)
    pipeline_active_node: int = 0
    pipeline_logs: list[str] = field(default_factory=list)
    safety_gate: str = "PENDING"

    intake_name: str = ""
    intake_concern: str = ""
    intake_context: str = ""
    outreach_target: str = ""
    outreach_message: str = ""
    stt_provider: str = ""
    checkin_mood_score: str = "5"
    checkin_anxiety_score: str = "5"
    checkin_sleep_hours: str = "7"
    checkin_energy_score: str = "5"
    checkin_stress_score: str = "5"
    checkin_cognition_score: str = "5"
    checkin_memory_score: str = "5"
    checkin_functioning_score: str = "5"
    checkin_medication_adherence: str = "100"
    checkin_side_effects: str = ""
    checkin_daily_update: str = ""
    checkin_safety_concerns: str = ""
    checkin_significant_events: str = ""
    checkin_clinical_summary: str = ""
    nancy_task_title: str = ""
    nancy_task_note: str = ""
    nancy_task_due: str = ""
    nancy_task_category: str = "Recovery"
    nancy_conversation_goal: str = ""
    nancy_patient_message: str = ""
    nancy_clinician_summary: str = ""
    nancy_observed_mood: str = ""
    nancy_functioning_note: str = ""
    nancy_cognition_note: str = ""
    nancy_medication_note: str = ""
    nancy_safety_note: str = ""
    nancy_recommended_follow_up: str = ""
    nancy_escalation_level: str = "routine"
    nancy_contact_mode: str = "voice"
    chat_message_body: str = ""
    chat_message_sender: str = "patient"
    sos_reason: str = ""
    sos_severity: str = "urgent"
    sos_location_district: str = "福田区"
    sos_notes: str = ""
    session_note_title: str = ""
    session_note_body: str = ""
    session_plan_update: str = ""
    session_disposition: str = "Continue current plan"

    # Somatic / Visual Behavioral State Vector (FaceMesh CV)
    camera_active: bool = False
    visual_bsv_facial_valence: float = 0.0
    visual_bsv_facial_arousal: float = 0.0
    visual_bsv_blink_rate: float = 0.0
    visual_bsv_flat_affect: float = 0.0
    visual_bsv_gaze_stability: float = 1.0
    visual_bsv_twitch_zones: list[str] = field(default_factory=list)
    visual_bsv_face_detected: bool = False
    visual_bsv_active_aus: list[str] = field(default_factory=list)


def on_load(e: me.LoadEvent):
    me.set_theme_mode("light")


def current_doctor_id() -> str:
    doctor_id = me.query_params.get("doctor", DEFAULT_DOCTOR_ID)
    if isinstance(doctor_id, list):
        return doctor_id[0] if doctor_id else DEFAULT_DOCTOR_ID
    return doctor_id or DEFAULT_DOCTOR_ID


def current_patient_id() -> str | None:
    patient_id = me.query_params.get("patient", "")
    if isinstance(patient_id, list):
        return patient_id[0] if patient_id else None
    return patient_id or None


def navigate_handler(path: str, doctor_id: str | None = None, patient_id: str | None = None):
    def handler(e: me.ClickEvent):
        params = {}
        resolved_doctor_id = doctor_id
        resolved_patient_id = patient_id

        if path in DOCTOR_SCOPED_ROUTES and not resolved_doctor_id:
            current_doctor = current_doctor_id()
            if current_doctor:
                resolved_doctor_id = current_doctor

        if path in PATIENT_CONTEXT_ROUTES and not resolved_patient_id:
            current_patient = current_patient_id()
            if current_patient:
                resolved_patient_id = current_patient

        if resolved_doctor_id:
            params["doctor"] = resolved_doctor_id
        if resolved_patient_id:
            params["patient"] = resolved_patient_id
        if not params:
            params = None
        me.navigate(path, query_params=params)

    # Mesop identifies handlers by __qualname__. Without unique names, all closures
    # from this factory share the same name and Mesop treats them as one handler,
    # always calling whichever was registered last in the render loop.
    safe_path = path.strip("/").replace("/", "_") or "home"
    uid = f"{safe_path}__{(doctor_id or 'none')}__{(patient_id or 'none')}"
    handler.__name__ = f"nav_to_{uid}"
    handler.__qualname__ = f"nav_to_{uid}"
    return handler


def reset_analysis_state(state: ApplicationState):
    state.transcript = ""
    state.is_recording = False
    state.is_processing = False
    state.stt_provider = ""
    state.bsv_valence = 0.0
    state.bsv_arousal = 0.0
    state.bsv_dominance = 0.0
    state.hypothesis_name = "Awaiting session analysis"
    state.hypothesis_confidence = ""
    state.hypothesis_reasoning = ""
    state.hypothesis_evidence = ""
    state.retrieved_evidence = []
    state.follow_up_questions = []
    state.treatment_plan = ""
    state.reference_cases = []
    state.traumatic_markers = []
    state.emotion_trajectory = []
    state.pipeline_active_node = 0
    state.pipeline_logs = []
    state.safety_gate = "PENDING"


def merge_patient_runtime_data(patient: dict | None, doctor_id: str) -> dict | None:
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
    async_alerts = get_async_alerts(doctor_id, patient_id=patient_id)
    outreach = [
        entry for entry in get_outreach_logs(doctor_id)
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

    return merged


def border_all(color: str) -> me.Border:
    return me.Border.all(me.BorderSide(width=1, color=color))


def border_none() -> me.Border:
    return me.Border.all(me.BorderSide(width=0, color="transparent"))


def card_style(background: str | None = None) -> me.Style:
    return me.Style(
        background=background or PALETTE["surface"],
        border=border_all(PALETTE["line"]),
        border_radius=24,
        padding=me.Padding.all(24),
        box_shadow=PALETTE["shadow"],
        backdrop_filter="blur(18px)",
    )


@contextmanager
def page_shell(active: str, doctor_id: str | None = None):
    with me.box(
        style=me.Style(
            min_height="100vh",
            background=(
                "radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 26%), "
                "radial-gradient(circle at top right, rgba(198, 106, 26, 0.08), transparent 22%), "
                + PALETTE["bg"]
            ),
            color=PALETTE["ink"],
            font_family="'IBM Plex Sans', sans-serif",
        )
    ):
        render_topbar(active, doctor_id)
        with me.box(
            style=me.Style(
                max_width="1280px",
                margin=me.Margin(left="auto", right="auto"),
                padding=me.Padding(top=32, bottom=40, left=28, right=28),
                display="flex",
                flex_direction="column",
                gap=28,
            )
        ):
            yield
        render_footer()


def render_topbar(active: str, doctor_id: str | None):
    doctor = get_doctor(doctor_id or DEFAULT_DOCTOR_ID)
    active_patient_id = current_patient_id()
    selected_patient = get_patient(doctor["id"], active_patient_id) if active_patient_id else None
    nav_items = [
        ("Platform", "/", None, None),
        ("Organizations", "/organizations", None, None),
        ("Portfolio", "/doctor", doctor["id"], selected_patient["id"] if selected_patient else None),
        ("Workspace", "/workspace", doctor["id"], selected_patient["id"] if selected_patient else None),
        ("Patient", "/patient", doctor["id"], selected_patient["id"] if selected_patient else None),
        ("Companion", "/companion", doctor["id"], selected_patient["id"] if selected_patient else None),
        ("Nancy", "/nancy", doctor["id"], selected_patient["id"] if selected_patient else None),
        ("Live Session", "/session", doctor["id"], selected_patient["id"] if selected_patient else None),
        ("Research", "/research", doctor["id"], selected_patient["id"] if selected_patient else None),
    ]

    with me.box(
        style=me.Style(
            position="sticky",
            top=0,
            z_index=10,
            backdrop_filter="blur(18px)",
            background="rgba(247, 241, 231, 0.82)",
            border=me.Border(bottom=me.BorderSide(width=1, color=PALETTE["line"])),
        )
    ):
        with me.box(
            style=me.Style(
                max_width="1280px",
                margin=me.Margin(left="auto", right="auto"),
                padding=me.Padding.symmetric(vertical=18, horizontal=28),
                display="flex",
                justify_content="space-between",
                align_items="center",
                gap=16,
                flex_wrap="wrap",
            )
        ):
            with me.box(style=me.Style(display="flex", align_items="center", gap=14, flex_wrap="wrap")):
                with me.box(
                    style=me.Style(
                        width=44,
                        height=44,
                        border_radius=14,
                        background="linear-gradient(135deg, #16384d 0%, #0f766e 100%)",
                        display="flex",
                        align_items="center",
                        justify_content="center",
                        box_shadow="0 10px 22px rgba(22, 56, 77, 0.22)",
                    )
                ):
                    me.text("M", style=me.Style(color="white", font_family="'Space Grotesk', sans-serif", font_weight="700", font_size=20))
                with me.box(style=me.Style(display="flex", flex_direction="column", gap=2)):
                    me.text("MindScape Clinical OS", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=20, font_weight="700"))
                    me.text(
                        "Where next-generation software meets longitudinal mental-health care.",
                        style=me.Style(color=PALETTE["muted"], font_size=13),
                    )

            with me.box(style=me.Style(display="flex", align_items="center", gap=10, flex_wrap="wrap")):
                for label, path, nav_doctor_id, nav_patient_id in nav_items:
                    is_active = active == path
                    me.button(
                        label=label,
                        type="flat",
                        on_click=navigate_handler(
                            path,
                            nav_doctor_id,
                            nav_patient_id,
                        ),
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=10),
                            background=PALETTE["navy"] if is_active else PALETTE["surface_strong"],
                            color="white" if is_active else PALETTE["ink"],
                            border=(border_none() if is_active else border_all(PALETTE["line"])),
                            font_weight="600",
                        ),
                    )

                with me.box(
                    style=me.Style(
                        margin=me.Margin(left=8),
                        padding=me.Padding.symmetric(horizontal=14, vertical=10),
                        border_radius=999,
                        background=PALETTE["teal_soft"],
                        border=border_all("rgba(15, 118, 110, 0.18)"),
                    )
                ):
                    me.text(doctor["name"], style=me.Style(color=PALETTE["teal"], font_weight="700", font_size=13))


def render_footer():
    with me.box(
        style=me.Style(
            margin=me.Margin(top=16),
            padding=me.Padding.symmetric(vertical=22, horizontal=28),
            border=me.Border(top=me.BorderSide(width=1, color=PALETTE["line"])),
            color=PALETTE["muted"],
            font_size=13,
        )
    ):
        with me.box(
            style=me.Style(
                max_width="1280px",
                margin=me.Margin(left="auto", right="auto"),
                display="flex",
                justify_content="space-between",
                align_items="center",
                gap=16,
                flex_wrap="wrap",
            )
        ):
            me.text("MindScape is a clinical decision-support layer and must be used with physician judgment.")
            me.text("Built for multi-doctor organizations, patient continuity, and fast-moving research translation.")


def render_hero(
    eyebrow: str,
    title: str,
    body: str,
    primary_label: str,
    primary_path: str,
    primary_doctor_id: str | None = None,
    primary_patient_id: str | None = None,
    secondary_label: str | None = None,
    secondary_path: str | None = None,
    secondary_doctor_id: str | None = None,
    secondary_patient_id: str | None = None,
):
    with me.box(
        style=me.Style(
            background=(
                "linear-gradient(135deg, rgba(22, 56, 77, 0.96) 0%, rgba(15, 118, 110, 0.92) 100%), "
                "linear-gradient(180deg, #16384d 0%, #0f766e 100%)"
            ),
            color="white",
            border_radius=32,
            padding=me.Padding.all(32),
            box_shadow="0 26px 60px rgba(22, 56, 77, 0.24)",
            display="flex",
            flex_direction="column",
            gap=20,
            overflow="hidden",
        )
    ):
        with me.box(style=me.Style(max_width="840px", display="flex", flex_direction="column", gap=14)):
            me.text(
                eyebrow,
                style=me.Style(
                    font_size=12,
                    font_weight="700",
                    text_transform="uppercase",
                    letter_spacing="1.2px",
                    color="rgba(255, 255, 255, 0.76)",
                ),
            )
            me.text(
                title,
                style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=46, line_height="1.05", font_weight="700"),
            )
            me.text(body, style=me.Style(font_size=18, line_height="1.65", color="rgba(255, 255, 255, 0.84)"))

        with me.box(style=me.Style(display="flex", gap=14, flex_wrap="wrap")):
            me.button(
                label=primary_label,
                on_click=navigate_handler(primary_path, primary_doctor_id, primary_patient_id),
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=20, vertical=13),
                    background="white",
                    color=PALETTE["navy"],
                    border=border_none(),
                    font_weight="700",
                ),
            )
            if secondary_label and secondary_path:
                me.button(
                    label=secondary_label,
                    on_click=navigate_handler(secondary_path, secondary_doctor_id, secondary_patient_id),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=20, vertical=13),
                        background="rgba(255, 255, 255, 0.08)",
                        color="white",
                        font_weight="700",
                        border=me.Border.all(me.BorderSide(width=1, style="solid", color="rgba(255, 255, 255, 0.22)")),
                    ),
                )


def render_section_header(title: str, body: str):
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=6)):
        me.text(title, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=28, font_weight="700"))
        me.text(body, style=me.Style(color=PALETTE["muted"], font_size=15, line_height="1.65"))


def render_stat_card(label: str, value: str, tone: str):
    tone_bg = {
        "teal": PALETTE["teal_soft"],
        "navy": PALETTE["navy_soft"],
        "amber": PALETTE["amber_soft"],
        "rose": PALETTE["rose_soft"],
    }.get(tone, PALETTE["surface"])
    tone_color = {
        "teal": PALETTE["teal"],
        "navy": PALETTE["navy"],
        "amber": PALETTE["amber"],
        "rose": PALETTE["rose"],
    }.get(tone, PALETTE["ink"])

    with me.box(style=card_style(tone_bg)):
        me.text(label, style=me.Style(color=PALETTE["muted"], font_size=13, text_transform="uppercase", letter_spacing="0.8px", font_weight="700"))
        me.text(value, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=30, font_weight="700", color=tone_color, margin=me.Margin(top=10)))


def render_bullet_list(items: list[str], tone: str = "teal"):
    tone_color = PALETTE["teal"] if tone == "teal" else PALETTE["navy"]
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=10)):
        for item in items:
            with me.box(style=me.Style(display="flex", align_items="flex-start", gap=10)):
                with me.box(
                    style=me.Style(
                        width=8,
                        height=8,
                        border_radius=999,
                        background=tone_color,
                        margin=me.Margin(top=8),
                        flex_shrink=0,
                    )
                ):
                    pass
                me.text(item, style=me.Style(color=PALETTE["ink"], font_size=15, line_height="1.6"))


def render_patient_card(patient: dict, doctor_id: str | None = None):
    risk_color = {
        "High": PALETTE["rose"],
        "Medium": PALETTE["amber"],
        "Low": PALETTE["success"],
    }.get(patient["risk"], PALETTE["teal"])

    with me.box(style=card_style()):
        with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="flex-start", gap=16, flex_wrap="wrap")):
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=6)):
                me.text(patient["name"], style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=23, font_weight="700"))
                me.text(patient["diagnosis"], style=me.Style(color=PALETTE["muted"], font_size=15))
            with me.box(
                style=me.Style(
                    padding=me.Padding.symmetric(horizontal=12, vertical=8),
                    border_radius=999,
                    background="rgba(255,255,255,0.9)",
                    border=me.Border.all(me.BorderSide(width=1, style="solid", color=risk_color)),
                )
            ):
                me.text(patient["risk"], style=me.Style(color=risk_color, font_weight="700", font_size=12))

        with me.box(style=me.Style(margin=me.Margin(top=16), display="flex", flex_direction="column", gap=10)):
            me.text(f"Status: {patient['status']}", style=me.Style(font_weight="600"))
            me.text(patient["history"], style=me.Style(color=PALETTE["muted"], font_size=14, line_height="1.6"))
            me.text(f"Next appointment: {patient['next_appointment']}", style=me.Style(font_size=14, color=PALETTE["navy"], font_weight="600"))
            if patient.get("last_daily_checkin_summary"):
                me.text(
                    f"Daily report: {patient['last_daily_checkin_summary']}",
                    style=me.Style(color=PALETTE["teal"], font_size=13, line_height="1.5"),
                )
            if patient.get("last_session_summary"):
                me.text(
                    f"Last consult insight: {patient['last_session_summary'][:160]}",
                    style=me.Style(color=PALETTE["amber"], font_size=13, line_height="1.5"),
                )

        with me.box(
            style=me.Style(
                margin=me.Margin(top=16),
                padding=me.Padding.all(14),
                border_radius=18,
                background=PALETTE["navy_soft"],
            )
        ):
            me.text(patient["last_update"], style=me.Style(font_size=14, color=PALETTE["ink"], line_height="1.55"))

        if doctor_id and patient.get("id"):
            with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", gap=10, flex_wrap="wrap")):
                me.button(
                    label="Patient Profile",
                    on_click=navigate_handler("/patient", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=14, vertical=10),
                        background=PALETTE["navy"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Start Session",
                    on_click=navigate_handler("/session", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=14, vertical=10),
                        background=PALETTE["surface_strong"],
                        color=PALETTE["ink"],
                        border=border_all(PALETTE["line"]),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Nancy",
                    on_click=navigate_handler("/nancy", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=14, vertical=10),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Companion",
                    on_click=navigate_handler("/companion", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=14, vertical=10),
                        background=PALETTE["amber"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )


def render_report_card(report: dict):
    with me.box(style=card_style()):
        me.text(report["title"], style=me.Style(font_weight="700", font_size=18))
        me.text(report["status"], style=me.Style(color=PALETTE["teal"], font_size=13, font_weight="700", margin=me.Margin(top=8)))
        me.text(f"Updated {report['updated']}", style=me.Style(color=PALETTE["muted"], font_size=13, margin=me.Margin(top=6)))


def render_data_point(label: str, value: str, tone: str = "navy"):
    tone_map = {
        "navy": PALETTE["navy_soft"],
        "teal": PALETTE["teal_soft"],
        "amber": PALETTE["amber_soft"],
        "rose": PALETTE["rose_soft"],
    }
    with me.box(
        style=me.Style(
            padding=me.Padding.all(16),
            border_radius=18,
            background=tone_map.get(tone, PALETTE["surface"]),
        )
    ):
        me.text(label, style=me.Style(color=PALETTE["muted"], font_size=12, text_transform="uppercase", font_weight="700"))
        me.text(value, style=me.Style(font_weight="700", margin=me.Margin(top=8), line_height="1.45"))


def render_patient_logs(title: str, items: list[dict], key_name: str):
    with me.box(style=card_style()):
        me.text(title, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
            for item in items:
                with me.box(
                    style=me.Style(
                        padding=me.Padding.all(16),
                        border_radius=18,
                        background="rgba(255, 255, 255, 0.72)",
                        border=border_all(PALETTE["line"]),
                    )
                ):
                    me.text(item.get(key_name, item.get("date", item.get("source", ""))), style=me.Style(font_weight="700", font_size=14))
                    content = item.get("summary") or item.get("entry") or item.get("note") or ""
                    if item.get("date") and key_name != "date":
                        me.text(item["date"], style=me.Style(color=PALETTE["muted"], font_size=12, margin=me.Margin(top=4)))
                    me.text(content, style=me.Style(color=PALETTE["ink"], font_size=14, line_height="1.55", margin=me.Margin(top=8)))


def render_metric_trend(title: str, items: list[dict], metric_key: str, tone: str):
    tone_soft = PALETTE[f"{tone}_soft"]
    tone_color = PALETTE[tone]
    recent = items[:5]
    if not recent:
        return

    with me.box(style=card_style()):
        me.text(title, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=20, font_weight="700"))
        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
            for item in reversed(recent):
                raw_value = item.get(metric_key, "0")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    value = 0.0
                width = max(6, min(100, int(value * 10)))
                with me.box(style=me.Style(display="grid", grid_template_columns="88px 1fr 42px", gap=12, align_items="center")):
                    me.text(item.get("date_label", item.get("date", "")), style=me.Style(color=PALETTE["muted"], font_size=12))
                    with me.box(style=me.Style(background="rgba(16, 33, 43, 0.10)", height=10, border_radius=999, overflow="hidden")):
                        with me.box(style=me.Style(background=tone_color, width=f"{width}%", height="100%", border_radius=999)):
                            pass
                    me.text(f"{value:.0f}", style=me.Style(color=tone_color, font_weight="700", font_size=12))
        with me.box(style=me.Style(margin=me.Margin(top=14), padding=me.Padding.all(14), border_radius=16, background=tone_soft)):
            me.text(f"Latest value: {recent[0].get(metric_key, 'N/A')}", style=me.Style(color=tone_color, font_weight="700"))


def render_daily_checkin_card(checkin: dict):
    with me.box(style=card_style("rgba(255, 255, 255, 0.76)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap")):
            me.text(f"{checkin.get('date_label', '')} {checkin.get('time_label', '')}", style=me.Style(font_weight="700"))
            me.text(
                f"Mood {checkin.get('mood_score', '')}/10 | Anxiety {checkin.get('anxiety_score', '')}/10 | Sleep {checkin.get('sleep_hours', '')}h",
                style=me.Style(color=PALETTE["navy"], font_size=13, font_weight="700"),
            )
        if checkin.get("clinical_summary"):
            me.text(checkin["clinical_summary"], style=me.Style(color=PALETTE["ink"], line_height="1.55", margin=me.Margin(top=10)))
        if checkin.get("daily_update"):
            me.text(checkin["daily_update"], style=me.Style(color=PALETTE["muted"], line_height="1.55", margin=me.Margin(top=10)))
        with me.box(style=me.Style(margin=me.Margin(top=14), display="grid", grid_template_columns="1fr 1fr 1fr", gap=10)):
            render_data_point("Energy", f"{checkin.get('energy_score', '')}/10", "amber")
            render_data_point("Cognition", f"{checkin.get('cognition_score', '')}/10", "teal")
            render_data_point("Memory", f"{checkin.get('memory_score', '')}/10", "navy")
        if checkin.get("significant_events"):
            with me.box(style=me.Style(margin=me.Margin(top=12), padding=me.Padding.all(14), border_radius=16, background=PALETTE["amber_soft"])):
                me.text("Interim event note", style=me.Style(color=PALETTE["amber"], font_weight="700", font_size=13))
                me.text(checkin["significant_events"], style=me.Style(line_height="1.55", margin=me.Margin(top=8)))
        if checkin.get("safety_concerns") or checkin.get("side_effects"):
            with me.box(style=me.Style(margin=me.Margin(top=12), display="flex", flex_direction="column", gap=8)):
                if checkin.get("safety_concerns"):
                    render_data_point("Safety concerns", checkin["safety_concerns"], "rose")
                if checkin.get("side_effects"):
                    render_data_point("Side effects", checkin["side_effects"], "amber")


def render_session_record_card(session_record: dict):
    with me.box(style=card_style("rgba(255, 255, 255, 0.76)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap", align_items="flex-start")):
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=6)):
                me.text(session_record.get("hypothesis_name", "Session analysis"), style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    f"{session_record.get('date_label', '')} {session_record.get('time_label', '')} | {session_record.get('transcription_provider', '')}",
                    style=me.Style(color=PALETTE["muted"], font_size=13),
                )
            with me.box(style=me.Style(display="flex", gap=10, flex_wrap="wrap")):
                render_data_point("Confidence", session_record.get("hypothesis_confidence", "N/A"), "navy")
                render_data_point("Safety gate", session_record.get("safety_gate", "FAIL"), "teal" if session_record.get("safety_gate") == "PASS" else "rose")

        if session_record.get("reasoning"):
            me.markdown(session_record["reasoning"], style=me.Style(color=PALETTE["ink"], line_height="1.65", margin=me.Margin(top=14)))

        with me.box(style=me.Style(margin=me.Margin(top=16), display="grid", grid_template_columns="1fr 1fr 1fr", gap=10)):
            bsv = session_record.get("bsv", {})
            render_data_point("Valence", f"{bsv.get('valence', 0.0):.2f}", "rose")
            render_data_point("Arousal", f"{bsv.get('arousal', 0.0):.2f}", "amber")
            render_data_point("Dominance", f"{bsv.get('dominance', 0.0):.2f}", "teal")

        if session_record.get("event_markers"):
            with me.box(style=me.Style(margin=me.Margin(top=16), display="flex", flex_direction="column", gap=10)):
                me.text("Detected session peaks", style=me.Style(font_weight="700"))
                for event in session_record["event_markers"][:5]:
                    with me.box(style=me.Style(padding=me.Padding.all(14), border_radius=16, background=PALETTE["navy_soft"])):
                        me.text(event.get("label", "Event"), style=me.Style(color=PALETTE["navy"], font_weight="700", font_size=13))
                        me.text(event.get("detail", ""), style=me.Style(line_height="1.55", margin=me.Margin(top=8)))
                        me.text(f"Observed response: {event.get('response', '')}", style=me.Style(color=PALETTE["muted"], font_size=13, margin=me.Margin(top=8)))

        if session_record.get("emotion_trajectory"):
            with me.box(style=me.Style(margin=me.Margin(top=16), display="grid", grid_template_columns="repeat(auto-fit, minmax(180px, 1fr))", gap=12)):
                for phase in session_record["emotion_trajectory"][:3]:
                    with me.box(style=me.Style(padding=me.Padding.all(14), border_radius=16, background=PALETTE["teal_soft"])):
                        me.text(phase.get("phase", "Phase"), style=me.Style(color=PALETTE["teal"], font_weight="700", font_size=13))
                        me.text(phase.get("dominant_emotion", "Unknown"), style=me.Style(font_weight="700", margin=me.Margin(top=6)))
                        me.text(phase.get("trigger", ""), style=me.Style(color=PALETTE["muted"], font_size=13, line_height="1.5", margin=me.Margin(top=8)))

        if session_record.get("treatment_plan"):
            with me.box(style=me.Style(margin=me.Margin(top=16), padding=me.Padding.all(14), border_radius=16, background=PALETTE["success_soft"])):
                me.text("Treatment pathway", style=me.Style(color=PALETTE["success"], font_weight="700", font_size=13))
                me.text(session_record["treatment_plan"], style=me.Style(line_height="1.55", margin=me.Margin(top=8)))


def render_session_note_card(session_note: dict):
    with me.box(style=card_style("rgba(255, 255, 255, 0.8)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap")):
            me.text(session_note.get("title", "Clinician review note"), style=me.Style(font_weight="700", font_size=18))
            me.text(
                f"{session_note.get('date_label', '')} {session_note.get('time_label', '')}",
                style=me.Style(color=PALETTE["muted"], font_size=13),
            )
        if session_note.get("disposition"):
            me.text(
                session_note["disposition"],
                style=me.Style(color=PALETTE["teal"], font_weight="700", font_size=13, margin=me.Margin(top=8)),
            )
        if session_note.get("note"):
            me.text(session_note["note"], style=me.Style(line_height="1.55", margin=me.Margin(top=10)))
        if session_note.get("plan_update"):
            with me.box(style=me.Style(margin=me.Margin(top=12), padding=me.Padding.all(14), border_radius=16, background=PALETTE["amber_soft"])):
                me.text("Plan update", style=me.Style(color=PALETTE["amber"], font_weight="700", font_size=13))
                me.text(session_note["plan_update"], style=me.Style(line_height="1.55", margin=me.Margin(top=8)))


def render_chat_thread(thread: dict):
    with me.box(style=card_style("rgba(255, 255, 255, 0.72)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap")):
            me.text(thread["patient"], style=me.Style(font_weight="700", font_size=17))
            me.text(thread["time"], style=me.Style(color=PALETTE["muted"], font_size=13))
        me.text(thread["channel"], style=me.Style(color=PALETTE["teal"], font_size=13, font_weight="700", margin=me.Margin(top=6)))
        me.text(thread["summary"], style=me.Style(color=PALETTE["ink"], font_size=14, line_height="1.55", margin=me.Margin(top=12)))


def render_forum_post(post: dict):
    with me.box(style=card_style()):
        me.text(post["community"], style=me.Style(color=PALETTE["amber"], font_size=12, text_transform="uppercase", font_weight="700", letter_spacing="0.8px"))
        me.text(post["title"], style=me.Style(font_weight="700", font_size=20, margin=me.Margin(top=10)))
        me.text(post["summary"], style=me.Style(color=PALETTE["muted"], line_height="1.6", margin=me.Margin(top=10)))
        me.text(post["activity"], style=me.Style(color=PALETTE["navy"], font_weight="700", font_size=13, margin=me.Margin(top=14)))


def render_status_banner(message: str, tone: str = "teal"):
    background = PALETTE["teal_soft"] if tone == "teal" else PALETTE["amber_soft"]
    color = PALETTE["teal"] if tone == "teal" else PALETTE["amber"]
    with me.box(
        style=me.Style(
            padding=me.Padding.symmetric(horizontal=18, vertical=14),
            border_radius=18,
            background=background,
            border=border_all("rgba(16, 33, 43, 0.10)"),
        )
    ):
        me.text(message, style=me.Style(color=color, font_weight="700"))


def render_code_panel(title: str, code_text: str):
    with me.box(style=card_style("rgba(22, 56, 77, 0.96)")):
        me.text(title, style=me.Style(color="white", font_family="'Space Grotesk', sans-serif", font_size=22, font_weight="700"))
        with me.box(
            style=me.Style(
                margin=me.Margin(top=16),
                padding=me.Padding.all(16),
                border_radius=18,
                background="rgba(8, 14, 18, 0.72)",
                border=border_all("rgba(255, 255, 255, 0.08)"),
            )
        ):
            preview_lines = code_text.splitlines()
            max_lines = 32
            for line in preview_lines[:max_lines]:
                me.text(
                    line,
                    style=me.Style(
                        color="rgba(255,255,255,0.9)",
                        font_family="'IBM Plex Sans', monospace",
                        font_size=12,
                        line_height="1.45",
                    ),
                )
            if len(preview_lines) > max_lines:
                me.text(
                    f"... {len(preview_lines) - max_lines} more lines hidden for stability.",
                    style=me.Style(color="rgba(255,255,255,0.68)", margin=me.Margin(top=10), font_size=12),
                )


def render_patient_route_hub(
    doctor: dict,
    patients: list[dict],
    title: str,
    body: str,
    primary_route: str,
    primary_label: str,
    tone: str = "teal",
):
    primary_background = PALETTE["teal"] if tone == "teal" else PALETTE["amber"]
    with me.box(style=card_style()):
        me.text(title, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        me.text(body, style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)))
        if not patients:
            with me.box(style=me.Style(margin=me.Margin(top=18), padding=me.Padding.all(16), border_radius=18, background=PALETTE["navy_soft"])):
                me.text("No patients are assigned to this doctor yet.", style=me.Style(color=PALETTE["navy"], font_weight="700"))
            return

        with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(280px, 1fr))", gap=16)):
            for panel_patient in patients:
                with me.box(
                    style=me.Style(
                        padding=me.Padding.all(18),
                        border_radius=20,
                        background=PALETTE["surface_strong"],
                        border=border_all(PALETTE["line"]),
                    )
                ):
                    me.text(panel_patient["name"], style=me.Style(font_weight="700", font_size=18))
                    me.text(panel_patient["diagnosis"], style=me.Style(color=PALETTE["muted"], font_size=14, margin=me.Margin(top=6)))
                    me.text(
                        panel_patient.get("last_update", "No recent update recorded yet."),
                        style=me.Style(font_size=14, line_height="1.55", margin=me.Margin(top=10)),
                    )
                    with me.box(style=me.Style(display="flex", gap=10, flex_wrap="wrap", margin=me.Margin(top=12))):
                        me.button(
                            label=primary_label,
                            on_click=navigate_handler(primary_route, doctor["id"], panel_patient["id"]),
                            style=me.Style(
                                border_radius=999,
                                padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                background=primary_background,
                                color="white",
                                border=border_none(),
                                font_weight="700",
                            ),
                        )
                        me.button(
                            label="Patient Record",
                            on_click=navigate_handler("/patient", doctor["id"], panel_patient["id"]),
                            style=me.Style(
                                border_radius=999,
                                padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                background=PALETTE["surface_strong"],
                                color=PALETTE["ink"],
                                border=border_all(PALETTE["line"]),
                                font_weight="700",
                            ),
                        )
                        me.button(
                            label="Session",
                            on_click=navigate_handler("/session", doctor["id"], panel_patient["id"]),
                            style=me.Style(
                                border_radius=999,
                                padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                background=PALETTE["navy"],
                                color="white",
                                border=border_none(),
                                font_weight="700",
                            ),
                        )


def render_nancy_task_card(task: dict):
    with me.box(style=card_style("rgba(255, 255, 255, 0.76)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap")):
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=6)):
                me.text(task.get("title", "Directive"), style=me.Style(font_weight="700", font_size=18))
                me.text(
                    f"{task.get('category', 'Recovery')} | {task.get('date_label', '')} {task.get('time_label', '')}",
                    style=me.Style(color=PALETTE["muted"], font_size=13),
                )
            with me.box(
                style=me.Style(
                    padding=me.Padding.symmetric(horizontal=12, vertical=8),
                    border_radius=999,
                    background=PALETTE["teal_soft"],
                )
            ):
                me.text(task.get("status", "Active"), style=me.Style(color=PALETTE["teal"], font_weight="700", font_size=12))
        me.text(task.get("instructions", ""), style=me.Style(line_height="1.55", margin=me.Margin(top=12)))
        if task.get("due_label"):
            me.text(f"Due / next revisit: {task['due_label']}", style=me.Style(color=PALETTE["amber"], font_weight="700", font_size=13, margin=me.Margin(top=10)))


def render_nancy_interaction_card(interaction: dict):
    escalation = interaction.get("escalation_level", "routine")
    tone = "rose" if escalation == "urgent" else ("amber" if escalation == "watch" else "teal")
    tone_soft = PALETTE[f"{tone}_soft"]
    tone_color = PALETTE[tone]

    with me.box(style=card_style("rgba(255, 255, 255, 0.76)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap", align_items="flex-start")):
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=6)):
                me.text(
                    f"Nancy {interaction.get('mode', 'voice').title()} Touchpoint",
                    style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=22, font_weight="700"),
                )
                me.text(
                    f"{interaction.get('date_label', '')} {interaction.get('time_label', '')}",
                    style=me.Style(color=PALETTE["muted"], font_size=13),
                )
            with me.box(
                style=me.Style(
                    padding=me.Padding.symmetric(horizontal=12, vertical=8),
                    border_radius=999,
                    background=tone_soft,
                )
            ):
                me.text(escalation.title(), style=me.Style(color=tone_color, font_weight="700", font_size=12))

        if interaction.get("conversation_goal"):
            me.text(
                f"Goal: {interaction['conversation_goal']}",
                style=me.Style(color=PALETTE["navy"], font_weight="700", font_size=13, margin=me.Margin(top=12)),
            )
        if interaction.get("patient_report"):
            me.text(interaction["patient_report"], style=me.Style(line_height="1.55", margin=me.Margin(top=10)))
        if interaction.get("clinician_summary"):
            with me.box(style=me.Style(margin=me.Margin(top=12), padding=me.Padding.all(14), border_radius=16, background=PALETTE["navy_soft"])):
                me.text("Doctor-facing summary", style=me.Style(color=PALETTE["navy"], font_weight="700", font_size=13))
                me.text(interaction["clinician_summary"], style=me.Style(line_height="1.55", margin=me.Margin(top=8)))

        with me.box(style=me.Style(margin=me.Margin(top=14), display="grid", grid_template_columns="1fr 1fr 1fr", gap=10)):
            render_data_point("Observed mood", interaction.get("observed_mood", "Not tagged"), tone)
            render_data_point("Functioning", interaction.get("functioning_note", "No note"), "navy")
            render_data_point("Cognition", interaction.get("cognition_note", "No note"), "amber")

        if interaction.get("medication_note") or interaction.get("safety_note") or interaction.get("recommended_follow_up"):
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", flex_direction="column", gap=8)):
                if interaction.get("medication_note"):
                    render_data_point("Medication / adherence", interaction["medication_note"], "teal")
                if interaction.get("safety_note"):
                    render_data_point("Safety note", interaction["safety_note"], "rose")
                if interaction.get("recommended_follow_up"):
                    render_data_point("Recommended follow-up", interaction["recommended_follow_up"], "amber")


def render_message_card(message: dict):
    sender = message.get("sender_role", "patient").title()
    recipient = message.get("recipient", "doctor").replace("_", " ").title()
    tone = {
        "patient": "navy",
        "doctor": "teal",
        "nancy": "amber",
        "system": "rose",
    }.get(message.get("sender_role", "patient"), "navy")

    with me.box(style=card_style("rgba(255, 255, 255, 0.76)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap")):
            me.text(f"{sender} -> {recipient}", style=me.Style(font_weight="700", font_size=16, color=PALETTE[tone]))
            me.text(
                f"{message.get('date_label', '')} {message.get('time_label', '')} | {message.get('channel', 'in-app chat')}",
                style=me.Style(color=PALETTE["muted"], font_size=13),
            )
        me.text(message.get("body", ""), style=me.Style(line_height="1.55", margin=me.Margin(top=10)))


def render_sos_event_card(event: dict):
    with me.box(style=card_style("rgba(255, 247, 246, 0.92)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap")):
            me.text(
                f"SOS {event.get('severity', 'urgent').title()} | {event.get('region', 'Shenzhen')}",
                style=me.Style(font_weight="700", font_size=18, color=PALETTE["rose"]),
            )
            me.text(
                f"{event.get('date_label', '')} {event.get('time_label', '')}",
                style=me.Style(color=PALETTE["muted"], font_size=13),
            )
        if event.get("reason"):
            me.text(event["reason"], style=me.Style(line_height="1.55", margin=me.Margin(top=10)))
        with me.box(style=me.Style(margin=me.Margin(top=14), display="grid", grid_template_columns="1fr 1fr", gap=10)):
            render_data_point("Emergency number", event.get("emergency_number", "120"), "rose")
            render_data_point("Safety / police", event.get("safety_number", "110"), "amber")
            render_data_point("Doctor hospital", event.get("doctor_hospital", "Unassigned"), "navy")
            render_data_point("Doctor department", event.get("doctor_department", "Unassigned"), "teal")
            render_data_point("Recommended hospital", event.get("recommended_hospital", "Unassigned"), "rose")
            render_data_point("Recommended department", event.get("recommended_department", "Emergency / Psychiatry"), "amber")
        if event.get("notes"):
            me.text(event["notes"], style=me.Style(color=PALETTE["muted"], line_height="1.55", margin=me.Margin(top=10)))


def render_async_alert_card(alert: dict, doctor_id: str | None = None):
    severity = alert.get("severity", "watch")
    tone = "rose" if severity == "urgent" else "amber"
    tone_soft = PALETTE[f"{tone}_soft"]
    tone_color = PALETTE[tone]
    status = alert.get("status", "new")

    with me.box(style=card_style("rgba(255, 255, 255, 0.84)")):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap", align_items="flex-start")):
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=6)):
                me.text(alert.get("title", "Async care alert"), style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=21, font_weight="700"))
                me.text(
                    f"{alert.get('patient_name', 'Patient')} | {alert.get('source', 'async care').title()} | {alert.get('date_label', '')} {alert.get('time_label', '')}",
                    style=me.Style(color=PALETTE["muted"], font_size=13),
                )
            with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=12, vertical=8), border_radius=999, background=tone_soft)):
                me.text(f"{severity.title()} / {status.title()}", style=me.Style(color=tone_color, font_weight="700", font_size=12))
        me.text(alert.get("summary", ""), style=me.Style(line_height="1.55", margin=me.Margin(top=12)))
        if alert.get("recommended_follow_up"):
            with me.box(style=me.Style(margin=me.Margin(top=12), padding=me.Padding.all(14), border_radius=16, background=PALETTE["navy_soft"])):
                me.text("Recommended follow-up", style=me.Style(color=PALETTE["navy"], font_weight="700", font_size=13))
                me.text(alert["recommended_follow_up"], style=me.Style(line_height="1.55", margin=me.Margin(top=8)))
        if alert.get("last_action_note"):
            me.text(
                f"Last action note: {alert['last_action_note']}",
                style=me.Style(color=PALETTE["muted"], font_size=13, margin=me.Margin(top=10)),
            )
        if doctor_id and alert.get("patient_id"):
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", gap=10, flex_wrap="wrap")):
                me.button(
                    label="Patient",
                    on_click=navigate_handler("/patient", doctor_id, alert["patient_id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                        background=PALETTE["navy"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Companion",
                    on_click=navigate_handler("/companion", doctor_id, alert["patient_id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                        background=PALETTE["amber"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Nancy",
                    on_click=navigate_handler("/nancy", doctor_id, alert["patient_id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                if status == "new":
                    me.button(
                        label="Acknowledge",
                        on_click=make_alert_status_handler(alert["id"], "acknowledged"),
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=12, vertical=8),
                            background=PALETTE["surface_strong"],
                            color=PALETTE["ink"],
                            border=border_all(PALETTE["line"]),
                            font_weight="700",
                        ),
                    )
                if status != "resolved":
                    me.button(
                        label="Resolve",
                        on_click=make_alert_status_handler(alert["id"], "resolved"),
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=12, vertical=8),
                            background=PALETTE["success"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )


def build_nancy_support_text(patient: dict, doctor: dict | None = None) -> str:
    doctor_name = (doctor or get_doctor(current_doctor_id())).get("name", "your doctor")
    if doctor_name.lower().startswith("dr. "):
        doctor_display_name = doctor_name
    else:
        doctor_display_name = f"Dr. {doctor_name}"
    latest_checkin = (patient.get("daily_checkins") or [None])[0]
    mood = ""
    anxiety = ""
    sleep = ""
    if latest_checkin:
        mood = latest_checkin.get("mood_score", "")
        anxiety = latest_checkin.get("anxiety_score", "")
        sleep = latest_checkin.get("sleep_hours", "")

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


def collect_recent_nancy_interactions(doctor_id: str, patients: list[dict]) -> list[dict]:
    recent_items = []
    for patient in patients:
        patient_id = patient.get("id")
        if not patient_id:
            continue
        for interaction in get_nancy_interactions(doctor_id, patient_id)[:2]:
            recent_items.append(
                {
                    **interaction,
                    "patient_name": patient.get("name", interaction.get("patient_name", "Unknown")),
                }
            )
    return sorted(recent_items, key=lambda item: item.get("created_at", ""), reverse=True)


def make_alert_status_handler(alert_id: str, status: str):
    def handler(e: me.ClickEvent):
        state = me.state(ApplicationState)
        doctor_id = current_doctor_id()
        updated = update_alert_status(doctor_id, alert_id, status)
        if not updated:
            state.status_message = "Alert update failed."
            return
        state.status_message = f"Alert marked {updated.status} for {updated.patient_name}."

    return handler


def summarize_checkin_payload(payload: dict) -> str:
    summary_bits = []
    for label, key, suffix in [
        ("mood", "mood_score", "/10"),
        ("anxiety", "anxiety_score", "/10"),
        ("sleep", "sleep_hours", "h"),
        ("cognition", "cognition_score", "/10"),
        ("memory", "memory_score", "/10"),
    ]:
        value = str(payload.get(key, "")).strip()
        if value:
            summary_bits.append(f"{label} {value}{suffix}")
    return "; ".join(summary_bits) if summary_bits else "Patient completed an async daily report."


def create_nancy_async_touchpoint(
    doctor: dict,
    patient: dict,
    plan: dict,
    patient_report: str,
    mode: str,
) -> dict:
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


def process_async_care_update(
    doctor: dict,
    patient: dict,
    checkin_payload: dict | None = None,
    patient_message: str = "",
    patient_message_recipient: str | None = None,
    patient_message_channel: str = "patient companion chat",
    auto_reply_channel: str = "nancy async companion",
    mode: str = "async companion",
) -> dict:
    if patient_message_recipient:
        return process_patient_message(
            doctor=doctor,
            patient=patient,
            body=patient_message,
            recipient=patient_message_recipient,
        )
    return process_daily_checkin(doctor=doctor, patient=patient, payload_dict=checkin_payload or {})


def build_async_care_summary(patient: dict, doctor: dict) -> dict:
    return summary_to_dict(service_build_async_care_summary(patient, doctor))


def build_patient_timeline(patient: dict) -> list[dict]:
    return [entry.__dict__ for entry in service_build_patient_timeline(patient)]


def render_timeline_entry(entry: dict):
    kind = entry.get("kind")
    payload = entry.get("payload", {})
    if kind == "checkin":
        render_daily_checkin_card(payload)
    elif kind == "nancy":
        render_nancy_interaction_card(payload)
    elif kind == "message":
        render_message_card(payload)
    elif kind == "session":
        render_session_record_card(payload)
    elif kind == "session_note":
        render_session_note_card(payload)
    elif kind == "sos":
        render_sos_event_card(payload)
    elif kind == "alert":
        render_async_alert_card(payload, current_doctor_id())
    elif kind == "outreach":
        render_chat_thread(payload)


def build_patient_context(patient: dict | None, doctor_id: str | None = None) -> str | None:
    if not patient:
        return None

    resolved_doctor = None
    if doctor_id:
        resolved_doctor = get_doctor(doctor_id)
    else:
        try:
            resolved_doctor = get_doctor(current_doctor_id())
        except Exception:
            resolved_doctor = None

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
        f"Supervising hospital: {(resolved_doctor or {}).get('hospital_name', 'Unknown')}",
        f"Supervising department: {(resolved_doctor or {}).get('department_name', 'Unknown')}",
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


def build_nancy_doctor_context(doctor: dict, patient: dict) -> str:
    goals = patient.get("goals", [])[:3]
    alerts = patient.get("alerts", [])[:3]
    tasks = patient.get("nancy_tasks", [])[:4]
    task_lines = [
        f"{task.get('title', 'Directive')}: {task.get('instructions', '')}"
        for task in tasks
    ]

    context_lines = [
        f"Supervising clinician: {doctor.get('name', 'Unknown')}",
        f"Specialty: {doctor.get('specialty', 'Unknown')}",
        f"Patient risk level: {patient.get('risk', 'Unknown')}",
        f"Care plan: {patient.get('care_plan', 'None')}",
        "Nancy must stay inside supportive monitoring, reminders, and doctor handoff.",
    ]
    if goals:
        context_lines.append("Top goals: " + " | ".join(goals))
    if alerts:
        context_lines.append("Known alerts: " + " | ".join(alerts))
    if task_lines:
        context_lines.append("Doctor-approved reminders: " + " | ".join(task_lines))
    return "\n".join(context_lines)


def build_nancy_reminders(patient: dict) -> list[str]:
    reminders = []
    for task in patient.get("nancy_tasks", [])[:5]:
        title = task.get("title", "").strip()
        instructions = task.get("instructions", "").strip()
        if title and instructions:
            reminders.append(f"{title}: {instructions}")
        elif instructions:
            reminders.append(instructions)
    for goal in patient.get("goals", [])[:2]:
        reminders.append(goal)
    return reminders[:6]


def build_nancy_settings_preview(doctor: dict, patient: dict) -> str:
    return build_nancy_settings_json(
        patient_context=build_patient_context(patient, doctor["id"]) or "",
        doctor_context=build_nancy_doctor_context(doctor, patient),
        reminders=build_nancy_reminders(patient),
    )


def update_intake_name(e):
    me.state(ApplicationState).intake_name = e.value


def update_intake_concern(e):
    me.state(ApplicationState).intake_concern = e.value


def update_intake_context(e):
    me.state(ApplicationState).intake_context = e.value


def update_outreach_target(e):
    me.state(ApplicationState).outreach_target = e.value


def update_outreach_message(e):
    me.state(ApplicationState).outreach_message = e.value


def _set_state_value(field_name: str):
    def handler(e):
        setattr(me.state(ApplicationState), field_name, e.value)

    return handler


update_checkin_mood_score = _set_state_value("checkin_mood_score")
update_checkin_anxiety_score = _set_state_value("checkin_anxiety_score")
update_checkin_sleep_hours = _set_state_value("checkin_sleep_hours")
update_checkin_energy_score = _set_state_value("checkin_energy_score")
update_checkin_stress_score = _set_state_value("checkin_stress_score")
update_checkin_cognition_score = _set_state_value("checkin_cognition_score")
update_checkin_memory_score = _set_state_value("checkin_memory_score")
update_checkin_functioning_score = _set_state_value("checkin_functioning_score")
update_checkin_medication_adherence = _set_state_value("checkin_medication_adherence")
update_checkin_side_effects = _set_state_value("checkin_side_effects")
update_checkin_daily_update = _set_state_value("checkin_daily_update")
update_checkin_safety_concerns = _set_state_value("checkin_safety_concerns")
update_checkin_significant_events = _set_state_value("checkin_significant_events")
update_checkin_clinical_summary = _set_state_value("checkin_clinical_summary")
update_nancy_task_title = _set_state_value("nancy_task_title")
update_nancy_task_note = _set_state_value("nancy_task_note")
update_nancy_task_due = _set_state_value("nancy_task_due")
update_nancy_task_category = _set_state_value("nancy_task_category")
update_nancy_conversation_goal = _set_state_value("nancy_conversation_goal")
update_nancy_patient_message = _set_state_value("nancy_patient_message")
update_nancy_clinician_summary = _set_state_value("nancy_clinician_summary")
update_nancy_observed_mood = _set_state_value("nancy_observed_mood")
update_nancy_functioning_note = _set_state_value("nancy_functioning_note")
update_nancy_cognition_note = _set_state_value("nancy_cognition_note")
update_nancy_medication_note = _set_state_value("nancy_medication_note")
update_nancy_safety_note = _set_state_value("nancy_safety_note")
update_nancy_recommended_follow_up = _set_state_value("nancy_recommended_follow_up")
update_nancy_escalation_level = _set_state_value("nancy_escalation_level")
update_nancy_contact_mode = _set_state_value("nancy_contact_mode")
update_chat_message_body = _set_state_value("chat_message_body")
update_chat_message_sender = _set_state_value("chat_message_sender")
update_sos_reason = _set_state_value("sos_reason")
update_sos_severity = _set_state_value("sos_severity")
update_sos_location_district = _set_state_value("sos_location_district")
update_sos_notes = _set_state_value("sos_notes")
update_session_note_title = _set_state_value("session_note_title")
update_session_note_body = _set_state_value("session_note_body")
update_session_plan_update = _set_state_value("session_plan_update")
update_session_disposition = _set_state_value("session_disposition")


def stage_intake(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    if not state.intake_name.strip() or not state.intake_concern.strip():
        state.status_message = "Patient name and presenting concern are required before intake can be staged."
        return

    patient_record = add_patient_record(
        doctor_id=doctor_id,
        name=state.intake_name,
        concern=state.intake_concern,
        context=state.intake_context,
    )
    state.status_message = f"Intake packet created for {patient_record['name']}. Opening patient portfolio."
    state.intake_name = ""
    state.intake_concern = ""
    state.intake_context = ""
    me.navigate("/patient", query_params={"doctor": doctor_id, "patient": patient_record["id"]})


def queue_outreach(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    patient = get_patient(doctor_id, current_patient_id()) if current_patient_id() else None
    target = state.outreach_target.strip() or (patient["name"] if patient else "")
    message = state.outreach_message.strip()
    if not target or not message:
        state.status_message = "Add both a target and a message before queueing a check-in."
        return

    add_outreach_log(
        doctor_id=doctor_id,
        target=target,
        message=message,
        patient_id=patient["id"] if patient else None,
    )
    state.status_message = f"Check-in note queued for {target}. Care team can send after final review."
    state.outreach_target = ""
    state.outreach_message = ""


def submit_daily_checkin(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    patient_id = current_patient_id()
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

    if not patient:
        state.status_message = "Open a patient profile before submitting a daily clinical questionnaire."
        return

    payload = {
        "mood_score": state.checkin_mood_score,
        "anxiety_score": state.checkin_anxiety_score,
        "sleep_hours": state.checkin_sleep_hours,
        "energy_score": state.checkin_energy_score,
        "stress_score": state.checkin_stress_score,
        "cognition_score": state.checkin_cognition_score,
        "memory_score": state.checkin_memory_score,
        "functioning_score": state.checkin_functioning_score,
        "medication_adherence": state.checkin_medication_adherence,
        "side_effects": state.checkin_side_effects,
        "daily_update": state.checkin_daily_update,
        "safety_concerns": state.checkin_safety_concerns,
        "significant_events": state.checkin_significant_events,
        "clinical_summary": state.checkin_clinical_summary,
    }

    async_result = process_daily_checkin(
        doctor=doctor,
        patient=patient,
        payload_dict=payload,
    )
    checkin = async_result["checkin"]
    level = async_result["plan"].get("escalation_level", "routine").title()
    alert = async_result.get("alert")
    state.status_message = (
        f"Daily clinical report saved for {patient['name']} at {checkin['time_label']}. "
        + (f"Nancy generated a {level} clinician handoff and opened a doctor review alert." if alert else f"Nancy generated a {level} clinician handoff.")
    )
    state.checkin_side_effects = ""
    state.checkin_daily_update = ""
    state.checkin_safety_concerns = ""
    state.checkin_significant_events = ""
    state.checkin_clinical_summary = ""


def add_nancy_directive(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    patient_id = current_patient_id()
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

    if not patient:
        state.status_message = "Open a patient before assigning Nancy directives."
        return
    if not state.nancy_task_title.strip() or not state.nancy_task_note.strip():
        state.status_message = "Nancy needs both a directive title and a guidance note."
        return

    task = add_nancy_task(
        doctor_id=doctor_id,
        patient_id=patient["id"],
        patient_name=patient["name"],
        title=state.nancy_task_title,
        instructions=state.nancy_task_note,
        category=state.nancy_task_category,
        due_label=state.nancy_task_due,
    )
    state.status_message = f"Nancy directive added for {patient['name']}: {task['title']}."
    state.nancy_task_title = ""
    state.nancy_task_note = ""
    state.nancy_task_due = ""


def log_nancy_touchpoint(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    patient_id = current_patient_id()
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

    if not patient:
        state.status_message = "Open a patient before logging a Nancy touchpoint."
        return
    if not state.nancy_patient_message.strip() and not state.nancy_clinician_summary.strip():
        state.status_message = "Add a patient update or a clinician summary before saving the Nancy touchpoint."
        return

    interaction = add_nancy_interaction(
        doctor_id=doctor_id,
        patient_id=patient["id"],
        patient_name=patient["name"],
        payload={
            "mode": state.nancy_contact_mode,
            "conversation_goal": state.nancy_conversation_goal,
            "patient_report": state.nancy_patient_message,
            "clinician_summary": state.nancy_clinician_summary,
            "observed_mood": state.nancy_observed_mood,
            "functioning_note": state.nancy_functioning_note,
            "cognition_note": state.nancy_cognition_note,
            "medication_note": state.nancy_medication_note,
            "safety_note": state.nancy_safety_note,
            "recommended_follow_up": state.nancy_recommended_follow_up,
            "escalation_level": state.nancy_escalation_level,
        },
    )
    state.status_message = (
        f"Nancy handoff saved for {patient['name']} at {interaction['time_label']}. "
        f"Escalation: {interaction['escalation_level'].title()}."
    )
    state.nancy_conversation_goal = ""
    state.nancy_patient_message = ""
    state.nancy_clinician_summary = ""
    state.nancy_observed_mood = ""
    state.nancy_functioning_note = ""
    state.nancy_cognition_note = ""
    state.nancy_medication_note = ""
    state.nancy_safety_note = ""
    state.nancy_recommended_follow_up = ""
    state.nancy_escalation_level = "routine"


def _send_patient_message(recipient: str):
    def handler(e: me.ClickEvent):
        state = me.state(ApplicationState)
        doctor_id = current_doctor_id()
        doctor = get_doctor(doctor_id)
        patient_id = current_patient_id()
        patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

        if not patient:
            state.status_message = "Open a patient before sending a message."
            return
        if not state.chat_message_body.strip():
            state.status_message = "Add a message before sending it."
            return

        message_body = state.chat_message_body.strip()
        sender_role = (state.chat_message_sender or "patient").strip().lower()

        if sender_role != "patient":
            message = add_patient_message(
                doctor_id=doctor_id,
                patient_id=patient["id"],
                patient_name=patient["name"],
                sender_role=sender_role,
                recipient=recipient,
                body=message_body,
            )
            state.status_message = (
                f"Message sent from {message['sender_role']} to {message['recipient']} for {patient['name']}."
            )
        else:
            if recipient == "doctor":
                result = process_patient_message(
                    doctor=doctor,
                    patient=patient,
                    body=message_body,
                    recipient="doctor",
                )
                message = result["message"]
                state.status_message = (
                    f"Patient message sent to {message['recipient']} for {patient['name']}."
                    + (" Doctor review alert opened." if result.get("alert") else "")
                )
            else:
                async_result = process_patient_message(
                    doctor=doctor,
                    patient=patient,
                    body=message_body,
                    recipient=recipient,
                )
                state.status_message = (
                    f"Patient message logged for {patient['name']}. "
                    f"Nancy replied and created a {async_result['plan'].get('escalation_level', 'routine')} handoff."
                    + (" Doctor review alert opened." if async_result.get("alert") else "")
                )
        state.chat_message_body = ""

    return handler


send_message_to_nancy = _send_patient_message("nancy")
send_message_to_doctor = _send_patient_message("doctor")
send_message_to_both = _send_patient_message("both")


def send_nancy_support_ping(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    patient_id = current_patient_id()
    doctor = get_doctor(doctor_id)
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

    if not patient:
        state.status_message = "Open a patient before sending a Nancy support text."
        return

    support_message = build_nancy_support_text(patient, doctor)
    process_nancy_proactive_ping(doctor, patient, support_message)
    state.status_message = f"Nancy supportive text queued for {patient['name']}."


def trigger_sos_escalation(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    patient_id = current_patient_id()
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

    if not patient:
        state.status_message = "Open a patient before triggering SOS escalation."
        return
    if not state.sos_reason.strip():
        state.status_message = "Document the SOS reason before escalating."
        return

    district = state.sos_location_district.strip() or doctor.get("hospital_district", "")
    recommended = choose_emergency_hospital(
        district=district,
        preferred_hospital=doctor.get("hospital_name", ""),
    )
    event = add_sos_event(
        doctor_id=doctor_id,
        patient_id=patient["id"],
        patient_name=patient["name"],
        payload={
            "region": "Shenzhen, Guangdong, China",
            "district": district,
            "severity": state.sos_severity,
            "reason": state.sos_reason,
            "rule": "shenzhen-psychiatric-escalation-v1",
            "doctor_hospital": doctor.get("hospital_name", ""),
            "doctor_department": doctor.get("department_name", ""),
            "recommended_hospital": recommended.get("name", doctor.get("hospital_name", "")),
            "recommended_department": doctor.get("department_name", "Emergency / Psychiatry"),
            "emergency_number": "120",
            "safety_number": "110",
            "notes": state.sos_notes,
            "status": "Escalated to hospital + doctor",
        },
    )
    add_async_alert(
        doctor_id=doctor_id,
        patient_id=patient["id"],
        patient_name=patient["name"],
        severity="urgent",
        source="sos escalation",
        title="Urgent SOS escalation opened",
        summary=(
            f"SOS escalation for {patient['name']}: {state.sos_reason.strip()}. "
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
        patient_id=patient["id"],
        patient_name=patient["name"],
        sender_role="system",
        recipient="doctor",
        body=(
            f"SOS triggered for {patient['name']}. Severity: {event['severity']}. "
            f"Route: 120, {event['recommended_hospital']}, {event['recommended_department']}."
        ),
        channel="sos escalation",
    )
    state.status_message = (
        f"SOS route prepared for {patient['name']}: call 120 and notify {event['recommended_hospital']}."
    )
    state.sos_reason = ""
    state.sos_notes = ""


def save_session_review_note(e: me.ClickEvent):
    state = me.state(ApplicationState)
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    patient_id = current_patient_id()
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None

    if not patient:
        state.status_message = "Open a patient context before saving a clinician session note."
        return
    if not state.session_note_body.strip():
        state.status_message = "Write a clinician session note before saving it."
        return

    note = save_clinician_session_note(
        doctor=doctor,
        patient=patient,
        title=state.session_note_title,
        note=state.session_note_body,
        plan_update=state.session_plan_update,
        disposition=state.session_disposition,
    )
    if not note:
        state.status_message = "Session note save failed."
        return

    state.status_message = f"Clinician session note saved for {patient['name']}."
    state.session_note_title = ""
    state.session_note_body = ""
    state.session_plan_update = ""


@me.page(path="/", stylesheets=APP_STYLESHEETS, on_load=on_load)
def landing_page():
    total_patients = sum(len(get_patients_for_doctor(doctor_id)) for doctor_id in DOCTORS)
    with page_shell("/"):
        render_hero(
            "Platform Vision",
            "A clinical operating system for psychiatrists, researchers, and care organizations.",
            "MindScape now becomes the front door to a multi-doctor mental-health platform: a place where patient continuity, live diagnostic support, research collaboration, and fast-moving evidence live in one coherent product.",
            "Explore Organizations",
            "/organizations",
            secondary_label="Enter Doctor Workspace",
            secondary_path="/workspace",
            secondary_doctor_id=DEFAULT_DOCTOR_ID,
        )

        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=18)):
            render_stat_card("Organizations", str(len(ORGANIZATIONS)), "navy")
            render_stat_card("Doctor Portfolios", str(len(DOCTORS)), "teal")
            render_stat_card("Patient Panels", str(total_patients), "amber")
            render_stat_card("Research Streams", "Weekly agent-curated", "rose")

        render_section_header(
            "Why This Product Shape Matters",
            "The current single diagnostic workflow is strong, but the harder problem is clinical continuity. This platform expands MindScape into a system that helps doctors act on insight, not just receive it.",
        )
        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(280px, 1fr))", gap=20)):
            for pillar in LANDING_PILLARS:
                with me.box(style=card_style()):
                    me.text(pillar["title"], style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    me.text(pillar["body"], style=me.Style(color=PALETTE["muted"], font_size=15, line_height="1.7", margin=me.Margin(top=14)))

        render_section_header(
            "Operating Model",
            "The platform is designed like elite healthcare software: every page exists to shorten the distance between detection, judgment, intervention, and learning.",
        )
        with me.box(style=me.Style(display="grid", grid_template_columns="1.2fr 1fr", gap=20, align_items="stretch")):
            with me.box(style=card_style()):
                render_bullet_list(PLATFORM_STEPS, tone="navy")
            with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(22,56,77,0.06) 100%)")):
                me.text("What Changes Immediately", style=me.Style(font_weight="700", font_size=20))
                render_bullet_list(
                    [
                        "Organizations become discoverable entities with their own care and research identity.",
                        "Doctors gain personal portfolios that act as the true entrance to their operating environment.",
                        "The current live diagnosis flow remains intact inside a dedicated session workspace.",
                        "Patient history, appointments, logs, and peer learning are positioned around the core engine.",
                    ]
                )


@me.page(path="/organizations", stylesheets=APP_STYLESHEETS, on_load=on_load)
def organizations_page():
    with page_shell("/organizations"):
        render_hero(
            "Network Portfolio",
            "Organizations and research leaders mapped into one coordinated mental-health network.",
            "This is the portfolio layer: institutions, their operating focus, and the doctors whose profiles lead into the main clinical workspace.",
            "Open Featured Doctor",
            "/doctor",
            primary_doctor_id=DEFAULT_DOCTOR_ID,
            secondary_label="Go To Workspace",
            secondary_path="/workspace",
            secondary_doctor_id=DEFAULT_DOCTOR_ID,
        )

        for organization in ORGANIZATIONS:
            with me.box(style=card_style()):
                with me.box(style=me.Style(display="flex", justify_content="space-between", gap=20, flex_wrap="wrap", align_items="flex-start")):
                    with me.box(style=me.Style(max_width="760px", display="flex", flex_direction="column", gap=10)):
                        me.text(organization["name"], style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=30, font_weight="700"))
                        me.text(organization["region"], style=me.Style(color=PALETTE["teal"], font_size=14, font_weight="700"))
                        me.text(organization["focus"], style=me.Style(font_size=15, color=PALETTE["muted"], line_height="1.65"))
                        me.text(
                            f"Signature program: {organization['signature_program']}",
                            style=me.Style(font_size=14, color=PALETTE["ink"], font_weight="600"),
                        )
                    with me.box(style=me.Style(display="grid", grid_template_columns="1fr", gap=10, min_width="240px")):
                        for label, value in organization["stats"].items():
                            with me.box(
                                style=me.Style(
                                    padding=me.Padding.all(14),
                                    border_radius=18,
                                    background=PALETTE["navy_soft"],
                                )
                            ):
                                me.text(label.title(), style=me.Style(color=PALETTE["muted"], font_size=12, text_transform="uppercase", font_weight="700"))
                                me.text(value, style=me.Style(font_weight="700", margin=me.Margin(top=6)))

                with me.box(style=me.Style(margin=me.Margin(top=22), display="grid", grid_template_columns="repeat(auto-fit, minmax(270px, 1fr))", gap=18)):
                    for doctor in get_doctors_for_organization(organization["id"]):
                        with me.box(style=card_style("rgba(255, 255, 255, 0.72)")):
                            me.text(doctor["name"], style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                            me.text(doctor["title"], style=me.Style(color=PALETTE["muted"], font_size=14, line_height="1.5", margin=me.Margin(top=8)))
                            me.text(doctor["specialty"], style=me.Style(color=PALETTE["teal"], font_size=13, font_weight="700", margin=me.Margin(top=10)))
                            me.text(doctor["tagline"], style=me.Style(font_size=14, color=PALETTE["ink"], line_height="1.6", margin=me.Margin(top=12)))
                            with me.box(style=me.Style(display="flex", gap=10, flex_wrap="wrap", margin=me.Margin(top=18))):
                                me.button(
                                    label="View Portfolio",
                                    on_click=navigate_handler("/doctor", doctor["id"]),
                                    style=me.Style(
                                        border_radius=999,
                                        padding=me.Padding.symmetric(horizontal=16, vertical=10),
                                        background=PALETTE["navy"],
                                        color="white",
                                        border=border_none(),
                                        font_weight="700",
                                    ),
                                )
                                me.button(
                                    label="Open Workspace",
                                    on_click=navigate_handler("/workspace", doctor["id"]),
                                    style=me.Style(
                                        border_radius=999,
                                        padding=me.Padding.symmetric(horizontal=16, vertical=10),
                                        background=PALETTE["surface_strong"],
                                        color=PALETTE["ink"],
                                        border=border_all(PALETTE["line"]),
                                        font_weight="700",
                                    ),
                                )


@me.page(path="/doctor", stylesheets=APP_STYLESHEETS, on_load=on_load)
def doctor_profile_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    organization = get_organization(doctor["organization_id"])
    panel_patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    panel_size = len(panel_patients)

    with page_shell("/doctor", doctor_id):
        render_hero(
            "Doctor Portfolio",
            f"{doctor['name']} leads a care-and-research practice built for modern psychiatry.",
            doctor["mission"],
            "Enter Doctor Workspace",
            "/workspace",
            primary_doctor_id=doctor_id,
            secondary_label="Start Session",
            secondary_path="/session",
            secondary_doctor_id=doctor_id,
        )

        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=18)):
            render_stat_card("Clinical Panel", f"{panel_size} tracked patients", "navy")
            render_stat_card("Active Watchlist", doctor["stats"]["watchlist"], "rose")
            render_stat_card("Research Load", doctor["stats"]["research"], "teal")
            render_stat_card("Follow-up Reliability", doctor["stats"]["followup"], "amber")

        with me.box(style=me.Style(display="grid", grid_template_columns="1.15fr 0.85fr", gap=20)):
            with me.box(style=card_style()):
                render_section_header("Clinical Identity", "The doctor's portfolio becomes the main entrance to the platform and orients the user before they dive into operations.")
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    me.text(f"Organization: {organization['name']}", style=me.Style(font_weight="700"))
                    me.text(f"Hospital: {doctor.get('hospital_name', 'Unassigned')}", style=me.Style(color=PALETTE["muted"]))
                    me.text(f"Department: {doctor.get('department_name', 'Unassigned')}", style=me.Style(color=PALETTE["muted"]))
                    me.text(f"Location: {doctor['location']}", style=me.Style(color=PALETTE["muted"]))
                    me.text(f"Experience: {doctor['years']}", style=me.Style(color=PALETTE["muted"]))
                    me.text(f"Specialty: {doctor['specialty']}", style=me.Style(color=PALETTE["ink"], font_weight="600"))
                    me.text(doctor["tagline"], style=me.Style(line_height="1.7"))

            with me.box(style=card_style("linear-gradient(180deg, rgba(198,106,26,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Focus Areas", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18))):
                    render_bullet_list(doctor["focus_areas"])

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style()):
                me.text("Patient Portfolio Snapshot", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                if panel_patients:
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                        for patient in panel_patients[:2]:
                            with me.box(
                                style=me.Style(
                                    padding=me.Padding.all(16),
                                    border_radius=18,
                                    background=PALETTE["teal_soft"],
                                )
                            ):
                                me.text(patient["name"], style=me.Style(font_weight="700"))
                                me.text(patient["diagnosis"], style=me.Style(color=PALETTE["muted"], font_size=14, margin=me.Margin(top=4)))
                                me.text(patient["last_update"], style=me.Style(font_size=14, line_height="1.55", margin=me.Margin(top=8)))
                                with me.box(style=me.Style(display="flex", gap=10, flex_wrap="wrap", margin=me.Margin(top=12))):
                                    me.button(
                                        label="Open Patient",
                                        on_click=navigate_handler("/patient", doctor_id, patient["id"]),
                                        style=me.Style(
                                            border_radius=999,
                                            padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                            background=PALETTE["navy"],
                                            color="white",
                                            border=border_none(),
                                            font_weight="700",
                                        ),
                                    )
                                    me.button(
                                        label="Start Session",
                                        on_click=navigate_handler("/session", doctor_id, patient["id"]),
                                        style=me.Style(
                                            border_radius=999,
                                            padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                            background=PALETTE["surface_strong"],
                                            color=PALETTE["ink"],
                                            border=border_all(PALETTE["line"]),
                                            font_weight="700",
                                        ),
                                    )
                                    me.button(
                                        label="Companion",
                                        on_click=navigate_handler("/companion", doctor_id, patient["id"]),
                                        style=me.Style(
                                            border_radius=999,
                                            padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                            background=PALETTE["amber"],
                                            color="white",
                                            border=border_none(),
                                            font_weight="700",
                                        ),
                                    )
                else:
                    me.text("Structured patient cards can be attached here as this doctor profile is expanded.", style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=18)))

            with me.box(style=card_style()):
                me.text("Research + Performance", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for item in doctor["performance"]:
                        with me.box(
                            style=me.Style(
                                padding=me.Padding.all(16),
                                border_radius=18,
                                background=PALETTE["navy_soft"],
                            )
                        ):
                            me.text(item["label"], style=me.Style(color=PALETTE["muted"], font_size=13, text_transform="uppercase", font_weight="700"))
                            me.text(item["value"], style=me.Style(font_weight="700", margin=me.Margin(top=8)))


@me.page(path="/patient", stylesheets=APP_STYLESHEETS, on_load=on_load)
def patient_profile_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    state = me.state(ApplicationState)
    patient = merge_patient_runtime_data(get_patient(doctor_id, current_patient_id()) or get_patient(doctor_id), doctor_id)
    async_summary = build_async_care_summary(patient, doctor) if patient else {}
    async_timeline = build_patient_timeline(patient) if patient else []
    async_alerts = patient.get("async_alerts", []) if patient else []
    outreach_logs = patient.get("outreach_logs", []) if patient else []
    daily_checkins = patient.get("daily_checkins", []) if patient else []
    session_records = patient.get("session_records", []) if patient else []
    nancy_tasks = patient.get("nancy_tasks", []) if patient else []
    nancy_interactions = patient.get("nancy_interactions", []) if patient else []
    patient_messages = patient.get("patient_messages", []) if patient else []
    sos_events = patient.get("sos_events", []) if patient else []
    latest_checkin = daily_checkins[0] if daily_checkins else None
    support_recommended = False
    if latest_checkin:
        try:
            mood_value = float(latest_checkin.get("mood_score", 5) or 5)
            anxiety_value = float(latest_checkin.get("anxiety_score", 5) or 5)
            support_recommended = mood_value <= 3 or anxiety_value >= 8 or bool(latest_checkin.get("safety_concerns", "").strip())
        except (TypeError, ValueError):
            support_recommended = False

    with page_shell("/patient", doctor_id):
        if not patient:
            render_hero(
                "Patient Profile",
                "No patient is currently selected for this doctor.",
                "Choose a patient from the doctor workspace to open history, reports, logs, and session context.",
                "Back To Workspace",
                "/workspace",
                primary_doctor_id=doctor_id,
            )
            render_patient_route_hub(
                doctor,
                [merge_patient_runtime_data(entry, doctor_id) for entry in get_patients_for_doctor(doctor_id)],
                "Open A Patient Record",
                "Patient routes no longer depend on a hidden active context. Pick any patient here and the route will open directly into the correct chart.",
                "/patient",
                "Open Record",
                tone="teal",
            )
            return

        render_hero(
            "Patient Portfolio",
            f"{patient['name']} is now a first-class clinical record inside the platform.",
            "This patient view pulls together health history, diagnosis reports, personal logs, care-team notes, next appointment planning, and direct entry into a live diagnostic session.",
            "Start Session",
            "/session",
            primary_doctor_id=doctor_id,
            primary_patient_id=patient["id"],
            secondary_label="Back To Workspace",
            secondary_path="/workspace",
            secondary_doctor_id=doctor_id,
        )

        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=18)):
            render_stat_card("Patient", patient["name"], "navy")
            render_stat_card("Risk", patient["risk"], "rose" if patient["risk"] == "High" else "amber")
            render_stat_card("Status", patient["status"], "teal")
            render_stat_card("Next Appointment", patient["next_appointment"], "amber")
            render_stat_card("Daily Reports", str(len(daily_checkins)), "navy")
            render_stat_card("Session Records", str(len(session_records)), "teal")

        with me.box(style=me.Style(display="flex", gap=12, flex_wrap="wrap")):
            me.button(
                label="Run Focused Session",
                on_click=navigate_handler("/session", doctor_id, patient["id"]),
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=18, vertical=12),
                    background=PALETTE["navy"],
                    color="white",
                    border=border_none(),
                    font_weight="700",
                ),
            )
            me.button(
                label="Open Nancy Console",
                on_click=navigate_handler("/nancy", doctor_id, patient["id"]),
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=18, vertical=12),
                    background=PALETTE["teal"],
                    color="white",
                    border=border_none(),
                    font_weight="700",
                ),
            )
            me.button(
                label="Open Companion Portal",
                on_click=navigate_handler("/companion", doctor_id, patient["id"]),
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=18, vertical=12),
                    background=PALETTE["amber"],
                    color="white",
                    border=border_none(),
                    font_weight="700",
                ),
            )
            me.button(
                label="Return To Doctor Workspace",
                on_click=navigate_handler("/workspace", doctor_id),
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=18, vertical=12),
                    background=PALETTE["surface_strong"],
                    color=PALETTE["ink"],
                    border=border_all(PALETTE["line"]),
                    font_weight="700",
                ),
            )

        with me.box(style=card_style("linear-gradient(180deg, rgba(22,56,77,0.06) 0%, rgba(255,255,255,0.88) 100%)")):
            me.text("Pre-Session Doctor Brief", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "This briefing is what the doctor sees before every session: the patient profile, recent history, ongoing medical issues, off-consultation updates, and the latest clinical signal from questionnaires and past sessions.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=12)):
                render_data_point("Medical record summary", patient.get("history", ""), "navy")
                render_data_point("Latest update", patient.get("last_update", ""), "amber")
                render_data_point("Last consultation insight", patient.get("last_session_summary", "No prior recorded session yet."), "teal")
                render_data_point("Off-consultation weekly update", patient.get("last_daily_checkin_summary", "No daily report submitted yet."), "rose")
                render_data_point("Latest Nancy handoff", patient.get("last_nancy_summary", "No Nancy interaction logged yet."), "navy")

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style()):
                me.text("Clinical Overview", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=12)):
                    render_data_point("Age", str(patient.get("age", "Unknown")), "navy")
                    render_data_point("Primary diagnosis", patient["diagnosis"], "teal")
                    render_data_point("Assigned doctor", doctor["name"], "amber")
                    render_data_point("Care plan", patient["care_plan"], "rose")
                    render_data_point("Hospital", doctor.get("hospital_name", "Unassigned"), "navy")
                    render_data_point("Department", doctor.get("department_name", "Unassigned"), "teal")

            with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Goals + Alerts", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=16)):
                    with me.box(style=me.Style(display="flex", flex_direction="column", gap=10)):
                        me.text("Goals", style=me.Style(font_weight="700"))
                        render_bullet_list(patient.get("goals", []))
                    with me.box(style=me.Style(display="flex", flex_direction="column", gap=10)):
                        me.text("Alerts", style=me.Style(font_weight="700"))
                        render_bullet_list(patient.get("alerts", []), tone="navy")

        with me.box(style=card_style()):
            me.text("Health History", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            with me.box(style=me.Style(margin=me.Margin(top=18))):
                render_bullet_list(patient.get("health_history", []), tone="navy")

        with me.box(style=card_style()):
            me.text("Daily Clinical Questionnaire", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "This is the off-consultation daily report. It captures mood, anxiety, sleep, cognition, memory, functioning, medication adherence, safety, and significant events in a clinically useful format for the doctor.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(180px, 1fr))", gap=12)):
                me.input(label="Mood 0-10", value=state.checkin_mood_score, on_input=update_checkin_mood_score, appearance="outline", type="number")
                me.input(label="Anxiety 0-10", value=state.checkin_anxiety_score, on_input=update_checkin_anxiety_score, appearance="outline", type="number")
                me.input(label="Sleep hours", value=state.checkin_sleep_hours, on_input=update_checkin_sleep_hours, appearance="outline", type="number")
                me.input(label="Energy 0-10", value=state.checkin_energy_score, on_input=update_checkin_energy_score, appearance="outline", type="number")
                me.input(label="Stress 0-10", value=state.checkin_stress_score, on_input=update_checkin_stress_score, appearance="outline", type="number")
                me.input(label="Cognition 0-10", value=state.checkin_cognition_score, on_input=update_checkin_cognition_score, appearance="outline", type="number")
                me.input(label="Memory / recall 0-10", value=state.checkin_memory_score, on_input=update_checkin_memory_score, appearance="outline", type="number")
                me.input(label="Functioning 0-10", value=state.checkin_functioning_score, on_input=update_checkin_functioning_score, appearance="outline", type="number")
                me.input(label="Medication adherence %", value=state.checkin_medication_adherence, on_input=update_checkin_medication_adherence, appearance="outline", type="number")
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", flex_direction="column", gap=12)):
                me.textarea(label="Daily clinical update", value=state.checkin_daily_update, on_input=update_checkin_daily_update, rows=3, appearance="outline")
                me.textarea(label="Intermediary cognitive / recall observations", value=state.checkin_clinical_summary, on_input=update_checkin_clinical_summary, rows=3, appearance="outline")
                me.textarea(label="Significant events, triggers, or interpersonal disclosures", value=state.checkin_significant_events, on_input=update_checkin_significant_events, rows=3, appearance="outline")
                me.textarea(label="Side effects / medication issues", value=state.checkin_side_effects, on_input=update_checkin_side_effects, rows=2, appearance="outline")
                me.textarea(label="Safety concerns", value=state.checkin_safety_concerns, on_input=update_checkin_safety_concerns, rows=2, appearance="outline")
                me.button(
                    label="Save Daily Report",
                    on_click=submit_daily_checkin,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )

        with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
            me.text("Nancy Between-Session Companion", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "Nancy is the doctor-supervised relationship layer between visits: a conversational care companion that checks in naturally, follows doctor-approved reminders, and writes useful handoffs back into the chart.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr 1fr", gap=12)):
                render_data_point("Active Nancy directives", str(len(nancy_tasks)), "teal")
                render_data_point("Nancy handoffs", str(len(nancy_interactions)), "navy")
                render_data_point("Latest Nancy summary", patient.get("last_nancy_summary", "No logged companion note yet."), "amber")
            if support_recommended:
                with me.box(style=me.Style(margin=me.Margin(top=16), padding=me.Padding.all(14), border_radius=16, background=PALETTE["amber_soft"])):
                    me.text(
                        "Nancy proactive support recommended from recent patient state.",
                        style=me.Style(color=PALETTE["amber"], font_weight="700"),
                    )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", gap=12, flex_wrap="wrap")):
                me.button(
                    label="Open Nancy Console",
                    on_click=navigate_handler("/nancy", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Start Session With Nancy Context",
                    on_click=navigate_handler("/session", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["surface_strong"],
                        color=PALETTE["ink"],
                        border=border_all(PALETTE["line"]),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Send Nancy Support Text",
                    on_click=send_nancy_support_ping,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["amber"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )

        with me.box(style=card_style("linear-gradient(180deg, rgba(198,106,26,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
            me.text("Unified Async Care Loop", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "This is the real between-session operating view: questionnaire updates, patient messages, Nancy responses, doctor alerts, and escalation signals all resolve into one timeline.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=12)):
                render_data_point("Risk lane", async_summary.get("risk_level", "routine").title(), "rose" if async_summary.get("risk_level") == "urgent" else ("amber" if async_summary.get("risk_level") == "watch" else "teal"))
                render_data_point("Doctor reviews needed", str(async_summary.get("review_needed", 0)), "navy")
                render_data_point("Messages in loop", str(async_summary.get("message_count", 0)), "amber")
                render_data_point("Active Nancy directives", str(async_summary.get("task_count", 0)), "teal")
                render_data_point("Open alerts", str(async_summary.get("open_alerts", 0)), "rose")
            if async_summary.get("risk_reasons"):
                with me.box(style=me.Style(margin=me.Margin(top=16))):
                    render_bullet_list(async_summary["risk_reasons"], tone="navy")
            with me.box(style=me.Style(margin=me.Margin(top=16), display="flex", gap=12, flex_wrap="wrap")):
                me.button(
                    label="Open Companion Portal",
                    on_click=navigate_handler("/companion", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["amber"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Open Nancy Console",
                    on_click=navigate_handler("/nancy", doctor_id, patient["id"]),
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )

        if async_timeline:
            with me.box(style=card_style()):
                me.text("Async Care Timeline", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Doctors no longer have to mentally stitch the week back together. The full between-session story is ordered here as one timeline.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    for entry in async_timeline[:10]:
                        render_timeline_entry(entry)

        if async_alerts:
            with me.box(style=card_style()):
                me.text("Outstanding Async Care Alerts", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "These are the explicit doctor-review signals generated from between-session activity.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    for alert in async_alerts[:6]:
                        render_async_alert_card(alert, doctor_id)

        if daily_checkins:
            with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
                render_metric_trend("Mood trend", daily_checkins, "mood_score", "navy")
                render_metric_trend("Anxiety trend", daily_checkins, "anxiety_score", "rose")
            with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
                render_metric_trend("Cognition trend", daily_checkins, "cognition_score", "teal")
                render_metric_trend("Memory / recall trend", daily_checkins, "memory_score", "amber")
            with me.box(style=card_style()):
                me.text("Recent Daily Reports", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for checkin in daily_checkins[:5]:
                        render_daily_checkin_card(checkin)

        sibling_patients = [entry for entry in get_patients_for_doctor(doctor_id) if entry.get("id") != patient.get("id")]
        if sibling_patients:
            with me.box(style=card_style()):
                me.text("Other Active Patients In This Panel", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(260px, 1fr))", gap=16)):
                    for sibling in sibling_patients:
                        with me.box(
                            style=me.Style(
                                padding=me.Padding.all(16),
                                border_radius=18,
                                background=PALETTE["navy_soft"],
                            )
                        ):
                            me.text(sibling["name"], style=me.Style(font_weight="700"))
                            me.text(sibling["diagnosis"], style=me.Style(color=PALETTE["muted"], font_size=14, margin=me.Margin(top=6)))
                            with me.box(style=me.Style(display="flex", gap=10, flex_wrap="wrap", margin=me.Margin(top=12))):
                                me.button(
                                    label="Open",
                                    on_click=navigate_handler("/patient", doctor_id, sibling["id"]),
                                    style=me.Style(
                                        border_radius=999,
                                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                        background=PALETTE["navy"],
                                        color="white",
                                        border=border_none(),
                                        font_weight="700",
                                    ),
                                )
                                me.button(
                                    label="Session",
                                    on_click=navigate_handler("/session", doctor_id, sibling["id"]),
                                    style=me.Style(
                                        border_radius=999,
                                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                        background=PALETTE["surface_strong"],
                                        color=PALETTE["ink"],
                                        border=border_all(PALETTE["line"]),
                                        font_weight="700",
                                    ),
                                )

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            render_patient_logs("Diagnosis Reports", patient.get("diagnosis_reports", []), "title")
            render_patient_logs("Personal Log Records", patient.get("personal_logs", []), "date")

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            render_patient_logs("Care Team Notes", patient.get("care_team_notes", []), "source")
            with me.box(style=card_style()):
                me.text("Why This Matters", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "This page turns the patient from a line item into a longitudinal story. The doctor can move from history to decision to live session without reconstructing context manually.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=18)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    render_data_point("Last update", patient["last_update"], "navy")
                    render_data_point("Session readiness", "Prepared for focused diagnostic review", "teal")

        if session_records:
            with me.box(style=card_style()):
                me.text("Post-Therapy Session Analytics", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Every analyzed session writes back into the patient record: diagnosis hypothesis, BSV, emotional fluctuations, disclosure peaks, and how the patient responded after high-salience moments.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    for session_record in session_records[:4]:
                        render_session_record_card(session_record)

        if nancy_tasks:
            with me.box(style=card_style()):
                me.text("Nancy Directives", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "These are the doctor-approved assignments, reminders, and behavioral follow-ups Nancy can gently weave into conversation between consultations.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for task in nancy_tasks[:4]:
                        render_nancy_task_card(task)

        if nancy_interactions:
            with me.box(style=card_style()):
                me.text("Nancy Handoffs", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Conversational check-ins, voice touchpoints, and between-session summaries are stored here so the doctor can see what shifted before the next live encounter.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    for interaction in nancy_interactions[:4]:
                        render_nancy_interaction_card(interaction)

        with me.box(style=card_style()):
            me.text("Patient Messaging Channel", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "Patients can message Nancy, the doctor, or both. This becomes the between-session communication spine that keeps human care and AI support in one shared thread.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 220px", gap=12)):
                me.textarea(label="Message body", value=state.chat_message_body, on_input=update_chat_message_body, rows=4, appearance="outline")
                me.input(label="Authored by", value=state.chat_message_sender, on_input=update_chat_message_sender, appearance="outline")
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", gap=12, flex_wrap="wrap")):
                me.button(
                    label="Talk To Nancy",
                    on_click=send_message_to_nancy,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Message Doctor",
                    on_click=send_message_to_doctor,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["navy"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Send To Both",
                    on_click=send_message_to_both,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["amber"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
            if patient_messages:
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for message in patient_messages[:6]:
                        render_message_card(message)

        with me.box(style=card_style("linear-gradient(180deg, rgba(181,71,92,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
            me.text("SOS Escalation Routing", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "Emergency rule for this visualization: Shenzhen psychiatric escalation routes first to emergency medical response and then into the supervising hospital and department.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr 1fr", gap=12)):
                render_data_point("Region", "Shenzhen, Guangdong, China", "rose")
                render_data_point("Medical emergency", "120", "amber")
                render_data_point("Immediate danger / police", "110", "navy")
                render_data_point("Doctor hospital", doctor.get("hospital_name", "Unassigned"), "rose")
                render_data_point("Department", doctor.get("department_name", "Unassigned"), "amber")
                render_data_point("Default district", state.sos_location_district, "navy")
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(180px, 1fr))", gap=12)):
                me.input(label="District", value=state.sos_location_district, on_input=update_sos_location_district, appearance="outline")
                me.input(label="Severity", value=state.sos_severity, on_input=update_sos_severity, appearance="outline")
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", flex_direction="column", gap=12)):
                me.textarea(label="SOS reason", value=state.sos_reason, on_input=update_sos_reason, rows=3, appearance="outline")
                me.textarea(label="Escalation notes", value=state.sos_notes, on_input=update_sos_notes, rows=3, appearance="outline")
                me.button(
                    label="Trigger SOS Escalation",
                    on_click=trigger_sos_escalation,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["rose"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
            if sos_events:
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for event in sos_events[:4]:
                        render_sos_event_card(event)

        if outreach_logs:
            with me.box(style=card_style()):
                me.text("Queued Follow-up Activity", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for entry in outreach_logs:
                        render_chat_thread(entry)


@me.page(path="/companion", stylesheets=APP_STYLESHEETS, on_load=on_load)
def companion_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    state = me.state(ApplicationState)
    patient = merge_patient_runtime_data(get_patient(doctor_id, current_patient_id()) or get_patient(doctor_id), doctor_id)

    with page_shell("/companion", doctor_id):
        if not patient:
            render_hero(
                "Patient Companion Portal",
                "No patient is currently selected.",
                "Choose a patient first to open the between-session companion workflow.",
                "Back To Workspace",
                "/workspace",
                primary_doctor_id=doctor_id,
            )
            render_patient_route_hub(
                doctor,
                [merge_patient_runtime_data(entry, doctor_id) for entry in get_patients_for_doctor(doctor_id)],
                "Open A Companion Flow",
                "Companion is now a stable patient route. Choose a patient and jump straight into their between-session care loop.",
                "/companion",
                "Open Companion",
                tone="amber",
            )
            return

        async_summary = build_async_care_summary(patient, doctor)
        async_timeline = build_patient_timeline(patient)
        async_alerts = patient.get("async_alerts", [])
        nancy_tasks = patient.get("nancy_tasks", [])
        latest_checkin = async_summary.get("latest_checkin")

        render_hero(
            "Patient Companion Portal",
            f"{patient['name']} now has a real between-session care loop, supervised by {doctor['name']}.",
            "This is the patient-side operating surface: conversational Nancy support, structured daily reporting, doctor-visible handoffs, and one continuous async timeline.",
            "Back To Patient Record",
            "/patient",
            primary_doctor_id=doctor_id,
            primary_patient_id=patient["id"],
            secondary_label="Open Nancy Console",
            secondary_path="/nancy",
            secondary_doctor_id=doctor_id,
            secondary_patient_id=patient["id"],
        )

        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=18)):
            render_stat_card("Patient", patient["name"], "navy")
            render_stat_card("Risk lane", async_summary.get("risk_level", "routine").title(), "rose" if async_summary.get("risk_level") == "urgent" else ("amber" if async_summary.get("risk_level") == "watch" else "teal"))
            render_stat_card("Doctor reviews needed", str(async_summary.get("review_needed", 0)), "amber")
            render_stat_card("Async messages", str(async_summary.get("message_count", 0)), "teal")
            render_stat_card("Active directives", str(async_summary.get("task_count", 0)), "navy")
            render_stat_card("Open alerts", str(async_summary.get("open_alerts", 0)), "rose")

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style()):
                me.text("Patient Context", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=12)):
                    render_data_point("Primary diagnosis", patient.get("diagnosis", "Unknown"), "navy")
                    render_data_point("Care plan", patient.get("care_plan", "None"), "teal")
                    render_data_point("Supervising doctor", doctor.get("name", "Unknown"), "amber")
                    render_data_point("Hospital", doctor.get("hospital_name", "Unassigned"), "rose")
                    render_data_point("Next appointment", patient.get("next_appointment", "Unknown"), "navy")
                    render_data_point("Latest handoff", async_summary.get("last_handoff", "No handoff yet."), "teal")

            with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Nancy's Approved Scope", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Nancy can check in, collect medically useful updates, follow doctor-approved reminders, and hand important changes back to the care team. She cannot diagnose, prescribe, or independently manage emergencies.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                if async_summary.get("risk_reasons"):
                    with me.box(style=me.Style(margin=me.Margin(top=18))):
                        render_bullet_list(async_summary["risk_reasons"], tone="navy")
                if nancy_tasks:
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=10)):
                        me.text("Current doctor-approved reminders", style=me.Style(font_weight="700"))
                        for task in nancy_tasks[:3]:
                            render_nancy_task_card(task)

        with me.box(style=card_style()):
            me.text("Daily Clinical Report", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "This report now triggers the full async loop: the patient update is stored, Nancy responds, and a doctor-facing handoff is generated automatically.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            if latest_checkin:
                with me.box(style=me.Style(margin=me.Margin(top=16), padding=me.Padding.all(14), border_radius=16, background=PALETTE["navy_soft"])):
                    me.text(
                        f"Latest report: mood {latest_checkin.get('mood_score', '')}/10, anxiety {latest_checkin.get('anxiety_score', '')}/10, sleep {latest_checkin.get('sleep_hours', '')}h.",
                        style=me.Style(font_weight="700", color=PALETTE["navy"]),
                    )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(180px, 1fr))", gap=12)):
                me.input(label="Mood 0-10", value=state.checkin_mood_score, on_input=update_checkin_mood_score, appearance="outline", type="number")
                me.input(label="Anxiety 0-10", value=state.checkin_anxiety_score, on_input=update_checkin_anxiety_score, appearance="outline", type="number")
                me.input(label="Sleep hours", value=state.checkin_sleep_hours, on_input=update_checkin_sleep_hours, appearance="outline", type="number")
                me.input(label="Energy 0-10", value=state.checkin_energy_score, on_input=update_checkin_energy_score, appearance="outline", type="number")
                me.input(label="Stress 0-10", value=state.checkin_stress_score, on_input=update_checkin_stress_score, appearance="outline", type="number")
                me.input(label="Cognition 0-10", value=state.checkin_cognition_score, on_input=update_checkin_cognition_score, appearance="outline", type="number")
                me.input(label="Memory / recall 0-10", value=state.checkin_memory_score, on_input=update_checkin_memory_score, appearance="outline", type="number")
                me.input(label="Functioning 0-10", value=state.checkin_functioning_score, on_input=update_checkin_functioning_score, appearance="outline", type="number")
                me.input(label="Medication adherence %", value=state.checkin_medication_adherence, on_input=update_checkin_medication_adherence, appearance="outline", type="number")
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", flex_direction="column", gap=12)):
                me.textarea(label="Daily clinical update", value=state.checkin_daily_update, on_input=update_checkin_daily_update, rows=3, appearance="outline")
                me.textarea(label="Cognition / recall observations", value=state.checkin_clinical_summary, on_input=update_checkin_clinical_summary, rows=3, appearance="outline")
                me.textarea(label="Significant events or triggers", value=state.checkin_significant_events, on_input=update_checkin_significant_events, rows=3, appearance="outline")
                me.textarea(label="Side effects / medication issues", value=state.checkin_side_effects, on_input=update_checkin_side_effects, rows=2, appearance="outline")
                me.textarea(label="Safety concerns", value=state.checkin_safety_concerns, on_input=update_checkin_safety_concerns, rows=2, appearance="outline")
                me.button(
                    label="Send Daily Report To Nancy + Doctor",
                    on_click=submit_daily_checkin,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )

        with me.box(style=card_style()):
            me.text("Talk To Nancy Or Doctor", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            me.text(
                "Patient messages can go to Nancy, the doctor, or both. If Nancy is included, the platform now generates a patient reply plus a structured clinician handoff.",
                style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
            )
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 220px", gap=12)):
                me.textarea(label="Message body", value=state.chat_message_body, on_input=update_chat_message_body, rows=4, appearance="outline")
                me.input(label="Authored by", value=state.chat_message_sender, on_input=update_chat_message_sender, appearance="outline")
            with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", gap=12, flex_wrap="wrap")):
                me.button(
                    label="Talk To Nancy",
                    on_click=send_message_to_nancy,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["amber"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Message Doctor",
                    on_click=send_message_to_doctor,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["navy"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )
                me.button(
                    label="Send To Both",
                    on_click=send_message_to_both,
                    style=me.Style(
                        border_radius=999,
                        padding=me.Padding.symmetric(horizontal=16, vertical=12),
                        background=PALETTE["teal"],
                        color="white",
                        border=border_none(),
                        font_weight="700",
                    ),
                )

        if async_timeline:
            with me.box(style=card_style()):
                me.text("Unified Async Care Timeline", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Every async touchpoint stays visible in one place: report, reply, handoff, alert, and doctor-facing record movement.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    for entry in async_timeline[:12]:
                        render_timeline_entry(entry)
        if async_alerts:
            with me.box(style=card_style()):
                me.text("Active Review Alerts", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    for alert in async_alerts[:6]:
                        render_async_alert_card(alert, doctor_id)


@me.page(path="/workspace", stylesheets=APP_STYLESHEETS, on_load=on_load)
def workspace_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    panel_patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    panel_size = len(panel_patients)
    state = me.state(ApplicationState)
    outreach_logs = get_outreach_logs(doctor_id)
    nancy_watchtower = collect_recent_nancy_interactions(doctor_id, panel_patients)
    doctor_alerts = get_async_alerts(doctor_id)
    open_alerts = [alert for alert in doctor_alerts if alert.get("status") != "resolved"]
    urgent_alerts = [alert for alert in open_alerts if alert.get("severity") == "urgent"]

    with page_shell("/workspace", doctor_id):
        render_hero(
            "Doctor Command Center",
            f"{doctor['name']} now enters through a clinical operating workspace, not a single demo screen.",
            "This page is the real daily entrance: patient panels, history snapshots, diagnosis reports, appointments, coordination chat, intake staging, and pathways into live sessions and research.",
            "Start Live Session",
            "/session",
            primary_doctor_id=doctor_id,
            secondary_label="Open Research Layer",
            secondary_path="/research",
            secondary_doctor_id=doctor_id,
        )

        render_status_banner(state.status_message)

        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=18)):
            render_stat_card("Active Panel", f"{panel_size} tracked patients", "navy")
            render_stat_card("Watchlist", doctor["stats"]["watchlist"], "rose")
            render_stat_card("Protocols", doctor["stats"]["research"], "teal")
            render_stat_card("Follow-up", doctor["stats"]["followup"], "amber")
            render_stat_card("Open Async Alerts", str(len(open_alerts)), "rose")
            render_stat_card("Urgent Alerts", str(len(urgent_alerts)), "amber")

        render_section_header(
            "Patient Intelligence Dashboard",
            "Every card below is designed to keep the doctor close to what matters: who needs attention, what changed, and which interventions should happen next.",
        )
        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))", gap=18)):
            for patient in panel_patients:
                render_patient_card(patient, doctor_id)

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style()):
                me.text("Upcoming Appointments", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for appointment in doctor["appointments"]:
                        with me.box(
                            style=me.Style(
                                padding=me.Padding.all(16),
                                border_radius=18,
                                background=PALETTE["amber_soft"],
                                display="flex",
                                justify_content="space-between",
                                gap=12,
                                flex_wrap="wrap",
                            )
                        ):
                            with me.box(style=me.Style(display="flex", flex_direction="column", gap=4)):
                                me.text(appointment["patient"], style=me.Style(font_weight="700"))
                                me.text(appointment["type"], style=me.Style(color=PALETTE["muted"], font_size=14))
                            with me.box(style=me.Style(display="flex", flex_direction="column", gap=4, align_items="flex-end")):
                                me.text(appointment["time"], style=me.Style(color=PALETTE["navy"], font_weight="700"))
                                me.text(appointment["mode"], style=me.Style(color=PALETTE["muted"], font_size=13))

            with me.box(style=card_style()):
                me.text("Latest Diagnosis Reports", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for report in doctor["reports"]:
                        render_report_card(report)

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style("linear-gradient(180deg, rgba(181,71,92,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Async Review Inbox", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "This is the doctor’s between-session review queue: the exact alerts opened by patient reports, Nancy handoffs, and SOS events.",
                    style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                    if open_alerts:
                        for alert in open_alerts[:5]:
                            render_async_alert_card(alert, doctor_id)
                    else:
                        me.text("No open async-care alerts right now.", style=me.Style(color=PALETTE["muted"]))

            with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Nancy Watchtower", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Between-session AI companion handoffs appear here first, so the doctor can spot worsening sleep, slipping adherence, cognitive drift, or new safety signals before the appointment begins.",
                    style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    if nancy_watchtower:
                        for interaction in nancy_watchtower[:3]:
                            with me.box(
                                style=me.Style(
                                    padding=me.Padding.all(16),
                                    border_radius=18,
                                    background=PALETTE["surface_strong"],
                                    border=border_all(PALETTE["line"]),
                                )
                            ):
                                me.text(
                                    interaction.get("patient_name", "Patient"),
                                    style=me.Style(font_weight="700"),
                                )
                                me.text(
                                    f"{interaction.get('mode', 'voice').title()} | {interaction.get('escalation_level', 'routine').title()} | {interaction.get('date_label', '')}",
                                    style=me.Style(color=PALETTE["muted"], font_size=13, margin=me.Margin(top=6)),
                                )
                                me.text(
                                    interaction.get("clinician_summary") or interaction.get("patient_report", ""),
                                    style=me.Style(font_size=14, line_height="1.55", margin=me.Margin(top=10)),
                                )
                                if interaction.get("patient_id"):
                                    with me.box(style=me.Style(margin=me.Margin(top=12), display="flex", gap=10, flex_wrap="wrap")):
                                        me.button(
                                            label="Open Nancy",
                                            on_click=navigate_handler("/nancy", doctor_id, interaction["patient_id"]),
                                            style=me.Style(
                                                border_radius=999,
                                                padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                                background=PALETTE["teal"],
                                                color="white",
                                                border=border_none(),
                                                font_weight="700",
                                            ),
                                        )
                                        me.button(
                                            label="Patient",
                                            on_click=navigate_handler("/patient", doctor_id, interaction["patient_id"]),
                                            style=me.Style(
                                                border_radius=999,
                                                padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                                background=PALETTE["surface_strong"],
                                                color=PALETTE["ink"],
                                                border=border_all(PALETTE["line"]),
                                                font_weight="700",
                                            ),
                                        )
                    else:
                        me.text("No Nancy handoffs are logged yet for this doctor panel.", style=me.Style(color=PALETTE["muted"]))

            with me.box(style=card_style()):
                me.text("Nancy Program Design", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Nancy is not a therapist replacement. She is a doctor-supervised relationship AI that handles supportive check-ins, questionnaires, reminders, and clinician handoffs with explicit boundaries.",
                    style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18))):
                    render_bullet_list(
                        [
                            "Conversationally gathers mood, anxiety, sleep, cognition, memory, adherence, and meaningful events.",
                            "References only doctor-approved reminders and never invents treatment instructions.",
                            "Stores voice or text touchpoints into the patient record for pre-session review.",
                            "Escalates safety-critical material instead of trying to manage emergencies itself.",
                        ],
                        tone="navy",
                    )

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style()):
                me.text("Care Coordination Stream", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "A lightweight chat layer helps doctors keep patient follow-up, therapist coordination, and family updates visible in one place.",
                    style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for thread in doctor["chat_threads"]:
                        render_chat_thread(thread)
                    for thread in outreach_logs:
                        render_chat_thread(thread)

                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    me.input(
                        label="Patient / Care Team",
                        value=state.outreach_target,
                        on_input=update_outreach_target,
                        appearance="outline",
                    )
                    me.textarea(
                        label="Check-in message",
                        value=state.outreach_message,
                        on_input=update_outreach_message,
                        rows=4,
                        appearance="outline",
                    )
                    me.button(
                        label="Queue Check-in",
                        on_click=queue_outreach,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["navy"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )

            with me.box(style=card_style()):
                me.text("New Patient Intake", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "This area is where the organization can log new patients, stage initial context, and push them into a live diagnostic session.",
                    style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    me.input(
                        label="Patient name",
                        value=state.intake_name,
                        on_input=update_intake_name,
                        appearance="outline",
                    )
                    me.input(
                        label="Presenting concern",
                        value=state.intake_concern,
                        on_input=update_intake_concern,
                        appearance="outline",
                    )
                    me.textarea(
                        label="Context for first review",
                        value=state.intake_context,
                        on_input=update_intake_context,
                        rows=5,
                        appearance="outline",
                    )
                    with me.box(style=me.Style(display="flex", gap=12, flex_wrap="wrap")):
                        me.button(
                            label="Stage Intake",
                            on_click=stage_intake,
                            style=me.Style(
                                border_radius=999,
                                padding=me.Padding.symmetric(horizontal=16, vertical=12),
                                background=PALETTE["teal"],
                                color="white",
                                border=border_none(),
                                font_weight="700",
                            ),
                        )
                        me.button(
                            label="Open Session Workspace",
                            on_click=navigate_handler("/session", doctor_id),
                            style=me.Style(
                                border_radius=999,
                                padding=me.Padding.symmetric(horizontal=16, vertical=12),
                                background=PALETTE["surface_strong"],
                                color=PALETTE["ink"],
                                border=border_all(PALETTE["line"]),
                                font_weight="700",
                            ),
                        )


@me.page(path="/nancy", stylesheets=APP_STYLESHEETS, on_load=on_load)
def nancy_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    state = me.state(ApplicationState)
    panel_patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    patient = merge_patient_runtime_data(get_patient(doctor_id, current_patient_id()), doctor_id) if current_patient_id() else None
    hospital_directory = load_shenzhen_hospital_directory(limit=120)
    psych_directory = load_shenzhen_psychiatric_directory()

    with page_shell("/nancy", doctor_id):
        if patient:
            nancy_tasks = patient.get("nancy_tasks", [])
            nancy_interactions = patient.get("nancy_interactions", [])
            patient_messages = patient.get("patient_messages", [])
            sos_events = patient.get("sos_events", [])
            async_alerts = patient.get("async_alerts", [])
            settings_preview = build_nancy_settings_preview(doctor, patient)

            render_hero(
                "Nancy Companion Console",
                f"Nancy extends {doctor['name']}'s care relationship with {patient['name']} between sessions.",
                "This is a doctor-supervised AI companion layer: conversational check-ins, memory and cognition touchpoints, task reminders, and structured handoffs back into the patient chart.",
                "Back To Patient",
                "/patient",
                primary_doctor_id=doctor_id,
                primary_patient_id=patient["id"],
                secondary_label="Start Session",
                secondary_path="/session",
                secondary_doctor_id=doctor_id,
                secondary_patient_id=patient["id"],
            )

            with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))", gap=18)):
                render_stat_card("Patient", patient["name"], "navy")
                render_stat_card("Nancy directives", str(len(nancy_tasks)), "teal")
                render_stat_card("Nancy handoffs", str(len(nancy_interactions)), "amber")
                render_stat_card("Latest escalation", (nancy_interactions[0].get("escalation_level", "routine").title() if nancy_interactions else "Routine"), "rose")
                render_stat_card("Open alerts", str(len([alert for alert in async_alerts if alert.get('status') != 'resolved'])), "amber")
                render_stat_card("Shenzhen hospitals", str(hospital_directory.get("total_count", 0)), "navy")
                render_stat_card("Psych hospitals", str(psych_directory.get("total_count", 0)), "teal")

            with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
                with me.box(style=card_style()):
                    me.text("Clinical Safety Charter", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    me.text(
                        "Nancy is intentionally constrained. She supports continuity and observation, not diagnosis or treatment authority.",
                        style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                    )
                    with me.box(style=me.Style(margin=me.Margin(top=18))):
                        render_bullet_list(
                            [
                                "Nancy can ask supportive, medically useful questions in a natural conversation.",
                                "Nancy can reference only doctor-approved reminders, tasks, and prior documented context.",
                                "Nancy cannot diagnose, prescribe, adjust medication, promise secrecy, or handle emergencies alone.",
                                "Any meaningful symptom shift, cognition change, safety concern, or adherence problem is written back for physician review.",
                            ],
                            tone="navy",
                        )

                with me.box(style=card_style("linear-gradient(180deg, rgba(198,106,26,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                    me.text("Doctor Context For Nancy", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=12)):
                        render_data_point("Care plan", patient.get("care_plan", "None"), "navy")
                        render_data_point("Risk level", patient.get("risk", "Unknown"), "rose" if patient.get("risk") == "High" else "amber")
                        render_data_point("Latest daily insight", patient.get("last_daily_checkin_summary", "No questionnaire yet."), "teal")
                        render_data_point("Latest session insight", patient.get("last_session_summary", "No prior session note yet."), "amber")
                        render_data_point("Supervising hospital", doctor.get("hospital_name", "Unassigned"), "navy")
                        render_data_point("Department", doctor.get("department_name", "Unassigned"), "teal")

            with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
                with me.box(style=card_style()):
                    me.text("Assign Doctor-Approved Nancy Directives", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    me.text(
                        "These are the exact between-session prompts Nancy is allowed to bring up with the patient: reading, routines, grounding, journaling, medication reminders, or other clinician-approved tasks.",
                        style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                    )
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                        me.input(label="Directive title", value=state.nancy_task_title, on_input=update_nancy_task_title, appearance="outline")
                        me.input(label="Category", value=state.nancy_task_category, on_input=update_nancy_task_category, appearance="outline")
                        me.input(label="Due / revisit date", value=state.nancy_task_due, on_input=update_nancy_task_due, appearance="outline")
                        me.textarea(label="Directive guidance for Nancy", value=state.nancy_task_note, on_input=update_nancy_task_note, rows=4, appearance="outline")
                        me.button(
                            label="Add Nancy Directive",
                            on_click=add_nancy_directive,
                            style=me.Style(
                                border_radius=999,
                                padding=me.Padding.symmetric(horizontal=16, vertical=12),
                                background=PALETTE["teal"],
                                color="white",
                                border=border_none(),
                                font_weight="700",
                            ),
                        )

                render_code_panel("Deepgram Voice Agent Settings Preview", settings_preview)

            with me.box(style=card_style()):
                me.text("Log Nancy Touchpoint / Handoff", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Use this when Nancy completes a voice or text check-in. The result becomes part of the patient timeline and the doctor’s pre-session briefing.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(180px, 1fr))", gap=12)):
                    me.input(label="Contact mode", value=state.nancy_contact_mode, on_input=update_nancy_contact_mode, appearance="outline")
                    me.input(label="Conversation goal", value=state.nancy_conversation_goal, on_input=update_nancy_conversation_goal, appearance="outline")
                    me.input(label="Observed mood", value=state.nancy_observed_mood, on_input=update_nancy_observed_mood, appearance="outline")
                    me.input(label="Escalation level", value=state.nancy_escalation_level, on_input=update_nancy_escalation_level, appearance="outline")
                with me.box(style=me.Style(margin=me.Margin(top=14), display="grid", grid_template_columns="1fr 1fr", gap=14)):
                    me.textarea(label="Patient-facing update captured by Nancy", value=state.nancy_patient_message, on_input=update_nancy_patient_message, rows=4, appearance="outline")
                    me.textarea(label="Doctor-facing summary", value=state.nancy_clinician_summary, on_input=update_nancy_clinician_summary, rows=4, appearance="outline")
                    me.textarea(label="Functioning note", value=state.nancy_functioning_note, on_input=update_nancy_functioning_note, rows=3, appearance="outline")
                    me.textarea(label="Cognition / memory note", value=state.nancy_cognition_note, on_input=update_nancy_cognition_note, rows=3, appearance="outline")
                    me.textarea(label="Medication / adherence note", value=state.nancy_medication_note, on_input=update_nancy_medication_note, rows=3, appearance="outline")
                    me.textarea(label="Safety note", value=state.nancy_safety_note, on_input=update_nancy_safety_note, rows=3, appearance="outline")
                with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", flex_direction="column", gap=12)):
                    me.textarea(label="Recommended follow-up for doctor / care team", value=state.nancy_recommended_follow_up, on_input=update_nancy_recommended_follow_up, rows=3, appearance="outline")
                    me.button(
                        label="Save Nancy Touchpoint",
                        on_click=log_nancy_touchpoint,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["navy"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )

            with me.box(style=card_style()):
                me.text("Messaging And Human Support", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Patients can talk to Nancy, message the doctor, or send one note to both. Nancy can also push a faster supportive outreach when recent mood signals suggest human contact may help.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 220px", gap=12)):
                    me.textarea(label="Message body", value=state.chat_message_body, on_input=update_chat_message_body, rows=4, appearance="outline")
                    me.input(label="Authored by", value=state.chat_message_sender, on_input=update_chat_message_sender, appearance="outline")
                with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", gap=12, flex_wrap="wrap")):
                    me.button(
                        label="Talk To Nancy",
                        on_click=send_message_to_nancy,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["teal"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )
                    me.button(
                        label="Message Doctor",
                        on_click=send_message_to_doctor,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["navy"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )
                    me.button(
                        label="Send To Both",
                        on_click=send_message_to_both,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["amber"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )
                    me.button(
                        label="Nancy Support Text",
                        on_click=send_nancy_support_ping,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["rose"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )
                if patient_messages:
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                        for message in patient_messages[:6]:
                            render_message_card(message)

            if async_alerts:
                with me.box(style=card_style("linear-gradient(180deg, rgba(198,106,26,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                    me.text("Doctor Review Queue", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    me.text(
                        "Nancy-generated and patient-generated review alerts stay visible here until the doctor acknowledges or resolves them.",
                        style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                    )
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                        for alert in async_alerts[:6]:
                            render_async_alert_card(alert, doctor_id)

            with me.box(style=card_style("linear-gradient(180deg, rgba(181,71,92,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Shenzhen SOS Rule", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Nancy can invoke the hospital escalation flow in-product: Shenzhen emergency medical routing, the supervising doctor, and the doctor’s hospital department all stay in the loop.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr 1fr", gap=12)):
                    render_data_point("Region", "Shenzhen, Guangdong, China", "rose")
                    render_data_point("Emergency medical", "120", "amber")
                    render_data_point("Immediate danger / police", "110", "navy")
                    render_data_point("Doctor hospital", doctor.get("hospital_name", "Unassigned"), "rose")
                    render_data_point("Department", doctor.get("department_name", "Unassigned"), "amber")
                    render_data_point("Psych hospitals loaded", str(psych_directory.get("total_count", 0)), "navy")
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(180px, 1fr))", gap=12)):
                    me.input(label="District", value=state.sos_location_district, on_input=update_sos_location_district, appearance="outline")
                    me.input(label="Severity", value=state.sos_severity, on_input=update_sos_severity, appearance="outline")
                with me.box(style=me.Style(margin=me.Margin(top=14), display="flex", flex_direction="column", gap=12)):
                    me.textarea(label="SOS reason", value=state.sos_reason, on_input=update_sos_reason, rows=3, appearance="outline")
                    me.textarea(label="Escalation notes", value=state.sos_notes, on_input=update_sos_notes, rows=3, appearance="outline")
                    me.button(
                        label="Trigger SOS Escalation",
                        on_click=trigger_sos_escalation,
                        style=me.Style(
                            border_radius=999,
                            padding=me.Padding.symmetric(horizontal=16, vertical=12),
                            background=PALETTE["rose"],
                            color="white",
                            border=border_none(),
                            font_weight="700",
                        ),
                    )
                if sos_events:
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                        for event in sos_events[:4]:
                            render_sos_event_card(event)

            with me.box(style=card_style()):
                me.text("Official Shenzhen Hospital Directory", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "The platform now loads a large official Shenzhen directory plus a dedicated official list of hospitals with psychiatry. This is what the SOS rule and hospital assignment layer can build on.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=20)):
                    with me.box(style=me.Style(display="flex", flex_direction="column", gap=12)):
                        me.text(
                            f"Licensed institutions loaded: {hospital_directory.get('total_count', 0)}",
                            style=me.Style(font_weight="700"),
                        )
                        for hospital in hospital_directory.get("hospitals", [])[:8]:
                            with me.box(style=me.Style(padding=me.Padding.all(14), border_radius=16, background=PALETTE["navy_soft"])):
                                me.text(hospital.get("name", ""), style=me.Style(font_weight="700"))
                                me.text(hospital.get("district", ""), style=me.Style(color=PALETTE["muted"], font_size=13, margin=me.Margin(top=6)))
                                me.text(hospital.get("address", ""), style=me.Style(font_size=13, line_height="1.5", margin=me.Margin(top=8)))
                    with me.box(style=me.Style(display="flex", flex_direction="column", gap=12)):
                        me.text(
                            f"Psychiatry-capable hospitals loaded: {psych_directory.get('total_count', 0)}",
                            style=me.Style(font_weight="700"),
                        )
                        for hospital in psych_directory.get("hospitals", [])[:8]:
                            with me.box(style=me.Style(padding=me.Padding.all(14), border_radius=16, background=PALETTE["teal_soft"])):
                                me.text(hospital.get("name", ""), style=me.Style(font_weight="700"))
                                me.text(hospital.get("district", ""), style=me.Style(color=PALETTE["muted"], font_size=13, margin=me.Margin(top=6)))
                                me.text(hospital.get("address", ""), style=me.Style(font_size=13, line_height="1.5", margin=me.Margin(top=8)))

            if nancy_tasks:
                with me.box(style=card_style()):
                    me.text("Active Nancy Directives", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                        for task in nancy_tasks[:6]:
                            render_nancy_task_card(task)

            if nancy_interactions:
                with me.box(style=card_style()):
                    me.text("Recent Nancy Handoffs", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                    with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                        for interaction in nancy_interactions[:6]:
                            render_nancy_interaction_card(interaction)
        else:
            render_hero(
                "Nancy Network Layer",
                f"{doctor['name']}'s between-session AI companion program spans the whole patient panel.",
                "Select a patient to inspect the exact Deepgram voice-agent configuration, assign directives, and review the conversational care trail Nancy is writing back into the clinical record.",
                "Return To Workspace",
                "/workspace",
                primary_doctor_id=doctor_id,
                secondary_label="Open Doctor Portfolio",
                secondary_path="/doctor",
                secondary_doctor_id=doctor_id,
            )

            with me.box(style=card_style()):
                me.text("What Nancy Adds To The Platform", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18))):
                    render_bullet_list(
                        [
                            "Conversational daily check-ins instead of sterile questionnaires.",
                            "Doctor-approved reminders that carry from session to session.",
                            "Voice-agent touchpoints that become structured doctor handoffs.",
                            "A safer bridge between patient, hospital, and physician when no live consultation is happening.",
                        ]
                    )

            render_patient_route_hub(
                doctor,
                panel_patients,
                "Open A Patient-Specific Nancy Console",
                "Nancy should always open into a concrete patient workflow. Choose the chart you want, and the route will bind Nancy to that patient explicitly.",
                "/nancy",
                "Open Nancy",
                tone="teal",
            )


@me.page(path="/research", stylesheets=APP_STYLESHEETS, on_load=on_load)
def research_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)

    with page_shell("/research", doctor_id):
        render_hero(
            "Research + Reference",
            "A doctor's corner where evidence, performance, and peer learning compound week after week.",
            "This is where the platform turns into a living clinical intelligence network: reference material, improvement signals, peer discussion, and weekly agent briefings that keep doctors updated on changing medical patterns.",
            "Return To Workspace",
            "/workspace",
            primary_doctor_id=doctor_id,
            secondary_label="Run Live Session",
            secondary_path="/session",
            secondary_doctor_id=doctor_id,
        )

        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=20)):
            with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Agent Weekly Brief", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Automated clinical agents surface what changed this week so the doctor does not have to manually scan the entire research landscape.",
                    style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18))):
                    render_bullet_list(doctor["weekly_brief"])

            with me.box(style=card_style()):
                me.text("Performance Improvement", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
                    for item in doctor["performance"]:
                        with me.box(
                            style=me.Style(
                                padding=me.Padding.all(16),
                                border_radius=18,
                                background=PALETTE["navy_soft"],
                            )
                        ):
                            me.text(item["label"], style=me.Style(color=PALETTE["muted"], font_size=13, text_transform="uppercase", font_weight="700"))
                            me.text(item["value"], style=me.Style(font_weight="700", margin=me.Margin(top=8)))

        render_section_header(
            "Doctor's Corner",
            "A forum-style layer where doctors compare strange patterns, discuss edge cases, and generate new research questions faster than traditional channels allow.",
        )
        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))", gap=18)):
            forum_posts = doctor["forum_posts"] or [
                {
                    "community": "Doctor's Corner",
                    "title": "No active threads yet for this specialty cluster",
                    "summary": "As more clinicians enter the network, this space can become the place where unusual patterns and emergent hypotheses are discussed safely.",
                    "activity": "Seed discussion",
                }
            ]
            for post in forum_posts:
                render_forum_post(post)

        with me.box(style=card_style()):
            me.text("Reference Library Strategy", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
            with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(240px, 1fr))", gap=16)):
                for library_card in [
                    ("Latest guideline refresh", "Agents can ingest new psychiatric guidance and summarize what materially changed for practice."),
                    ("Case pattern watchlist", "Cross-organization anomalies can be surfaced before they become obvious in a single clinic."),
                    ("Protocol refinement log", "Doctors can track which research-backed workflow changes improved outcomes over time."),
                ]:
                    with me.box(
                        style=me.Style(
                            padding=me.Padding.all(18),
                            border_radius=20,
                            background=PALETTE["amber_soft"],
                        )
                    ):
                        me.text(library_card[0], style=me.Style(font_weight="700"))
                        me.text(library_card[1], style=me.Style(color=PALETTE["muted"], font_size=14, line_height="1.6", margin=me.Margin(top=10)))


@me.page(path="/session", stylesheets=APP_STYLESHEETS, on_load=on_load)
def session_page():
    doctor_id = current_doctor_id()
    doctor = get_doctor(doctor_id)
    panel_patients = [merge_patient_runtime_data(patient, doctor_id) for patient in get_patients_for_doctor(doctor_id)]
    focused_patient = merge_patient_runtime_data(get_patient(doctor_id, current_patient_id()), doctor_id) if current_patient_id() else None
    session_prep = build_session_prep(doctor, focused_patient)
    session_notes = load_session_notes(doctor_id, current_patient_id())
    state = me.state(ApplicationState)

    with page_shell("/session", doctor_id):
        render_hero(
            "Start Session",
            "The current MindScape workflow now lives inside a richer physician workspace.",
            "This page preserves the existing recording and upload pipeline, while surrounding it with patient context, session preparation, and a more mature diagnostic interface.",
            "Back To Workspace",
            "/workspace",
            primary_doctor_id=doctor_id,
            secondary_label="Open Research",
            secondary_path="/research",
            secondary_doctor_id=doctor_id,
        )

        with me.box(style=me.Style(display="grid", grid_template_columns="repeat(auto-fit, minmax(240px, 1fr))", gap=18)):
            render_stat_card("Doctor", doctor["name"], "navy")
            render_stat_card("Primary Specialty", doctor["specialty"], "teal")
            render_stat_card("Hospital", doctor.get("hospital_name", "Unassigned"), "navy")
            render_stat_card("Patient Context", focused_patient["name"] if focused_patient else "General intake / unassigned", "navy")
            render_stat_card("Session Mode", "Live voice or uploaded audio", "amber")
            render_stat_card("Status", state.status_message, "rose")

        if focused_patient and session_prep:
            with me.box(style=card_style()):
                me.text("Focused Session Prep", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=16)):
                    render_data_point("Patient", session_prep.patient_name, "navy")
                    render_data_point("Diagnosis", session_prep.diagnosis, "teal")
                    render_data_point("Next appointment", session_prep.next_appointment, "amber")
                    render_data_point("Care plan", session_prep.care_plan, "rose")
                    render_data_point("Open async alerts", str(session_prep.open_async_alerts), "rose")
                    render_data_point("Urgent async alerts", str(session_prep.urgent_async_alerts), "amber")
                with me.box(style=me.Style(margin=me.Margin(top=18))):
                    render_bullet_list(session_prep.highlighted_alerts, tone="navy")
            with me.box(style=card_style("linear-gradient(180deg, rgba(15,118,110,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
                me.text("Doctor Brief Before Session", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text(
                    "Everything important before the conversation begins: patient history, last consultation insight, off-consultation reports, memory/cognition flags, and recent medical-record movement.",
                    style=me.Style(color=PALETTE["muted"], line_height="1.7", margin=me.Margin(top=10)),
                )
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="1fr 1fr", gap=12)):
                    render_data_point("History summary", session_prep.history_summary, "navy")
                    render_data_point("Recent medical summary", session_prep.recent_medical_summary, "amber")
                    render_data_point("Latest questionnaire insight", session_prep.latest_questionnaire_insight, "teal")
                    render_data_point("Last consultation insight", session_prep.last_consultation_insight, "rose")
                    render_data_point("Latest Nancy handoff", session_prep.latest_nancy_handoff, "navy")
                    render_data_point("Last clinician note", focused_patient.get("last_session_note_summary", "No clinician review note yet."), "amber")
        elif panel_patients:
            with me.box(style=card_style()):
                me.text("Session Prep Snapshot", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
                me.text("Choose a patient profile for a focused session, or run a general intake analysis from here.", style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10)))
                with me.box(style=me.Style(margin=me.Margin(top=18), display="grid", grid_template_columns="repeat(auto-fit, minmax(260px, 1fr))", gap=16)):
                    for patient in panel_patients:
                        with me.box(
                            style=me.Style(
                                padding=me.Padding.all(18),
                                border_radius=20,
                                background=PALETTE["teal_soft"],
                            )
                        ):
                            me.text(patient["name"], style=me.Style(font_weight="700"))
                            me.text(patient["diagnosis"], style=me.Style(color=PALETTE["muted"], font_size=14, margin=me.Margin(top=6)))
                            me.text(patient["last_update"], style=me.Style(font_size=14, line_height="1.55", margin=me.Margin(top=10)))
                            with me.box(style=me.Style(display="flex", gap=10, flex_wrap="wrap", margin=me.Margin(top=12))):
                                me.button(
                                    label="Focus Session",
                                    on_click=navigate_handler("/session", doctor_id, patient["id"]),
                                    style=me.Style(
                                        border_radius=999,
                                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                        background=PALETTE["navy"],
                                        color="white",
                                        border=border_none(),
                                        font_weight="700",
                                    ),
                                )
                                me.button(
                                    label="Open Patient",
                                    on_click=navigate_handler("/patient", doctor_id, patient["id"]),
                                    style=me.Style(
                                        border_radius=999,
                                        padding=me.Padding.symmetric(horizontal=12, vertical=8),
                                        background=PALETTE["surface_strong"],
                                        color=PALETTE["ink"],
                                        border=border_all(PALETTE["line"]),
                                        font_weight="700",
                                    ),
                                )

        render_status_banner("Live diagnostic workflow below still uses the existing engine, now embedded in the doctor operating system.", tone="amber")

        with me.box(style=me.Style(display="grid", grid_template_columns="1.2fr 0.8fr", gap=24)):
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
                render_session_controls(state, focused_patient)
                render_transcript_panel(state)
                if state.reference_cases:
                    render_reference_cases(state.reference_cases)
                if state.retrieved_evidence:
                    render_evidence_panel(state.retrieved_evidence)
                if state.follow_up_questions:
                    render_follow_up_panel(state.follow_up_questions)
                if state.treatment_plan:
                    render_treatment_panel(state.treatment_plan)
                if focused_patient:
                    render_session_review_notes(state, doctor, focused_patient, session_notes)

            with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
                render_bsv_panel(state)
                render_somatic_panel(state)
                render_pipeline_graph(state)
                render_hypothesis_card(state)


def render_session_controls(state: ApplicationState, patient: dict | None = None):
    with me.box(style=card_style("linear-gradient(180deg, rgba(22,56,77,0.05) 0%, rgba(255,255,255,0.88) 100%)")):
        me.text("Live Diagnostic Capture", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        me.text(
            "Start a live mic capture or upload an existing conversation recording. The current workflow stays operational here, but now it sits inside a broader clinical story.",
            style=me.Style(color=PALETTE["muted"], line_height="1.6", margin=me.Margin(top=12)),
        )
        if patient:
            with me.box(
                style=me.Style(
                    margin=me.Margin(top=14),
                    padding=me.Padding.symmetric(horizontal=16, vertical=12),
                    border_radius=18,
                    background=PALETTE["teal_soft"],
                )
            ):
                me.text(
                    f"Focused patient: {patient['name']} | {patient['diagnosis']} | Risk {patient['risk']}",
                    style=me.Style(color=PALETTE["teal"], font_weight="700"),
                )

        with me.box(
            style=me.Style(
                margin=me.Margin(top=20),
                display="flex",
                flex_direction="column",
                align_items="center",
                gap=16,
                padding=me.Padding.all(26),
                border_radius=24,
                background="rgba(255, 255, 255, 0.76)",
                border=border_all(PALETTE["line"]),
            )
        ):
            if state.is_recording:
                with me.box(
                    style=me.Style(
                        width=88,
                        height=88,
                        border_radius=999,
                        background=PALETTE["rose"],
                        border=border_none(),
                        display="flex",
                        align_items="center",
                        justify_content="center",
                        box_shadow="0 0 28px rgba(181, 71, 92, 0.35)",
                    )
                ):
                    me.icon("mic", style=me.Style(color="white", font_size=42))
                me.text("Listening for live speech...", style=me.Style(color=PALETTE["rose"], font_weight="700"))
            elif state.is_processing:
                me.progress_spinner()
                me.text("Analyzing session layers...", style=me.Style(color=PALETTE["teal"], font_weight="700"))
            else:
                with me.box(
                    style=me.Style(
                        width=88,
                        height=88,
                        border_radius=999,
                        background=PALETTE["navy"],
                        border=border_none(),
                        display="flex",
                        align_items="center",
                        justify_content="center",
                        cursor="pointer",
                    ),
                    on_click=toggle_recording,
                ):
                    me.icon("mic", style=me.Style(color="white", font_size=42))
                me.text("Tap to begin a 5-second live capture", style=me.Style(color=PALETTE["muted"]))

            me.uploader(
                label="Or upload an audio file",
                on_upload=handle_upload,
                accepted_file_types=["audio/*"],
                style=me.Style(
                    background=PALETTE["surface_strong"],
                    color=PALETTE["ink"],
                    border_radius=14,
                    font_weight="700",
                ),
            )

        # Camera enable / disable row
        with me.box(style=me.Style(margin=me.Margin(top=16), display="flex", align_items="center", gap=12)):
            cam_bg = PALETTE["teal"] if state.camera_active else PALETTE["surface_strong"]
            cam_fg = "white" if state.camera_active else PALETTE["ink"]
            cam_border = border_none() if state.camera_active else border_all(PALETTE["line"])
            me.button(
                label="Disable Camera" if state.camera_active else "Enable Somatic Camera",
                on_click=toggle_camera,
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=18, vertical=10),
                    background=cam_bg,
                    color=cam_fg,
                    border=cam_border,
                    font_weight="700",
                ),
            )
            if state.camera_active:
                dot_color = PALETTE["success"] if state.visual_bsv_face_detected else PALETTE["amber"]
                dot_label = "Face detected" if state.visual_bsv_face_detected else "No face"
                with me.box(style=me.Style(display="flex", align_items="center", gap=6)):
                    with me.box(style=me.Style(width=8, height=8, border_radius=999, background=dot_color)):
                        pass
                    me.text(dot_label, style=me.Style(color=PALETTE["muted"], font_size=13))

        if state.stt_provider:
            me.text(
                f"Transcribed via {state.stt_provider}",
                style=me.Style(color=PALETTE["muted"], font_size=12, margin=me.Margin(top=8)),
            )


def render_transcript_panel(state: ApplicationState):
    with me.box(style=card_style()):
        me.text("Session Transcript", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        me.text("Paralinguistic tags remain visible so clinicians can inspect the signal behind the text.", style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10)))
        if state.stt_provider:
            me.text(
                f"Transcription provider: {state.stt_provider}",
                style=me.Style(color=PALETTE["navy"], font_size=13, font_weight="700", margin=me.Margin(top=8)),
            )
        with me.box(style=me.Style(margin=me.Margin(top=18))):
            if state.transcript:
                render_tags(state.transcript)
            else:
                me.text("No speech captured yet.", style=me.Style(color=PALETTE["muted"], font_style="italic"))


def render_reference_cases(reference_cases: list[dict]):
    with me.box(style=card_style(PALETTE["amber_soft"])):
        me.text("Historical Case Matches", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=16)):
            for case in reference_cases:
                with me.box(
                    style=me.Style(
                        background="rgba(255,255,255,0.7)",
                        padding=me.Padding.all(16),
                        border_radius=18,
                        border=border_all(PALETTE["line"]),
                    )
                ):
                    me.text(case.get("title", "Historical Case"), style=me.Style(font_weight="700", font_size=17))
                    me.markdown(f"**Relevance:** {case.get('relevance', '')}", style=me.Style(color=PALETTE["ink"], margin=me.Margin(top=8)))
                    me.markdown(
                        f"**Historical pathway:** {case.get('historical_treatment', '')}",
                        style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=8)),
                    )


def render_evidence_panel(evidence_list: list[str]):
    with me.box(style=card_style(PALETTE["navy_soft"])):
        me.text("Retrieved Clinical Base", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
            for evidence in evidence_list:
                with me.box(
                    style=me.Style(
                        background="rgba(255,255,255,0.72)",
                        padding=me.Padding.all(14),
                        border_radius=16,
                    )
                ):
                    me.markdown(evidence, style=me.Style(color=PALETTE["ink"], font_size=14, line_height="1.55"))


def render_follow_up_panel(questions: list[str]):
    with me.box(style=card_style(PALETTE["teal_soft"])):
        me.text("Suggested Inquiry", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
            for question in questions:
                with me.box(
                    style=me.Style(
                        background="rgba(255,255,255,0.75)",
                        padding=me.Padding.all(14),
                        border_radius=16,
                    )
                ):
                    me.text(question, style=me.Style(font_size=14, color=PALETTE["ink"], font_style="italic"))


def render_treatment_panel(treatment_plan: str):
    with me.box(style=card_style("rgba(45, 123, 75, 0.09)")):
        me.text("First-Line Treatment Pathway", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        me.markdown(
            treatment_plan,
            style=me.Style(
                margin=me.Margin(top=18),
                padding=me.Padding.all(16),
                border_radius=18,
                background="rgba(255,255,255,0.76)",
                color=PALETTE["ink"],
                line_height="1.65",
            ),
        )


def render_session_review_notes(state: ApplicationState, doctor: dict, patient: dict, session_notes: list[dict]):
    with me.box(style=card_style("linear-gradient(180deg, rgba(198,106,26,0.08) 0%, rgba(255,255,255,0.88) 100%)")):
        me.text("Clinician Review Notes", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        me.text(
            "This closes the loop after the model output: clinician interpretation, disposition, and plan updates stay attached to the session slice.",
            style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10), line_height="1.6"),
        )
        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
            me.input(label="Note title", value=state.session_note_title, on_input=update_session_note_title, appearance="outline")
            me.textarea(label="Clinician review note", value=state.session_note_body, on_input=update_session_note_body, rows=4, appearance="outline")
            me.textarea(label="Care-plan update", value=state.session_plan_update, on_input=update_session_plan_update, rows=3, appearance="outline")
            me.input(label="Disposition", value=state.session_disposition, on_input=update_session_disposition, appearance="outline")
            me.button(
                label="Save Session Review Note",
                on_click=save_session_review_note,
                style=me.Style(
                    border_radius=999,
                    padding=me.Padding.symmetric(horizontal=16, vertical=12),
                    background=PALETTE["amber"],
                    color="white",
                    border=border_none(),
                    font_weight="700",
                ),
            )

        if session_notes:
            with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=14)):
                for note in session_notes[:5]:
                    render_session_note_card(note)


def render_bsv_panel(state: ApplicationState):
    with me.box(style=card_style()):
        me.text("Behavioral State Vector", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))
        me.text("Voice-derived affect features remain visible as a separate interpretability layer.", style=me.Style(color=PALETTE["muted"], margin=me.Margin(top=10)))
        with me.box(style=me.Style(margin=me.Margin(top=20))):
            bsv_meter("Valence", state.bsv_valence, -1, 1, [PALETTE["rose"], PALETTE["success"]])
            bsv_meter("Arousal", state.bsv_arousal, 0, 1, [PALETTE["navy"], PALETTE["amber"]])
            bsv_meter("Dominance", state.bsv_dominance, 0, 1, [PALETTE["amber"], PALETTE["teal"]])


def render_somatic_panel(state: ApplicationState):
    """Somatic Intelligence card — camera status and FaceMesh-derived BSV."""
    with me.box(style=card_style()):
        with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="center", margin=me.Margin(bottom=16))):
            me.text("Somatic Intelligence", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=22, font_weight="700"))
            dot_color = PALETTE["success"] if state.camera_active else PALETTE["line"]
            with me.box(style=me.Style(display="flex", align_items="center", gap=6)):
                with me.box(style=me.Style(width=9, height=9, border_radius=999, background=dot_color)):
                    pass
                me.text("Live" if state.camera_active else "Off", style=me.Style(color=PALETTE["muted"], font_size=13, font_weight="600"))

        me.text(
            "FaceMesh CV analyses muscular micro-expressions, Action Units, blink rate, and gaze stability frame-by-frame, adding a somatic channel to the behavioral state vector.",
            style=me.Style(color=PALETTE["muted"], font_size=14, line_height="1.6", margin=me.Margin(bottom=18)),
        )

        if not state.camera_active:
            me.text(
                "Enable the camera from the session controls to activate somatic analysis.",
                style=me.Style(color=PALETTE["muted"], font_style="italic", font_size=14),
            )
            return

        # Live annotated camera feed (MJPEG via embedded iframe)
        with me.box(style=me.Style(
            border_radius=14,
            overflow="hidden",
            margin=me.Margin(bottom=18),
            border=border_all("rgba(255,255,255,0.06)"),
            background="#0a0a0a",
        )):
            me.embed(
                src="http://localhost:5001/video_html",
                style=me.Style(width="100%", height=360, display="block"),
            )

        if not state.visual_bsv_face_detected:
            with me.box(style=me.Style(
                padding=me.Padding.symmetric(horizontal=14, vertical=10),
                border_radius=12,
                background=PALETTE["amber_soft"],
                margin=me.Margin(bottom=12),
            )):
                me.text("Position patient in frame — somatic inference active when face detected.", style=me.Style(color=PALETTE["amber"], font_weight="600", font_size=13))

        # Visual BSV meters
        bsv_meter("Facial Valence", state.visual_bsv_facial_valence, -1, 1, [PALETTE["rose"], PALETTE["success"]])
        bsv_meter("Agitation Index", state.visual_bsv_facial_arousal, 0, 1, [PALETTE["teal"], PALETTE["rose"]])
        bsv_meter("Gaze Stability", state.visual_bsv_gaze_stability, 0, 1, [PALETTE["amber"], PALETTE["success"]])

        # Stat chips: blink rate + flat affect
        with me.box(style=me.Style(display="flex", gap=12, margin=me.Margin(top=8, bottom=16), flex_wrap="wrap")):
            _somatic_stat("Blink Rate", f"{state.visual_bsv_blink_rate:.0f} /min", "15–20 normal", "navy")
            _somatic_stat("Flat Affect", f"{state.visual_bsv_flat_affect:.2f}", ">0.7 clinically significant", "amber")

        # Micro-twitch zones
        if state.visual_bsv_twitch_zones:
            me.text("Micro-twitch zones", style=me.Style(font_weight="700", font_size=13, margin=me.Margin(bottom=8)))
            with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap=6)):
                for zone in state.visual_bsv_twitch_zones:
                    with me.box(style=me.Style(
                        background=PALETTE["rose_soft"],
                        padding=me.Padding.symmetric(horizontal=10, vertical=5),
                        border_radius=999,
                        border=border_all("rgba(181, 71, 92, 0.22)"),
                    )):
                        me.text(zone.replace("_", " "), style=me.Style(color=PALETTE["rose"], font_size=12, font_weight="700"))

        # Active AUs
        if state.visual_bsv_active_aus:
            me.text("Active Action Units", style=me.Style(font_weight="700", font_size=13, margin=me.Margin(top=12, bottom=8)))
            with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap=6)):
                for au in state.visual_bsv_active_aus:
                    with me.box(style=me.Style(
                        background=PALETTE["navy_soft"],
                        padding=me.Padding.symmetric(horizontal=10, vertical=5),
                        border_radius=999,
                        border=border_all("rgba(22, 56, 77, 0.18)"),
                    )):
                        me.text(au, style=me.Style(color=PALETTE["navy"], font_size=12, font_weight="700"))


def _somatic_stat(label: str, value: str, note: str, tone: str):
    bg = PALETTE[f"{tone}_soft"]
    color = PALETTE[tone]
    with me.box(style=me.Style(
        background=bg,
        padding=me.Padding.all(14),
        border_radius=18,
        flex=1,
        min_width=120,
    )):
        me.text(label, style=me.Style(color=PALETTE["muted"], font_size=11, text_transform="uppercase", letter_spacing="0.6px", font_weight="700"))
        me.text(value, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700", color=color, margin=me.Margin(top=6)))
        me.text(note, style=me.Style(color=PALETTE["muted"], font_size=10, margin=me.Margin(top=4)))


def render_pipeline_graph(state: ApplicationState):
    with me.box(style=card_style()):
        me.text("System Execution Pipeline", style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=24, font_weight="700"))

        nodes = [
            (1, "Acoustic affect extraction"),
            (2, "Multimodal context fusion"),
            (3, "Hybrid clinical retrieval"),
            (4, "LLM synthesis and safety"),
            (5, "Analysis complete"),
        ]

        with me.box(style=me.Style(margin=me.Margin(top=18), display="flex", flex_direction="column", gap=12)):
            for node_id, label in nodes:
                is_active = state.pipeline_active_node == node_id
                is_done = state.pipeline_active_node > node_id
                dot_color = PALETTE["success"] if is_done else (PALETTE["teal"] if is_active else "rgba(16, 33, 43, 0.20)")
                bg_color = PALETTE["teal_soft"] if is_active else "transparent"

                with me.box(
                    style=me.Style(
                        display="flex",
                        align_items="center",
                        gap=12,
                        padding=me.Padding.all(10),
                        background=bg_color,
                        border_radius=16,
                    )
                ):
                    with me.box(
                        style=me.Style(
                            width=12,
                            height=12,
                            border_radius=999,
                            background=dot_color,
                            box_shadow="0 0 12px rgba(15, 118, 110, 0.28)" if is_active else "none",
                        )
                    ):
                        pass
                    me.text(
                        label,
                        style=me.Style(
                            color=PALETTE["ink"] if is_done or is_active else PALETTE["muted"],
                            font_weight="700" if is_active else "500",
                        ),
                    )

        if state.pipeline_logs:
            with me.box(
                style=me.Style(
                    margin=me.Margin(top=20),
                    padding=me.Padding.all(16),
                    border_radius=18,
                    background=PALETTE["navy"],
                    display="flex",
                    flex_direction="column",
                    gap=8,
                )
            ):
                me.text("Intelligence tracer", style=me.Style(color="rgba(255,255,255,0.72)", text_transform="uppercase", letter_spacing="0.9px", font_size=11, font_weight="700"))
                for pipeline_log in state.pipeline_logs:
                    me.text(f"> {pipeline_log}", style=me.Style(color="white", font_family="monospace", font_size=12, line_height="1.45"))


def render_hypothesis_card(state: ApplicationState):
    gate_color = PALETTE["success"] if state.safety_gate == "PASS" else PALETTE["rose"]
    gate_background = PALETTE["success_soft"] if state.safety_gate == "PASS" else PALETTE["rose_soft"]

    with me.box(
        style=me.Style(
            background=PALETTE["surface"],
            border=me.Border(
                top=me.BorderSide(width=1, color=PALETTE["line"]),
                right=me.BorderSide(width=1, color=PALETTE["line"]),
                bottom=me.BorderSide(width=1, color=PALETTE["line"]),
                left=me.BorderSide(width=5, color=gate_color),
            ),
            border_radius=24,
            padding=me.Padding.all(24),
            box_shadow=PALETTE["shadow"],
            display="flex",
            flex_direction="column",
            gap=18,
        )
    ):
        with me.box(style=me.Style(display="flex", justify_content="space-between", gap=12, flex_wrap="wrap", align_items="center")):
            me.text("Preliminary Diagnosis", style=me.Style(color=PALETTE["muted"], font_weight="700", text_transform="uppercase", font_size=12, letter_spacing="0.9px"))
            with me.box(
                style=me.Style(
                    padding=me.Padding.symmetric(horizontal=12, vertical=8),
                    border_radius=999,
                    background=gate_background,
                )
            ):
                me.text(f"Gate: {state.safety_gate}", style=me.Style(color=gate_color, font_weight="700", font_size=12))

        with me.box(style=me.Style(display="flex", gap=10, align_items="baseline", flex_wrap="wrap")):
            me.text(state.hypothesis_name, style=me.Style(font_family="'Space Grotesk', sans-serif", font_size=34, font_weight="700"))
            if state.hypothesis_confidence:
                me.text(state.hypothesis_confidence, style=me.Style(color=PALETTE["muted"], font_family="monospace", font_size=15))

        if state.hypothesis_reasoning:
            me.markdown(state.hypothesis_reasoning, style=me.Style(color=PALETTE["muted"], line_height="1.7"))

        me.text("Evidence", style=me.Style(font_weight="700", font_size=15))
        me.markdown(state.hypothesis_evidence or "Evidence will appear after the first processed session.", style=me.Style(color=PALETTE["ink"], line_height="1.65"))


def bsv_meter(label: str, value: float, min_value: float, max_value: float, colors: list[str]):
    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0.0, min(1.0, normalized))
    percentage = f"{normalized * 100:.0f}%"

    with me.box(style=me.Style(margin=me.Margin(bottom=18))):
        with me.box(style=me.Style(display="flex", justify_content="space-between", margin=me.Margin(bottom=8))):
            me.text(label, style=me.Style(font_weight="600"))
            me.text(f"{value:.2f}", style=me.Style(color=PALETTE["muted"], font_family="monospace"))

        with me.box(style=me.Style(background="rgba(16, 33, 43, 0.10)", height=8, border_radius=999, overflow="hidden")):
            with me.box(
                style=me.Style(
                    background=f"linear-gradient(90deg, {colors[0]}, {colors[1]})",
                    width=percentage,
                    height="100%",
                    border_radius=999,
                    transition="width 0.5s ease",
                )
            ):
                pass


def render_tags(text: str):
    parts = re.split(r"(<[^>]+>)", text)
    with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap=8, align_items="center")):
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("<") and part.endswith(">"):
                with me.box(
                    style=me.Style(
                        background=PALETTE["navy_soft"],
                        color=PALETTE["navy"],
                        padding=me.Padding.symmetric(horizontal=10, vertical=5),
                        border_radius=999,
                        border=border_all("rgba(22, 56, 77, 0.18)"),
                        font_size=12,
                        font_weight="700",
                    )
                ):
                    me.text(part)
            else:
                me.text(part, style=me.Style(font_size=16, line_height="1.7", color=PALETTE["ink"]))


def get_active_patient_context() -> tuple[dict | None, str | None]:
    doctor_id = current_doctor_id()
    patient_id = current_patient_id()
    patient = merge_patient_runtime_data(get_patient(doctor_id, patient_id), doctor_id) if patient_id else None
    return patient, build_patient_context(patient, doctor_id)


def toggle_recording(e: me.ClickEvent):
    state = me.state(ApplicationState)
    if state.is_recording:
        return

    reset_analysis_state(state)
    state.is_recording = True
    state.status_message = "Listening..."
    state.hypothesis_name = "Analyzing live capture"
    yield

    try:
        doctor = get_doctor(current_doctor_id())
        active_patient, patient_context = get_active_patient_context()
        state.is_recording = False
        state.is_processing = True
        state.status_message = "Processing acoustic layer..."
        yield

        result = {}
        transcript = ""
        provider = ""
        for update in analyze_live_capture(duration=5, patient_context=patient_context):
            if update.get("phase") == "transcribed":
                transcript = update["transcript"]
                provider = update["provider"]
                state.transcript = transcript
                state.stt_provider = provider
            if "node" in update:
                state.pipeline_active_node = update["node"]
                state.status_message = update["status"]
            if "log" in update:
                state.pipeline_logs.append(update["log"])
            if update.get("phase") == "completed":
                result = update["result"]
                apply_session_view_model(state, update["view_model"])
            yield

        persist_session_result(doctor, active_patient, transcript, provider, result)
        state.status_message = "Analysis complete"
    except Exception as ex:
        state.status_message = f"Session error: {str(ex)}"
        state.is_recording = False

    state.is_processing = False
    yield


def handle_upload(event: me.UploadEvent):
    state = me.state(ApplicationState)
    reset_analysis_state(state)
    state.is_processing = True
    state.status_message = "Uploading and analyzing..."
    state.hypothesis_name = "Analyzing uploaded session"
    yield

    try:
        doctor = get_doctor(current_doctor_id())
        active_patient, patient_context = get_active_patient_context()
        result = {}
        transcript = ""
        provider = ""
        for update in analyze_uploaded_file(event.file, event.file.name, patient_context=patient_context):
            if update.get("phase") == "transcribed":
                transcript = update["transcript"]
                provider = update["provider"]
                state.transcript = transcript
                state.stt_provider = provider
                state.status_message = "Running diagnostic pipeline..."
            if "node" in update:
                state.pipeline_active_node = update["node"]
                state.status_message = update["status"]
            if "log" in update:
                state.pipeline_logs.append(update["log"])
            if update.get("phase") == "completed":
                result = update["result"]
                apply_session_view_model(state, update["view_model"])
            yield

        persist_session_result(doctor, active_patient, transcript, provider, result)
        state.status_message = "Analysis complete"
    except Exception as ex:
        state.status_message = f"Upload error: {str(ex)}"

    state.is_processing = False
    yield


def apply_session_view_model(state: ApplicationState, view_model: dict):
    state.transcript = view_model.get("transcript", "")
    state.stt_provider = view_model.get("stt_provider", "")
    state.bsv_valence = float(view_model.get("bsv_valence", 0.0))
    state.bsv_arousal = float(view_model.get("bsv_arousal", 0.0))
    state.bsv_dominance = float(view_model.get("bsv_dominance", 0.0))
    state.hypothesis_name = view_model.get("hypothesis_name", "Unknown")
    state.hypothesis_confidence = view_model.get("hypothesis_confidence", "")
    state.hypothesis_reasoning = view_model.get("hypothesis_reasoning", "")
    state.hypothesis_evidence = view_model.get("hypothesis_evidence", "")
    state.retrieved_evidence = view_model.get("retrieved_evidence", [])
    state.follow_up_questions = view_model.get("follow_up_questions", [])
    state.treatment_plan = view_model.get("treatment_plan", "")
    state.reference_cases = view_model.get("reference_cases", [])
    state.traumatic_markers = view_model.get("traumatic_markers", [])
    state.emotion_trajectory = view_model.get("emotion_trajectory", [])
    state.safety_gate = view_model.get("safety_gate", "FAIL")
    state.visual_bsv_facial_valence = float(view_model.get("visual_bsv_facial_valence", 0.0))
    state.visual_bsv_facial_arousal = float(view_model.get("visual_bsv_facial_arousal", 0.0))
    state.visual_bsv_blink_rate = float(view_model.get("visual_bsv_blink_rate", 0.0))
    state.visual_bsv_flat_affect = float(view_model.get("visual_bsv_flat_affect", 0.0))
    state.visual_bsv_gaze_stability = float(view_model.get("visual_bsv_gaze_stability", 1.0))
    state.visual_bsv_twitch_zones = view_model.get("visual_bsv_twitch_zones", [])
    state.visual_bsv_face_detected = bool(view_model.get("visual_bsv_face_detected", False))
    state.visual_bsv_active_aus = view_model.get("visual_bsv_active_aus", [])

def toggle_camera(e: me.ClickEvent):
    state = me.state(ApplicationState)
    import mindscape_engine as _mindscape_engine  # noqa: PLC0415 — heavy stack only when camera is used

    analyzer = _mindscape_engine.get_visual_analyzer()
    if state.camera_active:
        analyzer.stop()
        state.camera_active = False
        state.visual_bsv_face_detected = False
    else:
        success = analyzer.start()
        state.camera_active = success
    yield
