from dataclasses import asdict, dataclass, field


@dataclass
class SessionPrepBrief:
    patient_name: str
    diagnosis: str
    care_plan: str
    next_appointment: str
    history_summary: str
    recent_medical_summary: str
    latest_questionnaire_insight: str
    last_consultation_insight: str
    latest_nancy_handoff: str
    open_async_alerts: int = 0
    urgent_async_alerts: int = 0
    highlighted_alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionAnalysisView:
    transcript: str
    stt_provider: str
    bsv_valence: float
    bsv_arousal: float
    bsv_dominance: float
    hypothesis_name: str
    hypothesis_confidence: str
    hypothesis_reasoning: str
    hypothesis_evidence: str
    retrieved_evidence: list[str]
    follow_up_questions: list[str]
    treatment_plan: str
    reference_cases: list[dict]
    traumatic_markers: list[str]
    emotion_trajectory: list[dict]
    safety_gate: str
    visual_bsv_facial_valence: float = 0.0
    visual_bsv_facial_arousal: float = 0.0
    visual_bsv_blink_rate: float = 0.0
    visual_bsv_flat_affect: float = 0.0
    visual_bsv_gaze_stability: float = 1.0
    visual_bsv_twitch_zones: list[str] = field(default_factory=list)
    visual_bsv_face_detected: bool = False
    visual_bsv_active_aus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimelineEntry:
    kind: str
    created_at: str
    payload: dict
