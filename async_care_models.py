from dataclasses import asdict, dataclass, field


@dataclass
class CheckinPayload:
    mood_score: str = ""
    anxiety_score: str = ""
    sleep_hours: str = ""
    energy_score: str = ""
    stress_score: str = ""
    cognition_score: str = ""
    memory_score: str = ""
    functioning_score: str = ""
    medication_adherence: str = ""
    side_effects: str = ""
    daily_update: str = ""
    safety_concerns: str = ""
    significant_events: str = ""
    clinical_summary: str = ""

    @classmethod
    def from_dict(cls, payload: dict | None):
        payload = payload or {}
        return cls(
            mood_score=str(payload.get("mood_score", "")),
            anxiety_score=str(payload.get("anxiety_score", "")),
            sleep_hours=str(payload.get("sleep_hours", "")),
            energy_score=str(payload.get("energy_score", "")),
            stress_score=str(payload.get("stress_score", "")),
            cognition_score=str(payload.get("cognition_score", "")),
            memory_score=str(payload.get("memory_score", "")),
            functioning_score=str(payload.get("functioning_score", "")),
            medication_adherence=str(payload.get("medication_adherence", "")),
            side_effects=str(payload.get("side_effects", "")),
            daily_update=str(payload.get("daily_update", "")),
            safety_concerns=str(payload.get("safety_concerns", "")),
            significant_events=str(payload.get("significant_events", "")),
            clinical_summary=str(payload.get("clinical_summary", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AsyncAlertRecord:
    id: str
    patient_id: str
    patient_name: str
    severity: str
    source: str
    title: str
    summary: str
    recommended_follow_up: str = ""
    route: str = "/patient"
    status: str = "new"
    created_at: str = ""
    updated_at: str = ""
    date_label: str = ""
    time_label: str = ""
    metadata: dict = field(default_factory=dict)
    last_action_by: str = ""
    last_action_note: str = ""

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(
            id=str(payload.get("id", "")),
            patient_id=str(payload.get("patient_id", "")),
            patient_name=str(payload.get("patient_name", "")),
            severity=str(payload.get("severity", "watch")),
            source=str(payload.get("source", "async care")),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            recommended_follow_up=str(payload.get("recommended_follow_up", "")),
            route=str(payload.get("route", "/patient")),
            status=str(payload.get("status", "new")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            date_label=str(payload.get("date_label", "")),
            time_label=str(payload.get("time_label", "")),
            metadata=payload.get("metadata", {}) or {},
            last_action_by=str(payload.get("last_action_by", "")),
            last_action_note=str(payload.get("last_action_note", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AsyncCareSummary:
    risk_level: str
    risk_reasons: list[str]
    last_handoff: str
    task_count: int
    message_count: int
    review_needed: int
    open_alerts: int
    urgent_alerts: int
    doctor_name: str
    latest_checkin: dict | None = None


@dataclass
class TimelineEntry:
    kind: str
    created_at: str
    payload: dict
