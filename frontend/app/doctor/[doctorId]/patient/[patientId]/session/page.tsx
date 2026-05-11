import { SessionNoteForm, SessionUploadForm } from "@/components/forms";
import { AppShell, Card, DataGrid } from "@/components/ui";
import { api } from "@/lib/api";

export default async function SessionPage({ params }: { params: Promise<{ doctorId: string; patientId: string }> }) {
  const { doctorId, patientId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}/patients/${patientId}/session`);
  const patient = payload.patient;
  const prep = payload.session_prep;
  const latestSession = payload.latest_session;
  const evidenceList = Array.isArray(latestSession?.retrieved_evidence)
    ? latestSession.retrieved_evidence
    : latestSession?.retrieved_evidence
      ? [latestSession.retrieved_evidence]
      : [];
  const followupList = Array.isArray(latestSession?.follow_up)
    ? latestSession.follow_up
    : latestSession?.follow_up
      ? [latestSession.follow_up]
      : [];
  const transcriptBlocks: Array<{ speaker?: string; text?: string }> = [];
  const sessionConfidence = latestSession?.hypothesis_confidence ?? "No analysis yet";
  const sessionBsv = latestSession?.bsv ?? {};
  const visualBsv = latestSession?.visual_bsv ?? {};
  const eventMarkers = Array.isArray(latestSession?.event_markers) ? latestSession.event_markers : [];
  const latestReasoning =
    latestSession?.reasoning ?? "Upload or capture a session to generate diagnostic reasoning and evidence-backed notes.";
  const latestPlan =
    latestSession?.treatment_plan ?? "No treatment plan has been generated for this patient yet.";

  return (
    <AppShell doctorId={doctorId} patientId={patientId} currentSection="session">
      <section className="session-hero">
        <div className="session-hero-copy">
          <div className="session-hero-eyebrow">
            Session Intelligence · {patient?.name ?? "Patient"}
          </div>
          <h1 className="session-hero-title">Every session becomes a structured clinical record.</h1>
          <p className="session-hero-body">
            Audio, transcript, affect, hypothesis, evidence, and review notes all land in one route
            backed by the existing clinical engine.
          </p>
        </div>
        <div className="session-hero-stats">
          <div className="session-metric">
            <div className="session-metric-label">Confidence</div>
            <div className="session-metric-value">{sessionConfidence}</div>
          </div>
          <div className="session-metric">
            <div className="session-metric-label">Evidence Sources</div>
            <div className="session-metric-value">{evidenceList.length}</div>
          </div>
          <div className="session-metric">
            <div className="session-metric-label">Open Async Alerts</div>
            <div className="session-metric-value">{prep?.open_async_alerts ?? 0}</div>
          </div>
          <div className="session-metric">
            <div className="session-metric-label">Urgent Alerts</div>
            <div className="session-metric-value">{prep?.urgent_async_alerts ?? 0}</div>
          </div>
        </div>
      </section>

      <div className="page-grid">
        <Card title="Focused Session Prep" className="session-card">
          {prep ? (
            <DataGrid
              items={[
                { label: "History", value: prep.history_summary },
                { label: "Recent Medical Summary", value: prep.recent_medical_summary },
                { label: "Questionnaire Insight", value: prep.latest_questionnaire_insight },
                { label: "Last Consultation", value: prep.last_consultation_insight },
                { label: "Latest Nancy Handoff", value: prep.latest_nancy_handoff },
                { label: "Care Plan", value: prep.care_plan },
              ]}
            />
          ) : (
            <p className="session-copy">No session prep available.</p>
          )}
        </Card>
        <Card title="Analyze Uploaded Audio" className="session-card">
          <div className="upload-zone" style={{ marginBottom: "1.5rem" }}>
            <div className="card-title" style={{ marginBottom: "0.35rem" }}>
              Drop session audio here
            </div>
            <p className="session-copy">
              Upload recorded audio, trigger transcription, and let the clinical engine generate
              analysis artifacts for review.
            </p>
          </div>
          <SessionUploadForm doctorId={doctorId} patientId={patientId} />
          <div style={{ marginTop: "1.5rem" }}>
            <div className="card-title" style={{ marginBottom: "0.75rem" }}>
              Processing Pipeline
            </div>
            <div className="pipeline">
              {[
                ["Transcription", "Speaker-attributed transcript and timing"],
                ["Acoustic affect extraction", "Emotion trajectory mapped across the session"],
                ["Diagnostic synthesis", "Hypothesis and reasoning structured for clinician review"],
                ["Evidence retrieval", "Clinical literature surfaced for supporting context"],
                ["Treatment plan and follow-up", "Suggested next-step prompts and note-ready output"],
              ].map(([title, body]) => (
                <div className="pipeline-step" key={title}>
                  <div className="pipeline-number">✓</div>
                  <div>
                    <div className="pipeline-title">{title}</div>
                    <div className="pipeline-body">{body}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      <div className="g-sidebar" style={{ alignItems: "start" }}>
        <Card title="Latest Session Analysis" className="session-card">
          {latestSession ? (
            <div className="panel-list">
              <div className="panel-list-item">
                <strong>Diagnostic hypothesis</strong>
                <p>
                  {latestSession.hypothesis_name ?? "No hypothesis available."}
                  {latestSession.hypothesis_confidence ? ` (${latestSession.hypothesis_confidence})` : ""}
                </p>
              </div>
              <div className="panel-list-item">
                <strong>Clinical reasoning</strong>
                <p>{latestReasoning}</p>
              </div>
              <div className="panel-list-item">
                <strong>Treatment plan</strong>
                <p>{latestPlan}</p>
              </div>
              <DataGrid
                items={[
                  { label: "Safety gate", value: latestSession.safety_gate ?? "Unknown" },
                  { label: "Valence", value: String(sessionBsv.valence ?? 0) },
                  { label: "Arousal", value: String(sessionBsv.arousal ?? 0) },
                  { label: "Dominance", value: String(sessionBsv.dominance ?? 0) },
                  { label: "Facial valence", value: String(visualBsv.facial_valence ?? 0) },
                  { label: "Blink rate", value: String(visualBsv.blink_rate_per_min ?? 0) },
                ]}
              />
              {evidenceList.length > 0 ? (
                <div>
                  <div className="card-title" style={{ marginTop: "0.5rem", marginBottom: "0.75rem" }}>
                    Retrieved Evidence
                  </div>
                  <div className="session-evidence-list">
                    {evidenceList.map((item: any, index: number) => (
                      <div className="session-evidence-item" key={`evidence-${index}`}>
                        {typeof item === "string" ? item : item.title ?? JSON.stringify(item)}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              {followupList.length > 0 ? (
                <div>
                  <div className="card-title" style={{ marginTop: "0.5rem", marginBottom: "0.75rem" }}>
                    Suggested Follow-up Questions
                  </div>
                  <div className="session-followup-list">
                    {followupList.map((item: any, index: number) => (
                      <div className="session-followup-item" key={`followup-${index}`}>
                        {typeof item === "string" ? item : item.question ?? JSON.stringify(item)}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              {eventMarkers.length > 0 ? (
                <div>
                  <div className="card-title" style={{ marginTop: "0.5rem", marginBottom: "0.75rem" }}>
                    Session Markers
                  </div>
                  <div className="session-evidence-list">
                    {eventMarkers.map((item: any, index: number) => (
                      <div className="session-evidence-item" key={`marker-${index}`}>
                        <strong>{item.label ?? "Marker"}</strong>
                        <div>{item.detail ?? item.response ?? "No marker detail available."}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="session-copy">
              No prior session record is stored yet. Upload audio above to create the first one.
            </p>
          )}
        </Card>

        <Card title="Transcript Surface" className="session-card">
          {transcriptBlocks.length > 0 ? (
            <div className="session-transcript">
              {transcriptBlocks.map((segment: any, index: number) => (
                <div
                  className={`session-transcript-line ${String(segment.speaker ?? "").toLowerCase().includes("doctor") ? "doctor" : "patient"}`}
                  key={segment.id ?? index}
                >
                  <div className="session-transcript-speaker">{segment.speaker ?? "Speaker"}</div>
                  {segment.text ?? segment.content ?? "No transcript text available."}
                </div>
              ))}
            </div>
          ) : (
            <div className="panel-list-item">
              <strong>Transcript excerpt</strong>
              <p>{latestSession?.transcript_excerpt ?? "No transcript excerpt is available yet."}</p>
            </div>
          )}
        </Card>
      </div>

      <div className="page-grid">
        <Card title="Clinician Review Note" className="session-card">
          <SessionNoteForm doctorId={doctorId} patientId={patientId} />
        </Card>
        <Card title="Session Notes Timeline" className="session-card">
          <div className="interaction-history">
            {payload.session_notes.map((entry: any, index: number) => (
              <div className="history-item" key={entry.id ?? index}>
                <div className="history-head">
                  <span className="history-kind">Session note</span>
                  <span className="history-time">{entry.created_at ?? "now"}</span>
                </div>
                <p className="history-body">
                  {entry.summary ?? entry.note ?? entry.body ?? "No note text available."}
                </p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
