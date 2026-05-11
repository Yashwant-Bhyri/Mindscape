import { DoctorInsightsVoice } from "@/components/DoctorInsightsVoice";
import { IntakeForm, OutreachForm } from "@/components/forms";
import { AppShell, ButtonLink, Card } from "@/components/ui";
import { api } from "@/lib/api";
import {
  doctorForumPath,
  doctorPatientChartPath,
  doctorPatientNancyConsolePath,
  doctorPatientSessionPath,
  patientPortalHomePath,
} from "@/lib/routes";

export default async function WorkspacePage({ params }: { params: Promise<{ doctorId: string }> }) {
  const { doctorId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}/workspace`);
  const doctor = payload.doctor;
  const initialsFor = (name: string) =>
    name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() ?? "")
      .join("");
  const riskTone = (risk?: string) => {
    const value = String(risk ?? "").toLowerCase();
    if (value.includes("high") || value.includes("urgent")) return "high";
    if (value.includes("medium") || value.includes("watch")) return "medium";
    return "low";
  };
  const alertTone = (alert: any) => {
    const level = String(alert.severity ?? alert.level ?? alert.risk_level ?? "").toLowerCase();
    if (level.includes("urgent") || level.includes("high")) return "urgent";
    if (level.includes("watch") || level.includes("medium")) return "watch";
    return "routine";
  };
  const alertLabel = (alert: any) =>
    String(alert.severity ?? alert.level ?? alert.risk_level ?? "routine").replaceAll("_", " ");
  const alertSummary = (alert: any) =>
    String(
      alert.summary ??
        alert.body ??
        alert.reason ??
        alert.patient_message ??
        alert.note ??
        "No details available.",
    );
  const watchtowerSummary = (entry: any) =>
    String(
      entry.clinician_summary ??
        entry.summary ??
        entry.patient_visible_summary ??
        entry.note ??
        entry.body ??
        "No summary available.",
    );
  const watchtowerTags = (entry: any) =>
    [
      entry.topic,
      entry.category,
      entry.signal,
      entry.safety_flag ? "Safety flagged" : null,
      entry.safe === true ? "No safety concerns" : null,
    ].filter(Boolean) as string[];

  return (
    <AppShell doctorId={doctorId} currentSection="workspace">
      <section className="workspace-hero">
        <div className="workspace-hero-copy">
          <div className="workspace-hero-eyebrow">
            Doctor Command Centre · {doctor?.name ?? "Clinician workspace"}
          </div>
          <h1 className="workspace-hero-title">
            The daily operating surface for your clinical panel.
          </h1>
          <p className="workspace-hero-body">
            Patients, async alerts, intake, outreach, and Nancy handoffs all route here. The
            workspace stays clinician-first while preserving the real backend contract.
          </p>
          <div className="hero-actions">
            <ButtonLink href={doctorForumPath(doctorId)} tone="secondary">
              Open Doctor&apos;s Corner
            </ButtonLink>
            <ButtonLink href={`/doctor/${doctorId}/research`} tone="primary">
              Brief and signals
            </ButtonLink>
          </div>
        </div>
        <div className="workspace-hero-stats">
          <div className="workspace-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Panel
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {payload.patients.length}
            </div>
          </div>
          <div className="workspace-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Open Alerts
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {payload.open_alerts.length}
            </div>
          </div>
          <div className="workspace-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Urgent
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {payload.urgent_alerts.length}
            </div>
          </div>
          <div className="workspace-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Handoffs
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {payload.nancy_watchtower.length}
            </div>
          </div>
        </div>
      </section>

      <section className="workspace-voice-panel">
        <div className="card-title">Doctor Insights Voice</div>
        <p className="section-copy" style={{ marginBottom: "1rem" }}>
          Ask your MindScape snapshot out loud, then pivot directly into patient review, Nancy
          oversight, or a session workflow.
        </p>
        <DoctorInsightsVoice doctorId={doctorId} doctorDisplayName={doctor?.name} />
      </section>

      <div className="g-wide" style={{ alignItems: "start" }}>
        <Card title="Patient Dashboard" tone="navy">
          <div className="workspace-panel-list">
            {payload.patients.map((patient: any) => (
              <div className="workspace-patient-row" key={patient.id}>
                <div className="workspace-patient-avatar">{initialsFor(patient.name)}</div>
                <div className="workspace-patient-info">
                  <div className="workspace-patient-name">{patient.name}</div>
                  <div className="workspace-patient-dx">{patient.diagnosis}</div>
                  <div className="workspace-patient-update">{patient.last_update}</div>
                </div>
                <div className="workspace-patient-chips">
                  <span className={`risk-badge ${riskTone(patient.risk_level ?? patient.risk)}`}>
                    {patient.risk_level ?? patient.risk ?? "Low risk"}
                  </span>
                  {patient.alert_count ? (
                    <span className="status-badge alert">{patient.alert_count} alerts</span>
                  ) : null}
                </div>
                <div className="doctor-card-actions">
                  <ButtonLink href={doctorPatientChartPath(doctorId, patient.id)} tone="primary">
                    Chart
                  </ButtonLink>
                  <ButtonLink
                    href={doctorPatientNancyConsolePath(doctorId, patient.id)}
                    tone="secondary"
                  >
                    Nancy
                  </ButtonLink>
                  <ButtonLink href={doctorPatientSessionPath(doctorId, patient.id)} tone="warm">
                    Session
                  </ButtonLink>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <div className="workspace-panel-list">
          <Card title="Async Review Inbox" tone="rose">
            <div className="alert-list">
              {payload.open_alerts.map((alert: any, index: number) => (
                <div className={`alert-item ${alertTone(alert)}`} key={alert.id ?? index}>
                  <div className="alert-head">
                    <span className={`status-badge ${alertTone(alert)}`}>
                      {alertLabel(alert)}
                    </span>
                    <span className="alert-time">{alert.created_at ?? "now"}</span>
                  </div>
                  <div className="alert-title">
                    {alert.title ?? alert.reason ?? "Clinical review required"}
                  </div>
                  <p className="alert-body">{alertSummary(alert)}</p>
                  <p className="alert-time" style={{ marginTop: "0.55rem" }}>
                    {alert.patient_name ?? alert.patient_id ?? "Patient"}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Nancy Watchtower" tone="teal">
            <div className="watchtower-list">
              {payload.nancy_watchtower.map((entry: any, index: number) => (
                <div className="watchtower-item" key={entry.id ?? index}>
                  <div className="history-head">
                    <div className="watchtower-title">
                      {entry.patient_name ?? entry.patient_id ?? "Patient handoff"}
                    </div>
                    <span className="history-time">{entry.created_at ?? "now"}</span>
                  </div>
                  <p className="watchtower-body">{watchtowerSummary(entry)}</p>
                  {watchtowerTags(entry).length > 0 ? (
                    <div className="hero-actions" style={{ marginTop: "0.65rem" }}>
                      {watchtowerTags(entry).map((tag) => (
                        <span className="status-badge ok" key={tag}>
                          {tag}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <div className="page-grid">
        <Card title="New Patient Intake" tone="amber">
          <IntakeForm doctorId={doctorId} />
        </Card>
        <Card title="Care Coordination Outreach" tone="navy">
          <OutreachForm doctorId={doctorId} />
        </Card>
      </div>
    </AppShell>
  );
}
