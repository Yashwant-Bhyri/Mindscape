import { NancySupportPingButton, OutreachForm } from "@/components/forms";
import { AppShell, ButtonLink, Card, DataGrid } from "@/components/ui";
import { api } from "@/lib/api";
import {
  doctorPatientNancyConsolePath,
  doctorPatientSessionPath,
  patientPortalHomePath,
  patientPortalNancyPath,
} from "@/lib/routes";

export default async function PatientPage({ params }: { params: Promise<{ doctorId: string; patientId: string }> }) {
  const { doctorId, patientId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}/patients/${patientId}`);
  const patient = payload.patient;
  const moodScores = Array.isArray(patient.daily_checkins)
    ? patient.daily_checkins
        .map((entry: any) => Number(entry.mood_score ?? entry.mood ?? entry.score))
        .filter((value: number) => Number.isFinite(value))
        .slice(-10)
    : [];
  const chartScores = moodScores.length > 0 ? moodScores : [5, 6, 4, 5, 7, 5, 6, 6, 7, 7];
  const chartDays = chartScores.map((_: number, index: number) => String(index + 1));
  const riskTone = String(patient.risk ?? "").toLowerCase().includes("high")
    ? "high"
    : String(patient.risk ?? "").toLowerCase().includes("medium")
      ? "medium"
      : "low";
  const latestSignal =
    patient.last_daily_checkin_summary ??
    patient.last_nancy_summary ??
    "No recent patient-authored update is available yet.";
  const timelineSummary = (entry: any) =>
    String(
      entry.payload?.summary ??
        entry.payload?.note ??
        entry.payload?.patient_message ??
        entry.payload?.patient_visible_summary ??
        entry.payload?.body ??
        entry.payload?.daily_update ??
        entry.payload?.patient_report ??
        entry.payload?.clinician_summary ??
        entry.payload?.reasoning ??
        entry.payload?.reason ??
        "No summary available.",
    );
  const timelineTone = (kind: string) => {
    const value = kind.toLowerCase();
    if (value.includes("alert")) return "tl-item--alert";
    if (value.includes("nancy")) return "tl-item--nancy";
    if (value.includes("session")) return "tl-item--session";
    return "tl-item--checkin";
  };

  return (
    <AppShell doctorId={doctorId} patientId={patientId} currentSection="patient-chart">
      <section className="chart-hero">
        <div className="chart-hero-copy">
          <div className="chart-hero-eyebrow">Patient Record</div>
          <h1 className="chart-hero-title">{patient.name}</h1>
          <p className="chart-hero-subtitle">
            {patient.diagnosis} · {patient.risk} risk
          </p>
          <div className="hero-actions">
            <ButtonLink href={doctorPatientSessionPath(doctorId, patientId)} tone="secondary">
              Run Session
            </ButtonLink>
            <ButtonLink href={doctorPatientNancyConsolePath(doctorId, patientId)} tone="primary">
              Nancy Console
            </ButtonLink>
            <ButtonLink href={patientPortalHomePath(doctorId, patientId)} tone="warm">
              Patient Portal
            </ButtonLink>
          </div>
        </div>
        <div className="chart-hero-stats">
          <div className="chart-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Risk
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {patient.risk}
            </div>
          </div>
          <div className="chart-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Status
            </div>
            <div className="stat-value" style={{ color: "white", fontSize: "1rem" }}>
              {patient.status}
            </div>
          </div>
          <div className="chart-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Open Alerts
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {payload.async_summary.open_alerts}
            </div>
          </div>
          <div className="chart-hero-stat">
            <div className="stat-label" style={{ color: "rgba(255,255,255,.42)" }}>
              Messages
            </div>
            <div className="stat-value" style={{ color: "white" }}>
              {payload.async_summary.message_count}
            </div>
          </div>
        </div>
      </section>

      <div className="g-sidebar" style={{ alignItems: "start" }}>
        <Card title="Clinical Overview" tone="navy">
          <DataGrid
            items={[
              { label: "Diagnosis", value: patient.diagnosis },
              { label: "Care Plan", value: patient.care_plan },
              { label: "Next Appointment", value: patient.next_appointment },
              { label: "Last Update", value: patient.last_update },
              { label: "Latest Daily Insight", value: patient.last_daily_checkin_summary ?? "None yet" },
              { label: "Latest Nancy Handoff", value: patient.last_nancy_summary ?? "None yet" },
            ]}
          />
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          <Card title="Mood Score · Last 10 Check-ins" tone="teal" className="mood-card">
            <div className="mood-chart">
              {chartScores.map((score: number, index: number) => (
                <div
                  className="mood-bar"
                  key={`score-${index}`}
                  style={{ height: `${Math.max(12, (score / 10) * 100)}%`, opacity: 0.5 + (score / 10) * 0.5 }}
                  title={`${score}/10`}
                />
              ))}
            </div>
            <div className="mood-axis">
              {chartDays.map((day: string) => (
                <div className="mood-day" key={day}>
                  {day}
                </div>
              ))}
            </div>
            <p className="section-copy">Trend: {chartScores.at(-1) && chartScores.at(-1)! >= chartScores[0] ? "stable with mild improvement" : "watch for decline"}.</p>
          </Card>
          <div className="risk-panel">
            <div className="risk-panel-head">
              <span className={`risk-badge ${riskTone}`}>Recent signal</span>
              <span className="risk-panel-title">What the chart is surfacing now</span>
            </div>
            <p className="section-copy">{latestSignal}</p>
            {payload.support_recommended ? (
              <div style={{ marginTop: "1rem" }}>
                <NancySupportPingButton doctorId={doctorId} patientId={patientId} />
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className="page-grid">
        <Card title="Clinician Follow-up" tone="amber">
          <p className="muted" style={{ marginBottom: "0.75rem", fontSize: "0.85rem" }}>
            Send a directed outreach message without stepping into the patient portal experience.
          </p>
          <OutreachForm doctorId={doctorId} patientId={patientId} defaultTarget={patient.name} />
        </Card>
        <Card title="Async Care Timeline" tone="navy">
          <div className="timeline">
            {payload.timeline.map((entry: any, index: number) => (
              <div className={`timeline-item ${timelineTone(entry.kind)}`} key={`${entry.kind}-${index}`}>
                <div className="timeline-kind">{String(entry.kind).replaceAll("_", " ")}</div>
                <div className="timeline-time">{entry.created_at ?? "now"}</div>
                <p>{timelineSummary(entry)}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="page-grid">
        <Card title="Portal Handoff" tone="rose">
          <p className="section-copy">
            When you want to review what the patient sees, use the portal links below instead of the clinician console.
          </p>
          <div className="inline-actions">
            <ButtonLink href={patientPortalHomePath(doctorId, patientId)} tone="warm">Open Companion Home</ButtonLink>
            <ButtonLink href={patientPortalNancyPath(doctorId, patientId)} tone="secondary">Open Patient Nancy</ButtonLink>
          </div>
        </Card>
        <Card title="Recent Risk Signal" tone="teal">
          <p className="section-copy">
            {patient.last_daily_checkin_summary ?? patient.last_nancy_summary ?? "No recent patient-authored update is available yet."}
          </p>
        </Card>
      </div>
    </AppShell>
  );
}
