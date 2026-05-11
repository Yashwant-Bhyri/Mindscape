import { AlertActionButtons, NancyDirectiveForm, NancySupportPingButton, NancyTouchpointForm, SosForm } from "@/components/forms";
import { NancyClinicianAssistant } from "@/components/NancyClinicianAssistant";
import { AppShell, ButtonLink, BulletList, Card, DataGrid } from "@/components/ui";
import { api } from "@/lib/api";
import { patientPortalNancyPath } from "@/lib/routes";

export default async function NancyPage({ params }: { params: Promise<{ doctorId: string; patientId: string }> }) {
  const { doctorId, patientId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}/patients/${patientId}/nancy`);
  const patient = payload.patient;
  const openAlerts = patient.async_alerts.filter((item: any) => item.status !== "resolved");
  const alertTone = (severity: string) => {
    const value = severity.toLowerCase();
    if (value.includes("urgent") || value.includes("high")) return "urgent";
    if (value.includes("watch") || value.includes("medium")) return "watch";
    return "routine";
  };
  const interactionSummary = (entry: any) =>
    String(
      entry.summary ??
        entry.note ??
        entry.patient_visible_summary ??
        entry.patient_message ??
        entry.clinician_summary ??
        entry.body ??
        "No summary available.",
    );

  return (
    <AppShell doctorId={doctorId} patientId={patientId} currentSection="nancy-console">
      <section className="nancy-console-hero">
        <div className="nancy-console-hero-content">
          <div className="nancy-console-eyebrow">Nancy Clinician Console · {patient.name}</div>
          <h1 className="nancy-console-title">
            Oversee Nancy&apos;s support lane without entering the patient conversation.
          </h1>
          <p className="nancy-console-body">
            Review alerts, issue directives, use the clinician copilot, and save structured
            handoffs. The patient-facing companion view stays separate.
          </p>
          <div className="hero-actions">
            <ButtonLink href={patientPortalNancyPath(doctorId, patientId)} tone="warm">
              Open Patient Nancy View
            </ButtonLink>
          </div>
        </div>
      </section>

      <section className="stat-grid">
        <article className="stat-card">
          <div className="stat-label">Patient</div>
          <div className="stat-value" style={{ fontSize: "1rem" }}>
            {patient.name}
          </div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Directives</div>
          <div className="stat-value">{patient.nancy_tasks.length}</div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Handoffs</div>
          <div className="stat-value">{patient.nancy_interactions.length}</div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Open Alerts</div>
          <div className="stat-value">{openAlerts.length}</div>
        </article>
      </section>

      <div className="page-grid">
        <Card title="Doctor Context" tone="amber">
          <DataGrid
            items={[
              { label: "Patient", value: patient.name },
              { label: "Doctor", value: payload.doctor.name },
              { label: "Hospital", value: payload.doctor.hospital_name },
              { label: "Care Plan", value: patient.care_plan },
              { label: "Latest Daily Insight", value: patient.last_daily_checkin_summary ?? "No questionnaire yet" },
              { label: "Latest Nancy Note", value: patient.last_nancy_summary ?? "No handoffs yet" },
            ]}
          />
          <div style={{ marginTop: "1rem" }}>
            <NancySupportPingButton doctorId={doctorId} patientId={patientId} />
          </div>
        </Card>
        <Card title="Portal Boundary" tone="teal" className="boundary-card">
          <BulletList
            items={[
              "Doctor portal: directives, clinician handoffs, alerts, and Nancy oversight.",
              "Patient portal: self-report, supportive messaging, and live Nancy conversation.",
              "Nancy copilot below listens for doctor intent and can turn it into structured Nancy actions.",
            ]}
          />
        </Card>
      </div>

      <div className="page-grid">
        <Card title="Nancy Clinician Copilot" tone="navy">
          <NancyClinicianAssistant doctorId={doctorId} patientId={patientId} patientName={patient.name} />
        </Card>
        <Card title="Assign Doctor Directives" tone="navy">
          <p className="muted" style={{ marginBottom: "0.75rem", fontSize: "0.85rem" }}>
            These directives are doctor-authored and loaded into patient-side Nancy conversations automatically.
          </p>
          <NancyDirectiveForm doctorId={doctorId} patientId={patientId} />
        </Card>
      </div>

      <div className="page-grid">
        <Card title="Doctor Review Queue" tone="rose">
          {openAlerts.length === 0 ? (
            <p className="muted">No alerts pending review.</p>
          ) : (
            <div className="review-queue">
              {openAlerts.map((alert: any) => (
                <div className={`alert-item ${alertTone(String(alert.severity ?? "routine"))}`} key={alert.id}>
                  <div className="alert-head">
                    <span className={`status-badge ${alertTone(String(alert.severity ?? "routine"))}`}>
                      {alert.severity}
                    </span>
                    <span className="alert-time">{alert.created_at}</span>
                  </div>
                  <div className="alert-title">{alert.title ?? "Clinical review required"}</div>
                  <p className="alert-body">{alert.summary}</p>
                  <AlertActionButtons doctorId={doctorId} alertId={alert.id} />
                </div>
              ))}
            </div>
          )}
        </Card>
        <Card title="SOS Escalation" tone="amber">
          <SosForm doctorId={doctorId} patientId={patientId} />
        </Card>
      </div>

      <div className="page-grid">
        <Card title="Log Manual Touchpoint" tone="teal">
          <NancyTouchpointForm doctorId={doctorId} patientId={patientId} />
        </Card>
        <Card title="Nancy Interaction History" tone="navy">
          <div className="interaction-history">
            {patient.nancy_interactions.map((entry: any, index: number) => (
              <div className="history-item" key={entry.id ?? index}>
                <div className="history-head">
                  <span className="history-kind">
                    {String(entry.kind ?? entry.mode ?? "nancy").replaceAll("_", " ")}
                  </span>
                  <span className="history-time">{entry.created_at ?? "now"}</span>
                </div>
                <p className="history-body">{interactionSummary(entry)}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
