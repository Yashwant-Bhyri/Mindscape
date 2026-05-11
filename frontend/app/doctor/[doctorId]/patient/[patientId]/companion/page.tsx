import Link from "next/link";

import { PatientPortalMessageComposer } from "@/components/patient-portal-client";
import { AppShell } from "@/components/ui";
import { api } from "@/lib/api";
import { patientPortalNancyPath } from "@/lib/routes";

export default async function CompanionPage({ params }: { params: Promise<{ doctorId: string; patientId: string }> }) {
  const { doctorId, patientId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}/patients/${patientId}/companion`);
  const patient = payload.patient;
  const dailyStatus = payload.daily_nancy_status;
  const patientDisplayName = String(patient.name).replace(/\.$/, "");
  const supportItems = Array.isArray(payload.patient_nancy_updates) ? payload.patient_nancy_updates : [];
  const inboxItems = Array.isArray(payload.patient_inbox) ? payload.patient_inbox : [];
  const tasks = Array.isArray(patient.nancy_tasks) ? patient.nancy_tasks : [];
  const taskLabel = (task: any) => task.title ?? task.category ?? task.topic ?? "Active support";
  const taskBody = (task: any) =>
    task.description ?? task.instructions ?? task.summary ?? task.message ?? "Nancy is actively supporting this area.";
  const inboxTone = (item: any) => {
    const source = String(item.kind ?? item.source ?? item.role ?? "").toLowerCase();
    if (source.includes("doctor") || source.includes("care")) return "doctor";
    if (source.includes("patient") || source.includes("self")) return "self";
    return "nancy";
  };
  const inboxSender = (item: any) => item.sender ?? item.role ?? item.kind ?? "Nancy";
  const inboxBody = (item: any) =>
    item.body ?? item.message ?? item.summary ?? item.patient_visible_summary ?? "No message body available.";
  const updateTitle = (item: any) => item.title ?? item.category ?? item.kind ?? "General support";
  const updateBody = (item: any) =>
    item.body ?? item.message ?? item.summary ?? item.patient_visible_summary ?? "No update available.";
  const hasTasks = tasks.length > 0;
  const hasInbox = inboxItems.length > 0;
  const hasSupport = supportItems.length > 0;

  return (
    <AppShell doctorId={doctorId} patientId={patientId} portalRole="patient" currentSection="companion-home">
      <section className="companion-hero">
        <div className="companion-hero-copy">
          <div className="companion-hero-eyebrow">Companion Home</div>
          <h1 className="companion-hero-title">
            A calmer between-session space for {patientDisplayName}.
          </h1>
          <p className="companion-hero-body">
            This portal is patient-first. Send updates, complete a daily check-in, and reach Nancy
            without doctor workflow tools in the way.
          </p>
          <div className="hero-actions">
            <Link className="patient-portal-btn patient-portal-btn--ghost-dark" href={patientPortalNancyPath(doctorId, patientId)}>
              Talk to Nancy
            </Link>
            <Link className="patient-portal-btn patient-portal-btn--amber" href="#messages">
              Open Inbox
            </Link>
          </div>
        </div>
        <div className="companion-hero-side">
          <div className="companion-ribbon">
            <div className="companion-chip">
              <span>Risk lane</span>
              <strong>{payload.async_summary.risk_level}</strong>
            </div>
            <div className="companion-chip">
              <span>Review needed</span>
              <strong>{String(payload.async_summary.review_needed)}</strong>
            </div>
            <div className="companion-chip">
              <span>Active tasks</span>
              <strong>{payload.async_summary.task_count}</strong>
            </div>
            <div className="companion-chip">
              <span>Open alerts</span>
              <strong>{payload.async_summary.open_alerts}</strong>
            </div>
          </div>
          <div className="companion-note">
            <strong>Patient-safe by default</strong>
            <p>Nancy relays only patient-facing guidance here. Clinician notes stay on the doctor side.</p>
          </div>
        </div>
      </section>

      {dailyStatus?.due ? (
        <section className="notice-banner">
          <div>
            <div className="notice-kicker">Nancy needs today&apos;s check-in</div>
            <h2 className="notice-title">Answer the daily questions once, conversationally.</h2>
            <p className="section-copy">
              Nancy will ask the core daily set plus any doctor-specific follow-up, then send your care team a clear summary.
            </p>
          </div>
          <Link className="patient-portal-btn patient-portal-btn--amber" href={`${patientPortalNancyPath(doctorId, patientId)}#daily-nancy`}>
            Start With Nancy
          </Link>
        </section>
      ) : (
        <section className="notice-banner">
          <div>
            <div className="notice-kicker">Today&apos;s check-in is done</div>
            <h2 className="notice-title">Nancy will not run the full questionnaire again today.</h2>
            <p className="section-copy">You can still message Nancy or your doctor if something changes.</p>
          </div>
          <Link className="patient-portal-btn patient-portal-btn--ghost-dark" href={patientPortalNancyPath(doctorId, patientId)}>
            Message Nancy
          </Link>
        </section>
      )}

      <div className="g-sidebar" style={{ alignItems: "start" }}>
        <section className="context-panel">
          <div className="panel-kicker teal">Care Snapshot</div>
          <h2 className="panel-title">Your care context</h2>
          <div className="context-rows">
            <div className="context-row"><span>Patient</span><strong>{patient.name}</strong></div>
            <div className="context-row"><span>Supervising doctor</span><strong>{payload.doctor.name}</strong></div>
            <div className="context-row"><span>Care organisation</span><strong>{payload.doctor.hospital_name}</strong></div>
            <div className="context-row"><span>Diagnosis</span><strong>{patient.diagnosis}</strong></div>
            <div className="context-row"><span>Care plan</span><strong>{patient.care_plan}</strong></div>
            <div className="context-row"><span>Latest Nancy handoff</span><strong>{payload.async_summary.last_handoff}</strong></div>
          </div>
        </section>
        <section className="task-panel">
          <div className="panel-kicker amber">Today with Nancy</div>
          <h2 className="panel-title">What Nancy is actively supporting</h2>
          <div className="task-stack">
            {hasTasks ? (
              tasks.map((task: any, index: number) => (
                <div className="task-item-card" key={task.id ?? index}>
                  <div className="task-item-top">
                    <span className="task-item-cat">{task.category ?? "Support"}</span>
                    <span className="status-badge ok">Active</span>
                  </div>
                  <div className="task-item-title">{taskLabel(task)}</div>
                  <p className="task-item-body">{taskBody(task)}</p>
                </div>
              ))
            ) : (
              <div className="patient-empty-state">
                <strong>No active Nancy tasks right now.</strong>
                <p>Nancy is available for support and will surface new guidance here when your care plan changes.</p>
              </div>
            )}
          </div>
        </section>
      </div>

      <div id="messages" className="g-sidebar-l" style={{ alignItems: "start" }}>
        <section className="message-composer-card">
          <div className="panel-kicker teal">Send an Update</div>
          <h2 className="panel-title">Message Nancy or your doctor</h2>
          <p className="section-copy">
            Use this for everyday updates, support requests, or quick notes. Nancy handoffs sent by your doctor will appear separately in the inbox thread beside this form.
          </p>
          <PatientPortalMessageComposer doctorId={doctorId} patientId={patientId} />
        </section>

        <section className="inbox-panel">
          <div className="panel-kicker muted">Inbox</div>
          <h2 className="panel-title">Messages from Nancy and your care team</h2>
          <div className="message-thread">
            {hasInbox ? (
              inboxItems.map((item: any, index: number) => (
                <div className={`message-thread-item ${inboxTone(item)}`} key={item.id ?? index}>
                  <div className="message-thread-meta">
                    <span className="task-item-title" style={{ margin: 0 }}>{inboxSender(item)}</span>
                    <span className="history-time">{item.created_at ?? item.timestamp ?? "now"}</span>
                  </div>
                  <p className="message-thread-body">{inboxBody(item)}</p>
                </div>
              ))
            ) : (
              <div className="patient-empty-state">
                <strong>Your inbox is clear.</strong>
                <p>Messages from Nancy and your care team will appear here as soon as they are sent.</p>
              </div>
            )}
          </div>
        </section>
      </div>

      <div id="daily-checkin" className="g-sidebar" style={{ alignItems: "start" }}>
        <section className="checkin-panel">
          <div className="panel-kicker amber">Daily Check-in</div>
          <h2 className="panel-title">Complete today&apos;s live Nancy check-in</h2>
          <p className="section-copy">
            The daily check-in runs as a live voice conversation with Nancy, not a text form. She gathers the signal naturally and sends the structured summary to your doctor.
          </p>
          <div className="patient-empty-state patient-live-checkin-card">
            <strong>{dailyStatus?.due ? "Nancy is ready for today&apos;s voice check-in." : "Today&apos;s voice check-in is already complete."}</strong>
            <p>
              {dailyStatus?.due
                ? "Open the Daily Conversation and speak naturally with Nancy."
                : "You can still open Nancy for support, but she will not repeat the full daily flow again today."}
            </p>
            <div className="patient-access-actions" style={{ marginTop: "1rem" }}>
              <Link
                className="patient-portal-btn patient-portal-btn--amber"
                href={`${patientPortalNancyPath(doctorId, patientId)}#daily-nancy`}
              >
                Open Daily Voice Check-in
              </Link>
            </div>
          </div>
        </section>

        <section className="support-panel">
          <div className="panel-kicker teal">Nancy Relay</div>
          <h2 className="panel-title">Recent Nancy guidance</h2>
          <div className="guidance-stack">
            {hasSupport ? (
              supportItems.map((item: any, index: number) => (
                <div className="guidance-item-card" key={item.id ?? index}>
                  <div className="support-card-meta">
                    <span className="support-card-cat">{updateTitle(item)}</span>
                    <span className="history-time">{item.created_at ?? item.timestamp ?? "now"}</span>
                  </div>
                  <p className="support-card-body">{updateBody(item)}</p>
                </div>
              ))
            ) : (
              <div className="patient-empty-state">
                <strong>No recent Nancy relay yet.</strong>
                <p>Nancy guidance will appear here after check-ins, support conversations, or doctor relays.</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </AppShell>
  );
}
