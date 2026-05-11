import Link from "next/link";

import { NancyVoice } from "@/components/NancyVoice";
import { SpeakWithNancy } from "@/components/NancyTextCompanion";
import { AppShell } from "@/components/ui";
import { api } from "@/lib/api";

export default async function CompanionNancyPage({
  params,
}: {
  params: Promise<{ doctorId: string; patientId: string }>;
}) {
  const { doctorId, patientId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}/patients/${patientId}/nancy`);
  const patient = payload.patient;
  const patientDisplayName = String(patient.name).replace(/\.$/, "");
  const dailyNancyStatus = payload.daily_nancy_status ?? {
    due: false,
    completed_today: false,
    question_count: payload.daily_nancy_questions?.length ?? 0,
    questions: payload.daily_nancy_questions ?? [],
  };
  const supportItems = Array.isArray(payload.patient_nancy_updates) ? payload.patient_nancy_updates : [];
  const tasks = Array.isArray(patient.nancy_tasks) ? patient.nancy_tasks : [];
  const openAlerts = Array.isArray(patient.async_alerts)
    ? patient.async_alerts.filter((item: any) => item.status !== "resolved").length
    : 0;
  const updateTitle = (item: any) => item.title ?? item.category ?? item.kind ?? "Guidance";
  const updateBody = (item: any) =>
    item.body ?? item.message ?? item.summary ?? item.patient_visible_summary ?? "No update available.";
  const taskLabel = (task: any) => task.title ?? task.category ?? task.topic ?? "Active support";
  const taskBody = (task: any) =>
    task.description ?? task.instructions ?? task.summary ?? task.message ?? "Nancy is actively supporting this area.";
  const conversationCount = Array.isArray(patient.nancy_interactions) ? patient.nancy_interactions.length : 0;
  const hasTasks = tasks.length > 0;
  const hasSupport = supportItems.length > 0;

  return (
    <AppShell
      doctorId={doctorId}
      patientId={patientId}
      portalRole="patient"
      currentSection="companion-nancy"
      mainClassName="patient-nancy-mainframe"
    >
      <section className="patient-nancy-shell">
        <div className="patient-nancy-layout">
          <div className="patient-nancy-main">
            <section className="companion-hero patient-nancy-hero-card">
              <div className="companion-hero-copy">
                <div className="companion-hero-eyebrow">Talk to Nancy</div>
                <h1 className="companion-hero-title">Nancy is ready to support you, {patientDisplayName}.</h1>
                <p className="companion-hero-body">
                  Use the live daily voice check-in when Nancy needs structured answers. Use the open
                  conversation for everything else.
                </p>
                <div className="hero-actions">
                  <Link className="patient-portal-btn patient-portal-btn--amber" href="#daily-nancy">
                    Daily Conversation
                  </Link>
                  <Link className="patient-portal-btn patient-portal-btn--ghost-dark" href="#speak-with-nancy">
                    Open Support
                  </Link>
                  <Link className="patient-portal-btn patient-portal-btn--ghost-dark" href="#voice-session">
                    Voice Session
                  </Link>
                </div>
              </div>
              <div className="companion-hero-side">
                <div className="companion-ribbon">
                  <div className="companion-chip"><span>Care plan</span><strong>{patient.care_plan}</strong></div>
                  <div className="companion-chip"><span>Active tasks</span><strong>{tasks.length}</strong></div>
                  <div className="companion-chip"><span>Conversations</span><strong>{conversationCount}</strong></div>
                  <div className="companion-chip"><span>Alerts</span><strong>{openAlerts}</strong></div>
                </div>
                <div className="companion-note">
                  <strong>Nancy stays in the patient lane</strong>
                  <p>This view is for support, check-ins, and relayed guidance only. Doctor-side planning stays separate.</p>
                </div>
              </div>
            </section>

            <section id="daily-nancy" className="patient-nancy-card">
              <div className="panel-kicker amber">Daily Conversation</div>
              <h2 className="panel-title" style={{ color: "white" }}>One meaningful check-in per day</h2>
              <div
                className="panel-list-item"
                style={{ background: "rgba(198,106,26,.08)", borderColor: "rgba(198,106,26,.18)", marginBottom: "0.85rem" }}
              >
                <strong style={{ color: "rgba(255,255,255,.82)" }}>
                  {dailyNancyStatus.due ? "Daily conversation due" : "Daily conversation complete"}
                </strong>
                <p>
                  {dailyNancyStatus.due
                    ? "Start the live voice check-in and Nancy will gather the daily clinical picture naturally."
                    : "Nancy will not repeat the full daily check-in again today, but you can still talk with her."}
                </p>
              </div>
              <p className="section-copy" style={{ marginTop: 0, marginBottom: "0.95rem" }}>
                This is the once-daily live voice flow. Nancy should speak with the patient naturally, gather the needed signal through conversation, and send the summary to the doctor.
              </p>
              <NancyVoice doctorId={doctorId} patientId={patientId} patientName={patient.name} mode="daily" />
            </section>

            <section id="speak-with-nancy" className="patient-nancy-card">
              <div className="panel-kicker teal">Open Support</div>
              <h2 className="panel-title" style={{ color: "white" }}>Everything outside the daily check-in</h2>
              <SpeakWithNancy doctorId={doctorId} patientId={patientId} patientName={patient.name} />
            </section>

            <section id="voice-session" className="patient-nancy-card">
              <div className="panel-kicker teal">Live Voice Session</div>
              <h2 className="panel-title" style={{ color: "white" }}>Speak naturally with Nancy</h2>
              <p className="section-copy">
                Use this for questions, concerns, complaints, logistics, or help outside the once-daily structured conversation.
              </p>
              <NancyVoice doctorId={doctorId} patientId={patientId} patientName={patient.name} mode="support" />
            </section>
          </div>

          <aside className="patient-nancy-sidebar">
            <section className="patient-nancy-card">
              <div className="panel-kicker teal">Care context</div>
              <h2 className="panel-title" style={{ color: "white" }}>
                Today with Nancy
              </h2>
              <div className="panel-list">
                <div className="panel-list-item">
                  <strong>Care plan</strong>
                  <p>{patient.care_plan}</p>
                </div>
                <div className="panel-list-item">
                  <strong>Active directives</strong>
                  <p>{tasks.length} active support tasks loaded into Nancy.</p>
                </div>
                <div className="panel-list-item">
                  <strong>Recent conversations</strong>
                  <p>{patient.nancy_interactions?.length ?? 0} recent Nancy interactions on record.</p>
                </div>
                <div className="panel-list-item">
                  <strong>Open alerts</strong>
                  <p>{openAlerts} open alerts on the clinician side.</p>
                </div>
              </div>
            </section>

            <section className="patient-nancy-card">
              <div className="panel-kicker amber">Active Nancy directives</div>
              <div className="panel-list">
                {hasTasks ? (
                  tasks.map((task: any, index: number) => (
                    <div className="panel-list-item" key={task.id ?? index}>
                      <strong>{taskLabel(task)}</strong>
                      <p>{taskBody(task)}</p>
                    </div>
                  ))
                ) : (
                  <div className="panel-list-item">
                    <strong>No active directives</strong>
                    <p>Nancy is ready for support and will reflect new clinician guidance here when it is added.</p>
                  </div>
                )}
              </div>
            </section>

            <section className="patient-nancy-card">
              <div className="panel-kicker teal">Recent relayed guidance</div>
              <div className="panel-list">
                {hasSupport ? (
                  supportItems.map((item: any, index: number) => (
                    <div className="panel-list-item" key={item.id ?? index}>
                      <strong>{updateTitle(item)}</strong>
                      <p>{updateBody(item)}</p>
                    </div>
                  ))
                ) : (
                  <div className="panel-list-item">
                    <strong>No recent relayed guidance</strong>
                    <p>Nancy relays from your doctor and recent support moments will appear here.</p>
                  </div>
                )}
              </div>
            </section>

            <section className="patient-nancy-card">
              <div className="panel-kicker amber">Good uses for Nancy</div>
              <div className="panel-list">
                <div className="panel-list-item">
                  <strong>Daily state</strong>
                  <p>Mood, sleep, energy, stress, and how your routine has felt since the last check-in.</p>
                </div>
                <div className="panel-list-item">
                  <strong>Medication and side effects</strong>
                  <p>Questions, discomfort, missed doses, or anything you want your doctor to know soon.</p>
                </div>
                <div className="panel-list-item">
                  <strong>Early warning changes</strong>
                  <p>Anything that feels different, concerning, or worth flagging before your next appointment.</p>
                </div>
              </div>
            </section>
          </aside>
        </div>
      </section>
    </AppShell>
  );
}
