import Link from "next/link";

import { AppShell } from "@/components/ui";
import { api } from "@/lib/api";
import { patientPortalHomePath, patientPortalNancyPath } from "@/lib/routes";

export default async function PatientGatewayPage() {
  const payload = await api<{
    doctors: Array<{
      id: string;
      name: string;
      specialty: string;
      hospital_name: string;
      patients: Array<{
        id: string;
        name: string;
        diagnosis: string;
        care_plan: string;
        risk: string;
        next_appointment: string;
        daily_nancy_due: boolean;
      }>;
    }>;
  }>("/patient-access");

  const patientCount = payload.doctors.reduce((total, doctor) => total + doctor.patients.length, 0);
  const dueCount = payload.doctors.reduce(
    (total, doctor) => total + doctor.patients.filter((patient) => patient.daily_nancy_due).length,
    0,
  );
  const initialsFor = (name: string) =>
    name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() ?? "")
      .join("");

  return (
    <AppShell portalRole="public" currentSection="patient-gateway">
      <section className="patient-gateway-hero">
        <div className="patient-gateway-copy">
          <div className="patient-gateway-eyebrow">Patient Portal · Companion Access</div>
          <h1 className="patient-gateway-title">Enter the patient side of MindScape.</h1>
          <p className="patient-gateway-body">
            This entrance is separate from the doctor workflow. Use it for daily updates, companion
            messaging, and talking to Nancy without clinician tools mixed into the experience.
          </p>
          <div className="hero-actions">
            <Link className="patient-portal-btn patient-portal-btn--amber" href="#patient-portals">
              Choose A Portal
            </Link>
          </div>
        </div>
        <div className="patient-gateway-side">
          <div className="patient-gateway-ribbon">
            <div className="patient-gateway-chip"><span>Patient portals</span><strong>{patientCount}</strong></div>
            <div className="patient-gateway-chip"><span>Care teams</span><strong>{payload.doctors.length}</strong></div>
            <div className="patient-gateway-chip"><span>Daily check-ins due</span><strong>{dueCount}</strong></div>
            <div className="patient-gateway-chip"><span>Mode</span><strong>Companion</strong></div>
          </div>
          <div className="patient-gateway-note">
            <strong>Separate from the doctor workflow</strong>
            <p>
              This side is for check-ins, messages, and support. Clinician tools and chart controls stay on the doctor side.
            </p>
          </div>
        </div>
      </section>

      <section className="patient-access-grid" id="patient-portals">
        {payload.doctors.map((doctor) => (
          <article className="patient-access-card-v2" key={doctor.id}>
            <div className="patient-access-card-head">
              <div className="patient-access-avatar">{initialsFor(doctor.name)}</div>
              <div>
                <div className="patient-access-card-title">{doctor.name} Care Panel</div>
                <div className="patient-access-card-subtitle">
                  {doctor.specialty} · {doctor.hospital_name}
                </div>
              </div>
            </div>

            <div className="patient-access-stack">
              {doctor.patients.map((patient) => (
                <article className="patient-access-entry-v2" key={patient.id}>
                  <div className="patient-access-entry-top">
                    <div>
                      <div className="patient-access-patient-name">{patient.name}</div>
                      <div className="patient-access-patient-dx">{patient.diagnosis}</div>
                    </div>
                    <span className={`patient-access-status ${patient.daily_nancy_due ? "patient-access-status--due" : ""}`}>
                      {patient.daily_nancy_due ? "Due today" : "Complete today"}
                    </span>
                  </div>

                  <div className="patient-access-plan">{patient.care_plan}</div>

                  <div className="patient-access-meta">
                    <span>{patient.risk} risk</span>
                    <span>Next appointment: {patient.next_appointment}</span>
                  </div>

                  <div className="patient-access-actions">
                    <Link className="patient-portal-btn patient-portal-btn--amber" href={patientPortalHomePath(doctor.id, patient.id)}>
                      Open Patient Portal
                    </Link>
                    <Link className="patient-portal-btn patient-portal-btn--teal" href={patientPortalNancyPath(doctor.id, patient.id)}>
                      Talk To Nancy
                    </Link>
                  </div>

                  <div className={`patient-access-status-note ${patient.daily_nancy_due ? "patient-access-status-note--due" : ""}`}>
                    <span>{patient.daily_nancy_due ? "Nancy check-in due today" : "Nancy check-in complete today"}</span>
                  </div>
                </article>
              ))}
            </div>
          </article>
        ))}
      </section>
    </AppShell>
  );
}
