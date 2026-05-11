import { AppShell, ButtonLink } from "@/components/ui";
import { api } from "@/lib/api";
import { doctorForumPath, doctorHomePath, doctorWorkspacePath } from "@/lib/routes";

export default async function DoctorGatewayPage() {
  const payload = await api<{
    default_doctor_id: string;
    doctors: Array<{
      id: string;
      name: string;
      title: string;
      specialty: string;
      hospital_name: string;
      patient_count: number;
      patients_preview: Array<{
        id: string;
        name: string;
        diagnosis: string;
        risk: string;
      }>;
    }>;
  }>("/doctors");
  const totalPatients = payload.doctors.reduce((sum, doctor) => sum + doctor.patient_count, 0);
  const initialsFor = (name: string) =>
    name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() ?? "")
      .join("");
  const riskTone = (risk: string) => {
    const value = risk.toLowerCase();
    if (value.includes("high") || value.includes("urgent")) return "high";
    if (value.includes("medium") || value.includes("watch")) return "medium";
    return "low";
  };

  return (
    <AppShell portalRole="public" currentSection="doctor-gateway">
      <section className="gateway-hero">
        <div className="gateway-hero-content">
          <div className="gateway-hero-eyebrow">Doctor Portal · Clinician Operations</div>
          <h1 className="gateway-hero-title">Enter the clinician side of MindScape.</h1>
          <p className="gateway-hero-body">
            Workspace, patient charts, Nancy oversight, session intelligence, and Doctor&apos;s
            Corner all begin here. Choose a clinician workspace below.
          </p>
        </div>
      </section>

      <section className="stat-grid">
        <article className="stat-card">
          <div className="stat-label">Doctor Workspaces</div>
          <div className="stat-value">{payload.doctors.length}</div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Active Patients</div>
          <div className="stat-value">{totalPatients}</div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Default Workspace</div>
          <div className="stat-value" style={{ fontSize: "1rem" }}>
            {payload.default_doctor_id}
          </div>
        </article>
        <article className="stat-card">
          <div className="stat-label">Mode</div>
          <div className="stat-value" style={{ fontSize: "1rem" }}>
            Operations
          </div>
        </article>
      </section>

      <section className="doctor-grid">
        {payload.doctors.map((doctor) => (
          <article className="doctor-card" key={doctor.id}>
            <div className="doctor-card-head">
              <div className="doctor-card-avatar">{initialsFor(doctor.name)}</div>
              <div>
                <div className="doctor-card-name">{doctor.name}</div>
                <div className="doctor-card-title">
                  {doctor.title} · {doctor.hospital_name}
                </div>
              </div>
            </div>

            <div className="doctor-card-meta">
              <div className="doctor-meta-item">
                <div className="doctor-meta-label">Specialty</div>
                <div className="doctor-meta-value">{doctor.specialty}</div>
              </div>
              <div className="doctor-meta-item">
                <div className="doctor-meta-label">Patients</div>
                <div className="doctor-meta-value">{doctor.patient_count} active</div>
              </div>
              <div className="doctor-meta-item">
                <div className="doctor-meta-label">Portal</div>
                <div className="doctor-meta-value">Clinician</div>
              </div>
              <div className="doctor-meta-item">
                <div className="doctor-meta-label">MindScape Mode</div>
                <div className="doctor-meta-value">
                  {doctor.id === payload.default_doctor_id ? "Default workspace" : "Operations"}
                </div>
              </div>
            </div>

            <div className="doctor-patient-list">
              {doctor.patients_preview.map((patient) => (
                <div className="doctor-patient-entry" key={patient.id}>
                  <div className="doctor-patient-avatar">{initialsFor(patient.name)}</div>
                  <span className="doctor-patient-name">{patient.name}</span>
                  <span className="doctor-patient-dx">{patient.diagnosis}</span>
                  <span className={`risk-badge ${riskTone(patient.risk)}`}>{patient.risk}</span>
                </div>
              ))}
            </div>

            <div className="doctor-card-actions">
              <ButtonLink href={doctorWorkspacePath(doctor.id)} tone="primary">
                Open Workspace
              </ButtonLink>
              <ButtonLink href={doctorForumPath(doctor.id)} tone="secondary">
                Doctor&apos;s Corner
              </ButtonLink>
              <ButtonLink href={doctorHomePath(doctor.id)} tone="warm">
                Doctor Home
              </ButtonLink>
            </div>
          </article>
        ))}
      </section>
    </AppShell>
  );
}
