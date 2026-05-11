import { api } from "@/lib/api";
import { AppShell, ButtonLink, Card, DataGrid, Hero, StatGrid } from "@/components/ui";
import { doctorForumPath, doctorPatientChartPath, doctorPatientSessionPath } from "@/lib/routes";

export default async function DoctorPage({ params }: { params: Promise<{ doctorId: string }> }) {
  const { doctorId } = await params;
  const payload = await api<any>(`/doctors/${doctorId}`);
  const doctor = payload.doctor;
  const organization = payload.organization;
  const patientCount = payload.patient_count ?? payload.patients_preview?.length ?? 0;
  const patientsPreview = payload.patients_preview ?? [];
  const stats = doctor?.stats ?? {};
  const focusAreas = doctor?.focus_areas ?? [];

  return (
    <AppShell doctorId={doctorId} currentSection="doctor-home">
      <Hero
        eyebrow="Doctor Portfolio"
        title={`${doctor.name}'s clinical operating surface.`}
        body={doctor.mission}
        actions={
          <>
            <ButtonLink href={`/doctor/${doctorId}/workspace`} tone="primary">Enter Workspace</ButtonLink>
            <ButtonLink href={doctorForumPath(doctorId)} tone="secondary">
              MH Forum
            </ButtonLink>
          </>
        }
      />

      <StatGrid
        stats={[
          { label: "Patients", value: patientCount },
          { label: "Watchlist", value: stats.watchlist ?? 0 },
          { label: "Research", value: stats.research ?? 0 },
          { label: "Follow-up", value: stats.followup ?? 0 },
        ]}
      />

      <div className="page-grid">
        <Card title="Clinical Identity">
          <DataGrid
            items={[
              { label: "Organization", value: doctor.organization_id },
              { label: "Organization Name", value: organization.name },
              { label: "Specialty", value: doctor.specialty },
              { label: "Hospital", value: doctor.hospital_name },
              { label: "Department", value: doctor.department_name },
              { label: "Location", value: doctor.location },
              { label: "Experience", value: doctor.years },
            ]}
          />
        </Card>
        <Card title="Focus Areas" tone="teal">
          <div className="bullet-list">
            {focusAreas.map((item: string) => (
              <div className="bullet-row" key={item}><span className="bullet-dot" /><span>{item}</span></div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Patient Portfolio Snapshot" tone="navy">
        <div className="patient-grid">
          {patientsPreview.map((patient: any) => (
            <Card key={patient.id} title={patient.name}>
              <p className="muted">{patient.diagnosis}</p>
              <p className="section-copy">{patient.last_update}</p>
              <div className="inline-actions">
                <ButtonLink href={doctorPatientChartPath(doctorId, patient.id)} tone="primary">Patient Chart</ButtonLink>
                <ButtonLink href={doctorPatientSessionPath(doctorId, patient.id)} tone="warm">Session</ButtonLink>
              </div>
            </Card>
          ))}
          {patientsPreview.length === 0 ? (
            <Card title="No patients yet" tone="amber">
              <p className="section-copy">This doctor profile does not have any patient previews available yet.</p>
            </Card>
          ) : null}
        </div>
      </Card>
    </AppShell>
  );
}
