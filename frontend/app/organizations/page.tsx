import { api } from "@/lib/api";
import { AppShell, ButtonLink, Card, Hero } from "@/components/ui";

export default async function OrganizationsPage() {
  const payload = await api<{ organizations: Array<{ id: string; name: string; focus: string; signature_program: string; researchers: string[] }> }>("/organizations");

  return (
    <AppShell>
      <Hero
        eyebrow="Organizations"
        title="Clinical groups and research-driven care teams."
        body="This page is the portfolio layer above the doctors. It shows the multi-doctor structure the old Mesop shell struggled to represent cleanly."
      />

      <div className="patient-grid">
        {payload.organizations.map((organization) => (
          <Card key={organization.id} title={organization.name} tone="amber">
            <p className="section-copy">{organization.focus}</p>
            <p className="muted">{organization.researchers.length} clinicians in this organization.</p>
            <ButtonLink href={`/doctor/${organization.researchers[0]}`} tone="primary">Open Lead Doctor</ButtonLink>
          </Card>
        ))}
      </div>
    </AppShell>
  );
}

