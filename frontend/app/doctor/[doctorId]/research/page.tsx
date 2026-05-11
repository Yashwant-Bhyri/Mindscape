import { api } from "@/lib/api";
import { DoctorForum, type ForumThread } from "@/components/DoctorForum";
import { AppShell, BulletList, Card } from "@/components/ui";

type ResearchPayload = {
  doctor?: { name?: string };
  weekly_brief: string[];
  performance: Array<{ label: string; value: string }>;
  forum_threads?: ForumThread[];
};

export default async function ResearchPage({
  params,
  searchParams,
}: {
  params: Promise<{ doctorId: string }>;
  searchParams: Promise<{ thread?: string }>;
}) {
  const { doctorId } = await params;
  const sp = await searchParams;
  const payload = await api<ResearchPayload>(`/doctors/${doctorId}/research`);

  const forumThreads = payload.forum_threads ?? [];

  return (
    <AppShell doctorId={doctorId} currentSection="forum">
      <section className="gateway-hero">
        <div className="gateway-hero-content">
          <div className="gateway-hero-eyebrow">
            Research-grade forum · live headlines · clinician-led threads
          </div>
          <h1 className="gateway-hero-title">
            The mental health intelligence loop society never had.
          </h1>
          <p className="gateway-hero-body">
            Doctor&apos;s Corner is MindScape&apos;s research-minded forum for psychiatrists and mental
            health clinicians: literature, trends, cases, near-misses, methods, screening, and
            hypotheses in one place without generic community noise.
          </p>
        </div>
      </section>

      <div className="page-grid">
        <Card title="Weekly agent brief" tone="teal">
          <BulletList items={payload.weekly_brief} />
        </Card>
        <Card title="Performance signals" tone="amber">
          <div className="bullet-list">
            {payload.performance.map((item) => (
              <div className="bullet-row" key={item.label}>
                <span className="bullet-dot" />
                <span>
                  {item.label}: {item.value}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <DoctorForum
        doctorId={doctorId}
        doctorDisplayName={payload.doctor?.name}
        initialOpenThreadId={typeof sp.thread === "string" ? sp.thread : undefined}
        initial={{
          forum_threads: forumThreads,
          weekly_brief: payload.weekly_brief,
          performance: payload.performance,
        }}
      />
    </AppShell>
  );
}
