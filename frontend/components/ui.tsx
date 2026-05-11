import Link from "next/link";
import { HTMLAttributes, ReactNode } from "react";

import {
  doctorGatewayPath,
  doctorHomePath,
  doctorPatientChartPath,
  doctorPatientNancyConsolePath,
  doctorPatientSessionPath,
  doctorForumPath,
  doctorWorkspacePath,
  patientGatewayPath,
  patientPortalHomePath,
  patientPortalNancyPath,
} from "@/lib/routes";

type NavProps = {
  doctorId?: string;
  patientId?: string;
  portalRole?: "doctor" | "patient" | "public";
  currentSection?: string;
  mainClassName?: string;
};

export function AppShell({
  children,
  doctorId,
  patientId,
  portalRole,
  currentSection,
  mainClassName,
}: { children: ReactNode } & NavProps) {
  const resolvedPortalRole: "doctor" | "patient" | "public" =
    portalRole ?? (patientId ? "patient" : doctorId ? "doctor" : "public");
  const publicItems: Array<{ key: string; label: string; href?: string; disabled?: boolean; hint?: string }> = [
    { key: "home", label: "Platform", href: "/" },
    { key: "doctor-gateway", label: "Doctor Portal", href: doctorGatewayPath() },
    { key: "patient-gateway", label: "Patient Portal", href: patientGatewayPath() },
    { key: "organizations", label: "Organizations", href: "/organizations" },
  ];
  const doctorItems: Array<{ key: string; label: string; href?: string; disabled?: boolean; hint?: string }> = [
    { key: "platform", label: "Platform", href: "/" },
    { key: "doctor-gateway", label: "Doctor Portal", href: doctorGatewayPath() },
    { key: "organizations", label: "Organizations", href: "/organizations" },
    { key: "doctor-home", label: "Doctor Home", href: doctorId ? doctorHomePath(doctorId) : "/", disabled: !doctorId },
    { key: "workspace", label: "Workspace", href: doctorId ? doctorWorkspacePath(doctorId) : "/", disabled: !doctorId },
    {
      key: "patient-chart",
      label: "Patient Chart",
      href: doctorId && patientId ? doctorPatientChartPath(doctorId, patientId) : undefined,
      disabled: !(doctorId && patientId),
      hint: "Open a patient from Workspace first",
    },
    {
      key: "nancy-console",
      label: "Clinician Nancy",
      href: doctorId && patientId ? doctorPatientNancyConsolePath(doctorId, patientId) : undefined,
      disabled: !(doctorId && patientId),
      hint: "Open a patient from Workspace first",
    },
    {
      key: "session",
      label: "Session",
      href: doctorId && patientId ? doctorPatientSessionPath(doctorId, patientId) : undefined,
      disabled: !(doctorId && patientId),
      hint: "Open a patient from Workspace first",
    },
    {
      key: "forum",
      label: "Forum",
      href: doctorId ? doctorForumPath(doctorId) : "/",
      disabled: !doctorId,
      hint: "Doctor's Corner — MH research & peer exchange",
    },
  ];
  const patientItems: Array<{ key: string; label: string; href?: string; disabled?: boolean; hint?: string }> = [
    {
      key: "patient-gateway",
      label: "Patient Portal",
      href: patientGatewayPath(),
    },
    {
      key: "companion-home",
      label: "Companion Home",
      href: doctorId && patientId ? patientPortalHomePath(doctorId, patientId) : undefined,
      disabled: !(doctorId && patientId),
    },
    {
      key: "messages",
      label: "Messages",
      href: doctorId && patientId ? `${patientPortalHomePath(doctorId, patientId)}#messages` : undefined,
      disabled: !(doctorId && patientId),
    },
    {
      key: "daily-checkin",
      label: "Daily Check-in",
      href: doctorId && patientId ? `${patientPortalHomePath(doctorId, patientId)}#daily-checkin` : undefined,
      disabled: !(doctorId && patientId),
    },
    {
      key: "companion-nancy",
      label: "Patient Nancy",
      href: doctorId && patientId ? patientPortalNancyPath(doctorId, patientId) : undefined,
      disabled: !(doctorId && patientId),
    },
  ];
  const items =
    resolvedPortalRole === "patient"
      ? patientItems
      : resolvedPortalRole === "doctor"
        ? doctorItems
        : publicItems;
  const roleBadge =
    resolvedPortalRole === "patient"
      ? { className: "role-badge role-badge--pat", label: "Patient" }
      : resolvedPortalRole === "doctor"
        ? { className: "role-badge role-badge--doc", label: "Clinician" }
        : null;

  return (
    <div className={`app-shell app-shell--${resolvedPortalRole}`}>
      <header className={`topbar topbar--${resolvedPortalRole}`}>
        <div className="brand-block brand">
          <div className="brand-mark">M</div>
          <div>
            <div className="brand-title brand-name">MindScape</div>
            <div className="brand-subtitle brand-sub">
              {resolvedPortalRole === "patient"
                ? "Patient companion portal"
                : resolvedPortalRole === "doctor"
                  ? "Clinician operations"
                  : "Mental health clinical operating system"}
            </div>
          </div>
        </div>
        <nav className="nav-strip nav">
          {items.map((item) =>
            item.disabled || !item.href ? (
              <span
                className="nav-pill nav-pill--dim"
                key={item.label}
                title={item.hint ?? "Not available"}
              >
                {item.label}
              </span>
            ) : (
              <Link
                className={`nav-pill ${item.key === currentSection ? "nav-pill--active" : ""}`}
                key={item.label}
                href={item.href}
                title={item.hint}
              >
                {item.label}
              </Link>
            )
          )}
        </nav>
        {roleBadge ? (
          <div className="topbar-end">
            <span className={roleBadge.className}>{roleBadge.label}</span>
          </div>
        ) : null}
      </header>
      <main className={`page-frame ${mainClassName ?? ""}`.trim()}>{children}</main>
    </div>
  );
}

export function Hero({
  eyebrow,
  title,
  body,
  actions,
}: {
  eyebrow: string;
  title: string;
  body: string;
  actions?: ReactNode;
}) {
  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="hero-body">{body}</p>
      </div>
      {actions ? <div className="hero-actions">{actions}</div> : null}
    </section>
  );
}

export function ButtonLink({
  href,
  children,
  tone = "primary",
}: {
  href: string;
  children: ReactNode;
  tone?: "primary" | "secondary" | "warm";
}) {
  return (
    <Link className={`button-link ${tone}`} href={href}>
      {children}
    </Link>
  );
}

export function Card({
  title,
  children,
  tone = "default",
  className = "",
  ...props
}: {
  title?: string;
  children: ReactNode;
  tone?: "default" | "teal" | "navy" | "amber" | "rose";
} & HTMLAttributes<HTMLElement>) {
  return (
    <section {...props} className={`card tone-${tone} ${className}`.trim()}>
      {title ? <h2 className="card-title">{title}</h2> : null}
      {children}
    </section>
  );
}

export function StatGrid({ stats }: { stats: Array<{ label: string; value: string | number }> }) {
  return (
    <section className="stat-grid">
      {stats.map((stat) => (
        <article className="stat-card" key={stat.label}>
          <div className="stat-label">{stat.label}</div>
          <div className="stat-value">{stat.value}</div>
        </article>
      ))}
    </section>
  );
}

export function BulletList({ items }: { items: string[] }) {
  return (
    <div className="bullet-list">
      {items.map((item) => (
        <div className="bullet-row" key={item}>
          <span className="bullet-dot" />
          <span>{item}</span>
        </div>
      ))}
    </div>
  );
}

export function DataGrid({ items }: { items: Array<{ label: string; value: ReactNode }> }) {
  return (
    <div className="data-grid">
      {items.map((item) => (
        <div className="data-card" key={item.label}>
          <div className="data-label">{item.label}</div>
          <div className="data-value">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

export function Timeline({ items }: { items: Array<{ kind: string; created_at?: string; payload?: Record<string, unknown> }> }) {
  return (
    <div className="timeline">
      {items.map((item, index) => {
        const payload = item.payload ?? {};
        const summary =
          String(
            payload.summary ??
              payload.note ??
              payload.patient_message ??
              payload.patient_visible_summary ??
              payload.body ??
              payload.daily_update ??
              payload.patient_report ??
              payload.clinician_summary ??
              payload.reasoning ??
              payload.reason ??
              "",
          ) || "No summary available.";

        return (
          <article className="timeline-item" key={`${item.kind}-${item.created_at ?? index}`}>
            <div className="timeline-kind">{item.kind.replaceAll("_", " ")}</div>
            <div className="timeline-time">{item.created_at ?? "now"}</div>
            <p>{summary}</p>
          </article>
        );
      })}
    </div>
  );
}

export function JsonPreview({ title, value }: { title: string; value: string }) {
  return (
    <Card title={title} tone="navy">
      <pre className="json-preview">{value}</pre>
    </Card>
  );
}
