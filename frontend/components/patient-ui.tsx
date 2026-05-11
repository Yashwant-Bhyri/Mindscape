import { ReactNode } from "react";

import { ButtonLink, Card } from "@/components/ui";

type PortalStat = {
  label: string;
  value: string | number;
};

type PortalAction = {
  href: string;
  label: string;
  tone?: "primary" | "secondary" | "warm";
};

type InboxMessage = {
  id?: string;
  created_at?: string;
  sender_role?: string;
  recipient?: string;
  body?: string;
  channel?: string;
};

type NancyUpdate = {
  id?: string;
  created_at?: string;
  patient_message?: string;
  patient_report?: string;
  conversation_goal?: string;
  recommended_follow_up?: string;
  observed_mood?: string;
};

type NancyTask = {
  id?: string;
  title?: string;
  instructions?: string;
  category?: string;
  due_label?: string;
  status?: string;
};

function formatTimestamp(value?: string) {
  if (!value) return "Just now";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

function senderLabel(role: string) {
  switch (role) {
    case "nancy":
      return "Nancy";
    case "doctor":
      return "Care team";
    case "caregiver":
      return "Caregiver";
    default:
      return "You";
  }
}

function recipientLabel(recipient: string) {
  switch (recipient) {
    case "nancy":
      return "Sent to Nancy";
    case "doctor":
      return "Sent to doctor";
    case "both":
      return "Sent to Nancy and doctor";
    case "patient":
      return "For you";
    default:
      return "Message";
  }
}

function toneClass(role: string) {
  if (role === "nancy") return "patient-thread-item--nancy";
  if (role === "doctor") return "patient-thread-item--doctor";
  if (role === "caregiver") return "patient-thread-item--caregiver";
  return "patient-thread-item--self";
}

export function PatientPortalHero({
  eyebrow,
  title,
  body,
  stats,
  actions = [],
  aside,
}: {
  eyebrow: string;
  title: string;
  body: string;
  stats: PortalStat[];
  actions?: PortalAction[];
  aside?: ReactNode;
}) {
  return (
    <section className="patient-hero-banner">
      <div className="patient-hero-copy">
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="hero-body">{body}</p>
        <div className="patient-hero-actions">
          {actions.map((action) => (
            <ButtonLink href={action.href} key={action.href} tone={action.tone ?? "primary"}>
              {action.label}
            </ButtonLink>
          ))}
        </div>
      </div>
      <div className="patient-hero-side">
        <div className="patient-stat-ribbon">
          {stats.map((stat) => (
            <div className="patient-stat-chip" key={stat.label}>
              <span>{stat.label}</span>
              <strong>{stat.value}</strong>
            </div>
          ))}
        </div>
        {aside ? <div className="patient-hero-aside">{aside}</div> : null}
      </div>
    </section>
  );
}

export function PatientContextPanel({
  title,
  lines,
}: {
  title: string;
  lines: Array<{ label: string; value: string | number | undefined }>;
}) {
  return (
    <Card className="patient-surface patient-context-panel">
      <div className="patient-panel-header">
        <p className="patient-panel-kicker">Care Snapshot</p>
        <h2>{title}</h2>
      </div>
      <div className="patient-context-grid">
        {lines.map((line) => (
          <div className="patient-context-row" key={line.label}>
            <span>{line.label}</span>
            <strong>{line.value ?? "Not available"}</strong>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function PatientInboxThread({ items }: { items: InboxMessage[] }) {
  if (!items.length) {
    return (
      <div className="patient-empty-state">
        <strong>No messages yet.</strong>
        <p>Messages from Nancy and your care team will appear here, separate from doctor-only chart notes.</p>
      </div>
    );
  }

  const ordered = [...items].reverse();

  return (
    <div className="patient-thread">
      {ordered.map((item, index) => {
        const role = (item.sender_role || "patient").toLowerCase();
        const recipient = (item.recipient || "doctor").toLowerCase();
        const body = item.body || "No message content.";
        const label = senderLabel(role);

        return (
          <article
            className={`patient-thread-item ${toneClass(role)}`.trim()}
            key={item.id ?? `${item.created_at ?? "message"}-${index}`}
          >
            <div className="patient-thread-meta">
              <div>
                <span className="patient-thread-sender">{label}</span>
                <span className="patient-thread-route">{recipientLabel(recipient)}</span>
              </div>
              <time>{formatTimestamp(item.created_at)}</time>
            </div>
            <p>{body}</p>
            <div className="patient-thread-tags">
              {role === "nancy" ? <span className="patient-tag">Relayed by Nancy</span> : null}
              <span className="patient-tag patient-tag--soft">{item.channel || "In-app chat"}</span>
            </div>
          </article>
        );
      })}
    </div>
  );
}

export function PatientSupportCards({
  title,
  items,
}: {
  title: string;
  items: NancyUpdate[];
}) {
  return (
    <Card className="patient-surface patient-support-panel">
      <div className="patient-panel-header">
        <p className="patient-panel-kicker">Nancy Relay</p>
        <h2>{title}</h2>
      </div>
      {items.length ? (
        <div className="patient-support-stack">
          {items.slice(0, 4).map((item, index) => (
            <article className="patient-support-card" key={item.id ?? `${item.created_at ?? "relay"}-${index}`}>
              <div className="patient-support-meta">
                <span>{item.conversation_goal || "Support update"}</span>
                <time>{formatTimestamp(item.created_at)}</time>
              </div>
              <p>{item.patient_message || item.patient_report || "Nancy has a fresh update for you."}</p>
              <div className="patient-support-tags">
                {item.observed_mood ? <span className="patient-tag">{item.observed_mood}</span> : null}
                {item.recommended_follow_up ? (
                  <span className="patient-tag patient-tag--soft">{item.recommended_follow_up}</span>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="patient-empty-state">
          <strong>No relayed Nancy guidance yet.</strong>
          <p>Once your care team hands something off through Nancy, it will appear here in patient-safe language.</p>
        </div>
      )}
    </Card>
  );
}

export function PatientTaskPanel({
  title,
  tasks,
}: {
  title: string;
  tasks: NancyTask[];
}) {
  return (
    <Card className="patient-surface patient-task-panel">
      <div className="patient-panel-header">
        <p className="patient-panel-kicker">Today With Nancy</p>
        <h2>{title}</h2>
      </div>
      {tasks.length ? (
        <div className="patient-task-stack">
          {tasks.slice(0, 4).map((task, index) => (
            <article className="patient-task-card" key={task.id ?? `${task.title ?? "task"}-${index}`}>
              <div className="patient-task-topline">
                <span>{task.category || "Support"}</span>
                <span>{task.status || "Active"}</span>
              </div>
              <h3>{task.title || "Care task"}</h3>
              <p>{task.instructions || "Nancy will help with this during your conversations and check-ins."}</p>
              {task.due_label ? <div className="patient-task-due">Next review: {task.due_label}</div> : null}
            </article>
          ))}
        </div>
      ) : (
        <div className="patient-empty-state">
          <strong>No active Nancy tasks right now.</strong>
          <p>That usually means Nancy is in a listening and support mode rather than following a specific handoff.</p>
        </div>
      )}
    </Card>
  );
}
