"use client";

import { FormEvent, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { postJson } from "@/lib/api";

type AssistantLine =
  | { role: "doctor" | "nancy"; text: string }
  | { role: "action"; text: string };

type AssistantAction = {
  type: string;
  status: string;
};

const ACTION_LABELS: Record<string, string> = {
  add_nancy_task: "Directive saved for Nancy",
  add_nancy_touchpoint: "Doctor handoff saved",
  create_alert: "Alert created",
  queue_support_ping: "Support ping queued",
};

const SUGGESTED_PROMPTS = [
  "Summarize the most important recent Nancy and patient updates for me.",
  "Turn this into a Nancy directive: check sleep nightly, ask about grounding practice, and flag panic anticipation if it worsens.",
  "Draft and queue a gentle Nancy support ping focused on sleep disruption and overwhelm.",
  "Log this doctor update as a structured Nancy touchpoint: patient sounded calmer, but concentration is still poor and mornings are hard.",
];

export function NancyClinicianAssistant({
  doctorId,
  patientId,
  patientName,
}: {
  doctorId: string;
  patientId: string;
  patientName: string;
}) {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const [feedback, setFeedback] = useState("");
  const [history, setHistory] = useState<AssistantLine[]>([
    {
      role: "nancy",
      text: `I’m in clinician mode for ${patientName}. I can summarize recent records, turn your instructions into Nancy directives, save a handoff, or queue a proactive support ping.`,
    },
  ]);
  const [isPending, startTransition] = useTransition();

  function formatActionLabel(action: AssistantAction) {
    const base = ACTION_LABELS[action.type] ?? action.type.replaceAll("_", " ");
    return action.status === "ok" ? base : `${base} failed`;
  }

  function submit(outbound: string) {
    const trimmed = outbound.trim();
    if (!trimmed) return;

    setFeedback("");
    setHistory((current) => [...current, { role: "doctor", text: trimmed }]);
    setMessage("");

    startTransition(async () => {
      try {
        const response = await postJson<{
          response: string;
          actions_taken: AssistantAction[];
        }>(`/doctors/${doctorId}/patients/${patientId}/nancy/clinician-chat`, {
          message: trimmed,
          history: history
            .filter((line) => line.role !== "action")
            .map((line) => ({
              role: line.role === "doctor" ? "user" : "assistant",
              content: line.text,
            })),
        });

        setHistory((current) => {
          const next: AssistantLine[] = [
            ...current,
            { role: "nancy", text: response.response },
          ];
          for (const action of response.actions_taken ?? []) {
            next.push({ role: "action", text: formatActionLabel(action) });
          }
          return next;
        });
        router.refresh();
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "Nancy clinician assistant request failed.");
      }
    });
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    submit(message);
  }

  return (
    <div className="nancy-copilot-shell">
      <p className="nancy-copilot-intro">
        Doctor-only copilot. Use this space to convert your instructions into Nancy directives, handoffs, alerts, and proactive outreach without entering the patient conversation surface.
      </p>

      <div className="nancy-copilot-prompts">
        {SUGGESTED_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            className="nancy-copilot-prompt"
            disabled={isPending}
            onClick={() => submit(prompt)}
            type="button"
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="nancy-copilot-thread">
        {history.map((line, index) =>
          line.role === "action" ? (
            <div key={`${line.text}-${index}`} className="nancy-copilot-action-row">
              <span className="nancy-copilot-action">{line.text}</span>
            </div>
          ) : (
            <div
              key={`${line.role}-${index}`}
              className={`nancy-copilot-bubble ${line.role === "doctor" ? "nancy-copilot-bubble--doctor" : "nancy-copilot-bubble--nancy"}`}
            >
              <div className="nancy-copilot-label">{line.role === "doctor" ? "Doctor" : "Nancy Copilot"}</div>
              <p>{line.text}</p>
            </div>
          ),
        )}
      </div>

      <form className="nancy-copilot-form" onSubmit={onSubmit}>
        <textarea
          placeholder="Ask Nancy to summarize records, save a handoff, draft a directive, or queue a support ping."
          rows={4}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />
        <button className="action-button primary" disabled={isPending || !message.trim()} type="submit">
          {isPending ? "Working..." : "Ask Nancy Copilot"}
        </button>
      </form>

      {feedback ? <p className="form-feedback">{feedback}</p> : null}
    </div>
  );
}
