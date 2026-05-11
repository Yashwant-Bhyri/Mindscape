"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState, useTransition } from "react";

import { postFormData, postJson } from "@/lib/api";

function useMutationFeedback() {
  const [message, setMessage] = useState("");
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  async function run(task: () => Promise<unknown>, success: string) {
    startTransition(async () => {
      try {
        await task();
        setMessage(success);
        router.refresh();
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Request failed");
      }
    });
  }

  return { message, isPending, run };
}

function Feedback({ message }: { message: string }) {
  if (!message) return null;
  return <p className="form-feedback">{message}</p>;
}

export function IntakeForm({ doctorId }: { doctorId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [name, setName] = useState("");
  const [concern, setConcern] = useState("");
  const [context, setContext] = useState("");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/intake`, { name, concern, context }),
      "Patient intake created.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <input placeholder="Patient name" value={name} onChange={(e) => setName(e.target.value)} />
      <input placeholder="Presenting concern" value={concern} onChange={(e) => setConcern(e.target.value)} />
      <textarea placeholder="Context for first review" rows={4} value={context} onChange={(e) => setContext(e.target.value)} />
      <button className="action-button primary" disabled={isPending} type="submit">Stage Intake</button>
      <Feedback message={message} />
    </form>
  );
}

export function OutreachForm({
  doctorId,
  patientId,
  defaultTarget = "",
}: {
  doctorId: string;
  patientId?: string;
  defaultTarget?: string;
}) {
  const { message, isPending, run } = useMutationFeedback();
  const [target, setTarget] = useState(defaultTarget);
  const [body, setBody] = useState("");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/outreach`, { target, message: body, patient_id: patientId }),
      "Follow-up queued.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <input placeholder="Patient or care team" value={target} onChange={(e) => setTarget(e.target.value)} />
      <textarea placeholder="Check-in message" rows={4} value={body} onChange={(e) => setBody(e.target.value)} />
      <button className="action-button secondary" disabled={isPending} type="submit">
        {patientId ? "Queue Patient Follow-up" : "Queue Check-in"}
      </button>
      <Feedback message={message} />
    </form>
  );
}

export function CheckinForm({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [form, setForm] = useState({
    mood_score: "5",
    anxiety_score: "5",
    sleep_hours: "7",
    energy_score: "5",
    stress_score: "5",
    cognition_score: "5",
    memory_score: "5",
    functioning_score: "5",
    medication_adherence: "100",
    side_effects: "",
    daily_update: "",
    safety_concerns: "",
    significant_events: "",
    clinical_summary: "",
  });

  const setField = (key: string, value: string) => setForm((current) => ({ ...current, [key]: value }));

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/patients/${patientId}/checkins`, form),
      "Daily clinical report sent to Nancy and doctor.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <div className="split-grid">
        {[
          ["mood_score", "Mood 0-10"],
          ["anxiety_score", "Anxiety 0-10"],
          ["sleep_hours", "Sleep hours"],
          ["energy_score", "Energy 0-10"],
          ["stress_score", "Stress 0-10"],
          ["cognition_score", "Cognition 0-10"],
          ["memory_score", "Memory 0-10"],
          ["functioning_score", "Functioning 0-10"],
          ["medication_adherence", "Medication adherence %"],
        ].map(([key, label]) => (
          <input key={key} placeholder={label} value={form[key as keyof typeof form]} onChange={(e) => setField(key, e.target.value)} />
        ))}
      </div>
      <textarea placeholder="Daily clinical update" rows={3} value={form.daily_update} onChange={(e) => setField("daily_update", e.target.value)} />
      <textarea placeholder="Cognition and recall observations" rows={3} value={form.clinical_summary} onChange={(e) => setField("clinical_summary", e.target.value)} />
      <textarea placeholder="Significant events or triggers" rows={3} value={form.significant_events} onChange={(e) => setField("significant_events", e.target.value)} />
      <textarea placeholder="Side effects or medication issues" rows={2} value={form.side_effects} onChange={(e) => setField("side_effects", e.target.value)} />
      <textarea placeholder="Safety concerns" rows={2} value={form.safety_concerns} onChange={(e) => setField("safety_concerns", e.target.value)} />
      <button className="action-button primary" disabled={isPending} type="submit">Send Daily Report</button>
      <Feedback message={message} />
    </form>
  );
}

export function MessageComposer({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [body, setBody] = useState("");
  const [recipient, setRecipient] = useState("nancy");
  const [senderRole, setSenderRole] = useState("patient");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/patients/${patientId}/messages`, { body, recipient, sender_role: senderRole }),
      "Message sent.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <textarea
        placeholder="Share an update, ask for support, or leave a note for Nancy or your doctor."
        rows={4}
        value={body}
        onChange={(e) => setBody(e.target.value)}
      />
      <div className="split-grid">
        <select value={recipient} onChange={(e) => setRecipient(e.target.value)}>
          <option value="nancy">Send to Nancy</option>
          <option value="doctor">Send to Doctor</option>
          <option value="both">Send to Both</option>
        </select>
        <select value={senderRole} onChange={(e) => setSenderRole(e.target.value)}>
          <option value="patient">I am the patient</option>
          <option value="caregiver">I am helping the patient</option>
        </select>
      </div>
      <button className="action-button secondary" disabled={isPending} type="submit">Send Update</button>
      <Feedback message={message} />
    </form>
  );
}

export function NancyDirectiveForm({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [title, setTitle] = useState("");
  const [instructions, setInstructions] = useState("");
  const [category, setCategory] = useState("Recovery");
  const [dueLabel, setDueLabel] = useState("");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () =>
        postJson(`/doctors/${doctorId}/patients/${patientId}/nancy/directives`, {
          title,
          instructions,
          category,
          due_label: dueLabel,
        }),
      "Nancy directive added.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <input placeholder="Directive title" value={title} onChange={(e) => setTitle(e.target.value)} />
      <input placeholder="Category" value={category} onChange={(e) => setCategory(e.target.value)} />
      <input placeholder="Due or revisit date" value={dueLabel} onChange={(e) => setDueLabel(e.target.value)} />
      <textarea placeholder="Directive guidance for Nancy" rows={4} value={instructions} onChange={(e) => setInstructions(e.target.value)} />
      <button className="action-button primary" disabled={isPending} type="submit">Add Directive</button>
      <Feedback message={message} />
    </form>
  );
}

export function NancyTouchpointForm({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [form, setForm] = useState({
    mode: "voice",
    conversation_goal: "",
    patient_report: "",
    patient_message: "",
    clinician_summary: "",
    observed_mood: "",
    functioning_note: "",
    cognition_note: "",
    medication_note: "",
    safety_note: "",
    recommended_follow_up: "",
    escalation_level: "routine",
    relay_to_patient: false,
  });

  const setField = (key: string, value: string | boolean) => setForm((current) => ({ ...current, [key]: value }));

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/patients/${patientId}/nancy/touchpoints`, form),
      "Nancy touchpoint saved.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <div className="split-grid">
        <input placeholder="Contact mode" value={form.mode} onChange={(e) => setField("mode", e.target.value)} />
        <input placeholder="Conversation goal" value={form.conversation_goal} onChange={(e) => setField("conversation_goal", e.target.value)} />
        <input placeholder="Observed mood" value={form.observed_mood} onChange={(e) => setField("observed_mood", e.target.value)} />
        <input placeholder="Escalation level" value={form.escalation_level} onChange={(e) => setField("escalation_level", e.target.value)} />
      </div>
      <textarea placeholder="Patient-facing update" rows={3} value={form.patient_report} onChange={(e) => setField("patient_report", e.target.value)} />
      <label className="muted" style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <input checked={form.relay_to_patient} onChange={(e) => setField("relay_to_patient", e.target.checked)} type="checkbox" />
        Relay a Nancy message to the patient from this touchpoint
      </label>
      {form.relay_to_patient ? (
        <textarea
          placeholder="Exact Nancy message the patient should receive. If left empty, Nancy uses the patient-facing update above."
          rows={3}
          value={form.patient_message}
          onChange={(e) => setField("patient_message", e.target.value)}
        />
      ) : null}
      <textarea placeholder="Doctor-facing summary" rows={3} value={form.clinician_summary} onChange={(e) => setField("clinician_summary", e.target.value)} />
      <textarea placeholder="Functioning note" rows={2} value={form.functioning_note} onChange={(e) => setField("functioning_note", e.target.value)} />
      <textarea placeholder="Cognition or memory note" rows={2} value={form.cognition_note} onChange={(e) => setField("cognition_note", e.target.value)} />
      <textarea placeholder="Medication note" rows={2} value={form.medication_note} onChange={(e) => setField("medication_note", e.target.value)} />
      <textarea placeholder="Safety note" rows={2} value={form.safety_note} onChange={(e) => setField("safety_note", e.target.value)} />
      <textarea placeholder="Recommended follow-up" rows={2} value={form.recommended_follow_up} onChange={(e) => setField("recommended_follow_up", e.target.value)} />
      <button className="action-button secondary" disabled={isPending} type="submit">Save Nancy Touchpoint</button>
      <Feedback message={message} />
    </form>
  );
}

export function NancySupportPingButton({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  return (
    <div className="stack-form">
      <button
        className="action-button warm"
        disabled={isPending}
        onClick={() => run(() => postJson(`/doctors/${doctorId}/patients/${patientId}/nancy/support-ping`, {}), "Nancy support ping queued.")}
        type="button"
      >
        Send Nancy Support Ping
      </button>
      <Feedback message={message} />
    </div>
  );
}

export function SosForm({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [reason, setReason] = useState("");
  const [severity, setSeverity] = useState("urgent");
  const [district, setDistrict] = useState("福田区");
  const [notes, setNotes] = useState("");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/patients/${patientId}/sos`, { reason, severity, district, notes }),
      "SOS escalation prepared.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <div className="split-grid">
        <input placeholder="District" value={district} onChange={(e) => setDistrict(e.target.value)} />
        <input placeholder="Severity" value={severity} onChange={(e) => setSeverity(e.target.value)} />
      </div>
      <textarea placeholder="SOS reason" rows={3} value={reason} onChange={(e) => setReason(e.target.value)} />
      <textarea placeholder="Escalation notes" rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} />
      <button className="action-button danger" disabled={isPending} type="submit">Trigger SOS Escalation</button>
      <Feedback message={message} />
    </form>
  );
}

export function AlertActionButtons({ doctorId, alertId }: { doctorId: string; alertId: string }) {
  const { message, isPending, run } = useMutationFeedback();

  return (
    <div className="inline-actions">
      <button className="action-button secondary" disabled={isPending} onClick={() => run(() => postJson(`/doctors/${doctorId}/alerts/${alertId}`, { status: "acknowledged" }), "Alert acknowledged.")} type="button">
        Acknowledge
      </button>
      <button className="action-button primary" disabled={isPending} onClick={() => run(() => postJson(`/doctors/${doctorId}/alerts/${alertId}`, { status: "resolved" }), "Alert resolved.")} type="button">
        Resolve
      </button>
      <Feedback message={message} />
    </div>
  );
}

export function SessionNoteForm({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [title, setTitle] = useState("");
  const [note, setNote] = useState("");
  const [planUpdate, setPlanUpdate] = useState("");
  const [disposition, setDisposition] = useState("Continue current plan");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    run(
      () => postJson(`/doctors/${doctorId}/patients/${patientId}/session/notes`, { title, note, plan_update: planUpdate, disposition }),
      "Session review note saved.",
    );
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <input placeholder="Note title" value={title} onChange={(e) => setTitle(e.target.value)} />
      <textarea placeholder="Clinician review note" rows={4} value={note} onChange={(e) => setNote(e.target.value)} />
      <textarea placeholder="Care-plan update" rows={3} value={planUpdate} onChange={(e) => setPlanUpdate(e.target.value)} />
      <input placeholder="Disposition" value={disposition} onChange={(e) => setDisposition(e.target.value)} />
      <button className="action-button primary" disabled={isPending} type="submit">Save Session Note</button>
      <Feedback message={message} />
    </form>
  );
}

export function SessionUploadForm({ doctorId, patientId }: { doctorId: string; patientId: string }) {
  const { message, isPending, run } = useMutationFeedback();
  const [result, setResult] = useState("");

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const input = form.elements.namedItem("audio") as HTMLInputElement | null;
    const file = input?.files?.[0];
    if (!file) {
      setResult("Choose an audio file first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    run(async () => {
      const payload = await postFormData<{ analysis: { view_model: { hypothesis_name: string; hypothesis_confidence: string; treatment_plan: string } } }>(
        `/doctors/${doctorId}/patients/${patientId}/session/analyze-upload`,
        formData,
      );
      setResult(
        `${payload.analysis.view_model.hypothesis_name} ${payload.analysis.view_model.hypothesis_confidence}\n${payload.analysis.view_model.treatment_plan}`,
      );
    }, "Session analysis completed.");
  }

  return (
    <form className="stack-form" onSubmit={onSubmit}>
      <input accept="audio/*" name="audio" type="file" />
      <button className="action-button secondary" disabled={isPending} type="submit">Analyze Uploaded Session</button>
      <Feedback message={message || result} />
    </form>
  );
}
