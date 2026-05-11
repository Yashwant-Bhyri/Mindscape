"use client";

import { FormEvent, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { postJson } from "@/lib/api";

function medicationPayload(value: string) {
  switch (value) {
    case "missed":
      return { medication_adherence: "0", side_effects: "" };
    case "side-effects":
      return { medication_adherence: "100", side_effects: "Patient reported side effects during portal check-in." };
    case "not-applicable":
      return { medication_adherence: "", side_effects: "" };
    default:
      return { medication_adherence: "100", side_effects: "" };
  }
}

export function PatientPortalMessageComposer({
  doctorId,
  patientId,
}: {
  doctorId: string;
  patientId: string;
}) {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const [recipient, setRecipient] = useState("nancy");
  const [feedback, setFeedback] = useState("");
  const [isPending, startTransition] = useTransition();

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!message.trim()) return;

    startTransition(async () => {
      try {
        await postJson(`/doctors/${doctorId}/patients/${patientId}/messages`, {
          body: message.trim(),
          recipient,
          sender_role: "patient",
        });
        setMessage("");
        setFeedback("Message sent.");
        router.refresh();
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "Message failed.");
      }
    });
  }

  return (
    <form className="mc-form" onSubmit={onSubmit}>
      <textarea
        className="mc-ta"
        placeholder="How are you feeling? Any changes since your last check-in? Nancy will relay clinically relevant updates to your doctor."
        rows={5}
        value={message}
        onChange={(event) => setMessage(event.target.value)}
      />
      <div className="mc-row">
        <select
          className="patient-select"
          value={recipient}
          onChange={(event) => setRecipient(event.target.value)}
        >
          <option value="nancy">Send to Nancy</option>
          <option value="doctor">Send to doctor</option>
          <option value="both">Send to both</option>
        </select>
        <button className="patient-portal-btn patient-portal-btn--teal" disabled={isPending} type="submit">
          {isPending ? "Sending..." : "Send"}
        </button>
      </div>
      {feedback ? <p className="form-feedback">{feedback}</p> : null}
    </form>
  );
}

export function PatientPortalCheckinForm({
  doctorId,
  patientId,
}: {
  doctorId: string;
  patientId: string;
}) {
  const router = useRouter();
  const [mood, setMood] = useState("7");
  const [sleep, setSleep] = useState("6");
  const [anxiety, setAnxiety] = useState("4");
  const [update, setUpdate] = useState("");
  const [medication, setMedication] = useState("as-prescribed");
  const [feedback, setFeedback] = useState("");
  const [isPending, startTransition] = useTransition();

  function onSubmit(event: FormEvent) {
    event.preventDefault();

    startTransition(async () => {
      try {
        const medicationState = medicationPayload(medication);
        await postJson(`/doctors/${doctorId}/patients/${patientId}/checkins`, {
          mood_score: mood,
          anxiety_score: anxiety,
          sleep_hours: sleep,
          energy_score: "",
          stress_score: "",
          cognition_score: "",
          memory_score: "",
          functioning_score: "",
          daily_update: update.trim(),
          significant_events: "",
          safety_concerns: "",
          clinical_summary: "Patient portal quick check-in submitted from companion home.",
          ...medicationState,
        });
        setFeedback("Daily clinical report sent to Nancy and doctor.");
        router.refresh();
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "Check-in failed.");
      }
    });
  }

  return (
    <form className="ci-form" onSubmit={onSubmit}>
      <div>
        <div className="ci-label" style={{ marginBottom: "0.5rem" }}>Mood today</div>
        <div className="ci-range-row">
          <input
            className="ci-range"
            max="10"
            min="1"
            type="range"
            value={mood}
            onChange={(event) => setMood(event.target.value)}
          />
          <span className="ci-range-val">{mood}</span>
          <span className="ci-scale">/ 10</span>
        </div>
      </div>

      <div>
        <div className="ci-label" style={{ marginBottom: "0.5rem" }}>Sleep last night</div>
        <div className="ci-range-row">
          <input
            className="ci-range"
            max="10"
            min="1"
            type="range"
            value={sleep}
            onChange={(event) => setSleep(event.target.value)}
          />
          <span className="ci-range-val">{sleep}</span>
          <span className="ci-scale">/ 10</span>
        </div>
      </div>

      <div>
        <div className="ci-label" style={{ marginBottom: "0.5rem" }}>Anxiety level</div>
        <div className="ci-range-row">
          <input
            className="ci-range"
            max="10"
            min="1"
            type="range"
            value={anxiety}
            onChange={(event) => setAnxiety(event.target.value)}
          />
          <span className="ci-range-val">{anxiety}</span>
          <span className="ci-scale">/ 10</span>
        </div>
      </div>

      <div>
        <div className="ci-label">What's been on your mind?</div>
        <textarea
          className="ci-input"
          placeholder="Anything you'd like your care team to know..."
          rows={5}
          style={{ marginTop: "0.4rem", resize: "vertical" }}
          value={update}
          onChange={(event) => setUpdate(event.target.value)}
        />
      </div>

      <div>
        <div className="ci-label">Medication taken today?</div>
        <select
          className="ci-select"
          style={{ marginTop: "0.4rem" }}
          value={medication}
          onChange={(event) => setMedication(event.target.value)}
        >
          <option value="as-prescribed">Yes, as prescribed</option>
          <option value="missed">Missed a dose</option>
          <option value="side-effects">Side effects to report</option>
          <option value="not-applicable">No medication on my plan</option>
        </select>
      </div>

      <button className="patient-portal-btn patient-portal-btn--amber" disabled={isPending} type="submit">
        {isPending ? "Submitting..." : "Submit Check-in"}
      </button>
      {feedback ? <p className="form-feedback">{feedback}</p> : null}
    </form>
  );
}
