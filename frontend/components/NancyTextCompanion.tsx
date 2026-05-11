"use client";

import { FormEvent, useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { postJson } from "@/lib/api";

type DailyQuestion = {
  key: string;
  label: string;
  prompt: string;
};

type DailyStatus = {
  due: boolean;
  completed_today: boolean;
  question_count: number;
  questions: DailyQuestion[];
};

type ThreadLine = {
  role: "patient" | "nancy" | "system";
  text: string;
};

const NUMERIC_FIELDS = new Set([
  "mood_score",
  "anxiety_score",
  "sleep_hours",
  "energy_score",
  "stress_score",
  "cognition_score",
  "memory_score",
  "functioning_score",
  "medication_adherence",
]);

function firstNumber(value: string, fallback = "") {
  const match = value.match(/-?\d+(\.\d+)?/);
  return match ? match[0] : fallback;
}

function buildCheckinPayload(questions: DailyQuestion[], answers: string[]) {
  const payload: Record<string, string> = {
    mood_score: "",
    anxiety_score: "",
    sleep_hours: "",
    energy_score: "",
    stress_score: "",
    cognition_score: "",
    memory_score: "",
    functioning_score: "",
    medication_adherence: "",
    side_effects: "",
    daily_update: "",
    safety_concerns: "",
    significant_events: "",
    clinical_summary: "",
  };

  questions.forEach((question, index) => {
    const answer = (answers[index] ?? "").trim();
    if (!answer) return;
    if (NUMERIC_FIELDS.has(question.key)) {
      payload[question.key] = firstNumber(answer, answer);
      if (answer !== payload[question.key]) {
        payload.clinical_summary = [payload.clinical_summary, `${question.label}: ${answer}`]
          .filter(Boolean)
          .join("\n");
      }
      return;
    }
    if (payload[question.key]) {
      payload[question.key] = `${payload[question.key]}\n${question.label}: ${answer}`;
    } else {
      payload[question.key] = answer;
    }
  });

  payload.clinical_summary = [
    payload.clinical_summary,
    "Nancy completed a conversational daily check-in and captured the answers one turn at a time.",
  ]
    .filter(Boolean)
    .join("\n");

  return payload;
}

export function NancyTextCompanion({
  doctorId,
  patientId,
  patientName,
  dailyStatus,
}: {
  doctorId: string;
  patientId: string;
  patientName: string;
  dailyStatus?: DailyStatus | null;
}) {
  const router = useRouter();
  const displayName = patientName.replace(/\.$/, "");
  const resolvedStatus = dailyStatus ?? {
    due: false,
    completed_today: false,
    question_count: 0,
    questions: [],
  };
  const questions = useMemo(() => resolvedStatus.questions ?? [], [resolvedStatus.questions]);
  const startsWithDaily = resolvedStatus.due && questions.length > 0;
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [feedback, setFeedback] = useState("");
  const [isPending, startTransition] = useTransition();
  const [thread, setThread] = useState<ThreadLine[]>(() => {
    if (startsWithDaily) {
      return [
        {
          role: "nancy",
          text: `Hi ${displayName}, I have today's check-in ready. I'll ask one thing at a time, then I’ll send a clear summary to your doctor.`,
        },
        { role: "nancy", text: questions[0].prompt },
      ];
    }
    return [
      {
        role: "nancy",
        text: resolvedStatus.completed_today
          ? `Today's check-in is already complete, ${displayName}. I won’t run the daily questions again today.`
          : `Hi ${displayName}, the daily check-in is not available right now, but I’m still here in the general Nancy conversation below.`,
      },
    ];
  });

  function finishDaily(nextAnswers: string[]) {
    const payload = buildCheckinPayload(questions, nextAnswers);
    startTransition(async () => {
      try {
        const response = await postJson<{
          result?: { plan?: { patient_response?: string } };
        }>(`/doctors/${doctorId}/patients/${patientId}/checkins`, payload);
        const nancyReply =
          response.result?.plan?.patient_response ??
          "Thank you. I sent today's check-in and summarized the important parts for your doctor.";
        setThread((current) => [
          ...current,
          { role: "nancy", text: nancyReply },
          { role: "system", text: "Daily check-in complete. Use Speak With Nancy below for anything else today." },
        ]);
        setStep(0);
        router.refresh();
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "Daily check-in failed.");
      }
    });
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || isPending || !startsWithDaily) return;

    setFeedback("");
    setDraft("");
    setThread((current) => [...current, { role: "patient", text }]);

    const nextAnswers = [...answers, text];
    setAnswers(nextAnswers);
    const nextStep = step + 1;
    if (nextStep < questions.length) {
      setStep(nextStep);
      setThread((current) => [...current, { role: "nancy", text: questions[nextStep].prompt }]);
    } else {
      finishDaily(nextAnswers);
    }
  }

  return (
    <div className="nancy-text-shell">
      <div className={`nancy-daily-banner ${resolvedStatus.due ? "nancy-daily-banner--due" : ""}`}>
        <div>
          <span>{resolvedStatus.due ? "Daily check-in due" : "Daily check-in complete"}</span>
          <strong>
            {resolvedStatus.due
              ? `${resolvedStatus.question_count} guided prompts, including doctor-specific follow-up.`
              : "Nancy will not ask the full questionnaire again today."}
          </strong>
        </div>
        {startsWithDaily ? <span className="nancy-step-pill">{step + 1} / {questions.length}</span> : null}
      </div>

      <div className="nancy-text-thread">
        {thread.map((line, index) => (
          <div
            className={`nancy-text-line nancy-text-line--${line.role}`}
            key={`${line.role}-${index}-${line.text.slice(0, 16)}`}
          >
            <span>{line.role === "patient" ? "You" : line.role === "system" ? "Status" : "Nancy"}</span>
            <p>{line.text}</p>
          </div>
        ))}
      </div>

      <form className="nancy-text-form" onSubmit={onSubmit}>
        <textarea
          placeholder={startsWithDaily ? "Answer Nancy's current question..." : "Daily check-in is not active."}
          rows={3}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={!startsWithDaily}
        />
        <button className="action-button primary" disabled={isPending || !draft.trim() || !startsWithDaily} type="submit">
          {isPending ? "Sending..." : "Answer Daily Check-In"}
        </button>
      </form>

      {feedback ? <p className="form-feedback">{feedback}</p> : null}
    </div>
  );
}

export function SpeakWithNancy({
  doctorId,
  patientId,
  patientName,
}: {
  doctorId: string;
  patientId: string;
  patientName: string;
}) {
  const router = useRouter();
  const displayName = patientName.replace(/\.$/, "");
  const [draft, setDraft] = useState("");
  const [feedback, setFeedback] = useState("");
  const [isPending, startTransition] = useTransition();
  const [thread, setThread] = useState<ThreadLine[]>([
    {
      role: "nancy",
      text: `Hi ${displayName}, this space is for questions, issues, complaints, help, or anything you want me to route to your care team. I won’t run the daily questionnaire here.`,
    },
  ]);

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || isPending) return;

    setFeedback("");
    setDraft("");
    setThread((current) => [...current, { role: "patient", text }]);

    startTransition(async () => {
      try {
        const response = await postJson<{ response: string }>(
          `/doctors/${doctorId}/patients/${patientId}/nancy/chat`,
          {
            message: text,
            history: thread
              .filter((line) => line.role !== "system")
              .slice(-8)
              .map((line) => ({
                role: line.role === "patient" ? "user" : "assistant",
                content: line.text,
              })),
          },
        );
        setThread((current) => [...current, { role: "nancy", text: response.response }]);
        router.refresh();
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "Nancy could not reply.");
      }
    });
  }

  return (
    <div className="nancy-text-shell">
      <div className="nancy-daily-banner">
        <div>
          <span>Open Nancy Conversation</span>
          <strong>For questions, symptoms, logistics, complaints, and help outside the daily check-in.</strong>
        </div>
      </div>

      <div className="nancy-text-thread">
        {thread.map((line, index) => (
          <div
            className={`nancy-text-line nancy-text-line--${line.role}`}
            key={`${line.role}-${index}-${line.text.slice(0, 16)}`}
          >
            <span>{line.role === "patient" ? "You" : line.role === "system" ? "Status" : "Nancy"}</span>
            <p>{line.text}</p>
          </div>
        ))}
      </div>

      <form className="nancy-text-form" onSubmit={onSubmit}>
        <textarea
          placeholder="Ask Nancy for help, report a change, or leave a note for your care team..."
          rows={3}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button className="action-button primary" disabled={isPending || !draft.trim()} type="submit">
          {isPending ? "Sending..." : "Send To Nancy"}
        </button>
      </form>

      {feedback ? <p className="form-feedback">{feedback}</p> : null}
    </div>
  );
}
