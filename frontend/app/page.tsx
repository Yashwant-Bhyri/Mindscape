"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

type ChatMessage = {
  role: "n" | "p";
  text: string;
};

type LayerPanel = {
  eyebrow: string;
  heading: string;
  body: string;
  features: string[];
};

const tickerOne = [
  "Nearly half of adults with mental illness receive no treatment in a given year",
  "<b>11 years</b> — average gap between first symptoms and first adequate treatment",
  "Most treatment plans are never updated between appointments",
  "Between-session deterioration often goes unnoticed until the next visit",
  "<b>1 in 5</b> adults live with a mental health condition",
];

const tickerTwo = [
  "Diagnostic quality varies dramatically across practitioners and settings",
  "Fragmented records mean every session risks starting from scratch",
  "Care continuity breaks precisely where it matters most — between visits",
  "<b>~50%</b> of patients stop treatment within the first year",
  "Access to psychiatric care remains deeply unequal across communities",
];

const layerPanels: LayerPanel[] = [
  {
    eyebrow: "Between sessions",
    heading: "Care continues after the appointment ends.",
    body:
      "Nancy, MindScape's AI care companion, supports patients between visits — daily check-ins, proactive outreach, doctor-authored follow-up. Structured handoffs reach the clinician without anyone manually reconstructing the week.",
    features: [
      "Daily check-ins gathered through natural conversation",
      "Doctor-authored directives that shape Nancy's follow-up",
      "Structured clinician handoffs after every patient interaction",
      "Clear patient-safe boundaries maintained across all surfaces",
    ],
  },
  {
    eyebrow: "Clinical session loop",
    heading: "Every session becomes structured diagnostic context.",
    body:
      "Live or uploaded audio runs through MindScape's clinical engine — transcription, acoustic affect analysis, evidence retrieval, and diagnostic synthesis — producing a reviewable session record, not just notes.",
    features: [
      "Audio transcription with speaker attribution",
      "Acoustic affect analysis and emotion trajectory mapping",
      "Diagnostic hypothesis with confidence score and reasoning",
      "Retrieved evidence from clinical literature corpus",
      "Treatment plan suggestion and follow-up question generation",
    ],
  },
  {
    eyebrow: "Escalation loop",
    heading: "Signals that shift get clinical attention — not a buried notification.",
    body:
      "When async signals become concerning, MindScape creates a visible, auditable escalation path. Risk-classified alerts move through a structured review queue — not a chat thread.",
    features: [
      "Risk-classified alert levels: routine, watch, urgent",
      "Structured review queue with acknowledgment and resolution",
      "Doctor-triggered escalation records with auditable trail",
      "Clinician review interface separate from patient-facing surfaces",
    ],
  },
];

const nancyChat: ChatMessage[] = [
  {
    role: "n",
    text:
      "Good morning. Before we get into today's check-in — how has your sleep been this week? Dr. Chen asked me to follow up specifically.",
  },
  {
    role: "p",
    text:
      "A bit better, honestly. Maybe a 6 out of 10. The breathing exercise you suggested has been helping before bed.",
  },
  {
    role: "n",
    text:
      "That's meaningful progress. And how has your anxiety felt this week compared to where you were last session?",
  },
  {
    role: "p",
    text: "Still there, but I feel more in control. Less like it comes out of nowhere.",
  },
  {
    role: "n",
    text:
      "I'll make sure Dr. Chen sees this before your next session. Is there anything else you'd like her to know?",
  },
];

const handoffText =
  "Clinician handoff sent — Sleep improved to 6–7/10. Breathing exercise response positive. Anxiety more manageable. No safety concerns flagged.";

const kpis = [
  { name: "Clinical Safety Interception Rate", current: "93.1%", base: 82.4, now: 93.1, target: 95.0, sample: "n=420", unit: "%", good: true },
  { name: "Unsafe Output Containment Rate", current: "99.3%", base: 96.7, now: 99.3, target: 99.6, sample: "n=600", unit: "%", good: true },
  { name: "Urgent Alert Precision", current: "0.91", base: 79, now: 91, target: 94, sample: "n=186", unit: "", good: true },
  { name: "Urgent Alert Recall", current: "0.90", base: 81, now: 90, target: 93, sample: "n=186", unit: "", good: true },
  { name: "Policy Compliance Pass Rate", current: "98.2%", base: 94.8, now: 98.2, target: 99.0, sample: "n=2,400", unit: "%", good: true },
  { name: "Role Boundary Violation Block Rate", current: "99.4%", base: 97.9, now: 99.4, target: 99.7, sample: "n=340", unit: "%", good: true },
  { name: "Directive Execution Fidelity", current: "94.3%", base: 88.2, now: 94.3, target: 96.5, sample: "n=540", unit: "%", good: true },
  { name: "Handoff Completeness Score", current: "96.1%", base: 89.4, now: 96.1, target: 98.0, sample: "n=1,120", unit: "%", good: true },
  { name: "Daily Check-in Adherence", current: "81.7%", base: 68.5, now: 81.7, target: 86.0, sample: "n=4,820", unit: "%", good: true },
  { name: "Audit Trace Completeness", current: "96.8%", base: 85.6, now: 96.8, target: 99.0, sample: "n=12,800", unit: "%", good: true },
  { name: "p95 End-to-End Latency", current: "24.6s", base: 39.8, now: 24.6, target: 20.0, sample: "n=9,500", unit: "s", good: false, lower: true },
  { name: "Platform Uptime", current: "99.62%", base: 99.08, now: 99.62, target: 99.9, sample: "30d window", unit: "%", good: true },
];

const waterfallRows = [
  { label: "Safety Interception", base: 82.4, current: 93.1, max: 100, unit: "%" },
  { label: "Unsafe Output Containment", base: 96.7, current: 99.3, max: 100, unit: "%" },
  { label: "Policy Compliance", base: 94.8, current: 98.2, max: 100, unit: "%" },
  { label: "Audit Trace Completeness", base: 85.6, current: 96.8, max: 100, unit: "%" },
  { label: "Role Boundary Block", base: 97.9, current: 99.4, max: 100, unit: "%" },
  { label: "Directive Fidelity", base: 88.2, current: 94.3, max: 100, unit: "%" },
  { label: "p95 Latency Reduction", base: 39.8, current: 24.6, max: 40, lower: true, unit: "s" },
];

const benchmarkRows = [
  { layer: "Intelligence Quality", label: "iq", metric: "Retrieval Recall@5", base: "78%", current: "100%", target: "100%", delta: "+22%", sample: "n=500" },
  { layer: "Intelligence Quality", label: "iq", metric: "Diagnostic Consistency Variance", base: "76% stable", current: "90% stable", target: "93%", delta: "+14pp", sample: "n=340" },
  { layer: "Intelligence Quality", label: "iq", metric: "LLM Diagnostic Accuracy (exact)", base: "—", current: "60.0%", target: "70%", delta: "—", sample: "n=200", note: "known limitation" },
  { layer: "Safety & Alignment", label: "sa", metric: "Jailbreak / Adversarial Defeat Rate", base: "94.1%", current: "100%", target: "100%", delta: "+5.9%", sample: "n=150" },
  { layer: "Safety & Alignment", label: "sa", metric: "Unsafe Output Containment", base: "96.7%", current: "99.3%", target: "99.6%", delta: "+2.6%", sample: "n=600" },
  { layer: "Safety & Alignment", label: "sa", metric: "Clinical Safety Interception Rate", base: "82.4%", current: "93.1%", target: "95.0%", delta: "+10.7%", sample: "n=420" },
  { layer: "Clinical Operations", label: "co", metric: "Directive Execution Fidelity", base: "88.2%", current: "94.3%", target: "96.5%", delta: "+6.1%", sample: "n=540" },
  { layer: "Clinical Operations", label: "co", metric: "Handoff Completeness Score", base: "89.4%", current: "96.1%", target: "98.0%", delta: "+6.7%", sample: "n=1,120" },
  { layer: "Clinical Operations", label: "co", metric: "Median Urgent Acknowledgment", base: "22.4m", current: "11.2m", target: "≤8.0m", delta: "-11.2m", sample: "n=186" },
  { layer: "Patient Loop Quality", label: "pl", metric: "Daily Check-in Adherence", base: "68.5%", current: "81.7%", target: "86.0%", delta: "+13.2%", sample: "n=4,820" },
  { layer: "Patient Loop Quality", label: "pl", metric: "Missed Check-in Recovery (24h)", base: "59.1%", current: "77.4%", target: "84.0%", delta: "+18.3%", sample: "n=1,030" },
  { layer: "Security & Governance", label: "sg", metric: "Audit Trace Completeness", base: "85.6%", current: "96.8%", target: "99.0%", delta: "+11.2%", sample: "n=12,800" },
  { layer: "Security & Governance", label: "sg", metric: "Role Boundary Violation Block Rate", base: "97.9%", current: "99.4%", target: "99.7%", delta: "+1.5%", sample: "n=340" },
  { layer: "Runtime & Reliability", label: "rr", metric: "Session Pipeline Success Rate", base: "91.3%", current: "97.1%", target: "98.5%", delta: "+5.8%", sample: "n=1,760" },
  { layer: "Runtime & Reliability", label: "rr", metric: "p95 End-to-End Latency", base: "39.8s", current: "24.6s", target: "≤20.0s", delta: "-15.2s", sample: "n=9,500" },
  { layer: "Runtime & Reliability", label: "rr", metric: "Platform Uptime", base: "99.08%", current: "99.62%", target: "99.90%", delta: "+0.54%", sample: "30d" },
];

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function computeBars(base: number, now: number, target: number, lower = false) {
  if (lower) {
    return {
      baseWidth: Math.max(20, (target / base) * 100),
      nowWidth: Math.max(30, (target / now) * 100),
      targetWidth: 100,
    };
  }

  return {
    baseWidth: base,
    nowWidth: now,
    targetWidth: target,
  };
}

export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false);
  const [activeLayer, setActiveLayer] = useState(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showHandoff, setShowHandoff] = useState(false);
  const radarRef = useRef<HTMLCanvasElement | null>(null);
  const nancySectionRef = useRef<HTMLElement | null>(null);
  const chatStartedRef = useRef(false);
  const timeoutIds = useRef<number[]>([]);

  const clearChatTimers = useCallback(() => {
    timeoutIds.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
    timeoutIds.current = [];
  }, []);

  const startChat = useCallback(() => {
    clearChatTimers();
    setMessages([]);
    setShowHandoff(false);

    let delay = 350;
    nancyChat.forEach((message) => {
      const timeoutId = window.setTimeout(() => {
        setMessages((current) => [...current, message]);
      }, delay);
      timeoutIds.current.push(timeoutId);
      delay += 700 + message.text.length * 7;
    });

    const handoffTimeout = window.setTimeout(() => {
      setShowHandoff(true);
    }, delay + 400);
    timeoutIds.current.push(handoffTimeout);
    chatStartedRef.current = true;
  }, [clearChatTimers]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" },
    );

    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const section = nancySectionRef.current;
    if (!section) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !chatStartedRef.current) {
            startChat();
          }
        });
      },
      { threshold: 0.25 },
    );

    observer.observe(section);
    return () => observer.disconnect();
  }, [startChat]);

  useEffect(() => {
    return () => clearChatTimers();
  }, [clearChatTimers]);

  useEffect(() => {
    const canvas = radarRef.current;
    if (!canvas) {
      return;
    }

    const ratio = window.devicePixelRatio || 1;
    const width = 340;
    const height = 300;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2 - 10;
    const radius = 110;
    const labels = [
      ["Intelligence", "Quality"],
      ["Safety &", "Alignment"],
      ["Clinical", "Ops"],
      ["Patient", "Loop"],
      ["Security &", "Gov"],
      ["Runtime &", "Reliability"],
    ];
    const baseline = [72, 85, 76, 68, 82, 79];
    const current = [88, 96, 91, 84, 95, 93];
    const target = [94, 98, 95, 88, 98, 97];
    const segments = labels.length;

    const getPoint = (index: number, value: number) => {
      const angle = (Math.PI * 2 * index) / segments - Math.PI / 2;
      return {
        x: centerX + radius * (value / 100) * Math.cos(angle),
        y: centerY + radius * (value / 100) * Math.sin(angle),
      };
    };

    const drawPolygon = (values: number[], stroke: string, fill: string) => {
      context.beginPath();
      values.forEach((value, index) => {
        const point = getPoint(index, value);
        if (index === 0) {
          context.moveTo(point.x, point.y);
        } else {
          context.lineTo(point.x, point.y);
        }
      });
      context.closePath();
      context.strokeStyle = stroke;
      context.lineWidth = 2;
      context.fillStyle = fill;
      context.fill();
      context.stroke();
    };

    [20, 40, 60, 80, 100].forEach((value) => {
      context.beginPath();
      for (let index = 0; index < segments; index += 1) {
        const point = getPoint(index, value);
        if (index === 0) {
          context.moveTo(point.x, point.y);
        } else {
          context.lineTo(point.x, point.y);
        }
      }
      context.closePath();
      context.strokeStyle = "rgba(16,33,43,0.08)";
      context.lineWidth = 1;
      context.stroke();
    });

    for (let index = 0; index < segments; index += 1) {
      const point = getPoint(index, 100);
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.lineTo(point.x, point.y);
      context.strokeStyle = "rgba(16,33,43,0.1)";
      context.lineWidth = 1;
      context.stroke();
    }

    drawPolygon(baseline, "rgba(16,33,43,0.3)", "rgba(16,33,43,0.05)");
    drawPolygon(current, "rgba(15,118,110,0.9)", "rgba(15,118,110,0.1)");
    drawPolygon(target, "rgba(198,106,26,0.65)", "rgba(198,106,26,0.06)");

    context.font = 'bold 10px "Avenir Next", "Segoe UI", sans-serif';
    context.fillStyle = "rgba(16,33,43,0.6)";
    context.textAlign = "center";
    labels.forEach((label, index) => {
      const point = getPoint(index, 122);
      label.forEach((line, lineIndex) => {
        context.fillText(line, point.x, point.y + lineIndex * 12 - (label.length - 1) * 6);
      });
    });

    context.font = 'bold 11px "Trebuchet MS", sans-serif';
    context.fillStyle = "rgba(15,118,110,0.85)";
    current.forEach((value, index) => {
      const point = getPoint(index, value);
      context.fillText(`${value}%`, point.x, point.y - 4);
    });
  }, []);

  return (
    <>
      <div className="landing-root">
        <nav className={cx("nav", scrolled && "scrolled")}>
          <div className="nav-brand">
            <div className="nav-mark">M</div>
            <span className="nav-name">MindScape</span>
          </div>
          <div className="nav-links">
            <Link href="/doctor" className="nav-ghost">
              For Clinicians
            </Link>
            <Link href="/patient" className="nav-solid">
              For Patients
            </Link>
          </div>
        </nav>

        <section className="hero">
          <div className="hero-glow" />
          <div className="hero-noise" />
          <div className="hero-content">
            <div className="hero-eyebrow">
              <div className="hero-eyebrow-dot" />
              Mental Health Clinical Operating System
            </div>
            <h1 className="hero-hl">
              Care that doesn&apos;t end when the <em>session</em> does.
            </h1>
            <div className="hero-row">
              <p className="hero-sub">
                MindScape connects live sessions, between-session support, risk signals, and peer
                clinical intelligence into one continuous care system — built for clinicians,
                designed for patients.
              </p>
              <div className="hero-ctas">
                <Link href="/doctor" className="hero-btn hero-btn--p">
                  Enter as a Clinician
                </Link>
                <Link href="/patient" className="hero-btn hero-btn--s">
                  I&apos;m a Patient
                </Link>
              </div>
            </div>
          </div>
          <div className="hero-scroll-hint">
            <span>Scroll</span>
            <div className="hero-scroll-line" />
          </div>
        </section>

        <div className="ticker-wrap">
          <div className="ticker-row">
            <div className="ticker-track">
              {[...tickerOne, ...tickerOne].map((item, index) => (
                <span
                  className="t-item"
                  dangerouslySetInnerHTML={{ __html: item }}
                  key={`t1-${index}`}
                />
              ))}
            </div>
          </div>
          <div className="ticker-row">
            <div className="ticker-track ticker-track--r">
              {[...tickerTwo, ...tickerTwo].map((item, index) => (
                <span
                  className="t-item"
                  dangerouslySetInnerHTML={{ __html: item }}
                  key={`t2-${index}`}
                />
              ))}
            </div>
          </div>
        </div>

        <section className="problem" id="problem">
          <div className="section-inner">
            <p className="ew ew--teal rv" data-reveal>
              The broken system
            </p>
            <h2 className="problem-hl rv rv--d2" data-reveal>
              Mental healthcare doesn&apos;t fail in the clinic. It fails between visits.
            </h2>
            <p className="problem-body rv rv--d3" data-reveal>
              Delayed diagnoses, fragmented records, and unsupported intervals erode what good
              sessions build. The signals that matter most arrive between appointments — and
              disappear there too.
            </p>
            <div className="problem-stats">
              <div className="p-stat rv rv--d2" data-reveal>
                <div className="p-num">11 yrs</div>
                <p className="p-label">
                  Average gap between first symptoms and first adequate treatment — a delay that
                  compounds silently across every untracked month.
                </p>
              </div>
              <div className="p-stat rv rv--d3" data-reveal>
                <div className="p-num">43%</div>
                <p className="p-label">
                  Of adults living with a mental health condition receive any care in a given year.
                  The majority navigate without clinical support.
                </p>
              </div>
              <div className="p-stat rv rv--d4" data-reveal>
                <div className="p-num">~50%</div>
                <p className="p-label">
                  Of patients stop treatment within the first year — often because care continuity
                  breaks down before trust has a chance to form.
                </p>
              </div>
            </div>
            <div className="problem-quote rv rv--d2" data-reveal>
              <blockquote>
                &quot;The most consequential moments in mental healthcare happen between appointments —
                and most systems treat them as silence.&quot;
              </blockquote>
            </div>
          </div>
        </section>

        <section className="thesis">
          <div className="thesis-inner">
            <p className="ew ew--light rv" data-reveal>
              The answer
            </p>
            <h2 className="thesis-hl rv rv--d2" data-reveal>
              A clinical operating system that keeps the <span>signal intact.</span>
            </h2>
            <p className="thesis-sub rv rv--d3" data-reveal>
              From live sessions to between-session care, from risk signals to peer intelligence —
              MindScape builds longitudinal clinical context that doesn&apos;t evaporate when the hour
              ends.
            </p>
            <div className="thesis-pills rv rv--d4" data-reveal>
              <span className="thesis-pill">Async care loop</span>
              <span className="thesis-pill">Clinical session loop</span>
              <span className="thesis-pill">Escalation &amp; review loop</span>
            </div>
          </div>
        </section>

        <section className="layers" id="layers">
          <div className="layers-hd">
            <p className="ew ew--light rv" data-reveal>
              The system, revealed
            </p>
            <h2 className="layers-hl rv rv--d2" data-reveal>
              One operating system. Three connected loops.
            </h2>
          </div>
          <div className="layers-tabs">
            {[
              { number: "Loop 01", name: "Async Care" },
              { number: "Loop 02", name: "Session Intelligence" },
              { number: "Loop 03", name: "Escalation & Review" },
            ].map((tab, index) => (
              <button
                aria-pressed={activeLayer === index}
                className={cx("l-tab", activeLayer === index && "on")}
                key={tab.name}
                onClick={() => setActiveLayer(index)}
                type="button"
              >
                <span className="l-tab-n">{tab.number}</span>
                <span className="l-tab-name">{tab.name}</span>
              </button>
            ))}
          </div>

          <div className="layers-body">
            <div className="l-panel panel-enter" key={activeLayer}>
              <div className="l-panel-copy">
                <p className="l-panel-ew">{layerPanels[activeLayer].eyebrow}</p>
                <h3 className="l-panel-hl">{layerPanels[activeLayer].heading}</h3>
                <p className="l-panel-body">{layerPanels[activeLayer].body}</p>
                <div className="l-feats">
                  {layerPanels[activeLayer].features.map((feature) => (
                    <div className="l-feat" key={feature}>
                      <div className="l-feat-d" />
                      <span>{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="l-panel-vis">
                {activeLayer === 0 ? (
                  <div className="mock mock--nancy">
                    <div className="mock-hd">
                      <div className="mock-av mock-av--n">N</div>
                      <div>
                        <div className="mock-av-name">Nancy</div>
                        <div className="mock-av-sub">Daily check-in · Between-session support</div>
                      </div>
                    </div>
                    <div className="mb mb--n">
                      <div className="mb-lbl">Nancy</div>
                      Good morning. Dr. Chen asked me to follow up on your sleep this week — how
                      has it been? Take your time.
                    </div>
                    <div className="mb mb--p">
                      <div className="mb-lbl">You</div>
                      Better, honestly. Maybe a 6 or 7. The breathing exercise has been helping
                      before bed.
                    </div>
                    <div className="mb mb--n">
                      <div className="mb-lbl">Nancy</div>
                      That&apos;s meaningful progress. I&apos;ll make sure Dr. Chen sees this before your
                      next session. Anything else you&apos;d like her to know?
                    </div>
                    <div className="mock-handoff">
                      <div className="mock-handoff-lbl">Clinician handoff generated</div>
                      Sleep improved to 6–7/10. Positive response to breathing exercise. Mood
                      stable. No safety concerns flagged. Ready for review.
                    </div>
                  </div>
                ) : null}

                {activeLayer === 1 ? (
                  <div className="mock mock--session">
                    <div className="mock-hd">
                      <div className="mock-av mock-av--s">S</div>
                      <div>
                        <div className="mock-av-name">Session · Dr. Chen / Patient #04</div>
                        <div className="mock-av-sub">Uploaded audio · 48 min · Analyzed</div>
                      </div>
                      <div className="mock-conf">87% confidence</div>
                    </div>
                    <div className="mf">
                      <div className="mf-lbl">Transcript excerpt</div>
                      <div className="mock-tx">
                        &quot;Lately even thinking about crowded spaces sets it off. My body reacts
                        before I can reason with it.&quot;
                      </div>
                    </div>
                    <div className="mf">
                      <div className="mf-lbl">Diagnostic hypothesis</div>
                      <div className="mf-val">
                        Panic Disorder with agoraphobic features. Anticipatory anxiety component
                        emerging. Prior CBT evidence supports interoceptive exposure as next
                        modality.
                      </div>
                    </div>
                    <div className="mf">
                      <div className="mf-lbl">Affect trajectory</div>
                      <div className="affect-row">
                        <span className="aff-chip aff-c">Calm</span>
                        <span className="aff-arr">→</span>
                        <span className="aff-chip aff-a">Anxious</span>
                        <span className="aff-arr">→</span>
                        <span className="aff-chip aff-c">Regulated</span>
                      </div>
                    </div>
                    <div className="mf">
                      <div className="mf-lbl">Retrieved evidence</div>
                      <div className="mf-val mf-small">
                        Clark &amp; Beck (2010) — Cognitive Therapy for Anxiety · Craske et al.
                        (2014) — Maximizing Exposure Therapy
                      </div>
                    </div>
                  </div>
                ) : null}

                {activeLayer === 2 ? (
                  <div className="alerts-stack">
                    <div className="al al--u">
                      <div className="al-hd">
                        <span className="al-badge">Urgent</span>
                        <span className="al-time">4 min ago</span>
                      </div>
                      <div className="al-title">Safety language flagged in daily check-in</div>
                      <p className="al-body">
                        Patient reported feeling &quot;not worth it&quot; during Nancy conversation.
                        Immediate clinician review recommended before next contact.
                      </p>
                    </div>
                    <div className="al al--w">
                      <div className="al-hd">
                        <span className="al-badge">Watch</span>
                        <span className="al-time">2 hrs ago</span>
                      </div>
                      <div className="al-title">
                        Mood score declined across three consecutive check-ins
                      </div>
                      <p className="al-body">
                        Downward trend: 8 → 5 → 3. Sleep disruption also reported. Suggested:
                        schedule session within 48 hours.
                      </p>
                    </div>
                    <div className="al al--r">
                      <div className="al-hd">
                        <span className="al-badge">Routine</span>
                        <span className="al-time">Yesterday</span>
                      </div>
                      <div className="al-title">Daily check-in missed — follow-up queued</div>
                      <p className="al-body">
                        Patient did not complete check-in. Nancy has queued a gentle outreach. No
                        concerning content in prior session record.
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </section>

        <section className="nancy" id="nancy" ref={nancySectionRef}>
          <div className="nancy-inner">
            <div>
              <p className="ew ew--amber rv" data-reveal>
                Nancy AI
              </p>
              <h2 className="nancy-hl rv rv--d2" data-reveal>
                Nancy is not a chatbot.
              </h2>
              <p className="nancy-body rv rv--d3" data-reveal>
                She&apos;s a care companion trained to support patients between visits — gathering
                structured daily signals, following doctor-authored directives, and relaying what
                matters to clinicians in their language. Every interaction is part of a clinical
                record.
              </p>
              <div className="n-feats rv rv--d4" data-reveal>
                {[
                  {
                    number: "01",
                    title: "Daily structured check-ins",
                    body:
                      "Natural conversation that captures mood, sleep, anxiety, cognition, and medication signals — structured for clinical use, not raw chat logs.",
                  },
                  {
                    number: "02",
                    title: "Doctor-authored directives",
                    body:
                      "Clinicians write specific follow-up tasks Nancy incorporates into future patient conversations between sessions.",
                  },
                  {
                    number: "03",
                    title: "Clinician-ready handoffs",
                    body:
                      "Every interaction produces a structured clinical summary — what Nancy observed, what the patient shared, what needs review.",
                  },
                  {
                    number: "04",
                    title: "Voice and text, two distinct modes",
                    body:
                      "Daily check-in and open-ended support run as separate conversation lanes — clear to the patient, structured for the clinician.",
                  },
                ].map((feature) => (
                  <div className="n-feat" key={feature.title}>
                    <div className="n-feat-ic">{feature.number}</div>
                    <div>
                      <div className="n-feat-title">{feature.title}</div>
                      <p className="n-feat-body">{feature.body}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="rv rv--d2" data-reveal>
              <button className="nancy-chat" onClick={startChat} title="Click to replay" type="button">
                <div className="nc-hd">
                  <div className="nc-av">N</div>
                  <div>
                    <div className="nc-name">Nancy</div>
                    <div className="nc-status">Daily check-in · Active</div>
                  </div>
                  <div className="nc-badge">Daily</div>
                </div>
                <div className="nc-msgs">
                  {messages.map((message, index) => (
                    <div className={cx("c-bbl", message.role === "n" ? "c-bbl--n" : "c-bbl--p")} key={`${message.role}-${index}-${message.text.slice(0, 12)}`}>
                      <div className="c-lbl">{message.role === "n" ? "Nancy" : "You"}</div>
                      {message.text}
                    </div>
                  ))}
                  {showHandoff ? (
                    <div className="c-handoff">
                      <div className="c-ho-d" />
                      <span>{handoffText}</span>
                    </div>
                  ) : null}
                </div>
                <div className="nc-hint">Click to replay conversation</div>
              </button>
            </div>
          </div>
        </section>

        <section className="session" id="session">
          <div className="session-inner">
            <div>
              <p className="ew ew--light rv" data-reveal>
                Session intelligence
              </p>
              <h2 className="session-hl rv rv--d2" data-reveal>
                Every session becomes a structured clinical record.
              </h2>
              <p className="session-body rv rv--d3" data-reveal>
                MindScape&apos;s diagnostic engine processes live or uploaded audio into transcript,
                affect analysis, hypothesis, retrieved evidence, and a reusable session record —
                before the next patient arrives.
              </p>
              <div className="s-feats rv rv--d4" data-reveal>
                {[
                  "Transcription with speaker attribution",
                  "Acoustic affect extraction and emotion trajectory",
                  "Diagnostic hypothesis with confidence score and reasoning",
                  "Retrieval over clinical literature for evidence-backed output",
                  "Treatment plan suggestion and follow-up question generation",
                  "All output persisted to the patient's longitudinal record",
                ].map((feature) => (
                  <div className="s-feat" key={feature}>
                    <div className="s-dot" />
                    <span>{feature}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="s-vis rv rv--d2" data-reveal>
              <div className="s-metrics">
                {[
                  { value: "87%", label: "Hypothesis confidence" },
                  { value: "14", label: "Evidence sources" },
                  { value: "3", label: "Follow-up questions" },
                ].map((metric) => (
                  <div className="s-met" key={metric.label}>
                    <div className="s-met-v">{metric.value}</div>
                    <div className="s-met-l">{metric.label}</div>
                  </div>
                ))}
              </div>
              <div className="s-card">
                <div className="s-card-lbl">Transcript excerpt</div>
                <div className="s-card-tx">
                  &quot;Lately even thinking about crowded spaces sets it off. My body reacts before I
                  can reason with it.&quot;
                </div>
              </div>
              <div className="s-card">
                <div className="s-card-lbl">Diagnostic hypothesis</div>
                <div className="s-card-val">
                  Panic Disorder with agoraphobic features. Anticipatory anxiety component emerging.
                  CBT evidence supports interoceptive exposure as next modality.
                </div>
              </div>
              <div className="s-card">
                <div className="s-card-lbl">Retrieved evidence</div>
                <div className="s-ev-row">
                  <span className="s-ev">
                    Clark &amp; Beck (2010) — Cognitive Therapy for Anxiety Disorders
                  </span>
                  <span className="s-ev">
                    Craske et al. (2014) — Maximizing Exposure Therapy: An Inhibitory Learning
                    Approach
                  </span>
                  <span className="s-ev">
                    Barlow et al. (2017) — Unified Protocol for Transdiagnostic Treatment
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="corner" id="corner">
          <div className="section-inner">
            <div className="corner-hd">
              <div>
                <p className="ew ew--teal rv" data-reveal>
                  Doctor&apos;s Corner
                </p>
                <h2 className="corner-hl rv rv--d2" data-reveal>
                  A clinical intelligence network for mental health professionals.
                </h2>
              </div>
              <p className="corner-body rv rv--d2" data-reveal>
                Peer case exchange, research threads, hypothesis crowdsourcing, and a curated
                weekly brief — built for the rigor and nuance that mental health work demands. Not
                generic community noise.
              </p>
            </div>
            <div className="corner-threads">
              {[
                {
                  flair: "Research",
                  flairClass: "ct-fr--research",
                  title:
                    "Does prolonged exposure actually outperform CPT for complex PTSD, or is the comparison confounded by dropout rates?",
                  body:
                    "Revisiting the landmark trials — the dropout problem may be masking treatment equivalence in the completers-only analysis. Looking for anyone who has run similar numbers.",
                  meta: ["Dr. M. Okonkwo", "24 replies", "3 hrs ago"],
                  revealClass: "",
                },
                {
                  flair: "Hypothesis",
                  flairClass: "ct-fr--hyp",
                  title:
                    "Anticipatory anxiety as a diagnostic marker earlier in GAD presentation — is there a reliable screening window?",
                  body:
                    "Proposing a structured intake protocol that probes anticipatory patterns before the standard GAD-7. Looking for peer critique and methods input before piloting.",
                  meta: ["Dr. S. Varga", "11 replies", "5 hrs ago"],
                  revealClass: "rv--d2",
                },
                {
                  flair: "Teaching case",
                  flairClass: "ct-fr--case",
                  title:
                    "Misdiagnosed bipolar II presenting as treatment-resistant depression — what the medication history revealed",
                  body:
                    "De-identified. Three years of antidepressant cycling before the mood-stabilizer switch. What the timeline looked like and how we finally got there.",
                  meta: ["Dr. A. Chen", "38 replies", "Yesterday"],
                  revealClass: "rv--d3",
                },
              ].map((thread) => (
                <div className={cx("ct", "rv", thread.revealClass)} data-reveal key={thread.title}>
                  <span className={cx("ct-flair", thread.flairClass)}>{thread.flair}</span>
                  <div className="ct-title">{thread.title}</div>
                  <p className="ct-body">{thread.body}</p>
                  <div className="ct-meta">
                    {thread.meta.map((item, index) => (
                      <div className="ct-meta-item" key={`${thread.title}-${item}`}>
                        {index > 0 ? <div className="ct-dot" /> : null}
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="arch" id="arch">
          <div className="section-inner">
            <div className="arch-intro">
              <div>
                <p className="ew ew--light rv" data-reveal>
                  How it works
                </p>
                <h2 className="arch-hl rv rv--d2" data-reveal>
                  Three loops. One continuous clinical record.
                </h2>
              </div>
              <p className="arch-body rv rv--d2" data-reveal>
                MindScape&apos;s three care loops are architecturally connected — patient context
                flows from sessions into Nancy, from Nancy into clinical alerts, and from alerts
                back into the clinician&apos;s workspace. Nothing is siloed.
              </p>
            </div>
            <div className="arch-loops">
              {[
                {
                  number: "01",
                  name: "Async Care Loop",
                  body:
                    "Between-session care, continuous and structured. Patient signals reach the right clinician without anyone manually reconstructing the week.",
                  items: [
                    "Daily patient check-ins via Nancy",
                    "Open support conversations",
                    "Structured clinician handoffs",
                    "Doctor-authored Nancy directives",
                  ],
                },
                {
                  number: "02",
                  name: "Session Intelligence Loop",
                  body:
                    "Every session is an input, not just a note. Audio becomes transcript, hypothesis, evidence, and a structured record the next session builds on.",
                  items: [
                    "Transcription and affect extraction",
                    "Diagnostic hypothesis synthesis",
                    "Clinical literature retrieval",
                    "Clinician session notes and review",
                  ],
                },
                {
                  number: "03",
                  name: "Escalation Loop",
                  body:
                    "When async signals shift, the system responds with a structured escalation path — not a notification badge. Clinicians see what matters and act with full context.",
                  items: [
                    "Risk-classified alert levels",
                    "Structured review and action queue",
                    "Doctor-triggered escalation records",
                    "Auditable escalation trail",
                  ],
                },
              ].map((loop, index) => (
                <div className={cx("a-loop", "rv", index === 1 ? "rv--d2" : index === 2 ? "rv--d3" : "")} data-reveal key={loop.name}>
                  <div className="a-n">{loop.number}</div>
                  <div className="a-name">{loop.name}</div>
                  <p className="a-body">{loop.body}</p>
                  <div className="a-items">
                    {loop.items.map((item) => (
                      <div className="a-item" key={item}>
                        <div className="a-item-d" />
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="tech-arch" id="engine">
          <div className="ta-hd">
            <p className="ew ew--light rv" data-reveal>
              MindScape Engine · Technical Architecture
            </p>
            <h2 className="ta-title rv rv--d2" data-reveal>
              Seven layers from signal capture
              <br />
              to clinical-grade reasoning.
            </h2>
            <p className="ta-sub rv rv--d3" data-reveal>
              An exploded view of the MindScape inference pipeline — from raw WebRTC audio through
              multi-modal perception, hybrid retrieval, and council reasoning, to an independently
              gated clinical output. Built for precision. Governed by independence.
            </p>
          </div>

          <div className="zone-legend rv rv--d2" data-reveal>
            <div className="zl-item">
              <div className="zl-dot zl-dot--capture" />
              L1 Client Capture
            </div>
            <div className="zl-item">
              <div className="zl-dot zl-dot--perception" />
              L2 Multi-Modal Perception
            </div>
            <div className="zl-item">
              <div className="zl-dot zl-dot--fusion" />
              L3 Behavioral Fusion
            </div>
            <div className="zl-item">
              <div className="zl-dot zl-dot--rag" />
              L4 Hybrid RAG
            </div>
            <div className="zl-item">
              <div className="zl-dot zl-dot--reason" />
              L5 Reasoning Core
            </div>
            <div className="zl-item">
              <div className="zl-dot zl-dot--safety" />
              L6 Safety Gate
            </div>
            <div className="zl-item">
              <div className="zl-dot zl-dot--output" />
              L7 Clinician Interface
            </div>
          </div>

          <div className="pipeline-scroll rv rv--d2" data-reveal>
            <div className="pipeline-row">
              {[
                {
                  className: "lc--l1",
                  number: "L1",
                  name: "Client Capture",
                  zone: "Ingestion",
                  components: [
                    { name: "WebRTC Audio", sub: "16 kHz · mono\nraw PCM stream" },
                    { name: "Rolling Buffer", sub: "500ms – 1000ms\nadaptive window" },
                  ],
                  io: "PCM frame chunks",
                },
                {
                  className: "lc--l2",
                  number: "L2",
                  name: "Multi-Modal Perception",
                  zone: "Acoustic + Affective",
                  components: [
                    { name: "SenseVoice Small", sub: "ASR + Events\nBranch A: Acoustic" },
                    { name: "MedCPT", sub: "768-dim\nclinical embedding" },
                    { name: "Emotion2Vec", sub: "Valence / Arousal\nBranch B: Affective" },
                  ],
                  io: "768-dim + 1024-dim vectors",
                },
                {
                  className: "lc--l3",
                  number: "L3",
                  name: "Behavioral State Fusion",
                  zone: "Cross-Modal Attention",
                  components: [
                    { name: "Event Sparse Vector", sub: "acoustic event flags\narousal delta" },
                    {
                      name: "Cross-Modal Attention Gate",
                      sub: "MedCPT 768 ⊕ Emotion2Vec 1024\nlearned fusion weights",
                    },
                    { name: "BSV Output", sub: "~1880-dim\nBehavioral State Vector" },
                  ],
                  io: "BSV ~1880-dim",
                  safety: "Safety checkpoint",
                },
                {
                  className: "lc--l4 lc-wide",
                  number: "L4",
                  name: "Precision Memory Hybrid RAG",
                  zone: "Dual-Index Retrieval",
                  components: [
                    { name: "MedCPT Float16 Encoder", sub: "query vectorisation" },
                    { name: "H-NSW Index", sub: "Semantic · approximate NN\n+ ElasticSearch BM25" },
                    { name: "BioLinkBERT-Large", sub: "Cross-Encoder re-ranking\ncontext enrichment" },
                    { name: "Top-5 Re-Ranked Evidence", sub: "Recall@5 = 100% · MRR 0.77" },
                  ],
                  io: "grounded context window",
                },
                {
                  className: "lc--l5",
                  number: "L5",
                  name: "Reasoning Core",
                  zone: "Council Chain-of-Thought",
                  components: [
                    { name: "DeepSeek-R1-Distill", sub: "Llama-8B backbone\nclinical fine-tune" },
                    { name: "LLM Council CoT", sub: "Multi-model deliberation\nSelf-Critique loop" },
                  ],
                  io: "hypothesis + plan draft",
                },
                {
                  className: "lc--l6",
                  number: "L6",
                  name: "Independent Safety Gate",
                  zone: "Validators + Confidence",
                  components: [
                    { name: "DeBERTa-v3", sub: "NLI Verifier\nfactual entailment check" },
                    { name: "DSM-5 Rule Engine", sub: "diagnostic policy\nconstraint enforcement" },
                    { name: "Confidence Gate", sub: "Score ≥ 0.9 → Green\nScore < 0.6 → Blocked" },
                    { name: "Fairness Monitor", sub: "demographic parity check\nbias signal logging" },
                  ],
                  io: "gated output packet",
                  safety: "Independent safety gate",
                },
                {
                  className: "lc--l7",
                  number: "L7",
                  name: "Clinician Interface",
                  zone: "Structured Output",
                  components: [
                    { name: "Session Record", sub: "transcript · hypothesis\nevidence · plan" },
                    { name: "Longitudinal Store", sub: "immutable audit trail\nevent lineage" },
                    { name: "Clinician UI Feedback", sub: "override signals\nRLHF loopback" },
                  ],
                  io: "feedback → L5 refinement",
                  ioArrow: "↺",
                },
              ].map((layer, index, allLayers) => (
                <div className="pipeline-segment" key={layer.number}>
                  <div className={cx("lc", layer.className)}>
                    <div className="lc-top-bar" />
                    <div className="lc-num">{layer.number}</div>
                    <div className="lc-name">{layer.name}</div>
                    <div className="lc-zone-lbl">{layer.zone}</div>
                    <hr className="lc-divider" />
                    <div className="lc-comps">
                      {layer.components.map((component) => (
                        <div className="lc-comp" key={`${layer.number}-${component.name}`}>
                          <div className="comp-n">{component.name}</div>
                          <div className="comp-s">
                            {component.sub.split("\n").map((line) => (
                              <span className="comp-line" key={`${component.name}-${line}`}>
                                {line}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="lc-io">
                      <span className="io-arrow">{layer.ioArrow ?? "→"}</span>
                      {layer.io}
                    </div>
                    {layer.safety ? <div className="safety-badge">⚑ {layer.safety}</div> : null}
                  </div>
                  {index < allLayers.length - 1 ? (
                    <div className="flow-arrow">
                      <div className="fa-track" />
                      <div className="fa-head" />
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          <div className="lifecycle-ribbon rv rv--d3" data-reveal>
            {["Capture", "Perceive", "Fuse", "Retrieve", "Reason", "Gate", "Act", "Monitor", "Improve"].map((step, index) => (
              <div className="lifecycle-fragment" key={step}>
                <div className={cx("lf-step", ["Retrieve", "Reason", "Gate"].includes(step) && "lf-step--active")}>
                  {step}
                </div>
                {index < 8 ? <div className="lf-step-sep">→</div> : null}
              </div>
            ))}
          </div>

          <div className="dc-adjacent rv rv--d3" data-reveal>
            <div className="dc-adj-card">
              <span className="dc-adj-badge">Adjacent Loop</span>
              <p className="dc-adj-text">
                Doctor&apos;s Corner operates as a peer intelligence layer feeding clinical signal
                back into the care ecosystem — separate from the four-engine core but enriching L5
                reasoning through practitioner-curated evidence, hypothesis exchange, and near-miss
                learning. Not part of the inference pipeline. Governing clinical culture, not
                clinical decisions.
              </p>
            </div>
          </div>
        </section>

        <section className="safety-sec" id="trust">
          <div className="ss-hd">
            <p className="ew ew--teal rv" data-reveal>
              AI Safety · Ethics · Alignment
            </p>
            <h2 className="ss-title rv rv--d2" data-reveal>
              Verified end-to-end. Measured against production-grade benchmarks.
            </h2>
            <p className="ss-sub rv rv--d3" data-reveal>
              MindScape&apos;s safety architecture is independently gated, auditable at every stage,
              and benchmarked across six clinical and operational layers. All figures represent
              internal pilot validation.
            </p>
            <div className="pilot-note rv rv--d4" data-reveal>
              ⚑ Pilot benchmarks — internal validation snapshot · May 2026 · n-sizes shown per
              metric
            </div>
          </div>

          <div className="kpi-grid rv rv--d2" data-reveal>
            {kpis.map((kpi) => {
              const bars = computeBars(kpi.base, kpi.now, kpi.target, kpi.lower);
              return (
                <div className="kpi-tile" key={kpi.name}>
                  <div className="kpi-name">{kpi.name}</div>
                  <div className={cx("kpi-current", kpi.good && "kpi-current--good")}>
                    {kpi.current}
                  </div>
                  <div className="kpi-stages">
                    <div className="kpi-stage">
                      <span className="ks-label">Base</span>
                      <div className="ks-bar-wrap">
                        <div className="ks-bar ks-bar--baseline" style={{ width: `${bars.baseWidth}%` }} />
                      </div>
                      <span className="ks-val">
                        {kpi.base}
                        {kpi.unit}
                      </span>
                    </div>
                    <div className="kpi-stage">
                      <span className="ks-label">Now</span>
                      <div className="ks-bar-wrap">
                        <div className="ks-bar ks-bar--current" style={{ width: `${bars.nowWidth}%` }} />
                      </div>
                      <span className="ks-val">
                        {kpi.now}
                        {kpi.unit}
                      </span>
                    </div>
                    <div className="kpi-stage">
                      <span className="ks-label">Target</span>
                      <div className="ks-bar-wrap">
                        <div className="ks-bar ks-bar--target" style={{ width: `${bars.targetWidth}%` }} />
                      </div>
                      <span className="ks-val">
                        {kpi.lower ? "≤" : ""}
                        {kpi.target}
                        {kpi.unit}
                      </span>
                    </div>
                  </div>
                  <div className="kpi-n">{kpi.sample} · ±2.1% CI</div>
                </div>
              );
            })}
          </div>

          <div className="chart-row rv rv--d2" data-reveal>
            <div className="chart-card">
              <div className="chart-title">6-Layer Benchmark Maturity</div>
              <div className="chart-sub">
                Baseline → Current → Target across all six architectural quality dimensions. Pilot
                validation · internal snapshot.
              </div>
              <div className="radar-wrap">
                <canvas height={300} ref={radarRef} width={340} />
              </div>
              <div className="radar-legend">
                <div className="rl-item">
                  <div className="rl-dot" style={{ background: "rgba(16,33,43,0.25)" }} />
                  Baseline
                </div>
                <div className="rl-item">
                  <div className="rl-dot" style={{ background: "#0f766e" }} />
                  Current
                </div>
                <div className="rl-item">
                  <div className="rl-dot" style={{ background: "rgba(198,106,26,0.65)" }} />
                  Target
                </div>
              </div>
            </div>
            <div className="chart-card">
              <div className="chart-title">Safety Guardrail Uplift — Baseline → Current</div>
              <div className="chart-sub">
                Waterfall of key safety metric gains from early prototype to current pilot state.
              </div>
              <div className="waterfall">
                {waterfallRows.map((row) => {
                  const delta = row.lower
                    ? (row.base - row.current).toFixed(1)
                    : (row.current - row.base).toFixed(1);
                  const baseWidth = row.lower
                    ? ((row.max - row.base) / row.max) * 100
                    : (row.base / row.max) * 100;
                  const gainWidth = row.lower
                    ? ((row.base - row.current) / row.max) * 100
                    : ((row.current - row.base) / row.max) * 100;

                  return (
                    <div className="wf-row" key={row.label}>
                      <span className="wf-label">{row.label}</span>
                      <div className="wf-bar-wrap">
                        <div className="wf-bar-bg" />
                        <div className="wf-bar-fill wf-bar--base" style={{ width: `${baseWidth.toFixed(1)}%` }} />
                        <div className="wf-bar-fill wf-bar--gain" style={{ width: `${(baseWidth + gainWidth).toFixed(1)}%` }}>
                          +{delta}
                          {row.unit}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="benchmark-wrap rv rv--d2" data-reveal>
            <div className="bm-title">6-Layer Benchmark Taxonomy — Baseline · Current · Target</div>
            <div className="bm-sub">
              Architecture-review grade coverage across intelligence quality, safety, operations,
              patient loop, governance, and runtime reliability.
            </div>
            <table className="bm-table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Metric</th>
                  <th>Baseline</th>
                  <th>Current</th>
                  <th>Target</th>
                  <th>Δ Gain</th>
                  <th>Sample</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkRows.map((row) => (
                  <tr key={`${row.layer}-${row.metric}`}>
                    <td>
                      <span className={cx("bm-layer", `bml--${row.label}`)}>{row.layer}</span>
                    </td>
                    <td className="bm-metric">
                      {row.metric}
                      {row.note ? <span className="bm-note"> ({row.note})</span> : null}
                    </td>
                    <td className="bm-val bm-base">{row.base}</td>
                    <td className="bm-val bm-current">{row.current}</td>
                    <td className="bm-val bm-target">{row.target}</td>
                    <td>
                      <span
                        className={cx(
                          "bm-delta",
                          row.delta.startsWith("-") ? "bm-delta--down" : "bm-delta--up",
                        )}
                      >
                        {row.delta}
                      </span>
                    </td>
                    <td className="bm-sample">{row.sample}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="limitation-card rv rv--d3" data-reveal>
            <div className="lim-head">⚠ Known Limitation — Current Pilot Scope</div>
            <p className="lim-body">
              All benchmarks reflect internal pilot validation on a controlled dataset (n as
              shown). Real-world clinical deployment will introduce distribution shift, multi-site
              variability, and edge-case volumes not yet represented. LLM Diagnostic Accuracy
              (exact-match) is currently 60.0% — intentionally conservative, reflecting the
              difficulty of exact-match evaluation on open-ended clinical reasoning. Human oversight
              and clinician review remain mandatory for all safety-critical decisions. Production
              readiness certification requires independent external audit.
            </p>
          </div>
        </section>

        <section className="cta" id="cta">
          <div className="section-inner">
            <div className="cta-hd">
              <p className="ew ew--teal rv" data-reveal>
                Where would you like to begin?
              </p>
              <h2 className="cta-hl rv rv--d2" data-reveal>
                Two portals. One care system.
              </h2>
            </div>
            <div className="cta-cards rv rv--d2" data-reveal>
              <Link href="/doctor" className="cc cc--doc">
                <span className="cc-tag">Clinician</span>
                <div className="cta-card-title">The operating workspace</div>
                <p className="cta-card-body">
                  Patient panel, session intelligence, Nancy orchestration, clinical alerts, and
                  Doctor&apos;s Corner — everything a clinician needs, in one connected environment.
                </p>
                <span className="cc-btn">Enter as a Clinician</span>
              </Link>
              <Link href="/patient" className="cc cc--pat">
                <span className="cc-tag">Patient</span>
                <div className="cta-card-title">Your care companion</div>
                <p className="cta-card-body">
                  Daily check-ins, messaging, and Nancy support — a calm space to stay connected
                  with your care between visits, at your own pace.
                </p>
                <span className="cc-btn">Enter as a Patient</span>
              </Link>
            </div>
          </div>
        </section>

        <footer>
          <div className="ft-brand">
            <div className="ft-mark">M</div>
            <span className="ft-name">MindScape Clinical OS</span>
          </div>
          <span className="ft-copy">Connecting care between sessions.</span>
        </footer>
      </div>

      <style jsx global>{`
        html {
          scroll-behavior: smooth;
        }

        body {
          margin: 0;
          background: #f7f2e8;
          color: #10212b;
        }
      `}</style>

      <style jsx>{`
        .landing-root {
          --navy: #0d1f2d;
          --navy-mid: #16384d;
          --navy-light: #1e4d69;
          --teal: #0f766e;
          --teal-light: #0d9488;
          --amber: #c66a1a;
          --rose: #b5475c;
          --cream: #f7f2e8;
          --surface: #fffcf7;
          --ink: #10212b;
          --muted: #5f6b75;
          --line: rgba(16, 33, 43, 0.1);
          --font-d: "Trebuchet MS", "Arial Narrow", sans-serif;
          --font-b: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
          font-family: var(--font-b);
          background: var(--cream);
          color: var(--ink);
          overflow-x: hidden;
        }

        .landing-root :global(*) {
          box-sizing: border-box;
        }

        .landing-root :global(a) {
          color: inherit;
          text-decoration: none;
        }

        .landing-root :global(button) {
          font: inherit;
          cursor: pointer;
          border: none;
          background: none;
        }

        .nav {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 100;
          padding: 1.1rem 2.5rem;
          display: flex;
          align-items: center;
          justify-content: space-between;
          transition: background 0.35s, backdrop-filter 0.35s, border-color 0.35s;
          border-bottom: 1px solid transparent;
        }

        .nav.scrolled {
          background: rgba(10, 20, 30, 0.88);
          backdrop-filter: blur(22px);
          border-color: rgba(255, 255, 255, 0.06);
        }

        .nav-brand {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: white;
        }

        .nav-mark {
          width: 2.2rem;
          height: 2.2rem;
          border-radius: 0.65rem;
          background: linear-gradient(135deg, var(--navy-mid), var(--teal));
          display: grid;
          place-items: center;
          font-family: var(--font-d);
          font-weight: 800;
          font-size: 0.95rem;
          color: white;
        }

        .nav-name {
          font-family: var(--font-d);
          font-weight: 700;
          font-size: 1rem;
          letter-spacing: -0.01em;
        }

        .nav-links {
          display: flex;
          gap: 0.5rem;
          align-items: center;
        }

        .nav-ghost {
          padding: 0.55rem 1.1rem;
          border-radius: 999px;
          font-size: 0.84rem;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.72);
          border: 1px solid rgba(255, 255, 255, 0.18);
          background: rgba(255, 255, 255, 0.06);
          transition: opacity 0.2s, transform 0.2s;
        }

        .nav-ghost:hover {
          opacity: 0.85;
          transform: translateY(-1px);
        }

        .nav-solid {
          padding: 0.55rem 1.25rem;
          border-radius: 999px;
          font-size: 0.84rem;
          font-weight: 700;
          background: var(--teal);
          color: white;
          transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s;
          box-shadow: 0 4px 18px rgba(15, 118, 110, 0.35);
        }

        .nav-solid:hover {
          opacity: 0.9;
          transform: translateY(-1px);
        }

        .hero {
          min-height: 100vh;
          background: var(--navy);
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          padding: 0 2.5rem 6rem;
          position: relative;
          overflow: hidden;
        }

        .hero-glow {
          position: absolute;
          inset: 0;
          pointer-events: none;
          background:
            radial-gradient(ellipse 70% 60% at 20% 55%, rgba(15, 118, 110, 0.28) 0%, transparent 60%),
            radial-gradient(ellipse 50% 40% at 82% 18%, rgba(198, 106, 26, 0.14) 0%, transparent 50%),
            radial-gradient(ellipse 40% 50% at 65% 75%, rgba(15, 118, 110, 0.12) 0%, transparent 55%);
          animation: glowBreath 9s ease-in-out infinite alternate;
        }

        .hero-noise {
          position: absolute;
          inset: 0;
          pointer-events: none;
          opacity: 0.03;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
        }

        .hero-content {
          position: relative;
          z-index: 1;
          max-width: 1240px;
          margin: 0 auto;
          width: 100%;
        }

        .hero-eyebrow {
          display: inline-flex;
          align-items: center;
          gap: 0.65rem;
          padding: 0.38rem 1rem;
          border-radius: 999px;
          margin-bottom: 2.25rem;
          border: 1px solid rgba(15, 118, 110, 0.4);
          background: rgba(15, 118, 110, 0.1);
          color: rgba(15, 183, 172, 0.85);
          font-size: 0.74rem;
          font-weight: 700;
          letter-spacing: 0.14em;
          text-transform: uppercase;
        }

        .hero-eyebrow-dot {
          width: 0.45rem;
          height: 0.45rem;
          border-radius: 50%;
          background: var(--teal-light);
          box-shadow: 0 0 8px rgba(13, 148, 136, 0.7);
          animation: pulse 2.2s ease-in-out infinite;
        }

        .hero-hl {
          font-family: var(--font-d);
          font-size: clamp(3.8rem, 8.5vw, 8rem);
          line-height: 0.92;
          letter-spacing: -0.025em;
          color: white;
          max-width: 14ch;
          margin-bottom: 2rem;
        }

        .hero-hl em {
          font-style: normal;
          color: rgba(13, 200, 190, 0.85);
        }

        .hero-row {
          display: flex;
          gap: 5rem;
          align-items: flex-end;
          flex-wrap: wrap;
        }

        .hero-sub {
          font-size: clamp(0.95rem, 1.5vw, 1.15rem);
          line-height: 1.75;
          color: rgba(255, 255, 255, 0.58);
          max-width: 46ch;
          flex-shrink: 0;
        }

        .hero-ctas {
          display: flex;
          gap: 0.85rem;
          flex-wrap: wrap;
          flex-shrink: 0;
        }

        .hero-btn {
          padding: 0.95rem 2rem;
          border-radius: 999px;
          font-weight: 700;
          font-size: 0.96rem;
          transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
          display: inline-block;
        }

        .hero-btn:hover {
          transform: translateY(-2px);
        }

        .hero-btn--p {
          background: var(--teal);
          color: white;
          box-shadow: 0 8px 28px rgba(15, 118, 110, 0.45);
        }

        .hero-btn--p:hover {
          box-shadow: 0 12px 36px rgba(15, 118, 110, 0.6);
        }

        .hero-btn--s {
          background: rgba(255, 255, 255, 0.09);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .hero-btn--s:hover {
          background: rgba(255, 255, 255, 0.15);
        }

        .hero-scroll-hint {
          position: absolute;
          bottom: 2.25rem;
          right: 2.5rem;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.5rem;
          color: rgba(255, 255, 255, 0.25);
          font-size: 0.72rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .hero-scroll-line {
          width: 1px;
          height: 2.5rem;
          background: linear-gradient(to bottom, rgba(255, 255, 255, 0.25), transparent);
          animation: scrollDrop 2.4s ease-in-out infinite;
        }

        .ticker-wrap {
          background: var(--navy);
          border-top: 1px solid rgba(255, 255, 255, 0.05);
          overflow: hidden;
        }

        .ticker-row {
          display: flex;
          overflow: hidden;
          padding: 0.8rem 0;
          border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .ticker-track {
          display: flex;
          gap: 0;
          flex-shrink: 0;
          white-space: nowrap;
          animation: tickL 55s linear infinite;
        }

        .ticker-track--r {
          animation: tickR 70s linear infinite;
        }

        .t-item {
          padding: 0 2rem;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.38);
          font-weight: 500;
          display: inline-flex;
          align-items: center;
          gap: 2rem;
          border-right: 1px solid rgba(255, 255, 255, 0.07);
        }

        .t-item :global(b) {
          color: rgba(255, 255, 255, 0.58);
          font-weight: 700;
        }

        .problem,
        .nancy,
        .corner,
        .cta {
          padding: 7rem 2.5rem;
        }

        .section-inner {
          max-width: 1240px;
          margin: 0 auto;
        }

        .ew {
          font-size: 0.74rem;
          font-weight: 700;
          letter-spacing: 0.13em;
          text-transform: uppercase;
          margin-bottom: 1.5rem;
        }

        .ew--teal {
          color: var(--teal);
        }

        .ew--amber {
          color: var(--amber);
        }

        .ew--light {
          color: rgba(15, 118, 110, 0.72);
        }

        .problem-hl,
        .nancy-hl,
        .session-hl,
        .arch-hl,
        .ss-title,
        .ta-title,
        .layers-hl,
        .corner-hl,
        .cta-hl {
          font-family: var(--font-d);
          letter-spacing: -0.02em;
        }

        .problem-hl {
          font-size: clamp(2.4rem, 4.5vw, 4.2rem);
          line-height: 1.02;
          max-width: 22ch;
          margin-bottom: 1.25rem;
        }

        .problem-body,
        .nancy-body,
        .corner-body,
        .ss-sub {
          font-size: 1.05rem;
          line-height: 1.72;
          color: var(--muted);
        }

        .problem-body {
          max-width: 50ch;
          margin-bottom: 4.5rem;
        }

        .problem-stats {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 1.5rem;
        }

        .p-stat {
          padding: 2.25rem 2rem;
          border-radius: 1.75rem;
          border: 1px solid var(--line);
          background: var(--surface);
          transition: transform 0.25s, box-shadow 0.25s;
        }

        .p-stat:hover,
        .ct:hover,
        .cc:hover {
          transform: translateY(-3px);
        }

        .p-stat:hover,
        .ct:hover {
          box-shadow: 0 16px 50px rgba(16, 33, 43, 0.1);
        }

        .p-num {
          font-family: var(--font-d);
          font-size: 3.8rem;
          font-weight: 700;
          line-height: 1;
          color: var(--navy-mid);
          margin-bottom: 0.75rem;
        }

        .p-label {
          font-size: 0.96rem;
          line-height: 1.65;
          color: var(--muted);
        }

        .problem-quote {
          margin-top: 4rem;
          padding: 2.5rem 3rem;
          border-radius: 1.75rem;
          border-left: 3px solid var(--teal);
          background: linear-gradient(135deg, rgba(15, 118, 110, 0.06), var(--surface));
        }

        .problem-quote :global(blockquote) {
          margin: 0;
          font-family: var(--font-d);
          font-size: clamp(1.5rem, 2.5vw, 2.2rem);
          line-height: 1.3;
          letter-spacing: -0.01em;
          color: var(--navy-mid);
          max-width: 36ch;
        }

        .thesis,
        .arch {
          background: var(--navy-mid);
          padding: 8rem 2.5rem;
          text-align: center;
        }

        .thesis-inner {
          max-width: 900px;
          margin: 0 auto;
        }

        .thesis-hl {
          font-family: var(--font-d);
          font-size: clamp(2.6rem, 5vw, 5rem);
          line-height: 1;
          letter-spacing: -0.025em;
          color: white;
          margin-bottom: 1.5rem;
        }

        .thesis-hl span {
          color: rgba(13, 200, 190, 0.75);
        }

        .thesis-sub {
          font-size: 1.1rem;
          line-height: 1.72;
          color: rgba(255, 255, 255, 0.55);
          max-width: 50ch;
          margin: 0 auto 3.5rem;
        }

        .thesis-pills {
          display: flex;
          justify-content: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .thesis-pill {
          padding: 0.6rem 1.4rem;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.13);
          background: rgba(255, 255, 255, 0.06);
          color: rgba(255, 255, 255, 0.68);
          font-size: 0.88rem;
          font-weight: 600;
        }

        .layers {
          background: #0a1a27;
          padding: 6rem 0 0;
        }

        .layers-hd,
        .ta-hd,
        .ss-hd {
          padding: 0 2.5rem;
          max-width: 1240px;
          margin: 0 auto 3rem;
        }

        .layers-hl {
          font-size: clamp(2rem, 3.8vw, 3.4rem);
          color: white;
          line-height: 1.06;
          max-width: 20ch;
          letter-spacing: -0.015em;
          margin-top: 0.75rem;
        }

        .layers-tabs {
          display: flex;
          gap: 0.75rem;
          flex-wrap: wrap;
          padding: 0 2.5rem;
          max-width: 1240px;
          margin: 0 auto 2.5rem;
        }

        .l-tab {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          padding: 0.7rem 1.25rem;
          border-radius: 0.85rem;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: transparent;
          color: rgba(255, 255, 255, 0.5);
          transition: all 0.25s;
          text-align: left;
        }

        .l-tab:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        .l-tab.on {
          background: rgba(15, 118, 110, 0.18);
          border-color: rgba(15, 118, 110, 0.35);
          color: white;
        }

        .l-tab-n {
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--teal);
        }

        .l-tab-name {
          font-size: 0.92rem;
          font-weight: 600;
        }

        .layers-body {
          position: relative;
        }

        .l-panel {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 4rem;
          padding: 2rem 2.5rem 5.5rem;
          max-width: 1240px;
          margin: 0 auto;
        }

        .panel-enter {
          animation: panelEnter 0.38s ease;
        }

        .l-panel-copy {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 1.35rem;
        }

        .l-panel-ew {
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--teal);
        }

        .l-panel-hl {
          font-family: var(--font-d);
          font-size: clamp(1.75rem, 2.8vw, 2.6rem);
          color: white;
          line-height: 1.1;
          letter-spacing: -0.015em;
        }

        .l-panel-body {
          font-size: 1rem;
          line-height: 1.72;
          color: rgba(255, 255, 255, 0.58);
        }

        .l-feats,
        .s-feats,
        .a-items,
        .s-ev-row {
          display: flex;
          flex-direction: column;
          gap: 0.55rem;
        }

        .l-feat,
        .s-feat,
        .a-item {
          display: flex;
          align-items: flex-start;
          gap: 0.7rem;
          line-height: 1.55;
        }

        .l-feat,
        .s-feat {
          font-size: 0.9rem;
        }

        .l-feat,
        .a-item {
          color: rgba(255, 255, 255, 0.7);
        }

        .s-feat {
          color: rgba(255, 255, 255, 0.65);
        }

        .l-feat-d,
        .s-dot,
        .a-item-d {
          width: 0.38rem;
          height: 0.38rem;
          border-radius: 50%;
          background: var(--teal);
          flex-shrink: 0;
          margin-top: 0.55rem;
        }

        .l-panel-vis,
        .s-vis {
          display: flex;
          align-items: center;
        }

        .mock {
          width: 100%;
          border-radius: 1.5rem;
          border: 1px solid rgba(255, 255, 255, 0.07);
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 0.9rem;
        }

        .mock--nancy {
          background: rgba(255, 252, 247, 0.04);
        }

        .mock--session {
          background: rgba(22, 56, 77, 0.3);
        }

        .mock-hd,
        .nc-hd {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .mock-hd {
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }

        .mock-av,
        .nc-av {
          display: grid;
          place-items: center;
          font-family: var(--font-d);
          font-weight: 800;
          color: white;
        }

        .mock-av {
          width: 2.4rem;
          height: 2.4rem;
          border-radius: 50%;
          flex-shrink: 0;
          font-size: 0.95rem;
        }

        .mock-av--n {
          background: linear-gradient(135deg, #c66a1a, #e8973a);
        }

        .mock-av--s {
          background: linear-gradient(135deg, var(--navy-mid), var(--teal));
        }

        .mock-av-name,
        .nc-name {
          font-size: 0.9rem;
          font-weight: 700;
          color: white;
        }

        .mock-av-sub,
        .nc-status {
          font-size: 0.74rem;
          color: rgba(255, 255, 255, 0.38);
        }

        .mock-conf,
        .nc-badge {
          margin-left: auto;
          padding: 0.28rem 0.72rem;
          border-radius: 999px;
          background: rgba(15, 118, 110, 0.2);
          border: 1px solid rgba(15, 118, 110, 0.3);
          font-size: 0.72rem;
          font-weight: 700;
          color: rgba(13, 200, 190, 0.9);
        }

        .mb,
        .c-bbl {
          padding: 0.8rem 0.95rem;
          border-radius: 1.1rem;
          font-size: 0.86rem;
          line-height: 1.6;
        }

        .mb--n {
          background: rgba(198, 106, 26, 0.16);
          border: 1px solid rgba(198, 106, 26, 0.18);
          color: rgba(255, 255, 255, 0.85);
          max-width: 90%;
        }

        .mb--p {
          background: rgba(15, 118, 110, 0.18);
          border: 1px solid rgba(15, 118, 110, 0.22);
          color: rgba(255, 255, 255, 0.85);
          margin-left: auto;
          max-width: 86%;
        }

        .mb-lbl,
        .mf-lbl,
        .c-lbl,
        .mock-handoff-lbl,
        .s-card-lbl,
        .al-badge,
        .cc-tag,
        .lim-head,
        .ks-label {
          font-size: 0.65rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .mb-lbl,
        .mf-lbl,
        .c-lbl {
          color: rgba(255, 255, 255, 0.3);
          margin-bottom: 0.28rem;
        }

        .mock-handoff {
          padding: 0.8rem 1rem;
          border-radius: 1rem;
          background: rgba(22, 56, 77, 0.45);
          border: 1px solid rgba(255, 255, 255, 0.07);
          font-size: 0.78rem;
          color: rgba(255, 255, 255, 0.5);
        }

        .mock-handoff-lbl {
          color: var(--teal);
          margin-bottom: 0.28rem;
        }

        .mf {
          display: flex;
          flex-direction: column;
          gap: 0.28rem;
        }

        .mf-val,
        .s-card-val {
          font-size: 0.86rem;
          line-height: 1.6;
          color: rgba(255, 255, 255, 0.75);
        }

        .mf-small {
          font-size: 0.78rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .mock-tx,
        .s-card-tx {
          font-size: 0.82rem;
          line-height: 1.7;
          color: rgba(255, 255, 255, 0.5);
          border-left: 2px solid rgba(15, 118, 110, 0.35);
          padding-left: 0.85rem;
        }

        .affect-row {
          display: flex;
          gap: 0.6rem;
          align-items: center;
          flex-wrap: wrap;
        }

        .aff-chip {
          padding: 0.25rem 0.65rem;
          border-radius: 999px;
          font-size: 0.76rem;
          font-weight: 700;
        }

        .aff-c {
          background: rgba(15, 118, 110, 0.2);
          color: rgba(13, 200, 190, 0.85);
        }

        .aff-a {
          background: rgba(198, 106, 26, 0.2);
          color: #e8973a;
        }

        .aff-arr {
          color: rgba(255, 255, 255, 0.22);
          font-size: 0.8rem;
        }

        .alerts-stack {
          display: flex;
          flex-direction: column;
          gap: 0.7rem;
          width: 100%;
        }

        .al {
          border-radius: 1.15rem;
          border: 1px solid;
          padding: 1rem 1.1rem;
        }

        .al--u {
          background: rgba(181, 71, 92, 0.15);
          border-color: rgba(181, 71, 92, 0.3);
        }

        .al--w {
          background: rgba(198, 106, 26, 0.12);
          border-color: rgba(198, 106, 26, 0.28);
        }

        .al--r {
          background: rgba(22, 56, 77, 0.28);
          border-color: rgba(255, 255, 255, 0.07);
        }

        .al-hd {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.45rem;
        }

        .al-badge {
          padding: 0.22rem 0.6rem;
          border-radius: 999px;
        }

        .al--u .al-badge {
          background: rgba(181, 71, 92, 0.25);
          color: #e87a8d;
        }

        .al--w .al-badge {
          background: rgba(198, 106, 26, 0.25);
          color: #e8973a;
        }

        .al--r .al-badge {
          background: rgba(255, 255, 255, 0.07);
          color: rgba(255, 255, 255, 0.4);
        }

        .al-time {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.28);
        }

        .al-title {
          font-size: 0.88rem;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.82);
          margin-bottom: 0.35rem;
        }

        .al-body {
          margin: 0;
          font-size: 0.78rem;
          line-height: 1.55;
          color: rgba(255, 255, 255, 0.45);
        }

        .nancy {
          padding-top: 8rem;
          background: var(--cream);
        }

        .nancy-inner,
        .session-inner {
          max-width: 1240px;
          margin: 0 auto;
          display: grid;
          grid-template-columns: 1fr 1.05fr;
          gap: 6rem;
          align-items: center;
        }

        .nancy-hl,
        .session-hl,
        .ta-title,
        .ss-title {
          font-size: clamp(2rem, 3.4vw, 3.2rem);
          line-height: 1.05;
          margin-bottom: 1.1rem;
        }

        .nancy-body {
          margin-bottom: 2rem;
        }

        .n-feats {
          display: flex;
          flex-direction: column;
          gap: 1.1rem;
        }

        .n-feat {
          display: flex;
          align-items: flex-start;
          gap: 0.85rem;
        }

        .n-feat-ic {
          width: 2rem;
          height: 2rem;
          border-radius: 0.5rem;
          flex-shrink: 0;
          background: rgba(198, 106, 26, 0.1);
          border: 1px solid rgba(198, 106, 26, 0.18);
          display: grid;
          place-items: center;
          font-size: 0.7rem;
          font-weight: 800;
          color: var(--amber);
        }

        .n-feat-title {
          font-size: 0.92rem;
          font-weight: 700;
          margin-bottom: 0.2rem;
        }

        .n-feat-body {
          margin: 0;
          font-size: 0.84rem;
          color: var(--muted);
          line-height: 1.55;
        }

        .nancy-chat {
          width: 100%;
          text-align: left;
          border-radius: 2rem;
          background: var(--surface);
          border: 1px solid rgba(16, 33, 43, 0.08);
          box-shadow: 0 32px 80px rgba(16, 33, 43, 0.12);
          overflow: hidden;
          cursor: pointer;
        }

        .nc-hd {
          padding: 1.25rem 1.5rem;
          background: linear-gradient(135deg, #1e4858, #0f5e61);
        }

        .nc-av {
          width: 2.75rem;
          height: 2.75rem;
          border-radius: 50%;
          background: linear-gradient(135deg, #c66a1a, #e8973a);
          font-size: 1.1rem;
        }

        .nc-badge {
          background: rgba(15, 118, 110, 0.35);
          color: rgba(255, 255, 255, 0.82);
        }

        .nc-msgs {
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 0.9rem;
          min-height: 280px;
          max-height: 340px;
          overflow: hidden;
        }

        .c-bbl,
        .c-handoff {
          animation: bubbleIn 0.45s ease both;
        }

        .c-bbl {
          max-width: 84%;
        }

        .c-bbl--n {
          background: linear-gradient(135deg, rgba(240, 142, 53, 0.13), rgba(255, 255, 255, 0.9));
          border: 1px solid rgba(198, 106, 26, 0.13);
          color: var(--ink);
        }

        .c-bbl--p {
          background: linear-gradient(135deg, rgba(15, 118, 110, 0.1), rgba(255, 255, 255, 0.9));
          border: 1px solid rgba(15, 118, 110, 0.13);
          color: var(--ink);
          margin-left: auto;
        }

        .c-lbl {
          color: var(--muted);
        }

        .c-handoff {
          padding: 0.85rem 1rem;
          border-radius: 1rem;
          background: rgba(22, 56, 77, 0.06);
          border: 1px solid rgba(22, 56, 77, 0.1);
          font-size: 0.78rem;
          color: var(--muted);
          display: flex;
          gap: 0.6rem;
          align-items: flex-start;
        }

        .c-ho-d {
          width: 0.4rem;
          height: 0.4rem;
          border-radius: 50%;
          background: var(--teal);
          flex-shrink: 0;
          margin-top: 0.35rem;
        }

        .nc-hint {
          text-align: center;
          padding: 0.65rem;
          font-size: 0.7rem;
          color: rgba(16, 33, 43, 0.25);
          letter-spacing: 0.04em;
          border-top: 1px solid rgba(16, 33, 43, 0.06);
        }

        .session {
          padding: 8rem 2.5rem;
          background: #0d2233;
        }

        .session-inner {
          grid-template-columns: 1fr 1.15fr;
          gap: 5rem;
          align-items: start;
        }

        .session-hl,
        .ta-title {
          color: white;
        }

        .session-body,
        .ta-sub,
        .arch-body {
          font-size: 1.05rem;
          line-height: 1.72;
          color: rgba(255, 255, 255, 0.55);
        }

        .s-vis {
          flex-direction: column;
          gap: 0.85rem;
        }

        .s-metrics {
          width: 100%;
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.75rem;
        }

        .s-met,
        .s-card,
        .lc,
        .chart-card,
        .kpi-tile {
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 1.25rem;
        }

        .s-met,
        .s-card {
          width: 100%;
          background: rgba(255, 255, 255, 0.04);
          padding: 1rem;
        }

        .s-met {
          text-align: center;
        }

        .s-met-v {
          font-family: var(--font-d);
          font-size: 1.9rem;
          font-weight: 700;
          color: var(--teal-light);
          line-height: 1;
        }

        .s-met-l {
          font-size: 0.68rem;
          text-transform: uppercase;
          letter-spacing: 0.07em;
          color: rgba(255, 255, 255, 0.3);
          margin-top: 0.3rem;
        }

        .s-card {
          padding: 1.25rem;
        }

        .s-card-lbl {
          color: rgba(15, 118, 110, 0.7);
          margin-bottom: 0.65rem;
        }

        .corner-hd,
        .arch-intro,
        .chart-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 4rem;
          margin-bottom: 4rem;
          align-items: end;
        }

        .corner-hl {
          font-size: clamp(2.5rem, 4.5vw, 4.2rem);
          line-height: 1;
          margin-top: 0.75rem;
        }

        .corner-body {
          max-width: 40ch;
        }

        .corner-threads,
        .arch-loops {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 1.25rem;
        }

        .ct {
          padding: 1.5rem 1.4rem;
          border-radius: 1.35rem;
          border: 1px solid var(--line);
          background: var(--surface);
          display: flex;
          flex-direction: column;
          gap: 0.7rem;
          transition: transform 0.25s, box-shadow 0.25s;
        }

        .ct-flair {
          display: inline-flex;
          width: fit-content;
          padding: 0.24rem 0.65rem;
          border-radius: 999px;
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .ct-fr--research {
          background: rgba(15, 118, 110, 0.1);
          color: var(--teal);
        }

        .ct-fr--hyp {
          background: rgba(198, 106, 26, 0.1);
          color: var(--amber);
        }

        .ct-fr--case {
          background: rgba(181, 71, 92, 0.1);
          color: var(--rose);
        }

        .ct-title {
          font-size: 0.96rem;
          font-weight: 700;
          line-height: 1.42;
        }

        .ct-body {
          margin: 0;
          font-size: 0.84rem;
          line-height: 1.6;
          color: var(--muted);
          flex: 1;
        }

        .ct-meta,
        .ct-meta-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .ct-meta {
          font-size: 0.76rem;
          color: var(--muted);
          padding-top: 0.35rem;
          border-top: 1px solid var(--line);
          flex-wrap: wrap;
        }

        .ct-dot {
          width: 0.3rem;
          height: 0.3rem;
          border-radius: 50%;
          background: var(--line);
        }

        .arch {
          text-align: left;
        }

        .arch-intro {
          margin-bottom: 4.5rem;
        }

        .a-loop {
          padding: 2rem;
          border-radius: 1.5rem;
          border: 1px solid rgba(255, 255, 255, 0.07);
          background: rgba(255, 255, 255, 0.04);
        }

        .a-n {
          font-family: var(--font-d);
          font-size: 3.5rem;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.06);
          line-height: 1;
          margin-bottom: 0.75rem;
        }

        .a-name {
          font-family: var(--font-d);
          font-size: 1.15rem;
          color: white;
          font-weight: 700;
          margin-bottom: 0.5rem;
        }

        .a-body {
          margin: 0 0 1.5rem;
          font-size: 0.88rem;
          line-height: 1.65;
          color: rgba(255, 255, 255, 0.45);
        }

        .tech-arch {
          padding: 8rem 0;
          background: #07111a;
          overflow: hidden;
        }

        .ta-title {
          margin: 0.65rem 0 0.9rem;
        }

        .ta-sub {
          max-width: 54ch;
        }

        .zone-legend {
          display: flex;
          gap: 1.5rem;
          padding: 0 2.5rem;
          max-width: 1240px;
          margin: 0 auto 2rem;
          flex-wrap: wrap;
        }

        .zl-item {
          display: flex;
          align-items: center;
          gap: 0.55rem;
          font-size: 0.76rem;
          color: rgba(255, 255, 255, 0.38);
          font-weight: 600;
          letter-spacing: 0.04em;
        }

        .zl-dot,
        .rl-dot {
          width: 0.55rem;
          height: 0.55rem;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .zl-dot--capture,
        .zl-dot--reason,
        .zl-dot--output {
          background: #0d9488;
        }

        .zl-dot--perception {
          background: #1e4d69;
        }

        .zl-dot--fusion {
          background: #7c3aed;
        }

        .zl-dot--rag {
          background: #c66a1a;
        }

        .zl-dot--safety {
          background: #b5475c;
        }

        .pipeline-scroll {
          overflow-x: auto;
          padding: 0 2.5rem 2.5rem;
        }

        .pipeline-row {
          display: flex;
          align-items: stretch;
          min-width: max-content;
        }

        .pipeline-segment {
          display: flex;
          align-items: stretch;
        }

        .lc {
          width: 195px;
          flex-shrink: 0;
          background: rgba(255, 255, 255, 0.04);
          padding: 1.25rem 1.1rem;
          display: flex;
          flex-direction: column;
          gap: 0.85rem;
          position: relative;
          transition: background 0.25s, border-color 0.25s;
        }

        .lc-wide {
          width: 230px;
        }

        .lc:hover {
          background: rgba(255, 255, 255, 0.07);
          border-color: rgba(255, 255, 255, 0.14);
        }

        .lc-top-bar {
          height: 3px;
          border-radius: 2px;
          position: absolute;
          top: 0;
          left: 1rem;
          right: 1rem;
        }

        .lc-num {
          font-family: var(--font-d);
          font-size: 0.65rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: rgba(255, 255, 255, 0.28);
        }

        .lc-name {
          font-family: var(--font-d);
          font-size: 0.92rem;
          font-weight: 700;
          line-height: 1.25;
        }

        .lc-zone-lbl {
          font-size: 0.58rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          padding: 0.18rem 0.55rem;
          border-radius: 999px;
          width: fit-content;
          background: rgba(255, 255, 255, 0.07);
          color: rgba(255, 255, 255, 0.35);
        }

        .lc-divider {
          width: 100%;
          border: 0;
          height: 1px;
          background: rgba(255, 255, 255, 0.07);
          margin: 0;
        }

        .lc-comps {
          display: flex;
          flex-direction: column;
          gap: 0.55rem;
          flex: 1;
        }

        .lc-comp {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 0.65rem;
          padding: 0.55rem 0.7rem;
        }

        .comp-n,
        .comp-s {
          font-family: "Courier New", monospace;
        }

        .comp-n {
          font-size: 0.78rem;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.82);
        }

        .comp-s {
          font-size: 0.68rem;
          color: rgba(255, 255, 255, 0.32);
          margin-top: 0.12rem;
          line-height: 1.4;
          display: flex;
          flex-direction: column;
        }

        .lc-io {
          font-size: 0.68rem;
          color: rgba(255, 255, 255, 0.28);
          border-top: 1px solid rgba(255, 255, 255, 0.06);
          padding-top: 0.65rem;
          font-family: "Courier New", monospace;
          line-height: 1.5;
        }

        .io-arrow {
          color: rgba(13, 200, 190, 0.5);
          margin-right: 0.3rem;
        }

        .safety-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.22rem 0.6rem;
          border-radius: 999px;
          width: fit-content;
          background: rgba(181, 71, 92, 0.18);
          border: 1px solid rgba(181, 71, 92, 0.3);
          font-size: 0.6rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #e87a8d;
          margin-top: 0.4rem;
        }

        .flow-arrow {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 38px;
          flex-shrink: 0;
          position: relative;
        }

        .fa-track {
          width: 100%;
          height: 2px;
          background: linear-gradient(90deg, rgba(13, 200, 190, 0.25), rgba(13, 200, 190, 0.5));
          position: relative;
          overflow: hidden;
        }

        .fa-track::after {
          content: "";
          position: absolute;
          top: 0;
          left: -100%;
          width: 50%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(13, 200, 190, 0.8), transparent);
          animation: flowPulse 2.2s linear infinite;
        }

        .fa-head {
          position: absolute;
          right: -1px;
          width: 0;
          height: 0;
          border-top: 4px solid transparent;
          border-bottom: 4px solid transparent;
          border-left: 6px solid rgba(13, 200, 190, 0.5);
        }

        .lifecycle-ribbon,
        .dc-adjacent,
        .benchmark-wrap,
        .limitation-card {
          max-width: 1240px;
          margin: 2.5rem auto 0;
          padding: 0 2.5rem;
        }

        .lifecycle-ribbon {
          display: flex;
          align-items: center;
          flex-wrap: wrap;
          gap: 0;
        }

        .lifecycle-fragment {
          display: flex;
          align-items: center;
        }

        .lf-step {
          padding: 0.7rem 0.85rem;
          text-align: center;
          font-size: 0.72rem;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.38);
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .lf-step--active {
          background: rgba(13, 200, 190, 0.08);
          color: rgba(13, 200, 190, 0.72);
          border-color: rgba(13, 200, 190, 0.18);
        }

        .lf-step-sep {
          width: 1rem;
          text-align: center;
          color: rgba(255, 255, 255, 0.18);
          font-size: 0.7rem;
        }

        .dc-adj-card {
          border: 1px dashed rgba(198, 106, 26, 0.3);
          border-radius: 1.25rem;
          padding: 1.1rem 1.5rem;
          background: rgba(198, 106, 26, 0.04);
          display: flex;
          align-items: center;
          gap: 1.25rem;
          flex-wrap: wrap;
        }

        .dc-adj-badge {
          padding: 0.3rem 0.8rem;
          border-radius: 999px;
          background: rgba(198, 106, 26, 0.12);
          border: 1px solid rgba(198, 106, 26, 0.25);
          font-size: 0.68rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #e8973a;
          white-space: nowrap;
        }

        .dc-adj-text {
          margin: 0;
          font-size: 0.88rem;
          color: rgba(255, 255, 255, 0.45);
          line-height: 1.6;
        }

        .safety-sec {
          padding: 8rem 2.5rem;
          background: var(--cream);
        }

        .ss-title {
          margin: 0.65rem 0 0.9rem;
        }

        .ss-sub {
          max-width: 54ch;
        }

        .pilot-note {
          display: inline-flex;
          align-items: center;
          gap: 0.55rem;
          padding: 0.45rem 1rem;
          border-radius: 999px;
          margin-top: 0.75rem;
          background: rgba(198, 106, 26, 0.08);
          border: 1px solid rgba(198, 106, 26, 0.18);
          font-size: 0.74rem;
          font-weight: 700;
          color: var(--amber);
        }

        .kpi-grid {
          max-width: 1240px;
          margin: 0 auto 3rem;
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1rem;
        }

        .kpi-tile,
        .chart-card {
          background: var(--surface);
          border: 1px solid var(--line);
          box-shadow: 0 8px 28px rgba(16, 33, 43, 0.07);
          position: relative;
          overflow: hidden;
        }

        .kpi-tile {
          border-radius: 1.4rem;
          padding: 1.25rem 1.3rem;
        }

        .kpi-tile::before {
          content: "";
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 2px;
          background: linear-gradient(90deg, var(--teal), var(--teal-light));
        }

        .kpi-name {
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          color: var(--muted);
          margin-bottom: 0.65rem;
          line-height: 1.4;
        }

        .kpi-current {
          font-family: var(--font-d);
          font-size: 2rem;
          font-weight: 700;
          color: var(--navy-mid);
          line-height: 1;
          margin-bottom: 0.6rem;
        }

        .kpi-current--good {
          color: var(--teal);
        }

        .kpi-stages {
          display: flex;
          flex-direction: column;
          gap: 0.3rem;
          margin-bottom: 0.65rem;
        }

        .kpi-stage {
          display: flex;
          align-items: center;
          gap: 0.55rem;
        }

        .ks-label {
          color: var(--muted);
          min-width: 3rem;
        }

        .ks-bar-wrap {
          flex: 1;
          height: 0.35rem;
          background: rgba(16, 33, 43, 0.07);
          border-radius: 999px;
          overflow: hidden;
        }

        .ks-bar {
          height: 100%;
          border-radius: 999px;
        }

        .ks-bar--baseline {
          background: rgba(16, 33, 43, 0.18);
        }

        .ks-bar--current {
          background: var(--teal);
        }

        .ks-bar--target {
          background: rgba(198, 106, 26, 0.5);
        }

        .ks-val {
          font-size: 0.68rem;
          font-weight: 700;
          color: var(--ink);
          min-width: 3rem;
          text-align: right;
        }

        .kpi-n {
          font-size: 0.62rem;
          color: var(--muted);
        }

        .chart-row {
          max-width: 1240px;
          margin: 0 auto 3rem;
          gap: 1.5rem;
          align-items: start;
        }

        .chart-card {
          border-radius: 1.6rem;
          padding: 1.75rem;
        }

        .chart-title {
          font-family: var(--font-d);
          font-size: 1.1rem;
          font-weight: 700;
          margin-bottom: 0.4rem;
        }

        .chart-sub,
        .bm-sub {
          font-size: 0.8rem;
          color: var(--muted);
          margin-bottom: 1.25rem;
          line-height: 1.5;
        }

        .radar-wrap {
          display: flex;
          justify-content: center;
        }

        .radar-legend {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
          margin-top: 1rem;
          justify-content: center;
        }

        .rl-item {
          display: flex;
          align-items: center;
          gap: 0.4rem;
          font-size: 0.74rem;
          color: var(--muted);
        }

        .waterfall {
          display: flex;
          flex-direction: column;
          gap: 0.6rem;
        }

        .wf-row {
          display: flex;
          align-items: center;
          gap: 0.65rem;
        }

        .wf-label {
          font-size: 0.78rem;
          color: var(--ink);
          font-weight: 600;
          min-width: 17ch;
          text-align: right;
        }

        .wf-bar-wrap {
          flex: 1;
          position: relative;
          height: 2rem;
        }

        .wf-bar-bg {
          height: 100%;
          background: rgba(16, 33, 43, 0.06);
          border-radius: 0.4rem;
        }

        .wf-bar-fill {
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          border-radius: 0.4rem;
          display: flex;
          align-items: center;
          justify-content: flex-end;
          padding-right: 0.5rem;
          font-size: 0.68rem;
          font-weight: 800;
          color: white;
        }

        .wf-bar--base {
          background: rgba(16, 33, 43, 0.22);
        }

        .wf-bar--gain {
          background: linear-gradient(90deg, var(--teal), var(--teal-light));
        }

        .bm-title {
          font-family: var(--font-d);
          font-size: 1.25rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .bm-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.82rem;
        }

        .bm-table th {
          padding: 0.65rem 0.9rem;
          text-align: left;
          font-size: 0.68rem;
          font-weight: 800;
          letter-spacing: 0.07em;
          text-transform: uppercase;
          color: var(--muted);
          background: rgba(16, 33, 43, 0.04);
          border-bottom: 1px solid var(--line);
        }

        .bm-table td {
          padding: 0.7rem 0.9rem;
          border-bottom: 1px solid rgba(16, 33, 43, 0.05);
          vertical-align: middle;
        }

        .bm-table tr:hover td {
          background: rgba(16, 33, 43, 0.02);
        }

        .bm-layer {
          font-size: 0.7rem;
          font-weight: 800;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          padding: 0.22rem 0.62rem;
          border-radius: 999px;
          width: fit-content;
          display: inline-flex;
        }

        .bml--iq {
          background: rgba(13, 200, 190, 0.1);
          color: var(--teal);
        }

        .bml--sa {
          background: rgba(181, 71, 92, 0.1);
          color: var(--rose);
        }

        .bml--co {
          background: rgba(22, 56, 77, 0.1);
          color: var(--navy-mid);
        }

        .bml--pl {
          background: rgba(198, 106, 26, 0.1);
          color: var(--amber);
        }

        .bml--sg {
          background: rgba(124, 58, 237, 0.1);
          color: #5b21b6;
        }

        .bml--rr {
          background: rgba(16, 33, 43, 0.08);
          color: var(--muted);
        }

        .bm-metric {
          font-size: 0.84rem;
          font-weight: 600;
        }

        .bm-note {
          font-size: 0.65rem;
          color: var(--muted);
        }

        .bm-val {
          font-family: var(--font-d);
          font-weight: 700;
        }

        .bm-base {
          color: var(--muted);
          font-size: 0.84rem;
        }

        .bm-current {
          color: var(--teal);
        }

        .bm-target {
          color: var(--amber);
          font-size: 0.84rem;
        }

        .bm-delta {
          font-size: 0.74rem;
          font-weight: 700;
          padding: 0.18rem 0.52rem;
          border-radius: 999px;
        }

        .bm-delta--up,
        .bm-delta--down {
          background: rgba(13, 200, 190, 0.1);
          color: var(--teal);
        }

        .bm-sample {
          font-size: 0.72rem;
          color: var(--muted);
        }

        .limitation-card {
          border-left: 3px solid var(--amber);
          background: linear-gradient(135deg, rgba(198, 106, 26, 0.06), var(--surface));
          border-radius: 1.25rem;
          padding: 1.25rem 1.5rem;
          border: 1px solid rgba(198, 106, 26, 0.18);
        }

        .lim-body {
          margin: 0;
          font-size: 0.88rem;
          line-height: 1.65;
          color: var(--muted);
        }

        .cta {
          padding-top: 8rem;
        }

        .cta-hd {
          text-align: center;
          margin-bottom: 4rem;
        }

        .cta-hl {
          font-size: clamp(2.5rem, 5vw, 4.2rem);
          line-height: 1.04;
          margin-top: 0.75rem;
        }

        .cta-cards {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
        }

        .cc {
          padding: 3rem;
          border-radius: 2rem;
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          transition: transform 0.28s, box-shadow 0.28s;
        }

        .cc--doc {
          background: linear-gradient(145deg, #0d1f30, #16384d);
          box-shadow: 0 12px 50px rgba(13, 31, 48, 0.3);
        }

        .cc--pat {
          background: linear-gradient(145deg, #0f4f52, #1e5c5f);
          box-shadow: 0 12px 50px rgba(15, 79, 82, 0.3);
        }

        .cc-tag {
          display: inline-flex;
          width: fit-content;
          padding: 0.32rem 0.82rem;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.16);
          background: rgba(255, 255, 255, 0.07);
          color: rgba(255, 255, 255, 0.55);
        }

        .cta-card-title {
          font-family: var(--font-d);
          font-size: clamp(1.8rem, 2.8vw, 2.5rem);
          line-height: 1.1;
          color: white;
        }

        .cta-card-body {
          margin: 0;
          font-size: 0.98rem;
          line-height: 1.65;
          color: rgba(255, 255, 255, 0.58);
          flex: 1;
        }

        .cc-btn {
          display: inline-flex;
          width: fit-content;
          padding: 0.88rem 1.75rem;
          border-radius: 999px;
          font-weight: 700;
          font-size: 0.92rem;
          color: white;
          transition: opacity 0.2s, transform 0.2s;
        }

        .cc--doc .cc-btn {
          background: var(--teal);
        }

        .cc--pat .cc-btn {
          background: var(--amber);
        }

        .cc:hover .cc-btn {
          opacity: 0.9;
          transform: translateY(-1px);
        }

        footer {
          background: #080f14;
          padding: 2.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        .ft-brand {
          display: flex;
          align-items: center;
          gap: 0.7rem;
        }

        .ft-mark {
          width: 1.9rem;
          height: 1.9rem;
          border-radius: 0.5rem;
          background: linear-gradient(135deg, var(--navy-mid), var(--teal));
          display: grid;
          place-items: center;
          font-family: var(--font-d);
          font-weight: 800;
          color: white;
          font-size: 0.82rem;
        }

        .ft-name {
          font-family: var(--font-d);
          font-weight: 700;
          color: rgba(255, 255, 255, 0.55);
          font-size: 0.88rem;
        }

        .ft-copy {
          font-size: 0.76rem;
          color: rgba(255, 255, 255, 0.22);
        }

        .rv {
          opacity: 0;
          transform: translateY(22px);
          transition: opacity 0.65s ease, transform 0.65s ease;
        }

        .rv.in {
          opacity: 1;
          transform: translateY(0);
        }

        .rv--d2 {
          transition-delay: 0.1s;
        }

        .rv--d3 {
          transition-delay: 0.2s;
        }

        .rv--d4 {
          transition-delay: 0.3s;
        }

        @media (max-width: 1200px) {
          .kpi-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
        }

        @media (max-width: 1000px) {
          .problem-stats,
          .arch-loops,
          .corner-threads,
          .cta-cards,
          .kpi-grid,
          .chart-row {
            grid-template-columns: 1fr;
          }

          .nancy-inner,
          .session-inner,
          .corner-hd,
          .arch-intro,
          .l-panel {
            grid-template-columns: 1fr;
            gap: 3rem;
          }

          .hero-row {
            flex-direction: column;
            gap: 2rem;
          }
        }

        @media (max-width: 760px) {
          .nav,
          .problem,
          .nancy,
          .session,
          .corner,
          .arch,
          .safety-sec,
          .cta,
          .layers-hd,
          .layers-tabs,
          .l-panel,
          .ta-hd,
          .zone-legend,
          .pipeline-scroll,
          .lifecycle-ribbon,
          .dc-adjacent,
          .benchmark-wrap,
          .limitation-card,
          .ss-hd {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
          }

          .hero {
            padding: 0 1.25rem 4rem;
          }

          .nav {
            gap: 1rem;
            flex-wrap: wrap;
          }

          .nav-links {
            width: 100%;
            justify-content: flex-end;
          }

          .hero-scroll-hint {
            display: none;
          }

          .problem-quote {
            padding: 1.5rem;
          }

          .s-metrics {
            grid-template-columns: 1fr;
          }

          footer {
            padding: 1.5rem 1.25rem;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.65rem;
          }

          .wf-row {
            flex-direction: column;
            align-items: flex-start;
          }

          .wf-label {
            min-width: 0;
            text-align: left;
          }
        }

        @keyframes glowBreath {
          from {
            opacity: 0.75;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes pulse {
          0%,
          100% {
            box-shadow: 0 0 8px rgba(13, 148, 136, 0.7);
          }
          50% {
            box-shadow: 0 0 14px rgba(13, 148, 136, 1);
          }
        }

        @keyframes scrollDrop {
          0% {
            transform: scaleY(0);
            transform-origin: top;
          }
          50% {
            transform: scaleY(1);
            transform-origin: top;
          }
          51% {
            transform-origin: bottom;
          }
          100% {
            transform: scaleY(0);
            transform-origin: bottom;
          }
        }

        @keyframes tickL {
          from {
            transform: translateX(0);
          }
          to {
            transform: translateX(-50%);
          }
        }

        @keyframes tickR {
          from {
            transform: translateX(-50%);
          }
          to {
            transform: translateX(0);
          }
        }

        @keyframes flowPulse {
          to {
            left: 200%;
          }
        }

        @keyframes panelEnter {
          from {
            opacity: 0;
            transform: translateY(14px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes bubbleIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </>
  );
}
