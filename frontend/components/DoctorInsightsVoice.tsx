"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** Same WebSocket / PCM protocol as `NancyVoice.tsx`; connects to `/ws/doctor-insights/{doctorId}` (OpenAI Realtime). */
type InsightEvent =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "action"; function: string; result: string }
  | { type: "UserStartedSpeaking" }
  | { type: "AgentStartedSpeaking" }
  | { type: "AgentAudioDone" }
  | { type: "ConversationText"; role: string; content: string }
  | { type: string; [key: string]: unknown };

type ChatLine = { role: "user" | "assistant" | "action"; text: string };

const REALTIME_SAMPLE_RATE = 24000;

const ACTION_LABELS: Record<string, string> = {
  refresh_mindscape_snapshot: "✓ Refreshed MindScape snapshot",
};

const WS_BASE =
  (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8002/api")
    .replace(/\/api$/, "")
    .replace(/^http/, "ws");

export function DoctorInsightsVoice({
  doctorId,
  doctorDisplayName,
}: {
  doctorId: string;
  /** Label for user transcript bubbles (defaults to "You"). */
  doctorDisplayName?: string;
}) {
  const userLabel = (doctorDisplayName || "You").trim() || "You";
  const [phase, setPhase] = useState<"idle" | "connecting" | "ready" | "user" | "nancy" | "error">("idle");
  const [chat, setChat] = useState<ChatLine[]>([]);
  const [error, setError] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sinkNodeRef = useRef<GainNode | null>(null);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const nextPlayTimeRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const blockMicUntilRef = useRef(0);
  const firstAgentTurnPendingRef = useRef(false);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chat]);

  function ensureAudioCtx(): AudioContext {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext();
    }
    if (audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }

  function stopPlayback() {
    activeSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch {
        /* noop */
      }
      source.disconnect();
    });
    activeSourcesRef.current = [];
    nextPlayTimeRef.current = 0;
  }

  function scheduleAudioChunk(pcmBuffer: ArrayBuffer) {
    const ctx = ensureAudioCtx();
    const int16 = new Int16Array(pcmBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const buffer = ctx.createBuffer(1, float32.length, REALTIME_SAMPLE_RATE);
    buffer.copyToChannel(float32, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    activeSourcesRef.current.push(source);
    source.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter((node) => node !== source);
    };

    const now = ctx.currentTime;
    const start = Math.max(now, nextPlayTimeRef.current);
    source.start(start);
    nextPlayTimeRef.current = start + buffer.duration;
  }

  function resampleToRate(samples: Float32Array, inputRate: number, outputRate: number): Float32Array {
    if (inputRate === outputRate) return samples;
    const outputLength = Math.max(1, Math.round((samples.length * outputRate) / inputRate));
    const output = new Float32Array(outputLength);
    const ratio = inputRate / outputRate;
    for (let i = 0; i < outputLength; i++) {
      const position = i * ratio;
      const left = Math.floor(position);
      const right = Math.min(left + 1, samples.length - 1);
      const mix = position - left;
      output[i] = samples[left] * (1 - mix) + samples[right] * mix;
    }
    return output;
  }

  function float32ToPcm16(samples: Float32Array): ArrayBuffer {
    const buf = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      buf[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    return buf.buffer;
  }

  async function startMic(ws: WebSocket) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    streamRef.current = stream;
    const ctx = ensureAudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const sink = ctx.createGain();
    sink.gain.value = 0;
    sinkNodeRef.current = sink;
    processor.onaudioprocess = (e) => {
      if (ws.readyState === WebSocket.OPEN) {
        if (performance.now() < blockMicUntilRef.current) {
          return;
        }
        const channelData = e.inputBuffer.getChannelData(0);
        const resampled = resampleToRate(channelData, ctx.sampleRate, REALTIME_SAMPLE_RATE);
        ws.send(float32ToPcm16(resampled));
      }
    };
    source.connect(processor);
    processor.connect(sink);
    sink.connect(ctx.destination);
    processorRef.current = processor;
  }

  function stopMic() {
    processorRef.current?.disconnect();
    processorRef.current = null;
    sinkNodeRef.current?.disconnect();
    sinkNodeRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  const connect = useCallback(async () => {
    setPhase("connecting");
    setError("");
    setChat([]);
    stopPlayback();
    nextPlayTimeRef.current = 0;

    const ws = new WebSocket(`${WS_BASE}/ws/doctor-insights/${doctorId}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onmessage = async (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        scheduleAudioChunk(ev.data);
        return;
      }
      const event = JSON.parse(ev.data as string) as InsightEvent;

      if (event.type === "ready") {
        try {
          await ensureAudioCtx().resume();
          // Keep mic uplink closed until the first assistant opener is complete.
          firstAgentTurnPendingRef.current = true;
          blockMicUntilRef.current = Number.POSITIVE_INFINITY;
          await startMic(ws);
          ws.send(JSON.stringify({ type: "client_ready" }));
          setPhase("ready");
        } catch {
          setError("Microphone access is required to start the voice session.");
          setPhase("error");
          ws.close();
          stopMic();
        }
      } else if (event.type === "error") {
        setError(typeof event.message === "string" ? event.message : "Insights voice encountered an error.");
        setPhase("error");
      } else if (event.type === "UserStartedSpeaking") {
        setPhase("user");
        stopPlayback();
      } else if (event.type === "AgentStartedSpeaking") {
        setPhase("nancy");
      } else if (event.type === "AgentAudioDone") {
        if (firstAgentTurnPendingRef.current) {
          firstAgentTurnPendingRef.current = false;
          // Small settle delay helps avoid residual echo bleed.
          blockMicUntilRef.current = performance.now() + 350;
        }
        setPhase("ready");
      } else if (event.type === "ConversationText") {
        const role = (event.role as string).toLowerCase();
        setChat((prev) => [
          ...prev,
          { role: role === "user" ? "user" : "assistant", text: event.content as string },
        ]);
      } else if (event.type === "action") {
        const label = ACTION_LABELS[event.function as string] ?? `✓ ${event.function}`;
        setChat((prev) => [...prev, { role: "action", text: label }]);
      }
    };

    ws.onclose = () => {
      stopMic();
      stopPlayback();
      setPhase("idle");
    };

    ws.onerror = () => {
      setError("WebSocket connection failed. Check the backend is running.");
      setPhase("error");
      stopMic();
      stopPlayback();
    };
  }, [doctorId]);

  function disconnect() {
    wsRef.current?.send(JSON.stringify({ type: "close" }));
    wsRef.current?.close();
    stopMic();
    stopPlayback();
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    setPhase("idle");
  }

  useEffect(
    () => () => {
      disconnect();
    },
    [], // eslint-disable-line react-hooks/exhaustive-deps -- run cleanup on unmount only
  );

  const orbClass =
    phase === "user"
      ? "orb--listening"
      : phase === "nancy"
        ? "orb--speaking"
        : phase === "connecting"
          ? "orb--thinking"
          : "orb--idle";

  const orbIcon =
    phase === "user" ? "🎙"
    : phase === "nancy" ? "🔊"
    : phase === "connecting" ? "⋯"
    : phase === "error" ? "✕"
    : "◆";

  const statusLabel =
    phase === "idle"
      ? "Start live Doctor Insights (OpenAI Realtime)"
      : phase === "connecting"
        ? "Connecting…"
        : phase === "ready"
          ? `Listening — speak naturally, ${userLabel}`
          : phase === "user"
            ? "Listening to you…"
            : phase === "nancy"
              ? "MindScape Insights is speaking…"
              : "Connection error";

  return (
    <div className="nancy-voice-shell">
      <div className="nancy-status-strip">
        <span className={`nancy-status-pill nancy-status-pill--${phase}`}>{statusLabel}</span>
        <span className="nancy-status-pill nancy-status-pill--soft">
          {chat.length ? `${chat.length} live updates` : "Transcript will appear here"}
        </span>
      </div>

      <div className={`nancy-orb ${orbClass}`}>
        <div className="orb-inner">{orbIcon}</div>
        <p className="orb-label">{statusLabel}</p>
      </div>

      <div className="nancy-control-row">
        {phase === "idle" || phase === "error" ? (
          <button className="mic-button" onClick={connect} type="button">
            Start voice session
          </button>
        ) : (
          <button
            className="mic-button"
            onClick={disconnect}
            type="button"
            style={{ background: "linear-gradient(135deg, var(--rose), var(--amber))" }}
          >
            End session
          </button>
        )}
      </div>

      {error && <p className="nancy-error-text">{error}</p>}

      {chat.length > 0 && (
        <div className="nancy-transcript" ref={scrollRef}>
          {chat.map((line, i) =>
            line.role === "action" ? (
              <div key={i} className="bubble-actions" style={{ justifyContent: "center" }}>
                <span className="action-badge">{line.text}</span>
              </div>
            ) : (
              <div
                key={i}
                className={`nancy-bubble ${line.role === "user" ? "bubble--user" : "bubble--nancy"}`}
              >
                <div className="bubble-label">{line.role === "user" ? userLabel : "Insights"}</div>
                <p>{line.text}</p>
              </div>
            ),
          )}
        </div>
      )}

      <p className="nancy-footnote muted">
        Same OpenAI Realtime voice stack as patient Nancy — PCM over WebSocket, server VAD, live transcript. Uses your
        MindScape snapshot (panel, alerts, forum); call “refresh” in chat to reload server data mid-session.
      </p>
    </div>
  );
}
