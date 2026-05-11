"use client";

import { DoctorInsightsVoice } from "@/components/DoctorInsightsVoice";
import { doctorForumPath } from "@/lib/routes";
import { FormEvent, useEffect, useMemo, useState, useTransition } from "react";

import { postJson } from "@/lib/api";

export type ForumReply = {
  id?: string;
  author: string;
  body: string;
  created?: string;
  date_label?: string;
};

export type ForumThread = {
  id: string;
  title: string;
  body: string;
  flair?: string;
  kind?: string;
  community?: string;
  author?: string;
  created?: string;
  date_label?: string;
  link?: string;
  link_title?: string;
  replies?: ForumReply[];
  reply_count?: number;
  activity?: string;
  vote_score?: number;
  live_headline?: boolean;
  source?: string;
  saved?: boolean;
};

type ResearchSnapshot = {
  forum_threads: ForumThread[];
  weekly_brief: string[];
  performance: Array<{ label: string; value: string }>;
};

/** Reddit-for-psychiatry style taxonomy — mental health screening, diagnosis, and treatment inefficiency is society-critical; this surface is for research-grade peer exchange. */
export const MH_POST_TYPES: Array<{ kind: string; flair: string; hint: string }> = [
  { kind: "discussion", flair: "Discussion", hint: "Open dialogue — any clinical angle" },
  { kind: "research", flair: "Research & trials", hint: "Papers, RCTs, preprints, systematic reviews" },
  { kind: "trend", flair: "Trend watch", hint: "Population signals, practice patterns, societal shifts" },
  { kind: "hypothesis", flair: "Hypothesis", hint: "Unsolved ideas — crowd-source critique & methods" },
  { kind: "success_case", flair: "Success / teaching case", hint: "What worked — de-identified, consent-aware" },
  { kind: "failure_case", flair: "Failure / near-miss", hint: "Safety culture; what broke and what you changed" },
  { kind: "methodology", flair: "Methodology", hint: "Protocols, measurement, implementation science" },
  { kind: "trial_error", flair: "Trial & error", hint: "Retired interventions, honest pivots" },
  { kind: "screening", flair: "Screening / diagnosis", hint: "Earlier detection, instruments, pathways" },
  { kind: "question", flair: "Case question", hint: "Specific patient fork — ask peers" },
];

const FILTER_CHIPS: Array<{ key: string; label: string }> = [
  { key: "all", label: "All" },
  { key: "saved", label: "Saved" },
  { key: "research", label: "Research" },
  { key: "trend", label: "Trends" },
  { key: "hypothesis", label: "Hypotheses" },
  { key: "success_case", label: "Wins" },
  { key: "failure_case", label: "Near-misses" },
  { key: "methodology", label: "Methods" },
  { key: "trial_error", label: "Trial & error" },
  { key: "screening", label: "Screening" },
  { key: "question", label: "Questions" },
  { key: "discussion", label: "Discussion" },
];

const PAGE_SIZE = 28;

function flairCssKind(kind: string | undefined): string {
  const k = (kind || "discussion").toLowerCase();
  if (
    [
      "discussion",
      "research",
      "trend",
      "hypothesis",
      "success_case",
      "failure_case",
      "methodology",
      "trial_error",
      "screening",
      "question",
    ].includes(k)
  ) {
    return k;
  }
  return "discussion";
}

export function DoctorForum({
  doctorId,
  doctorDisplayName,
  initial,
  initialOpenThreadId,
}: {
  doctorId: string;
  doctorDisplayName?: string;
  initial: ResearchSnapshot;
  /** Open & scroll from `/research?thread=<id>` share links */
  initialOpenThreadId?: string;
}) {
  const [snapshot, setSnapshot] = useState<ResearchSnapshot>(initial);
  const [filter, setFilter] = useState<string>("all");
  const [sortMode, setSortMode] = useState<"new" | "top">("new");
  const [searchQuery, setSearchQuery] = useState("");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [openId, setOpenId] = useState<string | null>(initial.forum_threads[0]?.id ?? null);
  const [message, setMessage] = useState("");
  const [copyNotice, setCopyNotice] = useState("");
  const [isPending, startTransition] = useTransition();

  const [composeTitle, setComposeTitle] = useState("");
  const [composeBody, setComposeBody] = useState("");
  const [composeTypeIdx, setComposeTypeIdx] = useState(0);
  const [composeLink, setComposeLink] = useState("");
  const [composeLinkTitle, setComposeLinkTitle] = useState("");
  const [composeCommunity, setComposeCommunity] = useState("MindScape · Doctor's Corner");

  const [replyDrafts, setReplyDrafts] = useState<Record<string, string>>({});

  const threads = snapshot.forum_threads;

  const selectedPostType = MH_POST_TYPES[composeTypeIdx] ?? MH_POST_TYPES[0];

  useEffect(() => {
    if (initialOpenThreadId && threads.some((t) => t.id === initialOpenThreadId)) {
      setOpenId(initialOpenThreadId);
    }
  }, [initialOpenThreadId, threads]);

  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
  }, [filter, sortMode, searchQuery]);

  const filteredSorted = useMemo(() => {
    let list = threads;
    if (filter === "saved") {
      list = list.filter((t) => t.saved);
    } else if (filter !== "all") {
      list = list.filter((t) => (t.kind || "discussion").toLowerCase() === filter);
    }

    const q = searchQuery.trim().toLowerCase();
    if (q) {
      list = list.filter((t) => {
        const blob = `${t.title} ${t.body} ${t.flair ?? ""} ${t.community ?? ""} ${t.author ?? ""}`.toLowerCase();
        return blob.includes(q);
      });
    }

    list = [...list];
    if (sortMode === "top") {
      list.sort((a, b) => (b.vote_score ?? 0) - (a.vote_score ?? 0));
    } else {
      list.sort((a, b) => (b.created || "").localeCompare(a.created || ""));
    }
    return list;
  }, [threads, filter, sortMode, searchQuery]);

  const visibleThreads = useMemo(() => filteredSorted.slice(0, visibleCount), [filteredSorted, visibleCount]);

  function applyResearch(next: ResearchSnapshot) {
    setSnapshot(next);
    setMessage("");
  }

  function vote(threadId: string) {
    startTransition(async () => {
      try {
        const res = await postJson<{ research: ResearchSnapshot }>(
          `/doctors/${doctorId}/research/threads/${encodeURIComponent(threadId)}/vote`,
          {},
        );
        applyResearch(res.research);
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not record vote");
      }
    });
  }

  function toggleSave(threadId: string) {
    startTransition(async () => {
      try {
        const res = await postJson<{ research: ResearchSnapshot }>(
          `/doctors/${doctorId}/research/threads/${encodeURIComponent(threadId)}/save`,
          {},
        );
        applyResearch(res.research);
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not update saved");
      }
    });
  }

  function copyThreadLink(threadId: string) {
    const path = `${doctorForumPath(doctorId)}?thread=${encodeURIComponent(threadId)}`;
    const absolute = typeof window !== "undefined" ? `${window.location.origin}${path}` : path;
    void navigator.clipboard.writeText(absolute).then(
      () => {
        setCopyNotice("Link copied for rounds / slides.");
        setTimeout(() => setCopyNotice(""), 2200);
      },
      () => setMessage("Clipboard unavailable — copy URL manually."),
    );
  }

  function submitThread(event: FormEvent) {
    event.preventDefault();
    const title = composeTitle.trim();
    const body = composeBody.trim();
    if (!title || !body) {
      setMessage("Add a title and body for your thread.");
      return;
    }
    const pt = selectedPostType;
    startTransition(async () => {
      try {
        const res = await postJson<{ research: ResearchSnapshot }>(
          `/doctors/${doctorId}/research/threads`,
          {
            title,
            body,
            flair: pt.flair,
            kind: pt.kind,
            author: "You",
            link: composeLink.trim(),
            link_title: composeLinkTitle.trim(),
            community: composeCommunity.trim() || "Doctor's Corner",
          },
        );
        applyResearch(res.research);
        setComposeTitle("");
        setComposeBody("");
        setComposeLink("");
        setComposeLinkTitle("");
        setOpenId(res.research.forum_threads[0]?.id ?? null);
        setSortMode("new");
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not create thread");
      }
    });
  }

  function submitReply(threadId: string) {
    const body = (replyDrafts[threadId] || "").trim();
    if (!body) return;
    startTransition(async () => {
      try {
        const res = await postJson<{ research: ResearchSnapshot }>(
          `/doctors/${doctorId}/research/threads/${encodeURIComponent(threadId)}/replies`,
          { body, author: "You" },
        );
        applyResearch(res.research);
        setReplyDrafts((d) => ({ ...d, [threadId]: "" }));
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not post reply");
      }
    });
  }

  const totalShown = visibleThreads.length;
  const totalMatched = filteredSorted.length;
  const hasMore = totalShown < totalMatched;

  return (
    <section className="forum-shell">
      <div className="forum-mission card-slab">
        <h3 className="forum-mission-title">Why this exists</h3>
        <p className="forum-mission-body">
          Mental health is among society&apos;s most pressing burdens — and among the most inefficiently diagnosed and triaged conditions. This forum is designed as a{" "}
          <strong>research-grade exchange</strong> for psychiatrists and mental-health clinicians: share literature, surface trends, teach from{" "}
          <strong>success and failure cases</strong>, stress-test <strong>hypotheses</strong>, document <strong>methodology</strong> and <strong>trial-and-error</strong> learning, and sharpen{" "}
          <strong>screening and diagnostic</strong> insight. By default, press-style headlines are not the main product — peer exchange and evidence-shaped posts are.
        </p>
      </div>

      <header className="forum-header">
        <div>
          <h2 className="forum-title">MindScape · Doctor&apos;s Corner</h2>
          <p className="forum-subtitle">
            The thread list is <strong>curated research-grade seed</strong> plus <strong>your live posts</strong> — not a Twitter firehose. Optional PubMed or headline digests (env-controlled) can augment the weekly brief only; the forum itself stays grounded in psychiatry and mental-health professional exchange.
          </p>
        </div>
      </header>

      {message ? <p className="form-feedback forum-banner">{message}</p> : null}
      {copyNotice ? <p className="form-feedback forum-banner forum-banner--ok">{copyNotice}</p> : null}

      <div className="card-slab" style={{ marginBottom: "1.25rem" }}>
        <h3 className="forum-mission-title" style={{ marginTop: 0 }}>
          Voice · Doctor Insights (Realtime)
        </h3>
        <p className="muted" style={{ fontSize: "0.88rem", marginBottom: "0.75rem" }}>
          Same OpenAI Realtime voice pipeline as patient Nancy — live audio, transcript, and optional snapshot refresh tool.
        </p>
        <DoctorInsightsVoice doctorId={doctorId} doctorDisplayName={doctorDisplayName} />
      </div>

      <form className="forum-composer card-slab" onSubmit={submitThread}>
        <div className="forum-composer-head">
          <span className="forum-new-badge">New thread</span>
          <span className="muted">De-identify patients; follow institutional and jurisdictional rules — auth & org gates ship next.</span>
        </div>
        <input
          placeholder="Title — e.g. Hypothesis: voice jitter rises 48h before self-harm ideation disclosure"
          value={composeTitle}
          onChange={(e) => setComposeTitle(e.target.value)}
        />
        <div className="forum-composer-row">
          <label className="forum-field forum-field--grow">
            <span className="forum-field-label">Post type</span>
            <select value={composeTypeIdx} onChange={(e) => setComposeTypeIdx(Number(e.target.value))}>
              {MH_POST_TYPES.map((pt, i) => (
                <option key={pt.kind} value={i}>
                  {pt.flair} — {pt.hint}
                </option>
              ))}
            </select>
          </label>
          <label className="forum-field">
            <span className="forum-field-label">Circle / channel</span>
            <input
              placeholder="e.g. Trauma · Mood · Community psychiatry"
              value={composeCommunity}
              onChange={(e) => setComposeCommunity(e.target.value)}
            />
          </label>
        </div>
        <p className="forum-type-hint muted">{selectedPostType.hint}</p>
        <textarea
          placeholder="Context, what you observed, what you need from peers — be specific enough to be useful."
          rows={6}
          value={composeBody}
          onChange={(e) => setComposeBody(e.target.value)}
        />
        <div className="forum-composer-row">
          <label className="forum-field">
            <span className="forum-field-label">Link (optional)</span>
            <input placeholder="https://…" value={composeLink} onChange={(e) => setComposeLink(e.target.value)} />
          </label>
          <label className="forum-field">
            <span className="forum-field-label">Link title</span>
            <input
              placeholder="Article or trial registry label"
              value={composeLinkTitle}
              onChange={(e) => setComposeLinkTitle(e.target.value)}
            />
          </label>
        </div>
        <button className="action-button primary" disabled={isPending} type="submit">
          Post to forum
        </button>
      </form>

      <div className="forum-toolbar forum-toolbar--stack">
        <label className="forum-search">
          <span className="muted forum-search-label">Search threads</span>
          <input
            className="forum-search-input"
            placeholder="Title, body, channel, author…"
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </label>
        <div className="forum-filters" role="tablist" aria-label="Filter by post type">
          {FILTER_CHIPS.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              className={`forum-filter${filter === key ? " forum-filter--active" : ""}`}
              onClick={() => setFilter(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="forum-sort">
          <span className="muted forum-sort-label">Sort</span>
          <button
            type="button"
            className={`forum-sort-btn${sortMode === "new" ? " forum-sort-btn--active" : ""}`}
            onClick={() => setSortMode("new")}
          >
            New
          </button>
          <button
            type="button"
            className={`forum-sort-btn${sortMode === "top" ? " forum-sort-btn--active" : ""}`}
            onClick={() => setSortMode("top")}
          >
            Top
          </button>
        </div>
      </div>

      <p className="muted forum-result-meta">
        Showing {totalShown} of {totalMatched} matching threads
        {searchQuery.trim() ? ` · search “${searchQuery.trim()}”` : ""}
      </p>

      <ul className="forum-thread-list">
        {visibleThreads.length === 0 ? (
          <li className="forum-empty">
            Nothing matches — adjust search or filters, or post the first thread.
          </li>
        ) : (
          visibleThreads.map((thread) => {
            const open = openId === thread.id;
            const rc = thread.reply_count ?? thread.replies?.length ?? 0;
            const meta = [thread.community, thread.author, thread.date_label || thread.created?.slice(0, 10)]
              .filter(Boolean)
              .join(" · ");
            const fk = flairCssKind(thread.kind);
            const score = thread.vote_score ?? 0;

            return (
              <li key={thread.id} className={`forum-thread card-slab${open ? " forum-thread--open" : ""}`}>
                <div className="forum-thread-row">
                  <div className="forum-vote-stack">
                    <button
                      type="button"
                      className="forum-upvote"
                      disabled={isPending}
                      title="Upvote — surface high-signal posts for peers"
                      aria-label="Upvote"
                      onClick={(e) => {
                        e.stopPropagation();
                        vote(thread.id);
                      }}
                    >
                      ▲
                    </button>
                    <span className="forum-score">{score}</span>
                  </div>

                  <div className="forum-thread-expand">
                    <button
                      type="button"
                      className="forum-thread-main"
                      onClick={() => setOpenId(open ? null : thread.id)}
                      aria-expanded={open}
                    >
                      <span className="forum-flair-row">
                        {thread.saved ? (
                          <span className="forum-saved-star" title="Saved">
                            ★
                          </span>
                        ) : null}
                        {thread.live_headline || thread.source === "medical_news" ? (
                          <span className="forum-live-pill" title="Live RSS / NewsAPI headline">
                            Live
                          </span>
                        ) : null}
                        <span className={`forum-flair forum-flair--${fk}`}>{thread.flair || fk}</span>
                      </span>
                      <span className="forum-thread-title">{thread.title}</span>
                      <span className="forum-thread-meta">
                        {meta}
                        {rc > 0 ? ` · ${rc} repl${rc === 1 ? "y" : "ies"}` : ""}
                      </span>
                      <span className="forum-thread-chevron">{open ? "▾" : "▸"}</span>
                    </button>

                    {open ? (
                      <div className="forum-thread-body">
                        <div className="forum-thread-actions">
                          <button
                            type="button"
                            className="forum-action-btn"
                            disabled={isPending}
                            onClick={() => copyThreadLink(thread.id)}
                          >
                            Copy share link
                          </button>
                          <button
                            type="button"
                            className={`forum-action-btn${thread.saved ? " forum-action-btn--active" : ""}`}
                            disabled={isPending}
                            onClick={() => toggleSave(thread.id)}
                          >
                            {thread.saved ? "Saved" : "Save thread"}
                          </button>
                        </div>
                        <p className="forum-thread-text">{thread.body}</p>
                        {thread.link ? (
                          <p className="forum-thread-link">
                            <a href={thread.link} rel="noreferrer" target="_blank">
                              {thread.link_title || thread.link}
                            </a>
                          </p>
                        ) : null}

                        <div className="forum-replies">
                          <div className="forum-replies-head">Thread discussion</div>
                          {(thread.replies || []).length === 0 ? (
                            <p className="muted forum-no-replies">No replies yet — add the first comment.</p>
                          ) : (
                            <ul className="forum-reply-list">
                              {(thread.replies || []).map((r) => (
                                <li key={r.id || `${r.author}-${r.body.slice(0, 12)}`} className="forum-reply">
                                  <div className="forum-reply-author">{r.author}</div>
                                  <div className="forum-reply-meta">{r.date_label}</div>
                                  <p className="forum-reply-body">{r.body}</p>
                                </li>
                              ))}
                            </ul>
                          )}
                          <div className="forum-reply-form">
                            <textarea
                              placeholder="Reply with clinical nuance — disagree constructively, cite experience."
                              rows={3}
                              value={replyDrafts[thread.id] || ""}
                              onChange={(e) =>
                                setReplyDrafts((d) => ({
                                  ...d,
                                  [thread.id]: e.target.value,
                                }))
                              }
                            />
                            <button
                              type="button"
                              className="action-button secondary"
                              disabled={isPending}
                              onClick={() => submitReply(thread.id)}
                            >
                              Reply
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              </li>
            );
          })
        )}
      </ul>

      {hasMore ? (
        <div className="forum-load-more-wrap">
          <button type="button" className="action-button secondary" onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}>
            Load more ({totalMatched - totalShown} remaining)
          </button>
        </div>
      ) : null}
    </section>
  );
}
