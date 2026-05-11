# Claude Design One-Pager

Use this as the short prompt/context handoff for Claude Design.

## Project

Redesign the MindScape UI into a premium, public-facing product experience.

## Important stack clarification

- The active product path is `Next.js + FastAPI`.
- Redesign the `frontend/` app.
- `app.py` / Mesop still exists in the repo as legacy context, but it is not the target frontend.
- If current copy mentions Mesop, migration, or engineering rewrites, treat that as stale internal language that should be removed.

## What MindScape is

MindScape is a mental-health clinical operating system with three core product slices:

1. Async care loop
   - between-session patient updates
   - Nancy support
   - clinician handoffs
   - alerts and review

2. Clinician session loop
   - patient-aware session prep
   - uploaded or live session analysis
   - structured hypothesis, evidence, BSV, and notes

3. Escalation loop
   - risk classification
   - doctor review queue
   - SOS routing and audit trail

It is not just a chatbot, wellness app, or internal dashboard.

## Primary audiences

- Clinicians: need clarity, confidence, fast triage, structured information, and high-signal workflows.
- Patients: need calm, compassionate, psychologically safe support surfaces.
- Public visitors: need a strong product story, trust, and a sense that this is a serious category-defining system.

## Most important product concepts

- Nancy: AI care companion for between-session support, daily check-ins, relayed guidance, and clinician handoffs.
- MindScape session intelligence: live/uploaded session analysis with transcript, affect, retrieval, evidence, and hypothesis output.
- Doctor's Corner: research and clinician-community layer, positioned like a high-signal intelligence network for mental-health professionals.
- Strong patient/doctor boundary: patient-safe surfaces must remain separate from clinician-only workflows and notes.

## What the redesign should achieve

### Public / landing

- Make the landing page feel like a legendary product reveal.
- Explain the failures of current mental-health diagnosis and care continuity.
- Reveal MindScape layer by layer.
- Spotlight Nancy, session intelligence, and Doctor's Corner.
- Use restrained motion, not flashy gimmicks.

### Clinician experience

- Make workspace, chart, session, and Nancy console feel elite, operational, and high-trust.
- Reduce prototype/admin-tool feeling.
- Improve information hierarchy and decision clarity.

### Patient experience

- Make companion and Nancy flows feel immaculate, calm, warm, and compassionate.
- Remove overly clinical or bureaucratic form language where possible.
- Preserve patient-safe boundaries.

## Non-negotiable design constraints

- Preserve separation between doctor portal and patient portal.
- Preserve separation between daily Nancy check-in and general Nancy conversation.
- Preserve separation between patient-visible and clinician-only Nancy content.
- Do not present SOS or emergency features as if the app is a live dispatch system.
- Do not present MindScape as replacing clinicians.
- Do not use copy that sounds like migration notes, engineering notes, or internal tooling.

## Current design/copy problems to fix

- Too much stack language: Next.js, FastAPI, Mesop, migration framing.
- Too much "portal/gateway" language without enough emotional or product payoff.
- Public story is underpowered relative to the real product depth.
- Some patient surfaces still sound too clinical or form-heavy.
- Some clinician surfaces feel like a prototype shell rather than a polished product.

## Visual direction

Aim for:

- polished
- editorial
- premium
- clinically trustworthy
- emotionally intelligent
- calm but ambitious

Avoid:

- hackathon/demo energy
- generic SaaS dashboard aesthetics
- overblown sci-fi visuals
- loud flashy motion
- internal admin-tool tone

## Highest-priority surfaces

1. Landing page
2. Doctor workspace
3. Patient companion home
4. Patient Nancy experience
5. Nancy clinician console
6. Session analysis workflow
7. Doctor's Corner

## Suggested deliverable

Produce a redesign that:

- upgrades the product story
- sharpens hierarchy
- improves copy everywhere
- makes clinician surfaces feel elite
- makes patient surfaces feel deeply cared for
- is ready to be shown publicly

If helpful, use `docs/CLAUDE_UI_REDESIGN_CONTEXT.md` as the full reference document.
