# MindScape MVP Recovery Plan

## Non-Negotiable Reset

We are no longer optimizing for number of features.
We are optimizing for three complete product slices.

Any work that does not strengthen one of these slices is secondary.

## The Three Slices

### Slice 1: Async Care Loop

Goal:
Between sessions, a patient can report status, talk to Nancy, and generate a clinician-useful handoff without the doctor reconstructing the week manually.

Must include:

- patient-facing companion entry point
- daily questionnaire submission
- patient-to-Nancy message flow
- Nancy response
- clinician handoff generation
- persistent timeline
- watch / urgent risk classification

Definition of done:

- patient can complete the flow from one route
- doctor can review the outcome from one route
- same event appears consistently in patient record, Nancy view, and workspace
- no manual “log this separately” step is required

### Slice 2: Clinician Session Loop

Goal:
Doctor opens a patient, gets a real pre-session brief, runs analysis, and has the result written back into the chart in a reusable format.

Must include:

- patient context briefing
- session audio upload or live recording
- engine result
- evidence display
- session record persistence
- visible post-session analytics

Definition of done:

- one clinician can complete this end to end without leaving the product shell
- session result is saved and visible on revisit
- patient longitudinal context is actually used during the run

### Slice 3: Escalation Loop

Goal:
When async signals become concerning, the system creates a serious, reviewable escalation path instead of just storing text.

Must include:

- risk lane classification
- doctor-visible alert
- hospital routing recommendation
- SOS event persistence
- clear timeline/audit trail

Definition of done:

- every escalation has a visible trigger, route, and outcome record
- escalation is visible in all relevant doctor-facing contexts
- emergency language never gets treated as a routine note

## What We Stop Doing

We stop:

- adding more portfolio-style pages
- expanding mock research layers
- building ornamental features that do not complete a slice
- calling prototype-only features “done”

## Engineering Definition Of Done

A feature is not done because a page exists.
A feature is done only if all of the following are true:

1. There is one clear user entry point.
2. There is one clear workflow handler/service.
3. Data is persisted in a stable shape.
4. The result is visible in the downstream place where it matters.
5. Failure states are handled intentionally.
6. The implementation can be described in one sentence without hand-waving.

## Current Status

### Slice 1: Async Care Loop

Status: partially real

What exists:

- patient companion route
- questionnaire input
- Nancy reply generation
- clinician handoff generation
- unified timeline foundation

What is still weak:

- no auth/identity model
- no message delivery model
- no notification model
- still local JSON backed

### Slice 2: Clinician Session Loop

Status: strongest slice

What exists:

- session route
- patient-aware context
- transcription
- retrieval-backed analysis
- post-session persistence

What is still weak:

- UI and workflow are still too coupled to `app.py`
- limited workflow verification
- no clinician-authored post-session note model

### Slice 3: Escalation Loop

Status: prototype but directionally correct

What exists:

- risk classification foundation
- SOS persistence
- Shenzhen routing logic
- doctor-hospital mapping

What is still weak:

- no external dispatch/communication integration
- no explicit alert inbox
- no escalation acknowledgement workflow

## Immediate Next Build Order

1. Harden Slice 1 completely
2. Refactor Slice 2 into service boundaries
3. Make Slice 3 visible and auditable from workspace + patient record

## The Standard Going Forward

Every future change should answer:

1. Which slice does this strengthen?
2. What user journey becomes more complete because of it?
3. What source of truth does it use?
4. Where does the result show up downstream?
5. How would we verify it end to end?

If we cannot answer those clearly, the work is probably not serious enough yet.
