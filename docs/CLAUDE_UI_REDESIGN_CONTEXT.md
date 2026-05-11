# MindScape UI Redesign Context For Claude

## Why this document exists

If a design model only sees the `frontend/` folder, it can understand the current screens but it will miss the actual product truth behind those screens.

MindScape is not just a marketing landing page plus a few dashboards. It is a mental-health clinical platform with real workflow semantics underneath:

- live session analysis
- between-session patient support through Nancy
- clinician review queues and alerts
- structured doctor directives for Nancy
- patient-safe vs clinician-only information boundaries
- SOS escalation routing
- research and peer-discussion surfaces for clinicians

This document is meant to be the single design-context handoff for Claude or any UI redesign model, so it does not need to read the full repo to understand what every route, section, card, button, and conversation surface is actually trying to do.

## Active stack status

This must be interpreted clearly:

- the current web product path in scope is `frontend/` plus `backend_api/`
- the active stack is `Next.js + FastAPI`
- `app.py` and the Mesop implementation still exist in the repo as legacy/original context
- Mesop should be treated as historical implementation context, not as the frontend Claude is redesigning
- any current UI copy that mentions Mesop, migration, or "not a Mesop screen" is stale internal-language leakage, not the desired public product narrative

For design work, assume:

- current UX target = the Next.js routes and components
- current backend contract = the FastAPI API and WebSocket surfaces
- Mesop references are useful only for background and should not appear in the redesigned public-facing copy

## Short product definition

MindScape is a precision mental-health operating system that combines:

- a clinician-facing workflow for live diagnostic sessions, patient review, async triage, and structured follow-up
- a patient-facing companion experience for daily check-ins, supportive messaging, and Nancy interactions
- an intelligence layer that turns fragmented signals into reviewable clinical handoffs, alerts, and longitudinal context
- a research and clinician-community layer that positions MindScape as a knowledge network, not just a charting tool

## The most important product truth

The repo itself says the strongest fully connected slices are:

1. The clinician session loop
2. The async care loop
3. The escalation loop

Everything in the redesign should make these three slices feel real, premium, and legible.

Do not redesign the product as if it is only:

- an AI chatbot
- a generic EHR
- a consumer wellness app
- a dashboard-only internal admin tool

It is a care-operating system with two very different emotional modes:

- clinician mode: confident, clear, structured, high-signal
- patient mode: calm, warm, safe, compassionate, human

## What is real vs what is still prototype-level

This is critical. The UI should feel public-facing and premium, but it should not misrepresent the current product maturity.

### Real enough to anchor the redesign

- FastAPI backend contract exists for the current web UI
- Next.js frontend routes exist for the main product shell
- patient intake, outreach, patient chart views, session prep, session note logging, async check-ins, messaging, Nancy directives, Nancy touchpoints, alerts, and SOS records all persist in local runtime JSON
- live or uploaded session analysis runs through the Python diagnostic engine
- Nancy has both text and realtime voice flows
- patient context is actually used during session analysis
- patient data is merged with runtime events like check-ins, Nancy handoffs, messages, sessions, notes, alerts, and SOS events

### Exists but should not be overclaimed

- authentication and role security are not real yet
- doctor/patient context is still URL selected
- organizations and some research/community layers are partly seed or mock-backed
- Nancy live voice exists as an implementation, but it is still MVP-grade
- SOS is routing logic plus persistence, not real emergency dispatch integration
- chat is persisted, but there is no full delivery, unread, or notification model

### Design implication

The redesign should present MindScape as an elite emerging product, not a finished hospital-wide regulated enterprise rollout.

That means:

- confident but not dishonest
- visionary but not fake
- premium without claiming operational maturity that the codebase does not yet have

## Public-facing design ambition

The current UI often sounds like an internal migration project or engineering demo. The redesign should instead feel like a world-class mental-health product reveal.

The direction requested by the user is:

- the landing page should feel legendary, like a product reveal and category introduction
- it should explain the problem with current mental-health diagnosis and treatment quality
- it should reveal MindScape layer by layer
- it should spotlight Nancy AI and the clinician community/forum
- it should use restrained motion, not flashy gimmicks
- the patient/Nancy conversation surfaces should feel immaculate, caring, and compassionate

### Good emotional territory

- polished, editorial, premium
- emotionally intelligent
- clinically trustworthy
- ambitious but calm
- sharp product craft, not startup toy aesthetics

### Bad emotional territory

- hackathon demo
- internal ops tooling
- admin dashboard genericness
- oversaturated wellness cliches
- neon futurism
- overloaded glassmorphism everywhere

## The core product narrative the redesign should express

The public story should be:

1. Mental healthcare still breaks down between visits.
2. Diagnosis is often slow, fragmented, or not context-rich enough.
3. Important daily deterioration signals get buried in text or missed entirely.
4. Clinicians need a better operating system, not more noise.
5. Patients need a support experience that feels humane, not mechanical.
6. MindScape connects live sessions, between-session support, risk review, escalation, and clinical intelligence into one system.
7. Nancy extends care between visits.
8. Doctor's Corner turns clinical intelligence into a peer network, not a static knowledge base.

That is the narrative spine for the landing page and for every section inside the product.

## The three product slices

### 1. Async care loop

Goal:
Between sessions, a patient can report status, talk to Nancy, and generate a clinician-useful handoff without the doctor manually reconstructing the week.

Inputs:

- patient daily check-ins
- freeform patient messages
- Nancy proactive support pings
- doctor-authored Nancy directives

Outputs:

- patient-facing Nancy replies
- clinician-facing Nancy handoffs
- async alerts
- unified patient timeline records

Design significance:

- this is not generic chat
- this is structured between-session care
- every message surface should imply continuity, safety, and real downstream impact

### 2. Clinician session loop

Goal:
Doctor opens a patient, sees prep context, runs analysis, and gets reusable diagnostic output written back to the chart.

Inputs:

- patient longitudinal context
- uploaded audio or live audio
- transcript
- acoustic affect
- retrieval-backed evidence
- optional visual/somatic BSV if camera is active

Outputs:

- transcript
- BSV
- diagnostic hypothesis
- retrieved evidence
- follow-up questions
- treatment-plan suggestion
- persisted session record
- clinician review note

Design significance:

- this is the most serious part of the product
- it should feel high-trust, evidence-linked, and clinically literate
- avoid consumer-AI styling here

### 3. Escalation loop

Goal:
When async signals become concerning, the system creates a visible and auditable escalation path instead of storing concerning text in a thread.

Inputs:

- severe check-in values
- patient safety language
- Nancy flagging
- doctor-triggered SOS

Outputs:

- urgent or watch alerts
- review queue entries
- SOS record
- hospital recommendation based on Shenzhen psychiatric-capable directory data

Design significance:

- this should feel operationally serious
- alerts must read as actionable signals, not decorative badges
- escalation UI must clearly distinguish routine vs watch vs urgent

## User roles and emotional modes

### Public visitor

What they need:

- category understanding
- confidence in the mission
- evidence that the platform is deeper than a chatbot
- a premium first impression

### Clinician

What they need:

- fast understanding of patient state
- risk visibility
- workflow clarity
- ability to act quickly
- confidence that information is structured and reviewable

Tone:

- decisive
- high-signal
- not emotionally cold
- not marketing-heavy once inside clinician workflow

### Patient

What they need:

- calm
- psychological safety
- support without confusion
- a clear distinction between what they can share and what Nancy will do with it

Tone:

- compassionate
- non-judgmental
- gentle
- never bureaucratic

### Care team / researcher / organization layer

What they need:

- portfolio and network context
- clinical credibility
- evidence-sharing and intelligence

Tone:

- editorial
- serious
- institution-grade

## System architecture in plain English

This matters because the UI should reflect the product's real architecture.

### Frontend

- `frontend/` is a Next.js 16 app
- it uses server-rendered route pages plus client components for mutation-heavy surfaces
- current visual primitives come from simple shared components like `AppShell`, `Hero`, `Card`, `StatGrid`, `Timeline`, and patient-specific UI wrappers

### Backend

- `backend_api/main.py` is the FastAPI contract for the frontend
- `backend_api/facade.py` assembles route payloads by merging seed data with runtime state
- the runtime state lives in `data/runtime/clinic_state.json`

### Domain/services

- `async_care_service.py` handles daily check-ins, patient messages, Nancy touchpoints, risk-based alerts, and async summaries
- `session_service.py` handles session prep, uploaded audio analysis, session result persistence, and clinician session notes
- `mindscape_engine.py` handles transcription, affect extraction, multimodal fusion, retrieval, LLM synthesis, and session diagnosis output
- `nancy_agent.py` defines Nancy's role, voice settings, boundaries, and Deepgram/OpenAI-compatible config
- `backend_api/nancy_voice.py` handles realtime Nancy voice conversations and tool execution during the call
- `backend_api/doctor_insights_context.py` builds the doctor voice/text insight context from workspace plus research/forum state
- `shenzhen_directory.py` loads official psychiatry-capable hospital directories and picks escalation destinations

### Product data split

- `product_data.py` holds seeded organizations, doctors, and seeded patients
- `clinic_state.py` holds mutable runtime records
- `backend_api/facade.py` merges them into route payloads

### Design implication

The UI is not just a static shell. Most sections correspond to real stored objects and real workflow transitions.

## The key domain objects the UI is actually visualizing

These are the main record types behind the current screens.

### Patient

Carries:

- diagnosis
- risk lane
- care plan
- next appointment
- history
- seeded logs, reports, goals, alerts

### Daily check-in

Captures:

- mood
- anxiety
- sleep
- energy
- stress
- cognition
- memory
- functioning
- medication adherence
- side effects
- daily update
- safety concerns
- significant events
- clinical summary

### Patient message

Represents:

- patient to Nancy
- patient to doctor
- Nancy to patient
- Nancy to doctor
- caregiver updates
- system-generated doctor notifications

Important visibility rule:

- not all messages are patient-visible

### Nancy task

Doctor-authored directive for Nancy to follow up on between sessions.

Examples:

- check sleep nightly
- ask about grounding practice
- revisit panic anticipation

### Nancy interaction

A structured handoff or touchpoint summarizing what Nancy observed or what a doctor wants Nancy to do.

Fields include:

- conversation goal
- patient report
- clinician summary
- mood and functioning notes
- cognition and medication notes
- safety note
- recommended follow-up
- escalation level
- whether it is patient-visible
- whether a patient-facing message was relayed

This object is extremely important for UI separation. It is the boundary between patient-safe language and clinician-only interpretation.

### Async alert

Represents doctor review work created by risk logic or explicit clinician action.

Fields include:

- severity: routine, watch, urgent
- source
- title
- summary
- recommended follow-up
- status: new, acknowledged, resolved

### Session record

Represents the structured output of a diagnostic session.

Includes:

- transcript excerpt
- hypothesis
- confidence
- reasoning
- treatment plan
- follow-up questions
- retrieved evidence
- BSV
- visual/somatic metrics if present
- emotion trajectory
- traumatic markers

### Session note

Clinician-authored post-session interpretation, plan update, or disposition.

### SOS event

Represents a serious escalation record with:

- severity
- reason
- district
- recommended hospital
- recommended department
- emergency numbers
- notes

### Forum thread

Represents a clinician-community post with:

- title
- body
- flair
- category
- author
- replies
- votes
- saved state
- optional live headline source

## Hard visibility boundaries the redesign must preserve

These boundaries are not cosmetic. They are core product logic.

### Boundary 1: doctor portal vs patient portal

Doctor portal contains:

- directives
- handoffs
- alerts
- chart review
- session prep and review
- structured clinical controls

Patient portal contains:

- daily check-ins
- patient-safe Nancy interactions
- support messaging
- relayed guidance

### Boundary 2: daily Nancy vs general Nancy

There are two separate Nancy conversation modes:

1. Daily conversation
   - structured
   - once per day
   - meant to gather clinical check-in data naturally

2. General support conversation
   - open-ended
   - for questions, concerns, symptoms, complaints, logistics, and support
   - explicitly not the daily questionnaire

The redesign should make this distinction obvious without making the experience feel mechanical.

### Boundary 3: patient-safe vs clinician-only information

Some Nancy touchpoints and summaries are clinician-only.

Patient-facing views should only show:

- patient-visible Nancy updates
- relayed messages
- patient-safe interpretations

Patient views must not expose:

- internal clinician reasoning
- doctor-only workflow notes
- operational triage language that is not intended for the patient

## Route-by-route UI inventory

This is the most important section for redesign planning.

### `/`
Current purpose:

- public landing / role gateway

Current meaning:

- introduces doctor vs patient portal split
- shows organizations
- currently still speaks in migration/internal-tool language

Current actions:

- `I Am A Doctor`
- `I Am A Patient`
- `View Portfolio`

Current copy problem:

- centered on routing and migration
- says things like "Choose the right entrance"
- highlights "What Changed" and references Mesop/FastAPI/Next.js

Redesign opportunity:

- turn this into a public category-defining landing page
- tell the mental-health systems story
- reveal the product in layers
- keep role selection, but after a real brand and mission narrative

### `/doctor`
Current purpose:

- doctor gateway / workspace selector

Current meaning:

- shows seeded clinicians and quick patient previews
- helps enter workspace, forum, or doctor home

Current actions:

- `Open Workspace`
- `MH Forum`
- `Doctor Home`

Current copy problem:

- still sounds like an internal entrance

Redesign opportunity:

- make this feel like a clinician entry experience, not a developer test menu

### `/patient`
Current purpose:

- patient gateway / patient portal selector

Current meaning:

- shows doctor care panels and patients under them
- provides entry to patient portal and Nancy

Current actions:

- `Open Patient Portal`
- `Talk To Nancy`

Current copy problem:

- still overly operational and directory-like

Redesign opportunity:

- make this feel like a calm entry experience for a patient or family member

### `/organizations`
Current purpose:

- organization portfolio layer

Current meaning:

- positions MindScape above single-doctor surfaces
- hints at networked institutions and research programs

Current actions:

- `Open Lead Doctor`

Current copy problem:

- currently undersold and too bare

Redesign opportunity:

- use this to support brand credibility and scale

### `/doctor/[doctorId]`
Current purpose:

- doctor home / profile / portfolio

Current meaning:

- clinician identity
- specialty and focus areas
- patient snapshot

Current actions:

- `Enter Workspace`
- `MH Forum`
- `Patient Chart`
- `Session`

Current copy problem:

- directly references "now has a real web entrance, not a Mesop screen"

Redesign opportunity:

- this should feel like a premium physician profile and command-center entry

### `/doctor/[doctorId]/workspace`
Current purpose:

- main clinician command center

Current meaning:

- this is the real operational heart of the web app
- patient panel overview
- alerts and watchtower
- intake
- outreach
- Doctor Insights voice
- forum entry

Key components:

- `DoctorInsightsVoice`
- `IntakeForm`
- `OutreachForm`
- patient cards
- async review inbox
- Nancy watchtower

Current actions:

- `Open MH Forum`
- `Brief & signals`
- `Chart`
- `Nancy Console`
- `Patient Portal`
- `Stage Intake`
- `Queue Check-in`

Current copy problem:

- too much "this now lives in a real routeable web shell"
- too much backend-tech framing

Redesign opportunity:

- treat this as the clinician operating system homepage
- patient list, alerts, and actions should feel powerful and high-trust

### `/doctor/[doctorId]/patient/[patientId]`
Current purpose:

- clinician-side patient chart

Current meaning:

- risk, status, async summary, timeline, follow-up, portal handoff

Current actions:

- `Run Session`
- `Nancy Console`
- `Patient Portal`
- `Send Nancy Support Ping`
- outreach form

Current copy problem:

- still framed around separation from Mesop and from the patient portal

Redesign opportunity:

- this should feel like a clear, premium patient intelligence summary

### `/doctor/[doctorId]/patient/[patientId]/nancy`
Current purpose:

- Nancy clinician console

Current meaning:

- doctor-side supervision and orchestration of Nancy
- clinician copilot
- directives
- touchpoints
- alerts
- SOS

Key components:

- `NancyClinicianAssistant`
- `NancyDirectiveForm`
- `NancyTouchpointForm`
- `NancySupportPingButton`
- `AlertActionButtons`
- `SosForm`

Current actions:

- `Open Patient Nancy View`
- `Ask Nancy Copilot`
- `Add Directive`
- `Save Nancy Touchpoint`
- `Acknowledge`
- `Resolve`
- `Trigger SOS Escalation`

Current copy problem:

- still partly framed as a separated console rather than a polished product system

Redesign opportunity:

- make this feel like an elite care-orchestration surface

### `/doctor/[doctorId]/patient/[patientId]/session`
Current purpose:

- session prep and analysis

Current meaning:

- pre-session context
- uploaded audio analysis
- clinician note-taking
- note timeline

Key components:

- `SessionUploadForm`
- `SessionNoteForm`

Current actions:

- upload audio
- `Analyze Uploaded Session`
- `Save Session Note`

Current copy problem:

- talks about living outside Mesop and running through a real web contract

Redesign opportunity:

- make this feel like the premium clinical intelligence core of MindScape

### `/doctor/[doctorId]/research`
Current purpose:

- Doctor's Corner / research and clinician forum

Current meaning:

- weekly brief
- performance signals
- live thread feed
- thread composition
- replies
- vote/save/share
- Doctor Insights voice

Key components:

- `DoctorForum`
- `DoctorInsightsVoice`

Current actions:

- post thread
- search
- filter
- sort
- vote
- save
- copy share link
- reply

Current copy problem:

- the ambition is strong, but it still reads like a feature pitch in places

Redesign opportunity:

- this can be one of the most distinctive parts of the public story
- it should feel like "clinical intelligence network", not just "forum"

### `/doctor/[doctorId]/patient/[patientId]/companion`
Current purpose:

- patient companion home

Current meaning:

- calmer between-session home for patient
- inbox
- message composer
- daily check-in form
- Nancy relayed guidance
- active Nancy tasks

Key components:

- `PatientPortalHero`
- `PatientContextPanel`
- `PatientTaskPanel`
- `PatientInboxThread`
- `PatientSupportCards`
- `MessageComposer`
- `CheckinForm`

Current actions:

- `Talk to Nancy`
- `Open Inbox`
- `Start With Nancy`
- `Message Nancy`
- `Send Update`
- `Send Daily Report`

Current copy problem:

- still over-explains that clinician notes stay elsewhere
- some copy is good in spirit but still too explicit and instructional

Redesign opportunity:

- this should feel like a sanctuary, not a patient dashboard

### `/doctor/[doctorId]/patient/[patientId]/companion/nancy`
Current purpose:

- patient-side Nancy experience

Current meaning:

- daily voice conversation
- open-ended text chat with Nancy
- support voice session
- patient-visible guidance and tasks

Key components:

- `NancyVoice`
- `NancyTextCompanion`
- `SpeakWithNancy`
- patient task and support cards

Current actions:

- `Daily Conversation`
- `Speak With Nancy`
- `Start Voice Session`
- `Answer Daily Check-In`
- `Send To Nancy`

Current copy problem:

- conceptually strong already, but still needs sharper product craft and more beauty

Redesign opportunity:

- this may be the most emotionally important patient-facing flow in the product

## Shared frontend components and what they really mean

### `AppShell`

Not just navigation. It encodes the three surface modes:

- public
- doctor
- patient

The redesign can change the nav system completely, but it should preserve the strong separation between those modes.

### `Hero`

Currently a generic large heading block.

Design opportunity:

- public hero should become brand storytelling
- clinician hero should become operational framing
- patient hero should become emotional grounding

### `PatientPortalHero`

This is already closer to the right direction. It contains:

- soft stat ribbon
- calmer tone
- patient-safe copy

The redesign should evolve this idea, not flatten it back into generic cards.

### `Timeline`

Current implementation collapses many record types into a generic list.

Real meaning:

- check-ins
- Nancy interactions
- messages
- sessions
- session notes
- SOS
- outreach
- alerts

Design opportunity:

- the same visual system should not be used for every event type

### `DoctorForum`

This is not a simple community feed. It is trying to be:

- Reddit for psychiatrists
- live mental-health intelligence
- case exchange
- hypothesis exchange
- peer learning
- save/share/reply workflow

The redesign should make this a signature surface.

### `NancyClinicianAssistant`

This is a doctor-only actioning copilot, not a public chatbot.

It can:

- summarize
- convert doctor instructions into Nancy tasks
- create Nancy touchpoints
- queue support pings
- create alerts
- relay patient-safe messages

The redesign should communicate that this is an orchestration tool, not a generic AI prompt box.

### `DoctorInsightsVoice`

This is a separate doctor voice assistant built from:

- workspace context
- alerts
- Nancy watchtower
- research/forum data

It should feel like a compact insight layer, not like a duplicate of Nancy.

## What the important actions actually do

This section exists so the redesign model understands the meaning of the main buttons, not just their labels.

### Public and gateway actions

- `I Am A Doctor`
  Sends the user into clinician entry and workspace selection.

- `I Am A Patient`
  Sends the user into patient-side access and Nancy entry.

- `View Portfolio`
  Jumps into organization-level credibility and doctor-network context.

- `Open Workspace`
  Opens the clinician's main operating surface.

- `MH Forum`
  Opens the clinician intelligence and community layer.

- `Doctor Home`
  Opens the doctor's profile-style overview.

- `Open Patient Portal`
  Opens the patient-facing companion home.

- `Talk To Nancy`
  Opens the patient-side Nancy experience directly.

### Workspace actions

- `Chart`
  Opens the clinician patient chart with timeline, risk state, and follow-up.

- `Nancy Console`
  Opens the doctor-side orchestration console for Nancy.

- `Patient Portal`
  Opens the patient's own companion view from the clinician side.

- `Stage Intake`
  Creates a new runtime patient record with concern, context, initial status, and seeded placeholder chart data.

- `Queue Check-in` and outreach actions
  Store a follow-up or outreach log for clinician coordination.

- `Open MH Forum`
  Opens Doctor's Corner from the operational workspace because research and clinical discussion are positioned as part of clinician workflow, not a separate app.

- `Brief & signals`
  Opens the research/forum route where weekly briefs, performance signals, threads, and Doctor Insights live.

### Patient chart actions

- `Run Session`
  Opens session prep, analysis, and post-session note workflow.

- `Nancy Console`
  Opens doctor-side Nancy supervision for that patient.

- `Patient Portal`
  Lets the clinician inspect the patient-facing experience.

- `Send Nancy Support Ping`
  Queues a proactive Nancy outreach message based on recent patient state.

- clinician outreach form submission
  Stores a directed follow-up record without using the patient portal chat UI.

### Nancy clinician console actions

- `Ask Nancy Copilot`
  Sends a doctor instruction or question into the clinician copilot, which may summarize records or convert intent into structured Nancy actions.

- `Add Directive`
  Creates a doctor-authored Nancy task that changes future Nancy follow-up behavior.

- `Save Nancy Touchpoint`
  Creates a structured Nancy handoff record and can optionally relay a patient-facing message.

- `Acknowledge`
  Marks an alert as seen by the doctor.

- `Resolve`
  Marks an alert as resolved.

- `Trigger SOS Escalation`
  Creates a serious escalation record and selects a psychiatry-capable hospital recommendation using Shenzhen routing logic.

- `Open Patient Nancy View`
  Opens the patient-facing Nancy side for this patient, not the clinician console.

### Session actions

- `Analyze Uploaded Session`
  Uploads audio, runs transcription plus MindScape diagnostic analysis, and persists the resulting session record.

- `Save Session Note`
  Stores a clinician-authored post-session note with title, interpretation, plan update, and disposition.

### Patient companion actions

- `Send Update`
  Stores the patient message, runs async-care reasoning, may trigger a Nancy reply, may generate a clinician handoff, and may create an alert.

- `Send Daily Report`
  Stores a structured patient check-in, generates a Nancy handoff, and may create a doctor review alert.

- `Start With Nancy`
  Opens the daily Nancy flow when today's structured conversation is due.

- `Message Nancy`
  Opens the general Nancy conversation lane when the daily flow is already complete.

### Patient Nancy actions

- `Start Daily Conversation`
  Opens the once-daily Nancy voice flow that naturally gathers check-in data and logs it.

- `Answer Daily Check-In`
  Progresses the text-based guided daily conversation and submits a structured check-in at completion.

- `Send To Nancy`
  Sends a freeform patient message into the open-ended Nancy support lane.

- `Start Voice Session`
  Opens the support-mode Nancy voice call for non-daily conversation.

### Doctor's Corner actions

- `Post to forum`
  Creates a runtime clinician thread.

- `Reply`
  Adds a reply to a thread.

- vote action
  Increases thread visibility score.

- `Save thread`
  Marks a thread for future retrieval.

- `Copy share link`
  Generates a direct link to a specific thread for rounds, slides, or committee discussion.

### Realtime voice actions

- Nancy voice tools can submit a daily check-in, create a doctor alert, or log an important patient statement while the conversation is happening.

- Doctor Insights voice can summarize workspace, alerts, watchtower, weekly brief, and forum context for the clinician.

## Backend behavior that the redesign should understand

### Daily check-ins trigger real logic

When a patient submits a daily check-in:

- it is stored
- risk is evaluated
- Nancy may generate a patient-facing response
- a Nancy interaction is created
- an alert may be created for the doctor

So the check-in surface is not a dead-end form.

### Patient messages are not plain chat

When a patient sends a message:

- it is stored
- async-care reasoning runs
- Nancy may reply to the patient
- Nancy may send a clinician-facing summary
- an alert may be created

So messaging is part of care orchestration.

### Nancy directives really change later patient conversations

Doctor-authored Nancy tasks are inserted into the daily Nancy question flow and become part of the future patient experience.

### Voice sessions can execute actions live

During Nancy voice calls, the realtime session can:

- submit a daily check-in
- create a doctor alert
- log patient statements

### Session analysis is the heavy clinical engine

The session engine does:

- transcription via Gemini, Whisper, or local SenseVoice
- acoustic affect extraction
- optional visual/somatic BSV
- retrieval over clinical corpus
- LLM synthesis into diagnosis hypothesis and evidence

This is one of the deepest product capabilities and should be visually positioned that way.

## Current UX and copy problems

The redesign should explicitly correct these.

### Problem 1: the UI explains itself like an internal migration project

Examples of the current tone:

- "now has a real web entrance"
- "not a Mesop screen"
- "outside Mesop"
- "the Python clinical runtime is preserved behind FastAPI"
- "OpenAI Realtime"
- "clean routeable web shell"

These are engineering migration notes, not public product copy.

### Problem 2: many pages are framed as portals or gateways instead of outcomes

"Doctor portal", "patient portal", and "role gateway" are functionally correct, but not emotionally strong enough as primary product language.

### Problem 3: too much explanation of internal separation

The doctor/patient boundary is important, but the current UI often says it too literally.

Example pattern:

- "this page is now doctor-facing only"
- "patient messaging lives elsewhere"

The redesign should imply this through structure and tone, not repeated defensive copy.

### Problem 4: public story is underdeveloped

The product has unusually strong raw ingredients:

- live session intelligence
- async AI companion
- risk review and escalation
- clinician community / Doctor's Corner
- longitudinal mental-health context

But the current landing and public narrative do not elevate them enough.

### Problem 5: too many surfaces feel equally important

The redesign should establish stronger hierarchy:

- public narrative and category story
- clinician operating system
- patient compassionate experience
- signature intelligence modules

## Copy strategy for the redesign

### Replace internal-tool language with product language

Avoid primary copy that sounds like:

- platform migration
- architecture notes
- route wiring
- "this lives here now"
- "this no longer depends on X"

Prefer copy that sounds like:

- product value
- clinical impact
- confidence
- care continuity
- intelligence with compassion

### Recommended tone by surface

#### Public / landing

- visionary
- category-defining
- elegant
- clear
- emotionally aware

#### Clinician surfaces

- concise
- crisp
- professional
- high-signal
- decision-friendly

#### Patient surfaces

- warm
- humane
- gentle
- reassuring
- never patronizing

## Landing page story architecture

This is the recommended narrative structure for the redesign.

### Beat 1: the problem

Show the failures of current mental-health systems:

- delayed diagnosis
- fragmented longitudinal context
- poor reach
- treatment irrelevance or mismatch over long periods
- between-session deterioration going unnoticed

The user requested subtle floating quotes or restrained moving headlines around these systemic failures. That can work, but keep it elegant and understated.

### Beat 2: the thesis

MindScape is the operating system that connects what breaks between visits.

### Beat 3: the layers of MindScape

Reveal the product progressively:

1. live session intelligence
2. between-session Nancy support
3. clinician review and alerts
4. escalation routing
5. Doctor's Corner intelligence network

### Beat 4: the emotional contrast

Make it clear that the product is both:

- clinically sharp for doctors
- emotionally safe for patients

### Beat 5: the distinctive features

Especially emphasize:

- Nancy AI
- Doctor's Corner
- longitudinal patient-aware session analysis

### Beat 6: architecture credibility

The landing page can include technical breakdown sections, but they must be productized and visual, not just engineering prose.

## Meaningful placeholder copy map

Use these as rewrite targets. The redesign model should replace current copy with content that fits these intents.

### Landing page

- `landing.hero.eyebrow`
  Short category statement, not "Portal Gateway"

- `landing.hero.title`
  The strongest statement of MindScape's mission

- `landing.hero.body`
  Explain how MindScape closes the gap between sessions, signals, and decisions

- `landing.problem.headlines[]`
  Short, credible lines about mental-health system failures

- `landing.layer_1.title`
  Live session intelligence

- `landing.layer_1.body`
  Real-time and uploaded sessions become structured diagnostic context

- `landing.layer_2.title`
  Nancy between visits

- `landing.layer_2.body`
  Patients get a calm companion; clinicians get structured follow-up

- `landing.layer_3.title`
  Signals that do not get lost

- `landing.layer_3.body`
  Alerts, handoffs, and escalation paths instead of buried text

- `landing.layer_4.title`
  Doctor's Corner

- `landing.layer_4.body`
  Clinician intelligence network, not generic community noise

- `landing.architecture.title`
  Technical credibility section

- `landing.architecture.body`
  Explain the clinical engine, retrieval, Nancy, and workflow system in product language

- `landing.primary_cta`
  Public-facing action, not just role routing

### Doctor gateway

- `doctor_gateway.hero.title`
  Clinician entry into the operating system

- `doctor_gateway.hero.body`
  Speak to psychiatrists, coordinators, and operators

- `doctor_gateway.card.subtitle`
  Specialty and institution

- `doctor_gateway.card.meta`
  Patient count and current operational posture

- `doctor_gateway.primary_cta`
  Workspace entry

### Patient gateway

- `patient_gateway.hero.title`
  Calm patient-side entry

- `patient_gateway.hero.body`
  Explain support, check-ins, and Nancy without clinician jargon

- `patient_gateway.panel.title`
  Care team relationship, not directory-only language

- `patient_gateway.primary_cta`
  Enter patient experience

### Doctor workspace

- `workspace.hero.title`
  The daily command center

- `workspace.hero.body`
  Frame around clarity, follow-up, and action

- `workspace.voice_module.title`
  Doctor Insights

- `workspace.patient_list.title`
  Patient panel overview

- `workspace.alerts.title`
  Review queue

- `workspace.watchtower.title`
  Recent Nancy handoffs

- `workspace.intake.title`
  New patient intake

- `workspace.outreach.title`
  Follow-up coordination

### Patient chart

- `patient_chart.hero.title`
  High-level patient intelligence summary

- `patient_chart.overview.title`
  Clinical overview

- `patient_chart.timeline.title`
  Longitudinal timeline

- `patient_chart.followup.title`
  Clinician follow-up

- `patient_chart.portal_handoff.title`
  What the patient sees

### Nancy clinician console

- `nancy_console.hero.title`
  Oversee and direct Nancy

- `nancy_console.copilot.title`
  Clinician copilot

- `nancy_console.directives.title`
  Doctor directives for Nancy

- `nancy_console.touchpoints.title`
  Structured handoffs

- `nancy_console.alerts.title`
  Review queue

- `nancy_console.sos.title`
  Escalation path

### Session page

- `session.hero.title`
  Session prep and clinical analysis

- `session.prep.title`
  What matters before the session

- `session.analysis.title`
  Analyze session audio

- `session.note.title`
  Clinician interpretation

- `session.timeline.title`
  Prior session notes and outputs

### Patient companion home

- `companion.hero.title`
  Calm between-session care space

- `companion.hero.body`
  Reassure, simplify, and humanize

- `companion.notice_due.title`
  Nancy check-in needed today

- `companion.notice_complete.title`
  Today's check-in is complete

- `companion.messages.title`
  Message Nancy or your care team

- `companion.inbox.title`
  Messages and relayed guidance

- `companion.checkin.title`
  Daily check-in

- `companion.guidance.title`
  Recent Nancy support

### Patient Nancy page

- `patient_nancy.hero.title`
  Nancy is ready

- `patient_nancy.daily.title`
  Daily conversation

- `patient_nancy.open_chat.title`
  Speak with Nancy

- `patient_nancy.voice.title`
  Live voice session

- `patient_nancy.guidance.title`
  Good uses for Nancy

### Doctor's Corner / research

- `research.hero.title`
  A mental-health intelligence network

- `research.hero.body`
  Peer exchange, literature, methods, cases, hypotheses

- `research.brief.title`
  Weekly intelligence brief

- `research.voice.title`
  Doctor Insights

- `research.composer.title`
  Start a thread

- `research.feed.title`
  Active clinical discourse

## Design recommendations by surface

### Public landing

- strongest visual ambition in the product
- editorial storytelling
- restrained ambient motion
- layered reveal of system capabilities
- premium typography and pacing

### Clinician surfaces

- denser information design
- fewer decorative gradients
- stronger hierarchy and signal management
- risk state should be obvious at a glance

### Patient surfaces

- softer pacing
- more whitespace
- warmer copy
- fewer sharp data tables
- more guided interaction framing

### Nancy surfaces

- should feel deeply cared-for, not sci-fi
- voice states should feel alive but calm
- realtime transcript elements should feel elegant

### Doctor's Corner

- can be bold and distinctive
- should feel like a premium intelligence network for clinicians
- not meme culture
- not generic community SaaS

## Features that should be visually spotlighted in the redesign

These are the most brand-defining capabilities.

### 1. Nancy AI

Why it matters:

- patient companion
- async support
- doctor handoffs
- directives
- voice and text

### 2. Live session intelligence

Why it matters:

- MindScape's strongest technical core
- differentiates the product from simple note-taking or messaging tools

### 3. Doctor's Corner

Why it matters:

- unique brand lever
- gives MindScape a network effect and intelligence story

### 4. Longitudinal patient context

Why it matters:

- everything in the system is stronger because patient state persists across time

### 5. Escalation and review

Why it matters:

- shows the product is not just about pretty AI conversations
- it has operational seriousness

## Things the redesign must not accidentally break conceptually

- doctor and patient surfaces must remain clearly separated
- patient-safe content rules must remain intact
- daily Nancy and general Nancy must remain distinct
- alerts must remain actionable, not cosmetic
- session analysis should remain the product's clinical core
- Nancy clinician console should remain an orchestration surface, not a second patient chat UI
- Doctor's Corner should remain clinically serious

## Optional but powerful design motifs

These fit the user's prompt and the product truth.

- restrained floating problem statements on landing
- layered "unfolding" reveal of MindScape capabilities
- contrast between fragmented care today vs coherent care with MindScape
- subtle continuity motifs across patient timelines, handoffs, and signals
- soft but precise motion around Nancy voice/listening states

## Suggested one-line brand territory

Not final copy, just direction:

- "MindScape brings intelligence, continuity, and compassion into mental-health care."
- "From live sessions to between-session care, MindScape keeps the signal intact."
- "A clinical operating system for mental-health care that does not lose the human being."

## Optional source anchors

Claude should not need to read these if it trusts this document, but these are the anchor files behind the product truth:

- `frontend/app/page.tsx`
- `frontend/app/doctor/[doctorId]/workspace/page.tsx`
- `frontend/app/doctor/[doctorId]/patient/[patientId]/page.tsx`
- `frontend/app/doctor/[doctorId]/patient/[patientId]/nancy/page.tsx`
- `frontend/app/doctor/[doctorId]/patient/[patientId]/session/page.tsx`
- `frontend/app/doctor/[doctorId]/patient/[patientId]/companion/page.tsx`
- `frontend/app/doctor/[doctorId]/patient/[patientId]/companion/nancy/page.tsx`
- `frontend/app/doctor/[doctorId]/research/page.tsx`
- `frontend/components/DoctorForum.tsx`
- `frontend/components/NancyVoice.tsx`
- `frontend/components/NancyTextCompanion.tsx`
- `frontend/components/NancyClinicianAssistant.tsx`
- `backend_api/main.py`
- `backend_api/facade.py`
- `async_care_service.py`
- `session_service.py`
- `mindscape_engine.py`
- `nancy_agent.py`
- `clinic_state.py`
- `product_data.py`
- `ARCHITECTURE.md`
- `MVP_RECOVERY_PLAN.md`

## Final instruction to the redesign model

Treat the current UI as a functional scaffold, not as the right narrative, tone, or product craft.

Preserve the workflow truth.
Upgrade the product story.
Sharpen the information hierarchy.
Make the patient experience feel cared for.
Make the clinician experience feel elite.
Make the public landing page worthy of being shown outside the team.
