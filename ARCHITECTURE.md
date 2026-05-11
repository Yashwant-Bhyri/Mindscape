# MindScape Architecture Reality Check

## Current Truth

This repository is not a finished clinical platform.
It is a hybrid of:

- one real core engine: session transcription + retrieval-backed diagnostic analysis
- one growing prototype shell: doctor, patient, Nancy, research, and organization pages
- one demo persistence layer: local JSON state under `data/runtime`

The biggest engineering problem is not lack of ideas.
It is lack of hard boundaries between:

- presentation
- workflow logic
- domain logic
- infrastructure
- mock/demo content

Right now those layers bleed into each other, especially inside `app.py`.

## What Is Actually Real

These parts behave like genuine product flows:

1. Session analysis pipeline
   - audio capture / upload
   - transcription
   - multimodal fusion
   - retrieval
   - LLM synthesis
   - result persistence into patient session records

2. Local runtime persistence
   - patient intake
   - daily check-ins
   - session records
   - Nancy directives
   - Nancy handoffs
   - patient messages
   - SOS event logs

3. Async care loop foundation
   - patient questionnaire submission
   - patient-to-Nancy messaging
   - Nancy-generated patient reply
   - clinician-facing handoff summary
   - unified async timeline

## What Is Still Prototype-Level

These surfaces exist, but should not be treated as production-complete:

1. Nancy live voice
   - prompt and provider config exist
   - browser or telephony voice session is not fully integrated

2. Chat
   - in-app persistence exists
   - there is no delivery system, identity model, unread state, or notification model

3. SOS
   - routing logic and records exist
   - there is no live emergency dispatch or hospital communication integration

4. Research / Doctor's Corner / Organizations
   - pages exist
   - data is mostly static or mock-backed
   - no authoring, syncing, moderation, retrieval, or collaboration workflow exists

5. Security / auth
   - none
   - doctor and patient context are still URL-selected

## Structural Problems

### 1. Monolithic app layer

`app.py` currently mixes:

- route definitions
- layout code
- state mutations
- workflow orchestration
- business logic
- infrastructure calls

This is the main source of fragility.

### 2. No domain contracts

Most flows pass around raw `dict` objects.
That makes it too easy for:

- required fields to be missing
- timelines to drift in shape
- UI assumptions to silently break

### 3. Demo data and runtime data are interleaved

`product_data.py` and `clinic_state.py` are both acting like source-of-truth layers.
That is acceptable for prototype speed, but dangerous unless explicitly managed.

### 4. Weak verification

Compile checks exist.
Systematic workflow verification does not.

## Target Architecture

This is the intended near-term architecture for a serious MVP.

### 1. Presentation

Mesop pages and components only.

Responsibilities:

- render state
- gather user input
- invoke application services
- show status and errors

Should not own:

- business rules
- risk logic
- record-shaping logic
- provider-specific AI orchestration

### 2. Application Services

Thin orchestration layer for product workflows.

Examples:

- `session_service`
- `async_care_service`
- `nancy_service`
- `patient_service`

Responsibilities:

- execute a user-facing workflow end to end
- call domain logic and persistence
- return structured results for UI

### 3. Domain Logic

Pure business rules.

Examples:

- risk evaluation
- async timeline composition
- summary generation
- event normalization
- escalation policy

This should be testable without Mesop.

### 4. Infrastructure

Adapters for:

- local JSON persistence
- LLM providers
- Deepgram settings/config
- Shenzhen directory loading
- session audio/vision engines

### 5. Demo Content

Static seeded organizations, doctors, and patients belong in a clearly-marked seed layer.

## Rules For Future Development

From this point on:

1. No new page or feature should be added unless it belongs to one of the three core MVP flows.
2. Every feature must have a defined source of truth.
3. Every user-facing workflow must have one service entry point.
4. Every feature must declare whether it is:
   - seed/mock-backed
   - local-runtime backed
   - external-provider backed
5. No more hidden dependencies on Mesop query params inside reusable helpers.

## Immediate Refactor Direction

The next serious engineering move is not “more UI.”
It is:

1. extract the async care flow into a first-class service boundary
2. extract the session workflow into a first-class service boundary
3. define typed record shapes for patient check-ins, Nancy handoffs, messages, session records, and SOS events
4. make the doctor workspace and patient profile read from those stable shapes

That is how the product stops behaving like a stitched demo and starts behaving like software with a backbone.
