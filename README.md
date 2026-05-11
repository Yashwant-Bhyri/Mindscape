# 🧠 MindScape Clinical OS

MindScape is a clinical-platform MVP in recovery. It combines a real session-analysis core with a broader multi-page product shell for psychiatrists, researchers, and mental-health organizations.

Important truth:

- the session-analysis pipeline is the strongest fully connected part
- the async-care layer is now becoming a serious end-to-end slice
- some surrounding product surfaces are still prototype-level and mock-backed

See [ARCHITECTURE.md](ARCHITECTURE.md) and [MVP_RECOVERY_PLAN.md](MVP_RECOVERY_PLAN.md) for the current engineering truth and build standard.

## ✨ Current Capabilities

- **Multi-Page Product Shell**: Landing page, organizations portfolio, doctor profiles, patient profiles, research area, and session workspace.
- **Doctor Workspace**: Patient dashboards, appointments, diagnosis reports, check-in drafting, and intake staging.
- **Patient Portfolios**: Health history, diagnosis reports, personal logs, care-team notes, alerts, daily questionnaires, and session entry points.
- **Local Runtime Persistence**: New patient intakes and queued outreach notes are stored in local runtime JSON data.
- **Daily Clinical Questionnaire**: Off-consultation reports capture mood, anxiety, sleep, cognition, memory, functioning, medication adherence, safety, and significant events.
- **Live Transcription**: Real-time audio processing using SenseVoiceSmall.
- **Behavioral State Vector (BSV)**: Analyzes Valence, Arousal, and Dominance.
- **Clinical Intelligence**: Maps transcript evidence to DSM-5 criteria using retrieval and a stronger LLM path.
- **Patient-Aware Sessions**: The live diagnosis flow can incorporate the selected patient's longitudinal context.
- **Session Analytics**: Post-therapy records preserve diagnosis hypotheses, BSV, emotional phase changes, disclosure peaks, and treatment pathway suggestions in the patient chart.
- **Research + Reference Layer**: Present in the UI, but still largely seed/mock-backed.
- **Nancy Care Companion**: Stronger async-care loop with conversational check-ins, reminders, patient handoffs, and a Deepgram voice-agent configuration layer.
- **Shared Messaging**: Persistent in-product messaging for patient, Nancy, and doctor workflow continuity.
- **Shenzhen SOS Routing**: In-product escalation records and hospital recommendation logic based on doctor hospital/department and official psychiatry-capable hospital listings.

## ⚠️ Product Maturity

Treat the repo as a serious MVP, not a finished production system.

Most mature:

- session analysis pipeline
- local persistence for clinical workflow records
- async-care loop foundation

Still prototype-level:

- live Nancy voice session integration
- research/community collaboration layer
- full organizational network workflows
- authentication and role security

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 20+
- npm 10+
- macOS recommended for local audio / vision workflows

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Yashwant-Bhyri/Mindscape.git
   cd Mindscape
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   Recommended default:
   - `OPENROUTER_API_KEY`
   - `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
   - `OPENAI_MODEL=deepseek/deepseek-v4-flash`
   - `DEEPGRAM_API_KEY` if you want to activate Nancy's live voice stack

### Running the App

Recommended local dev stack:

```bash
./start.sh
```

This starts:

- FastAPI backend on `http://localhost:8002`
- Next.js frontend on `http://localhost:3001`
- API docs on `http://localhost:8002/docs`

If you want to run the services manually:

```bash
uvicorn backend_api.main:app --reload --port 8002
```

```bash
cd frontend
npm run dev -- --port 3001
```

Legacy Mesop app:

```bash
mesop app.py
```

That path still exists for the older UI, but the current primary product shell is the Next.js frontend.

## 🗺 Routes

- `/` product landing page
- `/organizations` organization and researcher portfolio
- `/doctor` doctor gateway
- `/doctor/<doctorId>` doctor profile
- `/doctor/<doctorId>/workspace` doctor command center
- `/doctor/<doctorId>/research` research and Doctor's Corner
- `/doctor/<doctorId>/patient/<patientId>` clinician patient chart
- `/doctor/<doctorId>/patient/<patientId>/nancy` clinician Nancy console
- `/doctor/<doctorId>/patient/<patientId>/session` session intelligence
- `/doctor/<doctorId>/patient/<patientId>/companion` patient companion portal
- `/doctor/<doctorId>/patient/<patientId>/companion/nancy` patient Nancy surface

## 🛠 Tech Stack

- **Primary UI**: Next.js App Router frontend
- **Legacy UI**: [Mesop](https://google.github.io/mesop/)
- **Backend**: FastAPI
- **ASR**: [SenseVoiceSmall](https://github.com/alibaba-damo-academy/FunASR)
- **LLM**: OpenRouter + DeepSeek V4 Flash by default
- **Voice Agent**: Deepgram-ready Nancy settings payload with `nova-3-medical` listening and `aura-2-vesta-en` speaking
- **Fast Nancy Think Model**: `gpt-4o-mini` is the default Nancy reasoning model for lower latency conversational replies
- **Retrieval**: Hybrid evidence retrieval over the clinical corpus
- **Verification**: Cross-Encoder NLI

## 🚢 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for:

- local launch flow
- backend / frontend split
- environment variables
- production caveats
- what is and is not production-ready yet

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.
