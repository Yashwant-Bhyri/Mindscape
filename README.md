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
- macOS (recommended for MPS/Metal acceleration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/MindScape.git
   cd MindScape
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

4. **Configure Environment Variables**:
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

Start the Mesop server:
```bash
mesop app.py
```
Open your browser and navigate to `http://localhost:32123`.

## 🗺 Routes

- `/` product landing page
- `/organizations` organization and researcher portfolio
- `/doctor?doctor=<doctor-id>` doctor profile
- `/patient?doctor=<doctor-id>&patient=<patient-id>` patient portfolio
- `/companion?doctor=<doctor-id>&patient=<patient-id>` patient async-care companion portal
- `/workspace?doctor=<doctor-id>` doctor command center
- `/nancy?doctor=<doctor-id>&patient=<patient-id>` Nancy care-companion console
- `/session?doctor=<doctor-id>&patient=<patient-id>` live diagnostic session
- `/research?doctor=<doctor-id>` research and doctor's corner

## 🛠 Tech Stack

- **UI**: [Mesop](https://google.github.io/mesop/)
- **ASR**: [SenseVoiceSmall](https://github.com/alibaba-damo-academy/FunASR)
- **LLM**: OpenRouter + DeepSeek V4 Flash by default
- **Voice Agent**: Deepgram-ready Nancy settings payload with `nova-3-medical` listening and `aura-2-vesta-en` speaking
- **Fast Nancy Think Model**: `gpt-4o-mini` is the default Nancy reasoning model for lower latency conversational replies
- **Retrieval**: Hybrid evidence retrieval over the clinical corpus
- **Verification**: Cross-Encoder NLI

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.
