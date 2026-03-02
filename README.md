# 🧠 MindScape // 2.0

MindScape is an advanced clinical intelligence engine designed to assist psychiatrists. It analyzes live audio or uploaded recordings to interpret behavioral states and provide diagnostic hypotheses based on DSM-5 criteria.

## ✨ Features

- **Live Transcription**: Real-time audio processing using SenseVoiceSmall.
- **Behavioral State Vector (BSV)**: Analyzes Valence, Arousal, and Dominance.
- **Clinical Intelligence**: Maps transcript evidence to DSM-5 criteria using LLMs (Gemini or DeepSeek).
- **Safety Gate**: NLI-based verification of diagnostic evidence.
- **Silicon Valley Dark Mode**: A premium, glassmorphic UI built with Mesop.

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

### Running the App

Start the Mesop server:
```bash
mesop app.py
```
Open your browser and navigate to `http://localhost:32123`.

## 🛠 Tech Stack

- **UI**: [Mesop](https://google.github.io/mesop/)
- **ASR**: [SenseVoiceSmall](https://github.com/alibaba-damo-academy/FunASR)
- **LLM**: Google Gemini 2.0 Flash / DeepSeek Chat
- **Verification**: Cross-Encoder NLI

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.
