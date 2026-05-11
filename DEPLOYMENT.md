# Deployment Guide

## Current Shape

MindScape currently has two app surfaces:

- `frontend/`: the primary Next.js UI
- `backend_api/`: the primary FastAPI backend

There is also a legacy Mesop app in `app.py`, but the main product shell now lives in the Next.js frontend.

## Local Development

### 1. Python setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Frontend setup

```bash
cd frontend
npm install
cd ..
```

### 3. Environment variables

```bash
cp .env.example .env
```

Useful variables:

- `OPENROUTER_API_KEY`
- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_MODEL=deepseek/deepseek-v4-flash`
- `GEMINI_API_KEY`
- `WHISPER_API_KEY`
- `DEEPGRAM_API_KEY`
- `NANCY_OPENAI_MODEL=gpt-4o-mini`

Optional runtime switches:

- `MINDSCAPE_SESSION_FULL_DIAGNOSIS=1`
  enables the heavier session-analysis path instead of the lightweight stub
- `MINDSCAPE_RUNTIME_FILE=/path/to/clinic_state.json`
  overrides the local runtime JSON store

### 4. Start both services

```bash
./start.sh
```

Default ports:

- backend: `8002`
- frontend: `3001`
- OpenAPI docs: `http://localhost:8002/docs`

## Running Services Separately

Backend:

```bash
uvicorn backend_api.main:app --reload --port 8002
```

Frontend:

```bash
cd frontend
npm run dev -- --port 3001
```

If you run the frontend separately, set:

```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8002/api
```

## Production Notes

This repo is not production-hardened yet. The biggest caveats are:

- authentication and authorization are not complete
- local JSON runtime persistence is for MVP / local workflows, not multi-user production
- some research / community surfaces are still seed-backed
- voice and live multimodal paths may require machine-specific setup

## Recommended Deployment Split

### Frontend

The Next.js app can be deployed independently from `frontend/`.

Build check:

```bash
cd frontend
npm run build
```

Required env:

- `NEXT_PUBLIC_API_BASE`

### Backend

The FastAPI backend can be deployed as a separate Python service.

Run check:

```bash
pytest -q
```

Launch:

```bash
uvicorn backend_api.main:app --host 0.0.0.0 --port 8002
```

## Legacy Mesop Surface

If you need the older Mesop interface:

```bash
mesop app.py
```

That path should be treated as legacy / experimental relative to the current frontend.
