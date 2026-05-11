#!/bin/bash
# MindScape Clinical OS — Next.js + FastAPI
set -e
cd "$(dirname "$0")"

LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo ""
echo "  MindScape Clinical OS"
echo "  ─────────────────────────────────────────"
echo "  Backend API:  http://localhost:8002"
echo "  Frontend:     http://localhost:3001"
echo "  Network:      http://${LOCAL_IP}:3001"
echo "  API Docs:     http://localhost:8002/docs"
echo "  ─────────────────────────────────────────"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

# Start FastAPI backend
uvicorn backend_api.main:app --reload --port 8002 --host 0.0.0.0 &
BACKEND_PID=$!

cleanup() {
  echo ""
  echo "Stopping servers..."
  kill $BACKEND_PID 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

# Give backend a moment to start
sleep 1

# Start Next.js frontend (foreground)
cd frontend
NEXT_PUBLIC_API_BASE="http://${LOCAL_IP}:8002/api" npm run dev -- --port 3001 --hostname 0.0.0.0

# If Next.js exits, kill backend too
kill $BACKEND_PID 2>/dev/null || true
