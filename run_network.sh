#!/bin/bash
# Run MindScape accessible on the local network (same WiFi)
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
echo ""
echo "  MindScape starting on local network..."
echo "  Your friend can open: http://${LOCAL_IP}:32123"
echo ""
mesop --host=0.0.0.0 app.py
