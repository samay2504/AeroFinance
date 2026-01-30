#!/bin/bash
# Bash startup script - respects FASTAPI_PORT from .env
# Usage: chmod +x start.sh && ./start.sh

# Load .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Read values or use defaults
PORT=${FASTAPI_PORT:-9999}
HOST=${FASTAPI_HOST:-0.0.0.0}

echo "Starting AI-CA on ${HOST}:${PORT}"

# Start uvicorn with env-derived port
uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
