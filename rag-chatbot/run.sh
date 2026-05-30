#!/bin/bash
# run.sh — Convenience launcher for the RAG chatbot
# Usage: bash run.sh

PYTHON=/home/runner/workspace/.pythonlibs/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

# Step 1: Ingest documents if the database doesn't exist yet
if [ ! -d "chroma_db" ]; then
    echo "No database found. Running ingestion first..."
    $PYTHON ingest.py
    echo ""
fi

# Step 2: Start the chat
$PYTHON chat.py
