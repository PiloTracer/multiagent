#!/usr/bin/env bash
set -e

echo "Running ingestion step..."
python ingest_docs.py
echo "Ingestion done!"

echo "Starting FastAPI server..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
