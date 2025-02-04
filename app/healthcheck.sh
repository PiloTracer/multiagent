#!/usr/bin/env bash
# healthcheck.sh - Return 0 (healthy) if index.faiss exists, else 1 (unhealthy).

if [ -f "./faiss_store/index.faiss" ]; then
  exit 0
else
  exit 1
fi
