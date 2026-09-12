#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-ghcr.io/amperie/neuralsignal-runpod-base:latest}"
cd "$(dirname "$0")/.."
OUTPUT=--load
if [[ "${PUSH:-1}" == "1" ]]; then OUTPUT=--push; fi
docker buildx build --platform linux/amd64 -f Dockerfile.runpod -t "$IMAGE" "$OUTPUT" .
