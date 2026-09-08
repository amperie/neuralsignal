#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-ghcr.io/amperie/neuralsignal-runpod-base:latest}"
docker build -f Dockerfile.runpod -t "$IMAGE" .
if [[ "${PUSH:-0}" == "1" ]]; then
  docker push "$IMAGE"
fi

