#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-ghcr.io/amperie/neuralsignal-runpod-base:latest}"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${GH_TOKEN:?GH_TOKEN is required in .env}"
OWNER="${IMAGE#ghcr.io/}"
OWNER="${OWNER%%/*}"
printf "%s" "$GH_TOKEN" | docker login ghcr.io -u "$OWNER" --password-stdin
docker build -f Dockerfile.runpod -t "$IMAGE" .
docker push "$IMAGE"
