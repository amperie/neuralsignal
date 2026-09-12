#!/usr/bin/env bash
set -uo pipefail

RUN_ID="${NEURALSIGNAL_RUN_ID:-}"
if [[ -z "$RUN_ID" ]]; then
  previous=""
  for arg in "$@"; do
    if [[ "$previous" == "--run-id" ]]; then
      RUN_ID="$arg"
      break
    fi
    previous="$arg"
  done
fi
RUN_ID="${RUN_ID:-run}"
SESSION="neuralsignal-${RUN_ID}"
LOG_DIR="${NEURALSIGNAL_RUN_WORKDIR:-/workspace/neuralsignal-runs}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/runpod.log"
STATUS_FILE="${LOG_DIR}/exit_code"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

CMD=(uv run python -m neuralsignal.remote.job "$@")
printf -v TMUX_CMD ' %q' "${CMD[@]}"
TMUX_CMD="${TMUX_CMD:1}"

{
  echo "tmux session: ${SESSION}"
  echo "log file: ${LOG_FILE}"
  echo "attach with: tmux attach -t ${SESSION}"
  echo
} | tee -a "$LOG_FILE"

tmux new-session -d -s "$SESSION" "cd /opt/neuralsignal && set -o pipefail; ${TMUX_CMD} 2>&1 | tee -a '$LOG_FILE'; status=\${PIPESTATUS[0]}; echo \$status > '$STATUS_FILE'; exit \$status"

tail -n +1 -F "$LOG_FILE" &
TAIL_PID=$!
while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep 2
done
sleep 1
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

if [[ -f "$STATUS_FILE" ]]; then
  exit "$(cat "$STATUS_FILE")"
fi
exit 1
