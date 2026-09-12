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
SESSION="ns"
LOG_DIR="${NEURALSIGNAL_RUN_WORKDIR:-/workspace/neuralsignal-runs}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/runpod.log"
STATUS_FILE="${LOG_DIR}/exit_code"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

CMD=(/opt/neuralsignal/.venv/bin/python -m neuralsignal.remote.job "$@")
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${CMD[@]}"
fi
# RunPod supplies account public keys through PUBLIC_KEY for custom images.
SSH_KEYS="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [[ -n "$SSH_KEYS" ]]; then
  mkdir -p /root/.ssh /run/sshd
  chmod 700 /root/.ssh
  printf '%s\n' "$SSH_KEYS" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A
  if /usr/sbin/sshd -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password; then
    echo "SSH server started on port 22 (public-key authentication)"
  else
    echo "SSH server failed to start; see the error above. Feature collection will continue." >&2
  fi
else
  echo "SSH unavailable: add your public SSH key to RunPod account settings before launching."
fi

printf -v TMUX_CMD ' %q' "${CMD[@]}"
TMUX_CMD="${TMUX_CMD:1}"

{
  echo "tmux session: ${SESSION}"
  echo "log file: ${LOG_FILE}"
  echo "attach with: tmux attach -t ${SESSION}"
  echo
} | tee -a "$LOG_FILE"

printf -v WORKER_CMD 'cd /opt/neuralsignal && set -o pipefail; %s 2>&1 | tee -a %q; status=${PIPESTATUS[0]}; echo "$status" > %q; exit "$status"' "$TMUX_CMD" "$LOG_FILE" "$STATUS_FILE"
printf -v SHELL_CMD 'bash -c %q' "$WORKER_CMD"
tmux new-session -d -s "$SESSION" "$SHELL_CMD" || exit $?

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
