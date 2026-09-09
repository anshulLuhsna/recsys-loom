#!/usr/bin/env bash

set -Eeuo pipefail

LOG_PATH="${DOCKER_EVENT_LOG:-/opt/recsys-loom/logs/docker-events.log}"
install -d -m 0750 "$(dirname -- "$LOG_PATH")"
touch "$LOG_PATH"
chmod 0640 "$LOG_PATH"

exec docker events \
  --filter type=container \
  --filter event=die \
  --filter event=oom \
  --filter event=restart \
  --format '{{json .}}'
