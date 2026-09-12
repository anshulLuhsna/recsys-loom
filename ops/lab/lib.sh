#!/usr/bin/env bash

LAB_ROOT="/mnt/lab-faults"
LAB_MARKER="$LAB_ROOT/.recsys-loom-lab"
LAB_MARKER_CONTENT="recsys-loom-synthetic-lab"
LAB_APP_DIR="${LAB_APP_DIR:-/opt/recsys-loom/app}"
LAB_COMPOSE_FILE="$LAB_APP_DIR/docker-compose.prod.yml"
LAB_RUNTIME_ENV="$LAB_APP_DIR/.env.runtime"
LAB_DEPLOY_ENV="$LAB_APP_DIR/.env.deploy"

lab_fail() {
  echo "lab: $*" >&2
  exit 1
}

require_lab() {
  [[ "${LAB_MODE:-}" == "1" ]] ||
    lab_fail "set LAB_MODE=1 for an intentional synthetic drill"
  [[ -d "$LAB_ROOT" && ! -L "$LAB_ROOT" ]] ||
    lab_fail "$LAB_ROOT must be a real directory"
  [[ -f "$LAB_MARKER" && ! -L "$LAB_MARKER" ]] ||
    lab_fail "missing lab marker; run enable.sh first"
  [[ "$(cat "$LAB_MARKER")" == "$LAB_MARKER_CONTENT" ]] ||
    lab_fail "lab marker content is invalid"
}

require_compose() {
  command -v docker >/dev/null 2>&1 || lab_fail "docker is unavailable"
  docker compose version >/dev/null 2>&1 ||
    lab_fail "Docker Compose v2 is unavailable"
  [[ -f "$LAB_COMPOSE_FILE" ]] || lab_fail "missing $LAB_COMPOSE_FILE"
  [[ -f "$LAB_RUNTIME_ENV" ]] || lab_fail "missing $LAB_RUNTIME_ENV"
  [[ -f "$LAB_DEPLOY_ENV" ]] || lab_fail "missing $LAB_DEPLOY_ENV"
}

base_compose() {
  docker compose \
    --env-file "$LAB_RUNTIME_ENV" \
    --env-file "$LAB_DEPLOY_ENV" \
    -f "$LAB_COMPOSE_FILE" \
    "$@"
}

print_cleanup_recovery() {
  echo "Recover: LAB_MODE=1 $LAB_APP_DIR/lab/cleanup.sh"
}
