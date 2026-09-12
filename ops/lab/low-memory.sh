#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
require_compose

memory_mb="${1:-256}"
service="${2:-search-api}"
[[ "$memory_mb" =~ ^[0-9]+$ ]] ||
  lab_fail "memory limit must be an integer MiB value"
(( memory_mb >= 128 && memory_mb <= 1024 )) ||
  lab_fail "memory limit must be between 128 and 1024 MiB"
[[ "$service" == "recommendation-api" || "$service" == "search-api" ]] ||
  lab_fail "service must be recommendation-api or search-api"
reservation_mb=$((memory_mb / 2))

base_compose config --services | grep -Fxq "$service" ||
  lab_fail "Compose configuration has no $service service"

override="$LAB_ROOT/low-memory.override.yml"
cat > "$override" <<EOF
services:
  $service:
    mem_limit: ${memory_mb}m
    mem_reservation: ${reservation_mb}m
EOF

docker compose \
  --env-file "$LAB_RUNTIME_ENV" \
  --env-file "$LAB_DEPLOY_ENV" \
  -f "$LAB_COMPOSE_FILE" \
  -f "$override" \
  up -d --no-deps --force-recreate "$service"

container_id="$(base_compose ps -q "$service")"
echo "Applied a bounded ${memory_mb} MiB memory limit to $service."
echo "Verify: docker inspect --format '{{.HostConfig.Memory}}' $container_id"
echo "After failure, compare: docker inspect --format '{{.State.OOMKilled}} {{.State.ExitCode}}' $container_id"
echo "Host evidence: journalctl -k --since '-10 minutes' | grep -i -E 'oom|killed process'"
print_cleanup_recovery
