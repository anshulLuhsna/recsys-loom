#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
require_compose

service="${1:-}"
[[ "$service" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
  lab_fail "provide a valid Compose service name"
[[ "$service" == "recommendation-api" || "$service" == "search-api" ]] ||
  lab_fail "service must be recommendation-api or search-api"
base_compose config --services | grep -Fxq "$service" ||
  lab_fail "unknown Compose service: $service"

state_file="$LAB_ROOT/stopped-services.state"
if [[ ! -f "$state_file" ]] || ! grep -Fxq "$service" "$state_file"; then
  printf '%s\n' "$service" >> "$state_file"
fi
base_compose stop "$service"

echo "Stopped Compose service: $service"
echo "Verify: docker compose -f $LAB_COMPOSE_FILE ps $service"
print_cleanup_recovery
