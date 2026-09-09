#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
require_compose

service="${1:-}"
requested_network="${2:-}"
[[ "$service" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
  lab_fail "provide a valid Compose service name"
[[ "$service" == "recommendation-api" || "$service" == "search-api" ]] ||
  lab_fail "service must be recommendation-api or search-api"
base_compose config --services | grep -Fxq "$service" ||
  lab_fail "unknown Compose service: $service"

container_id="$(base_compose ps -q "$service")"
[[ -n "$container_id" ]] || lab_fail "service is not running: $service"

mapfile -t attached_networks < <(
  docker inspect \
    --format '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{"\n"}}{{end}}' \
    "$container_id"
)
(( ${#attached_networks[@]} > 0 )) ||
  lab_fail "container has no attached network"

if [[ -n "$requested_network" ]]; then
  network=""
  for attached_network in "${attached_networks[@]}"; do
    if [[ "$attached_network" == "$requested_network" ]]; then
      network="$attached_network"
      break
    fi
  done
  [[ -n "$network" ]] ||
    lab_fail "container is not attached to network: $requested_network"
else
  network="${attached_networks[0]}"
fi

docker network disconnect "$network" "$container_id"
printf '%s\t%s\n' "$container_id" "$network" >> "$LAB_ROOT/networks.state"

echo "Disconnected $service container $container_id from $network."
echo "Verify: docker inspect --format '{{json .NetworkSettings.Networks}}' $container_id"
print_cleanup_recovery
