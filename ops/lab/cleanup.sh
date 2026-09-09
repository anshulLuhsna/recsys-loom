#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
require_compose

disable=0
if [[ "${1:-}" == "--disable" ]]; then
  disable=1
elif [[ -n "${1:-}" ]]; then
  lab_fail "only --disable is supported"
fi

network_state="$LAB_ROOT/networks.state"
if [[ -f "$network_state" ]]; then
  while IFS=$'\t' read -r container_id network; do
    [[ "$container_id" =~ ^[0-9a-f]{12,64}$ ]] || continue
    [[ "$network" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || continue
    if docker inspect "$container_id" >/dev/null 2>&1 &&
      docker network inspect "$network" >/dev/null 2>&1; then
      docker network connect "$network" "$container_id" 2>/dev/null || true
    fi
  done < "$network_state"
fi

open_deleted_pid_file="$LAB_ROOT/open-deleted.pid"
if [[ -f "$open_deleted_pid_file" ]]; then
  process_id=""
  expected_start_time=""
  read -r process_id expected_start_time < "$open_deleted_pid_file" || true
  if [[ "$process_id" =~ ^[0-9]+$ && "$expected_start_time" =~ ^[0-9]+$ ]] &&
    [[ -r "/proc/$process_id/stat" ]]; then
    current_start_time="$(awk '{ print $22 }' "/proc/$process_id/stat")"
    if [[ "$current_start_time" == "$expected_start_time" ]]; then
      kill "$process_id" 2>/dev/null || true
    fi
  fi
fi

base_compose up -d --force-recreate --remove-orphans

rm -f -- \
  "$LAB_ROOT/bounded-fill.bin" \
  "$LAB_ROOT/bounded-fill.bin.tmp" \
  "$LAB_ROOT/low-memory.override.yml" \
  "$LAB_ROOT/missing-artifact.override.yml" \
  "$LAB_ROOT/missing-visual.override.yml" \
  "$LAB_ROOT/networks.state" \
  "$LAB_ROOT/open-deleted.pid" \
  "$LAB_ROOT/stopped-services.state"
rm -f -- "$LAB_ROOT/missing-artifact/manifest.json"
rmdir "$LAB_ROOT/missing-artifact" 2>/dev/null || true
if [[ -d "$LAB_ROOT/git-conflict" && ! -L "$LAB_ROOT/git-conflict" ]]; then
  rm -rf -- "$LAB_ROOT/git-conflict"
fi

drift_instance_file="$LAB_ROOT/terraform-drift.instance"
if [[ -f "$drift_instance_file" ]] && command -v aws >/dev/null 2>&1; then
  drifted_instance=""
  read -r drifted_instance < "$drift_instance_file" || true
  if [[ "$drifted_instance" =~ ^i-[0-9a-f]{8,17}$ ]]; then
    aws ec2 delete-tags \
      --resources "$drifted_instance" \
      --tags Key=LabDrift >/dev/null 2>&1 || true
  fi
  rm -f -- "$drift_instance_file"
fi

if (( disable == 1 )); then
  rm -f -- "$LAB_MARKER"
  rmdir "$LAB_ROOT" 2>/dev/null || true
  echo "Synthetic lab marker disabled."
fi

echo "Lab overrides removed and base Compose services recreated."
echo "Verify: docker compose --env-file $LAB_RUNTIME_ENV --env-file $LAB_DEPLOY_ENV -f $LAB_COMPOSE_FILE ps"
echo "Recover again if needed: LAB_MODE=1 $LAB_APP_DIR/lab/cleanup.sh"
