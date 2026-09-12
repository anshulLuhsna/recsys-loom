#!/usr/bin/env bash

set -Eeuo pipefail

LAB_ROOT="/mnt/lab-faults"
LAB_MARKER="$LAB_ROOT/.recsys-loom-lab"
LAB_MARKER_CONTENT="recsys-loom-synthetic-lab"

fail() {
  echo "lab-enable: $*" >&2
  exit 1
}

[[ "${LAB_MODE:-}" == "1" ]] ||
  fail "set LAB_MODE=1 for an intentional synthetic drill"
[[ "${LAB_CONFIRM:-}" == "SYNTHETIC_ONLY" ]] ||
  fail "set LAB_CONFIRM=SYNTHETIC_ONLY to create the lab marker"
[[ ! -L "$LAB_ROOT" ]] || fail "$LAB_ROOT must not be a symlink"
command -v findmnt >/dev/null 2>&1 || fail "findmnt is required"
mount_target="$(findmnt -n -o TARGET --target "$LAB_ROOT" 2>/dev/null || true)"
[[ "$mount_target" == "$LAB_ROOT" ]] ||
  fail "$LAB_ROOT must be a dedicated mounted filesystem, not the root volume"

install -d -m 2775 "$LAB_ROOT"
printf '%s\n' "$LAB_MARKER_CONTENT" > "$LAB_MARKER"
chmod 0644 "$LAB_MARKER"

echo "Synthetic lab mode enabled at $LAB_ROOT."
echo "Verify: cat $LAB_MARKER"
echo "Recover/disable: LAB_MODE=1 ${LAB_APP_DIR:-/opt/recsys-loom/app}/lab/cleanup.sh --disable"
