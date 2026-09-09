#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab

size_mb="${1:-64}"
[[ "$size_mb" =~ ^[0-9]+$ ]] || lab_fail "size must be an integer MiB value"
(( size_mb >= 1 && size_mb <= 8192 )) ||
  lab_fail "size must be between 1 and 8192 MiB"

available_mb="$(df -Pk "$LAB_ROOT" | awk 'NR == 2 { print int($4 / 1024) }')"
[[ "$available_mb" =~ ^[0-9]+$ ]] || lab_fail "could not determine free space"
(( available_mb - size_mb >= 1024 )) ||
  lab_fail "refusing to leave less than 1024 MiB free"

target="$LAB_ROOT/bounded-fill.bin"
temporary="$LAB_ROOT/bounded-fill.bin.tmp"
rm -f -- "$temporary"
dd \
  if=/dev/zero \
  of="$temporary" \
  bs=1048576 \
  count="$size_mb" \
  conv=fsync \
  status=none
mv -f -- "$temporary" "$target"

echo "Created a bounded ${size_mb} MiB synthetic file at $target."
echo "Verify: du -h $target && df -h $LAB_ROOT"
print_cleanup_recovery
