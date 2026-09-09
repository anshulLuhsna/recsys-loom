#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
command -v python3 >/dev/null 2>&1 || lab_fail "python3 is unavailable"

size_mb="${1:-64}"
[[ "$size_mb" =~ ^[0-9]+$ ]] || lab_fail "size must be an integer MiB value"
(( size_mb >= 1 && size_mb <= 256 )) ||
  lab_fail "size must be between 1 and 256 MiB"

target="$LAB_ROOT/open-deleted.bin"
pid_file="$LAB_ROOT/open-deleted.pid"
dd if=/dev/zero of="$target" bs=1048576 count="$size_mb" status=none

python3 - "$target" <<'PY' &
import os
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
with path.open("rb") as handle:
    path.unlink()
    os.fstat(handle.fileno())
    time.sleep(3600)
PY
process_id=$!
sleep 1
kill -0 "$process_id" 2>/dev/null || lab_fail "holder process exited early"
start_time="$(awk '{ print $22 }' "/proc/$process_id/stat")"
printf '%s %s\n' "$process_id" "$start_time" > "$pid_file"

echo "Deleted a bounded ${size_mb} MiB file while process $process_id holds it open."
echo "Verify: lsof +L1 $LAB_ROOT; compare df -h $LAB_ROOT with du -sh $LAB_ROOT"
print_cleanup_recovery
