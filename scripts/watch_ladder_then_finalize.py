#!/usr/bin/env python3
"""Wait for the research ladder PID, then freeze and evaluate."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: watch_ladder_then_finalize.py LADDER_PID")
    pid = int(sys.argv[1])
    print(f"Watching research ladder pid={pid}", flush=True)
    while alive(pid):
        time.sleep(30)
    print("Ladder process exited; running finalize.", flush=True)
    completed = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "run_overnight_finalize.py")],
        cwd=ROOT,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
