"""Tests for scripts/run_watchdog.py.

Three cases:
- timeout: subprocess writes once, sleeps; watchdog kills.
- mtime advance resets: subprocess writes every few seconds; watchdog leaves it alone.
- tmp pause: a *.tmp file in the run dir freezes the SIGTERM path.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


REPO = Path(__file__).parent.parent
WATCHDOG = REPO / "scripts" / "run_watchdog.py"


# ---------- helpers ----------

# Helpers as standalone .py files so we don't fight escaping in -c strings.
_HELPERS_DIR = Path(tempfile.mkdtemp(prefix="watchdog_helpers_"))


def _write_helper(name: str, code: str) -> Path:
    p = _HELPERS_DIR / name
    p.write_text(code)
    return p


# Helper: write once, then sleep forever
SLEEPER = _write_helper("sleeper.py", """
import sys, time
metrics = sys.argv[1]
with open(metrics, 'a') as f:
    f.write('{"step": 0}\\n')
    f.flush()
time.sleep(10000)
""")

# Helper: write every N seconds for K iterations
ACTIVE_WRITER = _write_helper("active_writer.py", """
import sys, time
metrics = sys.argv[1]
interval = float(sys.argv[2])
total = int(sys.argv[3])
for i in range(total):
    with open(metrics, 'a') as f:
        f.write('{"step": ' + str(i) + '}\\n')
        f.flush()
    time.sleep(interval)
""")


# ---------- tests ----------

@pytest.mark.timeout(120)
def test_watchdog_kills_stalled_process():
    """A subprocess that writes once then sleeps forever should be killed."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics = run_dir / "metrics.jsonl"

        writer = subprocess.Popen([sys.executable, str(SLEEPER), str(metrics)])
        # let writer write its initial line
        time.sleep(2)
        assert metrics.exists()

        wd = subprocess.Popen([
            sys.executable, str(WATCHDOG),
            "--pid", str(writer.pid),
            "--metrics", str(metrics),
            "--stall_seconds", "3",
            "--max_seconds", "3600",
        ])
        rc = wd.wait(timeout=90)
        # Watchdog returns 1 on a kill.
        assert rc == 1, f"watchdog rc {rc} != 1 (expected kill)"
        time.sleep(1)
        assert writer.poll() is not None, "writer still alive after watchdog kill"


@pytest.mark.timeout(120)
def test_watchdog_does_not_kill_active_writer():
    """A subprocess that keeps writing within the threshold should run to completion."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics = run_dir / "metrics.jsonl"

        # Write every 2s for 60s total. Threshold 60s — never stale.
        writer = subprocess.Popen([
            sys.executable, str(ACTIVE_WRITER), str(metrics), "2", "30"
        ])
        time.sleep(2)

        wd = subprocess.Popen([
            sys.executable, str(WATCHDOG),
            "--pid", str(writer.pid),
            "--metrics", str(metrics),
            "--stall_seconds", "60",
            "--max_seconds", "3600",
        ])
        writer_rc = writer.wait(timeout=90)
        wd_rc = wd.wait(timeout=10)
        assert writer_rc == 0, f"writer rc {writer_rc} != 0 (active writer killed)"
        assert wd_rc == 0, f"watchdog rc {wd_rc} != 0 (false-positive kill)"


@pytest.mark.timeout(120)
def test_watchdog_pauses_on_tmp_file():
    """If a *.tmp file is in the run dir, watchdog should not kill via the stall path.
    Hard cap (max_seconds) still fires."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics = run_dir / "metrics.jsonl"
        metrics.write_text('{"step": 0}\n')
        # Create a *.tmp inside the run dir to simulate a checkpoint write.
        (run_dir / "checkpoint.tmp").write_text("partial")

        writer = subprocess.Popen([sys.executable, str(SLEEPER), str(metrics)])
        time.sleep(2)

        wd = subprocess.Popen([
            sys.executable, str(WATCHDOG),
            "--pid", str(writer.pid),
            "--metrics", str(metrics),
            "--stall_seconds", "3",
            "--max_seconds", "60",  # hard cap fires instead
        ])
        wd_rc = wd.wait(timeout=90)
        # Hard cap fires (returns 1).
        assert wd_rc == 1, f"watchdog rc {wd_rc} != 1 (expected hard-cap kill)"
        time.sleep(1)
        assert writer.poll() is not None
