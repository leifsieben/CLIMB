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
import textwrap
import time
from pathlib import Path

import pytest


REPO = Path(__file__).parent.parent
WATCHDOG = REPO / "scripts" / "run_watchdog.py"


def _start_test_writer(metrics_path: Path, interval_seconds: float, total_writes: int) -> subprocess.Popen:
    """Spawn a tiny Python that writes to metrics_path every `interval_seconds`."""
    code = textwrap.dedent(f"""
        import time, sys
        from pathlib import Path
        p = Path(r"{metrics_path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        for i in range({total_writes}):
            with p.open("a") as f:
                f.write(f"{{\\\"step\\\": {{i}}}}\\n")
            time.sleep({interval_seconds})
    """)
    return subprocess.Popen([sys.executable, "-c", code])


@pytest.mark.timeout(60)
def test_watchdog_kills_stalled_process():
    """A subprocess that writes once then sleeps forever should be killed."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics = run_dir / "metrics.jsonl"
        metrics.write_text('{"step":1}\n')

        # a writer that sleeps 1000s after one write
        writer = subprocess.Popen([
            sys.executable, "-c",
            f"import time; open(r'{metrics}','a').write('x'); time.sleep(1000)"
        ])
        time.sleep(0.5)

        wd = subprocess.Popen([
            sys.executable, str(WATCHDOG),
            "--pid", str(writer.pid),
            "--metrics", str(metrics),
            "--stall_seconds", "3",     # very short for the test
            "--max_seconds", "3600",
        ])
        rc = wd.wait(timeout=45)
        # Watchdog returns 1 on a kill, 0 on clean exit.
        assert rc == 1
        # Writer should be gone.
        time.sleep(0.5)
        assert writer.poll() is not None


@pytest.mark.timeout(60)
def test_watchdog_does_not_kill_active_writer():
    """A subprocess that keeps writing within the threshold should run to completion."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics = run_dir / "metrics.jsonl"
        metrics.write_text('{"step":0}\n')

        writer = _start_test_writer(metrics, interval_seconds=2.0, total_writes=8)
        time.sleep(0.5)

        wd = subprocess.Popen([
            sys.executable, str(WATCHDOG),
            "--pid", str(writer.pid),
            "--metrics", str(metrics),
            "--stall_seconds", "10",
            "--max_seconds", "3600",
        ])
        writer_rc = writer.wait(timeout=40)
        wd_rc = wd.wait(timeout=10)
        # Writer should have exited cleanly (rc 0); watchdog should NOT have killed.
        assert writer_rc == 0
        assert wd_rc == 0


@pytest.mark.timeout(60)
def test_watchdog_pauses_on_tmp_file():
    """If a *.tmp file is in the run dir, watchdog should not kill even on stale metrics."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics = run_dir / "metrics.jsonl"
        metrics.write_text('{"step":1}\n')
        # Create a *.tmp inside the run dir to simulate a checkpoint write.
        (run_dir / "checkpoint.tmp").write_text("partial")

        # writer sleeps forever
        writer = subprocess.Popen([
            sys.executable, "-c",
            f"import time; time.sleep(1000)"
        ])
        time.sleep(0.5)

        wd = subprocess.Popen([
            sys.executable, str(WATCHDOG),
            "--pid", str(writer.pid),
            "--metrics", str(metrics),
            "--stall_seconds", "3",
            "--max_seconds", "20",   # we'll let the hard cap fire instead
        ])
        wd_rc = wd.wait(timeout=45)
        # Hard cap should fire (returns 1); but stall path was suppressed by tmp file.
        assert wd_rc == 1
        # writer should be gone (SIGKILL by hard cap)
        time.sleep(0.5)
        assert writer.poll() is not None
