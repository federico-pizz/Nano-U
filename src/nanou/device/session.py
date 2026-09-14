"""Lifetime of a serial/pty conversation with an attached device.

B4 turns this into a `DeviceSession`; for now it holds the one helper that the
profiling and evaluation paths share.
"""

import os
import select
import sys


def drain_pty(master, f_log):
    """Read any remaining data from PTY after the main loop exits."""
    while True:
        r, _, _ = select.select([master], [], [], 0.05)
        if not r:
            break
        try:
            data = os.read(master, 4096)
        except OSError:
            break
        if not data:
            break
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except Exception:
            pass
        f_log.write(data)
        f_log.flush()
