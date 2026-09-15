"""A conversation with the attached board: the pty, the cargo process, the log.

The cargo runner (`espflash flash --monitor`) only streams the board's output
when it believes it is attached to a terminal, so the process runs under a
pty. `DeviceSession` owns the three resources that implies - the pty's file
descriptor, the subprocess and an optional log file - and releases all of
them when the with-block ends, however it ends.
"""

import os
import pty
import select
import subprocess
import sys
import time


class DeviceSession:
    """Context manager around one flash-and-monitor run.

    Usage::

        with DeviceSession(["cargo", "run", "--release", "--bin", "inference"],
                           cwd=firmware_dir, env=env) as session:
            for line in session.lines():
                ...

    Everything the board sends is echoed to stdout and, with `log_path`, also
    written to that file byte for byte.
    """

    def __init__(self, cmd, cwd=None, env=None, log_path=None, timeout=300.0, echo=True):
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = env
        self.log_path = log_path
        self.timeout = timeout
        self.echo = echo
        # Set by lines() when it gave up waiting, so a caller can tell a
        # timeout apart from the board finishing.
        self.timed_out = False
        self._master = None
        self._proc = None
        self._log = None

    def __enter__(self):
        master, slave = pty.openpty()
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdin=slave,
                stdout=slave,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)
        self._master = master
        if self.log_path is not None:
            self._log = open(self.log_path, "wb")
        return self

    def __exit__(self, *exc_info):
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        finally:
            os.close(self._master)
            if self._log is not None:
                self._log.close()
        return False

    @property
    def running(self) -> bool:
        """Whether the cargo process is still alive."""
        return self._proc.poll() is None

    def _read(self, wait: float) -> bytes | None:
        """One chunk from the pty: None if nothing arrived in `wait` s, b"" once the pty is closed."""
        ready, _, _ = select.select([self._master], [], [], wait)
        if not ready:
            return None
        try:
            return os.read(self._master, 4096)
        except OSError:
            # EIO: the child closed its side of the pty.
            return b""

    def _record(self, data: bytes) -> None:
        if self.echo:
            out = getattr(sys.stdout, "buffer", None)
            if out is not None:
                out.write(data)
                out.flush()
            else:
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        if self._log is not None:
            self._log.write(data)
            self._log.flush()

    def drain(self) -> bytes:
        """Read what is still buffered in the pty, without waiting for more."""
        chunks = []
        while True:
            data = self._read(0.05)
            if not data:
                break
            self._record(data)
            chunks.append(data)
        return b"".join(chunks)

    def lines(self):
        """Yield the board's output one line at a time, decoded and stripped.

        Ends when the process exits (after draining what it left in the pty),
        when the pty closes, or after `timeout` seconds, which sets
        `timed_out`. Breaking out of the loop early is fine: the with-block
        stops the process.
        """
        buffer = b""
        start = time.monotonic()
        while True:
            if time.monotonic() - start > self.timeout:
                self.timed_out = True
                print(f"\nTimeout reached ({self.timeout:g}s).")
                break
            if self._proc.poll() is not None:
                buffer += self.drain()
                break
            data = self._read(0.1)
            if data is None:
                continue
            if not data:
                break
            self._record(data)
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace").strip()
        # Whatever the process left behind, including a last line with no newline.
        for line in buffer.split(b"\n"):
            if line.strip():
                yield line.decode("utf-8", errors="replace").strip()
