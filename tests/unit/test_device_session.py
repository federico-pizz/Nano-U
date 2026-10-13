"""DeviceSession owns a pty, a process and a log; these check it reads them right and lets go.

A short Python child stands in for `cargo run`, so no board is needed.
"""

import sys

import pytest

from nanou.device.session import DeviceSession


def _child(code):
    return [sys.executable, "-c", code]


def test_lines_are_decoded_stripped_and_complete(tmp_path):
    code = "print('BENCHMARK_START:2'); print('  padded  '); print('BENCHMARK_DONE')"
    with DeviceSession(_child(code), echo=False) as session:
        lines = list(session.lines())
    assert lines == ["BENCHMARK_START:2", "padded", "BENCHMARK_DONE"]
    assert not session.timed_out


def test_a_last_line_without_a_newline_is_not_lost():
    code = "import sys; print('first'); sys.stdout.write('tail'); sys.stdout.flush()"
    with DeviceSession(_child(code), echo=False) as session:
        assert list(session.lines()) == ["first", "tail"]


def test_the_log_gets_every_byte(tmp_path):
    log = tmp_path / "board.log"
    with DeviceSession(_child("print('STACK_PEAK: 1234')"), log_path=log, echo=False) as s:
        list(s.lines())
    assert b"STACK_PEAK: 1234" in log.read_bytes()


def test_a_silent_board_times_out_and_is_stopped():
    with DeviceSession(_child("import time; time.sleep(30)"), timeout=0.3, echo=False) as s:
        assert list(s.lines()) == []
    assert s.timed_out
    assert not s.running


def test_leaving_the_block_early_stops_the_process():
    code = "import time; print('ready', flush=True); time.sleep(30)"
    with DeviceSession(_child(code), echo=False) as s:
        assert next(s.lines()) == "ready"
    assert not s.running


def test_an_exception_in_the_block_still_stops_the_process():
    with pytest.raises(RuntimeError, match="boom"):
        with DeviceSession(_child("import time; time.sleep(30)"), echo=False) as s:
            raise RuntimeError("boom")
    assert not s.running
