"""Unit tests for the serial protocol parsing logic in scripts/eval_esp32.py.

These import the real parser functions used by run_inference_on_device, so the
tests track the production wire format instead of a private copy of the logic.
"""

import struct
import numpy as np
import pytest

from scripts.eval_esp32 import parse_idx_us_line, parse_stats_line, decode_hex_row


# ── IMG_PREPROCESS parsing ────────────────────────────────────────────────────

def test_parse_preprocess_basic():
    idx, us = parse_idx_us_line("IMG_PREPROCESS:3:1234us")
    assert idx == 3
    assert us == 1234


def test_parse_preprocess_zero_index():
    idx, us = parse_idx_us_line("IMG_PREPROCESS:0:99us")
    assert idx == 0
    assert us == 99


def test_parse_preprocess_large_value():
    _, us = parse_idx_us_line("IMG_PREPROCESS:10:999999us")
    assert us == 999999


# ── IMG_INFERENCE parsing (same wire format as IMG_PREPROCESS) ─────────────────

def test_parse_inference_basic():
    idx, us = parse_idx_us_line("IMG_INFERENCE:0:54321us")
    assert idx == 0
    assert us == 54321


# ── IMG_STATS parsing ─────────────────────────────────────────────────────────

def test_parse_stats_positive():
    idx, min_v, max_v = parse_stats_line("IMG_STATS:0:1.23,4.56")
    assert idx == 0
    assert abs(min_v - 1.23) < 1e-5
    assert abs(max_v - 4.56) < 1e-5


def test_parse_stats_negative_min():
    _, min_v, max_v = parse_stats_line("IMG_STATS:2:-3.14,2.71")
    assert abs(min_v - (-3.14)) < 1e-4
    assert abs(max_v - 2.71) < 1e-4


def test_parse_stats_index_preserved():
    for i in range(5):
        idx, _, _ = parse_stats_line(f"IMG_STATS:{i}:0.0,1.0")
        assert idx == i


# ── f32 hex row encoding round-trip ──────────────────────────────────────────

def _f32_to_firmware_hex(value: float) -> str:
    """Encode a float32 as 8 hex chars using the firmware's byte order.

    The firmware iterates byte_idx 0..3 (LSB first), emitting 2 hex chars per
    byte.  This is little-endian byte order written sequentially, so the result
    differs from Python's big-endian format() on the integer bits.
    """
    raw = struct.pack("<f", value)          # 4 LE bytes of the float
    return "".join(f"{b:02x}" for b in raw)


def test_hex_row_roundtrip_single_pixel():
    """One pixel: encode f32 → 8 hex chars (firmware byte order) → decode back."""
    value = np.float32(-1.234567)
    hex_str = _f32_to_firmware_hex(float(value))
    decoded = decode_hex_row(hex_str)
    assert abs(decoded[0] - value) < 1e-6


def test_hex_row_roundtrip_row_of_80():
    """Full 80-pixel row: encode then decode must recover original values."""
    rng = np.random.default_rng(42)
    values = rng.standard_normal(80).astype(np.float32)
    hex_str = "".join(_f32_to_firmware_hex(float(v)) for v in values)
    decoded = decode_hex_row(hex_str)
    np.testing.assert_array_almost_equal(decoded, values, decimal=6)


def test_hex_row_length_is_width_times_8():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(80).astype(np.float32)
    hex_str = "".join(_f32_to_firmware_hex(float(v)) for v in values)
    assert len(hex_str) == 80 * 8


def test_hex_row_special_values():
    for v in [0.0, 1.0, -1.0, float("inf"), float("-inf")]:
        hex_str = _f32_to_firmware_hex(v)
        decoded = decode_hex_row(hex_str)
        if np.isfinite(v):
            assert abs(decoded[0] - v) < 1e-6
        else:
            assert not np.isfinite(decoded[0])


# ── latency stats calculation (mirrors evaluate_esp_outputs logic) ────────────

def test_latency_stats_mean_min_max():
    latency_us = {
        0: {"preprocess": 100, "inference": 5000},
        1: {"preprocess": 120, "inference": 4800},
        2: {"preprocess": 110, "inference": 5200},
    }
    infer_times = [v["inference"] for v in latency_us.values() if "inference" in v]
    assert np.mean(infer_times) == pytest.approx(5000.0)
    assert min(infer_times) == 4800
    assert max(infer_times) == 5200


def test_latency_stats_missing_key_skipped():
    latency_us = {
        0: {"preprocess": 100},          # no inference key
        1: {"preprocess": 110, "inference": 5000},
    }
    infer_times = [v["inference"] for v in latency_us.values() if "inference" in v]
    assert len(infer_times) == 1
    assert infer_times[0] == 5000
