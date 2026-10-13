"""The device half of evaluation, without a device: the serial protocol and the row stacking."""

import numpy as np
import pytest

from nanou.device.evaluate import device_probs, read_device_outputs
from nanou.predictor import logits_to_probs


def _row(values):
    """One output row as the firmware prints it: little-endian f32s, lowercase hex."""
    return np.asarray(values, dtype="<f4").tobytes().hex()


def test_the_protocol_is_decoded():
    lines = [
        "BENCHMARK_START:1",
        "IMG_PREPROCESS:0:120us",
        "IMG_INFERENCE:0:5000us",
        "IMG_STATS:0:-1.5,2.0",
        "IMG_OUTPUT_START:0",
        _row([1.0, 2.0, 3.0]),
        _row([-1.5, 0.0, 2.0]),
        "IMG_OUTPUT_END:0",
    ]
    out = read_device_outputs(iter(lines), width=3)
    assert list(out["img_data"]) == [0]
    np.testing.assert_array_equal(out["img_data"][0][1], [-1.5, 0.0, 2.0])
    assert out["latency_us"] == {0: {"preprocess": 120, "inference": 5000}}
    assert out["stats"] == {0: {"min": -1.5, "max": 2.0}}


def test_reading_stops_at_benchmark_done():
    lines = ["IMG_OUTPUT_START:0", _row([1, 2, 3]), "BENCHMARK_DONE", "IMG_OUTPUT_START:1"]
    assert list(read_device_outputs(iter(lines), width=3)["img_data"]) == [0]


def test_stray_short_and_garbled_rows_are_not_kept():
    lines = [
        _row([9, 9, 9]),  # before any block: not an output row
        "IMG_OUTPUT_START:0",
        _row([1, 2]),  # wrong length
        "zz" * 12,  # right length, not hex
        _row([1, 2, 3]),
        "IMG_OUTPUT_END:0",
        _row([9, 9, 9]),  # after the block
    ]
    rows = read_device_outputs(iter(lines), width=3)["img_data"][0]
    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0], [1, 2, 3])


def _image(value, height=2, width=3):
    return [np.full(width, value, dtype="<f4") for _ in range(height)]


def test_device_probs_are_the_sigmoid_of_the_logits_in_index_order():
    probs, kept = device_probs({1: _image(2.0), 0: _image(-1.0)}, n_images=2, height=2)
    assert kept == [0, 1]
    assert probs.shape == (2, 2, 3, 1)
    np.testing.assert_array_equal(probs[1], logits_to_probs(np.full((2, 3, 1), 2.0, "<f4")))


def test_an_incomplete_image_is_dropped_not_scored():
    probs, kept = device_probs({0: _image(1.0), 1: _image(1.0)[:1]}, n_images=2, height=2)
    assert kept == [0]
    assert probs.shape[0] == 1


def test_an_index_beyond_the_split_raises():
    with pytest.raises(ValueError, match="different image set"):
        device_probs({5: _image(1.0)}, n_images=2, height=2)


def test_nothing_complete_raises():
    with pytest.raises(RuntimeError, match="no complete image"):
        device_probs({0: _image(1.0)[:1]}, n_images=1, height=2)
