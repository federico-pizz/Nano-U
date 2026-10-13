"""Exported size, INT8 input/output types, and the operator set.

The operator set is read from the flatbuffer's operator table rather than from
a successful host interpretation: a Flex (SELECT_TF_OPS) fallback runs fine on
the host and cannot run at all on the device, so "it loaded" is not the check
that matters here.
"""

import pytest

from tests.contract import oracle

pytestmark = pytest.mark.contract

# What the ESP32-S3 kernel set can execute. Anything outside it does not run
# on the device, whatever the host interpreter says.
ALLOWED_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "MAX_POOL_2D",
    "RESIZE_NEAREST_NEIGHBOR",
    "PAD",
    "ADD",
    "CONCATENATION",
}


def test_io_is_int8_end_to_end(dataset, artifacts):
    io = oracle.tflite_io(artifacts["tflite"])
    assert io["input"]["dtype"] == "int8", f"{dataset}: input dtype"
    assert io["output"]["dtype"] == "int8", f"{dataset}: output dtype"
    assert io["input"]["scale"] > 0.0 and io["output"]["scale"] > 0.0


def test_io_shapes_are_the_deployed_resolution(dataset, artifacts):
    io = oracle.tflite_io(artifacts["tflite"])
    assert io["input"]["shape"] == [1, 60, 80, 3], f"{dataset}: input shape"
    assert io["output"]["shape"] == [1, 60, 80, 1], f"{dataset}: output shape"


def test_size_matches_baseline(dataset, artifacts, record):
    """Byte size, not a bound: a size change means the graph changed."""
    live = oracle.tflite_io(artifacts["tflite"])["size_bytes"]
    assert live == record["tflite_io"]["size_bytes"], (
        f"{dataset}: exported graph size moved from "
        f"{record['tflite_io']['size_bytes']} to {live} bytes"
    )


def test_io_quantization_matches_baseline(dataset, artifacts, record):
    live = oracle.tflite_io(artifacts["tflite"])
    for side in ("input", "output"):
        assert live[side] == record["tflite_io"][side], f"{dataset}: {side} changed"


def test_no_flex_or_custom_ops(dataset, artifacts):
    ops = oracle.tflite_op_codes(artifacts["tflite"])
    custom = [op for op in ops if op.startswith("CUSTOM")]
    assert not custom, f"{dataset}: graph needs ops the device cannot run: {custom}"
    assert set(ops) <= ALLOWED_OPS, f"{dataset}: unexpected ops {sorted(set(ops) - ALLOWED_OPS)}"


def test_op_set_matches_baseline(dataset, artifacts, record):
    assert oracle.tflite_op_codes(artifacts["tflite"]) == record["tflite_ops"]
